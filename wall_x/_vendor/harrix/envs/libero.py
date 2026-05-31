"""LIBERO environment implementation.

The chunk-level API mirrors standard LIBERO rollout semantics:
- reset_episode enables rendering, sets initial state, and performs warmup steps.
- execute_chunk may skip intermediate image rendering but re-enables rendering
  before returning an observation.
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Optional

import numpy as np

from wall_x._vendor.harrix.envs.base import BaseEnv
from wall_x._vendor.harrix.envs.libero_common import (
    get_rollout_frame,
    model_action_to_libero_env,
    save_rollout_video,
)
from wall_x._vendor.harrix.envs.libero_sim import (
    create_libero_engine,
    find_image_observables,
    get_instruction,
    get_libero_dummy_action,
    get_task_suite,
    load_initial_states,
    pick_initial_state,
    resolve_task_info,
    set_render_enabled,
)
from wall_x._vendor.harrix.envs.registry import register_env
from wall_x._vendor.harrix.eval_config import EvalConfig

logger = logging.getLogger(__name__)


@register_env("libero")
class LiberoEnv(BaseEnv):

    def __init__(self, cfg: EvalConfig, worker_id: int) -> None:
        self._cfg = cfg
        self._libero_cfg = cfg.env.libero
        self._worker_id = worker_id
        self._seed = cfg.env.seed
        self._task_suite_name = self._libero_cfg.task_suite_name

        # Task-suite metadata.
        self._task_suite = get_task_suite(self._task_suite_name)
        self._num_tasks = self._task_suite.n_tasks
        self._custom_initial_states = load_initial_states(
            self._libero_cfg.initial_states_path
        )

        # Robosuite engine, lazily rebuilt on task changes.
        self._libero_env = None
        self._current_task_id: Optional[int] = None
        self._rebuild_env_per_episode = self._libero_cfg.rebuild_env_per_episode

        # Render-skip state.
        self._skip_intermediate_render = self._libero_cfg.skip_intermediate_render
        self._force_render_task_ids = set(
            self._libero_cfg.force_render_task_indices or []
        )
        self._effective_skip_render = self._skip_intermediate_render
        self._image_obs: list = []
        self._chunk_granular_render_toggle = (
            self._libero_cfg.chunk_granular_render_toggle
        )
        self._render_enabled_state = False

        # Optional bit-alignment dump for debugging.
        self._bit_dump_dir = os.environ.get("WALLX_BIT_DUMP_DIR", "").strip() or None
        if self._bit_dump_dir:
            os.makedirs(self._bit_dump_dir, exist_ok=True)

        rollout_dir = (self._libero_cfg.rollout_dir or "").strip()
        if not rollout_dir:
            rollout_dir = os.environ.get("WALLX_ROLLOUT_DIR", "").strip()
        if rollout_dir and os.environ.get("WALLX_DISABLE_ROLLOUT", "0") == "1":
            rollout_dir = ""
        self._rollout_dir = rollout_dir or None
        self._rollout_fps = int(self._libero_cfg.rollout_fps)
        if self._rollout_dir:
            worker_subdir = f"worker{worker_id}" if cfg.runtime.num_workers > 1 else ""
            self._rollout_dir = os.path.join(
                self._rollout_dir,
                self._task_suite_name,
                worker_subdir,
            )
            os.makedirs(self._rollout_dir, exist_ok=True)
            logger.info("Rollout MP4 saving enabled: %s", self._rollout_dir)

        # Per-episode state consumed by execute_chunk.
        self._current_ep: Optional[tuple] = None
        self._current_task_desc: str = ""
        self._current_instruction: str = ""
        self._chunk_counter_in_ep: int = 0
        self._last_obs_for_dump: Optional[dict] = None
        self._replay_images: list[np.ndarray] = []
        self._rollout_saved = False

    # ---- BaseEnv API ----

    @classmethod
    def enumerate_episodes(cls, cfg: EvalConfig) -> list[tuple]:
        libero_cfg = cfg.env.libero
        suite = libero_cfg.task_suite_name

        if libero_cfg.task_indices is not None:
            task_indices = [int(x) for x in libero_cfg.task_indices]
        else:
            ts = get_task_suite(suite)
            task_indices = list(range(ts.n_tasks))

        eps = []
        for tid in task_indices:
            for epi in range(libero_cfg.num_trials_per_task):
                eps.append((suite, tid, epi))
        return eps

    @property
    def robot_spec(self) -> dict:
        """Return robot spec for driver/adapter validation."""
        from wall_x._vendor.harrix.utils.train_config import (
            load_train_config_with_ckpt_overlay,
        )

        train_cfg = load_train_config_with_ckpt_overlay(
            self._cfg.model.train_config_path,
            self._cfg.model.checkpoint_path,
        )
        return {
            "dof_layout": train_cfg.get("dof_config", {}),
            "cam_names": list(self._cfg.model.cam_names),
            "norm_key": self._cfg.model.norm_key,
        }

    def _max_infer_rounds(self) -> int:
        return self._libero_cfg.max_infer_times

    def reset_episode(self, ep_id: tuple) -> dict:
        suite, task_id, ep_idx = ep_id
        if suite != self._task_suite_name:
            raise ValueError(
                f"env bound to suite={self._task_suite_name}, got ep with suite={suite}"
            )

        need_rebuild = self._rebuild_env_per_episode or self._current_task_id != task_id
        if need_rebuild:
            self._rebuild_env(task_id)

        task_desc, default_states = resolve_task_info(self._task_suite, task_id)
        init_state = pick_initial_state(
            self._libero_cfg.initial_states_path,
            self._custom_initial_states,
            task_desc,
            default_states,
            ep_idx,
        )

        if not need_rebuild:
            self._libero_env.reset()
        obs = self._libero_env.set_init_state(init_state)
        if obs is None:
            raise RuntimeError("set_init_state returned None")

        set_render_enabled(self._image_obs, True)
        self._render_enabled_state = True

        dummy_action = get_libero_dummy_action()
        for _ in range(10):
            obs, _, _, _ = self._libero_env.step(dummy_action)

        self._current_ep = (task_id, ep_idx)
        self._current_task_desc = task_desc
        self._current_instruction = get_instruction(task_desc)
        self._chunk_counter_in_ep = 0
        self._last_obs_for_dump = obs
        self._begin_rollout_capture(obs)

        return {
            "obs": obs,
            "instruction": self._current_instruction,
            "task_desc": task_desc,
        }

    def execute_chunk(self, actions: np.ndarray) -> dict:
        actions = np.asarray(actions, dtype=np.float32)
        H = actions.shape[0]

        if self._bit_dump_dir and self._current_ep is not None:
            self._dump_chunk_npz(
                (self._task_suite_name, *self._current_ep),
                self._chunk_counter_in_ep,
                self._last_obs_for_dump or {},
                actions,
            )
        self._chunk_counter_in_ep += 1

        skip_render = self._chunk_skip_render()
        # Disable rendering at the chunk start when render-skip is enabled.
        if skip_render and self._image_obs:
            if self._render_enabled_state:
                set_render_enabled(self._image_obs, False)
                self._render_enabled_state = False

        last_obs = None
        done = False
        steps = 0
        for step_idx in range(H):
            # Re-enable rendering before the final step to return a fresh image.
            if (
                skip_render
                and self._image_obs
                and step_idx == H - 1
                and not self._render_enabled_state
            ):
                set_render_enabled(self._image_obs, True)
                self._render_enabled_state = True

            action = model_action_to_libero_env(actions[step_idx].reshape(-1))
            obs, _, done_flag, _ = self._libero_env.step(action)
            last_obs = obs
            steps += 1
            self._append_rollout_frame(obs)
            if bool(done_flag):
                done = True
                # Do not add an extra simulator step on early success; just make
                # sure future rendering is enabled.
                if skip_render and self._image_obs and not self._render_enabled_state:
                    set_render_enabled(self._image_obs, True)
                    self._render_enabled_state = True
                break

        self._last_obs_for_dump = last_obs
        return {"obs": last_obs, "done": done, "steps": steps}

    def finalize_episode(self, success: bool) -> None:
        self._save_episode_rollout(success)

    def shutdown(self) -> None:
        if self._libero_env is not None:
            try:
                self._libero_env.close()
            except Exception:
                pass
            self._libero_env = None

    # ---- internals ----

    def _chunk_skip_render(self) -> bool:
        """Skip intermediate renders unless rollout MP4 saving needs every frame."""
        return self._effective_skip_render and self._rollout_dir is None

    def _begin_rollout_capture(self, obs: dict | None) -> None:
        self._replay_images = []
        self._rollout_saved = False
        if self._rollout_dir and obs is not None:
            if self._image_obs and not self._render_enabled_state:
                set_render_enabled(self._image_obs, True)
                self._render_enabled_state = True
            self._replay_images.append(get_rollout_frame(obs))

    def _append_rollout_frame(self, obs: dict | None) -> None:
        if self._rollout_dir and obs is not None:
            self._replay_images.append(get_rollout_frame(obs))

    def _save_episode_rollout(self, success: bool) -> None:
        if (
            not self._rollout_dir
            or self._rollout_saved
            or not self._replay_images
            or self._current_ep is None
        ):
            return
        task_id, ep_idx = self._current_ep
        try:
            mp4_path = save_rollout_video(
                self._rollout_dir,
                self._replay_images,
                task_id=task_id,
                episode_idx=ep_idx,
                success=success,
                task_description=self._current_task_desc,
                fps=self._rollout_fps,
            )
            self._rollout_saved = True
            if mp4_path:
                logger.info("Saved rollout MP4: %s", mp4_path)
        except Exception as exc:
            logger.warning(
                "Failed to save rollout MP4 for task%d ep%d: %s",
                task_id,
                ep_idx,
                exc,
            )

    def _rebuild_env(self, task_id: int) -> None:
        if self._libero_env is not None:
            try:
                self._libero_env.close()
            except Exception:
                pass
            self._libero_env = None
            gc.collect()

        self._libero_env = create_libero_engine(
            task_id=task_id,
            task_suite_name=self._task_suite_name,
            resolution=256,
            seed=self._seed,
        )
        self._current_task_id = task_id
        self._image_obs = find_image_observables(self._libero_env)
        self._effective_skip_render = (
            self._skip_intermediate_render
            and task_id not in self._force_render_task_ids
        )

    def _dump_chunk_npz(
        self, ep_id, chunk_idx: int, raw_obs: dict, chunk_actions: np.ndarray
    ) -> None:
        _, task_id, ep_idx = ep_id
        path = os.path.join(
            self._bit_dump_dir, f"t{task_id}_ep{ep_idx}_c{chunk_idx}.npz"
        )
        fields = {"action_chunk": np.asarray(chunk_actions, dtype=np.float32)}
        for k in (
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "agentview_image",
            "robot0_eye_in_hand_image",
        ):
            v = raw_obs.get(k)
            if v is not None:
                fields[k] = np.asarray(v)
        np.savez_compressed(path, **fields)
