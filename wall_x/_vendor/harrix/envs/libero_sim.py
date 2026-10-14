"""LIBERO benchmark and robosuite engine helpers.

This module may import robosuite and LIBERO. Adapter-side code should use
``libero_common.py`` instead, which only depends on NumPy.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# LIBERO config bootstrap. Called lazily (from the functions that actually use
# LIBERO), NOT at import time — importing this module only registers the env
# class and must not require LIBERO/robosuite or touch ~/.libero.
# ============================================================


def _ensure_libero_config() -> None:
    """Create ``$LIBERO_CONFIG_PATH/config.yaml`` (default ``~/.libero``) if absent
    so that importing ``libero.libero`` does not drop into its interactive
    ``input()`` prompt — which raises EOFError under non-interactive/mpirun runs.

    Locate the package via ``importlib`` instead of ``import libero.libero``:
    importing it is precisely what triggers the prompt when the config is missing,
    so the old import-to-find-the-path approach could never bootstrap a fresh node.
    Idempotent and safe to call on every LIBERO entry point.
    """
    import importlib.util

    libero_config_path = os.environ.get(
        "LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero")
    )
    config_file = os.path.join(libero_config_path, "config.yaml")
    if os.path.exists(config_file):
        return

    spec = importlib.util.find_spec("libero.libero")
    if spec is None or not spec.origin:
        return  # LIBERO not installed; nothing to pre-configure

    import yaml as _yaml

    benchmark_root = os.path.dirname(os.path.abspath(spec.origin))
    os.makedirs(libero_config_path, exist_ok=True)
    default_paths = {
        "benchmark_root": benchmark_root,
        "bddl_files": os.path.join(benchmark_root, "./bddl_files"),
        "init_states": os.path.join(benchmark_root, "./init_files"),
        "datasets": os.path.join(benchmark_root, "../datasets"),
        "assets": os.path.join(benchmark_root, "./assets"),
    }
    with open(config_file, "w") as f:
        _yaml.dump(default_paths, f)
    logger.info("Auto-created LIBERO config: %s", config_file)


# ============================================================
# task-suite entry point
# ============================================================


def get_task_suite(task_suite_name: str):
    """Load a LIBERO task suite."""
    _ensure_libero_config()
    from libero.libero import benchmark

    return benchmark.get_benchmark_dict()[task_suite_name]()


# ============================================================
# actions
# ============================================================


def get_libero_dummy_action() -> list[float]:
    """Return the 7-dof dummy action used for episode warmup."""
    return [0, 0, 0, 0, 0, 0, -1]


# ============================================================
# robosuite engine factory
# ============================================================


def _install_offsamples_hook(offsamples: int) -> None:
    """Set the MSAA sample count before the render context is created.

    EGL off-screen MSAA resolve is the only source of run-to-run
    nondeterminism in LIBERO evaluation: the first wrist-camera frame already
    differs by a few pixel LSBs, which closed-loop control amplifies into
    success-rate noise. With ``offsamples=0`` repeated full evaluations are
    bit-identical.

    The value must be applied before robosuite creates its render context,
    because the context allocates the MSAA buffer from
    ``vis.quality.offsamples`` at construction time and ``OffScreenRenderEnv``
    creates it in its constructor. The hook is therefore installed on
    ``MjRenderContext.__init__``.
    """
    from robosuite.utils import binding_utils

    if getattr(binding_utils.MjRenderContext, "_wallx_offsamples_hook", None) == offsamples:
        return
    orig_init = binding_utils.MjRenderContext.__init__

    def patched_init(self, sim, *args, **kwargs):
        raw = getattr(sim.model, "_model", sim.model)
        raw.vis.quality.offsamples = offsamples
        return orig_init(self, sim, *args, **kwargs)

    binding_utils.MjRenderContext.__init__ = patched_init
    binding_utils.MjRenderContext._wallx_offsamples_hook = offsamples


def create_libero_engine(
    task_id: int,
    task_suite_name: str,
    resolution: int = 256,
    seed: int = 7,
) -> Any:
    """Construct one LIBERO robosuite engine."""
    _ensure_libero_config()
    from robosuite.wrappers import VisualizationWrapper
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    offsamples = os.environ.get("WALLX_RENDER_OFFSAMPLES", "").strip()
    if offsamples:
        _install_offsamples_hook(int(offsamples))

    task_suite = get_task_suite(task_suite_name)
    task = task_suite.get_task(task_id)
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    # The seed still affects object poses even when an initial state is fixed.
    env.seed(seed)
    env.env = VisualizationWrapper(env.env)
    env.env.set_visualization_setting(setting="grippers", visible=False)
    return env


# ============================================================
# task metadata / initial states
# ============================================================


def load_initial_states(initial_states_path: str) -> Optional[dict]:
    """Load custom initial states, or return None for suite defaults."""
    if initial_states_path == "DEFAULT":
        return None
    with open(initial_states_path, "r") as f:
        return json.load(f)


def resolve_task_info(task_suite, task_id: int) -> tuple[str, Any]:
    """Return ``(task_desc, default_initial_states)`` for one task id."""
    num_tasks = task_suite.n_tasks
    if task_id < 0 or task_id >= num_tasks:
        raise ValueError(f"invalid task_id={task_id}, num_tasks={num_tasks}")
    task = task_suite.get_task(task_id)
    return task.language, task_suite.get_task_init_states(task_id)


def pick_initial_state(
    initial_states_path: str,
    custom_initial_states: Optional[dict],
    task_desc: str,
    default_states: Any,
    episode_idx: int,
) -> np.ndarray:
    """Pick one initial state from suite defaults or a custom states file."""
    if initial_states_path == "DEFAULT":
        if default_states is None:
            raise ValueError("default states missing for DEFAULT mode")
        return default_states[episode_idx]

    if custom_initial_states is None:
        raise ValueError(
            f"custom initial states not loaded for {initial_states_path}"
        )
    key = task_desc.replace(" ", "_")
    ep_key = f"demo_{episode_idx}"
    record = custom_initial_states[key][ep_key]
    if not record["success"]:
        raise ValueError(f"expert demo failed for {ep_key}")
    return np.array(record["initial_state"])


def get_instruction(task_desc: str) -> str:
    """Return the instruction text for a LIBERO task description."""
    return task_desc


# ============================================================
# render-skip: directly assign obs._enabled to avoid set_enabled() side effects.
# ============================================================


def find_image_observables(env) -> list:
    """Find image observables along the env.env wrapper chain."""
    cur = env
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, "_observables") and isinstance(cur._observables, dict):
            return [
                obs
                for obs in cur._observables.values()
                if getattr(obs, "modality", None) == "image"
            ]
        cur = getattr(cur, "env", None)
    return []


def set_render_enabled(image_obs_list, enabled: bool) -> None:
    for obs in image_obs_list:
        obs._enabled = enabled
