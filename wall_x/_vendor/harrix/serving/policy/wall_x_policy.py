import base64
import logging
from typing import Dict, Any, List

import cv2
import numpy as np
import torch
from wall_x._vendor.harrix.serving.websocket_policy_server import BasePolicy
from wall_x._vendor.harrix.infer.infer_config import InferConfig
from wall_x._vendor.harrix.infer.model_wrapper import WallxModelWrapper
from wall_x._vendor.harrix.infer.robot import (
    DesktopRobotPreprocessor,
    TurtleRobotPreprocessor,
    EX001RobotPreprocessor,
)
from wall_x._vendor.harrix.serving.protocol import rtc
from wall_x.utils.timers import ScopeTimer

logger = logging.getLogger(__name__)

INFER_MODE_FLOW = "flow"
INFER_MODE_AR = "ar"
INFER_MODE_FLOW_WITH_SUBTASK = "flow_with_subtask"
INFER_MODE_DLLM_FLOW = "dllm"
INFER_MODE_DLLM_DD = "discrete_diffusion"
INFER_MODE_FLOW_RTC = "flow_rtc"
ACTION_INFER_MODES = (
    INFER_MODE_FLOW,
    INFER_MODE_AR,
    INFER_MODE_FLOW_WITH_SUBTASK,
    INFER_MODE_DLLM_FLOW,
    INFER_MODE_DLLM_DD,
    INFER_MODE_FLOW_RTC,
)


class WallXPolicy(BasePolicy):
    """Policy wrapper for Wall-X model that implements the BasePolicy interface."""

    def __init__(
        self,
        config: InferConfig,
        image_passing_mode: str = "base64",
        default_infer_mode: str = INFER_MODE_FLOW,
        serialize_actions: bool = True,
        ar_do_sample: bool = False,
        ar_temperature: float = 1.0,
        instruction_override: str | None = None,
    ):
        """Initialize the Wall-X policy.

        Args:
            config: Inference configuration dataclass.
            image_passing_mode: How images are passed from client ('base64' or 'numpy').
            default_infer_mode: Default inference mode ('flow', 'ar', 'flow_with_subtask', etc.).
            serialize_actions: If True, actions are serialized via robot_preprocessor;
                if False, raw model output is returned directly.
            ar_do_sample: Default AR sampling switch (False = greedy, accuracy-optimal
                for RVQ AR). Per-request obs["ar_do_sample"] overrides this.
            ar_temperature: Default AR sampling temperature (inert unless do_sample).
                Per-request obs["ar_temperature"] overrides this.
            instruction_override: If set, REPLACES the client-sent obs["instruction"]
                for every request (a server-side prompt hack). Per-request
                obs["instruction_override"] takes precedence over this.
        """
        self.config = config
        self.model_wrapper = WallxModelWrapper(config)
        self.robot_preprocessor = self._register_robot_preprocessor()
        self.image_passing_mode = image_passing_mode
        self.default_infer_mode = default_infer_mode
        self.serialize_actions = serialize_actions
        # AR sampling defaults; per-request obs can override (default greedy).
        self.ar_do_sample = ar_do_sample
        self.ar_temperature = ar_temperature
        # Server-side instruction override (None = use whatever the client sends).
        self.instruction_override = instruction_override
        logger.info(
            "Image passing mode: %s, robot_type: %s, default_infer_mode: %s, "
            "ar_do_sample: %s, ar_temperature: %s, instruction_override: %r",
            self.image_passing_mode,
            config.robot_type,
            self.default_infer_mode,
            self.ar_do_sample,
            self.ar_temperature,
            self.instruction_override,
        )
        self._maybe_warmup_rtc()

    def _maybe_warmup_rtc(self) -> None:
        """Pre-run RTC inference for the delays listed in env RTC_WARMUP_DELAYS.

        Delays (comma or space separated) are still read per request from the
        client observation; warmup only removes the first-hit latency for each
        delay. Best effort: failures are logged as warnings and never block
        serving startup.
        """
        import os as _os
        raw = _os.getenv("RTC_WARMUP_DELAYS", "").strip()
        if not raw:
            return
        try:
            delays = sorted({int(x) for x in raw.replace(",", " ").split() if x.strip()})
        except ValueError:
            logger.warning("failed to parse RTC_WARMUP_DELAYS (ignored): %r", raw)
            return
        if not delays:
            return
        logger.info("RTC startup warmup: delays=%s (delay is still read per request)", delays)
        try:
            import numpy as _np

            res = 224
            if self.image_passing_mode == "base64":
                import base64 as _b64

                _img = _np.zeros((res, res, 3), dtype=_np.uint8)
                _ok, _buf = cv2.imencode(".jpg", _img)
                _v = _b64.b64encode(_buf).decode()
                views_tmpl = {"camera_front": _v, "camera_left": _v, "camera_right": _v}
            else:
                views_tmpl = {
                    c: _np.zeros((1, res, res, 3), dtype=_np.uint8)
                    for c in ("camera_front", "camera_left", "camera_right")
                }
            state_tmpl = {
                "follow1_pos": [0.0] * 7, "follow2_pos": [0.0] * 7,
                "follow1_gripper": [0.0], "follow2_gripper": [0.0],
                "follow1_joints_cur": [0.0] * 7, "follow2_joints_cur": [0.0] * 7,
                "head_pos": [0.0, 0.0], "lift": [0.0],
                "velocity_decomposed": [0.0, 0.0, 0.0],
                "follow1_end_effort_force": [0.0] * 3, "follow1_end_effort_torque": [0.0] * 3,
                "follow2_end_effort_force": [0.0] * 3, "follow2_end_effort_torque": [0.0] * 3,
            }
        except Exception as e:  # pragma: no cover
            logger.warning("RTC warmup: failed to build dummy input (skipped): %s", e)
            return
        for d in delays:
            try:
                obs = {
                    "state": dict(state_tmpl),
                    "views": {
                        k: (v.copy() if hasattr(v, "copy") else v)
                        for k, v in views_tmpl.items()
                    },
                    "instruction": "warmup",
                    "use_rtc": True,
                    "prev_action_chunk": [[0.0] * 14 for _ in range(d + 1)],
                    "inference_delay": d,
                }
                self.infer(obs)
                logger.info("RTC warmup delay=%d done", d)
            except Exception as e:  # best-effort
                logger.warning("RTC warmup delay=%d failed (skipped, startup continues): %s", d, e)
        logger.info("RTC warmup finished")

    def _resolve_instruction(self, obs: Dict) -> str | Dict[str, str]:
        """Effective instruction: per-request override > launch override > client.

        The per-request key obs["instruction_override"] lets a single frame force a
        prompt; self.instruction_override is the server-wide default override; else
        fall back to the client-sent obs["instruction"]. Structured
        wallx_task_instruction input uses {"task": str, "subtask": str}; the
        legacy "detail" key is also accepted.
        """
        return (
            obs.get("instruction_override")
            or self.instruction_override
            or obs["instruction"]
        )

    def _register_robot_preprocessor(self):
        """Select the robot preprocessor from ``config.robot_type``."""
        if self.config.robot_type == "desktop":
            return DesktopRobotPreprocessor(self.config)
        if self.config.robot_type == "turtle":
            return TurtleRobotPreprocessor(self.config)
        if self.config.robot_type == "ex001":
            return EX001RobotPreprocessor(self.config)
        raise ValueError(f"Invalid robot_type: {self.config.robot_type!r}")

    def reset(self) -> None:
        """Reset the policy state."""
        self.action_buffer = []
        self.buffer_index = 0
        logger.debug("Policy reset")

    def _run_action_infer(
        self,
        observation: Dict,
        instruction: str,
        mode: str,
        ar_do_sample=None,
        ar_temperature=None,
        rtc_kwargs=None,
    ) -> Dict[str, Any]:
        """Run action inference for ``mode`` and return a model_output with robot_state_action_data.

        Supported modes: flow | ar | flow_with_subtask | flow_rtc | dllm |
        discrete_diffusion. ar_do_sample / ar_temperature default to the policy
        settings (greedy) when None. rtc_kwargs (flow_rtc only) carries
        prev_action_chunk (already relative+6D), inference_delay, and an
        optional num_steps.
        """
        if mode == INFER_MODE_FLOW:
            return self.model_wrapper.infer_flow_action(observation, instruction)
        if mode == INFER_MODE_FLOW_RTC:
            if rtc_kwargs is None:
                raise ValueError("flow_rtc requires rtc_kwargs")
            return self.model_wrapper.infer_flow_action_rtc(
                observation,
                instruction,
                prev_action_chunk=rtc_kwargs["prev_action_chunk"],
                inference_delay=rtc_kwargs["inference_delay"],
                num_steps=rtc_kwargs.get("num_steps"),
            )
        if mode == INFER_MODE_AR:
            return self.model_wrapper.infer_ar_action(
                observation,
                instruction,
                do_sample=(
                    self.ar_do_sample if ar_do_sample is None else bool(ar_do_sample)
                ),
                temperature=(
                    self.ar_temperature
                    if ar_temperature is None
                    else float(ar_temperature)
                ),
            )
        if mode == INFER_MODE_FLOW_WITH_SUBTASK:
            with ScopeTimer("infer_subtask"):
                subtask = self.model_wrapper.infer_subtask(observation, instruction)
            with ScopeTimer("infer_flow_action"):
                model_output = self.model_wrapper.infer_flow_action(
                    observation, subtask
                )
            model_output["subtask"] = subtask
            return model_output
        if mode == INFER_MODE_DLLM_FLOW:
            return self.model_wrapper.infer_dllm_action(
                observation, instruction, use_ar_action=False
            )
        if mode == INFER_MODE_DLLM_DD:
            return self.model_wrapper.infer_dllm_action(
                observation, instruction, use_ar_action=True
            )
        raise ValueError(
            f"Unsupported infer_mode={mode!r}, expected one of {ACTION_INFER_MODES}"
        )

    def _rtc_dof_segments(self) -> "rtc.DofSegments":
        """Build the DoF segment table of the server action layout (for RTC abs-to-relative conversion)."""
        from wall_x._vendor.harrix.infer.base_dataclass import dof_dims

        data_config = getattr(self.config, "data_config", None)
        keys = getattr(data_config, "predict_action_keys", None)
        if keys is None and isinstance(data_config, dict):
            keys = data_config.get("predict_action_keys")
        if not keys:
            raise RuntimeError(
                "WallXPolicy RTC: cannot find predict_action_keys in "
                f"config.data_config (type={type(data_config).__name__})"
            )
        train_cfg = getattr(self.config, "train_config", {}) or {}
        train_dof = train_cfg.get("dof_config") or (
            train_cfg.get("task", {}) or {}
        ).get("dof_config", {})
        return rtc.dof_segments_from_keys(list(keys), dof_dims, train_dof)

    def _parse_rtc_obs(self, obs: Dict):
        """Parse RTC fields; return None (standard flow) when RTC is off or fields are missing."""
        try:
            action_horizon = int(self.config.action_horizon)
        except (AttributeError, TypeError, ValueError):
            action_horizon = 0
        max_rtc_steps = rtc.get_training_max_rtc_steps(
            getattr(self.config, "train_config", {}) or {},
            getattr(self.config, "model_config", None),
        )
        return rtc.parse_rtc_obs(
            obs, action_horizon=action_horizon, max_rtc_steps=max_rtc_steps
        )

    def infer(self, obs: Dict) -> Dict:
        """Infer action from observation.

        Args:
            obs: Dictionary containing:
                - 'state': Robot state
                - 'views': Camera views (keyed by camera name)
                - 'instruction': Task instruction
                - Optional: 'infer_mode' — one of 'flow' | 'ar' | 'flow_with_subtask'
                - Optional: 'robot_action_start_ratio' / 'robot_action_end_ratio' /
                  'robot_action_interpolate_multiplier' — override config for action trim/interpolate

        Returns:
            When serialize_actions=True (default): Serialized action dict
                (e.g. follow1_pos/follow2_pos or follow1_joints/follow2_joints).
            When serialize_actions=False: Raw model_output dict.
            If infer_mode is flow_with_subtask, also includes 'subtask'.
        """
        import time

        wall_clock_start = time.perf_counter()

        rtc_kwargs = self._parse_rtc_obs(obs)
        if rtc_kwargs is not None:
            obs_state = obs.get("state", {}) or {}
            f1 = np.asarray(
                obs_state.get("follow1_pos"), dtype=np.float32
            ).reshape(-1)[:7]
            f2 = np.asarray(
                obs_state.get("follow2_pos"), dtype=np.float32
            ).reshape(-1)[:7]
            sensor_state = np.concatenate([f1, f2])  # [14]
            with ScopeTimer("rtc_transform"):
                rel_chunk, state_override = rtc.transform_abs_chunk_to_relative(
                    rtc_kwargs["prev_action_chunk"],
                    sensor_state,
                    self._rtc_dof_segments(),
                )
            rtc_kwargs["prev_action_chunk"] = rel_chunk
            obs = rtc.apply_state_override(obs, state_override)
            logger.info(
                "RTC enabled: prev_chunk_len=%d, delay=%d, ref_source=%s",
                rel_chunk.shape[0],
                rtc_kwargs["inference_delay"],
                "chunk[0]" if rtc.USE_CHUNK_STATE_ENV else "sensor",
            )

        state = obs["state"]
        views = obs["views"]
        instruction = self._resolve_instruction(obs)
        if self.image_passing_mode == "base64":
            for k, v in views.items():
                img_bytes = base64.b64decode(v)
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                decoded_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                decoded_img = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)
                views[k] = np.expand_dims(decoded_img, axis=0)
        else:
            # Robot preprocessors consume batched ``(1, H, W, C)`` views and
            # select the leading sample.  The raw-numpy WebSocket adapter sends
            # ``(H, W, C)``; without this expansion ``views[k][0]`` becomes one
            # image row, producing a severely distorted vision input.
            for k, image in views.items():
                if isinstance(image, np.ndarray) and image.ndim == 3:
                    views[k] = np.expand_dims(image, axis=0)

        with ScopeTimer("get_observation"):
            observation = self.robot_preprocessor.get_observation(state, views)

        if rtc_kwargs is not None:
            mode = INFER_MODE_FLOW_RTC
        else:
            mode = obs.get("infer_mode", self.default_infer_mode)
        # Optional per-request AR sampling override (None -> policy defaults).
        ar_do_sample = obs.get("ar_do_sample")
        ar_temperature = obs.get("ar_temperature")
        with ScopeTimer(f"infer_{mode}"):
            model_output = self._run_action_infer(
                observation,
                instruction,
                mode,
                ar_do_sample=ar_do_sample,
                ar_temperature=ar_temperature,
                rtc_kwargs=rtc_kwargs,
            )

        if not self.serialize_actions:
            return model_output

        serialized = self.robot_preprocessor.get_serialized_actions(
            model_output, robot_action_interpolate_multiplier=1
        )
        infer_ms = (time.perf_counter() - wall_clock_start) * 1000.0
        return rtc.attach_rtc_outputs(serialized, infer_ms=infer_ms)

    # ── Batch inference ──────────────────────────────────────────────

    def _preprocess_obs(self, obs: Dict):
        """Preprocess a single observation dict into (observation, instruction).

        Handles both base64 and raw image modes.
        """
        state = obs["state"]
        views = obs["views"]
        instruction = self._resolve_instruction(obs)

        if self.image_passing_mode == "base64":
            import cv2

            for k, v in views.items():
                img_bytes = base64.b64decode(v)
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                decoded_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                decoded_img = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)
                views[k] = np.expand_dims(decoded_img, axis=0)
        else:
            # raw numpy: ensure (1, H, W, C)
            for k, img in views.items():
                if isinstance(img, np.ndarray) and img.ndim == 3:
                    views[k] = np.expand_dims(img, axis=0)

        observation = self.robot_preprocessor.get_observation(state, views)
        return observation, instruction

    def infer_batch(
        self, obs_list: List[Dict[str, Any]], skip_serialize: bool = False
    ) -> List[Dict[str, Any]]:
        """Perform batch inference.

        Args:
            obs_list: List of observations, each containing:
                - "views": dict of camera images
                - "state": robot state dict or array
                - "instruction": text instruction

        Returns:
            List of action dicts.
        """
        batch_size = len(obs_list)
        logger.info(f"WallXPolicy.infer_batch: processing {batch_size} observations")

        # Route RTC regardless of skip_serialize: the plain-flow batch path
        # would degrade it (and, under skip_serialize warmup, would prime the
        # wrong CUDA-graph bucket). Honor skip_serialize's discard contract by
        # returning empty dicts after running infer() for its side effects.
        if any(o.get("use_rtc") for o in obs_list):
            logger.info("WallXPolicy.infer_batch: RTC request, delegating to per-request infer()")
            outs = [self.infer(o) for o in obs_list]
            return [{}] * len(outs) if skip_serialize else outs

        try:
            observations = []
            instructions = []
            for obs in obs_list:
                observation, instruction = self._preprocess_obs(obs)
                observations.append(observation)
                instructions.append(instruction)

            mode = obs_list[0].get("infer_mode", self.default_infer_mode)
            # Batch-level AR sampling override (from the first obs; None -> defaults).
            ar_do_sample = obs_list[0].get("ar_do_sample")
            ar_temperature = obs_list[0].get("ar_temperature")

            with torch.no_grad():
                if mode == INFER_MODE_FLOW:
                    model_outputs = self.model_wrapper.infer_flow_action_batch(
                        observations, instructions
                    )
                else:
                    # AR / flow_with_subtask: fall back to per-sample inference
                    model_outputs = []
                    for obs_dict, instruction in zip(observations, instructions):
                        output = self._run_action_infer(
                            obs_dict,
                            instruction,
                            mode,
                            ar_do_sample=ar_do_sample,
                            ar_temperature=ar_temperature,
                        )
                        model_outputs.append(output)

            if skip_serialize:
                return [{}] * len(model_outputs)

            results = []
            for model_output in model_outputs:
                if self.serialize_actions:
                    action = self.robot_preprocessor.get_serialized_actions(
                        model_output, robot_action_interpolate_multiplier=1
                    )
                    # Attach the RTC feedback field (action_pred) exactly like the
                    # single infer() path. The server runs mode=batched, so real
                    # robots always hit infer_batch; without action_pred the RTC
                    # async client (DesktopRTCAsyncClient) never seeds its
                    # _prev_action_pred, so it never sends use_rtc=True and RTC
                    # silently never starts (deadlock). end_pose only; joints is
                    # skipped inside attach_rtc_outputs.
                    rtc.attach_rtc_outputs(action)
                    results.append(action)
                else:
                    predict_action = model_output["predict_action"]
                    if isinstance(predict_action, torch.Tensor):
                        predict_action = predict_action.cpu().numpy()
                    results.append({"action": predict_action})

            return results

        except Exception as e:
            logger.error(f"Batch inference failed: {e}", exc_info=True)
            return [{"action": {}, "error": str(e)} for _ in obs_list]

    @property
    def metadata(self) -> Dict[str, Any]:
        return {"batch_enabled": True, "model": "wall-x"}
