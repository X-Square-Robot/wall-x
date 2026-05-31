import base64
import logging
from typing import Dict, Any, List

import cv2
import numpy as np
import torch
from wall_x._vendor.harrix.serving.websocket_policy_server import BasePolicy
from wall_x._vendor.harrix.serving._wallx_infer.infer_config import InferConfig
from wall_x._vendor.harrix.serving._wallx_infer.model_wrapper import WallxModelWrapper
from wall_x._vendor.harrix.serving._wallx_infer.robot import (
    DesktopRobotPreprocessor,
    TurtleRobotPreprocessor,
    EX001RobotPreprocessor,
)
from wall_x._vendor.harrix.serving.policy._smoothing import apply_smoothing
from wall_x.utils.timers import ScopeTimer

logger = logging.getLogger(__name__)

# Inference modes aligned with model_wrapper; extend for vqa / batch as needed.
INFER_MODE_FLOW = "flow"
INFER_MODE_AR = "ar"
INFER_MODE_FLOW_WITH_SUBTASK = "flow_with_subtask"
INFER_MODE_DLLM_FLOW = "dllm"
INFER_MODE_DLLM_DD = "discrete_diffusion"
ACTION_INFER_MODES = (
    INFER_MODE_FLOW,
    INFER_MODE_AR,
    INFER_MODE_FLOW_WITH_SUBTASK,
    INFER_MODE_DLLM_FLOW,
    INFER_MODE_DLLM_DD,
)


class WallXPolicy(BasePolicy):
    """Policy wrapper for Wall-X model that implements the BasePolicy interface."""

    def __init__(
        self,
        config: InferConfig,
        image_passing_mode: str = "base64",
        default_infer_mode: str = INFER_MODE_FLOW,
        serialize_actions: bool = True,
    ):
        """Initialize the Wall-X policy.

        Args:
            config: Inference configuration dataclass.
            image_passing_mode: How images are passed from client ('base64' or 'numpy').
            default_infer_mode: Default inference mode ('flow', 'ar', 'flow_with_subtask', etc.).
            serialize_actions: If True, actions are serialized via robot_preprocessor;
                if False, raw model output is returned directly.
        """
        self.config = config
        self.model_wrapper = WallxModelWrapper(config)
        self.robot_preprocessor = self._register_robot_preprocessor()
        self.image_passing_mode = image_passing_mode
        self.default_infer_mode = default_infer_mode
        self.serialize_actions = serialize_actions
        logger.info(
            "Image passing mode: %s, robot_type: %s, default_infer_mode: %s, "
            "serialize_actions: %s, smooth_action: %s, smooth_gripper: %s",
            self.image_passing_mode,
            config.robot_type,
            self.default_infer_mode,
            self.serialize_actions,
            config.smooth_action,
            config.smooth_gripper,
        )

    def _register_robot_preprocessor(self):
        """Select preprocessor by config.robot_type (mirrors env._register_robot)."""
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

    def _get_dof_config(self) -> dict:
        train_config = self.config.train_config or {}
        return (
            train_config.get("dof_config")
            or train_config.get("task", {}).get("dof_config", {})
            or {}
        )

    def _is_single_arm_right_only(self) -> bool:
        """True for single-arm LIBERO-style configs (right arm only, no left arm)."""
        train_config = self.config.train_config or {}
        agent_pos = (
            train_config.get("agent_pos_config")
            or train_config.get("task", {}).get("agent_pos_config")
            or {}
        )
        skip = {"action_padding"}
        has_left = any(k.startswith("follow_left_") for k in agent_pos if k not in skip)
        has_right = any(
            k.startswith("follow_right_") for k in agent_pos if k not in skip
        )
        return has_right and not has_left

    def _pack_action_chunk_response(
        self, model_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Return a msgpack-safe action chunk for websocket clients."""
        predict_action = model_output["predict_action"]
        if isinstance(predict_action, torch.Tensor):
            predict_action = predict_action.detach().cpu().numpy()
        response: Dict[str, Any] = {
            "predict_action": np.asarray(predict_action, dtype=np.float32),
        }
        if "subtask" in model_output:
            response["subtask"] = model_output["subtask"]
        return response

    def _get_predict_action_keys(self) -> List[str]:
        """Resolve flat action key order for smoothing / serialization."""
        data_cfg = self.config.data_config
        keys = None
        if isinstance(data_cfg, dict):
            keys = data_cfg.get("predict_action_keys")
        else:
            keys = getattr(data_cfg, "predict_action_keys", None)
        if keys:
            return list(keys)
        return list(self._get_dof_config().keys())

    def _apply_smoothing(self, model_output: Dict[str, Any]) -> None:
        """Apply Laplacian smoothing on predict_action before 6D->euler conversion."""
        cfg = self.config
        if not getattr(cfg, "smooth_action", False):
            return
        dof_config = self._get_dof_config()
        apply_smoothing(
            model_output,
            smooth_action=True,
            smooth_gripper=cfg.smooth_gripper,
            predict_action_keys=self._get_predict_action_keys(),
            action_padding_dof=dof_config.get("action_padding"),
            action_dim=cfg.action_dim,
        )

    def _run_action_infer(
        self, observation: Dict, instruction: str, mode: str
    ) -> Dict[str, Any]:
        """Run action inference via model_wrapper; returns model_output with robot_state_action_data.

        Supports mode: flow | ar | flow_with_subtask. VQA can be added here later (different return shape).
        """
        if mode == INFER_MODE_FLOW:
            return self.model_wrapper.infer_flow_action(observation, instruction)
        if mode == INFER_MODE_AR:
            return self.model_wrapper.infer_ar_action(observation, instruction)
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
        state = obs["state"]
        views = obs["views"]
        instruction = obs.get("instruction") or self.config.default_instruction or ""
        if self.image_passing_mode == "base64":
            for k, v in views.items():
                img_bytes = base64.b64decode(v)
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                decoded_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                decoded_img = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)
                views[k] = np.expand_dims(decoded_img, axis=0)

        with ScopeTimer("get_observation"):
            observation = self.robot_preprocessor.get_observation(state, views)

        mode = obs.get("infer_mode", self.default_infer_mode)
        with ScopeTimer(f"infer_{mode}"):
            model_output = self._run_action_infer(observation, instruction, mode)

        self._apply_smoothing(model_output)

        # Single-arm LIBERO has no left-arm EE to serialize; dual-arm follow1/follow2
        # layout would fail in get_serialized_actions. Return raw action chunks instead.
        if not self.serialize_actions or self._is_single_arm_right_only():
            return self._pack_action_chunk_response(model_output)

        return self.robot_preprocessor.get_serialized_actions(
            model_output, robot_action_interpolate_multiplier=1
        )  # interpolation is done on the websocket robot client

    # ── Batch inference ──────────────────────────────────────────────

    def _preprocess_obs(self, obs: Dict):
        """Preprocess a single observation dict into (observation, instruction).

        Handles both base64 and raw image modes.
        """
        state = obs["state"]
        views = obs["views"]
        instruction = obs.get("instruction") or self.config.default_instruction or ""

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

        try:
            observations = []
            instructions = []
            for obs in obs_list:
                observation, instruction = self._preprocess_obs(obs)
                observations.append(observation)
                instructions.append(instruction)

            mode = obs_list[0].get("infer_mode", self.default_infer_mode)

            with torch.no_grad():
                if mode == INFER_MODE_FLOW:
                    model_outputs = self.model_wrapper.infer_flow_action_batch(
                        observations, instructions
                    )
                else:
                    # AR / flow_with_subtask: fall back to per-sample inference
                    model_outputs = []
                    for obs_dict, instruction in zip(observations, instructions):
                        output = self._run_action_infer(obs_dict, instruction, mode)
                        model_outputs.append(output)

            if skip_serialize:
                return [{}] * len(model_outputs)

            results = []
            for model_output in model_outputs:
                self._apply_smoothing(model_output)
                if self.serialize_actions and not self._is_single_arm_right_only():
                    action = self.robot_preprocessor.get_serialized_actions(
                        model_output, robot_action_interpolate_multiplier=1
                    )
                    results.append(action)
                else:
                    results.append(self._pack_action_chunk_response(model_output))

            return results

        except Exception as e:
            logger.error(f"Batch inference failed: {e}", exc_info=True)
            return [{"action": {}, "error": str(e)} for _ in obs_list]

    @property
    def metadata(self) -> Dict[str, Any]:
        return {"batch_enabled": True, "model": "wall-x"}
