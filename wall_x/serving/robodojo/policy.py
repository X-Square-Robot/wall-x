"""Wall-X flow policy behind the RoboDojo (XPolicyLab) policy interface.

RoboDojo observations carry three RGB cameras under ``vision`` and, per arm,
an end-effector pose ``[x, y, z, qw, qx, qy, qz]`` plus a one-value gripper
state under ``state``. Actions go back in the same layout, one dict per
simulator step.

End-effector rotations reach the model as row-major 6D (the first two rows of
the rotation matrix). Relative actions are composed with the observed state
in the frame named by the checkpoint's ``relative_action_frame``
(``base`` or ``body``).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from wall_x.serving.robodojo.protocol.exceptions import ErrorCode, WsError

logger = logging.getLogger(__name__)

CAMERAS = {
    "face_view": ("cam_head", "cam_high", "head_camera", "camera_front"),
    "left_wrist_view": ("cam_left_wrist", "left_camera", "camera_left"),
    "right_wrist_view": ("cam_right_wrist", "right_camera", "camera_right"),
}
IMAGE_FIELDS = ("color", "rgb", "image", "image_rgb")
SIDES = ("left", "right")
# Non-arm action dimensions that RoboDojo arms never execute.
MASKED_ACTION_KEYS = ("head_actions", "height", "velocity_decomposed")


def _invalid(message: str):
    raise WsError(ErrorCode.INVALID_FRAME, message)


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        _invalid("end-effector quaternion must be nonzero")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Row-major 6D -> rotation matrix via Gram-Schmidt on the two rows."""
    r = np.asarray(rot6d, dtype=np.float64).reshape(6)
    r1 = r[:3] / (np.linalg.norm(r[:3]) + 1e-12)
    r2 = r[3:] - np.dot(r1, r[3:]) * r1
    r2 = r2 / (np.linalg.norm(r2) + 1e-12)
    return np.stack([r1, r2, np.cross(r1, r2)])


def matrix_to_quat_wxyz(m: np.ndarray) -> np.ndarray:
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12)) * 2.0
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12)) * 2.0
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12)) * 2.0
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def _image(value: Any, camera: str) -> np.ndarray:
    if isinstance(value, Mapping):
        for name in IMAGE_FIELDS:
            if name in value:
                value = value[name]
                break
        else:
            _invalid(f"camera {camera!r} has none of {IMAGE_FIELDS}")
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8:
        _invalid(f"camera {camera!r} must be a uint8 ndarray")
    if value.ndim != 3 or value.shape[2] not in (3, 4):
        _invalid(f"camera {camera!r} must have shape [H, W, 3]")
    return np.ascontiguousarray(value[..., :3])


def observation_views(obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    vision = obs.get("vision")
    if not isinstance(vision, Mapping):
        _invalid("observation has no 'vision' mapping")
    views = {}
    for view, candidates in CAMERAS.items():
        for name in candidates:
            if name in vision:
                views[view] = _image(vision[name], name)
                break
        else:
            _invalid(f"missing camera for {view}; available: {sorted(vision)}")
    return views


def observation_arms(obs: Mapping[str, Any]) -> dict[str, tuple[np.ndarray, float]]:
    state = obs.get("state", obs)
    if not isinstance(state, Mapping):
        _invalid("observation 'state' must be a mapping")
    arms = {}
    for side in SIDES:
        try:
            pose = np.asarray(state[f"{side}_ee_pose"], dtype=np.float64).reshape(-1)
            grip = np.asarray(state[f"{side}_ee_joint_state"], dtype=np.float64).reshape(-1)
        except (KeyError, TypeError, ValueError) as exc:
            _invalid(f"invalid {side} end-effector state: {exc}")
        if pose.shape != (7,) or not np.isfinite(pose).all():
            _invalid(f"{side}_ee_pose must be 7 finite values (xyz + wxyz)")
        if grip.size != 1 or not np.isfinite(grip).all():
            _invalid(f"{side}_ee_joint_state must be one finite value")
        arms[side] = (pose, float(grip[0]))
    return arms


@dataclass
class RoboDojoPolicyConfig:
    checkpoint_path: str
    train_config_path: str | None = None
    norm_key: str = "ex_normal"
    model_device: str = "cuda"
    robot_type: str = "desktop"
    action_horizon: int | None = None
    num_inference_timesteps: int = 10
    # Returned chunk = predicted steps [action_start_index, + action_chunk_length).
    # ``None`` means replan_steps for the plain policy and the full horizon for
    # keyframe-memory checkpoints (the platform wants the complete chunk).
    action_start_index: int = 0
    replan_steps: int = 16
    action_chunk_length: int | None = None
    default_instruction: str = ""


def resolve_action_steps(
    config: RoboDojoPolicyConfig, horizon: int, *, full_chunk: bool
) -> int:
    """How many predicted steps a request returns.

    The platform asks for the whole checkpoint chunk (``len(actions) ==
    action_chunk_length``) and advances the policy by ``replan_steps``. Keeping
    the two apart is what lets a memory policy return 32 actions and move 16 —
    the same split the in-house service serves with.
    """
    available = int(horizon) - int(config.action_start_index)
    steps = config.action_chunk_length
    if steps is None:
        steps = available if full_chunk else int(config.replan_steps)
    steps = int(steps)
    if steps < 1:
        raise ValueError("action chunk length must be positive")
    if steps > available:
        raise ValueError(
            f"action chunk length {steps} exceeds the {available} predicted "
            f"steps left after action_start_index {config.action_start_index}"
        )
    if config.replan_steps > steps:
        raise ValueError(
            f"replan_steps ({config.replan_steps}) must not exceed the returned "
            f"action chunk ({steps})"
        )
    return steps


def build_infer_config(config: RoboDojoPolicyConfig):
    """Load the checkpoint's inference config (configs only, no weights)."""
    from wall_x._vendor.harrix.infer.infer_config import InferConfig

    infer_kwargs = dict(
        checkpoint_path=config.checkpoint_path,
        train_config_path=config.train_config_path,
        robot_type=config.robot_type,
        norm_key=config.norm_key,
        model_device=config.model_device,
        num_inference_timesteps=config.num_inference_timesteps,
    )
    if config.action_horizon is not None:
        infer_kwargs["action_horizon"] = config.action_horizon
    return InferConfig(**infer_kwargs)


class WallXRoboDojoPolicy:
    """Stateless flow policy: each request is answered from its own observation."""

    #: Plain policy keeps its published convention: one replan's worth of actions.
    returns_full_chunk = False

    def __init__(self, config: RoboDojoPolicyConfig, infer_config=None):
        from wall_x._vendor.harrix.utils.dataset_v2 import (
            resolve_relative_action_frame,
        )

        self.config = config
        self.infer_config = infer_config or build_infer_config(config)
        horizon = int(self.infer_config.action_horizon)
        if not 0 <= config.action_start_index < horizon:
            raise ValueError(f"action_start_index must be in [0, {horizon})")
        if config.replan_steps < 1:
            raise ValueError("replan_steps must be positive")
        self.action_steps = resolve_action_steps(
            config, horizon, full_chunk=self.returns_full_chunk
        )
        self.relative_action_frame = resolve_relative_action_frame(
            self.infer_config.data_config
        )
        self.dof_mask = self._build_dof_mask()
        self.model = self._build_model(self.infer_config)
        self._pending = None
        self._pending_batch = None
        logger.info(
            "RoboDojo policy ready: horizon=%d start=%d chunk=%d replan=%d "
            "frame=%s norm_key=%s",
            horizon,
            config.action_start_index,
            self.action_steps,
            config.replan_steps,
            self.relative_action_frame,
            config.norm_key,
        )

    def _build_model(self, infer_config):
        from wall_x._vendor.harrix.infer.model_wrapper import WallxModelWrapper

        return WallxModelWrapper(infer_config)

    # ------------------------------------------------------------------
    # XPolicyLab model interface
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._pending = None
        self._pending_batch = None

    def update_obs(self, obs: Mapping[str, Any]) -> None:
        self._pending = self._convert(obs)

    def get_action(self) -> list[dict[str, np.ndarray]]:
        if self._pending is None:
            raise RuntimeError("get_action called before update_obs")
        observation, instruction = self._pending
        output = self.model.infer_flow_action(observation, instruction)
        return self._actions(output["robot_state_action_data"])

    def update_obs_batch(self, observations: list[Mapping[str, Any]]) -> None:
        self._pending_batch = [self._convert(obs) for obs in observations]

    def get_action_batch(self, env_ids=None) -> list[list[dict[str, np.ndarray]]]:
        if not self._pending_batch:
            raise RuntimeError("get_action_batch called before update_obs_batch")
        observations = [observation for observation, _ in self._pending_batch]
        instructions = [instruction for _, instruction in self._pending_batch]
        outputs = self.model.infer_flow_action_batch(observations, instructions)
        return [self._actions(out["robot_state_action_data"]) for out in outputs]

    # ------------------------------------------------------------------
    # Observation / action conversion
    # ------------------------------------------------------------------
    def _build_dof_mask(self) -> np.ndarray:
        dof_config = self.infer_config.train_config["dof_config"]
        mask = np.ones(
            (1, int(self.infer_config.action_horizon), sum(dof_config.values()))
        )
        start = 0
        for key, size in dof_config.items():
            if key in MASKED_ACTION_KEYS:
                mask[:, :, start : start + size] = 0
            start += size
        return mask

    def _convert(self, obs: Mapping[str, Any]):
        from wall_x._vendor.harrix.infer.base_dataclass import RobotStateActionData

        if not isinstance(obs, Mapping):
            _invalid("observation must be a mapping")
        views = observation_views(obs)
        arms = observation_arms(obs)
        instruction = obs.get("instruction") or self.config.default_instruction
        if not isinstance(instruction, str) or not instruction.strip():
            _invalid("observation needs a nonempty 'instruction'")

        data = RobotStateActionData(config=self.infer_config)
        for side, (pose, grip) in arms.items():
            rot6d = quat_wxyz_to_matrix(pose[3:])[:2].reshape(6)
            data.save_state_data_with_key(
                pose[None, :3].astype(np.float32), f"follow_{side}_ee_cartesian_pos"
            )
            data.save_state_data_with_key(
                rot6d[None].astype(np.float32), f"follow_{side}_ee_rotation_6D"
            )
            data.save_state_data_with_key(
                np.array([[grip]], dtype=np.float32), f"follow_{side}_gripper"
            )
        data.dof_mask = self.dof_mask
        return {**views, "robot_state_action_data": data}, instruction

    def _absolute_arm(self, data, side: str) -> tuple[np.ndarray, np.ndarray]:
        from wall_x._vendor.harrix.infer import body_frame
        from wall_x._vendor.x2robot_utils import geometry

        values = data.data
        state_pos = np.asarray(values[f"state_{side}_ee_cartesian_pos"], dtype=np.float64)
        state_rot = np.asarray(values[f"state_{side}_ee_rotation_6D"], dtype=np.float64)
        rel_pos = dict.get(values, f"action_{side}_ee_cartesian_pos_relative")
        rel_rot = dict.get(values, f"action_{side}_ee_rotation_6D_relative")
        if rel_pos is not None and rel_rot is not None:
            rel_pos = np.asarray(rel_pos, dtype=np.float64).reshape(-1, 3)
            rel_rot = np.asarray(rel_rot, dtype=np.float64).reshape(-1, 6)
            if self.relative_action_frame == "body":
                pos = body_frame.compose_body_delta_to_abs_pos(rel_pos, state_pos, state_rot)
                rot = body_frame.compose_body_delta_to_abs_6d(rel_rot, state_rot)
            else:
                pos = rel_pos + state_pos.reshape(1, 3)
                rot = geometry.compose_state_and_delta_to_abs_6d(rel_rot, state_rot[0])
            return pos, rot
        abs_pos = dict.get(values, f"action_{side}_ee_cartesian_pos")
        abs_rot = dict.get(values, f"action_{side}_ee_rotation_6D")
        if abs_pos is None or abs_rot is None:
            raise ValueError(
                f"checkpoint predicts no {side} end-effector position + 6D rotation"
            )
        return (
            np.asarray(abs_pos, dtype=np.float64).reshape(-1, 3),
            np.asarray(abs_rot, dtype=np.float64).reshape(-1, 6),
        )

    def _actions(self, data) -> list[dict[str, np.ndarray]]:
        start = self.config.action_start_index
        stop = start + self.action_steps
        per_side = {}
        for side in SIDES:
            pos, rot = self._absolute_arm(data, side)
            grip = np.asarray(data.data[f"action_{side}_gripper"], dtype=np.float64)
            quat = np.stack([matrix_to_quat_wxyz(rot6d_to_matrix(r)) for r in rot])
            pose = np.concatenate([pos, quat], axis=1)[start:stop]
            grip = grip.reshape(-1, 1)[start:stop]
            if not (np.isfinite(pose).all() and np.isfinite(grip).all()):
                raise ValueError(f"non-finite {side} action")
            per_side[side] = (pose.astype(np.float32), grip.astype(np.float32))
        return [
            {
                f"{side}_ee_pose": per_side[side][0][t]
                for side in SIDES
            }
            | {f"{side}_ee_joint_state": per_side[side][1][t] for side in SIDES}
            for t in range(len(per_side["left"][0]))
        ]
