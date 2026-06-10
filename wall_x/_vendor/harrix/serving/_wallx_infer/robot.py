import numpy as np
from abc import ABC
import re

from wall_x._vendor.harrix.serving._wallx_infer.infer_config import InferConfig
from wall_x._vendor.harrix.serving._wallx_infer.utils import VehiclePoseHandler, UnifiedTrajectoryProcessor
from wall_x._vendor.harrix.serving._wallx_infer.base_dataclass import (
    RobotStateActionData,
    dof_dims,
)
from wall_x._vendor.harrix.serving._wallx_infer.socket_controller import RobotController

from wall_x._vendor.harrix.serving._wallx_infer.logger import InferLogger

robot_action_key_mapping = {
    "follow_left_ee_cartesian_pos": "follow1_pos[:3]",
    "follow_left_ee_rotation": "follow1_pos[3:6]",
    "follow_left_gripper": "follow1_pos[6:7]",
    "follow_right_ee_cartesian_pos": "follow2_pos[:3]",
    "follow_right_ee_rotation": "follow2_pos[3:6]",
    "follow_right_gripper": "follow2_pos[6:7]",
    "head_actions": "head_pos",
    "head_rotation": "head_pos",  # match ex001
    "height": "lift",
    "velocity_decomposed": "velocity_decomposed",
    "velocity_decomposed_odom": "velocity_decomposed_odom",  # match ex001
    "follow_left_arm_joint_cur": "follow1_joints_cur",  # 1 -> 7
    "follow_right_arm_joint_cur": "follow2_joints_cur",  # 1 -> 7
    "follow_left_gripper_cur": "follow1_joints_cur[-1:]",
    "follow_right_gripper_cur": "follow2_joints_cur[-1:]",
    "follow_left_arm_joint_pos": "follow1_joints",
    "follow_right_arm_joint_pos": "follow2_joints",
    "follow_left_wrench_ext_local_force": "follow1_end_effort_force",
    "follow_left_wrench_ext_local_torque": "follow1_end_effort_torque",
    "follow_right_wrench_ext_local_force": "follow2_end_effort_force",
    "follow_right_wrench_ext_local_torque": "follow2_end_effort_torque",
    "follow_left_wrench_ext_local_force_from_joint": "follow1_end_effort_force_from_joint",
    "follow_left_wrench_ext_local_torque_from_joint": "follow1_end_effort_torque_from_joint",
    "follow_right_wrench_ext_local_force_from_joint": "follow2_end_effort_force_from_joint",
    "follow_right_wrench_ext_local_torque_from_joint": "follow2_end_effort_torque_from_joint",
    "follow_left_wrench_ext_world_force": "follow1_wrench_ext_world_force",
    "follow_left_wrench_ext_world_torque": "follow1_wrench_ext_world_torque",
    "follow_left_wrench_ext_world_force_from_joint": "follow1_wrench_ext_world_force_from_joint",
    "follow_left_wrench_ext_world_torque_from_joint": "follow1_wrench_ext_world_torque_from_joint",
    "follow_right_wrench_ext_world_force": "follow2_wrench_ext_world_force",
    "follow_right_wrench_ext_world_torque": "follow2_wrench_ext_world_torque",
    "follow_right_wrench_ext_world_force_from_joint": "follow2_wrench_ext_world_force_from_joint",
    "follow_right_wrench_ext_world_torque_from_joint": "follow2_wrench_ext_world_torque_from_joint",
    "follow_left_arm_joint_dev": "follow1_joints_dev",
    "follow_right_arm_joint_dev": "follow2_joints_dev",
}


def _parse_follow_pos(follow_pos) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse websocket ``follow{1,2}_pos``: pos3 + euler3 + one or more gripper dims."""
    arr = np.asarray(follow_pos, dtype=np.float64).reshape(-1)
    if arr.shape[0] < 7:
        raise ValueError(f"follow_pos expects 7 dims, got shape {arr.shape}")
    return arr[:3], arr[3:6], arr[6:]


def ingest_websocket_follow_pos(
    state: dict,
    config: InferConfig,
    robot_state_action_data: RobotStateActionData,
) -> None:
    """Convert client ``follow1_pos``/``follow2_pos`` (14D) into train-config proprio."""
    from wall_x._vendor.harrix.utils.train_config import resolve_agent_pos_config
    from wall_x._vendor.x2robot_utils import geometry as geom

    agent_cfg = resolve_agent_pos_config(config.train_config)
    arm_ws_keys = (
        ("left", "follow1_pos", "follow_left_"),
        ("right", "follow2_pos", "follow_right_"),
    )
    for _side, ws_key, cfg_prefix in arm_ws_keys:
        if ws_key not in state or state[ws_key] is None:
            continue
        pos, euler, grip = _parse_follow_pos(state[ws_key])
        rot6d = geom.euler_to_matrix_zyx_6d_nb(euler.reshape(1, 3))
        for key in agent_cfg:
            if key == "action_padding" or not key.startswith(cfg_prefix):
                continue
            if "cartesian_pos" in key:
                robot_state_action_data.save_state_data_with_key(pos, key)
            elif "rotation_6d" in key.lower():
                robot_state_action_data.save_state_data_with_key(
                    rot6d.reshape(1, -1), key
                )
            elif "rotation" in key:
                robot_state_action_data.save_state_data_with_key(euler, key)
            elif "gripper" in key:
                robot_state_action_data.save_state_data_with_key(
                    grip[: int(agent_cfg[key])], key, gt_dim=int(agent_cfg[key])
                )

    for key, dim in agent_cfg.items():
        if key == "action_padding" or key.startswith(("follow_left_", "follow_right_")):
            continue
        norm_key = key.replace("follow_", "").replace("master_", "")
        if robot_state_action_data.data.get(f"state_{norm_key}") is None:
            robot_state_action_data.save_state_data_with_key(
                np.zeros(int(dim), dtype=np.float64), key
            )


def export_websocket_follow_pos(
    robot_state_action_data: RobotStateActionData,
) -> dict[str, list]:
    """Convert model ``robot_state_action_data`` back to ``follow1_pos``/``follow2_pos``."""
    left_arm_action = _stack_state_action_series(robot_state_action_data, side="left")
    right_arm_action = _stack_state_action_series(robot_state_action_data, side="right")
    return {
        "follow1_pos": left_arm_action.tolist(),
        "follow2_pos": right_arm_action.tolist(),
    }


def _zero_state_series(key: str) -> np.ndarray:
    """Fallback proprio when websocket state omits an arm (e.g. only follow2_pos)."""
    dim = dof_dims.get(key, 1)
    return np.zeros((1, dim), dtype=np.float64)


def _identity_state_rotation_6d() -> np.ndarray:
    from wall_x._vendor.x2robot_utils import geometry as geom

    return geom.euler_to_matrix_zyx_6d_nb(np.zeros((1, 3), dtype=np.float64))


def _as_2d_series(value, name: str) -> np.ndarray:
    """Normalize state/action arrays to (T, D) for follow*_pos serialization."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1) if arr.size <= 7 else arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.ndim != 2:
        raise ValueError(f"{name}: expected 2D series after normalize, got {arr.shape}")
    return arr


def _stack_state_action_series(
    robot_state_action_data: RobotStateActionData,
    *,
    side: str,
    pos_key: str = "ee_cartesian_pos",
    rot_key: str = "ee_rotation",
    grip_key: str = "gripper",
) -> np.ndarray:
    """Build (1+horizon, 7) follow arm pose: pos3 + rot3 + grip1."""
    from wall_x._vendor.x2robot_utils import geometry as geom

    data = robot_state_action_data.data
    prefix = "left" if side == "left" else "right"

    state_pos = data.get(f"state_{prefix}_{pos_key}")
    action_pos = data.get(f"action_{prefix}_{pos_key}")
    rel_pos = data.get(f"action_{prefix}_{pos_key}_relative")
    if action_pos is None and rel_pos is not None:
        base_pos = (
            _as_2d_series(state_pos, f"state_{prefix}_{pos_key}")
            if state_pos is not None
            else _zero_state_series(f"{prefix}_{pos_key}")
        )
        action_pos = base_pos + np.asarray(rel_pos, dtype=np.float64)
    if state_pos is None:
        state_pos = _zero_state_series(f"{prefix}_{pos_key}")

    state_rot = data.get(f"state_{prefix}_{rot_key}")
    action_rot = data.get(f"action_{prefix}_{rot_key}")

    rot6d = data.get(f"action_{prefix}_{rot_key}_6D")
    if action_rot is None and rot6d is not None:
        action_rot = geom.so3_to_euler_zyx_batch_nb(np.asarray(rot6d, dtype=np.float64))
    if action_rot is None:
        rel6d = data.get(f"action_{prefix}_{rot_key}_6D_relative")
        state6d = data.get(f"state_{prefix}_{rot_key}_6D")
        if state6d is None and state_rot is not None:
            state6d = geom.euler_to_matrix_zyx_6d_nb(
                _as_2d_series(state_rot, f"state_{prefix}_{rot_key}")
            )
        if state6d is None:
            state6d = _identity_state_rotation_6d()
        if rel6d is not None:
            rot6d = geom.compose_state_and_delta_to_abs_6d(
                np.asarray(rel6d, dtype=np.float64),
                np.asarray(state6d, dtype=np.float64).reshape(-1),
            )
            action_rot = geom.so3_to_euler_zyx_batch_nb(rot6d)

    if state_rot is None:
        state6d = data.get(f"state_{prefix}_{rot_key}_6D")
        if state6d is not None:
            state_rot = geom.so3_to_euler_zyx_batch_nb(
                np.asarray(state6d, dtype=np.float64)
            )
        else:
            state_rot = _zero_state_series(f"{prefix}_{rot_key}")

    state_grip = data.get(f"state_{prefix}_{grip_key}")
    if state_grip is None:
        state_grip = _zero_state_series(f"{prefix}_{grip_key}")
    action_grip = data.get(f"action_{prefix}_{grip_key}")

    if action_pos is None or action_rot is None or action_grip is None:
        missing = [
            name
            for name, value in (
                (f"action_{prefix}_{pos_key}", action_pos),
                (f"action_{prefix}_{rot_key}", action_rot),
                (f"action_{prefix}_{grip_key}", action_grip),
            )
            if value is None
        ]
        raise ValueError(
            f"Cannot serialize {prefix} arm action; missing action fields: {missing}"
        )

    pos = np.concatenate(
        [
            _as_2d_series(state_pos, f"state_{prefix}_{pos_key}"),
            _as_2d_series(action_pos, f"action_{prefix}_{pos_key}"),
        ],
        axis=0,
    )
    rot = np.concatenate(
        [
            _as_2d_series(state_rot, f"state_{prefix}_{rot_key}"),
            _as_2d_series(action_rot, f"action_{prefix}_{rot_key}"),
        ],
        axis=0,
    )
    grip = np.concatenate(
        [
            _as_2d_series(state_grip, f"state_{prefix}_{grip_key}"),
            _as_2d_series(action_grip, f"action_{prefix}_{grip_key}"),
        ],
        axis=0,
    )
    return np.concatenate([pos, rot, grip], axis=1)


def _views_to_camera_observation(
    config: InferConfig, views: dict
) -> dict[str, np.ndarray]:
    """Map websocket camera keys to model observation keys.

    Only cameras listed in ``config.cam_names`` are required (e.g. LIBERO uses
    face_view + right_wrist_view without left_wrist_view).
    """
    camera_mappings = {
        "face_view": [config.camera_front_key, "face_view", "front_view"],
        "left_wrist_view": [
            config.camera_left_key,
            "left_wrist_view",
            "left_view",
        ],
        "right_wrist_view": [
            config.camera_right_key,
            "right_wrist_view",
            "right_view",
        ],
    }
    observation: dict[str, np.ndarray] = {}
    for obs_key in config.cam_names:
        possible_keys = camera_mappings.get(obs_key, [obs_key])
        for view_key in possible_keys:
            if view_key in views and views[view_key] is not None:
                view_data = views[view_key]
                if (
                    isinstance(view_data, np.ndarray)
                    and view_data.ndim == 4
                    and view_data.shape[0] == 1
                ):
                    observation[obs_key] = view_data[0]
                else:
                    observation[obs_key] = view_data
                break
        else:
            raise KeyError(f"Missing view for {obs_key!r}, tried keys: {possible_keys}")
    return observation


class Robot(ABC):
    def __init__(self, config: InferConfig, robot_id=10) -> None:
        self.config = config

        self.robot_controller = RobotController(
            robot_id=robot_id, host=config.robot_host, port=config.robot_port
        )

        self.robot_controller.connect()

        self.is_received = False  # blocking flag
        self.current_robot_state_action = None

        self.logger = InferLogger.get_robot_logger("Robot")

    def _get_views_and_state(self):
        state = self.robot_controller.recv_action()
        views = self.robot_controller.recv_image(
            [
                self.config.camera_left_key,
                self.config.camera_front_key,
                self.config.camera_right_key,
            ]
        )
        self.is_received = True
        return state, views

    def _get_dof_mask(self):
        dof_config = self.config.train_config["dof_config"]
        total_dof = sum(dof_config.values())
        dof_mask = np.ones((1, self.config.action_horizon, total_dof))
        return dof_mask

    @staticmethod
    def _try_compose_arm_joint_pos_with_gripper(state: dict, key: str):
        """Build 7D arm joint position from runtime state when possible.

        Why this is needed:
        - Joint-control training expects `*_arm_joint_pos` to be 7D
          (6 arm joints + 1 gripper).
        - Some online robot states expose arm joints as 6D (`follow*_joints`)
          and gripper separately in `follow*_pos[6]`.
        - If we pass raw 6D joints directly, `save_state_data_with_key` will fail
          shape validation against dof_dims=7.

        Behavior:
        - Only handles `follow_left_arm_joint_pos` / `follow_right_arm_joint_pos`.
        - Returns a composed 7D vector when both required sources exist and shapes
          match expected runtime format.
        - Returns None for all other keys or incompatible state shapes, so caller
          falls back to original mapping logic.
        """

        if key == "follow_left_arm_joint_pos":
            joints_key = "follow1_joints"
            ee_key = "follow1_pos"
        elif key == "follow_right_arm_joint_pos":
            joints_key = "follow2_joints"
            ee_key = "follow2_pos"
        else:
            return None

        if joints_key not in state or ee_key not in state:
            return None

        joints = np.asarray(state[joints_key]).reshape(-1)
        ee = np.asarray(state[ee_key]).reshape(-1)
        if joints.shape[0] == 6 and ee.shape[0] >= 7:
            return np.concatenate([joints, ee[6:7]], axis=0)

        return None

    def get_observation(self):
        state, views = self._get_views_and_state()

        robot_state_action_data = RobotStateActionData(config=self.config)
        for key in robot_action_key_mapping.keys():
            state_key_str = robot_action_key_mapping[key]
            value = None
            # Prefer a robust 7D composition path for arm_joint_pos:
            # online state often provides `follow*_joints` as 6D plus gripper
            # in `follow*_pos[6]`. Compose first to match model/dof schema.
            composed_value = self._try_compose_arm_joint_pos_with_gripper(state, key)
            if composed_value is not None:
                value = composed_value
                robot_state_action_data.save_state_data_with_key(
                    np.asarray(value)[None], key
                )
                continue

            match = re.match(r"(\w+)\[(.*)\]", state_key_str)
            if match:
                base_key = match.group(1)
                if base_key in state:
                    slicing_str = match.group(2)
                    slice_parts = slicing_str.split(":")
                    slice_args = [(int(p) if p.strip() else None) for p in slice_parts]
                    s = slice(*slice_args)
                    value = state[base_key][s]
            else:
                if state_key_str in state:
                    value = state[state_key_str]
            if value is not None:
                robot_state_action_data.save_state_data_with_key(
                    np.asarray(value)[None], key
                )

        dof_mask = self._get_dof_mask()
        robot_state_action_data.dof_mask = dof_mask
        self.current_robot_state_action = robot_state_action_data
        return {
            "robot_state_action_data": robot_state_action_data,
            **_views_to_camera_observation(self.config, views),
        }

    def go_home(self):
        if self.current_robot_state_action is None:
            self.get_observation()

        self.current_robot_state_action.save_action_data_with_key(
            np.zeros((1, 3)), "left_ee_cartesian_pos"
        )
        self.current_robot_state_action.save_action_data_with_key(
            np.zeros((1, 3)), "left_ee_rotation"
        )
        self.current_robot_state_action.save_action_data_with_key(
            np.zeros((1, 1)), "left_gripper"
        )
        self.current_robot_state_action.save_action_data_with_key(
            np.zeros((1, 3)), "right_ee_cartesian_pos"
        )
        self.current_robot_state_action.save_action_data_with_key(
            np.zeros((1, 3)), "right_ee_rotation"
        )
        self.current_robot_state_action.save_action_data_with_key(
            np.zeros((1, 1)), "right_gripper"
        )

        action_dict = {"robot_state_action_data": self.current_robot_state_action}
        self.apply_action(
            action_dict,
            robot_action_interpolate_multiplier=150,
            robot_action_start_ratio=0,
            robot_action_end_ratio=1,
        )
        self.logger.info("robot returned to initial pose")

    def _get_left_arm_action(
        self, robot_state_action_data: RobotStateActionData
    ) -> np.ndarray:
        if not self.config.robot_use_joint_angle_control:
            return _stack_state_action_series(robot_state_action_data, side="left")
        left_arm_joint_pos = np.concatenate(
            [
                _as_2d_series(
                    robot_state_action_data.data["state_left_arm_joint_pos"],
                    "state_left_arm_joint_pos",
                ),
                _as_2d_series(
                    robot_state_action_data.data["action_left_arm_joint_pos"],
                    "action_left_arm_joint_pos",
                ),
            ],
            axis=0,
        )
        return left_arm_joint_pos

    def _get_right_arm_action(
        self, robot_state_action_data: RobotStateActionData
    ) -> np.ndarray:
        if not self.config.robot_use_joint_angle_control:
            return _stack_state_action_series(robot_state_action_data, side="right")
        right_arm_joint_pos = np.concatenate(
            [
                _as_2d_series(
                    robot_state_action_data.data["state_right_arm_joint_pos"],
                    "state_right_arm_joint_pos",
                ),
                _as_2d_series(
                    robot_state_action_data.data["action_right_arm_joint_pos"],
                    "action_right_arm_joint_pos",
                ),
            ],
            axis=0,
        )
        return right_arm_joint_pos

    def get_serialized_actions(
        self,
        input: dict,
        robot_action_interpolate_multiplier=None,
        robot_action_start_ratio=None,
        robot_action_end_ratio=None,
    ) -> dict:
        """Build trimmed+interpolated serialized actions from model_output (with robot_state_action_data); does not send.

        Used by apply_action and serving policy; returns a dict ready for robot or client.
        """
        assert "robot_state_action_data" in input

        left_arm_action = self._get_left_arm_action(input["robot_state_action_data"])
        right_arm_action = self._get_right_arm_action(input["robot_state_action_data"])

        # Trim action
        action_length = len(left_arm_action)
        if robot_action_start_ratio is None:
            robot_action_start_ratio = self.config.robot_action_start_ratio
        if robot_action_end_ratio is None:
            robot_action_end_ratio = self.config.robot_action_end_ratio
        start_frame = int(robot_action_start_ratio * action_length)
        end_frame = int(robot_action_end_ratio * action_length)

        left_arm_action = left_arm_action[start_frame:end_frame]
        right_arm_action = right_arm_action[start_frame:end_frame]

        # Interpolate
        if robot_action_interpolate_multiplier is None:
            robot_action_interpolate_multiplier = (
                self.config.robot_action_interpolate_multiplier
            )
        target_length = robot_action_interpolate_multiplier * len(left_arm_action)
        left_arm_action, right_arm_action = (
            UnifiedTrajectoryProcessor.interpolate_trajectory_batch(
                [left_arm_action, right_arm_action], target_length
            )
        )

        if not self.config.robot_use_joint_angle_control:
            return {
                "follow1_pos": left_arm_action.tolist(),
                "follow2_pos": right_arm_action.tolist(),
            }
        return {
            "follow1_joints": left_arm_action.tolist(),
            "follow2_joints": right_arm_action.tolist(),
        }

    def apply_action(
        self,
        input: dict,
        robot_action_interpolate_multiplier=None,
        robot_action_start_ratio=None,
        robot_action_end_ratio=None,
    ) -> None:
        serialized_actions = self.get_serialized_actions(
            input,
            robot_action_interpolate_multiplier=robot_action_interpolate_multiplier,
            robot_action_start_ratio=robot_action_start_ratio,
            robot_action_end_ratio=robot_action_end_ratio,
        )
        self._send_actions(serialized_actions)

    def _send_actions(self, serialized_actions: dict) -> None:
        """Send actions to robot controller"""
        self.robot_controller.robot_comm.send_dict(serialized_actions)
        self.is_received = False
        self.current_robot_state_action = None


class DesktopRobot(Robot):
    def __init__(self, config: InferConfig, robot_id=10) -> None:
        self.is_received = False  # blocking flag
        self.current_robot_state_action = None

        self.logger = InferLogger.get_robot_logger("Robot")

    def _get_dof_mask(self):
        from wall_x._vendor.harrix.utils.train_config import resolve_dof_config

        dof_config = resolve_dof_config(self.config.train_config)
        total_dof = sum(dof_config.values())
        dof_mask = np.ones((1, self.config.action_horizon, total_dof))

        # For desktop, mask out head_actions, height, and velocity_decomposed
        mask_keys = ["head_actions", "height", "velocity_decomposed"]
        start_idx = 0
        for key, dof_size in dof_config.items():
            if key in mask_keys:
                # Set mask for these dims to 0
                dof_mask[:, :, start_idx : start_idx + dof_size] = 0
            start_idx += dof_size

        return dof_mask


class DesktopRobotVGA(Robot):
    def __init__(self, config: InferConfig, robot_id=10) -> None:
        super().__init__(config, robot_id)
        self.is_received = False  # blocking flag
        self.current_robot_state_action = None

        self.logger = InferLogger.get_robot_logger("Robot")

    def _get_dof_mask(self):
        dof_config = self.config.train_config["data"]["dof_config"]
        total_dof = sum(dof_config.values())
        dof_mask = np.ones((1, self.config.action_horizon, total_dof))

        # For desktop, mask out head_actions, height, and velocity_decomposed
        mask_keys = ["head_actions", "height", "velocity_decomposed"]
        start_idx = 0
        for key, dof_size in dof_config.items():
            if key in mask_keys:
                # Set mask for these dims to 0
                dof_mask[:, :, start_idx : start_idx + dof_size] = 0
            start_idx += dof_size

        return dof_mask


class DesktopRobotPreprocessor(DesktopRobot):
    def __init__(self, config: InferConfig, robot_id=10) -> None:
        self.config = config

        self.is_received = False  # blocking flag
        self.current_robot_state_action = None
        self.logger = InferLogger.get_robot_logger("Robot")

    def _send_actions(self, serialized_actions: dict) -> None:
        # bypass
        return

    def get_observation(self, state, views):
        robot_state_action_data = RobotStateActionData(config=self.config)
        if "follow1_pos" in state or "follow2_pos" in state:
            ingest_websocket_follow_pos(state, self.config, robot_state_action_data)
        for key in robot_action_key_mapping.keys():
            if key.startswith(("follow_left_", "follow_right_")):
                continue
            state_key_str = robot_action_key_mapping[key]
            value = None
            match = re.match(r"(\w+)\[(.*)\]", state_key_str)
            if match:
                base_key = match.group(1)
                if base_key in state:
                    slicing_str = match.group(2)
                    slice_parts = slicing_str.split(":")
                    slice_args = [(int(p) if p.strip() else None) for p in slice_parts]
                    s = slice(*slice_args)
                    value = state[base_key][s]
            else:
                if state_key_str in state:
                    value = state[state_key_str]
            if value is not None:
                robot_state_action_data.save_state_data_with_key(
                    np.asarray(value)[None], key
                )

        dof_mask = self._get_dof_mask()
        robot_state_action_data.dof_mask = dof_mask
        self.current_robot_state_action = robot_state_action_data
        return {
            "robot_state_action_data": robot_state_action_data,
            **_views_to_camera_observation(self.config, views),
        }


class TurtleRobot(Robot):
    def __init__(self, config: InferConfig, robot_id=10) -> None:
        self.vehicle_pose_handler = VehiclePoseHandler()
        self.last_speed = [0, 0, 0]

    def _get_state_from_controller(self):
        state = super()._get_state_from_controller()

        self.vehicle_pose_handler.update_pose(state["car_pose"])
        state["velocity_decomposed"] = self.last_speed

        if self.config.turtle_as_desktop:
            state["head_pos"] = [0, -1]
            state["lift"] = [0.4]

        return state

    def _calculate_car_pose(self, base_velocity_pred):
        # Vectorized integrate velocity to pose
        dt = 1 / 20
        current_pose = (
            self.vehicle_pose_handler.current_pose.copy()
            if self.vehicle_pose_handler.current_pose is not None
            else np.array([0.0, 0.0, 0.0])
        )

        # Batch integrate positions
        poses_frames = []
        for i in range(len(base_velocity_pred)):
            current_pose = self.vehicle_pose_handler.velocity_to_pose(
                base_velocity_pred[i, 0],
                base_velocity_pred[i, 1],
                base_velocity_pred[i, 2],
                dt,
                current_pose,
            )
            poses_frames.append(current_pose.copy())

        return poses_frames

    def _get_car_velocity(
        self, robot_state_action_data: RobotStateActionData
    ) -> np.ndarray:
        car_velocity = robot_state_action_data.data["action_velocity_decomposed"]
        self.last_speed = car_velocity[-1, :].copy().tolist()
        return car_velocity

    def _get_head_action(
        self, robot_state_action_data: RobotStateActionData
    ) -> np.ndarray:
        head_actions = np.concatenate(
            [
                robot_state_action_data.data["state_head_actions"],
                robot_state_action_data.data["action_head_actions"],
            ],
            axis=0,
        )
        return head_actions

    def _get_height_action(
        self, robot_state_action_data: RobotStateActionData
    ) -> np.ndarray:
        height = np.concatenate(
            [
                robot_state_action_data.data["state_height"],
                robot_state_action_data.data["action_height"],
            ],
            axis=0,
        )
        return height

    def go_home(self):
        if self.current_robot_state_action is None:
            self.get_observation()

        self.current_robot_state_action.save_action_data_with_key(
            np.zeros((1, 3)), "left_ee_cartesian_pos"
        )
        self.current_robot_state_action.save_action_data_with_key(
            np.zeros((1, 3)), "left_ee_rotation"
        )
        self.current_robot_state_action.save_action_data_with_key(
            np.zeros((1, 1)), "left_gripper"
        )
        self.current_robot_state_action.save_action_data_with_key(
            np.zeros((1, 3)), "right_ee_cartesian_pos"
        )
        self.current_robot_state_action.save_action_data_with_key(
            np.zeros((1, 3)), "right_ee_rotation"
        )
        self.current_robot_state_action.save_action_data_with_key(
            np.zeros((1, 1)), "right_gripper"
        )

        self.current_robot_state_action.save_action_data_with_key(
            np.array([[0, -1]]), "head_actions"
        )
        self.current_robot_state_action.save_action_data_with_key(
            np.array([[0.4]]), "height"
        )
        self.current_robot_state_action.save_action_data_with_key(
            np.array([self.last_speed]), "velocity_decomposed"
        )

        action_dict = {"robot_state_action_data": self.current_robot_state_action}
        self.apply_action(
            action_dict,
            robot_action_interpolate_multiplier=150,
            robot_action_start_ratio=0,
            robot_action_end_ratio=1,
        )
        self.logger.info("robot returned to initial pose")

    def get_serialized_actions(
        self,
        input: dict,
        robot_action_interpolate_multiplier=None,
        robot_action_start_ratio=None,
        robot_action_end_ratio=None,
    ) -> dict:
        """Build serialized Turtle actions from model_output (dual arms + head/lift/car_pose); does not send."""
        assert "robot_state_action_data" in input

        left_arm_action = self._get_left_arm_action(input["robot_state_action_data"])
        right_arm_action = self._get_right_arm_action(input["robot_state_action_data"])
        action_length = len(left_arm_action)
        if robot_action_start_ratio is None:
            robot_action_start_ratio = self.config.robot_action_start_ratio
        if robot_action_end_ratio is None:
            robot_action_end_ratio = self.config.robot_action_end_ratio
        start_frame = int(robot_action_start_ratio * action_length)
        end_frame = int(robot_action_end_ratio * action_length)

        left_arm_action = left_arm_action[start_frame:end_frame]
        right_arm_action = right_arm_action[start_frame:end_frame]
        if robot_action_interpolate_multiplier is None:
            robot_action_interpolate_multiplier = (
                self.config.robot_action_interpolate_multiplier
            )
        target_length = robot_action_interpolate_multiplier * action_length
        (
            left_arm_action,
            right_arm_action,
        ) = UnifiedTrajectoryProcessor.interpolate_trajectory_batch(
            [left_arm_action, right_arm_action],
            target_length,
        )
        serialized_actions = {
            "follow1_pos": left_arm_action.tolist(),
            "follow2_pos": right_arm_action.tolist(),
        }

        if not self.config.turtle_as_desktop:
            head_action = self._get_head_action(input["robot_state_action_data"])
            height_action = self._get_height_action(input["robot_state_action_data"])
            car_velocity_action = self._get_car_velocity(
                input["robot_state_action_data"]
            )
            head_action = head_action[start_frame:end_frame]
            height_action = height_action[start_frame:end_frame]
            car_velocity_action = car_velocity_action[start_frame:end_frame]
            car_pose_action = np.array(self._calculate_car_pose(car_velocity_action))
            (
                head_action,
                height_action,
                car_pose_action,
            ) = UnifiedTrajectoryProcessor.interpolate_trajectory_batch(
                [head_action, height_action, car_pose_action],
                target_length,
            )
            serialized_actions["head_pos"] = head_action.tolist()
            serialized_actions["lift"] = height_action.tolist()
            serialized_actions["car_pose"] = car_pose_action.tolist()
        else:
            serialized_actions["head_pos"] = [
                [0, -1] for _ in range(len(left_arm_action))
            ]
            serialized_actions["lift"] = [0.4 for _ in range(len(left_arm_action))]
            serialized_actions["car_pose"] = [
                [0.0, 0.0, 0.0] for _ in range(len(left_arm_action))
            ]
        return serialized_actions

    def apply_action(
        self,
        input: dict,
        robot_action_interpolate_multiplier=None,
        robot_action_start_ratio=None,
        robot_action_end_ratio=None,
    ) -> None:
        serialized_actions = self.get_serialized_actions(
            input,
            robot_action_interpolate_multiplier=robot_action_interpolate_multiplier,
            robot_action_start_ratio=robot_action_start_ratio,
            robot_action_end_ratio=robot_action_end_ratio,
        )
        self._send_actions(serialized_actions)


class TurtleRobotPreprocessor(TurtleRobot):
    """Turtle preprocessor: no real robot; builds observations and get_serialized_actions for serving."""

    def __init__(self, config: InferConfig, robot_id=10) -> None:
        self.config = config
        self.is_received = False
        self.current_robot_state_action = None
        self.logger = InferLogger.get_robot_logger("Robot")
        self.vehicle_pose_handler = VehiclePoseHandler()
        self.last_speed = [0, 0, 0]

    def _send_actions(self, serialized_actions: dict) -> None:
        return

    def get_observation(self, state, views):
        robot_state_action_data = RobotStateActionData(config=self.config)
        for key in robot_action_key_mapping.keys():
            state_key_str = robot_action_key_mapping[key]
            value = None
            match = re.match(r"(\w+)\[(.*)\]", state_key_str)
            if match:
                base_key = match.group(1)
                if base_key in state:
                    slicing_str = match.group(2)
                    slice_parts = slicing_str.split(":")
                    slice_args = [(int(p) if p.strip() else None) for p in slice_parts]
                    s = slice(*slice_args)
                    value = state[base_key][s]
            else:
                if state_key_str in state:
                    value = state[state_key_str]
            if value is not None:
                robot_state_action_data.save_state_data_with_key(
                    np.asarray(value)[None], key
                )
        dof_mask = self._get_dof_mask()
        robot_state_action_data.dof_mask = dof_mask
        self.current_robot_state_action = robot_state_action_data
        return {
            "robot_state_action_data": robot_state_action_data,
            **_views_to_camera_observation(self.config, views),
        }


class EX001Robot(Robot):
    """EX001 robot: no pose integration; predict from velocity_decomposed_odom state and send to client."""

    def __init__(self, config: InferConfig, robot_id=10) -> None:
        super().__init__(config, robot_id)
        self.last_speed = [0, 0, 0]

    def _get_velocity_decomposed(
        self, robot_state_action_data: RobotStateActionData
    ) -> np.ndarray:
        """Get decomposed velocity actions: concat state and action velocity_decomposed_odom"""
        velocity_decomposed = np.concatenate(
            [
                robot_state_action_data.data["state_velocity_decomposed_odom"],
                robot_state_action_data.data["action_velocity_decomposed_odom"],
            ],
            axis=0,
        )
        self.last_speed = velocity_decomposed[-1, :].copy().tolist()
        return velocity_decomposed

    def _get_head_action(
        self, robot_state_action_data: RobotStateActionData
    ) -> np.ndarray:
        """Get head actions: concat state and action head sequence; supports head_rotation / head_actions"""
        if robot_state_action_data.data.get("state_head_rotation") is not None:
            state_key, action_key = "state_head_rotation", "action_head_rotation"
        else:
            state_key, action_key = "state_head_actions", "action_head_actions"
        head_actions = np.concatenate(
            [
                robot_state_action_data.data[state_key],
                robot_state_action_data.data[action_key],
            ],
            axis=0,
        )
        return head_actions

    def _get_height_action(
        self, robot_state_action_data: RobotStateActionData
    ) -> np.ndarray:
        """Get lift height actions: concat state and action height sequence"""
        height = np.concatenate(
            [
                robot_state_action_data.data["state_height"],
                robot_state_action_data.data["action_height"],
            ],
            axis=0,
        )
        return height

    def go_home(self):
        """Return to initial pose"""
        if self.current_robot_state_action is None:
            self.get_observation()

        self.current_robot_state_action.save_action_data_with_key(
            np.zeros((1, 3)), "left_ee_cartesian_pos"
        )
        self.current_robot_state_action.save_action_data_with_key(
            np.zeros((1, 3)), "left_ee_rotation"
        )
        self.current_robot_state_action.save_action_data_with_key(
            np.zeros((1, 1)), "left_gripper"
        )
        self.current_robot_state_action.save_action_data_with_key(
            np.zeros((1, 3)), "right_ee_cartesian_pos"
        )
        self.current_robot_state_action.save_action_data_with_key(
            np.zeros((1, 3)), "right_ee_rotation"
        )
        self.current_robot_state_action.save_action_data_with_key(
            np.zeros((1, 1)), "right_gripper"
        )

        self.current_robot_state_action.save_action_data_with_key(
            np.array([[0, -1]]), "head_actions"
        )
        self.current_robot_state_action.save_action_data_with_key(
            np.array([[0.4]]), "height"
        )
        self.current_robot_state_action.save_action_data_with_key(
            np.array([self.last_speed]), "velocity_decomposed_odom"
        )

        action_dict = {"robot_state_action_data": self.current_robot_state_action}
        self.apply_action(
            action_dict,
            robot_action_interpolate_multiplier=150,
            robot_action_start_ratio=0,
            robot_action_end_ratio=1,
        )
        self.logger.info("robot returned to initial pose")

    def get_serialized_actions(
        self,
        input: dict,
        robot_action_interpolate_multiplier=None,
        robot_action_start_ratio=None,
        robot_action_end_ratio=None,
    ) -> dict:
        """Build serialized EX001 actions from model_output (dual arms + head/lift/velocity_decomposed_odom); does not send.

        Uses predicted velocity_decomposed_odom directly; no pose integration.
        """
        assert "robot_state_action_data" in input

        left_arm_action = self._get_left_arm_action(input["robot_state_action_data"])
        right_arm_action = self._get_right_arm_action(input["robot_state_action_data"])
        action_length = len(left_arm_action)

        # action_length is 1 state frame + action_horizon action frames
        # Actual action length is action_length - 1
        actual_action_length = action_length - 1

        if robot_action_start_ratio is None:
            robot_action_start_ratio = self.config.robot_action_start_ratio
        if robot_action_end_ratio is None:
            robot_action_end_ratio = self.config.robot_action_end_ratio

        # Trim indices in action section (skip frame 0 state)
        start_frame = 1 + int(robot_action_start_ratio * actual_action_length)
        end_frame = 1 + int(robot_action_end_ratio * actual_action_length)

        left_arm_action = left_arm_action[start_frame:end_frame]
        right_arm_action = right_arm_action[start_frame:end_frame]

        if robot_action_interpolate_multiplier is None:
            robot_action_interpolate_multiplier = (
                self.config.robot_action_interpolate_multiplier
            )
        target_length = robot_action_interpolate_multiplier * len(left_arm_action)

        (
            left_arm_action,
            right_arm_action,
        ) = UnifiedTrajectoryProcessor.interpolate_trajectory_batch(
            [left_arm_action, right_arm_action],
            target_length,
        )

        serialized_actions = {
            "follow1_pos": left_arm_action.tolist(),
            "follow2_pos": right_arm_action.tolist(),
        }

        # Get head, lift, and velocity actions
        head_action = self._get_head_action(input["robot_state_action_data"])
        height_action = self._get_height_action(input["robot_state_action_data"])
        velocity_decomposed_action = self._get_velocity_decomposed(
            input["robot_state_action_data"]
        )

        # Trim actions (same start_frame and end_frame)
        head_action = head_action[start_frame:end_frame]
        height_action = height_action[start_frame:end_frame]
        velocity_decomposed_action = velocity_decomposed_action[start_frame:end_frame]

        # Interpolate
        (
            head_action,
            height_action,
            velocity_decomposed_action,
        ) = UnifiedTrajectoryProcessor.interpolate_trajectory_batch(
            [head_action, height_action, velocity_decomposed_action],
            target_length,
        )

        serialized_actions["head_pos"] = head_action.tolist()
        serialized_actions["lift"] = height_action.tolist()
        # Send velocity_decomposed_odom directly; no pose integration
        serialized_actions["velocity_decomposed_odom"] = (
            velocity_decomposed_action.tolist()
        )
        return serialized_actions

    def apply_action(
        self,
        input: dict,
        robot_action_interpolate_multiplier=None,
        robot_action_start_ratio=None,
        robot_action_end_ratio=None,
    ) -> None:
        serialized_actions = self.get_serialized_actions(
            input,
            robot_action_interpolate_multiplier=robot_action_interpolate_multiplier,
            robot_action_start_ratio=robot_action_start_ratio,
            robot_action_end_ratio=robot_action_end_ratio,
        )
        self._send_actions(serialized_actions)


class EX001RobotPreprocessor(EX001Robot):
    """EX001 preprocessor: no real robot; builds observations and get_serialized_actions for serving."""

    def __init__(self, config: InferConfig, robot_id=10) -> None:
        self.config = config
        self.is_received = False
        self.current_robot_state_action = None
        self.logger = InferLogger.get_robot_logger("Robot")
        self.last_speed = [0, 0, 0]

    def _send_actions(self, serialized_actions: dict) -> None:
        """No action send; preprocessing only"""
        return

    def get_observation(self, state, views):
        """Build observation from external state and views"""
        robot_state_action_data = RobotStateActionData(config=self.config)
        for key in robot_action_key_mapping.keys():
            state_key_str = robot_action_key_mapping[key]
            value = None

            # Keep the same 7D adaptation behavior as Robot.get_observation(),
            # so offline/serving preprocessor path is consistent with runtime path.
            composed_value = self._try_compose_arm_joint_pos_with_gripper(state, key)
            if composed_value is not None:
                value = composed_value
                robot_state_action_data.save_state_data_with_key(
                    np.asarray(value)[None], key
                )
                continue

            match = re.match(r"(\w+)\[(.*)\]", state_key_str)
            if match:
                base_key = match.group(1)
                if base_key in state:
                    slicing_str = match.group(2)
                    slice_parts = slicing_str.split(":")
                    slice_args = [(int(p) if p.strip() else None) for p in slice_parts]
                    s = slice(*slice_args)
                    value = state[base_key][s]
            else:
                if state_key_str in state:
                    value = state[state_key_str]
            if value is not None:
                robot_state_action_data.save_state_data_with_key(
                    np.asarray(value)[None], key
                )
        dof_mask = self._get_dof_mask()
        robot_state_action_data.dof_mask = dof_mask
        self.current_robot_state_action = robot_state_action_data

        # Build observation; handle key name mismatches
        observation = {
            "robot_state_action_data": robot_state_action_data,
        }

        # Try multiple possible key names
        camera_mappings = {
            "face_view": [
                self.config.camera_front_key,
                "face_view",
                "front_view",
            ],
            "left_wrist_view": [
                self.config.camera_left_key,
                "left_wrist_view",
                "left_view",
            ],
            "right_wrist_view": [
                self.config.camera_right_key,
                "right_wrist_view",
                "right_view",
            ],
        }

        for obs_key in self.config.cam_names:
            possible_view_keys = camera_mappings.get(obs_key, [obs_key])
            found = False
            for view_key in possible_view_keys:
                if view_key in views and views[view_key] is not None:
                    view_data = views[view_key]
                    # If view_data is (1, H, W, 3), take [0]
                    if (
                        isinstance(view_data, np.ndarray)
                        and len(view_data.shape) == 4
                        and view_data.shape[0] == 1
                    ):
                        observation[obs_key] = view_data[0]
                    else:
                        observation[obs_key] = view_data
                    found = True
                    self.logger.info(
                        f"EX001RobotPreprocessor: using '{view_key}' as '{obs_key}'"
                    )
                    break
            if not found:
                self.logger.error(
                    f"EX001RobotPreprocessor: no view for '{obs_key}', tried keys: {possible_view_keys}"
                )
                # Black image placeholder
                observation[obs_key] = np.zeros((480, 640, 3), dtype=np.uint8)

        return observation
