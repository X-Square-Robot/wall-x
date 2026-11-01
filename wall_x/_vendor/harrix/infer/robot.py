import numpy as np
from abc import ABC
import re

from wall_x._vendor.harrix.infer.infer_config import InferConfig
from wall_x._vendor.harrix.infer.utils import VehiclePoseHandler, UnifiedTrajectoryProcessor
from wall_x._vendor.harrix.infer.base_dataclass import RobotStateActionData
from wall_x._vendor.harrix.infer.socket_controller import RobotController

from wall_x._vendor.harrix.infer.logger import InferLogger

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
    # "follow_left_arm_joint_cur": "follow1_joints_cur",  # 1 -> 7
    # "follow_right_arm_joint_cur": "follow2_joints_cur",  # 1 -> 7
    # "follow_left_gripper_cur": "follow1_joints_cur[-1:]",
    # "follow_right_gripper_cur": "follow2_joints_cur[-1:]",
    # "follow_left_arm_joint_pos": "follow1_joints",
    # "follow_right_arm_joint_pos": "follow2_joints",
    "follow_left_wrench_ext_local_force": "follow1_end_effort_force",
    "follow_left_wrench_ext_local_torque": "follow1_end_effort_torque",
    "follow_right_wrench_ext_local_force": "follow2_end_effort_force",
    "follow_right_wrench_ext_local_torque": "follow2_end_effort_torque",
    # "follow_left_wrench_ext_local_force_from_joint": "follow1_end_effort_force_from_joint",
    # "follow_left_wrench_ext_local_torque_from_joint": "follow1_end_effort_torque_from_joint",
    # "follow_right_wrench_ext_local_force_from_joint": "follow2_end_effort_force_from_joint",
    # "follow_right_wrench_ext_local_torque_from_joint": "follow2_end_effort_torque_from_joint",
    "follow_left_wrench_ext_world_force": "follow1_wrench_ext_world_force",
    "follow_left_wrench_ext_world_torque": "follow1_wrench_ext_world_torque",
    # "follow_left_wrench_ext_world_force_from_joint": "follow1_wrench_ext_world_force_from_joint",
    # "follow_left_wrench_ext_world_torque_from_joint": "follow1_wrench_ext_world_torque_from_joint",
    "follow_right_wrench_ext_world_force": "follow2_wrench_ext_world_force",
    "follow_right_wrench_ext_world_torque": "follow2_wrench_ext_world_torque",
    # "follow_right_wrench_ext_world_force_from_joint": "follow2_wrench_ext_world_force_from_joint",
    # "follow_right_wrench_ext_world_torque_from_joint": "follow2_wrench_ext_world_torque_from_joint",
    # "follow_left_arm_joint_dev": "follow1_joints_dev",
    # "follow_right_arm_joint_dev": "follow2_joints_dev",
}


class Robot(ABC):
    def __init__(self, config: InferConfig, robot_id=10) -> None:
        self.config = config

        self.robot_controller = RobotController(
            robot_id=robot_id, host=config.robot_host, port=config.robot_port
        )

        self.robot_controller.connect()

        self.is_received = False
        self.current_robot_state_action = None

        self.logger = InferLogger.get_robot_logger("Robot")

    def _get_views_and_state(self):
        state = self.robot_controller.recv_action()
        views = self.robot_controller.recv_image(
            ["camera_left", "camera_front", "camera_right"]
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
            "face_view": views["camera_front"][0],  # (1, 480, 640, 3)
            "left_wrist_view": views["camera_left"][0],  # (1, 480, 640, 3)
            "right_wrist_view": views["camera_right"][0],  # (1, 480, 640, 3)
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
        self.logger.info("robot returned to home position")

    def _get_left_arm_action(
        self, robot_state_action_data: RobotStateActionData
    ) -> np.ndarray:
        if not self.config.robot_use_joint_angle_control:
            left_ee_cartesian_pos = np.concatenate(
                [
                    robot_state_action_data.data["state_left_ee_cartesian_pos"],
                    robot_state_action_data.data["action_left_ee_cartesian_pos"],
                ],
                axis=0,
            )
            left_ee_rotation = np.concatenate(
                [
                    robot_state_action_data.data["state_left_ee_rotation"],
                    robot_state_action_data.data["action_left_ee_rotation"],
                ],
                axis=0,
            )
            left_gripper = np.concatenate(
                [
                    robot_state_action_data.data["state_left_gripper"],
                    robot_state_action_data.data["action_left_gripper"],
                ],
                axis=0,
            )
            return np.concatenate(
                [left_ee_cartesian_pos, left_ee_rotation, left_gripper], axis=1
            )
        else:
            left_arm_joint_pos = np.concatenate(
                [
                    robot_state_action_data.data["state_left_arm_joint_pos"],
                    robot_state_action_data.data["action_left_arm_joint_pos"],
                ],
                axis=0,
            )
            return left_arm_joint_pos

    def _get_right_arm_action(
        self, robot_state_action_data: RobotStateActionData
    ) -> np.ndarray:
        if not self.config.robot_use_joint_angle_control:
            right_ee_cartesian_pos = np.concatenate(
                [
                    robot_state_action_data.data["state_right_ee_cartesian_pos"],
                    robot_state_action_data.data["action_right_ee_cartesian_pos"],
                ],
                axis=0,
            )
            right_ee_rotation = np.concatenate(
                [
                    robot_state_action_data.data["state_right_ee_rotation"],
                    robot_state_action_data.data["action_right_ee_rotation"],
                ],
                axis=0,
            )
            right_gripper = np.concatenate(
                [
                    robot_state_action_data.data["state_right_gripper"],
                    robot_state_action_data.data["action_right_gripper"],
                ],
                axis=0,
            )
            return np.concatenate(
                [right_ee_cartesian_pos, right_ee_rotation, right_gripper], axis=1
            )
        else:
            right_arm_joint_pos = np.concatenate(
                [
                    robot_state_action_data.data["state_right_arm_joint_pos"],
                    robot_state_action_data.data["action_right_arm_joint_pos"],
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
        """Build trimmed and interpolated serialized actions from ``model_output``.

        ``model_output`` must contain ``robot_state_action_data``. Nothing is
        sent; ``apply_action`` and the serving policy forward the returned dict
        to the robot or the client.
        """
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
        """Send actions to the robot controller."""
        self.robot_controller.robot_comm.send_dict(serialized_actions)
        self.is_received = False
        self.current_robot_state_action = None


class DesktopRobot(Robot):
    def __init__(self, config: InferConfig, robot_id=10) -> None:
        self.is_received = False
        self.current_robot_state_action = None

        self.logger = InferLogger.get_robot_logger("Robot")

    def _get_dof_mask(self):
        dof_config = self.config.train_config["dof_config"]
        total_dof = sum(dof_config.values())
        dof_mask = np.ones((1, self.config.action_horizon, total_dof))

        mask_keys = ["head_actions", "height", "velocity_decomposed"]
        start_idx = 0
        for key, dof_size in dof_config.items():
            if key in mask_keys:
                dof_mask[:, :, start_idx : start_idx + dof_size] = 0
            start_idx += dof_size

        return dof_mask


class DesktopRobotPreprocessor(DesktopRobot):
    def __init__(self, config: InferConfig, robot_id=10) -> None:
        self.config = config

        self.is_received = False
        self.current_robot_state_action = None
        self.logger = InferLogger.get_robot_logger("Robot")

    def _send_actions(self, serialized_actions: dict) -> None:
        # bypass
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
            "face_view": views["camera_front"][0],  # (1, 480, 640, 3)
            "left_wrist_view": views["camera_left"][0],  # (1, 480, 640, 3)
            "right_wrist_view": views["camera_right"][0],  # (1, 480, 640, 3)
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
            state["head_pos"] = list(self.config.fixed_head_pos)
            state["lift"] = [self.config.fixed_lift]

        return state

    def _calculate_car_pose(self, base_velocity_pred):
        dt = 1 / 20
        current_pose = (
            self.vehicle_pose_handler.current_pose.copy()
            if self.vehicle_pose_handler.current_pose is not None
            else np.array([0.0, 0.0, 0.0])
        )

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

        head_pos = (
            self.config.fixed_head_pos if self.config.turtle_as_desktop else (0, -1)
        )
        lift = self.config.fixed_lift if self.config.turtle_as_desktop else 0.4
        self.current_robot_state_action.save_action_data_with_key(
            np.array([head_pos]), "head_actions"
        )
        self.current_robot_state_action.save_action_data_with_key(
            np.array([[lift]]), "height"
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
        self.logger.info("robot returned to home position")

    def get_serialized_actions(
        self,
        input: dict,
        robot_action_interpolate_multiplier=None,
        robot_action_start_ratio=None,
        robot_action_end_ratio=None,
    ) -> dict:
        """Build Turtle serialized actions (both arms + head/lift/car_pose) without sending."""
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
                list(self.config.fixed_head_pos) for _ in range(len(left_arm_action))
            ]
            serialized_actions["lift"] = [
                self.config.fixed_lift for _ in range(len(left_arm_action))
            ]
            serialized_actions["car_pose"] = [
                list(self.config.fixed_car_pose) for _ in range(len(left_arm_action))
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
    """Turtle preprocessor for serving: builds observations and serialized actions without a robot connection."""

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
            "face_view": views["camera_front"][0],
            "left_wrist_view": views["camera_left"][0],
            "right_wrist_view": views["camera_right"][0],
        }


class EX001Robot(Robot):
    """EX001 robot: predicts velocity_decomposed_odom directly from state, without integrating a base pose."""

    def __init__(self, config: InferConfig, robot_id=10) -> None:
        super().__init__(config, robot_id)
        self.last_speed = [0, 0, 0]

    def _get_velocity_decomposed(
        self, robot_state_action_data: RobotStateActionData
    ) -> np.ndarray:
        """Concatenate state and action velocity_decomposed_odom sequences."""
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
        """Concatenate state and action head sequences (head_rotation or head_actions)."""
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
        """Concatenate state and action lift height sequences."""
        height = np.concatenate(
            [
                robot_state_action_data.data["state_height"],
                robot_state_action_data.data["action_height"],
            ],
            axis=0,
        )
        return height

    def go_home(self):
        """Return to the home position."""
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

        if self.config.ex001_as_desktop:
            head_pos = self.config.fixed_head_pos
            lift = self.config.fixed_lift
            base_velocity = self.config.fixed_base_velocity
        else:
            head_pos = (0, -1)
            lift = 0.4
            base_velocity = self.last_speed
        self.current_robot_state_action.save_action_data_with_key(
            np.array([head_pos]), "head_actions"
        )
        self.current_robot_state_action.save_action_data_with_key(
            np.array([[lift]]), "height"
        )
        self.current_robot_state_action.save_action_data_with_key(
            np.array([base_velocity]), "velocity_decomposed_odom"
        )

        action_dict = {"robot_state_action_data": self.current_robot_state_action}
        self.apply_action(
            action_dict,
            robot_action_interpolate_multiplier=150,
            robot_action_start_ratio=0,
            robot_action_end_ratio=1,
        )
        self.logger.info("robot returned to home position")

    def get_serialized_actions(
        self,
        input: dict,
        robot_action_interpolate_multiplier=None,
        robot_action_start_ratio=None,
        robot_action_end_ratio=None,
    ) -> dict:
        """Build EX001 serialized actions (both arms + head/lift/velocity_decomposed_odom) without sending.

        The predicted velocity_decomposed_odom is used directly; no base pose is integrated.
        """
        assert "robot_state_action_data" in input

        left_arm_action = self._get_left_arm_action(input["robot_state_action_data"])
        right_arm_action = self._get_right_arm_action(input["robot_state_action_data"])
        action_length = len(left_arm_action)

        actual_action_length = action_length - 1

        if robot_action_start_ratio is None:
            robot_action_start_ratio = self.config.robot_action_start_ratio
        if robot_action_end_ratio is None:
            robot_action_end_ratio = self.config.robot_action_end_ratio

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

        if self.config.ex001_as_desktop:
            serialized_actions["head_pos"] = [
                list(self.config.fixed_head_pos) for _ in range(target_length)
            ]
            # Preserve EX001's existing (N, 1) lift wire shape.
            serialized_actions["lift"] = [
                [self.config.fixed_lift] for _ in range(target_length)
            ]
            serialized_actions["velocity_decomposed_odom"] = [
                list(self.config.fixed_base_velocity) for _ in range(target_length)
            ]
        else:
            head_action = self._get_head_action(input["robot_state_action_data"])
            height_action = self._get_height_action(input["robot_state_action_data"])
            velocity_decomposed_action = self._get_velocity_decomposed(
                input["robot_state_action_data"]
            )

            head_action = head_action[start_frame:end_frame]
            height_action = height_action[start_frame:end_frame]
            velocity_decomposed_action = velocity_decomposed_action[
                start_frame:end_frame
            ]

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
    """EX001 preprocessor for serving: builds observations and serialized actions without a robot connection."""

    def __init__(self, config: InferConfig, robot_id=10) -> None:
        self.config = config
        self.is_received = False
        self.current_robot_state_action = None
        self.logger = InferLogger.get_robot_logger("Robot")
        self.last_speed = [0, 0, 0]

    def _send_actions(self, serialized_actions: dict) -> None:
        """Preprocessing only; actions are not sent."""
        return

    def get_observation(self, state, views):
        """Build an observation from externally provided state and views."""
        if self.config.ex001_as_desktop:
            # Do not mutate the websocket request object in place. Fixed values
            # remain valid proprioception and therefore keep agent_pos_mask=1.
            state = dict(state)
            state["head_pos"] = list(self.config.fixed_head_pos)
            state["lift"] = [self.config.fixed_lift]
            state["velocity_decomposed_odom"] = list(
                self.config.fixed_base_velocity
            )
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

        observation = {
            "robot_state_action_data": robot_state_action_data,
        }

        camera_mappings = {
            "face_view": ["camera_front", "face_view", "front_view"],
            "left_wrist_view": ["camera_left", "left_wrist_view", "left_view"],
            "right_wrist_view": ["camera_right", "right_wrist_view", "right_view"],
        }

        for obs_key, possible_view_keys in camera_mappings.items():
            found = False
            for view_key in possible_view_keys:
                if view_key in views and views[view_key] is not None:
                    view_data = views[view_key]
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
                    f"EX001RobotPreprocessor: no view found for '{obs_key}', tried keys: {possible_view_keys}"
                )
                observation[obs_key] = np.zeros((480, 640, 3), dtype=np.uint8)

        return observation
