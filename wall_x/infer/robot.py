import numpy as np
from abc import ABC
import re

from wall_x.infer.infer_config import InferConfig
from wall_x.infer.utils import VehiclePoseHandler, UnifiedTrajectoryProcessor
from wall_x.infer.base_dataclass import RobotStateActionData
from wall_x.infer.socket_controller import RobotController

# from wall_x.infer.socket_controller import DummyRobotController
from wall_x.infer.logger import InferLogger

robot_action_key_mapping = {
    "follow_left_ee_cartesian_pos": "follow1_pos[:3]",
    "follow_left_ee_rotation": "follow1_pos[3:6]",
    "follow_left_gripper": "follow1_pos[6:7]",
    "follow_right_ee_cartesian_pos": "follow2_pos[:3]",
    "follow_right_ee_rotation": "follow2_pos[3:6]",
    "follow_right_gripper": "follow2_pos[6:7]",
    "head_actions": "head_pos",
    "height": "lift",
    "velocity_decomposed": "velocity_decomposed",
    "follow_left_arm_joint_cur": "follow1_joints_cur[-1:]",
    "follow_right_arm_joint_cur": "follow2_joints_cur[-1:]",
    "follow_left_gripper_cur": "follow1_joints_cur[-1:]",
    "follow_right_gripper_cur": "follow2_joints_cur[-1:]",
    "follow_left_arm_joint_pos": "follow1_joints",
    "follow_right_arm_joint_pos": "follow2_joints",
}


class Robot(ABC):
    def __init__(self, config: InferConfig, robot_id=10) -> None:
        self.config = config

        # DummyRobotController 用于测试
        # self.robot_controller = DummyRobotController(
        #     robot_id=robot_id,
        #     host=config.robot_host,
        #     port=config.robot_port
        # )
        # RobotController 用于实际推理
        self.robot_controller = RobotController(
            robot_id=robot_id, host=config.robot_host, port=config.robot_port
        )

        self.robot_controller.connect()

        self.is_received = False  # 阻塞标志位
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

    def get_observation(self):
        state, views = self._get_views_and_state()

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
        self.logger.info("机器人回到初始位置完成")

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

    def apply_action(
        self,
        input: dict,
        robot_action_interpolate_multiplier=None,
        robot_action_start_ratio=None,
        robot_action_end_ratio=None,
    ) -> None:
        assert "robot_state_action_data" in input

        left_arm_action = self._get_left_arm_action(input["robot_state_action_data"])
        right_arm_action = self._get_right_arm_action(input["robot_state_action_data"])

        # 截取action
        action_length = len(left_arm_action)
        if robot_action_start_ratio is None:
            robot_action_start_ratio = self.config.robot_action_start_ratio
        if robot_action_end_ratio is None:
            robot_action_end_ratio = self.config.robot_action_end_ratio
        start_frame = int(robot_action_start_ratio * action_length)
        end_frame = int(robot_action_end_ratio * action_length)

        left_arm_action = left_arm_action[start_frame:end_frame]
        right_arm_action = right_arm_action[start_frame:end_frame]

        # 插值
        if robot_action_interpolate_multiplier is None:
            robot_action_interpolate_multiplier = (
                self.config.robot_action_interpolate_multiplier
            )
        target_length = robot_action_interpolate_multiplier * action_length
        left_arm_action, right_arm_action = (
            UnifiedTrajectoryProcessor.interpolate_trajectory_batch(
                [left_arm_action, right_arm_action], target_length
            )
        )

        if not self.config.robot_use_joint_angle_control:
            serialized_actions = {
                "follow1_pos": left_arm_action.tolist(),
                "follow2_pos": right_arm_action.tolist(),
            }
        else:
            serialized_actions = {
                "follow1_joints": left_arm_action.tolist(),
                "follow2_joints": right_arm_action.tolist(),
            }

        self._send_actions(serialized_actions)

    def _send_actions(self, serialized_actions: dict) -> None:
        """发送动作到机器人控制器"""
        self.robot_controller.robot_comm.send_dict(serialized_actions)
        self.is_received = False
        self.current_robot_state_action = None


class DesktopRobot(Robot):
    def __init__(self, config: InferConfig, robot_id=10) -> None:
        self.is_received = False  # 阻塞标志位
        self.current_robot_state_action = None

        self.logger = InferLogger.get_robot_logger("Robot")

    def _get_dof_mask(self):
        dof_config = self.config.train_config["dof_config"]
        total_dof = sum(dof_config.values())
        dof_mask = np.ones((1, self.config.action_horizon, total_dof))

        # 对于desktop，需要mask掉head_actions、height和velocity_decomposed
        mask_keys = ["head_actions", "height", "velocity_decomposed"]
        start_idx = 0
        for key, dof_size in dof_config.items():
            if key in mask_keys:
                # 将这些维度的mask设置为0
                dof_mask[:, :, start_idx : start_idx + dof_size] = 0
            start_idx += dof_size

        return dof_mask


class DesktopRobotPreprocessor(DesktopRobot):
    def __init__(self, config: InferConfig, robot_id=10) -> None:
        self.config = config

        self.is_received = False  # 阻塞标志位
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
        super().__init__(config, robot_id)
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
        # 向量化积分速度为位置
        dt = 1 / 20
        current_pose = (
            self.vehicle_pose_handler.current_pose.copy()
            if self.vehicle_pose_handler.current_pose is not None
            else np.array([0.0, 0.0, 0.0])
        )

        # 批量积分计算位置
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
        self.logger.info("机器人回到初始位置完成")

    def apply_action(
        self,
        input: dict,
        robot_action_interpolate_multiplier=None,
        robot_action_start_ratio=None,
        robot_action_end_ratio=None,
    ) -> None:
        assert "robot_state_action_data" in input

        left_arm_action = self._get_left_arm_action(input["robot_state_action_data"])
        right_arm_action = self._get_right_arm_action(input["robot_state_action_data"])
        head_action = self._get_head_action(input["robot_state_action_data"])
        height_action = self._get_height_action(input["robot_state_action_data"])
        car_velocity_action = self._get_car_velocity(input["robot_state_action_data"])

        # 截取action
        action_length = len(left_arm_action)
        if robot_action_start_ratio is None:
            robot_action_start_ratio = self.config.robot_action_start_ratio
        if robot_action_end_ratio is None:
            robot_action_end_ratio = self.config.robot_action_end_ratio
        start_frame = int(robot_action_start_ratio * action_length)
        end_frame = int(robot_action_end_ratio * action_length)

        left_arm_action = left_arm_action[start_frame:end_frame]
        right_arm_action = right_arm_action[start_frame:end_frame]
        head_action = head_action[start_frame:end_frame]
        height_action = height_action[start_frame:end_frame]
        car_velocity_action = car_velocity_action[start_frame:end_frame]

        # car velocity to car pose
        car_pose_action = np.array(self._calculate_car_pose(car_velocity_action))

        # 插值
        if robot_action_interpolate_multiplier is None:
            robot_action_interpolate_multiplier = (
                self.config.robot_action_interpolate_multiplier
            )
        target_length = robot_action_interpolate_multiplier * action_length
        (
            left_arm_action,
            right_arm_action,
            head_action,
            height_action,
            car_pose_action,
        ) = UnifiedTrajectoryProcessor.interpolate_trajectory_batch(
            [
                left_arm_action,
                right_arm_action,
                head_action,
                height_action,
                car_pose_action,
            ],
            target_length,
        )

        serialized_actions = {
            "follow1_pos": left_arm_action.tolist(),
            "follow2_pos": right_arm_action.tolist(),
            "head_pos": head_action.tolist(),
            "lift": height_action.tolist(),
            "car_pose": car_pose_action.tolist(),
        }

        if self.config.turtle_as_desktop:
            serialized_actions["head_pos"] = [
                [0, -1] for _ in range(len(left_arm_action))
            ]
            serialized_actions["lift"] = [0.4 for _ in range(len(left_arm_action))]
            serialized_actions["car_pose"] = [
                [0.0, 0.0, 0.0] for _ in range(len(left_arm_action))
            ]

        self._send_actions(serialized_actions)
