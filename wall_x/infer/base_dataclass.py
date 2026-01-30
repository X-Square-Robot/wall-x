from wall_x.infer.infer_config import InferConfig
from typing import Optional, List
from dataclasses import dataclass, field
import numpy as np
import torch
import wall_x.infer.data_utils as data_utils
from wall_x.infer.logger import InferLogger

dof_dims = {
    "left_ee_cartesian_pos": 3,
    "left_ee_cartesian_pos_relative": 3,
    "left_ee_rotation": 3,
    "left_ee_rotation_relative": 3,
    "left_ee_rotation_6D": 6,
    "left_ee_rotation_6D_relative": 6,
    "left_arm_joint_pos": 7,
    "left_gripper": 1,
    "left_gripper_cur": 1,
    "left_arm_joint_cur": 1,
    "right_ee_cartesian_pos": 3,
    "right_ee_cartesian_pos_relative": 3,
    "right_ee_rotation": 3,
    "right_ee_rotation_relative": 3,
    "right_ee_rotation_6D": 6,
    "right_ee_rotation_6D_relative": 6,
    "right_arm_joint_pos": 7,
    "right_gripper": 1,
    "right_gripper_cur": 1,
    "right_arm_joint_cur": 1,
    "head_actions": 2,
    "height": 1,
    "car_pose": 3,
    "velocity_decomposed": 3,
}


class ComputedDict(dict):
    """智能字典，支持注册计算规则，在get时自动计算None值"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._compute_rules = {}  # key -> compute_function

    def register_compute_rule(self, key, compute_func):
        """
        注册计算规则

        Args:
            key: 需要计算的键
            compute_func: 计算函数，接收self作为参数，返回计算结果
        """
        self._compute_rules[key] = compute_func

    def get(self, key, default=None):
        """重载get方法，支持自动计算"""
        value = super().get(key, default)

        # 如果值为None且有计算规则，尝试计算
        if value is None and key in self._compute_rules:
            try:
                computed_value = self._compute_rules[key](self)
                if computed_value is not None:
                    # 缓存计算结果
                    self[key] = computed_value
                    return computed_value
            except Exception:
                pass  # 计算失败则返回None或default

        return value if value is not None else default

    def __getitem__(self, key):
        """重载[]操作符，支持自动计算"""
        value = super().get(key, None)

        # 如果值为None且有计算规则，尝试计算
        if value is None and key in self._compute_rules:
            try:
                computed_value = self._compute_rules[key](self)
                if computed_value is not None:
                    # 缓存计算结果
                    self[key] = computed_value
                    return computed_value
            except Exception:
                pass  # 计算失败则抛出原始KeyError或返回None

        if key in self:
            return super().__getitem__(key)
        raise KeyError(key)


@dataclass
class RobotStateActionData:
    config: InferConfig = None
    data: ComputedDict = field(
        default_factory=lambda: ComputedDict(
            {
                # State (原pose) - 使用 state_ 前缀
                "state_left_ee_cartesian_pos": None,  # (1, 3)
                "state_left_ee_rotation": None,  # (1, 3)
                "state_left_ee_rotation_6D": None,
                "state_left_arm_joint_pos": None,
                "state_left_gripper": None,  # (1, 1)
                "state_left_arm_joint_cur": None,
                "state_left_gripper_cur": None,
                "state_right_ee_cartesian_pos": None,  # (1, 3)
                "state_right_ee_rotation": None,
                "state_right_ee_rotation_6D": None,  # (1, 6)
                "state_right_arm_joint_pos": None,
                "state_right_gripper": None,
                "state_right_gripper_cur": None,
                "state_right_arm_joint_cur": None,  # (1, 1)
                "state_head_actions": None,
                "state_height": None,
                "state_car_pose": None,
                "state_velocity_decomposed": None,
                # Action - 使用 action_ 前缀
                "action_left_ee_cartesian_pos": None,
                "action_left_ee_cartesian_pos_relative": None,
                "action_left_ee_rotation": None,
                "action_left_ee_rotation_relative": None,
                "action_left_ee_rotation_6D": None,
                "action_left_ee_rotation_6D_relative": None,
                "action_left_gripper": None,
                "action_left_arm_joint_pos": None,
                "action_right_ee_cartesian_pos": None,
                "action_right_ee_cartesian_pos_relative": None,
                "action_right_ee_rotation": None,
                "action_right_ee_rotation_relative": None,
                "action_right_ee_rotation_6D": None,
                "action_right_ee_rotation_6D_relative": None,
                "action_right_gripper": None,
                "action_right_arm_joint_pos": None,
                "action_head_actions": None,
                "action_height": None,
                "action_car_pose": None,
                "action_velocity_decomposed": None,
            }
        )
    )
    dof_mask: np.ndarray = None
    logger = InferLogger.get_robot_logger("RobotStateActionData")

    def __post_init__(self):
        """注册计算规则"""
        # State的计算规则 - euler angles -> 6D rotation
        self.data.register_compute_rule(
            "state_left_ee_rotation_6D",
            lambda d: (
                data_utils.euler_to_matrix_zyx_6d_nb(d["state_left_ee_rotation"])
                if d.get("state_left_ee_rotation") is not None
                else None
            ),
        )
        self.data.register_compute_rule(
            "state_right_ee_rotation_6D",
            lambda d: (
                data_utils.euler_to_matrix_zyx_6d_nb(d["state_right_ee_rotation"])
                if d.get("state_right_ee_rotation") is not None
                else None
            ),
        )

        # Action的计算规则 - absolute position从relative + state计算
        self.data.register_compute_rule(
            "action_left_ee_cartesian_pos",
            lambda d: (
                d.get("state_left_ee_cartesian_pos")
                + d.get("action_left_ee_cartesian_pos_relative")
                if d.get("state_left_ee_cartesian_pos") is not None
                and d.get("action_left_ee_cartesian_pos_relative") is not None
                else None
            ),
        )
        self.data.register_compute_rule(
            "action_right_ee_cartesian_pos",
            lambda d: (
                d.get("state_right_ee_cartesian_pos")
                + d.get("action_right_ee_cartesian_pos_relative")
                if d.get("state_right_ee_cartesian_pos") is not None
                and d.get("action_right_ee_cartesian_pos_relative") is not None
                else None
            ),
        )

        # Action的计算规则 - 得到abs rpy
        self.data.register_compute_rule(  # delta rpy -> abs rpy
            "action_left_ee_rotation",
            lambda d: (
                data_utils.compose_state_and_delta_to_abs_rpy(
                    d["action_left_ee_rotation_relative"],
                    d["state_left_ee_rotation"][0],
                )
                if d.get("action_left_ee_rotation_relative") is not None
                and d.get("state_left_ee_rotation") is not None
                else None
            ),
        )
        self.data.register_compute_rule(  # abs 6D -> abs rpy
            "action_left_ee_rotation",
            lambda d: (
                data_utils.so3_to_euler_zyx_batch_nb(d["action_left_ee_rotation_6D"])
                if d.get("action_left_ee_rotation_6D") is not None
                else None
            ),
        )
        self.data.register_compute_rule(  # delta 6D -> abs 6D -> abs rpy
            "action_left_ee_rotation_6D",
            lambda d: (
                data_utils.compose_state_and_delta_to_abs_rpy(
                    d["action_left_ee_rotation_6D_relative"],
                    d["state_left_ee_rotation_6D"][0],
                )
                if d.get("action_left_ee_rotation_6D_relative") is not None
                and d.get("state_left_ee_rotation_6D") is not None
                else None
            ),
        )

        self.data.register_compute_rule(  # delta rpy -> abs rpy
            "action_right_ee_rotation",
            lambda d: (
                data_utils.compose_state_and_delta_to_abs_rpy(
                    d["action_right_ee_rotation_relative"],
                    d["state_right_ee_rotation"][0],
                )
                if d.get("action_right_ee_rotation_relative") is not None
                and d.get("state_right_ee_rotation") is not None
                else None
            ),
        )
        self.data.register_compute_rule(  # abs 6D -> abs rpy
            "action_right_ee_rotation",
            lambda d: (
                data_utils.so3_to_euler_zyx_batch_nb(d["action_right_ee_rotation_6D"])
                if d.get("action_right_ee_rotation_6D") is not None
                else None
            ),
        )
        self.data.register_compute_rule(  # delta 6D -> abs 6D -> abs rpy
            "action_right_ee_rotation_6D",
            lambda d: (
                data_utils.compose_state_and_delta_to_abs_rpy(
                    d["action_right_ee_rotation_6D_relative"],
                    d["state_right_ee_rotation_6D"][0],
                )
                if d.get("action_right_ee_rotation_6D_relative") is not None
                and d.get("state_right_ee_rotation_6D") is not None
                else None
            ),
        )

    def get_agent_pos(self, obs_action_keys=None):
        if obs_action_keys is None:
            obs_action_keys = self.config.train_config["data"]["obs_action_keys"]

        agent_pose_data = []
        for key in obs_action_keys:
            # 移除follow_或master_前缀
            if key.startswith("follow_"):
                key = key.replace("follow_", "")
            elif key.startswith("master_"):
                key = key.replace("master_", "")

            # 添加state_前缀来访问state数据
            state_key = f"state_{key}"

            if state_key in self.data:
                # 使用get方法，会自动处理None值的计算
                value = self.data.get(state_key)
                if value is None:
                    # 如果计算后仍为None，使用零向量
                    agent_pose_data.append(np.zeros((1, dof_dims[key])))
                else:
                    agent_pose_data.append(value)
            else:
                raise ValueError(f"Key {state_key} not found in data")

        agent_pose_data = np.concatenate(agent_pose_data, axis=1)[None]  # (1, 1, D)

        return agent_pose_data

    def get_agent_pos_mask(self, obs_action_keys=None):
        if obs_action_keys is None:
            obs_action_keys = self.config.train_config["data"]["obs_action_keys"]

        agent_pos_mask_data = []
        for key in obs_action_keys:
            # 移除follow_或master_前缀
            if key.startswith("follow_"):
                key = key.replace("follow_", "")
            elif key.startswith("master_"):
                key = key.replace("master_", "")

            # 添加state_前缀来访问state数据
            state_key = f"state_{key}"

            if state_key in self.data:
                # 使用get方法，会自动处理None值的计算
                value = self.data.get(state_key)
                if value is None:
                    agent_pos_mask_data.append(np.zeros((1, dof_dims[key])))
                else:
                    agent_pos_mask_data.append(np.ones((1, dof_dims[key])))
            else:
                raise ValueError(f"Key {state_key} not found in data")

        return np.concatenate(agent_pos_mask_data, axis=1)[None]  # (1, 1, D)

    def save_state_data_with_key(self, value, key):
        # 移除follow_或master_前缀
        key = key.replace("follow_", "")
        key = key.replace("master_", "")

        # if torch, convert to numpy
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()

        if f"state_{key}" not in self.data:  # TODO： joint angle control
            self.logger.warning(f"{key} 不是合法的state key, 不被记录")
            return

        # 对value进行形状检测，预期形状为（1， D）
        if value.shape == (1, dof_dims[key]):
            self.data[f"state_{key}"] = value
        elif value.shape == (1, 1, dof_dims[key]):
            self.data[f"state_{key}"] = value[0]
        elif value.shape == (dof_dims[key],):
            self.data[f"state_{key}"] = value[None]
        else:
            raise ValueError(f"Value shape {value.shape} is not legal")

    def save_action_data_with_key(self, value, key):
        key = key.replace("follow_", "")
        key = key.replace("master_", "")

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()

        if value.shape == (dof_dims[key],):
            self.data[f"action_{key}"] = value[None]
        else:
            self.data[f"action_{key}"] = value

    def save_action_data(
        self, predict_action, predict_action_keys: Optional[List[str]] = None
    ):
        if predict_action_keys is None:
            predict_action_keys = self.config.data_config["predict_action_keys"]

        if isinstance(predict_action, torch.Tensor):
            predict_action = predict_action.detach().cpu().numpy()

        if predict_action.ndim == 3:
            predict_action = predict_action[0]

        dof_start = 0
        for action_key in predict_action_keys:
            action_key = action_key.replace("follow_", "")
            action_key = action_key.replace("master_", "")
            dof_dim = dof_dims[action_key]
            action_key = f"action_{action_key}"
            self.data[action_key] = predict_action[:, dof_start : dof_start + dof_dim]
            dof_start += dof_dim

    # 为了兼容性，提供便捷的属性访问
    @property
    def agent_pos(self):
        return self.get_agent_pos()

    @property
    def agent_pos_mask(self):
        return self.get_agent_pos_mask()

    @property
    def action(self):
        pass  # TODO: support action access
