import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R  # TODO：转换成numba的函数
from collections import deque
import threading

from wall_x.infer.logger import InferLogger


class KeyboardThread(threading.Thread):
    """
    简单的键盘监听线程，提供 stop 和 reset 功能
    """

    def __init__(self):
        self.should_reset = False
        self.should_stop = False
        self.new_instruction_index = None  # 用于存储新的指令索引
        self.logger = InferLogger.get_utils_logger("KeyboardThread")

        super(KeyboardThread, self).__init__(name="keyboard-thread", daemon=True)
        self.show_help()
        self.start()

    def run(self):
        """监听键盘输入"""
        while True:
            try:
                user_input = input().strip().lower()

                if user_input in ["s", "stop"]:
                    self.should_stop = not self.should_stop
                    self.logger.info("[键盘] 停止信号已发送")

                elif user_input in ["r", "reset"]:
                    self.logger.info("[键盘] 执行重置...")
                    self.should_reset = True
                    self.logger.info("[键盘] 重置信号已发送")

                elif user_input.isdigit():
                    # 处理数字输入，切换到对应的指令索引
                    index = int(user_input)
                    self.new_instruction_index = index
                    self.logger.info(f"[键盘] 切换到指令索引: {index}")

                else:
                    self.logger.info(f"[键盘] 收到输入: {user_input}. 没有执行动作.")

            except EOFError:
                break
            except Exception as e:
                self.logger.error(f"[键盘] 错误: {e}")

    def show_help(self):
        self.logger.info(
            "[键盘] 键盘控制: 输入 's' 停止, 'r' 重置, '数字' 切换指令索引"
        )


# 机械臂轨迹参数
ARM_MAX_VELOCITY = 0.02
ARM_EXECUTION_HZ = 20
ARM_MIN_EXECUTION_TIME = 5.0
ARM_MAX_EXECUTION_TIME = 15.0


class UnifiedTrajectoryProcessor:
    """统一轨迹处理器"""

    @staticmethod
    def interpolate_trajectory_batch(trajectories, target_length, smooth=True):
        """
        批量插值多个轨迹到统一长度
        Args:
            trajectories: list of np.array, 每个数组shape为(N, D)
            target_length: int, 目标长度
            smooth: bool, 是否平滑
        Returns:
            list of np.array, 插值后的轨迹
        """
        if not trajectories:
            return []

        results = []
        for traj in trajectories:
            if len(traj) == 0:
                results.append(np.zeros((target_length, traj.shape[1])))
                continue

            if len(traj) == target_length:
                results.append(traj)
                continue

            # 向量化插值
            original_indices = np.linspace(0, len(traj) - 1, len(traj))
            target_indices = np.linspace(0, len(traj) - 1, target_length)

            # 处理不同类型的数据
            if traj.shape[1] == 7:  # 机械臂数据 [x,y,z,rx,ry,rz,gripper]
                interpolated = UnifiedTrajectoryProcessor._interpolate_arm_trajectory(
                    traj, original_indices, target_indices, target_length
                )
            else:  # 其他数据(高度、电流等)
                interpolated = np.zeros((target_length, traj.shape[1]))
                for i in range(traj.shape[1]):
                    interpolated[:, i] = np.interp(
                        target_indices, original_indices, traj[:, i]
                    )

            # 平滑处理
            if smooth and len(interpolated) >= 5:
                interpolated = UnifiedTrajectoryProcessor._smooth_trajectory(
                    interpolated
                )

            results.append(interpolated)

        return results

    @staticmethod
    def _interpolate_arm_trajectory(
        traj, original_indices, target_indices, target_length
    ):
        """优化的机械臂轨迹插值"""
        interpolated = np.zeros((target_length, 7))

        # 向量化插值位置和夹爪
        for i in [0, 1, 2, 6]:  # x, y, z, gripper
            interpolated[:, i] = np.interp(target_indices, original_indices, traj[:, i])

        # 四元数插值(向量化)
        quaternions = R.from_euler("xyz", traj[:, 3:6]).as_quat()
        interpolated_quats = np.zeros((target_length, 4))
        for i in range(4):
            interpolated_quats[:, i] = np.interp(
                target_indices, original_indices, quaternions[:, i]
            )

        # 批量归一化
        norms = np.linalg.norm(interpolated_quats, axis=1, keepdims=True)
        interpolated_quats = interpolated_quats / norms

        # 批量转换回欧拉角
        interpolated[:, 3:6] = R.from_quat(interpolated_quats).as_euler("xyz")

        return interpolated

    @staticmethod
    def _interpolate_position_trajectory(
        traj, original_indices, target_indices, target_length
    ):
        """优化的位置轨迹插值"""
        interpolated = np.zeros((target_length, 3))
        for i in range(3):
            interpolated[:, i] = np.interp(target_indices, original_indices, traj[:, i])
        return interpolated

    @staticmethod
    def _smooth_trajectory(trajectory):
        """向量化平滑处理"""
        if len(trajectory) < 5:
            return trajectory

        try:
            # 批量平滑所有维度
            smoothed = np.zeros_like(trajectory)
            for dim in range(trajectory.shape[1]):
                smoothed[:, dim] = savgol_filter(
                    trajectory[:, dim],
                    min(
                        5,
                        (
                            len(trajectory)
                            if len(trajectory) % 2 == 1
                            else len(trajectory) - 1
                        ),
                    ),
                    3,
                    mode="nearest",
                )
            return smoothed
        except Exception:
            return trajectory

    @staticmethod
    def calculate_optimal_trajectory_length(left_traj, right_traj):
        """计算最优轨迹长度"""

        # 向量化距离计算
        def calc_distance(traj):
            if len(traj) < 2:
                return 0.0
            pos_diff = traj[1:, :3] - traj[:-1, :3]
            return np.sum(np.linalg.norm(pos_diff, axis=1))

        distances = [calc_distance(left_traj), calc_distance(right_traj)]
        max_distance = max(distances)

        if max_distance > 1e-6:
            execution_time = np.clip(
                max_distance / ARM_MAX_VELOCITY,
                ARM_MIN_EXECUTION_TIME,
                ARM_MAX_EXECUTION_TIME,
            )
        else:
            execution_time = ARM_MIN_EXECUTION_TIME

        return max(int(execution_time * ARM_EXECUTION_HZ), len(left_traj))


class VehiclePoseHandler:
    """车辆位姿和速度计算"""

    def __init__(self):
        self.current_pose = None
        self.previous_pose = None
        self.pose_history = deque(maxlen=10)

    def update_pose(self, new_pose):
        """更新车辆位姿"""
        if new_pose is not None:
            self.previous_pose = self.current_pose
            self.current_pose = np.array(new_pose)
            self.pose_history.append(self.current_pose.copy())
            print("current_pose", self.current_pose, flush=True)
        return self.current_pose

    def velocity_to_pose(self, vx_body, vy_body, vyaw, dt, start_pose=None):
        """将本体坐标系速度转换为全局坐标系位置"""
        if start_pose is None:
            if self.current_pose is not None:
                start_pose = self.current_pose.copy()
            else:
                start_pose = np.array([0.0, 0.0, 0.0])

        x, y, theta = start_pose

        # 本体坐标系速度转换为全局坐标系位移
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # 坐标变换：本体坐标系 -> 全局坐标系
        dx_global = (vx_body * cos_theta - vy_body * sin_theta) * dt
        dy_global = (vx_body * sin_theta + vy_body * cos_theta) * dt
        dtheta = vyaw * dt

        # 计算新位置
        x_new = x + dx_global
        y_new = y + dy_global
        theta_new = theta + dtheta

        # 将角度限制在[-pi, pi]范围内
        theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi

        return np.array([x_new, y_new, theta_new])

    def compute_body_velocities_from_poses(
        self, current_pose, previous_pose, dt=1 / 20
    ):
        """从位姿变化计算本体坐标系速度"""
        if current_pose is None or previous_pose is None:
            return np.array([0.0, 0.0, 0.0])

        # 计算全局坐标系下的位移
        dx_global = current_pose[0] - previous_pose[0]
        dy_global = current_pose[1] - previous_pose[1]
        dtheta = current_pose[2] - previous_pose[2]

        # 使用前一帧的角度进行坐标变换
        theta = previous_pose[2]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # 全局坐标系位移转换为本体坐标系速度
        vx_body = (dx_global * cos_theta + dy_global * sin_theta) / dt
        vy_body = (-dx_global * sin_theta + dy_global * cos_theta) / dt
        vyaw = dtheta / dt

        return np.array([vx_body, vy_body, vyaw])
