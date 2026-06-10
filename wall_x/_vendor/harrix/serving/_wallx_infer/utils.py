import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R  # TODO: convert to numba
from collections import deque
import threading

from wall_x._vendor.harrix.serving._wallx_infer.logger import InferLogger


class KeyboardThread(threading.Thread):
    """
    Simple keyboard listener thread with stop and reset
    """

    def __init__(self):
        self.should_reset = False
        self.should_stop = False
        self.new_instruction_index = None  # Stores new instruction index
        self.logger = InferLogger.get_utils_logger("KeyboardThread")

        super(KeyboardThread, self).__init__(name="keyboard-thread", daemon=True)
        self.show_help()
        self.start()

    def run(self):
        """Listen for keyboard input"""
        while True:
            try:
                user_input = input().strip().lower()

                if user_input in ["s", "stop"]:
                    self.should_stop = not self.should_stop
                    self.logger.info("[keyboard] stop signal sent")

                elif user_input in ["r", "reset"]:
                    self.logger.info("[keyboard] resetting...")
                    self.should_reset = True
                    self.logger.info("[keyboard] reset signal sent")

                elif user_input.isdigit():
                    # Numeric input: switch instruction index
                    index = int(user_input)
                    self.new_instruction_index = index
                    self.logger.info(f"[keyboard] switched to instruction index: {index}")

                else:
                    self.logger.info(f"[keyboard] input received: {user_input}. no action taken.")

            except EOFError:
                break
            except Exception as e:
                self.logger.error(f"[keyboard] error: {e}")

    def show_help(self):
        self.logger.info(
            "[keyboard] controls: 's' stop, 'r' reset, digit switches instruction index"
        )


# Arm trajectory parameters
ARM_MAX_VELOCITY = 0.02
ARM_EXECUTION_HZ = 20
ARM_MIN_EXECUTION_TIME = 5.0
ARM_MAX_EXECUTION_TIME = 15.0


class UnifiedTrajectoryProcessor:
    """Unified trajectory processor"""

    @staticmethod
    def interpolate_trajectory_batch(trajectories, target_length, smooth=True):
        """
        Interpolate multiple trajectories to a common length
        Args:
            trajectories: list of np.array, each shape (N, D)
            target_length: int, target length
            smooth: bool, whether to smooth
        Returns:
            list of np.array, interpolated trajectories
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

            # Vectorized interpolation
            original_indices = np.linspace(0, len(traj) - 1, len(traj))
            target_indices = np.linspace(0, len(traj) - 1, target_length)

            # Handle different data types
            if traj.shape[1] == 7:  # Arm data [x,y,z,rx,ry,rz,gripper]
                interpolated = UnifiedTrajectoryProcessor._interpolate_arm_trajectory(
                    traj, original_indices, target_indices, target_length
                )
            else:  # Other data (height, current, etc.)
                interpolated = np.zeros((target_length, traj.shape[1]))
                for i in range(traj.shape[1]):
                    interpolated[:, i] = np.interp(
                        target_indices, original_indices, traj[:, i]
                    )

            # Smoothing
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
        """Optimized arm trajectory interpolation"""
        interpolated = np.zeros((target_length, 7))

        # Vectorized interp for position and gripper
        for i in [0, 1, 2, 6]:  # x, y, z, gripper
            interpolated[:, i] = np.interp(target_indices, original_indices, traj[:, i])

        # Quaternion interpolation (vectorized)
        quaternions = R.from_euler("xyz", traj[:, 3:6]).as_quat()
        interpolated_quats = np.zeros((target_length, 4))
        for i in range(4):
            interpolated_quats[:, i] = np.interp(
                target_indices, original_indices, quaternions[:, i]
            )

        # Batch normalize
        norms = np.linalg.norm(interpolated_quats, axis=1, keepdims=True)
        interpolated_quats = interpolated_quats / norms

        # Batch convert back to euler
        interpolated[:, 3:6] = R.from_quat(interpolated_quats).as_euler("xyz")

        return interpolated

    @staticmethod
    def _interpolate_position_trajectory(
        traj, original_indices, target_indices, target_length
    ):
        """Optimized position trajectory interpolation"""
        interpolated = np.zeros((target_length, 3))
        for i in range(3):
            interpolated[:, i] = np.interp(target_indices, original_indices, traj[:, i])
        return interpolated

    @staticmethod
    def _smooth_trajectory(trajectory):
        """Vectorized smoothing"""
        if len(trajectory) < 5:
            return trajectory

        try:
            # Smooth all dimensions in batch
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
        """Compute optimal trajectory length"""

        # Vectorized distance computation
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
    """Vehicle pose and velocity computation"""

    def __init__(self):
        self.current_pose = None
        self.previous_pose = None
        self.pose_history = deque(maxlen=10)
        self.logger = InferLogger.get_utils_logger("VehiclePoseHandler")

    def update_pose(self, new_pose):
        """Update vehicle pose"""
        if new_pose is not None:
            self.previous_pose = self.current_pose
            self.current_pose = np.array(new_pose)
            self.pose_history.append(self.current_pose.copy())
            self.logger.info("current_pose %s", self.current_pose)
        return self.current_pose

    def velocity_to_pose(self, vx_body, vy_body, vyaw, dt, start_pose=None):
        """Convert body-frame velocity to global pose"""
        if start_pose is None:
            if self.current_pose is not None:
                start_pose = self.current_pose.copy()
            else:
                start_pose = np.array([0.0, 0.0, 0.0])

        x, y, theta = start_pose

        # Body velocity to global displacement
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Transform: body frame -> global frame
        dx_global = (vx_body * cos_theta - vy_body * sin_theta) * dt
        dy_global = (vx_body * sin_theta + vy_body * cos_theta) * dt
        dtheta = vyaw * dt

        # Compute new pose
        x_new = x + dx_global
        y_new = y + dy_global
        theta_new = theta + dtheta

        # Wrap angle to [-pi, pi]
        theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi

        return np.array([x_new, y_new, theta_new])

    def compute_body_velocities_from_poses(
        self, current_pose, previous_pose, dt=1 / 20
    ):
        """Compute body-frame velocity from pose delta"""
        if current_pose is None or previous_pose is None:
            return np.array([0.0, 0.0, 0.0])

        # Global-frame displacement
        dx_global = current_pose[0] - previous_pose[0]
        dy_global = current_pose[1] - previous_pose[1]
        dtheta = current_pose[2] - previous_pose[2]

        # Use previous frame angle for transform
        theta = previous_pose[2]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Global displacement to body velocity
        vx_body = (dx_global * cos_theta + dy_global * sin_theta) / dt
        vy_body = (-dx_global * sin_theta + dy_global * cos_theta) / dt
        vyaw = dtheta / dt

        return np.array([vx_body, vy_body, vyaw])
