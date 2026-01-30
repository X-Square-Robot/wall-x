import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R  # TODO: Convert to numba functions
from collections import deque
import threading

from wall_x.infer.logger import InferLogger


class KeyboardThread(threading.Thread):
    """
    Simple keyboard listening thread that provides stop and reset functionality
    """

    def __init__(self):
        self.should_reset = False
        self.should_stop = False
        self.new_instruction_index = None  # Used to store new instruction index
        self.logger = InferLogger.get_utils_logger("KeyboardThread")

        super(KeyboardThread, self).__init__(name="keyboard-thread", daemon=True)
        self.show_help()
        self.start()

    def run(self):
        """Listen to keyboard input"""
        while True:
            try:
                user_input = input().strip().lower()

                if user_input in ["s", "stop"]:
                    self.should_stop = not self.should_stop
                    self.logger.info("[Keyboard] Stop signal sent")

                elif user_input in ["r", "reset"]:
                    self.logger.info("[Keyboard] Executing reset...")
                    self.should_reset = True
                    self.logger.info("[Keyboard] Reset signal sent")

                elif user_input.isdigit():
                    # Handle digit input, switch to corresponding instruction index
                    index = int(user_input)
                    self.new_instruction_index = index
                    self.logger.info(f"[Keyboard] Switched to instruction index: {index}")

                else:
                    self.logger.info(f"[Keyboard] Received input: {user_input}. No action taken.")

            except EOFError:
                break
            except Exception as e:
                self.logger.error(f"[Keyboard] Error: {e}")

    def show_help(self):
        self.logger.info(
            "[Keyboard] Keyboard control: Enter 's' to stop, 'r' to reset, 'number' to switch instruction index"
        )


# Robot arm trajectory parameters
ARM_MAX_VELOCITY = 0.02
ARM_EXECUTION_HZ = 20
ARM_MIN_EXECUTION_TIME = 5.0
ARM_MAX_EXECUTION_TIME = 15.0


class UnifiedTrajectoryProcessor:
    """Unified trajectory processor"""

    @staticmethod
    def interpolate_trajectory_batch(trajectories, target_length, smooth=True):
        """
        Batch interpolate multiple trajectories to unified length
        Args:
            trajectories: list of np.array, each array with shape (N, D)
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

            # Handle different types of data
            if traj.shape[1] == 7:  # Robot arm data [x,y,z,rx,ry,rz,gripper]
                interpolated = UnifiedTrajectoryProcessor._interpolate_arm_trajectory(
                    traj, original_indices, target_indices, target_length
                )
            else:  # Other data (height, current, etc.)
                interpolated = np.zeros((target_length, traj.shape[1]))
                for i in range(traj.shape[1]):
                    interpolated[:, i] = np.interp(
                        target_indices, original_indices, traj[:, i]
                    )

            # Smooth processing
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
        """Optimized robot arm trajectory interpolation"""
        interpolated = np.zeros((target_length, 7))

        # Vectorized interpolation for position and gripper
        for i in [0, 1, 2, 6]:  # x, y, z, gripper
            interpolated[:, i] = np.interp(target_indices, original_indices, traj[:, i])

        # Quaternion interpolation (vectorized)
        quaternions = R.from_euler("xyz", traj[:, 3:6]).as_quat()
        interpolated_quats = np.zeros((target_length, 4))
        for i in range(4):
            interpolated_quats[:, i] = np.interp(
                target_indices, original_indices, quaternions[:, i]
            )

        # Batch normalization
        norms = np.linalg.norm(interpolated_quats, axis=1, keepdims=True)
        interpolated_quats = interpolated_quats / norms

        # Batch convert back to Euler angles
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
        """Vectorized smooth processing"""
        if len(trajectory) < 5:
            return trajectory

        try:
            # Batch smooth all dimensions
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
        """Calculate optimal trajectory length"""

        # Vectorized distance calculation
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
    """Vehicle pose and velocity calculation"""

    def __init__(self):
        self.current_pose = None
        self.previous_pose = None
        self.pose_history = deque(maxlen=10)

    def update_pose(self, new_pose):
        """Update vehicle pose"""
        if new_pose is not None:
            self.previous_pose = self.current_pose
            self.current_pose = np.array(new_pose)
            self.pose_history.append(self.current_pose.copy())
            print("current_pose", self.current_pose, flush=True)
        return self.current_pose

    def velocity_to_pose(self, vx_body, vy_body, vyaw, dt, start_pose=None):
        """Convert body frame velocity to global frame position"""
        if start_pose is None:
            if self.current_pose is not None:
                start_pose = self.current_pose.copy()
            else:
                start_pose = np.array([0.0, 0.0, 0.0])

        x, y, theta = start_pose

        # Convert body frame velocity to global frame displacement
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Coordinate transformation: body frame -> global frame
        dx_global = (vx_body * cos_theta - vy_body * sin_theta) * dt
        dy_global = (vx_body * sin_theta + vy_body * cos_theta) * dt
        dtheta = vyaw * dt

        # Calculate new position
        x_new = x + dx_global
        y_new = y + dy_global
        theta_new = theta + dtheta

        # Constrain angle to [-pi, pi] range
        theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi

        return np.array([x_new, y_new, theta_new])

    def compute_body_velocities_from_poses(
        self, current_pose, previous_pose, dt=1 / 20
    ):
        """Compute body frame velocity from pose changes"""
        if current_pose is None or previous_pose is None:
            return np.array([0.0, 0.0, 0.0])

        # Calculate displacement in global frame
        dx_global = current_pose[0] - previous_pose[0]
        dy_global = current_pose[1] - previous_pose[1]
        dtheta = current_pose[2] - previous_pose[2]

        # Use previous frame's angle for coordinate transformation
        theta = previous_pose[2]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Convert global frame displacement to body frame velocity
        vx_body = (dx_global * cos_theta + dy_global * sin_theta) / dt
        vy_body = (-dx_global * sin_theta + dy_global * cos_theta) / dt
        vyaw = dtheta / dt

        return np.array([vx_body, vy_body, vyaw])
