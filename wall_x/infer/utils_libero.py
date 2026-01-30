"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os
from enum import Enum
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import BatchFeature
import random
import time

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


# Initialize important constants
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")

# Configure NumPy print settings
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


def set_seed_everywhere(seed: int) -> None:
    """
    Set random seed for all random number generators for reproducibility.

    Args:
        seed: The random seed to use
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """
    Normalize gripper action from [0,1] to [-1,+1] range.

    This is necessary for some environments because the dataset wrapper
    standardizes gripper actions to [0,1]. Note that unlike the other action
    dimensions, the gripper action is not normalized to [-1,+1] by default.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1

    Args:
        action: Action array with gripper action in the last dimension
        binarize: Whether to binarize gripper action to -1 or +1

    Returns:
        np.ndarray: Action array with normalized gripper action
    """
    # Create a copy to avoid modifying the original
    normalized_action = action.copy()

    # Normalize the last action dimension to [-1,+1]
    orig_low, orig_high = 0.0, 1.0
    normalized_action[..., -1] = (
        2 * (normalized_action[..., -1] - orig_low) / (orig_high - orig_low) - 1
    )

    if binarize:
        # Binarize to -1 or +1
        normalized_action[..., -1] = np.sign(normalized_action[..., -1])

    return normalized_action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """
    Flip the sign of the gripper action (last dimension of action vector).

    This is necessary for environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.

    Args:
        action: Action array with gripper action in the last dimension

    Returns:
        np.ndarray: Action array with inverted gripper action
    """
    # Create a copy to avoid modifying the original
    inverted_action = action.copy()

    # Invert the gripper action
    inverted_action[..., -1] *= -1.0

    return inverted_action


def move_to_cuda(obj, device="cuda"):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, (dict, BatchFeature)):
        return {k: move_to_cuda(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_cuda(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_cuda(v, device) for v in obj)
    else:
        return obj


def get_libero_env(task, model_family, resolution=256, seed=7):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(
        seed
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs):
    """Extracts third-person image from observations and preprocesses it."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def get_libero_wrist_image(obs):
    """Extracts wrist camera image from observations and preprocesses it."""
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def save_rollout_video(
    rollout_dir,
    rollout_images,
    idx,
    success,
    task_description,
    log_file=None,
    model_family="openvla_oft",
):
    """Saves an MP4 replay of an episode."""
    processed_task_description = (
        task_description.lower()
        .replace(" ", "_")
        .replace("\n", "_")
        .replace(".", "_")[:50]
    )
    mp4_path = f"{rollout_dir}/episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def save_rollout_data(
    rollout_dir,
    rollout_data,
    idx,
    success,
    task_description,
    log_file=None,
    model_family="openvla_oft",
):
    """
    Saves an NPY file of the rollout data.
    """

    # Process task description to make it suitable for filename
    processed_task_description = (
        task_description.lower()
        .replace(" ", "_")
        .replace("\n", "_")
        .replace(".", "_")[:50]
    )

    # Build .npy file path
    npy_path = f"{rollout_dir}/episode={idx}--success={success}--task={processed_task_description}--action.npy"

    # Save rollout_data as .npy file
    np.save(npy_path, rollout_data)
    print(f"Saved rollout data at path {npy_path}")

    fig, axes = plt.subplots(nrows=1, ncols=7, figsize=(20, 3))

    titles = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

    for i in range(rollout_data.shape[1]):
        ax = axes[i]  # Select the i-th subplot
        ax.plot(rollout_data[:, i], label=f"Feature {i+1}")  # Plot line chart
        ax.set_title(titles[i])  # Set subplot title
        ax.set_xlabel("Time in one episode")  # Set x-axis label

    axes[-1].legend(["predicted action"], loc="upper right")

    plt.tight_layout()
    png_path = f"{rollout_dir}/episode={idx}--success={success}--task={processed_task_description}--action.png"
    plt.savefig(png_path, dpi=300)  # Save image as PNG file

    # If log file is provided, record the save path
    if log_file is not None:
        log_file.write(f"Saved rollout data at path {npy_path}\n")

    return npy_path


def save_rollout_observation(
    rollout_dir,
    rollout_data,
    idx,
    success,
    task_description,
    log_file=None,
    model_family="openvla_oft",
):
    """
    Saves an NPY file of the rollout data.
    """

    # Process task description to make it suitable for filename
    processed_task_description = (
        task_description.lower()
        .replace(" ", "_")
        .replace("\n", "_")
        .replace(".", "_")[:50]
    )

    # Build .npy file path
    npy_path = f"{rollout_dir}/episode={idx}--success={success}--task={processed_task_description}--observation.npy"

    # Save rollout_data as .npy file
    np.save(npy_path, rollout_data)
    print(f"Saved rollout data at path {npy_path}")

    if rollout_data.shape[1] == 7:
        fig, axes = plt.subplots(nrows=1, ncols=7, figsize=(20, 3))
        titles = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]
    else:
        fig, axes = plt.subplots(nrows=1, ncols=8, figsize=(20, 3))
        titles = ["x", "y", "z", "roll", "pitch", "yaw", "-", "grasp"]

    for i in range(rollout_data.shape[1]):
        ax = axes[i]  # Select the i-th subplot
        ax.plot(rollout_data[:, i], label=f"Feature {i+1}")  # Plot line chart
        ax.set_title(titles[i])  # Set subplot title
        ax.set_xlabel("Time in one episode")  # Set x-axis label

    axes[-1].legend(["predicted action"], loc="upper right")

    plt.tight_layout()
    png_path = f"{rollout_dir}/episode={idx}--success={success}--task={processed_task_description}--observation.png"
    plt.savefig(png_path, dpi=300)  # Save image as PNG file

    # If log file is provided, record the save path
    if log_file is not None:
        log_file.write(f"Saved rollout data at path {npy_path}\n")

    return npy_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
