"""
Convert x2robot internal format to LeRobotDataset v3.0.

Requires ``lerobot >= 0.3`` with ``CODEBASE_VERSION "v3.0"``.
After conversion, ``dataset.finalize()`` is called in ``main()``; call it before
loading the dataset elsewhere.

Usage
-----
1. Install dependencies::

       pip install "lerobot>=0.3" numpy pandas torch torchvision scipy numba

2. Run from the repo root. All dataset-specific settings are passed via CLI or JSON
   (no hardcoded paths or episode schema in this script).

   Single JSON config (recommended)::

       python x2robot2lerobot.py --config path/to/conversion.json

   Example ``conversion.json``::

       {
         "output_path": "/path/to/lerobot_output",
         "repo_id": "my_dataset",
         "src_path_list": ["/path/to/collection1"],
         "fps": 20,
         "features": { "...": "LeRobot v3.0 feature schema" },
         "episode_template": {
           "cam_mapping": {"faceImg": "face_view", "leftImg": "left_wrist_view"},
           "type": "x2_normal",
           "predict_action_keys": ["follow_left_ee_cartesian_pos", "..."],
           "obs_action_keys": ["follow_left_ee_cartesian_pos", "..."]
         }
       }

   Or pass the same fields via CLI flags (``--features`` / ``--episode-config``
   point to JSON files; see ``--help``).

Input layout (each entry in ``src_path_list``)::

    /path/to/collection/
        instruction.json          # { "<episode_folder_name>": { "instruction": "..." }, ... }
        <episode_name>/           # one folder per episode (``record/`` is skipped)
            <episode_name>.json   # raw action JSON with a top-level ``data`` list
            faceImg.mp4           # camera files; stems must match ``cam_mapping`` keys
            leftImg.mp4
            rightImg.mp4

Output::

    <output_path>/              # LeRobot v3.0 dataset (meta/, data/, videos/)
        ...

Notes
-----
- Videos are copied as-is (no re-encode); action/state are written from JSON.
- Rotation keys in ``predict_action_keys`` / ``obs_action_keys`` are converted to
  6D rotation (first two rows of the rotation matrix).
- Action keys must exist in ``_ACTION_KEY_FULL_MAPPING`` or be added there first.
"""

import argparse
import json
import logging
import os
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchvision
from lerobot.datasets.compute_stats import compute_episode_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import validate_episode_buffer
from numba import jit, prange
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)

_REQUIRED_EPISODE_TEMPLATE_KEYS = (
    "cam_mapping",
    "predict_action_keys",
    "obs_action_keys",
)


def _load_json_file(path: str | Path, label: str) -> Any:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


@jit(nopython=True, parallel=True)
def euler_to_matrix_zyx_6d_nb(eulers):
    """
    Numba: euler angles (N, 3) -> first two rows flattened (N, 6)
    """
    N = eulers.shape[0]
    R6 = np.empty((N, 6), dtype=np.float64)
    for i in prange(N):
        roll = eulers[i, 0]
        pitch = eulers[i, 1]
        yaw = eulers[i, 2]

        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        r00 = cy * cp
        r01 = cy * sp * sr - sy * cr
        r02 = cy * sp * cr + sy * sr

        r10 = sy * cp
        r11 = sy * sp * sr + cy * cr
        r12 = sy * sp * cr - cy * sr

        R6[i, 0] = r00
        R6[i, 1] = r01
        R6[i, 2] = r02
        R6[i, 3] = r10
        R6[i, 4] = r11
        R6[i, 5] = r12
    return R6


_DOF_DIM_MAPPING = {
    "follow_left_position": 3,
    "follow_left_rotation": 3,
    "follow_left_gripper": 1,
    "follow_right_position": 3,
    "follow_right_rotation": 3,
    "follow_right_gripper": 1,
    "head_rotation": 2,
    "lifting_mechanism_position": 1,
    "car_pose": 3,
}


# Unified action key mapping: model key -> raw data key
_ACTION_KEY_FULL_MAPPING = {
    # ARX arm series
    "follow_right_arm_joint_pos": "follow_right_joint_pos",
    "follow_right_arm_joint_dev": "follow_right_joint_dev",
    "follow_right_arm_joint_cur": "follow_right_joint_cur",
    "follow_right_ee_cartesian_pos": "follow_right_position",
    "follow_right_ee_rotation": "follow_right_rotation",
    "follow_right_gripper": "follow_right_gripper",
    "master_right_arm_joint_pos": "master_right_joint_pos",
    "master_right_arm_joint_dev": "master_right_joint_dev",
    "master_right_arm_joint_cur": "master_right_joint_cur",
    "master_right_ee_cartesian_pos": "master_right_position",
    "master_right_ee_rotation": "master_right_rotation",
    "master_right_gripper": "master_right_gripper",
    "follow_left_arm_joint_pos": "follow_left_joint_pos",
    "follow_left_arm_joint_dev": "follow_left_joint_dev",
    "follow_left_arm_joint_cur": "follow_left_joint_cur",
    "follow_left_ee_cartesian_pos": "follow_left_position",
    "follow_left_ee_rotation": "follow_left_rotation",
    "follow_left_gripper": "follow_left_gripper",
    "master_left_arm_joint_pos": "master_left_joint_pos",
    "master_left_arm_joint_dev": "master_left_joint_dev",
    "master_left_arm_joint_cur": "master_left_joint_cur",
    "master_left_ee_cartesian_pos": "master_left_position",
    "master_left_ee_rotation": "master_left_rotation",
    "master_left_gripper": "master_left_gripper",
    # JAKA arm series - raw data mapping
    "follow_left_ee_cartesian_pos_jaka": "follow_left_position",
    "follow_right_ee_cartesian_pos_jaka": "follow_right_position",
    "follow_left_ee_rotation_jaka": "follow_left_rotation",
    "follow_right_ee_rotation_jaka": "follow_right_rotation",
    "follow_left_arm_joint_pos_jaka": "follow_left_joint_pos",
    "follow_right_arm_joint_pos_jaka": "follow_right_joint_pos",
    "follow_left_arm_joint_cur_jaka": "follow_left_joint_cur",
    "follow_right_arm_joint_cur_jaka": "follow_right_joint_cur",
    # Hand control
    "follow_left_hand_joint_pos": "follow_left_hand_joint_pos",
    "follow_left_hand_joint_dev": "follow_left_hand_joint_dev",
    "follow_right_hand_joint_pos": "follow_right_hand_joint_pos",
    "follow_right_hand_joint_dev": "follow_right_hand_joint_dev",
    # Gripper force - last joint of arm_joint_cur
    "follow_left_gripper_cur": "follow_left_joint_cur[-1]",
    "follow_right_gripper_cur": "follow_right_joint_cur[-1]",
    # Other
    "base_movement": "base_movement",
    "car_pose": "car_pose",
    "velocity_decomposed": "velocity_decomposed",
    "head_actions": "head_rotation",
    "height": "lifting_mechanism_position",
}


_ACTION_KEY_FULL_MAPPING_INV = {v: k for k, v in _ACTION_KEY_FULL_MAPPING.items()}


def process_action(
    file_path,
    raw_key2model_key=_ACTION_KEY_FULL_MAPPING_INV,
    filter_angle_outliers=True,
    fps=20,
):
    # ======== Step 1: normalize input type ========
    mappings = raw_key2model_key
    # Wrap a single dict in a one-element list for uniform handling
    if isinstance(mappings, dict):
        mappings = [mappings]
        return_list = False
    else:
        return_list = True

    # ======== Step 2: load raw action data ========
    if isinstance(file_path, str):
        file_name = os.path.basename(file_path)
        action_path = os.path.join(file_path, f"{file_name}.json")
        with open(action_path, "r") as file:
            actions = json.load(file)
    else:
        actions = file_path

    data = actions.get("data", [])
    if not data:
        raise ValueError("No 'data' field found in action file")

    # ======== Step 3: pre-compile regex ========
    index_pattern = re.compile(r"^(.+)\[(-?\d+)\]$")

    # ======== Step 4: collect all raw keys needed for mappings ========
    all_raw_keys = set()
    special_mappings_info = []  # metadata for indexed/special mappings

    for mapping_idx, mapping_dict in enumerate(mappings):
        for raw_key in mapping_dict:
            # Handle string keys (may include index suffix)
            if isinstance(raw_key, str):
                match = index_pattern.match(raw_key)
                if match:
                    base_key = match.group(1)
                    all_raw_keys.add(base_key)
                    # Record special mapping metadata
                    special_mappings_info.append(
                        {
                            "mapping_idx": mapping_idx,
                            "base_raw_key": base_key,
                            "index": int(match.group(2)),
                            "model_key": mapping_dict[raw_key],
                            "original_key": raw_key,
                        }
                    )
                else:
                    all_raw_keys.add(raw_key)
            # Handle non-string keys
            else:
                all_raw_keys.add(raw_key)

    # ======== Step 5: aggregate raw data ========
    unknown_keys = all_raw_keys - _DOF_DIM_MAPPING.keys()
    if unknown_keys:
        logger.warning(
            "Unknown keys not in _DOF_DIM_MAPPING, skipping: %s",
            sorted(unknown_keys),
        )
    keys_to_aggregate = {key for key in all_raw_keys if key in _DOF_DIM_MAPPING}

    aggregated_raw_data = defaultdict(list)
    for action in data:
        for key in keys_to_aggregate:
            if key in action:
                aggregated_raw_data[key].append(action[key])
            else:
                # Pad missing values with NaN for consistent array shapes
                aggregated_raw_data[key].append(
                    [float("nan") for _ in range(_DOF_DIM_MAPPING[key])]
                )

    # Convert to NumPy arrays
    for key in aggregated_raw_data:
        aggregated_raw_data[key] = np.array(aggregated_raw_data[key], dtype=np.float32)

    # ======== Step 6: pre-allocate trajectory dicts ========
    trajectories_list = [dict() for _ in range(len(mappings))]

    # ======== Step 7: apply plain key mappings ========
    for mapping_idx, mapping_dict in enumerate(mappings):
        for raw_key, model_key in mapping_dict.items():
            # Skip indexed/special mapping keys
            if isinstance(raw_key, str) and index_pattern.match(raw_key):
                continue

            if raw_key in aggregated_raw_data:
                # Clone to avoid in-place mutation
                trajectories_list[mapping_idx][model_key] = aggregated_raw_data[
                    raw_key
                ].copy()

    # ======== Step 8: apply indexed/special mappings ========
    for info in special_mappings_info:
        base_key = info["base_raw_key"]
        idx = info["mapping_idx"]

        if base_key in aggregated_raw_data:
            data_arr = aggregated_raw_data[base_key]
            # Only handle 2D arrays (N x M)
            if data_arr.ndim == 2:
                # Handle negative indices
                if info["index"] < 0:
                    target_index = data_arr.shape[1] + info["index"]
                else:
                    target_index = info["index"]

                # Extract column by index safely
                if 0 <= target_index < data_arr.shape[1]:
                    trajectories_list[idx][info["model_key"]] = data_arr[
                        :, target_index : target_index + 1
                    ]
                else:
                    logger.warning(
                        "Invalid index %s for key %s",
                        info["index"],
                        info["original_key"],
                    )
            else:
                logger.warning("Base key %s is not 2D array for index access", base_key)
        else:
            logger.warning("Base key %s not found in raw data", base_key)

    # ======== Step 9: post-processing ========
    final_trajectories = []
    for traj in trajectories_list:
        # Quaternion -> euler angles
        traj = quat2euler(traj)
        # Compute base velocity from car_pose
        traj = calculate_base_velocity(traj, fps=fps)
        if filter_angle_outliers:
            processed = smooth_action(traj)
            final_trajectories.append(processed)
        else:
            final_trajectories.append(traj)

    # ======== Step 10: return results ========
    return final_trajectories if return_list else final_trajectories[0]


def quat2euler(traj):
    if (
        "follow_right_ee_rotation" in traj
        and traj["follow_right_ee_rotation"].shape[-1] == 4
    ):
        traj["follow_right_ee_rotation"] = R.from_quat(
            traj["follow_right_ee_rotation"]
        ).as_euler("xyz")
    if (
        "follow_left_ee_rotation" in traj
        and traj["follow_left_ee_rotation"].shape[-1] == 4
    ):
        traj["follow_left_ee_rotation"] = R.from_quat(
            traj["follow_left_ee_rotation"]
        ).as_euler("xyz")
    if (
        "master_right_ee_rotation" in traj
        and traj["master_right_ee_rotation"].shape[-1] == 4
    ):
        traj["master_right_ee_rotation"] = R.from_quat(
            traj["master_right_ee_rotation"]
        ).as_euler("xyz")
    if (
        "master_left_ee_rotation" in traj
        and traj["master_left_ee_rotation"].shape[-1] == 4
    ):
        traj["master_left_ee_rotation"] = R.from_quat(
            traj["master_left_ee_rotation"]
        ).as_euler("xyz")
    return traj


def calculate_base_velocity(traj, fps=20):
    if (
        "car_pose" in traj
        and traj["car_pose"].shape[-1] == 3
        and not np.isnan(traj["car_pose"]).any()
    ):
        velocity = process_car_pose_to_base_velocity(traj["car_pose"], fps=fps)
        if velocity["valid"]:
            traj["car_pose"] = velocity["base_velocity_decomposed"]
        else:
            logger.warning("car_pose is invalid, dropping key")
            traj.pop("car_pose")
    return traj


def smooth_action(action):
    def _filter(traj, threshold=3, alpha=0.05, window=10):
        # Convert to pandas Series but preserve the original dtype
        orig_dtype = traj.dtype
        data = pd.Series(traj)
        derivatives = np.diff(data)

        spike_indices = np.where(abs(derivatives) > threshold)[0]
        if len(spike_indices) > 0:
            ema = data.ewm(alpha=alpha, adjust=True).mean()

            # Fix: Ensure the slice indices are within bounds
            start_idx = max(0, spike_indices[0] - window)
            end_idx = min(len(data), spike_indices[-1] + window + 1)

            # Get the corresponding segment from the EMA
            modified_seg = ema.iloc[start_idx:end_idx]

            # Ensure the lengths match before assignment and explicitly convert to the original dtype
            if len(modified_seg) > 0:
                # Convert values back to the original dtype before assignment
                data.iloc[start_idx:end_idx] = modified_seg.values.astype(orig_dtype)

        return data.to_numpy().astype(orig_dtype)  # Ensure we return the same dtype

    for key in ["follow_right_ee_rotation", "follow_left_ee_rotation"]:
        if key in action:  # Check if the key exists in the action dictionary
            try:
                # Process each dimension separately while preserving dtype
                orig_dtype = action[key].dtype
                filtered_traj = np.stack(
                    [_filter(action[key][:, i]) for i in range(3)], axis=1
                )
                if not np.isnan(filtered_traj).any():
                    action[key] = filtered_traj.astype(
                        orig_dtype
                    )  # Ensure consistent dtype
            except (IndexError, ValueError) as e:
                logger.warning("Could not smooth %s: %s", key, e)

    return action


def process_car_pose_to_base_velocity(
    car_pose,
    outlier_threshold=3,
    jump_threshold=1.0,
    smooth_iterations=3,
    strong_smooth=True,
    fps=20,
):
    """
    Process car_pose into body-frame base_velocity_decomposed (matches batch_process_json_data.py).

        Includes outlier removal, angle unwrap, jump correction, smoothing, and body-frame velocity.

        Args:
            car_pose: Input array, shape (n, 3) [x, y, angle].
            outlier_threshold: Z-score threshold for outliers (default 3).
            jump_threshold: Jump detection threshold (default 1.0).
            smooth_iterations: Number of smoothing passes (default 3).
            strong_smooth: Use stronger smoothing when True (default True).
            fps: Sampling rate in Hz for velocity computation (default 20).

        Returns:
            dict with:
                - 'base_velocity_decomposed': shape (n, 3) [vx_body, vy_body, vyaw] in body frame
                - 'valid': bool, whether velocities pass range checks
    """
    # Velocity limits (same as data_analysis_filter.py)
    velocity_limits = {
        "vx": {"min": -0.5, "max": 0.5},
        "vy": {"min": -0.5, "max": 0.5},
        "vyaw": {"min": -1.6, "max": 1.6},
    }

    # Handle empty or single-point input
    if len(car_pose) == 0:
        return {"base_velocity_decomposed": np.zeros((0, 3)), "valid": False}

    if len(car_pose) == 1:
        return {
            "base_velocity_decomposed": np.zeros((1, 3)),
            "valid": True,  # single point treated as valid
        }

    # Step 1: extract position/angle and unwrap angles
    x_values = car_pose[:, 0].copy()
    y_values = car_pose[:, 1].copy()
    angle_values = car_pose[:, 2].copy()

    # Unwrap angles to avoid pi jumps
    angle_values_unwrapped = np.unwrap(angle_values)

    # Steps 2-4: outlier removal, jump correction, smoothing
    # Outlier removal
    x_filtered = remove_outliers(x_values, outlier_threshold)
    y_filtered = remove_outliers(y_values, outlier_threshold)
    angle_filtered = remove_outliers(angle_values_unwrapped, outlier_threshold)

    # Jump correction
    x_filtered = remove_jumps(x_filtered, jump_threshold)
    y_filtered = remove_jumps(y_filtered, jump_threshold)
    angle_filtered = remove_jumps(angle_filtered, jump_threshold)

    # Smoothing
    window_length = min(51 if strong_smooth else 21, len(x_filtered) - 1)
    if window_length % 2 == 0:
        window_length -= 1
    window_length = max(3, window_length)

    x_smooth = smooth_data(
        x_filtered,
        window_length,
        polyorder=2 if strong_smooth else 3,
        iterations=smooth_iterations,
        strong_smooth=strong_smooth,
    )
    y_smooth = smooth_data(
        y_filtered,
        window_length,
        polyorder=2 if strong_smooth else 3,
        iterations=smooth_iterations,
        strong_smooth=strong_smooth,
    )
    angle_smooth = smooth_data(
        angle_filtered,
        window_length,
        polyorder=2 if strong_smooth else 3,
        iterations=smooth_iterations,
        strong_smooth=strong_smooth,
    )

    # Step 5: body-frame velocity (same as data_processor.py)
    dt = 1.0 / fps

    # Global displacement deltas
    x_diff = np.diff(x_smooth)
    y_diff = np.diff(y_smooth)
    angle_diff = np.diff(angle_smooth)

    # Current heading for coordinate transform
    current_theta = angle_smooth[:-1]  # shape: (n-1,)

    # Transform global deltas to body frame
    cos_theta = np.cos(current_theta)
    sin_theta = np.sin(current_theta)

    # Body-frame velocities
    vx_body = (
        x_diff * cos_theta + y_diff * sin_theta
    ) / dt  # forward velocity (body frame)
    vy_body = (
        -x_diff * sin_theta + y_diff * cos_theta
    ) / dt  # lateral velocity (body frame)
    vyaw = angle_diff / dt  # yaw rate

    # Prepend zero velocity to match original length
    vx_array = np.concatenate([[0], vx_body])
    vy_array = np.concatenate([[0], vy_body])
    vyaw_array = np.concatenate([[0], vyaw])

    base_velocity_decomposed = np.stack([vx_array, vy_array, vyaw_array], axis=1)

    # Step 6: velocity range check (data_analysis_filter.py)
    valid = True

    if (
        abs(x_values).max() > 6
        or abs(y_values).max() > 6
        or abs(angle_values).max() > 6
    ):
        valid = False

    # Check each velocity component against limits
    if valid:
        for vx_val in vx_body:
            if (
                vx_val < velocity_limits["vx"]["min"]
                or vx_val > velocity_limits["vx"]["max"]
            ):
                valid = False
                break

    if valid:  # check vy only if vx passed
        for vy_val in vy_body:
            if (
                vy_val < velocity_limits["vy"]["min"]
                or vy_val > velocity_limits["vy"]["max"]
            ):
                valid = False
                break

    if valid:  # check vyaw only if vx and vy passed
        for vyaw_val in vyaw:
            if (
                vyaw_val < velocity_limits["vyaw"]["min"]
                or vyaw_val > velocity_limits["vyaw"]["max"]
            ):
                valid = False
                break

    return {"base_velocity_decomposed": base_velocity_decomposed, "valid": valid}


def remove_outliers(data, threshold=3):
    """
    Remove outliers via Z-score.

        Args:
            data: Input 1D array.
            threshold: Z-score threshold (default 3).

        Returns:
            Filtered array.
    """
    # Too few points or constant series: return as-is
    if len(data) < 3 or np.all(data == data[0]):
        return data.copy()

    # Compute std; avoid catastrophic cancellation
    std = np.std(data)
    if std < 1e-10:  # near-zero std: return as-is
        return data.copy()

    # Compute Z-scores
    try:
        z_scores = np.abs(stats.zscore(data))
    except (FloatingPointError, ValueError):
        # Cannot compute Z-score: return as-is
        return data.copy()

    filtered_data = data.copy()

    # Flag outliers
    mask = z_scores > threshold

    # Too many outliers: keep original (likely high natural variance)
    if np.sum(mask) > len(data) * 0.4:  # >40% flagged: keep original
        return data.copy()

    # Mark outliers as NaN
    filtered_data[mask] = np.nan

    # All NaN: revert to original
    if np.all(np.isnan(filtered_data)):
        return data.copy()

    # Interpolate NaN values
    nan_mask = np.isnan(filtered_data)

    # Need at least one non-NaN for interpolation
    if np.any(~nan_mask):
        filtered_data[nan_mask] = np.interp(
            np.flatnonzero(nan_mask),
            np.flatnonzero(~nan_mask),
            filtered_data[~nan_mask],
        )

    return filtered_data


def remove_jumps(data, threshold=1.0):
    """
    Detect and fix sudden jumps in the series.

        Args:
            data: Input 1D array.
            threshold: Jump detection threshold (default 1.0).

        Returns:
            Corrected array.
    """
    # Too few points for jump detection: return as-is
    if len(data) < 3:
        return data.copy()

    result = data.copy()

    # Absolute differences between consecutive samples
    try:
        diffs = np.abs(np.diff(result))
    except (ValueError, TypeError):
        # Cannot compute diffs: return as-is
        return data.copy()

    # Indices where diff exceeds threshold
    jump_indices = np.where(diffs > threshold)[0]

    # Too many jumps: keep original
    if len(jump_indices) > len(data) * 0.3:  # >30% jumps: keep original
        return data.copy()

    # Fix each jump (idx indexes into diffs; corrected sample is result[idx + 1])
    for idx in jump_indices:
        if idx > 0 and idx + 2 < len(result):
            result[idx + 1] = (result[idx] + result[idx + 2]) / 2
        elif idx + 1 < len(result):
            result[idx + 1] = result[idx]

    return result


def smooth_data(
    data, window_length=None, polyorder=3, iterations=1, strong_smooth=False
):
    """
    Smooth data with a Savitzky-Golay filter.

        Args:
            data: Input 1D array.
            window_length: Window size (auto if None).
            polyorder: Polynomial order (default 3).
            iterations: Number of passes (default 1).
            strong_smooth: Use stronger smoothing (default False).

        Returns:
            Smoothed array.
    """
    # Need at least 3 points
    if len(data) < 3:
        return data.copy()

    # Choose window length
    if window_length is None:
        if strong_smooth:
            # Strong smooth: larger window
            window_length = min(51, len(data) - 1)
        else:
            window_length = min(21, len(data) - 1)

    # Odd window length, capped by series length
    window_length = min(window_length, len(data) - 1)
    if window_length % 2 == 0:  # enforce odd window
        window_length -= 1

    # Minimum window length 3
    window_length = max(3, window_length)

    # Series shorter than window: use Gaussian filter
    if window_length >= len(data):
        sigma = 3.0 if strong_smooth else 1.0
        return gaussian_filter1d(data, sigma=sigma)

    # Strong smooth: lower polynomial order
    if strong_smooth:
        polyorder = min(2, polyorder)

    # polyorder must be < window_length
    polyorder = min(polyorder, window_length - 1)

    smooth_data_result = data.copy()

    try:
        # Multiple smoothing passes
        for _ in range(iterations):
            smooth_data_result = savgol_filter(
                smooth_data_result, window_length, polyorder
            )

        # Optional extra Gaussian pass for strong smooth
        if strong_smooth:
            smooth_data_result = gaussian_filter1d(smooth_data_result, sigma=2.0)

        return smooth_data_result

    except (ValueError, TypeError) as e:
        # Savgol failed: fall back to Gaussian
        logger.warning("Savgol filter failed (%s), using Gaussian filter instead", e)
        sigma = 3.0 if strong_smooth else 1.0
        return gaussian_filter1d(data, sigma=sigma)


class X2LeRobotDataset(LeRobotDataset):
    """LeRobot v3.0 writer: copies existing x2robot mp4 files instead of re-encoding frames."""

    def add_frame(self, frame: dict, task: str, timestamp: float | None = None) -> None:
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        frame_index = self.episode_buffer["size"]
        if timestamp is None:
            timestamp = frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)
        self.episode_buffer["task"].append(task)

        for key in frame:
            if key not in self.features:
                raise ValueError(
                    f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'."
                )
            if self.features[key]["dtype"] in ["image", "video"]:
                continue
            self.episode_buffer[key].append(frame[key])

        # Placeholders for video keys (validated in buffer, written in save_episode)
        for key in self.meta.video_keys:
            self.episode_buffer[key].append("")

        self.episode_buffer["size"] += 1

    def save_episode(
        self,
        videos: dict[str, str | Path],
        episode_data: dict | None = None,
        parallel_encoding: bool = True,
    ) -> None:
        episode_buffer = (
            episode_data if episode_data is not None else self.episode_buffer
        )

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(
            self.meta.total_frames, self.meta.total_frames + episode_length
        )
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        self.meta.save_episode_tasks(episode_tasks)
        episode_buffer["task_index"] = np.array(
            [self.meta.get_task_index(task) for task in tasks]
        )

        for key, ft in self.features.items():
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in [
                "image",
                "video",
            ]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        for key in self.meta.video_keys:
            episode_buffer.pop(key, None)

        ep_stats = compute_episode_stats(episode_buffer, self.features)
        ep_metadata = self._save_episode_data(episode_buffer)

        for video_key, src_path in videos.items():
            ep_metadata.update(
                self._save_episode_video_from_file(video_key, episode_index, src_path)
            )

        self.meta.save_episode(
            episode_index, episode_length, episode_tasks, ep_stats, ep_metadata
        )

        if episode_data is None:
            self.clear_episode_buffer(delete_images=False)

    def _save_episode_video_from_file(
        self, video_key: str, episode_index: int, src_path: str | Path
    ) -> dict:
        temp_dir = Path(tempfile.mkdtemp(dir=self.root))
        temp_path = temp_dir / f"{video_key}_{episode_index:06d}.mp4"
        try:
            shutil.copy2(src_path, temp_path)
            return self._save_episode_video(
                video_key, episode_index, temp_path=temp_path
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


def convert_euler_to_6D(euler_angle):
    """
    Convert euler angle to 6D rotation
    Input:
        euler_angle: numpy array of shape [low_dim_obs_horizon+horizon, 3] or [3]
    Output:
        rotation_6d: numpy array of shape [low_dim_obs_horizon+horizon, 6] or [6]
    """
    if len(euler_angle.shape) == 1:
        euler_angle = euler_angle.reshape(1, 3)
    rotation_matrix = R.from_euler("xyz", euler_angle).as_matrix()  # [horizon, 3, 3]
    # Convert rotation matrix to 6D rotation(first 2 columns of rotation matrix)
    rotation_6d = np.zeros((euler_angle.shape[0], 6))
    rotation_6d[:, :3] = rotation_matrix[:, :, 0]
    rotation_6d[:, 3:] = rotation_matrix[:, :, 1]
    assert rotation_6d.shape == (
        euler_angle.shape[0],
        6,
    ), f"rotation_6d shape is not correct, you get {rotation_6d.shape}"
    return rotation_6d.squeeze() if len(euler_angle.shape) == 1 else rotation_6d


def get_video_num_frames(video_path: str):
    """
    Read all frames from a video to get its length.

    Note: This iterates through every frame (does not cache them) to match the
    requirement "read each frame video".
    """
    reader = torchvision.io.VideoReader(video_path, stream="video")
    count = 0
    first_frame_shape = None
    for frame in reader:
        if first_frame_shape is None:
            # Typically a torch Tensor with shape [C, H, W]
            first_frame_shape = tuple(frame["data"].shape)
        count += 1
    return count, first_frame_shape


def _stack_action_keys(data_dict: dict, keys: list[str]) -> np.ndarray:
    """Stack per-key action/state arrays into a single (T, D) matrix."""
    parts = []
    for key in keys:
        if "rotation" in key:
            parts.append(euler_to_matrix_zyx_6d_nb(data_dict[key]))
        else:
            arr = data_dict[key]
            parts.append(arr[:, None] if arr.ndim == 1 else arr)
    return np.concatenate(parts, axis=1)


def load_local_dataset(episode_item):
    frames = []
    videos = {}

    cam_mapping = episode_item["cam_mapping"]
    folder_path = episode_item["path"]

    # Load mp4 files for all cameras in cam_mapping
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # filename = file.split(".")[0]
            filename = ".".join(file.split(".")[:-1])
            if file.endswith(".mp4") and filename in cam_mapping.keys():
                videos[f"observation.images.{filename}"] = os.path.join(root, file)

    # Build raw-key -> model-key maps for predict and obs
    action_keys_raw2predict = {
        _ACTION_KEY_FULL_MAPPING[k.replace("6D", "")]: k
        for k in episode_item["predict_action_keys"]
    }
    action_keys_raw2obs = {
        _ACTION_KEY_FULL_MAPPING[k.replace("6D", "")]: k
        for k in episode_item["obs_action_keys"]
    }

    # Load and process action trajectories from episode JSON
    action_data, action_data_obs = process_action(
        folder_path,
        raw_key2model_key=[action_keys_raw2predict, action_keys_raw2obs],
        filter_angle_outliers=True,
        fps=episode_item["fps"],
    )

    action = _stack_action_keys(action_data, episode_item["predict_action_keys"])
    obs = _stack_action_keys(action_data_obs, episode_item["obs_action_keys"])

    assert action.shape[0] == obs.shape[0], (
        f"action ({action.shape[0]}) and obs ({obs.shape[0]}) length mismatch "
        f"in {folder_path}"
    )

    frames = [
        {
            "action": action[i + 1],
            "observation.state": obs[i],
        }
        for i in range(action.shape[0] - 1)
    ]

    return frames, videos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert x2robot collections to LeRobotDataset v3.0.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
example:
  %(prog)s --config path/to/conversion.json
  %(prog)s --output-path /out --repo-id my_ds --src-path /data \\
      --features features.json --episode-config episode.json --fps 20
""",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "JSON config with output_path, repo_id, src_path_list, fps, features, "
            "and episode_template. CLI flags override values from this file."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Directory where the LeRobot v3.0 dataset is written.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Dataset name (subfolder under output_path).",
    )
    parser.add_argument(
        "--src-path",
        action="append",
        dest="src_paths",
        default=None,
        help="x2robot collection root (repeatable).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Sampling rate in Hz for car_pose velocity.",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="JSON file defining LeRobot v3.0 features schema.",
    )
    parser.add_argument(
        "--episode-config",
        type=str,
        default=None,
        help=(
            "JSON file with per-episode settings: cam_mapping, predict_action_keys, "
            "obs_action_keys, and optional type."
        ),
    )
    return parser.parse_args()


def _validate_episode_template(episode_template: dict) -> dict:
    missing = [
        key for key in _REQUIRED_EPISODE_TEMPLATE_KEYS if key not in episode_template
    ]
    if missing:
        raise ValueError(
            "episode_template missing required keys: "
            + ", ".join(missing)
            + " (provide via --episode-config or config.episode_template)"
        )
    return episode_template


def resolve_run_config(args: argparse.Namespace) -> dict:
    cfg: dict = {}
    if args.config:
        cfg = _load_json_file(args.config, "config")

    output_path = args.output_path or cfg.get("output_path")
    if not output_path:
        raise ValueError(
            "--output-path is required (or provide output_path in --config JSON)"
        )

    repo_id = args.repo_id or cfg.get("repo_id")
    if not repo_id:
        raise ValueError("--repo-id is required (or provide repo_id in --config JSON)")

    src_path_list = args.src_paths or cfg.get("src_path_list")
    if not src_path_list:
        raise ValueError(
            "At least one --src-path is required (or provide src_path_list in --config JSON)"
        )

    fps = args.fps if args.fps is not None else cfg.get("fps")
    if fps is None:
        raise ValueError("--fps is required (or provide fps in --config JSON)")

    if args.features:
        features = _load_json_file(args.features, "features")
    else:
        features = cfg.get("features")
    if not isinstance(features, dict) or not features:
        raise ValueError(
            "--features JSON file is required (or provide features in --config JSON)"
        )

    if args.episode_config:
        episode_template = _load_json_file(args.episode_config, "episode config")
    else:
        episode_template = cfg.get("episode_template")
    if not isinstance(episode_template, dict) or not episode_template:
        raise ValueError(
            "--episode-config JSON file is required "
            "(or provide episode_template in --config JSON)"
        )
    episode_template = _validate_episode_template(episode_template)

    return {
        "output_path": Path(output_path),
        "repo_id": repo_id,
        "src_path_list": list(src_path_list),
        "fps": float(fps),
        "features": features,
        "episode_template": episode_template,
    }


def run_conversion(
    output_path: Path,
    repo_id: str,
    src_path_list: list[str],
    fps: float,
    features: dict,
    episode_template: dict,
) -> list[str]:
    dataset = X2LeRobotDataset.create(
        repo_id=repo_id,
        root=output_path,
        features=features,
        fps=fps,
    )

    failed_episodes: list[str] = []

    for src_path in src_path_list:
        if not src_path or not os.path.exists(src_path):
            raise ValueError(f"Please provide a valid src_path: {src_path}")

        episode_paths = []
        for item in os.listdir(src_path):
            if item == "record":
                continue
            item_path = os.path.join(src_path, item)
            if os.path.isdir(item_path):
                episode_paths.append(item_path)

        logger.info("Found %d episode folders in %s", len(episode_paths), src_path)

        instructions_file = os.path.join(src_path, "instruction.json")
        with open(instructions_file, encoding="utf-8") as f:
            instruction_dict = json.load(f)

        for episode_path in episode_paths:
            episode_item = {
                **episode_template,
                "path": episode_path,
                "fps": fps,
                "instruction_info": instruction_dict.get(
                    os.path.basename(episode_path), {}
                ).get("instruction", ""),
            }
            logger.info("Processing episode: %s", episode_path)
            try:
                frames, videos = load_local_dataset(episode_item)
            except Exception:
                logger.exception("Failed to load episode: %s", episode_path)
                failed_episodes.append(episode_path)
                continue

            logger.info(
                "Loaded %d frames and %d videos from %s",
                len(frames),
                len(videos),
                episode_path,
            )
            for frame in frames:
                dataset.add_frame(frame, task=episode_item["instruction_info"])
            dataset.save_episode(videos)

    dataset.finalize()
    logger.info("LeRobot v3.0 dataset saved to %s", output_path)
    if failed_episodes:
        logger.error(
            "Skipped %d episode(s) due to errors:\n%s",
            len(failed_episodes),
            "\n".join(failed_episodes),
        )
    return failed_episodes


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_cfg = resolve_run_config(parse_args())
    failed = run_conversion(**run_cfg)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
