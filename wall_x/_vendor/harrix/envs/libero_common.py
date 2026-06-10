"""Env-adapter IO helpers for single-arm LIBERO tasks.

Env code extracts a compact ndarray payload from raw LIBERO observations.
Adapters then build proprioception, masks, and 7-dof right-arm action chunks
from that payload. This module intentionally depends only on NumPy so adapter
processes can import it without importing robosuite or LIBERO.
"""

from __future__ import annotations

import math
import os

import numpy as np


# Fallback single-arm LIBERO dof_config when train_config does not provide one.
_LIBERO_FALLBACK_DOF_CONFIG = {
    "follow_right_ee_cartesian_pos": 3,
    "follow_right_ee_rotation": 3,
    "follow_right_gripper": 1,
}

_LIBERO_FALLBACK_AGENT_POS_CONFIG = dict(_LIBERO_FALLBACK_DOF_CONFIG)
_VIRTUAL_TAIL_KEYS = frozenset(("action_padding",))


def _resolve_dof_config(train_config: dict) -> dict:
    return (
        train_config.get("dof_config")
        or train_config.get("task", {}).get("dof_config")
        or _LIBERO_FALLBACK_DOF_CONFIG
    )


def _resolve_agent_pos_config(train_config: dict) -> dict:
    return (
        train_config.get("agent_pos_config")
        or train_config.get("task", {}).get("agent_pos_config")
        or _LIBERO_FALLBACK_AGENT_POS_CONFIG
    )


def _move_virtual_keys_to_tail(layout: dict) -> dict:
    head = {k: v for k, v in layout.items() if k not in _VIRTUAL_TAIL_KEYS}
    tail = {k: v for k, v in layout.items() if k in _VIRTUAL_TAIL_KEYS}
    return {**head, **tail}


def _effective_agent_pos_config(train_config: dict, state_values: dict) -> dict:
    config = _move_virtual_keys_to_tail(dict(_resolve_agent_pos_config(train_config)))
    gripper_key = next(
        (
            key
            for key in config
            if key.replace("follow_", "").replace("master_", "") == "right_gripper"
        ),
        None,
    )
    if gripper_key is None:
        return config

    old_dim = int(config[gripper_key])
    target_dim = None
    override = os.environ.get("WALLX_LIBERO_STATE_GRIPPER_DIM")
    if override:
        target_dim = int(override)
    elif os.environ.get("WALLX_LIBERO_AUTO_STATE_GRIPPER_DIM", "1") != "0":
        norm_dim = int(train_config.get("_libero_proprio_norm_dim") or 0)
        real_dim = sum(v for k, v in config.items() if k not in _VIRTUAL_TAIL_KEYS)
        if norm_dim == real_dim + 1:
            target_dim = old_dim + 1

    available_dim = state_values["right_gripper"].shape[1]
    if target_dim is None or target_dim == old_dim or target_dim > available_dim:
        return config

    config[gripper_key] = target_dim
    delta = target_dim - old_dim
    if "action_padding" in config:
        config["action_padding"] = max(0, int(config["action_padding"]) - delta)
    return config


def _build_right_arm_state_values(obs_ndarrays: dict) -> dict[str, np.ndarray]:
    """Bare-key state tensors for single-arm LIBERO proprio construction."""
    rot3 = np.asarray(obs_ndarrays["eef_axisangle"], dtype=np.float32).reshape(1, 3)
    values: dict[str, np.ndarray] = {
        "right_ee_cartesian_pos": np.asarray(
            obs_ndarrays["eef_pos"], dtype=np.float32
        ).reshape(1, 3),
        "right_ee_rotation": rot3,
        "right_gripper": np.asarray(obs_ndarrays["gripper"], dtype=np.float32).reshape(
            1, -1
        ),
    }
    from wall_x._vendor.x2robot_utils.geometry import euler_to_matrix_zyx_6d_nb

    rot6d = euler_to_matrix_zyx_6d_nb(rot3.astype(np.float64)).reshape(1, 6)
    values["right_ee_rotation_6D"] = rot6d.astype(np.float32)
    return values


# Auxiliary action keys that should be masked out for single-arm LIBERO.
_DOF_MASK_ZERO_KEYS = frozenset(
    (
        "follow_left_ee_cartesian_pos",
        "follow_left_ee_rotation",
        "follow_left_ee_rotation_6D",
        "follow_left_gripper",
        "head_actions",
        "height",
        "velocity_decomposed",
        "action_padding",
    )
)


# ============================================================
# LIBERO raw observation decoding helpers.
# ============================================================


def _get_libero_image(obs: dict) -> np.ndarray:
    """Return the third-person camera image, rotated to match preprocessing."""
    return obs["agentview_image"][::-1, ::-1]


def get_rollout_frame(obs: dict) -> np.ndarray:
    """Return one RGB frame for rollout MP4 saving."""
    return np.asarray(_get_libero_image(obs), dtype=np.uint8)


def _get_libero_wrist_image(obs: dict) -> np.ndarray:
    """Return the wrist camera image, rotated to match preprocessing."""
    return obs["robot0_eye_in_hand_image"][::-1, ::-1]


def _quat2axisangle(quat) -> np.ndarray:
    """Convert an xyzw quaternion to a 3D axis-angle vector."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


# ============================================================
# Env-side to adapter-side observation encoding.
# ============================================================


def encode_raw_obs(raw_obs: dict) -> dict:
    """Extract the minimal ndarray payload from a raw LIBERO observation.

    The payload contains three 1-D state arrays and two rotated image arrays.
    """
    if "agentview_image" not in raw_obs:
        raise KeyError(
            "agentview_image missing in raw_obs; render-skip may have returned "
            "a stale observation"
        )
    return {
        "eef_pos": np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32),
        "eef_axisangle": np.asarray(
            _quat2axisangle(raw_obs["robot0_eef_quat"]), dtype=np.float32
        ),
        "gripper": np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32),
        "face_view": _get_libero_image(raw_obs),
        "wrist_view": _get_libero_wrist_image(raw_obs),
    }


# ============================================================
# Adapter-side proprioception and mask construction.
# ============================================================


def encode_proprio(
    obs_ndarrays: dict,
    train_config: dict,
    action_horizon: int,
) -> dict:
    """Convert a single-arm LIBERO ndarray payload into model input fields.

    Returned fields include proprioception, agent_pos_mask, dof_mask,
    face_view, and right_wrist_view.
    """
    state_values = _build_right_arm_state_values(obs_ndarrays)
    agent_pos_config = _effective_agent_pos_config(train_config, state_values)
    dof_config = _move_virtual_keys_to_tail(dict(_resolve_dof_config(train_config)))

    propri_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    for key, dim in agent_pos_config.items():
        bare = key.replace("follow_", "").replace("master_", "")
        if bare in state_values:
            v = state_values[bare]
            if bare == "right_gripper" and v.shape[1] > dim:
                v = v[:, :dim]
            if v.shape[1] != dim:
                raise ValueError(
                    f"agent_pos_config[{key!r}]={dim} does not match "
                    f"observation dimension {v.shape[1]}"
                )
            propri_parts.append(v)
            mask_parts.append(np.ones((1, dim), dtype=np.float32))
        else:
            propri_parts.append(np.zeros((1, dim), dtype=np.float32))
            mask_parts.append(np.zeros((1, dim), dtype=np.float32))

    # (1, 1, D)
    proprioception = np.concatenate(propri_parts, axis=1)[None]
    agent_pos_mask = np.concatenate(mask_parts, axis=1)[None]

    # dof_mask: (1, T, D_action)
    total_dof = sum(dof_config.values())
    dof_mask = np.ones((1, action_horizon, total_dof))
    start = 0
    for key, dim in dof_config.items():
        if key in _DOF_MASK_ZERO_KEYS:
            dof_mask[:, :, start : start + dim] = 0
        start += dim

    return {
        "proprioception": proprioception.astype(np.float32),
        "agent_pos_mask": agent_pos_mask.astype(np.float32),
        "dof_mask": dof_mask,
        "face_view": obs_ndarrays["face_view"],
        "right_wrist_view": obs_ndarrays["wrist_view"],
    }


# ============================================================
# Adapter-side action decoding.
# ============================================================


def decode_chunk(predict_action: np.ndarray, train_config: dict) -> np.ndarray:
    """Extract a 7-dof right-arm chunk from model action output."""
    if predict_action.ndim == 3:
        predict_action = predict_action[0]

    dof_config = _resolve_dof_config(train_config)
    slices: dict[str, slice] = {}
    start = 0
    for key, dim in dof_config.items():
        bare = key.replace("follow_", "").replace("master_", "")
        slices[bare] = slice(start, start + dim)
        start += dim

    pos = predict_action[:, slices["right_ee_cartesian_pos"]]
    grip = predict_action[:, slices["right_gripper"]]

    if "right_ee_rotation_6D" in slices:
        from wall_x._vendor.x2robot_utils.geometry import so3_to_euler_zyx_batch_nb

        rot6d = np.asarray(
            predict_action[:, slices["right_ee_rotation_6D"]], dtype=np.float64
        )
        rot = so3_to_euler_zyx_batch_nb(rot6d).astype(np.float32)
    elif "right_ee_rotation" in slices:
        rot = predict_action[:, slices["right_ee_rotation"]]
    else:
        raise KeyError(
            "dof_config has no right-arm rotation slice "
            f"(keys={list(slices.keys())})"
        )

    return np.concatenate([pos, rot, grip], axis=1)


def gripper_model_to_libero_osc(grip_2d: np.ndarray) -> float:
    """Map model gripper output to robosuite OSC_POSE gripper command in [-1, 1]."""
    g = np.asarray(grip_2d, dtype=np.float64).reshape(-1)
    if g.size == 0:
        raise ValueError("empty gripper action")
    cmd = float(g[0])
    if os.environ.get("WALLX_LIBERO_GRIPPER_BINARIZE", "0") == "1":
        if abs(cmd) < 1e-6:
            cmd = -1.0
        else:
            cmd = float(np.sign(cmd))
    if os.environ.get("WALLX_LIBERO_INVERT_GRIPPER", "0") == "1":
        cmd *= -1.0
    return cmd


def _sanitize_task_description(task_description: str, max_len: int = 50) -> str:
    return (
        task_description.lower()
        .replace(" ", "_")
        .replace("\n", "_")
        .replace(".", "_")[:max_len]
    )


def save_rollout_video(
    rollout_dir: str,
    rollout_images: list[np.ndarray],
    *,
    task_id: int,
    episode_idx: int,
    success: bool,
    task_description: str,
    fps: int = 30,
) -> str | None:
    """Save an MP4 replay of one LIBERO episode."""
    if not rollout_images:
        return None

    import imageio

    os.makedirs(rollout_dir, exist_ok=True)
    task_slug = _sanitize_task_description(task_description)
    mp4_path = os.path.join(
        rollout_dir,
        f"task{task_id}_ep{episode_idx}--success={int(success)}--{task_slug}.mp4",
    )
    writer = imageio.get_writer(mp4_path, fps=fps, macro_block_size=1)
    try:
        for img in rollout_images:
            writer.append_data(np.asarray(img, dtype=np.uint8))
    finally:
        writer.close()
    return mp4_path


def model_action_to_libero_env(action: np.ndarray) -> np.ndarray:
    """Convert model output to the robosuite OSC_POSE 7D action.

    Internal LIBERO evaluation passes the model's 7D chunk directly to
    ``env.step``. Keep that as the public default; conversion modes remain
    available only for ablations through environment variables.
    """
    from scipy.spatial.transform import Rotation as R

    a = np.asarray(action, dtype=np.float64).reshape(-1)
    if a.size not in (7, 8):
        raise ValueError(f"expected 7D or 8D model action, got shape {a.shape}")

    pos_delta = a[:3]
    rot_mode = os.environ.get("WALLX_LIBERO_ROT_MODE", "direct").strip()
    if rot_mode == "euler_zyx_to_rotvec":
        rot_aa = R.from_euler("zyx", a[3:6]).as_rotvec()
    elif rot_mode == "direct":
        rot_aa = a[3:6]
    else:
        raise ValueError(
            "WALLX_LIBERO_ROT_MODE must be 'euler_zyx_to_rotvec' or 'direct', "
            f"got {rot_mode!r}"
        )
    grip = gripper_model_to_libero_osc(a[6:8] if a.size == 8 else a[6:7])
    return np.concatenate([pos_delta, rot_aa, np.array([grip], dtype=np.float64)])
