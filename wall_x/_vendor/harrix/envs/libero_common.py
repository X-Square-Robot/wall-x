"""Env-adapter IO helpers for single-arm LIBERO tasks.

Env code extracts a compact ndarray payload from raw LIBERO observations.
Adapters then build proprioception, masks, and 7-dof right-arm action chunks
from that payload. This module intentionally depends only on NumPy so adapter
processes can import it without importing robosuite or LIBERO.
"""

from __future__ import annotations

import math

import numpy as np


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
        "gripper": np.asarray(
            raw_obs["robot0_gripper_qpos"][0:1], dtype=np.float32
        ),
        "face_view": _get_libero_image(raw_obs),
        "wrist_view": _get_libero_wrist_image(raw_obs),
    }


# ============================================================
# Adapter-side proprioception and mask construction.
# ============================================================


def encode_proprio(
    obs_ndarrays: dict,
    task,
    action_horizon: int,
) -> dict:
    """Convert a single-arm LIBERO ndarray payload into model input fields.

    ``task`` is the typed ``TaskConfig`` (single source of the action
    contract). Returned fields include proprioception, agent_pos_mask,
    dof_mask, face_view, and right_wrist_view.
    """
    agent_pos_config = task.agent_pos_config
    dof_config = task.dof_config

    # Canonical state values with bare keys.
    state_values: dict[str, np.ndarray] = {
        "right_ee_cartesian_pos": obs_ndarrays["eef_pos"][None],     # (1, 3)
        "right_ee_rotation": obs_ndarrays["eef_axisangle"][None],    # (1, 3)
        "right_gripper": obs_ndarrays["gripper"][None],              # (1, 1)
    }

    propri_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    for key, dim in agent_pos_config.items():
        bare = key.replace("follow_", "").replace("master_", "")
        if bare in state_values:
            v = state_values[bare]
            if bare == "right_gripper" and v.shape[1] < dim:
                # The env emits one gripper qpos but some checkpoints train
                # with both finger dims, which mirror each other
                # (qpos[1] ~= -qpos[0]); mirror-pad instead of OOD zeros.
                pad = np.repeat(-v[:, :1], dim - v.shape[1], axis=1)
                v = np.concatenate([v, pad], axis=1).astype(np.float32)
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


def decode_chunk(predict_action: np.ndarray, task) -> np.ndarray:
    """Extract a 7-dof right-arm chunk from model action output.

    ``task`` is the typed ``TaskConfig``. Only the first gripper dim is
    kept: 2-dim-gripper checkpoints carry a mirrored second finger dim
    (see ``encode_proprio``).
    """
    if predict_action.ndim == 3:
        predict_action = predict_action[0]

    dof_config = task.dof_config
    slices: dict[str, slice] = {}
    start = 0
    for key, dim in dof_config.items():
        bare = key.replace("follow_", "").replace("master_", "")
        slices[bare] = slice(start, start + dim)
        start += dim

    pos = predict_action[:, slices["right_ee_cartesian_pos"]]
    rot = predict_action[:, slices["right_ee_rotation"]]
    grip = predict_action[:, slices["right_gripper"]][:, :1]
    return np.concatenate([pos, rot, grip], axis=1)
