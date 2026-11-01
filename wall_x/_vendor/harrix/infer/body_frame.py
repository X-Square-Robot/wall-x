"""Decode body-frame relative end-effector actions into absolute poses.

A body-frame delta is expressed in the end-effector frame of the current
state. With ``R_state`` the state rotation matrix:

* position: ``p_abs = p_state + R_state @ delta``
* rotation: ``R_abs = R_state @ R_delta``

Rotations use the row-major 6D layout (first two rows of the rotation
matrix) and ZYX Euler angles of ``wall_x._vendor.x2robot_utils.geometry``.
"""

from __future__ import annotations

import numpy as np

from wall_x._vendor.x2robot_utils import geometry as _geom


def _as_rows(values: np.ndarray, width: int, name: str) -> np.ndarray:
    rows = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if rows.shape[-1] != width:
        raise ValueError(f"{name} must end in dimension {width}, got {rows.shape}")
    return rows.reshape(-1, width)


def state_rotation_matrix(state_rotation: np.ndarray) -> np.ndarray:
    """Return the 3x3 matrix of one Euler-3 or row-6D state rotation."""
    rotation = np.asarray(state_rotation, dtype=np.float64)
    width = rotation.shape[-1]
    if width == 3:
        matrices = _geom.euler_to_matrix_zyx_batch_nb(_as_rows(rotation, 3, "state"))
    elif width == 6:
        matrices = _geom.so3_to_matrix_batch_nb(_as_rows(rotation, 6, "state"))
    else:
        raise ValueError(
            "state rotation must be Euler-3 or row-6D, "
            f"got shape {rotation.shape}"
        )
    if matrices.shape[0] != 1:
        raise ValueError(
            "body-frame decoding needs exactly one state rotation, "
            f"got shape {rotation.shape}"
        )
    return matrices[0]


def compose_body_delta_to_abs_pos(
    delta: np.ndarray, state_position: np.ndarray, state_rotation: np.ndarray
) -> np.ndarray:
    """Return absolute positions ``(N, 3)`` for body-frame deltas ``(N, 3)``."""
    delta_rows = _as_rows(delta, 3, "position delta")
    position = _as_rows(state_position, 3, "state position")
    if position.shape[0] != 1:
        raise ValueError(
            "body-frame decoding needs exactly one state position, "
            f"got shape {np.shape(state_position)}"
        )
    rotation = state_rotation_matrix(state_rotation)
    return delta_rows @ rotation.T + position[0]


def compose_body_delta_to_abs_6d(
    delta: np.ndarray, state_rotation: np.ndarray
) -> np.ndarray:
    """Return absolute row-6D rotations ``(N, 6)`` for body-frame deltas ``(N, 6)``."""
    delta_matrices = _geom.so3_to_matrix_batch_nb(_as_rows(delta, 6, "rotation delta"))
    rotation = state_rotation_matrix(state_rotation)
    absolute = np.einsum("ij,njk->nik", rotation, delta_matrices)
    return absolute[:, :2, :].reshape(-1, 6)


__all__ = [
    "state_rotation_matrix",
    "compose_body_delta_to_abs_pos",
    "compose_body_delta_to_abs_6d",
]
