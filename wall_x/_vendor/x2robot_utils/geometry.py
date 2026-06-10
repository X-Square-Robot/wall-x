"""Rotation / pose geometry utilities (pure numpy + numba).

Shared between wall-x and internal_dataset_backend:
- ``euler_to_matrix_zyx_6d_nb``: ZYX Euler -> flattened top-2-rows of R (N, 6)
- ``so3_to_euler_zyx_batch_nb``: 6D rotation -> ZYX Euler (canonicalized)
- ``compose_state_and_delta_to_abs_{rpy,6d}``: state + delta -> absolute pose
"""

import numpy as np
from numba import jit, prange


@jit(nopython=True)
def euler_to_matrix_zyx_6d_nb(eulers):
    """Euler angles (N, 3) -> flattened top two rows of rotation matrix (N, 6)."""
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


@jit(nopython=True)
def euler_to_matrix_zyx_batch_nb(eulers):
    N = eulers.shape[0]
    R = np.empty((N, 3, 3), dtype=np.float64)
    for i in prange(N):
        roll = eulers[i, 0]
        pitch = eulers[i, 1]
        yaw = eulers[i, 2]

        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        R[i, 0, 0] = cy * cp
        R[i, 0, 1] = cy * sp * sr - sy * cr
        R[i, 0, 2] = cy * sp * cr + sy * sr

        R[i, 1, 0] = sy * cp
        R[i, 1, 1] = sy * sp * sr + cy * cr
        R[i, 1, 2] = sy * sp * cr - cy * sr

        R[i, 2, 0] = -sp
        R[i, 2, 1] = cp * sr
        R[i, 2, 2] = cp * cr
    return R


@jit(nopython=True)
def matrix_to_euler_zyx_batch_nb(Rs):
    """R = Rz(yaw) * Ry(pitch) * Rx(roll) -> (roll, pitch, yaw)."""
    N = Rs.shape[0]
    eulers = np.empty((N, 3), dtype=np.float64)
    for i in prange(N):
        r00 = Rs[i, 0, 0]
        r10 = Rs[i, 1, 0]
        r20 = Rs[i, 2, 0]
        r21 = Rs[i, 2, 1]
        r22 = Rs[i, 2, 2]

        x = -r20
        if x > 1.0:
            x = 1.0
        elif x < -1.0:
            x = -1.0

        pitch = np.arcsin(x)
        roll = np.arctan2(r21, r22)
        yaw = np.arctan2(r10, r00)

        eulers[i, 0] = roll
        eulers[i, 1] = pitch
        eulers[i, 2] = yaw
    return eulers


@jit(nopython=True)
def so3_to_matrix_batch_nb(batch_so3):
    N = batch_so3.shape[0]
    R_all = np.empty((N, 3, 3), dtype=np.float64)
    eps = 1e-12
    for i in prange(N):
        r1x, r1y, r1z = batch_so3[i, 0], batch_so3[i, 1], batch_so3[i, 2]
        r2x, r2y, r2z = batch_so3[i, 3], batch_so3[i, 4], batch_so3[i, 5]

        n1 = np.sqrt(r1x * r1x + r1y * r1y + r1z * r1z) + eps
        r1x /= n1
        r1y /= n1
        r1z /= n1

        dot12 = r1x * r2x + r1y * r2y + r1z * r2z
        r2x -= dot12 * r1x
        r2y -= dot12 * r1y
        r2z -= dot12 * r1z
        n2 = np.sqrt(r2x * r2x + r2y * r2y + r2z * r2z) + eps
        r2x /= n2
        r2y /= n2
        r2z /= n2

        r3x = r1y * r2z - r1z * r2y
        r3y = r1z * r2x - r1x * r2z
        r3z = r1x * r2y - r1y * r2x

        R_all[i, 0, 0] = r1x
        R_all[i, 0, 1] = r1y
        R_all[i, 0, 2] = r1z
        R_all[i, 1, 0] = r2x
        R_all[i, 1, 1] = r2y
        R_all[i, 1, 2] = r2z
        R_all[i, 2, 0] = r3x
        R_all[i, 2, 1] = r3y
        R_all[i, 2, 2] = r3z
    return R_all


@jit(nopython=True)
def canonicalize_euler_zyx_batch_nb(rpy_batch):
    """Canonicalize ZYX Euler angles so each component falls in (-pi, pi]."""
    N = rpy_batch.shape[0]
    out = np.empty_like(rpy_batch)
    two_pi = 2.0 * np.pi

    for i in prange(N):
        r = rpy_batch[i, 0]
        p = rpy_batch[i, 1]
        y = rpy_batch[i, 2]

        r = (r + np.pi) % two_pi - np.pi
        p = (p + np.pi) % two_pi - np.pi
        y = (y + np.pi) % two_pi - np.pi

        if p > np.pi / 2.0:
            p = np.pi - p
            r = r + np.pi
            y = y + np.pi
        elif p <= -np.pi / 2.0:
            p = -np.pi - p
            r = r + np.pi
            y = y + np.pi

        r = (r + np.pi) % two_pi - np.pi
        p = (p + np.pi) % two_pi - np.pi
        y = (y + np.pi) % two_pi - np.pi

        out[i, 0] = r
        out[i, 1] = p
        out[i, 2] = y

    return out


def so3_to_euler_zyx_batch_nb(batch_so3):
    matrix = so3_to_matrix_batch_nb(batch_so3)
    eulers = matrix_to_euler_zyx_batch_nb(matrix)
    return canonicalize_euler_zyx_batch_nb(eulers)


@jit(nopython=True)
def compose_state_and_delta_to_abs_rpy(delta, state):
    """Compose a delta (ZYX rpy or 6D) with an absolute state -> absolute rpy(ZYX).

    delta: (N, 3) deltarpy or (N, 6) delta6D. state: (3,) rpy or (6,) 6D.
    Output: (N, 3) rpy canonicalized into (-pi, pi].
    """
    if delta.shape[-1] == 3:
        R_delta = euler_to_matrix_zyx_batch_nb(delta)
    elif delta.shape[-1] == 6:
        R_delta = so3_to_matrix_batch_nb(delta)
    else:
        raise ValueError(f"delta last dim must be 3 or 6, got {delta.shape[-1]}")

    if state.shape[-1] == 3:
        R_state = euler_to_matrix_zyx_batch_nb(state[np.newaxis, :])[0]
    elif state.shape[-1] == 6:
        R_state = so3_to_matrix_batch_nb(state[np.newaxis, :])[0]
    else:
        raise ValueError(f"state last dim must be 3 or 6, got {state.shape[-1]}")

    N = R_delta.shape[0]
    R_abs = np.empty((N, 3, 3), dtype=np.float64)

    S00 = R_state[0, 0]
    S01 = R_state[0, 1]
    S02 = R_state[0, 2]
    S10 = R_state[1, 0]
    S11 = R_state[1, 1]
    S12 = R_state[1, 2]
    S20 = R_state[2, 0]
    S21 = R_state[2, 1]
    S22 = R_state[2, 2]

    for i in prange(N):
        A00 = R_delta[i, 0, 0]
        A01 = R_delta[i, 0, 1]
        A02 = R_delta[i, 0, 2]
        A10 = R_delta[i, 1, 0]
        A11 = R_delta[i, 1, 1]
        A12 = R_delta[i, 1, 2]
        A20 = R_delta[i, 2, 0]
        A21 = R_delta[i, 2, 1]
        A22 = R_delta[i, 2, 2]

        R_abs[i, 0, 0] = A00 * S00 + A01 * S10 + A02 * S20
        R_abs[i, 0, 1] = A00 * S01 + A01 * S11 + A02 * S21
        R_abs[i, 0, 2] = A00 * S02 + A01 * S12 + A02 * S22

        R_abs[i, 1, 0] = A10 * S00 + A11 * S10 + A12 * S20
        R_abs[i, 1, 1] = A10 * S01 + A11 * S11 + A12 * S21
        R_abs[i, 1, 2] = A10 * S02 + A11 * S12 + A12 * S22

        R_abs[i, 2, 0] = A20 * S00 + A21 * S10 + A22 * S20
        R_abs[i, 2, 1] = A20 * S01 + A21 * S11 + A22 * S21
        R_abs[i, 2, 2] = A20 * S02 + A21 * S12 + A22 * S22

    abs_rpy = matrix_to_euler_zyx_batch_nb(R_abs)
    abs_rpy = canonicalize_euler_zyx_batch_nb(abs_rpy)

    return abs_rpy


@jit(nopython=True)
def compose_state_and_delta_to_abs_6d(delta, state):
    """Compose a 6D delta with a 6D state -> absolute 6D rotation.

    delta: (N, 6). state: (6,). Output: (N, 6).
    """
    R_delta = so3_to_matrix_batch_nb(delta)
    R_state = so3_to_matrix_batch_nb(state[np.newaxis, :])[0]

    N = R_delta.shape[0]
    R_abs = np.empty((N, 3, 3), dtype=np.float64)

    S00 = R_state[0, 0]
    S01 = R_state[0, 1]
    S02 = R_state[0, 2]
    S10 = R_state[1, 0]
    S11 = R_state[1, 1]
    S12 = R_state[1, 2]
    S20 = R_state[2, 0]
    S21 = R_state[2, 1]
    S22 = R_state[2, 2]

    for i in prange(N):
        A00 = R_delta[i, 0, 0]
        A01 = R_delta[i, 0, 1]
        A02 = R_delta[i, 0, 2]
        A10 = R_delta[i, 1, 0]
        A11 = R_delta[i, 1, 1]
        A12 = R_delta[i, 1, 2]
        A20 = R_delta[i, 2, 0]
        A21 = R_delta[i, 2, 1]
        A22 = R_delta[i, 2, 2]

        R_abs[i, 0, 0] = A00 * S00 + A01 * S10 + A02 * S20
        R_abs[i, 0, 1] = A00 * S01 + A01 * S11 + A02 * S21
        R_abs[i, 0, 2] = A00 * S02 + A01 * S12 + A02 * S22

        R_abs[i, 1, 0] = A10 * S00 + A11 * S10 + A12 * S20
        R_abs[i, 1, 1] = A10 * S01 + A11 * S11 + A12 * S21
        R_abs[i, 1, 2] = A10 * S02 + A11 * S12 + A12 * S22

        R_abs[i, 2, 0] = A20 * S00 + A21 * S10 + A22 * S20
        R_abs[i, 2, 1] = A20 * S01 + A21 * S11 + A22 * S21
        R_abs[i, 2, 2] = A20 * S02 + A21 * S12 + A22 * S22

    abs_6d = np.empty((N, 6), dtype=np.float64)
    for i in prange(N):
        abs_6d[i, 0] = R_abs[i, 0, 0]
        abs_6d[i, 1] = R_abs[i, 0, 1]
        abs_6d[i, 2] = R_abs[i, 0, 2]
        abs_6d[i, 3] = R_abs[i, 1, 0]
        abs_6d[i, 4] = R_abs[i, 1, 1]
        abs_6d[i, 5] = R_abs[i, 1, 2]

    return abs_6d


__all__ = [
    "euler_to_matrix_zyx_6d_nb",
    "euler_to_matrix_zyx_batch_nb",
    "matrix_to_euler_zyx_batch_nb",
    "so3_to_matrix_batch_nb",
    "canonicalize_euler_zyx_batch_nb",
    "so3_to_euler_zyx_batch_nb",
    "compose_state_and_delta_to_abs_rpy",
    "compose_state_and_delta_to_abs_6d",
]
