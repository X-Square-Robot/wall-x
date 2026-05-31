import torch
import numpy as np

from scipy import stats
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

# from internal_dataset_backend.common.constants import ACTION_KEY_RANGES
from numba import jit, prange


@jit(nopython=True, parallel=True)
def euler_to_matrix_zyx_6d_nb(eulers):
    """
    Numba: Euler angles (N, 3) -> first two rows flattened (N, 6)
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


@jit(nopython=True, parallel=True)
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


def so3_to_euler_zyx_batch_nb(batch_so3):
    matrix = so3_to_matrix_batch_nb(batch_so3)
    eulers = matrix_to_euler_zyx_batch_nb(matrix)
    return canonicalize_euler_zyx_batch_nb(eulers)


@jit(nopython=True, parallel=True)
def matrix_to_euler_zyx_batch_nb(Rs):
    """
    R = Rz(yaw) * Ry(pitch) * Rx(roll)
    Extract:
      pitch = asin(-R[2,0])
      roll  = atan2(R[2,1], R[2,2])
      yaw   = atan2(R[1,0], R[0,0])
    """
    N = Rs.shape[0]
    eulers = np.empty((N, 3), dtype=np.float64)
    for i in prange(N):
        r00 = Rs[i, 0, 0]
        # r01 = Rs[i, 0, 1]
        # r02 = Rs[i, 0, 2]
        r10 = Rs[i, 1, 0]
        # r11 = Rs[i, 1, 1]
        # r12 = Rs[i, 1, 2]
        r20 = Rs[i, 2, 0]
        r21 = Rs[i, 2, 1]
        r22 = Rs[i, 2, 2]

        # clamp to [-1, 1] for numerical stability
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


@jit(nopython=True, parallel=True)
def so3_to_matrix_batch_nb(batch_so3):
    N = batch_so3.shape[0]
    R_all = np.empty((N, 3, 3), dtype=np.float64)
    eps = 1e-12
    for i in prange(N):
        r1x, r1y, r1z = batch_so3[i, 0], batch_so3[i, 1], batch_so3[i, 2]
        r2x, r2y, r2z = batch_so3[i, 3], batch_so3[i, 4], batch_so3[i, 5]

        # normalize r1
        n1 = np.sqrt(r1x * r1x + r1y * r1y + r1z * r1z) + eps
        r1x /= n1
        r1y /= n1
        r1z /= n1

        # orthogonalize r2 to r1, then normalize
        dot12 = r1x * r2x + r1y * r2y + r1z * r2z
        r2x -= dot12 * r1x
        r2y -= dot12 * r1y
        r2z -= dot12 * r1z
        n2 = np.sqrt(r2x * r2x + r2y * r2y + r2z * r2z) + eps
        r2x /= n2
        r2y /= n2
        r2z /= n2

        # r3 = r1 x r2
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


@jit(nopython=True, parallel=True)
def compute_delta_from_state_and_abs_rot(rotations, state):
    if rotations.shape[-1] == 3:
        rotations_matrix = euler_to_matrix_zyx_batch_nb(rotations)
        out_is_euler = True
    elif rotations.shape[-1] == 6:
        rotations_matrix = so3_to_matrix_batch_nb(rotations)
        out_is_euler = False
    else:
        raise ValueError(
            f"Only support 3D euler angle or 6D rotation, but got {rotations.shape[-1]}D"
        )

    if state.shape[-1] == 3:
        state_matrix = euler_to_matrix_zyx_batch_nb(state[np.newaxis, :])[0]
    elif state.shape[-1] == 6:
        state_matrix = so3_to_matrix_batch_nb(state[np.newaxis, :])[0]
    else:
        raise ValueError(
            f"Only support 3D euler angle or 6D rotation, but got {state.shape[-1]}D"
        )

    ST = np.empty((3, 3), dtype=np.float64)
    ST[0, 0] = state_matrix[0, 0]
    ST[0, 1] = state_matrix[1, 0]
    ST[0, 2] = state_matrix[2, 0]
    ST[1, 0] = state_matrix[0, 1]
    ST[1, 1] = state_matrix[1, 1]
    ST[1, 2] = state_matrix[2, 1]
    ST[2, 0] = state_matrix[0, 2]
    ST[2, 1] = state_matrix[1, 2]
    ST[2, 2] = state_matrix[2, 2]
    N = rotations_matrix.shape[0]
    R_rel = np.empty((N, 3, 3), dtype=np.float64)
    for i in prange(N):
        A00 = rotations_matrix[i, 0, 0]
        A01 = rotations_matrix[i, 0, 1]
        A02 = rotations_matrix[i, 0, 2]
        A10 = rotations_matrix[i, 1, 0]
        A11 = rotations_matrix[i, 1, 1]
        A12 = rotations_matrix[i, 1, 2]
        A20 = rotations_matrix[i, 2, 0]
        A21 = rotations_matrix[i, 2, 1]
        A22 = rotations_matrix[i, 2, 2]

        R_rel[i, 0, 0] = A00 * ST[0, 0] + A01 * ST[1, 0] + A02 * ST[2, 0]
        R_rel[i, 0, 1] = A00 * ST[0, 1] + A01 * ST[1, 1] + A02 * ST[2, 1]
        R_rel[i, 0, 2] = A00 * ST[0, 2] + A01 * ST[1, 2] + A02 * ST[2, 2]
        R_rel[i, 1, 0] = A10 * ST[0, 0] + A11 * ST[1, 0] + A12 * ST[2, 0]
        R_rel[i, 1, 1] = A10 * ST[0, 1] + A11 * ST[1, 1] + A12 * ST[2, 1]
        R_rel[i, 1, 2] = A10 * ST[0, 2] + A11 * ST[1, 2] + A12 * ST[2, 2]
        R_rel[i, 2, 0] = A20 * ST[0, 0] + A21 * ST[1, 0] + A22 * ST[2, 0]
        R_rel[i, 2, 1] = A20 * ST[0, 1] + A21 * ST[1, 1] + A22 * ST[2, 1]
        R_rel[i, 2, 2] = A20 * ST[0, 2] + A21 * ST[1, 2] + A22 * ST[2, 2]

    if out_is_euler:
        d_euler = matrix_to_euler_zyx_batch_nb(R_rel)
        return canonicalize_euler_zyx_batch_nb(d_euler)
    else:
        out6 = np.empty((N, 6), dtype=np.float64)
        for i in prange(N):
            out6[i, 0] = R_rel[i, 0, 0]
            out6[i, 1] = R_rel[i, 0, 1]
            out6[i, 2] = R_rel[i, 0, 2]
            out6[i, 3] = R_rel[i, 1, 0]
            out6[i, 4] = R_rel[i, 1, 1]
            out6[i, 5] = R_rel[i, 1, 2]
        return out6


@jit(nopython=True, parallel=True)
def canonicalize_euler_zyx_batch_nb(rpy_batch):
    """
    Batch ZYX Euler normalization (parallel)
    Input:  rpy_batch (N, 3) [roll, pitch, yaw] (rad)
    Output: out (N, 3) on one branch, each in (-pi, pi]
    Rules:
      1) wrap each component to (-pi, pi]
      2) if p >  pi/2:  p = pi - p; r += pi; y += pi
         if p <= -pi/2: p = -pi - p; r += pi; y += pi
      3) wrap each component to (-pi, pi] again
    """
    N = rpy_batch.shape[0]
    out = np.empty_like(rpy_batch)
    two_pi = 2.0 * np.pi

    for i in prange(N):
        r = rpy_batch[i, 0]
        p = rpy_batch[i, 1]
        y = rpy_batch[i, 2]

        # first wrap
        r = (r + np.pi) % two_pi - np.pi
        p = (p + np.pi) % two_pi - np.pi
        y = (y + np.pi) % two_pi - np.pi

        # branch canonicalization
        if p > np.pi / 2.0:
            p = np.pi - p
            r = r + np.pi
            y = y + np.pi
        elif p <= -np.pi / 2.0:
            p = -np.pi - p
            r = r + np.pi
            y = y + np.pi

        # final wrap
        r = (r + np.pi) % two_pi - np.pi
        p = (p + np.pi) % two_pi - np.pi
        y = (y + np.pi) % two_pi - np.pi

        out[i, 0] = r
        out[i, 1] = p
        out[i, 2] = y

    return out


@jit(nopython=True, parallel=True)
def compose_state_and_delta_to_abs_rpy(delta, state):
    """
    Input:
      delta: (N,3) -> delta rpy (ZYX) or (N,6) -> delta 6D (first two rows)
      state: (3,) -> rpy (ZYX) or (6,) -> 6D (first two rows)
    Output:
      abs_rpy: (N,3) absolute rpy (ZYX, rad), normalized to (-pi, pi]
    """
    if delta.shape[-1] == 3:
        R_delta = euler_to_matrix_zyx_batch_nb(delta)  # (N,3,3)
    elif delta.shape[-1] == 6:
        R_delta = so3_to_matrix_batch_nb(delta)  # (N,3,3)
    else:
        raise ValueError(f"delta last dim must be 3 or 6, got {delta.shape[-1]}")

    if state.shape[-1] == 3:
        R_state = euler_to_matrix_zyx_batch_nb(state[np.newaxis, :])[0]  # (3,3)
    elif state.shape[-1] == 6:
        R_state = so3_to_matrix_batch_nb(state[np.newaxis, :])[0]  # (3,3)
    else:
        raise ValueError(f"state last dim must be 3 or 6, got {state.shape[-1]}")

    N = R_delta.shape[0]
    R_abs = np.empty((N, 3, 3), dtype=np.float64)

    # copy R_state to avoid view aliasing
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


@jit(nopython=True, parallel=True)
def compose_state_and_delta_to_abs_6d(delta, state):
    """
    Input:
      delta: (N,6) -> delta 6D rotation (first two rows)
      state: (6,) -> 6D (first two rows) current rotation
    Output:
      abs_6d: (N,6) absolute 6D rotation (first two rows)
    """
    R_delta = so3_to_matrix_batch_nb(delta)  # (N,3,3)
    R_state = so3_to_matrix_batch_nb(state[np.newaxis, :])[0]  # (3,3)

    N = R_delta.shape[0]
    R_abs = np.empty((N, 3, 3), dtype=np.float64)

    # copy R_state to avoid view aliasing
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

    # rotation matrix -> 6D (first two rows flattened)
    abs_6d = np.empty((N, 6), dtype=np.float64)
    for i in prange(N):
        abs_6d[i, 0] = R_abs[i, 0, 0]
        abs_6d[i, 1] = R_abs[i, 0, 1]
        abs_6d[i, 2] = R_abs[i, 0, 2]
        abs_6d[i, 3] = R_abs[i, 1, 0]
        abs_6d[i, 4] = R_abs[i, 1, 1]
        abs_6d[i, 5] = R_abs[i, 1, 2]

    return abs_6d


@jit(nopython=True, parallel=True)
def normalize_angle_rad_batch_nb2(angles):
    """
    angles: (N, 2) -> (N, 2), each wrapped to (-pi, pi]
    """
    N = angles.shape[0]
    out = np.empty_like(angles)
    two_pi = 2.0 * np.pi
    for i in prange(N):
        a0 = angles[i, 0]
        a1 = angles[i, 1]
        a0 = (a0 + np.pi) % two_pi - np.pi
        a1 = (a1 + np.pi) % two_pi - np.pi
        out[i, 0] = a0
        out[i, 1] = a1
    return out


@jit(nopython=True, parallel=True)
def compute_head_delta_from_state_and_abs_nb(abs_py, state_py):
    """
    abs_py:   (N, 2)
    state_py: (2,)
    return:   (N, 2)  delta = wrap(abs - state)
    """
    N = abs_py.shape[0]
    out = np.empty((N, 2), dtype=np.float64)

    if state_py.ndim == 1:
        s0, s1 = state_py[0], state_py[1]
        for i in prange(N):
            d0 = abs_py[i, 0] - s0
            d1 = abs_py[i, 1] - s1
            out[i, 0] = d0
            out[i, 1] = d1
    elif state_py.ndim == 2:
        if state_py.shape[0] != N and state_py.shape[0] != 1:
            raise ValueError("state_py must be shape (2,) or (N,2) or (1,2)")
        if state_py.shape[0] == 1:
            s0, s1 = state_py[0, 0], state_py[0, 1]
            for i in prange(N):
                d0 = abs_py[i, 0] - s0
                d1 = abs_py[i, 1] - s1
                out[i, 0] = d0
                out[i, 1] = d1
        else:  # (N,2)
            for i in prange(N):
                d0 = abs_py[i, 0] - state_py[i, 0]
                d1 = abs_py[i, 1] - state_py[i, 1]
                out[i, 0] = d0
                out[i, 1] = d1
    else:
        raise ValueError("state_py.ndim must be 1 or 2")

    return normalize_angle_rad_batch_nb2(out)


@jit(nopython=True, parallel=True)
def compose_state_and_delta_to_abs_head_nb(delta_py, state_py):
    """
    delta_py: (N, 2) relative pitch/yaw (rad)
    state_py: (2,) or (N, 2)
    return:   (N, 2)  abs = wrap(state + delta)
    """
    N = delta_py.shape[0]
    out = np.empty((N, 2), dtype=np.float64)

    if state_py.ndim == 1:
        s0, s1 = state_py[0], state_py[1]
        for i in prange(N):
            a0 = delta_py[i, 0] + s0
            a1 = delta_py[i, 1] + s1
            out[i, 0] = a0
            out[i, 1] = a1
    elif state_py.ndim == 2:
        if state_py.shape[0] != N and state_py.shape[0] != 1:
            raise ValueError("state_py must be shape (2,) or (N,2) or (1,2)")
        if state_py.shape[0] == 1:
            s0, s1 = state_py[0, 0], state_py[0, 1]
            for i in prange(N):
                a0 = delta_py[i, 0] + s0
                a1 = delta_py[i, 1] + s1
                out[i, 0] = a0
                out[i, 1] = a1
        else:  # (N,2)
            for i in prange(N):
                a0 = delta_py[i, 0] + state_py[i, 0]
                a1 = delta_py[i, 1] + state_py[i, 1]
                out[i, 0] = a0
                out[i, 1] = a1
    else:
        raise ValueError("state_py.ndim must be 1 or 2")

    return normalize_angle_rad_batch_nb2(out)


def convert_euler_to_Lang(euler_angle):
    """
    Convert Euler angles to Lang angle.

    Input:
        euler_angle: pytorch tensor of shape [batch, 3] (Euler angles in radians)
    Output:
        lang_angle: numpy array of shape [batch] (Lang angles)
    """
    # Convert the PyTorch tensor to a NumPy array
    if isinstance(euler_angle, torch.Tensor):
        euler_angle_numpy = euler_angle.cpu().numpy()
    else:
        euler_angle_numpy = np.array(euler_angle)

    if len(euler_angle_numpy.shape) == 3:
        euler_angle_numpy = euler_angle_numpy.reshape(-1, 3)

    # Convert Euler angles to rotation matrix using scipy
    rotation_matrix = Rotation.from_euler(
        "xyz", euler_angle_numpy
    ).as_matrix()  # Shape: [batch, 3, 3]

    # Extract the relevant elements M00, M11, M22 for each rotation matrix
    M00 = rotation_matrix[:, 0, 0]  # First column, first row
    M11 = rotation_matrix[:, 1, 1]  # Second column, second row
    M22 = rotation_matrix[:, 2, 2]  # Third column, third row

    # Calculate the Lang angle
    lang_angle = np.arccos((M00 + M11 + M22 - 1) / 2)  # Shape: [batch]

    return lang_angle


def convert_6D_to_Lang(rotation_6d):
    """
    Convert 6D rotation to Lang angle. (Don't ask me why it is called Lang angle, quick coding)
    """
    if isinstance(rotation_6d, torch.Tensor):
        rotation_6d_numpy = rotation_6d.cpu().numpy()
    else:
        rotation_6d_numpy = np.array(rotation_6d)
    euler_angle = convert_6D_to_euler(rotation_6d_numpy)
    lang_angle = convert_euler_to_Lang(euler_angle)
    return lang_angle


def convert_euler_to_6D(euler_angle):
    """
    Convert euler angle to 6D rotation
    Input:
        euler_angle: numpy array of shape [low_dim_obs_horizon+horizon, 3] or [3]
    Output:
        rotation_6d: numpy array of shape [low_dim_obs_horizon+horizon, 6] or [6]
    """
    # TODO: find more elegent way
    # Convert euler angle to rotation matrix
    if len(euler_angle.shape) == 1:
        euler_angle = euler_angle.reshape(1, 3)
    rotation_matrix = Rotation.from_euler(
        "xyz", euler_angle
    ).as_matrix()  # [horizon, 3, 3]
    # Convert rotation matrix to 6D rotation(first 2 columns of rotation matrix)
    rotation_6d = np.zeros((euler_angle.shape[0], 6))
    rotation_6d[:, :3] = rotation_matrix[:, :, 0]
    rotation_6d[:, 3:] = rotation_matrix[:, :, 1]
    assert rotation_6d.shape == (
        euler_angle.shape[0],
        6,
    ), f"rotation_6d shape is not correct, you get {rotation_6d.shape}"
    return rotation_6d.squeeze() if len(euler_angle.shape) == 1 else rotation_6d


def convert_6D_to_euler(rotation_6d):
    """
    Convert 6D rotation to euler angle
    Input:
        rotation_6d: numpy array of shape [low_dim_obs_horizon+horizon, 6] or [6]
    Output:
        euler_angle: numpy array of shape [low_dim_obs_horizon+horizon, 3]
    """
    if rotation_6d.shape[0] == 6:
        rotation_6d = rotation_6d.reshape(1, 6)
    if len(rotation_6d.shape) == 3:
        rotation_6d = rotation_6d.reshape(-1, 6)
    # Convert 6D rotation to rotation matrix
    rotation_matrix = np.zeros((rotation_6d.shape[0], 3, 3))
    rotation_matrix[:, :, 0] = rotation_6d[:, :3]
    rotation_matrix[:, :, 1] = rotation_6d[:, 3:6]
    # get the third column of rotation matrix
    rotation_matrix[:, :, 2] = np.cross(
        rotation_matrix[:, :, 0], rotation_matrix[:, :, 1]
    )
    assert rotation_matrix.shape == (
        rotation_6d.shape[0],
        3,
        3,
    ), "rotation_matrix shape is not correct"
    # Convert rotation matrix to euler angle
    euler_angle = Rotation.from_matrix(rotation_matrix).as_euler("xyz")
    assert euler_angle.shape == (
        rotation_6d.shape[0],
        3,
    ), "euler_angle shape is not correct"
    return euler_angle


def convert_xyzrpy_to_matrix(position, orientation, data_config):
    """
    Convert xyzrpy to matrix
    Input:
        postion: np.array [3]
        orientation: np.array [3] for euler angle or [6] for 6D representation
        data_config: X2...
    Output:
        configuration matrix: SE(3) np.array [4,4]
    """
    configuration_matrix = np.eye(4)
    configuration_matrix[0:3, 3] = position
    if data_config.use_6D_rotation is True:
        orientation = convert_6D_to_euler(orientation)
    configuration_matrix[0:3, 0:3] = Rotation.from_euler("xyz", orientation).as_matrix()
    return configuration_matrix


def pose_to_transformation_matrix(
    pose, xyz_start_end_idx, rotation_start_end_idx, data_config
):
    """
    Input:
        pose: [action_dim], np.array
        xyz_start_end_idx: (,) tuple
        rotation_start_end_idx: (,) tuple
    Output:
        transformation_matrix: [4*4]
    """
    position = pose[xyz_start_end_idx[0] : xyz_start_end_idx[1]]
    rotation = pose[rotation_start_end_idx[0] : rotation_start_end_idx[1]]
    transformation_matrix = convert_xyzrpy_to_matrix(position, rotation, data_config)
    return transformation_matrix


def actions_to_relative(
    actions, add_noise=False, noise_scale=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
):
    """Convert absolute actions to relative actions

    Args:
        actions: numpy array of shape [horizon, action_dim]
                action_dim=14, [left_arm(7), right_arm(7)]
                Each arm: [x,y,z,roll,pitch,yaw,gripper]
        add_noise: bool, whether to add noise to the first frame
        noise_scale: list of 6 numbers, scale of noise for [x,y,z,roll,pitch,yaw]

    Returns:
        relative_actions: numpy array of shape [horizon, action_dim]
    """
    horizon, _ = actions.shape
    relative_actions = np.zeros_like(actions)

    # left and right arms
    for arm_idx in range(2):
        start_idx = arm_idx * 7

        # first-frame transform
        ref_pos = actions[0, start_idx : start_idx + 3]
        ref_rot = Rotation.from_euler("xyz", actions[0, start_idx + 3 : start_idx + 6])
        ref_matrix = np.eye(4)
        ref_matrix[:3, :3] = ref_rot.as_matrix()
        ref_matrix[:3, 3] = ref_pos

        # optional noise on first frame
        if add_noise:
            noise = np.random.normal(scale=noise_scale, size=6)
            noisy_pos = ref_pos + noise[:3]
            noisy_rot = Rotation.from_euler(
                "xyz", actions[0, start_idx + 3 : start_idx + 6] + noise[3:]
            )
            noisy_matrix = np.eye(4)
            noisy_matrix[:3, :3] = noisy_rot.as_matrix()
            noisy_matrix[:3, 3] = noisy_pos
            ref_matrix_inv = np.linalg.inv(noisy_matrix)
        else:
            ref_matrix_inv = np.linalg.inv(ref_matrix)

        # relative transforms for all frames (incl. first)
        for i in range(horizon):
            # current-frame transform
            curr_pos = actions[i, start_idx : start_idx + 3]
            curr_rot = Rotation.from_euler(
                "xyz", actions[i, start_idx + 3 : start_idx + 6]
            )
            curr_matrix = np.eye(4)
            curr_matrix[:3, :3] = curr_rot.as_matrix()
            curr_matrix[:3, 3] = curr_pos

            # relative transform
            relative_matrix = ref_matrix_inv @ curr_matrix

            # relative position and rotation
            relative_actions[i, start_idx : start_idx + 3] = relative_matrix[:3, 3]
            relative_actions[i, start_idx + 3 : start_idx + 6] = Rotation.from_matrix(
                relative_matrix[:3, :3]
            ).as_euler("xyz")

            # gripper unchanged
            relative_actions[i, start_idx + 6] = actions[i, start_idx + 6]

    return relative_actions


def relative_to_actions(relative_actions, start_pose, one_by_one_relative=False):
    """Convert relative actions back to absolute actions

    Args:
        relative_actions: numpy array of shape [horizon-1, action_dim]
                         relative pose sequence from frame 2
                         action_dim=14, [left_arm(7), right_arm(7)]
                         Each arm: [x,y,z,roll,pitch,yaw,gripper]
        start_pose: numpy array of shape [action_dim]
                   absolute pose of first frame

    Returns:
        actions: numpy array of shape [horizon, action_dim]
                start_pose plus converted absolute pose sequence
    """
    horizon = (
        relative_actions.shape[0] + 1
    )  # +1 because relative_actions excludes frame 0
    actions = np.zeros((horizon, relative_actions.shape[1]))

    # set frame 0 to start_pose
    actions[0] = start_pose

    # left and right arms
    for arm_idx in range(2):
        start_idx = arm_idx * 7

        # reference transform from start_pose
        ref_pos = start_pose[start_idx : start_idx + 3]
        ref_rot = Rotation.from_euler("xyz", start_pose[start_idx + 3 : start_idx + 6])
        ref_matrix = np.eye(4)
        ref_matrix[:3, :3] = ref_rot.as_matrix()
        ref_matrix[:3, 3] = ref_pos

        # absolute poses from frame 2
        for i in range(horizon - 1):  # length of relative_actions
            # relative pose transform
            relative_pos = relative_actions[i, start_idx : start_idx + 3]
            relative_rot = Rotation.from_euler(
                "xyz", relative_actions[i, start_idx + 3 : start_idx + 6]
            )
            relative_matrix = np.eye(4)
            relative_matrix[:3, :3] = relative_rot.as_matrix()
            relative_matrix[:3, 3] = relative_pos

            # absolute transform
            abs_matrix = ref_matrix @ relative_matrix

            # store absolute pos/rot at frame i+1
            actions[i + 1, start_idx : start_idx + 3] = abs_matrix[:3, 3]
            actions[i + 1, start_idx + 3 : start_idx + 6] = Rotation.from_matrix(
                abs_matrix[:3, :3]
            ).as_euler("xyz")

            # gripper unchanged
            actions[i + 1, start_idx + 6] = relative_actions[i, start_idx + 6]

            # relative action w.r.t. previous frame
            if one_by_one_relative:
                ref_matrix = abs_matrix

    return actions[1:]  # excludes start_pose


def remove_outliers(data, threshold=3):
    """
    Remove outliers via Z-score.

    Args:
        data: input array
        threshold: Z-score threshold (default 3)

    Returns:
        filtered data
    """
    # too few points or constant -> return as-is
    if len(data) < 3 or np.all(data == data[0]):
        return data.copy()

    # std with guard against cancellation
    std = np.std(data)
    if std < 1e-10:  # near-zero std -> return as-is
        return data.copy()

    # Z-score
    try:
        z_scores = np.abs(stats.zscore(data))
    except (ValueError, FloatingPointError, TypeError):
        # cannot compute Z-score -> return as-is
        return data.copy()

    filtered_data = data.copy()

    # mark outliers
    mask = z_scores > threshold

    # too many outliers -> keep original
    if np.sum(mask) > len(data) * 0.4:  # >40% flagged -> keep original
        return data.copy()

    # set outliers to NaN
    filtered_data[mask] = np.nan

    # ensure not all NaN
    if np.all(np.isnan(filtered_data)):
        return data.copy()

    # interpolate NaNs
    nan_mask = np.isnan(filtered_data)

    # need non-NaN values for interpolation
    if np.any(~nan_mask):
        filtered_data[nan_mask] = np.interp(
            np.flatnonzero(nan_mask),
            np.flatnonzero(~nan_mask),
            filtered_data[~nan_mask],
        )

    return filtered_data


def remove_jumps(data, threshold=1.0):
    """
    Detect and fix sudden jumps in data.

    Args:
        data: input array
        threshold: jump threshold (default 1.0)

    Returns:
        corrected data
    """
    # too few points -> return as-is
    if len(data) < 3:
        return data.copy()

    result = data.copy()

    # consecutive differences
    try:
        diffs = np.abs(np.diff(result))
    except (ValueError, FloatingPointError, TypeError):
        # cannot diff -> return as-is
        return data.copy()

    # indices above threshold
    jump_indices = np.where(diffs > threshold)[0]

    # too many jumps -> keep original
    if len(jump_indices) > len(data) * 0.3:  # >30% jumps -> keep original
        return data.copy()

    # fix each jump
    for idx in jump_indices:
        # replace jump with neighbor average
        if idx > 0 and idx < len(result) - 1:
            # average neighbors
            result[idx + 1] = (result[idx] + result[idx + 2]) / 2
        elif idx == len(result) - 2:  # second-to-last point
            result[idx + 1] = result[idx]  # copy previous value

    return result


def smooth_data(
    data, window_length=None, polyorder=3, iterations=1, strong_smooth=False
):
    """
    Smooth data with Savitzky-Golay filter.

    Args:
        data: input array
        window_length: window length (auto if None)
        polyorder: polynomial order (default 3)
        iterations: smoothing passes (default 1)
        strong_smooth: stronger smoothing (default False)

    Returns:
        smoothed data
    """
    # need at least 3 points
    if len(data) < 3:
        return data.copy()

    # choose window length
    if window_length is None:
        if strong_smooth:
            # larger window in strong mode
            window_length = min(51, len(data) - 1)
        else:
            window_length = min(21, len(data) - 1)

    # odd window <= len(data)
    window_length = min(window_length, len(data) - 1)
    if window_length % 2 == 0:  # force odd window
        window_length -= 1

    # minimum window 3
    window_length = max(3, window_length)

    # fallback to Gaussian if data shorter than window
    if window_length >= len(data):
        sigma = 3.0 if strong_smooth else 1.0
        return gaussian_filter1d(data, sigma=sigma)

    # lower polyorder in strong mode
    if strong_smooth:
        polyorder = min(2, polyorder)

    # polyorder < window_length
    polyorder = min(polyorder, window_length - 1)

    smooth_data_result = data.copy()

    try:
        # multiple passes
        for _ in range(iterations):
            smooth_data_result = savgol_filter(
                smooth_data_result, window_length, polyorder
            )

        # extra Gaussian pass when strong_smooth
        if strong_smooth:
            smooth_data_result = gaussian_filter1d(smooth_data_result, sigma=2.0)

        return smooth_data_result

    except Exception as e:
        # fallback to Gaussian on savgol failure
        print(f"Savgol filter failed: {e}, using Gaussian filter instead")
        sigma = 3.0 if strong_smooth else 1.0
        return gaussian_filter1d(data, sigma=sigma)


def process_car_pose_to_base_velocity(
    car_pose,
    outlier_threshold=3,
    jump_threshold=1.0,
    smooth_iterations=3,
    strong_smooth=True,
):
    """
    Process car_pose into body-frame base_velocity_decomposed (matches batch_process_json_data.py).
    Includes outlier handling, angle unwrap, filtering, smoothing, and body-frame velocity.

    Args:
        car_pose: input (n, 3) [x, y, angle]
        outlier_threshold: outlier Z threshold (default 3)
        jump_threshold: jump threshold (default 1.0)
        smooth_iterations: smooth passes (default 3)
        strong_smooth: strong smoothing (default True)

    Returns:
        dict with processed arrays
            - 'base_velocity_decomposed': shape (n, 3) [vx_body, vy_body, vyaw] (body frame)
            - 'valid': bool, passes velocity range check
    """
    # velocity limits (same as data_analysis_filter.py)
    velocity_limits = {
        "vx": {"min": -0.5, "max": 0.5},
        "vy": {"min": -0.5, "max": 0.5},
        "vyaw": {"min": -1.6, "max": 1.6},
    }

    # empty or single-point input
    if len(car_pose) == 0:
        return {"base_velocity_decomposed": np.zeros((0, 3)), "valid": False}

    if len(car_pose) == 1:
        return {
            "base_velocity_decomposed": np.zeros((1, 3)),
            "valid": True,  # single point treated as valid
        }

    # Step 1: extract pose and unwrap angles
    x_values = car_pose[:, 0].copy()
    y_values = car_pose[:, 1].copy()
    angle_values = car_pose[:, 2].copy()

    # unwrap angles to avoid pi jumps
    angle_values_unwrapped = np.unwrap(angle_values)

    # Steps 2-4: outliers, jumps, smoothing
    # outlier removal
    x_filtered = remove_outliers(x_values, outlier_threshold)
    y_filtered = remove_outliers(y_values, outlier_threshold)
    angle_filtered = remove_outliers(angle_values_unwrapped, outlier_threshold)

    # jump correction
    x_filtered = remove_jumps(x_filtered, jump_threshold)
    y_filtered = remove_jumps(y_filtered, jump_threshold)
    angle_filtered = remove_jumps(angle_filtered, jump_threshold)

    # smoothing
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
    dt = 1 / 20  # 20 Hz sampling

    # global displacement
    x_diff = np.diff(x_smooth)
    y_diff = np.diff(y_smooth)
    angle_diff = np.diff(angle_smooth)

    # current heading for frame transform
    current_theta = angle_smooth[:-1]  # shape: (n-1,)

    # global -> body frame (data_processor.py)
    cos_theta = np.cos(current_theta)
    sin_theta = np.sin(current_theta)

    # body-frame velocities
    vx_body = (x_diff * cos_theta + y_diff * sin_theta) / dt  # forward (body frame)
    vy_body = (-x_diff * sin_theta + y_diff * cos_theta) / dt  # lateral (body frame)
    vyaw = angle_diff / dt  # yaw rate

    # prepend zero velocity to match length
    vx_array = np.concatenate([[0], vx_body])
    vy_array = np.concatenate([[0], vy_body])
    vyaw_array = np.concatenate([[0], vyaw])

    base_velocity_decomposed = np.stack([vx_array, vy_array, vyaw_array], axis=1)

    # Step 6: velocity range check
    valid = True

    if (
        abs(x_values).max() > 6
        or abs(y_values).max() > 6
        or abs(angle_values).max() > 6
    ):
        valid = False

    # per-component range check
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

    if valid:  # check vyaw only if vx/vy passed
        for vyaw_val in vyaw:
            if (
                vyaw_val < velocity_limits["vyaw"]["min"]
                or vyaw_val > velocity_limits["vyaw"]["max"]
            ):
                valid = False
                break

    return {"base_velocity_decomposed": base_velocity_decomposed, "valid": valid}
