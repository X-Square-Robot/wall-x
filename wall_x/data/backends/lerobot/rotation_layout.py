"""Layout helpers when config expects 6D rotation but LeRobot stores 3D Euler."""

from __future__ import annotations

import numpy as np

from wall_x._vendor.x2robot_utils.geometry import euler_to_matrix_zyx_6d_nb

LAYOUT_SKIP_KEYS = frozenset(
    {"velocity_decomposed", "height", "head_actions", "action_padding"}
)
ROTATION_KEYWORD = "rotation"
ROTATION_6D_KEYWORD = "6D"


def layout_uses_6d_rotation(layout_config: dict) -> bool:
    for key, dim in layout_config.items():
        if key in LAYOUT_SKIP_KEYS:
            continue
        if ROTATION_KEYWORD in key and ROTATION_6D_KEYWORD in key and dim == 6:
            return True
    return False


def euler_layout_dim(layout_config: dict) -> int:
    """Vector width in LeRobot when rotation slices are still 3D Euler."""
    total = 0
    for key, dim in layout_config.items():
        if key in LAYOUT_SKIP_KEYS:
            continue
        if ROTATION_KEYWORD in key and ROTATION_6D_KEYWORD in key and dim == 6:
            total += 3
        else:
            total += int(dim)
    return total


def convert_euler_to_6d(vec: np.ndarray, layout_config: dict) -> np.ndarray:
    """Rewrite [pos, euler(3), tail...] to [pos, rot6d(6), tail...] per layout."""
    vec = np.asarray(vec, dtype=np.float64)
    single = vec.ndim == 1
    if single:
        vec = vec[np.newaxis, :]

    out_rows = []
    for row in vec:
        parts: list[np.ndarray] = []
        raw_cur = 0
        for key, dim in layout_config.items():
            if key in LAYOUT_SKIP_KEYS:
                continue
            dim = int(dim)
            if ROTATION_KEYWORD in key and ROTATION_6D_KEYWORD in key and dim == 6:
                euler = row[raw_cur : raw_cur + 3]
                rot6d = euler_to_matrix_zyx_6d_nb(euler.reshape(1, 3)).reshape(6)
                parts.append(rot6d)
                raw_cur += 3
            else:
                parts.append(row[raw_cur : raw_cur + dim])
                raw_cur += dim
        out_rows.append(np.concatenate(parts, axis=0))

    out = np.stack(out_rows, axis=0)
    return out[0] if single else out


def maybe_convert_norm_stats_vector(
    values,
    layout_config: dict,
    enabled: bool | None = None,
):
    """Convert a 1D norm-stat vector (q01/q99/mean/std) from Euler layout to 6D."""
    if enabled is None:
        enabled = layout_uses_6d_rotation(layout_config)
    if not enabled or not layout_config:
        return values
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        return values
    raw_dim = euler_layout_dim(layout_config)
    if arr.shape[0] != raw_dim:
        return values
    return convert_euler_to_6d(arr, layout_config).astype(np.float32)


def maybe_convert_euler_to_6d(
    vec: np.ndarray, layout_config: dict, enabled: bool
) -> np.ndarray:
    if not enabled or not layout_config:
        return vec
    raw_dim = euler_layout_dim(layout_config)
    arr = np.asarray(vec)
    if arr.shape[-1] != raw_dim:
        return vec
    return convert_euler_to_6d(arr, layout_config).astype(np.float32)
