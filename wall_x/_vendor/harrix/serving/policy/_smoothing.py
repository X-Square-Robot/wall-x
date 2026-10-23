"""Test-time Laplacian smoothing for serving model outputs.

Mirrors the open-loop implementation in
`wallx_bus2604/wall-x/run_scripts/infer_openloop.py`
(_laplacian_smooth @ 203-209, smooth_action / smooth_gripper fields @ 186-187,
application @ 458-463). Invoked by WallXPolicy before downstream 6D->euler
conversion so that smoothed rotations propagate through serialization.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch

from wall_x._vendor.harrix.serving._wallx_infer.base_dataclass import dof_dims


def _laplacian_smooth(a: np.ndarray, lam: float = 1.0, iters: int = 30) -> np.ndarray:
    """Iterative Laplacian smoothing along axis 0; endpoints pinned."""
    a = a.copy()
    orig = a.copy()
    for _ in range(iters):
        a[1:-1] = (orig[1:-1] + lam * (a[:-2] + a[2:])) / (1 + 2 * lam)
    return a


def _gripper_column_indices(
    predict_action_keys: Sequence[str],
    action_padding_dof: Optional[int] = None,
) -> List[int]:
    """Column indices in the flat (T, D) layout that correspond to gripper dofs."""
    cols: List[int] = []
    dof_start = 0
    for key in predict_action_keys:
        if key == "action_padding":
            dof_start += action_padding_dof or 0
            continue
        short = key.replace("follow_", "").replace("master_", "")
        width = dof_dims[short]
        if "gripper" in short:
            cols.extend(range(dof_start, dof_start + width))
        dof_start += width
    return cols


_LAZY_ACTION_KEYS = (
    "action_left_ee_cartesian_pos",
    "action_right_ee_cartesian_pos",
    "action_left_ee_rotation",
    "action_right_ee_rotation",
    "action_left_ee_rotation_6D",
    "action_right_ee_rotation_6D",
)


def apply_smoothing(
    model_output: dict,
    smooth_action: bool,
    smooth_gripper: bool,
    predict_action_keys: Sequence[str],
    action_padding_dof: Optional[int] = None,
    action_dim: Optional[int] = None,
) -> None:
    """Smooth `model_output['predict_action']` in place and refresh per-arm keys."""
    if not smooth_action:
        return
    pa = model_output.get("predict_action")
    if pa is None:
        return

    was_tensor = isinstance(pa, torch.Tensor)
    arr = pa.detach().cpu().numpy() if was_tensor else np.asarray(pa)

    orig_ndim = arr.ndim
    if orig_ndim == 3:
        if arr.shape[0] != 1:
            return
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[0] < 3:
        return
    if action_dim is not None and arr.shape[-1] != action_dim:
        return

    orig = arr.copy()
    smoothed = _laplacian_smooth(arr)

    if not smooth_gripper:
        for c in _gripper_column_indices(predict_action_keys, action_padding_dof):
            if c < smoothed.shape[-1]:
                smoothed[:, c] = orig[:, c]

    out = smoothed[None] if orig_ndim == 3 else smoothed
    if was_tensor:
        out = torch.from_numpy(out).to(device=pa.device, dtype=pa.dtype)
    model_output["predict_action"] = out

    rsd = model_output.get("robot_state_action_data")
    if rsd is not None:
        for k in _LAZY_ACTION_KEYS:
            if k in rsd.data:
                rsd.data[k] = None
        rsd.save_action_data(out)
