"""Load checkpoint weights and apply fused-format conversion when needed.

Model-instance operations such as ``load_state_dict`` and ``set_normalizer``
are intentionally left to the adapter.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import torch
from safetensors.torch import load_file


def _noop_log(_msg: str, **_kw) -> None:
    pass


def _align_checkpoint_tensor(
    param: torch.Tensor,
    target: torch.Tensor,
    name: str,
    log_fn: Callable,
) -> torch.Tensor | None:
    """Crop or pad a checkpoint tensor to match the current model parameter."""
    if param.shape == target.shape:
        return param
    if param.ndim != target.ndim:
        log_fn(
            f"Skipping '{name}': ndim mismatch "
            f"checkpoint={param.ndim} model={target.ndim}"
        )
        return None

    overlap = tuple(
        slice(0, min(src, dst)) for src, dst in zip(param.shape, target.shape)
    )

    if all(src >= dst for src, dst in zip(param.shape, target.shape)):
        aligned = param[overlap].contiguous()
        log_fn(
            f"Cropped '{name}': checkpoint {tuple(param.shape)} "
            f"-> model {tuple(aligned.shape)}"
        )
        return aligned

    if all(src <= dst for src, dst in zip(param.shape, target.shape)):
        aligned = target.detach().clone()
        aligned[overlap] = param[overlap]
        log_fn(
            f"Padded '{name}': checkpoint {tuple(param.shape)} "
            f"-> model {tuple(aligned.shape)} (tail keeps model init)"
        )
        return aligned

    aligned = target.detach().clone()
    aligned[overlap] = param[overlap]
    log_fn(
        f"Partially aligned '{name}': checkpoint {tuple(param.shape)} "
        f"-> model {tuple(aligned.shape)} (non-overlap keeps model init)"
    )
    return aligned


def reshape_compatible_state_dict(
    state_dict: dict, model_sd: dict, log_fn: Optional[Callable] = None
) -> dict:
    """Align checkpoint tensors to the target model shapes via crop / pad."""
    log_fn = log_fn or _noop_log
    out = {}
    for name, param in state_dict.items():
        if name not in model_sd:
            log_fn(f"Not used parameter: {name}")
            continue
        target = model_sd[name]
        if param.shape == target.shape:
            out[name] = param
            continue
        aligned = _align_checkpoint_tensor(param, target, name, log_fn)
        if aligned is not None:
            out[name] = aligned
    return out


def load_state_dict(checkpoint_path: str, model_class) -> dict:
    """Load a state dict from a checkpoint directory.

    Supported formats:
      - pytorch_model_fsdp.bin, optionally wrapped as {"state_dict": ...}
      - model.safetensors

    If the model class reports that the state dict is not fused, it is converted
    through ``model_class.convert_to_fused``.
    """
    fsdp_ckpt = os.path.join(checkpoint_path, "pytorch_model_fsdp.bin")
    safetensor_ckpt = os.path.join(checkpoint_path, "model.safetensors")

    if os.path.exists(fsdp_ckpt):
        state_dict = torch.load(fsdp_ckpt, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
    elif os.path.exists(safetensor_ckpt):
        state_dict = load_file(safetensor_ckpt, device="cpu")
    else:
        raise FileNotFoundError(
            "checkpoint contains neither pytorch_model_fsdp.bin nor model.safetensors: "
            f"{checkpoint_path}"
        )

    if not model_class.is_fused(state_dict):
        state_dict = model_class.convert_to_fused(state_dict)

    return state_dict


def read_global_step(checkpoint_path: str) -> int | None:
    """Read ``global_step.pth`` when present."""
    p = os.path.join(checkpoint_path, "global_step.pth")
    if not os.path.exists(p):
        return None
    payload = torch.load(p)
    return int(payload["global_step"])


def _dir_has_weights(path: str) -> bool:
    return os.path.exists(
        os.path.join(path, "pytorch_model_fsdp.bin")
    ) or os.path.exists(os.path.join(path, "model.safetensors"))


def resolve_checkpoint_dir(checkpoint_path: str) -> str:
    """Return a directory that directly contains model weights.

    Training saves under a root such as ``libero6/`` with step subdirs
    ``libero6/0/``, ``libero6/3/``, etc. Inference callers may pass either the
    root or a concrete step directory.
    """
    if os.path.isfile(checkpoint_path):
        checkpoint_path = os.path.dirname(checkpoint_path)

    if _dir_has_weights(checkpoint_path):
        return checkpoint_path

    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"checkpoint path does not exist: {checkpoint_path}")

    candidates: list[tuple[int, float, str]] = []
    for entry in os.listdir(checkpoint_path):
        sub = os.path.join(checkpoint_path, entry)
        if not os.path.isdir(sub) or not _dir_has_weights(sub):
            continue
        step = read_global_step(sub)
        sort_step = step if step is not None else -1
        candidates.append((sort_step, os.path.getmtime(sub), sub))

    if not candidates:
        return checkpoint_path

    candidates.sort()
    resolved = candidates[-1][2]
    if resolved != checkpoint_path:
        import logging

        logging.getLogger(__name__).info(
            "Resolved checkpoint root %s -> %s (global_step=%s)",
            checkpoint_path,
            resolved,
            read_global_step(resolved),
        )
    return resolved
