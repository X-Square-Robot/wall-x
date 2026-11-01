"""Load checkpoint weights and apply fused-format conversion when needed.

Model-instance operations such as ``load_state_dict`` and ``set_normalizer``
are intentionally left to the adapter.
"""

from __future__ import annotations

import os

import torch
from safetensors.torch import load_file


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
