"""Device selection for inference on CUDA, Intel XPU or CPU."""

import torch


def get_default_device() -> torch.device:
    """Return CUDA when available, then Intel XPU, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def default_vision_attn_implementation() -> str:
    """Use flash-attn for the vision tower on CUDA and PyTorch SDPA elsewhere."""
    return "flash_attention_2" if torch.cuda.is_available() else "sdpa"
