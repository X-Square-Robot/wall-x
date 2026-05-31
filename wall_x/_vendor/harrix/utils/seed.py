"""Seed helpers for driver and environment worker processes."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed_everywhere(seed: int) -> None:
    """Set random seeds and deterministic backend options."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Deterministic cuBLAS workspace plus warning-only deterministic op checks.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
