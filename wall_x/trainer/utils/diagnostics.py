"""Training-run diagnostic helpers (CUDA memory, etc.)."""

from __future__ import annotations

from typing import Callable, Optional

import torch


def log_gpu_memory(
    device: torch.device,
    rank: int,
    *,
    stage: str = "",
    log_fn: Optional[Callable] = None,
) -> None:
    """Log per-rank GPU memory (allocated / reserved / total) via ``log_fn``.

    Calls ``torch.cuda.synchronize`` on ``device`` so the numbers reflect
    the actual post-op usage, not pending work. If ``log_fn`` is None this
    is a no-op.
    """
    if log_fn is None:
        return
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    tag = f"[{stage}] " if stage else ""
    log_fn(
        f"{tag}GPU memory rank{rank} "
        f"| allocated {allocated:.2f} GiB"
        f" | reserved {reserved:.2f} GiB"
        f" | peak_allocated {peak_allocated:.2f} GiB"
        f" | peak_reserved {peak_reserved:.2f} GiB"
        f" | total {total:.2f} GiB",
        main_process_only=False,
    )
