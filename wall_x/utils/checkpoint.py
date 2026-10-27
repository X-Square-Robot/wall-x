"""Conditional activation checkpointing helper (``maybe_checkpoint``).

Thin wrapper around ``torch.utils.checkpoint.checkpoint`` that lets callers
toggle ckpt per-call via an ``enabled`` kwarg instead of writing if/else at
every call site. Selection policy (which layers ckpt, every-Nth, etc.)
stays with the caller — this helper only does the conditional dispatch.

Defaults follow the wall-x convention:
- ``use_reentrant=False`` — session-19/24 实测固化。``use_reentrant=True`` 在
  wall-x FSDP2 nested ``fully_shard`` 下不兼容、显存反而暴涨（lab24 v2 +11 GiB
  OOM 确认）。``False`` 路径下 external_accel ``autograd.Function`` 在 chained ckpt 段
  的漏显存另行用 ``use_fused_geglu`` / pure-PT op 治（minimal repro 见
  `docs/inspection/repros/repro_ckpt_function_leak.py`）。**不要切 True，也不要
  移植 Megatron CheckpointFunction（同 reentrant 机制，一样 OOM）。**
- ``preserve_rng_state=False`` — our recomputed forward has no RNG ops
  (dropout=0, no stochastic depth), so the RNG state save/restore is just
  overhead.
"""

from __future__ import annotations

from typing import Any, Callable

import torch.utils.checkpoint as _ckpt


def maybe_checkpoint(
    func: Callable[..., Any],
    *args: Any,
    enabled: bool = True,
    use_reentrant: bool = False,  # PROBE session-19 reverted (lab24: FSDP2 incompat confirmed); see module docstring
    preserve_rng_state: bool = False,
    **kwargs: Any,
) -> Any:
    """Call ``func`` via ``torch.utils.checkpoint`` when ``enabled``, else direct.

    Args:
        func: The function to run (and recompute on backward when enabled).
        *args, **kwargs: Forwarded to ``func``.
        enabled: When False, calls ``func(*args, **kwargs)`` directly with no
            checkpointing — the activation stays in memory.
        use_reentrant: Forwarded to ``torch.utils.checkpoint``. Default False;
            True is incompatible with FSDP2 nested fully_shard and OOMs
            (session-19/24, see module docstring). Do not switch.
        preserve_rng_state: Forwarded to ``torch.utils.checkpoint``. Default
            False — our forward paths have no RNG ops.

    Returns:
        Whatever ``func`` returns.
    """
    if not enabled:
        return func(*args, **kwargs)
    return _ckpt.checkpoint(
        func,
        *args,
        use_reentrant=use_reentrant,
        preserve_rng_state=preserve_rng_state,
        **kwargs,
    )
