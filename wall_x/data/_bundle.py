"""Dataset bundle returned by data backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional


def _noop_set_epoch(_epoch: int) -> None:
    """Default ``set_epoch`` for backends with epoch-agnostic shuffling."""
    return None


@dataclass
class DataBundle:
    """Container returned by every backend ``build()``.

    Attributes:
        dataset: backend-private; trainer code should treat it as opaque.
        train_loader: anything iterable that yields training batches.
        val_loader: val iterable or None if the backend does not split val.
        train_iters: one-epoch step count for the train loader. Backends
            with dynamic resizing should set this to a stable snapshot
            and expose the live value separately on ``dataset``.
        val_iters: one-epoch step count for the val loader; 0 if no val.
        set_epoch: per-epoch seed hook. Called before each epoch by
            the trainer. Backends that don't need per-epoch reshuffle
            should use ``_noop_set_epoch``.
    """

    dataset: Any
    train_loader: Iterable
    val_loader: Optional[Iterable] = None
    train_iters: int = 0
    val_iters: int = 0
    set_epoch: Callable[[int], None] = field(default=_noop_set_epoch)


__all__ = ["DataBundle"]
