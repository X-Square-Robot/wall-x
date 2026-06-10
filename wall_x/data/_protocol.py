"""Shared data backend protocol definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

from wall_x.data._bundle import DataBundle


@dataclass
class BuildContext:
    """Shared runtime state passed to every backend ``build()``.

    Fields are Optional so backends not using a particular piece can
    simply leave it ``None``. The trainer populates whatever it has.
    """

    rank: int = 0
    world_size: int = 1
    tokenizer: Optional[Any] = None
    processor: Optional[Any] = None
    tokenizer_mixin: Optional[Any] = None
    normalizer_action: Optional[Any] = None
    normalizer_propri: Optional[Any] = None
    model_config: Optional[Any] = None
    resume_state: Optional[dict] = None


@runtime_checkable
class DatasetBackend(Protocol):
    """Callable every backend registers under its ``dataset_type`` name."""

    def __call__(self, cfg: Any, ctx: BuildContext) -> DataBundle: ...


__all__ = ["BuildContext", "DatasetBackend"]
