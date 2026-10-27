"""Keyframe-memory serving configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class KeyframeMemoryConfig:
    """Geometry of the selection pass and the memory bank.

    The defaults are the values the released keyframe-memory checkpoints were
    trained with; ``from_checkpoint`` reads the prompt switches from the
    checkpoint's ``keyframe_memory_packing`` epilogue.
    """

    # Selection: every ``select_stride`` frames, ``num_candidates`` uniformly
    # spaced frames of the last ``window`` frames are offered as <|R_i|>.
    window: int = 48
    num_candidates: int = 16
    select_stride: int = 48
    # Only candidates newer than ``write_tail`` frames (and newer than the last
    # selection) may enter memory; older ones are context for the decision.
    write_tail: int = 48
    max_writes: int = 4
    # A new keyframe within this many frames of a stored one is a duplicate.
    dedup_radius: int = 3
    capacity: int = 22
    # Raw head-camera frames the client sends back each request: [now - span, now].
    ring_span: int = 60

    # Action prompt (``keyframe_memory_packing`` epilogue params).
    memory_in_prompt: bool = True
    memory_label: str = "progress memory"
    frame_stamp: bool = False
    # Crop memory frames to the object box decoded with them (box tasks only).
    box_crop: bool = False
    box_tasks: frozenset = frozenset({"imitate_sorting_sequence"})
    # Per-task short-term taps: (number of frames, stride) before the current one.
    short_term: Mapping[str, tuple] = field(
        default_factory=lambda: {"match_and_pick_from_conveyor": (2, 4)}
    )
    # Tasks that run selection and see memory; ``None`` means every task.
    memory_tasks: Optional[frozenset] = None

    def __post_init__(self):
        if not 0 < self.select_stride <= self.window:
            raise ValueError("select_stride must be in (0, window]")
        if not 0 < self.write_tail <= self.window:
            raise ValueError("write_tail must be in (0, window]")
        if self.ring_span < self.window:
            raise ValueError("ring_span must cover the selection window")
        if min(self.num_candidates, self.capacity, self.max_writes) < 1:
            raise ValueError("num_candidates, capacity and max_writes must be positive")

    @classmethod
    def from_checkpoint(cls, epilogue_params: Mapping[str, Any], **overrides):
        """Read the action-prompt switches the checkpoint was trained with."""
        params = dict(epilogue_params or {})
        base = cls(
            memory_in_prompt=bool(params.get("memory_in_prompt", True)),
            memory_label=str(params.get("memory_display", "progress memory")),
            frame_stamp=bool(params.get("now_frame_stamp", True)),
            box_crop=bool(params.get("memory_box_crop", False)),
        )
        return replace(base, **overrides)

    @staticmethod
    def task_family(task_id: str) -> str:
        return task_id.removesuffix("_random")

    def uses_memory(self, task_id: str) -> bool:
        return self.memory_tasks is None or self.task_family(task_id) in self.memory_tasks

    def uses_boxes(self, task_id: str) -> bool:
        return self.box_crop and self.task_family(task_id) in self.box_tasks

    def short_term_taps(self, task_id: str) -> tuple:
        return tuple(self.short_term.get(self.task_family(task_id), (0, 1)))

    def ring_plan(self, task_id: str) -> dict:
        """What a client needs to send only the ring frames a request reads."""
        return {
            "span": self.ring_span,
            "window": self.window,
            "num_candidates": self.num_candidates,
            "select_stride": self.select_stride,
            "short_term": list(self.short_term_taps(task_id)),
            "selection": self.uses_memory(task_id),
        }
