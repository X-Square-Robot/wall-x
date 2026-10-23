"""Episode memory and the frame arithmetic of the selection pass.

Everything here is pure bookkeeping over frame indices, so it can be checked
without a model. Frame indices are simulator steps; frame 0 is the observation
before the first action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from wall_x.serving.keyframe_memory.config import KeyframeMemoryConfig

# Object box in 0..1000 coordinates of the frame: (x1, y1, x2, y2).
Box = Tuple[int, int, int, int]


@dataclass(frozen=True)
class Frame:
    index: int
    image: np.ndarray


@dataclass
class MemoryState:
    """One episode's memory, as carried by the client between requests.

    ``ring`` holds every raw head-camera frame of the last ``ring_span`` steps
    (oldest first, ending at the current frame); ``memory`` the confirmed
    keyframes, oldest first. ``boxes`` maps a memory frame index to the object
    box decoded when it was written (box tasks only).
    """

    ring: List[Frame]
    memory: List[Frame] = field(default_factory=list)
    last_select_frame: int = 0
    consecutive_skips: int = 0
    boxes: Dict[int, Box] = field(default_factory=dict)
    # First frame of the logical ring, ``max(0, now - ring_span)``. A client may
    # send only the ring frames a request needs (see ``needed_ring_frames``), so
    # ``ring[0]`` is not necessarily the oldest frame of the window.
    ring_start: Optional[int] = None

    @property
    def now(self) -> int:
        return self.ring[-1].index

    @property
    def oldest(self) -> int:
        return self.ring[0].index if self.ring_start is None else self.ring_start


@dataclass(frozen=True)
class Selection:
    """Decoded answer of one selection pass, in candidate-slot indices."""

    write: bool
    slots: Tuple[int, ...] = ()
    boxes: Dict[int, Box] = field(default_factory=dict)
    margin: float = 0.0  # logit(write) - logit(skip) at the decision token


@dataclass(frozen=True)
class SelectionWindow:
    """Inputs of one selection pass at frame ``now``."""

    now: int
    candidates: Tuple[Frame, ...]
    memory: Tuple[Frame, ...]
    writable: Tuple[int, ...]


def _nearest(ring: Sequence[Frame], target: int) -> Frame:
    # First match wins on ties; the ring is in increasing frame order.
    return min(ring, key=lambda frame: abs(frame.index - target))


class KeyframeBank:
    """Selection timing, candidate sampling and memory updates."""

    def __init__(self, config: KeyframeMemoryConfig):
        self.config = config

    def selection_due(self, state: MemoryState) -> bool:
        return state.now - state.last_select_frame >= self.config.select_stride

    def window(self, state: MemoryState) -> SelectionWindow:
        cfg = self.config
        now = state.now
        start = max(0, now - cfg.window)
        span = max(1, now - start)
        candidates = tuple(
            _nearest(state.ring, start + int((i + 0.5) * span / cfg.num_candidates))
            for i in range(cfg.num_candidates)
        )
        # The selector only sees memory older than its candidate window, so a
        # frame never appears both as [M_j] and as <|R_i|>.
        memory = tuple(f for f in state.memory if f.index < start)[-cfg.capacity :]
        write_start = max(0, state.last_select_frame, now - cfg.write_tail)
        writable = tuple(
            i for i, f in enumerate(candidates) if write_start <= f.index < now
        )
        return SelectionWindow(now, candidates, memory, writable)

    def commit(
        self, state: MemoryState, window: SelectionWindow, selection: Selection
    ) -> List[int]:
        """Apply a selection to ``state`` in place; returns the frames written."""
        cfg = self.config
        if selection.write:
            state.consecutive_skips = 0
        else:
            state.consecutive_skips += 1
        written = []
        for slot in sorted(s for s in selection.slots if s in window.writable):
            frame = window.candidates[slot]
            if any(abs(frame.index - m.index) <= cfg.dedup_radius for m in state.memory):
                continue
            state.memory.append(frame)
            if slot in selection.boxes:
                state.boxes[frame.index] = selection.boxes[slot]
            written.append(frame.index)
        state.memory = state.memory[-cfg.capacity :]
        kept = {f.index for f in state.memory}
        state.boxes = {k: b for k, b in state.boxes.items() if k in kept}
        state.last_select_frame = window.now
        return written

    def action_memory(self, state: MemoryState) -> List[Frame]:
        return state.memory[-self.config.capacity :]

    @staticmethod
    def short_term(state: MemoryState, count: int, stride: int) -> List[Frame]:
        """Ring frames at ``now - stride * k`` for k = count..1, clamped to the ring."""
        if count <= 0:
            return []
        oldest = state.oldest
        return [
            _nearest(state.ring, max(oldest, state.now - k * stride))
            for k in range(count, 0, -1)
        ]


def needed_ring_frames(plan: Dict, now: int, last_select_frame: int) -> List[int]:
    """Ring frames a request at ``now`` reads, given the ``ring_plan`` of its task.

    These are the current frame, the selection candidates when a selection is
    due, and the short-term taps. Over a complete ring every one of them is an
    exact frame index (the window lies inside the ring), so a request that
    carries just these frames gives the server the same inputs as the full ring.
    The client mirrors this arithmetic to decide what to send.
    """
    oldest = max(0, now - int(plan["span"]))
    needed = {now}
    if plan["selection"] and now - last_select_frame >= int(plan["select_stride"]):
        start = max(0, now - int(plan["window"]))
        span = max(1, now - start)
        count = int(plan["num_candidates"])
        needed.update(
            min(max(start + int((i + 0.5) * span / count), oldest), now) for i in range(count)
        )
    taps, stride = (int(v) for v in plan["short_term"])
    needed.update(max(oldest, now - k * stride) for k in range(taps, 0, -1))
    return sorted(needed)


def box_for(state: MemoryState, frame: Frame) -> Optional[Box]:
    return state.boxes.get(frame.index)
