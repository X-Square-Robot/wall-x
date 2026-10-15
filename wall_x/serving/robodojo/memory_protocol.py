"""Wire format of client-held keyframe memory (``kfmem_client_state_v1``).

The client keeps the episode's raw head-camera frames and sends back, with each
observation, the state returned by the previous response plus the images it
references. The server therefore holds no per-episode state: reconnects,
retries and concurrent episodes cannot mix memories.

Request fields (inside the observation)::

    sim_frame               current simulator step
    kfmem_history           every head frame since the previous request
                            [{sim_frame, face_view}], including the current one
    kfmem_client_state_v1   {version: 1, context_key, prev_sim_frame,
                             ring: [{sim_frame, face_view}] for max(0,now-60)..now,
                             mem: [{sim_frame, face_view}] for the memory frames,
                             last_sel_frame, consec_skip, dump_count, rng_state,
                             mem_boxes (optional, box tasks)}

Response field ``kfmem_state``: ``version``, ``prev_sim_frame``,
``mem_frame_ids``, ``last_sel_frame``, ``consec_skip``, ``dump_count``,
``rng_state`` and, for box tasks, ``mem_boxes`` as ``{"<frame>": [x1,y1,x2,y2]}``.
``dump_count`` is not used by this server and is echoed unchanged.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from wall_x.serving.keyframe_memory.config import KeyframeMemoryConfig
from wall_x.serving.keyframe_memory.state import Box, Frame, MemoryState
from wall_x.serving.robodojo.protocol.exceptions import ErrorCode, WsError

STATE_KEY = "kfmem_client_state_v1"
HISTORY_KEY = "kfmem_history"
RESPONSE_KEY = "kfmem_state"
VERSION = 1
_CONTEXT_KEY = re.compile(r"[A-Za-z0-9_-]{1,128}")
# The first request may arrive after a warm-up of at most this many steps, and
# later requests may advance at most this many steps: the history a request
# carries is bounded by it.
MAX_ADVANCE = 48


def _invalid(message: str):
    raise WsError(ErrorCode.INVALID_FRAME, message)


def _frame_id(value: Any, field: str = "sim_frame") -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        _invalid(f"{field} must be a nonnegative integer")
    return int(value)


def _image(value: Any) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8:
        _invalid("camera images must be uint8 ndarrays")
    if value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) <= 0:
        _invalid("camera images must have shape [H, W, 3]")
    return value


def _frames(entries: Any, what: str):
    if not isinstance(entries, list):
        _invalid(f"{what} must be a list")
    frames = []
    for item in entries:
        if not isinstance(item, Mapping):
            _invalid(f"{what} entries must be mappings")
        frames.append(Frame(_frame_id(item.get("sim_frame")), _image(item.get("face_view"))))
    return frames


@dataclass
class ClientRequest:
    """A validated request: the episode state plus the fields echoed back."""

    state: MemoryState
    previous_frame: Optional[int]
    context_key: str
    rng_state: Optional[bytes]
    dump_count: int
    # Boxes the client echoed; ``None`` if it sent none (older clients).
    client_boxes: Optional[Dict[int, Box]]


def initial_state(rng_state: Optional[bytes]) -> Dict[str, Any]:
    """``reset`` result: the state a client sends with an episode's first request."""
    return {
        "version": VERSION,
        "prev_sim_frame": None,
        "mem_frame_ids": [],
        "last_sel_frame": 0,
        "consec_skip": 0,
        "dump_count": 0,
        "rng_state": rng_state,
    }


def parse_request(
    obs: Mapping[str, Any], *, max_advance: int, config: KeyframeMemoryConfig
) -> ClientRequest:
    """Validate the memory fields of one observation; raises ``WsError``."""
    now = _frame_id(obs.get("sim_frame"))
    raw = obs.get(STATE_KEY)
    if not isinstance(raw, Mapping) or raw.get("version") != VERSION:
        _invalid(f"{STATE_KEY}.version must equal {VERSION}")

    previous = raw.get("prev_sim_frame")
    if previous is not None:
        previous = _frame_id(previous, "prev_sim_frame")
        if previous >= now:
            _invalid("prev_sim_frame must be less than sim_frame")
    _check_history(obs.get(HISTORY_KEY, []), now, previous, max_advance)

    ring_raw = raw.get("ring")
    if not isinstance(ring_raw, list) or len(ring_raw) > config.ring_span + 1:
        _invalid(f"ring must be a list of at most {config.ring_span + 1} frames")
    ring = _frames(ring_raw, "ring")
    expected = list(range(max(0, now - config.ring_span), now + 1))
    if [f.index for f in ring] != expected:
        _invalid(f"ring must contain every raw frame from max(0,now-{config.ring_span}) through now")

    mem_raw = raw.get("mem")
    if not isinstance(mem_raw, list) or len(mem_raw) > config.capacity:
        _invalid("mem must be a list within model capacity")
    memory = _frames(mem_raw, "mem")
    if any(f.index > now for f in memory):
        _invalid("mem frame must not exceed sim_frame")
    if len({f.index for f in memory}) != len(memory):
        _invalid("mem frame IDs must be unique")

    last_select = _frame_id(raw.get("last_sel_frame"), "last_sel_frame")
    if last_select > now:
        _invalid("last_sel_frame must not exceed sim_frame")
    context_key = raw.get("context_key")
    if not isinstance(context_key, str) or not _CONTEXT_KEY.fullmatch(context_key):
        _invalid("context_key must match [A-Za-z0-9_-]{1,128}")
    rng_state = raw.get("rng_state")
    if rng_state is not None and not isinstance(rng_state, bytes):
        _invalid("rng_state must be bytes or null")

    return ClientRequest(
        state=MemoryState(
            ring=ring,
            memory=memory,
            last_select_frame=last_select,
            consecutive_skips=_frame_id(raw.get("consec_skip"), "consec_skip"),
        ),
        previous_frame=previous,
        context_key=context_key,
        rng_state=rng_state,
        dump_count=_frame_id(raw.get("dump_count"), "dump_count"),
        client_boxes=_parse_boxes(raw.get("mem_boxes"), [f.index for f in memory]),
    )


def _check_history(history: Any, now: int, previous: Optional[int], max_advance: int):
    if not isinstance(history, list) or len(history) > MAX_ADVANCE:
        _invalid(f"{HISTORY_KEY} must be a list of at most {MAX_ADVANCE} execution frames")
    frames = [f.index for f in _frames(history, "history")]
    if previous is None:
        if now > MAX_ADVANCE:
            _invalid(f"first observation must start within the first {MAX_ADVANCE} steps")
        expected = list(range(1, now + 1))
    else:
        if now - previous > max_advance:
            _invalid("sim_frame must advance by the executed chunk length; reset between episodes")
        expected = list(range(previous + 1, now + 1))
    if frames != expected:
        _invalid(
            f"{HISTORY_KEY} must contain every executed frame in increasing order, "
            "including current frame"
        )


def _parse_boxes(raw: Any, memory_frames: list) -> Optional[Dict[int, Box]]:
    """``{"<frame>": [x1,y1,x2,y2]}`` (or a list aligned with ``mem``) -> boxes."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        items = []
        for key, box in raw.items():
            if not isinstance(key, str) or not key.isdecimal() or int(key) not in memory_frames:
                _invalid("mem_boxes keys must be decimal frame ids of retained memory frames")
            items.append((int(key), box))
    elif isinstance(raw, list) and len(raw) == len(memory_frames):
        items = list(zip(memory_frames, raw))
    else:
        _invalid("mem_boxes must be a {frame: box} map or a list aligned with mem")
    boxes = {}
    for frame, box in items:
        if box is None:
            continue
        if (
            not isinstance(box, (list, tuple))
            or len(box) != 4
            or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in box)
            or not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000)
        ):
            _invalid(
                "mem_boxes entries must be null or [x1,y1,x2,y2] ints in 0..1000 "
                "with x1<x2, y1<y2"
            )
        boxes[frame] = tuple(int(v) for v in box)
    return boxes


def response_state(
    request: ClientRequest, *, rng_state: Optional[bytes], include_boxes: bool
) -> Dict[str, Any]:
    state = request.state
    out = {
        "version": VERSION,
        "prev_sim_frame": state.now,
        "mem_frame_ids": [f.index for f in state.memory],
        "last_sel_frame": state.last_select_frame,
        "consec_skip": state.consecutive_skips,
        "dump_count": request.dump_count,
        "rng_state": rng_state,
    }
    if include_boxes:
        out["mem_boxes"] = {
            str(f.index): list(state.boxes[f.index])
            for f in state.memory
            if f.index in state.boxes
        }
    return out
