"""Sparse ring: a request may carry only the ring frames the server reads."""

import numpy as np
import pytest

from wall_x.serving.keyframe_memory.config import KeyframeMemoryConfig
from wall_x.serving.keyframe_memory.state import (
    Frame,
    KeyframeBank,
    MemoryState,
    needed_ring_frames,
)
from wall_x.serving.robodojo.memory_protocol import parse_request, response_state
from wall_x.serving.robodojo.protocol.exceptions import WsError

IMG = np.zeros((4, 6, 3), dtype=np.uint8)
CONVEYOR = "match_and_pick_from_conveyor"  # the task with short-term taps


def frames(indices):
    return [Frame(i, np.full((4, 6, 3), i % 251, dtype=np.uint8)) for i in indices]


@pytest.mark.parametrize("task", ["swap_blocks", CONVEYOR])
@pytest.mark.parametrize("now", [0, 5, 15, 47, 48, 63, 64, 95, 111, 127, 300])
@pytest.mark.parametrize("last", [0, 16, 48, 63])
def test_sparse_ring_reads_the_same_frames(task, now, last):
    cfg = KeyframeMemoryConfig()
    if last > now:
        return
    bank = KeyframeBank(cfg)
    oldest = max(0, now - cfg.ring_span)
    full = MemoryState(ring=frames(range(oldest, now + 1)), last_select_frame=last, ring_start=oldest)
    needed = needed_ring_frames(cfg.ring_plan(task), now, last)
    sparse = MemoryState(ring=frames(needed), last_select_frame=last, ring_start=oldest)
    assert sparse.now == full.now == now
    if bank.selection_due(full):
        assert [f.index for f in bank.window(sparse).candidates] == [
            f.index for f in bank.window(full).candidates
        ]
    taps = cfg.short_term_taps(task)
    assert [f.index for f in bank.short_term(sparse, *taps)] == [
        f.index for f in bank.short_term(full, *taps)
    ]


def sparse_request(now, prev, *, ring, last=0, history=()):
    return {
        "sim_frame": now,
        "task_id": "swap_blocks",
        "kfmem_history": [{"sim_frame": f, "face_view": IMG} for f in history],
        "kfmem_client_state_v1": {
            "version": 1,
            "context_key": "ep_0",
            "prev_sim_frame": prev,
            "ring": [{"sim_frame": f, "face_view": IMG} for f in ring],
            "mem": [],
            "last_sel_frame": last,
            "consec_skip": 0,
            "dump_count": 0,
            "rng_state": None,
            "ring_sparse": True,
        },
    }


def test_sparse_request_round_trip():
    cfg = KeyframeMemoryConfig()
    needed = needed_ring_frames(cfg.ring_plan("swap_blocks"), 63, 0)
    assert len(needed) == 1 + cfg.num_candidates  # selection due: now + candidates
    req = parse_request(sparse_request(63, 47, ring=needed), max_advance=16, config=cfg)
    assert req.state.now == 63 and req.state.oldest == 3
    out = response_state(req, rng_state=None, include_boxes=False, ring_plan=cfg.ring_plan("swap_blocks"))
    assert out["ring_plan"]["select_stride"] == cfg.select_stride
    # no selection due: the current frame is all the server reads
    assert needed_ring_frames(cfg.ring_plan("swap_blocks"), 79, 63) == [79]


@pytest.mark.parametrize(
    "obs, message",
    [
        (sparse_request(63, 47, ring=[63]), "lacks frames"),
        (sparse_request(15, None, ring=[15]), "first request"),
        (sparse_request(79, 63, ring=[79], last=63, history=range(64, 80)), "must not send"),
        (sparse_request(79, 63, ring=[70], last=63), "ending at now"),
    ],
)
def test_sparse_request_rejects(obs, message):
    with pytest.raises(WsError, match=message):
        parse_request(obs, max_advance=16, config=KeyframeMemoryConfig())
