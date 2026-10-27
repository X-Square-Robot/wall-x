"""Model-free checks of keyframe-memory serving."""

import numpy as np
import pytest
import torch

from wall_x.serving.keyframe_memory.boxes import MIN_SIDE, crop_rect, crop_to_box
from wall_x.serving.keyframe_memory.config import KeyframeMemoryConfig
from wall_x.serving.keyframe_memory.grammar import (
    SelectionGrammar,
    SelectionTokens,
    parse_selection,
)
from wall_x.serving.keyframe_memory.prompts import IMAGE, inject_memory, selection_prompt
from wall_x.serving.keyframe_memory.state import (
    Frame,
    KeyframeBank,
    MemoryState,
    Selection,
)
from wall_x.serving.robodojo.memory_protocol import (
    initial_state,
    parse_request,
    response_state,
)
from wall_x.serving.robodojo.protocol.exceptions import WsError

IMG = np.zeros((4, 6, 3), dtype=np.uint8)


def ring(now, span=60):
    return [Frame(i, IMG) for i in range(max(0, now - span), now + 1)]


# ----------------------------------------------------------------------
# Frame arithmetic
# ----------------------------------------------------------------------
def test_window_candidates_memory_and_writable():
    bank = KeyframeBank(KeyframeMemoryConfig())
    state = MemoryState(ring=ring(96), memory=[Frame(10, IMG), Frame(60, IMG)], last_select_frame=48)
    window = bank.window(state)
    # start=48, span=48: candidate i sits at 48 + int((i + 0.5) * 3)
    assert [f.index for f in window.candidates] == [48 + int((i + 0.5) * 3) for i in range(16)]
    # memory inside the candidate window is hidden from the selector
    assert [f.index for f in window.memory] == [10]
    # writable: newer than the last selection (48) and older than now
    assert window.writable == tuple(range(16))


def test_writable_excludes_frames_before_last_selection():
    bank = KeyframeBank(KeyframeMemoryConfig())
    state = MemoryState(ring=ring(64), last_select_frame=48)
    window = bank.window(state)
    assert all(window.candidates[i].index >= 48 for i in window.writable)
    assert any(f.index < 48 for f in window.candidates)


def test_commit_dedups_trims_and_counts_skips():
    cfg = KeyframeMemoryConfig(capacity=2)
    bank = KeyframeBank(cfg)
    state = MemoryState(ring=ring(96), memory=[Frame(50, IMG)], boxes={50: (1, 2, 3, 4)})
    window = bank.window(state)
    # slot 0 -> frame 49 duplicates 50; slots 5 and 10 are new
    written = bank.commit(state, window, Selection(write=True, slots=(10, 0, 5), boxes={5: (5, 6, 7, 8)}))
    assert written == [window.candidates[5].index, window.candidates[10].index]
    assert [f.index for f in state.memory] == written  # capacity 2 drops frame 50
    assert state.boxes == {window.candidates[5].index: (5, 6, 7, 8)}
    assert state.last_select_frame == 96 and state.consecutive_skips == 0
    bank.commit(state, bank.window(state), Selection(write=False))
    assert state.consecutive_skips == 1


def test_short_term_taps_clamp_to_ring():
    state = MemoryState(ring=ring(5))
    assert [f.index for f in KeyframeBank.short_term(state, 2, 4)] == [0, 1]


# ----------------------------------------------------------------------
# Selection grammar
# ----------------------------------------------------------------------
WRITE, SKIP, END, OPEN, COMMA, MID, CLOSE = 100, 101, 102, 103, 104, 105, 106
SLOTS = tuple(range(200, 204))
DIGITS = tuple(range(10, 20))
TOKENS = SelectionTokens(WRITE, SKIP, END, SLOTS, OPEN, COMMA, MID, CLOSE, DIGITS)
VOCAB = 256


def greedy(grammar, preferences, max_steps=60):
    """Greedy decode where ``preferences`` ranks tokens (earlier = higher logit)."""
    generated = []
    for _ in range(max_steps):
        scores = torch.full((1, VOCAB), -1e3)
        for rank, tok in enumerate(preferences(generated)):
            scores[0, tok] = 100.0 - rank
        ids = torch.tensor([[0] * grammar.prompt_length + generated])
        nxt = int(grammar(ids, scores).argmax())
        generated.append(nxt)
        if nxt in TOKENS.stop:
            break
    return generated


def test_skip_and_write_without_boxes():
    g = SelectionGrammar(TOKENS, 3, max_writes=2, boxes=False)
    assert greedy(g, lambda gen: [SKIP, WRITE]) == [SKIP]
    out = greedy(g, lambda gen: [WRITE, SLOTS[2], SLOTS[0], SLOTS[1], END])
    # budget of two slots, then the answer must end
    assert out == [WRITE, SLOTS[2], SLOTS[0], END]
    assert parse_selection(TOKENS, out, 0.0).slots == (2, 0)


def test_box_digits_and_budget_merge():
    g = SelectionGrammar(TOKENS, 0, max_writes=1, boxes=True)
    digits = lambda s: [DIGITS[int(c)] for c in s]
    answer = [WRITE, SLOTS[1], OPEN] + digits("12") + [COMMA] + digits("5") + [MID] + digits("300") + [COMMA] + digits("40") + [CLOSE]

    def prefs(gen):
        nxt = answer[len(gen)] if len(gen) < len(answer) else END
        return [nxt, END]

    out = greedy(g, prefs)
    assert out == answer + [END]
    sel = parse_selection(TOKENS, out, 0.0)
    assert sel.slots == (1,) and sel.boxes == {1: (12, 5, 300, 40)}

    # Budget spent: a slot preferred over the box opener ends the answer.
    out = greedy(g, lambda gen: [WRITE] if not gen else [SLOTS[1], SLOTS[2], OPEN, END])
    assert out == [WRITE, SLOTS[1], END]


def test_digit_count_is_bounded():
    g = SelectionGrammar(TOKENS, 0, max_writes=1, boxes=True)
    allowed, _ = g.allowed([WRITE, SLOTS[0], OPEN] + [DIGITS[1]] * 4)
    assert allowed == [COMMA]


def test_degenerate_box_is_dropped():
    out = [WRITE, SLOTS[0], OPEN, DIGITS[5], COMMA, DIGITS[5], MID, DIGITS[1], COMMA, DIGITS[9], CLOSE, END]
    sel = parse_selection(TOKENS, out, 0.0)
    assert sel.slots == (0,) and sel.boxes == {}


# ----------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------
def test_selection_prompt_layout():
    text = selection_prompt("stack", num_memory=1, num_candidates=2)
    assert text == (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nTask: stack\n"
        f"Progress memory: [M_0]: {IMAGE}\n"
        f"Recent frames: <|R_0|>: {IMAGE} <|R_1|>: {IMAGE}\n"
        "Which recent frames record newly confirmed task progress that must be kept "
        "in long-term memory?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert "Progress memory: (empty)" in selection_prompt("x", 0, 2)


def test_inject_memory():
    prefix = "user\nObservation: face_view: X"
    assert inject_memory(prefix, num_memory=0, num_short_term=0, frame=None, label="m") == prefix
    assert inject_memory(prefix, num_memory=0, num_short_term=0, frame=7, label="m") == (
        "user\nObservation: now[f7]: face_view: X"
    )
    assert inject_memory(prefix, num_memory=2, num_short_term=1, frame=None, label="progress memory") == (
        f"user\nObservation: progress memory: [M_0]: {IMAGE} [M_1]: {IMAGE}"
        f" Recent view: {IMAGE} now: face_view: X"
    )


# ----------------------------------------------------------------------
# Boxes
# ----------------------------------------------------------------------
def test_crop_rect_expands_and_keeps_minimum_side():
    assert crop_rect((400, 400, 600, 600), 1000, 1000) == (360, 360, 640, 640)
    x1, y1, x2, y2 = crop_rect((0, 0, 10, 10), 448, 336)
    assert (x1, y1) == (0, 0) and x2 - x1 >= MIN_SIDE and y2 - y1 >= MIN_SIDE
    image = np.random.default_rng(0).integers(0, 255, (336, 448, 3), dtype=np.uint8)
    assert crop_to_box(image, (100, 100, 400, 500)).shape == image.shape


# ----------------------------------------------------------------------
# Wire protocol
# ----------------------------------------------------------------------
def request(now, prev, *, mem=(), last=0, history=None, **state):
    hist = history if history is not None else list(range(1 if prev is None else prev + 1, now + 1))
    return {
        "sim_frame": now,
        "kfmem_history": [{"sim_frame": f, "face_view": IMG} for f in hist],
        "kfmem_client_state_v1": {
            "version": 1,
            "context_key": "ep_0",
            "prev_sim_frame": prev,
            "ring": [{"sim_frame": f, "face_view": IMG} for f in range(max(0, now - 60), now + 1)],
            "mem": [{"sim_frame": f, "face_view": IMG} for f in mem],
            "last_sel_frame": last,
            "consec_skip": 0,
            "dump_count": 3,
            "rng_state": None,
            **state,
        },
    }


def test_protocol_round_trip():
    cfg = KeyframeMemoryConfig()
    req = parse_request(request(80, 64, mem=(30,), last=48, mem_boxes={"30": [1, 2, 3, 4]}), max_advance=16, config=cfg)
    assert req.state.now == 80 and [f.index for f in req.state.memory] == [30]
    assert req.client_boxes == {30: (1, 2, 3, 4)}
    req.state.boxes = dict(req.client_boxes)
    out = response_state(req, rng_state=b"x", include_boxes=True)
    assert out["prev_sim_frame"] == 80 and out["mem_frame_ids"] == [30]
    assert out["dump_count"] == 3 and out["mem_boxes"] == {"30": [1, 2, 3, 4]}
    assert initial_state(None)["prev_sim_frame"] is None


@pytest.mark.parametrize(
    "obs, message",
    [
        (request(80, 60), "advance"),
        (request(80, 64, history=list(range(66, 81))), "every executed frame"),
        (request(60, None, history=list(range(1, 49))), "first 48"),
        (request(80, 64, mem=(30, 30)), "unique"),
        (request(80, 64, mem_boxes={"31": [1, 2, 3, 4]}, mem=(30,)), "retained memory"),
    ],
)
def test_protocol_rejects(obs, message):
    with pytest.raises(WsError, match=message):
        parse_request(obs, max_advance=16, config=KeyframeMemoryConfig())


def test_protocol_rejects_short_ring():
    obs = request(80, 64)
    obs["kfmem_client_state_v1"]["ring"].pop(0)
    with pytest.raises(WsError, match="ring"):
        parse_request(obs, max_advance=16, config=KeyframeMemoryConfig())


def test_protocol_names_the_missing_field():
    obs = request(80, 64)
    del obs["kfmem_client_state_v1"]["consec_skip"]
    with pytest.raises(WsError, match="consec_skip must be"):
        parse_request(obs, max_advance=16, config=KeyframeMemoryConfig())


# ----------------------------------------------------------------------
# RoboDojo memory policy (no model)
# ----------------------------------------------------------------------
from types import SimpleNamespace

from wall_x.serving.robodojo.memory_policy import ClientRng, KeyframeMemoryRoboDojoPolicy
from wall_x.serving.robodojo.policy import RoboDojoPolicyConfig
from wall_x.serving.robodojo.serve import trained_with_keyframe_memory


def test_replan_steps_beyond_protocol_bound_is_rejected_before_loading():
    with pytest.raises(ValueError, match="replan_steps <= 48"):
        KeyframeMemoryRoboDojoPolicy(RoboDojoPolicyConfig(checkpoint_path="/nonexistent", replan_steps=64))


def test_client_rng_without_cuda():
    rng = ClientRng("cpu")
    rng.validate(None)
    with pytest.raises(WsError, match="unavailable"):
        rng.validate(b"x")
    with rng.loaded(None) as export:
        assert export() is None


class _FakeMemoryModel:
    def __init__(self, config):
        self.memory_config = config
        self.seen_boxes = []

    def infer_with_memory(self, observation, instruction, task_id, state):
        self.seen_boxes.append(dict(state.boxes))
        state.memory.append(state.ring[-1])
        state.boxes[state.ring[-1].index] = (1, 2, 3, 4)
        return {"robot_state_action_data": None}, None


def _policy():
    policy = object.__new__(KeyframeMemoryRoboDojoPolicy)
    policy.config = RoboDojoPolicyConfig(checkpoint_path="x", replan_steps=16)
    policy.model = _FakeMemoryModel(KeyframeMemoryConfig(box_crop=True))
    policy._rng = ClientRng("cpu")
    policy._box_cache = __import__("collections").OrderedDict()
    policy._pending = None
    policy._actions = lambda data: ["action"]
    return policy


def test_policy_returns_state_and_falls_back_to_cached_boxes():
    policy = _policy()
    first = request(16, None, history=list(range(1, 17)))
    policy._pending = (None, "go", "imitate_sorting_sequence",
                       parse_request(first, max_advance=16, config=policy.memory_config))
    out = policy.get_action()
    assert out["actions"] == ["action"]
    assert out["kfmem_state"]["mem_frame_ids"] == [16]
    assert out["kfmem_state"]["mem_boxes"] == {"16": [1, 2, 3, 4]}
    # next request without echoed boxes: the server-side cache supplies them
    second = request(32, 16, mem=(16,))
    policy._pending = (None, "go", "imitate_sorting_sequence",
                       parse_request(second, max_advance=16, config=policy.memory_config))
    policy.get_action()
    assert policy.model.seen_boxes[-1] == {16: (1, 2, 3, 4)}


def test_memory_autodetection_reads_the_resolved_epilogue():
    def cfg(epilogue):
        return SimpleNamespace(data_config=SimpleNamespace(epilogue_processor_type=epilogue))
    assert trained_with_keyframe_memory(cfg("keyframe_memory_packing"))
    assert not trained_with_keyframe_memory(cfg("wallx"))
