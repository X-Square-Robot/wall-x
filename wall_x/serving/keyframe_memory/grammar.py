"""Constrained decoding of the selection answer.

The selector answers one of::

    <|kf_skip|>
    <|kf_write|><|R_i|><|R_j|>...<|kf_end|>
    <|kf_write|><|R_i|> (x1,y1),(x2,y2)<|R_j|>...<|kf_end|>     (box tasks)

with distinct slots, at most ``max_writes`` of them, and box corners as 0..1000
integers of at most four digits. :class:`SelectionGrammar` is a logits
processor that masks everything else, so greedy decoding can only produce
strings the training renderer could have produced; :func:`parse_selection`
turns the generated tokens back into a :class:`Selection`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from wall_x.serving.keyframe_memory.state import Box, Selection

BOX_SCALE = 1000
_MAX_DIGITS = 4
# Training-format box text and its expected tokenization; checked against the
# checkpoint tokenizer so the grammar matches the trained answer format.
_BOX_SAMPLE = " (212,405),(378,592)"
_BOX_PIECES = [" (", "2", "1", "2", ",", "4", "0", "5", "),(", "3", "7", "8", ",", "5", "9", "2", ")"]


@dataclass(frozen=True)
class SelectionTokens:
    write: int
    skip: int
    end: int
    slots: Tuple[int, ...]
    # Box grammar: " (" d+ "," d+ "),(" d+ "," d+ ")"
    box_open: Optional[int] = None
    comma: Optional[int] = None
    middle: Optional[int] = None
    box_close: Optional[int] = None
    digits: Tuple[int, ...] = ()

    @classmethod
    def from_tokenizer(cls, tokenizer, num_candidates: int, boxes: bool):
        ids = {
            name: tokenizer.convert_tokens_to_ids(token)
            for name, token in (
                ("write", "<|kf_write|>"),
                ("skip", "<|kf_skip|>"),
                ("end", "<|kf_end|>"),
            )
        }
        slots = tuple(
            tokenizer.convert_tokens_to_ids(f"<|R_{i}|>") for i in range(num_candidates)
        )
        unknown = {tokenizer.unk_token_id, None}
        if set(ids.values()) & unknown or set(slots) & unknown:
            raise ValueError(
                "tokenizer lacks the keyframe-memory special tokens "
                "(<|kf_write|>, <|kf_skip|>, <|kf_end|>, <|R_i|>); this checkpoint "
                "was not trained for keyframe memory"
            )
        if not boxes:
            return cls(slots=slots, **ids)
        sample = tokenizer(_BOX_SAMPLE, add_special_tokens=False)["input_ids"]
        digits = tuple(tokenizer.convert_tokens_to_ids(str(d)) for d in range(10))
        if [tokenizer.decode([t]) for t in sample] != _BOX_PIECES or len(set(digits)) != 10:
            raise ValueError("unexpected box tokenization; the box grammar would not match training")
        return cls(
            slots=slots,
            box_open=sample[0],
            comma=sample[4],
            middle=sample[8],
            box_close=sample[16],
            digits=digits,
            **ids,
        )

    @property
    def stop(self) -> List[int]:
        return [self.skip, self.end]


class _Cursor:
    """Where a partial answer stands in the grammar."""

    def __init__(self):
        self.started = False  # <|kf_write|> emitted
        self.chosen: List[int] = []
        self.after_slot = False  # last token was a slot: a box may open
        self.box_seps: Optional[List[int]] = None  # separators still to come
        self.num_digits = 0


def _walk(tokens: SelectionTokens, generated: Sequence[int]) -> _Cursor:
    cursor = _Cursor()
    for tok in generated:
        if not cursor.started:
            cursor.started = tok == tokens.write
            continue
        if cursor.box_seps is not None:
            if tok == cursor.box_seps[0]:
                cursor.box_seps.pop(0)
                cursor.num_digits = 0
                if not cursor.box_seps:
                    cursor.box_seps = None
            else:
                cursor.num_digits += 1
            continue
        if tok == tokens.box_open and cursor.after_slot:
            cursor.box_seps = [tokens.comma, tokens.middle, tokens.comma, tokens.box_close]
            cursor.num_digits = 0
            cursor.after_slot = False
        elif tok in tokens.slots:
            cursor.chosen.append(tokens.slots.index(tok))
            cursor.after_slot = True
    return cursor


class SelectionGrammar:
    """``LogitsProcessor`` restricting greedy decoding to valid answers."""

    def __init__(self, tokens: SelectionTokens, prompt_length: int, max_writes: int, boxes: bool):
        self.tokens = tokens
        self.prompt_length = prompt_length
        self.max_writes = max_writes
        self.boxes = boxes and tokens.box_open is not None

    def allowed(self, generated: Sequence[int]) -> Tuple[List[int], List[int]]:
        """Return ``(allowed, merged_into_end)`` for the next token.

        Tokens in ``merged_into_end`` compete for the argmax but stand for
        stopping: once the write budget is spent the decoder still compares a
        box against every alternative, and any alternative ends the answer.
        """
        t = self.tokens
        if not generated:
            return [t.write, t.skip], []
        cursor = _walk(t, generated)
        if cursor.box_seps is not None:
            allowed = list(t.digits) if cursor.num_digits < _MAX_DIGITS else []
            if cursor.num_digits:
                allowed.append(cursor.box_seps[0])
            return allowed, []
        remaining = [s for i, s in enumerate(t.slots) if i not in cursor.chosen]
        box = [t.box_open] if self.boxes and cursor.after_slot else []
        if len(cursor.chosen) >= self.max_writes:
            return box + [t.end], remaining
        return box + remaining + [t.end], []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        masked = torch.full_like(scores, float("-inf"))
        for row in range(input_ids.shape[0]):
            allowed, merged = self.allowed(input_ids[row, self.prompt_length :].tolist())
            masked[row, allowed] = scores[row, allowed]
            if merged:
                end = self.tokens.end
                masked[row, end] = torch.max(scores[row, merged + [end]])
        return masked


def parse_selection(tokens: SelectionTokens, generated: Sequence[int], margin: float) -> Selection:
    if not generated or generated[0] != tokens.write:
        return Selection(write=False, margin=margin)
    slots: List[int] = []
    boxes: Dict[int, Box] = {}
    digit_of = {tok: d for d, tok in enumerate(tokens.digits)}
    values: List[int] = []
    current = ""
    in_box = False
    for tok in generated[1:]:
        if in_box:
            if tok in digit_of:
                current += str(digit_of[tok])
                continue
            values.append(int(current))
            current = ""
            if tok == tokens.box_close:
                in_box = False
                x1, y1, x2, y2 = (min(BOX_SCALE, v) for v in values)
                # Training skips degenerate annotation boxes; such a frame
                # stays uncropped rather than guessed.
                if x1 < x2 and y1 < y2:
                    boxes[slots[-1]] = (x1, y1, x2, y2)
            continue
        if tok == tokens.box_open:
            in_box, values = True, []
        elif tok in tokens.slots:
            slots.append(tokens.slots.index(tok))
    return Selection(write=True, slots=tuple(slots), boxes=boxes, margin=margin)
