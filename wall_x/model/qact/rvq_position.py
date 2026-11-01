"""Wall-VLM compatible RVQ position semantics for Qwen3.5 training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from wall_x.model.core.ops.seq_rvq import qwen35_rvq_position_ids


RVQ_POSITION_ENCODINGS = frozenset({"text_1d", "time_level_2d"})
RVQ_TOKEN_LAYOUTS = frozenset({"time_major", "layer_major"})


def normalize_rvq_position_encoding(value: Any) -> str:
    encoding = str(value or "text_1d")
    if encoding not in RVQ_POSITION_ENCODINGS:
        raise ValueError(
            "rvq_position_encoding must be one of "
            f"{sorted(RVQ_POSITION_ENCODINGS)}, got {encoding!r}"
        )
    return encoding


def normalize_rvq_token_layout(value: Any) -> str:
    layout = str(value or "layer_major")
    if layout not in RVQ_TOKEN_LAYOUTS:
        raise ValueError(
            f"rvq_token_layout must be one of {sorted(RVQ_TOKEN_LAYOUTS)}, "
            f"got {layout!r}"
        )
    return layout


def describe_rvq_tokenizer_geometry(
    codec: Any,
    *,
    action_horizon: int,
    token_layout: str,
    action_dim: int | None = None,
) -> dict[str, int | str]:
    values = {
        "action_dim": int(
            getattr(codec, "action_dim", 0) if action_dim is None else action_dim
        ),
        "action_horizon": int(action_horizon),
        "codebook_size": int(codec.codebook_size),
        "compression_ratio": int(codec.compression_ratio),
        "num_quantizers": int(codec.num_quantizers),
    }
    if min(values.values()) <= 0:
        raise ValueError(f"RVQ tokenizer geometry must be positive, got {values}")
    values["num_latents"] = (
        values["action_horizon"] + values["compression_ratio"] - 1
    ) // values["compression_ratio"]
    values["token_layout"] = normalize_rvq_token_layout(token_layout)
    return values


def validate_rvq_tokenizer_geometry(
    actual: Mapping[str, Any] | None,
    expected: Mapping[str, Any],
    *,
    context: str,
) -> None:
    if actual is None:
        raise ValueError(f"{context} has no rvq_tokenizer_geometry metadata")
    mismatches = [
        f"{key}: saved={actual.get(key)!r}, current={value!r}"
        for key, value in expected.items()
        if actual.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            f"RVQ tokenizer geometry differs from {context}: " + "; ".join(mismatches)
        )


@dataclass(frozen=True)
class RVQPositionMapper:
    """Map one serialized RVQ span to ``[causal,time,level,level]``."""

    start_token_id: int
    quantizer_token_ids: tuple[frozenset[int], ...]
    num_latents: int
    token_layout: str

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: Any,
        codec: Any,
        *,
        action_horizon: int,
        token_layout: str,
        action_dim: int | None = None,
    ) -> "RVQPositionMapper":
        geometry = describe_rvq_tokenizer_geometry(
            codec,
            action_horizon=action_horizon,
            token_layout=token_layout,
            action_dim=action_dim,
        )
        unk_id = getattr(tokenizer, "unk_token_id", None)

        def token_id(token: str) -> int:
            value = tokenizer.convert_tokens_to_ids(token)
            if value is None or (unk_id is not None and int(value) == int(unk_id)):
                raise ValueError(f"tokenizer is missing required RVQ token {token!r}")
            return int(value)

        quantizer_token_id_lists = tuple(
            tuple(
                token_id(f"<rvq_r{q}_{code}>")
                for code in range(int(codec.codebook_size))
            )
            for q in range(int(codec.num_quantizers))
        )
        all_ids = [token_id("<rvq_start>")]
        all_ids.extend(token for ids in quantizer_token_id_lists for token in ids)
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("RVQ special tokens must map to distinct tokenizer IDs")
        quantizer_token_ids = tuple(frozenset(ids) for ids in quantizer_token_id_lists)
        for level, ids in enumerate(quantizer_token_ids):
            ordered = sorted(ids)
            if ordered != list(range(ordered[0], ordered[-1] + 1)):
                raise ValueError(
                    f"RVQ level {level} token IDs must form one contiguous span"
                )
        return cls(
            start_token_id=all_ids[0],
            quantizer_token_ids=quantizer_token_ids,
            num_latents=int(geometry["num_latents"]),
            token_layout=str(geometry["token_layout"]),
        )

    @property
    def num_quantizers(self) -> int:
        return len(self.quantizer_token_ids)

    @property
    def num_code_tokens(self) -> int:
        return self.num_latents * self.num_quantizers

    def _time_level(self, code_slot: int) -> tuple[int, int]:
        if self.token_layout == "time_major":
            return divmod(code_slot, self.num_quantizers)
        level, time = divmod(code_slot, self.num_latents)
        return time, level

    def _level_lut_for(self, device: torch.device | str) -> torch.Tensor:
        """Per-device cached int8 lookup ``token id -> quantizer level`` (-1
        for non-code tokens), built lazily from ``quantizer_token_ids``."""
        cache = self.__dict__.get("_level_lut_cache")
        if cache is None:
            cache = {}
            # frozen dataclass: bypass the setattr guard for the lazy cache
            object.__setattr__(self, "_level_lut_cache", cache)
        device = torch.device(device)
        lut = cache.get(device)
        if lut is None:
            cpu = torch.device("cpu")
            base = cache.get(cpu)
            if base is None:
                if self.num_quantizers > 127:
                    raise ValueError(
                        "RVQ level lookup table is int8 and supports at most "
                        f"127 quantizer levels, got {self.num_quantizers}"
                    )
                size = max(max(ids) for ids in self.quantizer_token_ids) + 1
                base = torch.full((size,), -1, dtype=torch.int8)
                for level, ids in enumerate(self.quantizer_token_ids):
                    base[sorted(ids)] = level
                cache[cpu] = base
            lut = base.to(device)
            cache[device] = lut
        return lut

    def apply(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        """Thin shell over the ``qwen35_rvq_position_ids`` operator; the old
        per-span loop is kept as :meth:`_apply_reference` (error-detail replay
        and test reference)."""
        return qwen35_rvq_position_ids(
            input_ids,
            position_ids,
            start_token_id=self.start_token_id,
            level_lut=self._level_lut_for(input_ids.device),
            num_latents=self.num_latents,
            num_quantizers=self.num_quantizers,
            time_major=self.token_layout == "time_major",
            reference=lambda: self._apply_reference(input_ids, position_ids),
        )

    def _apply_reference(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        """Original per-span loop implementation (many host syncs). Retained
        as the semantic reference for tests and to reproduce detailed error
        messages when the vectorized operator flags an invalid span."""
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B,L], got {input_ids.shape}")
        if position_ids.ndim != 3 or position_ids.shape[0] not in (3, 4):
            raise ValueError(
                "position_ids must have shape [3|4,B,L], got " f"{position_ids.shape}"
            )
        if tuple(position_ids.shape[-2:]) != tuple(input_ids.shape):
            raise ValueError(
                f"input/position shapes differ: {input_ids.shape} vs "
                f"{position_ids.shape}"
            )

        output = (
            torch.cat((position_ids[:1], position_ids), dim=0).clone()
            if position_ids.shape[0] == 3
            else position_ids.clone()
        )
        for batch_idx in range(input_ids.shape[0]):
            starts = (input_ids[batch_idx] == self.start_token_id).nonzero().flatten()
            for start in starts.tolist():
                end = start + 1 + self.num_code_tokens
                if end > input_ids.shape[1]:
                    raise ValueError(
                        f"RVQ span at token {start} needs {self.num_code_tokens} "
                        "code tokens but reaches the sequence end"
                    )
                anchor = output[1, batch_idx, start]
                slots = torch.arange(
                    self.num_code_tokens,
                    device=input_ids.device,
                    dtype=torch.long,
                )
                if self.token_layout == "time_major":
                    times = torch.div(slots, self.num_quantizers, rounding_mode="floor")
                    levels = slots.remainder(self.num_quantizers)
                else:
                    levels = torch.div(slots, self.num_latents, rounding_mode="floor")
                    times = slots.remainder(self.num_latents)

                span_tokens = input_ids[batch_idx, start + 1 : end]
                valid = torch.zeros_like(span_tokens, dtype=torch.bool)
                for level, ids in enumerate(self.quantizer_token_ids):
                    level_slots = levels == level
                    valid[level_slots] = (span_tokens[level_slots] >= min(ids)) & (
                        span_tokens[level_slots] <= max(ids)
                    )
                if not bool(valid.all()):
                    bad_slot = int((~valid).nonzero()[0])
                    bad_level = int(levels[bad_slot])
                    bad_token = int(span_tokens[bad_slot])
                    raise ValueError(
                        f"RVQ level {bad_level} slot contains token id {bad_token}, "
                        "which belongs to a different level"
                    )

                output[1, batch_idx, start + 1 : end] = anchor + 1 + times
                output[2, batch_idx, start + 1 : end] = anchor + 1 + levels
                output[3, batch_idx, start + 1 : end] = anchor + 1 + levels
        return output


__all__ = [
    "RVQPositionMapper",
    "describe_rvq_tokenizer_geometry",
    "normalize_rvq_position_encoding",
    "normalize_rvq_token_layout",
    "validate_rvq_tokenizer_geometry",
]
