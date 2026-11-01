"""Qwen3.5 RVQ position-id rewrite operator (``qwen35_rvq_position_ids``).

Rewrites mrope position ids for serialized RVQ action spans: every
``<rvq_start>`` token anchors a span of ``num_latents * num_quantizers`` code
tokens whose temporal route (dim 1) becomes ``anchor + 1 + time`` and whose
level routes (dims 2 and 3) become ``anchor + 1 + level``, where ``anchor`` is
the pre-rewrite dim-1 position of the ``<rvq_start>`` token and ``time`` /
``level`` are derived from the slot index by ``token_layout``.

The PyTorch fallback is fully vectorized: the hot path issues **exactly one**
device→host sync (the final ``bool(err.any())`` validation check) instead of
the per-span ``nonzero``/``tolist``/``.all()`` cascade of the old
``RVQPositionMapper.apply`` loop. On the (rare) error path it re-runs a slow
reference check to reproduce the old, detailed error messages.
"""

import logging

import torch

logger = logging.getLogger(__name__)

# ops is the sole boundary that touches external_accel: import the kernel directly
# here (capability-based — present in the build => used); without a build,
# dispatch resolves to the PyTorch fallback below.
from wall_x.model.core.ops.base import load_kernels

_external_accel_rvq_position_ids = getattr(load_kernels(), "qwen35_rvq_position_ids", None)


def qwen35_rvq_position_ids_available() -> bool:
    """True iff the loaded external_accel build provides the fused kernel."""
    return _external_accel_rvq_position_ids is not None


def _check_rvq_position_inputs(
    input_ids: torch.Tensor, position_ids: torch.Tensor
) -> None:
    # Same messages as the historical RVQPositionMapper.apply checks.
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


def _raise_rvq_span_error(
    input_ids: torch.Tensor,
    *,
    start_token_id: int,
    level_lut: torch.Tensor,
    num_latents: int,
    num_quantizers: int,
    time_major: bool,
) -> None:
    """Slow re-check that reproduces the reference implementation's error
    messages (and its batch → span → slot iteration order). Only reached after
    the vectorized validation flagged an error, so syncs are fine here."""
    num_code_tokens = num_latents * num_quantizers
    ids_cpu = input_ids.detach().cpu()
    lut_cpu = level_lut.detach().cpu()
    lut_len = lut_cpu.numel()
    seq_len = ids_cpu.shape[1]
    for batch_idx in range(ids_cpu.shape[0]):
        starts = (ids_cpu[batch_idx] == start_token_id).nonzero().flatten().tolist()
        for start in starts:
            end = start + 1 + num_code_tokens
            if end > seq_len:
                raise ValueError(
                    f"RVQ span at token {start} needs {num_code_tokens} "
                    "code tokens but reaches the sequence end"
                )
            for slot in range(num_code_tokens):
                if time_major:
                    level = slot % num_quantizers
                else:
                    level = slot // num_latents
                token = int(ids_cpu[batch_idx, start + 1 + slot])
                token_level = int(lut_cpu[token]) if 0 <= token < lut_len else -1
                if token_level != level:
                    raise ValueError(
                        f"RVQ level {level} slot contains token id {token}, "
                        "which belongs to a different level"
                    )
    raise RuntimeError(
        "qwen35_rvq_position_ids flagged an invalid RVQ span, but the detailed "
        "re-check found none; level_lut and span geometry disagree"
    )


def qwen35_rvq_position_ids_pytorch(
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    start_token_id: int,
    level_lut: torch.Tensor,
    num_latents: int,
    num_quantizers: int,
    time_major: bool,
    reference=None,
):
    """Vectorized PyTorch fallback.

    Args:
        input_ids: ``[B, L]`` token ids.
        position_ids: ``[3|4, B, L]``; a 3-route input is expanded to 4 routes
            by duplicating route 0.
        start_token_id: id of ``<rvq_start>``.
        level_lut: 1-D int8 lookup ``token id -> quantizer level`` (``-1`` for
            non-code tokens). Ids outside the table are treated as non-code.
        num_latents / num_quantizers: span geometry; a span holds
            ``num_latents * num_quantizers`` code tokens.
        time_major: slot order — ``time, level = divmod(slot, num_quantizers)``
            when True, else ``level, time = divmod(slot, num_latents)``.
        reference: optional zero-arg callable invoked (and expected to raise
            with the detailed legacy message) when validation fails; when None
            an internal re-check raises the same messages.

    The hot path performs exactly one device→host sync (``bool(err.any())``).
    """
    _check_rvq_position_inputs(input_ids, position_ids)
    num_code_tokens = num_latents * num_quantizers
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    output = (
        torch.cat((position_ids[:1], position_ids), dim=0).clone()
        if position_ids.shape[0] == 3
        else position_ids.clone()
    )

    idx = torch.arange(seq_len, device=device, dtype=torch.long)
    start_mask = input_ids == start_token_id
    # Nearest span start at or before each position (-1 where none): positions
    # are increasing, so a running max over (start ? idx : -1) yields the
    # latest start index seen so far.
    last_start = torch.cummax(torch.where(start_mask, idx, -1), dim=1).values
    offset = idx - last_start  # 0 at the start token itself, 1.. in the span
    in_span = (last_start >= 0) & (offset >= 1) & (offset <= num_code_tokens)
    slot = (offset - 1).clamp(min=0)
    if time_major:
        times = torch.div(slot, num_quantizers, rounding_mode="floor")
        levels = slot.remainder(num_quantizers)
    else:
        levels = torch.div(slot, num_latents, rounding_mode="floor")
        times = slot.remainder(num_latents)

    # Anchor = pre-rewrite dim-1 position of the span's start token. Start
    # positions are never rewritten, so gathering from the original values is
    # order-independent.
    anchor = output[1].gather(1, last_start.clamp(min=0))

    # ---- validation (single fused device→host sync) ----
    lut = level_lut.to(device=device)
    safe_ids = input_ids.clamp(min=0, max=lut.numel() - 1)
    token_level = torch.where(
        (input_ids >= 0) & (input_ids < lut.numel()), lut[safe_ids].long(), -1
    )
    err_token = in_span & (token_level != levels)
    # A span running past the row end.
    err_tail = start_mask & (idx >= seq_len - num_code_tokens)
    # A start inside a previous span (start positions have offset 0 and skip
    # the in_span slot check above, so overlap needs its own detector).
    prev_start = torch.cat(
        (last_start.new_full((batch_size, 1), -1), last_start[:, :-1]), dim=1
    )
    err_overlap = start_mask & (prev_start >= 0) & (idx - prev_start <= num_code_tokens)
    err = err_token | err_tail | err_overlap
    if bool(err.any()):  # the one allowed sync; True only on the error path
        if reference is not None:
            reference()
            raise RuntimeError(
                "qwen35_rvq_position_ids flagged an invalid RVQ span, but the "
                "reference implementation did not raise"
            )
        _raise_rvq_span_error(
            input_ids,
            start_token_id=start_token_id,
            level_lut=level_lut,
            num_latents=num_latents,
            num_quantizers=num_quantizers,
            time_major=time_major,
        )

    output[1] = torch.where(in_span, anchor + 1 + times, output[1])
    output[2] = torch.where(in_span, anchor + 1 + levels, output[2])
    output[3] = torch.where(in_span, anchor + 1 + levels, output[3])
    return output


def qwen35_rvq_position_ids(
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    start_token_id: int,
    level_lut: torch.Tensor,
    num_latents: int,
    num_quantizers: int,
    time_major: bool,
    reference=None,
):
    """Capability-dispatched entry: external_accel kernel when the loaded build
    provides it (and inputs are on CUDA), else the PyTorch fallback."""
    _check_rvq_position_inputs(input_ids, position_ids)
    if (
        _external_accel_rvq_position_ids is not None
        and input_ids.is_cuda
        and position_ids.is_cuda
    ):
        try:
            return _external_accel_rvq_position_ids(
                input_ids,
                position_ids,
                start_token_id=start_token_id,
                level_lut=level_lut.to(device=input_ids.device),
                num_latents=num_latents,
                num_quantizers=num_quantizers,
                time_major=time_major,
            )
        except Exception as exc:
            # Never swallow silently: a failing kernel degrading to the slow
            # path must be visible in logs.
            logger.warning(
                "external_accel qwen35_rvq_position_ids failed (%s); "
                "using PyTorch fallback",
                exc,
            )
    return qwen35_rvq_position_ids_pytorch(
        input_ids,
        position_ids,
        start_token_id=start_token_id,
        level_lut=level_lut,
        num_latents=num_latents,
        num_quantizers=num_quantizers,
        time_major=time_major,
        reference=reference,
    )
