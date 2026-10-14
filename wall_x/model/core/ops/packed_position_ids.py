"""Packing-aware Qwen3.5 3-D mRoPE position ids."""

import torch

from wall_x.model.core.ops.qwen35_index import qwen35_get_rope_index


def packed_position_ids_available() -> bool:
    return False


def _normalize_pack_seq_lens(pack_seq_lens, batch_size: int):
    if torch.is_tensor(pack_seq_lens):
        pack_seq_lens = pack_seq_lens.detach().cpu().tolist()
    if not pack_seq_lens:
        raise ValueError("pack_seq_lens is empty for a packed Qwen3.5 batch")
    if all(isinstance(value, (int, float)) for value in pack_seq_lens):
        pack_seq_lens = [pack_seq_lens]
    if len(pack_seq_lens) != batch_size:
        raise ValueError(
            "pack_seq_lens batch mismatch: "
            f"got {len(pack_seq_lens)}, expected {batch_size}"
        )
    return [[int(value) for value in row if int(value) > 0] for row in pack_seq_lens]


def derive_mm_token_type_ids(
    input_ids,
    image_grid_thw,
    video_grid_thw,
    *,
    image_token_id,
    video_token_id,
):
    if image_grid_thw is None and video_grid_thw is None:
        return None
    result = torch.zeros_like(input_ids, dtype=torch.int32)
    if image_grid_thw is not None:
        result.masked_fill_(input_ids == image_token_id, 1)
    if video_grid_thw is not None:
        result.masked_fill_(input_ids == video_token_id, 2)
    return result


def packed_position_ids(
    input_ids,
    attention_mask,
    pack_seq_lens,
    *,
    image_token_id,
    video_token_id,
    spatial_merge_size,
    mm_token_type_ids=None,
    image_grid_thw=None,
    video_grid_thw=None,
):
    """Restart Qwen3.5 mRoPE at zero for every sub-sample in a packed row."""
    batch_size, seq_len = input_ids.shape
    lengths = _normalize_pack_seq_lens(pack_seq_lens, batch_size)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    if mm_token_type_ids is None:
        mm_token_type_ids = derive_mm_token_type_ids(
            input_ids,
            image_grid_thw,
            video_grid_thw,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
        )
    if mm_token_type_ids is None:
        mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)

    blocks = []
    for batch_idx, row_lengths in enumerate(lengths):
        offset = 0
        for length in row_lengths:
            block_end = offset + length
            if block_end > seq_len:
                raise ValueError(
                    "pack_seq_lens exceeds sequence length: "
                    f"row={batch_idx} end={block_end} seq_len={seq_len}"
                )
            blocks.append((batch_idx, offset, length))
            offset = block_end

    result = torch.zeros(
        (3, batch_size, seq_len),
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    if not blocks:
        return result
    max_len = max(length for _, _, length in blocks)
    device = input_ids.device
    rows = torch.tensor([row for row, _, _ in blocks], device=device)
    starts = torch.tensor([start for _, start, _ in blocks], device=device)
    block_lengths = torch.tensor([length for _, _, length in blocks], device=device)
    within = torch.arange(max_len, device=device)
    valid = within[None, :] < block_lengths[:, None]
    source_cols = (starts[:, None] + within[None, :]).clamp(max=seq_len - 1)
    source_rows = rows[:, None].expand_as(source_cols)
    valid_long = valid.long()
    block_input_ids = input_ids[source_rows, source_cols] * valid_long
    block_types = mm_token_type_ids[source_rows, source_cols] * valid_long
    block_attention = attention_mask[source_rows, source_cols].long() * valid_long
    # There is no packing-specific fused kernel contract yet. Force the known
    # correct PyTorch implementation instead of letting the generic proxy bind
    # permanently to a CUDA-only external_accel function after its first call.
    block_positions, _ = qwen35_get_rope_index.call_with_backend(
        "pytorch",
        block_input_ids.contiguous(),
        block_types.contiguous(),
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=block_attention.contiguous(),
        spatial_merge_size=spatial_merge_size,
    )
    result[:, source_rows[valid], source_cols[valid]] = block_positions[:, valid].to(
        result.dtype
    )
    return result


packed_position_ids_pytorch = packed_position_ids
