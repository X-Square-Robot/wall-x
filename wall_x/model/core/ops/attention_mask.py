import logging

import torch

logger = logging.getLogger(__name__)

# ops is the sole boundary that touches external_accel: import the kernels directly
# here (capability-based — present in the loaded build => used) and expose the
# operators (kernel + PyTorch fallback) so higher layers import only from ops.
try:
    from external_accel import (
        qwen35_update_joint_attention_flash_mask as _external_accel_update_flash_mask,
        qwen35_update_position_ids as _external_accel_update_position_ids,
    )
except Exception:
    _external_accel_update_flash_mask = None
    _external_accel_update_position_ids = None


def _handle_external_accel_attention_mask_error(exc):
    # Never swallow silently: an intermittently failing kernel degrading to the
    # slow path must be visible in logs.
    logger.warning(
        "external_accel attention-mask kernel failed (%s); using PyTorch fallback", exc
    )


def find_first_last_ones(tensor):
    """
    Input: tensor of shape (bs, seq_len) containing 0s and 1s
    Output: (first_indices, last_indices), each of shape (bs,)
    first_indices[i] is the first index of 1 in batch i, or -1 if none exists
    last_indices[i] is the last index of 1 in batch i, or -1 if none exists
    """
    # Branchless: the old boolean-mask writes (`first[has_ones] = ...`)
    # synchronized host and device on every call; `torch.where` keeps the
    # whole computation on-device with identical results (long dtype, -1
    # sentinel for rows without ones).
    bs, seq_len = tensor.shape
    masks = tensor == 1
    has_ones = masks.any(dim=1)
    none_found = torch.full((bs,), -1, dtype=torch.long, device=tensor.device)

    masks_i8 = masks.to(torch.int8)
    first = torch.where(has_ones, masks_i8.argmax(dim=1), none_found)
    last = torch.where(
        has_ones, seq_len - 1 - masks_i8.flip(dims=[1]).argmax(dim=1), none_found
    )

    return first, last


def update_position_ids(position_ids, moe_token_types, positional_masks):
    """Extracted from ActionModelMixMin._update_position_ids (was @staticmethod)."""
    if positional_masks is None:
        return position_ids

    if (
        _external_accel_update_position_ids is not None
        and torch.is_tensor(position_ids)
        and position_ids.is_cuda
        and torch.is_tensor(moe_token_types)
        and moe_token_types.is_cuda
    ):
        try:
            return _external_accel_update_position_ids(
                position_ids,
                moe_token_types,
                ar_predict_token_positions=positional_masks.get(
                    "ar_predict_token_positions"
                ),
                flow_block_id=positional_masks.get("flow_block_id"),
                rtc_pinned=positional_masks.get("rtc_pinned"),
                rtc_rope_gap=positional_masks.get("rtc_rope_gap", 0),
            )
        except Exception as exc:
            _handle_external_accel_attention_mask_error(exc)

    new_position_ids = position_ids.clone()
    flow_mask = moe_token_types == 1

    if "ar_predict_token_positions" in positional_masks:
        ar_predict_token_positions = positional_masks["ar_predict_token_positions"]

        start_ar_pos, end_ar_pos = find_first_last_ones(ar_predict_token_positions)
        start_flow_pos, _ = find_first_last_ones(flow_mask)

        for bs_i in range(position_ids.shape[1]):
            if start_ar_pos[bs_i] != -1 and end_ar_pos[bs_i] != -1:
                start_ar_ids = new_position_ids[:, bs_i, start_ar_pos[bs_i]]
                start_flow_ids = new_position_ids[:, bs_i, start_flow_pos[bs_i]]
                diff = start_flow_ids - start_ar_ids
                new_position_ids[:, bs_i, start_flow_pos[bs_i] :] = position_ids[
                    :, bs_i, start_flow_pos[bs_i] :
                ] - diff.unsqueeze(-1)

    # 修复: K 个 flow block 共享 baseline 位置(数据驱动多样本 A′)。dataset 把 input_ids
    # tile 成 K 个 flow block 但不产出 position_ids; compute_3d 在【扩展后】序列上算出递增
    # 位置 → K 个并行去噪样本拿到各不相同的 RoPE 位置, 破坏"同一段轨迹的并行样本共享
    # baseline"。用 dataset 提供的 flow_block_id 把每个 block reset 成 block 0 的位置; 在
    # rope_gap 之前(gap 是各 block 基于自身 rtc_pinned 的后续修饰)。
    # 注: 仅在会运行 _update_position_ids 的注意力实现下生效(flash_mask / sdpa /
    # flash_attention_2_ki / flash_attention_2_triton); 普通 flash_attention_2 本就不支持
    # 多样本(flow block 无隔离), 见 docs/multisample_rtc_attention_support.md。
    flow_block_id = positional_masks.get("flow_block_id")
    if flow_block_id is not None:
        for bs_i in range(new_position_ids.shape[1]):
            fb = flow_block_id[bs_i]
            if not bool((fb >= 0).any()):
                continue
            n_blocks = int(fb[fb >= 0].max().item()) + 1
            block0 = (fb == 0).nonzero(as_tuple=True)[0]
            for blk in range(1, n_blocks):
                blk_pos = (fb == blk).nonzero(as_tuple=True)[0]
                if blk_pos.numel() != block0.numel():
                    raise ValueError(
                        "flow_block_id requires equal-sized flow blocks for "
                        "position id reset: "
                        f"batch={bs_i}, block=0 has {block0.numel()} tokens, "
                        f"block={blk} has {blk_pos.numel()} tokens"
                    )
                new_position_ids[:, bs_i, blk_pos] = new_position_ids[:, bs_i, block0]

    rtc_rope_gap = positional_masks.get("rtc_rope_gap", 0)
    if isinstance(rtc_rope_gap, torch.Tensor):
        rtc_rope_gap = rtc_rope_gap.item()
    rtc_rope_gap = int(rtc_rope_gap or 0)
    if rtc_rope_gap > 0 and "rtc_pinned" in positional_masks:
        rtc_pinned = positional_masks["rtc_pinned"].bool()
        rtc_suffix_flow = flow_mask & ~rtc_pinned
        if rtc_suffix_flow.any():
            new_position_ids = new_position_ids + (
                rtc_suffix_flow.unsqueeze(0).to(new_position_ids.dtype) * rtc_rope_gap
            )

    return new_position_ids


def update_position_ids_packed(
    position_ids, moe_token_types, positional_masks, pack_seq_lens
):
    """Apply legacy action-position transforms independently per packed segment."""
    if torch.is_tensor(pack_seq_lens):
        pack_seq_lens = pack_seq_lens.detach().cpu().tolist()
    if pack_seq_lens and all(
        isinstance(value, (int, float)) for value in pack_seq_lens
    ):
        pack_seq_lens = [pack_seq_lens]
    result = position_ids.clone()
    for batch_idx, lengths in enumerate(pack_seq_lens):
        offset = 0
        for length in lengths:
            length = int(length)
            end = offset + length
            segment_masks = {}
            for key, value in (positional_masks or {}).items():
                if torch.is_tensor(value) and value.ndim >= 2:
                    segment_masks[key] = value[batch_idx : batch_idx + 1, offset:end]
                elif key != "pack_seq_lens":
                    segment_masks[key] = value
            result[:, batch_idx : batch_idx + 1, offset:end] = update_position_ids(
                position_ids[:, batch_idx : batch_idx + 1, offset:end],
                moe_token_types[batch_idx : batch_idx + 1, offset:end],
                segment_masks,
            )
            offset = end
    return result


def _apply_multisample_flow_blocks_flash(
    LTS,
    UTE,
    moe_token_types,
    flow_block_id,
    rtc_pinned=None,
    pack_block_start=None,
):
    """Restrict FlashMask visibility by dataset-provided flow block ids."""
    i32 = LTS.dtype
    B, S = LTS.shape
    flow_seq = moe_token_types == 1
    if not bool(flow_seq.any()):
        return

    valid_block = flow_seq & (flow_block_id >= 0)
    if not bool(valid_block.any()):
        return

    n_local_blocks = int(flow_block_id[valid_block].max().item()) + 1
    if pack_block_start is not None:
        group_id = flow_block_id + pack_block_start.long() * n_local_blocks
    else:
        group_id = flow_block_id
    n_blocks = int(group_id[valid_block].max().item()) + 1
    safe_block_id = torch.where(valid_block, group_id, 0).long()
    pos = torch.arange(S, device=flow_block_id.device).expand(B, S)

    block_start = torch.full((B, n_blocks), S, dtype=torch.long, device=pos.device)
    block_end = torch.full((B, n_blocks), -1, dtype=torch.long, device=pos.device)
    block_start.scatter_reduce_(
        1,
        safe_block_id,
        torch.where(valid_block, pos, S),
        reduce="amin",
        include_self=True,
    )
    block_end.scatter_reduce_(
        1,
        safe_block_id,
        torch.where(valid_block, pos, -1),
        reduce="amax",
        include_self=True,
    )

    UTE[valid_block] = block_start.gather(1, safe_block_id)[valid_block].to(i32)
    LTS[valid_block] = (block_end.gather(1, safe_block_id)[valid_block] + 1).to(i32)

    if rtc_pinned is not None:
        pinned = rtc_pinned.bool() & valid_block
        suffix = valid_block & ~rtc_pinned.bool()
        suffix_start = torch.full((B, n_blocks), S, dtype=torch.long, device=pos.device)
        suffix_start.scatter_reduce_(
            1,
            safe_block_id,
            torch.where(suffix, pos, S),
            reduce="amin",
            include_self=True,
        )

        ones = torch.ones_like(safe_block_id, dtype=torch.long)
        pinned_counts = torch.zeros((B, n_blocks), dtype=torch.long, device=pos.device)
        suffix_counts = torch.zeros_like(pinned_counts)
        pinned_counts.scatter_add_(1, safe_block_id, torch.where(pinned, ones, 0))
        suffix_counts.scatter_add_(1, safe_block_id, torch.where(suffix, ones, 0))
        mixed_block = (pinned_counts > 0) & (suffix_counts > 0)
        suffix_in_mixed_block = suffix & mixed_block.gather(1, safe_block_id)
        UTE[suffix_in_mixed_block] = suffix_start.gather(1, safe_block_id)[
            suffix_in_mixed_block
        ].to(i32)


def _apply_multisample_flow_blocks_2d(attention_mask, flow_block_id, rtc_pinned=None):
    """Apply per-block flow attention and RTC asymmetric masking to dense masks."""
    flow_seq = flow_block_id >= 0
    if not bool(flow_seq.any()):
        return attention_mask

    zero = torch.zeros((), dtype=attention_mask.dtype, device=attention_mask.device)
    one = torch.ones((), dtype=attention_mask.dtype, device=attention_mask.device)

    flow_pairs = flow_seq[:, :, None] & flow_seq[:, None, :]
    attention_mask = torch.where(flow_pairs, zero, attention_mask)

    same_block_pairs = flow_pairs & (
        flow_block_id[:, :, None] == flow_block_id[:, None, :]
    )
    attention_mask = torch.where(same_block_pairs, one, attention_mask)

    if rtc_pinned is not None:
        pinned_query = rtc_pinned.bool()[:, :, None]
        suffix_key = ~rtc_pinned.bool()[:, None, :]
        pinned_to_suffix = same_block_pairs & pinned_query & suffix_key
        attention_mask = torch.where(pinned_to_suffix, zero, attention_mask)

    return attention_mask


def update_joint_attention_mask_2d(
    attention_mask,
    moe_token_types,
    positional_masks,
    causal_action_attention_mask=False,
):
    """Extracted from ActionModelMixMin._update_joint_attention_mask_2d.
    The only self attribute used was self.config.causal_action_attention_mask, now passed as parameter.
    """
    if attention_mask.dim() == 3:  # bs, seq_len, seq_len
        return attention_mask

    bs, seq_len = moe_token_types.shape[0], moe_token_types.shape[1]
    # Create a lower-triangular causal mask
    causal_mask = torch.tril(
        torch.ones(
            (seq_len, seq_len), dtype=torch.bfloat16, device=moe_token_types.device
        )
    )
    # Expand to the batch dimension
    attention_mask = causal_mask.unsqueeze(0).expand(bs, -1, -1)

    if positional_masks is not None and "padding_positions" in positional_masks:
        padding_positions = positional_masks["padding_positions"]
        # Set padding rows to zero
        attention_mask = torch.where(
            padding_positions[:, None, :],
            torch.zeros_like(attention_mask),
            attention_mask,
        )
        # Set padding columns to zero
        attention_mask = torch.where(
            padding_positions[:, :, None],
            torch.zeros_like(attention_mask),
            attention_mask,
        )

    # Set the moe1 region to 1 and mask it from the fast region
    moe1_mask = (moe_token_types[:, :, None]) & (moe_token_types[:, None, :])

    if (
        not causal_action_attention_mask
    ):  # If causal action attention mask is disabled, set the whole moe1 region to 1
        attention_mask = torch.where(
            moe1_mask, torch.ones_like(attention_mask), attention_mask
        )

    if (
        positional_masks is not None
        and "ar_predict_token_positions" in positional_masks
    ):
        ar_predict_token_positions = positional_masks["ar_predict_token_positions"]
        moe1_mask = (moe_token_types[:, :, None]) & (
            ar_predict_token_positions[:, None, :]
        )
        attention_mask = torch.where(
            moe1_mask, torch.zeros_like(attention_mask), attention_mask
        )

    # AR and flow are bidirectional
    if positional_masks is not None and "ar_action_mask" in positional_masks:
        ar_action_mask = positional_masks["ar_action_mask"] != 0
        flow_positions = moe_token_types == 1
        if positional_masks.get("ar_visible", True):
            flow_ar_position = ar_action_mask | flow_positions
            flow_ar_mask = flow_ar_position[:, :, None] & flow_ar_position[:, None, :]
            attention_mask = torch.where(
                flow_ar_mask, torch.ones_like(attention_mask), attention_mask
            )
        else:
            flow_flow_mask = flow_positions[:, :, None] & flow_positions[:, None, :]
            ar_ar_mask = ar_action_mask[:, :, None] & ar_action_mask[:, None, :]
            flow_ar_mask = flow_flow_mask | ar_ar_mask

            affected = ar_action_mask | flow_positions  # (B, N)
            affected_pair = affected[:, :, None] & affected[:, None, :]
            attention_mask = attention_mask.masked_fill(affected_pair, 0)

            attention_mask = torch.where(
                flow_ar_mask, torch.ones_like(attention_mask), attention_mask
            )

    if positional_masks is not None and "flow_block_id" in positional_masks:
        rtc_pinned = positional_masks.get("rtc_pinned")
        attention_mask = _apply_multisample_flow_blocks_2d(
            attention_mask, positional_masks["flow_block_id"], rtc_pinned
        )

    if (
        positional_masks is not None
        and "valid_flow_action_positions" in positional_masks
    ):
        # true in moe_token_types but false in valid_flow_action_positions
        nonvalid_flow_action_positions = (
            moe_token_types & ~positional_masks["valid_flow_action_positions"]
        )
        attention_mask = torch.where(
            nonvalid_flow_action_positions[:, None, :],
            torch.zeros_like(attention_mask),
            attention_mask,
        )
        attention_mask = torch.where(
            nonvalid_flow_action_positions[:, :, None],
            torch.zeros_like(attention_mask),
            attention_mask,
        )

    return attention_mask


def pack_block_bounds(seq_lens_per_sample, seq_len, device):
    """Return per-token packed segment bounds as int32 ``[B, S]`` tensors."""
    if torch.is_tensor(seq_lens_per_sample):
        seq_lens_per_sample = seq_lens_per_sample.detach().cpu().tolist()
    if seq_lens_per_sample and all(
        isinstance(value, (int, float)) for value in seq_lens_per_sample
    ):
        seq_lens_per_sample = [seq_lens_per_sample]
    start = torch.zeros(
        (len(seq_lens_per_sample), seq_len), dtype=torch.int32, device=device
    )
    end = torch.full_like(start, seq_len)
    for batch_idx, lengths in enumerate(seq_lens_per_sample):
        offset = 0
        for length in lengths:
            length = int(length)
            if length <= 0:
                continue
            block_end = offset + length
            if block_end > seq_len:
                raise ValueError(
                    "pack_seq_lens exceeds sequence length: "
                    f"row={batch_idx} end={block_end} seq_len={seq_len}"
                )
            start[batch_idx, offset:block_end] = offset
            end[batch_idx, offset:block_end] = block_end
            offset = block_end
    return start, end


def update_joint_attention_mask_2d_packed(
    attention_mask,
    moe_token_types,
    positional_masks,
    pack_block_start,
    pack_block_end,
    causal_action_attention_mask=False,
):
    """Build the legacy dense mask, then apply strict block-diagonal isolation."""
    if attention_mask is not None and attention_mask.dim() == 3:
        result = attention_mask
    else:
        if moe_token_types is None:
            if attention_mask is None:
                raise ValueError("packed 2-D attention requires attention_mask")
            moe_token_types = torch.zeros_like(attention_mask, dtype=torch.bool)
        result = update_joint_attention_mask_2d(
            attention_mask,
            moe_token_types,
            positional_masks,
            causal_action_attention_mask=causal_action_attention_mask,
        )
    key = torch.arange(result.shape[-1], device=result.device).view(1, 1, -1)
    same_block = (key >= pack_block_start[:, :, None]) & (
        key < pack_block_end[:, :, None]
    )
    return torch.where(same_block, result, torch.zeros_like(result))


def update_joint_attention_flash_mask(
    attention_mask,
    moe_token_types,
    positional_masks,
    causal_action_attention_mask=False,
    debug=False,
    batch_size=None,
    seq_len=None,
    device=None,
    pack_block_start=None,
    pack_block_end=None,
):
    """Extracted from ActionModelMixMin._update_joint_attention_flash_mask.
    The only self attribute used was self.config.causal_action_attention_mask, now passed as parameter.
    """
    if moe_token_types is None:
        if batch_size is None or seq_len is None:
            if attention_mask is None:
                raise ValueError(
                    "`batch_size` and `seq_len` are required when "
                    "`moe_token_types` and `attention_mask` are both None."
                )
            batch_size, seq_len = attention_mask.shape[:2]
        if device is None:
            if attention_mask is not None:
                device = attention_mask.device
            elif positional_masks:
                for value in positional_masks.values():
                    if hasattr(value, "device"):
                        device = value.device
                        break
                if device is None:
                    raise ValueError(
                        "`device` is required when `positional_masks` has no tensors."
                    )
            else:
                raise ValueError("`device` is required when `moe_token_types` is None.")
    else:
        device = moe_token_types.device
        batch_size, seq_len = moe_token_types.shape

    B, S = batch_size, seq_len
    i32 = torch.int32

    if (
        _external_accel_update_flash_mask is not None
        and moe_token_types is not None
        and torch.is_tensor(moe_token_types)
        and moe_token_types.is_cuda
        and pack_block_start is None
    ):
        masks = positional_masks or {}
        try:
            return _external_accel_update_flash_mask(
                moe_token_types,
                padding_positions=masks.get("padding_positions"),
                ar_predict_token_positions=masks.get("ar_predict_token_positions"),
                valid_flow_action_positions=masks.get("valid_flow_action_positions"),
                ar_action_mask=masks.get("ar_action_mask"),
                flow_block_id=masks.get("flow_block_id"),
                rtc_pinned=masks.get("rtc_pinned"),
                causal_action_attention_mask=causal_action_attention_mask,
                ar_visible=masks.get("ar_visible", True),
            )
        except Exception as exc:
            _handle_external_accel_attention_mask_error(exc)

    # ---- Return-vector initialization ----
    LTS = torch.ones((B, S), device=device, dtype=i32) * S
    UTE = torch.arange(S, device=device, dtype=i32).unsqueeze(0).expand(B, S).clone()

    # Handle padding positions
    if positional_masks is not None and "padding_positions" in positional_masks:
        padding_positions = positional_masks["padding_positions"]
        LTS[padding_positions] = 0
        UTE[padding_positions] = S

    # Handle AR predict tokens
    if (
        positional_masks is not None
        and "ar_predict_token_positions" in positional_masks
    ):
        start_ar_pos, end_ar_pos = find_first_last_ones(
            positional_masks["ar_predict_token_positions"]
        )
        for bs_i in range(B):
            if end_ar_pos[bs_i] != -1:
                LTS[bs_i, positional_masks["ar_predict_token_positions"][bs_i]] = (
                    end_ar_pos[bs_i].to(i32) + 1
                )

    flow_mask = moe_token_types == 1 if moe_token_types is not None else None
    if positional_masks is None or "ar_action_mask" not in positional_masks:
        # Handle bidirectional flow action masks
        if flow_mask is not None and not causal_action_attention_mask:
            start_flow_pos, _ = find_first_last_ones(flow_mask)
            for bs_i in range(B):
                if start_flow_pos[bs_i] != -1:
                    UTE[bs_i, flow_mask[bs_i]] = start_flow_pos[bs_i].to(i32)
    else:
        # AR and flow are bidirectional
        ar_action_mask = positional_masks["ar_action_mask"] != 0
        if positional_masks.get("ar_visible", True):
            flow_ar_position = (
                ar_action_mask if flow_mask is None else ar_action_mask | flow_mask
            )
            for bs_i in range(B):
                idx = flow_ar_position[bs_i].nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                block_start = idx.min()
                block_end = idx.max() + 1
                # Set the visible range of every token in the block to [block_start, block_end)
                UTE[bs_i, idx] = block_start.to(i32)
                LTS[bs_i, idx] = block_end.to(i32)
        else:
            for bs_i in range(B):
                # 1) Flow sub-block: flow attends only to flow
                if flow_mask is not None:
                    flow_idx = flow_mask[bs_i].nonzero(as_tuple=True)[0]
                    if flow_idx.numel() > 0:
                        flow_start = flow_idx.min()
                        flow_end = flow_idx.max() + 1
                        # Only flow-token rows are set to [flow_start, flow_end)
                        UTE[bs_i, flow_idx] = flow_start.to(i32)
                        LTS[bs_i, flow_idx] = flow_end.to(i32)

                # 2) AR sub-block: AR attends only to AR
                ar_idx = ar_action_mask[bs_i].nonzero(as_tuple=True)[0]
                if ar_idx.numel() > 0:
                    ar_start = ar_idx.min()
                    ar_end = ar_idx.max() + 1
                    # Only AR-token rows are set to [ar_start, ar_end)
                    UTE[bs_i, ar_idx] = ar_start.to(i32)
                    LTS[bs_i, ar_idx] = ar_end.to(i32)

    if (
        positional_masks is not None
        and "flow_block_id" in positional_masks
        and flow_mask is not None
    ):
        _apply_multisample_flow_blocks_flash(
            LTS,
            UTE,
            moe_token_types,
            positional_masks["flow_block_id"],
            positional_masks.get("rtc_pinned"),
            pack_block_start=pack_block_start,
        )

    # Handle validation flow
    if (
        positional_masks is not None
        and "valid_flow_action_positions" in positional_masks
        and flow_mask is not None
    ):
        nonvalid_flow_action_positions = (
            flow_mask & ~positional_masks["valid_flow_action_positions"]
        )
        if nonvalid_flow_action_positions.any():
            LTS[nonvalid_flow_action_positions] = 0
            UTE[nonvalid_flow_action_positions] = S

    if pack_block_start is not None:
        if pack_block_end is None:
            raise ValueError("packed FlashMask requires both block bounds")
        UTE = torch.maximum(UTE, pack_block_start.to(device=device, dtype=i32))
        LTS = torch.minimum(LTS, pack_block_end.to(device=device, dtype=i32))

    LTS = LTS.unsqueeze(-1)
    UTE = UTE.unsqueeze(-1)

    startend_row_indices = torch.cat([LTS, UTE], dim=-1)

    # add num_heads dimension
    startend_row_indices = startend_row_indices.unsqueeze(1)

    return startend_row_indices


def update_joint_attention_flash_mask_packed(
    attention_mask,
    moe_token_types,
    positional_masks,
    pack_block_start,
    pack_block_end,
    **kwargs,
):
    """Packing-safe FlashMask using the Python path, never legacy external_accel."""
    return update_joint_attention_flash_mask(
        attention_mask,
        moe_token_types,
        positional_masks,
        pack_block_start=pack_block_start,
        pack_block_end=pack_block_end,
        **kwargs,
    )
