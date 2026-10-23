import torch


def find_first_last_ones(tensor):
    """
    Input: tensor of shape (bs, seq_len) containing 0s and 1s
    Output: (first_indices, last_indices), each of shape (bs,)
    first_indices[i] is the first index of 1 in batch i, or -1 if none exists
    last_indices[i] is the last index of 1 in batch i, or -1 if none exists
    """
    bs, seq_len = tensor.shape
    masks = tensor == 1
    has_ones = masks.any(dim=1)

    first = torch.full((bs,), -1, dtype=torch.long, device=tensor.device)
    last = first.clone()

    # Compute the first index of 1
    first[has_ones] = torch.argmax(masks[has_ones].float(), dim=1)

    # Compute the last index of 1
    flipped_masks = masks.flip(dims=[1])
    last_argmax = torch.argmax(flipped_masks[has_ones].float(), dim=1)
    last[has_ones] = seq_len - 1 - last_argmax

    return first, last


def update_position_ids(position_ids, moe_token_types, positional_masks):
    """Extracted from ActionModelMixMin._update_position_ids (was @staticmethod)."""
    if positional_masks is None or "ar_predict_token_positions" not in positional_masks:
        return position_ids

    new_position_ids = position_ids.clone()
    ar_predict_token_positions = positional_masks["ar_predict_token_positions"]
    flow_mask = moe_token_types == 1

    start_ar_pos, end_ar_pos = find_first_last_ones(ar_predict_token_positions)
    start_flow_pos, end_flow_pos = find_first_last_ones(flow_mask)

    for bs_i in range(position_ids.shape[1]):
        if start_ar_pos[bs_i] != -1 and end_ar_pos[bs_i] != -1:
            start_ar_ids = new_position_ids[:, bs_i, start_ar_pos[bs_i]]
            start_flow_ids = new_position_ids[:, bs_i, start_flow_pos[bs_i]]
            diff = start_flow_ids - start_ar_ids
            new_position_ids[:, bs_i, start_flow_pos[bs_i] :] = position_ids[
                :, bs_i, start_flow_pos[bs_i] :
            ] - diff.unsqueeze(-1)

    return new_position_ids


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

    return attention_mask


def update_joint_attention_flash_mask(
    attention_mask,
    moe_token_types,
    positional_masks,
    causal_action_attention_mask=False,
    debug=False,
):
    """Extracted from ActionModelMixMin._update_joint_attention_flash_mask.
    The only self attribute used was self.config.causal_action_attention_mask, now passed as parameter.
    """
    device = moe_token_types.device
    B, S = moe_token_types.shape
    i32 = torch.int32

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

    flow_mask = moe_token_types == 1
    start_flow_pos, end_flow_pos = find_first_last_ones(flow_mask)
    if positional_masks is None or "ar_action_mask" not in positional_masks:
        # Handle bidirectional flow action masks
        if not causal_action_attention_mask:
            for bs_i in range(B):
                if start_flow_pos[bs_i] != -1:
                    UTE[bs_i, flow_mask[bs_i]] = start_flow_pos[bs_i].to(i32)
    else:
        # AR and flow are bidirectional
        ar_action_mask = positional_masks["ar_action_mask"] != 0
        flow_mask = moe_token_types == 1
        if positional_masks.get("ar_visible", True):
            flow_ar_position = ar_action_mask | flow_mask
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

    # Handle validation flow
    if (
        positional_masks is not None
        and "valid_flow_action_positions" in positional_masks
    ):
        flow_mask = moe_token_types == 1
        nonvalid_flow_action_positions = (
            flow_mask & ~positional_masks["valid_flow_action_positions"]
        )
        if nonvalid_flow_action_positions.any():
            LTS[nonvalid_flow_action_positions] = 0
            UTE[nonvalid_flow_action_positions] = S

    LTS = LTS.unsqueeze(-1)
    UTE = UTE.unsqueeze(-1)

    startend_row_indices = torch.cat([LTS, UTE], dim=-1)

    # add num_heads dimension
    startend_row_indices = startend_row_indices.unsqueeze(1)

    return startend_row_indices
