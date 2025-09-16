"""
High-performance C++ backend interface for optimized matrix operations.

This module provides Python bindings for custom CUDA kernels optimized for
transformer and MoE (Mixture of Experts) operations, including:
- Asymmetric dual expert operations
- Token permutation/unpermutation for MoE routing
- RoPE (Rotary Position Embedding) operations
"""

import torch
from typing import Tuple, Optional
import wallx_csrc as backend


def _allocate_asymmetric_dual_outputs(
    input_expert0: torch.Tensor,
    input_expert1: torch.Tensor,
    weight_expert0: torch.Tensor,
    weight_expert1: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Allocate output tensors for asymmetric dual expert GEMM operations.

    This function handles the case where two experts may have different output
    dimensions, which is common in heterogeneous MoE architectures.

    Args:
        input_expert0 (torch.Tensor): Expert 0 input tensor of shape [m0, k]
        input_expert1 (torch.Tensor): Expert 1 input tensor of shape [m1, k]
        weight_expert0 (torch.Tensor): Expert 0 weight tensor of shape [k, n0]
        weight_expert1 (torch.Tensor): Expert 1 weight tensor of shape [k, n1]

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Pre-allocated output tensors
            - output_expert0: Shape [m0, n0]
            - output_expert1: Shape [m1, n1]

    Raises:
        AssertionError: If tensor dimensions are incompatible
    """
    # Validate input tensor dimensions
    assert input_expert0.ndim == 2, "Expected 2D tensor for input_expert0"
    assert input_expert1.ndim == 2, "Expected 2D tensor for input_expert1"
    assert weight_expert0.ndim == 2, "Expected 2D tensor for weight_expert0"
    assert weight_expert1.ndim == 2, "Expected 2D tensor for weight_expert1"

    # Verify dimension compatibility for matrix multiplication
    assert input_expert0.size(1) == weight_expert0.size(
        0
    ), f"Input expert0 K dimension {input_expert0.size(1)} != weight expert0 K dimension {weight_expert0.size(0)}"
    assert input_expert1.size(1) == weight_expert1.size(
        0
    ), f"Input expert1 K dimension {input_expert1.size(1)} != weight expert1 K dimension {weight_expert1.size(0)}"

    # Calculate output shapes: [m, k] × [k, n] = [m, n]
    m0, n0 = input_expert0.size(0), weight_expert0.size(1)
    m1, n1 = input_expert1.size(0), weight_expert1.size(1)

    # Allocate output tensors with matching device and dtype
    output_expert0 = torch.empty(
        m0, n0, device=input_expert0.device, dtype=input_expert0.dtype
    )
    output_expert1 = torch.empty(
        m1, n1, device=input_expert1.device, dtype=input_expert1.dtype
    )

    return output_expert0, output_expert1


def asym_dual_gmm_separated(
    input_expert0: torch.Tensor,
    input_expert1: torch.Tensor,
    weight_expert0: torch.Tensor,
    weight_expert1: torch.Tensor,
    output_expert0: Optional[torch.Tensor] = None,
    output_expert1: Optional[torch.Tensor] = None,
    trans_a: bool = False,
    trans_b: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Asymmetric dual expert grouped GEMM with separated inputs and outputs.

    This is the recommended interface for maximum flexibility and performance when
    dealing with two experts that may have different intermediate dimensions.
    The operation is equivalent to:
        output_expert0 = input_expert0 @ weight_expert0
        output_expert1 = input_expert1 @ weight_expert1
    But optimized as a single fused kernel call.

    Args:
        input_expert0 (torch.Tensor): Expert 0 input tensor of shape [m0, k]
        input_expert1 (torch.Tensor): Expert 1 input tensor of shape [m1, k]
        weight_expert0 (torch.Tensor): Expert 0 weight tensor of shape [k, n0]
        weight_expert1 (torch.Tensor): Expert 1 weight tensor of shape [k, n1]
                                      Note: n0 can be different from n1
        output_expert0 (torch.Tensor, optional): Pre-allocated output for expert 0 [m0, n0]
        output_expert1 (torch.Tensor, optional): Pre-allocated output for expert 1 [m1, n1]
        trans_a (bool, optional): Whether to transpose input tensors. Defaults to False.
        trans_b (bool, optional): Whether to transpose weight tensors. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Output tensors (output_expert0, output_expert1)

    Example:
        >>> # Two experts with different output dimensions
        >>> input0 = torch.randn(512, 1024, device='cuda')  # 512 tokens for expert 0
        >>> input1 = torch.randn(256, 1024, device='cuda')  # 256 tokens for expert 1
        >>> weight0 = torch.randn(1024, 2048, device='cuda')  # Expert 0: 1024->2048
        >>> weight1 = torch.randn(1024, 4096, device='cuda')  # Expert 1: 1024->4096
        >>> out0, out1 = asym_dual_gmm_separated(input0, input1, weight0, weight1)
    """
    # Allocate outputs if not provided
    if output_expert0 is None or output_expert1 is None:
        alloc_out0, alloc_out1 = _allocate_asymmetric_dual_outputs(
            input_expert0, input_expert1, weight_expert0, weight_expert1
        )
        if output_expert0 is None:
            output_expert0 = alloc_out0
        if output_expert1 is None:
            output_expert1 = alloc_out1

    # Call optimized C++ backend kernel
    backend.asym_dual_gmm(
        input_expert0,
        input_expert1,
        weight_expert0,
        weight_expert1,
        output_expert0,
        output_expert1,
        trans_a,
        trans_b,
    )

    return output_expert0, output_expert1


def permute(
    input: torch.Tensor,
    indices: torch.Tensor,
    num_out_tokens: int,
    workspace: torch.Tensor,
    max_expanded_token_num: int,
) -> torch.Tensor:
    """
    Permute input tokens according to expert assignment indices for MoE routing.

    This function reorders tokens based on their assigned experts to enable
    efficient grouped processing. Used in the forward pass of MoE layers.

    Args:
        input (torch.Tensor): Input tokens to permute
        indices (torch.Tensor): Expert assignment indices for each token
        num_out_tokens (int): Number of output tokens after expansion
        workspace (torch.Tensor): Temporary workspace tensor for intermediate computations
        max_expanded_token_num (int): Maximum number of tokens after top-k expansion

    Returns:
        torch.Tensor: Permuted tokens grouped by expert assignment

    Note:
        This is typically used with top-k expert selection where each token
        can be routed to multiple experts.
    """
    return backend.permute(
        input, indices, num_out_tokens, workspace, max_expanded_token_num
    )


def unpermute(
    input: torch.Tensor,
    row_id_map: torch.Tensor,
    prob: torch.Tensor,
    max_tokens: int,
    num_topK: int,
) -> torch.Tensor:
    """
    Unpermute expert outputs back to original token order with probability weighting.

    This function reverses the permutation applied in the forward pass and combines
    outputs from multiple experts using their routing probabilities.

    Args:
        input (torch.Tensor): Permuted expert outputs to unpermute
        row_id_map (torch.Tensor): Mapping from permuted positions to original positions
        prob (torch.Tensor): Expert routing probabilities for weighted combination
        max_tokens (int): Maximum number of tokens in the sequence
        num_topK (int): Number of top experts selected per token

    Returns:
        torch.Tensor: Unpermuted tokens in original order with expert outputs combined

    Note:
        The output combines multiple expert predictions for each token using
        the routing probabilities as weights.
    """
    return backend.unpermute(input, row_id_map, prob, max_tokens, num_topK)


def unpermute_bwd(
    input_bwd: torch.Tensor,
    input_fwd: torch.Tensor,
    row_id_map: torch.Tensor,
    prob: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Backward pass for unpermute operation with gradient flow.

    This function handles the backward pass through the unpermute operation,
    ensuring proper gradient flow for training MoE models.

    Args:
        input_bwd (torch.Tensor): Backward gradients from the next layer
        input_fwd (torch.Tensor): Forward pass inputs (for gradient computation)
        row_id_map (torch.Tensor): Row mapping used in forward unpermute
        prob (torch.Tensor, optional): Expert probabilities. If None, uniform weights are used.

    Returns:
        torch.Tensor: Gradients with respect to the input of unpermute forward pass

    Note:
        If prob is None, uniform probabilities are assumed for gradient computation.
    """
    # Handle case where probabilities are not provided
    if prob is None:
        prob = torch.ones(
            [input_bwd.size(0), 1], dtype=torch.float32, device=input_bwd.device
        )

    return backend.unpermute_bwd(input_bwd, input_fwd, row_id_map, prob)


def rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    mrope_section_doubled: bool,
) -> None:
    """
    Apply RoPE (Rotary Position Embedding) to query and key tensors.

    Applies rotary position embeddings to query and key tensors using precomputed
    cosine and sine values. Supports both standard RoPE and multi-dimensional RoPE (mRoPE).

    Args:
        q (torch.Tensor): Query tensor to apply RoPE to
        k (torch.Tensor): Key tensor to apply RoPE to
        cos (torch.Tensor): Precomputed cosine values for rotation
        sin (torch.Tensor): Precomputed sine values for rotation
        q_out (torch.Tensor): Output tensor for rotated queries (in-place operation supported)
        k_out (torch.Tensor): Output tensor for rotated keys (in-place operation supported)
        mrope_section_doubled (bool): Whether using multi-dimensional RoPE with doubled sections

    Note:
        This function performs in-place operations if q_out and k_out point to the same
        memory as q and k respectively. The rotation is applied using the standard
        RoPE formulation with complex number rotation.
    """
    return backend.rope(q, k, cos, sin, q_out, k_out, mrope_section_doubled)


def rope_bwd(
    grad_q_out: torch.Tensor,
    grad_k_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    grad_q: torch.Tensor,
    grad_k: torch.Tensor,
    mrope_section_doubled: bool,
) -> None:
    """
    Backward pass for RoPE operation with gradient computation.

    Computes gradients with respect to the input query and key tensors
    for the RoPE operation used in transformer attention mechanisms.

    Args:
        grad_q_out (torch.Tensor): Gradient with respect to output queries
        grad_k_out (torch.Tensor): Gradient with respect to output keys
        q (torch.Tensor): Original query tensor from forward pass
        k (torch.Tensor): Original key tensor from forward pass
        cos (torch.Tensor): Cosine values used in forward pass
        sin (torch.Tensor): Sine values used in forward pass
        grad_q (torch.Tensor): Output tensor for query gradients
        grad_k (torch.Tensor): Output tensor for key gradients
        mrope_section_doubled (bool): Whether using multi-dimensional RoPE configuration

    Note:
        This function computes the analytical gradient of the RoPE operation,
        which involves the inverse rotation compared to the forward pass.
    """
    return backend.rope_bwd(
        grad_q_out, grad_k_out, q, k, cos, sin, grad_q, grad_k, mrope_section_doubled
    )


def get_rope_index(
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor],
    video_grid_thw: Optional[torch.Tensor],
    second_per_grid_ts: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    tokens_per_second: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate position indices for multimodal RoPE (Rotary Position Embedding).

    This function computes 3D position indices for text, image, and video tokens
    to enable proper spatial-temporal position encoding in multimodal transformers.

    Args:
        input_ids (torch.Tensor): Input token IDs of shape [batch_size, seq_len]
        image_grid_thw (torch.Tensor, optional): Image grid specifications of shape [num_images, 3] (T, H, W)
        video_grid_thw (torch.Tensor, optional): Video grid specifications of shape [num_videos, 3] (T, H, W)
        second_per_grid_ts (torch.Tensor, optional): Temporal scaling per video grid of shape [num_videos]
        attention_mask (torch.Tensor, optional): Attention mask of shape [batch_size, seq_len]
        spatial_merge_size (int): Spatial dimension merge factor for patch grouping
        image_token_id (int): Token ID representing image patches
        video_token_id (int): Token ID representing video frames
        vision_start_token_id (int): Token ID marking vision sequence start
        tokens_per_second (float): Temporal scaling factor for video sequences

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - position_ids: 3D position indices of shape [3, batch_size, seq_len]
            - mrope_deltas: Position deltas for multimodal RoPE of shape [batch_size, 1]

    Note:
        When both image_grid_thw and video_grid_thw are None, returns standard
        text-only position indices based on attention_mask or sequence order.
    """
    return backend.rope_index(
        input_ids,
        image_grid_thw,
        video_grid_thw,
        second_per_grid_ts,
        attention_mask,
        spatial_merge_size,
        image_token_id,
        video_token_id,
        vision_start_token_id,
        tokens_per_second,
    )


def rot_pos_emb(
    inv_freq: torch.Tensor,
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> torch.Tensor:
    """
    Compute fused rotary position embeddings for multimodal grids.

    This function efficiently computes rotary position embeddings for spatial-temporal
    grids using a fused CUDA kernel, supporting both int32 and int64 grid specifications.

    Args:
        inv_freq (torch.Tensor): Inverse frequencies for RoPE of shape [dim/2]
                                Must be float32 dtype on CUDA device
        grid_thw (torch.Tensor): Grid specifications of shape [num_grids, 3] (T, H, W)
                                Supports int32 or int64 dtype on CUDA device
        spatial_merge_size (int): Merge factor for spatial dimensions (must be positive)

    Returns:
        torch.Tensor: Computed rotary embeddings of shape [total_tokens, dim]
                     where total_tokens is determined by grid layouts and spatial_merge_size

    Example:
        >>> inv_freq = torch.randn(64, device='cuda', dtype=torch.float32)  # 128-dim model
        >>> grids = torch.tensor([[8, 14, 14], [16, 7, 7]], device='cuda', dtype=torch.int32)
        >>> embeddings = rot_pos_emb(inv_freq, grids, spatial_merge_size=2)
        >>> print(embeddings.shape)  # [computed_tokens, 128]

    Note:
        The function automatically dispatches to int32 or int64 implementations
        based on the dtype of grid_thw. Output is always float32.
    """
    return backend.rot_pos_emb(inv_freq, grid_thw, spatial_merge_size)


def get_window_index(
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    vit_merger_window_size: int,
    patch_size: int,
    spatial_merge_unit: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate window attention indices for Vision Transformer architectures.

    Computes window-based attention indices for hierarchical processing of vision
    tokens, enabling efficient sliding window attention patterns in ViT models.

    Args:
        grid_thw (torch.Tensor): Grid specifications of shape [num_grids, 3] (T, H, W)
                                Must be int32 dtype on CUDA device
        spatial_merge_size (int): Spatial dimension merge factor
        vit_merger_window_size (int): Size of attention windows for ViT processing
        patch_size (int): Size of vision patches in pixels
        spatial_merge_unit (int): Unit size for spatial merging operations

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - window_indices: Flattened window indices of shape [total_elements]
            - cu_window_seqlens: Cumulative window sequence lengths of shape [num_windows + 1]

    Example:
        >>> grids = torch.tensor([[1, 14, 14]], device='cuda', dtype=torch.int32)
        >>> indices, seqlens = get_window_index(
        ...     grids, spatial_merge_size=2, vit_merger_window_size=7,
        ...     patch_size=16, spatial_merge_unit=4
        ... )

    Note:
        Returns empty tensors if input grid is empty or no valid windows can be formed.
        The cu_window_seqlens tensor enables efficient batched attention computation.
    """
    return backend.get_window_index(
        grid_thw,
        spatial_merge_size,
        vit_merger_window_size,
        patch_size,
        spatial_merge_unit,
    )
