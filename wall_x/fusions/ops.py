import torch
import warnings
from wall_x.fusions import backend


class AsymmetricDualExpertGemm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, input_expert0, input_expert1, weight_expert0, weight_expert1, trans_b=False
    ):
        """
        Forward pass for asymmetric dual expert GEMM.

        Args:
            input_expert0: Expert 0 input [m0, k]
            input_expert1: Expert 1 input [m1, k]
            weight_expert0: Expert 0 weight [k, n0] or [n0, k] if trans_b=True
            weight_expert1: Expert 1 weight [k, n1] or [n1, k] if trans_b=True
            trans_b: Whether to transpose the weight matrices

        Returns:
            Tuple of (output_expert0, output_expert1)
        """
        # Validate inputs
        assert input_expert0.dim() == 2, "input_expert0 must be 2D"
        assert input_expert1.dim() == 2, "input_expert1 must be 2D"
        assert weight_expert0.dim() == 2, "weight_expert0 must be 2D"
        assert weight_expert1.dim() == 2, "weight_expert1 must be 2D"

        # Dimension validation depends on trans_b
        if trans_b:
            assert input_expert0.size(1) == weight_expert0.size(
                1
            ), "Expert 0 dimension mismatch (trans_b=True)"
            assert input_expert1.size(1) == weight_expert1.size(
                1
            ), "Expert 1 dimension mismatch (trans_b=True)"
        else:
            assert input_expert0.size(1) == weight_expert0.size(
                0
            ), "Expert 0 dimension mismatch (trans_b=False)"
            assert input_expert1.size(1) == weight_expert1.size(
                0
            ), "Expert 1 dimension mismatch (trans_b=False)"

        # Save tensors and trans_b for backward pass
        ctx.save_for_backward(
            input_expert0, input_expert1, weight_expert0, weight_expert1
        )
        ctx.trans_b = trans_b

        # Allocate output tensors
        m0 = input_expert0.size(0)
        m1 = input_expert1.size(0)
        n0 = weight_expert0.size(0) if trans_b else weight_expert0.size(1)
        n1 = weight_expert1.size(0) if trans_b else weight_expert1.size(1)

        output_expert0 = torch.empty(
            m0, n0, device=input_expert0.device, dtype=input_expert0.dtype
        )
        output_expert1 = torch.empty(
            m1, n1, device=input_expert1.device, dtype=input_expert1.dtype
        )

        # Call the backend C++ function
        backend.asym_dual_gmm_separated(
            input_expert0,
            input_expert1,
            weight_expert0,
            weight_expert1,
            output_expert0,
            output_expert1,
            trans_b=trans_b,
        )

        return output_expert0, output_expert1

    @staticmethod
    def backward(ctx, grad_output_expert0, grad_output_expert1):
        """
        Optimized backward pass using specialized kernels.
        Always computes all gradients to minimize kernel calls.
        """
        grad_output_expert0 = grad_output_expert0.contiguous()
        grad_output_expert1 = grad_output_expert1.contiguous()

        input_expert0, input_expert1, weight_expert0, weight_expert1 = ctx.saved_tensors
        trans_b = ctx.trans_b

        # Always allocate all gradient tensors (no conditional computation)
        grad_input_expert0 = torch.empty_like(input_expert0)
        grad_input_expert1 = torch.empty_like(input_expert1)
        grad_weight_expert0 = torch.empty_like(weight_expert0)
        grad_weight_expert1 = torch.empty_like(weight_expert1)

        # Compute input gradients: grad_input = grad_output @ weight^T (if trans_b=False)
        #                                    = grad_output @ weight   (if trans_b=True)
        backend.asym_dual_gmm_separated(
            grad_output_expert0,
            grad_output_expert1,
            weight_expert0,
            weight_expert1,
            grad_input_expert0,
            grad_input_expert1,
            trans_a=False,
            trans_b=not trans_b,
        )

        # Compute weight gradients
        if trans_b:
            # When trans_b=True in forward: output = input @ weight^T
            # So grad_weight^T = input^T @ grad_output
            # Which means grad_weight = grad_output^T @ input
            backend.asym_dual_gmm_separated(
                grad_output_expert0,
                grad_output_expert1,
                input_expert0,
                input_expert1,
                grad_weight_expert0,
                grad_weight_expert1,
                trans_a=True,
                trans_b=False,
            )
        else:
            # When trans_b=False in forward: output = input @ weight
            # So grad_weight = input^T @ grad_output
            backend.asym_dual_gmm_separated(
                input_expert0,
                input_expert1,
                grad_output_expert0,
                grad_output_expert1,
                grad_weight_expert0,
                grad_weight_expert1,
                trans_a=True,
                trans_b=False,
            )

        return (
            grad_input_expert0,
            grad_input_expert1,
            grad_weight_expert0,
            grad_weight_expert1,
            None,
        )


def asym_dual_gmm(
    input_expert0, input_expert1, weight_expert0, weight_expert1, trans_b=False
):
    """
    Convenience function for asymmetric dual expert GEMM.

    Args:
        input_expert0: Expert 0 input [m0, k]
        input_expert1: Expert 1 input [m1, k]
        weight_expert0: Expert 0 weight [k, n0] or [n0, k] if trans_b=True
        weight_expert1: Expert 1 weight [k, n1] or [n1, k] if trans_b=True
        trans_b: Whether to transpose the weight matrices

    Returns:
        Tuple of (output_expert0, output_expert1)
    """
    return AsymmetricDualExpertGemm.apply(
        input_expert0, input_expert1, weight_expert0, weight_expert1, trans_b
    )


################################################################################################
##
## PermuteMoE topK
##
################################################################################################


class PermuteMoE_topK(torch.autograd.Function):

    workspace_fw = None
    dtype = None
    max_expanded_token_num = 0

    @staticmethod
    def forward(
        ctx,
        input_act: torch.Tensor,
        indices: torch.Tensor,
        num_out_tokens: int,
        max_token_num: int,
    ):
        """
        indices: for topK=1, indices in a 1-d tensor of shape [num_tokens],
                 otherwise, it's a 2-d tensor of shape [num_tokens, topK]
        """
        # Empty input check
        if not input_act.numel():
            return input_act, None

        # For top1 case, view the indices as 2D tensor to unify the shape for topk>=2 cases.
        if indices.dim() == 1:
            indices = indices.view(-1, 1)

        # Device check
        if input_act.is_cpu:
            raise RuntimeError(
                "[Error] The input `input_act` of permute_topK op is on the device: CPU!"
            )
        if indices.is_cpu:
            warnings.warn(
                "The input `indices` of permute_topK op is on the device: CPU!"
            )

        # Shape check
        if input_act.size(0) != indices.size(0):
            raise RuntimeError(
                f"[Error] permute_topK op input `indices` shape mismatch! "
                f"Expect {input_act.size(0)}, but got {indices.size(0)}."
            )

        # Data type check
        if indices.dtype != torch.int32:
            warnings.warn(
                f"The data type of the input `indices` of permute_topK op is {indices.dtype}! "
                "The recommended type is torch.int32."
            )
            indices = indices.to(torch.int32)

        # Contiguous check
        if not input_act.is_contiguous():
            warnings.warn("The input `input_act` of permute_topK op is discontiguous!")
            input_act = input_act.contiguous()
        if not indices.is_contiguous():
            warnings.warn("The input `indices` of permute_topK op is discontiguous!")
            indices = indices.contiguous()

        num_topK = indices.size(1)

        input_max_expanded_token_num = max(max_token_num, input_act.size(0)) * num_topK
        if PermuteMoE_topK.max_expanded_token_num < input_max_expanded_token_num:
            PermuteMoE_topK.max_expanded_token_num = input_max_expanded_token_num
            PermuteMoE_topK.workspace_fw = []

        if PermuteMoE_topK.dtype != input_act.dtype:
            PermuteMoE_topK.dtype = input_act.dtype
            PermuteMoE_topK.workspace_fw = []

        permuted_act, row_id_map, PermuteMoE_topK.workspace_fw = backend.permute(
            input_act,
            indices,
            num_out_tokens,
            PermuteMoE_topK.workspace_fw,
            PermuteMoE_topK.max_expanded_token_num,
        )

        ctx.row_id_map = row_id_map
        ctx.num_tokens = indices.size(0)
        ctx.num_topK = num_topK
        return permuted_act, row_id_map

    @staticmethod
    def backward(ctx, permuted_act_grad, _):
        # Empty input check
        if not permuted_act_grad.numel():
            return permuted_act_grad, None, None, None

        if not permuted_act_grad.is_contiguous():
            permuted_act_grad = permuted_act_grad.contiguous()

        row_id_map = ctx.row_id_map
        num_tokens = ctx.num_tokens
        num_topK = ctx.num_topK

        unpermuted_act_grad = backend.unpermute(
            permuted_act_grad, row_id_map, torch.tensor([]), num_tokens, num_topK
        )
        return unpermuted_act_grad, None, None, None


################################################################################################
##
## UnpermuteMoE topK
##
################################################################################################


class UnpermuteMoE_topK(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        input_act: torch.Tensor,
        row_id_map: torch.Tensor,
        probs: torch.Tensor = None,
    ):
        # Empty input check
        if not input_act.numel():
            ctx.probs = probs
            return input_act

        # Device check
        if input_act.is_cpu:
            raise RuntimeError(
                "[Error] The input `input_act` of unpermute_topK op is on the device: CPU!"
            )
        if row_id_map.is_cpu:
            warnings.warn(
                "The input `row_id_map` of unpermute_topK op is on the device: CPU!"
            )
            row_id_map = row_id_map.cuda()
        if probs is not None and probs.is_cpu:
            warnings.warn(
                "The input `probs` of unpermute_topK op is on the device: CPU!"
            )
            probs = probs.cuda()

        # Shape check
        if probs is not None and row_id_map.size(0) != probs.size(0) * probs.size(1):
            raise RuntimeError(
                f"[Error] unpermute_topK op input `probs` shape mismatch! "
                f"Expect {row_id_map.size(0)}, but got {probs.size(0) * probs.size(1)}."
            )

        # Data type check
        if row_id_map.dtype != torch.int32:
            warnings.warn(
                f"The data type of the input `row_id_map` of unpermute_topK op is {row_id_map.dtype}! "
                "The recommended type is torch.int32."
            )
            row_id_map = row_id_map.to(torch.int32)
        if probs is not None and probs.dtype != torch.float32:
            warnings.warn(
                f"The data type of the input `probs` of unpermute_topK op is {probs.dtype}! "
                "The recommended type is torch.float32."
            )
            probs = probs.to(torch.float32)

        # Contiguous check
        if not input_act.is_contiguous():
            warnings.warn(
                "The input `input_act` of unpermute_topK op is discontiguous!"
            )
            input_act = input_act.contiguous()
        if not row_id_map.is_contiguous():
            warnings.warn(
                "The input `row_id_map` of unpermute_topK op is discontiguous!"
            )
            row_id_map = row_id_map.contiguous()
        if probs is not None and not probs.is_contiguous():
            warnings.warn("The input `probs` of unpermute_topK op is discontiguous!")
            probs = probs.contiguous()

        num_tokens = probs.size(0) if probs is not None else input_act.size(0)
        num_topK = probs.size(1) if probs is not None else 1

        unpermuted_output = backend.unpermute(
            input_act,
            row_id_map,
            probs if probs is not None else torch.tensor([]),
            num_tokens,
            num_topK,
        )

        ctx.save_for_backward(input_act, row_id_map, probs)
        return unpermuted_output

    @staticmethod
    def backward(ctx, unpermuted_act_grad):
        # Empty input check
        if not unpermuted_act_grad.numel():
            return unpermuted_act_grad, None, ctx.probs

        if not unpermuted_act_grad.is_contiguous():
            unpermuted_act_grad = unpermuted_act_grad.contiguous()

        input_act, row_id_map, probs = ctx.saved_tensors

        act_grad = None
        if ctx.needs_input_grad[0]:
            act_grad, prob_grad = backend.unpermute_bwd(
                unpermuted_act_grad, input_act, row_id_map, probs
            )

        if not ctx.needs_input_grad[2]:
            prob_grad = None
        return act_grad, None, prob_grad


def permute(input_act, indices, num_out_tokens=None, max_token_num=0):
    num_out_tokens = 0 if num_out_tokens is None else num_out_tokens
    return PermuteMoE_topK.apply(input_act, indices, num_out_tokens, max_token_num)


def unpermute(input_act, row_id_map, probs=None):
    return UnpermuteMoE_topK.apply(input_act, row_id_map, probs)


################################################################################################
##
## mutlimodal RoPE
##
################################################################################################


class MultimodalRoPE(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mrope_section: list,
    ):

        # Device check
        if q.is_cpu:
            raise RuntimeError(
                "[Error] The input `q` of multimodal_rope op is on the device: CPU!"
            )
        if k.is_cpu:
            raise RuntimeError(
                "[Error] The input `k` of multimodal_rope op is on the device: CPU!"
            )
        if cos.is_cpu:
            raise RuntimeError(
                "[Error] The input `cos` of multimodal_rope op is on the device: CPU!"
            )
        if sin.is_cpu:
            raise RuntimeError(
                "[Error] The input `sin` of multimodal_rope op is on the device: CPU!"
            )
        if len(mrope_section) != 3:
            raise RuntimeError(
                "[Error] The input `mrope_section` of multimodal_rope op must be a list of 3 integers!"
            )

        # Contiguous check
        if not q.is_contiguous():
            warnings.warn("The input `q` of multimodal_rope op is discontiguous!")
            q = q.contiguous()
        if not k.is_contiguous():
            warnings.warn("The input `k` of multimodal_rope op is discontiguous!")
            k = k.contiguous()
        if not cos.is_contiguous():
            warnings.warn("The input `cos` of multimodal_rope op is discontiguous!")
            cos = cos.contiguous()
        if not sin.is_contiguous():
            warnings.warn("The input `sin` of multimodal_rope op is discontiguous!")
            sin = sin.contiguous()

        # Prepare mrope_section_doubled
        mrope_section_doubled = [x * 2 for x in mrope_section]

        # Create output tensors
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)
        backend.rope(q, k, cos, sin, q_out, k_out, mrope_section_doubled)

        ctx.save_for_backward(q, k, cos, sin)
        ctx.mrope_section_doubled = mrope_section_doubled

        return q_out, k_out

    @staticmethod
    def backward(ctx, grad_q_out, grad_k_out):

        if not grad_q_out.is_contiguous():
            grad_q_out = grad_q_out.contiguous()
        if not grad_k_out.is_contiguous():
            grad_k_out = grad_k_out.contiguous()

        q, k, cos, sin = ctx.saved_tensors

        grad_q = None
        grad_k = None
        if ctx.needs_input_grad[0]:
            grad_q = torch.empty_like(q)
        if ctx.needs_input_grad[1]:
            grad_k = torch.empty_like(k)

        if grad_q is not None or grad_k is not None:
            backend.rope_bwd(
                grad_q_out,
                grad_k_out,
                q,
                k,
                cos,
                sin,
                grad_q if grad_q is not None else torch.empty_like(q),
                grad_k if grad_k is not None else torch.empty_like(k),
                ctx.mrope_section_doubled,
            )

        return grad_q, grad_k, None, None, None


def multimodal_rope(q, k, cos, sin, mrope_section):
    return MultimodalRoPE.apply(q, k, cos, sin, mrope_section)


################################################################################################
##
## RoPE Index 3D
##
################################################################################################


def get_rope_index(
    input_ids: torch.Tensor,
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    tokens_per_second: float,
    image_grid_thw: torch.Tensor = None,
    video_grid_thw: torch.Tensor = None,
    second_per_grid_ts: torch.Tensor = None,
    attention_mask: torch.Tensor = None,
):
    """
    Generate 3D RoPE position indices for multimodal transformer inputs.

    Computes position indices for text, image, and video tokens to enable proper
    spatial-temporal position encoding in multimodal transformers with RoPE.

    Args:
        input_ids (torch.Tensor): Input token sequence of shape [batch_size, seq_len]
                                 Must be LongTensor on CUDA device
        spatial_merge_size (int): Spatial merge size for patch grouping (must be positive)
        image_token_id (int): Token ID representing image patches
        video_token_id (int): Token ID representing video frames
        vision_start_token_id (int): Token ID marking start of vision sequences
        tokens_per_second (float): Temporal scaling factor for video sequences (must be positive)
        image_grid_thw (torch.Tensor, optional): Image grid dimensions of shape [num_images, 3] (T, H, W)
        video_grid_thw (torch.Tensor, optional): Video grid dimensions of shape [num_videos, 3] (T, H, W)
        second_per_grid_ts (torch.Tensor, optional): Video time intervals of shape [num_videos]
        attention_mask (torch.Tensor, optional): Attention mask of shape [batch_size, seq_len]

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - position_ids: 3D position indices of shape [3, batch_size, seq_len]
            - mrope_position_deltas: mRoPE position deltas of shape [batch_size, 1]

    Raises:
        TypeError: If input_ids is not a torch.Tensor
        ValueError: If input dimensions are incorrect or tensors not on CUDA
    """
    # Input validation
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("input_ids must be a torch.Tensor")

    if input_ids.dim() != 2:
        raise ValueError("input_ids must be 2D tensor (batch_size, seq_len)")

    if not input_ids.is_cuda:
        raise ValueError("input_ids must be on CUDA device")

    # Parameter validation
    if not isinstance(spatial_merge_size, int) or spatial_merge_size <= 0:
        raise ValueError(
            f"spatial_merge_size must be positive integer, got {spatial_merge_size}"
        )

    if not isinstance(tokens_per_second, (int, float)) or tokens_per_second <= 0:
        raise ValueError(
            f"tokens_per_second must be positive number, got {tokens_per_second}"
        )

    return backend.get_rope_index(
        input_ids,
        image_grid_thw,
        video_grid_thw,
        second_per_grid_ts,
        attention_mask,
        spatial_merge_size,
        image_token_id,
        video_token_id,
        vision_start_token_id,
        float(tokens_per_second),
    )


################################################################################################
##
## Fused Rotary Position Embedding
##
################################################################################################


def rot_pos_emb(
    inv_freq: torch.Tensor,
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> torch.Tensor:
    """
    Compute fused rotary position embeddings using optimized CUDA kernel.

    This function fuses all rotary position embedding computations into a single
    CUDA kernel for improved performance with spatial-temporal grids.

    Args:
        inv_freq (torch.Tensor): Inverse frequencies tensor of shape [dim/2]
                                Contains precomputed 1.0 / (theta ** (torch.arange(0, dim, 2) / dim))
                                Must be float32 on CUDA device
        grid_thw (torch.Tensor): Grid dimensions tensor of shape [num_grids, 3]
                                Each row contains (T, H, W) for temporal, height, width dimensions
                                Supports int32 or int64 on CUDA device
        spatial_merge_size (int): Spatial merge size for token grouping (must be positive)

    Returns:
        torch.Tensor: Rotary position embeddings of shape [total_tokens, dim]
                     where dim = 2 * len(inv_freq)
                     First half contains h_pos frequencies, second half contains w_pos frequencies

    Raises:
        TypeError: If inputs are not torch.Tensor or spatial_merge_size not int
        ValueError: If tensor dimensions incorrect, not on CUDA, or devices mismatch
        RuntimeError: If CUDA kernel execution fails
    """
    # Type checking
    if not isinstance(inv_freq, torch.Tensor):
        raise TypeError(f"inv_freq must be a torch.Tensor, got {type(inv_freq)}")

    if not isinstance(grid_thw, torch.Tensor):
        raise TypeError(f"grid_thw must be a torch.Tensor, got {type(grid_thw)}")

    # Dimension checking
    if inv_freq.dim() != 1:
        raise ValueError(
            f"inv_freq must be 1-dimensional, got {inv_freq.dim()}D tensor"
        )

    if grid_thw.dim() != 2:
        raise ValueError(
            f"grid_thw must be 2-dimensional, got {grid_thw.dim()}D tensor"
        )

    if grid_thw.size(1) != 3:
        raise ValueError(
            f"grid_thw must have shape [num_grids, 3], got shape {list(grid_thw.shape)}"
        )

    # Device checking
    if not inv_freq.is_cuda:
        raise ValueError("inv_freq must be on CUDA device")

    if not grid_thw.is_cuda:
        raise ValueError("grid_thw must be on CUDA device")

    # Ensure both tensors are on the same device
    if inv_freq.device != grid_thw.device:
        raise ValueError(
            f"inv_freq and grid_thw must be on the same device, "
            f"got {inv_freq.device} and {grid_thw.device}"
        )

    # Parameter validation
    if not isinstance(spatial_merge_size, int):
        raise TypeError(
            f"spatial_merge_size must be an integer, got {type(spatial_merge_size)}"
        )

    if spatial_merge_size <= 0:
        raise ValueError(
            f"spatial_merge_size must be positive, got {spatial_merge_size}"
        )

    # Ensure inv_freq is float32 (the kernel expects float)
    if inv_freq.dtype != torch.float32:
        inv_freq = inv_freq.to(torch.float32)

    # Call the CUDA backend
    try:
        return backend.rot_pos_emb(inv_freq, grid_thw, spatial_merge_size)
    except RuntimeError as e:
        raise RuntimeError(f"CUDA kernel execution failed: {str(e)}")


################################################################################################
##
## Fused Window Index Generation
##
################################################################################################


def get_window_index(
    grid_thw: torch.Tensor,
    window_size: int,
    spatial_merge_size: int,
    patch_size: int,
    spatial_merge_unit: int = 1,
):
    """
    Generate window attention indices for Vision Transformer architectures.

    Computes window-based attention indices for hierarchical processing of vision
    tokens, enabling efficient sliding window attention patterns in ViT models.

    Args:
        grid_thw (torch.Tensor): Grid specifications of shape [num_grids, 3] (T, H, W)
                                Must be or will be converted to int32 on CUDA device
        window_size (int): Window size for attention computation
        spatial_merge_size (int): Spatial merge size for patch grouping
        patch_size (int): Size of vision patches in pixels
        spatial_merge_unit (int, optional): Spatial merging unit size. Defaults to 1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - window_index: Window indices tensor of shape [total_elements]
            - cu_window_seqlens: Cumulative window sequence lengths of shape [num_windows + 1]

    Raises:
        AssertionError: If grid_thw dimensions are incorrect

    Note:
        Returns empty tensors if input grid is empty or no valid windows can be formed.
        The function automatically converts input to CUDA int32 if needed.
    """
    # Input validation
    assert (
        grid_thw.dim() == 2 and grid_thw.size(1) == 3
    ), f"grid_thw must have shape (num_grids, 3), got {grid_thw.shape}"

    # Ensure input is on CUDA and int32 type
    if not grid_thw.is_cuda:
        grid_thw = grid_thw.cuda()

    if grid_thw.dtype != torch.int32:
        grid_thw = grid_thw.to(torch.int32)

    # Calculate vit_merger_window_size
    vit_merger_window_size = window_size // spatial_merge_size // patch_size

    # Call CUDA backend
    window_index, cu_window_seqlens = backend.get_window_index(
        grid_thw,
        spatial_merge_size,
        vit_merger_window_size,
        patch_size,
        spatial_merge_unit,
    )

    return window_index, cu_window_seqlens
