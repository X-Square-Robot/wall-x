"""MoE routing operators (permute/unpermute).

These operators reorder tokens by expert assignment for efficient
Mixture-of-Experts processing.
"""

import logging

import torch
from wall_x.model.core.action.layout import TwoExpertLayout
from wall_x.model.core.ops.base import OpsProxy

logger = logging.getLogger(__name__)
_SLOW_TWO_EXPERT_FALLBACK_WARNED = set()


def _warn_slow_two_expert_fallback(op_name: str) -> None:
    if op_name in _SLOW_TWO_EXPERT_FALLBACK_WARNED:
        return
    _SLOW_TWO_EXPERT_FALLBACK_WARNED.add(op_name)
    logger.warning(
        "%s: using PyTorch fallback with per-batch Python loops. "
        "This path is much slower than the external_accel CUDA kernel; install/build "
        "external_accel for production training.",
        op_name,
    )


class PermuteOp(OpsProxy):
    """Reorder tokens by expert assignment for MoE processing.

    Signature: permute(tokens, indices, num_out_tokens=None, max_token_num=0) -> (permuted_tokens, sorted_indices)
    """

    @property
    def _external_accel_name(self):
        return "permute"

    def _get_cuda_kernel(self):
        try:
            from wall_x.model.core.ops._cuda_wrappers import permute_kernel

            return permute_kernel
        except ImportError:
            return None
        except Exception as e:
            logger.warning("PermuteOp: CUDA kernel load failed: %s", e)
            return None

    def _pytorch_fallback(self, tokens, indices, num_out_tokens=None, max_token_num=0):
        """PyTorch fallback for permute.

        Args:
            num_out_tokens: If set, truncate output to this many tokens (matches
                external_accel behavior of discarding padding expert assignments).
            max_token_num: Unused, kept for external_accel API compatibility.
        """
        del max_token_num  # unused, external_accel API compat
        if indices.dim() == 1:
            indices = indices.view(-1, 1)
        expand_factor = indices.size(1)
        flatten_indices = indices.view(-1)
        # Keep int64 throughout to avoid precision loss on large token counts
        sorted_indices = torch.argsort(flatten_indices, stable=True)
        permuted_tokens = tokens.index_select(0, sorted_indices // expand_factor)
        if num_out_tokens is not None:
            permuted_tokens = permuted_tokens[:num_out_tokens]
            sorted_indices = sorted_indices[:num_out_tokens]
        return permuted_tokens, sorted_indices


class UnpermuteOp(OpsProxy):
    """Restore tokens to original order after MoE processing.

    Signature: unpermute(permuted_tokens, sorted_indices, probs=None) -> restored_tokens
    """

    @property
    def _external_accel_name(self):
        return "unpermute"

    def _get_cuda_kernel(self):
        try:
            from wall_x.model.core.ops._cuda_wrappers import unpermute_kernel

            return unpermute_kernel
        except ImportError:
            return None
        except Exception as e:
            logger.warning("UnpermuteOp: CUDA kernel load failed: %s", e)
            return None

    def _pytorch_fallback(self, permuted_tokens, sorted_indices, probs=None):
        if probs is not None:
            merge_factor = probs.size(1)
        else:
            merge_factor = 1
        unpermuted_tokens = torch.zeros_like(permuted_tokens)
        unpermuted_tokens.index_copy_(0, sorted_indices.long(), permuted_tokens)
        unpermuted_tokens = unpermuted_tokens.reshape(
            -1, merge_factor, permuted_tokens.size(-1)
        )
        if probs is not None:
            unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)
        unpermuted_tokens = unpermuted_tokens.sum(dim=1)
        return unpermuted_tokens


permute = PermuteOp()
unpermute = UnpermuteOp()


class _PermGatherPack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, flat, perm, perm_inv):
        ctx.save_for_backward(perm_inv)
        return flat.index_select(0, perm)

    @staticmethod
    def backward(ctx, grad_packed):
        (perm_inv,) = ctx.saved_tensors
        return grad_packed.index_select(0, perm_inv), None, None


class _PermGatherUnpack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, packed, perm, perm_inv):
        ctx.save_for_backward(perm)
        return packed.index_select(0, perm_inv)

    @staticmethod
    def backward(ctx, grad_out):
        (perm,) = ctx.saved_tensors
        return grad_out.index_select(0, perm), None, None


class TwoExpertPackOp(OpsProxy):
    """Pack fixed two-expert VLA tokens without sorting.

    Input shape is [B, S, D]. Tokens in [action_starts[b], action_ends[b])
    are routed to expert 1/action; all other tokens are routed to expert 0/LLM. Output
    shape is [B*S, D], ordered as [all LLM tokens][all action tokens].
    """

    @property
    def _external_accel_name(self):
        return "two_expert_pack"

    def __call__(
        self,
        tokens,
        layout_or_action_starts,
        action_ends=None,
        llm_offsets=None,
        action_offsets=None,
    ):
        if isinstance(layout_or_action_starts, TwoExpertLayout):
            layout = layout_or_action_starts
            if layout.token_perm is not None:
                if tokens.dim() != 3:
                    raise RuntimeError("two_expert_pack expects input shape [B, S, D]")
                flat = tokens.reshape(-1, tokens.shape[-1])
                if layout.token_perm_inv is None:
                    raise RuntimeError("packed TwoExpertLayout is missing inverse permutation")
                perm = layout.token_perm.to(device=flat.device)
                perm_inv = layout.token_perm_inv.to(device=flat.device)
                if perm.numel() != flat.shape[0]:
                    raise RuntimeError("packed TwoExpertLayout does not cover all tokens")
                return _PermGatherPack.apply(flat, perm, perm_inv)
            return super().__call__(
                tokens,
                layout.action_starts,
                layout.action_ends,
                layout.llm_offsets,
                layout.action_offsets,
            )
        if llm_offsets is None or action_offsets is None:
            raise RuntimeError("two_expert_pack requires precomputed layout offsets")
        return super().__call__(
            tokens,
            layout_or_action_starts,
            action_ends,
            llm_offsets,
            action_offsets,
        )

    def _pytorch_fallback(
        self, tokens, action_starts, action_ends, llm_offsets, action_offsets
    ):
        _warn_slow_two_expert_fallback("two_expert_pack")
        if tokens.dim() != 3:
            raise RuntimeError("two_expert_pack expects input shape [B, S, D]")
        bsz, seq_length, hidden_dim = tokens.shape
        action_starts = torch.as_tensor(
            action_starts, dtype=torch.int64, device=tokens.device
        ).reshape(-1)
        action_ends = torch.as_tensor(
            action_ends, dtype=torch.int64, device=tokens.device
        ).reshape(-1)
        if action_starts.numel() != bsz or action_ends.numel() != bsz:
            raise RuntimeError("two_expert_pack expects [B] action ranges")
        if llm_offsets.numel() != bsz + 1 or action_offsets.numel() != bsz + 1:
            raise RuntimeError("two_expert_pack expects [B+1] layout offsets")

        llm_tokens = []
        action_tokens = []
        for batch in range(bsz):
            start = int(action_starts[batch].item())
            end = int(action_ends[batch].item())
            if start < 0 or end < start or end > seq_length:
                raise RuntimeError(
                    f"invalid action range [{start}, {end}) for seq_length={seq_length}"
                )
            llm_tokens.append(
                torch.cat((tokens[batch, :start], tokens[batch, end:]), dim=0)
            )
            action_tokens.append(tokens[batch, start:end])

        return torch.cat(
            (
                torch.cat(llm_tokens, dim=0).reshape(-1, hidden_dim),
                torch.cat(action_tokens, dim=0).reshape(-1, hidden_dim),
            ),
            dim=0,
        )


class TwoExpertUnpackOp(OpsProxy):
    """Restore fixed two-expert packed tokens back to [B, S, D]."""

    @property
    def _external_accel_name(self):
        return "two_expert_unpack"

    def __call__(
        self,
        packed_tokens,
        batch_size: int,
        seq_length: int,
        layout_or_action_starts,
        action_ends=None,
        llm_offsets=None,
        action_offsets=None,
    ):
        if isinstance(layout_or_action_starts, TwoExpertLayout):
            layout = layout_or_action_starts
            if layout.token_perm is not None:
                if packed_tokens.dim() != 2:
                    raise RuntimeError("two_expert_unpack expects input shape [B*S, D]")
                if layout.token_perm_inv is None:
                    raise RuntimeError("packed TwoExpertLayout is missing inverse permutation")
                perm = layout.token_perm.to(device=packed_tokens.device)
                perm_inv = layout.token_perm_inv.to(device=packed_tokens.device)
                total = int(batch_size) * int(seq_length)
                if perm.numel() != total or packed_tokens.shape[0] != total:
                    raise RuntimeError("packed TwoExpertLayout does not cover all tokens")
                out = _PermGatherUnpack.apply(packed_tokens, perm, perm_inv)
                return out.view(int(batch_size), int(seq_length), packed_tokens.shape[-1])
            return super().__call__(
                packed_tokens,
                batch_size,
                seq_length,
                layout.action_starts,
                layout.action_ends,
                layout.llm_offsets,
                layout.action_offsets,
            )
        if llm_offsets is None or action_offsets is None:
            raise RuntimeError("two_expert_unpack requires precomputed layout offsets")
        return super().__call__(
            packed_tokens,
            batch_size,
            seq_length,
            layout_or_action_starts,
            action_ends,
            llm_offsets,
            action_offsets,
        )

    def _pytorch_fallback(
        self,
        packed_tokens,
        batch_size: int,
        seq_length: int,
        action_starts,
        action_ends,
        llm_offsets,
        action_offsets,
    ):
        _warn_slow_two_expert_fallback("two_expert_unpack")
        if packed_tokens.dim() != 2:
            raise RuntimeError("two_expert_unpack expects input shape [B*S, D]")
        batch_size = int(batch_size)
        seq_length = int(seq_length)
        if packed_tokens.size(0) != batch_size * seq_length:
            raise RuntimeError(
                f"packed rows mismatch: got {packed_tokens.size(0)}, expected {batch_size * seq_length}"
            )

        hidden_dim = packed_tokens.size(1)
        action_starts = torch.as_tensor(
            action_starts, dtype=torch.int64, device=packed_tokens.device
        ).reshape(-1)
        action_ends = torch.as_tensor(
            action_ends, dtype=torch.int64, device=packed_tokens.device
        ).reshape(-1)
        if action_starts.numel() != batch_size or action_ends.numel() != batch_size:
            raise RuntimeError("two_expert_unpack expects [B] action ranges")
        if llm_offsets.numel() != batch_size + 1 or action_offsets.numel() != (
            batch_size + 1
        ):
            raise RuntimeError("two_expert_unpack expects [B+1] layout offsets")
        llm_offsets = llm_offsets.to(device=packed_tokens.device, dtype=torch.int64)
        action_offsets = action_offsets.to(
            device=packed_tokens.device, dtype=torch.int64
        )
        total_llm = int(llm_offsets[-1].item())

        output = torch.empty(
            (batch_size, seq_length, hidden_dim),
            dtype=packed_tokens.dtype,
            device=packed_tokens.device,
        )
        for batch in range(batch_size):
            start = int(action_starts[batch].item())
            end = int(action_ends[batch].item())
            if start < 0 or end < start or end > seq_length:
                raise RuntimeError(
                    f"invalid action range [{start}, {end}) for seq_length={seq_length}"
                )
            llm_start = int(llm_offsets[batch].item())
            llm_count = int(llm_offsets[batch + 1].item()) - llm_start
            action_start = total_llm + int(action_offsets[batch].item())
            llm_tokens = packed_tokens[llm_start : llm_start + llm_count]
            action_tokens = packed_tokens[action_start : action_start + (end - start)]
            if start > 0:
                output[batch, :start, :] = llm_tokens[:start]
            output[batch, start:end, :] = action_tokens
            if end < seq_length:
                output[batch, end:, :] = llm_tokens[start:]
        return output


two_expert_pack = TwoExpertPackOp()
two_expert_unpack = TwoExpertUnpackOp()
