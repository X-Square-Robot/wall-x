"""MoE routing operators (permute/unpermute).

These operators reorder tokens by expert assignment for efficient
Mixture-of-Experts processing.
"""

import logging

import torch
from wall_x.model.core.ops.base import OpsProxy

logger = logging.getLogger(__name__)


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
