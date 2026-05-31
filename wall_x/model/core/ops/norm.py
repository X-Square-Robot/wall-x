"""RMS normalization operator."""

import logging

import torch
from wall_x.model.core.ops.base import OpsProxy

logger = logging.getLogger(__name__)


class RMSNormOp(OpsProxy):
    """RMS normalization: rmsnorm(x, weight, eps).

    Accepts ``rmsnorm(x, weight, eps)``.
    """

    @property
    def _external_accel_name(self):
        return "rmsnorm"

    def _get_cuda_kernel(self):
        try:
            from wall_x.model.core.ops._cuda_wrappers import Rmsnorm

            return Rmsnorm()
        except ImportError:
            return None
        except Exception as e:
            logger.warning("RMSNormOp: CUDA kernel load failed: %s", e)
            return None

    def _pytorch_fallback(self, hidden_states, weight, eps=1e-6):
        """Pure PyTorch RMSNorm. eps default matches external_accel (1e-6, not PyTorch's 1e-5)."""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + eps)
        return (weight * hidden_states).to(input_dtype)


rmsnorm = RMSNormOp()
