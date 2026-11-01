"""Normalization operators (RMSNorm + LayerNorm)."""

import inspect
import logging

import torch
import torch.nn.functional as F
from wall_x.model.core.ops.base import OpsProxy

logger = logging.getLogger(__name__)


class RMSNormOp(OpsProxy):
    """RMS normalization: rmsnorm(x, weight, eps).

    Accepts ``rmsnorm(x, weight, eps)``.
    """

    @property
    def _external_accel_name(self):
        return "rmsnorm"

    def _probe_external_accel(self):
        fn = super()._probe_external_accel()
        if fn is None:
            return None

        try:
            supports_zero_centered = "zero_centered" in inspect.signature(fn).parameters
        except (TypeError, ValueError):
            supports_zero_centered = True

        if supports_zero_centered:
            return fn

        try:
            _supports_inference = "inference" in inspect.signature(fn).parameters
        except (TypeError, ValueError):
            _supports_inference = True

        def legacy_rmsnorm(
            hidden_states, weight, eps=1e-6, inference=False, zero_centered=False
        ):
            if zero_centered:
                return self._pytorch_fallback(
                    hidden_states, weight, eps, zero_centered=True
                )
            # Some legacy external_accel rmsnorm kernels predate the `inference` kwarg
            # (added alongside zero_centered); only pass it when accepted.
            if _supports_inference:
                return fn(hidden_states, weight, eps, inference=inference)
            return fn(hidden_states, weight, eps)

        return legacy_rmsnorm

    def _get_cuda_kernel(self):
        try:
            from wall_x.model.core.ops._cuda_wrappers import Rmsnorm

            return Rmsnorm()
        except ImportError:
            return None
        except Exception as e:
            logger.warning("RMSNormOp: CUDA kernel load failed: %s", e)
            return None

    def _pytorch_fallback(self, hidden_states, weight, eps=1e-6, zero_centered=False):
        """Pure PyTorch RMSNorm. eps default matches external_accel (1e-6, not PyTorch's 1e-5)."""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + eps)
        if zero_centered:
            weight = weight + 1.0
        return (weight * hidden_states).to(input_dtype)


class AddRMSNormOp(OpsProxy):
    """Fused residual-add + RMSNorm.

    add_rmsnorm(residual, input, weight, eps, zero_centered=False)
        -> (sum_output, norm_output)
    where sum_output = residual + input (materialized, reused by the next
    residual add) and norm_output = RMSNorm(sum_output) * weight_effective.
    The external_accel backward reuses the forward's saved rstd (fast path for
    hidden_size <= 4096). See docs qwen35_mfu_report (AddRMSNorm fast backward).
    """

    @property
    def _external_accel_name(self):
        return "add_rmsnorm"

    def _pytorch_fallback(
        self, residual, input, weight, eps=1e-6, inference=False, zero_centered=False
    ):
        sum_output = residual + input
        input_dtype = sum_output.dtype
        x = sum_output.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        if zero_centered:
            weight = weight + 1.0
        norm_output = (weight * x).to(input_dtype)
        return sum_output, norm_output


class GatedRMSNormOp(OpsProxy):
    """Gated RMSNorm used at the end of GatedDeltaNet:

        gated_rmsnorm(input, gate, weight, eps)
            = (RMSNorm(input) * weight) * silu(gate)

    Prefers the external_accel fused kernel when the loaded build provides it
    forces the PyTorch fallback below.
    See docs qwen35_mfu_report (GatedDeltaNet gated RMSNorm via external_accel).
    """

    @property
    def _external_accel_name(self):
        return "gated_rmsnorm"

    def _pytorch_fallback(self, input, gate, weight, eps=1e-6, inference=False):
        input_dtype = input.dtype
        hidden_states = input.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + eps)
        hidden_states = weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * torch.nn.functional.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


rmsnorm = RMSNormOp()
add_rmsnorm = AddRMSNormOp()
gated_rmsnorm = GatedRMSNormOp()


class LayerNormOp(OpsProxy):
    """Layer normalization: layernorm(input, weight, bias, eps).

    external_accel signature (external_accel/kernel/layernorm.py):
      layernorm(input_tensor, weight, bias, epsilon) -> Tensor
      bias is required (external_accel kernel always applies affine).

    Currently unused. Kept as a reference for the proxy pattern; safe to
    remove if no other consumer appears.
    """

    @property
    def _external_accel_name(self):
        return "layernorm"

    def _pytorch_fallback(self, input_tensor, weight, bias, eps):
        # F.layer_norm normalizes over weight.shape (the trailing dims).
        return F.layer_norm(input_tensor, weight.shape, weight, bias, eps)


layernorm = LayerNormOp()


class ScaledAddLayerNormOp(OpsProxy):
    """Fused layerscale-mul + residual-add + LayerNorm; returns ``(sum, y)``.

        sum = residual + branch * lambda      # lambda per-channel [H]
        y   = LayerNorm(sum) * weight + bias

    One kernel each way replaces addcmul + native_layer_norm and their
    backward chain (written for the DINOv3 tower where the pattern occurs
    per sub-block; same fusion structure as the decoder's add_rmsnorm).
    The external_accel backend carries its own autograd; parameter grads use
    fp32 atomics (row-order nondeterministic, same class as aten's norm
    backwards). Fallback is the exact unfused composition.
    """

    @property
    def _external_accel_name(self):
        return "scaled_add_layernorm"

    def _pytorch_fallback(self, residual, branch, lambda_, weight, bias, eps):
        total = torch.addcmul(residual, branch, lambda_)
        normed = F.layer_norm(total, weight.shape, weight, bias, eps)
        return total, normed


scaled_add_layernorm = ScaledAddLayerNormOp()
