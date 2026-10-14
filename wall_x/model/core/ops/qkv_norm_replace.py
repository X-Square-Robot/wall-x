"""Fused Qwen3.5 VLA qkv norm-replace operator (kernel + PyTorch fallback).

One op splits a fused text/action qkv tensor into q/gate/k/v, applies the
zero-centered q/k RMSNorm (raw norm weights; the `+1` is added here), and
replaces the action-token rows of each batch with the action expert's values,
emitting a FlashMask-friendly layout.

``qkv_norm_replace`` dispatches by capability: the fused external_accel kernel when it
is available and the inputs are on CUDA, otherwise the self-contained PyTorch
fallback in this module. The fallback is the code path the open-source export
runs (it ships without external_accel) and is what the bitexact test exercises.

Input layout (matches the kernel, per external_accel ``_check_inputs``):
``text_qkv`` / ``action_qkv`` are ``[B, S, C]`` with
``C = (num_heads*2 + num_key_value_heads*2) * head_dim`` laid out as
``[q|gate head-interleaved (H*2D)] [key (Hkv*D)] [value (Hkv*D)]``; within each
head's ``2*head_dim`` slice the first ``head_dim`` is query, the next is gate.
Action tokens are contiguous rows ``[action_starts[b]:action_ends[b])`` per
batch; ``action_offsets`` is the ``[B+1]`` cumsum mapping each batch's rows into
the flattened action buffer.

Returns ``(query, key, value, gate)`` with shapes
``[B,S,H,D] / [B,S,Hkv,D] / [B,S,Hkv,D] / [B,S,H*D]``.
"""

import torch

from wall_x.model.core.ops.norm import rmsnorm as _rmsnorm

try:
    from external_accel import qwen35_vla_qkv_norm_replace_kernel as _kernel
except Exception:
    _kernel = None


def qkv_norm_replace_available(reference_tensor: torch.Tensor) -> bool:
    """True iff the fused kernel can run for tensors like *reference_tensor*."""
    return _kernel is not None and reference_tensor.is_cuda


def _zero_centered_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    # Same math as Qwen3_5RMSNorm: float32 upcast, (1 + weight) scale, cast back.
    return _rmsnorm._pytorch_fallback(x, weight.float(), eps, zero_centered=True)


def qkv_norm_replace_pytorch(
    text_qkv: torch.Tensor,
    action_qkv: torch.Tensor,
    action_starts: torch.Tensor,
    action_ends: torch.Tensor,
    action_offsets: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_action_weight: torch.Tensor,
    k_action_weight: torch.Tensor,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    epsilon: float,
):
    """Pure-tensor equivalent of the fused kernel (see module docstring)."""
    H, Hkv, D = int(num_heads), int(num_key_value_heads), int(head_dim)
    q_block = H * 2 * D
    k_block = Hkv * D

    def split_norm(qkv, qw, kw):
        bsz, seqlen, _ = qkv.shape
        qg = qkv[..., :q_block].reshape(bsz, seqlen, H, 2 * D)
        query = _zero_centered_rmsnorm(qg[..., :D].contiguous(), qw, epsilon)
        gate = qg[..., D:].reshape(bsz, seqlen, H * D)
        key = _zero_centered_rmsnorm(
            qkv[..., q_block : q_block + k_block]
            .reshape(bsz, seqlen, Hkv, D)
            .contiguous(),
            kw,
            epsilon,
        )
        value = qkv[..., q_block + k_block :].reshape(bsz, seqlen, Hkv, D).contiguous()
        return query, key, value, gate

    query, key, value, gate = split_norm(text_qkv, q_weight, k_weight)
    B = query.shape[0]

    a_query, a_key, a_value, a_gate = split_norm(
        action_qkv, q_action_weight, k_action_weight
    )
    total_action = action_qkv.shape[0] * action_qkv.shape[1]
    a_query = a_query.reshape(total_action, H, D)
    a_key = a_key.reshape(total_action, Hkv, D)
    a_value = a_value.reshape(total_action, Hkv, D)
    a_gate = a_gate.reshape(total_action, H * D)

    # Replace each batch's contiguous action rows with the action expert's,
    # indexing the flattened action buffer via action_offsets (matches kernel).
    query, key, value, gate = query.clone(), key.clone(), value.clone(), gate.clone()
    starts = action_starts.tolist()
    ends = action_ends.tolist()
    offsets = action_offsets.tolist()
    for b in range(B):
        s, e = int(starts[b]), int(ends[b])
        o0, o1 = int(offsets[b]), int(offsets[b + 1])
        if e > s:
            query[b, s:e] = a_query[o0:o1]
            key[b, s:e] = a_key[o0:o1]
            value[b, s:e] = a_value[o0:o1]
            gate[b, s:e] = a_gate[o0:o1]
    return query, key, value, gate


def qkv_norm_replace(*args, **kwargs):
    """Capability-dispatched fused qkv norm-replace.

    Uses the external_accel kernel when it is available and the fused qkv tensor is on
    CUDA; otherwise runs :func:`qkv_norm_replace_pytorch`.
    """
    text_qkv = args[0] if args else kwargs["text_qkv"]
    if _kernel is not None and text_qkv.is_cuda:
        return _kernel(*args, **kwargs)
    return qkv_norm_replace_pytorch(*args, **kwargs)
