"""SlimFA3: minimal-overhead FA3 forward/backward for the DINO fixed-shape case.

Bypasses flash_attn_interface's python Function + torch.library custom_op
wrapper layers (~165us host per fwd+bwd pair measured; see
dev/runs/WRAPPER_SLIMMING_GUIDE.md) and calls the raw registered ops
directly. Scope guards (enforced by the caller / this Function):

  * non-causal, dropout == 0, no varlen/paged, fixed softmax scale
  * q/k/v (B, S, H, D) bf16/fp16 with last-dim stride 1 (the packed-qkv
    unbind views from DinoFusedRopeAttention qualify -- verified bitwise
    against the full wrapper on the real strided views)

Raw-op signatures are pinned to FA3 3.0.0 (fwd 34 args / bwd 22 args) and
asserted at import: a silent signature drift after an FA3 upgrade would
otherwise corrupt argument binding. torch.compile CANNOT trace this (no
fake-tensor support by design) -- keep WALLX_SLIM_FA3=0 as the escape hatch.
"""

import os

import torch

SLIM_FA3_AVAILABLE = False
_op_fwd = None
_op_bwd = None

if os.environ.get("WALLX_SLIM_FA3", "1") != "0":
    try:
        import flash_attn_3._C  # noqa: F401 -- registers torch.ops.flash_attn_3.*

        # (the package __init__ does NOT import _C; and _C needs torch loaded
        # first for libc10. dir() on the lazy op namespace lists nothing --
        # only attribute access resolves ops.)

        _op_fwd = torch.ops.flash_attn_3.fwd
        _op_bwd = torch.ops.flash_attn_3.bwd
        _n_fwd = len(_op_fwd.default._schema.arguments)
        _n_bwd = len(_op_bwd.default._schema.arguments)
        if (_n_fwd, _n_bwd) != (34, 22):
            print(
                "[SlimFA3] raw op arity mismatch (fwd=%d bwd=%d, pinned 34/22 "
                "for FA3 3.0.0) -- falling back to the full wrapper. Re-pin "
                "the argument lists and RERUN fa_grad_check before trusting a "
                "new FA3 version." % (_n_fwd, _n_bwd)
            )
        else:
            SLIM_FA3_AVAILABLE = True
    except Exception as _exc:  # noqa: BLE001
        print(f"[SlimFA3] unavailable: {type(_exc).__name__}: {_exc}")


class SlimFA3(torch.autograd.Function):
    """See module docstring. Returns ``out`` only (no lse exposure)."""

    @staticmethod
    def forward(ctx, q, k, v, scale):
        out, lse, _, _ = _op_fwd(
            q,
            k,
            v,
            None,
            None,
            None,
            None,  # k_new, v_new, qv, out_
            None,
            None,
            None,
            None,
            None,
            None,
            None,  # cu_seqlens*, seqused*, max_seqlen*
            None,
            None,
            None,
            None,
            None,
            None,  # page_table..seqlens_rotary
            None,
            None,
            None,  # q/k/v_descale
            scale,
            False,
            -1,
            -1,
            0,
            0.0,
            True,
            None,
            1,
            None,
            0,
        )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        _op_bwd(
            dout,
            q,
            k,
            v,
            out,
            lse,
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            ctx.scale,
            False,
            -1,
            -1,
            0.0,
            False,
            0,
        )
        return dq, dk, dv, None


class SlimFA3Varlen(torch.autograd.Function):
    """Varlen twin of ``SlimFA3`` for the packed multi-image DINO forward.

    q/k/v are ``[total_tokens, H, D]`` with per-image attention boundaries in
    ``cu_seqlens`` (int32, [n_images+1]); self-attention only, so one
    cu_seqlens serves both q and k. Same scope guards and raw-op arity
    pinning as the dense Function above.
    """

    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens, max_seqlen, scale):
        out, lse, _, _ = _op_fwd(
            q,
            k,
            v,
            None,
            None,
            None,
            None,  # k_new, v_new, qv, out_
            cu_seqlens,
            cu_seqlens,
            None,  # cu_seqlens_q/k, cu_seqlens_k_new
            None,
            None,  # seqused_q/k
            max_seqlen,
            max_seqlen,  # max_seqlen_q/k
            None,
            None,
            None,
            None,
            None,
            None,  # page_table..seqlens_rotary
            None,
            None,
            None,  # q/k/v_descale
            scale,
            False,
            -1,
            -1,
            0,
            0.0,
            True,
            None,
            1,
            None,
            0,
        )
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens)
        ctx.scale = scale
        ctx.max_seqlen = max_seqlen
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse, cu_seqlens = ctx.saved_tensors
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        _op_bwd(
            dout,
            q,
            k,
            v,
            out,
            lse,
            dq,
            dk,
            dv,
            cu_seqlens,
            cu_seqlens,
            None,
            None,  # cu_seqlens_q/k, seqused_q/k
            ctx.max_seqlen,
            ctx.max_seqlen,
            ctx.scale,
            False,
            -1,
            -1,
            0.0,
            False,
            0,
        )
        return dq, dk, dv, None, None, None
