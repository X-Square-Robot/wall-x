"""GatedDeltaNet post-conv qkv pack / preprocess with backend dispatch.

Model code calls :func:`pack_gated_delta_qkv` / :func:`preprocess_gated_delta_inputs`
and stays oblivious to which backend runs. Backend selection is capability-based
and lives entirely here: the fused external_accel kernels are used when the loaded
build provides them (and inputs are CUDA), otherwise the pure-PyTorch
``*_pytorch`` fallbacks in this module run (also what the open-source export
ships and what the bitexact test exercises).

Note on qk L2 norm: the fused *preprocess* kernel applies the query/key L2 norm
itself and reports ``use_qk_l2norm_in_kernel=False`` so the downstream delta-rule
kernel skips it; the PyTorch path defers the L2 norm to the delta-rule kernel and
reports ``True``. So for equal downstream results,
``l2norm(pytorch_query) == kernel_query`` (and likewise for key).
"""

import os

import torch
import torch.nn.functional as F

try:
    from external_accel import gated_delta_post_conv_pack as _external_accel_pack
except Exception:
    _external_accel_pack = None

try:
    from external_accel import (
        gated_delta_post_conv_pack_preprocess as _external_accel_preprocess,
    )
except Exception:
    _external_accel_preprocess = None


def gated_delta_pack_available(reference_tensor: torch.Tensor) -> bool:
    """True iff the fused pack kernel can run for tensors like *reference_tensor*."""
    return _external_accel_pack is not None and reference_tensor.is_cuda


def gated_delta_preprocess_available(reference_tensor: torch.Tensor) -> bool:
    """True iff the fused preprocess kernel can run for *reference_tensor*."""
    return _external_accel_preprocess is not None and reference_tensor.is_cuda


def pack_gated_delta_qkv_pytorch(
    module,
    mixed_qkv: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch split of post-conv mixed_qkv into per-head q/k/v."""
    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv,
        [
            module.key_dim,
            module.key_dim,
            module.value_dim,
        ],
        dim=-1,
    )

    query = query.reshape(batch_size, seq_len, -1, module.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, module.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, module.head_v_dim)
    if module.num_v_heads // module.num_k_heads > 1:
        query = query.repeat_interleave(module.num_v_heads // module.num_k_heads, dim=2)
        key = key.repeat_interleave(module.num_v_heads // module.num_k_heads, dim=2)
    return query, key, value


def preprocess_gated_delta_inputs_pytorch(
    module,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Pure-PyTorch pack + beta/g gates; defers qk L2 norm (returns ``True``)."""
    query, key, value = pack_gated_delta_qkv_pytorch(
        module, mixed_qkv, batch_size, seq_len
    )
    beta = b.sigmoid()
    # If the model is loaded in fp16, without the .float() here, A might be -inf
    g = -module.A_log.float().exp() * F.softplus(a.float() + module.dt_bias)
    return query, key, value, beta, g, True


def pack_gated_delta_qkv(
    module,
    mixed_qkv: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split post-conv mixed_qkv into per-head q/k/v (fused kernel or PyTorch)."""
    if _external_accel_pack is not None and mixed_qkv.is_cuda:
        return _external_accel_pack(
            mixed_qkv,
            module.num_k_heads,
            module.num_v_heads,
            module.head_k_dim,
            module.head_v_dim,
        )
    return pack_gated_delta_qkv_pytorch(module, mixed_qkv, batch_size, seq_len)


def preprocess_gated_delta_inputs(
    module,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Pack q/k/v and compute beta/g gates (fused kernel or PyTorch).

    Returns ``(query, key, value, beta, g, use_qk_l2norm_in_kernel)``: the fused
    kernel already applies the qk L2 norm, so downstream must skip it
    (``use_qk_l2norm_in_kernel=False``); the PyTorch path leaves it to the
    delta-rule kernel (``True``).
    """
    if _external_accel_preprocess is not None and mixed_qkv.is_cuda:
        query, key, value, beta, g = _external_accel_preprocess(
            mixed_qkv,
            b,
            a,
            module.A_log.float().contiguous(),
            module.dt_bias.float().contiguous(),
            module.num_k_heads,
            module.num_v_heads,
            module.head_k_dim,
            module.head_v_dim,
        )
        return query, key, value, beta, g, False

    return preprocess_gated_delta_inputs_pytorch(
        module, mixed_qkv, b, a, batch_size, seq_len
    )


def _resolve_chunk_gated_delta_rule():
    """Resolve the GDN chunk kernel: FlashQLA first, fla's Triton fallback.

    FlashQLA (QwenLM, TileLang) is call-compatible with
    ``fla.ops.gated_delta_rule.chunk_gated_delta_rule`` and measures ~1.25x
    layer fwd+bwd on H100 (16x128 k / 32x128 v heads) with far fewer
    launches. Only the chunked-prefill kernel is covered -- the decode path
    (fused_recurrent) stays on fla. First use JIT-compiles ~5 TileLang
    kernels (~90 s; the cache keys on the cu_seqlens segment count, not on
    sequence length, so fixed-bs batch-mode training compiles once).

    Resolution lives entirely HERE, including the fla fallback import:
    consumers import the resolved ``chunk_gated_delta_rule`` from this
    module and never touch either backend directly (the fla import block is
    duplicated across three modeling files, and patching only one of them
    silently benchmarked the wrong kernel once already). Opt out with
    WALLX_DISABLE_FLASH_QLA=1. The fallback is deliberately loud. Returns
    None when no backend is installed (GDN layers are unusable then anyway;
    the decode-path symbols come from fla alone).
    """
    try:
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule as fla_chunk_fn,
        )
    except Exception:  # noqa: BLE001 -- fla optional
        return None
    if os.environ.get("WALLX_DISABLE_FLASH_QLA", "0") == "1":
        return fla_chunk_fn
    try:
        from flash_qla import chunk_gated_delta_rule as qla_chunk_fn

        return qla_chunk_fn
    except Exception as exc:  # noqa: BLE001 -- loud: a silent revert to the
        # fla kernel would invalidate benchmarks unnoticed.
        print(
            "[WallX] WARNING: flash_qla unavailable, GDN chunk kernel falls "
            f"back to fla: {type(exc).__name__}: {exc}"
        )
        return fla_chunk_fn


chunk_gated_delta_rule = _resolve_chunk_gated_delta_rule()
