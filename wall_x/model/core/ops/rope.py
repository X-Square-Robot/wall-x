"""Rotary position embedding operators."""

import logging
from typing import List

import torch
from wall_x.model.core.ops.base import OpsProxy

logger = logging.getLogger(__name__)


def _rotate_half(x):
    """Rotate half: split last dim in two halves, negate-swap, concat."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_interleave(x):
    """Interleaved rotation: pairs of (even, odd) elements."""
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


class RoPEOp(OpsProxy):
    """Standard rotary position embedding.

    Signature: rope(q, k, cos, sin, interleave=False) -> (q_embed, k_embed)
    """

    @property
    def _external_accel_name(self):
        return "rope"

    def _get_cuda_kernel(self):
        try:
            from wall_x.model.core.ops._cuda_wrappers import Rope

            return Rope()
        except ImportError:
            return None
        except Exception as e:
            logger.warning("RoPEOp: CUDA kernel load failed: %s", e)
            return None

    def _pytorch_fallback(self, q, k, cos, sin, interleave=False, **kwargs):
        cos = cos.float()
        sin = sin.float()
        rotary_dim = cos.size(-1) * 2
        head_dim = q.size(-1)
        if rotary_dim > head_dim:
            raise ValueError(
                f"rotary_dim ({rotary_dim}) > head_dim ({head_dim}): "
                f"cos last dim ({cos.size(-1)}) is too large for q"
            )
        partial = rotary_dim < head_dim
        if partial:
            q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
            k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        else:
            q_rot, k_rot = q, k
        cos = cos.unsqueeze(-2)  # (..., 1, half_dim)
        sin = sin.unsqueeze(-2)
        if interleave:
            cos = cos.repeat_interleave(2, dim=-1)
            sin = sin.repeat_interleave(2, dim=-1)
            q_embed = q_rot.float() * cos + _rotate_interleave(q_rot.float()) * sin
            k_embed = k_rot.float() * cos + _rotate_interleave(k_rot.float()) * sin
        else:
            cos = torch.cat((cos, cos), dim=-1)
            sin = torch.cat((sin, sin), dim=-1)
            q_embed = q_rot.float() * cos + _rotate_half(q_rot.float()) * sin
            k_embed = k_rot.float() * cos + _rotate_half(k_rot.float()) * sin
        if partial:
            q_embed = torch.cat([q_embed.to(q.dtype), q_pass], dim=-1)
            k_embed = torch.cat([k_embed.to(k.dtype), k_pass], dim=-1)
        else:
            q_embed = q_embed.to(q.dtype)
            k_embed = k_embed.to(k.dtype)
        return q_embed, k_embed

    def full(self, q, k, cos, sin, interleave=False, rotary_dim=None):
        """Partial-rotary RoPE that also emits the pass-through part.

        Fuses the rotary-prefix rotation and the pass-through copy into a single
        external_accel kernel, avoiding the Python-side ``torch.cat([rotated, pass])``.
        q/k are the FULL head tensors ([..., head_dim]); cos/sin are the compact
        half-dim (rotary_dim // 2). Falls back to slice + rope + cat when the
        external_accel ``.full`` kernel is unavailable.
        See docs qwen35_mfu_report (RoPE full fused kernel).
        """
        if self._resolved_fn is None:
            self._resolve()
        # external_accel rope_full fuses the rotary-prefix rotation and pass-through
        # copy into one kernel. Verified numerically equal to a pure-PyTorch
        # partial-rotary reference (bf16 tolerance). Used whenever the loaded
        # backend provides .full. See docs qwen35_mfu_report (RoPE full).
        use_external_accel_full = hasattr(self._resolved_fn, "full")
        if use_external_accel_full:
            try:
                return self._resolved_fn.full(
                    q, k, cos, sin, interleave=interleave, rotary_dim=rotary_dim
                )
            except Exception as exc:
                logger.warning(
                    "RoPEOp.full: external_accel path failed (%s); using PyTorch fallback",
                    exc,
                )
        return self._pytorch_full_fallback(
            q, k, cos, sin, interleave=interleave, rotary_dim=rotary_dim
        )

    def _pytorch_full_fallback(self, q, k, cos, sin, interleave=False, rotary_dim=None):
        if rotary_dim is None:
            rotary_dim = cos.size(-1) * 2
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_embed, k_embed = self(q_rot, k_rot, cos, sin, interleave=interleave)
        q_embed = torch.cat([q_embed, q_pass], dim=-1)
        k_embed = torch.cat([k_embed, k_pass], dim=-1)
        return q_embed, k_embed

    def pack(self, *args, **kwargs):
        """Apply RoPE to packed qkv through the selected backend.

        Callers rely on three contracts a bare ``return args`` breaks: some use
        the return value directly (qwen3_5 vision), some truth-test it
        (qwen2_5 vision), and the infer paths ignore the return and rely on the
        in-place q/k writes. The fallback must therefore apply RoPE in place
        and return the qkv tensor.
        """
        if self._resolved_fn is None:
            self._resolve()
        if hasattr(self._resolved_fn, "pack"):
            return self._resolved_fn.pack(*args, **kwargs)
        return self._pytorch_pack_fallback(*args, **kwargs)

    def _pytorch_pack_fallback(
        self, qkv, q_num_heads, kv_num_heads, cos, sin, interleave=False, **kwargs
    ):
        if qkv.dim() != 2:
            raise ValueError(f"Expected packed qkv to be 2D, got {qkv.dim()}D")
        if qkv.stride(-1) != 1:
            raise ValueError("Expected packed qkv last dimension to have stride 1")
        if cos.dtype != torch.float32 or sin.dtype != torch.float32:
            raise ValueError("Expected rope cos/sin to be float32")
        if not cos.is_contiguous() or not sin.is_contiguous():
            raise ValueError("Expected rope cos/sin to be contiguous")

        head_dim = cos.size(-1) * 2
        q_dim = q_num_heads * head_dim
        kv_dim = kv_num_heads * head_dim
        if qkv.size(-1) != q_dim + 2 * kv_dim:
            raise ValueError(
                f"Expected packed qkv last dim {q_dim + 2 * kv_dim}, got {qkv.size(-1)}"
            )

        q = qkv[..., :q_dim].view(*qkv.shape[:-1], q_num_heads, head_dim)
        k = qkv[..., q_dim : q_dim + kv_dim].view(
            *qkv.shape[:-1], kv_num_heads, head_dim
        )

        if qkv.dim() == 2 and cos.dim() == 3 and cos.shape[0] == 1:
            cos = cos.squeeze(0)
            sin = sin.squeeze(0)

        q_embed, k_embed = self._pytorch_fallback(q, k, cos, sin, interleave=interleave)
        q.copy_(q_embed)
        k.copy_(k_embed)
        return qkv


class MRoPEOp(OpsProxy):
    """Multi-head rotary position embedding (used by Qwen2.5-VL models).

    Signature: m_rope(query_states, key_states, cos, sin, mrope_section, interleaved=False) -> (q_embed, k_embed)

    cos/sin shape: (3, B, S, D//2). mrope_section like [16, 24, 24] specifies
    how many head dims each of T/H/W occupies. After split by mrope_section * 2,
    uses i % 3 to select from corresponding temporal/height/width rows.
    """

    @property
    def _external_accel_name(self):
        return "m_rope"

    def _get_cuda_kernel(self):
        try:
            from wall_x.model.core.ops._cuda_wrappers import MRope

            return MRope()
        except ImportError:
            return None
        except Exception as e:
            logger.warning("MRoPEOp: CUDA kernel load failed: %s", e)
            return None

    def _pytorch_fallback(
        self,
        query_states,
        key_states,
        cos,
        sin,
        mrope_section: List[int],
        interleaved=False,
        **kwargs,
    ):
        cos = cos.float()
        sin = sin.float()
        # Double cos/sin along last dim (matching external_accel kernel's internal behavior)
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
        # List concatenation: [16,24,24] -> [16,24,24,16,24,24] (NOT element-wise multiply)
        mrope_section_doubled = mrope_section + mrope_section
        cos_split = torch.cat(
            [m[i % 3] for i, m in enumerate(cos.split(mrope_section_doubled, dim=-1))],
            dim=-1,
        ).unsqueeze(
            2
        )  # (B, S, 1, D)
        sin_split = torch.cat(
            [m[i % 3] for i, m in enumerate(sin.split(mrope_section_doubled, dim=-1))],
            dim=-1,
        ).unsqueeze(2)
        q_embed = (query_states.float() * cos_split) + (
            _rotate_half(query_states.float()) * sin_split
        )
        k_embed = (key_states.float() * cos_split) + (
            _rotate_half(key_states.float()) * sin_split
        )
        return q_embed.to(query_states.dtype), k_embed.to(key_states.dtype)

    def pack(self, *args, **kwargs):
        """Apply MRoPE to packed qkv through the selected backend.

        See RoPEOp.pack: the infer kvcache path ignores the return value and
        relies on the in-place q/k writes, so the fallback must apply MRoPE in
        place rather than return the raw args.
        """
        if self._resolved_fn is None:
            self._resolve()
        if hasattr(self._resolved_fn, "pack"):
            return self._resolved_fn.pack(*args, **kwargs)
        return self._pytorch_pack_fallback(*args, **kwargs)

    def _pytorch_pack_fallback(
        self, qkv, q_num_heads, kv_num_heads, cos, sin, mrope_section, **kwargs
    ):
        if qkv.dim() != 3:
            raise ValueError(f"Expected packed qkv to be 3D, got {qkv.dim()}D")
        if qkv.stride(-1) != 1:
            raise ValueError("Expected packed qkv last dimension to have stride 1")

        head_dim = sum(mrope_section) * 2
        q_dim = q_num_heads * head_dim
        kv_dim = kv_num_heads * head_dim
        if qkv.size(-1) != q_dim + 2 * kv_dim:
            raise ValueError(
                f"Expected packed qkv last dim {q_dim + 2 * kv_dim}, got {qkv.size(-1)}"
            )

        q = qkv[..., :q_dim].view(*qkv.shape[:-1], q_num_heads, head_dim)
        k = qkv[..., q_dim : q_dim + kv_dim].view(
            *qkv.shape[:-1], kv_num_heads, head_dim
        )

        q_embed, k_embed = self._pytorch_fallback(q, k, cos, sin, mrope_section)
        q.copy_(q_embed)
        k.copy_(k_embed)
        return qkv


class RotPosEmbOp(OpsProxy):
    """Rotary position embedding computation for ViT (used by Qwen2.5-VL vision encoder).

    Signature: rot_pos_emb(inv_freq, grid_thw, spatial_merge_size) -> rotary_pos_emb
    """

    @property
    def _external_accel_name(self):
        return "rot_pos_emb"

    def _get_cuda_kernel(self):
        try:
            from wall_x.model.core.ops._cuda_wrappers import RotPos

            return RotPos()
        except ImportError:
            return None
        except Exception as e:
            logger.warning("RotPosEmbOp: CUDA kernel load failed: %s", e)
            return None

    def _pytorch_fallback(
        self, inv_freq, grid_thw, spatial_merge_size, total_output_tokens=None
    ):
        # total_output_tokens lets the external_accel backend skip a GPU->CPU sync;
        # the PyTorch fallback derives sizes from grid_thw directly.
        del total_output_tokens
        if inv_freq.dtype != torch.float32:
            inv_freq = inv_freq.to(torch.float32)
        pos_ids = []
        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            if h % spatial_merge_size != 0 or w % spatial_merge_size != 0:
                raise ValueError(
                    f"grid h={h}, w={w} must be divisible by spatial_merge_size={spatial_merge_size}"
                )
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = (
                hpos_ids.reshape(
                    h // spatial_merge_size,
                    spatial_merge_size,
                    w // spatial_merge_size,
                    spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = (
                wpos_ids.reshape(
                    h // spatial_merge_size,
                    spatial_merge_size,
                    w // spatial_merge_size,
                    spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0).to(inv_freq.device)
        max_grid_size = grid_thw[:, 1:].max()
        seq = torch.arange(max_grid_size, device=inv_freq.device, dtype=inv_freq.dtype)
        rotary_pos_emb_full = torch.outer(seq, inv_freq)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb.to(torch.float)

    def pack(self, *args, **kwargs):
        """Delegate to resolved backend's pack method if available."""
        if self._resolved_fn is None:
            self._resolve()
        if hasattr(self._resolved_fn, "pack"):
            return self._resolved_fn.pack(*args, **kwargs)
        return args


rope = RoPEOp()
m_rope = MRoPEOp()
rot_pos_emb = RotPosEmbOp()
