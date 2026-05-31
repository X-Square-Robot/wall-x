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

    def pack(self, *args, **kwargs):
        """Delegate to resolved backend's pack method if available."""
        if self._resolved_fn is None:
            self._resolve()
        if hasattr(self._resolved_fn, "pack"):
            return self._resolved_fn.pack(*args, **kwargs)
        return args


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
        """Delegate to resolved backend's pack method if available."""
        if self._resolved_fn is None:
            self._resolve()
        if hasattr(self._resolved_fn, "pack"):
            return self._resolved_fn.pack(*args, **kwargs)
        return args


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

    def _pytorch_fallback(self, inv_freq, grid_thw, spatial_merge_size):
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
