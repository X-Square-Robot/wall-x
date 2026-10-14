"""
MEM-based (Multi-Scale Embodied Memory) Vision Encoder for Qwen3.5 VLA.

Wraps the original Qwen3.5 ViT to support multi-frame history encoding with:
- Fixed sinusoidal temporal position encoding (t=0 → zero for current frame)
- Spatiotemporal separated attention (spatial bidirectional + temporal causal)
- Current-frame-only token output (constant token count regardless of history)

Preserves the original temporal_patch_size=2 by default so that pretrained
Conv3d weights are loaded without any approximation.  Each history frame is
processed as a single image by the image_processor (which duplicates it to
fill the temporal patch), exactly like the standard Qwen3.5 pipeline.

Degrades to standard Qwen3.5 ViT when no history frames are provided.

temporal_param_mode controls temporal attention parameters:
  "shared"      — reuse spatial attn weights, no new parameters (default).
  "independent" — dedicated zero-init temporal_qkv/proj + learnable time embed.
"""

import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils.generic import (
    is_flash_attention_requested,
    merge_with_config_defaults,
)
from transformers.utils.output_capturing import capture_outputs
from transformers.utils import logging

from wall_x.model.core.ops import (
    pos_embed_interp as pos_embed_interp_op,
    rope as rope_op,
    rot_pos_emb as rot_pos_emb_op,
)
from wall_x.model.qact.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from wall_x.model.qact.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5VisionPatchEmbed,
    Qwen3_5VisionPatchMerger,
    Qwen3_5VisionRotaryEmbedding,
    Qwen3_5VisionAttention,
    Qwen3_5VisionMLP,
    Qwen3_5PreTrainedModel,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
)

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class Qwen3_5VLAVisionConfig(Qwen3_5VisionConfig):
    """Vision config extended for MEM-style multi-frame history encoding."""

    model_type = "qwen3_5_vla_vision"

    def __init__(
        self,
        temporal_attn_interval: int = 4,
        temporal_param_mode: str = "shared",
        temporal_max_frames: int = 16,
        temporal_max_dt: int = 128,
        temporal_history_interval: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.temporal_attn_interval = temporal_attn_interval
        if temporal_param_mode not in ("shared", "independent"):
            raise ValueError(
                f"temporal_param_mode must be 'shared' or 'independent', "
                f"got {temporal_param_mode!r}"
            )
        self.temporal_param_mode = temporal_param_mode
        self.temporal_max_frames = temporal_max_frames
        self.temporal_max_dt = temporal_max_dt
        self.temporal_history_interval = temporal_history_interval


# ---------------------------------------------------------------------------
# Sinusoidal temporal position encoding
# ---------------------------------------------------------------------------


def sinusoidal_time_embed(t_indices: torch.Tensor, dim: int) -> torch.Tensor:
    """Fixed sinusoidal temporal position encoding.

    Args:
        t_indices: (total_tokens,) integer time-step index per token.
                   0 = current frame, 1 = one step in the past, etc.
        dim:       embedding dimension (must be even).

    Returns:
        (total_tokens, dim) encoding.  t=0 maps to the zero vector so that
        single-frame input is identical to the original ViT.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, dtype=torch.float32, device=t_indices.device)
        / half
    )
    angles = t_indices.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)  # (N, dim)
    emb[t_indices == 0] = 0.0
    return emb


# ---------------------------------------------------------------------------
# Vision Block with optional temporal causal attention
# ---------------------------------------------------------------------------


def _rope_pack_available() -> bool:
    """True when the resolved rope backend exposes the packed-qkv fast path."""
    if rope_op._resolved_fn is None:
        rope_op._resolve()
    available = hasattr(rope_op._resolved_fn, "pack")
    if not available:
        logger.warning_once(
            "external_accel rope.pack unavailable — VLA vision attention falls back "
            "to the PyTorch RoPE path (slower, bit-identical)."
        )
    return available


class Qwen3_5VLAVisionAttention(Qwen3_5VisionAttention):
    """VLA vision attention with a PyTorch RoPE fallback.

    The VLA vision path always builds compact half-dim cos/sin (the external_accel
    ``rope.pack`` contract, same as the base class). When the packed kernel is
    available we delegate to the base class; otherwise RoPE is applied via the
    PyTorch vision helper, which is bit-identical to the kernel (see
    tests/test_vision_ops_bitexact.py) — unlike the base class, which would
    silently skip RoPE without a pack backend.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | None = None,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if (
            position_embeddings[0].shape[-1] == self.head_dim // 2
            and _rope_pack_available()
        ):
            return super().forward(
                hidden_states,
                cu_seqlens,
                max_seqlen=max_seqlen,
                rotary_pos_emb=rotary_pos_emb,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        del rotary_pos_emb
        seq_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states)
        query_states, key_states, value_states = qkv.view(
            seq_length, 3, self.num_heads, -1
        ).unbind(1)

        cos, sin = position_embeddings
        if cos.shape[-1] == self.head_dim:
            cos = cos[..., : self.head_dim // 2]
            sin = sin[..., : self.head_dim // 2]
        elif cos.shape[-1] != self.head_dim // 2:
            raise ValueError(
                f"Unexpected VLA vision RoPE dim: cos.shape[-1]={cos.shape[-1]}, "
                f"expected {self.head_dim // 2} or {self.head_dim}"
            )
        query_states, key_states = apply_rotary_pos_emb_vision(
            query_states, key_states, cos, sin
        )

        if is_flash_attention_requested(self.config):
            from flash_attn import flash_attn_varlen_func

            orig_dtype = query_states.dtype
            if orig_dtype == torch.float32:
                query_states = query_states.to(torch.bfloat16)
                key_states = key_states.to(torch.bfloat16)
                value_states = value_states.to(torch.bfloat16)

            if max_seqlen is None:
                max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
            attn_output = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=0.0 if not self.training else self.attention_dropout,
                softmax_scale=self.scaling,
                causal=False,
                deterministic=self.deterministic,
            )
            if orig_dtype == torch.float32:
                attn_output = attn_output.to(orig_dtype)
        else:
            query_states = query_states.transpose(0, 1).unsqueeze(0)
            key_states = key_states.transpose(0, 1).unsqueeze(0)
            value_states = value_states.transpose(0, 1).unsqueeze(0)
            attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, eager_attention_forward
            )
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2)
                for tensor in (query_states, key_states, value_states)
            ]
            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)
            attn_output = attn_output.squeeze(0).transpose(0, 1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)


class Qwen3_5VLAVisionBlock(GradientCheckpointingLayer):
    """VisionBlock that supports an optional second-pass temporal causal attention.

    Spatial attention (phase 1) runs every layer using per-frame cu_seqlens.
    Temporal attention (phase 2) runs on selected layers, reusing the *same*
    QKV projection weights, and applies causal masking so that each spatial
    position can only attend to itself at the current and past time-steps.
    """

    def __init__(
        self,
        config: Qwen3_5VLAVisionConfig,
        use_selective_recompute: bool = False,
    ) -> None:
        super().__init__()
        self.use_selective_recompute = use_selective_recompute
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3_5VLAVisionAttention(config=config)
        self.mlp = Qwen3_5VisionMLP(
            config=config, use_selective_recompute=use_selective_recompute
        )

        self.temporal_param_mode = getattr(config, "temporal_param_mode", "shared")
        if self.temporal_param_mode == "independent":
            dim = config.hidden_size
            self.temporal_qkv = nn.Linear(dim, 3 * dim, bias=False)
            self.temporal_proj = nn.Linear(dim, dim, bias=False)
            nn.init.zeros_(self.temporal_proj.weight)
            self.temporal_time_embed = nn.Embedding(
                getattr(config, "temporal_max_dt", 128), dim
            )
            nn.init.normal_(self.temporal_time_embed.weight, std=0.02)
            self.temporal_norm = nn.LayerNorm(dim, eps=1e-6)

    def _apply_norm1(self, hidden_states):
        if self.use_selective_recompute:
            return cp.checkpoint(self.norm1, hidden_states, use_reentrant=False)
        return self.norm1(hidden_states)

    def _apply_norm2(self, hidden_states):
        if self.use_selective_recompute:
            return cp.checkpoint(self.norm2, hidden_states, use_reentrant=False)
        return self.norm2(hidden_states)

    def _apply_temporal_norm(self, hidden_states):
        if self.use_selective_recompute:
            return cp.checkpoint(self.temporal_norm, hidden_states, use_reentrant=False)
        return self.temporal_norm(hidden_states)

    # -- helpers for temporal attention --------------------------------

    @staticmethod
    def _rearrange_for_temporal(
        hidden_states: torch.Tensor,
        frame_token_counts: list[int],
        num_frames_per_image: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
        """Rearrange from frame-major to position-major layout for temporal attn.

        Input layout  (frame-major):
            [img0_frame0(N0), img0_frame1(N0), ..., img1_frame0(N1), ...]
        Output layout (position-major, per image):
            [img0_pos0(K0), img0_pos1(K0), ..., img1_pos0(K1), ...]

        Returns:
            rearranged: (total_tokens, D) in position-major order
            temporal_cu_seqlens: for flash-attention variable-length batching
            tokens_per_frame_per_img: N_i for each image
            num_frames_list: K_i for each image
        """
        device = hidden_states.device
        D = hidden_states.shape[-1]

        chunks: list[torch.Tensor] = []
        temporal_group_sizes: list[int] = []

        offset = 0
        ftc_idx = 0
        for K in num_frames_per_image:
            N = frame_token_counts[ftc_idx]
            frame_tokens = hidden_states[offset : offset + K * N].view(K, N, D)
            # (K, N, D) -> (N, K, D) -> (N*K, D)
            chunks.append(frame_tokens.transpose(0, 1).reshape(N * K, D))
            temporal_group_sizes.extend([K] * N)
            offset += K * N
            ftc_idx += K

        rearranged = torch.cat(chunks, dim=0)
        cu = torch.tensor(
            temporal_group_sizes, dtype=torch.int32, device=device
        ).cumsum(0)
        temporal_cu_seqlens = F.pad(cu, (1, 0), value=0).to(torch.int32)
        return rearranged, temporal_cu_seqlens

    @staticmethod
    def _build_position_major_valid_mask(
        frame_valid_mask: list[bool],
        frame_token_counts: list[int],
        num_frames_per_image: list[int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build token validity and packed temporal lengths in position-major layout."""
        if not num_frames_per_image:
            return (
                torch.empty(0, dtype=torch.bool, device=device),
                torch.empty(0, dtype=torch.int32, device=device),
            )

        frame_mask = torch.tensor(frame_valid_mask, dtype=torch.bool, device=device)
        num_frames = torch.tensor(num_frames_per_image, dtype=torch.long, device=device)
        frame_start_list: list[int] = []
        frame_offset = 0
        for num_image_frames in num_frames_per_image:
            frame_start_list.append(frame_offset)
            frame_offset += num_image_frames
        frame_starts = torch.tensor(frame_start_list, dtype=torch.long, device=device)
        tokens_per_frame = torch.tensor(
            [frame_token_counts[start] for start in frame_start_list],
            dtype=torch.long,
            device=device,
        )

        group_sizes = tokens_per_frame * num_frames
        total_tokens = int(group_sizes.sum().item())
        if total_tokens == 0:
            valid_mask = torch.empty(0, dtype=torch.bool, device=device)
        else:
            image_ids = torch.repeat_interleave(
                torch.arange(num_frames.numel(), device=device), group_sizes
            )
            group_starts = F.pad(group_sizes.cumsum(0)[:-1], (1, 0), value=0)
            offsets = (
                torch.arange(total_tokens, device=device) - group_starts[image_ids]
            )
            frame_indices = frame_starts[image_ids] + offsets.remainder(
                num_frames[image_ids]
            )
            valid_mask = frame_mask[frame_indices]

        valid_counts = torch.zeros_like(num_frames, dtype=torch.int32)
        frame_image_ids = torch.repeat_interleave(
            torch.arange(num_frames.numel(), device=device), num_frames
        )
        valid_counts.scatter_add_(0, frame_image_ids, frame_mask.to(torch.int32))

        packed_images = valid_counts > 1
        if packed_images.any():
            packed_lengths = torch.repeat_interleave(
                valid_counts[packed_images], tokens_per_frame[packed_images]
            )
        else:
            packed_lengths = torch.empty(0, dtype=torch.int32, device=device)
        return valid_mask, packed_lengths

    @staticmethod
    def _rearrange_from_temporal(
        hidden_states: torch.Tensor,
        frame_token_counts: list[int],
        num_frames_per_image: list[int],
    ) -> torch.Tensor:
        """Inverse of _rearrange_for_temporal: position-major -> frame-major."""
        D = hidden_states.shape[-1]
        chunks: list[torch.Tensor] = []
        offset = 0
        ftc_idx = 0
        for K in num_frames_per_image:
            N = frame_token_counts[ftc_idx]
            pos_tokens = hidden_states[offset : offset + N * K].view(N, K, D)
            # (N, K, D) -> (K, N, D) -> (K*N, D)
            chunks.append(pos_tokens.transpose(0, 1).reshape(K * N, D))
            offset += N * K
            ftc_idx += K
        return torch.cat(chunks, dim=0)

    @staticmethod
    def _rearrange_dt_for_temporal(
        dt_per_token: torch.Tensor,
        frame_token_counts: list[int],
        num_frames_per_image: list[int],
    ) -> torch.Tensor:
        """Rearrange per-token dt from frame-major to position-major layout."""
        parts: list[torch.Tensor] = []
        ftc_idx = 0
        offset = 0
        for K in num_frames_per_image:
            N = frame_token_counts[ftc_idx]
            frame_dt = dt_per_token[offset : offset + N * K].view(K, N)
            parts.append(frame_dt.T.reshape(-1))
            offset += N * K
            ftc_idx += K
        return torch.cat(parts, dim=0)

    def _temporal_zero_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Zero temporal contribution that still ties the temporal parameters
        into the autograd graph.

        Every degenerate early-exit of ``_temporal_attention`` MUST return via
        this helper in 'independent' mode: whether temporal_norm/time_embed/
        qkv/proj participate in backward is data-dependent (per-rank batch
        composition), and under FSDP a parameter used on some ranks but not
        others desynchronizes the gradient reduce-scatter (NCCL hang). The
        ``dummy.sum() * 0.0`` term is numerically zero but registers usage of
        all four parameter groups on every rank.
        """
        out = torch.zeros_like(hidden_states)
        if self.temporal_param_mode == "independent":
            dummy = self._apply_temporal_norm(hidden_states[:1])
            dummy_idx = torch.zeros(1, dtype=torch.long, device=hidden_states.device)
            dummy = dummy + self.temporal_time_embed(dummy_idx).to(dummy.dtype)
            dummy_v = self.temporal_qkv(dummy).chunk(3, dim=-1)[2]
            dummy = self.temporal_proj(dummy_v)
            out = out + dummy.sum() * 0.0
        return out

    def _temporal_attention(
        self,
        hidden_states: torch.Tensor,
        frame_token_counts: list[int],
        num_frames_per_image: list[int],
        frame_valid_mask: list[bool] | None = None,
        dt_per_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run causal temporal attention (same-position, across frames)."""
        if all(K <= 1 for K in num_frames_per_image):
            return self._temporal_zero_output(hidden_states)

        independent = self.temporal_param_mode == "independent"
        normed = (
            self._apply_temporal_norm(hidden_states)
            if independent
            else self._apply_norm1(hidden_states)
        )

        rearranged, temporal_cu = self._rearrange_for_temporal(
            normed, frame_token_counts, num_frames_per_image
        )

        # Build per-token dt in position-major layout for Layer 2 embedding.
        # When dt_per_token is provided, rearrange it the same way as hidden_states.
        # When absent, fall back to ordinal slot indices [0, 1, ..., K-1].
        temporal_embed_idx = None
        if independent:
            if dt_per_token is not None:
                temporal_embed_idx = self._rearrange_dt_for_temporal(
                    dt_per_token, frame_token_counts, num_frames_per_image
                )
            else:
                slot_parts: list[torch.Tensor] = []
                ftc_idx = 0
                for K in num_frames_per_image:
                    N = frame_token_counts[ftc_idx]
                    slots = torch.arange(K, device=rearranged.device, dtype=torch.long)
                    slot_parts.append(slots.repeat(N))
                    ftc_idx += K
                temporal_embed_idx = torch.cat(slot_parts, dim=0)

        output_valid_mask = None
        if frame_valid_mask is not None:
            output_valid_mask, packed_lengths = self._build_position_major_valid_mask(
                frame_valid_mask,
                frame_token_counts,
                num_frames_per_image,
                rearranged.device,
            )
            if packed_lengths.numel() == 0:
                # Frames exist structurally (some K>1) but every history frame
                # is masked invalid (e.g. an all-episode-start batch): same
                # FSDP unused-parameter hazard as the all-K<=1 exit above.
                return self._temporal_zero_output(hidden_states)

            group_keep_mask = []
            offset = 0
            ftc_idx = 0
            for K in num_frames_per_image:
                N = frame_token_counts[ftc_idx]
                valid_for_image = output_valid_mask[offset : offset + N * K].view(N, K)
                keep_for_image = valid_for_image.sum(dim=1) > 1
                group_keep_mask.append(
                    keep_for_image.unsqueeze(1).expand(N, K).reshape(-1)
                    & valid_for_image.reshape(-1)
                )
                offset += N * K
                ftc_idx += K

            output_valid_mask = torch.cat(group_keep_mask, dim=0)
            rearranged = rearranged[output_valid_mask]
            if temporal_embed_idx is not None:
                temporal_embed_idx = temporal_embed_idx[output_valid_mask]
            temporal_cu = F.pad(packed_lengths.cumsum(0), (1, 0), value=0).to(
                torch.int32
            )

        if independent:
            max_idx = self.temporal_time_embed.num_embeddings - 1
            if temporal_embed_idx.max() > max_idx:
                logger.warning_once(
                    "temporal_embed_idx max (%d) >= temporal_max_dt (%d), "
                    "values will be clamped. Consider increasing temporal_max_dt.",
                    temporal_embed_idx.max().item(),
                    self.temporal_time_embed.num_embeddings,
                )
            t_emb = self.temporal_time_embed(temporal_embed_idx.clamp_max(max_idx)).to(
                rearranged.dtype
            )
            rearranged = rearranged + t_emb

        seq_length = rearranged.shape[0]
        attn = self.attn
        qkv_proj = self.temporal_qkv if independent else attn.qkv

        query_states, key_states, value_states = (
            qkv_proj(rearranged)
            .reshape(seq_length, 3, attn.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )

        # No spatial RoPE for temporal attention — temporal ordering comes
        # from the sinusoidal embedding already added to hidden_states.
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            attn.config._attn_implementation, None
        )

        if is_flash_attention_requested(attn.config):
            max_seqlen = int((temporal_cu[1:] - temporal_cu[:-1]).max().item())
            attn_output, _ = attention_interface(
                attn,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=attn.scaling,
                dropout=0.0,
                cu_seq_lens_q=temporal_cu,
                cu_seq_lens_k=temporal_cu,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=True,
                deterministic=getattr(attn, "deterministic", False),
            )
        else:
            # Fallback: chunk-by-chunk with causal mask
            lengths = temporal_cu[1:] - temporal_cu[:-1]
            splits = [
                torch.split(t, lengths.tolist(), dim=2)
                for t in (query_states, key_states, value_states)
            ]
            attn_outputs = []
            for q, k, v in zip(*splits):
                K_len = q.shape[2]
                causal_mask = torch.triu(
                    torch.full(
                        (K_len, K_len), float("-inf"), device=q.device, dtype=q.dtype
                    ),
                    diagonal=1,
                )
                out, _ = attention_interface(
                    attn,
                    q,
                    k,
                    v,
                    attention_mask=causal_mask.unsqueeze(0).unsqueeze(0),
                    scaling=attn.scaling,
                    dropout=0.0,
                    is_causal=False,
                    deterministic=getattr(attn, "deterministic", False),
                )
                attn_outputs.append(out)
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = (self.temporal_proj if independent else attn.proj)(attn_output)

        if output_valid_mask is not None:
            full_attn_output = torch.zeros(
                output_valid_mask.shape[0],
                attn_output.shape[-1],
                dtype=attn_output.dtype,
                device=attn_output.device,
            )
            full_attn_output[output_valid_mask] = attn_output
            attn_output = full_attn_output

        # Rearrange back to frame-major order
        attn_output = self._rearrange_from_temporal(
            attn_output, frame_token_counts, num_frames_per_image
        )
        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        # MEM-specific
        temporal_attn_enable: bool = False,
        frame_token_counts: list[int] | None = None,
        num_frames_per_image: list[int] | None = None,
        frame_valid_mask: list[bool] | None = None,
        dt_per_token: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self._apply_norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        if temporal_attn_enable and frame_token_counts is not None:
            hidden_states = hidden_states + self._temporal_attention(
                hidden_states,
                frame_token_counts,
                num_frames_per_image,
                frame_valid_mask=frame_valid_mask,
                dt_per_token=dt_per_token,
            )

        hidden_states = hidden_states + self.mlp(self._apply_norm2(hidden_states))
        return hidden_states


# ---------------------------------------------------------------------------
# Main Vision Model
# ---------------------------------------------------------------------------


class Qwen3_5VLAVisionModel(Qwen3_5PreTrainedModel):
    """MEM-augmented Qwen3.5 Vision Encoder.

    Extends the standard Qwen3_5VisionModel with:
    - Per-frame independent patch embedding (temporal_patch_size=1)
    - Sinusoidal temporal position encoding
    - Spatiotemporal separated attention
    - Current-frame-only token output
    """

    config: Qwen3_5VLAVisionConfig
    input_modalities = ("image", "video")
    _no_split_modules = ["Qwen3_5VLAVisionBlock"]

    def __init__(
        self,
        config: Qwen3_5VLAVisionConfig,
        *inputs,
        use_selective_recompute: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.use_selective_recompute = use_selective_recompute
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.temporal_attn_interval = config.temporal_attn_interval

        self.patch_embed = Qwen3_5VisionPatchEmbed(config=config)

        self.pos_embed = nn.Embedding(
            config.num_position_embeddings, config.hidden_size
        )
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [
                Qwen3_5VLAVisionBlock(
                    config, use_selective_recompute=use_selective_recompute
                )
                for _ in range(config.depth)
            ]
        )
        self.merger = Qwen3_5VisionPatchMerger(
            config=config,
            use_postshuffle_norm=False,
        )

        self.gradient_checkpointing = False
        self.post_init()

    # -- reuse parent helpers ------------------------------------------

    def rot_pos_emb(
        self, grid_thw: torch.Tensor, total_output_tokens: int | None = None
    ) -> torch.Tensor:
        # Bit-identical to _rot_pos_emb_fallback for arbitrary (variable-K)
        # grids — see tests/test_vision_ops_bitexact.py.
        return rot_pos_emb_op(
            self.rotary_pos_emb.inv_freq,
            grid_thw,
            self.spatial_merge_size,
            total_output_tokens=total_output_tokens,
        )

    def _rot_pos_emb_fallback(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        grid_thw_list = grid_thw.tolist()

        max_hw = max(max(h, w) for _, h, w in grid_thw_list)
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device

        total_tokens = sum(t * h * w for t, h, w in grid_thw_list)
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw_list:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            row_idx = (
                block_rows[:, None, None, None] * merge_size
                + intra_row[None, None, :, None]
            )
            col_idx = (
                block_cols[None, :, None, None] * merge_size
                + intra_col[None, None, None, :]
            )
            row_idx = row_idx.expand(
                merged_h, merged_w, merge_size, merge_size
            ).reshape(-1)
            col_idx = col_idx.expand(
                merged_h, merged_w, merge_size, merge_size
            ).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids].flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(
        self, grid_thw: torch.Tensor, total_output_tokens: int | None = None
    ) -> torch.Tensor:
        # Bit-identical to _fast_pos_embed_interpolate_fallback for arbitrary
        # (variable-K) grids — see tests/test_vision_ops_bitexact.py.
        return pos_embed_interp_op(
            grid_thw,
            self.pos_embed.weight,
            self.num_grid_per_side,
            self.config.spatial_merge_size,
            total_output_tokens=total_output_tokens,
        )

    def _fast_pos_embed_interpolate_fallback(self, grid_thw: torch.Tensor):
        pos_embed_weight = self.pos_embed.weight
        grid_thw_list = grid_thw.tolist()
        grid_ts = [row[0] for row in grid_thw_list]
        grid_hs = [row[1] for row in grid_thw_list]
        grid_ws = [row[2] for row in grid_thw_list]
        device = pos_embed_weight.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in grid_thw_list:
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(
            weight_list, dtype=pos_embed_weight.dtype, device=device
        )
        pos_embeds = pos_embed_weight[idx_tensor] * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split(
            [h * w for h, w in zip(grid_hs, grid_ws)]
        )

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(
                    t, h // merge_size, merge_size, w // merge_size, merge_size, -1
                )
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    # -- build temporal metadata ---------------------------------------

    @staticmethod
    def _build_temporal_metadata(
        grid_thw: torch.Tensor,
        num_frames_per_image: list[int],
        device: torch.device,
        dt_per_frame: list[int] | None = None,
    ) -> tuple[torch.Tensor, list[int]]:
        """Build per-token dt indices and per-frame token counts.

        Supports per-image variable frame counts (K_i + 1 each).

        Args:
            dt_per_frame: per-frame dt in raw-data-frame units (len = total_frames).
                Current frames must have dt=0. If None, falls back to ordinal
                frame indices [K_i, K_i-1, ..., 1, 0] per image.

        Returns:
            dt_indices: (total_tokens,) int tensor — dt values per token
            frame_token_counts: list of token counts per frame
        """
        grid_thw_list = grid_thw.tolist()

        dt_indices_parts: list[torch.Tensor] = []
        frame_token_counts: list[int] = []

        frame_idx = 0
        dt_idx = 0
        for frames_this_image in num_frames_per_image:
            K_i = frames_this_image - 1

            for local_f in range(frames_this_image):
                t_val, h_val, w_val = grid_thw_list[frame_idx]
                n_tokens = int(t_val * h_val * w_val)
                frame_token_counts.append(n_tokens)

                if dt_per_frame is not None:
                    dt_val = dt_per_frame[dt_idx]
                else:
                    dt_val = K_i - local_f
                dt_indices_parts.append(
                    torch.full((n_tokens,), dt_val, dtype=torch.long, device=device)
                )
                frame_idx += 1
                dt_idx += 1

        dt_indices = torch.cat(dt_indices_parts)
        return dt_indices, frame_token_counts

    # -- combine current & history pixels ------------------------------

    def _combine_inputs(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        pixel_values_history: torch.Tensor | None,
        image_grid_thw_history: torch.Tensor | None,
        num_history_per_image: list[int],
        history_frame_mask: list[bool] | None = None,
        history_dt: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], list[bool], list[int] | None]:
        """Interleave history and current frame tokens per image.

        Supports per-image variable history length.  Images with K_i=0
        contribute only their current frame (no history interleaving).

        Returns:
            combined_hidden, combined_grid_thw, num_frames_per_image,
            frame_valid_mask, combined_dt (None when history_dt is None)
        """
        num_images = grid_thw.shape[0]
        total_history = sum(num_history_per_image)

        if total_history > 0:
            if pixel_values_history is None or image_grid_thw_history is None:
                raise ValueError(
                    "pixel_values_history and image_grid_thw_history must be "
                    "provided when any image has history frames"
                )
            if image_grid_thw_history.shape[0] != total_history:
                raise ValueError(
                    f"image_grid_thw_history has {image_grid_thw_history.shape[0]} "
                    f"rows but sum(num_history_per_image) = {total_history}"
                )

        if history_frame_mask is not None and len(history_frame_mask) != total_history:
            raise ValueError(
                f"history_frame_mask length {len(history_frame_mask)} != "
                f"sum(num_history_per_image) {total_history}"
            )
        if history_dt is not None and len(history_dt) != total_history:
            raise ValueError(
                f"history_dt length {len(history_dt)} != "
                f"sum(num_history_per_image) {total_history}"
            )
        if history_dt is not None and any(int(dt) < 1 for dt in history_dt):
            # dt == 0 is reserved for current frames: the output-token filter
            # keeps tokens where dt_indices == 0, so a history frame carrying
            # dt = 0 (e.g. episode-start padding replicated from the current
            # frame) would leak into the output tokens and corrupt the merger
            # token count. Padded/invalid frames must be excluded via
            # history_frame_mask, not encoded as dt = 0.
            raise ValueError(
                f"history_dt must be >= 1 for every history frame, got "
                f"{sorted(set(int(dt) for dt in history_dt))[:5]}; mark padded "
                "frames invalid via history_frame_mask instead of dt=0"
            )

        current_tokens = self.patch_embed(pixel_values)
        current_counts = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()
        current_splits = torch.split(current_tokens, current_counts)

        history_splits: tuple[torch.Tensor, ...] = ()
        if total_history > 0:
            history_tokens = self.patch_embed(pixel_values_history)
            history_counts = (
                image_grid_thw_history[:, 0]
                * image_grid_thw_history[:, 1]
                * image_grid_thw_history[:, 2]
            ).tolist()
            history_splits = torch.split(history_tokens, history_counts)

        combined_parts: list[torch.Tensor] = []
        combined_grid_parts: list[torch.Tensor] = []
        frame_valid_mask: list[bool] = []
        num_frames_per_image: list[int] = []
        combined_dt: list[int] | None = [] if history_dt is not None else None

        hist_offset = 0
        for img_i in range(num_images):
            K_i = num_history_per_image[img_i]
            for k in range(K_i):
                combined_parts.append(history_splits[hist_offset + k])
                combined_grid_parts.append(
                    image_grid_thw_history[hist_offset + k].unsqueeze(0)
                )
                if history_frame_mask is not None:
                    frame_valid_mask.append(history_frame_mask[hist_offset + k])
                else:
                    frame_valid_mask.append(True)
                if combined_dt is not None:
                    combined_dt.append(history_dt[hist_offset + k])
            combined_parts.append(current_splits[img_i])
            combined_grid_parts.append(grid_thw[img_i].unsqueeze(0))
            frame_valid_mask.append(True)
            if combined_dt is not None:
                combined_dt.append(0)
            num_frames_per_image.append(K_i + 1)
            hist_offset += K_i

        combined_hidden = torch.cat(combined_parts, dim=0)
        combined_grid_thw = torch.cat(combined_grid_parts, dim=0)
        return (
            combined_hidden,
            combined_grid_thw,
            num_frames_per_image,
            frame_valid_mask,
            combined_dt,
        )

    # -- forward -------------------------------------------------------

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        pixel_values_history: torch.Tensor | None = None,
        image_grid_thw_history: torch.Tensor | None = None,
        history_image_mask: torch.Tensor | None = None,
        num_history_per_image: list[int] | None = None,
        history_dt: list[int] | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        """
        Args:
            hidden_states: Current-frame pixel values (flattened patches).
            grid_thw: (num_images, 3) — T, H, W per current-frame image.
            pixel_values_history: History-frame pixel values, or None.
            image_grid_thw_history: Grid metadata for history frames.
            history_image_mask: Legacy (num_images, K) bool tensor for uniform-K.
                Ignored when num_history_per_image is provided.
            num_history_per_image: Per-image history count [K_0, K_1, ...].
                When None, derived from image_grid_thw_history for backward compat.
            history_dt: Per-history-frame dt in raw-data-frame units.
                len = sum(num_history_per_image). None → ordinal fallback.
        """
        has_history = (
            pixel_values_history is not None and image_grid_thw_history is not None
        ) or (num_history_per_image is not None and sum(num_history_per_image) > 0)
        if (pixel_values_history is None) ^ (image_grid_thw_history is None):
            raise ValueError(
                "pixel_values_history and image_grid_thw_history must be provided together"
            )
        total_output_tokens = kwargs.pop("total_output_tokens", None)

        if has_history:
            num_images = grid_thw.shape[0]

            # Backward compat: derive num_history_per_image from uniform-K layout
            if num_history_per_image is None:
                num_history_total = image_grid_thw_history.shape[0]
                if num_images == 0 or num_history_total % num_images != 0:
                    raise ValueError(
                        "image_grid_thw_history rows must be divisible by "
                        "num_images when num_history_per_image is not provided"
                    )
                K = num_history_total // num_images
                num_history_per_image = [K] * num_images

            # Auto-compute dt from config when not provided by data pipeline
            interval = getattr(self.config, "temporal_history_interval", 0)
            if history_dt is None and interval > 0:
                history_dt = []
                for K_i in num_history_per_image:
                    for k in range(K_i):
                        history_dt.append((K_i - k) * interval)

            # Convert legacy history_image_mask tensor to flat list[bool]
            history_frame_mask: list[bool] | None = None
            if history_image_mask is not None:
                mask_t = history_image_mask.to(dtype=torch.bool).flatten()
                history_frame_mask = mask_t.tolist()

            (
                hidden_states,
                combined_grid_thw,
                num_frames_per_image,
                frame_valid_mask,
                combined_dt,
            ) = self._combine_inputs(
                hidden_states,
                grid_thw,
                pixel_values_history,
                image_grid_thw_history,
                num_history_per_image=num_history_per_image,
                history_frame_mask=history_frame_mask,
                history_dt=history_dt,
            )

            # Temporal metadata
            dt_indices, frame_token_counts = self._build_temporal_metadata(
                combined_grid_thw,
                num_frames_per_image,
                hidden_states.device,
                dt_per_frame=combined_dt,
            )

            # Spatial position embedding (repeats per frame automatically)
            pos_embeds = self.fast_pos_embed_interpolate(
                combined_grid_thw, total_output_tokens=total_output_tokens
            )
            hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)

            # Sinusoidal temporal position encoding (Layer 1)
            time_emb = sinusoidal_time_embed(dt_indices, self.config.hidden_size)
            hidden_states = hidden_states + time_emb.to(hidden_states.dtype)

            # dt_per_token for Block._temporal_attention (Layer 2, independent mode)
            dt_per_token = dt_indices

            # RoPE (spatial only, shared across frames)
            rotary_pos_emb = self.rot_pos_emb(
                combined_grid_thw, total_output_tokens=total_output_tokens
            )
            active_grid_thw = combined_grid_thw
        else:
            # Standard single-frame path (no history)
            hidden_states = self.patch_embed(hidden_states)
            pos_embeds = self.fast_pos_embed_interpolate(
                grid_thw, total_output_tokens=total_output_tokens
            )
            hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)
            rotary_pos_emb = self.rot_pos_emb(
                grid_thw, total_output_tokens=total_output_tokens
            )
            active_grid_thw = grid_thw

            dt_indices = None
            frame_token_counts = None
            num_frames_per_image = None
            frame_valid_mask = None
            dt_per_token = None

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        # Compact half-dim cos/sin — the rope.pack contract (same as the base
        # vision model). The PyTorch fallback duplicates to full head_dim
        # internally, so both attention paths consume this layout.
        position_embeddings = (
            rotary_pos_emb.cos().float(),
            rotary_pos_emb.sin().float(),
        )

        # cu_seqlens: per-frame grouping (each frame's H*W tokens as one group)
        cu_seqlens = torch.repeat_interleave(
            active_grid_thw[:, 1] * active_grid_thw[:, 2],
            active_grid_thw[:, 0],
        ).cumsum(
            dim=0,
            dtype=active_grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

        # Run through transformer blocks
        for layer_idx, blk in enumerate(self.blocks):
            temporal_enable = has_history and (
                layer_idx % self.temporal_attn_interval == 0
            )
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
                temporal_attn_enable=temporal_enable,
                frame_token_counts=frame_token_counts,
                num_frames_per_image=num_frames_per_image,
                frame_valid_mask=frame_valid_mask,
                dt_per_token=dt_per_token,
                **kwargs,
            )

        # Filter: keep only current-frame tokens
        if has_history and dt_indices is not None:
            current_mask = dt_indices == 0
            hidden_states = hidden_states[current_mask]

        # PatchMerger
        merged_hidden_states = self.merger(hidden_states)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
        )

    # -- pretrained weight compatibility -------------------------------

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Handle Conv3d weight folding when loading from a checkpoint with
        a different temporal_patch_size.

        If the checkpoint has temporal_patch_size=2 but the model uses
        temporal_patch_size=1 (or vice versa), fold/repeat the temporal
        dimension to match.  When temporal_patch_size matches (the common
        case), this is a no-op.
        """
        conv_key = prefix + "patch_embed.proj.weight"
        if conv_key in state_dict:
            w = state_dict[conv_key]
            model_tps = self.config.temporal_patch_size
            ckpt_tps = w.shape[2]
            if ckpt_tps != model_tps:
                if ckpt_tps == 2 and model_tps == 1:
                    logger.info(
                        f"Folding Conv3d temporal dimension: {w.shape} -> "
                        f"({w.shape[0]}, {w.shape[1]}, 1, {w.shape[3]}, {w.shape[4]})"
                    )
                    state_dict[conv_key] = w.mean(dim=2, keepdim=True)
                else:
                    logger.warning(
                        f"Conv3d temporal_patch_size mismatch: checkpoint={ckpt_tps}, "
                        f"model={model_tps}. Skipping automatic conversion."
                    )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
