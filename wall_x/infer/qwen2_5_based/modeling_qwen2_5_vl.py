import torch.nn as nn
import torch
from flash_attn import (
    flash_attn_varlen_func,
)
import torch.nn.functional as F
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLVisionBlock,
    Qwen2_5_VLMLP,
    Qwen2_5_VLPatchMerger,
)
from typing import Optional, Tuple
from xcompute import rot_pos_emb, get_window_index, rope, rmsnorm


class Qwen2_5_VLVisionFlashAttention2(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: Optional[int] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states)
        q, k, v = qkv.split(
            [
                self.dim,
                self.dim,
                self.dim,
            ],
            dim=-1,
        )
        q = q.reshape(1, q.shape[0], self.num_heads, -1)
        k = k.reshape(1, k.shape[0], self.num_heads, -1)
        v = v.reshape(v.shape[0], self.num_heads, -1)

        cos, sin = position_embeddings
        q, k = rope(q.float(), k.float(), cos, sin)
        q = q.squeeze(0).to(torch.bfloat16)
        k = k.squeeze(0).to(torch.bfloat16)

        attn_output = flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen
        ).reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        normed_inputs = rmsnorm(hidden_states, self.weight, self.variance_epsilon)
        return normed_inputs


class Qwen2_5_VLPatchMergerOpt(Qwen2_5_VLPatchMerger):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__(dim, context_dim, spatial_merge_size)
        self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed_x = self.ln_q(x)
        x = self.mlp(normed_x.view(-1, self.hidden_size))
        return x


class Qwen2_5_VLVisionBlockOpt(Qwen2_5_VLVisionBlock):
    def __init__(
        self,
        config,
        attn_implementation: str = "sdpa",
        use_selective_recompute: bool = False,
    ):
        super().__init__(config, attn_implementation, use_selective_recompute)

        self.norm1 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)

        self.attn = Qwen2_5_VLVisionFlashAttention2(
            config.hidden_size, num_heads=config.num_heads
        )
        self.mlp = Qwen2_5_VLMLP(
            config, bias=True, use_selective_recompute=use_selective_recompute
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: Optional[int] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        normed_hidden_states = self.norm1(hidden_states)

        hidden_states = hidden_states + self.attn(
            normed_hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
        )

        normed_hidden_states = self.norm2(hidden_states)

        hidden_states = hidden_states + self.mlp(normed_hidden_states)

        return hidden_states


class Qwen2_5_VisionTransformerPretrainedModelOpt(
    Qwen2_5_VisionTransformerPretrainedModel
):

    def __init__(
        self, config, use_selective_recompute=False, *inputs, **kwargs
    ) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.blocks = nn.ModuleList(
            [
                Qwen2_5_VLVisionBlockOpt(
                    config, config._attn_implementation, use_selective_recompute
                )
                for _ in range(config.depth)
            ]
        )
        self.merger = Qwen2_5_VLPatchMergerOpt(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )
        self.gradient_checkpointing = False

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = rot_pos_emb(
            self.rotary_pos_emb.inv_freq, grid_thw, self.spatial_merge_size
        )
        window_index, cu_window_seqlens = get_window_index(
            grid_thw=grid_thw.to(torch.int32),
            window_size=self.window_size,
            spatial_merge_size=self.spatial_merge_size,
            patch_size=self.patch_size,
            spatial_merge_unit=self.spatial_merge_unit,
        )

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = rotary_pos_emb
        position_embeddings = (emb.cos().unsqueeze(0), emb.sin().unsqueeze(0))

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(
            dim=0,
            dtype=torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        max_seqlen_full = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        max_seqlen_window = (
            (cu_window_seqlens[1:] - cu_window_seqlens[:-1]).max().item()
        )

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
                max_seqlen_now = max_seqlen_full
            else:
                cu_seqlens_now = cu_window_seqlens
                max_seqlen_now = max_seqlen_window

            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                max_seqlen=max_seqlen_now,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states
