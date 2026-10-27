"""Operator proxy layer with runtime fallback backends."""

from wall_x.model.core.ops.norm import (
    rmsnorm,
    layernorm,
    add_rmsnorm,
    gated_rmsnorm,
    scaled_add_layernorm,
)
from wall_x.model.core.ops.adarmsnorm import adarmsnorm
from wall_x.model.core.ops.activation import silu_and_mul
from wall_x.model.core.ops.position import pos_embed_interp
from wall_x.model.core.ops.rope import rope, m_rope, rot_pos_emb
from wall_x.model.core.ops.moe import (
    permute,
    unpermute,
    two_expert_pack,
    two_expert_unpack,
)
from wall_x.model.core.ops.index import get_rope_index, get_window_index
from wall_x.model.core.ops.qwen35_index import qwen35_get_rope_index

__all__ = [
    "rmsnorm",
    "layernorm",
    "scaled_add_layernorm",
    "adarmsnorm",
    "add_rmsnorm",
    "gated_rmsnorm",
    "silu_and_mul",
    "pos_embed_interp",
    "rope",
    "m_rope",
    "rot_pos_emb",
    "permute",
    "unpermute",
    "two_expert_pack",
    "two_expert_unpack",
    "get_rope_index",
    "qwen35_get_rope_index",
    "get_window_index",
]
