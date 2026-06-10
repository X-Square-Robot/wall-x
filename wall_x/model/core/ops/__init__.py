"""Operator proxy layer with runtime fallback backends."""

from wall_x.model.core.ops.index import get_rope_index, get_window_index
from wall_x.model.core.ops.moe import permute, unpermute
from wall_x.model.core.ops.norm import rmsnorm
from wall_x.model.core.ops.rope import m_rope, rope, rot_pos_emb

__all__ = [
    "rmsnorm",
    "rope",
    "m_rope",
    "rot_pos_emb",
    "permute",
    "unpermute",
    "get_rope_index",
    "get_window_index",
]
