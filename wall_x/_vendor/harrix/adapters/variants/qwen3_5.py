"""Qwen3.5 inference adapter registration."""

from wall_x._vendor.harrix.adapters.qwen_vlact import QwenVLActInferAdapter
from wall_x._vendor.harrix.adapters.registry import register_adapter
from wall_x.model.qact.qwen3_5.adapter import Qwen3_5Adapter


@register_adapter("qwen3_5")
class Qwen3_5InferAdapter(QwenVLActInferAdapter):
    @classmethod
    def _training_adapter(cls):
        return Qwen3_5Adapter
