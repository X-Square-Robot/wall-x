"""Qwen2.5 inference adapter registration.

The shared inference logic lives in ``harrix.adapters.qwen_vlact``. This module
binds the Qwen2.5 variant key to Wall-X's training-side adapter class.
"""

from wall_x._vendor.harrix.adapters.qwen_vlact import QwenVLActInferAdapter
from wall_x._vendor.harrix.adapters.registry import register_adapter


@register_adapter("qwen2_5")
class Qwen2_5InferAdapter(QwenVLActInferAdapter):
    @classmethod
    def _training_adapter(cls):
        from wall_x.model.qact.qwen2_5.adapter import Qwen2_5Adapter

        return Qwen2_5Adapter
