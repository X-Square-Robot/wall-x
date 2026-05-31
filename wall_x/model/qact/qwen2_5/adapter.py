"""Qwen2.5 VLA adapter — variant-specific overrides on top of VLAdapter."""

from wall_x.model.registry import register_model
from wall_x.trainer.adapters.vla_model_adapter import VLAdapter


@register_model("qwen2_5")
class Qwen2_5Adapter(VLAdapter):
    MODEL_TYPE = "qwen2_5"

    @classmethod
    def model_class(cls):
        from wall_x.model.qact.qwen2_5 import Qwen2_5_VLMoEForAction

        return Qwen2_5_VLMoEForAction

    @classmethod
    def config_class(cls):
        from wall_x.model.qact.qwen2_5 import Qwen2_5_VLConfig

        return Qwen2_5_VLConfig

    @classmethod
    def inference_model_class(cls):
        return cls.model_class()

    def get_transformer_layer_cls(self):
        layer_classes = set()
        try:
            from transformers.models.qwen2_vl.modeling_qwen2_vl import (
                Qwen2VLDecoderLayer,
            )

            layer_classes.add(Qwen2VLDecoderLayer)
        except ImportError:
            pass
        try:
            from wall_x.model.qact.qwen2_5.modeling_qwen2_5_vl import (
                Qwen2_5_VLDecoderLayer,
            )

            layer_classes.add(Qwen2_5_VLDecoderLayer)
        except ImportError:
            pass
        return layer_classes if layer_classes else None

    @staticmethod
    def log_attention_implementation(logger, model):
        logger.info(
            f"*** model attention implementation: "
            f"{model.model._attn_implementation} ***"
        )
        logger.info(
            f"*** model.visual attention implementation: "
            f"{model.visual.config._attn_implementation} ***"
        )
