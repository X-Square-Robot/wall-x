"""Qwen3.5 VLA adapter with variant-specific overrides on top of VLAdapter."""

from wall_x.model.registry import register_model
from wall_x.trainer.adapters.vla_model_adapter import VLAdapter


@register_model("qwen3_5")
class Qwen3_5Adapter(VLAdapter):
    MODEL_TYPE = "qwen3_5"

    @classmethod
    def model_class(cls):
        from wall_x.model.qact.qwen3_5 import Qwen3_5_VLAForConditionalGeneration

        return Qwen3_5_VLAForConditionalGeneration

    @classmethod
    def config_class(cls):
        from wall_x.model.qact.qwen3_5 import Qwen3_5Config

        return Qwen3_5Config

    @staticmethod
    def _resolve_ar_action_loss_weight(setting, global_step, *, training):
        if not isinstance(setting, dict):
            return float(setting)
        target = float(setting["target"])
        if not training:
            return target
        initial = float(setting["initial"])
        progress = min(
            max(int(global_step), 0) / int(setting["warmup_steps"]),
            1.0,
        )
        return initial + (target - initial) * progress

    def forward(self, model, batch, **kwargs):
        self._last_batch_info = self._extract_batch_info(batch)
        mode = kwargs.get("mode", "train")
        weight = getattr(self.cfg.model, "ar_action_loss_weight", None)
        if weight is None:
            return model(**batch, mode=mode)
        resolved_weight = self._resolve_ar_action_loss_weight(
            weight,
            kwargs.get("global_step", 0),
            training=model.training and mode == "train",
        )
        return model(**batch, mode=mode, ar_action_loss_weight=resolved_weight)

    @property
    def default_action_lr_keywords(self) -> list[str]:
        # qwen3_5 has its own action-expert naming scheme (vs qwen2_5/qwen3).
        return [
            "action_preprocessor",
            "action_processor",
            "q_proj_action",
            "k_proj_action",
            "v_proj_action",
            "o_proj_action",
            "mlp_action_expert",
            "input_layernorm_action",
            "post_attention_layernorm_action",
            "norm_action",
        ]

    def get_transformer_layer_cls(self):
        layer_classes = set()
        try:
            from wall_x.model.qact.qwen3_5.modeling_qwen3_5_act import (
                Qwen3_5VLADecoderLayer,
            )

            layer_classes.add(Qwen3_5VLADecoderLayer)
        except ImportError:
            pass
        return layer_classes if layer_classes else None

    def validate_pretrained_load_result(self, model, load_result):
        if model.config.vision_config.model_type != "qwen3_5_dino_vision":
            return

        missing = list(load_result.missing_keys)
        unexpected = list(load_result.unexpected_keys)
        visual_missing = [key for key in missing if key.startswith("model.visual.")]
        visual_unexpected = [
            key for key in unexpected if key.startswith("model.visual.")
        ]
        if visual_missing or visual_unexpected:
            raise RuntimeError(
                "DINO visual checkpoint did not load completely: "
                f"missing={visual_missing[:8]}, unexpected={visual_unexpected[:8]}"
            )

        allowed_missing_markers = (
            "action_processor",
            "_action",
            "action_expert",
            "layernorm_action",
            "norm_action",
        )
        invalid_missing = [
            key
            for key in missing
            if not any(marker in key for marker in allowed_missing_markers)
        ]
        if invalid_missing or unexpected:
            raise RuntimeError(
                "DINO VLM checkpoint is incompatible with the Qwen3.5 VLA: "
                f"non_action_missing={invalid_missing[:8]}, "
                f"unexpected={unexpected[:8]}"
            )

        visual_tensor_count = sum(
            key.startswith("model.visual.") for key in model.state_dict()
        )
        if visual_tensor_count == 0:
            raise RuntimeError("constructed DINO model contains no visual tensors")
        if self.logger:
            self.logger.info(
                "Loaded DINO VLM checkpoint strictly: %d visual tensors; "
                "%d expected VLA-only tensors initialized",
                visual_tensor_count,
                len(missing),
            )

    @staticmethod
    def log_attention_implementation(logger, model):
        logger.info(
            f"*** model attention implementation: "
            f"{model.model.language_model.config._attn_implementation} ***"
        )
        logger.info(
            f"*** model.visual attention implementation: "
            f"{model.model.visual.config._attn_implementation} ***"
        )
