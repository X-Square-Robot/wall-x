"""Base model config dataclasses.

Optional model variants register their config dataclasses from their own
packages via ``wall_x.config.registry.register_model_config``.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .registry import register_model_config


@dataclass
class ModelConfig:
    """Base model architecture fields."""

    use_ema: bool = False
    attn_implementation: Optional[str] = None
    attn_deterministic: Optional[bool] = None
    ar_loss_weight: float = 1.0


@register_model_config("qwen2_5")
@dataclass
class QActModelConfig(ModelConfig):
    """QAct model config for the Qwen2.5 VLA path."""

    config_path: str = ""
    processor_path: str = ""
    pretrained_path: Optional[str] = None
    backbone: str = "qwen2_5"
    action_tokenizer_type: Optional[str] = None
    action_tokenizer_path: Optional[str] = None
    action_tokenizer_checkpoint_path: Optional[str] = None
    action_tokenizer_config_dir: Optional[str] = None
    new_special_tokens: Optional[List[str]] = None
    flow_loss_weight: float = 1.0
    enable_customized_robot_config: bool = False
    customized_robot_config: Optional[Dict[str, Any]] = None


__all__ = ["ModelConfig", "QActModelConfig"]
