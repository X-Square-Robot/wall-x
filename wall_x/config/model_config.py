"""Base model config dataclasses.

Optional model variants register their config dataclasses from their own
packages via ``wall_x.config.registry.register_model_config``.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from .registry import register_model_config


@dataclass
class ModelConfig:
    """Base model architecture fields."""

    use_ema: bool = False
    attn_implementation: Optional[str] = None
    attn_deterministic: Optional[bool] = None
    ar_loss_weight: float = 1.0
    ar_action_loss_weight: Optional[Union[float, Dict[str, Any]]] = None


@register_model_config("qwen2_5", "qwen3_5")
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
    rvq_token_protocol: Optional[str] = None
    rvq_token_layout: str = "layer_major"
    rvq_position_encoding: str = "time_level_2d"
    rvq_action_horizon: Optional[int] = None
    rvq_checkpoint_contract: str = "strict"
    new_special_tokens: Optional[List[str]] = None
    flow_loss_weight: float = 1.0
    enable_customized_robot_config: bool = False
    customized_robot_config: Optional[Dict[str, Any]] = None


__all__ = ["ModelConfig", "QActModelConfig"]
