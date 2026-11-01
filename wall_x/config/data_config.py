"""Public data config dataclasses.

Only data backends shipped in the public package should define config classes
here. Internal backends register their dataclasses from their own packages via
``wall_x.config.registry.register_data_config``.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .registry import register_data_config


@dataclass
class DataConfig:
    """Base fields shared by data backends.

    ``normalizer_config`` may contain:
    - ``min_key``: stats key for the minimum value.
    - ``delta_key``: stats key for the value range.
    - ``customized_action_statistic_dof``: explicit action-stats JSON path.
    """

    dataset_type: str = "lerobot"
    resolution: Dict[str, int] = field(
        default_factory=lambda: {
            "face_view": 256,
            "left_wrist_view": 256,
            "right_wrist_view": 256,
        }
    )
    train_test_split: float = 0.95
    normalizer_config: Optional[Dict[str, Any]] = None


@register_data_config("lerobot")
@dataclass
class LeRobotDataConfig(DataConfig):
    """LeRobot data config.

    ``lerobot_config`` is expected to contain fields such as ``repo_id`` and
    ``root`` for a HuggingFace LeRobot dataset. ``norm_stats_path`` points to
    explicit action normalizer stats; the core package does not bundle private
    defaults.
    """

    dataset_type: str = "lerobot"
    dataset_config_path: Optional[str] = None
    lerobot_config: Optional[Dict[str, Any]] = None
    key_mappings: Optional[Dict[str, Any]] = None
    norm_stats_path: Optional[str] = None
    priority_order: Optional[Dict[str, float]] = None
    camera_name_mapping: Optional[Dict[str, str]] = None
    num_workers: int = 4
    action_tokenizer_path: Optional[str] = None
    use_fast_tokenizer: bool = False
    padding_side: str = "left"
    noise_scheduler: Optional[Dict[str, Any]] = None
    # --- qwen preprocess knobs (read by prompt/state encoding) ---
    state_bins: int = 256
    use_embodied_system_prompt_ratio: float = 0.0
    # --- wall-wm lerobot fields ---
    image_history_length: int = 1
    obs_history_stride: int = 0
    event_level: bool = False
    image_horizon: int = 16
    future_image_length: int = 16
    future_image_interval: int = 2
    action_history_length: int = 1


__all__ = [
    "DataConfig",
    "LeRobotDataConfig",
]
