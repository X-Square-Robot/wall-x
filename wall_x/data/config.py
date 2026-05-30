# Copyright (c) 2025, X-Square-Robot
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from qwen_vl_utils.vision_process import MIN_PIXELS, MAX_PIXELS, IMAGE_FACTOR


# Tactile sensor file mapping for data processing
TACTILE_FILE_MAPPING = {
    "tactile_data_left": "left_tactile",
    "tactile_data_right": "right_tactile",
}

# Supported action datasets
ACTION_DATASET_NAMES = [
    "x2_normal",
    "agibotworld_alpha",
    "droid",
    "fractal",
    "bridge_data_v2",
    "DobbE",
    "RH20T",
    "UMI-biarm",
    "austin_buds",
    "austin_sailor",
    "austin_sirius",
    "bc_z",
    "berkeley_autolab_ur5",
    "berkeley_cable_routing",
    "berkeley_fanuc_manipulation",
    "dlr_edan_shared_control",
    "fmb",
    "furniture_bench",
    "jaco_play",
    "nyu_rot",
    "stanford_hydra",
    "stanford_kuka_multimodal",
    "taco_play",
    "utaustin_mutex",
    "viola",
    "physical-intelligence/libero",
    "lerobot/aloha_mobile_cabinet",
]

# Supported multimodal datasets
MULTIMODAL_DATASET_NAMES = [
    "x2_multimodal_from_action",
    "x2_multimodal",
    "x2_subtask_generation",
    "multimodal_CapsFusion",
    "multimodal_Robo2VLM",
    "multimodal_RoboPoint",
    "multimodal_EQA",
    "multimodal_Cambrian",
    "multimodal_pixmo",
    "multimodal_VQAv2",
    "multimodal_COCO",
]


@dataclass
class X2RDataProcessingConfig:
    """Configuration class for X2R data processing pipeline.

    This class contains all the necessary parameters for processing robotic data
    including camera mappings, tactile sensor configurations, action predictions,
    and various processing options.
    """

    # Action prediction configuration
    predict_action_keys: List[str] = field(default_factory=list)
    obs_action_keys: List[str] = field(default_factory=list)

    # Image resolution settings for different views
    resolution: Dict[str, int] = field(
        default_factory=lambda: {
            "face_view": -1,
            "left_wrist_view": 128,
            "right_wrist_view": 128,
        }
    )

    # Dataset splitting
    train_test_split: float = 0.9
    split_seed: int = 42

    # Instruction handling
    priority_order: Optional[Dict[str, float]] = None

    # Vision model parameters
    model_type: str = "qwen2_5"
    max_pixels: int = MAX_PIXELS
    min_pixels: int = MIN_PIXELS
    image_factor: int = IMAGE_FACTOR

    generate_subtask_ratio: float = 0.0

    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Validate train/test split
        if not 0 < self.train_test_split < 1:
            raise ValueError(
                f"train_test_split must be between 0 and 1, got {self.train_test_split}"
            )

    def as_dict(self) -> Dict:
        """Convert configuration to dictionary format.

        Returns:
            Dict: Configuration as dictionary
        """
        return self.__dict__

    def update(self, **kwargs) -> "X2RDataProcessingConfig":
        """Update configuration parameters.

        Args:
            **kwargs: Key-value pairs to update

        Returns:
            X2RDataProcessingConfig: Updated configuration instance
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        return self
