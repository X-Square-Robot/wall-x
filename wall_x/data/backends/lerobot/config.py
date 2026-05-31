from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from qwen_vl_utils.vision_process import MIN_PIXELS, MAX_PIXELS, IMAGE_FACTOR


@dataclass
class LerobotConfig:
    """Configuration for the LeRobot preprocessing pipeline.

    Dataset-specific camera display names are optional config inputs. Other
    dataset behavior is derived from the current LeRobot sample.
    """

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
    seed: int = 42

    # Instruction handling
    priority_order: Optional[Dict[str, float]] = None
    camera_name_mapping: Optional[Dict[str, str]] = None

    # Vision model parameters
    model_type: str = "qwen2_5"
    max_pixels: int = MAX_PIXELS
    min_pixels: int = MIN_PIXELS
    image_factor: int = IMAGE_FACTOR

    generate_subtask_ratio: float = 0.0

    # When True, discretized proprioception replaces ``<|propri|>`` in the
    # prompt instead of feeding a continuous proprioception tensor only.
    use_state_string_representation: bool = False
    state_bins: int = 256

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

    def update(self, **kwargs) -> "LerobotConfig":
        """Update configuration parameters.

        Args:
            **kwargs: Key-value pairs to update

        Returns:
            LerobotConfig: Updated configuration instance
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        return self

    def __getitem__(self, key: str):
        return getattr(self, key)

    @classmethod
    def from_yaml_dict(cls, yaml_dict: Dict[str, Any]) -> "LerobotConfig":
        """
        Build a LerobotConfig instance from a YAML dictionary.

        Supports two styles:

        1) Top-level fields:
            train_test_split: 0.8
            model_type: qwen2_5

        2) Nested under `data:` (higher priority):
            data:
              train_test_split: 0.8
              model_type: qwen2_5

        Keys inside `data:` override top-level keys.
        """

        data_config = yaml_dict.get("data", {})

        task_config = yaml_dict.get("task") or {}

        def get(key: str, default: Any = None):
            """
            Helper function:
            Read from ``data:``, then ``task:``, then top-level YAML.
            """
            if key in data_config:
                return data_config[key]
            if key in task_config:
                return task_config[key]
            return yaml_dict.get(key, default)

        # Construct only fields that actually exist in LerobotConfig
        params: Dict[str, Any] = {
            # Action prediction settings
            # Image resolution per camera view
            "resolution": get(
                "resolution",
                {
                    "face_view": -1,
                    "left_wrist_view": 128,
                    "right_wrist_view": 128,
                },
            ),
            # Dataset train/test split configuration
            "train_test_split": get("train_test_split", 0.9),
            "seed": get("seed", 42),
            # Instruction priority ordering (optional)
            "priority_order": get("priority_order", None),
            "camera_name_mapping": get("camera_name_mapping", None),
            # Vision model parameters
            "model_type": get("model_type", "qwen2_5"),
            "max_pixels": get("max_pixels", MAX_PIXELS),
            "min_pixels": get("min_pixels", MIN_PIXELS),
            "image_factor": get("image_factor", IMAGE_FACTOR),
            # Subtask generation ratio
            "generate_subtask_ratio": get("generate_subtask_ratio", 0.0),
            "use_state_string_representation": get(
                "use_state_string_representation", False
            ),
            "state_bins": get("state_bins", 256),
        }

        # Keep only valid dataclass fields (ignore unknown YAML keys)
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_params = {k: v for k, v in params.items() if k in valid_fields}

        return cls(**filtered_params)
