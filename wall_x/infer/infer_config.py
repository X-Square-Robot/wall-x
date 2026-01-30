import yaml
import os
from wall_x.model.model_utils import update_model_config
# from x2robot_dataset.configs.config import X2RDataConfig

import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field


@dataclass
class X2RDataConfig:
    """
    Unified X2Robot data configuration class (reorganized by README's 5 modules):
      1) Data I/O and caching
      2) Visual input and sampling (image/camera)
      3) Action and time series
      4) Instruction and multimodal
      5) Data cleaning and alignment (validation/augmentation/framework constraints)
    """

    # ----------------------------------------------------------------------
    # 1) Data I/O and caching
    # ----------------------------------------------------------------------
    cache_dir: str = "~/.cache/dataset_cache"
    dataset_config_path: Optional[str] = None
    use_cache: bool = True
    check_mode: bool = True
    preload_size: int = 128
    buffer_size: int = 20000
    batch_size: int = 32
    train_test_split: float = 0.9
    seed: int = 42
    episode_chunk_size: int = 500  # Commonly used on VG side (number of frames for episode chunking)

    # ----------------------------------------------------------------------
    # 2) Visual input and sampling (image/camera)
    # ----------------------------------------------------------------------
    # Camera mapping
    cam_mapping: Dict[str, str] = field(
        default_factory=lambda: {
            "faceImg": "face_view",
            "leftImg": "left_wrist_view",
            "rightImg": "right_wrist_view",
        }
    )
    # Image and augmentation
    resolution: Dict[str, int] = field(
        default_factory=lambda: {
            "face_view": -1,
            "left_wrist_view": 128,
            "right_wrist_view": 128,
        }
    )
    cam_augmentation_list: List[str] = field(default_factory=list)

    # Image time series (history/future)
    image_horizon: int = 1
    image_history_length: int = 0
    image_history_interval: int = 1
    future_image_length: int = 0
    future_image_interval: int = 1
    future_image_indices: Optional[List[int]] = (
        None  # If provided, length must equal image_horizon
    )

    # Smart scaling
    max_pixels: int = field(
        default_factory=lambda: 1280 * 28 * 28
    )  # Will be replaced with MAX_PIXELS in __post_init__
    min_pixels: int = field(
        default_factory=lambda: 4 * 28 * 28
    )  # Will be replaced with MIN_PIXELS in __post_init__
    image_factor: int = 28  # Will be replaced with IMAGE_FACTOR in __post_init__

    # ----------------------------------------------------------------------
    # 3) Action and time series
    # ----------------------------------------------------------------------
    predict_action_keys: List[str] = field(default_factory=list)
    obs_action_keys: List[str] = field(default_factory=list)

    # Action window
    action_horizon: int = 21
    action_history_length: int = 0
    action_horizon_flow: int = 32
    action_horizon_ar: int = 0

    # Padding strategy
    left_padding: bool = True
    right_padding: bool = True

    # Dimension configuration
    dof_config: Dict[str, int] = field(default_factory=dict)  # Input degrees of freedom
    agent_pos_config: Dict[str, int] = field(default_factory=dict)  # Output degrees of freedom

    # State augmentation
    state_augmentation_ratio: float = 1.0  # Ratio of augmented states
    state_augmentation_prob: float = 0.1  # Random dimension masking probability for state string
    state_drop_prob: float = 0.0  # Probability of dropping entire state

    # ----------------------------------------------------------------------
    # 4) Instruction and multimodal
    # ----------------------------------------------------------------------
    default_instruction: str = ""
    instruction_path: Optional[str] = None
    instruction_key: Optional[List[Dict]] = None

    multimodal_chunk_size: int = 500
    generate_subtask_ratio: float = 0.0
    cot_ratio: float = 0.0
    multimodal_data_ratio: float = 0.25  # Multimodal data ratio per batch in VLA dataset
    instruction_key_prob: Optional[Dict[str, float]] = None
    trunc_action_with_instruction: bool = True
    use_embodied_system_prompt_ratio: float = 0.0

    # ----------------------------------------------------------------------
    # 5) Data cleaning and alignment (validation/augmentation/framework constraints)
    # ----------------------------------------------------------------------
    filter_angle_outliers: bool = False
    trim_stationary: bool = False
    use_state_string_representation: bool = False
    pad_prefix_to_same_length: bool = False
    put_ar_predict_in_postfix: bool = (
        False  # Whether to put ar prediction in postfix, set to True in prediction mode, False in training
    )
    pad_to_128_multiple: bool = False  # Triton Attention requirement (deprecated, always set to False)
    max_seqlen: int = 768
    model_type: Optional[str] = None  # qwen2_5, qwen2
    model_config_path: Optional[str] = None  # Model config path (used to derive PaddingSide)
    low_dim_obs_horizon: int = 1  # To be deprecated

    # ----------------------------------------------------------------------
    # Validation and post-processing
    # ----------------------------------------------------------------------
    def __post_init__(self):
        # TODO: Determine VGA model type validation here
        # assert self.model_type in ["qwen2_5", "qwen3"], f"Unsupported model type: {self.model_type}"

        if self.model_type == "qwen2_5":
            self.max_pixels = 16384 * 28 * 28
            self.min_pixels = 4 * 28 * 28
            self.image_factor = 28
        elif self.model_type == "qwen3":
            self.max_pixels = 16384 * 32 * 32
            self.min_pixels = 4 * 32 * 32
            self.image_factor = 32

        # Future image indices validation
        if (
            self.future_image_indices
            and len(self.future_image_indices) != self.image_horizon
        ):
            raise ValueError(
                f"future_image_indices length must equal image_horizon: "
                f"{len(self.future_image_indices)} != {self.image_horizon}"
            )

        # Auto-derive action window
        if self.action_horizon == 0:
            self.action_horizon = max(self.action_horizon_flow, self.action_horizon_ar)

        # Auto-derive action keys
        if not self.obs_action_keys:
            self.obs_action_keys = list(self.agent_pos_config.keys())
        if not self.predict_action_keys:
            self.predict_action_keys = list(self.dof_config.keys())

        # Derive PaddingSide
        # @Ryan: Only FlashAttention can use RightPadding, other AttnImpl use LeftPadding
        if self.model_config_path is not None:
            with open(self.model_config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            attn_impl = cfg["_attn_implementation"]

            if attn_impl == "flash_attention_2":
                self.padding_side = "right"
            else:
                self.padding_side = "left"

    # Convenience properties
    @property
    def use_6D_rotation(self) -> bool:
        """Whether to use 6D rotation (auto-determined from predict_action_keys)"""
        if hasattr(self, "_use_6D_rotation"):
            return self._use_6D_rotation
        self._use_6D_rotation = any("6D" in key for key in self.predict_action_keys)
        return self._use_6D_rotation

    @property
    def use_relative_action(self) -> bool:
        """Whether to use relative action (auto-determined from predict_action_keys)"""
        if hasattr(self, "_use_relative_action"):
            return self._use_relative_action
        self._use_relative_action = any(
            "relative" in key for key in self.predict_action_keys
        )
        return self._use_relative_action

    # ----------------------------------------------------------------------
    # YAML initialization
    # ----------------------------------------------------------------------
    @classmethod
    def from_yaml_dict(cls, yaml_dict: Dict[str, Any]) -> "X2RDataConfig":
        """
        Create config object from YAML config dict. Prioritizes data sub-config, then top-level fields.
        """
        data_config = yaml_dict.get("data", {})
        params: Dict[str, Any] = {}

        # 1) Data I/O and caching
        params.update(
            {
                "cache_dir": data_config.get(
                    "cache_dir", yaml_dict.get("cache_dir", "~/.cache/dataset_cache")
                ),
                "dataset_config_path": data_config.get(
                    "dataset_config_path", yaml_dict.get("dataset_config_path", None)
                ),
                "use_cache": data_config.get(
                    "use_cache", yaml_dict.get("use_cache", True)
                ),
                "check_mode": data_config.get(
                    "check_mode", yaml_dict.get("check_mode", True)
                ),
                "preload_size": data_config.get(
                    "preload_size", yaml_dict.get("preload_size", 128)
                ),
                "buffer_size": data_config.get(
                    "buffer_size", yaml_dict.get("buffer_size", 20000)
                ),
                "batch_size": data_config.get(
                    "batch_size",
                    yaml_dict.get(
                        "batch_size_per_gpu", yaml_dict.get("batch_size", 32)
                    ),
                ),
                "train_test_split": data_config.get("train_test_split", 0.9),
                "seed": yaml_dict.get("seed", 42),
                "episode_chunk_size": data_config.get("episode_chunk_size", 500),
            }
        )

        # 2) Visual input and sampling (image/camera)
        params.update(
            {
                "cam_mapping": data_config.get(
                    "cam_mapping",
                    {
                        "faceImg": "face_view",
                        "leftImg": "left_wrist_view",
                        "rightImg": "right_wrist_view",
                    },
                ),
                "resolution": data_config.get(
                    "resolution",
                    {"face_view": -1, "left_wrist_view": 128, "right_wrist_view": 128},
                ),
                "cam_augmentation_list": data_config.get("cam_augmentation_list", []),
                "image_horizon": data_config.get("image_horizon", 1),
                "image_history_length": data_config.get("image_history_length", 0),
                "image_history_interval": data_config.get("image_history_interval", 1),
                "future_image_length": data_config.get("future_image_length", 0),
                "future_image_interval": data_config.get("future_image_interval", 1),
                "future_image_indices": data_config.get("future_image_indices", None),
                "max_pixels": data_config.get("max_pixels", 1280 * 28 * 28),
                "min_pixels": data_config.get("min_pixels", 4 * 28 * 28),
                "image_factor": data_config.get("image_factor", 28),
            }
        )

        # 3) Action and time series
        params.update(
            {
                "predict_action_keys": data_config.get("predict_action_keys", []),
                "obs_action_keys": data_config.get("obs_action_keys", []),
                "action_horizon": data_config.get("action_horizon", 0),
                "action_history_length": data_config.get("action_history_length", 0),
                "action_horizon_flow": data_config.get(
                    "action_horizon_flow", yaml_dict.get("action_horizon_flow", 32)
                ),
                "action_horizon_ar": data_config.get("action_horizon_ar", 0),
                "left_padding": data_config.get("left_padding", True),
                "right_padding": data_config.get("right_padding", True),
                "dof_config": yaml_dict.get(
                    "dof_config", data_config.get("dof_config", {})
                ),
                "agent_pos_config": yaml_dict.get(
                    "agent_pos_config", data_config.get("agent_pos_config", {})
                ),
                "state_augmentation_prob": data_config.get(
                    "state_augmentation_prob", 0.05
                ),
                "state_drop_prob": data_config.get("state_drop_prob", 0.0),
            }
        )

        # 4) Instruction and multimodal
        params.update(
            {
                "default_instruction": data_config.get("default_instruction", ""),
                "instruction_path": data_config.get("instruction_path", None),
                "instruction_key": data_config.get("instruction_key", None),
                "multimodal_chunk_size": data_config.get("multimodal_chunk_size", 500),
                "generate_subtask_ratio": data_config.get(
                    "generate_subtask_ratio", 0.0
                ),
                "cot_ratio": data_config.get("cot_ratio", 0.0),
                "multimodal_data_ratio": data_config.get("multimodal_data_ratio", 0.25),
                "instruction_key_prob": data_config.get("instruction_key_prob", None),
                "trunc_action_with_instruction": data_config.get(
                    "trunc_action_with_instruction", True
                ),
                "use_embodied_system_prompt_ratio": data_config.get(
                    "use_embodied_system_prompt_ratio",
                    yaml_dict.get("use_embodied_system_prompt_ratio", 0.0),
                ),
            }
        )

        # 5) Data cleaning and alignment (validation/augmentation/framework constraints)
        params.update(
            {
                "filter_angle_outliers": data_config.get(
                    "filter_angle_outliers", False
                ),
                "trim_stationary": data_config.get("trim_stationary", False),
                "use_state_string_representation": data_config.get(
                    "use_state_string_representation",
                    yaml_dict.get("use_state_string_representation", False),
                ),
                "pad_prefix_to_same_length": data_config.get(
                    "pad_prefix_to_same_length", False
                ),
                "put_ar_predict_in_postfix": data_config.get(
                    "put_ar_predict_in_postfix", False
                ),
                # "pad_to_128_multiple": data_config.get("pad_to_128_multiple", True),
                "padding_side": data_config.get("padding_side", "left"),
                "max_seqlen": yaml_dict.get("max_seqlen", 768),
                "model_type": yaml_dict.get("model_type", "qwen2_5"),
                "model_config_path": yaml_dict.get("qwen_vl_act_config_path", None),
                "low_dim_obs_horizon": data_config.get("low_dim_obs_horizon", 1),
            }
        )

        # Only keep valid fields defined in dataclass
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in params.items() if k in valid_fields}
        return cls(**filtered)

    # ----------------------------------------------------------------------
    # Dict-style access (for compatibility with existing calls)
    # ----------------------------------------------------------------------
    def __getitem__(self, key: str):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"'{key}' not found in {self.__class__.__name__}")

    def __setitem__(self, key: str, value):
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()


class InferConfig:
    def __init__(
        self,
        checkpoint_path: str | None = None,
        train_config_path: str | None = None,
        robot_host: str = "0.0.0.0",
        robot_port: int = 33723,
        robot_id: str = "10053",
        robot_type: str = "desktop",  # ["desktop", "turtle"]
        robot_action_start_ratio: float = 0,  # Action execution start ratio
        robot_action_end_ratio: float = 0.8,  # Action execution end ratio
        robot_action_interpolate_multiplier: int = 70,  # Action interpolation
        robot_use_joint_angle_control: bool = False,  # Use joint control (model must be joint prediction model)
        turtle_as_desktop: bool = False,  # Use turtle body for desktop operation, fixed chassis head movement, head camera, and chassis height
        action_horizon: int = 10,  # Please correctly fill in the model's horizon
        action_dim: int | None = None,
        model_device: str = "cuda:0",
        num_inference_timesteps: int = 10,
        norm_key: str = "x2_normal",
        cam_names: list[str] = ["face_view", "right_wrist_view"],
    ):
        # Private attribute for storing path
        assert checkpoint_path is not None
        self._checkpoint_path = checkpoint_path
        if os.path.exists(os.path.join(checkpoint_path, "normalizer_action.pth")):
            self.normalizer_action_path = os.path.join(
                checkpoint_path, "normalizer_action.pth"
            )
        if os.path.exists(os.path.join(checkpoint_path, "normalizer_propri.pth")):
            self.normalizer_propri_path = os.path.join(
                checkpoint_path, "normalizer_propri.pth"
            )
        
        self.model_path=checkpoint_path
        self.action_tokenizer_path="/x2robot_v2/Models/fast/"


        # Other configuration attributes
        self.robot_host = robot_host
        self.robot_port = robot_port
        self.robot_type = robot_type  # ["desktop", "turtle"]
        self.robot_id = robot_id
        self.robot_action_start_ratio = robot_action_start_ratio
        self.robot_action_end_ratio = robot_action_end_ratio
        self.robot_action_interpolate_multiplier = robot_action_interpolate_multiplier
        self.robot_use_joint_angle_control = (
            robot_use_joint_angle_control  # Use joint angle control
        )
        self.turtle_as_desktop = turtle_as_desktop

        self._action_horizon = (
            action_horizon  # Default controlled by train config's flow action horizon
        )
        self._action_dim = action_dim  # Default determined by train config's dof config
        
        self.action_dim = action_dim
        self.pred_horizon = action_horizon
        self.predict_mode = "diffusion"
        self.camera_key = cam_names

        self.model_device = model_device
        self.num_inference_timesteps = (
            num_inference_timesteps  # flow matching related config
        )

        # Initialize config objects
        self.train_config: dict = {}
        self.model_config = None
        self.data_config = None
        self.norm_key = norm_key
        self.cam_names = cam_names
        # Load all configs
        self._load_all_configs(train_config_path)

    @property
    def checkpoint_path(self) -> str | None:
        return self._checkpoint_path

    @checkpoint_path.setter
    def checkpoint_path(self, value: str | None):
        """When checkpoint_path is updated, reload all configs"""
        if self._checkpoint_path != value:
            self._checkpoint_path = value
            self._load_all_configs()

    @property
    def action_horizon(self) -> int:
        return self._action_horizon

    @action_horizon.setter
    def action_horizon(self, value: int):
        self._action_horizon = value

    @property
    def action_dim(self) -> int | None:
        return self._action_dim

    @action_dim.setter
    def action_dim(self, value: int | None):
        self._action_dim = value

    def _load_all_configs(self, train_config_path=None):
        """Unified entry point for loading all configs"""
        self._load_train_config(train_config_path)
        self._load_model_config()
        self._load_data_config()

        # Update action_horizon and action_dim (if needed)
        if self._action_horizon is None:
            self._action_horizon = self.train_config.get("data", {}).get(
                "action_horizon_flow", 32
            )
        assert self._action_horizon is not None and self._action_horizon > 0

        if self._action_dim is None:
            self._action_dim = sum(self.train_config.get("dof_config", {}).values())

    def _load_train_config(self, train_config_path):
        if train_config_path is None:
            train_config_path = os.path.join(self._checkpoint_path, "config.yml")
        with open(train_config_path, "r") as f:
            self.train_config = yaml.load(f, Loader=yaml.FullLoader)

        ckpt_dir = self._checkpoint_path
        preprocessor_file = os.path.join(ckpt_dir, "preprocessor_config.json")
        if os.path.exists(preprocessor_file):
            print(f"[LoadConfig] Found {preprocessor_file}, override processor_path.")
            self.train_config["processor_path"] = ckpt_dir

        tokenizer_file = os.path.join(ckpt_dir, "tokenizer.json")
        tokenizer_config_file = os.path.join(ckpt_dir, "tokenizer_config.json")
        if "action_tokenizer_path" in self.train_config and not os.path.exists(
            self.train_config["action_tokenizer_path"]
        ):
            if os.path.exists(tokenizer_file) and os.path.exists(tokenizer_config_file):
                print(
                    f"[LoadConfig] Found tokenizer files in {ckpt_dir}, override action_tokenizer_path."
                )
                self.train_config["action_tokenizer_path"] = ckpt_dir
            else:
                print("[LoadConfig] Cannot load action tokenizer! ")

    def _load_model_config(self):
        ckpt_config_path = os.path.join(self._checkpoint_path, "config.json")
        resolved_cfg_path = None

        if os.path.exists(ckpt_config_path):
            # Prefer checkpoint config
            resolved_cfg_path = ckpt_config_path
            print(f"[LoadModelConfig] Using checkpoint config.json: {ckpt_config_path}")
        else:
            # Fallback to original config path
            fallback_cfg = self.train_config.get("qwen_vl_act_config_path", None)
            if fallback_cfg is not None:
                resolved_cfg_path = fallback_cfg
                print(f"[LoadModelConfig] Using fallback act config: {fallback_cfg}")

        if resolved_cfg_path is None or (not os.path.exists(resolved_cfg_path)):
            raise ValueError(
                f"[LoadModelConfig] Cannot load model config! "
                f"Checked:\n"
                f" - Checkpoint config.json: {ckpt_config_path}\n"
                f" - Fallback path: {self.train_config.get('qwen_vl_act_config_path', None)}"
            )

        # Save back to config for consistency
        self.train_config["qwen_vl_act_config_path"] = resolved_cfg_path

        model_type = self.train_config["model_type"]
        if model_type == "qwen2_5":
            from wall_x.model.qwen2_5_based import Qwen2_5_VLConfig

            ConfigClass = Qwen2_5_VLConfig

        # elif model_type == "qwen3":
        #     from wall_x.model.qwen3_based import Qwen3VLConfig

        #     ConfigClass = Qwen3VLConfig

        else:
            raise ValueError(f"[LoadModelConfig] Unsupported model type: {model_type}")

        print(f"[LoadModelConfig] Loading model config from: {resolved_cfg_path}")
        self.model_config = ConfigClass.from_pretrained(resolved_cfg_path)

        self.model_config = update_model_config(self.train_config, self.model_config)

        self.model_config._attn_implementation = "sdpa"
        self.model_config.vision_config._attn_implementation = "flash_attention_2"

        print("[LoadModelConfig] Model config loaded and updated successfully.")

    def _load_data_config(self):
        self.data_config = X2RDataConfig.from_yaml_dict(self.train_config)


if __name__ == "__main__":
    config = InferConfig()
    print(config.train_config)
