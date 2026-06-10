import logging
import os
from dataclasses import dataclass, field
from typing import Any

import yaml
from qwen_vl_utils.vision_process import IMAGE_FACTOR, MAX_PIXELS, MIN_PIXELS

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class InferenceDataConfig:
    """Minimal data config needed by online inference.

    Serving does not build datasets, so it should not require closed-source
    data backends just to read image resize settings from a checkpoint yaml.
    """

    resolution: dict[str, int] = field(default_factory=dict)
    model_type: str = "qwen2_5"
    max_pixels: int = MAX_PIXELS
    min_pixels: int = MIN_PIXELS
    image_factor: int = IMAGE_FACTOR
    use_relative_action: bool = False
    predict_action_keys: list[str] = field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)


class InferConfig:
    def __init__(
        self,
        checkpoint_path: str | None = None,
        train_config_path: str | None = None,
        robot_host: str = "0.0.0.0",
        robot_port: int = 41776,
        robot_type: str = "desktop",  # ["desktop", "turtle", "ex001"]
        robot_action_start_ratio: float = 0,  # Start ratio for trimming executed actions
        robot_action_end_ratio: float = 0.8,  # End ratio for trimming executed actions
        robot_action_interpolate_multiplier: int = 10,  # Action interpolation multiplier
        robot_use_joint_angle_control: bool = False,  # Joint control (model must predict joints)
        turtle_as_desktop: bool = False,  # Use turtle as desktop: fixed base/head motion, head camera, base height
        action_horizon: int = 32,  # Set to the model's action horizon
        action_dim: int | None = None,
        ar_action_dim: int | None = None,
        model_device: str = "cuda:0",
        num_inference_timesteps: int = 10,
        num_inference_steps: int | None = None,
        cfg_scale: float | None = None,
        seed: int | None = None,
        norm_key: str = "x2_normal",  # ["x2_normal", "ex_normal"]
        cam_names: list[str] | None = None,
        camera_front_key: str = "camera_front",
        camera_left_key: str = "camera_left",
        camera_right_key: str = "camera_right",
        default_instruction: str | None = None,
        prompt_template: str | None = None,
        qwen25_prompt_template: str | None = None,
        prompt_priority_order: str | None = None,
        save_video_dir: str = "./videos",
        robot_id: str = "10000",
        model_type: str = "wallx",  # ["wallx", "vga"]
        smooth_action: bool = False,
        smooth_gripper: bool = True,
    ):
        # Private attributes for paths
        assert checkpoint_path is not None
        from wall_x._vendor.harrix.utils.ckpt_load import resolve_checkpoint_dir

        if not os.path.isdir(checkpoint_path) and not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint path not found: {checkpoint_path!r}. "
                "Serving requires a directory containing model.safetensors and "
                "normalizer_action.pth / normalizer_propri.pth (or norm_stats.json)."
            )
        self._checkpoint_path = resolve_checkpoint_dir(checkpoint_path)
        action_pth = os.path.join(self._checkpoint_path, "normalizer_action.pth")
        propri_pth = os.path.join(self._checkpoint_path, "normalizer_propri.pth")
        if os.path.exists(action_pth):
            self.normalizer_action_path = action_pth
        if os.path.exists(propri_pth):
            self.normalizer_propri_path = propri_pth

        # Other config attributes
        self.robot_host = robot_host
        self.robot_port = robot_port
        self.robot_type = robot_type  # ["desktop", "turtle", "ex001"]
        self.robot_action_start_ratio = robot_action_start_ratio
        self.robot_action_end_ratio = robot_action_end_ratio
        self.robot_action_interpolate_multiplier = robot_action_interpolate_multiplier
        self.robot_use_joint_angle_control = (
            robot_use_joint_angle_control  # Joint-angle control
        )
        self.turtle_as_desktop = turtle_as_desktop
        self.robot_id = robot_id
        self._action_horizon = (
            action_horizon  # Default from train config flow action horizon
        )
        self._action_dim = action_dim  # Default from train config dof config
        self.ar_action_dim = ar_action_dim  # Default from train config ar dof config
        self.model_device = model_device
        self.num_inference_timesteps = (
            num_inference_timesteps  # flow matching related config
        )
        self.num_inference_steps = num_inference_steps
        self.cfg_scale = cfg_scale
        self.seed = seed
        self.save_video_dir = save_video_dir
        self.model_type = model_type  # ["wallx", "vga"]
        # Initialize config objects
        self.train_config: dict = {}
        self.model_config = None
        self.data_config = None
        self.norm_key = norm_key
        self.cam_names = cam_names or [
            "face_view",
            "left_wrist_view",
            "right_wrist_view",
        ]
        self.camera_front_key = camera_front_key
        self.camera_left_key = camera_left_key
        self.camera_right_key = camera_right_key
        self.default_instruction = default_instruction
        self.prompt_template = prompt_template
        self.qwen25_prompt_template = qwen25_prompt_template
        self.prompt_priority_order = prompt_priority_order
        self.smooth_action = _env_bool("WALLX_SMOOTH_ACTION", smooth_action)
        self.smooth_gripper = _env_bool("WALLX_SMOOTH_GRIPPER", smooth_gripper)
        # Load all configs
        self._load_all_configs(train_config_path)
        self._apply_cam_names_from_train_config(cam_names)

    def _apply_cam_names_from_train_config(
        self, cam_names: list[str] | None
    ) -> None:
        """Use train YAML camera mapping when CLI did not override cam_names."""
        if cam_names is not None:
            return
        from wall_x._vendor.harrix.utils.train_config import (
            resolve_cam_names_from_train_config,
        )

        resolved = resolve_cam_names_from_train_config(self.train_config)
        if resolved:
            self.cam_names = resolved
            logger.info(
                "[InferConfig] cam_names from train key_mappings: %s",
                self.cam_names,
            )

    @property
    def checkpoint_path(self) -> str | None:
        return self._checkpoint_path

    @checkpoint_path.setter
    def checkpoint_path(self, value: str | None):
        """Reload all configs when checkpoint_path is updated"""
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

    @property
    def ar_action_dim(self) -> int | None:
        return self._ar_action_dim

    @ar_action_dim.setter
    def ar_action_dim(self, value: int | None):
        self._ar_action_dim = value

    def _load_all_configs(self, train_config_path=None):
        """Unified entry to load all configs"""
        self._load_train_config(train_config_path)
        if self.model_type != "vga":
            self._load_model_config_for_wallx()
        self._load_data_config()

        # Update action_horizon and action_dim if needed
        if self._action_horizon is None:
            self._action_horizon = self.train_config.get("data", {}).get(
                "action_horizon_flow", 32
            )
        assert self._action_horizon is not None and self._action_horizon > 0

        if self._action_dim is None:
            dof_config = (
                self.train_config.get("dof_config")
                or self.train_config.get("task", {}).get("dof_config")
                or self.train_config.get("data", {}).get("dof_config", {})
            )
            self._action_dim = sum(dof_config.values())

        if self._ar_action_dim is None:
            ar_dof_config = self.train_config.get("ar_dof_config") or self.train_config.get(
                "task", {}
            ).get("ar_dof_config", {})
            self._ar_action_dim = sum(ar_dof_config.values())

    def _load_train_config(self, train_config_path):
        if train_config_path is None:
            for fname in ("config.yml", "config.yaml"):
                candidate = os.path.join(self._checkpoint_path, fname)
                if os.path.exists(candidate):
                    train_config_path = candidate
                    break
            else:
                raise FileNotFoundError(
                    f"No config.yml/config.yaml found in {self._checkpoint_path}"
                )
        with open(train_config_path, "r") as f:
            self.train_config = yaml.load(f, Loader=yaml.FullLoader)

        ckpt_dir = self._checkpoint_path
        preprocessor_file = os.path.join(ckpt_dir, "preprocessor_config.json")
        if os.path.exists(preprocessor_file):
            logger.info(
                "[LoadConfig] Found %s, override processor_path.",
                preprocessor_file,
            )
            self.train_config["processor_path"] = ckpt_dir

        tokenizer_file = os.path.join(ckpt_dir, "tokenizer.json")
        tokenizer_config_file = os.path.join(ckpt_dir, "tokenizer_config.json")
        if self.train_config.get(
            "action_tokenizer_path", None
        ) is not None and not os.path.exists(
            self.train_config.get("action_tokenizer_path", None)
        ):
            if os.path.exists(tokenizer_file) and os.path.exists(tokenizer_config_file):
                logger.info(
                    "[LoadConfig] Found tokenizer files in %s, override action_tokenizer_path.",
                    ckpt_dir,
                )
                self.train_config["action_tokenizer_path"] = ckpt_dir
            else:
                logger.warning("[LoadConfig] Cannot load action tokenizer! ")

        from wall_x._vendor.harrix.utils.train_config import (
            normalize_train_config_for_inference,
            strip_action_tokenizer_fields,
        )

        self.train_config = normalize_train_config_for_inference(
            self.train_config, train_config_path
        )
        # Flow serving does not load AR action tokenizers; crop embed_tokens instead.
        strip_action_tokenizer_fields(self.train_config)
        self._train_config_path = train_config_path

    def _load_model_config_for_wallx(self):
        model_type = self.train_config["model_type"]
        if not os.path.isdir(self._checkpoint_path):
            return
        # For Qwen models
        ckpt_config_path = os.path.join(self._checkpoint_path, "config.json")
        resolved_cfg_path = None

        if os.path.exists(ckpt_config_path):
            # Prefer checkpoint config
            resolved_cfg_path = ckpt_config_path
            logger.info(
                "[LoadModelConfig] Using checkpoint config.json: %s",
                ckpt_config_path,
            )
        else:
            # Fallback to original config path
            fallback_cfg = self.train_config.get("qwen_vl_act_config_path", None)
            if fallback_cfg is not None:
                resolved_cfg_path = fallback_cfg
                logger.info(
                    "[LoadModelConfig] Using fallback act config: %s",
                    fallback_cfg,
                )

        if resolved_cfg_path is None or (not os.path.exists(resolved_cfg_path)):
            raise ValueError(
                f"[LoadModelConfig] Cannot load model config! "
                f"Checked:\n"
                f" - Checkpoint config.json: {ckpt_config_path}\n"
                f" - Fallback path: {self.train_config.get('qwen_vl_act_config_path', None)}"
            )

        # Save back to config for consistency
        self.train_config["qwen_vl_act_config_path"] = resolved_cfg_path

        from wall_x.trainer.adapters import resolve_adapter

        adapter_cls = resolve_adapter(model_type)
        ConfigClass = adapter_cls.config_class()

        logger.info(
            "[LoadModelConfig] Loading model config from: %s", resolved_cfg_path
        )
        if resolved_cfg_path.endswith(".json"):
            self.model_config = ConfigClass.from_json_file(resolved_cfg_path)
        else:
            self.model_config = ConfigClass.from_pretrained(resolved_cfg_path)

        self.model_config.update_model_config(self.train_config)

        self.model_config._attn_implementation = "sdpa"
        vision_attn = os.environ.get("WALLX_VISION_ATTN_IMPLEMENTATION", "flash_attention_2")
        self.model_config.vision_config._attn_implementation = vision_attn
        logger.info(
            "[LoadModelConfig] vision _attn_implementation=%s (override via WALLX_VISION_ATTN_IMPLEMENTATION)",
            vision_attn,
        )

        logger.info("[LoadModelConfig] Model config loaded and updated successfully.")

    def _load_data_config(self):
        # Prefer typed TrainConfig path (handles new 8-section schema where
        # dof_config lives under task:). Fall back to legacy raw-dict path
        # for old flat yamls. See trainer/adapters/base_adapter.py.
        typed_dcfg = self._try_typed_data_config()
        if typed_dcfg is not None:
            self.data_config = typed_dcfg
        elif self.model_type != "vga":
            self.data_config = self._build_inference_data_config()
        else:
            # TEMPORARY: direct call to the private _set_data_backend.
            # VGA inference still uses the legacy x2robot data config object.
            # Wall-X online inference uses the lightweight InferenceDataConfig
            # above so serving does not require excluded data backends.
            from wall_x.data._registry import _set_data_backend

            _set_data_backend(self.train_config.get("dataset_type", "x2robot_v1"))
            from wall_x.model.vga.openloop_visualization import get_data_configs

            dataload_config = get_data_configs(self.train_config.get("data", {}))
            dataload_config["predict_action_keys"] = list(
                dataload_config.get("dof_config", {}).keys()
            )
            self.data_config = dataload_config

        self._ensure_predict_action_keys()

    def _ensure_predict_action_keys(self) -> None:
        dof_keys = list((self.train_config.get("dof_config") or {}).keys())
        if not dof_keys:
            return
        if isinstance(self.data_config, InferenceDataConfig):
            if not self.data_config.predict_action_keys:
                self.data_config.predict_action_keys = dof_keys
        elif isinstance(self.data_config, dict):
            self.data_config.setdefault("predict_action_keys", dof_keys)

    def _build_inference_data_config(self) -> InferenceDataConfig:
        data = self.train_config.get("data", {}) or {}

        def get(key: str, default: Any) -> Any:
            return data.get(key, self.train_config.get(key, default))

        resolution = get("resolution", {}) or {}
        dof_config = self.train_config.get("dof_config") or {}
        return InferenceDataConfig(
            resolution=dict(resolution),
            model_type=get(
                "model_type", self.train_config.get("model_type", "qwen2_5")
            ),
            max_pixels=get("max_pixels", MAX_PIXELS),
            min_pixels=get("min_pixels", MIN_PIXELS),
            image_factor=get("image_factor", IMAGE_FACTOR),
            use_relative_action=get("use_relative_action", False),
            predict_action_keys=list(dof_config.keys()),
        )

    def _try_typed_data_config(self):
        """Attempt to load TrainConfig and build X2RDataConfig via typed path.

        Returns the X2RDataConfig on success, or None if the yaml is not in
        TrainConfig schema (legacy flat yaml).
        """
        yml_path = getattr(self, "_train_config_path", None)
        if yml_path is None:
            for fname in ("config.yml", "config.yaml"):
                candidate = os.path.join(self._checkpoint_path, fname)
                if os.path.exists(candidate):
                    yml_path = candidate
                    break
        if yml_path is None:
            return None
        try:
            from wall_x.config.loader import load_config
            from wall_x.trainer.adapters.base_adapter import load_trainer_data_config

            typed_cfg = load_config(yml_path)
        except Exception as e:
            logger.warning(
                "[InferConfig] Typed config load failed for %s: %s",
                yml_path,
                e,
            )
            return None
        try:
            return load_trainer_data_config(typed_cfg)
        except Exception as e:
            logger.warning(
                "[InferConfig] Typed config load failed for %s: %s",
                yml_path,
                e,
            )
            return None


if __name__ == "__main__":
    config = InferConfig()
    logger.info("%s", config.train_config)
