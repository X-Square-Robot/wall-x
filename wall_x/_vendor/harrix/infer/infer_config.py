import dataclasses
import logging
import os
from typing import Any

from wall_x._vendor.harrix.utils.dataset_v2 import (
    InferenceDataConfig,
    build_inference_data_config,
    merge_private_data_config,
    normalize_dataset_config_for_inference,
)

logger = logging.getLogger(__name__)

_SUPPORTED_ROBOT_TYPES = ("desktop", "turtle", "ex001")


def resolve_prompt_dataset_family(norm_key: str) -> str:
    """Return the dataset family used to build the inference prompt."""
    return norm_key


def resolve_embodiment_type(robot_type: str, norm_key: str) -> str | None:
    """Public builds carry no embodiment table; prompts stay generic."""
    del robot_type, norm_key
    return None


def validate_as_desktop_config(
    robot_type: str,
    turtle_as_desktop: bool,
    ex001_as_desktop: bool,
    fixed_head_pos,
    fixed_car_pose,
    fixed_base_velocity,
) -> tuple[
    tuple[float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """Validate and normalize fixed-target configuration for mobile robots."""
    if turtle_as_desktop and robot_type != "turtle":
        raise ValueError("turtle_as_desktop requires robot_type='turtle'")
    if ex001_as_desktop and robot_type != "ex001":
        raise ValueError("ex001_as_desktop requires robot_type='ex001'")
    if len(fixed_head_pos) != 2:
        raise ValueError("fixed_head_pos must contain [yaw, pitch]")
    if len(fixed_car_pose) != 3:
        raise ValueError("fixed_car_pose must contain [x, y, yaw]")
    if len(fixed_base_velocity) != 3:
        raise ValueError("fixed_base_velocity must contain [vx, vy, angular_velocity]")
    return (
        tuple(float(v) for v in fixed_head_pos),
        tuple(float(v) for v in fixed_car_pose),
        tuple(float(v) for v in fixed_base_velocity),
    )


def validate_robot_identity(
    robot_type: str, robot_id: str | int | None, norm_key: str
) -> str | None:
    """Validate ``robot_type``; ``robot_id`` is passed through unchanged."""
    del norm_key
    if robot_type not in _SUPPORTED_ROBOT_TYPES:
        supported = ", ".join(_SUPPORTED_ROBOT_TYPES)
        raise ValueError(
            f"Unsupported robot_type={robot_type!r}; expected one of: {supported}"
        )
    return None if robot_id is None else str(robot_id)


class InferConfig:
    def __init__(
        self,
        checkpoint_path: str | None = None,
        train_config_path: str | None = None,
        robot_host: str = "0.0.0.0",
        robot_port: int = 41776,
        robot_type: str = "desktop",  # ["desktop", "turtle", "ex001"]
        robot_action_start_ratio: float = 0,  # start ratio of the action chunk to execute
        robot_action_end_ratio: float = 0.8,  # end ratio of the action chunk to execute
        robot_action_interpolate_multiplier: int = 10,  # action interpolation
        robot_use_joint_angle_control: bool = False,  # use joint control (note: the model must be a joint-prediction model)
        turtle_as_desktop: bool = False,  # use the turtle embodiment for desktop manipulation; fixed base, head movement, head camera, and base height
        ex001_as_desktop: bool = False,
        fixed_head_pos: tuple[float, float] = (0.0, -1.0),
        fixed_lift: float = 0.4,
        fixed_car_pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
        fixed_base_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        action_horizon: (
            int | None
        ) = None,  # defaults to the flow horizon from the training config
        action_dim: int | None = None,
        ar_action_dim: int | None = None,
        model_device: str = "cuda:0",
        num_inference_timesteps: int = 10,
        norm_key: str = "x2_normal",  # ["x2_normal", "ex_normal"]
        # Optional end_effector_type for x2robot_v2 ckpts whose normalizer is keyed
        # by the composite "dataset_type/ee_type". When set, the normalizer key is resolved as
        # make_norm_key(norm_key, end_effector_type). Leave None for a single /
        # auto-resolvable end_effector_type or for lerobot / x2robot_v1.
        end_effector_type: str | None = None,
        # Optional serving-host override for Dataset V2 normalizer JSON.
        normalizer_stats_path: str | None = None,
        # Drop the discretized state string from the text prompt.  This keeps
        # the checkpoint's model topology unchanged and only affects online
        # text preprocessing.
        state_drop: bool = False,
        cam_names: list[str] = ["face_view", "left_wrist_view", "right_wrist_view"],
        save_video_dir: str = "./videos",
        robot_id: str | None = None,
        model_type: str = "wallx",
        artifact_path_remaps: dict[str, str] | None = None,
    ):
        # private attributes for storing paths
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

        # other config attributes
        self.robot_host = robot_host
        self.robot_port = robot_port
        self.robot_type = robot_type  # ["desktop", "turtle", "ex001"]
        self.robot_action_start_ratio = robot_action_start_ratio
        self.robot_action_end_ratio = robot_action_end_ratio
        self.robot_action_interpolate_multiplier = robot_action_interpolate_multiplier
        self.robot_use_joint_angle_control = (
            robot_use_joint_angle_control  # use joint angle control
        )
        self.turtle_as_desktop = turtle_as_desktop
        self.ex001_as_desktop = ex001_as_desktop
        (
            self.fixed_head_pos,
            self.fixed_car_pose,
            self.fixed_base_velocity,
        ) = validate_as_desktop_config(
            robot_type,
            turtle_as_desktop,
            ex001_as_desktop,
            fixed_head_pos,
            fixed_car_pose,
            fixed_base_velocity,
        )
        self.fixed_lift = float(fixed_lift)
        self.robot_id = robot_id
        self._action_horizon = action_horizon
        self._action_dim = action_dim  # defaults to the dof_config in the train config
        self.ar_action_dim = (
            ar_action_dim  # defaults to the ar_dof_config in the train config
        )
        self.model_device = model_device
        self.num_inference_timesteps = (
            num_inference_timesteps  # flow matching related config
        )
        self.save_video_dir = save_video_dir
        self.model_type = model_type
        self.artifact_path_remaps = dict(artifact_path_remaps or {})
        # initialize config objects
        # cfg: typed TrainConfig, the single source of truth.
        self.cfg = None
        self.train_config: dict = {}
        self.model_config = None
        self.data_config = None
        self.dataset_config_source = "legacy"
        self._dataset_processors: dict[str, Any] = {}
        self.norm_key = norm_key
        self.end_effector_type = end_effector_type
        self.normalizer_stats_path = normalizer_stats_path
        self.state_drop = bool(state_drop)
        self.cam_names = cam_names
        self.robot_id = validate_robot_identity(
            self.robot_type, self.robot_id, self.norm_key
        )
        self.embodiment_type = resolve_embodiment_type(
            self.robot_type, self.norm_key
        )
        # load all configs
        self._load_all_configs(train_config_path)

    @property
    def checkpoint_path(self) -> str | None:
        return self._checkpoint_path

    @checkpoint_path.setter
    def checkpoint_path(self, value: str | None):
        """Reload all configs when checkpoint_path is updated."""
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
        """Unified entry point for loading all configs."""
        self._load_train_config(train_config_path)
        self._load_model_config_for_wallx()
        self._load_data_config()

        # update action_horizon and action_dim (if needed)
        if self._action_horizon is None:
            data = self.train_config.get("data", {})
            self._action_horizon = data.get(
                "action_horizon_flow",
                self.train_config.get("action_horizon_flow", 32),
            )
        assert self._action_horizon is not None and self._action_horizon > 0

        if self._action_dim is None:
            dof_config = self.train_config.get("dof_config") or self.train_config.get(
                "data", {}
            ).get("dof_config", {})
            self._action_dim = sum(dof_config.values())

        if self._ar_action_dim is None:
            self._ar_action_dim = sum(
                (self.cfg.task.ar_dof_config or {}).values()
            )

    def _load_train_config(self, train_config_path):
        """Load through the single wall_x loader.

        ``self.cfg`` is the typed TrainConfig (schema-validated, checkpoint
        processor/tokenizer overlays applied, data backend registered);
        ``self.train_config`` is its dict view with the full task contract
        spread at the top level (training-side ``build_data_loader_dict``
        shape), kept for the legacy dict consumers in the serving stack.
        """
        from wall_x.config.loader import dump_config, load_train_config_from_checkpoint

        self.cfg = load_train_config_from_checkpoint(
            self._checkpoint_path, train_config_path
        )
        self.train_config = dump_config(self.cfg)
        # ``dump_config`` intentionally omits TrainConfig's private raw fields,
        # but Dataset V2 stores its fully materialized processor contract in
        # ``_raw_data.dataset_config_effective``. Preserve it for serving and
        # let typed/materialized data fields win only at the same level.
        merge_private_data_config(self.cfg, self.train_config)
        self.train_config.update(dataclasses.asdict(self.cfg.task))

        self._normalize_refactor_train_config_for_inference()

    def _normalize_refactor_train_config_for_inference(self) -> None:
        """Expose refactor-schema fields expected by the legacy infer stack."""
        model = self.train_config.get("model") or {}
        for key in (
            "processor_path",
            "pretrained_path",
            "action_tokenizer_type",
            "action_tokenizer_path",
            "action_tokenizer_checkpoint_path",
            "action_tokenizer_config_dir",
            "attn_deterministic",
            "flow_loss_weight",
            "use_ema",
        ):
            if key in model and self.train_config.get(key) is None:
                self.train_config[key] = model[key]
        if (
            self.train_config.get("qwen_vl_act_config_path") is None
            and model.get("config_path") is not None
        ):
            self.train_config["qwen_vl_act_config_path"] = model["config_path"]

        self._resolve_model_artifact_paths()

        # Mirror task keys into data: (training-side build_data_loader_dict
        # shape) -- update_model_config reads e.g. data.action_horizon_flow /
        # data.use_state_string_representation.
        data = self.train_config.setdefault("data", {})
        for key in (
            "dof_config",
            "agent_pos_config",
            "action_horizon",
            "action_horizon_flow",
            "ar_dof_config",
            "use_state_string_representation",
        ):
            value = self.train_config.get(key)
            if value is not None:
                data.setdefault(key, value)

        self._normalize_dataset_config_for_inference(data)

    def _normalize_dataset_config_for_inference(self, data: dict[str, Any]) -> None:
        (
            self._dataset_processors,
            self.dataset_config_source,
        ) = normalize_dataset_config_for_inference(self.train_config)

        logger.info(
            "[InferConfig] Dataset processors loaded from %s: text=%s vision=%s "
            "epilogue=%s",
            self.dataset_config_source,
            (self._dataset_processors.get("text") or {}).get("type"),
            (self._dataset_processors.get("vision") or {}).get("type"),
            (self._dataset_processors.get("epilogue") or {}).get("type"),
        )

    def _resolve_model_artifact_paths(self) -> None:
        """Relocate stale training-machine paths without changing valid paths."""
        model = self.train_config.get("model") or {}
        known_remaps = dict(self.artifact_path_remaps)

        for key in (
            "action_tokenizer_path",
            "action_tokenizer_checkpoint_path",
            "action_tokenizer_config_dir",
        ):
            original = self.train_config.get(key) or model.get(key)
            if not original or os.path.exists(original):
                continue
            resolved = None
            for old_prefix, new_prefix in known_remaps.items():
                if original == old_prefix or original.startswith(old_prefix + "/"):
                    candidate = new_prefix + original[len(old_prefix) :]
                    if os.path.exists(candidate):
                        resolved = candidate
                        break
            if resolved is None:
                continue
            logger.warning(
                "[InferConfig] Relocated %s from %s to %s",
                key,
                original,
                resolved,
            )
            self.train_config[key] = resolved
            model[key] = resolved
            if hasattr(self.cfg.model, key):
                setattr(self.cfg.model, key, resolved)

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

        from wall_x.trainer.adapters import ADAPTER_REGISTRY, format_adapter_error

        adapter_cls = ADAPTER_REGISTRY.get(model_type)
        if adapter_cls is None:
            raise ValueError(f"[LoadModelConfig] {format_adapter_error(model_type)}")
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
        self.model_config.vision_config._attn_implementation = "flash_attention_2"

        logger.info("[LoadModelConfig] Model config loaded and updated successfully.")

    def _load_data_config(self):
        self.data_config = self._build_inference_data_config()
        # Public configs declare the delta frame as data.relative_action_frame.
        frame = (self.train_config.get("data") or {}).get("relative_action_frame")
        if frame is not None:
            self.data_config.action_processor_params.setdefault(
                "relative_action_frame", frame
            )

        self._ensure_predict_action_keys()

    def _ensure_predict_action_keys(self) -> None:
        dof_keys = list((self.train_config.get("dof_config") or {}).keys())
        obs_keys = list((self.train_config.get("agent_pos_config") or {}).keys())
        if isinstance(self.data_config, InferenceDataConfig):
            if not self.data_config.predict_action_keys:
                self.data_config.predict_action_keys = dof_keys
            if not self.data_config.obs_action_keys:
                self.data_config.obs_action_keys = obs_keys
        elif isinstance(self.data_config, dict):
            self.data_config.setdefault("predict_action_keys", dof_keys)
            self.data_config.setdefault("obs_action_keys", obs_keys)

    def _build_inference_data_config(self) -> InferenceDataConfig:
        return build_inference_data_config(
            self.train_config,
            self._dataset_processors,
            self.dataset_config_source,
        )


if __name__ == "__main__":
    config = InferConfig()
    logger.info("%s", config.train_config)
