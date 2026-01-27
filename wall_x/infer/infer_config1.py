import yaml
import os
from wall_x.trainer.trainer_utils import update_model_config
from x2robot_dataset.configs.config import X2RDataConfig


class InferConfig:
    def __init__(
        self,
        checkpoint_path: str | None = None,
        train_config_path: str | None = None,
        robot_host: str = "0.0.0.0",
        robot_port: int = 33723,
        robot_type: str = "desktop",  # ["desktop", "turtle"]
        robot_action_start_ratio: float = 0,  # 截取动作执行的开始比例
        robot_action_end_ratio: float = 0.8,  # 截取动作执行的最终比例
        robot_action_interpolate_multiplier: int = 70,  # 动作插值
        robot_use_joint_angle_control: bool = False,  # 使用关节控制（注意这时候模型需要是关节预测模型）
        turtle_as_desktop: bool = False,  # 使用乌龟本体做桌面操作，固定底盘头移动，头部相机，和底盘高度
        action_horizon: int = 32,  # 请正确填写模型的horizon
        action_dim: int | None = None,
        model_device: str = "cuda:0",
        num_inference_timesteps: int = 10,
        norm_key: str = "x2_normal",
        cam_names: list[str] = ["face_view", "left_wrist_view", "right_wrist_view"],
        save_video_dir: str = "./videos",
        robot_id: str = "10053",
    ):
        # 私有属性用于存储路径
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

        # 其他配置属性
        self.robot_host = robot_host
        self.robot_port = robot_port
        self.robot_type = robot_type  # ["desktop", "turtle"]
        self.robot_id = robot_id
        self.robot_action_start_ratio = robot_action_start_ratio
        self.robot_action_end_ratio = robot_action_end_ratio
        self.robot_action_interpolate_multiplier = robot_action_interpolate_multiplier
        self.robot_use_joint_angle_control = (
            robot_use_joint_angle_control  # 使用关节角度控制
        )
        self.turtle_as_desktop = turtle_as_desktop
        self.robot_id = robot_id
        self._action_horizon = (
            action_horizon  # 默认由train config 的flow action horizon控制
        )
        self._action_dim = action_dim  # 默认由train config 的dof config决定

        self.model_device = model_device
        self.num_inference_timesteps = (
            num_inference_timesteps  # flow matching related config
        )
        self.save_video_dir = save_video_dir
        # 初始化配置对象
        self.train_config: dict = {}
        self.model_config = None
        self.data_config = None
        self.norm_key = norm_key
        self.cam_names = cam_names
        # 加载所有配置
        self._load_all_configs(train_config_path)

    @property
    def checkpoint_path(self) -> str | None:
        return self._checkpoint_path

    @checkpoint_path.setter
    def checkpoint_path(self, value: str | None):
        """当checkpoint_path更新时，重新加载所有配置"""
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
        """加载所有配置的统一入口"""
        self._load_train_config(train_config_path)
        self._load_model_config()
        self._load_data_config()

        # 更新 action_horizon 和 action_dim（如果需要）
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
        if self._checkpoint_path.endswith(".safetensors"):
            print(
                "[LoadModelConfig] Model is a safetensors file. Model config will NOT be loaded from the checkpoint path."
            )
            return
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

        elif model_type == "qwen3":
            from wall_x.model.qwen3_based import Qwen3VLConfig

            ConfigClass = Qwen3VLConfig

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
