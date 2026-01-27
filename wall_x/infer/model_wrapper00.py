import os
import torch
import json
import copy
from safetensors.torch import load_file
import numpy as np
from PIL import Image
from qwen_vl_utils.vision_process import smart_resize
from transformers import BatchFeature

from wall_x.trainer.trainer_utils import load_wallx_processors, update_model_config
from x2robot_dataset.utils.preprocessing import preprocesser_call
from x2robot_dataset.utils.text_utils import get_prologue_with_embodied_information
from x2robot_dataset.constant import _CAM_NAME_MAPPING
from x2robot_dataset.utils.grounding_utils import (
    reverse_grounding_points,
    extract_grounding_points,
)

from wall_x.infer.infer_config import InferConfig
from wall_x.model.action_head import Normalizer
from wall_x.utils.constant import action_statistic_dof as default_action_statistic_dof
from wall_x.infer.logger import InferLogger
from wall_x.utils.timers import timer, ScopeTimer
from workspace.weight_convert.weight_loader import WeightLoader

ENABLE_FAST_PREPROCESS = os.getenv("ENABLE_FAST_PREPROCESS", "False").lower() == "true"
ENABLE_EXPERIMENTAL_INFERENCE_ENGINE = (
    os.getenv("ENABLE_EXPERIMENTAL_INFERENCE_ENGINE", "False").lower() == "true"
)


def move_to_cuda(obj, device="cuda"):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, (dict, BatchFeature)):
        return {k: move_to_cuda(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_cuda(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_cuda(v, device) for v in obj)
    else:
        return obj


class WallxModelWrapper:
    def __init__(self, config: InferConfig):
        self.config = config
        self.logger = InferLogger.get_model_logger("WallxModelWrapper")
        self.norm_key = self.config.norm_key
        self._register_normalizers()
        self.logger.info(f"normalizers {self.norm_key} 注册完成")
        self._load_processor()
        self._load_model()
        self.load_ckpt()
        self.logger.info(f"模型 {self.config.checkpoint_path} 加载完成")

        self.role_start_symbol = "<|im_start|>"
        self.role_end_symbol = "<|im_end|>"
        self.vision_start_symbol = "<|vision_start|>"
        self.vision_end_symbol = "<|vision_end|>"
        self.image_pad_symbol = "<|image_pad|>"
        self.propri_symbol = "<|propri|>"
        self.action_symbol = "<|action|>"

        self.cam_names = self.config.cam_names

    def _load_processor(self):
        processors_dict = load_wallx_processors(self.config.train_config)
        self.processor = processors_dict["processor"]
        self.action_tokenizer = processors_dict["train_action_tokenizer"]
        self.action_mapper = processors_dict["action_mapper"]

    def _load_model(self):
        model_type = self.config.train_config["model_type"]
        if model_type == "qwen2_5":
            from wall_x.model.qwen2_5_based import Qwen2_5_VLMoEForAction

            ModelClass = Qwen2_5_VLMoEForAction
            self.ModelClass = Qwen2_5_VLMoEForAction
        elif model_type == "qwen3":
            from wall_x.model.qwen3_based import Qwen3_VLMoEForAction

            ModelClass = Qwen3_VLMoEForAction
            self.ModelClass = Qwen3_VLMoEForAction
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.logger.info(f"初始化模型: {model_type}")
        self.config.model_config = update_model_config(
            self.config.train_config, self.config.model_config
        )
        self.model = ModelClass(
            self.config.model_config,
            self.config.train_config.get("action_tokenizer_type", None),
            self.processor,
            self.action_tokenizer,
            self.action_mapper,
        )

        # log attention implementation
        if model_type == "qwen2_5":
            self.logger.info(
                f"model attention implementation: {self.model.model._attn_implementation}"
            )
            self.logger.info(
                f"model.visual attention implementation: {self.model.visual.config._attn_implementation}"
            )
        elif model_type == "qwen3":
            self.logger.info(
                f"model attention implementation: {self.model.model.language_model._attn_implementation}"
            )
            self.logger.info(
                f"model.visual attention implementation: {self.model.model.visual.config._attn_implementation}"
            )

        # self.logger.info(f"加载qwen预训练权重")
        # if self.config.train_config.get("pretrained_qwen_vl_path", None):
        #     model, err = load_qwen_pretrain_weight(
        #         self.model, self.config.train_config["pretrained_qwen_vl_path"]
        #     )
        #     self.logger.info(f"load qwen pretrain weight errors: {err}")

        self.logger.info("调整模型token embedding")
        self.model.resize_token_embeddings(len(self.processor.tokenizer))
        self.logger.info("调整模型token embedding完成")
        self.logger.info("调整部分模型参数为bfloat16")
        self.model.to_bfloat16_for_selected_params()
        self.logger.info("调整部分模型参数为bfloat16完成")

    def load_ckpt(self, checkpoint_path: str = None):
        if checkpoint_path is None:
            checkpoint_path = self.config.checkpoint_path

        if os.path.exists(os.path.join(checkpoint_path, "global_step.pth")):
            global_step = torch.load(os.path.join(checkpoint_path, "global_step.pth"))[
                "global_step"
            ]
            self.logger.info(f"读取ckpt的训练步数为: {global_step}")

        fsdp_ckpt = os.path.join(checkpoint_path, "pytorch_model_fsdp.bin")
        safetensor_ckpt = os.path.join(checkpoint_path, "model.safetensors")
        if os.path.exists(fsdp_ckpt):
            self.logger.info(f"Loading FSDP checkpoint: {fsdp_ckpt}")
            state_dict = torch.load(fsdp_ckpt, map_location="cpu")
            # 若最外层包了一层 (例如 {'state_dict': {...}})
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                self.logger.info(
                    "Using nested state_dict inside pytorch_model_fsdp.bin"
                )
                state_dict = state_dict["state_dict"]
        elif os.path.exists(safetensor_ckpt):
            self.logger.info(f"读取ckpt的safetensors权重: {safetensor_ckpt}")
            state_dict = load_file(safetensor_ckpt, device="cpu")
        else:
            raise FileNotFoundError(
                f"❌ No checkpoint found under {checkpoint_path}. "
                "Expecting either pytorch_model_fsdp.bin or model.safetensors."
            )

        if not self.ModelClass.is_fused(state_dict):
            self.logger.info(
                "Converting non-fused weights to fused format...",
            )
            state_dict = self.ModelClass.convert_to_fused(state_dict)
        else:
            self.logger.info(
                "The weights is fused, skipping conversion.",
            )

        msg = self.model.load_state_dict(state_dict, strict=False)
        self.state_dict = state_dict
        self.model.set_normalizer(
            copy.deepcopy(self.normalizer_action), copy.deepcopy(self.normalizer_propri)
        )
        self.logger.info(f"load_state_dict result: {msg}")
        self.model.eval()
        self.model.to(self.config.model_device)
        self.model.to_bfloat16_for_selected_params()

        if ENABLE_EXPERIMENTAL_INFERENCE_ENGINE:
            loader = WeightLoader(
                source_path=safetensor_ckpt,
                expert_index=1,  # Use expert 1 as specified
                verbose=True,
            )
            prefill_loader = WeightLoader(
                source_path=safetensor_ckpt,
                expert_index=0,  # Use expert 0 as specified
                verbose=True,
            )
            self.model.converted_weights = loader.convert()
            self.model.prefill_converted_weights = prefill_loader.convert()

    def _register_normalizers(self):
        if self.config.train_config.get("customized_action_statistic_dof", None):
            action_statistic_dof = json.load(
                open(self.config.train_config["customized_action_statistic_dof"], "r")
            )
        else:
            action_statistic_dof = default_action_statistic_dof
        # 如果有ckpt，优先从ckpt中加载normalizer
        if hasattr(self.config, "normalizer_action_path"):
            self.normalizer_action = Normalizer.from_ckpt(
                self.config.normalizer_action_path
            )
        else:
            self.normalizer_action = Normalizer(
                action_statistic_dof,
                self.config.train_config["dof_config"],
                min_key=self.config.train_config.get("min_key", "min"),
                delta_key=self.config.train_config.get("delta_key", "delta"),
            )
        if hasattr(self.config, "normalizer_propri_path"):
            self.normalizer_propri = Normalizer.from_ckpt(
                self.config.normalizer_propri_path
            )
        else:
            self.normalizer_propri = Normalizer(
                action_statistic_dof,
                self.config.train_config["agent_pos_config"],
                min_key=self.config.train_config.get("min_key", "min"),
                delta_key=self.config.train_config.get("delta_key", "delta"),
            )

    @timer
    def construct_model_input(self, observation, prefix_text, postfix_text):
        batch_size = len(observation)
        dataset_names = [self.norm_key] * batch_size

        additional_inputs = {}

        # -------- proprioception / masks (batch) --------
        agent_pos_list = []
        agent_pos_mask_list = []
        dof_mask_list = []
        for obs in observation:
            if "robot_state_action_data" in obs:
                robot_state_action_data = obs["robot_state_action_data"]

                agent_pos = torch.from_numpy(robot_state_action_data.agent_pos)
                # 统一到 [1, T, D] 便于 cat 成 batch
                if agent_pos.dim() == 2:
                    agent_pos = agent_pos.unsqueeze(0)
                agent_pos_list.append(agent_pos)

                agent_pos_mask = torch.from_numpy(
                    robot_state_action_data.agent_pos_mask
                )
                if agent_pos_mask.dim() == 2:
                    agent_pos_mask = agent_pos_mask.unsqueeze(0)
                agent_pos_mask_list.append(agent_pos_mask)

                dof_mask = torch.from_numpy(robot_state_action_data.dof_mask)
                if dof_mask.dim() == 1:
                    dof_mask = dof_mask.unsqueeze(0)
                dof_mask_list.append(dof_mask)

        # cat: [B, T, D] / [B, ...]
        if len(agent_pos_list) > 0:
            agent_pos = torch.cat(agent_pos_list, dim=0)
            agent_pos_mask = torch.cat(agent_pos_mask_list, dim=0)
            dof_mask = torch.cat(dof_mask_list, dim=0)

            if self.normalizer_propri is not None:
                agent_pos = self.normalizer_propri.normalize_data(
                    agent_pos, dataset_names
                )

            additional_inputs["proprioception"] = agent_pos.detach()
            additional_inputs["agent_pos_mask"] = agent_pos_mask
            additional_inputs["dof_mask"] = dof_mask

        # -------- images (flattened, in placeholder scan order) --------
        with ScopeTimer("resize_images"):
            # TODO[KC]: optimize this using batch processing
            image_inputs = []
            all_image_sizes = []
            if ENABLE_FAST_PREPROCESS:
                for obs in observation:
                    current_image_inputs = self._resize_images_fast(obs)
                    image_inputs.extend(current_image_inputs)
                    # tensor shape is (H, W, C), convert to (W, H) to match PIL.size format
                    all_image_sizes.extend(
                        [
                            (image_i.shape[1], image_i.shape[0])
                            for image_i in current_image_inputs
                        ]
                    )
            else:
                for obs in observation:
                    current_image_inputs = self._resize_images(obs)
                    image_inputs.extend(current_image_inputs)
                    # tensor shape is (H, W, C), convert to (W, H) to match PIL.size format
                    all_image_sizes.extend(
                        [
                            (image_i.shape[1], image_i.shape[0])
                            for image_i in current_image_inputs
                        ]
                    )

        additional_inputs["image_size"] = all_image_sizes

        with ScopeTimer("preprocesser_call"):
            inputs = preprocesser_call(
                processor=self.model.processor,
                prefix_text=prefix_text,
                postfix_text=postfix_text,
                images=image_inputs,
                videos=None,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=1000,
                pad_to_128_multiple=False,
                pad_prefix_to_same_length=False,
                norm_state=(
                    additional_inputs["proprioception"]
                    if "proprioception" in additional_inputs
                    and self.config.train_config.get(
                        "use_state_string_representation", False
                    )
                    else None
                ),
                agent_pos_mask=(
                    additional_inputs["agent_pos_mask"]
                    if "agent_pos_mask" in additional_inputs
                    else None
                ),
                state_augmentation_prob=0.0,
                state_drop_prob=0.0,
                state_augmentation_ratio=0.0,
            )

        with ScopeTimer("convert_action_token_id and post "):
            action_token_id = self.model.processor.tokenizer.convert_tokens_to_ids(
                "<|action|>"
            )
            flow_action_mask = inputs["input_ids"] == action_token_id
            additional_inputs["moe_token_types"] = flow_action_mask
            additional_inputs["dataset_names"] = dataset_names

            inputs.update(additional_inputs)
            inputs = move_to_cuda(inputs, self.config.model_device)

        return inputs

    @timer
    def get_text_for_action(self, instruction):
        if (
            self.config.train_config["data"].get("use_embodied_system_prompt_ratio", 0)
            > 0
        ):
            if self.norm_key != "x2_normal":
                self.config.robot_id = 0
            cam_name_mapping = {cam_name: cam_name for cam_name in self.cam_names}
            prologue = get_prologue_with_embodied_information(
                dataset_name=self.norm_key,
                cam_mapping=cam_name_mapping,
                robot_id=self.config.robot_id,
                config=self.config.train_config,
            )
        else:
            prologue = f"{self.role_start_symbol}system\nYou are a helpful assistant.{self.role_end_symbol}\n"
        user_request = f"{self.role_start_symbol}user\nObservation:"
        for cam_name in self.cam_names:
            user_request += f" {_CAM_NAME_MAPPING[cam_name]}: {self.vision_start_symbol}{self.image_pad_symbol}{self.vision_end_symbol}"
        user_request += "\nInstruction:"
        text_prompt = f"\nPredict the next action in robot action.\nProprioception: {self.propri_symbol}\n"
        user_message = (
            f"{user_request} {instruction}{text_prompt}{self.role_end_symbol}\n"
        )
        assistant_message = f"{self.role_start_symbol}assistant\n"
        flow_action = f"{self.action_symbol * self.config.action_horizon}"

        prefix_text = prologue + user_message + assistant_message
        postfix_text = flow_action

        return prefix_text, postfix_text

    @timer
    def get_text_for_subtask_generation(self, instruction):
        prologue = f"{self.role_start_symbol}system\nYou are a helpful assistant.{self.role_end_symbol}\n"
        user_request = f"{self.role_start_symbol}user\nObservation:"
        for cam_name in self.cam_names:
            user_request += f" {_CAM_NAME_MAPPING[cam_name]}: {self.vision_start_symbol}{self.image_pad_symbol}{self.vision_end_symbol}"
        user_request += "\nInstruction:"
        text_prompt = "\nPredict the next action in language.\n"
        user_message = (
            f"{user_request} {instruction}{text_prompt}{self.role_end_symbol}\n"
        )
        assistant_message = f"{self.role_start_symbol}assistant\n"

        prefix_text = prologue + user_message + assistant_message
        postfix_text = ""

        return prefix_text, postfix_text

    @timer
    def _resize_images(self, observation):
        image_inputs = []
        for key in self.cam_names:
            if key not in observation:
                continue
            # 1. 获取原始图像
            # current_obs = observation[key].permute(1, 2, 0)
            current_obs = observation[key]
            if isinstance(current_obs, np.ndarray):
                img_pil = Image.fromarray(current_obs)
            elif isinstance(current_obs, Image.Image):
                img_pil = current_obs
            else:
                raise ValueError(f"Unsupported image type: {type(current_obs)}")
            orig_width, orig_height = img_pil.size

            target_size = self.config.data_config.resolution.get(key, -1)
            if target_size != -1:
                # 保持宽高比的限制逻辑
                if orig_width > orig_height:  # 横向图像
                    new_width = target_size
                    new_height = int(target_size * orig_height / orig_width)
                else:  # 纵向图像
                    new_height = target_size
                    new_width = int(target_size * orig_width / orig_height)
                img_pil = img_pil.resize((new_width, new_height))

            # 3. 应用智能缩放（qwen逻辑）
            current_width, current_height = img_pil.size
            resized_height, resized_width = smart_resize(
                current_height,
                current_width,
                factor=self.config.data_config.image_factor,
                min_pixels=self.config.data_config.min_pixels,
                max_pixels=self.config.data_config.max_pixels,
            )
            resized_img = img_pil.resize((resized_width, resized_height))
            resized_img = torch.from_numpy(np.array(resized_img)).to(
                self.config.model_device
            )
            image_inputs.append(resized_img)

        return image_inputs

    def _resize_images_fast(self, observation):
        import cv2

        image_inputs = []
        for key in ["face_view", "left_wrist_view", "right_wrist_view"]:
            if key not in observation:
                continue
            # 1. 获取原始图像
            # current_obs = observation[key].permute(1, 2, 0)
            current_obs = observation[key]
            orig_height, orig_width, _ = current_obs.shape

            target_size = self.config.data_config.resolution.get(key, -1)
            current_width, current_height = orig_width, orig_height
            if target_size != -1:
                # 保持宽高比的限制逻辑
                if orig_width > orig_height:  # 横向图像
                    new_width = target_size
                    new_height = int(target_size * orig_height / orig_width)
                else:  # 纵向图像
                    new_height = target_size
                    new_width = int(target_size * orig_width / orig_height)
                # img_pil = img_pil.resize((new_width, new_height))
                current_width = new_width
                current_height = new_height

            # 3. 应用智能缩放（qwen逻辑）
            resized_height, resized_width = smart_resize(
                current_height,
                current_width,
                factor=self.config.data_config.image_factor,  # FIXME
                min_pixels=self.config.data_config.min_pixels,  # FIXME
                max_pixels=self.config.data_config.max_pixels,  # FIXME
            )

            resized_img = cv2.resize(
                current_obs,
                (resized_width, resized_height),
                interpolation=cv2.INTER_CUBIC,
            )
            resized_img = torch.from_numpy(resized_img).to(self.config.model_device)
            image_inputs.append(resized_img)

        return image_inputs

    def infer_flow_action(self, observation, instruction):
        self.logger.info("开始生成flow action")
        self.logger.info(f"当前flow action的instruction为: {instruction}")

        prefix_text, postfix_text = self.get_text_for_action(instruction)
        model_input = self.construct_model_input(
            [observation], [prefix_text], [postfix_text]
        )

        padding = (
            torch.zeros_like(
                self.normalizer_action.delta[model_input["dataset_names"][0]]
            )
            .unsqueeze(0)
            .to("cpu")
        )
        padding_action = self.normalizer_action.normalize_data(
            padding, model_input["dataset_names"]
        ).to(model_input["input_ids"].device)

        with ScopeTimer("generate_flow_action"):
            model_output = self.model.generate_flow_action(
                action_horizon=self.config.action_horizon,
                action_dim=self.config.action_dim,
                num_inference_timesteps=self.config.num_inference_timesteps,
                padding_action=padding_action,
                **model_input,
            )

        self.logger.info("flow action生成完成")
        model_output["robot_state_action_data"] = observation["robot_state_action_data"]
        model_output["robot_state_action_data"].save_action_data(
            model_output["predict_action"]
        )
        self.logger.info("保存flow action到robot_state_action_data")
        return model_output

    def infer_flow_action_batch(self, observations, instructions):
        """
        支持批量flow action推理：
        - observations: List[Dict]，每个元素与单条推理的observation格式一致
        - instructions: List[str]，与observations一一对应
        返回 List[model_output]，长度等于batch size
        """
        assert len(observations) == len(
            instructions
        ), "observations与instructions长度需一致"
        batch_size = len(observations)

        prefix_list = []
        postfix_list = []
        for ins in instructions:
            prefix_text, postfix_text = self.get_text_for_action(ins)
            prefix_list.append(prefix_text)
            postfix_list.append(postfix_text)

        # 批量构造输入（一次 preprocesser_call），避免逐样本处理与手动 cat 合并
        batch_inputs = self.construct_model_input(
            observations, prefix_list, postfix_list
        )

        padding_list = []
        for ds_name in batch_inputs["dataset_names"]:
            padding = (
                torch.zeros_like(self.normalizer_action.delta[ds_name])
                .unsqueeze(0)
                .to("cpu")
            )
            padding_list.append(padding)
        padding = torch.cat(padding_list, dim=0)
        padding_action = self.normalizer_action.normalize_data(
            padding, batch_inputs["dataset_names"]
        ).to(batch_inputs["input_ids"].device)

        with ScopeTimer("generate_flow_action_batch"):
            model_output = self.model.generate_flow_action(
                action_horizon=self.config.action_horizon,
                action_dim=self.config.action_dim,
                num_inference_timesteps=self.config.num_inference_timesteps,
                padding_action=padding_action,
                **batch_inputs,
            )

        predict_action = model_output["predict_action"]  # [B, H, D]
        outputs = []
        for i in range(batch_size):
            single_output = {
                "predict_action": predict_action[i : i + 1],  # 保留batch维便于后续处理
                "robot_state_action_data": observations[i]["robot_state_action_data"],
            }
            single_output["robot_state_action_data"].save_action_data(
                single_output["predict_action"]
            )
            outputs.append(single_output)

        return outputs

    def infer_ar_action(self, observation, instruction):
        self.logger.info("开始生成ar action")
        self.logger.info(f"当前ar action的instruction为: {instruction}")

        prefix_text, _ = self.get_text_for_action(instruction)
        model_input = self.construct_model_input([observation], [prefix_text], [""])

        model_output = self.model.generate_ar_action(
            action_horizon=self.config.action_horizon,
            action_dim=self.config.action_dim,
            num_inference_timesteps=self.config.num_inference_timesteps,
            **model_input,
        )
        self.logger.info("ar action生成完成")
        model_output["robot_state_action_data"] = observation["robot_state_action_data"]
        model_output["robot_state_action_data"].save_action_data(
            model_output["predict_action"]
        )
        self.logger.info("保存ar action到robot_state_action_data")
        return model_output

    def infer_subtask(self, observation, instruction):
        self.logger.info("开始生成subtask")
        self.logger.info(f"当前subtask的instruction为: {instruction}")

        prefix_text, postfix_text = self.get_text_for_subtask_generation(instruction)
        model_input = self.construct_model_input([observation], [prefix_text], [""])

        model_output = self.model.generate_text(**model_input)
        subtask = model_output["predict_output_text"][0].split("<|im_end|>")[0].strip()
        self.logger.info(f"subtask生成完成, 生成的subtask为: {subtask}")
        return subtask

    def infer_vqa(self, observation, instruction):
        self.logger.info("开始生成vqa答案")

        if isinstance(observation["multi_modal"], list):
            orig_size = observation["multi_modal"][0].size
        else:
            orig_size = observation["multi_modal"].size

        prefix_text = instruction
        model_input = self.construct_model_input([observation], [prefix_text], [""])

        model_output = self.model.generate_text(**model_input)
        answer = model_output["predict_output_text"][0].split("<|im_end|>")[0].strip()
        self.logger.info(f"vqa答案生成完成, 生成的答案为: {answer}")
        answer = reverse_grounding_points(
            answer,
            orig_size[1],
            orig_size[0],
            model_input["image_size"][0][1],
            model_input["image_size"][0][0],
            self.config.data_config.model_type,
        )
        points = extract_grounding_points(answer)

        return {
            "answer": answer,
            "points": points,
        }
