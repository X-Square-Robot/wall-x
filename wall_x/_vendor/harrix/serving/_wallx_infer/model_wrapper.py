import os
import torch
import copy
from safetensors.torch import load_file
import numpy as np
from PIL import Image
from qwen_vl_utils.vision_process import smart_resize
from transformers import BatchFeature

from wall_x.trainer.trainer_utils import load_wallx_processors

from wall_x._vendor.x2robot_utils.text_templates import (
    preprocesser_call,
    get_prologue_with_embodied_information,
)
from wall_x._vendor.x2robot_utils.grounding import (
    reverse_grounding_points,
    extract_grounding_points,
)

from wall_x._vendor.harrix.serving._wallx_infer.infer_config import InferConfig
from wall_x._vendor.harrix.utils.ckpt_load import reshape_compatible_state_dict
from wall_x._vendor.harrix.utils.train_config import (
    resolve_state_bins,
    resolve_use_state_string_representation,
)
from wall_x._vendor.harrix.serving._wallx_infer.logger import InferLogger
from wall_x.utils.timers import timer, ScopeTimer


def _camera_label(cam_name: str) -> str:
    return cam_name.replace("_", " ")


ENABLE_FAST_PREPROCESS = os.getenv("ENABLE_FAST_PREPROCESS", "False").lower() == "true"


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
        self.logger.info(f"normalizers {self.norm_key} registered")
        self._load_processor()
        self._load_model()
        self.load_ckpt()
        self.logger.info(f"Model loaded from {self.config.checkpoint_path}")

        self.norm_key = self.config.norm_key
        self._register_normalizers()
        self.logger.info(f"normalizers {self.norm_key} registered")

        # init robot_type_id (for v3.1 delta tokenizer)
        self.robot_type_id = None
        if self.tokenizer_mixin is not None:
            robot_type = getattr(self.config, "robot_type", None)
            if robot_type:
                self.tokenizer_mixin.init_inference(robot_type=robot_type)
                if hasattr(self.tokenizer_mixin, "robot_type_id"):
                    self.robot_type_id = self.tokenizer_mixin.robot_type_id
                    if self.robot_type_id is not None:
                        self.logger.info(
                            f"Initialized robot_type_id: {self.robot_type_id} from robot_type: {robot_type}"
                        )

        self.role_start_symbol = "<|im_start|>"
        self.role_end_symbol = "<|im_end|>"
        self.vision_start_symbol = "<|vision_start|>"
        self.vision_end_symbol = "<|vision_end|>"
        self.image_pad_symbol = "<|image_pad|>"
        self.propri_symbol = "<|propri|>"
        self.action_symbol = "<|action|>"

        self.cam_names = self.config.cam_names

    def _load_processor(self):
        # load tokenizer on model_device at inference
        device = getattr(self.config, "model_device", "cuda")
        processors_dict = load_wallx_processors(self.config.train_config, device=device)
        self.processor = processors_dict["processor"]
        self.action_mapper = processors_dict["action_mapper"]
        self.tokenizer_mixin = processors_dict.get("tokenizer_mixin")

    def _load_model(self):
        from wall_x.trainer.adapters import resolve_adapter

        model_type = self.config.train_config["model_type"]
        adapter_cls = resolve_adapter(model_type)
        ModelClass = adapter_cls.inference_model_class()
        self.ModelClass = ModelClass

        self.logger.info(f"Initializing model: {model_type}")

        self.model = ModelClass(
            self.config.model_config,
            self.processor,
            self.tokenizer_mixin,
        )

        # log attention implementation — variant-specific layout dispatched
        # via the adapter (qwen2_5 uses model.model._attn_implementation;
        # qwen3 / qwen3_5 use model.model.language_model.config._attn_impl).
        adapter_cls.log_attention_implementation(self.logger, self.model)
        self.model_type = model_type

        self.logger.info("Resizing model token embeddings")
        self.model.resize_token_embeddings(len(self.processor.tokenizer))
        self.logger.info("Token embedding resize done")
        self.logger.info("Casting selected parameters to bfloat16")
        self.model.to_bfloat16_for_selected_params()
        self.logger.info("bfloat16 cast done")

    def load_ckpt(self, checkpoint_path: str = None):
        if checkpoint_path is None:
            checkpoint_path = self.config.checkpoint_path

        if os.path.exists(os.path.join(checkpoint_path, "global_step.pth")):
            global_step = torch.load(os.path.join(checkpoint_path, "global_step.pth"))[
                "global_step"
            ]
            self.logger.info(f"Checkpoint global_step: {global_step}")

        fsdp_ckpt = os.path.join(checkpoint_path, "pytorch_model_fsdp.bin")
        safetensor_ckpt = os.path.join(checkpoint_path, "model.safetensors")
        if os.path.exists(fsdp_ckpt):
            self.logger.info(f"Loading FSDP checkpoint: {fsdp_ckpt}")
            state_dict = torch.load(fsdp_ckpt, map_location="cpu")
            # unwrap outer dict wrapper (e.g. {'state_dict': {...}})
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                self.logger.info(
                    "Using nested state_dict inside pytorch_model_fsdp.bin"
                )
                state_dict = state_dict["state_dict"]
        elif os.path.exists(safetensor_ckpt):
            self.logger.info(f"Loading safetensors weights: {safetensor_ckpt}")
            state_dict = load_file(safetensor_ckpt, device="cpu")
        else:
            raise FileNotFoundError(
                f"❌ No checkpoint found under {checkpoint_path}. "
                "Expecting either pytorch_model_fsdp.bin or model.safetensors."
            )

        # Qwen models need fused weight conversion
        if not self.ModelClass.is_fused(state_dict):
            self.logger.info(
                "Converting non-fused weights to fused format...",
            )
            state_dict = self.ModelClass.convert_to_fused(state_dict)
        else:
            self.logger.info(
                "The weights is fused, skipping conversion.",
            )

        state_dict = reshape_compatible_state_dict(
            state_dict, self.model.state_dict(), log_fn=self.logger.info
        )
        msg = self.model.load_state_dict(state_dict, strict=False)
        self.model.set_normalizer(
            copy.deepcopy(self.normalizer_action),
            copy.deepcopy(self.normalizer_propri),
        )
        self.logger.info(f"load_state_dict result: {msg}")
        self.model.eval()
        self.model.to(self.config.model_device)
        self.model.to_bfloat16_for_selected_params()

        if hasattr(self.model, "load_optimized_weights"):
            self.model.load_optimized_weights(state_dict)

    def _register_normalizers(self):
        from wall_x._vendor.harrix.utils.normalizer import build_normalizers

        self.normalizer_action, self.normalizer_propri, resolved = build_normalizers(
            self.config.checkpoint_path,
            self.config.train_config,
            self.config.norm_key,
        )
        self.norm_key = resolved
        self.config.norm_key = resolved
        self._log_norm_debug(self.norm_key)

    def _log_norm_debug(self, norm_key):
        if not hasattr(self, "normalizer_action") or not hasattr(
            self, "normalizer_propri"
        ):
            self.logger.warning("[NormDebug] normalizer is not initialized yet")
            return

        if (
            norm_key in self.normalizer_action.min
            and norm_key in self.normalizer_propri.min
        ):
            self.logger.debug(
                "[NormDebug] norm_key=%s action(min/delta) shape=%s/%s",
                norm_key,
                tuple(self.normalizer_action.min[norm_key].shape),
                tuple(self.normalizer_action.delta[norm_key].shape),
            )
            self.logger.debug(
                "[NormDebug] action min(all)=%s delta(all)=%s",
                self.normalizer_action.min[norm_key].detach().cpu().tolist(),
                self.normalizer_action.delta[norm_key].detach().cpu().tolist(),
            )
            self.logger.debug(
                "[NormDebug] propri(min/delta) shape=%s/%s",
                tuple(self.normalizer_propri.min[norm_key].shape),
                tuple(self.normalizer_propri.delta[norm_key].shape),
            )
            self.logger.debug(
                "[NormDebug] propri min(all)=%s delta(all)=%s",
                self.normalizer_propri.min[norm_key].detach().cpu().tolist(),
                self.normalizer_propri.delta[norm_key].detach().cpu().tolist(),
            )
        else:
            self.logger.warning(
                "[NormDebug] norm_key=%s not found in normalizer", norm_key
            )

    @timer
    def construct_model_input(
        self, observation, prefix_text, postfix_text, pad_prefix=False
    ):
        batch_size = len(observation)
        dataset_names = [self.norm_key] * batch_size
        self.logger.debug("[NormDebug] dataset_names=%s", dataset_names)

        additional_inputs = {}

        # -------- proprioception / masks (batch) --------
        agent_pos_list = []
        agent_pos_mask_list = []
        dof_mask_list = []
        for obs in observation:
            if "robot_state_action_data" in obs:
                robot_state_action_data = obs["robot_state_action_data"]

                agent_pos = torch.from_numpy(robot_state_action_data.agent_pos)
                # reshape to [1, T, D] for batch cat
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
                pad_prefix_to_same_length=pad_prefix,
                norm_state=(
                    additional_inputs["proprioception"]
                    if "proprioception" in additional_inputs
                    and resolve_use_state_string_representation(
                        self.config.train_config
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
                state_bins=resolve_state_bins(self.config.train_config),
                inference_mode=True,
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

    def get_text_for_dllm_action(self, instruction):
        if (
            self.config.train_config["data"].get("use_embodied_system_prompt_ratio", 0)
            > 0
        ):
            if self.norm_key != "x2_normal" and self.norm_key != "ex_normal":
                self.config.robot_id = 0
            cam_name_mapping = {cam_name: cam_name for cam_name in self.cam_names}
            prologue = get_prologue_with_embodied_information(
                dataset_name=self.norm_key,
                cam_mapping=cam_name_mapping,
                robot_id=self.config.robot_id,
                uid="",
                config=self.config.data_config,
            )
        else:
            prologue = f"{self.role_start_symbol}system\nYou are a helpful assistant.{self.role_end_symbol}\n"
        user_request = f"{self.role_start_symbol}user\nObservation:"
        for cam_name in self.cam_names:
            user_request += f" {_camera_label(cam_name)}: {self.vision_start_symbol}{self.image_pad_symbol}{self.vision_end_symbol}"
        user_request += "\nInstruction:"
        text_prompt = f"\nPredict the next action in robot action.\nProprioception: {self.propri_symbol}\n"
        user_message = (
            f"{user_request} {instruction}{text_prompt}{self.role_end_symbol}\n"
        )
        placeholder_seq = self.tokenizer_mixin.get_placeholder_for_dllm()
        ar_token = "".join(placeholder_seq) + "<|im_end|>\n"
        assistant_message = f"{self.role_start_symbol}assistant\n{ar_token}"
        flow_action = f"{self.action_symbol * self.config.action_horizon}"

        prefix_text = prologue + user_message + assistant_message
        postfix_text = flow_action

        return prefix_text, postfix_text

    @timer
    def get_text_for_action(self, instruction):
        if (
            self.config.train_config["data"].get("use_embodied_system_prompt_ratio", 0)
            > 0
        ):
            if self.norm_key not in ["x2_normal", "ex_normal"]:
                self.config.robot_id = 0
            cam_name_mapping = {cam_name: cam_name for cam_name in self.cam_names}
            prologue = get_prologue_with_embodied_information(
                dataset_name=self.norm_key,
                cam_mapping=cam_name_mapping,
                robot_id=self.config.robot_id,
                uid="",
                config=self.config.data_config,
            )
        else:
            prologue = f"{self.role_start_symbol}system\nYou are a helpful assistant.{self.role_end_symbol}\n"
        user_request = f"{self.role_start_symbol}user\nObservation:"
        for cam_name in self.cam_names:
            user_request += f" {_camera_label(cam_name)}: {self.vision_start_symbol}{self.image_pad_symbol}{self.vision_end_symbol}"
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
            user_request += f" {_camera_label(cam_name)}: {self.vision_start_symbol}{self.image_pad_symbol}{self.vision_end_symbol}"
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
                # aspect-ratio preserving resize logic
                if orig_width > orig_height:  # landscape
                    new_width = target_size
                    new_height = int(target_size * orig_height / orig_width)
                else:  # portrait
                    new_height = target_size
                    new_width = int(target_size * orig_width / orig_height)
                img_pil = img_pil.resize((new_width, new_height))

            # 3. smart resize (Qwen logic)
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
            current_obs = observation[key]
            orig_height, orig_width, _ = current_obs.shape

            target_size = self.config.data_config.resolution.get(key, -1)
            current_width, current_height = orig_width, orig_height
            if target_size != -1:
                # aspect-ratio preserving resize logic
                if orig_width > orig_height:  # landscape
                    new_width = target_size
                    new_height = int(target_size * orig_height / orig_width)
                else:  # portrait
                    new_height = target_size
                    new_width = int(target_size * orig_width / orig_height)
                current_width = new_width
                current_height = new_height

            # 3. smart resize (Qwen logic)
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
        self.logger.info("Generating flow action")
        self.logger.info(f"Flow action instruction: {instruction}")

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

        self.logger.info("Flow action generation done")

        model_output["robot_state_action_data"] = observation["robot_state_action_data"]
        model_output["robot_state_action_data"].save_action_data(
            model_output["predict_action"]
        )
        self.logger.info("Saving flow action to robot_state_action_data")
        return model_output

    def infer_flow_action_batch(self, observations, instructions):
        """
        Batch flow action inference:
        - observations: List[Dict], same format as single inference
        - instructions: List[str], aligned with observations
        Returns List[model_output] with batch size length
        """
        assert len(observations) == len(
            instructions
        ), "observations and instructions must have the same length"
        batch_size = len(observations)

        prefix_list = []
        postfix_list = []
        for ins in instructions:
            prefix_text, postfix_text = self.get_text_for_action(ins)
            prefix_list.append(prefix_text)
            postfix_list.append(postfix_text)

        # batch inputs in one preprocesser_call (avoid per-sample cat)
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
            single_action = predict_action[i : i + 1]
            single_output = {
                "predict_action": single_action,
                "robot_state_action_data": observations[i]["robot_state_action_data"],
            }
            single_output["robot_state_action_data"].save_action_data(
                single_output["predict_action"]
            )
            outputs.append(single_output)

        return outputs

    def infer_ar_action(self, observation, instruction):
        self.logger.info("Generating AR action")
        self.logger.info(f"AR action instruction: {instruction}")

        prefix_text, _ = self.get_text_for_action(instruction)
        model_input = self.construct_model_input([observation], [prefix_text], [""])

        model_output = self.model.generate_ar_action(
            action_horizon=self.config.action_horizon,
            action_dim=self.config.ar_action_dim,
            num_inference_timesteps=self.config.num_inference_timesteps,
            robot_type_id=self.robot_type_id,
            **model_input,
        )
        self.logger.info("AR action generation done")

        model_output["robot_state_action_data"] = observation["robot_state_action_data"]
        model_output["robot_state_action_data"].save_action_data(
            model_output["predict_action"]
        )
        self.logger.info("Saving AR action to robot_state_action_data")
        return model_output

    def infer_subtask(self, observation, instruction):
        self.logger.info("Generating subtask")
        self.logger.info(f"Subtask instruction: {instruction}")

        prefix_text, postfix_text = self.get_text_for_subtask_generation(instruction)
        model_input = self.construct_model_input([observation], [prefix_text], [""])

        model_output = self.model.generate_text(**model_input)
        subtask = model_output["predict_output_text"][0].split("<|im_end|>")[0].strip()
        self.logger.info(f"Subtask done: {subtask}")
        return subtask

    def infer_vqa(self, observation, instruction):
        self.logger.info("Generating VQA answer")

        if isinstance(observation["multi_modal"], list):
            orig_size = observation["multi_modal"][0].size
        else:
            orig_size = observation["multi_modal"].size

        prefix_text = instruction
        model_input = self.construct_model_input([observation], [prefix_text], [""])

        model_output = self.model.generate_text(**model_input)
        answer = model_output["predict_output_text"][0].split("<|im_end|>")[0].strip()
        self.logger.info(f"VQA answer done: {answer}")
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

    def infer_dllm_action(
        self, observation, instruction, use_ar_action=False, dataset_name="x2_normal"
    ):
        assert self.model_type in ["qwen2_5"], "DLLM only supports qwen2_5"
        prefix_text, postfix_text = self.get_text_for_dllm_action(instruction)
        model_input = self.construct_model_input(
            [observation], [prefix_text], [postfix_text], pad_prefix=True
        )
        model_input = self.model.update_infer_dllm_position_mask(model_input)
        total_ar_step = self.tokenizer_mixin.inference_ar_steps_for_dllm
        cnt = 0
        while cnt < 3:  # retry up to 3 times for fast
            model_output = self.model.generate_dllm_action(
                action_horizon=self.config.action_horizon,
                action_dim=self.config.action_dim,
                ar_action_dim=self.config.ar_action_dim,
                num_inference_timesteps=self.config.num_inference_timesteps,
                use_ar_action=use_ar_action,
                total_ar_step=total_ar_step,
                robot_type_id=self.robot_type_id,  # for v3.1 delta decode
                **model_input,
            )
            if model_output["predict_action"] is not None:
                break
            cnt += 1
            self.logger.warning(f"DLLM action failed, retry {cnt}")
        self.logger.info("DLLM action generation done")

        model_output["robot_state_action_data"] = observation["robot_state_action_data"]
        model_output["robot_state_action_data"].save_action_data(
            model_output["predict_action"]
        )
        self.logger.info("Saving DLLM action to robot_state_action_data")
        return model_output
