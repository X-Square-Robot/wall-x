import contextlib
import copy
import json
import os
from collections import defaultdict
from collections.abc import Mapping

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils.vision_process import smart_resize as qwen_smart_resize
from safetensors.torch import load_file
from transformers import BatchFeature

from wall_x.trainer.trainer_utils import load_wallx_processors_from_cfg

from wall_x._vendor.x2robot_utils.text_templates import (
    preprocesser_call,
    get_prologue_with_embodied_information,
)
from wall_x._vendor.x2robot_utils.text_templates import (
    _PUBLIC_CAMERA_LABELS as _CAM_NAME_MAPPING,
)
from wall_x._vendor.x2robot_utils.grounding import (
    reverse_grounding_points,
    extract_grounding_points,
)

from wall_x._vendor.harrix.infer.infer_config import (
    InferConfig,
    resolve_embodiment_type,
    resolve_prompt_dataset_family,
)
from wall_x._vendor.harrix.utils.dataset_v2 import (
    InferenceDataConfig,
    build_dataset_v2_text_processor,
    dataset_v2_smart_resize,
    inject_dataset_v2_state_strings,
    replace_state_schema_placeholder,
    validate_inference_data_semantics,
)
from wall_x.model.core.refiner import RefinerConfig, verify_loaded
from wall_x.model.core.action.normalizer import (
    NORM_KEY_SEP,
    Normalizer,
    make_norm_key,
)
from wall_x._vendor.harrix.infer.logger import InferLogger
from wall_x._vendor.harrix.utils.normalizer import pad_norm_stats_like_collator
from wall_x.utils.timers import timer, ScopeTimer
from wall_x.data.backends.lerobot.utils import NormStats


ENABLE_FAST_PREPROCESS = os.getenv("ENABLE_FAST_PREPROCESS", "False").lower() == "true"
# Move the image resize (currently CPU cv2/PIL BICUBIC) onto the GPU with
# torch bicubic + antialias. Sizing (dataset_v2_smart_resize) is unchanged, so
# image_grid_thw is identical; only the interpolation kernel differs, which is
# NOT bit-identical to PIL BICUBIC (see docs/gpu_resize verification).
# Default ON: a seeded 5-episode 5090 A/B showed the resulting open-loop L1
# change sits inside the GPU compute-nondeterminism floor (|Δ|≈8e-4, ~1% of
# L1) — no task-performance cost — while saving ~9.6 ms/infer (resize
# 11.3->1.7 ms, 6.8x). Set ENABLE_GPU_RESIZE=0 for bit-exact PIL parity
# (golden regression / strict train-alignment).
ENABLE_GPU_RESIZE = os.getenv("ENABLE_GPU_RESIZE", "True").lower() == "true"


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


def _load_norm_stats(norm_stats_path, action_key):
    with open(norm_stats_path, "r") as f:
        norm_stats = json.load(f)

    q01 = torch.tensor(norm_stats["norm_stats"][action_key]["q01"])
    q99 = torch.tensor(norm_stats["norm_stats"][action_key]["q99"])
    delta = q99 - q01

    return NormStats(min=q01, max=q99, delta=delta)


class WallxModelWrapper:
    def __init__(self, config: InferConfig):
        self.config = config
        if isinstance(config.data_config, InferenceDataConfig):
            validate_inference_data_semantics(config.data_config, config.action_horizon)
        self.logger = InferLogger.get_model_logger("WallxModelWrapper")
        self.norm_key = self.config.norm_key
        self._register_normalizers()
        self.logger.info(
            "normalizers registered: requested=%s end_effector_type=%s resolved=%s",
            self.norm_key,
            getattr(self.config, "end_effector_type", None),
            self.norm_dataset_key,
        )
        self._load_processor()
        self._load_model()
        self.load_ckpt()
        self.logger.info(f"model {self.config.checkpoint_path} loaded")

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

    def _resolve_norm_key(self):
        # ``norm_key`` stays the bare dataset_type (used e.g. for the embodied
        # prompt). ``norm_dataset_key`` is the key actually used to index the
        # normalizer -- possibly the composite "dataset_type/end_effector_type"
        # (x2robot_v2 stats are keyed by both). Resolution order: explicit
        # end_effector_type -> exact match -> unique composite -> legacy fallback.
        available_action_keys = set(self.normalizer_action.min.keys())
        available_propri_keys = set(self.normalizer_propri.min.keys())
        available_keys = sorted(available_action_keys & available_propri_keys)
        self.norm_dataset_key = self.norm_key

        if not available_keys:
            self.logger.warning(
                "[NormDebug] no available norm keys found in normalizer"
            )
            return

        # 1) explicit end_effector_type -> composite key (one ckpt, many EEs).
        eet = getattr(self.config, "end_effector_type", None)
        if eet:
            composite = make_norm_key(self.norm_key, eet)
            if composite in available_keys:
                self.norm_dataset_key = composite
                return
            raise RuntimeError(
                "Explicit end_effector_type requires an exact composite "
                f"normalizer key, but {composite!r} is unavailable. "
                f"available_keys={available_keys}"
            )

        # 2) exact match (flat stats entry, or lerobot / x2robot_v1 bare key).
        if self.norm_key in available_keys:
            self.norm_dataset_key = self.norm_key
            return

        # 3) bare norm_key -> unique composite "dataset_type/<ee>".
        composite_matches = [
            k for k in available_keys if k.startswith(f"{self.norm_key}{NORM_KEY_SEP}")
        ]
        if len(composite_matches) == 1:
            self.norm_dataset_key = composite_matches[0]
            return
        if len(composite_matches) > 1:
            raise RuntimeError(
                f"norm_key={self.norm_key!r} matches multiple end_effector_types "
                f"{composite_matches}; set InferConfig.end_effector_type "
                "(e.g. --model-config.end-effector-type <ee>) to disambiguate."
            )

        # 4) legacy "_" prefix / single-key fallback.
        prefix_matches = [
            k for k in available_keys if k.startswith(f"{self.norm_key}_")
        ]
        if len(prefix_matches) == 1:
            self.norm_dataset_key = prefix_matches[0]
            return
        if len(available_keys) == 1:
            self.norm_dataset_key = available_keys[0]
            return

        self.logger.warning(
            "[NormDebug] norm_key=%s not found in normalizer. available_keys=%s",
            self.norm_key,
            available_keys,
        )

    def _load_processor(self):
        device = getattr(self.config, "model_device", "cuda")
        # Use the same typed model config and the checkpoint normalizer as
        # training. In particular, delta action tokenizers must not silently
        # fall back to the repository-wide default statistics.
        processors_dict = load_wallx_processors_from_cfg(
            self.config.cfg,
            normalizer=self.normalizer_action,
            device=device,
        )
        self.processor = processors_dict["processor"]
        self.action_mapper = processors_dict["action_mapper"]
        self.tokenizer_mixin = processors_dict.get("tokenizer_mixin")
        if self.config.train_config.get("model_type") == "qwen3_5":
            processor_name = type(self.processor).__name__
            if processor_name != "Qwen3VLProcessor":
                raise TypeError(
                    "qwen3_5 checkpoint requires Qwen3VLProcessor, got "
                    f"{type(self.processor).__module__}.{processor_name}"
                )

    def _load_model(self):
        from wall_x.trainer.adapters import ADAPTER_REGISTRY, format_adapter_error

        model_type = self.config.train_config["model_type"]
        adapter_cls = ADAPTER_REGISTRY.get(model_type)
        if adapter_cls is None:
            raise ValueError(format_adapter_error(model_type))
        ModelClass = adapter_cls.model_class()
        self.ModelClass = ModelClass

        self.logger.info(f"initializing model: {model_type}")

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

        # Grow only: some checkpoints pad the vocabulary past len(tokenizer)
        # and save the padded rows, so shrinking would break weight loading.
        current_vocab = self.model.get_input_embeddings().weight.shape[0]
        if len(self.processor.tokenizer) > current_vocab:
            self.logger.info("resizing model token embeddings")
            self.model.resize_token_embeddings(len(self.processor.tokenizer))
            self.logger.info("model token embeddings resized")
        self.logger.info("casting selected model parameters to bfloat16")
        self.model.to_bfloat16_for_selected_params()
        self.logger.info("selected model parameters cast to bfloat16")

    def load_ckpt(self, checkpoint_path: str = None):
        if checkpoint_path is None:
            checkpoint_path = self.config.checkpoint_path

        if os.path.exists(os.path.join(checkpoint_path, "global_step.pth")):
            global_step = torch.load(os.path.join(checkpoint_path, "global_step.pth"))[
                "global_step"
            ]
            self.logger.info(f"checkpoint training step: {global_step}")

        fsdp_ckpt = os.path.join(checkpoint_path, "pytorch_model_fsdp.bin")
        safetensor_ckpt = os.path.join(checkpoint_path, "model.safetensors")
        if os.path.exists(fsdp_ckpt):
            self.logger.info(f"Loading FSDP checkpoint: {fsdp_ckpt}")
            state_dict = torch.load(fsdp_ckpt, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                self.logger.info(
                    "Using nested state_dict inside pytorch_model_fsdp.bin"
                )
                state_dict = state_dict["state_dict"]
        elif os.path.exists(safetensor_ckpt):
            self.logger.info(f"loading checkpoint safetensors weights: {safetensor_ckpt}")
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

        embedding_keys = [
            key
            for key, value in state_dict.items()
            if key.endswith("embed_tokens.weight") and value.ndim == 2
        ]
        if embedding_keys:
            checkpoint_vocab_size = max(
                int(state_dict[key].shape[0]) for key in embedding_keys
            )
            tokenizer_size = len(self.processor.tokenizer)
            if checkpoint_vocab_size < tokenizer_size:
                raise RuntimeError(
                    "checkpoint token embedding is smaller than the tokenizer: "
                    f"checkpoint={checkpoint_vocab_size}, tokenizer={tokenizer_size}. "
                    "Refusing to load because valid tokenizer IDs would be out of range."
                )
            current_vocab_size = int(
                self.model.get_input_embeddings().weight.shape[0]
            )
            if current_vocab_size != checkpoint_vocab_size:
                self.logger.warning(
                    "Resizing token embeddings from %d to checkpoint size %d "
                    "(tokenizer size %d)",
                    current_vocab_size,
                    checkpoint_vocab_size,
                    tokenizer_size,
                )
                self.model.resize_token_embeddings(checkpoint_vocab_size)

        msg = self.model.load_state_dict(state_dict, strict=False)
        self._verify_refiner_load(state_dict, msg)
        self.model.set_normalizer(
            copy.deepcopy(self.normalizer_action),
            copy.deepcopy(self.normalizer_propri),
        )
        self.logger.info(f"load_state_dict result: {msg}")
        self.model.eval()
        self.model.to(self.config.model_device)
        self.model.to_bfloat16_for_selected_params()

    def _verify_refiner_load(self, state_dict, load_result):
        vla_config = getattr(self.model.config, "vla_config", None)
        loaded = verify_loaded(
            getattr(self.model, "model", self.model),
            state_dict.keys(),
            load_result.missing_keys,
            RefinerConfig.from_vla_config(vla_config),
        )
        if loaded:
            self.logger.info("residual refiner: loaded %d twin tensors", loaded)

    def _register_normalizers(self):
        missing = object()
        action_statistic_dof = missing

        def load_action_statistic_dof():
            nonlocal action_statistic_dof
            if action_statistic_dof is not missing:
                return action_statistic_dof
            if self.config.train_config.get("customized_action_statistic_dof", None):
                with open(
                    self.config.train_config["customized_action_statistic_dof"], "r"
                ) as f:
                    action_statistic_dof = json.load(f)
            else:
                raise FileNotFoundError(
                    "No normalizer statistics: expected norm_stats.json or "
                    "normalizer_action.pth / normalizer_propri.pth in the "
                    "checkpoint, or customized_action_statistic_dof in the "
                    "train config."
                )
            return action_statistic_dof

        checkpoint_path = self.config.checkpoint_path
        norm_key = self.config.norm_key
        norm_stats_path = os.path.join(checkpoint_path, "norm_stats.json")
        if os.path.exists(norm_stats_path):
            self.logger.info(
                f"loading LeRobot norm stats from checkpoint {checkpoint_path}: {norm_stats_path}"
            )
            propri_stats = pad_norm_stats_like_collator(
                _load_norm_stats(norm_stats_path, "observation.state"),
                sum(self.config.train_config["agent_pos_config"].values()),
            )
            self.normalizer_propri = Normalizer.from_lerobot_norm_stats(
                propri_stats, norm_key
            )
            action_stats = pad_norm_stats_like_collator(
                _load_norm_stats(norm_stats_path, "action"),
                sum(self.config.train_config["dof_config"].values()),
            )
            self.normalizer_action = Normalizer.from_lerobot_norm_stats(
                action_stats, norm_key
            )
            self._resolve_norm_key()
            self._log_norm_debug(self.norm_dataset_key)
            return

        if hasattr(self.config, "normalizer_action_path"):
            self.normalizer_action = Normalizer.from_ckpt(
                self.config.normalizer_action_path
            )
        else:
            action_statistic_dof = load_action_statistic_dof()
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
            action_statistic_dof = load_action_statistic_dof()
            self.normalizer_propri = Normalizer(
                action_statistic_dof,
                self.config.train_config["agent_pos_config"],
                min_key=self.config.train_config.get("min_key", "min"),
                delta_key=self.config.train_config.get("delta_key", "delta"),
            )

        self._resolve_norm_key()
        self._log_norm_debug(self.norm_dataset_key)

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
        dataset_names = [self.norm_dataset_key] * batch_size
        self.logger.debug("[NormDebug] dataset_names=%s", dataset_names)
        is_dataset_v2 = self.config.cfg.data.dataset_type == "x2robot_v2"
        # ``padding_side`` is a real field on the data config, resolved from the
        # checkpoint (see build_inference_data_config); no dataset_type guess.
        padding_side = self.config.data_config.padding_side
        state_bins = getattr(
            self.config.data_config,
            "state_bins",
            self.config.train_config.get("data", {}).get("state_bins", 256),
        )
        use_state_string = getattr(
            self.config.data_config,
            "use_state_string_representation",
            self.config.train_config.get(
                "use_state_string_representation", False
            ),
        )
        state_keep_dims = (
            self.config.data_config.epilogue_params.get(
                "state_string_keep_dims_by_dataset", {}
            )
            if isinstance(self.config.data_config, InferenceDataConfig)
            else {}
        )

        additional_inputs = {}

        # -------- proprioception / masks (batch) --------
        agent_pos_list = []
        agent_pos_mask_list = []
        dof_mask_list = []
        for obs in observation:
            if "robot_state_action_data" in obs:
                robot_state_action_data = obs["robot_state_action_data"]

                agent_pos = torch.from_numpy(robot_state_action_data.agent_pos)
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
            for obs in observation:
                current_image_inputs = self._observation_images(obs)
                image_inputs.extend(current_image_inputs)
                all_image_sizes.extend(
                    [
                        image_i.size
                        if isinstance(image_i, Image.Image)
                        else (image_i.shape[1], image_i.shape[0])
                        for image_i in current_image_inputs
                    ]
                )

        additional_inputs["image_size"] = all_image_sizes

        state_was_injected = bool(
            use_state_string
            and state_keep_dims
            and "proprioception" in additional_inputs
        )
        if state_was_injected:
            prefix_text = inject_dataset_v2_state_strings(
                prefix_text,
                additional_inputs["proprioception"],
                additional_inputs["agent_pos_mask"],
                dataset_names,
                state_bins=state_bins,
                keep_dims_by_dataset=state_keep_dims,
                state_drop=bool(getattr(self.config, "state_drop", False)),
            )

        with ScopeTimer("preprocesser_call"):
            inputs = preprocesser_call(
                processor=self.model.processor,
                prefix_text=prefix_text,
                postfix_text=postfix_text,
                images=image_inputs,
                videos=None,
                padding=True,
                padding_side=padding_side,
                # Right truncation would drop the trailing action tokens.
                truncation=False,
                return_tensors="pt",
                max_length=None,
                pad_to_128_multiple=False,
                pad_prefix_to_same_length=pad_prefix,
                norm_state=(
                    additional_inputs["proprioception"]
                    if "proprioception" in additional_inputs
                    and use_state_string
                    and not state_was_injected
                    else None
                ),
                agent_pos_mask=(
                    additional_inputs["agent_pos_mask"]
                    if "agent_pos_mask" in additional_inputs
                    else None
                ),
                state_augmentation_prob=0.0,
                state_drop_prob=(
                    1.0 if getattr(self.config, "state_drop", False) else 0.0
                ),
                state_augmentation_ratio=0.0,
                state_bins=state_bins,
                inference_mode=True,
            )

        if is_dataset_v2:
            max_seq_length = self.config.data_config.max_seq_length
            sequence_lengths = inputs["attention_mask"].sum(dim=1)
            overlong = sequence_lengths > max_seq_length
            if bool(overlong.any().item()):
                rows = overlong.nonzero(as_tuple=True)[0].tolist()
                details = {row: int(sequence_lengths[row].item()) for row in rows}
                raise ValueError(
                    "Inference input exceeds the training max_seq_length without "
                    "truncation: "
                    f"limit={max_seq_length}, rows={details}, "
                    f"images={len(image_inputs)}, "
                    f"action_horizon={self.config.action_horizon}"
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
        text_type = getattr(self.config.data_config, "text_processor_type", "")
        text_params = getattr(self.config.data_config, "text_processor_params", {})
        dataset_text_processor = self._get_dataset_v2_text_processor(text_type)
        prologue = self._get_action_prologue(
            text_params, dataset_text_processor
        )
        user_request = f"{self.role_start_symbol}user\nObservation:"
        for cam_name in self.cam_names:
            display_name = (
                dataset_text_processor.cam_name_mapping.get(cam_name, cam_name)
                if dataset_text_processor is not None
                else _CAM_NAME_MAPPING[cam_name]
            )
            user_request += (
                f" {display_name}: {self.vision_start_symbol}"
                f"{self.image_pad_symbol}{self.vision_end_symbol}"
            )
        user_request += f"\n{self._instruction_label()}:"
        text_prompt = f"\nPredict the next action in robot action.\nProprioception: {self.propri_symbol}\n"
        user_message = (
            f"{user_request} {instruction}{text_prompt}{self.role_end_symbol}\n"
        )
        placeholder_seq = self.tokenizer_mixin.get_placeholder_for_dllm()
        ar_token = "".join(placeholder_seq) + "<|im_end|>\n"
        assistant_message = f"{self.role_start_symbol}assistant\n{ar_token}"
        flow_action = f"{self.action_symbol * self.config.action_horizon}"

        prefix_text = prologue + user_message + assistant_message
        if isinstance(self.config.data_config, InferenceDataConfig):
            prefix_text = replace_state_schema_placeholder(
                prefix_text, self.config.data_config
            )
        postfix_text = flow_action

        return prefix_text, postfix_text

    @timer
    def get_text_for_action(self, instruction):
        text_type = getattr(self.config.data_config, "text_processor_type", "")
        text_params = getattr(self.config.data_config, "text_processor_params", {})
        dataset_text_processor = self._get_dataset_v2_text_processor(text_type)
        prologue = self._get_action_prologue(
            text_params, dataset_text_processor
        )
        user_request = f"{self.role_start_symbol}user\nObservation:"
        for cam_name in self.cam_names:
            if dataset_text_processor is not None:
                display_name = dataset_text_processor.cam_name_mapping.get(
                    cam_name, cam_name
                )
            else:
                display_name = _CAM_NAME_MAPPING[cam_name]
            user_request += (
                f" {display_name}: {self.vision_start_symbol}"
                f"{self.image_pad_symbol}{self.vision_end_symbol}"
            )
        text_prompt = f"\nPredict the next action in robot action.\nProprioception: {self.propri_symbol}\n"

        if text_type == "wallx_task_instruction":
            if isinstance(instruction, Mapping):
                task = self._sanitize_runtime_instruction(
                    instruction.get("task", ""), text_params
                )
                subtask = self._sanitize_runtime_instruction(
                    instruction.get("subtask", instruction.get("detail", "")),
                    text_params,
                )
            else:
                task = self._sanitize_runtime_instruction(instruction, text_params)
                subtask = ""
            instruction_lines = []
            if task:
                instruction_lines.append(f"Task: {task}")
            if subtask:
                instruction_lines.append(f"Instruction: {subtask}")
            instruction_text = "\n".join(instruction_lines)
            user_message = (
                f"{user_request}\n{instruction_text}{text_prompt}"
                f"{self.role_end_symbol}\n"
            )
        else:
            user_message = (
                f"{user_request}\n{self._instruction_label()}: {instruction}{text_prompt}"
                f"{self.role_end_symbol}\n"
            )
        assistant_message = f"{self.role_start_symbol}assistant\n"
        configured_horizon = int(
            text_params.get("action_chunk_size", self.config.action_horizon)
        )
        if configured_horizon != self.config.action_horizon:
            raise ValueError(
                "text processor action_chunk_size does not match inference "
                f"action_horizon: {configured_horizon} != {self.config.action_horizon}"
            )
        # Dataset V2 strips <|action_ar|><|im_end|> when no ground-truth AR
        # action is present. Online flow inference follows that exact branch.
        flow_action = f"{self.action_symbol * configured_horizon}"

        prefix_text = prologue + user_message + assistant_message
        if isinstance(self.config.data_config, InferenceDataConfig):
            prefix_text = replace_state_schema_placeholder(
                prefix_text, self.config.data_config
            )
        postfix_text = flow_action

        return prefix_text, postfix_text

    def _configured_system_prompt(self):
        return (self.config.train_config.get("data") or {}).get("system_prompt")

    def _instruction_label(self):
        data = self.config.train_config.get("data") or {}
        return data.get("instruction_label") or "Instruction"

    def _get_action_prologue(self, text_params, dataset_text_processor):
        """Build the shared flow/AR/RTC/DLLM inference system prompt."""
        system_prompt = self._configured_system_prompt()
        if system_prompt:
            return (
                f"{self.role_start_symbol}system\n"
                f"{system_prompt}{self.role_end_symbol}\n"
            )
        use_embodied_ratio = text_params.get(
            "use_embodied_prompt_ratio",
            self.config.train_config["data"].get(
                "use_embodied_system_prompt_ratio", 0
            ),
        )
        if float(use_embodied_ratio) <= 0:
            return (
                f"{self.role_start_symbol}system\n"
                f"You are a helpful assistant.{self.role_end_symbol}\n"
            )

        prompt_dataset_family = resolve_prompt_dataset_family(self.norm_key)
        cam_name_mapping = {cam_name: cam_name for cam_name in self.cam_names}
        if dataset_text_processor is not None:
            embodiment_type = getattr(self.config, "embodiment_type", None)
            if embodiment_type is None:
                embodiment_type = resolve_embodiment_type(
                    getattr(self.config, "robot_type", "desktop"),
                    prompt_dataset_family,
                )
            return dataset_text_processor._build_embodied_prologue(
                prompt_dataset_family,
                cam_name_mapping,
                str(getattr(self.config, "robot_id", "")),
                "",
                embodiment_type=embodiment_type,
            )
        return get_prologue_with_embodied_information(
            dataset_name=prompt_dataset_family,
            cam_mapping=cam_name_mapping,
            robot_id=self.config.robot_id,
            uid="",
            config=self.config.data_config,
        )

    @staticmethod
    def _sanitize_runtime_instruction(value, text_params):
        words = str(value or "").strip().split()
        max_words = int(text_params.get("max_instruction_words", 256))
        if len(words) > max_words:
            raise ValueError(
                "Runtime instruction exceeds the Dataset V2 text processor "
                f"limit: words={len(words)}, max_instruction_words={max_words}"
            )
        return " ".join(words)

    def _get_dataset_v2_text_processor(self, text_type):
        """Public builds have no Dataset V2 text processor; use generic prompts."""
        del text_type
        return None

    @timer
    def get_text_for_subtask_generation(self, instruction):
        system_prompt = (
            self._configured_system_prompt() or "You are a helpful assistant."
        )
        prologue = f"{self.role_start_symbol}system\n{system_prompt}{self.role_end_symbol}\n"
        user_request = f"{self.role_start_symbol}user\nObservation:"
        for cam_name in self.cam_names:
            user_request += f" {_CAM_NAME_MAPPING[cam_name]}: {self.vision_start_symbol}{self.image_pad_symbol}{self.vision_end_symbol}"
        user_request += f"\n{self._instruction_label()}:"
        text_prompt = "\nPredict the next action in language.\n"
        user_message = (
            f"{user_request} {instruction}{text_prompt}{self.role_end_symbol}\n"
        )
        assistant_message = f"{self.role_start_symbol}assistant\n"

        prefix_text = prologue + user_message + assistant_message
        postfix_text = ""

        return prefix_text, postfix_text

    def _observation_images(self, observation):
        """Model images of one observation, in prompt placeholder order."""
        return self._resize_camera_images(observation)

    def _resize_camera_images(self, observation):
        if getattr(self.config, "image_resize", "training") == "serving":
            return self._resize_images_serving(observation)
        if ENABLE_GPU_RESIZE:
            return self._resize_images_gpu(observation)
        if ENABLE_FAST_PREPROCESS:
            return self._resize_images_fast(observation)
        return self._resize_images(observation)

    def _action_prompt(self, observation, instruction):
        """(prefix, postfix) text of a flow-action request for ``observation``."""
        return self.get_text_for_action(instruction)

    def _resize_images_serving(self, observation):
        """Reproduce the deployed preprocessing exactly.

        The in-house server resizes twice -- once to the configured camera
        resolution, once to the ``smart_resize`` grid -- with Pillow's default
        (bicubic) filter, then hands the GPU tensor to the image processor. A
        single resize from the original image is the training-time behaviour and
        is what ``image_resize="training"`` does, but the two differ by a grey
        level or two, which is enough to move a contact-rich rollout. Serving a
        checkpoint that was evaluated this way therefore needs this path.
        """
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
                if orig_width > orig_height:
                    new_width = target_size
                    new_height = int(target_size * orig_height / orig_width)
                else:
                    new_height = target_size
                    new_width = int(target_size * orig_width / orig_height)
                img_pil = img_pil.resize((new_width, new_height))

            current_width, current_height = img_pil.size
            resized_height, resized_width = qwen_smart_resize(
                current_height,
                current_width,
                factor=self.config.data_config.image_factor,
                min_pixels=self.config.data_config.min_pixels,
                max_pixels=self.config.data_config.max_pixels,
            )
            resized_img = img_pil.resize((resized_width, resized_height))
            image_inputs.append(
                torch.from_numpy(np.array(resized_img)).to(self.config.model_device)
            )
        return image_inputs

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
                if orig_width > orig_height:
                    current_width = target_size
                    current_height = int(target_size * orig_height / orig_width)
                else:
                    current_height = target_size
                    current_width = int(target_size * orig_width / orig_height)
            else:
                current_width, current_height = orig_width, orig_height

            use_dataset_resize = (
                getattr(self.config.data_config, "dataset_type", "")
                == "x2robot_v2"
                or getattr(self.config.data_config, "vision_processor_type", "")
                in ("qwen3", "qwen3_5")
            )
            resize_fn = (
                dataset_v2_smart_resize if use_dataset_resize else qwen_smart_resize
            )
            resized_height, resized_width = resize_fn(
                current_height,
                current_width,
                factor=self.config.data_config.image_factor,
                min_pixels=self.config.data_config.min_pixels,
                max_pixels=self.config.data_config.max_pixels,
            )
            # Dataset V2 validation performs one resize from the original image
            # using torchvision BICUBIC. PIL's BICUBIC path is equivalent here
            # and avoids the old two-stage interpolation.
            resized_img = img_pil.resize(
                (resized_width, resized_height), Image.Resampling.BICUBIC
            )
            if not self._passes_pil_images():
                resized_img = torch.from_numpy(np.array(resized_img)).to(
                    self.config.model_device
                )
            image_inputs.append(resized_img)

        return image_inputs

    def _resize_images_gpu(self, observation):
        """GPU counterpart of ``_resize_images``.

        Identical target sizing (``dataset_v2_smart_resize`` / ``qwen_smart_resize``)
        so ``image_grid_thw`` is unchanged; only the resample runs on the GPU via
        ``F.interpolate(mode="bicubic", antialias=True)`` instead of PIL BICUBIC.
        The GPU bicubic kernel is not bit-identical to PIL, so pixel_values differ
        at the ~1 gray-level scale (quantified in the verification report). The
        result is returned as a PIL image so the downstream ``preprocesser_call``
        path is byte-for-byte the same as the CPU route.
        """
        import torch.nn.functional as F

        device = self.config.model_device
        is_qwen3 = getattr(self.config.data_config, "vision_processor_type", "") in (
            "qwen3",
            "qwen3_5",
        )
        use_dataset_resize = (
            getattr(self.config.data_config, "dataset_type", "") == "x2robot_v2"
            or is_qwen3
        )
        resize_fn = (
            dataset_v2_smart_resize if use_dataset_resize else qwen_smart_resize
        )

        image_inputs = []
        for key in self.cam_names:
            if key not in observation:
                continue
            current_obs = observation[key]
            if isinstance(current_obs, np.ndarray):
                arr = current_obs
            elif isinstance(current_obs, Image.Image):
                arr = np.asarray(current_obs)
            else:
                raise ValueError(f"Unsupported image type: {type(current_obs)}")

            orig_height, orig_width = arr.shape[0], arr.shape[1]
            target_size = self.config.data_config.resolution.get(key, -1)
            if target_size != -1:
                if orig_width > orig_height:
                    current_width = target_size
                    current_height = int(target_size * orig_height / orig_width)
                else:
                    current_height = target_size
                    current_width = int(target_size * orig_width / orig_height)
            else:
                current_width, current_height = orig_width, orig_height

            resized_height, resized_width = resize_fn(
                current_height,
                current_width,
                factor=self.config.data_config.image_factor,
                min_pixels=self.config.data_config.min_pixels,
                max_pixels=self.config.data_config.max_pixels,
            )

            # uint8 HxWxC -> 1xCxHxW float on GPU, single bicubic resample from
            # the original image (matches the CPU one-shot PIL resize), then back
            # to uint8. antialias=True approximates PIL's antialiased BICUBIC.
            t = torch.from_numpy(np.ascontiguousarray(arr)).to(
                device, non_blocking=True
            )
            t = t.permute(2, 0, 1).unsqueeze(0).float()
            t = F.interpolate(
                t,
                size=(resized_height, resized_width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            t = t.round_().clamp_(0, 255).to(torch.uint8)
            resized_arr = t.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
            resized_img = Image.fromarray(resized_arr)

            if not self._passes_pil_images():
                resized_img = torch.from_numpy(np.array(resized_img)).to(device)
            image_inputs.append(resized_img)

        return image_inputs

    def _passes_pil_images(self):
        if getattr(self.config, "image_resize", "training") == "serving":
            return False
        qwen_types = ("qwen3", "qwen3_5")
        return (
            getattr(self.config.data_config, "vision_processor_type", "")
            in qwen_types
            or self.config.train_config.get("model_type") in qwen_types
        )

    def _resize_images_fast(self, observation):
        if (
            self._passes_pil_images()
            or getattr(self.config.data_config, "is_dino_vision", False)
        ):
            # Keep the Dataset V2/Qwen3.5 path pixel-consistent. The cv2 fast
            # path has different cubic coefficients and used a hard-coded
            # camera list.
            return self._resize_images(observation)

        import cv2

        image_inputs = []
        for key in self.cam_names:
            if key not in observation:
                continue
            current_obs = observation[key]
            orig_height, orig_width, _ = current_obs.shape

            target_size = self.config.data_config.resolution.get(key, -1)
            current_width, current_height = orig_width, orig_height
            if target_size != -1:
                if orig_width > orig_height:
                    new_width = target_size
                    new_height = int(target_size * orig_height / orig_width)
                else:
                    new_height = target_size
                    new_width = int(target_size * orig_width / orig_height)
                current_width = new_width
                current_height = new_height

            resized_height, resized_width = qwen_smart_resize(
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
        self.logger.info("generating flow action")
        self.logger.info(f"flow action instruction: {instruction}")

        prefix_text, postfix_text = self._action_prompt(observation, instruction)
        model_input = self.construct_model_input(
            [observation], [prefix_text], [postfix_text]
        )

        with ScopeTimer("generate_flow_action"):
            model_output = self._generate_flow_from_model_input(model_input)

        self.logger.info("flow action generated")

        model_output["robot_state_action_data"] = observation["robot_state_action_data"]
        model_output["robot_state_action_data"].save_action_data(
            model_output["predict_action"]
        )
        self.logger.info("saved flow action to robot_state_action_data")
        return model_output

    def infer_flow_action_rtc(
        self,
        observation,
        instruction,
        prev_action_chunk,
        inference_delay: int,
        num_steps=None,
    ):
        """Flow inference for RTC: the first ``inference_delay`` steps are pinned to
        ``prev_action_chunk``.

        Args:
            observation: observation dict, same format as ``infer_flow_action``
            instruction: text instruction
            prev_action_chunk: np.ndarray [H, D] or [1, H, D] in the relative+6D
                layout, unnormalized (the policy converts it from the absolute
                protocol format)
            inference_delay: int, number of leading steps pinned to the prefix
            num_steps: int | None, denoising steps; None uses
                config.num_inference_timesteps
        """
        self.logger.info(
            f"generating RTC flow action, delay={inference_delay}, num_steps={num_steps}"
        )
        self.logger.info(f"RTC instruction: {instruction}")

        prefix_text, postfix_text = self._action_prompt(observation, instruction)
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

        prefix = torch.as_tensor(prev_action_chunk, dtype=torch.float32)
        if prefix.dim() == 2:
            prefix = prefix.unsqueeze(0)  # [1, L, D]
        if prefix.dim() != 3:
            # ValueError (not assert): stays active under `python -O`, where a
            # bad shape would otherwise slip through into confusing downstream
            # errors.
            raise ValueError(
                f"prev_action_chunk must be [L,D] or [1,L,D], "
                f"got shape={tuple(prefix.shape)}"
            )
        _na_min = self.normalizer_action.min[model_input["dataset_names"][0]]
        _chunk_size = int(_na_min.shape[0]) if _na_min.dim() == 2 else 0
        _L = prefix.shape[1]
        if _na_min.dim() == 2 and _L != _chunk_size:
            if _L < _chunk_size:
                _pad = torch.zeros(
                    (prefix.shape[0], _chunk_size - _L, prefix.shape[2]),
                    dtype=prefix.dtype,
                )
                prefix_in = torch.cat([prefix, _pad], dim=1)
            else:
                prefix_in = prefix[:, :_chunk_size]
            prefix = self.normalizer_action.normalize_data(
                prefix_in, model_input["dataset_names"]
            )[:, :_L].to(model_input["input_ids"].device)
        else:
            prefix = self.normalizer_action.normalize_data(
                prefix, model_input["dataset_names"]
            ).to(model_input["input_ids"].device)

        # Align the RTC prefix with the training distribution for the dims the
        # model masks on this embodiment (desktop: velocity_decomposed/height/
        # head_actions). The 14-dim absolute-EE RTC protocol carries no data for
        # those dims, so ``transform_abs_chunk_to_relative`` left them raw 0 and
        # the normalizer above mapped raw 0 to normalize(0) — which is far from
        # 0 for some dims (e.g. x2_normal height -> -1.0, head_pitch -> +0.5).
        # Training instead pins the prefix to the normalized action_chunk, whose
        # masked dims are set to exactly 0.0 by ``nan_to_num(0.0)`` (dof_mask=0,
        # see WallxEpilogueProcessor._prepare_actions). Multiplying by dof_mask
        # reproduces that: active dims (real EE) are untouched, masked dummy dims
        # are pinned to 0.0 in normalized space. No-op on embodiments that keep
        # these dims active (dof_mask=1), e.g. turtle/ex001.
        pinned_masked_dims = False
        dof_mask_t = model_input.get("dof_mask")
        if dof_mask_t is not None:
            dm = dof_mask_t.to(device=prefix.device, dtype=prefix.dtype)
            # Normalize dof_mask to [B_or_1, 1, D] so it broadcasts over the
            # prefix time axis PER SAMPLE. dof_mask carries no time axis: it is
            # (D,) or, per construct_model_input, (B, D) (each obs's (D,) stacked
            # on dim 0). A bare unsqueeze(0) on a (B, D) mask would wrongly treat
            # the batch as time and collapse to sample 0's mask for the whole
            # batch — fine only at batch==1 (RTC's per-request path), wrong for a
            # heterogeneous batch. Insert the time axis instead.
            if dm.dim() == 1:  # (D,) -> broadcast over batch & time
                dm = dm.view(1, 1, -1)
            elif dm.dim() == 2:  # (B, D) -> (B, 1, D): per-sample, over time
                dm = dm.unsqueeze(1)
            # dof_mask is constant along the time axis, so a single step
            # broadcasts over the full prefix length regardless of horizon.
            if dm.dim() == 3 and dm.shape[-1] == prefix.shape[-1]:
                prefix = prefix * dm[:, :1, :]
                pinned_masked_dims = True
        if not pinned_masked_dims:
            # Surface the silent skip: without this pin the masked non-EE dims
            # (velocity/height/head) stay at normalize(raw 0) — OOD for
            # embodiments that mask them — silently reintroducing the bug the
            # pin fixes. Not a hard error (full-dof embodiments legitimately
            # have no masked dims), but must not pass unnoticed.
            self.logger.warning(
                "RTC prefix: dof_mask missing or shape-mismatched "
                "(dof_mask=%s, prefix_dim=%d) — masked non-EE dims NOT pinned "
                "to 0; prefix may be OOD if this embodiment masks "
                "velocity/height/head.",
                None if dof_mask_t is None else tuple(dof_mask_t.shape),
                prefix.shape[-1],
            )

        inference_delay = int(max(0, min(inference_delay, self.config.action_horizon)))
        num_inference_timesteps = (
            int(num_steps) if num_steps else self.config.num_inference_timesteps
        )

        with ScopeTimer("generate_flow_action_rtc"), self._inference_autocast():
            model_output = self.model.generate_flow_action(
                action_horizon=self.config.action_horizon,
                action_dim=self.config.action_dim,
                num_inference_timesteps=num_inference_timesteps,
                padding_action=padding_action,
                prefix_actions=prefix,
                inference_delay=inference_delay,
                **model_input,
            )

        self.logger.info("RTC flow action generated")

        model_output["robot_state_action_data"] = observation["robot_state_action_data"]
        model_output["robot_state_action_data"].save_action_data(
            model_output["predict_action"]
        )
        return model_output

    def _generate_flow_from_model_input(self, model_input):
        generate_kwargs = {
            "action_horizon": self.config.action_horizon,
            "action_dim": self.config.action_dim,
            "num_inference_timesteps": self.config.num_inference_timesteps,
            **model_input,
        }
        if self.model_type != "qwen3_5":
            padding_list = []
            for dataset_name in model_input["dataset_names"]:
                padding_list.append(
                    torch.zeros_like(self.normalizer_action.delta[dataset_name])
                    .unsqueeze(0)
                    .to("cpu")
                )
            padding = torch.cat(padding_list, dim=0)
            generate_kwargs["padding_action"] = self.normalizer_action.normalize_data(
                padding, model_input["dataset_names"]
            ).to(model_input["input_ids"].device)
        with self._inference_autocast():
            return self.model.generate_flow_action(**generate_kwargs)

    def _inference_autocast(self):
        """bf16 autocast for Qwen3.5 on CUDA, as in training.

        The model intentionally keeps selected normalization parameters in
        fp32, so invoking it without autocast produces a bf16/float LayerNorm
        dtype mismatch.
        """
        # The reference serving stack runs the fp32 weights without autocast;
        # WALLX_INFERENCE_AUTOCAST=0 reproduces that exactly.
        if os.environ.get("WALLX_INFERENCE_AUTOCAST", "1") == "0":
            return contextlib.nullcontext()
        device_type = str(self.config.model_device).split(":", 1)[0]
        if self.model_type == "qwen3_5" and device_type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    @staticmethod
    def _merge_model_inputs(model_inputs):
        """Merge independently preprocessed, shape-compatible samples."""
        merged = {}
        for key in model_inputs[0]:
            values = [item[key] for item in model_inputs]
            if all(value is None for value in values):
                merged[key] = None
            elif all(torch.is_tensor(value) for value in values):
                merged[key] = torch.cat(values, dim=0)
            elif key in ("dataset_names", "image_size"):
                merged[key] = [entry for value in values for entry in value]
            elif all(value == values[0] for value in values):
                merged[key] = values[0]
            else:
                raise ValueError(f"Cannot merge model input field {key!r}")
        return merged

    def infer_flow_action_batch(self, observations, instructions):
        """Batched flow action inference.

        - observations: List[Dict], each in the single-sample observation format
        - instructions: List[str | Mapping]; a Mapping uses
          {"task": str, "subtask": str} (the legacy "detail" key is accepted)

        Returns List[model_output] with one entry per sample.
        """
        assert len(observations) == len(
            instructions
        ), "observations and instructions must have the same length"
        batch_size = len(observations)

        if self.model_type == "qwen3_5":
            # Qwen3.5's cached flow path requires every row to begin its action
            # block at the same position and to have no right-padding after it.
            # Preprocess samples independently, bucket identical layouts, then
            # restore caller order. This preserves Dataset V2 positions instead
            # of inserting artificial padding between prompt and action.
            buckets = defaultdict(list)
            for index, (observation, instruction) in enumerate(
                zip(observations, instructions)
            ):
                prefix_text, postfix_text = self._action_prompt(
                    observation, instruction
                )
                sample_input = self.construct_model_input(
                    [observation], [prefix_text], [postfix_text]
                )
                action_positions = sample_input["moe_token_types"][0].nonzero(
                    as_tuple=True
                )[0]
                if action_positions.numel() == 0:
                    raise ValueError(f"sample {index} contains no flow action tokens")
                layout = (
                    int(action_positions[0].item()),
                    int(sample_input["input_ids"].shape[1]),
                )
                buckets[layout].append((index, sample_input))

            predicted_by_index = {}
            with ScopeTimer("generate_flow_action_batch"):
                for entries in buckets.values():
                    batch_inputs = self._merge_model_inputs(
                        [sample_input for _, sample_input in entries]
                    )
                    batch_output = self._generate_flow_from_model_input(batch_inputs)
                    for row, (index, _) in enumerate(entries):
                        predicted_by_index[index] = batch_output["predict_action"][
                            row : row + 1
                        ]
            predict_action = torch.cat(
                [predicted_by_index[index] for index in range(batch_size)], dim=0
            )
        else:
            prefix_list = []
            postfix_list = []
            for observation, instruction in zip(observations, instructions):
                prefix_text, postfix_text = self._action_prompt(
                    observation, instruction
                )
                prefix_list.append(prefix_text)
                postfix_list.append(postfix_text)
            batch_inputs = self.construct_model_input(
                observations, prefix_list, postfix_list
            )
            with ScopeTimer("generate_flow_action_batch"):
                model_output = self._generate_flow_from_model_input(batch_inputs)
            predict_action = model_output["predict_action"]

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

    def infer_ar_action(
        self,
        observation,
        instruction,
        max_new_tokens=None,
        do_sample=False,
        temperature=1.0,
    ):
        self.logger.info("generating AR action")
        self.logger.info(f"AR action instruction: {instruction}")

        prefix_text, _ = self.get_text_for_action(instruction)
        model_input = self.construct_model_input([observation], [prefix_text], [""])
        prefix_length = int(model_input["attention_mask"][0].sum().item())

        gen_kwargs = dict(model_input)
        gen_kwargs.update(
            action_horizon=self.config.action_horizon,
            action_dim=self.config.action_dim,
            robot_type_id=self.robot_type_id,
            prefix_length=prefix_length,
            do_sample=bool(do_sample),
            temperature=float(temperature),
        )
        if max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = int(max_new_tokens)

        with self._inference_autocast():
            model_output = self.model.generate_ar_action(**gen_kwargs)
        self.logger.info("AR action generated")

        decode_ok = model_output.get("decode_success")
        if decode_ok is not None and not all(bool(x) for x in decode_ok):
            raise RuntimeError(
                f"AR action decode failed (decode_success={decode_ok}); refusing "
                "to serialize a fallback pose."
            )

        got_dim = int(model_output["predict_action"].shape[-1])
        assert got_dim == self.config.action_dim, (
            f"AR predict_action last dim {got_dim} != action_dim "
            f"{self.config.action_dim} (=sum(dof_config)); save_action_data "
            "slicing would be wrong."
        )
        self.logger.info(
            "ar token_accuracy=%s decode_success=%s",
            model_output.get("token_accuracy"),
            decode_ok,
        )

        model_output["robot_state_action_data"] = observation["robot_state_action_data"]
        model_output["robot_state_action_data"].save_action_data(
            model_output["predict_action"]
        )
        self.logger.info("saved AR action to robot_state_action_data")
        return model_output

    def infer_subtask(self, observation, instruction):
        self.logger.info("generating subtask")
        self.logger.info(f"subtask instruction: {instruction}")

        prefix_text, postfix_text = self.get_text_for_subtask_generation(instruction)
        model_input = self.construct_model_input([observation], [prefix_text], [""])

        model_output = self.model.generate_text(**model_input)
        subtask = model_output["predict_output_text"][0].split("<|im_end|>")[0].strip()
        self.logger.info(f"subtask generated: {subtask}")
        return subtask

    def infer_vqa(self, observation, instruction):
        self.logger.info("generating VQA answer")

        if isinstance(observation["multi_modal"], list):
            orig_size = observation["multi_modal"][0].size
        else:
            orig_size = observation["multi_modal"].size

        prefix_text = instruction
        model_input = self.construct_model_input([observation], [prefix_text], [""])

        model_output = self.model.generate_text(**model_input)
        answer = model_output["predict_output_text"][0].split("<|im_end|>")[0].strip()
        self.logger.info(f"VQA answer generated: {answer}")
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
        assert self.model_type in ["qwen2_5"], "DLLM inference supports qwen2_5 only"
        prefix_text, postfix_text = self.get_text_for_dllm_action(instruction)
        model_input = self.construct_model_input(
            [observation], [prefix_text], [postfix_text], pad_prefix=True
        )
        model_input = self.model.update_infer_dllm_position_mask(model_input)
        total_ar_step = self.tokenizer_mixin.inference_ar_steps_for_dllm
        cnt = 0
        while cnt < 3:
            model_output = self.model.generate_dllm_action(
                action_horizon=self.config.action_horizon,
                action_dim=self.config.action_dim,
                ar_action_dim=self.config.ar_action_dim,
                num_inference_timesteps=self.config.num_inference_timesteps,
                use_ar_action=use_ar_action,
                total_ar_step=total_ar_step,
                robot_type_id=self.robot_type_id,
                **model_input,
            )
            if model_output["predict_action"] is not None:
                break
            cnt += 1
            self.logger.warning(f"DLLM action generation failed, retry {cnt}")
        self.logger.info("DLLM action generated")

        model_output["robot_state_action_data"] = observation["robot_state_action_data"]
        model_output["robot_state_action_data"].save_action_data(
            model_output["predict_action"]
        )
        self.logger.info("saved DLLM action to robot_state_action_data")
        return model_output
