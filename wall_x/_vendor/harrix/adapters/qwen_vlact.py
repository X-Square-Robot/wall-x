"""Shared adapter base for Qwen-VL action models.

Variant-specific subclasses live in ``harrix.adapters.variants`` and provide
the Wall-X training adapter class via ``_training_adapter``. The shared base
handles checkpoint loading, normalizers, LIBERO observation encoding, prompt
construction, and flow-action inference.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping

import numpy as np
import torch
from PIL import Image

from qwen_vl_utils.vision_process import smart_resize as qwen_smart_resize

from wall_x._vendor.x2robot_utils.text_templates import (
    get_prologue_with_embodied_information,
    preprocesser_call,
)

from wall_x._vendor.harrix.adapters.base import BaseInferAdapter
from wall_x._vendor.harrix.envs.libero_common import decode_chunk, encode_proprio
from wall_x._vendor.harrix.eval_config import EvalConfig
from wall_x._vendor.harrix.infer.infer_config import (
    resolve_embodiment_type,
    resolve_prompt_dataset_family,
)
from wall_x._vendor.harrix.utils.ckpt_load import load_state_dict
from wall_x._vendor.harrix.utils.dataset_v2 import (
    InferenceDataConfig,
    build_dataset_v2_text_processor,
    dataset_v2_smart_resize,
    inject_dataset_v2_state_strings,
    replace_state_schema_placeholder,
    validate_inference_data_semantics,
)
from wall_x._vendor.harrix.utils.normalizer import build_normalizers
from wall_x._vendor.harrix.utils.train_config import (
    build_data_config,
    build_model_config,
    load_train_config_with_ckpt_overlay,
)
from wall_x.trainer.trainer_utils import load_wallx_processors_from_cfg


_SUPPORTED_MODES = {"flow", "ar", "dllm", "vqa", "subtask"}
_IMPLEMENTED_MODES = {"flow"}

# Special tokens used by the Qwen-VL action model.
_ROLE_START = "<|im_start|>"
_ROLE_END = "<|im_end|>"
_VISION_START = "<|vision_start|>"
_VISION_END = "<|vision_end|>"
_IMAGE_PAD = "<|image_pad|>"
_PROPRI = "<|propri|>"
_ACTION = "<|action|>"
logger = logging.getLogger(__name__)


_PUBLIC_CAMERA_LABELS = {
    "face_view": "front view",
    "right_wrist_view": "right wrist view",
    "left_wrist_view": "left wrist view",
}


def _camera_label(cam_name: str) -> str:
    return _PUBLIC_CAMERA_LABELS.get(cam_name, cam_name.replace("_", " "))


def _normalizer_width(normalizer, norm_key: str) -> int | None:
    if normalizer is None or norm_key not in normalizer.delta:
        return None
    return int(normalizer.delta[norm_key].shape[0])


def _normalize_real_prefix(normalizer, tensor, dataset_names):
    if normalizer is None:
        return tensor
    widths = [_normalizer_width(normalizer, name) for name in dataset_names]
    if any(width is None for width in widths) or len(set(widths)) != 1:
        return normalizer.normalize_data(tensor, dataset_names)
    width = widths[0]
    if tensor.shape[-1] == width:
        return normalizer.normalize_data(tensor, dataset_names)
    if tensor.shape[-1] < width:
        raise ValueError(
            f"normalizer width {width} exceeds tensor dim {tensor.shape[-1]}"
        )
    out = tensor.clone()
    out[..., :width] = normalizer.normalize_data(
        tensor[..., :width], dataset_names
    )
    out[..., width:] = 0
    return out

class QwenVLActInferAdapter(BaseInferAdapter):
    """Shared constructor and batched flow inference implementation."""

    # ---- subclass hook ----

    @classmethod
    def _training_adapter(cls):
        """Return the training-side ModelAdapter subclass."""
        raise NotImplementedError(
            f"{cls.__name__} must override _training_adapter()"
        )

    # ---- ctor ----

    def __init__(self, cfg: EvalConfig) -> None:
        mode = cfg.model.action_mode
        if mode not in _SUPPORTED_MODES:
            raise ValueError(
                f"{type(self).__name__}: unknown action_mode={mode!r}, "
                f"supported={sorted(_SUPPORTED_MODES)}"
            )
        if mode not in _IMPLEMENTED_MODES:
            raise NotImplementedError(
                f"{type(self).__name__}: action_mode={mode!r} is not implemented; "
                "currently supported="
                f"{sorted(_IMPLEMENTED_MODES)}"
            )
        self._action_mode = mode
        self._state_drop = bool(getattr(cfg.model, "state_drop", False))

        ta = self._training_adapter()
        device = "cuda"
        self._device = device

        # 1) typed cfg through the single wall_x loader (registers the data
        # backend, applies ckpt overlays).
        typed_cfg, train_config = load_train_config_with_ckpt_overlay(
            cfg.model.train_config_path, cfg.model.checkpoint_path
        )
        self._typed_cfg = typed_cfg

        # 2) data_config. Image resizing needs resolution/image_factor/min/max.
        self._data_config = build_data_config(typed_cfg, train_config)

        # 3) HF model config
        ConfigClass = ta.config_class()
        self._model_config = build_model_config(
            ConfigClass, cfg.model.checkpoint_path, typed_cfg, train_config
        )

        # 4) normalizers, including norm_key fallback.
        # Built before load_wallx_processors: OSS wall-x (refactor_2604) requires
        # an explicit normalizer for the action tokenizer; older builds accept it
        # via the same kwarg.
        normalizer_action, normalizer_propri, resolved_norm_key = build_normalizers(
            cfg.model.checkpoint_path,
            typed_cfg,
            cfg.model.norm_key,
            end_effector_type=getattr(cfg.model, "end_effector_type", None),
        )
        self._normalizer_action = normalizer_action
        self._normalizer_propri = normalizer_propri
        # Keep the bare dataset key for prompt semantics. Only normalization
        # should see the Dataset V2 composite dataset/end-effector key.
        self._prompt_norm_key = cfg.model.norm_key
        self._norm_key = resolved_norm_key

        # 5) processor + tokenizer_mixin (from the typed cfg, so ckpt
        # processor/tokenizer overlays are honored).
        procs = load_wallx_processors_from_cfg(
            typed_cfg, normalizer=normalizer_action, device=device
        )
        self._processor = procs["processor"]
        self._tokenizer_mixin = procs.get("tokenizer_mixin")
        if (
            ta.MODEL_TYPE == "qwen3_5"
            and type(self._processor).__name__ != "Qwen3VLProcessor"
        ):
            raise TypeError(
                "qwen3_5 checkpoint requires Qwen3VLProcessor, got "
                f"{type(self._processor).__module__}.{type(self._processor).__name__}"
            )

        # 6) Public variants run the training adapter's model class.
        ModelClass = ta.inference_model_class()
        self._model_class = ModelClass
        model = ModelClass(self._model_config, self._processor, self._tokenizer_mixin)
        current_vocab = model.get_input_embeddings().weight.shape[0]
        if len(self._processor.tokenizer) > current_vocab:
            model.resize_token_embeddings(len(self._processor.tokenizer))
        model.to_bfloat16_for_selected_params()

        # 7) checkpoint weights and model finalization.
        state_dict = load_state_dict(cfg.model.checkpoint_path, ModelClass)
        msg = model.load_state_dict(state_dict, strict=False)
        logger.info(
            "%s load_state_dict: missing=%s unexpected=%s",
            type(self).__name__,
            len(msg.missing_keys),
            len(msg.unexpected_keys),
        )
        model.set_normalizer(
            copy.deepcopy(normalizer_action),
            copy.deepcopy(normalizer_propri),
        )
        model.eval()
        model.to(device)
        model.to_bfloat16_for_selected_params()
        self._model = model

        # 8) cached runtime fields
        self._cam_names = list(cfg.model.cam_names)
        self._action_horizon = int(cfg.model.action_horizon)
        self._action_dim = sum(typed_cfg.task.dof_config.values())
        self._num_inference_timesteps = 10
        self._robot_id = "10000"
        self._robot_type = cfg.model.robot_type
        self._embodiment_type = resolve_embodiment_type(
            self._robot_type, self._prompt_norm_key
        )
        if isinstance(self._data_config, InferenceDataConfig):
            validate_inference_data_semantics(self._data_config, self._action_horizon)

    # ---- BaseInferAdapter API ----

    @property
    def chunk_horizon(self) -> int:
        return self._action_horizon

    @property
    def action_mode(self) -> str:
        return self._action_mode

    def predict_batch(self, payloads: list[dict]) -> list[np.ndarray]:
        if self._action_mode == "flow":
            return self._flow_batch(payloads)
        raise NotImplementedError(
            f"{type(self).__name__}.predict_batch: action_mode={self._action_mode!r} "
            "is not dispatched"
        )

    # ---- batched flow inference ----

    def _flow_batch(self, payloads: list[dict]) -> list[np.ndarray]:
        # 1) Encode each observation into proprioception, masks, and views.
        observations: list[dict] = []
        instructions: list[str] = []
        noises: list[np.ndarray | None] = []
        any_noise = False
        for p in payloads:
            observations.append(
                encode_proprio(
                    p["observation"], self._typed_cfg.task, self._action_horizon
                )
            )
            instructions.append(p["instruction"])
            n = p.get("noise")
            if n is not None:
                any_noise = True
            noises.append(n)

        # 2) Stack noise when provided; all-None delegates sampling to the model.
        if any_noise:
            if any(n is None for n in noises):
                raise ValueError(
                    "payload noise must be provided for every payload or for none"
                )
            batch_noise = torch.stack(
                [torch.from_numpy(n).to(dtype=torch.float32) for n in noises], dim=0
            )
        else:
            batch_noise = None

        # 3) Prompts and model inputs.
        prefix_list, postfix_list = [], []
        for payload, ins in zip(payloads, instructions):
            prefix, postfix = self._get_flow_prompt(
                ins, robot_id=payload.get("robot_id", self._robot_id)
            )
            prefix_list.append(prefix)
            postfix_list.append(postfix)
        batch_inputs = self._construct_model_input(
            observations, prefix_list, postfix_list
        )

        # 4) Normalized zero action as the flow starting point.
        padding = torch.zeros(
            (
                len(batch_inputs["dataset_names"]),
                1,
                self._action_dim,
            ),
            dtype=torch.float32,
        )
        padding_action = _normalize_real_prefix(
            self._normalizer_action, padding, batch_inputs["dataset_names"]
        ).to(batch_inputs["input_ids"].device)

        # 5) Flow forward.
        generate_kwargs = dict(
            action_horizon=self._action_horizon,
            action_dim=self._action_dim,
            num_inference_timesteps=self._num_inference_timesteps,
            padding_action=padding_action,
            init_noise=batch_noise,
            **batch_inputs,
        )
        model_output = self._generate_flow_action(**generate_kwargs)

        # 6) Decode one action chunk per payload.
        predict_action = model_output["predict_action"]  # (B, H, D_action)
        if isinstance(predict_action, torch.Tensor):
            predict_action = predict_action.detach().cpu().numpy()
        return [
            decode_chunk(predict_action[i : i + 1], self._typed_cfg.task)
            for i in range(len(payloads))
        ]

    def _generate_flow_action(self, **kwargs):
        """Run flow generation under the train-time Qwen3.5 dtype contract."""
        if self._typed_cfg.model_type == "qwen3_5":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return self._model.generate_flow_action(**kwargs)
        return self._model.generate_flow_action(**kwargs)

    # ---- prompt template ----

    def _get_flow_prompt(
        self,
        instruction,
        *,
        robot_id: str | int | None = None,
    ) -> tuple[str, str]:
        """Build the Dataset V2 post-epilogue flow prompt."""
        text_type = getattr(self._data_config, "text_processor_type", "")
        text_params = getattr(self._data_config, "text_processor_params", {})
        text_processor = self._get_dataset_v2_text_processor()
        embodied_ratio = text_params.get(
            "use_embodied_prompt_ratio",
            getattr(self._typed_cfg.data, "use_embodied_system_prompt_ratio", 0),
        )
        raw_data = getattr(self._typed_cfg, "_raw_data", None) or {}
        system_prompt = raw_data.get("system_prompt")
        instruction_label = raw_data.get("instruction_label") or "Instruction"
        if system_prompt:
            prologue = f"{_ROLE_START}system\n{system_prompt}{_ROLE_END}\n"
        elif float(embodied_ratio) > 0:
            prompt_norm_key = resolve_prompt_dataset_family(
                getattr(self, "_prompt_norm_key", self._norm_key)
            )
            cam_name_mapping = {cn: cn for cn in self._cam_names}
            if text_processor is not None:
                embodiment_type = getattr(self, "_embodiment_type", None)
                if embodiment_type is None:
                    embodiment_type = resolve_embodiment_type(
                        getattr(self, "_robot_type", "desktop"), prompt_norm_key
                    )
                prologue = text_processor._build_embodied_prologue(
                    prompt_norm_key,
                    cam_name_mapping,
                    str(robot_id),
                    "",
                    embodiment_type=embodiment_type,
                )
            else:
                prologue = get_prologue_with_embodied_information(
                    dataset_name=prompt_norm_key,
                    cam_mapping=cam_name_mapping,
                    robot_id=robot_id,
                    uid="",
                    config=self._data_config,
                )
        else:
            prologue = f"{_ROLE_START}system\nYou are a helpful assistant.{_ROLE_END}\n"
        user_request = f"{_ROLE_START}user\nObservation:"
        # LeRobot training prompts use the raw camera key unless the training
        # config sets data.camera_name_mapping.
        lerobot_cam_mapping = (
            (self._typed_cfg.data.camera_name_mapping or {})
            if self._typed_cfg.data.dataset_type == "lerobot"
            else None
        )
        for cn in self._cam_names:
            if text_processor is not None:
                display_name = text_processor.cam_name_mapping.get(cn, cn)
            elif lerobot_cam_mapping is not None:
                display_name = lerobot_cam_mapping.get(cn, cn)
            else:
                display_name = _camera_label(cn)
            user_request += (
                f" {display_name}: " f"{_VISION_START}{_IMAGE_PAD}{_VISION_END}"
            )
        text_prompt = (
            f"\nPredict the next action in robot action.\nProprioception: {_PROPRI}\n"
        )
        if text_type == "wallx_task_instruction":
            if isinstance(instruction, Mapping):
                task = self._sanitize_instruction(instruction.get("task", ""))
                subtask = self._sanitize_instruction(
                    instruction.get("subtask", instruction.get("detail", ""))
                )
            else:
                task = self._sanitize_instruction(instruction)
                subtask = ""
            lines = [f"Task: {task}"] if task else []
            if subtask:
                lines.append(f"Instruction: {subtask}")
            instruction_text = "\n".join(lines)
            user_message = (
                f"{user_request}\n{instruction_text}{text_prompt}{_ROLE_END}\n"
            )
        else:
            instruction_text = self._sanitize_instruction(instruction)
            user_message = (
                f"{user_request}\n{instruction_label}: {instruction_text}"
                f"{text_prompt}{_ROLE_END}\n"
            )
        assistant_message = f"{_ROLE_START}assistant\n"
        flow_action = _ACTION * self._action_horizon

        prefix_text = prologue + user_message + assistant_message
        if isinstance(self._data_config, InferenceDataConfig):
            prefix_text = replace_state_schema_placeholder(
                prefix_text, self._data_config
            )
        postfix_text = flow_action
        return prefix_text, postfix_text

    def _get_dataset_v2_text_processor(self):
        if not isinstance(self._data_config, InferenceDataConfig):
            return None
        cached = getattr(self, "_dataset_v2_text_processor", None)
        if cached is None:
            cached = build_dataset_v2_text_processor(self._data_config)
            self._dataset_v2_text_processor = cached
        return cached

    def _sanitize_instruction(self, value) -> str:
        text = " ".join(str(value or "").strip().split())
        max_words = int(
            getattr(self._data_config, "text_processor_params", {}).get(
                "max_instruction_words", 256
            )
        )
        if len(text.split()) > max_words:
            raise ValueError(
                "Runtime instruction exceeds the Dataset V2 text processor "
                f"limit: max_instruction_words={max_words}"
            )
        return text

    # ---- batched model input construction ----

    def _construct_model_input(
        self,
        observations: list[dict],
        prefix_list: list[str],
        postfix_list: list[str],
    ) -> dict:
        """Build model inputs for the flow inference path."""
        batch_size = len(observations)
        dataset_names = [self._norm_key] * batch_size

        # Proprioception and masks are prepared as ndarrays in encode_proprio.
        agent_pos = torch.cat(
            [torch.from_numpy(o["proprioception"]) for o in observations], dim=0
        )
        agent_pos_mask = torch.cat(
            [torch.from_numpy(o["agent_pos_mask"]) for o in observations], dim=0
        )
        dof_mask = torch.cat(
            [torch.from_numpy(o["dof_mask"]) for o in observations], dim=0
        )
        agent_pos = _normalize_real_prefix(
            self._normalizer_propri, agent_pos, dataset_names
        )

        # Resize images per sample.
        image_inputs: list[torch.Tensor | Image.Image] = []
        all_image_sizes: list[tuple[int, int]] = []
        for o in observations:
            for cn in self._cam_names:
                if cn not in o:
                    continue
                image = self._resize_image(o[cn], cn)
                image_inputs.append(image)
                all_image_sizes.append(
                    image.size
                    if isinstance(image, Image.Image)
                    else (image.shape[1], image.shape[0])
                )

        is_dataset_v2 = isinstance(self._data_config, InferenceDataConfig)
        state_keep_dims = (
            self._data_config.epilogue_params.get(
                "state_string_keep_dims_by_dataset", {}
            )
            if is_dataset_v2
            else {}
        )
        use_state_string = (
            self._data_config.use_state_string_representation
            if is_dataset_v2
            else self._typed_cfg.task.use_state_string_representation
        )
        state_was_injected = bool(use_state_string and state_keep_dims)
        if state_was_injected:
            prefix_list = inject_dataset_v2_state_strings(
                prefix_list,
                agent_pos,
                agent_pos_mask,
                dataset_names,
                state_bins=(
                    self._data_config.state_bins
                    if is_dataset_v2
                    else getattr(self._typed_cfg.data, "state_bins", 256)
                ),
                keep_dims_by_dataset=state_keep_dims,
                state_drop=bool(self._state_drop),
            )
        inputs = preprocesser_call(
            processor=self._processor,
            prefix_text=prefix_list,
            postfix_text=postfix_list,
            images=image_inputs,
            videos=None,
            padding=True,
            padding_side=(self._data_config.padding_side if is_dataset_v2 else "left"),
            truncation=not is_dataset_v2,
            return_tensors="pt",
            max_length=None if is_dataset_v2 else 1000,
            pad_to_128_multiple=False,
            pad_prefix_to_same_length=bool(
                self._data_config.epilogue_params.get(
                    "pad_prefix_to_same_length", False
                )
                if is_dataset_v2
                else False
            ),
            norm_state=(
                agent_pos
                if use_state_string and not state_was_injected
                else None
            ),
            agent_pos_mask=agent_pos_mask,
            state_augmentation_prob=0.0,
            state_drop_prob=1.0 if self._state_drop else 0.0,
            state_augmentation_ratio=0.0,
            state_bins=(
                self._data_config.state_bins
                if is_dataset_v2
                else getattr(self._typed_cfg.data, "state_bins", 256)
            ),
            inference_mode=True,
        )

        if is_dataset_v2:
            sequence_lengths = inputs["attention_mask"].sum(dim=1)
            overlong = sequence_lengths > self._data_config.max_seq_length
            if bool(overlong.any().item()):
                rows = overlong.nonzero(as_tuple=True)[0].tolist()
                details = {row: int(sequence_lengths[row].item()) for row in rows}
                raise ValueError(
                    "Inference input exceeds the training max_seq_length without "
                    f"truncation: limit={self._data_config.max_seq_length}, "
                    f"rows={details}"
                )

        action_token_id = self._processor.tokenizer.convert_tokens_to_ids(_ACTION)
        moe_token_types = inputs["input_ids"] == action_token_id

        extra = {
            "proprioception": agent_pos.detach(),
            "agent_pos_mask": agent_pos_mask,
            "dof_mask": dof_mask,
            "image_size": all_image_sizes,
            "moe_token_types": moe_token_types,
            "dataset_names": dataset_names,
        }
        inputs.update(extra)
        return _move_to_device(inputs, self._device)

    # ---- image resizing ----

    def _resize_image(
        self, img: np.ndarray, cam_name: str
    ) -> torch.Tensor | Image.Image:
        """Resize one image with the train-time image config."""
        if isinstance(img, np.ndarray):
            pil = Image.fromarray(img)
        elif isinstance(img, Image.Image):
            pil = img
        else:
            raise ValueError(f"unsupported image type: {type(img)}")
        orig_w, orig_h = pil.size

        is_dataset_v2 = isinstance(self._data_config, InferenceDataConfig)
        if (
            is_dataset_v2
            and self._data_config.resolution
            and cam_name not in self._data_config.resolution
        ):
            raise ValueError(
                f"camera {cam_name!r} is absent from the effective Dataset V2 "
                "vision resolution config"
            )
        target = self._data_config.resolution.get(cam_name, -1)
        if target != -1:
            if orig_w > orig_h:
                new_w, new_h = target, int(target * orig_h / orig_w)
            else:
                new_h, new_w = target, int(target * orig_w / orig_h)
        else:
            new_w, new_h = orig_w, orig_h

        resize_fn = dataset_v2_smart_resize if is_dataset_v2 else qwen_smart_resize
        resized_h, resized_w = resize_fn(
            new_h,
            new_w,
            factor=self._data_config.image_factor,
            min_pixels=self._data_config.min_pixels,
            max_pixels=self._data_config.max_pixels,
        )
        resized = pil.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
        if is_dataset_v2 and self._data_config.vision_processor_type in (
            "qwen3",
            "qwen3_5",
        ):
            return resized
        return torch.from_numpy(np.array(resized)).to(self._device)


def _move_to_device(obj, device):
    """Recursively move tensors inside common containers to ``device``."""
    from transformers import BatchFeature

    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, (dict, BatchFeature)):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_device(v, device) for v in obj)
    return obj
