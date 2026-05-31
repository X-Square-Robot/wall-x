"""Shared adapter base for Qwen-VL action models.

Variant-specific subclasses live in ``harrix.adapters.variants`` and provide
the Wall-X training adapter class via ``_training_adapter``. The shared base
handles checkpoint loading, normalizers, LIBERO observation encoding, prompt
construction, and flow-action inference.
"""

from __future__ import annotations

import copy
import logging

import numpy as np
import torch
from PIL import Image

from qwen_vl_utils.vision_process import smart_resize

from wall_x._vendor.x2robot_utils.text_templates import (
    get_prologue_with_embodied_information,
    preprocesser_call,
)

from wall_x._vendor.harrix.adapters.base import BaseInferAdapter
from wall_x._vendor.harrix.envs.libero_common import decode_chunk, encode_proprio
from wall_x._vendor.harrix.eval_config import EvalConfig
from wall_x._vendor.harrix.utils.ckpt_load import (
    load_state_dict,
    resolve_checkpoint_dir,
)
from wall_x._vendor.harrix.utils.normalizer import build_normalizers
from wall_x._vendor.harrix.utils.train_config import (
    build_data_config,
    build_model_config,
    load_train_config_with_ckpt_overlay,
    normalize_train_config_for_inference,
    register_data_backend,
    resolve_state_bins,
    resolve_use_state_string_representation,
)
from wall_x.trainer.trainer_utils import load_wallx_processors


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


def _camera_label(cam_name: str) -> str:
    return cam_name.replace("_", " ")


class QwenVLActInferAdapter(BaseInferAdapter):
    """Shared constructor and batched flow inference implementation."""

    # ---- subclass hook ----

    @classmethod
    def _training_adapter(cls):
        """Return the training-side ModelAdapter subclass."""
        raise NotImplementedError(f"{cls.__name__} must override _training_adapter()")

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

        ta = self._training_adapter()
        device = "cuda"
        self._device = device
        self._checkpoint_path = resolve_checkpoint_dir(cfg.model.checkpoint_path)

        # 1) train_config with checkpoint overlays, then data backend registration.
        train_config = load_train_config_with_ckpt_overlay(
            cfg.model.train_config_path, self._checkpoint_path
        )
        train_config = normalize_train_config_for_inference(
            train_config, cfg.model.train_config_path
        )
        register_data_backend(train_config)
        self._train_config = train_config

        # 2) normalizers first — action_tokenizer setup needs them during processor load.
        normalizer_action, normalizer_propri, resolved_norm_key = build_normalizers(
            self._checkpoint_path, train_config, cfg.model.norm_key
        )
        self._normalizer_action = normalizer_action
        self._normalizer_propri = normalizer_propri
        self._norm_key = resolved_norm_key

        # 3) data_config. Image resizing needs resolution/image_factor/min/max.
        self._data_config = build_data_config(cfg.model.train_config_path, train_config)

        # 4) HF model config
        ConfigClass = ta.config_class()
        self._model_config = build_model_config(
            ConfigClass,
            self._checkpoint_path,
            train_config,
            cfg.model.train_config_path,
        )

        # 5) processor + tokenizer_mixin (may extend vocab via action_tokenizer)
        procs = load_wallx_processors(
            train_config, normalizer=normalizer_action, device=device
        )
        self._processor = procs["processor"]
        self._tokenizer_mixin = procs.get("tokenizer_mixin")
        logger.info(
            "processor vocab size: %s (action_tokenizer_type=%r)",
            len(self._processor.tokenizer),
            train_config.get("action_tokenizer_type"),
        )

        # 6) model
        ModelClass = ta.inference_model_class()
        self._model_class = ModelClass
        model = ModelClass(self._model_config, self._processor, self._tokenizer_mixin)
        model.resize_token_embeddings(len(self._processor.tokenizer))
        model.to_bfloat16_for_selected_params()

        # 7) checkpoint weights and model finalization.
        state_dict = load_state_dict(self._checkpoint_path, ModelClass)
        embed_key = "model.embed_tokens.weight"
        if embed_key in state_dict:
            ckpt_vocab = state_dict[embed_key].shape[0]
            cur_vocab = model.model.embed_tokens.weight.shape[0]
            if cur_vocab != ckpt_vocab:
                logger.info(
                    "resize_token_embeddings from %d to %d to match checkpoint",
                    cur_vocab,
                    ckpt_vocab,
                )
                model.resize_token_embeddings(ckpt_vocab)
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
        if hasattr(model, "load_optimized_weights"):
            model.load_optimized_weights(state_dict)
        self._model = model

        # 8) cached runtime fields
        self._cam_names = list(cfg.model.cam_names)
        self._action_horizon = int(cfg.model.action_horizon)
        self._action_dim = sum(train_config["dof_config"].values())
        self._num_inference_timesteps = 10
        self._robot_id = "10000"

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
                    p["observation"], self._train_config, self._action_horizon
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
        for ins in instructions:
            prefix, postfix = self._get_flow_prompt(ins)
            prefix_list.append(prefix)
            postfix_list.append(postfix)
        batch_inputs = self._construct_model_input(
            observations, prefix_list, postfix_list
        )

        # 4) Normalized zero action as the flow starting point.
        padding_list = []
        for ds_name in batch_inputs["dataset_names"]:
            padding_list.append(
                torch.zeros_like(self._normalizer_action.delta[ds_name])
                .unsqueeze(0)
                .to("cpu")
            )
        padding = torch.cat(padding_list, dim=0)
        padding_action = self._normalizer_action.normalize_data(
            padding, batch_inputs["dataset_names"]
        ).to(batch_inputs["input_ids"].device)

        # 5) Flow forward.
        model_output = self._model.generate_flow_action(
            action_horizon=self._action_horizon,
            action_dim=self._action_dim,
            num_inference_timesteps=self._num_inference_timesteps,
            padding_action=padding_action,
            noise=batch_noise,
            **batch_inputs,
        )

        # 6) Decode one action chunk per payload.
        predict_action = model_output["predict_action"]  # (B, H, D_action)
        if isinstance(predict_action, torch.Tensor):
            predict_action = predict_action.detach().cpu().numpy()
        return [
            decode_chunk(predict_action[i : i + 1], self._train_config)
            for i in range(len(payloads))
        ]

    # ---- prompt template ----

    def _get_flow_prompt(self, instruction: str) -> tuple[str, str]:
        """Build the flow-action prompt."""
        if self._train_config["data"].get("use_embodied_system_prompt_ratio", 0) > 0:
            robot_id = (
                self._robot_id if self._norm_key in ("x2_normal", "ex_normal") else 0
            )
            cam_name_mapping = {cn: cn for cn in self._cam_names}
            prologue = get_prologue_with_embodied_information(
                dataset_name=self._norm_key,
                cam_mapping=cam_name_mapping,
                robot_id=robot_id,
                uid="",
                config=self._data_config,
            )
        else:
            prologue = f"{_ROLE_START}system\nYou are a helpful assistant.{_ROLE_END}\n"
        user_request = f"{_ROLE_START}user\nObservation:"
        for cn in self._cam_names:
            user_request += (
                f" {_camera_label(cn)}: " f"{_VISION_START}{_IMAGE_PAD}{_VISION_END}"
            )
        user_request += "\nInstruction:"
        text_prompt = (
            f"\nPredict the next action in robot action.\nProprioception: {_PROPRI}\n"
        )
        user_message = f"{user_request} {instruction}{text_prompt}{_ROLE_END}\n"
        assistant_message = f"{_ROLE_START}assistant\n"
        flow_action = _ACTION * self._action_horizon

        prefix_text = prologue + user_message + assistant_message
        postfix_text = flow_action
        return prefix_text, postfix_text

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
        if self._normalizer_propri is not None:
            agent_pos = self._normalizer_propri.normalize_data(agent_pos, dataset_names)

        # Resize images per sample.
        image_inputs: list[torch.Tensor] = []
        all_image_sizes: list[tuple[int, int]] = []
        for o in observations:
            for cn in self._cam_names:
                if cn not in o:
                    continue
                tensor = self._resize_image(o[cn], cn)
                image_inputs.append(tensor)
                # Tensor (H, W, C) to PIL-compatible (W, H).
                all_image_sizes.append((tensor.shape[1], tensor.shape[0]))

        inputs = preprocesser_call(
            processor=self._processor,
            prefix_text=prefix_list,
            postfix_text=postfix_list,
            images=image_inputs,
            videos=None,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=1000,
            pad_to_128_multiple=False,
            pad_prefix_to_same_length=False,
            norm_state=(
                agent_pos
                if resolve_use_state_string_representation(self._train_config)
                else None
            ),
            agent_pos_mask=agent_pos_mask,
            state_augmentation_prob=0.0,
            state_drop_prob=0.0,
            state_augmentation_ratio=0.0,
            state_bins=resolve_state_bins(self._train_config),
            inference_mode=True,
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

    def _resize_image(self, img: np.ndarray, cam_name: str) -> torch.Tensor:
        """Resize one image with the train-time image config."""
        if isinstance(img, np.ndarray):
            pil = Image.fromarray(img)
        elif isinstance(img, Image.Image):
            pil = img
        else:
            raise ValueError(f"unsupported image type: {type(img)}")
        orig_w, orig_h = pil.size

        target = self._data_config.resolution.get(cam_name, -1)
        if target != -1:
            if orig_w > orig_h:
                new_w, new_h = target, int(target * orig_h / orig_w)
            else:
                new_h, new_w = target, int(target * orig_w / orig_h)
            pil = pil.resize((new_w, new_h))

        cur_w, cur_h = pil.size
        resized_h, resized_w = smart_resize(
            cur_h,
            cur_w,
            factor=self._data_config.image_factor,
            min_pixels=self._data_config.min_pixels,
            max_pixels=self._data_config.max_pixels,
        )
        resized = pil.resize((resized_w, resized_h))
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
