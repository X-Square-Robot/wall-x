"""Processors for the per-patch DINOv3 Qwen3.5 visual tower."""

from __future__ import annotations

import json
import math
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.video_processing_utils import BaseVideoProcessor

from .dino_vision import patchify_image


class DinoImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(
        self,
        patch_size: int = 16,
        merge_size: int = 1,
        image_mean: Optional[list] = None,
        image_std: Optional[list] = None,
        rescale_factor: float = 1.0 / 255.0,
        max_pixels: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = int(patch_size)
        self.image_mean = list(image_mean or [0.485, 0.456, 0.406])
        self.image_std = list(image_std or [0.229, 0.224, 0.225])
        self.rescale_factor = float(rescale_factor)
        self.max_pixels = max_pixels
        self.merge_size = int(merge_size)

    def _size(self, height: int, width: int) -> tuple[int, int]:
        patch = self.patch_size
        # Each merged visual token consumes merge_size x merge_size patches,
        # so both patch-grid dimensions must be divisible by merge_size.
        quantum = patch * self.merge_size
        target_h = max(quantum, round(height / quantum) * quantum)
        target_w = max(quantum, round(width / quantum) * quantum)
        if self.max_pixels and target_h * target_w > self.max_pixels:
            scale = math.sqrt(self.max_pixels / (height * width))
            target_h = max(quantum, int(height * scale / quantum) * quantum)
            target_w = max(quantum, int(width * scale / quantum) * quantum)
        return target_h, target_w

    @staticmethod
    def _to_chw(image) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            tensor = image.detach().float()
            if tensor.ndim == 3 and tensor.shape[0] not in (1, 3):
                tensor = tensor.permute(2, 0, 1)
        else:
            array = np.asarray(
                image.convert("RGB") if hasattr(image, "convert") else image
            )
            tensor = torch.from_numpy(np.array(array, dtype=np.float32))
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(-1).repeat(1, 1, 3)
            tensor = tensor.permute(2, 0, 1)
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        return tensor

    def preprocess(self, images, return_tensors=None, **kwargs) -> BatchFeature:
        if not isinstance(images, (list, tuple)):
            images = [images]
        blocks, grids = [], []
        for image in images:
            tensor = self._to_chw(image)
            height, width = tensor.shape[-2:]
            target_h, target_w = self._size(height, width)
            if (target_h, target_w) != (height, width):
                tensor = F.interpolate(
                    tensor.unsqueeze(0),
                    size=(target_h, target_w),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(0)
            tensor = tensor * self.rescale_factor
            mean = tensor.new_tensor(self.image_mean).view(3, 1, 1)
            std = tensor.new_tensor(self.image_std).view(3, 1, 1)
            patches, (grid_h, grid_w) = patchify_image(
                (tensor - mean) / std, self.patch_size
            )
            blocks.append(patches)
            grids.append([1, grid_h, grid_w])
        return BatchFeature(
            {
                "pixel_values": torch.cat(blocks),
                "image_grid_thw": torch.tensor(grids, dtype=torch.long),
            },
            tensor_type=return_tensors,
        )

    def __call__(self, images=None, **kwargs):
        return self.preprocess(images, **kwargs)


class DinoVideoProcessor(BaseVideoProcessor):
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(
        self, image_processor: DinoImageProcessor, merge_size: int | None = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.image_processor = image_processor
        self.merge_size = int(
            image_processor.merge_size if merge_size is None else merge_size
        )
        self.temporal_patch_size = 1

    def __call__(self, videos=None, **kwargs):
        if videos is None:
            return BatchFeature(data={})
        blocks, grids = [], []
        for video in videos:
            encoded = self.image_processor(images=list(video))
            frame_grids = encoded["image_grid_thw"]
            if not bool((frame_grids[:, 1:] == frame_grids[0, 1:]).all()):
                raise ValueError("all frames in a DINO video must have the same size")
            blocks.append(encoded["pixel_values"])
            grids.append(
                [
                    int(frame_grids.shape[0]),
                    int(frame_grids[0, 1]),
                    int(frame_grids[0, 2]),
                ]
            )
        return BatchFeature(
            {
                "pixel_values_videos": torch.cat(blocks),
                "video_grid_thw": torch.tensor(grids, dtype=torch.long),
            }
        )


def load_dino_qwen35_processor(path: str):
    """Rebuild a processor saved by wall-vlm's pluggable DINO path."""
    with open(os.path.join(path, "processor_config.json")) as file:
        processor_config = json.load(file)
    image_config = processor_config["image_processor"]
    image_processor = DinoImageProcessor(
        **{
            key: image_config[key]
            for key in (
                "patch_size",
                "image_mean",
                "image_std",
                "rescale_factor",
                "max_pixels",
                "merge_size",
            )
            if key in image_config
        }
    )
    tokenizer = AutoTokenizer.from_pretrained(path)
    chat_template = getattr(tokenizer, "chat_template", None)
    template_path = os.path.join(path, "chat_template.jinja")
    if chat_template is None and os.path.isfile(template_path):
        with open(template_path) as file:
            chat_template = file.read()

    try:
        from transformers.models.qwen3_5.processing_qwen3_5 import (
            Qwen3_5Processor as Processor,
        )
    except ImportError:
        from transformers.models.qwen3_vl.processing_qwen3_vl import (
            Qwen3VLProcessor as Processor,
        )
    return Processor(
        image_processor=image_processor,
        video_processor=DinoVideoProcessor(
            image_processor,
            merge_size=processor_config.get("video_processor", {}).get(
                "merge_size", image_processor.merge_size
            ),
        ),
        tokenizer=tokenizer,
        chat_template=chat_template,
    )


def is_dino_processor_checkpoint(path: str) -> bool:
    processor_path = os.path.join(path, "processor_config.json")
    if not os.path.isfile(processor_path):
        return False
    with open(processor_path) as file:
        config = json.load(file)
    return (
        config.get("image_processor", {}).get("image_processor_type")
        == "PluggableImageProcessor"
    )
