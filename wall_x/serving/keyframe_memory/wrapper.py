"""Two-pass keyframe-memory inference on top of :class:`WallxModelWrapper`.

Per request, with the episode's :class:`MemoryState` supplied by the caller:

1. **Selection** (every ``select_stride`` frames): the model sees the task, the
   memory older than the candidate window and ``num_candidates`` frames of the
   recent window, and answers ``<|kf_skip|>`` or which candidates to keep.
   Newly observed candidates it names are appended to memory.
2. **Action**: the ordinary flow-action request, with the memory images
   prepended to the camera images and referenced from the prompt::

       Observation: progress memory: [M_0]: <image> ... now: <cameras>
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import LogitsProcessorList

from wall_x._vendor.harrix.infer.model_wrapper import WallxModelWrapper, move_to_cuda
from wall_x.serving.keyframe_memory.boxes import crop_to_box
from wall_x.serving.keyframe_memory.config import KeyframeMemoryConfig
from wall_x.serving.keyframe_memory.grammar import (
    SelectionGrammar,
    SelectionTokens,
    parse_selection,
)
from wall_x.serving.keyframe_memory.prompts import inject_memory, selection_prompt
from wall_x.serving.keyframe_memory.state import (
    Box,
    Frame,
    KeyframeBank,
    MemoryState,
    Selection,
    SelectionWindow,
)

logger = logging.getLogger(__name__)

# Observation key carrying the memory of one action pass.
MEMORY_KEY = "keyframe_memory"


@dataclass(frozen=True)
class ActionMemory:
    frames: Tuple[Frame, ...] = ()
    short_term: Tuple[Frame, ...] = ()
    # Crop boxes for memory frames (box tasks only).
    boxes: Dict[int, Box] = field(default_factory=dict)
    frame_stamp: Optional[int] = None


@dataclass(frozen=True)
class SelectionReport:
    now: int
    candidates: Tuple[int, ...]
    writable: Tuple[int, ...]
    selection: Selection
    written: Tuple[int, ...]


class KeyframeMemoryModelWrapper(WallxModelWrapper):
    def __init__(self, config, memory_config: KeyframeMemoryConfig):
        super().__init__(config)
        self.memory_config = memory_config
        self.bank = KeyframeBank(memory_config)
        self.selection_tokens = SelectionTokens.from_tokenizer(
            self.processor.tokenizer,
            memory_config.num_candidates,
            boxes=memory_config.box_crop,
        )
        logger.info("keyframe memory: %s", memory_config)

    # ------------------------------------------------------------------
    # Request entry point
    # ------------------------------------------------------------------
    def infer_with_memory(
        self,
        observation: Dict[str, Any],
        instruction: str,
        task_id: str,
        state: MemoryState,
    ) -> Tuple[Dict[str, Any], Optional[SelectionReport]]:
        """Run one request; ``state`` is updated in place by the selection pass."""
        memory, report = self.prepare_memory(instruction, task_id, state)
        output = self.infer_flow_action({**observation, MEMORY_KEY: memory}, instruction)
        return output, report

    def infer_with_memory_batch(
        self,
        requests: List[Tuple[Dict[str, Any], str, str, MemoryState]],
        init_noises: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Optional[SelectionReport]]]:
        """Run several independent requests, one ``(observation, instruction,
        task_id, state)`` each.

        Selection runs request by request (it is greedy and touches only that
        request's ``state``); the flow-action pass runs as one batch. With the
        noise each request would have drawn on its own, every row gets the same
        inputs as :meth:`infer_with_memory`.
        """
        prepared = [
            self.prepare_memory(instruction, task_id, state)
            for _, instruction, task_id, state in requests
        ]
        observations = [
            {**observation, MEMORY_KEY: memory}
            for (observation, _, _, _), (memory, _) in zip(requests, prepared)
        ]
        instructions = [instruction for _, instruction, _, _ in requests]
        outputs = self.infer_flow_action_batch(
            observations, instructions, init_noises=init_noises
        )
        return outputs, [report for _, report in prepared]

    def prepare_memory(
        self, instruction: str, task_id: str, state: MemoryState
    ) -> Tuple[ActionMemory, Optional[SelectionReport]]:
        """Selection pass when due, then the memory the action pass shows."""
        cfg = self.memory_config
        report = None
        memory = ActionMemory(frame_stamp=state.now if cfg.frame_stamp else None)
        if cfg.uses_memory(task_id):
            boxes = cfg.uses_boxes(task_id)
            if self.bank.selection_due(state):
                window = self.bank.window(state)
                selection = self.select(instruction, window, state, boxes=boxes)
                written = self.bank.commit(state, window, selection)
                report = SelectionReport(
                    now=window.now,
                    candidates=tuple(f.index for f in window.candidates),
                    writable=window.writable,
                    selection=selection,
                    written=tuple(written),
                )
            memory = ActionMemory(
                frames=tuple(self.bank.action_memory(state)),
                short_term=tuple(
                    self.bank.short_term(state, *cfg.short_term_taps(task_id))
                ),
                boxes=dict(state.boxes) if boxes else {},
                frame_stamp=memory.frame_stamp,
            )
        return memory, report

    # ------------------------------------------------------------------
    # Selection pass
    # ------------------------------------------------------------------
    @torch.no_grad()
    def select(
        self,
        instruction: str,
        window: SelectionWindow,
        state: MemoryState,
        boxes: bool,
    ) -> Selection:
        cfg = self.memory_config
        tokens = self.selection_tokens
        prompt = selection_prompt(instruction, len(window.memory), len(window.candidates))
        images = [
            self._memory_image(f.image, state.boxes.get(f.index) if boxes else None)
            for f in window.memory
        ] + [self._head_image(f.image) for f in window.candidates]
        inputs = self.processor(
            text=[prompt],
            images=[self._to_numpy(image) for image in images],
            return_tensors="pt",
            padding=False,
        )
        inputs = move_to_cuda(dict(inputs), self.config.model_device)
        prompt_length = inputs["input_ids"].shape[1]
        grammar = SelectionGrammar(tokens, prompt_length, cfg.max_writes, boxes=boxes)
        # Longest answer: write, then per slot the slot, a box opener and four
        # numbers with their separators, then the end token.
        max_new_tokens = 2 + cfg.max_writes * (2 + 4 * 5)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self.model.generate(
                **inputs,
                logits_processor=LogitsProcessorList([grammar]),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                eos_token_id=tokens.stop,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                output_logits=True,
                return_dict_in_generate=True,
            )
        generated = out.sequences[0, prompt_length:].tolist()
        first = out.logits[0][0].float()
        margin = float(first[tokens.write] - first[tokens.skip])
        return parse_selection(tokens, generated, margin)

    # ------------------------------------------------------------------
    # Action pass: memory blocks in the prompt, memory images before cameras
    # ------------------------------------------------------------------
    def _action_prompt(self, observation, instruction):
        prefix, postfix = super()._action_prompt(observation, instruction)
        memory = observation.get(MEMORY_KEY)
        if memory is not None:
            cfg = self.memory_config
            prefix = inject_memory(
                prefix,
                num_memory=len(memory.frames),
                num_short_term=len(memory.short_term),
                frame=memory.frame_stamp,
                label=cfg.memory_label,
                memory_in_prompt=cfg.memory_in_prompt,
            )
        return prefix, postfix

    def _observation_images(self, observation):
        cameras = super()._observation_images(observation)
        memory = observation.get(MEMORY_KEY)
        if memory is None or not self.memory_config.memory_in_prompt:
            return cameras
        extra: List[Any] = [
            self._memory_image(f.image, memory.boxes.get(f.index)) for f in memory.frames
        ]
        extra += [self._head_image(f.image) for f in memory.short_term]
        return extra + cameras

    def _head_image(self, image):
        """A head-camera frame resized exactly like the live head camera."""
        return self._resize_camera_images({"face_view": image})[0]

    def _memory_image(self, image, box: Optional[Box]):
        resized = self._head_image(image)
        if box is None:
            return resized
        if torch.is_tensor(resized):
            cropped = crop_to_box(resized.cpu().numpy(), box)
            return torch.from_numpy(np.ascontiguousarray(cropped)).to(resized.device)
        return crop_to_box(resized, box)

    @staticmethod
    def _to_numpy(image):
        return image.cpu().numpy() if torch.is_tensor(image) else np.asarray(image)
