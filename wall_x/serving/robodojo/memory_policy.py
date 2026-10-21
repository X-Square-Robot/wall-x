"""RoboDojo policy for keyframe-memory checkpoints, with client-held memory.

Each observation carries the episode's memory in the ``kfmem_client_state_v1``
format (see :mod:`wall_x.serving.robodojo.memory_protocol`); the response
returns the updated state next to the actions. Nothing about an episode is
kept on the server except a small fallback cache of box-task boxes for clients
that do not echo them.
"""

from __future__ import annotations

import contextlib
import logging
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, Dict, Optional

import torch

from wall_x.serving.keyframe_memory.config import KeyframeMemoryConfig
from wall_x.serving.robodojo.memory_protocol import (
    MAX_ADVANCE,
    RESPONSE_KEY,
    ClientRequest,
    initial_state,
    parse_request,
    response_state,
)
from wall_x.serving.robodojo.policy import (
    RoboDojoPolicyConfig,
    WallXRoboDojoPolicy,
    _invalid,
)

logger = logging.getLogger(__name__)

_BOX_CACHE_SIZE = 512


class ClientRng:
    """CUDA generator state carried by the client from request to request.

    Flow sampling draws its noise from the CUDA generator. Loading the client's
    state for the request and returning the advanced state makes an episode's
    noise sequence independent of whatever else the server ran in between.
    """

    def __init__(self, device: str):
        self.enabled = str(device).startswith("cuda") and torch.cuda.is_available()
        # The generator of the device the model samples on, not the current one.
        self.device = torch.device(device) if self.enabled else None
        self.initial = self._get() if self.enabled else None

    def _get(self) -> bytes:
        return bytes(torch.cuda.get_rng_state(self.device).numpy())

    def validate(self, raw: Optional[bytes]) -> None:
        if not self.enabled:
            if raw is not None:
                _invalid("CUDA RNG state is unavailable on this model host")
        elif raw is None:
            _invalid("rng_state is required for CUDA client-state mode")
        elif len(raw) != len(self.initial):
            _invalid("rng_state byte length does not match CUDA generator state")

    @contextlib.contextmanager
    def loaded(self, raw: Optional[bytes]):
        """Run with the client's generator; yields a function returning the new state."""
        if not self.enabled:
            yield lambda: None
            return
        saved = torch.cuda.get_rng_state(self.device)
        torch.cuda.set_rng_state(
            torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone(), self.device
        )
        try:
            yield self._get
        finally:
            torch.cuda.set_rng_state(saved, self.device)


class KeyframeMemoryRoboDojoPolicy(WallXRoboDojoPolicy):
    # Selection decisions are per episode; requests are served one at a time.
    supports_batch_infer = False
    # The client advances by ``replan_steps``; the platform still wants the
    # complete chunk, so every request returns the whole prediction horizon.
    returns_full_chunk = True

    def __init__(
        self,
        config: RoboDojoPolicyConfig,
        infer_config=None,
        memory: Optional[KeyframeMemoryConfig] = None,
        memory_overrides: Optional[Dict[str, Any]] = None,
    ):
        if config.replan_steps > MAX_ADVANCE:
            # The client sends every frame since the previous request; the
            # protocol bounds that history at MAX_ADVANCE frames.
            raise ValueError(
                f"keyframe-memory serving supports replan_steps <= {MAX_ADVANCE}, "
                f"got {config.replan_steps}"
            )
        self._memory = memory
        self._memory_overrides = dict(memory_overrides or {})
        self._box_cache: "OrderedDict[str, dict]" = OrderedDict()
        super().__init__(config, infer_config)
        self._rng = ClientRng(self.infer_config.model_device)

    @property
    def memory_config(self) -> KeyframeMemoryConfig:
        return self.model.memory_config

    def _build_model(self, infer_config):
        from wall_x.serving.keyframe_memory.wrapper import KeyframeMemoryModelWrapper

        memory = self._memory or KeyframeMemoryConfig.from_checkpoint(
            infer_config.data_config.epilogue_params, **self._memory_overrides
        )
        return KeyframeMemoryModelWrapper(infer_config, memory)

    # ------------------------------------------------------------------
    # XPolicyLab model interface
    # ------------------------------------------------------------------
    def reset(self) -> Dict[str, Any]:
        self._pending = None
        return initial_state(self._rng.initial)

    def update_obs(self, obs: Mapping[str, Any]) -> None:
        if not isinstance(obs, Mapping):
            _invalid("observation must be a mapping")
        task_id = obs.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            _invalid("observation needs a nonempty 'task_id'")
        request = parse_request(
            obs, max_advance=self.config.replan_steps, config=self.memory_config
        )
        self._rng.validate(request.rng_state)
        observation, instruction = self._convert(obs)
        self._pending = (observation, instruction, task_id, request)

    def get_action(self) -> Dict[str, Any]:
        if self._pending is None:
            raise RuntimeError("get_action called before update_obs")
        observation, instruction, task_id, request = self._pending
        self._pending = None
        boxes = self.memory_config.uses_boxes(task_id)
        request.state.boxes = self._boxes_for(request) if boxes else {}
        with self._rng.loaded(request.rng_state) as export_rng:
            output, report = self.model.infer_with_memory(
                observation, instruction, task_id, request.state
            )
            rng_state = export_rng()
        if report is not None:
            logger.info(
                "[%s] select t=%d %s slots=%s written=%s margin=%+.3f memory=%s",
                request.context_key,
                report.now,
                "write" if report.selection.write else "skip",
                list(report.selection.slots),
                list(report.written),
                report.selection.margin,
                [f.index for f in request.state.memory],
            )
        if boxes:
            self._remember_boxes(request)
        return {
            "actions": self._actions(output["robot_state_action_data"]),
            RESPONSE_KEY: response_state(request, rng_state=rng_state, include_boxes=boxes),
        }

    # ------------------------------------------------------------------
    # Box tasks: boxes come back from the client; the cache serves clients
    # that do not echo ``mem_boxes``.
    # ------------------------------------------------------------------
    def _boxes_for(self, request: ClientRequest) -> dict:
        key = request.context_key
        if request.previous_frame is None:
            self._box_cache.pop(key, None)
        if request.client_boxes is not None:
            return dict(request.client_boxes)
        cached = self._box_cache.get(key, {})
        frames = [f.index for f in request.state.memory]
        missing = [f for f in frames if f not in cached]
        if missing:
            logger.warning(
                "[%s] no boxes for memory frames %s; they stay uncropped", key, missing
            )
        return {f: cached[f] for f in frames if f in cached}

    def _remember_boxes(self, request: ClientRequest) -> None:
        self._box_cache[request.context_key] = dict(request.state.boxes)
        self._box_cache.move_to_end(request.context_key)
        while len(self._box_cache) > _BOX_CACHE_SIZE:
            self._box_cache.popitem(last=False)
