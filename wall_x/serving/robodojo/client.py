"""Synchronous environment-side adapter for the policy websocket protocol."""

from __future__ import annotations

import asyncio
from typing import Any, cast

from wall_x.serving.robodojo.protocol.client import (
    PolicyEvalClient,
    PolicyEvalClientConfig,
)


class WsModelClient:
    def __init__(
        self,
        *,
        url: str,
        evaluation_id: str,
        trial_id: str,
        action_case_id: str | None = None,
        repeat_index: int | None = None,
        client: Any | None = None,
    ):
        self.action_case_id = action_case_id
        self.trial_id = trial_id
        self.repeat_index = repeat_index
        self._step = 0
        self._latest_obs: Any | None = None
        self._latest_obs_batch: list[Any] | None = None
        self._latest_env_ids: list[int] | None = None
        self._loop = asyncio.new_event_loop()
        self._client = client or PolicyEvalClient(
            PolicyEvalClientConfig(url=url, evaluation_id=evaluation_id)
        )
        hello = self._loop.run_until_complete(self._client.connect(handshake=True))
        capabilities = hello.payload.get("capabilities", []) if hello is not None else []
        self.supports_batch_infer = "batch_infer" in capabilities

    def call(self, func_name: str | None = None, obs: Any = None, **kwargs: Any) -> Any:
        if func_name == "prepare_case":
            if self.action_case_id is None:
                raise ValueError("prepare_case requires action_case_id")
            response = self._loop.run_until_complete(
                self._client.prepare_case(
                    self.action_case_id,
                    case_meta=obs if isinstance(obs, dict) else None,
                )
            )
            return response.payload.get("result")

        if func_name == "reset":
            self._step = 0
            self._latest_obs = None
            self._latest_obs_batch = None
            self._latest_env_ids = None
            response = self._loop.run_until_complete(
                self._client.reset(
                    trial_id=self.trial_id,
                    action_case_id=self.action_case_id,
                    repeat_index=self.repeat_index,
                    payload=obs if isinstance(obs, dict) else None,
                )
            )
            return response.payload.get("result")

        if func_name == "update_obs":
            self._latest_obs = obs
            return None

        if func_name == "get_action":
            observation = obs if obs is not None else self._latest_obs
            if observation is None:
                raise ValueError(
                    "get_action requires obs or a previous update_obs call"
                )
            response = self._loop.run_until_complete(
                self._client.infer(
                    cast(dict[str, Any], observation),
                    trial_id=self.trial_id,
                    action_case_id=self.action_case_id,
                    step=self._step,
                )
            )
            self._step += 1
            return response.payload.get("actions")

        if func_name == "update_obs_batch":
            if not self.supports_batch_infer:
                raise RuntimeError("websocket policy server does not support batch_infer")
            self._latest_obs_batch = list(obs) if obs is not None else []
            env_ids = kwargs.get("env_ids")
            self._latest_env_ids = list(env_ids) if env_ids is not None else None
            return None

        if func_name == "get_action_batch":
            if not self.supports_batch_infer:
                raise RuntimeError("websocket policy server does not support batch_infer")
            observations = self._latest_obs_batch
            if observations is None:
                raise ValueError(
                    "get_action_batch requires a previous update_obs_batch call"
                )
            env_ids = kwargs.get("env_ids", obs)
            if env_ids is None:
                env_ids = self._latest_env_ids
            response = self._loop.run_until_complete(
                self._client.infer_batch(
                    [cast(dict[str, Any], observation) for observation in observations],
                    env_ids=list(env_ids) if env_ids is not None else None,
                    trial_id=self.trial_id,
                    action_case_id=self.action_case_id,
                    step=self._step,
                )
            )
            self._step += 1
            return response.payload.get("actions_batch")

        if func_name == "trial_end":
            response = self._loop.run_until_complete(
                self._client.trial_end(
                    trial_id=self.trial_id,
                    action_case_id=self.action_case_id,
                    result=obs if isinstance(obs, dict) else None,
                )
            )
            return response.payload.get("result")

        raise NotImplementedError(f"unsupported websocket model call: {func_name}")

    def close(self) -> None:
        if self._loop.is_closed():
            return
        try:
            self._loop.run_until_complete(self._client.close())
        finally:
            self._loop.close()

    def __enter__(self) -> WsModelClient:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
