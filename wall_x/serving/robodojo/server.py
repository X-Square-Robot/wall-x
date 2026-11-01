"""WebSocket policy server."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import msgpack
from websockets.asyncio.server import Server, ServerConnection, serve

from wall_x.serving.robodojo.protocol.codec import decode_envelope, encode_frame
from wall_x.serving.robodojo.protocol.exceptions import ErrorCode, WsError
from wall_x.serving.robodojo.protocol.messages import (
    CAPABILITY_JPEG_NDARRAY_V1,
    MessageType,
)
from wall_x.serving.robodojo.protocol.schemas import Frame

logger = logging.getLogger(__name__)


def _ok_payload(result: Any = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"ok": True}
    if result is not None:
        payload["result"] = result
    return payload


@dataclass
class PolicyServerConfig:
    host: str = "0.0.0.0"
    port: int = 19000
    ws_ping_interval_s: float | None = 20.0
    ws_ping_timeout_s: float | None = 20.0
    jpeg_ndarray_v1: bool = False
    # None declines permessage-deflate, which saves client CPU on large
    # observation frames; payload bytes are unchanged.
    ws_compression: str | None = "deflate"

    def __post_init__(self) -> None:
        if not isinstance(self.jpeg_ndarray_v1, bool):
            raise TypeError("jpeg_ndarray_v1 must be a bool")


@dataclass
class PolicyServer:
    model: Any
    config: PolicyServerConfig = field(default_factory=PolicyServerConfig)
    _server: Server | None = field(default=None, init=False)
    _model_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock,
        init=False,
        repr=False,
    )

    async def start(self) -> None:
        if self._server is not None:
            return
        self._server = await serve(
            self._handle_connection,
            self.config.host,
            self.config.port,
            max_size=None,
            ping_interval=self.config.ws_ping_interval_s,
            ping_timeout=self.config.ws_ping_timeout_s,
            compression=self.config.ws_compression,
        )
        logger.info("websocket policy server listening on %s", self.url)

    async def stop(self) -> None:
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None

    async def serve_forever(self) -> None:
        await self.start()
        assert self._server is not None
        await self._server.serve_forever()

    @property
    def url(self) -> str:
        port = self.config.port
        if self._server is not None and self._server.sockets:
            port = self._server.sockets[0].getsockname()[1]
        return f"ws://{self.config.host}:{port}"

    async def _handle_connection(self, websocket: ServerConnection) -> None:
        send_lock = asyncio.Lock()
        pending: set[asyncio.Task[None]] = set()

        async def respond(
            frame: Frame,
            *,
            decode_ms: float,
            request_bytes: int,
        ) -> None:
            dispatch_start = time.perf_counter()
            try:
                response = await self.process_frame(frame)
                dispatch_ms = (time.perf_counter() - dispatch_start) * 1000.0
                if response is None:
                    await websocket.close()
                    return
                encode_start = time.perf_counter()
                encoded = encode_frame(response)
                encode_ms = (time.perf_counter() - encode_start) * 1000.0
                send_start = time.perf_counter()
                async with send_lock:
                    await websocket.send(encoded)
                send_ms = (time.perf_counter() - send_start) * 1000.0
                logger.info(
                    "[WS_TIMING] type=%s request_id=%s request_bytes=%d "
                    "response_bytes=%d decode_ms=%.3f dispatch_ms=%.3f "
                    "encode_ms=%.3f send_ms=%.3f",
                    frame.message_type.value,
                    frame.request_id,
                    request_bytes,
                    len(encoded),
                    decode_ms,
                    dispatch_ms,
                    encode_ms,
                    send_ms,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("failed to send policy server response")

        try:
            async for raw in websocket:
                if not isinstance(raw, (bytes, bytearray)):
                    continue
                decode_start = time.perf_counter()
                try:
                    frame = decode_envelope(bytes(raw))
                except WsError as exc:
                    decode_ms = (time.perf_counter() - decode_start) * 1000.0
                    logger.error(
                        "[WS_TIMING] invalid request request_bytes=%d "
                        "decode_ms=%.3f error=%s",
                        len(raw),
                        decode_ms,
                        exc,
                    )
                    error_response = self._invalid_frame_reply_from_raw(
                        bytes(raw),
                        exc,
                    )
                    if error_response is not None:
                        async with send_lock:
                            await websocket.send(encode_frame(error_response))
                    continue
                decode_ms = (time.perf_counter() - decode_start) * 1000.0
                task = asyncio.create_task(
                    respond(
                        frame,
                        decode_ms=decode_ms,
                        request_bytes=len(raw),
                    )
                )
                pending.add(task)
                task.add_done_callback(pending.discard)
        finally:
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    @staticmethod
    def _invalid_frame_reply_from_raw(data: bytes, exc: WsError) -> Frame | None:
        """Build a correlated error when a bad payload still has a valid header."""

        try:
            wire = msgpack.unpackb(data, raw=False, strict_map_key=False)
        except Exception:
            return None
        if not isinstance(wire, Mapping):
            return None

        request_id = wire.get("message_id")
        evaluation_id = wire.get("evaluation_id")
        if request_id is None or evaluation_id is None:
            return None
        return Frame(
            message_type=MessageType.ERROR,
            request_id=str(request_id),
            evaluation_id=str(evaluation_id),
            payload={
                "code": ErrorCode.INVALID_FRAME.value,
                "message": exc.message,
                "details": exc.details or {},
            },
        )

    async def process_frame(self, frame: Frame) -> Frame | None:
        try:
            return await self._dispatch_frame(frame)
        except WsError as exc:
            return self._error_reply(frame, exc.code, exc.message, exc.details)
        except Exception as exc:
            logger.exception("policy server request failed")
            return self._error_reply(frame, ErrorCode.INTERNAL, str(exc))

    def _reply(
        self,
        frame: Frame,
        message_type: MessageType,
        payload: dict[str, Any] | None = None,
    ) -> Frame:
        return Frame(
            message_type=message_type,
            request_id=frame.request_id,
            evaluation_id=frame.evaluation_id,
            action_case_id=frame.action_case_id,
            trial_id=frame.trial_id,
            repeat_index=frame.repeat_index,
            step=frame.step,
            payload=payload or {},
        )

    def _error_reply(
        self,
        frame: Frame,
        code: ErrorCode,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> Frame:
        return self._reply(
            frame,
            MessageType.ERROR,
            {
                "code": code.value,
                "message": message,
                "details": details or {},
            },
        )

    async def _invoke_method(self, method, *args: Any) -> Any:
        if inspect.iscoroutinefunction(method):
            return await method(*args)
        result = await self._run_sync(method, *args)
        if inspect.isawaitable(result):
            return await result
        return result

    @staticmethod
    async def _run_sync(method, *args: Any) -> Any:
        """Keep a cancelled caller waiting until its model thread has stopped."""

        task = asyncio.create_task(asyncio.to_thread(method, *args))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            try:
                await task
            except Exception:
                logger.exception("model method failed after caller cancellation")
            raise

    async def _call_model_method(self, method, *args: Any) -> Any:
        async with self._model_lock:
            return await self._invoke_method(method, *args)

    def _supports_batch_infer(self) -> bool:
        declared_support = getattr(self.model, "supports_batch_infer", None)
        if declared_support is not None:
            return bool(declared_support)
        return callable(getattr(self.model, "update_obs_batch", None)) and callable(
            getattr(self.model, "get_action_batch", None)
        )

    @staticmethod
    def _requested_capabilities(frame: Frame) -> set[str]:
        requested = frame.payload.get("capabilities", [])
        if requested is None:
            return set()
        if not isinstance(requested, (list, tuple)) or any(
            not isinstance(capability, str) for capability in requested
        ):
            raise WsError(
                ErrorCode.INVALID_FRAME,
                "hello payload capabilities must be a sequence of strings",
            )
        return set(requested)

    @staticmethod
    def _batch_length(value: Any, field_name: str) -> int:
        if isinstance(value, (str, bytes, bytearray, Mapping)):
            raise WsError(
                ErrorCode.INVALID_FRAME,
                f"infer payload field {field_name!r} must be a batch sequence",
            )
        try:
            size = len(value)
        except (TypeError, AttributeError) as exc:
            raise WsError(
                ErrorCode.INVALID_FRAME,
                f"infer payload field {field_name!r} must be a batch sequence",
            ) from exc
        if size <= 0:
            raise WsError(
                ErrorCode.INVALID_FRAME,
                f"infer payload field {field_name!r} must not be empty",
            )
        return size

    @staticmethod
    def _validate_observation(observation: Any, field_name: str) -> None:
        if not isinstance(observation, Mapping):
            raise WsError(
                ErrorCode.INVALID_FRAME,
                f"infer payload field {field_name!r} must be a map",
            )

    @classmethod
    def _validate_batch(cls, observations: Any, env_ids: Any) -> int:
        batch_size = cls._batch_length(observations, "observations")
        for index, observation in enumerate(observations):
            cls._validate_observation(observation, f"observations[{index}]")
        if env_ids is not None:
            env_count = cls._batch_length(env_ids, "env_ids")
            if any(
                isinstance(env_id, bool) or not isinstance(env_id, int)
                for env_id in env_ids
            ):
                raise WsError(
                    ErrorCode.INVALID_FRAME,
                    "infer payload field 'env_ids' must contain integers",
                )
            if len(set(env_ids)) != env_count:
                raise WsError(
                    ErrorCode.INVALID_FRAME,
                    "infer payload field 'env_ids' must not contain duplicates",
                )
            if env_count != batch_size:
                raise WsError(
                    ErrorCode.INVALID_FRAME,
                    "infer payload batch length mismatch: "
                    f"len(observations)={batch_size}, len(env_ids)={env_count}",
                )
        return batch_size

    async def _dispatch_frame(self, frame: Frame) -> Frame | None:
        if frame.message_type == MessageType.HELLO:
            requested_capabilities = self._requested_capabilities(frame)
            capabilities = []
            if self._supports_batch_infer():
                capabilities.append("batch_infer")
            if (
                self.config.jpeg_ndarray_v1
                and CAPABILITY_JPEG_NDARRAY_V1 in requested_capabilities
            ):
                capabilities.append(CAPABILITY_JPEG_NDARRAY_V1)
            return self._reply(
                frame,
                MessageType.HELLO_ACK,
                {
                    "ok": True,
                    "server": "wall_x",
                    "capabilities": capabilities,
                },
            )
        if frame.message_type == MessageType.PREPARE_CASE:
            return await self._handle_prepare_case(frame)
        if frame.message_type == MessageType.RESET:
            return await self._handle_reset(frame)
        if frame.message_type == MessageType.INFER:
            return await self._handle_infer(frame)
        if frame.message_type == MessageType.TRIAL_END:
            return await self._handle_trial_end(frame)
        if frame.message_type == MessageType.HEARTBEAT:
            return self._reply(frame, MessageType.HEARTBEAT_ACK, {"ok": True})
        if frame.message_type == MessageType.CLOSE:
            return None
        raise WsError(
            ErrorCode.UNKNOWN_MESSAGE_TYPE,
            f"unsupported request type: {frame.message_type.value}",
        )

    async def _handle_prepare_case(self, frame: Frame) -> Frame:
        case_meta = dict(frame.payload)
        if frame.action_case_id is not None:
            case_meta.setdefault("action_case_id", frame.action_case_id)

        hook = getattr(self.model, "prepare_case", None)
        hook_result = (
            await self._call_model_method(hook, case_meta) if callable(hook) else None
        )
        return self._reply(
            frame, MessageType.PREPARE_CASE_ACK, _ok_payload(hook_result)
        )

    async def _handle_reset(self, frame: Frame) -> Frame:
        method = getattr(self.model, "reset", None)
        if not callable(method):
            raise WsError(ErrorCode.RESET_FAILED, "model.reset is not callable")
        try:
            result = await self._call_model_method(method)
        except Exception as exc:
            raise WsError(ErrorCode.RESET_FAILED, str(exc)) from exc
        return self._reply(frame, MessageType.RESET_RESULT, _ok_payload(result))

    async def _handle_infer_batch(self, frame: Frame, observations: Any) -> Frame:
        env_ids = frame.payload.get("env_ids")
        batch_size = self._validate_batch(observations, env_ids)
        if not self._supports_batch_infer():
            raise WsError(
                ErrorCode.INVALID_FRAME,
                "server model does not advertise batch_infer",
            )

        start = time.perf_counter()
        update_obs_ms = 0.0
        get_action_ms = 0.0
        batch_profile: dict[str, Any] = {}
        try:
            async with self._model_lock:
                update_obs_batch = getattr(self.model, "update_obs_batch", None)
                get_action_batch = getattr(self.model, "get_action_batch", None)
                if inspect.iscoroutinefunction(
                    update_obs_batch
                ) or inspect.iscoroutinefunction(get_action_batch):
                    update_start = time.perf_counter()
                    upd = update_obs_batch(observations)
                    if inspect.isawaitable(upd):
                        await upd
                    update_obs_ms = (time.perf_counter() - update_start) * 1000.0
                    action_start = time.perf_counter()
                    result = get_action_batch(env_ids)
                    if inspect.isawaitable(result):
                        result = await result
                    get_action_ms = (time.perf_counter() - action_start) * 1000.0
                else:
                    def run_batch_infer() -> tuple[Any, float, float]:
                        update_start = time.perf_counter()
                        upd = update_obs_batch(observations)
                        if inspect.isawaitable(upd):
                            raise TypeError(
                                "update_obs_batch returned an awaitable but is not async"
                            )
                        update_elapsed_ms = (
                            time.perf_counter() - update_start
                        ) * 1000.0
                        action_start = time.perf_counter()
                        res = get_action_batch(env_ids)
                        if inspect.isawaitable(res):
                            raise TypeError(
                                "get_action_batch returned an awaitable but is not async"
                            )
                        action_elapsed_ms = (
                            time.perf_counter() - action_start
                        ) * 1000.0
                        return res, update_elapsed_ms, action_elapsed_ms

                    result, update_obs_ms, get_action_ms = await self._run_sync(
                        run_batch_infer
                    )
                profile = getattr(self.model, "last_batch_profile", None)
                if isinstance(profile, Mapping):
                    batch_profile = dict(profile)
            try:
                action_count = len(result)
            except (TypeError, AttributeError) as exc:
                raise TypeError("get_action_batch must return a batch sequence") from exc
            if action_count != batch_size:
                raise ValueError(
                    "batch inference result length mismatch: "
                    f"len(observations)={batch_size}, len(actions_batch)={action_count}"
                )
        except WsError:
            raise
        except Exception as exc:
            logger.exception(
                "[BATCH_TIMING] failed B=%d env_ids=%s "
                "update_obs_batch_ms=%.3f get_action_batch_ms=%.3f",
                batch_size,
                env_ids,
                update_obs_ms,
                get_action_ms,
            )
            raise WsError(ErrorCode.INFER_FAILED, str(exc)) from exc
        latency_ms = (time.perf_counter() - start) * 1000.0
        timings_ms = {
            "server_infer_total": latency_ms,
            "update_obs_batch": update_obs_ms,
            "get_action_batch": get_action_ms,
        }
        logger.info(
            "[BATCH_TIMING] B=%d env_ids=%s timings_ms=%s profile=%s",
            batch_size,
            env_ids,
            timings_ms,
            batch_profile,
        )
        payload = {
            "actions_batch": result,
            "latency_ms": latency_ms,
            "timings_ms": timings_ms,
            "batch_profile": batch_profile,
        }
        return self._reply(frame, MessageType.INFER_RESULT, payload)

    async def _handle_infer(self, frame: Frame) -> Frame:
        if "observation" in frame.payload and "observations" in frame.payload:
            raise WsError(
                ErrorCode.INVALID_FRAME,
                "infer payload must contain exactly one of observation or observations",
            )
        observations = frame.payload.get("observations")
        if observations is not None:
            return await self._handle_infer_batch(frame, observations)
        observation = frame.payload.get("observation")
        if observation is None:
            raise WsError(ErrorCode.INVALID_FRAME, "infer payload missing observation")
        self._validate_observation(observation, "observation")

        start = time.perf_counter()
        try:
            async with self._model_lock:
                update_obs = getattr(self.model, "update_obs", None)
                get_action = getattr(self.model, "get_action", None)
                if callable(update_obs) and callable(get_action):
                    if inspect.iscoroutinefunction(
                        update_obs
                    ) or inspect.iscoroutinefunction(get_action):
                        update_result = update_obs(observation)
                        if inspect.isawaitable(update_result):
                            await update_result
                        result = get_action()
                        if inspect.isawaitable(result):
                            result = await result
                    else:
                        def run_legacy_infer() -> Any:
                            update_result = update_obs(observation)
                            if inspect.isawaitable(update_result):
                                raise TypeError(
                                    "update_obs returned an awaitable but is not async"
                                )
                            result = get_action()
                            if inspect.isawaitable(result):
                                raise TypeError(
                                    "get_action returned an awaitable but is not async"
                                )
                            return result

                        result = await self._run_sync(run_legacy_infer)
                else:
                    infer = getattr(self.model, "infer", None)
                    if not callable(infer):
                        raise AttributeError(
                            "model must implement update_obs()/get_action() or infer(observation)"
                        )
                    result = await self._invoke_method(infer, observation)
        except WsError:
            raise
        except Exception as exc:
            raise WsError(ErrorCode.INFER_FAILED, str(exc)) from exc

        latency_ms = (time.perf_counter() - start) * 1000.0
        if isinstance(result, Mapping) and "actions" in result:
            payload: dict[str, Any] = dict(result)
            payload.setdefault("latency_ms", latency_ms)
            payload.setdefault("actions_batch", [payload["actions"]])
            payload.setdefault("timings_ms", {"server_infer_total": latency_ms})
            payload.setdefault("batch_profile", {"batch_size": 1, "session_capacity": 1})
        else:
            payload = {"actions": result, "latency_ms": latency_ms}

        return self._reply(frame, MessageType.INFER_RESULT, payload)

    async def _handle_trial_end(self, frame: Frame) -> Frame:
        hook = getattr(self.model, "on_trial_end", None)
        hook_result = (
            await self._call_model_method(hook, dict(frame.payload))
            if callable(hook)
            else None
        )
        return self._reply(frame, MessageType.TRIAL_END_ACK, _ok_payload(hook_result))
