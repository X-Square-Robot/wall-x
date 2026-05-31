import asyncio
import http
import logging
import time
from typing import Dict, Any, List

try:
    import msgpack
    import msgpack_numpy as m

    m.patch()
except ImportError:
    logging.warning(
        "msgpack-numpy not installed. Install with: pip install msgpack-numpy"
    )
    msgpack = None

import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class BasePolicy:
    """Base class for policies that can be served."""

    def infer(self, obs: Dict) -> Dict:
        """Infer actions from observations."""
        raise NotImplementedError

    def infer_batch(self, obs_list: List[Dict]) -> List[Dict]:
        """Batch inference. Default implementation calls infer() per sample."""
        return [self.infer(obs) for obs in obs_list]

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        pass

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return metadata about the policy."""
        return {}


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol.

    Implements a websocket server that:
    1. Sends policy metadata on connection
    2. Receives observations
    3. Returns predicted actions (single or batched)
    4. Tracks timing information

    When batching parameters are provided, dynamic batching is enabled:
    requests from multiple clients are queued and processed in batches.
    """

    def __init__(
        self,
        policy: BasePolicy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: Dict | None = None,
        # Dynamic batching parameters (None = disabled)
        max_batch_size: int | None = None,
        max_wait_time_ms: float | None = None,
        max_queue_size: int = 100,
        timeout_ms: float = 30000,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}

        # Dynamic batching
        self._scheduler = None
        if max_batch_size is not None:
            from .scheduler import RequestScheduler

            self._scheduler = RequestScheduler(
                policy=policy,
                max_batch_size=max_batch_size,
                max_wait_time_ms=(
                    max_wait_time_ms if max_wait_time_ms is not None else 0
                ),
                max_queue_size=max_queue_size,
                timeout_ms=timeout_ms,
            )

        logging.getLogger("websockets.server").setLevel(logging.INFO)

    @property
    def batching_enabled(self) -> bool:
        return self._scheduler is not None

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        # Start the scheduler if batching is enabled
        if self._scheduler is not None:
            await self._scheduler.start()

        try:
            async with _server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                ping_interval=None,
                ping_timeout=None,
                process_request=_health_check,
            ) as server:
                mode_str = "batched" if self.batching_enabled else "single"
                logger.info(
                    f"Server started on {self._host}:{self._port} (mode={mode_str})"
                )
                await server.serve_forever()
        finally:
            if self._scheduler is not None:
                await self._scheduler.stop()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")

        if msgpack is None:
            await websocket.close(
                code=websockets.frames.CloseCode.INTERNAL_ERROR,
                reason="msgpack-numpy not installed on server",
            )
            return

        # Send metadata to client
        await websocket.send(msgpack.packb(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack.unpackb(await websocket.recv())

                infer_time = time.monotonic()

                if self._scheduler is not None:
                    # Dynamic batching path
                    action = await self._scheduler.add_request(obs)
                else:
                    # Single inference path
                    action = await asyncio.to_thread(self._policy.infer, obs)

                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(msgpack.packb(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception as e:
                logger.error(f"Error handling request: {e}", exc_info=True)
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error.",
                )
                raise


def _health_check(
    connection: _server.ServerConnection, request: _server.Request
) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    if request.path == "/v2/health/ready":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
