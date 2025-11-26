import asyncio
import http
import logging
import time
import traceback
from typing import Any, Dict, Optional

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
    3. Returns predicted actions
    4. Tracks timing information
    """

    def __init__(
        self,
        policy: BasePolicy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: Optional[Dict] = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            ping_interval=None,  # Disable automatic ping for long-running inference
            ping_timeout=None,  # Disable ping timeout
            process_request=_health_check,
        ) as server:
            logger.info(f"Server started on {self._host}:{self._port}")
            await server.serve_forever()

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
                action = self._policy.infer(obs)
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
                logger.error(f"Error handling request: {e}")
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(
    connection: _server.ServerConnection, request: _server.Request
) -> Optional[_server.Response]:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
