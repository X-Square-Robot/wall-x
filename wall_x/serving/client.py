#!/usr/bin/env python3
"""
Client for Wall-X model server with sync support.
"""

import asyncio
import logging
from typing import Dict
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import msgpack
    import msgpack_numpy as m

    m.patch()
except ImportError:
    logger.error("Please install msgpack-numpy: pip install msgpack-numpy")
    raise SystemExit(1)

try:
    import websockets
except ImportError:
    logger.error("Please install websockets: pip install websockets")
    raise SystemExit(1)


class RobotClient:
    """Client for connecting to Wall-X model server."""

    def __init__(self, uri: str = "ws://localhost:8000"):
        """Initialize client.

        Args:
            uri: WebSocket URI of the server (e.g., ws://localhost:8000)
        """
        self.uri = uri
        self.websocket = None
        self.metadata = None
        self._loop = None
        self._thread = None

    async def connect(self):
        """Connect to the server and receive metadata."""
        logger.info(f"Connecting to {self.uri}...")
        self.websocket = await websockets.connect(
            self.uri,
            ping_interval=None,
            ping_timeout=None,
            max_size=None,
        )

        self.metadata = msgpack.unpackb(await self.websocket.recv())
        logger.info(f"Connected! Server metadata: {self.metadata}")

    async def predict(self, obs: Dict) -> Dict:
        """Get action prediction from observation.

        Args:
            obs: Observation dictionary containing:
                - 'image': Image array (H, W, C)
                - 'prompt': Optional text prompt
                - 'state': Optional robot state

        Returns:
            Dictionary with:
                - 'action': Predicted action array
                - 'server_timing': Timing information
        """
        if self.websocket is None:
            raise RuntimeError("Not connected. Call connect() first.")

        packed = msgpack.packb(obs)
        logger.debug(f"Payload size: {len(packed) / 1024:.1f} KB")
        await self.websocket.send(packed)
        response = msgpack.unpackb(await self.websocket.recv())
        return response

    async def close(self):
        """Close the connection."""
        if self.websocket:
            await self.websocket.close()
            logger.info("Connection closed")

    async def reset(self):
        """Reset the policy (if supported)."""
        pass

    # ============ Synchronous methods (using independent thread event loop) ============

    def _start_background_loop(self):
        """Start event loop in background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _ensure_loop(self):
        """Ensure background event loop is running."""
        if self._loop is None or not self._loop.is_running():
            self._thread = threading.Thread(
                target=self._start_background_loop, daemon=True
            )
            self._thread.start()
            # Wait for loop to start
            import time

            while self._loop is None:
                time.sleep(0.01)

    def _run_async(self, coro):
        """Run coroutine in background event loop."""
        self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def connect_sync(self):
        """Synchronously connect to server."""
        return self._run_async(self.connect())

    def predict_sync(self, obs: Dict) -> Dict:
        """Synchronous prediction method.

        Args:
            obs: Observation dictionary

        Returns:
            Prediction result dictionary
        """
        return self._run_async(self.predict(obs))

    def close_sync(self):
        """Synchronously close connection."""
        result = self._run_async(self.close())
        # Stop event loop, join thread, close loop to avoid resource leaks
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
        if self._loop and not self._loop.is_running():
            self._loop.close()
        self._loop = None
        self._thread = None
        return result
