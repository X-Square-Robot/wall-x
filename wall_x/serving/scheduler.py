import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class Request:
    request_id: str
    obs: Dict[str, Any]
    future: asyncio.Future
    timestamp: float


class RequestScheduler:
    def __init__(
        self,
        policy,
        max_batch_size: int = 8,
        max_wait_time_ms: float = 100,
        max_queue_size: int = 128,
        timeout_ms: float = 5000,
    ):
        self.policy = policy
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time_ms / 1000.0
        self.max_queue_size = max_queue_size
        self.timeout = timeout_ms / 1000.0

        self.queue = deque()
        self.queue_lock = asyncio.Lock()
        self.queue_not_empty = asyncio.Condition(self.queue_lock)

        self.running = False
        self.batch_task = None

    async def start(self):
        self.running = True
        self.batch_task = asyncio.create_task(self._batch_loop())
        logger.info(
            f"RequestScheduler started: max_batch={self.max_batch_size}, max_wait={self.max_wait_time*1000}ms"
        )

    async def stop(self):
        # Signal batch_loop to stop under the lock so _collect_batch
        # wakes up and sees running=False atomically.
        async with self.queue_lock:
            self.running = False
            self.queue_not_empty.notify_all()

        # Wait for batch_loop to fully exit before draining.
        if self.batch_task:
            await self.batch_task

        # Now drain any remaining requests (no race — batch_loop is done).
        async with self.queue_lock:
            remaining = list(self.queue)
            self.queue.clear()
        if remaining:
            logger.info(
                f"Graceful shutdown: processing {len(remaining)} remaining request(s)"
            )
            for i in range(0, len(remaining), self.max_batch_size):
                batch = remaining[i : i + self.max_batch_size]
                await self._process_batch(batch)
        logger.info("RequestScheduler stopped")

    async def add_request(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        request_id = str(uuid.uuid4())
        future = asyncio.Future()
        request = Request(
            request_id=request_id, obs=obs, future=future, timestamp=time.monotonic()
        )

        async with self.queue_lock:
            if len(self.queue) >= self.max_queue_size:
                raise RuntimeError(
                    f"Queue full: {len(self.queue)}/{self.max_queue_size}"
                )
            self.queue.append(request)
            self.queue_not_empty.notify()

        try:
            result = await asyncio.wait_for(future, timeout=self.timeout)
            return result
        except asyncio.TimeoutError:
            logger.error(f"Request {request_id} timeout after {self.timeout}s")
            raise

    async def _batch_loop(self):
        while self.running:
            try:
                batch = await self._collect_batch()
                if batch:
                    await self._process_batch(batch)
            except Exception as e:
                logger.error(f"Unexpected error in batch loop: {e}", exc_info=True)

    async def _collect_batch(self) -> List[Request]:
        async with self.queue_not_empty:
            while self.running and len(self.queue) == 0:
                await self.queue_not_empty.wait()

            if not self.running:
                return []

            batch = []
            deadline = time.monotonic() + self.max_wait_time

            while len(batch) < self.max_batch_size:
                if len(self.queue) > 0:
                    batch.append(self.queue.popleft())

                if len(batch) >= self.max_batch_size:
                    break

                if len(self.queue) == 0:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        await asyncio.wait_for(
                            self.queue_not_empty.wait(), timeout=remaining
                        )
                    except asyncio.TimeoutError:
                        break

            return batch

    async def _process_batch(self, batch: List[Request]):
        if not batch:
            return

        start_time = time.monotonic()
        logger.info(f"Processing batch of {len(batch)} requests")

        try:
            obs_list = [req.obs for req in batch]
            results = await asyncio.to_thread(self.policy.infer_batch, obs_list)

            for req, result in zip(batch, results):
                if not req.future.done():
                    req.future.set_result(result)

            infer_time = time.monotonic() - start_time
            logger.info(
                f"Batch processed in {infer_time*1000:.1f}ms, throughput: {len(batch)/infer_time:.1f} req/s"
            )

        except Exception as e:
            logger.error(f"Batch processing failed: {e}", exc_info=True)
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)
