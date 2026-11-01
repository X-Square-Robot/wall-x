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
    obs: Any
    future: asyncio.Future
    timestamp: float
    packed: bool = False


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
        self.started_at = None
        self.total_requests_enqueued = 0
        self.total_requests_completed = 0
        self.total_requests_failed = 0
        self.total_timeouts = 0
        self.queue_full_count = 0
        self.total_batches = 0
        self.total_batch_items = 0
        self.total_batch_latency_ms = 0.0
        self.last_batch_size = 0
        self.last_batch_latency_ms = None
        self.last_batch_started_at = None
        self.last_batch_finished_at = None

    async def start(self):
        self.running = True
        self.started_at = time.time()
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

    async def add_request(
        self, obs: Dict[str, Any] | bytes, packed: bool = False
    ) -> Dict[str, Any]:
        request_id = str(uuid.uuid4())
        future = asyncio.Future()
        request = Request(
            request_id=request_id,
            obs=obs,
            future=future,
            timestamp=time.monotonic(),
            packed=packed,
        )

        async with self.queue_lock:
            if len(self.queue) >= self.max_queue_size:
                self.queue_full_count += 1
                raise RuntimeError(
                    f"Queue full: {len(self.queue)}/{self.max_queue_size}"
                )
            self.queue.append(request)
            self.total_requests_enqueued += 1
            self.queue_not_empty.notify()

        try:
            result = await asyncio.wait_for(future, timeout=self.timeout)
            return result
        except asyncio.TimeoutError:
            self.total_timeouts += 1
            logger.error(f"Request {request_id} timeout after {self.timeout}s")
            raise

    def stats_snapshot(self) -> Dict[str, Any]:
        queue_size = len(self.queue)
        average_batch_latency_ms = None
        if self.total_batches > 0:
            average_batch_latency_ms = self.total_batch_latency_ms / self.total_batches
        return {
            "running": self.running,
            "started_at": self.started_at,
            "queue_size": queue_size,
            "max_queue_size": self.max_queue_size,
            "max_batch_size": self.max_batch_size,
            "max_wait_time_ms": self.max_wait_time * 1000,
            "timeout_ms": self.timeout * 1000,
            "total_requests_enqueued": self.total_requests_enqueued,
            "total_requests_completed": self.total_requests_completed,
            "total_requests_failed": self.total_requests_failed,
            "total_timeouts": self.total_timeouts,
            "queue_full_count": self.queue_full_count,
            "total_batches": self.total_batches,
            "total_batch_items": self.total_batch_items,
            "last_batch_size": self.last_batch_size,
            "last_batch_latency_ms": self.last_batch_latency_ms,
            "average_batch_latency_ms": average_batch_latency_ms,
            "last_batch_started_at": self.last_batch_started_at,
            "last_batch_finished_at": self.last_batch_finished_at,
        }

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
        self.last_batch_started_at = time.time()
        logger.debug("Processing batch of %d requests", len(batch))

        try:
            results = await asyncio.to_thread(self._infer_batch_payloads, batch)
            if len(results) != len(batch):
                raise RuntimeError(
                    "Policy infer_batch returned "
                    f"{len(results)} result(s) for {len(batch)} request(s)"
                )

            # Split queue-wait (time spent in the batch queue) from model-
            # forward time (this batch's processing). The old single infer_ms
            # conflated both, hiding queue buildup during rollover.
            batch_ms = (time.monotonic() - start_time) * 1000.0
            completed_count = 0
            for req, result in zip(batch, results):
                if isinstance(result, dict):
                    result["__sched_timing__"] = {
                        "queue_ms": round((start_time - req.timestamp) * 1000.0, 2),
                        "batch_ms": round(batch_ms, 2),
                        "batch_size": len(batch),
                    }
                if not req.future.done():
                    req.future.set_result(result)
                    completed_count += 1

            infer_time = time.monotonic() - start_time
            infer_time_ms = infer_time * 1000
            self.total_batches += 1
            self.total_batch_items += len(batch)
            self.total_requests_completed += completed_count
            self.total_batch_latency_ms += infer_time_ms
            self.last_batch_size = len(batch)
            self.last_batch_latency_ms = infer_time_ms
            self.last_batch_finished_at = time.time()
            logger.debug(
                "Batch processed in %.1fms, throughput: %.1f req/s",
                infer_time_ms,
                len(batch) / infer_time,
            )

        except Exception as e:
            infer_time_ms = (time.monotonic() - start_time) * 1000
            failed_count = 0
            self.last_batch_size = len(batch)
            self.last_batch_latency_ms = infer_time_ms
            self.last_batch_finished_at = time.time()
            logger.error(f"Batch processing failed: {e}", exc_info=True)
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)
                    failed_count += 1
            self.total_requests_failed += failed_count

    def _infer_batch_payloads(self, batch: List[Request]) -> List[Dict[str, Any]]:
        obs_list = []
        has_packed = any(req.packed for req in batch)
        if has_packed:
            try:
                import msgpack
                import msgpack_numpy as m

                m.patch()
            except ImportError as exc:
                raise RuntimeError(
                    "msgpack and msgpack-numpy are required for packed requests"
                ) from exc

        for req in batch:
            if req.packed:
                obs_list.append(msgpack.unpackb(req.obs))
            else:
                obs_list.append(req.obs)
        return self.policy.infer_batch(obs_list)
