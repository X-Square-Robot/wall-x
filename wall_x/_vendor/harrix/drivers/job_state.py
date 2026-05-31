"""Episode work queue with JSONL persistence.

Drivers enumerate episode ids into ``pending``; env handles claim work and
return results through ``complete`` or ``fail``. Every state transition is
appended to ``state.jsonl``, so completed episodes can be skipped on restart.

Two scheduling modes are supported:
- FIFO: workers claim the next pending episode.
- batch_sync: episodes are grouped into task-local frames. A new frame is not
  released until the previous frame is complete, which gives deterministic
  lockstep batches at the cost of possible idle workers.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Optional


class JobState:
    def __init__(
        self,
        all_episodes: list[tuple],
        log_path: str,
        batch_sync_mode: bool = False,
        batch_size: int = 1,
    ):
        """
        all_episodes: list of (suite_name, task_idx, ep_idx) tuples
        log_path: state.jsonl path; existing completed episodes are skipped
        batch_sync_mode: enable task-local frame barriers
        batch_size: number of episodes per frame
        """
        self._lock = threading.Lock()
        self._log_path = log_path
        self._log_fh = None
        self._batch_sync_mode = bool(batch_sync_mode)
        self._batch_size = int(batch_size)

        completed_set = (
            self._load_completed(log_path) if os.path.exists(log_path) else set()
        )
        remaining: list[tuple] = [
            tuple(ep) for ep in all_episodes if tuple(ep) not in completed_set
        ]

        self._in_progress: dict[tuple, int] = {}
        self._completed: dict[tuple, dict] = {}
        self._failed: dict[tuple, str] = {}

        if self._batch_sync_mode:
            self._frames: list[dict] = self._build_frames(remaining, self._batch_size)
            self._cur_frame_idx: int = 0
            self._cur_frame_inflight: int = 0
            self._pending = None
        else:
            self._frames = []
            self._cur_frame_idx = 0
            self._cur_frame_inflight = 0
            self._pending: list[tuple] = remaining

        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self._log_fh = open(log_path, "a")

        log_entry = {
            "event": "session_start",
            "pending": (
                len(self._pending)
                if not self._batch_sync_mode
                else sum(len(f["eps"]) for f in self._frames)
            ),
            "skipped_completed": len(completed_set),
            "batch_sync_mode": self._batch_sync_mode,
        }
        if self._batch_sync_mode:
            log_entry["num_frames"] = len(self._frames)
            log_entry["batch_size"] = self._batch_size
        self._append_log(log_entry)

    @staticmethod
    def _build_frames(remaining: list[tuple], batch_size: int) -> list[dict]:
        """Group episodes by ``(suite, task_idx)`` into fixed-size frames.

        Partial tail frames are kept.
        """
        frames: list[dict] = []
        cur_key: Optional[tuple] = None
        cur_bucket: list[tuple] = []
        for ep in remaining:
            key = (ep[0], ep[1])
            if key != cur_key and cur_bucket:
                for i in range(0, len(cur_bucket), batch_size):
                    frames.append(
                        {"eps": cur_bucket[i : i + batch_size], "next_slot": 0}
                    )
                cur_bucket = []
            cur_key = key
            cur_bucket.append(ep)
        if cur_bucket:
            for i in range(0, len(cur_bucket), batch_size):
                frames.append({"eps": cur_bucket[i : i + batch_size], "next_slot": 0})
        return frames

    @staticmethod
    def _load_completed(log_path: str) -> set[tuple]:
        completed = set()
        with open(log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("event") == "completed" and "ep_id" in rec:
                    completed.add(tuple(rec["ep_id"]))
        return completed

    def _append_log(self, extra: dict):
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}
        if "ep_id" in rec and isinstance(rec["ep_id"], tuple):
            rec["ep_id"] = list(rec["ep_id"])
        self._log_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._log_fh.flush()

    # ---- claim API ----

    def claim(self, worker_id: int) -> Optional[list]:
        """Claim one episode id.

        FIFO returns ``None`` when the pending queue is empty. In batch-sync
        mode, ``None`` can also mean the current frame has been fully issued but
        still has in-flight episodes.
        """
        with self._lock:
            if self._batch_sync_mode:
                return self._claim_sync(worker_id)
            return self._claim_fifo(worker_id)

    def claim_frame(self) -> list[list]:
        """Claim one complete frame for lockstep in-process evaluation.

        In FIFO mode this returns up to ``batch_size`` pending episodes.
        """
        with self._lock:
            if self._batch_sync_mode:
                return self._claim_frame_sync()
            return self._claim_frame_fifo()

    def _claim_fifo(self, worker_id: int) -> Optional[list]:
        if not self._pending:
            return None
        ep = self._pending.pop(0)
        self._in_progress[ep] = worker_id
        self._append_log({"event": "claimed", "ep_id": ep, "worker": worker_id})
        return list(ep)

    def _claim_sync(self, worker_id: int) -> Optional[list]:
        while self._cur_frame_idx < len(self._frames):
            frame = self._frames[self._cur_frame_idx]
            if frame["next_slot"] < len(frame["eps"]):
                ep = frame["eps"][frame["next_slot"]]
                frame["next_slot"] += 1
                self._cur_frame_inflight += 1
                self._in_progress[ep] = worker_id
                self._append_log(
                    {
                        "event": "claimed",
                        "ep_id": ep,
                        "worker": worker_id,
                        "frame": self._cur_frame_idx,
                    }
                )
                return list(ep)
            if self._cur_frame_inflight > 0:
                return None
            self._append_log(
                {
                    "event": "frame_done",
                    "frame": self._cur_frame_idx,
                    "size": len(frame["eps"]),
                }
            )
            self._cur_frame_idx += 1
        return None

    def _claim_frame_sync(self) -> list[list]:
        # Advance completed frames.
        while self._cur_frame_idx < len(self._frames):
            frame = self._frames[self._cur_frame_idx]
            if frame["next_slot"] < len(frame["eps"]):
                break
            if self._cur_frame_inflight > 0:
                # The previous frame still has in-flight episodes.
                return []
            self._append_log(
                {
                    "event": "frame_done",
                    "frame": self._cur_frame_idx,
                    "size": len(frame["eps"]),
                }
            )
            self._cur_frame_idx += 1
        if self._cur_frame_idx >= len(self._frames):
            return []
        frame = self._frames[self._cur_frame_idx]
        out: list[list] = []
        while frame["next_slot"] < len(frame["eps"]):
            ep = frame["eps"][frame["next_slot"]]
            frame["next_slot"] += 1
            self._cur_frame_inflight += 1
            self._in_progress[ep] = -1
            self._append_log(
                {"event": "claimed", "ep_id": ep, "frame": self._cur_frame_idx}
            )
            out.append(list(ep))
        return out

    def _claim_frame_fifo(self) -> list[list]:
        if not self._pending:
            return []
        n = min(self._batch_size, len(self._pending))
        out: list[list] = []
        for _ in range(n):
            ep = self._pending.pop(0)
            self._in_progress[ep] = -1
            self._append_log({"event": "claimed", "ep_id": ep})
            out.append(list(ep))
        return out

    # ---- complete / fail ----

    def complete(self, ep_id: list, result: dict) -> None:
        ep = tuple(ep_id)
        with self._lock:
            if ep in self._in_progress:
                self._in_progress.pop(ep, None)
                if self._batch_sync_mode:
                    self._cur_frame_inflight = max(0, self._cur_frame_inflight - 1)
            self._completed[ep] = result
            self._append_log({"event": "completed", "ep_id": ep, **result})

    def fail(self, ep_id: list, error: str) -> None:
        ep = tuple(ep_id)
        with self._lock:
            if ep in self._in_progress:
                self._in_progress.pop(ep, None)
                if self._batch_sync_mode:
                    self._cur_frame_inflight = max(0, self._cur_frame_inflight - 1)
            self._failed[ep] = error
            self._append_log({"event": "failed", "ep_id": ep, "error": error})

    # ---- status queries ----

    def get_frame_inflight(self) -> int:
        with self._lock:
            return self._cur_frame_inflight if self._batch_sync_mode else 0

    def is_drained(self) -> bool:
        with self._lock:
            if self._batch_sync_mode:
                return (
                    self._cur_frame_idx >= len(self._frames)
                    and self._cur_frame_inflight == 0
                )
            return len(self._pending) == 0 and len(self._in_progress) == 0

    def progress(self) -> dict:
        with self._lock:
            base = {
                "in_progress": len(self._in_progress),
                "completed": len(self._completed),
                "failed": len(self._failed),
            }
            if self._batch_sync_mode:
                pending = sum(
                    len(f["eps"]) - f["next_slot"]
                    for f in self._frames[self._cur_frame_idx :]
                )
                base["pending"] = pending
                base["frame"] = f"{self._cur_frame_idx}/{len(self._frames)}"
                base["frame_inflight"] = self._cur_frame_inflight
            else:
                base["pending"] = len(self._pending)
            return base

    def dump_final(self, report_path: str) -> None:
        with self._lock:
            per_task: dict[tuple, dict] = {}
            for ep, result in self._completed.items():
                key = (ep[0], ep[1])
                d = per_task.setdefault(
                    key, {"attempted": 0, "successes": 0, "steps": []}
                )
                d["attempted"] += 1
                if result.get("success"):
                    d["successes"] += 1
                if "steps" in result:
                    d["steps"].append(result["steps"])
            for ep in self._failed:
                key = (ep[0], ep[1])
                d = per_task.setdefault(
                    key, {"attempted": 0, "successes": 0, "steps": []}
                )
                d["attempted"] += 1

            total_attempted = sum(d["attempted"] for d in per_task.values())
            total_successes = sum(d["successes"] for d in per_task.values())
            overall_rate = total_successes / max(1, total_attempted)

            report = {
                "overall": {
                    "attempted": total_attempted,
                    "successes": total_successes,
                    "success_rate": overall_rate,
                    "failed": len(self._failed),
                },
                "per_task": {
                    f"{suite}_t{task_idx}": {
                        **d,
                        "success_rate": d["successes"] / max(1, d["attempted"]),
                        "avg_steps": (
                            (sum(d["steps"]) / max(1, len(d["steps"])))
                            if d["steps"]
                            else None
                        ),
                    }
                    for (suite, task_idx), d in sorted(per_task.items())
                },
            }
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            self._append_log(
                {
                    "event": "session_end",
                    "report_path": report_path,
                    "overall_success_rate": overall_rate,
                }
            )
