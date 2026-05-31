"""Training metric buffering and wandb emission helpers."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple


class MetricsLogger:
    """Buffer per-step stats and emit averaged batches to wandb.

    Also formats the per-step console line via ``format_training_line``,
    which consumes an ``(stat_key, pretty_label, fmt_spec)`` triple list
    from ``adapter.console_fields()``.

    ``smooth_window`` controls a separate rolling-mean buffer used purely
    for display smoothing (console + tqdm). It is independent of the
    ``log_interval`` wandb-flush buffer; the smoothed dict is returned by
    :meth:`smooth` for the caller to assign to ``_current_step_stats``.
    1 (default) preserves historical per-step values.
    """

    def __init__(self, *, wandb_run, log_interval: int, smooth_window: int = 1):
        self._wandb_run = wandb_run
        self._log_interval = log_interval
        self._buffer: List[Dict[str, Any]] = []
        self._smooth_window = max(1, int(smooth_window))
        self._smooth_buffers: Dict[str, Deque[float]] = {}

    def record_step(self, step_stats: Dict[str, Any], *, is_main: bool) -> None:
        """Push a per-step stats dict into the rank-0 buffer (no-op off rank-0)."""
        if is_main:
            self._buffer.append(step_stats)

    def flush_if_due(self, global_step: int) -> Optional[Dict[str, Any]]:
        """At log_interval boundaries, average + emit; return avg dict or None."""
        if global_step % self._log_interval != 0 or not self._buffer:
            return None
        avg = self._average(self._buffer)
        if self._wandb_run is not None and hasattr(self._wandb_run, "log"):
            self._wandb_run.log(avg, step=global_step)
        self._buffer = []
        return avg

    def smooth(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Return *stats* with numeric entries replaced by their rolling mean.

        Uses one ``deque(maxlen=smooth_window)`` per metric key. Non-numeric
        entries pass through unchanged. When ``smooth_window <= 1`` this is
        the identity (historical behavior).
        """
        if self._smooth_window <= 1:
            return stats
        out: Dict[str, Any] = {}
        for key, val in stats.items():
            if not isinstance(val, (int, float)):
                out[key] = val
                continue
            buf = self._smooth_buffers.get(key)
            if buf is None or buf.maxlen != self._smooth_window:
                buf = deque(maxlen=self._smooth_window)
                self._smooth_buffers[key] = buf
            buf.append(float(val))
            out[key] = sum(buf) / len(buf)
        return out

    @staticmethod
    def _average(buf: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_keys: set = set()
        for stats in buf:
            all_keys.update(stats.keys())
        avg: Dict[str, Any] = {}
        for key in all_keys:
            values = [s[key] for s in buf if key in s]
            if values:
                avg[key] = sum(values) / len(values)
        return avg

    def format_training_line(
        self,
        *,
        epoch: int,
        total_epoch: int,
        current_iter: int,
        total_iter: int,
        loss: float,
        lr: float,
        time_per_step: float,
        stats: Dict[str, Any],
        fields: List[Tuple[str, str, str]],
        mfu_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Render a line matching pre-refactor training_log output exactly.

        Layout:  " epoch E/T | iter I/T | loss L | [adapter fields] | lr L |
                  time_current_backward_step Ts | [MFU P% |]"
        """
        parts = [
            " epoch {:3d}/{:3d} |".format(epoch, total_epoch),
            " iter {:6d}/{:6d} |".format(current_iter, total_iter),
            " loss {:.6f} |".format(loss),
        ]
        for key, label, fmt in fields:
            value = stats.get(key)
            if value is not None:
                parts.append(" {} {:{fmt}} |".format(label, value, fmt=fmt))
        parts.append(" lr {:.6f} |".format(lr))
        parts.append(" time_current_backward_step {:.6f}s |".format(time_per_step))
        if mfu_info is not None:
            parts.append(" MFU {:.2f}% |".format(mfu_info["mfu"] * 100))
        return "".join(parts)
