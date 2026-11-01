"""Prometheus metrics exposition for the model service.

Dependency-free: renders the existing router / worker stats snapshots into the
Prometheus text exposition format (version 0.0.4). The router and each worker
serve this at ``/metrics``; Prometheus distinguishes workers by the scrape
``instance`` label, and the router additionally labels per-worker series by
``worker_id`` / ``device``.

Counters (``*_total``) come straight from the monotonic counters already in
the snapshots; gauges are instantaneous values. Request latency is exposed as
a Prometheus Histogram (``*_bucket`` / ``*_sum`` / ``*_count``) so the unified
platform can derive mean and any quantile, aggregated correctly across workers:

    mean : rate(..._sum[5m]) / rate(..._count[5m])
    p50  : histogram_quantile(0.50, sum by (le) (rate(..._bucket[5m])))
    p95  : histogram_quantile(0.95, sum by (le) (rate(..._bucket[5m])))
    p99  : histogram_quantile(0.99, sum by (le) (rate(..._bucket[5m])))

We render here (not prometheus_client objects) so there is no cross-process
registry to manage in multi-instance; the only instrumentation is a cheap
``LatencyHistogram.observe`` at the two points latency is already recorded.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

Sample = Tuple[Mapping[str, Any], Any]

# Latency histogram bucket upper bounds in SECONDS (Prometheus convention).
# Dense around 50-500ms where policy inference lives, with a tail to 5s for
# queue-induced spikes.
DEFAULT_LATENCY_BUCKETS_S: Tuple[float, ...] = (
    0.025,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    5.0,
)


@dataclass
class LatencyHistogram:
    """Minimal cumulative-bucket latency histogram.

    Lock-free by design — not by tolerating races, but because there are
    none: observe is only ever called from its process's single asyncio
    event-loop thread (router: _finish_worker_request, under the router
    lock; worker: _record_request_latency in the connection handler; the
    model forward in asyncio.to_thread never touches the histogram), and
    observe contains no await — so its _sum / _count /
    _counts[idx] increments cannot be interleaved with another coroutine
    and are effectively atomic. It also never raises (bad / NaN inputs are
    dropped). NOTE: this is NOT thread-safe; if observe ever gets called
    from a real worker thread (e.g. inside asyncio.to_thread), guard the
    three increments with a threading.Lock.
    """

    bounds: Tuple[float, ...] = DEFAULT_LATENCY_BUCKETS_S
    _counts: List[int] = field(default_factory=list)
    _sum: float = 0.0
    _count: int = 0

    def __post_init__(self) -> None:
        # one extra slot for the implicit +Inf overflow bucket
        if not self._counts:
            self._counts = [0] * (len(self.bounds) + 1)

    def observe(self, seconds: Optional[float]) -> None:
        if seconds is None:
            return
        try:
            s = float(seconds)
        except (TypeError, ValueError):
            return
        if s != s:  # NaN
            return
        self._sum += s
        self._count += 1
        idx = len(self.bounds)  # default to +Inf overflow bucket
        for i, b in enumerate(self.bounds):
            if s <= b:
                idx = i
                break
        self._counts[idx] += 1

    def snapshot(self) -> Dict[str, Any]:
        """Return cumulative (le, count) buckets + sum + count for rendering."""
        cum: List[Tuple[float, int]] = []
        running = 0
        for i, b in enumerate(self.bounds):
            running += self._counts[i]
            cum.append((b, running))
        running += self._counts[-1]
        cum.append((float("inf"), running))
        return {"buckets": cum, "sum": self._sum, "count": self._count}


def _escape_label(value: Any) -> str:
    s = str(value)
    return s.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _format_labels(labels: Optional[Mapping[str, Any]]) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{_escape_label(v)}"' for k, v in labels.items())
    return "{" + inner + "}"


def _num(value: Any) -> Any:
    """5.0 -> 5, 0.025 -> 0.025 (compact, valid Prometheus number)."""
    f = float(value)
    return int(f) if f.is_integer() else f


def _coerce(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


class _Writer:
    def __init__(self) -> None:
        self._chunks: List[str] = []

    def family(
        self, name: str, mtype: str, help_text: str, samples: Iterable[Sample]
    ) -> None:
        lines = []
        for labels, value in samples:
            v = _coerce(value)
            if v is None:
                continue
            lines.append(f"{name}{_format_labels(labels)} {_num(v)}")
        if not lines:
            return
        self._chunks.append(f"# HELP {name} {help_text}")
        self._chunks.append(f"# TYPE {name} {mtype}")
        self._chunks.extend(lines)

    def scalar(
        self,
        name: str,
        mtype: str,
        help_text: str,
        value: Any,
        labels: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.family(name, mtype, help_text, [(labels or {}, value)])

    def histogram(
        self,
        name: str,
        help_text: str,
        series: Sequence[Tuple[Mapping[str, Any], Optional[Dict[str, Any]]]],
    ) -> None:
        """series: list of (labels, snapshot-dict-or-None). Emits one HELP/TYPE
        then ``_bucket``/``_sum``/``_count`` lines for every series."""
        valid = [(lbl, s) for lbl, s in series if s and s.get("count") is not None]
        if not valid:
            return
        self._chunks.append(f"# HELP {name} {help_text}")
        self._chunks.append(f"# TYPE {name} histogram")
        for labels, snap in valid:
            base = dict(labels or {})
            for le, cum in snap["buckets"]:
                le_str = "+Inf" if le == float("inf") else _num(le)
                self._chunks.append(
                    f"{name}_bucket{_format_labels({**base, 'le': le_str})} {int(cum)}"
                )
            self._chunks.append(f"{name}_sum{_format_labels(base)} {_num(snap['sum'])}")
            self._chunks.append(
                f"{name}_count{_format_labels(base)} {int(snap['count'])}"
            )

    def render(self) -> str:
        return "\n".join(self._chunks) + "\n"


# --------------------------------------------------------------------------- #
# Router: fleet + per-worker routing view (single public scrape target).
# --------------------------------------------------------------------------- #
def render_router_metrics(snapshot: Dict[str, Any]) -> str:
    w = _Writer()
    router = snapshot.get("router", {})
    w.scalar(
        "model_service_router_workers_total",
        "gauge",
        "Total worker instances behind the router.",
        router.get("total_workers"),
    )
    w.scalar(
        "model_service_router_workers_healthy",
        "gauge",
        "Worker instances passing health checks.",
        router.get("healthy_workers"),
    )
    w.scalar(
        "model_service_router_workers_available",
        "gauge",
        "Worker instances available for routing (healthy and not draining).",
        router.get("available_workers"),
    )
    w.scalar(
        "model_service_router_active_connections",
        "gauge",
        "Active client (robot) connections currently held by the router "
        "(number of robots being served).",
        router.get("active_connections"),
    )

    ru = snapshot.get("rolling_update") or {}
    w.scalar(
        "model_service_rolling_update_running",
        "gauge",
        "1 while a rolling update is in progress, else 0.",
        1 if ru.get("running") else 0,
    )
    wu = snapshot.get("weight_update") or {}
    w.scalar(
        "model_service_weight_update_running",
        "gauge",
        "1 while a rolling weight/actor update (rollover) is in progress, " "else 0.",
        1 if wu.get("running") else 0,
    )

    workers = snapshot.get("workers", []) or []

    def labels(wk):
        return {"worker_id": str(wk.get("worker_id")), "device": str(wk.get("device"))}

    def per(name: str, mtype: str, help_text: str, key: str, transform=lambda x: x):
        w.family(
            name,
            mtype,
            help_text,
            [(labels(wk), transform(wk.get(key))) for wk in workers],
        )

    per(
        "model_service_worker_active_connections",
        "gauge",
        "Active client connections proxied to this worker.",
        "active_connections",
    )
    per(
        "model_service_worker_in_flight",
        "gauge",
        "Requests currently being processed by this worker.",
        "in_flight",
    )
    per(
        "model_service_worker_requests_total",
        "counter",
        "Requests proxied to this worker.",
        "total_requests",
    )
    per(
        "model_service_worker_connections_total",
        "counter",
        "Connections proxied to this worker.",
        "total_connections",
    )
    per(
        "model_service_worker_errors_total",
        "counter",
        "Proxy/inference errors for this worker.",
        "error_count",
    )
    per(
        "model_service_worker_healthy",
        "gauge",
        "1 if the worker is healthy, else 0.",
        "healthy",
        lambda x: 1 if x else 0,
    )
    per(
        "model_service_worker_draining",
        "gauge",
        "1 if the worker is draining (rolling update), else 0.",
        "draining",
        lambda x: 1 if x else 0,
    )
    per(
        "model_service_worker_available",
        "gauge",
        "1 if the worker is available for routing, else 0.",
        "available_for_routing",
        lambda x: 1 if x else 0,
    )
    per(
        "model_service_worker_reloading",
        "gauge",
        "1 while the worker is reloading weights in place (rollover), else 0.",
        "reloading",
        lambda x: 1 if x else 0,
    )
    per(
        "model_service_worker_restart_count",
        "counter",
        "Process restarts for this worker.",
        "restart_count",
    )
    per(
        "model_service_worker_rolling_restart_count",
        "counter",
        "Rolling-update restarts for this worker.",
        "rolling_restart_count",
    )

    w.family(
        "model_service_worker_module_version",
        "gauge",
        "Per-worker loaded version of each model module "
        "(base / encoder / actor); steps up worker-by-worker during rollover.",
        [
            ({**labels(wk), "module": str(mod)}, ver)
            for wk in workers
            for mod, ver in (wk.get("module_versions") or {}).items()
        ],
    )
    # Per-worker request round-trip latency (router's view) as a histogram.
    w.histogram(
        "model_service_worker_request_duration_seconds",
        "Request round-trip latency (seconds) per worker, as seen by the router.",
        [(labels(wk), wk.get("latency_histogram")) for wk in workers],
    )
    return w.render()


# --------------------------------------------------------------------------- #
# Worker: its own detailed view incl. scheduler / batching (per-port scrape).
# --------------------------------------------------------------------------- #
def render_worker_metrics(snapshot: Dict[str, Any]) -> str:
    w = _Writer()
    wk = snapshot.get("worker", {})
    rq = snapshot.get("requests", {})
    sch = snapshot.get("scheduler")

    w.scalar("model_service_up", "gauge", "1 if the worker server is up.", 1)
    w.scalar(
        "model_service_uptime_seconds",
        "gauge",
        "Seconds since the worker started.",
        wk.get("uptime_s"),
    )
    w.scalar(
        "model_service_batching_enabled",
        "gauge",
        "1 if dynamic batching is enabled on this worker.",
        1 if wk.get("batching_enabled") else 0,
    )

    w.scalar(
        "model_service_requests_total",
        "counter",
        "Inference requests handled by this worker.",
        rq.get("total_requests"),
    )
    w.scalar(
        "model_service_connections_total",
        "counter",
        "Connections accepted by this worker.",
        rq.get("total_connections"),
    )
    w.scalar(
        "model_service_errors_total",
        "counter",
        "Request errors on this worker.",
        rq.get("error_count"),
    )
    w.scalar(
        "model_service_in_flight",
        "gauge",
        "Requests currently in flight on this worker.",
        rq.get("in_flight"),
    )
    w.scalar(
        "model_service_active_connections",
        "gauge",
        "Active client connections on this worker.",
        rq.get("active_connections"),
    )

    # Per-request handling latency as a histogram (mean/p50/p95/p99 derived in
    # Grafana; see module docstring for the queries).
    w.histogram(
        "model_service_request_duration_seconds",
        "Per-request handling latency (seconds) on this worker.",
        [({}, rq.get("latency_histogram"))],
    )

    if sch:
        w.scalar(
            "model_service_scheduler_queue_size",
            "gauge",
            "Requests waiting in the batching queue.",
            sch.get("queue_size"),
        )
        w.scalar(
            "model_service_scheduler_max_batch_size",
            "gauge",
            "Configured max batch size.",
            sch.get("max_batch_size"),
        )
        w.scalar(
            "model_service_scheduler_batches_total",
            "counter",
            "Batches executed.",
            sch.get("total_batches"),
        )
        w.scalar(
            "model_service_scheduler_batch_items_total",
            "counter",
            "Total requests folded into batches.",
            sch.get("total_batch_items"),
        )
        w.scalar(
            "model_service_scheduler_last_batch_size",
            "gauge",
            "Size of the most recent batch.",
            sch.get("last_batch_size"),
        )
        w.scalar(
            "model_service_scheduler_requests_completed_total",
            "counter",
            "Requests completed via the scheduler.",
            sch.get("total_requests_completed"),
        )
        w.scalar(
            "model_service_scheduler_requests_failed_total",
            "counter",
            "Requests failed in the scheduler.",
            sch.get("total_requests_failed"),
        )
        w.scalar(
            "model_service_scheduler_timeouts_total",
            "counter",
            "Requests that hit the scheduler timeout.",
            sch.get("total_timeouts"),
        )
        w.scalar(
            "model_service_scheduler_queue_full_total",
            "counter",
            "Requests rejected because the queue was full.",
            sch.get("queue_full_count"),
        )
    return w.render()
