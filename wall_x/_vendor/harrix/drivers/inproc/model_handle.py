"""In-process model handle.

The model is constructed in the driver process and calls the adapter directly.
"""

from __future__ import annotations


class DirectModelHandle:
    def __init__(self, cfg):
        # Trigger adapter registration.
        import wall_x._vendor.harrix.adapters  # noqa: F401
        from wall_x._vendor.harrix.adapters.registry import build_adapter

        self._adapter = build_adapter(cfg)

    @property
    def chunk_horizon(self) -> int:
        return self._adapter.chunk_horizon

    def predict_batch(self, payloads):
        return self._adapter.predict_batch(payloads)

    def shutdown(self) -> None:
        pass
