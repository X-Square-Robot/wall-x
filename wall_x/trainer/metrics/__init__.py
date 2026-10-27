"""Training metrics: computation (``mfu/``, ``norms``) and transport (``metrics_logger``).

Dependency rule: this package is a leaf. It must not import from
``wall_x.trainer.adapters`` or ``wall_x.trainer.fsdp_trainer`` -- both of
those import it, and a back-edge would be a circular import (the old
``fsdp_trainer/metrics`` location had exactly that problem: adapters
imported it through the eagerly-importing ``fsdp_trainer`` package).
"""

from wall_x.trainer.metrics.metrics_logger import MetricsLogger
from wall_x.trainer.metrics.norms import DistributedNormAccumulator

__all__ = ["MetricsLogger", "DistributedNormAccumulator"]
