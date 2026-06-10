import torch.nn as nn

from .utils import get_dmuon_optimizer


def is_dmuon_model(model: nn.Module) -> bool:
    """True if ``dmuon.dedicate_params()`` has been applied to this model.

    Checked via an attribute the external ``dmuon`` package attaches to
    the root module, so this predicate works without importing ``dmuon``
    and returns ``False`` for ordinary (non-DMuon) models.
    """
    return hasattr(model, "_dedicated_comm_ctx")


__all__ = ["get_dmuon_optimizer", "is_dmuon_model"]
