"""Twin action-branch weights for the residual refiner.

The refiner is a second set of action-branch weights that runs one extra pass
over a finished flow sample. In this model the action branch is not a tower of
its own: every decoder layer holds ``*_action`` projections and norms plus a
``mlp_action_expert`` next to the language weights, and a single layer call
drives both streams. The refiner therefore keeps its weights *beside* the
originals, as ``<name>_refiner`` siblings, and a refinement pass swaps them in
for its duration.

Registering the twins as ordinary sibling modules fixes the module tree at
construction time, so a checkpoint loads into them like any other weight and
``state_dict()`` round-trips without special cases.
"""

from __future__ import annotations

import contextlib
import copy
from typing import Iterator, List, Tuple

import torch.nn as nn

TWIN_SUFFIX = "_refiner"
# The action processor embeds the candidate and owns the output head. It is not
# named ``*_action``, so it is listed explicitly. Its twin is used directly by
# the refinement pass rather than swapped in.
PROCESSOR_ATTR = "action_processor"

_ACTION_SUFFIX = "_action"
_ACTION_EXACT = ("mlp_action_expert",)
# ``normalizer_action`` ends in ``_action`` but holds min/delta statistics, not
# weights; swapping it would renormalize the pass with a different table.
_EXCLUDED_SUBSTRINGS = ("normalizer",)
_SWAPPED_FLAG = "_refiner_swapped"


def is_action_branch(attr: str) -> bool:
    """Whether a child attribute name belongs to the per-layer action branch."""
    if attr.endswith(TWIN_SUFFIX) or any(s in attr for s in _EXCLUDED_SUBSTRINGS):
        return False
    return attr.endswith(_ACTION_SUFFIX) or attr in _ACTION_EXACT


def is_twin_key(state_dict_key: str) -> bool:
    return any(part.endswith(TWIN_SUFFIX) for part in state_dict_key.split("."))


def base_key(twin_key: str) -> str:
    """State-dict key of the frozen weight a twin key mirrors."""
    return ".".join(
        part[: -len(TWIN_SUFFIX)] if part.endswith(TWIN_SUFFIX) else part
        for part in twin_key.split(".")
    )


def _zero_(linear: nn.Module) -> None:
    nn.init.zeros_(linear.weight)
    if getattr(linear, "bias", None) is not None:
        nn.init.zeros_(linear.bias)


def install_twins(model: nn.Module) -> List[str]:
    """Register a ``*_refiner`` twin next to every action-branch submodule.

    ``model`` is the module that owns ``action_processor`` and
    ``language_model`` (the inner VLA model). Twins start as copies of the
    current weights; the processor twin's output head is zeroed so that an
    untrained refiner leaves the sample unchanged. Idempotent.
    """
    created: List[str] = []
    for module_name, module in list(model.named_modules()):
        for attr, child in list(module._modules.items()):
            if child is None:
                continue
            is_processor = module is model and attr == PROCESSOR_ATTR
            if not (is_action_branch(attr) or is_processor):
                continue
            twin_attr = attr + TWIN_SUFFIX
            if twin_attr in module._modules:
                continue
            twin = copy.deepcopy(child)
            if is_processor:
                _zero_(twin.action_proj_back)
            module.add_module(twin_attr, twin)
            created.append(f"{module_name}.{twin_attr}" if module_name else twin_attr)
    if not created and processor_twin(model, required=False) is None:
        raise ValueError(
            f"{type(model).__name__} has no action-branch submodules to twin"
        )
    return created


def has_twins(model: nn.Module) -> bool:
    return processor_twin(model, required=False) is not None


def processor_twin(model: nn.Module, required: bool = True):
    twin = model._modules.get(PROCESSOR_ATTR + TWIN_SUFFIX)
    if twin is None and required:
        raise RuntimeError(
            f"{type(model).__name__} has no refiner twins; the refiner must be "
            "enabled in vla_config when the model is constructed"
        )
    return twin


def _swap_slots(model: nn.Module) -> List[Tuple[nn.Module, str]]:
    return [
        (module, attr)
        for module in model.modules()
        for attr in list(module._modules)
        if is_action_branch(attr) and attr + TWIN_SUFFIX in module._modules
    ]


@contextlib.contextmanager
def swapped_in(model: nn.Module) -> Iterator[int]:
    """Run the action branch on the twin weights for the duration.

    The originals are restored in ``finally``: a forward that raised half-way
    would otherwise leave the policy running on refiner weights, which yields
    plausible but wrong actions instead of an error. Yields the number of
    swapped submodules.
    """
    if model.__dict__.get(_SWAPPED_FLAG):
        raise RuntimeError("refiner twins are already swapped in (not re-entrant)")
    slots = _swap_slots(model)
    if not slots:
        raise RuntimeError("no refiner twins installed")
    originals = [module._modules[attr] for module, attr in slots]
    model.__dict__[_SWAPPED_FLAG] = True
    try:
        for module, attr in slots:
            module._modules[attr] = module._modules[attr + TWIN_SUFFIX]
        yield len(slots)
    finally:
        for (module, attr), original in zip(slots, originals):
            module._modules[attr] = original
        model.__dict__.pop(_SWAPPED_FLAG, None)
