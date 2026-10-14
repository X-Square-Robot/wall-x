"""Wall-X DMuon parameter policy helpers."""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import torch
from torch import nn


RouteHintFn = Callable[[str, nn.Parameter], Optional[str]]

WALLX_ACTION_FP32_PREFIXES = (
    "action_processor.",
    "action_preprocessor.",
    "model.action_processor.",
    "model.action_preprocessor.",
)
WALLX_ACTION_FP32_NAME_PARTS = WALLX_ACTION_FP32_PREFIXES


def make_dmuon_param_policy_fn(
    route_hint_fn: RouteHintFn,
    *,
    fp32_name_prefixes: tuple[str, ...] = (),
    fp32_name_parts: tuple[str, ...] = (),
    default_param_dtype: torch.dtype = torch.bfloat16,
) -> Callable[[str, nn.Parameter], Optional[dict[str, object]]]:
    """Convert Wall-X route selection into DMuon's structured param policy."""

    fp32_name_prefixes = tuple(prefix for prefix in fp32_name_prefixes if prefix)
    # Backward-compatible alias for older call sites. The new path uses explicit
    # prefixes so qwen action-expert matrices stay on bf16 Tensor Core kernels.
    if not fp32_name_prefixes and fp32_name_parts:
        fp32_name_prefixes = tuple(part for part in fp32_name_parts if part)

    def param_policy_fn(name: str, param: nn.Parameter) -> Optional[dict[str, object]]:
        route = route_hint_fn(name, param)
        if route is None:
            return None

        policy: dict[str, object] = {
            "route": route,
            "param_dtype": default_param_dtype,
        }
        if any(name.startswith(prefix) for prefix in fp32_name_prefixes):
            policy.update(
                {
                    "param_dtype": torch.float32,
                    "grad_dtype": torch.float32,
                    "cast_forward_inputs": True,
                }
            )
        return policy

    return param_policy_fn


def make_wallx_action_fp32_dmuon_param_policy_fn(
    route_hint_fn: RouteHintFn,
) -> Callable[[str, nn.Parameter], Optional[dict[str, object]]]:
    """Keep WallX action processor/head compute in fp32 under pure DMuon.

    The route still comes from the model-specific route selector. This helper
    only overlays dtype policy: ordinary WallX parameters materialize in bf16,
    while action processor/head parameters materialize gradients and forward
    weights in fp32.
    """

    return make_dmuon_param_policy_fn(
        route_hint_fn,
        fp32_name_prefixes=WALLX_ACTION_FP32_PREFIXES,
    )


def summarize_dmuon_param_policy(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    param_policy_fn: Callable[[str, nn.Parameter], Optional[dict[str, object]]],
    *,
    max_examples_per_key: int = 6,
) -> tuple[dict[str, int], dict[str, list[str]]]:
    """Return compact policy counts/examples for training logs."""

    counts: dict[str, int] = {}
    examples: dict[str, list[str]] = {}
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        policy = param_policy_fn(name, param)
        if policy is None:
            key = "unmanaged"
        else:
            route = policy.get("route")
            param_dtype = policy.get("param_dtype")
            grad_dtype = policy.get("grad_dtype")
            key = f"route={route},param_dtype={param_dtype}," f"grad_dtype={grad_dtype}"
        counts[key] = counts.get(key, 0) + 1
        bucket = examples.setdefault(key, [])
        if len(bucket) < max_examples_per_key:
            bucket.append(name)
    return counts, examples


__all__ = [
    "WALLX_ACTION_FP32_NAME_PARTS",
    "WALLX_ACTION_FP32_PREFIXES",
    "make_dmuon_param_policy_fn",
    "make_wallx_action_fp32_dmuon_param_policy_fn",
    "summarize_dmuon_param_policy",
]
