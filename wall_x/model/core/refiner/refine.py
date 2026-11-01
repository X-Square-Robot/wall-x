"""One-pass residual refinement of a finished flow sample.

The base policy integrates its flow for ``num_steps`` Euler steps and produces a
candidate chunk ``a``. The refiner then runs the action branch once more, on its
own twin weights, over the same observation and returns::

    a_refined = a + r(observation, a)

The pass conditions on a fixed timestep (the candidate is treated as clean
data) and reads the observation through the unchanged attention, so on the
cached sampler it reuses the prefix K/V that integration already built.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import torch

from wall_x.model.core.refiner.twins import (
    base_key,
    has_twins,
    is_twin_key,
    processor_twin,
    swapped_in,
)


@dataclass(frozen=True)
class RefinerConfig:
    """``vla_config.refiner``.

    ``num_steps`` is the integration budget the refiner was trained on. The
    refiner learned the residual of *that* candidate distribution, so serving
    with a different budget is refused rather than silently degraded.

    ``refine_loops`` chains the pass: each iteration refines the previous
    estimate, which is why loop checkpoints trained that way cannot be served
    as a single pass. ``refine_step_scale`` is the fraction of each pass's
    residual taken (``1.0`` with one loop is the plain ``candidate + residual``
    pass every pre-loop checkpoint was trained with, and stays bit-identical).
    """

    enabled: bool = False
    num_steps: Optional[int] = None
    timestep: float = 1.0
    refine_loops: int = 1
    refine_step_scale: float = 1.0

    @classmethod
    def from_vla_config(cls, vla_config: Any) -> "RefinerConfig":
        raw = getattr(vla_config, "refiner", None)
        if raw is None:
            return cls()
        if isinstance(raw, RefinerConfig):
            return raw
        get = raw.get if isinstance(raw, dict) else (lambda k, d=None: getattr(raw, k, d))
        cfg = cls(
            enabled=bool(get("enabled", False)),
            num_steps=None if get("num_steps") is None else int(get("num_steps")),
            timestep=float(get("timestep", 1.0)),
            refine_loops=int(get("refine_loops", 1)),
            refine_step_scale=float(get("refine_step_scale", 1.0)),
        )
        if cfg.enabled and (cfg.num_steps is None or cfg.num_steps < 1):
            raise ValueError("vla_config.refiner.num_steps must be a positive integer")
        if cfg.refine_loops < 1:
            raise ValueError("vla_config.refiner.refine_loops must be >= 1")
        if not cfg.refine_step_scale > 0.0:
            raise ValueError(
                "vla_config.refiner.refine_step_scale must be positive"
            )
        return cfg


def refine(
    model: torch.nn.Module,
    candidate: torch.Tensor,
    *,
    dof_mask: Optional[torch.Tensor],
    embed_dtype: torch.dtype,
    run_lm: Callable[[torch.Tensor, Any], Any],
    config: RefinerConfig,
) -> torch.Tensor:
    """Refine ``candidate`` with ``config.refine_loops`` residual passes.

    ``model`` owns ``action_processor`` and the twins. ``run_lm(action_embeds,
    adarms_conds)`` drives the decoder the same way the sampler's own velocity
    evaluations do, so the refiner sees exactly the observation the candidate
    was sampled from.

    Each pass conditions on the current estimate and returns its residual; the
    estimate moves by ``refine_step_scale`` of it. With one pass at scale 1.0
    (every pre-loop checkpoint) this is ``candidate + residual``, unchanged.
    """
    batch, horizon, dim = candidate.shape
    processor = processor_twin(model)
    timestep = torch.full(
        (batch,), config.timestep, device=candidate.device, dtype=torch.float32
    )
    action_hidden = model.config.vla_config.dim_inputs[1]
    head = processor.action_proj_back
    refined = candidate
    for _ in range(config.refine_loops):
        with swapped_in(model):
            embeds, adarms = processor.step(
                timestep=timestep, noisy_action=refined, dof_mask=dof_mask
            )
            output = run_lm(embeds[:, :, :action_hidden].to(embed_dtype), adarms)
        residual = head(output.action_embeds.to(head.weight.dtype)).float()
        residual = residual.reshape(batch, horizon, dim)
        # Scale 1.0 takes the residual as-is, so one-pass checkpoints stay
        # bit-identical to the single-pass release.
        if config.refine_step_scale == 1.0:
            refined = refined + residual
        else:
            refined = refined + config.refine_step_scale * residual
    return refined


def verify_loaded(
    model: torch.nn.Module,
    checkpoint_keys: Iterable[str],
    missing_keys: Iterable[str],
    config: RefinerConfig,
) -> int:
    """Refuse any load that would quietly serve the base policy.

    With ``strict=False`` loading, a refiner switch that did not reach the
    serving config drops every twin tensor as unexpected, and twins that were
    never filled run as an untrained (identity) refiner. Both raise here.
    A twin of a module the checkpoint does not carry at all is left alone:
    the frozen branch has the same gap, so nothing is lost. Returns the number
    of twin tensors in the checkpoint.
    """
    checkpoint_keys = set(checkpoint_keys)
    twin_keys = [k for k in checkpoint_keys if is_twin_key(k)]
    if not config.enabled:
        if twin_keys:
            raise RuntimeError(
                f"checkpoint carries {len(twin_keys)} refiner tensors "
                f"(e.g. {twin_keys[0]}) but vla_config.refiner is disabled; "
                "they would be dropped and the base policy served instead"
            )
        return 0
    if not has_twins(model):
        raise RuntimeError("vla_config.refiner is enabled but the model has no twins")
    if not twin_keys:
        raise RuntimeError(
            "vla_config.refiner is enabled but the checkpoint has no refiner "
            "weights; serving would run an untrained refiner"
        )
    unfilled = [
        k for k in missing_keys if is_twin_key(k) and base_key(k) in checkpoint_keys
    ]
    if unfilled:
        raise RuntimeError(
            f"{len(unfilled)} refiner tensors were not loaded (e.g. {unfilled[0]})"
        )
    return len(twin_keys)
