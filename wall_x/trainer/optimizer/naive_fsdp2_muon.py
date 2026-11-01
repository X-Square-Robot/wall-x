"""Naive FSDP2 Muon baseline.

This is intentionally inefficient: FSDP2 keeps parameters sharded, but every
rank all-gathers each Muon gradient to a full matrix, runs Newton-Schulz
locally, then drops the full update after redistributing it back to the
parameter's DTensor layout.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterable, Optional

import torch
import torch.distributed as dist
from torch.optim import Optimizer

from wall_x.config.hyperparams_config import NaiveFSDP2MuonConfig
from wall_x.trainer.optimizer.dmuon.utils import _build_ns_backend
from wall_x.trainer.optimizer.utils import register_optimizer

try:
    from torch.distributed.tensor import DTensor, Replicate
except ImportError:  # pragma: no cover - FSDP2 runtime provides this.
    DTensor = None
    Replicate = None

_logger = logging.getLogger(__name__)

_BLOCKED_MUON_NAME_PARTS = (
    "embed_tokens",
    "pos_embed",
    "lm_head",
    "norm",
)


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _is_dtensor(tensor: torch.Tensor) -> bool:
    return DTensor is not None and isinstance(tensor, DTensor)


def _replicate_dtensor(tensor: torch.Tensor) -> torch.Tensor:
    if not _is_dtensor(tensor):
        return tensor
    mesh = tensor.device_mesh
    return tensor.redistribute(placements=[Replicate()] * mesh.ndim)


def _full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    replicated = _replicate_dtensor(tensor)
    if _is_dtensor(replicated):
        return replicated.to_local()
    return replicated


def _redistribute_like(full_tensor: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    if not _is_dtensor(like):
        return full_tensor
    mesh = like.device_mesh
    replicated = DTensor.from_local(
        full_tensor.contiguous(),
        mesh,
        [Replicate()] * mesh.ndim,
        run_check=False,
        shape=like.shape,
        stride=tuple(full_tensor.stride()),
    )
    return replicated.redistribute(placements=like.placements)


def _is_muon_target(name: str, param: torch.Tensor) -> bool:
    if not getattr(param, "requires_grad", False):
        return False
    if param.ndim != 2 or not name.endswith(".weight"):
        return False
    return not any(part in name for part in _BLOCKED_MUON_NAME_PARTS)


def _named_params(model: torch.nn.Module) -> list[tuple[str, torch.Tensor]]:
    return [
        (name, param)
        for name, param in model.named_parameters()
        if getattr(param, "requires_grad", False)
    ]


def _normalize_param_groups(
    model: torch.nn.Module,
    opt_cfg: NaiveFSDP2MuonConfig,
    param_groups: Optional[list[dict]],
) -> list[dict]:
    id_to_name = {id(param): name for name, param in _named_params(model)}

    if param_groups is None:
        muon_params = []
        adamw_params = []
        for name, param in _named_params(model):
            if _is_muon_target(name, param):
                muon_params.append(param)
            else:
                adamw_params.append(param)
        groups = []
        if muon_params:
            groups.append(
                {
                    "params": muon_params,
                    "param_names": [
                        id_to_name.get(id(param), "<unknown>") for param in muon_params
                    ],
                    "group_name": "base/muon",
                    "use_muon": True,
                    "lr": opt_cfg.muon_lr,
                    "weight_decay": opt_cfg.muon_weight_decay,
                    "momentum": opt_cfg.momentum,
                    "nesterov": opt_cfg.nesterov,
                    "ns_steps": opt_cfg.ns_steps,
                }
            )
        if adamw_params:
            groups.append(
                {
                    "params": adamw_params,
                    "param_names": [
                        id_to_name.get(id(param), "<unknown>") for param in adamw_params
                    ],
                    "group_name": "base/adamw",
                    "use_muon": False,
                    "lr": opt_cfg.adamw_lr,
                    "weight_decay": opt_cfg.adamw_weight_decay,
                    "betas": tuple(opt_cfg.adamw_betas),
                    "eps": opt_cfg.adamw_eps,
                }
            )
        return groups

    groups = []
    for idx, user_group in enumerate(param_groups):
        raw_params = [
            param
            for param in user_group.get("params", [])
            if getattr(param, "requires_grad", False)
        ]
        semantic_name = str(
            user_group.get("group_name", user_group.get("name", f"group_{idx}"))
        )
        muon_params = []
        adamw_params = []
        for param in raw_params:
            name = id_to_name.get(id(param), "")
            if _is_muon_target(name, param):
                muon_params.append(param)
            else:
                adamw_params.append(param)

        if muon_params:
            groups.append(
                {
                    "params": muon_params,
                    "param_names": [
                        id_to_name.get(id(param), "<unknown>") for param in muon_params
                    ],
                    "group_name": f"{semantic_name}/muon",
                    "use_muon": True,
                    "lr": user_group.get(
                        "muon_lr", user_group.get("lr", opt_cfg.muon_lr)
                    ),
                    "weight_decay": user_group.get(
                        "muon_weight_decay",
                        user_group.get("weight_decay", opt_cfg.muon_weight_decay),
                    ),
                    "momentum": user_group.get("momentum", opt_cfg.momentum),
                    "nesterov": user_group.get("nesterov", opt_cfg.nesterov),
                    "ns_steps": user_group.get("ns_steps", opt_cfg.ns_steps),
                }
            )
        if adamw_params:
            groups.append(
                {
                    "params": adamw_params,
                    "param_names": [
                        id_to_name.get(id(param), "<unknown>") for param in adamw_params
                    ],
                    "group_name": f"{semantic_name}/adamw",
                    "use_muon": False,
                    "lr": user_group.get(
                        "adamw_lr", user_group.get("lr", opt_cfg.adamw_lr)
                    ),
                    "weight_decay": user_group.get(
                        "adamw_weight_decay",
                        user_group.get("weight_decay", opt_cfg.adamw_weight_decay),
                    ),
                    "betas": tuple(user_group.get("adamw_betas", opt_cfg.adamw_betas)),
                    "eps": user_group.get("adamw_eps", opt_cfg.adamw_eps),
                }
            )
    return groups


def _param_layout_summary(param: torch.Tensor) -> str:
    if not _is_dtensor(param):
        return f"type={type(param).__name__}, shape={tuple(param.shape)}"
    placements = ",".join(str(placement) for placement in param.placements)
    return (
        f"type=DTensor, shape={tuple(param.shape)}, "
        f"local_shape={tuple(param.to_local().shape)}, placements=[{placements}]"
    )


class NaiveFSDP2Muon(Optimizer):
    def __init__(
        self,
        param_groups: list[dict],
        *,
        ns_backend,
        record_step_profile: bool = False,
        log_fn=None,
    ) -> None:
        defaults = {"lr": 1e-3}
        super().__init__(param_groups, defaults)
        self._ns = ns_backend
        self._record_step_profile = bool(record_step_profile)
        self._debug_steps = _env_int("WALLX_NAIVE_FSDP2_MUON_DEBUG_STEPS", 1)
        self._debug_ranks = _env_bool("WALLX_NAIVE_FSDP2_MUON_DEBUG_RANKS", True)
        self._step_index = 0
        self._log_fn = log_fn
        self._last_step_profile: dict[str, object] = {}
        self._last_step_profile_events: list[
            tuple[str, torch.cuda.Event, torch.cuda.Event]
        ] = []
        self._step_profile_enabled = False

    def _log(self, message: str) -> None:
        if self._log_fn is not None:
            self._log_fn(message)
        else:
            _logger.info(message)

    def _profile_begin_step(self) -> None:
        self._step_profile_enabled = (
            self._record_step_profile and torch.cuda.is_available()
        )
        self._last_step_profile_events = []
        self._last_step_profile = {
            "enabled": self._step_profile_enabled,
            "ns_matrix_count": 0,
            "ns_input_numel": 0,
        }

    def _profile_event_start(self, name: str):
        if not self._step_profile_enabled:
            return None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        return name, start, end

    def _profile_event_end(self, token) -> None:
        if token is None:
            return
        _name, _start, end = token
        end.record()
        self._last_step_profile_events.append(token)

    @contextmanager
    def _profile_phase(self, name: str):
        token = self._profile_event_start(name)
        try:
            yield
        finally:
            self._profile_event_end(token)

    def _profile_add(self, key: str, value: int | float) -> None:
        if not self._step_profile_enabled:
            return
        current = self._last_step_profile.get(key, 0)
        self._last_step_profile[key] = current + value

    def consume_last_step_profile(self) -> dict[str, object]:
        profile = dict(self._last_step_profile)
        if not profile:
            return {}
        if not profile.get("enabled"):
            self._last_step_profile_events = []
            return profile
        event_totals: dict[str, float] = {}
        for name, start, end in self._last_step_profile_events:
            if isinstance(start, torch.cuda.Event) and not start.query():
                start.synchronize()
            if isinstance(end, torch.cuda.Event) and not end.query():
                end.synchronize()
            event_totals[name] = event_totals.get(name, 0.0) + float(
                start.elapsed_time(end)
            )
        for name, value in event_totals.items():
            profile[f"{name}_ms"] = round(value, 6)
        profile["ns_compute_ms"] = round(event_totals.get("ns_compute", 0.0), 6)
        self._last_step_profile_events = []
        return profile

    @staticmethod
    def _adjusted_muon_lr(lr: float, shape: Iterable[int]) -> float:
        shape = tuple(int(dim) for dim in shape)
        rows = shape[0]
        cols = 1
        for dim in shape[1:]:
            cols *= dim
        return lr * 0.2 * (max(rows, cols) ** 0.5)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._profile_begin_step()
        full_gather_count = 0
        full_gather_numel = 0
        ns_matrix_count = 0
        muon_param_count = 0
        adamw_param_count = 0

        for group in self.param_groups:
            if group.get("use_muon", False):
                with self._profile_phase("muon"):
                    stats = self._step_muon_group(group)
                full_gather_count += stats["full_gather_count"]
                full_gather_numel += stats["full_gather_numel"]
                ns_matrix_count += stats["ns_matrix_count"]
                muon_param_count += stats["param_count"]
            else:
                with self._profile_phase("adamw"):
                    adamw_param_count += self._step_adamw_group(group)

        if self._step_index < self._debug_steps and (self._debug_ranks or _rank() == 0):
            print(
                "[NaiveFSDP2Muon] "
                f"rank={_rank()}/{_world_size()} step={self._step_index} "
                f"muon_params={muon_param_count} adamw_params={adamw_param_count} "
                f"full_gathers={full_gather_count} "
                f"full_gather_numel={full_gather_numel} "
                f"ns_matrices={ns_matrix_count}",
                flush=True,
            )
        self._step_index += 1
        return loss

    def _step_muon_group(self, group: dict) -> dict[str, int]:
        lr = float(group["lr"])
        weight_decay = float(group.get("weight_decay", 0.0))
        momentum = float(group["momentum"])
        nesterov = bool(group.get("nesterov", True))
        ns_steps = int(group.get("ns_steps", 5))
        stats = {
            "full_gather_count": 0,
            "full_gather_numel": 0,
            "ns_matrix_count": 0,
            "param_count": 0,
        }

        for param in group["params"]:
            grad = param.grad
            if grad is None:
                continue
            stats["param_count"] += 1

            state = self.state[param]
            buf = state.get("momentum_buffer")
            if (
                buf is None
                or tuple(buf.shape) != tuple(grad.shape)
                or _is_dtensor(buf) != _is_dtensor(grad)
            ):
                buf = torch.zeros_like(grad)
                state["momentum_buffer"] = buf
            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(grad)
            if nesterov:
                full_grad = _full_tensor(grad).detach()
                stats["full_gather_count"] += 1
                stats["full_gather_numel"] += int(full_grad.numel())
                full_buf = _full_tensor(buf).detach()
                stats["full_gather_count"] += 1
                stats["full_gather_numel"] += int(full_buf.numel())
                ns_input = full_grad.add(full_buf, alpha=momentum)
            else:
                ns_input = _full_tensor(buf).detach()
                stats["full_gather_count"] += 1
                stats["full_gather_numel"] += int(ns_input.numel())
            if ns_input.ndim > 2:
                ns_input = ns_input.view(ns_input.shape[0], -1)
            with self._profile_phase("ns_compute"):
                update = self._ns.local(ns_input, ns_steps)
            stats["ns_matrix_count"] += 1
            self._profile_add("ns_matrix_count", 1)
            self._profile_add("ns_input_numel", int(ns_input.numel()))

            full_shape = tuple(int(dim) for dim in param.shape)
            update = update.view(full_shape).to(
                device=ns_input.device,
                dtype=(
                    _full_tensor(param).dtype
                    if not _is_dtensor(param)
                    else param.to_local().dtype
                ),
            )
            update = _redistribute_like(update, param)

            if weight_decay > 0:
                param.mul_(1.0 - lr * weight_decay)
            param.add_(update, alpha=-self._adjusted_muon_lr(lr, full_shape))
            if nesterov:
                del full_grad, full_buf
            del ns_input, update

        return stats

    def _step_adamw_group(self, group: dict) -> int:
        lr = float(group["lr"])
        weight_decay = float(group.get("weight_decay", 0.0))
        beta1, beta2 = tuple(group.get("betas", (0.9, 0.999)))
        eps = float(group.get("eps", 1e-8))
        param_count = 0

        for param in group["params"]:
            grad = param.grad
            if grad is None:
                continue
            param_count += 1
            state = self.state[param]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(grad)
                state["exp_avg_sq"] = torch.zeros_like(grad)
            state["step"] += 1
            step = state["step"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.lerp_(grad, 1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            denom = exp_avg_sq.sqrt().add_(eps)
            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            step_size = lr / bias_correction1 * (bias_correction2**0.5)
            if weight_decay > 0:
                param.mul_(1.0 - lr * weight_decay)
            param.addcdiv_(exp_avg, denom, value=-step_size)

        return param_count


def get_naive_fsdp2_muon_optimizer(
    model: torch.nn.Module,
    *,
    opt_cfg,
    param_groups=None,
    log_fn=None,
):
    if not isinstance(opt_cfg, NaiveFSDP2MuonConfig):
        raise TypeError(
            "get_naive_fsdp2_muon_optimizer expects "
            f"NaiveFSDP2MuonConfig, got {type(opt_cfg).__name__}"
        )

    dmuon = __import__("dmuon")
    ns_backend = _build_ns_backend(dmuon, opt_cfg)
    if isinstance(ns_backend, str):
        ns_backend = dmuon.NewtonSchulz(backend=ns_backend)
    groups = _normalize_param_groups(model, opt_cfg, param_groups)
    muon_count = sum(len(group["params"]) for group in groups if group["use_muon"])
    adamw_count = sum(len(group["params"]) for group in groups if not group["use_muon"])
    if _rank() == 0:
        muon_examples = []
        adamw_examples = []
        for group in groups:
            target = muon_examples if group["use_muon"] else adamw_examples
            for name, param in zip(group.get("param_names", []), group["params"]):
                if len(target) >= 6:
                    break
                target.append(f"{name} ({_param_layout_summary(param)})")
        message = (
            "NaiveFSDP2Muon: ordinary FSDP2 wrapping; every rank all-gathers "
            "each Muon gradient/momentum to full matrix and runs Newton-Schulz "
            "locally; optimizer momentum state remains sharded. "
            f"muon_params={muon_count}, adamw_params={adamw_count}, "
            f"ns_backend={opt_cfg.ns_backend}, ns_steps={opt_cfg.ns_steps}, "
            f"muon_examples={muon_examples}, adamw_examples={adamw_examples}"
        )
        if log_fn is not None:
            log_fn(message)
        else:
            _logger.info(message)

    return NaiveFSDP2Muon(
        groups,
        ns_backend=ns_backend,
        record_step_profile=bool(getattr(opt_cfg, "record_step_profile", False)),
        log_fn=log_fn,
    )


register_optimizer("naive_fsdp2_muon", get_naive_fsdp2_muon_optimizer)
