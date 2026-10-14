"""Distribution strategies for model wrapping and gradient coordination."""

from __future__ import annotations

import inspect
import logging
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from importlib import import_module
from typing import NamedTuple, Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    OffloadPolicy,
)
from torch.nn.parallel import DistributedDataParallel as DDP

_logger = logging.getLogger(__name__)


class FSDP2Layout(NamedTuple):
    """Per-call FSDP2 wrap configuration produced by ``FSDPStrategy``."""

    mesh: DeviceMesh
    dp_process_group: Optional[dist.ProcessGroup]
    shard_process_group: Optional[dist.ProcessGroup]
    replicate_process_group: Optional[dist.ProcessGroup]
    trainer_process_group: Optional[dist.ProcessGroup]
    mp_policy: Optional[MixedPrecisionPolicy]
    offload_policy: Optional[OffloadPolicy]
    reshard_after_forward: bool


class DMuonLayout(NamedTuple):
    """Pure DMuon wrap configuration produced by ``DMuonStrategy``."""

    mesh: DeviceMesh
    reshard_after_forward: bool


class DistributionStrategy(ABC):
    """Common interface for FSDP2 / pure DMuon / DDP gradient operations."""

    _cfg: dict
    last_grad_clip_stats: Optional[dict] = None

    def _maybe_enable_grad_ckpt(self, model: torch.nn.Module) -> None:
        """Call ``gradient_checkpointing_enable`` if the model exposes it.

        Passes ``num_layers=cfg["gradient_checkpoint_num_layers"]`` when the
        model's enable signature accepts it (partial-ckpt support); otherwise
        falls back to a no-arg call (legacy models = ckpt all layers).
        """
        if not self._cfg.get("use_gradient_checkpointing", False):
            return
        method = getattr(model, "gradient_checkpointing_enable", None)
        if method is None:
            _logger.warning(
                "use_gradient_checkpointing=true but %s has no "
                "gradient_checkpointing_enable method. If this model "
                "routes recomputation through its config the request may "
                "still be honored; otherwise "
                "gradient checkpointing is effectively disabled for "
                "this run.",
                type(model).__name__,
            )
            return
        n_layers = self._cfg.get("gradient_checkpoint_num_layers", None)
        sig_params = inspect.signature(method).parameters
        kwargs = {}
        if "num_layers" in sig_params:
            kwargs["num_layers"] = n_layers
        # Models with config-driven selective recompute take the plan here
        # too: this strategy-level call runs AFTER the adapter's and re-arms
        # every layer, so it must re-apply the same plan to stay idempotent.
        if "recompute" in sig_params:
            kwargs["recompute"] = self._cfg.get("recompute")
        method(**kwargs)

    @abstractmethod
    def wrap(
        self,
        model: torch.nn.Module,
        *,
        use_dedicated_params: bool = False,
        dmuon_owner_kwargs: Optional[dict] = None,
    ) -> torch.nn.Module:
        """Wrap *model* with the active distribution strategy."""

    @abstractmethod
    def clip_grad_norm(
        self,
        model: torch.nn.Module,
        max_norm: float,
        *,
        optimizer=None,
    ) -> torch.Tensor:
        """Clip gradient L2 norm; return total norm before clipping."""

    @abstractmethod
    def no_sync(self, model: torch.nn.Module):
        """Context manager that disables cross-rank gradient sync.

        Used by the trainer for gradient accumulation micro-batches.
        FSDP2 toggles ``set_requires_gradient_sync``; DDP yields
        ``model.no_sync()``.
        """

    @property
    @abstractmethod
    def params_sharded(self) -> bool:
        """Whether parameters are actually sharded across ranks.

        Used by adapters to decide whether per-rank norm contributions need
        an all-reduce.
        """


class FSDPStrategy(DistributionStrategy):
    def __init__(self, config: dict):
        self._cfg = config
        self._layout: Optional[FSDP2Layout] = None
        self.last_grad_clip_stats = None

    @property
    def trainer_process_group(self) -> Optional[dist.ProcessGroup]:
        """Process group for trainer-level collectives.

        Returns `dist.group.WORLD` (the implicit all-rank group created
        by `init_process_group`) so that trainer barriers / metric
        reductions behave consistently regardless of FSDP mesh shape
        (1D for full_shard / shard_grad_op, 2D for hybrid_shard /
        _hybrid_shard_zero2). No extra NCCL communicator is allocated.
        """
        return self._layout.trainer_process_group if self._layout else None

    @staticmethod
    def _build_trainer_process_group(
        world_size: int,
    ) -> Optional[dist.ProcessGroup]:
        if not dist.is_initialized() or world_size <= 1:
            return None
        return dist.group.WORLD

    def _build_fsdp2_layout(self) -> FSDP2Layout:
        """Construct the FSDP2 wrap layout from yaml config.

        Builds the device mesh (1D for full_shard / shard_grad_op, 2D
        for hybrid_shard / _hybrid_shard_zero2), explicit FSDP mesh process
        groups, a trainer-level global process group, the matching
        mixed-precision policy, offload policy, and reshard_after_forward flag.
        """
        name = self._cfg.get("sharding_strategy", "full_shard")
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        dp_pg: Optional[dist.ProcessGroup] = None
        shard_pg: Optional[dist.ProcessGroup] = None
        replicate_pg: Optional[dist.ProcessGroup] = None

        # 2D mesh for HSDP variants; 1D for full_shard / shard_grad_op.
        if name in ("hybrid_shard", "_hybrid_shard_zero2"):
            replicate_size = self._cfg.get("hsdp_replicate_size") or int(
                os.environ.get("LOCAL_WORLD_SIZE", "0")
            )
            if replicate_size <= 0:
                # Fallback: assume one replica per node, equal-sized shard groups.
                # Caller should set hsdp_replicate_size explicitly when this
                # default is wrong (e.g. uneven node sizes).
                replicate_size = max(1, world_size // 8) if world_size >= 8 else 1
            if world_size % replicate_size != 0:
                raise ValueError(
                    f"world_size ({world_size}) not divisible by HSDP replicate "
                    f"size ({replicate_size}); set hsdp_replicate_size."
                )
            shard_size = world_size // replicate_size
            mesh = init_device_mesh(
                "cuda",
                (replicate_size, shard_size),
                mesh_dim_names=("replicate", "shard"),
            )
            replicate_pg = mesh.get_group("replicate")
            shard_pg = mesh.get_group("shard")
        else:
            mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
            dp_pg = mesh.get_group("dp")

        trainer_pg = self._build_trainer_process_group(world_size)

        reshard_after_forward = name not in (
            "shard_grad_op",
            "_hybrid_shard_zero2",
        )

        offload_policy: Optional[OffloadPolicy] = (
            CPUOffloadPolicy(pin_memory=True)
            if self._cfg.get("cpu_offload", False)
            else None
        )

        mp_policy: Optional[MixedPrecisionPolicy]
        if self._cfg.get("use_mixed_precision", True):
            dtype = torch.bfloat16 if self._cfg.get("bf16", True) else torch.float16
            reduce_dtype = (
                torch.float32
                if self._cfg.get("reduce_dtype", "bf16") == "fp32"
                else dtype
            )
            mp_policy = MixedPrecisionPolicy(
                param_dtype=dtype,
                reduce_dtype=reduce_dtype,
                cast_forward_inputs=False,
            )
        else:
            mp_policy = None

        assert mp_policy is not None, (
            "FSDP2 requires a MixedPrecisionPolicy in this torch version. "
            "Set distributed.use_mixed_precision=true."
        )

        return FSDP2Layout(
            mesh=mesh,
            dp_process_group=dp_pg,
            shard_process_group=shard_pg,
            replicate_process_group=replicate_pg,
            trainer_process_group=trainer_pg,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=reshard_after_forward,
        )

    def _wrap_fsdp(
        self,
        model: torch.nn.Module,
        *,
        use_dmuon: bool = False,
        dmuon_owner_kwargs: Optional[dict] = None,
    ) -> torch.nn.Module:
        if not hasattr(model, "convert_to_fsdp"):
            raise NotImplementedError(
                f"Model {model.__class__.__name__} has no convert_to_fsdp method"
            )
        self._maybe_enable_grad_ckpt(model)
        layout = self._build_fsdp2_layout()
        self._layout = layout
        convert_kwargs = dict(
            mesh=layout.mesh,
            mp_policy=layout.mp_policy,
            offload_policy=layout.offload_policy,
            reshard_after_forward=layout.reshard_after_forward,
            use_dmuon=use_dmuon,
            dmuon_owner_kwargs=dmuon_owner_kwargs,
        )
        wrapped = model.convert_to_fsdp(**convert_kwargs)
        torch.cuda.empty_cache()
        return wrapped

    def wrap(
        self,
        model: torch.nn.Module,
        *,
        use_dedicated_params: bool = False,
        dmuon_owner_kwargs: Optional[dict] = None,
    ) -> torch.nn.Module:
        return self._wrap_fsdp(
            model,
            use_dmuon=use_dedicated_params,
            dmuon_owner_kwargs=dmuon_owner_kwargs,
        )

    def clip_grad_norm(
        self,
        model: torch.nn.Module,
        max_norm: float,
        *,
        optimizer=None,
    ) -> torch.Tensor:
        del optimizer
        self.last_grad_clip_stats = None
        # Separate DTensor (FSDP2) from regular tensor params — PyTorch's
        # foreach_mul_ in clip_grad_norm_ can't mix them.
        try:
            from torch.distributed.tensor import DTensor
        except ImportError:
            DTensor = None
        dtensor_params = []
        regular_params = []
        for p in model.parameters():
            if p.grad is None:
                continue
            if DTensor is not None and isinstance(p, DTensor):
                dtensor_params.append(p)
            else:
                regular_params.append(p)

        device = next(iter(p for p in model.parameters())).device
        total_sq = torch.tensor(0.0, device=device)
        if dtensor_params:
            n = torch.nn.utils.clip_grad_norm_(dtensor_params, max_norm)
            total_sq = total_sq + n.to(device).pow(2)
        if regular_params:
            n = torch.nn.utils.clip_grad_norm_(regular_params, max_norm)
            total_sq = total_sq + n.to(device).pow(2)
        return total_sq.sqrt()

    @contextmanager
    def no_sync(self, model: torch.nn.Module):
        if hasattr(model, "_dedicated_comm_ctx"):
            try:
                import dmuon
            except ImportError:
                _logger.warning(
                    "DMuon model detected but dmuon.no_sync is unavailable; "
                    "falling back to FSDP2 gradient-sync gating only."
                )
            else:
                dmuon_no_sync = getattr(dmuon, "no_sync", None)
                if dmuon_no_sync is not None:
                    with dmuon_no_sync(model):
                        yield
                    return
                _logger.warning(
                    "DMuon model detected but installed dmuon has no no_sync(); "
                    "falling back to FSDP2 gradient-sync gating only."
                )

        toggled = hasattr(model, "set_requires_gradient_sync")
        if toggled:
            model.set_requires_gradient_sync(False)
        try:
            yield
        finally:
            if toggled:
                model.set_requires_gradient_sync(True)

    @property
    def params_sharded(self) -> bool:
        return True


class DDPStrategy(DistributionStrategy):
    """PyTorch DistributedDataParallel backend for debug/small-model runs."""

    def __init__(
        self,
        config: dict,
        *,
        device: torch.device,
        local_rank: int,
    ):
        self._cfg = config
        self._device = device
        self._local_rank = local_rank
        self.find_unused_parameters = config.get("find_unused_parameters", False)
        self.broadcast_buffers = config.get("broadcast_buffers", True)
        self.bucket_cap_mb = config.get("bucket_cap_mb", 25)
        self.last_grad_clip_stats = None

    def wrap(
        self,
        model: torch.nn.Module,
        *,
        use_dedicated_params: bool = False,
        dmuon_owner_kwargs: Optional[dict] = None,
    ) -> torch.nn.Module:
        if use_dedicated_params or dmuon_owner_kwargs:
            raise ValueError("DDPStrategy does not support DMuon dedicated params")
        self._maybe_enable_grad_ckpt(model)
        model = model.to(self._device)
        return DDP(
            model,
            device_ids=[self._local_rank],
            output_device=self._local_rank,
            find_unused_parameters=self.find_unused_parameters,
            broadcast_buffers=self.broadcast_buffers,
            bucket_cap_mb=self.bucket_cap_mb,
        )

    def clip_grad_norm(
        self,
        model: torch.nn.Module,
        max_norm: float,
        *,
        optimizer=None,
    ) -> torch.Tensor:
        del optimizer
        self.last_grad_clip_stats = None
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    @contextmanager
    def no_sync(self, model: torch.nn.Module):
        with model.no_sync():
            yield

    @property
    def params_sharded(self) -> bool:
        return False


class DMuonStrategy(DistributionStrategy):
    """Pure DMuon dedicated-parameter runtime, without FSDP2 wrapping."""

    def __init__(self, config: dict):
        self._cfg = config
        self._layout: Optional[DMuonLayout] = None
        self.last_grad_clip_stats = None
        # dmuon clip stats are 0-dim device tensors (defer-only contract);
        # every consumer below must keep them on-device -- the trainer
        # materializes all clip scalars in ONE packed D2H at metrics build.
        # Static clip-partition cache, built on the first clip call:
        # dedicated ownership is fixed at wrap time, so the non-dedicated
        # candidate list and the model device never change per step.
        self._clip_cache_model_id: Optional[int] = None
        self._clip_regular_candidates: list = []
        self._clip_device: Optional[torch.device] = None
        # Official dmuon fast path for "the Wall-X pattern" (see dmuon
        # docs/guides/training.md "Fused CUDA clip"): one CUDA pass + ONE
        # packed all_reduce + ONE packed D2H for all three buckets, with
        # per-bucket semantics identical to the manual path below. Resolved
        # ONCE here (extension availability is a static install property);
        # the per-step code never probes. None -> manual fallback path.
        self._buckets_clip_fn = None
        try:
            from dmuon.fast_clip import _load_fast_clip_extension
            from dmuon import try_clip_optimizer_grad_norm_buckets_

            if _load_fast_clip_extension() is not None:
                self._buckets_clip_fn = try_clip_optimizer_grad_norm_buckets_
        except Exception as _fc_exc:  # noqa: BLE001
            print(
                "[WallX DMuon] fast-clip probe failed, using manual clip "
                f"path: {type(_fc_exc).__name__}: {_fc_exc}"
            )
        print(
            "[WallX DMuon] grad-clip path: "
            + (
                "dmuon segmented CUDA fastpath"
                if self._buckets_clip_fn
                else "manual (foreach/defer)"
            )
        )

    @staticmethod
    def _runtime():
        return import_module("dmuon")

    @staticmethod
    def _debug_sharded_adamw_steps() -> int:
        try:
            return max(
                0, int(os.environ.get("WALLX_DMUON_SHARDED_ADAMW_DEBUG_STEPS", "0"))
            )
        except ValueError:
            return 0

    @staticmethod
    def _debug_sharded_adamw_log_limit() -> int:
        try:
            return max(
                0,
                int(os.environ.get("WALLX_DMUON_SHARDED_ADAMW_DEBUG_LOG_LIMIT", "24")),
            )
        except ValueError:
            return 24

    @staticmethod
    def _tensor_finite_stats(tensor: torch.Tensor) -> dict:
        detached = tensor.detach()
        finite_mask = torch.isfinite(detached)
        all_finite = bool(finite_mask.all().item())
        stats = {
            "finite": all_finite,
            "numel": int(detached.numel()),
            "dtype": str(detached.dtype),
            "shape": tuple(int(dim) for dim in detached.shape),
            "nan": 0,
            "inf": 0,
            "amax": float("nan"),
        }
        if all_finite:
            if detached.numel() > 0:
                stats["amax"] = float(detached.float().abs().max().cpu().item())
            else:
                stats["amax"] = 0.0
            return stats
        stats["nan"] = int(torch.isnan(detached).sum().cpu().item())
        stats["inf"] = int(torch.isinf(detached).sum().cpu().item())
        return stats

    @staticmethod
    def _format_finite_stats(stats: dict) -> str:
        return (
            f"finite={stats['finite']} shape={stats['shape']} "
            f"dtype={stats['dtype']} numel={stats['numel']} "
            f"nan={stats['nan']} inf={stats['inf']} amax={stats['amax']}"
        )

    @staticmethod
    def _maybe_install_sharded_adamw_debug_hooks(model: torch.nn.Module) -> None:
        debug_steps = DMuonStrategy._debug_sharded_adamw_steps()
        if debug_steps <= 0:
            return
        comm_ctx = getattr(model, "_dedicated_comm_ctx", None)
        if comm_ctx is None:
            return
        installed = 0
        for state in getattr(comm_ctx, "all_states", []):
            group = getattr(state, "group", None)
            if group is None:
                continue
            group_name = str(
                getattr(group, "_debug_name", None) or f"group_{id(group):x}"
            )
            for dparam in getattr(group, "params", []):
                if getattr(dparam, "_dmuon_route", None) != "sharded_adamw":
                    continue
                param = getattr(dparam, "_unsharded_param", None)
                if param is None:
                    continue
                if getattr(dparam, "_wallx_sharded_adamw_debug_hook", None) is not None:
                    continue
                name = f"{group_name}.{getattr(dparam, 'param_name', '<unnamed>')}"
                dparam._wallx_sharded_adamw_debug_name = name
                dparam._wallx_sharded_adamw_raw_calls = 0
                dparam._wallx_sharded_adamw_raw_stats = None

                def _make_hook(dp):
                    def _hook(grad):
                        calls = int(getattr(dp, "_wallx_sharded_adamw_raw_calls", 0))
                        if calls >= debug_steps:
                            return grad
                        stats = DMuonStrategy._tensor_finite_stats(grad)
                        dp._wallx_sharded_adamw_raw_calls = calls + 1
                        dp._wallx_sharded_adamw_raw_stats = stats
                        if not stats["finite"]:
                            rank = dist.get_rank() if dist.is_initialized() else 0
                            print(
                                "[WallX DMuon ShardedAdamW Debug] "
                                f"rank={rank} phase=raw_grad call={calls} "
                                f"name={getattr(dp, '_wallx_sharded_adamw_debug_name', '<unknown>')} "
                                f"{DMuonStrategy._format_finite_stats(stats)}",
                                flush=True,
                            )
                        return grad

                    return _hook

                dparam._wallx_sharded_adamw_debug_hook = param.register_hook(
                    _make_hook(dparam)
                )
                installed += 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(
            "[WallX DMuon ShardedAdamW Debug] "
            f"rank={rank} installed_hooks={installed} debug_steps={debug_steps}",
            flush=True,
        )

    @property
    def trainer_process_group(self) -> Optional[dist.ProcessGroup]:
        return None

    def _build_dmuon_layout(self) -> DMuonLayout:
        name = self._cfg.get("sharding_strategy", "shard_grad_op")
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        if name in ("hybrid_shard", "_hybrid_shard_zero2"):
            replicate_size = self._cfg.get("hsdp_replicate_size") or int(
                os.environ.get("LOCAL_WORLD_SIZE", "0")
            )
            if replicate_size <= 0:
                replicate_size = max(1, world_size // 8) if world_size >= 8 else 1
            if world_size % replicate_size != 0:
                raise ValueError(
                    f"world_size ({world_size}) not divisible by DMuon HSDP "
                    f"replicate size ({replicate_size}); set "
                    "hsdp_replicate_size."
                )
            mesh = init_device_mesh(
                "cuda",
                (replicate_size, world_size // replicate_size),
                mesh_dim_names=("replicate", "shard"),
            )
        else:
            mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))

        reshard_after_forward = name not in (
            "shard_grad_op",
            "_hybrid_shard_zero2",
        )
        return DMuonLayout(
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
        )

    def wrap(
        self,
        model: torch.nn.Module,
        *,
        use_dedicated_params: bool = False,
        dmuon_owner_kwargs: Optional[dict] = None,
    ) -> torch.nn.Module:
        del use_dedicated_params
        if not hasattr(model, "convert_to_dmuon"):
            raise NotImplementedError(
                f"Model {model.__class__.__name__} has no convert_to_dmuon method"
            )
        self._maybe_enable_grad_ckpt(model)
        layout = self._build_dmuon_layout()
        self._layout = layout
        wrapped = model.convert_to_dmuon(
            mesh=layout.mesh,
            reshard_after_forward=layout.reshard_after_forward,
            dmuon_owner_kwargs=dmuon_owner_kwargs,
        )
        self._maybe_install_sharded_adamw_debug_hooks(wrapped)
        torch.cuda.empty_cache()
        return wrapped

    @staticmethod
    def _iter_dedicated_adamw_grads(optimizer):
        seen = set()
        for dedicated_params in getattr(optimizer, "_adamw_group_dps", {}).values():
            for dparam in dedicated_params:
                dparam_id = id(dparam)
                if dparam_id in seen:
                    continue
                seen.add(dparam_id)
                if getattr(dparam, "_dmuon_route", None) == "sharded_adamw":
                    grad = getattr(dparam, "_sharded_adamw_grad", None)
                else:
                    grad = getattr(dparam, "_reduced_grad", None)
                if grad is not None:
                    yield dparam, grad

    @staticmethod
    def _maybe_log_sharded_adamw_reduce_debug(optimizer, entries) -> None:
        debug_steps = DMuonStrategy._debug_sharded_adamw_steps()
        if debug_steps <= 0:
            return
        step = int(getattr(optimizer, "_wallx_sharded_adamw_debug_step", 0))
        if step >= debug_steps:
            return
        log_limit = DMuonStrategy._debug_sharded_adamw_log_limit()
        rank = dist.get_rank() if dist.is_initialized() else 0
        total = 0
        raw_nonfinite = 0
        shard_nonfinite = 0
        logged = 0
        for dparam, grad in entries:
            if getattr(dparam, "_dmuon_route", None) != "sharded_adamw":
                continue
            total += 1
            shard_stats = DMuonStrategy._tensor_finite_stats(grad)
            reduce_input = getattr(dparam, "_sharded_adamw_reduce_input", None)
            reduce_input_stats = (
                None
                if reduce_input is None
                else DMuonStrategy._tensor_finite_stats(reduce_input)
            )
            raw_stats = getattr(dparam, "_wallx_sharded_adamw_raw_stats", None)
            raw_finite = None if raw_stats is None else bool(raw_stats["finite"])
            if raw_stats is not None and not raw_stats["finite"]:
                raw_nonfinite += 1
            if not shard_stats["finite"]:
                shard_nonfinite += 1
            should_log = (
                not shard_stats["finite"] or raw_finite is False or logged < log_limit
            )
            if should_log:
                logged += 1
                name = getattr(
                    dparam,
                    "_wallx_sharded_adamw_debug_name",
                    getattr(dparam, "param_name", "<unknown>"),
                )
                raw_msg = (
                    "raw_grad=missing"
                    if raw_stats is None
                    else "raw_grad=" + DMuonStrategy._format_finite_stats(raw_stats)
                )
                reduce_input_msg = (
                    "reduce_input=missing"
                    if reduce_input_stats is None
                    else "reduce_input="
                    + DMuonStrategy._format_finite_stats(reduce_input_stats)
                )
                print(
                    "[WallX DMuon ShardedAdamW Debug] "
                    f"rank={rank} step={step} phase=post_reduce "
                    f"name={name} {raw_msg} "
                    f"{reduce_input_msg} "
                    f"shard_grad={DMuonStrategy._format_finite_stats(shard_stats)}",
                    flush=True,
                )
        print(
            "[WallX DMuon ShardedAdamW Debug] "
            f"rank={rank} step={step} phase=post_reduce_summary "
            f"total={total} raw_nonfinite={raw_nonfinite} "
            f"shard_nonfinite={shard_nonfinite}",
            flush=True,
        )
        optimizer._wallx_sharded_adamw_debug_step = step + 1

    @staticmethod
    @torch.no_grad()
    def _clip_dedicated_adamw_grad_norm(
        optimizer,
        max_norm: float,
        *,
        device: torch.device,
    ) -> dict:
        entries = list(DMuonStrategy._iter_dedicated_adamw_grads(optimizer))
        DMuonStrategy._maybe_log_sharded_adamw_reduce_debug(optimizer, entries)
        grads = [grad for _dparam, grad in entries]

        # Fallback path (fast-clip extension unavailable). foreach-batched
        # norms: one multi-tensor pass per (device, dtype) group. NOTE
        # _foreach_norm rounds each per-tensor norm once in the input dtype;
        # the logged adamw_total_norm can differ from an exact-fp32
        # accumulation in the ~3rd decimal (training-neutral).
        local_sq = torch.zeros((), device=device, dtype=torch.float32)
        by_dev_dtype: dict = {}
        for grad in grads:
            by_dev_dtype.setdefault((grad.device, grad.dtype), []).append(grad)
        for (_gdev, _gdt), grad_list in by_dev_dtype.items():
            norms = torch._foreach_norm(grad_list, 2.0)
            local_sq = local_sq + torch.stack(norms).float().square().sum().to(device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
        total_norm = torch.sqrt(local_sq)
        clip_coef = float(max_norm) / (total_norm + 1e-6)
        # branchless nan guard: keeps the clip path free of D2H syncs
        clip_coef_clamped = torch.nan_to_num(torch.clamp(clip_coef, max=1.0), nan=1.0)
        for (gdev, _gdt), grad_list in by_dev_dtype.items():
            torch._foreach_mul_(grad_list, clip_coef_clamped.to(device=gdev))

        # Stats stay ON-DEVICE (0-d tensors); the trainer materializes all
        # clip scalars in one packed D2H at metrics-build time.
        finite_t = torch.isfinite(total_norm)
        return {
            "total_norm": total_norm.detach().float(),
            "clip_coef": clip_coef_clamped.detach().float(),
            "clipped": (finite_t & (clip_coef.detach() < 1.0)).float(),
            "param_count": len(grads),
            "found_inf": (~finite_t).float(),
        }

    def clip_grad_norm(
        self,
        model: torch.nn.Module,
        max_norm: float,
        *,
        optimizer=None,
    ) -> torch.Tensor:
        self.last_grad_clip_stats = None
        if self._clip_cache_model_id != id(model):
            # one-time partition: dedicated ownership never changes post-wrap
            self._clip_regular_candidates = [
                p for p in model.parameters() if not hasattr(p, "_dedicated_owner_rank")
            ]
            try:
                self._clip_device = next(iter(p for p in model.parameters())).device
            except StopIteration:
                self._clip_device = torch.device("cuda", torch.cuda.current_device())
            self._clip_cache_model_id = id(model)
        device = self._clip_device

        # per-step: only the grad-presence check, over the (in pure DMuon
        # mode: empty) cached candidate list
        regular_params = [
            p for p in self._clip_regular_candidates if p.grad is not None
        ]

        if self._buckets_clip_fn is not None and optimizer is not None:
            # dedicated grads (_reduced_grad/_sharded_adamw_grad) only
            # materialize at the reduce rendezvous -- run it BEFORE
            # collecting the adamw bucket. Idempotent: the fast-clip call
            # below re-invokes it and hits the _grads_ready guard.
            ensure = getattr(optimizer, "_ensure_grads_ready", None)
            if ensure is not None:
                try:
                    ensure(coalesce_wait=True)
                except TypeError:
                    ensure()
            adamw_entries = list(self._iter_dedicated_adamw_grads(optimizer))
            result = self._buckets_clip_fn(
                optimizer,
                regular_grads=[p.grad for p in regular_params],
                adamw_grads=[grad for _dp, grad in adamw_entries],
                max_norm=max_norm,
            )
            if result is not None:
                stats = result.stats_by_name
                muon_s = stats.get("muon")
                adamw_s = stats.get("adamw")
                merged = muon_s.as_dict() if muon_s is not None else {}
                merged.pop("name", None)
                if adamw_s is not None:
                    merged.update(
                        {
                            "adamw_total_norm": adamw_s.total_norm,
                            "adamw_clip_coef": adamw_s.clip_coef,
                            "adamw_clipped": adamw_s.clipped,
                            "adamw_param_count": adamw_s.param_count,
                            "adamw_found_inf": adamw_s.found_inf,
                        }
                    )
                self.last_grad_clip_stats = merged
                total = result.total_norm
                return (
                    total.to(device)
                    if torch.is_tensor(total)
                    else torch.tensor(float(total), device=device)
                )
            # rank-consistent None (extension refused): fall through to the
            # manual path -- consistent across ranks per dmuon's contract

        if regular_params:
            total_norm = torch.nn.utils.clip_grad_norm_(regular_params, max_norm)
            total_norm_t = total_norm.to(device)
        else:
            total_norm_t = torch.tensor(0.0, device=device)

        if optimizer is None:
            return total_norm_t

        clip_stats = self._runtime().clip_grad_norm_(optimizer, max_norm)
        adamw_clip_stats = self._clip_dedicated_adamw_grad_norm(
            optimizer,
            max_norm,
            device=device,
        )
        self.last_grad_clip_stats = clip_stats.as_dict()
        self.last_grad_clip_stats.update(
            {
                "adamw_total_norm": adamw_clip_stats["total_norm"],
                "adamw_clip_coef": adamw_clip_stats["clip_coef"],
                "adamw_clipped": adamw_clip_stats["clipped"],
                "adamw_param_count": adamw_clip_stats["param_count"],
                "adamw_found_inf": adamw_clip_stats["found_inf"],
            }
        )
        # dmuon clip stats arrive as 0-dim device tensors (defer-only
        # contract); combine everything on-device, no host round-trips.
        muon_norm = clip_stats.total_norm
        dedicated_norm_t = (
            muon_norm.to(device=device, dtype=torch.float32)
            if torch.is_tensor(muon_norm)
            else torch.tensor(float(muon_norm), device=device)
        )
        dedicated_adamw_norm_t = adamw_clip_stats["total_norm"].to(
            device=device, dtype=torch.float32
        )
        return (
            total_norm_t.pow(2)
            + dedicated_norm_t.pow(2)
            + dedicated_adamw_norm_t.pow(2)
        ).sqrt()

    @contextmanager
    def no_sync(self, model: torch.nn.Module):
        with self._runtime().no_sync(model):
            yield

    @property
    def params_sharded(self) -> bool:
        return True


def build_strategy(
    config: dict,
    *,
    device: torch.device,
    local_rank: int,
    use_dmuon: bool | None = None,
    optimizer_type: str | None = None,
) -> DistributionStrategy:
    """Pick the public distributed backend: FSDP2, pure DMuon, or DDP."""

    backend = str(config.get("backend", "fsdp2"))
    if use_dmuon is None:
        use_dmuon = optimizer_type == "dmuon"
    if optimizer_type == "naive_fsdp2_muon" and backend != "fsdp2":
        raise ValueError(
            "optimizer_type='naive_fsdp2_muon' requires distributed.backend='fsdp2'."
        )
    common = {
        "use_gradient_checkpointing": config.get("use_gradient_checkpointing", False),
        "use_mixed_precision": config.get("use_mixed_precision", True),
        "bf16": config.get("bf16", True),
        # Selective-recompute plan (plain dict after dataclasses.asdict);
        # models whose enable method accepts ``recompute`` rehydrate it.
        "recompute": config.get("recompute"),
    }
    # Forward top-level distributed keys the strategies consume directly. The
    # current schema nests sharding/replicate under ``fsdp:``/``dmuon:``, but
    # this model-level knob lives at distributed top level; without forwarding
    # it, _maybe_enable_grad_ckpt silently falls back to checkpointing ALL
    # layers. ``if k in config`` keeps new-schema configs (keys under the
    # active section) untouched, since the section overlay below wins.
    for _k in ("gradient_checkpoint_num_layers",):
        if _k in config:
            common[_k] = config[_k]
    if backend == "fsdp2":
        if use_dmuon:
            raise ValueError(
                "FSDP2 + DMuon dedicated params is no longer supported. "
                "Use distributed.backend=dmuon for pure DMuon."
            )
        return FSDPStrategy({**common, **dict(config.get("fsdp", {}))})
    if backend == "dmuon":
        if optimizer_type is not None and optimizer_type != "dmuon":
            raise ValueError(
                "distributed.backend=dmuon requires optimizer_type='dmuon'."
            )
        if not use_dmuon:
            raise ValueError(
                "distributed.backend=dmuon requires optimizer_type='dmuon'."
            )
        return DMuonStrategy({**common, **dict(config.get("dmuon", {}))})
    if backend == "ddp":
        if use_dmuon or optimizer_type == "dmuon":
            raise ValueError(
                "distributed.backend=ddp is not compatible with optimizer_type='dmuon'."
            )
        return DDPStrategy(
            {**common, **dict(config.get("ddp", {}))},
            device=device,
            local_rank=local_rank,
        )
    raise ValueError(
        f"Unknown distributed.backend={backend!r}; expected 'fsdp2', "
        "'dmuon', or 'ddp'."
    )
