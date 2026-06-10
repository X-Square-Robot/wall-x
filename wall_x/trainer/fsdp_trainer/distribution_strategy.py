"""Distribution strategies for model wrapping and gradient coordination."""

from __future__ import annotations

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


class DistributionStrategy(ABC):
    """Common interface for FSDP2 / DDP wrapping + gradient operations."""

    _cfg: dict
    last_grad_clip_stats: Optional[dict] = None

    def _maybe_enable_grad_ckpt(self, model: torch.nn.Module) -> None:
        """Call ``gradient_checkpointing_enable`` if the model exposes it."""
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
        method()

    @abstractmethod
    def wrap(self, model: torch.nn.Module) -> torch.nn.Module:
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

        FSDP2 always shards (every ``fully_shard`` unit is sharded across
        the mesh). DDP never shards. Used by adapters to decide whether
        per-rank norm contributions need an all-reduce.
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
        name = self._cfg.get("fsdp_sharding_strategy", "full_shard")
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        dp_pg: Optional[dist.ProcessGroup] = None
        shard_pg: Optional[dist.ProcessGroup] = None
        replicate_pg: Optional[dist.ProcessGroup] = None

        # 2D mesh for HSDP variants; 1D for full_shard / shard_grad_op.
        if name in ("hybrid_shard", "_hybrid_shard_zero2"):
            replicate_size = self._cfg.get("fsdp_hsdp_replicate_size") or int(
                os.environ.get("LOCAL_WORLD_SIZE", "0")
            )
            if replicate_size <= 0:
                # Fallback: assume one replica per node, equal-sized shard groups.
                # Caller should set fsdp_hsdp_replicate_size explicitly when this
                # default is wrong (e.g. uneven node sizes).
                replicate_size = max(1, world_size // 8) if world_size >= 8 else 1
            if world_size % replicate_size != 0:
                raise ValueError(
                    f"world_size ({world_size}) not divisible by HSDP replicate "
                    f"size ({replicate_size}); set fsdp_hsdp_replicate_size."
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
            OffloadPolicy(pin_memory=True)
            if self._cfg.get("fsdp_cpu_offload", False)
            else None
        )

        mp_policy: Optional[MixedPrecisionPolicy]
        if self._cfg.get("use_mixed_precision", True):
            dtype = torch.bfloat16 if self._cfg.get("bf16", True) else torch.float16
            reduce_dtype = (
                torch.float32
                if self._cfg.get("fsdp_reduce_dtype", "bf16") == "fp32"
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
    ) -> torch.nn.Module:
        if not hasattr(model, "convert_to_fsdp"):
            raise NotImplementedError(
                f"Model {model.__class__.__name__} has no convert_to_fsdp method"
            )
        self._maybe_enable_grad_ckpt(model)
        layout = self._build_fsdp2_layout()
        self._layout = layout
        wrapped = model.convert_to_fsdp(
            mesh=layout.mesh,
            mp_policy=layout.mp_policy,
            offload_policy=layout.offload_policy,
            reshard_after_forward=layout.reshard_after_forward,
            use_dmuon=use_dmuon,
        )
        torch.cuda.empty_cache()
        return wrapped

    def wrap(self, model: torch.nn.Module) -> torch.nn.Module:
        return self._wrap_fsdp(model, use_dmuon=False)

    def clip_grad_norm(
        self,
        model: torch.nn.Module,
        max_norm: float,
        *,
        optimizer=None,
    ) -> torch.Tensor:
        del optimizer
        self.last_grad_clip_stats = None
        # Separate DTensor (FSDP2) from regular tensor params - PyTorch's
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


class DMuonFSDPStrategy(FSDPStrategy):
    """FSDP2 plus the DMuon dedicated-parameter runtime."""

    @staticmethod
    def _runtime():
        return import_module("dmuon")

    def wrap(self, model: torch.nn.Module) -> torch.nn.Module:
        return self._wrap_fsdp(model, use_dmuon=True)

    def clip_grad_norm(
        self,
        model: torch.nn.Module,
        max_norm: float,
        *,
        optimizer=None,
    ) -> torch.Tensor:
        total_norm = super().clip_grad_norm(model, max_norm, optimizer=optimizer)
        self.last_grad_clip_stats = None
        if optimizer is None:
            return total_norm

        clip_stats = self._runtime().clip_grad_norm_(optimizer, max_norm)
        self.last_grad_clip_stats = clip_stats.as_dict()
        device = next(iter(p for p in model.parameters())).device
        total_norm_t = (
            total_norm.to(device)
            if torch.is_tensor(total_norm)
            else torch.tensor(float(total_norm), device=device)
        )
        dedicated_norm_t = torch.tensor(float(clip_stats.total_norm), device=device)
        return (total_norm_t.pow(2) + dedicated_norm_t.pow(2)).sqrt()

    @contextmanager
    def no_sync(self, model: torch.nn.Module):
        with self._runtime().no_sync(model):
            yield


class DDPStrategy(DistributionStrategy):
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

    def wrap(self, model: torch.nn.Module) -> torch.nn.Module:
        # Adapter.convert_to_mix_precision_hint already placed the model on
        # self._device; this .to() is a defensive no-op (same as the original
        # wrap_ddp_model).
        model = model.to(self._device)
        self._maybe_enable_grad_ckpt(model)
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


def build_strategy(
    config: dict,
    *,
    device: torch.device,
    local_rank: int,
    use_dmuon: bool = False,
) -> DistributionStrategy:
    """Pick FSDP vs DDP based on ``use_fsdp`` flag.

    ``use_gradient_checkpointing`` is read from the config dict by
    each strategy's ``_maybe_enable_grad_ckpt``; not a separate kwarg here.
    """
    if config.get("use_fsdp", False):
        if use_dmuon:
            return DMuonFSDPStrategy(config)
        return FSDPStrategy(config)
    if use_dmuon:
        raise NotImplementedError(
            "DMuon dedicated-parameter training requires "
            "distributed.use_fsdp=true in this trainer."
        )
    return DDPStrategy(
        config,
        device=device,
        local_rank=local_rank,
    )
