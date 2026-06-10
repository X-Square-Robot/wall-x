"""
Base Trainer for Pure PyTorch Distributed Training (DDP/FSDP)
No accelerate dependency - can be launched directly with torchrun
"""

import contextlib
import gc
import logging
import os
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any, Optional

import torch
import torch.distributed as dist

from wall_x.config.hyperparams_config import AdamWConfig
from wall_x.config.train_config import TrainConfig
from wall_x.trainer.trainer_utils import seed_all
from wall_x.utils.logger import DistributedLogger
from wall_x.utils.timers import Timers


def setup_distributed():
    """Initialize distributed training environment.

    NCCL watchdog default is 10 min - some dataset backends (e.g. lerobot's
    per-rank sequential LeRobotDataset indexing) can exceed that during the
    first build, causing barrier #2 to time out on the waiting rank. Allow
    the timeout to be raised via ``WALLX_DIST_TIMEOUT_MINUTES`` (default 30).
    """
    if not dist.is_initialized():
        timeout_min = int(os.environ.get("WALLX_DIST_TIMEOUT_MINUTES", "30"))
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=timeout_min),
        )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    return local_rank


def cleanup_distributed():
    """Clean up distributed training environment"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    """Get current process rank"""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes"""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_local_rank() -> int:
    """Get local rank (GPU index on current node)"""
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0)"""
    return get_rank() == 0


# Trainer-level collectives should use an explicit global process group.
# FSDP's mesh groups may be 1D or 2D and have sharding-specific semantics,
# while trainer barriers/metric reductions need all-rank semantics.
_trainer_process_group = None


def set_trainer_process_group(pg):
    global _trainer_process_group
    _trainer_process_group = pg


def barrier():
    """Synchronize all processes"""
    if dist.is_initialized():
        dist.barrier(group=_trainer_process_group)


def all_reduce(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """All-reduce tensor across all processes"""
    if dist.is_initialized():
        dist.all_reduce(tensor, op=op, group=_trainer_process_group)
    return tensor


def all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather tensor from all processes"""
    if not dist.is_initialized():
        return tensor

    group = _trainer_process_group
    world_size = dist.get_world_size(group=group)
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor, group=group)
    return torch.stack(gathered)


class BaseDistributedTrainer(ABC):
    """
    Abstract base class for distributed training without accelerate.
    Supports both DDP and FSDP through subclasses.

    Launch with torchrun:
        torchrun --nproc_per_node=8 train.py --config config.yaml
    """

    @abstractmethod
    def train_loop(self, epoch: int):
        """Training loop for one epoch"""
        raise NotImplementedError("train_loop must be implemented")

    @abstractmethod
    def val_loop(self):
        """Validation loop"""
        raise NotImplementedError("val_loop must be implemented")

    @abstractmethod
    def save_checkpoint(self, epoch: int, step: int = 0):
        """Save model checkpoint"""
        raise NotImplementedError("save_checkpoint must be implemented")

    @abstractmethod
    def load_model(self):
        """Load model"""
        raise NotImplementedError("load_model must be implemented")

    @abstractmethod
    def load_dataset(self):
        """Load dataset"""
        raise NotImplementedError("load_dataset must be implemented")

    @abstractmethod
    def backward(self, loss: torch.Tensor):
        """Perform backward pass"""
        raise NotImplementedError("backward must be implemented")

    @abstractmethod
    def clip_grad_norm(self, max_norm: float) -> torch.Tensor:
        """Clip gradient norm"""
        raise NotImplementedError("clip_grad_norm must be implemented")


class DistributedTrainer(BaseDistributedTrainer):
    """
    Concrete base trainer with common functionality for DDP/FSDP.
    Subclasses should implement wrap_model, backward, clip_grad_norm, etc.
    """

    def __init__(
        self,
        train_config: TrainConfig,
        wandb_run: Optional[Any] = None,
    ):
        self.cfg = train_config

        # wandb Run object (metric recorder) - None on non-main ranks and
        # when use_wandb is disabled. Distinct from self.logger below.
        self.wandb_run = wandb_run

        # Initialize distributed environment
        self.local_rank = setup_distributed()
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.device = torch.device(f"cuda:{self.local_rank}")

        # Dataset config: optional backend-provided trainer config.
        from wall_x.trainer.adapters.base_adapter import load_trainer_data_config

        self.data_config = load_trainer_data_config(self.cfg)

        # Text logger
        self.logger = DistributedLogger(
            name=self.__class__.__name__,
            save_path=self.cfg.checkpoint.save_path,
        )

        # Training state
        self.seed = self.cfg.hyperparams.seed
        self.logger.info(f"seed {self.seed}")

        seed_all(self.seed)
        self.start_epoch = 0
        self.global_step = 0
        self.micro_step = 0
        self.num_epoch = self.cfg.hyperparams.num_epoch
        self.dataset_config_path = getattr(self.cfg.data, "dataset_config_path", None)
        self.initial_step = 0

        # Optimizer hyperparameters (betas / weight_decay / eps)
        # used by build_optimizer for both AdamW and Muon paths.
        opt = self.cfg.hyperparams.optimizer
        if isinstance(opt, AdamWConfig):
            self.adamw_betas = tuple(opt.betas)
            self.adamw_weight_decay = opt.weight_decay
            self.adamw_eps = opt.eps
        elif getattr(opt, "optimizer_type", None) == "muon":
            self.adamw_betas = (opt.beta_1, opt.beta_2)
            self.adamw_weight_decay = opt.weight_decay
            self.adamw_eps = opt.eps
        else:
            self.adamw_betas = (0.9, 0.98)
            self.adamw_weight_decay = 1e-8
            self.adamw_eps = 1e-8

        # Performance options
        self.nvtx = self.cfg.debug.nvtx
        self.timers = Timers(log_level=0, log_option="minmax")

        # Training hyperparameters
        self.max_grad_norm = opt.max_grad_norm
        self.grad_accum_steps = self.cfg.hyperparams.gradient_accumulation_steps
        self.log_interval = self.cfg.logging.log_interval
        self.log_stats_buffer = []

        # Model and optimizer (to be set by subclass)
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None

        # Mixed precision
        dist_cfg = self.cfg.distributed
        self.use_amp = dist_cfg.use_amp
        self.amp_dtype = torch.bfloat16 if dist_cfg.bf16 else torch.float16
        self.grad_scaler = None
        if self.use_amp and self.amp_dtype == torch.float16:
            self.grad_scaler = torch.amp.GradScaler("cuda")

    def log(
        self, message: str, level: int = logging.INFO, main_process_only: bool = True
    ):
        """Log a text message via self.logger.

        Kept for backward compat with call sites that still use ``self.log(...)``
        (new code should call ``self.logger.info/warning/error(...)`` directly).
        Under the new DistributedLogger the ``main_process_only`` flag is
        redundant for stdout (stdout is rank-0-only by construction); it still
        controls whether non-zero ranks write the message to their file log.
        """
        if main_process_only and not is_main_process():
            return
        self.logger.log(message, level=level, main_process_only=main_process_only)

    def fit(self):
        """Main training loop"""
        barrier()

        # Optional: validate before training
        if self.cfg.checkpoint.validate_first:
            self.val_loop()
            barrier()

        if self.nvtx:
            torch.cuda.cudart().cudaProfilerStart()

        # Profiler setup
        enable_profiling = self.cfg.debug.profile
        profiler = (
            self._setup_profiler() if enable_profiling else contextlib.nullcontext()
        )

        for epoch in range(self.start_epoch, self.num_epoch):
            # Training
            self.train_loop(epoch, profiler=profiler)
            barrier()
            num_training_steps = getattr(self, "num_training_steps", 0)
            if num_training_steps > 0 and self.global_step >= num_training_steps:
                break

            # Save checkpoint
            if (epoch + 1) % self.cfg.logging.epoch_save_interval == 0:
                self.save_checkpoint(epoch)

            # Validation
            self.val_loop()
            barrier()

            gc.collect()
            torch.cuda.empty_cache()

        if self.nvtx:
            torch.cuda.cudart().cudaProfilerStop()

    def _setup_profiler(self):
        """Setup PyTorch profiler"""
        return torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=self.cfg.debug.profile_wait_iters,
                warmup=self.cfg.debug.profile_warmup_iters,
                active=self.cfg.debug.profile_active_iters,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                self.cfg.debug.profile_save_path,
                worker_name=f"worker{self.rank}",
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )

    def gather_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather tensor from all processes"""
        return all_gather(tensor)

    def reduce_tensor(self, tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
        """Reduce tensor across all processes"""
        tensor = tensor.clone()
        all_reduce(tensor)
        if average:
            tensor = tensor / self.world_size
        return tensor

    def sync_gradients(self) -> bool:
        """Check if gradients should be synchronized (for gradient accumulation)"""
        return (self.micro_step + 1) % self.grad_accum_steps == 0

    def optimizer_zero_grad(self):
        """Zero optimizer gradients"""
        self.optimizer.zero_grad(set_to_none=True)

    def optimizer_step(self):
        """Perform optimizer step"""
        if self.grad_scaler is not None:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()

    def lr_scheduler_step(self):
        """Perform learning rate scheduler step"""
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def get_lr(self) -> float:
        """Get current learning rate"""
        if self.lr_scheduler is not None:
            return self.lr_scheduler.get_last_lr()[0]
        return self.optimizer.param_groups[0]["lr"]

    def autocast_context(self):
        """Get autocast context for mixed precision"""
        if self.use_amp:
            return torch.amp.autocast("cuda", dtype=self.amp_dtype)
        return contextlib.nullcontext()

    def training_log(
        self,
        current_epoch: int,
        total_epoch: int,
        current_iter: int,
        total_iter: int,
        loss: torch.Tensor,
        lr: float,
        time_per_step: float,
        show_time_details: bool = False,
    ):
        """Log training progress"""
        if not is_main_process():
            return

        # Prefer the cross-rank reduced loss (train_loss) from
        # _current_step_stats so the console line matches what wall_x_2509's
        # fsdp_trainer prints and what wandb sees. Falls back to the raw
        # rank-0 local tensor when stats haven't been populated yet.
        _smoothed = getattr(self, "_current_step_stats", None) or {}
        _loss_to_print = _smoothed.get("train_loss", float(loss))

        log_string = ""
        log_string += f" epoch {current_epoch:3d}/{total_epoch:3d} |"
        log_string += f" iter {current_iter:6d}/{total_iter:6d} |"
        log_string += f" loss {_loss_to_print:.10f} |"
        log_string += f" lr {lr:.6f} |"
        log_string += f" time {time_per_step:.4f}s |"

        self.log(log_string)

        if show_time_details:
            timers_to_log = [
                "interval-time",
                "data-load",
                "forward-compute",
                "backward-compute",
                "optimizer",
            ]
            self.timers.log(timers_to_log, normalizer=1)

    def true_gather(self, value):
        """Gather values across all processes and compute mean over non-None values."""
        device = next(self.model.parameters()).device

        # CRITICAL: must clone() here, not just detach().
        # detach() only disconnects the computation graph but still shares the
        # same underlying storage.  dist.all_reduce() is **in-place** - it
        # overwrites that storage with the sum across all ranks.  If the caller
        # passes a tensor that is also referenced elsewhere (e.g. outputs["loss"]
        # and outputs["flow_loss"] pointing to the same scalar_loss), the
        # in-place all_reduce silently mutates the original, causing downstream
        # readers (like training_log) to see a value multiplied by world_size.
        value_to_gather = (
            value.detach().clone()
            if value is not None
            else torch.tensor(0.0, device=device)
        )
        count_to_gather = torch.tensor(1.0 if value is not None else 0.0, device=device)

        # all_reduce is cheaper than all_gather because only the sum is needed.
        dist.all_reduce(value_to_gather, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_to_gather, op=dist.ReduceOp.SUM)

        total_processes = count_to_gather.item()
        if total_processes > 0:
            return value_to_gather.item() / total_processes
        else:
            return None
