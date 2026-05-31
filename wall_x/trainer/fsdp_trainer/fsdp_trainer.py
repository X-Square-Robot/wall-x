"""FSDP trainer implementation."""

import dataclasses
import gc
import os
import torch
import time
import logging
import contextlib
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, Any, Tuple
from wall_x.trainer.optimizer import get_optimizer

from wall_x.trainer.fsdp_trainer.base_trainer import (
    DistributedTrainer,
    is_main_process,
    barrier,
)
from wall_x.trainer.adapters import resolve_adapter
from wall_x.trainer.fsdp_trainer import checkpoint_io as _ckpt_io
from wall_x.trainer.fsdp_trainer.distribution_strategy import build_strategy
from wall_x.trainer.fsdp_trainer.metrics_logger import MetricsLogger
from wall_x.trainer.utils import move_batch_to_device

from wall_x.model.core.action.normalizer import Normalizer, create_normalizers
from wall_x.trainer.scheduler.scheduler import get_scheduler


class FSDPTrainer(DistributedTrainer):
    """
    FSDP-based Trainer using pure PyTorch FSDP2 (``fully_shard``).

    Features:
    - No accelerate dependency
    - Direct torchrun launch
    - Mixed precision training support (bfloat16/float16)
    - Gradient accumulation
    - Full/Sharded state dict saving
    - CPU offload support
    - Activation checkpointing

    Launch:
        torchrun --nproc_per_node=8 --master_port=29500 train_fsdp.py --config config.yaml
    """

    def __init__(
        self,
        train_config,
        wandb_run: Optional[Any] = None,
    ):
        if train_config is None:
            raise ValueError("train_config is required")

        self.model_type = train_config.model_type
        super().__init__(train_config, wandb_run)

        adapter_cls = resolve_adapter(self.model_type)
        self.adapter = adapter_cls(
            cfg=self.cfg,
            logger=self.logger,
            model_type=self.model_type,
        )

        self.strategy = build_strategy(
            dataclasses.asdict(self.cfg.distributed),
            device=self.device,
            local_rank=self.local_rank,
            use_dmuon=self.cfg.hyperparams.optimizer.use_dedicated_params,
        )

        # Model config
        self.action_dim = self.cfg.action_dim
        self.use_selective_recompute = self.cfg.distributed.use_selective_recompute
        self.show_time_details = self.cfg.debug.show_time_details

        # num_training_steps: read from any scheduler that exposes it
        # (CosineSchedulerConfig currently; future schedulers may add it).
        # Used in train_loop to trigger loss_guard_should_stop — works with
        # constant scheduler too once the field is set.
        sched = self.cfg.hyperparams.scheduler
        self.num_training_steps = int(getattr(sched, "num_training_steps", 0) or 0)
        self.metrics_logger = MetricsLogger(
            wandb_run=self.wandb_run,
            log_interval=self.log_interval,
            smooth_window=self.cfg.logging.loss_log_smooth_window,
        )

        # Loss tracking
        self.base_l1_loss = None
        self.base_l1_loss_detail = {}

        self.load_normalizer()
        self.load_processor()
        self.load_model()

        # DDP needs to see correct requires_grad at wrap time, so freeze first.
        self._freeze_params_if_needed(self.model)

        # Frozen-submodule prefixes for checkpoint filtering. Must be computed
        # before FSDP wrapping, while child modules still have stable names.
        self._frozen_prefixes = self._compute_frozen_prefixes()

        # Original shapes captured before FSDP flattening.
        self.model._orig_param_shapes = {
            name: p.shape for name, p in self.model.named_parameters()
        }

        if self._resume_from_single_file():
            self.load_state_dict(
                self.model,
                {"ckpt": self.cfg.checkpoint.resume_from},
            )

        self._wrap_model(self.model)
        self._create_optimizer()
        self._create_scheduler()

        if self._resume_from_training_checkpoint():
            self.resume_from_checkpoint()

        self.load_dataset()

        self.adapter.init_validation(self.normalizer_action, self.normalizer_propri)

    def load_normalizer(self):
        norm_cfg = dict(self.cfg.data.normalizer_config or {})
        raw_data = getattr(self.cfg, "_raw_data", {}) or {}

        custom_path = norm_cfg.get("customized_action_statistic_dof") or raw_data.get(
            "customized_action_statistic_dof"
        )
        merged = {
            "dof_config": self.cfg.task.dof_config,
            "agent_pos_config": self.cfg.task.agent_pos_config,
            "customized_action_statistic_dof": custom_path,
            "min_key": norm_cfg.get("min_key", "min"),
            "delta_key": norm_cfg.get("delta_key", "delta"),
        }
        if custom_path:
            (
                self.normalizer_action,
                self.normalizer_propri,
                self._action_statistic_dof,
            ) = create_normalizers(merged)
            return

        # LeRobot finetune: reuse collator stats (q01/q99 JSON) for the model
        # normalizer. ``dataset_names`` in the batch is ``lerobot_config.repo_id``.
        norm_stats_path = getattr(
            self.cfg.data, "norm_stats_path", None
        ) or raw_data.get("norm_stats_path")
        lerobot_cfg = getattr(self.cfg.data, "lerobot_config", None) or raw_data.get(
            "lerobot_config", {}
        )
        key_mappings = getattr(self.cfg.data, "key_mappings", None) or raw_data.get(
            "key_mappings"
        )
        repo_id = (lerobot_cfg or {}).get("repo_id")
        if norm_stats_path and key_mappings and repo_id:
            from wall_x.data.backends.lerobot.utils import load_norm_stats

            stats = load_norm_stats(
                norm_stats_path,
                key_mappings,
                dof_config=self.cfg.task.dof_config,
                agent_pos_config=self.cfg.task.agent_pos_config,
            )
            dataset_name = str(repo_id)
            self.normalizer_action = Normalizer.from_lerobot_norm_stats(
                stats["action"], dataset_name
            )
            self.normalizer_propri = Normalizer.from_lerobot_norm_stats(
                stats["state"], dataset_name
            )
            self._action_statistic_dof = {}
            if is_main_process():
                self.logger.info(
                    "Built model normalizers from norm_stats_path=%s "
                    "(dataset key=%r)",
                    norm_stats_path,
                    dataset_name,
                )
            return

        # Pretrain / resume: sibling normalizer checkpoints next to resume_from.
        resume_from = getattr(self.cfg.checkpoint, "resume_from", None)
        if resume_from:
            ckpt_dir = os.path.dirname(resume_from)
            action_pth = os.path.join(ckpt_dir, "normalizer_action.pth")
            propri_pth = os.path.join(ckpt_dir, "normalizer_propri.pth")
            if os.path.isfile(action_pth) and os.path.isfile(propri_pth):
                self.normalizer_action = Normalizer.from_ckpt(action_pth)
                self.normalizer_propri = Normalizer.from_ckpt(propri_pth)
                self._action_statistic_dof = {}
                if is_main_process():
                    self.logger.info(
                        "Loaded model normalizers from checkpoint dir %s",
                        ckpt_dir,
                    )
                return

        self.normalizer_action, self.normalizer_propri, self._action_statistic_dof = (
            create_normalizers(merged)
        )

    def backward(self, loss: torch.Tensor):
        """Perform backward pass."""
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()

    def clip_grad_norm(self, max_norm: float) -> torch.Tensor:
        """Unscale and clip gradient norm via the active distribution strategy."""
        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(self.optimizer)

        total_norm = self.strategy.clip_grad_norm(
            self.model,
            max_norm,
            optimizer=self.optimizer,
        )
        self._dedicated_param_grad_clip_stats = getattr(
            self.strategy,
            "last_grad_clip_stats",
            None,
        )
        return total_norm

    def load_model(self):
        """Load and prepare model, optimizer, and scheduler"""
        self.model_config = self.adapter.build_model_config()
        self.model = self.adapter.create_model(
            self.processor, self.tokenizer_mixin, self.model_config
        )
        type(self.adapter).log_attention_implementation(self.logger, self.model)
        self.adapter.load_weights(
            self.model,
            self.normalizer_action,
            self.normalizer_propri,
            processor=self.processor,
        )

    def load_processor(self):
        """Load processor and tokenizers"""
        self.adapter.normalizer_action = self.normalizer_action
        self.adapter.normalizer_propri = self.normalizer_propri
        processors_dict = self.adapter.load_processor(self._action_statistic_dof)
        self.processor = processors_dict["processor"]
        self.train_action_tokenizer = processors_dict["train_action_tokenizer"]
        self.val_action_tokenizer = processors_dict["val_action_tokenizer"]
        self.action_mapper = processors_dict["action_mapper"]
        self.tokenizer_mixin = processors_dict.get("tokenizer_mixin")

    def _freeze_params_if_needed(self, model: torch.nn.Module):
        """Freeze non-action parameters when train_action_expert_only is set.

        Must be called BEFORE wrapping with DDP/FSDP so the wrapper sees the
        correct requires_grad flags and does not expect gradients for frozen params.
        """
        from wall_x.trainer.optimizer.utils import resolve_lr_group_configs

        opt = self.cfg.hyperparams.optimizer
        if not opt.train_action_expert_only:
            return
        lr_groups = resolve_lr_group_configs(
            opt, self.adapter.default_action_lr_keywords
        )
        if not lr_groups:
            self.log(
                "WARNING: train_action_expert_only is True but no optimizer LR "
                "group is set. No parameters will be frozen.",
                level=logging.WARNING,
            )
            return

        frozen_count = 0
        grouped_count = 0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            matches_group = any(
                any(keyword in name for keyword in group.include) for group in lr_groups
            )
            if matches_group:
                grouped_count += 1
            else:
                param.requires_grad = False
                frozen_count += 1

        if grouped_count == 0:
            raise ValueError(
                "No grouped params found for train_action_expert_only. "
                "Please check optimizer.lr_groups or legacy action_lr_keywords."
            )

        self.log(
            f"*** train_action_expert_only: frozen {frozen_count} base params, "
            f"keeping {grouped_count} grouped params trainable ***"
        )

    def _wrap_model(self, model: torch.nn.Module):
        """Prepare model dtype/placement via adapter, then wrap via strategy.

        If the strategy created an explicit trainer process group, register
        it so trainer-level barriers and metric collectives keep all-rank
        semantics independent of the FSDP mesh topology.
        """
        self.adapter.convert_to_mix_precision_hint(
            model,
            device=self.device,
            use_fsdp=self.cfg.distributed.use_fsdp,
            log_fn=self.log,
        )
        self.model = self.strategy.wrap(model)

        trainer_pg = getattr(self.strategy, "trainer_process_group", None)
        if trainer_pg is not None:
            from wall_x.trainer.fsdp_trainer.base_trainer import (
                set_trainer_process_group,
            )

            set_trainer_process_group(trainer_pg)
            self.log("[FSDP2] routing trainer collectives through trainer PG")

    def _create_optimizer(self):
        """Create optimizer for FSDP wrapped model."""
        from wall_x.trainer.optimizer.utils import (
            build_lr_param_groups,
            resolve_lr_group_configs,
            uses_legacy_action_lr_groups,
        )

        opt_cfg = self.cfg.hyperparams.optimizer

        param_groups = None
        lr_group_configs = resolve_lr_group_configs(
            opt_cfg, self.adapter.default_action_lr_keywords
        )
        if lr_group_configs:
            if opt_cfg.optimizer_type not in ("adamw", "muon", "dmuon"):
                raise ValueError(
                    "optimizer.lr_groups are only supported with adamw, muon, "
                    "or dmuon"
                )
            base_group_name = (
                "base_lr_group" if uses_legacy_action_lr_groups(opt_cfg) else "base"
            )
            param_groups = build_lr_param_groups(
                self.model,
                opt_cfg,
                lr_group_configs,
                base_group_name=base_group_name,
            )
            summary = ", ".join(
                f"{group.name}={group.lr} include={group.include}"
                for group in lr_group_configs
            )
            self.log(
                f"setting optimizer LR groups: base={opt_cfg.learning_rate}; "
                f"{summary} ({opt_cfg.optimizer_type})"
            )

        optimizer_kwargs = {
            "opt_cfg": opt_cfg,
            "param_groups": param_groups,
        }
        if opt_cfg.optimizer_type == "dmuon":
            optimizer_kwargs["log_fn"] = self.log

        self.optimizer = get_optimizer(
            opt_cfg.optimizer_type,
            self.model,
            **optimizer_kwargs,
        )

    def _create_scheduler(self):
        """Create learning rate scheduler"""
        from wall_x.config.hyperparams_config import (
            ConstantSchedulerConfig,
            CosineSchedulerConfig,
        )

        sched = self.cfg.hyperparams.scheduler
        lr = self.cfg.hyperparams.optimizer.learning_rate

        if isinstance(sched, ConstantSchedulerConfig):
            self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer, factor=1.0, total_iters=0
            )
        elif isinstance(sched, CosineSchedulerConfig):
            if sched.num_training_steps <= 0:
                raise ValueError(
                    "num_training_steps must be > 0 for cosine scheduler. "
                    "Please set it explicitly in the config."
                )
            min_lr = sched.min_lr if sched.min_lr is not None else 0.1 * lr
            self.lr_scheduler = get_scheduler(
                optimizer=self.optimizer,
                lr_scheduler_type="cosine",
                num_warmup_steps=sched.num_warmup_steps,
                num_training_steps=sched.num_training_steps,
                peak_lr=lr,
                end_lr=min_lr,
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {type(sched).__name__}")

    def load_dataset(self):
        """Load training and validation datasets"""
        torch.cuda.empty_cache()
        barrier()
        # Dispatch to adapter with unified signature (supports model-specific kwargs)
        extra_dataset_kwargs = dict(
            model_config=self.model_config,
            tokenizer_mixin=self.tokenizer_mixin,
            normalizer_action=self.normalizer_action,
            normalizer_propri=self.normalizer_propri,
        )

        resume_path = self.cfg.checkpoint.resume_from
        if resume_path is not None and os.path.isdir(resume_path):
            # Step and epoch restore are generic. Backend-owned episode
            # metadata is optional and only loaded when the adapter exposes
            # the hook.
            is_incomplete_epoch = False
            if hasattr(self.adapter, "_load_episode_indices"):
                resume_indices = self.adapter._load_episode_indices(
                    resume_path, self.rank
                )
                is_incomplete_epoch = resume_indices["is_incomplete_epoch"]
                extra_dataset_kwargs["resume_indices"] = resume_indices

            train_state = self.adapter.load_step_and_epoch(
                resume_path, is_incomplete_epoch
            )
            self.global_step = train_state["global_step"]
            self.start_epoch = train_state["start_epoch"]
            self.log(
                f"global_step: {self.global_step}, start_epoch: "
                f"{self.start_epoch} from {resume_path}"
            )
            # global_step records completed steps, so resume from the next step
            # to avoid triggering _should_save_checkpoint() immediately.
            self.global_step += 1
        (
            self.dataset,
            self.train_dataloader,
            self.train_num,
        ) = self.adapter.load_dataset(
            self.data_config,
            self.processor,
            self.rank,
            self.world_size,
            **extra_dataset_kwargs,
        )

    def train_loop(self, epoch: int, profiler=contextlib.nullcontext()):
        """Execute training for a single epoch"""
        self.model.train()

        # VGDynamicRobotDataset (v1 path) doesn't expose set_epoch; other
        # dataset backends (v2, lerobot) do. Dispatch conditionally.
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)
        total = len(self.train_dataloader)
        # Drop last
        stop_step = total - total % self.grad_accum_steps

        # Disable automatic GC to prevent random ~200ms stalls during compute
        gc_interval = self.cfg.logging.gc_interval_steps
        gc.disable()

        t0 = time.time()
        with profiler:
            self.timers("interval-time", log_level=0).start(barrier=True)
            self.timers("data-load", log_level=0).start(barrier=True)

            for i, batch in enumerate(self.train_dataloader, self.initial_step):
                self.timers("data-load").stop()

                # Save first batch for offline profiling (rank 0 only)
                debug_batch_path = self.cfg.debug.save_debug_batch_path
                if debug_batch_path and i == self.initial_step and self.rank == 0:
                    os.makedirs(os.path.dirname(debug_batch_path) or ".", exist_ok=True)
                    torch.save({"batch": batch}, debug_batch_path)
                    self.log(f"Saved debug batch to {debug_batch_path}")

                # Move batch to device
                batch = move_batch_to_device(batch, self.device)

                if self.cfg.debug.enable_mfu_profile and self.global_step == 0:
                    self.log(
                        "[MFU] Forward FLOPs profiling is not available in the public package.",
                        level=logging.INFO,
                    )

                # Forward pass with autocast
                self.timers("forward-compute", log_level=0).start(barrier=False)
                with self.autocast_context():
                    outputs = self.adapter.forward(
                        self.model, batch, global_step=self.global_step
                    )
                self.timers("forward-compute").stop()

                loss = self.adapter.extract_loss(outputs)

                # Check for NaN loss -- replace with 0 instead of `continue`
                # to avoid skipping collective ops (backward, all_reduce, etc.)
                # which would cause NCCL deadlock across ranks.
                nan_loss = torch.isnan(loss)
                if nan_loss:
                    self.log(
                        f"Warning: nan in loss at epoch: {epoch}, step: {i}",
                        level=logging.WARNING,
                    )
                    loss = torch.zeros_like(loss)

                # Backward pass
                self.timers("backward-compute", log_level=0).start(barrier=True)

                context = (
                    contextlib.nullcontext()
                    if self.sync_gradients()
                    else self.strategy.no_sync(self.model)
                )
                with context:
                    scaled_loss = loss / self.grad_accum_steps
                    self.backward(scaled_loss)
                self.timers("backward-compute").stop()

                # Gradient sync and optimizer step
                if self.sync_gradients():
                    self.timers("optimizer", log_level=0).start(barrier=True)

                    # Per-component grad norms (before clipping)
                    self.timers("optimizer-grad-norms", log_level=0).start(
                        barrier=False
                    )
                    self._component_grad_norms = {}
                    self.adapter.collect_grad_norms(
                        self.model,
                        self._component_grad_norms,
                        device=self.device,
                        reduce_tensor_fn=self.reduce_tensor,
                        params_sharded=self.strategy.params_sharded,
                    )
                    self.timers("optimizer-grad-norms").stop()

                    # Clip gradients
                    self.timers("optimizer-clip", log_level=0).start(barrier=False)
                    if self.cfg.hyperparams.optimizer.enable_grad_clip:
                        total_norm = self.clip_grad_norm(self.max_grad_norm)
                    else:
                        self._dedicated_param_grad_clip_stats = None
                        total_norm = 0
                    self.timers("optimizer-clip").stop()

                    self.timers("optimizer-step", log_level=0).start(barrier=False)
                    self.optimizer_step()
                    self.timers("optimizer-step").stop()

                    self.timers("optimizer-zero-grad", log_level=0).start(barrier=False)
                    self.optimizer_zero_grad()
                    self.timers("optimizer-zero-grad").stop()
                    self.timers("optimizer").stop()

                    # Scheduler step
                    self.lr_scheduler_step()

                    # Logging
                    self.timers("logging", log_level=0).start(barrier=True)
                    _t_metrics_start = time.time()
                    self._log_training_metrics(
                        epoch, i, total, loss, total_norm, outputs
                    )
                    _t_metrics_ms = (time.time() - _t_metrics_start) * 1000

                    t1 = time.time()
                    self.training_log(
                        epoch,
                        self.num_epoch,
                        i,
                        total,
                        loss,
                        self.get_lr(),
                        t1 - t0,
                        self.show_time_details,
                    )
                    self.timers("logging").stop()
                    t0 = time.time()

                    # Optional: log per-step breakdown to pinpoint spikes (e.g. param_norms every 100 steps)
                    if (
                        self.show_time_details
                        and _t_metrics_ms > 5000
                        and is_main_process()
                    ):
                        self.log(
                            f"[Step time breakdown] _log_training_metrics took {_t_metrics_ms:.0f} ms at global_step={self.global_step}",
                            level=logging.INFO,
                        )

                    # Checkpoint saving (FSDP full state_dict can take 10-20s every save_interval steps)
                    if self._should_save_checkpoint():
                        _t_save_start = time.time()
                        self.save_checkpoint(epoch, self.global_step)
                        if is_main_process():
                            self.log(
                                f"[Step time breakdown] save_checkpoint took {(time.time() - _t_save_start):.1f} s at global_step={self.global_step}",
                                level=logging.INFO,
                            )

                    # Validation
                    if self._should_validate():
                        self.val_loop()

                    self.global_step += 1
                    self.micro_step = 0

                    # Manual GC outside timing window to avoid random stalls (can add 1-5s every gc_interval_steps)
                    if self.global_step % gc_interval == 0:
                        _t_gc_start = time.time()
                        gc.collect()
                        if is_main_process() and self.show_time_details:
                            self.log(
                                f"[Step time breakdown] gc.collect took {(time.time() - _t_gc_start):.1f} s at global_step={self.global_step}",
                                level=logging.INFO,
                            )
                else:
                    self.micro_step += 1

                del batch
                self.timers("interval-time").stop()

                # Drop last
                if i == stop_step:
                    break

                if not isinstance(profiler, contextlib.nullcontext):
                    profiler.step()

                if (
                    self.num_training_steps > 0
                    and self.global_step >= self.num_training_steps
                ):
                    break

                # Setup timers for next iteration
                if i < total - 1:
                    self.timers("interval-time", log_level=0).start(barrier=True)
                    self.timers("data-load", log_level=0).start(barrier=True)

        # Re-enable automatic GC after training loop
        gc.enable()

        # Reset dataloader for next epoch
        self.train_dataloader = self.dataset.get_train_dataloader()

    def _log_training_metrics(self, epoch, step, total, loss, total_norm, outputs):
        """Collect per-step stats, delegate buffering + wandb emission."""
        lr = self.get_lr()
        train_loss = self.reduce_tensor(loss.detach()).item()

        step_stats = {
            "lr": lr,
            "train_loss": train_loss,
            "grad_norm": (
                total_norm.item() if torch.is_tensor(total_norm) else float(total_norm)
            ),
        }
        for idx, group in enumerate(self.optimizer.param_groups):
            group_name = group.get("group_name", f"group_{idx}")
            step_stats[f"lr_group/{group_name}"] = float(group["lr"])

        # Model-family-specific auxiliary losses / accuracies.
        self.adapter.collect_output_stats(
            outputs,
            step_stats,
            reduce_tensor_fn=self.reduce_tensor,
            true_gather_fn=self.true_gather,
            tokenizer_mixin=self.tokenizer_mixin,
        )

        # Per-component grad norms (captured before clip & optimizer.zero_grad).
        if hasattr(self, "_component_grad_norms"):
            step_stats.update(self._component_grad_norms)

        dedicated_clip_stats = getattr(self, "_dedicated_param_grad_clip_stats", None)
        if dedicated_clip_stats is not None:
            step_stats.update(
                {
                    "muon_grad_norm": dedicated_clip_stats["total_norm"],
                    "muon_grad_clip_coef": dedicated_clip_stats["clip_coef"],
                    "muon_grad_clipped": float(dedicated_clip_stats["clipped"]),
                }
            )

        if self.global_step % 100 == 0:
            self.adapter.collect_param_norms(
                self.model,
                step_stats,
                device=self.device,
                reduce_tensor_fn=self.reduce_tensor,
                params_sharded=self.strategy.params_sharded,
            )

        self._current_step_raw_stats = step_stats

        # Display-smoothing rolling window (DZ-style)
        self._current_step_stats = self.metrics_logger.smooth(step_stats)

    def training_log(
        self,
        current_epoch,
        total_epoch,
        current_train_iter,
        total_train_iter,
        loss,
        lr,
        time_per_step,
        show_time_details=False,
    ):
        # timers.log() contains all_gather — must run on ALL ranks before the
        # is_main_process() guard to avoid NCCL deadlock.
        if show_time_details:
            self.timers.log(
                [
                    "interval-time",
                    "data-load",
                    "forward-compute",
                    "backward-compute",
                    "optimizer",
                    "optimizer-grad-norms",
                    "optimizer-clip",
                    "optimizer-step",
                    "optimizer-zero-grad",
                    "logging",
                ],
                normalizer=1,
            )

        main = is_main_process()
        stats = getattr(self, "_current_step_stats", None) or {}

        # MFU is computed after step time is known, then merged into both the
        # console stats and the MetricsLogger buffer so wandb records it.
        mfu_info = None
        if self.cfg.debug.enable_mfu and hasattr(self.adapter, "compute_mfu"):
            unwrapped = (
                self.model.module if hasattr(self.model, "module") else self.model
            )
            mfu_info = self.adapter.compute_mfu(unwrapped, time_per_step)
            if mfu_info is not None:
                self.log(
                    "[MFU-rank] rank={} mfu={:.2f}% step={:.3f}s "
                    "flops_step={:.3f}T flops_fwd={:.3f}T "
                    "seq={} latent={} source={}".format(
                        getattr(self, "rank", 0),
                        mfu_info["mfu"] * 100.0,
                        time_per_step,
                        mfu_info["flops_per_step_T"],
                        mfu_info.get("flops_forward_T", 0.0) or 0.0,
                        mfu_info.get("seq_dims"),
                        mfu_info.get("latent_dims"),
                        mfu_info.get("latent_source"),
                    ),
                    main_process_only=False,
                )
            if main and mfu_info is not None and stats is not None:
                mfu_stats = {
                    "mfu": mfu_info["mfu"],
                    "flops_per_step_T": mfu_info["flops_per_step_T"],
                    "flops_forward_T": mfu_info.get("flops_forward_T"),
                }
                profile_fwd_flops = mfu_info.get("profile_fwd_flops_T")
                if profile_fwd_flops is not None:
                    mfu_stats["profile_fwd_flops_T"] = profile_fwd_flops
                stats.update(mfu_stats)
                raw_stats = getattr(self, "_current_step_raw_stats", None)
                if raw_stats is not None:
                    raw_stats.update(mfu_stats)

        if main:
            raw_stats = getattr(self, "_current_step_raw_stats", None)
            if raw_stats is not None:
                self.metrics_logger.record_step(raw_stats, is_main=True)
                avg_stats = self.metrics_logger.flush_if_due(self.global_step)
                if avg_stats is not None:
                    self.logger.info(
                        f"[FSDP Train] Step {self.global_step}: {avg_stats}"
                    )

        if not main:
            return

        loss_to_print = stats.get(
            "train_loss", loss.item() if torch.is_tensor(loss) else float(loss)
        )
        fields = self.adapter.console_fields(tokenizer_mixin=self.tokenizer_mixin)

        line = self.metrics_logger.format_training_line(
            epoch=current_epoch,
            total_epoch=total_epoch,
            current_iter=current_train_iter,
            total_iter=total_train_iter,
            loss=loss_to_print,
            lr=lr,
            time_per_step=time_per_step,
            stats=stats,
            fields=fields,
            mfu_info=mfu_info,
        )
        self.log(line)

    def _should_save_checkpoint(self) -> bool:
        """Check if checkpoint should be saved"""
        log = self.cfg.logging
        return (
            self.global_step >= log.ignore_until_interval
            and self.global_step % log.save_interval == 0
            and self.global_step != 0
        )

    def _should_validate(self) -> bool:
        """Check if validation should be performed"""
        log = self.cfg.logging
        return (
            self.global_step >= log.ignore_until_interval
            and self.global_step % log.val_interval == 0
            and self.global_step > 0
        )

    @torch.no_grad()
    def val_loop(self):
        """Delegate the full validation run to the adapter.

        Skips silently when the dataset has no val split — adapter's
        run_validation iterates ``val_dataloader`` with ``tqdm(...,
        total=len(...))`` and cannot accept None. v2 returns None here
        when the YAML only declares a train split.
        """
        self.val_dataloader = self.dataset.get_val_dataloader()
        if self.val_dataloader is None:
            if is_main_process():
                self.logger.info("No val split configured, skipping validation.")
            return
        save_path = self.cfg.checkpoint.save_path
        self.adapter.run_validation(
            model=self.model,
            val_dataloader=self.val_dataloader,
            rank=self.rank,
            world_size=self.world_size,
            device=self.device,
            autocast_context=self.autocast_context,
            reduce_fn=self.reduce_tensor,
            gather_fn=self.true_gather,
            logger=self.wandb_run if is_main_process() else None,
            global_step=self.global_step,
            output_path=os.path.join(save_path, f"val_rank_{self.rank}"),
            tokenizer_mixin=self.tokenizer_mixin,
        )
        barrier()

    def _compute_frozen_prefixes(self) -> Optional[Tuple[str, ...]]:
        """Return state-dict key prefixes for submodules that are fully frozen.

        For models with a ``pipe`` container, any child not present in
        ``cfg.model.trainable_models`` is treated as frozen. The matching key
        prefixes let checkpoint_io drop those entries from the saved file.
        Empty / missing ``trainable_models`` disables filtering and preserves
        "save everything" behavior.
        """
        if not hasattr(self.model, "pipe"):
            return None
        trainable_models = getattr(self.cfg.model, "trainable_models", None)
        if trainable_models is None:
            return None
        if isinstance(trainable_models, str):
            trainable_set = {
                s.strip() for s in trainable_models.split(",") if s.strip()
            }
        else:
            trainable_set = set(trainable_models)
        if not trainable_set:
            self.log(
                "WARNING: trainable_models is empty; not filtering frozen "
                "entries from checkpoint to avoid saving an empty file."
            )
            return None
        frozen_prefixes = tuple(
            f"pipe.{name}."
            for name, _ in self.model.pipe.named_children()
            if name not in trainable_set
        )
        if frozen_prefixes:
            self.log(
                f"Frozen submodule prefixes excluded from checkpoints: "
                f"{list(frozen_prefixes)}"
            )
            return frozen_prefixes
        return None

    def save_checkpoint(self, epoch: int, step: int = 0):
        """Save model checkpoint via checkpoint_io (dispatches on model wrapper type)."""
        save_path = self.cfg.checkpoint.save_path
        ckpt_path = f"{save_path}/{epoch}_{step}" if step else f"{save_path}/{epoch}"
        _ckpt_io.save_checkpoint(
            ckpt_path=ckpt_path,
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            config=dataclasses.asdict(self.cfg),
            rank=self.rank,
            is_main=is_main_process(),
            epoch=epoch,
            global_step=self.global_step,
            seed=self.seed,
            normalizer_action=self.normalizer_action,
            normalizer_propri=self.normalizer_propri,
            dataset=getattr(self, "dataset", None) if step != 0 else None,
            grad_scaler=self.grad_scaler,
            log_fn=self.log,
            frozen_prefixes=self._frozen_prefixes,
        )
        _ckpt_io.finalize_save()
        self.log(f"Saved checkpoint to {ckpt_path}")

    def load_state_dict(self, model, resume_config):
        """Load state dict with fused-weight conversion + try_harder support."""
        return _ckpt_io.load_weights(
            model=model,
            resume_config=resume_config,
            model_class=self._checkpoint_model_class(),
            log_fn=self.log,
        )

    def _checkpoint_model_class(self):
        model_class = getattr(self, "ModelClass", None)
        if model_class is None and hasattr(type(self.adapter), "model_class"):
            model_class = type(self.adapter).model_class()
        return model_class

    def _resume_from_single_file(self) -> bool:
        path = self.cfg.checkpoint.resume_from
        return bool(path) and str(path).endswith((".safetensors", ".pth"))

    def _resume_from_training_checkpoint(self) -> bool:
        path = self.cfg.checkpoint.resume_from
        return bool(path) and not self._resume_from_single_file()

    def resume_from_checkpoint(self):
        """Resume training from checkpoint via checkpoint_io."""
        _ckpt_io.resume_from_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            resume_config={"ckpt": self.cfg.checkpoint.resume_from},
            rank=self.rank,
            grad_scaler=self.grad_scaler,
            model_class=self._checkpoint_model_class(),
            log_fn=self.log,
        )
        barrier()

    def predict_action_loop(
        self,
        current_step=0,
        max_iteration=None,
        prediction_type="flow_action",
        mode="acc",
    ):
        """Delegate action prediction to the adapter."""
        del mode  # unused in current dispatch; kept for signature compatibility
        return self.adapter.predict(
            prediction_type,
            model=self._unwrapped_model_for_inference(),
            val_dataloader=self.dataset.get_val_dataloader(),
            rank=self.rank,
            world_size=self.world_size,
            device=self.device,
            processor=self.processor,
            tokenizer_mixin=self.tokenizer_mixin,
            logger=self.wandb_run,
            current_step=current_step,
            max_iteration=max_iteration,
        )

    def predict_text_loop(self, current_step=0, max_samples=None, save_dir="./"):
        """Delegate text prediction to the adapter."""
        return self.adapter.predict(
            "text",
            model=self._unwrapped_model_for_inference(),
            val_dataloader=self.dataset.get_val_dataloader(),
            rank=self.rank,
            world_size=self.world_size,
            device=self.device,
            processor=self.processor,
            tokenizer_mixin=self.tokenizer_mixin,
            logger=self.wandb_run,
            current_step=current_step,
            max_samples=max_samples,
            save_dir=save_dir,
        )

    def _unwrapped_model_for_inference(self):
        """Return the underlying module for inference: unwrap DDP, pass FSDP through."""
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model
