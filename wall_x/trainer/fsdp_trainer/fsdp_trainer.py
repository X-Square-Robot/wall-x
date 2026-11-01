"""FSDP trainer implementation."""

import dataclasses
import copy
import gc
import json
import os
import torch
import time
import logging
import contextlib
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, Any, Tuple
from wall_x.config.loader import dump_config
from wall_x.trainer.optimizer import get_optimizer
from wall_x.utils.profile_zones import zone, Z

from wall_x.trainer.fsdp_trainer.base_trainer import (
    DistributedTrainer,
    is_main_process,
    barrier,
    get_trainer_process_group,
)
from wall_x.trainer.adapters import ADAPTER_REGISTRY, format_adapter_error
from wall_x.trainer.fsdp_trainer import checkpoint_io as _ckpt_io
from wall_x.trainer.fsdp_trainer import checkpoint_transaction as _ckpt_transaction
from wall_x.trainer.fsdp_trainer import portable_bundle as _portable_bundle
from wall_x.trainer.fsdp_trainer.distribution_strategy import build_strategy
from wall_x.trainer.metrics import MetricsLogger
from wall_x.trainer.utils import move_batch_to_device

from wall_x.model.core.action.normalizer import create_normalizers
from wall_x.trainer.scheduler.scheduler import get_scheduler


def _summarize_input_shapes(value, *, max_items: int = 96, max_depth: int = 4):
    """Return a bounded, JSON-serializable shape summary without tensor values."""
    summary = {}

    def visit(item, path: str, depth: int):
        if len(summary) >= max_items:
            return
        if isinstance(item, torch.Tensor):
            summary[path] = {
                "shape": list(item.shape),
                "dtype": str(item.dtype),
                "device": str(item.device),
                "requires_grad": item.requires_grad,
            }
            return
        if depth >= max_depth:
            summary[path] = {"type": type(item).__name__, "truncated": True}
            return
        if isinstance(item, dict):
            summary[path] = {"type": "dict", "len": len(item)}
            for key, child in item.items():
                visit(child, f"{path}.{key}", depth + 1)
            return
        if isinstance(item, (list, tuple)):
            summary[path] = {"type": type(item).__name__, "len": len(item)}
            for index, child in enumerate(item[:8]):
                visit(child, f"{path}[{index}]", depth + 1)
            return
        summary[path] = {"type": type(item).__name__}

    visit(value, "batch", 0)
    if len(summary) >= max_items:
        summary["<summary>"] = {"truncated_at_items": max_items}
    return summary


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

        self._run_rank0_portable_step(
            lambda: _portable_bundle.validate_checkpoint_root(
                self.cfg.checkpoint.save_path
            ),
            "validate checkpoint root",
        )

        # CUDA memory snapshot — start recording BEFORE adapter / strategy / model
        # construction so the dumped pickle covers:
        #   * adapter init (processor, tokenizer)
        #   * strategy build (FSDP2 mesh / process group)
        #   * model load + ckpt weight load
        #   * FSDP/DDP wrap (param sharding + DTensor wiring)
        #   * optimizer + scheduler creation (AdamW state ≈ 8B × #params, large)
        #   * dataset / dataloader build
        #   * the first few training steps
        # Rank 0 only — multi-rank snapshots would 8× the .pickle size for no extra
        # signal (the per-rank pattern is identical).
        if self.cfg.debug.enable_memory_snapshot and self.rank == 0:
            torch.cuda.memory._record_memory_history(
                max_entries=self.cfg.debug.memory_snapshot_max_entries
            )
            self.log(
                f"[MemSnap] recording started in __init__ (max_entries="
                f"{self.cfg.debug.memory_snapshot_max_entries})"
            )

        if self.model_type not in ADAPTER_REGISTRY:
            raise ValueError(format_adapter_error(self.model_type))
        adapter_cls = ADAPTER_REGISTRY[self.model_type]
        self.adapter = adapter_cls(
            cfg=self.cfg,
            logger=self.logger,
            model_type=self.model_type,
        )

        self.strategy = build_strategy(
            dataclasses.asdict(self.cfg.distributed),
            device=self.device,
            local_rank=self.local_rank,
            optimizer_type=self.cfg.hyperparams.optimizer.optimizer_type,
        )

        # Model config
        self.action_dim = self.cfg.action_dim
        self.use_selective_recompute = self.cfg.distributed.use_selective_recompute
        self.show_time_details = self.cfg.debug.show_time_details
        if self.show_time_details:
            self.timers._log_level = max(getattr(self.timers, "_log_level", -1), 0)
            self.log(
                "[TimeDetailsTimersEnabled] timers log_level set to 0",
                level=logging.INFO,
            )

        # num_training_steps: read from any scheduler that exposes it
        # (CosineSchedulerConfig currently; future schedulers may add it).
        # Used in train_loop to trigger loss_guard_should_stop — works with
        # constant scheduler too once the field is set.
        sched = self.cfg.hyperparams.scheduler
        self.num_training_steps = int(getattr(sched, "num_training_steps", 0) or 0)

        # ── loss_guard: optional per-step train_loss dump ─────────────
        # Set ``LOSS_GUARD_PATH`` env to opt in; trainer appends
        # ``{step, train_loss}`` each optimizer step (main rank only) and
        # writes a JSON at end-of-training or when ``num_training_steps`` is
        # reached. Used for cross-code-version loss-curve alignment (e.g.
        # acc_pi/wall-x baseline ↔ online_rl/wall-x verify).
        # Mirror of acc_pi fsdp_trainer.py:130-138, 1094-1106, 1268-1281.
        self.loss_guard_path = os.environ.get("LOSS_GUARD_PATH")
        self.loss_guard_records: list = []
        self.loss_guard_meta = {
            "id": os.environ.get("LOSS_GUARD_ID"),
            "name": os.environ.get("LOSS_GUARD_NAME"),
        }
        self.loss_guard_should_stop = False
        self.metrics_logger = MetricsLogger(
            wandb_run=self.wandb_run,
            log_interval=self.log_interval,
            smooth_window=self.cfg.logging.loss_log_smooth_window,
        )

        # Loss tracking
        self.base_l1_loss = None
        self.base_l1_loss_detail = {}

        startup_t0 = time.time()
        stage_t0 = time.time()
        self.load_normalizer()
        self.log("[StartupTiming] normalizer_s={:.3f}".format(time.time() - stage_t0))

        stage_t0 = time.time()
        self.load_processor()
        self.log("[StartupTiming] processor_s={:.3f}".format(time.time() - stage_t0))

        stage_t0 = time.time()
        self.load_model()
        self.log("[StartupTiming] model_s={:.3f}".format(time.time() - stage_t0))

        # DDP needs to see correct requires_grad at wrap time, so freeze first.
        stage_t0 = time.time()
        self._freeze_params_if_needed(self.model)
        self.log(
            "[StartupTiming] freeze_params_s={:.3f}".format(time.time() - stage_t0)
        )

        # Frozen-submodule prefixes for checkpoint filtering. Must be computed
        # before FSDP wrapping, while child modules still have stable names.
        stage_t0 = time.time()
        self._frozen_prefixes = self._compute_frozen_prefixes()

        # Original shapes captured before FSDP flattening.
        self.model._orig_param_shapes = {
            name: p.shape for name, p in self.model.named_parameters()
        }
        self.log(
            "[StartupTiming] param_metadata_s={:.3f}".format(time.time() - stage_t0)
        )

        if self._resume_from_single_file():
            stage_t0 = time.time()
            self.load_state_dict(
                self.model,
                {
                    "ckpt": self.cfg.checkpoint.resume_from,
                    "try_harder": self.cfg.checkpoint.try_harder,
                },
            )
            self.log(
                "[StartupTiming] single_file_resume_s={:.3f} path={}".format(
                    time.time() - stage_t0,
                    self.cfg.checkpoint.resume_from,
                )
            )

        stage_t0 = time.time()
        self._wrap_model(self.model)
        self.log("[StartupTiming] wrap_model_s={:.3f}".format(time.time() - stage_t0))

        stage_t0 = time.time()
        self._create_optimizer()
        self.log("[StartupTiming] optimizer_s={:.3f}".format(time.time() - stage_t0))

        stage_t0 = time.time()
        self._create_scheduler()
        self.log("[StartupTiming] scheduler_s={:.3f}".format(time.time() - stage_t0))

        if self._resume_from_training_checkpoint():
            stage_t0 = time.time()
            self.resume_from_checkpoint()
            self.log(
                "[StartupTiming] training_checkpoint_resume_s={:.3f}".format(
                    time.time() - stage_t0
                )
            )

        stage_t0 = time.time()
        self.load_dataset()
        self.log("[StartupTiming] dataset_s={:.3f}".format(time.time() - stage_t0))

        stage_t0 = time.time()
        self.adapter.init_validation(self.normalizer_action, self.normalizer_propri)
        self.log(
            "[StartupTiming] validation_init_s={:.3f} total_s={:.3f}".format(
                time.time() - stage_t0,
                time.time() - startup_t0,
            )
        )

    def load_normalizer(self):
        norm_cfg = self.cfg.data.normalizer_config or {}
        custom_stats_path = norm_cfg.get("customized_action_statistic_dof")
        if not custom_stats_path and self.cfg.data.dataset_type == "lerobot":
            from wall_x.data.backends.lerobot.build import load_lerobot_normalizers

            loaded = load_lerobot_normalizers(self.cfg)
            if loaded is not None:
                self.normalizer_action, self.normalizer_propri = (
                    loaded[0],
                    loaded[1],
                )
                self._action_statistic_dof = None
                self.logger.info(
                    "Loaded LeRobot normalizers from %s with dataset key %s",
                    loaded[2],
                    loaded[3],
                )
                return

        merged = {
            "dof_config": self.cfg.task.dof_config,
            "agent_pos_config": self.cfg.task.agent_pos_config,
            "normalizer_config": {
                "customized_action_statistic_dof": custom_stats_path,
                "min_key": norm_cfg.get("min_key", "min"),
                "delta_key": norm_cfg.get("delta_key", "delta"),
            },
        }
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
        model_t0 = time.time()
        self.model_config = self.adapter.build_model_config()
        self.log("[StartupTiming] model_config_s={:.3f}".format(time.time() - model_t0))
        create_t0 = time.time()
        self.model = self.adapter.create_model(
            self.processor, self.tokenizer_mixin, self.model_config
        )
        self.log(
            "[StartupTiming] model_create_s={:.3f}".format(time.time() - create_t0)
        )
        type(self.adapter).log_attention_implementation(self.logger, self.model)
        weights_t0 = time.time()
        self.adapter.load_weights(
            self.model,
            self.normalizer_action,
            self.normalizer_propri,
            processor=self.processor,
        )
        self.log(
            "[StartupTiming] adapter_load_weights_s={:.3f} model_total_s={:.3f}".format(
                time.time() - weights_t0,
                time.time() - model_t0,
            )
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
            backend=self.cfg.distributed.backend,
            log_fn=self.log,
        )
        from wall_x.trainer.optimizer.dmuon.utils import dmuon_owner_assignment_kwargs

        dmuon_owner_kwargs = dmuon_owner_assignment_kwargs(
            self.cfg.hyperparams.optimizer
        )
        if dmuon_owner_kwargs:
            self.log(
                "[DMuon] owner assignment config: "
                f"owner_strategy={dmuon_owner_kwargs['owner_strategy']}, "
                f"owner_cost_model={dmuon_owner_kwargs['owner_cost_model']}, "
                f"owner_pack_small_params={dmuon_owner_kwargs['pack_small_params']}, "
                f"hsdp_column_balance={dmuon_owner_kwargs['hsdp_column_balance']}"
            )
        self.model = self.strategy.wrap(
            model,
            use_dedicated_params=self.cfg.hyperparams.optimizer.use_dedicated_params,
            dmuon_owner_kwargs=dmuon_owner_kwargs,
        )

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
            if opt_cfg.optimizer_type not in (
                "adamw",
                "muon",
                "dmuon",
                "naive_fsdp2_muon",
            ):
                raise ValueError(
                    "optimizer.lr_groups are only supported with adamw, muon, "
                    "dmuon, or naive_fsdp2_muon"
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
                f"{group.name}(lr={group.lr if group.lr is not None else 'default'}, "
                f"weight_decay={group.weight_decay if group.weight_decay is not None else 'default'}) "
                f"include={group.include}"
                for group in lr_group_configs
            )
            self.log(
                f"setting optimizer parameter groups: base_lr={opt_cfg.learning_rate}; "
                f"{summary} ({opt_cfg.optimizer_type})"
            )

        optimizer_kwargs = {
            "opt_cfg": opt_cfg,
            "param_groups": param_groups,
        }
        if opt_cfg.optimizer_type in ("dmuon", "naive_fsdp2_muon"):
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
            self.lr_scheduler = get_scheduler(
                optimizer=self.optimizer,
                lr_scheduler_type="constant",
                num_warmup_steps=sched.num_warmup_steps,
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
        self._resume_checkpoint_path = None
        self._resume_loaded_global_step = None
        self._resume_next_global_step = self.global_step
        self._resume_consumed_samples = None
        self._resume_committed_batches = None
        self._resume_iter_offset = 0
        self._display_train_iter_total = None
        # Dispatch to adapter with unified signature (supports model-specific kwargs)
        extra_dataset_kwargs = dict(
            model_config=self.model_config,
            tokenizer_mixin=self.tokenizer_mixin,
            train_action_tokenizer=self.train_action_tokenizer,
            val_action_tokenizer=self.val_action_tokenizer,
            normalizer_action=self.normalizer_action,
            normalizer_propri=self.normalizer_propri,
            process_group=get_trainer_process_group(),
        )

        resume_path = self.cfg.checkpoint.resume_from
        skip_steps = int(self.cfg.checkpoint.resume_data_skip_steps)
        skip_samples_per_rank = int(
            self.cfg.checkpoint.resume_data_skip_samples_per_rank
        )
        if skip_steps < 0:
            raise ValueError(
                "checkpoint.resume_data_skip_steps must be >= 0, " f"got {skip_steps}"
            )
        if skip_samples_per_rank < 0:
            raise ValueError(
                "checkpoint.resume_data_skip_samples_per_rank must be >= 0, "
                f"got {skip_samples_per_rank}"
            )
        if skip_steps and skip_samples_per_rank:
            raise ValueError(
                "checkpoint.resume_data_skip_steps and "
                "checkpoint.resume_data_skip_samples_per_rank are mutually exclusive"
            )
        if (skip_steps or skip_samples_per_rank) and (
            resume_path is None or not os.path.isdir(resume_path)
        ):
            raise ValueError(
                "Data cursor skipping requires checkpoint.resume_from to point "
                "to an existing checkpoint directory"
            )
        if resume_path is not None and os.path.isdir(resume_path):
            resume_load_start = time.time()
            self._resume_checkpoint_path = resume_path
            # Step and epoch restore are generic. Backend-owned episode
            # metadata is optional and only loaded when the adapter exposes
            # the hook.
            has_data_cursor = False
            sampler_state_path = os.path.join(
                resume_path, f"sampler_state_rank_{self.rank}.pt"
            )
            if os.path.exists(sampler_state_path):
                has_data_cursor = True
                extra_dataset_kwargs["sampler_state_path"] = sampler_state_path

            dataset_state = _ckpt_io.load_dataset_state(
                checkpoint_path=resume_path,
                rank=self.rank,
                log_fn=self.log,
            )
            if skip_steps or skip_samples_per_rank:
                if dataset_state is None:
                    raise ValueError(
                        "Data cursor skipping requires a compatible "
                        "dataset_state_rank_<rank>.pt in the resume checkpoint"
                    )
                if skip_steps:
                    skip_samples_per_rank = (
                        skip_steps
                        * int(self.cfg.hyperparams.batch_size_per_gpu)
                        * int(self.grad_accum_steps)
                    )
                extra_dataset_kwargs["dataset_skip_samples_per_rank"] = (
                    skip_samples_per_rank
                )
                if skip_steps:
                    self.log(
                        "[Resume] resume_data_skip_steps={} batch_size_per_gpu={} "
                        "gradient_accumulation_steps={} "
                        "skip_samples_per_rank={}".format(
                            skip_steps,
                            self.cfg.hyperparams.batch_size_per_gpu,
                            self.grad_accum_steps,
                            skip_samples_per_rank,
                        )
                    )
                else:
                    self.log(
                        "[Resume] resume_data_skip_samples_per_rank={}".format(
                            skip_samples_per_rank
                        )
                    )
            if dataset_state is not None:
                has_data_cursor = True
                extra_dataset_kwargs["dataset_state"] = dataset_state

            if hasattr(self.adapter, "_load_episode_indices"):
                resume_indices = self.adapter._load_episode_indices(
                    resume_path, self.rank
                )
                if resume_indices is not None:
                    has_data_cursor = bool(
                        has_data_cursor
                        or resume_indices["is_incomplete_epoch"]
                    )
                    extra_dataset_kwargs["resume_indices"] = resume_indices

            train_state = _ckpt_io.load_trainer_state(resume_path)
            _ckpt_io.validate_resume_data_cursor(
                train_state, has_data_cursor=has_data_cursor
            )
            self.global_step = train_state["global_step"]
            self._resume_loaded_global_step = self.global_step
            self._resume_next_global_step = self.global_step
            self.start_epoch = train_state["start_epoch"]
            self.log(
                "[Resume] checkpoint={} completed_steps={} "
                "next_step_index={} start_epoch={} trainer_state_format={} "
                "dataset_state_loaded={} "
                "consumed_samples_per_rank={} load_state_s={:.1f}".format(
                    resume_path,
                    self._resume_loaded_global_step,
                    self._resume_next_global_step,
                    self.start_epoch,
                    train_state["format_version"],
                    dataset_state is not None,
                    self._resume_consumed_samples,
                    time.time() - resume_load_start,
                )
            )
        dataset_load_start = time.time()
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
        self._packing_enabled = bool(
            getattr(self.train_dataloader, "_wallx_packing_enabled", False)
        )
        resume_progress = getattr(self.dataset, "resume_progress", None)
        if callable(resume_progress):
            progress = resume_progress()
            self._resume_consumed_samples = int(
                progress.get("consumed_samples", 0) or 0
            )
            self._resume_committed_batches = int(
                progress.get("committed_batches", 0) or 0
            )
        if self._packing_enabled:
            if self.grad_accum_steps != 1:
                raise ValueError(
                    "Packed training currently requires "
                    "hyperparams.gradient_accumulation_steps=1"
                )
            ar_weight = getattr(self.cfg.model, "ar_action_loss_weight", None)
            if ar_weight is not None:
                raise ValueError(
                    "model.ar_action_loss_weight is not supported with packing "
                    "yet; packed channel-level loss accounting is a TODO"
                )
        self._prepare_train_iter_display(len(self.train_dataloader))
        if self._resume_checkpoint_path is not None:
            self.log(
                "[Resume] dataloader_epoch_iter_estimate={} iter_offset={} "
                "display_total_iters={} dataloader_batch_size={} "
                "dataset_build_s={:.1f}".format(
                    len(self.train_dataloader),
                    self._resume_iter_offset,
                    self._display_train_iter_total,
                    self._infer_dataloader_batch_size(),
                    time.time() - dataset_load_start,
                )
            )

    def _maybe_dump_oom_memsnap(self, exc):
        """Rescue dump of the CUDA allocator history when OOM hits before the
        normal step-3 snapshot has fired.

        Only fires when global_step < memory_snapshot_until_step, because:
        - After until_step, the normal dump already wrote a baseline pickle
          and ``_record_memory_history(enabled=None)`` stopped recording —
          a late dump here would be empty / stale.
        - And by then the allocator-history ring buffer is so long the
          memory_viz tool can't open the pickle anyway.

        Best-effort: any failure during dump is logged but swallowed, so we
        still re-raise the original OOM at the caller.
        """
        if not (
            self.cfg.debug.enable_memory_snapshot
            and self.rank == 0
            and self.global_step < self.cfg.debug.memory_snapshot_until_step
        ):
            return
        try:
            os.makedirs(self.cfg.debug.memory_snapshot_dir, exist_ok=True)
            snap_path = os.path.join(
                self.cfg.debug.memory_snapshot_dir,
                f"memory_snapshot_oom_step{self.global_step}.pickle",
            )
            torch.cuda.memory._dump_snapshot(snap_path)
            torch.cuda.memory._record_memory_history(enabled=None)
            self.log(
                f"[MemSnap-OOM] dumped to {snap_path} after "
                f"{type(exc).__name__} at step {self.global_step}; "
                f"recording stopped"
            )
        except Exception as dump_err:
            self.log(
                f"[MemSnap-OOM] dump failed at step {self.global_step}: "
                f"{type(dump_err).__name__}: {dump_err}",
                level=logging.WARNING,
            )

    @contextlib.contextmanager
    def _oom_dump_guard(self):
        """Context manager that rescues a CUDA memsnap on OOM, then re-raises.

        Wrap the GPU-compute-heavy sections (forward / backward / optimizer
        step) so that an early-step OOM produces a viz-openable allocator
        snapshot before the process dies. See ``_maybe_dump_oom_memsnap``
        for the gate condition.
        """
        try:
            yield
        except torch.OutOfMemoryError as e:
            self._maybe_dump_oom_memsnap(e)
            raise

    def train_loop(self, epoch: int, profiler=contextlib.nullcontext()):
        """Execute training for a single epoch"""
        self.model.train()

        # VGDynamicRobotDataset (v1 path) doesn't expose set_epoch; other
        # dataset backends (v2, lerobot) do. Dispatch conditionally.
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)
        total = len(self.train_dataloader)
        if self._resume_checkpoint_path is not None and epoch == self.start_epoch:
            display_total = self._display_train_iter_total
        else:
            self._resume_iter_offset = 0
            self._display_train_iter_total = total
            display_total = total
        # Drop last
        stop_step = total - total % self.grad_accum_steps

        # Disable automatic GC to prevent random ~200ms stalls during compute
        gc_interval = self.cfg.logging.gc_interval_steps
        gc.disable()

        t0 = time.time()
        pending_dataset_samples = 0
        pending_data_commit_tokens = []
        self._current_ordinary_token_observation = None
        adaptive_flow_weighted_sum = 0.0
        adaptive_flow_sample_count = 0
        adaptive_flow_min = None
        adaptive_flow_max = None
        adaptive_flow_enabled = False
        first_batch_timing_logged = False
        with profiler:
            self.timers("interval-time", log_level=0).start(
                barrier=self.show_time_details
            )
            self.timers("data-load", log_level=0).start(barrier=self.show_time_details)

            batch_iterator = self._iter_train_batches_synced_if_packed()
            for i, batch in enumerate(batch_iterator, self.initial_step):
                # With distributed training, a slow dataloader on one rank makes
                # faster ranks wait at the first forward/backward collective.
                # When detailed timing is enabled, charge that cross-rank skew
                # to data-load instead of compute so rank-local logs are easier
                # to interpret.
                self.timers("data-load").stop(barrier=self.show_time_details)
                if self.show_time_details and not first_batch_timing_logged:
                    first_batch_timing_logged = True
                    self.log(
                        "[DataTiming] first_batch_wait_s={:.3f} "
                        "local_iter={} display_iter={}/{} global_step={} "
                        "resume_checkpoint={}".format(
                            time.time() - t0,
                            i,
                            self._resume_iter_offset + i,
                            display_total,
                            self.global_step,
                            getattr(self, "_resume_checkpoint_path", None),
                        )
                    )
                batch_sample_count = self._batch_sample_count(batch)
                if self._packing_enabled and i == self.initial_step:
                    self._validate_packed_batch_protocol(batch)
                self._current_packed_throughput_observation = (
                    self._packed_throughput_observation(batch)
                    if self._packing_enabled
                    else None
                )
                self._accumulate_ordinary_token_observation(batch)
                data_commit_token = self._pop_data_commit_token(batch)
                adaptive_flow_observation = (
                    self._pop_adaptive_flow_sample_observation(batch)
                )
                if adaptive_flow_observation is not None:
                    adaptive_flow_enabled = True
                    flow_k, action_sample_count = adaptive_flow_observation
                    if action_sample_count > 0:
                        adaptive_flow_weighted_sum += flow_k * action_sample_count
                        adaptive_flow_sample_count += action_sample_count
                        adaptive_flow_min = (
                            flow_k
                            if adaptive_flow_min is None
                            else min(adaptive_flow_min, flow_k)
                        )
                        adaptive_flow_max = (
                            flow_k
                            if adaptive_flow_max is None
                            else max(adaptive_flow_max, flow_k)
                        )
                self._current_batch_input_length_stats = self._batch_input_length_stats(
                    batch
                )

                # Save first batch for offline profiling (rank 0 only)
                debug_batch_path = self.cfg.debug.save_debug_batch_path
                if debug_batch_path and i == self.initial_step and self.rank == 0:
                    os.makedirs(os.path.dirname(debug_batch_path) or ".", exist_ok=True)
                    torch.save({"batch": batch}, debug_batch_path)
                    self.log(f"Saved debug batch to {debug_batch_path}")

                # Move batch to device
                with zone(Z.DATA_TO_DEVICE):
                    self.timers("batch-to-device", log_level=0).start(barrier=False)
                    batch = move_batch_to_device(batch, self.device)
                    self.timers("batch-to-device").stop(sync=True)

                # Keep only tensor metadata so an exception can identify the exact
                # rank/microbatch shape without retaining another copy of inputs.
                self._last_batch_debug_context = {
                    "epoch": epoch,
                    "dataloader_iter": i,
                    "global_step": self.global_step,
                    "micro_step": getattr(self, "micro_step", None),
                    "batch_sample_count": batch_sample_count,
                    "inputs": _summarize_input_shapes(batch),
                }

                if self.cfg.debug.enable_mfu_profile and self.global_step == 0:
                    self.log(
                        "[MFU] Forward FLOPs profiling is not available in the public package.",
                        level=logging.INFO,
                    )

                # Forward pass with autocast
                self.timers("forward-compute", log_level=0).start(
                    barrier=self.show_time_details
                )
                with zone(Z.FORWARD), self._oom_dump_guard(), self.autocast_context():
                    outputs = self.adapter.forward(
                        self.model, batch, global_step=self.global_step
                    )
                self.timers("forward-compute").stop(sync=True)
                if self.show_time_details:
                    try:
                        _free_bytes, _total_bytes = torch.cuda.mem_get_info(self.device)
                        _alloc = torch.cuda.memory_allocated(self.device) / (1024**3)
                        _reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
                        _peak_alloc = torch.cuda.max_memory_allocated(self.device) / (
                            1024**3
                        )
                        _peak_reserved = torch.cuda.max_memory_reserved(self.device) / (
                            1024**3
                        )
                        self.log(
                            "[MemoryDetails] "
                            f"step={self.global_step} rank={self.rank} "
                            f"after_forward_allocated_gib={_alloc:.3f} "
                            f"after_forward_reserved_gib={_reserved:.3f} "
                            f"peak_allocated_gib={_peak_alloc:.3f} "
                            f"peak_reserved_gib={_peak_reserved:.3f} "
                            f"free_gib={_free_bytes / (1024 ** 3):.3f} "
                            f"total_gib={_total_bytes / (1024 ** 3):.3f}",
                            level=logging.INFO,
                        )
                    except Exception as _mem_exc:
                        self.log(
                            f"[MemoryDetails] failed: {type(_mem_exc).__name__}: {_mem_exc}",
                            level=logging.WARNING,
                        )

                loss = self.adapter.extract_loss(outputs)

                # Check for NaN loss -- use ``nan_to_num`` instead of
                # ``zeros_like`` or ``loss * 0``:
                #  - ``zeros_like(loss)`` returns a fresh leaf tensor with no
                #    grad_fn → backward crashes with "element 0 of tensors
                #    does not require grad and does not have a grad_fn".
                #  - ``loss * 0`` preserves grad_fn but ``NaN * 0 = NaN`` per
                #    IEEE 754, so grads still propagate as NaN.
                # ``torch.nan_to_num`` is differentiable: its forward replaces
                # NaN→0, its backward zeros the upstream grad at NaN positions
                # → clean grads, no NCCL deadlock, no NaN-poisoning of params.
                # Skipping the step is the caller's responsibility (e.g., via
                # GradScaler).
                nan_loss = torch.isnan(loss)
                if nan_loss:
                    self.log(
                        f"Warning: nan in loss at epoch: {epoch}, step: {i}",
                        level=logging.WARNING,
                    )
                    loss = torch.nan_to_num(loss, nan=0.0)

                # Backward pass
                self.timers("backward-compute", log_level=0).start(
                    barrier=self.show_time_details
                )

                context = (
                    contextlib.nullcontext()
                    if self.sync_gradients()
                    else self.strategy.no_sync(self.model)
                )
                with zone(Z.BACKWARD), context, self._oom_dump_guard():
                    scaled_loss = loss / self.grad_accum_steps
                    self.backward(scaled_loss)
                self.timers("backward-compute").stop()
                pending_dataset_samples += batch_sample_count
                pending_data_commit_tokens.append(data_commit_token)

                # Capture sync flag before the if/else below mutates micro_step /
                # global_step. Used at the end of the loop to gate profiler.step()
                # on the gbs boundary so 1 ProfilerStep = grad_accum_steps × (fwd+bwd)
                # + reduce-scatter + optim, comparable across grad_accum settings.
                did_optim_step = self.sync_gradients()

                # Gradient sync and optimizer step
                if did_optim_step:
                    self._current_adaptive_flow_sample_stats = (
                        self._reduce_adaptive_flow_sample_stats(
                            enabled=adaptive_flow_enabled,
                            weighted_sum=adaptive_flow_weighted_sum,
                            sample_count=adaptive_flow_sample_count,
                            local_min=adaptive_flow_min,
                            local_max=adaptive_flow_max,
                        )
                    )
                    adaptive_flow_weighted_sum = 0.0
                    adaptive_flow_sample_count = 0
                    adaptive_flow_min = None
                    adaptive_flow_max = None
                    adaptive_flow_enabled = False

                    self.timers("optimizer", log_level=0).start(
                        barrier=self.show_time_details
                    )

                    # Per-component grad norms (before clipping)
                    self.timers("optimizer-grad-norms", log_level=0).start(
                        barrier=False
                    )
                    self._component_grad_norms = {}
                    with zone(Z.GRAD_NORMS):
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
                    with zone(Z.GRAD_CLIP):
                        if self.cfg.hyperparams.optimizer.enable_grad_clip:
                            total_norm = self.clip_grad_norm(self.max_grad_norm)
                        else:
                            self._dedicated_param_grad_clip_stats = None
                            total_norm = 0
                    self.timers("optimizer-clip").stop()

                    self.timers("optimizer-step", log_level=0).start(barrier=False)
                    with zone(Z.OPTIMIZER_STEP), self._oom_dump_guard():
                        optimizer_updated = self.optimizer_step()
                    self.timers("optimizer-step").stop()

                    self.timers("optimizer-zero-grad", log_level=0).start(barrier=False)
                    with zone(Z.ZERO_GRAD):
                        self.optimizer_zero_grad()
                    self.timers("optimizer-zero-grad").stop()
                    self.timers("optimizer").stop()

                    self._require_packed_optimizer_update(optimizer_updated)

                    # Scheduler step
                    self.lr_scheduler_step()

                    # The optimizer/scheduler and dataset cursor form one
                    # completed training step.  Advance both counters before
                    # logging, validation, or publishing a checkpoint so every
                    # persisted component describes the same prefix.
                    self._commit_dataset_progress(
                        tokens=pending_data_commit_tokens,
                        samples=pending_dataset_samples,
                    )
                    pending_dataset_samples = 0
                    pending_data_commit_tokens = []
                    self.global_step += 1

                    # Logging
                    self.timers("logging", log_level=0).start(
                        barrier=self.show_time_details
                    )
                    with zone(Z.LOGGING):
                        _t_metrics_start = time.time()
                        self._log_training_metrics(
                            epoch, i, total, loss, total_norm, outputs
                        )
                        _t_metrics_ms = (time.time() - _t_metrics_start) * 1000

                    t1 = time.time()
                    self.training_log(
                        epoch,
                        self.num_epoch,
                        self._resume_iter_offset + i,
                        display_total,
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
                        with zone(Z.SAVE_CHECKPOINT):
                            self.save_checkpoint(epoch, self.global_step)
                        if is_main_process():
                            self.log(
                                f"[Step time breakdown] save_checkpoint took {(time.time() - _t_save_start):.1f} s at global_step={self.global_step}",
                                level=logging.INFO,
                            )

                    # Validation
                    if self._should_validate():
                        self.val_loop()

                    if (
                        self.cfg.debug.max_steps
                        and self.global_step >= self.cfg.debug.max_steps
                    ):
                        self.log(
                            f"[MaxSteps] debug.max_steps="
                            f"{self.cfg.debug.max_steps} reached, stopping"
                        )
                        self.log(
                            "[MaxMem] allocated="
                            f"{torch.cuda.max_memory_allocated() / 2**30:.2f}GB "
                            f"reserved={torch.cuda.max_memory_reserved() / 2**30:.2f}GB"
                        )
                        break
                    self.micro_step = 0

                    # Dump memory snapshot at the configured step (rank 0).
                    if (
                        self.cfg.debug.enable_memory_snapshot
                        and self.rank == 0
                        and self.global_step
                        == self.cfg.debug.memory_snapshot_until_step
                    ):
                        os.makedirs(self.cfg.debug.memory_snapshot_dir, exist_ok=True)
                        snap_path = os.path.join(
                            self.cfg.debug.memory_snapshot_dir,
                            "memory_snapshot.pickle",
                        )
                        torch.cuda.memory._dump_snapshot(snap_path)
                        torch.cuda.memory._record_memory_history(enabled=None)
                        self.log(
                            f"[MemSnap] dumped to {snap_path} at step "
                            f"{self.global_step}; recording stopped"
                        )

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
                if not self._packing_enabled and i == stop_step:
                    break

                # Fire profiler.step() per optim cycle (gbs boundary). With
                # grad_accum_steps > 1 this groups all micro fwd+bwd + the
                # cross-node reduce-scatter + optim into one ProfilerStep.
                # Without this guard, ProfilerStep tracked micro-batches and
                # traces from different grad_accum settings were not comparable.
                if did_optim_step and not isinstance(profiler, contextlib.nullcontext):
                    profiler.step()

                if (
                    self.num_training_steps > 0
                    and self.global_step >= self.num_training_steps
                ):
                    # Set the flag so base_trainer.train() exits the outer
                    # epoch loop instead of starting another epoch.
                    self.loss_guard_should_stop = True
                    break

                # Setup timers for next iteration
                if i < total - 1:
                    self.timers("interval-time", log_level=0).start(
                        barrier=self.show_time_details
                    )
                    self.timers("data-load", log_level=0).start(
                        barrier=self.show_time_details
                    )

        # Re-enable automatic GC after training loop
        gc.enable()

        # loss_guard JSON dump on main rank (only when the LOSS_GUARD_PATH env
        # is set). Writes when training is ending — either we hit
        # num_training_steps this epoch or this is the last configured epoch.
        if (
            is_main_process()
            and self.loss_guard_path
            and (self.loss_guard_should_stop or epoch == self.num_epoch - 1)
        ):
            self._write_loss_guard()

        # Reset dataloader for next epoch
        self.train_dataloader = self.dataset.get_train_dataloader()

    def _iter_train_batches_synced_if_packed(self):
        """Yield packed batches until every DP rank still has one available.

        Async packing makes ``len(loader)`` an estimate and different ranks can
        exhaust after different numbers of bins. All ranks therefore perform
        one availability reduction before entering model collectives. The
        ordinary DataLoader path is intentionally returned untouched.
        """
        if not getattr(self, "_packing_enabled", False):
            yield from self.train_dataloader
            return

        iterator = iter(self.train_dataloader)
        try:
            while True:
                try:
                    batch = next(iterator)
                    has_batch = 1
                except StopIteration:
                    batch = None
                    has_batch = 0
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    available = torch.tensor(
                        has_batch, dtype=torch.int32, device=self.device
                    )
                    torch.distributed.all_reduce(
                        available, op=torch.distributed.ReduceOp.MIN
                    )
                    has_batch = int(available.item())
                if not has_batch:
                    break
                yield batch
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()

    @staticmethod
    def _validate_packed_batch_protocol(batch) -> None:
        """Fail before forward if Dataset V2's packing contract is incomplete."""
        if not isinstance(batch, dict):
            raise TypeError("Packed loader must yield a dictionary batch")
        input_ids = batch.get("input_ids")
        if not torch.is_tensor(input_ids) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                "Packed Wall-X expects input_ids with shape [1, S], got "
                f"{getattr(input_ids, 'shape', None)}"
            )
        positional_masks = batch.get("positional_masks")
        pack_seq_lens = (
            positional_masks.get("pack_seq_lens")
            if isinstance(positional_masks, dict)
            else None
        )
        if pack_seq_lens is None:
            raise ValueError(
                "Packed batch is missing positional_masks.pack_seq_lens; this "
                "field is the sole model-side packing feature gate"
            )
        if torch.is_tensor(pack_seq_lens):
            pack_seq_lens = pack_seq_lens.detach().cpu().tolist()
        if len(pack_seq_lens) == 1 and isinstance(pack_seq_lens[0], (list, tuple)):
            pack_seq_lens = pack_seq_lens[0]
        lengths = [int(value) for value in pack_seq_lens]
        if not lengths or any(value <= 0 for value in lengths):
            raise ValueError(f"Invalid packed sequence lengths: {lengths}")
        if sum(lengths) > int(input_ids.shape[1]):
            raise ValueError(
                "Packed sequence lengths exceed padded input length: "
                f"sum={sum(lengths)} S={input_ids.shape[1]}"
            )
        packed_count = int(batch.get("_packed_sample_count", len(lengths)))
        if packed_count != len(lengths):
            raise ValueError(
                "_packed_sample_count does not match pack_seq_lens: "
                f"{packed_count} != {len(lengths)}"
            )
        requested = int(batch.get("_requested_batch_size", packed_count))
        if requested < packed_count:
            raise ValueError(
                "_requested_batch_size cannot be smaller than packed samples: "
                f"{requested} < {packed_count}"
            )
        if batch.get("_data_commit_token") is None:
            raise ValueError(
                "Packed batch is missing _data_commit_token; Wall-X cannot "
                "checkpoint a model/data-consistent prefix"
            )

    def _write_loss_guard(self):
        """Dump collected per-step train_loss records to LOSS_GUARD_PATH.

        Idempotent: re-callable across epochs without harm. Layout matches
        acc_pi/wall-x ``fsdp_trainer._write_loss_guard`` so the two JSONs are
        diffable directly.
        """
        if not self.loss_guard_path:
            return
        dirname = os.path.dirname(self.loss_guard_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        payload = {
            "id": self.loss_guard_meta.get("id"),
            "name": self.loss_guard_meta.get("name"),
            "seed": getattr(self, "seed", None),
            "num_epochs": self.num_epoch,
            "global_step": int(self.global_step),
            "losses": [r["train_loss"] for r in self.loss_guard_records],
            "records": self.loss_guard_records,
        }
        with open(self.loss_guard_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _batch_sample_count(self, batch) -> int:
        """Return sampler indices consumed by this per-rank DataLoader batch.

        ``_requested_batch_size`` is captured by dataset collation before bad
        samples or epilogue filters remove model rows.  Cursor advancement must
        use that pre-filter count so dropped indices are not retried on resume.
        """
        if isinstance(batch, dict):
            requested_batch_size = batch.get("_requested_batch_size")
            if requested_batch_size is not None:
                requested_batch_size = int(requested_batch_size)
                if requested_batch_size <= 0:
                    raise ValueError(
                        "_requested_batch_size must be > 0 for dataset cursor "
                        f"advancement, got {requested_batch_size}"
                    )
                return requested_batch_size

        # Backwards compatibility for fixed-size DataLoaders built by older
        # dataset versions that do not emit _requested_batch_size.  A custom
        # variable-size batch_sampler exposes DataLoader.batch_size == None and
        # must provide the explicit metadata above.
        fixed_batch_size = getattr(self.train_dataloader, "batch_size", None)
        if fixed_batch_size is not None:
            fixed_batch_size = int(fixed_batch_size)
            if fixed_batch_size <= 0:
                raise ValueError(
                    "DataLoader.batch_size must be > 0 for dataset cursor "
                    f"advancement, got {fixed_batch_size}"
                )
            return fixed_batch_size

        if torch.is_tensor(batch) and batch.ndim > 0:
            return int(batch.shape[0])
        if isinstance(batch, (list, tuple)):
            return len(batch)
        raise RuntimeError(
            "Cannot determine sampler cursor advancement for a variable-size "
            "batch. Dataset collation must provide '_requested_batch_size'."
        )

    def _batch_input_length_stats(self, batch) -> dict:
        """Return local/global input_ids.shape[1] stats for diagnostics."""
        if not isinstance(batch, dict):
            return {}
        input_ids = batch.get("input_ids")
        if not torch.is_tensor(input_ids) or input_ids.ndim < 2:
            return {}

        local_len = int(input_ids.shape[1])
        stats = {"input_ids_len_local": local_len}
        length_tensor = torch.tensor(
            float(local_len),
            dtype=torch.float32,
            device=self.device if torch.cuda.is_available() else torch.device("cpu"),
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            min_len = length_tensor.clone()
            max_len = length_tensor.clone()
            torch.distributed.all_reduce(min_len, op=torch.distributed.ReduceOp.MIN)
            torch.distributed.all_reduce(max_len, op=torch.distributed.ReduceOp.MAX)
            stats["input_ids_len_min"] = int(min_len.item())
            stats["input_ids_len_max"] = int(max_len.item())
        else:
            stats["input_ids_len_min"] = local_len
            stats["input_ids_len_max"] = local_len
        return stats

    @staticmethod
    def _packed_throughput_observation(batch) -> tuple[int, int, int]:
        """Return model-fed samples plus useful and physical token counts.

        Packed rows have a physical batch dimension of one, so neither
        ``input_ids.shape[0]`` nor the configured ordinary-loader batch size
        describes their sample throughput.  ``_packed_sample_count`` is the
        number of member samples actually concatenated into the row, while
        ``pack_seq_lens`` excludes fixed padding up to ``pack_cutoff``.
        """
        if not isinstance(batch, dict):
            raise TypeError("Packed throughput accounting requires a dict batch")

        packed_count = batch.get("_packed_sample_count")
        positional_masks = batch.get("positional_masks")
        pack_seq_lens = (
            positional_masks.get("pack_seq_lens")
            if isinstance(positional_masks, dict)
            else None
        )
        if packed_count is None or pack_seq_lens is None:
            raise ValueError(
                "Packed throughput accounting requires _packed_sample_count "
                "and positional_masks.pack_seq_lens"
            )
        if torch.is_tensor(pack_seq_lens):
            pack_seq_lens = pack_seq_lens.detach().cpu().tolist()
        if len(pack_seq_lens) == 1 and isinstance(pack_seq_lens[0], (list, tuple)):
            pack_seq_lens = pack_seq_lens[0]

        lengths = [int(value) for value in pack_seq_lens]
        packed_count = int(packed_count)
        if packed_count <= 0 or packed_count != len(lengths):
            raise ValueError(
                "Packed throughput sample count does not match pack_seq_lens: "
                f"{packed_count} != {len(lengths)}"
            )
        if any(length <= 0 for length in lengths):
            raise ValueError(f"Invalid packed throughput sequence lengths: {lengths}")
        input_ids = batch.get("input_ids")
        if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
            raise ValueError("Packed throughput accounting requires 2-D input_ids")
        return packed_count, sum(lengths), int(input_ids.numel())

    @staticmethod
    def _ordinary_token_observation(batch) -> tuple[int, int]:
        """Return physical and non-padding token counts for an ordinary batch."""
        if not isinstance(batch, dict):
            raise TypeError("Token throughput accounting requires a dict batch")
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
            raise ValueError("Token throughput accounting requires 2-D input_ids")
        if (
            not torch.is_tensor(attention_mask)
            or attention_mask.shape != input_ids.shape
        ):
            raise ValueError(
                "Token throughput accounting requires attention_mask to match input_ids"
            )
        return int(input_ids.numel()), int(torch.count_nonzero(attention_mask).item())

    def _accumulate_ordinary_token_observation(self, batch) -> None:
        """Accumulate token counts over every microbatch in one optimizer cycle."""
        if self._packing_enabled or os.environ.get("WALLX_LOG_TOKEN_PERF", "0") != "1":
            return
        physical_tokens, useful_tokens = self._ordinary_token_observation(batch)
        accumulated = self._current_ordinary_token_observation or (0, 0)
        self._current_ordinary_token_observation = (
            accumulated[0] + physical_tokens,
            accumulated[1] + useful_tokens,
        )

    def _require_packed_optimizer_update(self, optimizer_updated: bool) -> None:
        """Keep a skipped fp16 update from advancing an exact packed cursor."""
        if not self._packing_enabled or optimizer_updated:
            return
        raise RuntimeError(
            "GradScaler skipped a packed optimizer update after detecting "
            "non-finite gradients. Dataset progress, scheduler state, and "
            "global_step were not advanced; restart from the last checkpoint "
            "so the uncommitted packed batch is replayed exactly."
        )

    def _log_ordinary_token_interval(self, time_per_step: float) -> None:
        """Log exact per-rank token counters without adding a collective."""
        observation = getattr(self, "_current_ordinary_token_observation", None)
        if observation is None:
            return
        # One observation covers the complete optimizer accumulation cycle.
        # Consume it exactly once even if this helper is called again.
        self._current_ordinary_token_observation = None

        skip_steps = int(getattr(self.cfg.logging, "perf_log_skip_steps", 0) or 0)
        if self.global_step < skip_steps:
            self._ordinary_perf_physical_tokens = 0
            self._ordinary_perf_useful_tokens = 0
            self._ordinary_perf_seconds = 0.0
            self._ordinary_perf_steps = 0
            return

        physical_tokens, useful_tokens = observation
        self._ordinary_perf_physical_tokens = int(
            getattr(self, "_ordinary_perf_physical_tokens", 0)
        ) + int(physical_tokens)
        self._ordinary_perf_useful_tokens = int(
            getattr(self, "_ordinary_perf_useful_tokens", 0)
        ) + int(useful_tokens)
        self._ordinary_perf_seconds = float(
            getattr(self, "_ordinary_perf_seconds", 0.0)
        ) + float(time_per_step)
        self._ordinary_perf_steps = int(getattr(self, "_ordinary_perf_steps", 0)) + 1

        if self.global_step % int(self.cfg.logging.log_interval) != 0:
            return

        self.log(
            "[TokenPerf-rank] rank={} steps={} physical_tokens={} "
            "useful_tokens={} seconds={:.9f}".format(
                getattr(self, "rank", 0),
                self._ordinary_perf_steps,
                self._ordinary_perf_physical_tokens,
                self._ordinary_perf_useful_tokens,
                self._ordinary_perf_seconds,
            ),
            main_process_only=False,
        )
        self._ordinary_perf_physical_tokens = 0
        self._ordinary_perf_useful_tokens = 0
        self._ordinary_perf_seconds = 0.0
        self._ordinary_perf_steps = 0

    def _packed_interval_throughput_stats(self, time_per_step: float) -> dict:
        """Reduce exact packed throughput once per logging interval.

        Each rank accumulates Python scalars on every optimizer step.  At the
        normal metric-emission cadence, one small all-gather obtains global
        member/token counts and the slowest rank's accumulated training time.
        The collective is intentionally outside the measured step-time window
        and is not executed on intervening steps.
        """
        if not getattr(self, "_packing_enabled", False):
            return {}

        observation = getattr(self, "_current_packed_throughput_observation", None)
        if observation is None:
            raise RuntimeError(
                "Packed training step is missing its throughput observation"
            )

        skip_steps = int(getattr(self.cfg.logging, "perf_log_skip_steps", 0) or 0)
        if self.global_step < skip_steps:
            self._packed_perf_local_samples = 0
            self._packed_perf_local_tokens = 0
            self._packed_perf_local_physical_tokens = 0
            self._packed_perf_local_seconds = 0.0
            return {}

        samples, tokens, physical_tokens = observation
        self._packed_perf_local_samples = int(
            getattr(self, "_packed_perf_local_samples", 0)
        ) + int(samples)
        self._packed_perf_local_tokens = int(
            getattr(self, "_packed_perf_local_tokens", 0)
        ) + int(tokens)
        self._packed_perf_local_physical_tokens = int(
            getattr(self, "_packed_perf_local_physical_tokens", 0)
        ) + int(physical_tokens)
        self._packed_perf_local_seconds = float(
            getattr(self, "_packed_perf_local_seconds", 0.0)
        ) + float(time_per_step)

        log_interval = int(self.cfg.logging.log_interval)
        if self.global_step % log_interval != 0:
            return {}

        device = self.device if torch.cuda.is_available() else torch.device("cpu")
        local = torch.tensor(
            [
                float(self._packed_perf_local_samples),
                float(self._packed_perf_local_tokens),
                float(self._packed_perf_local_physical_tokens),
                self._packed_perf_local_seconds,
            ],
            dtype=torch.float64,
            device=device,
        )
        gathered = self.gather_tensor(local)
        if gathered.ndim == 1:
            gathered = gathered.unsqueeze(0)
        values = gathered.detach().cpu()
        global_samples = float(values[:, 0].sum().item())
        global_tokens = float(values[:, 1].sum().item())
        global_physical_tokens = float(values[:, 2].sum().item())
        interval_seconds = float(values[:, 3].max().item())

        self._packed_perf_local_samples = 0
        self._packed_perf_local_tokens = 0
        self._packed_perf_local_physical_tokens = 0
        self._packed_perf_local_seconds = 0.0

        denominator = max(interval_seconds, 1e-9)
        packed_samples_per_sec = global_samples / denominator
        return {
            # Preserve the public metric name while making it exact for packed
            # rows. The explicit alias makes dashboards self-describing.
            "samples_per_sec": packed_samples_per_sec,
            "packed_samples_per_sec": packed_samples_per_sec,
            "packed_raw_tokens_per_sec": global_tokens / denominator,
            "packed_physical_tokens_per_sec": global_physical_tokens / denominator,
            "packed_non_padded_ratio": global_tokens / max(global_physical_tokens, 1.0),
        }

    @staticmethod
    def _pop_adaptive_flow_sample_observation(batch):
        """Remove trainer-only adaptive-flow metadata and return ``(K, N)``.

        ``N`` is the number of action rows, so the reported mean is weighted by
        samples that actually receive flow supervision.  Absence of the private
        key is the feature gate: fixed-flow runs incur no reduction or logging.
        """
        if not isinstance(batch, dict):
            return None
        value = batch.pop("_num_flow_samples_effective", None)
        if value is None:
            return None
        try:
            flow_k = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "_num_flow_samples_effective must be an integer, "
                f"got {value!r}"
            ) from exc
        if flow_k < 1:
            raise ValueError(
                "_num_flow_samples_effective must be >= 1, "
                f"got {flow_k}"
            )

        action_chunk = batch.get("action_chunk")
        action_sample_count = (
            int(action_chunk.shape[0])
            if torch.is_tensor(action_chunk) and action_chunk.ndim >= 1
            else 0
        )
        return flow_k, action_sample_count

    def _reduce_adaptive_flow_sample_stats(
        self,
        *,
        enabled: bool,
        weighted_sum: float,
        sample_count: int,
        local_min,
        local_max,
    ) -> dict:
        """Aggregate one optimizer step across gradient accumulation and ranks."""
        if not enabled:
            return {}

        device = self.device if torch.cuda.is_available() else torch.device("cpu")
        totals = torch.tensor(
            [float(weighted_sum), float(sample_count)],
            dtype=torch.float32,
            device=device,
        )
        extrema = torch.tensor(
            [
                float(local_min) if local_min is not None else float("inf"),
                -float(local_max) if local_max is not None else float("inf"),
            ],
            dtype=torch.float32,
            device=device,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
            # MIN over [min(K), -max(K)] obtains both extrema in one collective.
            torch.distributed.all_reduce(extrema, op=torch.distributed.ReduceOp.MIN)

        global_sum, global_count = totals.cpu().tolist()
        if global_count <= 0:
            return {}
        global_min, neg_global_max = extrema.cpu().tolist()
        return {
            "adaptive_flow_samples/mean": global_sum / global_count,
            "adaptive_flow_samples/min": global_min,
            "adaptive_flow_samples/max": -neg_global_max,
        }

    def _infer_dataloader_batch_size(self) -> Optional[int]:
        """Best-effort batch-size inference for resume progress display."""
        cfg_data = getattr(self.cfg, "data", None)
        for value in (
            getattr(self.train_dataloader, "batch_size", None),
            getattr(self.cfg.hyperparams, "batch_size_per_gpu", None),
            getattr(cfg_data, "batch_size", None),
        ):
            if value is None:
                continue
            try:
                batch_size = int(value)
            except (TypeError, ValueError):
                continue
            if batch_size > 0:
                return batch_size
        return None

    def _prepare_train_iter_display(self, loader_iters: int) -> None:
        """Compute display-only epoch iter offsets after dataset resume."""
        self._resume_iter_offset = 0
        committed_batches = getattr(self, "_resume_committed_batches", None)
        if getattr(self, "_packing_enabled", False) and committed_batches is not None:
            self._resume_iter_offset = int(committed_batches)
            # AsyncPackedDataLoader.__len__ is a cached full-epoch estimate,
            # including after a v7 frontier is loaded.  Adding the committed
            # prefix would therefore count that prefix twice.
            self._display_train_iter_total = int(loader_iters)
            return
        consumed_samples = getattr(self, "_resume_consumed_samples", None)
        batch_size = self._infer_dataloader_batch_size()
        if consumed_samples is not None and batch_size:
            self._resume_iter_offset = consumed_samples // batch_size
            remainder = consumed_samples % batch_size
            if remainder:
                self.log(
                    "[Resume] consumed_samples_per_rank={} is not divisible by "
                    "dataloader_batch_size={}; display iter offset is rounded down".format(
                        consumed_samples,
                        batch_size,
                    ),
                    level=logging.WARNING,
                )
        self._display_train_iter_total = self._resume_iter_offset + int(loader_iters)

    @staticmethod
    def _pop_data_commit_token(batch):
        """Remove loader-only commit metadata before model/device handling."""
        if not isinstance(batch, dict):
            return None
        return batch.pop("_data_commit_token", None)

    def _commit_dataset_progress(self, *, tokens, samples: int) -> None:
        """Commit exactly the data prefix included by a successful step."""
        dataset = getattr(self, "dataset", None)
        if getattr(self, "_packing_enabled", False):
            if len(tokens) != 1 or tokens[0] is None:
                raise RuntimeError(
                    "Packed optimizer steps require exactly one data commit token"
                )
            if dataset is None or not hasattr(dataset, "commit_data"):
                raise RuntimeError(
                    "Packed dataset wrapper does not expose commit_data(token=...)"
                )
            dataset.commit_data(token=tokens[0], samples=int(samples))
            return

        if dataset is not None and hasattr(dataset, "commit_data"):
            dataset.commit_data(token=None, samples=int(samples))
            return
        self._advance_dataset(samples)

    def _advance_dataset(self, n: int) -> None:
        """Advance backend-owned dataset cursors after a committed train step."""
        if n <= 0:
            return
        dataset = getattr(self, "dataset", None)
        if dataset is not None and hasattr(dataset, "advance"):
            dataset.advance(n)
        elif dataset is not None and hasattr(dataset, "advance_train_sampler"):
            dataset.advance_train_sampler(n)

    def _log_training_metrics(self, epoch, step, total, loss, total_norm, outputs):
        """Collect per-step stats, delegate buffering + wandb emission."""
        lr = self.get_lr()
        train_loss = self.reduce_tensor(loss.detach()).item()

        # ── loss_guard: append per-step train_loss on main rank ──
        # Stops appending once we've recorded num_training_steps points so the
        # JSON payload is bounded even if extra epochs slip through.
        if is_main_process() and self.loss_guard_path:
            if (
                self.num_training_steps <= 0
                or len(self.loss_guard_records) < self.num_training_steps
            ):
                self.loss_guard_records.append(
                    {"step": int(self.global_step), "train_loss": float(train_loss)}
                )

        step_stats = {
            "lr": lr,
            "train_loss": train_loss,
            "grad_norm": total_norm,  # materialized in the packed D2H below
        }
        input_length_stats = getattr(self, "_current_batch_input_length_stats", None)
        if input_length_stats:
            step_stats.update(input_length_stats)
        adaptive_flow_stats = getattr(
            self, "_current_adaptive_flow_sample_stats", None
        )
        if adaptive_flow_stats:
            step_stats.update(adaptive_flow_stats)
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

        # Single packed D2H for every deferred 0-d tensor stat (grad_norm +
        # the dmuon clip stats below). Each stat used to sync separately --
        # before the dmuon defer-only stats contract there were 5-6 pipeline drains per
        # step in the clip/step region; now there is exactly one, here.
        def _materialize_scalar_stats(mapping: dict) -> None:
            keys = [
                k for k, v in mapping.items() if torch.is_tensor(v) and v.numel() == 1
            ]
            if not keys:
                return
            def _local(t):
                if hasattr(t, "full_tensor"):
                    t = t.full_tensor()
                return t.detach().float().squeeze()

            packed = (
                torch.stack([_local(mapping[k]) for k in keys])
                .cpu()
                .tolist()
            )
            for k, v in zip(keys, packed):
                mapping[k] = float(v)

        dedicated_clip_stats = getattr(self, "_dedicated_param_grad_clip_stats", None)
        if dedicated_clip_stats is not None:
            # Either bucket may be absent (dmuon reports stats only for
            # buckets that had grads); guard both symmetrically.
            if "total_norm" in dedicated_clip_stats:
                step_stats.update(
                    {
                        "muon_grad_norm": dedicated_clip_stats["total_norm"],
                        "muon_grad_clip_coef": dedicated_clip_stats["clip_coef"],
                        "muon_grad_clipped": dedicated_clip_stats["clipped"],
                    }
                )
            if "adamw_total_norm" in dedicated_clip_stats:
                step_stats.update(
                    {
                        "dmuon_adamw_grad_norm": dedicated_clip_stats[
                            "adamw_total_norm"
                        ],
                        "dmuon_adamw_grad_clip_coef": dedicated_clip_stats[
                            "adamw_clip_coef"
                        ],
                        "dmuon_adamw_grad_clipped": dedicated_clip_stats[
                            "adamw_clipped"
                        ],
                        "dmuon_adamw_grad_found_inf": dedicated_clip_stats[
                            "adamw_found_inf"
                        ],
                    }
                )

        _materialize_scalar_stats(step_stats)

        param_norm_interval = int(
            getattr(self.cfg.logging, "param_norm_interval", 100) or 0
        )
        if param_norm_interval > 0 and self.global_step % param_norm_interval == 0:
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
        timer_names = [
            "interval-time",
            "data-load",
            "batch-to-device",
            "forward-compute",
            "backward-compute",
            "optimizer",
            "optimizer-grad-norms",
            "optimizer-clip",
            "optimizer-step",
            "optimizer-zero-grad",
            "logging",
        ]
        timer_detail_stats = {}
        # timers.log() contains all_gather — must run on ALL ranks before the
        # is_main_process() guard to avoid NCCL deadlock.
        if show_time_details:
            timer_detail_stats = self._collect_timer_detail_stats(
                timer_names, time_per_step
            )
            self.timers.log(timer_names, normalizer=1)

        main = is_main_process()
        stats = getattr(self, "_current_step_stats", None) or {}
        time_stats = {"time_per_iter": float(time_per_step)}
        if timer_detail_stats:
            time_stats.update(timer_detail_stats)
        stats.update(time_stats)
        raw_stats = getattr(self, "_current_step_raw_stats", None)
        if raw_stats is not None:
            raw_stats.update(time_stats)

        # MFU is computed after step time is known, then merged into both the
        # console stats and the MetricsLogger buffer so wandb records it.
        mfu_info = None
        if self.cfg.debug.enable_mfu and hasattr(self.adapter, "compute_mfu"):
            unwrapped = (
                self.model.module if hasattr(self.model, "module") else self.model
            )
            mfu_info = self.adapter.compute_mfu(unwrapped, time_per_step)
            # Per-rank [MFU-rank] line is verbose (N_rank × N_step entries).
            # Gate it on log_interval so file logs match wandb emission cadence.
            log_mfu_line = (
                mfu_info is not None
                and self.global_step % self.cfg.logging.log_interval == 0
            )
            if log_mfu_line:
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
                }
                flops_forward = mfu_info.get("flops_forward_T")
                if flops_forward is not None:
                    mfu_stats["flops_forward_T"] = flops_forward
                profile_fwd_flops = mfu_info.get("profile_fwd_flops_T")
                if profile_fwd_flops is not None:
                    mfu_stats["profile_fwd_flops_T"] = profile_fwd_flops
                stats.update(mfu_stats)
                raw_stats = getattr(self, "_current_step_raw_stats", None)
                if raw_stats is not None:
                    raw_stats.update(mfu_stats)

        # All ranks must enter this cadence-gated collective. It communicates
        # only four float64 scalars per rank and runs outside time_per_step.
        packed_throughput_stats = self._packed_interval_throughput_stats(time_per_step)
        self._log_ordinary_token_interval(time_per_step)

        if main:
            # Expose per-step wall time on the same wandb track as loss/lr/mfu.
            # samples_per_sec 是全局吞吐（所有 rank 求和），跟 micro/accum 切分无关，
            # 只看 global batch / step time。
            inv_time = 1.0 / max(float(time_per_step), 1e-9)
            step_time_stats = {"step_time": float(time_per_step)}
            if self._packing_enabled:
                step_time_stats.update(packed_throughput_stats)
            else:
                samples_per_step = (
                    self.cfg.hyperparams.batch_size_per_gpu
                    * self.world_size
                    * self.grad_accum_steps
                )
                step_time_stats["samples_per_sec"] = samples_per_step * inv_time
            if stats is not None:
                stats.update(step_time_stats)
            raw_stats = getattr(self, "_current_step_raw_stats", None)
            if raw_stats is not None:
                # 跳过 warm-up：前 N 步 step_time / samples_per_sec 不进 wandb，
                # 否则 step 0 ~85s 把 y 轴拉爆后续看不清。控制台 line（stats）
                # 仍然显示原值。其他 key（loss / lr / mfu / grad_norm）照常写。
                if self.global_step >= self.cfg.logging.perf_log_skip_steps:
                    raw_stats.update(step_time_stats)
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

    def _collect_timer_detail_stats(self, timer_names, time_per_step):
        """Collect per-step timer details before ``timers.log`` resets them."""
        try:
            elapsed = self.timers._get_global_min_max_time(
                timer_names,
                reset=False,
                barrier=False,
                normalizer=1.0,
            )
        except Exception as exc:
            self.log(
                f"[TimeDetails] failed to collect timer stats: {type(exc).__name__}: {exc}",
                level=logging.WARNING,
            )
            return {}

        step_time = max(float(time_per_step), 1e-9)
        stats = {}
        for name, (min_time, max_time) in elapsed.items():
            key = name.replace("-", "_")
            stats[f"time/{key}_min_s"] = float(min_time)
            stats[f"time/{key}_max_s"] = float(max_time)
            stats[f"time/{key}_max_pct"] = float(max_time) / step_time
        return stats

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
            log.enable_validation
            and self.global_step >= log.ignore_until_interval
            and self.global_step % log.val_interval == 0
            and self.global_step > 0
        )

    def _checkpoint_config_dict(self) -> dict:
        # dump_config (not asdict): persist the effective config in the same
        # typed yaml schema as hand-written configs.
        config = dump_config(self.cfg)
        effective_data_config = getattr(
            getattr(self, "dataset", None), "effective_data_config", None
        )
        if effective_data_config is None:
            return config

        config.setdefault("data", {})["dataset_config_effective"] = (
            self._dataset_config_params_only(effective_data_config)
        )
        return config

    @staticmethod
    def _dataset_config_params_only(effective_data_config: dict) -> dict:
        data_config = copy.deepcopy(effective_data_config)
        data_config.pop("tasks", None)

        dataset_config = data_config.get("dataset")
        if not isinstance(dataset_config, dict):
            return data_config

        dataset_config.pop("sources", None)
        dataset_config.pop("sources_include", None)

        sampler = dataset_config.get("sampler")
        if not isinstance(sampler, dict):
            return data_config

        task_balance = sampler.get("task_balance")
        if not isinstance(task_balance, dict):
            return data_config

        task_balance.pop("counts_per_task", None)
        task_balance.pop("report_path", None)
        params = task_balance.get("params")
        if isinstance(params, dict):
            params.pop("counts_per_task", None)
            if not params:
                task_balance.pop("params", None)
        return data_config

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
        runtime_config = self._checkpoint_config_dict()
        effective_data_config = getattr(
            getattr(self, "dataset", None), "effective_data_config", None
        )
        include_pretrained = bool(self._frozen_prefixes)
        self._run_rank0_portable_step(
            lambda: _portable_bundle.ensure_portable_assets(
                save_path=save_path,
                config=runtime_config,
                processor=self.processor,
                effective_data_config=effective_data_config,
                include_pretrained=include_pretrained,
            ),
            "create portable checkpoint assets",
        )
        staging_path = self._begin_checkpoint_transaction(ckpt_path)
        if is_main_process():
            _portable_bundle.link_checkpoint_compat_assets(staging_path)
        barrier()
        checkpoint_config = _portable_bundle.portable_checkpoint_config(
            runtime_config, include_pretrained=include_pretrained
        )
        _ckpt_io.save_checkpoint(
            ckpt_path=staging_path,
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            config=checkpoint_config,
            rank=self.rank,
            is_main=is_main_process(),
            epoch=epoch,
            global_step=self.global_step,
            epoch_complete=step == 0,
            seed=self.seed,
            normalizer_action=self.normalizer_action,
            normalizer_propri=self.normalizer_propri,
            dataset=getattr(self, "dataset", None) if step != 0 else None,
            grad_scaler=self.grad_scaler,
            log_fn=self.log,
            frozen_prefixes=self._frozen_prefixes,
        )
        # Do not expose the final path until every rank has finished its
        # per-rank RNG/data files and collective state-dict work.
        barrier()
        self._run_rank0_portable_step(
            lambda: _ckpt_transaction.publish(staging_path, ckpt_path),
            "publish checkpoint",
        )
        _ckpt_io.finalize_save()
        self.log(f"Saved checkpoint to {ckpt_path}")

    def _begin_checkpoint_transaction(self, ckpt_path: str) -> str:
        """Create a rank-0 staging directory and share its path."""
        staging_path = None

        def _begin():
            nonlocal staging_path
            staging_path = _ckpt_transaction.begin(ckpt_path)

        self._run_rank0_portable_step(_begin, "begin checkpoint transaction")
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            payload = [staging_path]
            torch.distributed.broadcast_object_list(payload, src=0)
            staging_path = payload[0]
        if not staging_path:
            raise RuntimeError("Rank 0 did not provide a checkpoint staging path")
        return str(staging_path)

    def _run_rank0_portable_step(self, operation, description: str) -> None:
        """Run filesystem work on rank 0 and propagate failure to every rank."""
        error = None
        if is_main_process():
            try:
                operation()
            except Exception as exc:  # broadcast the message before raising
                error = f"{type(exc).__name__}: {exc}"
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            payload = [error]
            torch.distributed.broadcast_object_list(payload, src=0)
            error = payload[0]
        if error is not None:
            raise RuntimeError(f"Failed to {description}: {error}")

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
            resume_config={
                "ckpt": self.cfg.checkpoint.resume_from,
                "try_harder": self.cfg.checkpoint.try_harder,
                "resume_optimizer_state": (
                    self.cfg.checkpoint.resume_optimizer_state
                ),
            },
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
