import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict

import torch
import torch.distributed as dist
from tqdm import tqdm

from wall_x.trainer.utils import move_batch_to_device

logger = logging.getLogger(__name__)


def load_trainer_data_config(cfg):
    """Return optional backend-specific trainer data config.

    Public backends build datasets directly from TrainConfig. A backend that
    needs an additional trainer data config may expose
    ``load_trainer_data_config(cfg)``.
    """
    from wall_x.data import data_backend

    backend = data_backend()
    if backend.supports("load_trainer_data_config"):
        return backend.load_trainer_data_config(cfg)
    return None


class ModelAdapter(ABC):
    """Pure strategy that encapsulates model-type-specific behavior.

    The adapter is stateless w.r.t. the trainer: every method declares
    its inputs as parameters and communicates results via return values.
    The trainer orchestrates calls and manages its own state.

    Abstract methods (subclasses MUST implement):
        load_processor          -- load processor / tokenizer artifacts
        build_model_config      -- build model-specific config object
        create_model            -- instantiate the model
        load_weights            -- load pretrained weights & set normalizers
        get_transformer_layer_cls -- layer classes for FSDP auto-wrap
        load_dataset            -- build dataset & training dataloader
        forward                 -- run a forward pass
        extract_loss            -- extract scalar loss from model outputs

    Concrete methods with default behaviour (override when needed):
        load_step_and_epoch     -- restore global_step / epoch from checkpoint
        log_model_info          -- log model-specific info (no-op)
        collect_output_stats    -- collect auxiliary loss / accuracy stats (no-op)
        collect_param_norms     -- compute per-group param L2 norms (no-op)
        get_fwd_flops           -- optional forward FLOPs hook
        get_output_field        -- retrieve a named field from outputs (static)
    """

    def __init__(self, *, cfg=None, logger=None, model_type=None):
        self.cfg = cfg
        self.logger = logger

    # ---- processor ----
    @abstractmethod
    def load_processor(self, action_statistic_dof):
        """Load processor and associated tokenizer artifacts.

        Reads model/data config from ``self.cfg``.

        Args:
            action_statistic_dof: normalizer statistics dict.

        Returns:
            dict with keys:
                "processor", "data_config", "tokenizer_mixin",
                "train_action_tokenizer", "val_action_tokenizer",
                "action_mapper", "num_added_tokens"
        """
        ...

    # ---- model ----
    @abstractmethod
    def create_model(self, processor, tokenizer_mixin, model_config):
        """Instantiate the model (before weight loading / FSDP wrap).

        Reads model config from ``self.cfg``.
        """
        ...

    @abstractmethod
    def build_model_config(self):
        """Build the model-specific config object from ``self.cfg.model``."""
        ...

    @abstractmethod
    def load_weights(self, model, normalizer_action, normalizer_propri, **kwargs):
        """Load pretrained weights, resize embeddings, set normalizers, etc."""
        ...

    # ---- FSDP wrapping ----
    @abstractmethod
    def get_transformer_layer_cls(self):
        """Return layer classes for FSDP transformer_auto_wrap_policy."""
        ...

    # ---- dataset ----
    @abstractmethod
    def load_dataset(self, data_config, processor, rank, world_size, **kwargs):
        """Load dataset and build the training dataloader.

        Args:
            data_config: parsed data config from load_processor().
            processor:   processor instance.
            rank:        current process rank.
            world_size:  total number of processes.
            **kwargs:    extra model-specific args.

        Returns:
            (dataset, train_dataloader, train_num) tuple.
        """
        ...

    def load_step_and_epoch(self, checkpoint_path: str, is_incomplete_epoch: bool):
        global_step = 0
        start_epoch = 0

        global_step_path = os.path.join(checkpoint_path, "global_step.pth")
        if os.path.exists(global_step_path):
            global_step = torch.load(global_step_path)["global_step"]

        current_epoch_path = os.path.join(checkpoint_path, "current_epoch.pth")
        if os.path.exists(current_epoch_path):
            start_epoch = torch.load(current_epoch_path)["current_epoch"]
            if not is_incomplete_epoch:
                start_epoch = start_epoch + 1

        return {"global_step": global_step, "start_epoch": start_epoch}

    # ---- forward / loss ----
    @abstractmethod
    def forward(self, model, batch, **kwargs):
        """Run a forward pass.

        Args:
            model: the (possibly wrapped) model.
            batch: dict of tensors already on device.
            **kwargs: extra context (e.g. global_step, mode).

        Returns:
            Raw model outputs (dict or object).
        """
        ...

    @abstractmethod
    def extract_loss(self, outputs):
        """Extract the scalar training loss from model outputs.

        Returns:
            torch.Tensor (scalar).
        """
        ...

    # ---- optional hooks (override when needed) ----
    def log_model_info(self, model, log_fn):
        """Log model-specific info (e.g. attention implementation).

        Args:
            model:  the unwrapped model.
            log_fn: callable(str) for logging.
        """
        pass

    def collect_output_stats(
        self,
        outputs,
        step_stats,
        reduce_tensor_fn,
        true_gather_fn,
        tokenizer_mixin=None,
    ):
        """Collect model-output-specific statistics into step_stats.

        Extracts auxiliary losses (cross_entropy_loss, flow_loss),
        per-dataset channel losses, and action accuracy metrics from
        the model outputs and merges them into *step_stats* in-place.

        This is an optional hook - subclasses override it when their
        output format carries extra fields.  The default implementation
        is a no-op.

        Args:
            outputs:          raw model outputs (dict or object).
            step_stats:       dict being built by the trainer; mutated in-place.
            reduce_tensor_fn: callable(tensor, average=True) -> scalar tensor,
                              wraps dist.all_reduce for the current parallelism.
            true_gather_fn:   callable(tensor) -> scalar | None,
                              gathers a value to rank-0 (returns None on
                              non-main ranks or when input is None).
            tokenizer_mixin:  optional ActionTokenizerMixin used to retrieve
                              extra accuracy keys (e.g. RVQ layer accuracy).
        """
        pass

    def collect_param_norms(
        self, model, step_stats, device, reduce_tensor_fn, params_sharded=False
    ):
        """Compute per-group parameter L2 norms and merge into step_stats.

        Groups parameters by name keyword:
          - "visual"  -> visual_param_norm
          - "action"  -> action_expert_param_norm
          - (others)  -> org_vlm_param_norm
          - all       -> total_param_norm

        When parameters are sharded across ranks (FSDP FULL_SHARD /
        HYBRID_SHARD), the local squared-sum is partial and must be
        all_reduce'd (sum, not avg) before taking the square root.
        When parameters are replicated (DDP, or FSDP with SHARD_GRAD_OP /
        NO_SHARD), all_reduce would inflate the result by world_size.

        This is an optional hook - the default implementation is a no-op.

        Args:
            model:            the (possibly FSDP/DDP-wrapped) model.
            step_stats:       dict being built by the trainer; mutated in-place.
            device:           torch.device for tensor allocation.
            reduce_tensor_fn: callable(tensor, average=False) -> tensor,
                              wraps dist.all_reduce for the current parallelism.
            params_sharded:   True when parameters are actually sharded across
                              ranks (FSDP FULL_SHARD / HYBRID_SHARD).
        """
        pass

    def collect_grad_norms(
        self, model, step_stats, device, reduce_tensor_fn, params_sharded=False
    ):
        """Compute per-component gradient L2 norms and merge into step_stats.

        Must be called while gradients are still available (before
        optimizer.zero_grad).  Under FSDP each rank holds a gradient shard,
        so local squared-sums must be all_reduce'd (sum, not avg) before
        taking the square root.

        Default is a no-op; subclasses override for model-specific groupings.
        """
        pass

    def get_fwd_flops(self, autocast_context, model, batch, global_step):
        """Forward FLOPs profiling is disabled in the public package."""
        return None

    @staticmethod
    def get_output_field(outputs, key, default=None):
        """Retrieve a named field from model outputs (dict or object)."""
        if isinstance(outputs, dict):
            return outputs.get(key, default)
        return getattr(outputs, key, default)

    # ---- console logging fields ----
    def console_fields(self, tokenizer_mixin=None) -> list:
        """Per-step console fields as (stat_key, pretty_label, fmt_spec) triples.

        MetricsLogger renders them in order, skipping any key not present
        in step_stats. Public adapters may override this method to add
        task-specific fields.
        """
        fields = [
            ("video_loss", "vid_loss", ".6f"),
            ("action_loss", "act_loss", ".6f"),
            ("action_accuracy", "accuracy", ".4f"),
            ("flow_loss", "flow_loss", ".6f"),
        ]
        if tokenizer_mixin is not None:
            for key in tokenizer_mixin.get_accuracy_keys():
                if key != "action_accuracy":
                    short_key = key.replace("action_accuracy_", "acc_")
                    fields.append((key, short_key, ".4f"))
        fields.extend(
            [
                ("action_grad_norm", "act_gnorm", ".4f"),
                ("video_grad_norm", "vid_gnorm", ".4f"),
            ]
        )
        return fields

    # ---- distribution hooks ----
    def convert_to_mix_precision_hint(
        self,
        model,
        *,
        device,
        use_fsdp: bool,
        log_fn=None,
    ):
        """Prepare *model* dtype + placement before FSDP/DDP wrap.

        FSDP path: cast to fp32 on device; MixedPrecision policy handles the
        bf16 compute cast at forward time.  DDP path: delegate to the model's
        ``convert_to_mix_precision`` method (manual bf16 conversion, since
        DDP has no equivalent of MixedPrecision), then move to device.

        Adapters that handle placement themselves may override this hook.
        """
        _log = log_fn or (lambda _msg: None)
        if use_fsdp:
            model.to(dtype=torch.float32)
            _log("FSDP mode: all params fp32, MixedPrecision handles bf16 compute")
            model.to(device)
        else:
            model.convert_to_mix_precision()
            _log("DDP mode: params converted to bf16 (no MixedPrecision policy)")
            model.to(device)

    # ---- validation ----
    def run_validation(
        self,
        *,
        model,
        val_dataloader,
        rank,
        world_size,
        device,
        autocast_context,
        reduce_fn,
        gather_fn,
        logger,
        global_step,
        output_path,
        tokenizer_mixin=None,
    ):
        """Default validation: forward-pass loss + adapter.collect_output_stats.

        Public VLA adapters may override this for task-specific metrics.

        ``output_path`` is accepted for interface uniformity but not used
        in this default.
        """
        del output_path
        model.eval()
        log_dict = defaultdict(float)
        pbar = tqdm(
            val_dataloader,
            desc="Validating",
            total=len(val_dataloader),
            disable=rank != 0,
        )

        for batch in pbar:
            batch = move_batch_to_device(batch, device)
            with autocast_context():
                outputs = self.forward(model, batch, global_step=0)

            loss = outputs["total_loss"] if "total_loss" in outputs else outputs["loss"]
            log_dict["val_loss"] += reduce_fn(loss.detach()).item()

            val_ce_loss = self.get_output_field(outputs, "cross_entropy_loss")
            if val_ce_loss is not None:
                log_ce_loss = gather_fn(val_ce_loss)
                if log_ce_loss is not None and rank == 0:
                    log_dict["val_cross_entropy_loss"] += log_ce_loss
            val_flow_loss = self.get_output_field(outputs, "flow_loss")
            if val_flow_loss is not None:
                log_flow_loss = gather_fn(val_flow_loss)
                if log_flow_loss is not None and rank == 0:
                    log_dict["val_flow_loss"] += log_flow_loss

            for k, v in collect_channel_loss_stats(
                outputs,
                prefix="val_",
                tokenizer_mixin=tokenizer_mixin,
            ).items():
                log_dict[k] += v

        log_dict = {k: v / len(val_dataloader) for k, v in log_dict.items()}

        if rank == 0:
            if self.logger is not None:
                self.logger.info(f"[FSDP Val] Step {global_step}: {log_dict}")
            if logger is not None and hasattr(logger, "log"):
                logger.log(log_dict, step=global_step)

            val_info = f"[Validation] step {global_step}"
            if "val_loss" in log_dict:
                val_info += f" | val_loss {log_dict['val_loss']:.4f}"
            if "val_cross_entropy_loss" in log_dict:
                val_info += f" | val_ce_loss {log_dict['val_cross_entropy_loss']:.4f}"
            if "val_flow_loss" in log_dict:
                val_info += f" | val_flow_loss {log_dict['val_flow_loss']:.6f}"
            if "val_action_accuracy" in log_dict:
                val_info += f" | val_accuracy {log_dict['val_action_accuracy']:.4f}"
            if tokenizer_mixin is not None:
                for key in tokenizer_mixin.get_accuracy_keys():
                    if key != "action_accuracy":
                        val_key = f"val_{key}"
                        if val_key in log_dict:
                            short_key = key.replace("action_accuracy_", "acc_")
                            val_info += f" | val_{short_key} {log_dict[val_key]:.4f}"
            if self.logger is not None:
                self.logger.info(val_info)

        model.train()
        return log_dict

    def init_validation(self, normalizer_action, normalizer_propri):
        """Optional pre-training-loop setup for validation. Default is a no-op."""
        del normalizer_action, normalizer_propri

    # ---- prediction / inference ----
    def predict(
        self,
        prediction_type: str,
        *,
        model,
        val_dataloader,
        rank,
        world_size,
        device,
        processor,
        tokenizer_mixin,
        logger,
        current_step,
        max_iteration=None,
        save_dir=None,
        max_samples=None,
    ):
        """Run an inference loop for the given ``prediction_type``.

        Subclasses implement the dispatch table (flow_action / ar_action /
        dllm_action / text / ...). Default raises - only VLAdapter has
        meaningful predict paths today.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support predict(prediction_type=...)"
        )

    # ---- optimizer-related defaults ----
    @property
    def default_action_lr_keywords(self) -> list[str]:
        """Keyword list used to identify action-expert parameters.

        The trainer splits parameters into two groups when
        ``action_expert_learning_rate`` is set: names containing any of these
        keywords go into the action-LR group. Subclasses override this when
        the architecture uses different naming conventions.
        """
        return [
            "action_preprocessor",
            "moe.experts.1",
            "qkv_proj_experts.1",
            "o_proj_experts.1",
            "input_layernorms.1",
            "post_attention_layernorms.1",
            "model.norms.1",
        ]


def collect_channel_loss_stats(outputs, prefix="", tokenizer_mixin=None):
    """Gather per-dataset channel losses + accuracy metrics via dist.all_reduce.

    Mirrors the legacy fsdp_trainer._collect_channel_loss_stats so training
    and validation stats come out with identical keys.  Returns {} when the
    model output dict lacks ``channel_loss_dict``.
    """
    stats = {}
    channel_loss_dict = (
        outputs.get("channel_loss_dict")
        if isinstance(outputs, dict)
        else getattr(outputs, "channel_loss_dict", None)
    )
    if channel_loss_dict is None:
        return stats
    channel_loss_count_dict = (
        outputs.get("channel_loss_count_dict")
        if isinstance(outputs, dict)
        else getattr(outputs, "channel_loss_count_dict", None)
    )
    for dataset_name_i in channel_loss_dict:
        count_tensor = channel_loss_count_dict[dataset_name_i].clone()
        loss_tensor = channel_loss_dict[dataset_name_i].detach().clone()
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        count_sum = count_tensor.item()
        if count_sum >= 0.5:
            stats[f"{prefix}channel_loss_{dataset_name_i}"] = (
                loss_tensor.item() / count_sum
            )
    if "action_accuracy" in channel_loss_dict and tokenizer_mixin is not None:
        acc_tensor = channel_loss_dict["action_accuracy"].detach().clone()
        dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        stats[f"{prefix}action_accuracy"] = acc_tensor.item() / world_size
        for key in tokenizer_mixin.get_accuracy_keys():
            if key != "action_accuracy" and key in channel_loss_dict:
                rvq_acc_tensor = channel_loss_dict[key].detach().clone()
                dist.all_reduce(rvq_acc_tensor, op=dist.ReduceOp.SUM)
                stats[f"{prefix}{key}"] = rvq_acc_tensor.item() / world_size
    return stats
