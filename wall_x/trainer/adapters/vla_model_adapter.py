import os

import torch
import torch.distributed as dist
from tqdm import tqdm

from wall_x.trainer.adapters.base_adapter import ModelAdapter
from wall_x.trainer.optimizer.dmuon import is_dmuon_model
from wall_x.trainer.trainer_utils import (
    compute_action_metrics,
    load_qwen_pretrain_weight,
    load_wallx_processors_from_cfg,
    save_text_results_to_file,
)


class VLAdapter(ModelAdapter):
    """Base adapter for public Qwen2.5 VLA models.

    Concrete subclasses define the model/config classes, FSDP wrap layers,
    and optional task-specific logging hooks.
    """

    #: Subclass must set this. Identifies the variant in ``ADAPTER_REGISTRY``.
    MODEL_TYPE: str = ""

    def __init__(self, *, cfg=None, logger=None, model_type=None):
        # Concrete subclasses set MODEL_TYPE; tests may pass model_type explicitly
        # (e.g. to swap a stub class into a different slot).
        resolved = model_type or self.MODEL_TYPE
        super().__init__(cfg=cfg, logger=logger, model_type=resolved)
        if not resolved:
            raise ValueError(
                f"{type(self).__name__} requires a MODEL_TYPE class attribute "
                "or an explicit model_type kwarg; got neither."
            )
        self.model_type = resolved
        self._dmuon_module = None
        self._dmuon_import_checked = False
        self._dmuon_named_param_cache = None

    def _get_dmuon_module(self):
        if not self._dmuon_import_checked:
            try:
                import dmuon
            except ImportError:
                dmuon = None
            self._dmuon_module = dmuon
            self._dmuon_import_checked = True
        return self._dmuon_module

    def _get_named_dmuon_dedicated_params(self, model):
        cache = self._dmuon_named_param_cache
        model_id = id(model)
        if cache is not None and cache[0] == model_id:
            return cache[1]

        dmuon = self._get_dmuon_module()
        if dmuon is None:
            named_params = []
        else:
            module_to_name = {
                id(module): module_name for module_name, module in model.named_modules()
            }
            named_params = []
            for dparam in dmuon.get_dedicated_params(model):
                prefix = module_to_name.get(id(dparam.module), "")
                name = f"{prefix}.{dparam.param_name}" if prefix else dparam.param_name
                named_params.append((name, dparam))

        self._dmuon_named_param_cache = (model_id, named_params)
        return named_params

    # ---- variant hooks (subclass overrides) ----
    @classmethod
    def model_class(cls):
        """Return the training-side model class for this variant."""
        raise NotImplementedError(f"{cls.__name__} must override model_class()")

    @classmethod
    def config_class(cls):
        """Return the HF PretrainedConfig class for this variant."""
        raise NotImplementedError(f"{cls.__name__} must override config_class()")

    @classmethod
    def inference_model_class(cls):
        """Return the model class to use for inference."""
        return cls.model_class()

    # ---- helpers ----
    def _get_model_and_config_class(self):
        """Resolve (ModelClass, ConfigClass) for this variant via classmethods."""
        cls = type(self)
        return cls.model_class(), cls.config_class()

    # ---- processor ----
    def _build_processor_dict(self) -> dict:
        """Flat-dict shape expected by legacy ``load_wallx_processors`` /
        ``update_model_config`` / prediction loops.

        Derived from typed TrainConfig. Kept as the one-place adapter layer
        between typed configs and legacy dict-based APIs; drop me when those
        downstream APIs are typed-ified.
        """
        import dataclasses

        flat = dataclasses.asdict(self.cfg.model)
        flat["model_type"] = self.cfg.model_type
        flat["data"] = dict(self.cfg._raw_data or {})
        flat["dof_config"] = self.cfg.task.dof_config
        flat["agent_pos_config"] = self.cfg.task.agent_pos_config
        if self.cfg.task.ar_dof_config is not None:
            flat["ar_dof_config"] = self.cfg.task.ar_dof_config
        flat["batch_size_per_gpu"] = self.cfg.hyperparams.batch_size_per_gpu
        return flat

    def load_processor(self, action_statistic_dof):
        processors_dict = load_wallx_processors_from_cfg(
            self.cfg,
            normalizer=getattr(self, "normalizer_action", None),
            action_statistic_dof=action_statistic_dof,
        )
        self.logger.info(
            f"processor vocab size: {len(processors_dict['processor'].tokenizer.vocab)}"
        )
        self.logger.info(
            f"num added tokens to processor: {processors_dict['num_added_tokens']}"
        )
        return {
            "processor": processors_dict["processor"],
            "data_config": self._build_processor_dict(),
            "tokenizer_mixin": processors_dict.get("tokenizer_mixin"),
            "train_action_tokenizer": processors_dict["train_action_tokenizer"],
            "val_action_tokenizer": processors_dict["val_action_tokenizer"],
            "action_mapper": processors_dict["action_mapper"],
            "num_added_tokens": processors_dict["num_added_tokens"],
        }

    # ---- model ----
    def build_model_config(self):
        _, ConfigClass = self._get_model_and_config_class()
        qwen_vl_act_config_path = self.cfg.model.config_path
        if qwen_vl_act_config_path.endswith(".json"):
            model_config = ConfigClass.from_json_file(qwen_vl_act_config_path)
        else:
            model_config = ConfigClass.from_pretrained(qwen_vl_act_config_path)

        assert self.model_type in model_config.model_type, (
            f"Mismatch of model type: model type in config file is "
            f"{model_config.model_type}, but the model type is {self.model_type}."
        )

        model_config.update_model_config(self._build_processor_dict())
        return model_config

    def create_model(self, processor, tokenizer_mixin, model_config):
        ModelClass, _ = self._get_model_and_config_class()
        use_selective_recompute = self.cfg.distributed.use_selective_recompute
        return ModelClass(
            model_config,
            processor,
            tokenizer_mixin=tokenizer_mixin,
            use_selective_recompute=use_selective_recompute,
        )

    def load_weights(self, model, normalizer_action, normalizer_propri, **kwargs):
        import copy

        processor = kwargs.get("processor")

        if self.cfg.model.pretrained_path:
            model, err = load_qwen_pretrain_weight(
                model, self.cfg.model.pretrained_path
            )

        if processor is not None:
            model.resize_token_embeddings(len(processor.tokenizer))

        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        elif (
            hasattr(model, "get_input_embeddings")
            and model.get_input_embeddings() is not None
        ):

            def _make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(
                _make_inputs_require_grad
            )

        if hasattr(model, "set_normalizer"):
            model.set_normalizer(
                copy.deepcopy(normalizer_action),
                copy.deepcopy(normalizer_propri),
            )

        return model

    # ---- FSDP wrapping ----
    def get_transformer_layer_cls(self):
        """Return the FSDP transformer-wrap layer classes for this variant.

        Subclass override - see the per-variant adapter for the actual
        layer classes.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override get_transformer_layer_cls()"
        )

    # ---- dataset ----
    def load_dataset(self, data_config, processor, rank, world_size, **kwargs):
        """Dispatch to the backend named by ``cfg.data.dataset_type``.

        The backend-specific wiring (resume indices, pool offsets,
        processor wrapping) lives inside each backend's ``build()``;
        this method only assembles a :class:`BuildContext` and forwards.
        """
        import copy

        from wall_x.data import BuildContext, build_data

        resume_state = None
        resume_batches = kwargs.get("resume_batches", 0)
        indices = kwargs.get("resume_indices")
        if indices is None and self.cfg.checkpoint.resume_from:
            checkpoint_path = self.cfg.checkpoint.resume_from
            if os.path.isfile(checkpoint_path):
                checkpoint_path = os.path.dirname(checkpoint_path)
            if os.path.isdir(checkpoint_path):
                indices = self._load_episode_indices(checkpoint_path, rank)

        if resume_batches or indices or kwargs.get("episode_container_checkpoint_path"):
            resume_state = {
                "resume_batches": resume_batches,
                "indices": indices,
                "episode_container_checkpoint_path": kwargs.get(
                    "episode_container_checkpoint_path"
                ),
            }

        ctx = BuildContext(
            rank=rank,
            world_size=world_size,
            processor=processor,
            tokenizer_mixin=kwargs.get("tokenizer_mixin"),
            normalizer_action=copy.deepcopy(kwargs.get("normalizer_action")),
            normalizer_propri=copy.deepcopy(kwargs.get("normalizer_propri")),
            model_config=kwargs.get("model_config"),
            resume_state=resume_state,
        )

        bundle = build_data(self.cfg, ctx)
        return bundle.dataset, bundle.train_loader, bundle.train_iters

    # ---- forward / loss ----
    def forward(self, model, batch, **kwargs):
        self._last_batch_info = self._extract_batch_info(batch)
        mode = kwargs.get("mode", "train")
        return model(**batch, mode=mode)

    def extract_loss(self, outputs):
        return outputs.loss

    def collect_output_stats(
        self,
        outputs,
        step_stats,
        reduce_tensor_fn,
        true_gather_fn,
        tokenizer_mixin=None,
    ):
        """Collect VL-specific auxiliary losses and accuracy metrics.

        Handles:
          - cross_entropy_loss / flow_loss  (all-reduce + true_gather)
          - per-dataset channel losses      (all-reduce via count-weighted avg)
          - action_accuracy and extra RVQ layer accuracies
        """
        import torch.distributed as dist

        cross_entropy_loss = self.get_output_field(outputs, "cross_entropy_loss")
        if cross_entropy_loss is not None:
            step_stats["cross_entropy_loss"] = reduce_tensor_fn(
                cross_entropy_loss
            ).item()

        flow_loss = self.get_output_field(outputs, "flow_loss")
        if flow_loss is not None:
            step_stats["flow_loss"] = reduce_tensor_fn(flow_loss).item()

        log_ce_loss = true_gather_fn(
            self.get_output_field(outputs, "cross_entropy_loss")
        )
        if log_ce_loss is not None:
            step_stats["cross_entropy_loss"] = log_ce_loss

        log_flow_loss = true_gather_fn(self.get_output_field(outputs, "flow_loss"))
        if log_flow_loss is not None:
            step_stats["flow_loss"] = log_flow_loss

        channel_loss_dict = self.get_output_field(outputs, "channel_loss_dict")
        channel_loss_count_dict = self.get_output_field(
            outputs, "channel_loss_count_dict"
        )
        if channel_loss_dict is not None:
            for dataset_name_i in channel_loss_dict:
                count_tensor = channel_loss_count_dict[dataset_name_i].clone()
                loss_tensor = channel_loss_dict[dataset_name_i].detach().clone()

                dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

                cout_sum = count_tensor.item()
                if cout_sum >= 0.5:
                    step_stats[f"channel_loss_{dataset_name_i}"] = (
                        loss_tensor.item() / cout_sum
                    )

            if "action_accuracy" in channel_loss_dict and tokenizer_mixin is not None:
                acc_tensor = channel_loss_dict["action_accuracy"].detach().clone()
                dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)
                world_size = dist.get_world_size() if dist.is_initialized() else 1
                step_stats["action_accuracy"] = acc_tensor.item() / world_size

                for key in tokenizer_mixin.get_accuracy_keys():
                    if key != "action_accuracy" and key in channel_loss_dict:
                        rvq_acc_tensor = channel_loss_dict[key].detach().clone()
                        dist.all_reduce(rvq_acc_tensor, op=dist.ReduceOp.SUM)
                        step_stats[key] = rvq_acc_tensor.item() / world_size

    def collect_param_norms(
        self, model, step_stats, device, reduce_tensor_fn, params_sharded=False
    ):
        """Compute per-group parameter L2 norms for VL models.

        Groups:
          "visual"  -> visual_param_norm
          "action"  -> action_expert_param_norm
          others    -> org_vlm_param_norm
          all       -> total_param_norm
        """
        total_sq = torch.tensor(0.0, dtype=torch.float32, device=device)
        visual_sq = torch.tensor(0.0, dtype=torch.float32, device=device)
        action_sq = torch.tensor(0.0, dtype=torch.float32, device=device)
        org_vlm_sq = torch.tensor(0.0, dtype=torch.float32, device=device)
        dedicated_sq = torch.tensor(0.0, dtype=torch.float32, device=device)
        dedicated_params = []

        def add_named_sq(name, sq):
            nonlocal total_sq, visual_sq, action_sq, org_vlm_sq
            total_sq += sq
            if "visual" in name:
                visual_sq += sq
            elif "action" in name:
                action_sq += sq
            else:
                org_vlm_sq += sq

        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if param.numel() == 0 or hasattr(param, "_dedicated_owner_rank"):
                    continue
                tensor = param.detach()
                if hasattr(tensor, "to_local"):
                    tensor = tensor.to_local()
                    if tensor.numel() == 0:
                        continue
                sq = torch.sum(tensor.float() ** 2)
                add_named_sq(name, sq)

            if is_dmuon_model(model):
                dedicated_params = self._get_named_dmuon_dedicated_params(model)
                # DMuon replaces dedicated params with placeholders in
                # named_parameters(). Count the authoritative dedicated
                # storage here so total/category norms cover the full model.
                for name, dparam in dedicated_params:
                    if getattr(dparam, "_dmuon_route", None) == "sharded_adamw":
                        replicate_group = getattr(dparam, "replicate_group", None)
                        # replicate_group=None is 1D shard-only mode: each rank
                        # contributes a distinct shard and the later reduce sums
                        # them into the global norm.
                        if (
                            replicate_group is not None
                            and replicate_group.rank()
                            != getattr(dparam, "owner_replicate", 0)
                        ):
                            continue
                        tensor = getattr(dparam, "_sharded_adamw_data", None)
                        if tensor is None:
                            continue
                        valid_numel = int(
                            getattr(
                                dparam,
                                "_sharded_adamw_valid_numel",
                                tensor.numel(),
                            )
                        )
                        tensor = tensor[:valid_numel]
                    else:
                        if not bool(getattr(dparam, "is_owner", False)):
                            continue
                        tensor = getattr(dparam, "_owned_data", None)
                        if tensor is None:
                            continue

                    if tensor.numel() == 0:
                        continue
                    sq = torch.sum(tensor.detach().float() ** 2)
                    dedicated_sq += sq
                    add_named_sq(name, sq)

        if params_sharded:
            total_sq = reduce_tensor_fn(total_sq, average=False)
            visual_sq = reduce_tensor_fn(visual_sq, average=False)
            action_sq = reduce_tensor_fn(action_sq, average=False)
            org_vlm_sq = reduce_tensor_fn(org_vlm_sq, average=False)
            dedicated_sq = reduce_tensor_fn(dedicated_sq, average=False)

        step_stats["total_param_norm"] = torch.sqrt(total_sq).item()
        step_stats["visual_param_norm"] = torch.sqrt(visual_sq).item()
        step_stats["action_expert_param_norm"] = torch.sqrt(action_sq).item()
        step_stats["org_vlm_param_norm"] = torch.sqrt(org_vlm_sq).item()
        if dedicated_params:
            step_stats["dmuon_dedicated_param_norm"] = torch.sqrt(dedicated_sq).item()

    # ---- MFU computation ----
    @staticmethod
    def _extract_batch_info(batch):
        """Extract token counts from a training batch."""
        info = {}
        input_ids = batch.get("input_ids")
        if input_ids is not None:
            info["batch_size"] = input_ids.shape[0]
            info["seq_length"] = input_ids.shape[1]

        moe_token_types = batch.get("moe_token_types")
        if moe_token_types is not None:
            info["num_lang_tokens"] = (
                int((moe_token_types == 0).sum().item()) // info["batch_size"]
            )
            info["num_action_tokens"] = (
                int((moe_token_types == 1).sum().item()) // info["batch_size"]
            )
        else:
            info["num_lang_tokens"] = info.get("seq_length", 0)
            info["num_action_tokens"] = 0

        pixel_values = batch.get("pixel_values")
        info["vision_seq_length"] = (
            pixel_values.shape[0] if pixel_values is not None else 0
        )

        labels = batch.get("labels")
        if labels is not None:
            info["num_loss_tokens"] = int((labels[..., 1:] != -100).sum().item())
        else:
            info["num_loss_tokens"] = None

        return info

    def _compute_detailed_flops(self, model, config):
        """Compute detailed per-module FLOPs for one training step.

        Uses the same formulas as scripts/profile_forward.py:compute_module_flops,
        with fwd+bwd multipliers:
          - Frozen modules (e.g. ViT if frozen): 1x forward
          - Trainable modules: 3x forward (fwd + 2x bwd)
        """

        batch_info = getattr(self, "_last_batch_info", None)
        if not batch_info:
            return None

        model_config = model.config if hasattr(model, "config") else None
        if model_config is None:
            return None

        B = batch_info["batch_size"]
        S = batch_info["seq_length"]
        num_lang = batch_info["num_lang_tokens"]
        num_act = batch_info["num_action_tokens"]
        N_lang = B * num_lang
        N_act = B * num_act
        N_total = N_lang + N_act
        Nv = batch_info["vision_seq_length"]
        num_loss_tokens = batch_info["num_loss_tokens"]

        H = model_config.hidden_size
        num_layers = model_config.num_hidden_layers
        num_heads = model_config.num_attention_heads
        num_kv = model_config.num_key_value_heads
        vocab_size = getattr(model_config, "padded_vocab_size", model_config.vocab_size)

        use_mot = getattr(model_config, "attention_moe", False)
        use_moe_mlp = getattr(model_config, "mlp_moe", False)
        dim_inputs = getattr(model_config, "dim_inputs", (H, H))
        dim_lang, dim_act = dim_inputs

        grad_accum = self.cfg.hyperparams.gradient_accumulation_steps

        vit_frozen = True
        if hasattr(model, "visual"):
            for p in model.visual.parameters():
                if p.requires_grad:
                    vit_frozen = False
                    break
        vit_mult = 1.0 if vit_frozen else 3.0

        vit_fwd_flops = 0
        if hasattr(model_config, "vision_config") and Nv > 0:
            vcfg = model_config.vision_config
            Hv = vcfg.hidden_size
            Iv = vcfg.intermediate_size
            out_hidden = vcfg.out_hidden_size
            depth_v = vcfg.depth

            F_qkv_v = 6 * Nv * Hv * Hv
            F_o_v = 2 * Nv * Hv * Hv
            F_mlp_v = 6 * Nv * Hv * Iv + 2 * Nv * Iv
            F_linear_per_layer = F_qkv_v + F_o_v + F_mlp_v

            fullatt_set = set(getattr(vcfg, "fullatt_block_indexes", []))
            num_images = max(Nv // 768, 1)
            si = Nv // num_images
            sum_si_sq = num_images * si * si
            sum_wi_sq = sum_si_sq // 16

            for i in range(depth_v):
                if i in fullatt_set:
                    vit_fwd_flops += F_linear_per_layer + 4 * Hv * sum_si_sq
                else:
                    vit_fwd_flops += F_linear_per_layer + 4 * Hv * sum_wi_sq

            merge_unit = getattr(vcfg, "spatial_merge_size", 2) ** 2
            merger_hidden = Hv * merge_unit
            Nv_merged = Nv // merge_unit
            vit_fwd_flops += (
                2 * Nv_merged * merger_hidden * merger_hidden
                + 2 * Nv_merged * merger_hidden * out_hidden
            )

        kv_ratio = num_kv / num_heads
        F_matmul_fwd = 4 * B * (S**2) * H

        if not use_mot:
            F_attn_fwd = (
                2 * N_total * H * H
                + 4 * N_total * H * H * kv_ratio
                + 2 * N_total * H * H
                + F_matmul_fwd
            )
        else:
            F_attn_fwd = (
                N_lang * dim_lang * H * (4 + 4 * kv_ratio)
                + N_act * dim_act * H * (4 + 4 * kv_ratio)
                + F_matmul_fwd
            )

        if not use_moe_mlp:
            ffn_hidden = model_config.intermediate_size
            F_mlp_fwd = 6 * N_total * H * ffn_hidden + 2 * N_total * ffn_hidden
        else:
            hid_lang = model_config.experts[0]["intermediate_size"]
            hid_act = model_config.experts[1]["intermediate_size"]
            F_mlp_fwd = (6 * N_lang * dim_lang * hid_lang + 2 * N_lang * hid_lang) + (
                6 * N_act * dim_act * hid_act + 2 * N_act * hid_act
            )

        decoder_fwd_flops = num_layers * (F_attn_fwd + F_mlp_fwd)

        N_lm = num_loss_tokens if num_loss_tokens is not None else N_total
        lm_head_fwd_flops = 2 * N_lm * H * vocab_size

        total_flops = (
            vit_mult * vit_fwd_flops + 3.0 * decoder_fwd_flops + 3.0 * lm_head_fwd_flops
        ) * grad_accum

        return {
            "total_flops": total_flops,
            "vit_fwd_flops": vit_fwd_flops,
            "decoder_fwd_flops": decoder_fwd_flops,
            "lm_head_fwd_flops": lm_head_fwd_flops,
            "vit_mult": vit_mult,
        }

    def compute_mfu(self, model, step_time_seconds):
        """Compute detailed Model FLOPs Utilization for VL transformer training."""
        if step_time_seconds <= 0:
            return None
        try:
            config = self._build_processor_dict()
            flops_info = self._compute_detailed_flops(model, config)
            if flops_info is None:
                return None

            gpu_peak_tflops = 312.0  # TODO: move to DebugConfig if needed
            peak_flops = gpu_peak_tflops * 1e12

            # Data-parallel: each GPU independently processes its own micro-batch,
            # so per-GPU MFU = per_gpu_flops / (step_time * per_gpu_peak).
            # No division by num_gpus needed.
            total_flops = flops_info["total_flops"]
            mfu = total_flops / (step_time_seconds * peak_flops)

            return {
                "mfu": mfu,
                "flops_per_step_T": total_flops / 1e12,
            }
        except Exception:
            return None

    # ---- optional hooks ----
    @staticmethod
    def log_attention_implementation(logger, model):
        """Log the attention implementation name. Variant-specific because the
        model layout (model.model vs model.model.language_model) differs."""
        raise NotImplementedError(
            "VLAdapter subclasses must override log_attention_implementation()"
        )

    def _load_episode_indices(self, checkpoint_path: str, rank: int):
        """Read backend-managed per-rank resume offsets if supported."""
        from wall_x.data import data_backend

        backend = data_backend()
        if not backend.supports("load_episode_indices"):
            return None
        return backend.load_episode_indices(checkpoint_path, rank)

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
        """Dispatch VLA inference by prediction_type.

        flow_action / dllm_action return per-rank L1 action metrics; ar_action
        is reserved (was unimplemented pre-refactor too); text runs text
        generation with optional point-L1 distance scoring.
        """
        config = self._build_processor_dict()
        if prediction_type == "flow_action":
            return self._predict_flow_action(
                model=model,
                val_dataloader=val_dataloader,
                rank=rank,
                world_size=world_size,
                device=device,
                config=config,
                logger=logger,
                current_step=current_step,
                max_iteration=max_iteration,
            )
        if prediction_type == "ar_action":
            return None  # matches pre-refactor behaviour (was pass)
        if prediction_type == "dllm_action":
            return self._predict_dllm_action(
                model=model,
                val_dataloader=val_dataloader,
                rank=rank,
                world_size=world_size,
                device=device,
                config=config,
                processor=processor,
                tokenizer_mixin=tokenizer_mixin,
                logger=logger,
                current_step=current_step,
                max_iteration=max_iteration,
            )
        if prediction_type == "text":
            return self._predict_text(
                model=model,
                val_dataloader=val_dataloader,
                rank=rank,
                device=device,
                logger=logger,
                current_step=current_step,
                max_samples=max_samples,
                save_dir=save_dir,
            )
        raise ValueError(f"Unsupported prediction type: {prediction_type}")

    @torch.no_grad()
    def _predict_flow_action(
        self,
        *,
        model,
        val_dataloader,
        rank,
        world_size,
        device,
        config,
        logger,
        current_step,
        max_iteration,
    ):
        if dist.is_initialized():
            dist.barrier()

        total_num = len(val_dataloader)
        if max_iteration:
            total_num = min(max_iteration, total_num)
        model.eval()

        all_preds, all_actions = [], []
        pepoch = tqdm(
            total=total_num,
            desc=f"Predicting ckpt at step {current_step}",
            disable=rank != 0,
        )
        for batch_idx, batch in enumerate(val_dataloader):
            if batch_idx >= total_num:
                break
            batch = _move_batch(batch, device)

            model_output = model.generate_flow_action(
                action_horizon=config["data"]["action_horizon_flow"],
                action_dim=model.action_preprocessor.action_dim,
                **batch,
            )
            pred_action, gt_action = (
                model_output["predict_action"],
                model_output["gt_action"],
            )

            pred_list = [torch.zeros_like(pred_action) for _ in range(world_size)]
            gt_list = [torch.zeros_like(gt_action) for _ in range(world_size)]
            dist.all_gather(pred_list, pred_action)
            dist.all_gather(gt_list, gt_action)

            if dist.is_initialized():
                dist.barrier()
            if rank == 0:
                all_preds.append(torch.cat(pred_list, dim=0).cpu())
                all_actions.append(torch.cat(gt_list, dim=0).cpu())
                pepoch.update(1)

        pepoch.close()

        if rank == 0:
            step_log = {}
            if all_preds:
                all_preds = torch.cat(all_preds, dim=0)
                all_actions = torch.cat(all_actions, dim=0)
                step_log = compute_action_metrics(
                    all_preds, all_actions, config=config, step_log=step_log
                )
            if logger and step_log:
                logger.log(step_log, step=current_step)
            if self.logger is not None:
                self.logger.info(
                    f"Step {current_step}, Validation L1 Loss: {step_log.get('val_action_l1', -1)}"
                )
            return step_log
        return None

    @torch.no_grad()
    def _predict_dllm_action(
        self,
        *,
        model,
        val_dataloader,
        rank,
        world_size,
        device,
        config,
        processor,
        tokenizer_mixin,
        logger,
        current_step,
        max_iteration,
    ):
        if dist.is_initialized():
            dist.barrier()

        total_num = len(val_dataloader)
        if max_iteration:
            total_num = min(max_iteration, total_num)
        model.eval()

        all_preds, all_actions = [], []
        pepoch = tqdm(
            total=total_num,
            desc=f"Predicting ckpt at step {current_step}",
            disable=rank != 0,
        )
        for batch_idx, batch in enumerate(val_dataloader):
            if batch_idx >= total_num:
                break
            batch = _move_batch(batch, device)
            batch = self._preprocess_dllm_batch(batch, processor, tokenizer_mixin)

            if hasattr(model.action_tokenizer, "max_waypoints"):
                total_ar_step = model.action_tokenizer.max_waypoints
            elif hasattr(model.action_tokenizer, "max_length"):
                total_ar_step = model.action_tokenizer.max_length
            else:
                raise ValueError(
                    "Unknown action_tokenizer type for dllm action inference"
                )
            model_output = model.generate_dllm_action(
                action_dim=7,
                action_horizon=config["data"]["action_horizon_flow"],
                use_ar_action=False,
                total_ar_step=total_ar_step,
                **batch,
            )
            pred_action, gt_action = (
                model_output["predict_action"],
                model_output["gt_action"],
            )

            pred_list = [torch.zeros_like(pred_action) for _ in range(world_size)]
            gt_list = [torch.zeros_like(gt_action) for _ in range(world_size)]
            dist.all_gather(pred_list, pred_action)
            dist.all_gather(gt_list, gt_action)

            if dist.is_initialized():
                dist.barrier()
            if rank == 0:
                all_preds.append(torch.cat(pred_list, dim=0).cpu())
                all_actions.append(torch.cat(gt_list, dim=0).cpu())
                pepoch.update(1)

        pepoch.close()

        if rank == 0:
            step_log = {}
            if all_preds:
                all_preds = torch.cat(all_preds, dim=0)
                all_actions = torch.cat(all_actions, dim=0)
                step_log = compute_action_metrics(
                    all_preds, all_actions, config=config, step_log=step_log
                )
            if logger and step_log:
                logger.log(step_log, step=current_step)
            if self.logger is not None:
                self.logger.info(
                    f"Step {current_step}, Validation L1 Loss: {step_log.get('val_action_l1', -1)}"
                )
            return step_log
        return None

    @staticmethod
    def _preprocess_dllm_batch(batch, processor, tokenizer_mixin):
        """Patch the dllm action placeholders into input_ids (in place)."""
        input_ids = batch["input_ids"]
        prefix_length = batch["prefix_length"]
        placeholder_str = tokenizer_mixin.get_placeholder_for_dllm()
        placeholder_seq = torch.tensor(
            processor.tokenizer.convert_tokens_to_ids(placeholder_str)
        )
        placeholder_len = placeholder_seq.shape[0]
        input_ids[:, prefix_length - placeholder_len - 2 : prefix_length - 2] = (
            placeholder_seq
        )
        batch.update({"input_ids": input_ids})
        return batch

    @torch.no_grad()
    def _predict_text(
        self,
        *,
        model,
        val_dataloader,
        rank,
        device,
        logger,
        current_step,
        max_samples,
        save_dir,
    ):
        from wall_x._vendor.x2robot_utils.grounding import calculate_point_l1_distance

        if dist.is_initialized():
            dist.barrier()

        total_num = (
            len(val_dataloader)
            if max_samples is None
            else min(max_samples, len(val_dataloader))
        )
        model.eval()

        all_input_texts, all_gt_texts, all_pred_texts = [], [], []
        all_point_l1_distances = []

        pepoch = tqdm(
            total=total_num,
            desc=f"Predicting text at step {current_step}",
            disable=rank != 0,
        )

        for batch_idx, batch in enumerate(val_dataloader):
            if batch_idx >= total_num:
                break
            batch = _move_batch(batch, device)

            model_output = model.generate_text(
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
                moe_token_types=batch.get("moe_token_types"),
                pixel_values=batch.get("pixel_values"),
                image_grid_thw=batch.get("image_grid_thw"),
                proprioception=batch.get("proprioception"),
                dataset_names=batch.get("dataset_names"),
                dof_mask=batch.get("dof_mask"),
                agent_pos_mask=batch.get("agent_pos_mask"),
                prefix_length=batch.get("prefix_length"),
                positional_masks=batch.get("positional_masks"),
                re_generate=False,
            )

            input_texts = model_output["input_text"]
            gt_texts = model_output["gt_output_text"]
            pred_texts = model_output["predict_output_text"]

            if rank == 0:
                all_input_texts.extend(input_texts)
                all_gt_texts.extend(gt_texts)
                all_pred_texts.extend(pred_texts)
                for gt_text, pred_text in zip(gt_texts, pred_texts):
                    gt_clean = gt_text[0] if isinstance(gt_text, list) else gt_text
                    pred_clean = (
                        pred_text[0] if isinstance(pred_text, list) else pred_text
                    )
                    point_l1_dist = calculate_point_l1_distance(gt_clean, pred_clean)
                    if point_l1_dist is not None:
                        all_point_l1_distances.append(point_l1_dist)

            pepoch.update(1)

        pepoch.close()

        if rank == 0 and all_pred_texts:
            if all_point_l1_distances:
                avg_point_l1 = sum(all_point_l1_distances) / len(all_point_l1_distances)
                min_point_l1 = min(all_point_l1_distances)
                max_point_l1 = max(all_point_l1_distances)
                point_stats = {
                    "text_prediction_point_l1_avg": avg_point_l1,
                    "text_prediction_point_l1_min": min_point_l1,
                    "text_prediction_point_l1_max": max_point_l1,
                    "text_prediction_point_samples_count": len(all_point_l1_distances),
                }
                if logger is not None:
                    logger.log(point_stats, step=current_step)
                if self.logger is not None:
                    self.logger.info(
                        f"Point L1 Statistics - Avg: {avg_point_l1:.4f}, "
                        f"Min: {min_point_l1:.4f}, Max: {max_point_l1:.4f}, "
                        f"Samples: {len(all_point_l1_distances)}"
                    )
            save_text_results_to_file(
                current_step,
                all_input_texts,
                all_gt_texts,
                all_pred_texts,
                save_dir if save_dir is not None else "./",
            )


def _move_batch(batch, device):
    """Recursively move dict/list/tensor to device (VLA-local copy to keep
    VLAdapter self-contained; base_adapter has a similar helper used by
    validation)."""
    if isinstance(batch, dict):
        return {k: _move_batch(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [_move_batch(v, device) for v in batch]
    if torch.is_tensor(batch):
        return batch.to(device)
    return batch
