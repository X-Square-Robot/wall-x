from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from peft import LoraConfig, get_peft_model
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision as MP
from transformers import GenerationMixin
from transformers.utils import logging

from wall_x.model.core.action.moe import SparseMoeBlock, TokenTypeRouter
from wall_x.model.core.action.processor import ActionProcessor
from wall_x.model.core.attention.mask import (
    update_joint_attention_flash_mask,
    update_joint_attention_mask_2d,
    update_position_ids,
)
from wall_x.utils.constant import is_action_dataset_name

logger = logging.get_logger(__name__)


class ActionModelMixMin:
    # config: Qwen2_5_VLConfig
    action_preprocessor: ActionProcessor
    router: "TokenTypeRouter"
    moe: "SparseMoeBlock"

    def __init__(self, config, action_preprocessor, router, moe):
        self.config = config
        self.action_preprocessor = action_preprocessor
        self.router = router
        self.moe = moe
        self._mot_opt_warned = False

    def set_normalizer(self, normalizer_action, normalizer_propri):
        if hasattr(self, "action_preprocessor"):
            self.action_preprocessor.set_normalizer(
                normalizer_action, normalizer_propri
            )
        else:
            # WARNING: normalizer cannot be set when action_preprocessor is missing
            logger.warning(
                "ActionModelMixMin.set_normalizer is called but action_preprocessor is not set"
            )

    def _apply_mlp_moe(self, hidden_states, token_types, start_indices, end_indices):
        if self.config.mlp_moe:
            hidden_states = self.moe(
                hidden_states, token_types, start_indices, end_indices
            )
        else:
            hidden_states = self.mlp(hidden_states)
        return hidden_states

    def _apply_norm_moe(
        self,
        hidden_states,
        token_types,
        adarms_conds,
        norms,  # list of norm layers (expert-wise)
        norm,  # shared norm if not norm_moe
        start_indices=None,
        end_indices=None,
        use_selective_recompute=False,
    ):
        """
        MoE-aware LayerNorm with optional selective activation recomputation.

        Only activation math is recomputed. No GEMM is recomputed.
        Safe for FSDP (use_reentrant=False).
        """

        gate = None
        gate_mask = None

        # -------------------------
        # Case 1: norm_moe=True (expert-wise norm)
        # -------------------------
        if self.config.norm_moe:

            # ---------------------------------------------------------
            # Case 1A: mot_opt=True (segments assigned by start/end)
            # ---------------------------------------------------------
            if self.config.mot_opt:
                new_hidden_states = torch.zeros_like(hidden_states)

                for expert_idx, expert_norm in enumerate(norms):
                    start = start_indices[expert_idx]
                    end = end_indices[expert_idx]
                    if start == end:
                        continue

                    dim_input = self.config.dim_inputs[expert_idx]
                    selected = hidden_states[start:end]  # [K, D]

                    # ====== reshape if adarms on flow expert ======
                    if self.config.use_adarms and expert_idx == 1:
                        selected = selected.view(
                            -1,
                            self.config.action_horizon_flow,
                            selected.shape[-1],
                        )
                        input_slice = selected[:, :, :dim_input]
                        cond = adarms_conds[expert_idx]
                    else:
                        input_slice = selected[:, :dim_input]
                        cond = adarms_conds[expert_idx]

                    if use_selective_recompute:

                        def norm_chunk(t_x, t_cond, expert_norm=expert_norm):
                            if t_cond is None or (
                                isinstance(t_cond, torch.Tensor) and t_cond.numel() == 0
                            ):
                                out, _ = expert_norm(t_x)
                            else:
                                out, _ = expert_norm(t_x, t_cond)
                            return out

                        cond_for_cp = (
                            cond
                            if cond is not None
                            else torch.empty(0, device=input_slice.device)
                        )
                        processed = cp.checkpoint(
                            norm_chunk,
                            input_slice,
                            cond_for_cp,
                            use_reentrant=False,
                        )
                    else:
                        processed, gate = expert_norm(input_slice, cond)

                    # reshape back if needed
                    if self.config.use_adarms and expert_idx == 1:
                        processed = processed.view(-1, dim_input)

                    new_hidden_states[start:end, :dim_input] = processed.to(
                        hidden_states.dtype
                    )

                hidden_states = new_hidden_states

            # ---------------------------------------------------------
            # Case 1B: mot_opt=False (token-level mask)
            # ---------------------------------------------------------
            else:

                new_hidden_states = torch.zeros_like(hidden_states)
                B, S, D = hidden_states.shape

                for expert_idx, expert_norm in enumerate(norms):
                    mask = token_types == expert_idx
                    if mask.sum() == 0:
                        continue

                    dim_input = self.config.dim_inputs[expert_idx]
                    selected = hidden_states[mask]  # [K, D]

                    if self.config.use_adarms and expert_idx == 1:
                        gate_mask = mask
                        selected = selected.view(
                            -1,
                            self.config.action_horizon_flow,
                            selected.shape[-1],
                        )
                        input_slice = selected[:, :, :dim_input]
                        cond = adarms_conds[expert_idx]
                    else:
                        input_slice = selected[:, :dim_input]
                        cond = adarms_conds[expert_idx]

                    if use_selective_recompute:

                        def norm_chunk(t_x, t_cond, expert_norm=expert_norm):
                            if t_cond is None or (
                                isinstance(t_cond, torch.Tensor) and t_cond.numel() == 0
                            ):
                                out, _ = expert_norm(t_x)
                            else:
                                out, _ = expert_norm(t_x, t_cond)
                            return out

                        cond_for_cp = (
                            cond
                            if cond is not None
                            else torch.empty(0, device=input_slice.device)
                        )

                        processed = cp.checkpoint(
                            norm_chunk,
                            input_slice,
                            cond_for_cp,
                            use_reentrant=False,
                        )
                    else:
                        processed, gate = expert_norm(input_slice, cond)

                    if self.config.use_adarms and expert_idx == 1:
                        processed = processed.view(-1, dim_input)

                    # scatter back
                    b_id, s_id = torch.where(mask)
                    new_hidden_states[b_id, s_id, :dim_input] = processed.to(
                        hidden_states.dtype
                    )

                hidden_states = new_hidden_states

        # -------------------------
        # Case 2: norm_moe=False (single LN)
        # -------------------------
        else:

            def norm_chunk_shared(t_x, dummy, norm_module=norm):
                out, _ = norm_module(t_x)
                return out

            if use_selective_recompute:
                dummy = torch.empty(0, device=hidden_states.device)
                hidden_states = cp.checkpoint(
                    norm_chunk_shared,
                    hidden_states,
                    dummy,
                    use_reentrant=False,
                )
            else:
                hidden_states, gate = norm(hidden_states)

        return hidden_states, gate, gate_mask

    def _gated_residual(self, x, y, gate, start_indices=None, end_indices=None):
        """
        Applies gated residual connection with optional gate parameter.

        Args:
            x: Input tensor (residual)
            y: Output tensor to be added
            gate: Optional gate tensor to modulate the addition

        Returns:
            x + y if gate is None, otherwise x + y * gate
        """
        if x is None and y is None:
            return None
        if x is None or y is None:
            return x if x is not None else y
        if gate is None:
            return x + y

        new_y = y.clone()
        selected_y = y[start_indices[1] : end_indices[1]]
        selected_y = selected_y.view(
            -1, self.config.action_horizon_flow, selected_y.shape[-1]
        )[:, :, : self.config.dim_inputs[1]]
        selected_y = selected_y.to(torch.float32) * gate
        new_y[start_indices[1] : end_indices[1], : self.config.dim_inputs[1]] = (
            selected_y.view(-1, self.config.dim_inputs[1]).to(new_y.dtype)
        )

        return x + new_y

    def scatter_proprioception_embeddings(
        self, input_ids, inputs_embeds, proprioception, dataset_names, agent_pos_mask
    ):
        use_state_string_representation = getattr(
            self.config, "use_state_string_representation", False
        )
        if proprioception is not None and not use_state_string_representation:
            proprioception = proprioception.to(inputs_embeds.device).to(
                inputs_embeds.dtype
            )
            agent_pos_mask = agent_pos_mask.to(inputs_embeds.device).to(
                inputs_embeds.dtype
            )
            proprioception = self.action_preprocessor.proprioception_proj(
                proprioception,
                dataset_names,
                agent_pos_mask,
                use_history=proprioception.shape[1] > 1,
            )
            mask = input_ids == self.action_token_id_set["propri_token_id"]
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            proprioception_mask = mask_expanded.to(inputs_embeds.device)

            proprioception = proprioception.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                proprioception_mask, proprioception
            )

        return inputs_embeds

    def scatter_flow_action_embeddings(
        self,
        input_ids,
        inputs_embeds,
        action_chunk,
        dataset_names,
        sample_time,
        dof_mask,
    ):
        if not self.config.use_flow_action_expert:
            return inputs_embeds, None, None
        adarms_cond, flow = None, None
        if action_chunk is not None:
            action_chunk = action_chunk.to(inputs_embeds.device)
            dof_mask = dof_mask.to(inputs_embeds.device)
            noisy_action_emb, flow, adarms_cond = self.action_preprocessor(
                action_chunk, dataset_names, sample_time, dof_mask
            )
            mask = input_ids == self.action_token_id_set["action_token_id"]
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            action_mask = mask_expanded.to(inputs_embeds.device)

            noisy_action_emb = noisy_action_emb.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(action_mask, noisy_action_emb)

        return inputs_embeds, flow, adarms_cond

    @staticmethod
    def _update_position_ids(position_ids, moe_token_types, positional_masks):
        return update_position_ids(position_ids, moe_token_types, positional_masks)

    def _update_joint_attention_mask_2d(
        self, attention_mask, moe_token_types, positional_masks
    ):
        return update_joint_attention_mask_2d(
            attention_mask,
            moe_token_types,
            positional_masks,
            causal_action_attention_mask=self.config.causal_action_attention_mask,
        )

    def _update_joint_attention_flash_mask(
        self, attention_mask, moe_token_types, positional_masks, debug=False
    ):
        return update_joint_attention_flash_mask(
            attention_mask,
            moe_token_types,
            positional_masks,
            causal_action_attention_mask=self.config.causal_action_attention_mask,
            debug=debug,
        )


class ActionGenerationMixin(GenerationMixin):
    action_preprocessor: ActionProcessor

    def to_bfloat16_for_selected_params(self, fsdp_plugin=None, accelerator=None):
        """
        Keep selected model parameters in float32 and cast the rest to bfloat16.
        - If fsdp_plugin exists, use FSDP v1 mixed_precision wrapping.
        - Otherwise, modify parameter dtypes directly.
        """

        def _assign_child(root_module, dotted_name: str, new_child):
            parts = dotted_name.split(".")
            parent = root_module
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], new_child)

        if fsdp_plugin:
            fsdp_version = getattr(fsdp_plugin, "fsdp_version", None)
            if fsdp_version != 1:
                raise RuntimeError("Only FSDP v1 (fsdp_version=1) is supported.")

            device = getattr(
                accelerator, "device", torch.device("cuda", torch.cuda.current_device())
            )
            if isinstance(device, torch.device) and device.type == "cuda":
                if device.index is not None:
                    torch.cuda.set_device(device.index)
            device_id = device.index

            # Move the model to the target device first
            self = self.to(device)

            # Define the mixed precision policy
            bf16_policy = MP(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.bfloat16,
                cast_forward_inputs=False,
                cast_root_forward_inputs=False,
            )

            fp32_policy = MP(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
                cast_forward_inputs=False,
                cast_root_forward_inputs=False,
            )

            from wall_x.model.qact.qwen2_5.modeling_qwen2_5_vl import (
                Qwen2_5_VLVisionBlock,
            )
            from wall_x.model.qact.qwen2_5.modeling_qwen2_5_vl_act import (
                Qwen2_5_VLDecoderLayer_with_MoE,
            )

            target_classes = (
                Qwen2_5_VLDecoderLayer_with_MoE,
                Qwen2_5_VLVisionBlock,
            )

            # Step 1: Find top-level ActionProcessor modules and wrap them separately with FSDP (FP32)
            for name, module in list(self.named_modules()):
                if isinstance(module, nn.Module) and any(
                    k in name.lower()
                    for k in [
                        "input_layernorm",
                        "post_attention_layernorm",
                        "model.norm",
                        "action_preprocessor",
                    ]
                ):
                    if any(
                        True for _ in module.children()
                    ):  # Wrap only leaves to avoid parent-child duplication
                        continue
                    if getattr(module, "_fsdp_wrapped", False):
                        continue

                    logger.info("[FSDP v1] wrapping module in FP32: %s", name)
                    wrapped = FSDP(
                        module,
                        mixed_precision=fp32_policy,
                        sharding_strategy=torch.distributed.fsdp.ShardingStrategy.SHARD_GRAD_OP,
                        backward_prefetch="BACKWARD_PRE",
                        device_id=device_id,
                        use_orig_params=True,
                    )
                    _assign_child(self, name, wrapped)
                    setattr(wrapped, "_fsdp_wrapped", True)

            for name, module in list(self.named_modules()):
                if isinstance(module, target_classes):

                    if getattr(module, "_fsdp_wrapped", False):
                        continue

                    logger.info("[FSDP v1] wrapping module in BF16: %s", name)
                    wrapped = FSDP(
                        module,
                        mixed_precision=bf16_policy,
                        sharding_strategy=torch.distributed.fsdp.ShardingStrategy.SHARD_GRAD_OP,
                        backward_prefetch="BACKWARD_PRE",
                        device_id=device_id,
                        use_orig_params=True,
                    )
                    _assign_child(self, name, wrapped)
                    setattr(wrapped, "_fsdp_wrapped", True)

            # Step 2: Wrap the outer module with FSDP using the BF16 policy
            logger.info("[FSDP v1] wrapping root model with bf16 mixed precision...")
            self = FSDP(
                self,
                mixed_precision=bf16_policy,
                sharding_strategy=torch.distributed.fsdp.ShardingStrategy.SHARD_GRAD_OP,
                backward_prefetch="BACKWARD_PRE",
                device_id=device_id,
                use_orig_params=True,
            )

            return self

        # ----------------- Non-FSDP path -----------------
        else:
            logger.info("Running manual dtype conversion (no FSDP).")
            params_to_keep_float32 = []
            for name, _ in self.named_parameters():
                if any(
                    k in name
                    for k in [
                        "input_layernorm",
                        "post_attention_layernorm",
                        "model.norm",
                        "action_preprocessor",
                        "action_processor",
                    ]
                ):
                    params_to_keep_float32.append(name)

            for name, param in self.named_parameters():
                if name not in params_to_keep_float32:
                    param.data = param.data.to(torch.bfloat16)
                if name in params_to_keep_float32:
                    param.data = param.data.to(torch.float32)

            return self

    def define_action_token_id(self):
        # Get the action token list through tokenizer_mixin; compatible with pure flow mode
        if self.processor is not None:
            action_token_list = []
            if self.tokenizer_mixin is not None:
                action_token_list = self.tokenizer_mixin.get_action_token_list(
                    self.processor
                )

            action_token_id = self.processor.tokenizer.convert_tokens_to_ids(
                "<|action|>"
            )
            propri_token_id = self.processor.tokenizer.convert_tokens_to_ids(
                "<|propri|>"
            )
            self.action_token_id_set = {
                "action_token_list": action_token_list,
                "propri_token_id": propri_token_id,
                "action_token_id": action_token_id,
            }

    def add_lora(
        self, r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.1
    ):
        """Add LoRA adapters"""
        config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, config)
        # Log trainable parameter information
        self.model.print_trainable_parameters()

    def compute_loss(
        self,
        hidden_states,
        logits,
        input_ids=None,
        dataset_names=None,
        labels=None,
        action_chunk=None,
        dof_mask=None,
        flow=None,
        flow_loss_mask=None,
        _lm_loss_mask=None,
        action_hidden_states=None,
        **kwargs,
    ):
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape

        loss = 0
        cross_entropy_loss, flow_loss = None, None

        if dataset_names is not None:
            unique_datasets_name = list(set(dataset_names))
            _device = hidden_states.device
            _flow_channel_names = [
                f"{name}_flow"
                for name in unique_datasets_name
                if is_action_dataset_name(name)
            ]
            channel_loss_dict = {
                name: torch.tensor(0.0, device=_device)
                for name in unique_datasets_name + _flow_channel_names
            }
            channel_loss_count_dict = {
                name: torch.tensor(0, device=_device)
                for name in unique_datasets_name + _flow_channel_names
            }
        else:
            unique_datasets_name, channel_loss_dict, channel_loss_count_dict = (
                None,
                None,
                None,
            )

        if labels is not None:
            if _lm_loss_mask is not None:
                # ===== Optimized path: logits already gathered for loss tokens only =====
                # logits: [N_loss, V] or None, _lm_loss_mask: [B, S-1]
                if logits is not None:
                    shift_logits = logits.to(torch.float32)  # [N_loss, V]
                    shift_labels = labels[..., 1:].contiguous()
                    shift_labels_flat = shift_labels[_lm_loss_mask]  # [N_loss]
                    shift_labels_flat = shift_labels_flat.to(shift_logits.device)

                    _cross_entropy_loss = self.loss_fct(shift_logits, shift_labels_flat)
                    cross_entropy_loss = _cross_entropy_loss.mean()

                    # compute channel loss
                    if unique_datasets_name is not None:
                        batch_idx = (
                            torch.arange(batch_size, device=_lm_loss_mask.device)
                            .unsqueeze(1)
                            .expand_as(_lm_loss_mask)
                        )
                        loss_batch_idx = batch_idx[_lm_loss_mask]  # [N_loss]
                        for dataset_name_i in unique_datasets_name:
                            ds_mask = torch.tensor(
                                [name == dataset_name_i for name in dataset_names],
                                device=_lm_loss_mask.device,
                                dtype=torch.bool,
                            )
                            tok_ds_mask = ds_mask[loss_batch_idx]
                            channel_loss_dict[dataset_name_i] = (
                                _cross_entropy_loss[tok_ds_mask].sum()
                                if tok_ds_mask.any()
                                else torch.tensor(0.0, device=shift_logits.device)
                            )
                            channel_loss_count_dict[dataset_name_i] += tok_ds_mask.sum()
                else:
                    cross_entropy_loss = torch.tensor(0.0, device=hidden_states.device)
            else:
                # ===== Original path (inference / no labels optimization) =====
                shift_logits = logits[..., :-1, :].contiguous().to(torch.float32)
                shift_labels = labels[..., 1:].contiguous()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                non_ignored_mask = shift_labels != -100
                _cross_entropy_loss = self.loss_fct(shift_logits, shift_labels)
                cross_entropy_loss = (
                    _cross_entropy_loss[non_ignored_mask].mean()
                    if non_ignored_mask.any()
                    else torch.tensor(0.0, device=shift_logits.device)
                )

                # compute channel loss
                _cross_entropy_loss = _cross_entropy_loss.view(
                    batch_size, seq_length - 1
                )
                non_ignored_mask = non_ignored_mask.view(batch_size, seq_length - 1)
                if unique_datasets_name is not None:
                    for dataset_name_i in unique_datasets_name:
                        dataset_mask = torch.tensor(
                            [name == dataset_name_i for name in dataset_names],
                            device=logits.device,
                        )
                        combined_mask = dataset_mask.unsqueeze(1) & non_ignored_mask
                        channel_loss_dict[dataset_name_i] = (
                            _cross_entropy_loss[combined_mask].sum()
                            if combined_mask.any()
                            else torch.tensor(0.0, device=shift_logits.device)
                        )
                        channel_loss_count_dict[dataset_name_i] += combined_mask.sum()

            if not torch.isnan(cross_entropy_loss):
                loss += cross_entropy_loss * self.config.ar_loss_weight
            else:
                with torch.no_grad():
                    cross_entropy_loss.detach()

            # compute action token accuracy(computed uniformly through tokenizer_mixin)
            # if self.tokenizer_mixin is not None and self.action_mapper is not None:
            #     accuracy_dict = self.tokenizer_mixin.compute_accuracy(
            #         logits, labels, self.action_mapper, self.action_token_id_set
            #     )
            #     channel_loss_dict.update(accuracy_dict)

        if action_chunk is not None:
            action_mask = input_ids == self.action_token_id_set["action_token_id"]
            if action_mask.any():
                if action_hidden_states is None:
                    action_hidden_states = hidden_states[action_mask].to(torch.float32)
                else:
                    action_hidden_states = action_hidden_states.reshape(
                        -1, action_hidden_states.shape[-1]
                    ).to(torch.float32)
                flow = flow.reshape(-1, flow.shape[-1])
                _flow_loss = self.action_preprocessor.flow_loss(
                    action_hidden_states, flow, action_chunk, dof_mask, flow_loss_mask
                )
                if isinstance(_flow_loss, torch.Tensor):
                    # Compute the valid-element mask as the intersection of dof_mask and flow_loss_mask
                    valid_mask = (
                        dof_mask.reshape(-1, dof_mask.shape[-1])
                        if dof_mask is not None
                        else None
                    )
                    if flow_loss_mask is not None:
                        flow_mask_expanded = (
                            flow_loss_mask.unsqueeze(-1)
                            .reshape(-1, 1)
                            .expand(-1, _flow_loss.shape[-1])
                        )
                        valid_mask = (
                            valid_mask * flow_mask_expanded
                            if valid_mask is not None
                            else flow_mask_expanded
                        )
                    if valid_mask is not None and not valid_mask.all():
                        flow_loss = _flow_loss.sum() / valid_mask.sum()
                    else:
                        flow_loss = _flow_loss.mean()
                    loss += flow_loss
                    _flow_loss = _flow_loss.view(
                        dof_mask.shape[0], dof_mask.shape[1], dof_mask.shape[2]
                    )

                    # compute flow channel loss
                    if unique_datasets_name is not None:
                        B, T, D = _flow_loss.shape
                        action_ds_names = [
                            name
                            for name in dataset_names
                            if is_action_dataset_name(name)
                        ]
                        if valid_mask is not None:
                            valid_mask_3d = valid_mask.view(B, T, D)
                        else:
                            valid_mask_3d = torch.ones_like(
                                _flow_loss, dtype=torch.bool, device=_flow_loss.device
                            )
                        for dataset_name_i in unique_datasets_name:
                            ds_mask = torch.tensor(
                                [name == dataset_name_i for name in action_ds_names],
                                device=_flow_loss.device,
                                dtype=torch.bool,
                            )
                            if not ds_mask.any():
                                continue
                            flow_key = f"{dataset_name_i}_flow"
                            ds_mask_3d = ds_mask.view(-1, 1, 1).expand(B, T, D)
                            flow_loss_sum = (_flow_loss * ds_mask_3d).sum()
                            flow_count = (valid_mask_3d * ds_mask_3d).sum()
                            channel_loss_dict[flow_key] = (
                                channel_loss_dict[flow_key] + flow_loss_sum
                            )
                            channel_loss_count_dict[flow_key] = (
                                channel_loss_count_dict[flow_key] + flow_count
                            )

        return (
            loss,
            cross_entropy_loss,
            flow_loss,
            channel_loss_dict,
            channel_loss_count_dict,
        )
