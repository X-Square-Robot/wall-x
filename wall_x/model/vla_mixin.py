import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from torch.distributed.fsdp import MixedPrecision as MP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from wall_x.fusions import ops

from peft import LoraConfig, get_peft_model
from typing import Optional, Union, Dict
from packaging import version

from transformers import GenerationMixin
from transformers.activations import ACT2FN
from transformers.modeling_utils import AttentionInterface

from transformers.utils import logging, is_torch_xla_available

from wall_x.model.action_head import ActionProcessor
from wall_x.model.model_utils import find_first_last_ones

ALL_ATTENTION_FUNCTIONS: AttentionInterface = AttentionInterface()
logger = logging.get_logger(__name__)


X2ROBOT_ATTENTION_FUNCTIONS = []
ATTENTION_TYPES_WITH_2D_MASK = [
    "sdpa",
]
ATTENTION_TYPES_WITH_FLASH_MASK = []


class TokenTypeRouter(nn.Module):
    def __init__(self, num_experts: int):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, token_types: torch.Tensor) -> torch.Tensor:
        """
        Assigns tokens to different experts based on `token_type`. 
        Args:
            token_types (torch.Tensor): A tensor of shape (batch_size, seq_length) representing the type of each token. 

        Returns:
            experts_indices (torch.Tensor): A tensor of shape (batch_size, seq_length) representing the expert index assigned to each token. 
        """
        experts_indices = token_types % self.num_experts
        return experts_indices


class BlockSparseMLP(nn.Module):
    def __init__(self, config, use_selective_recompute: bool = False):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config["intermediate_size"]
        self.hidden_act = config["hidden_act"]

        self.use_selective_recompute = use_selective_recompute

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        self.act_fn = ACT2FN[self.hidden_act]

    def _full_mlp(self, hidden_state):
        gate_out = self.gate_proj(hidden_state)
        up_out = self.up_proj(hidden_state)
        act_out = self.act_fn(gate_out) * up_out
        return self.down_proj(act_out)

    def forward(self, hidden_state):
        if self.use_selective_recompute:
            # Perform checkpoint recalculation for the entire expert MLP.
            return cp.checkpoint(
                self._full_mlp,
                hidden_state,
                use_reentrant=False,
            )
        else:
            return self._full_mlp(hidden_state)


class SparseMoeBlock(nn.Module):
    def __init__(self, config, num_experts: int, use_selective_recompute: bool = False):
        super().__init__()
        self.num_experts = num_experts
        self.use_selective_recompute = use_selective_recompute

        # Pass the `use_selective_recompute` parameter to each expert.
        self.experts = nn.ModuleList(
            [
                BlockSparseMLP(
                    config.experts[i], use_selective_recompute=use_selective_recompute
                )
                for i in range(num_experts)
            ]
        )

        if not hasattr(config, "dim_inputs") or not config.dim_inputs:
            raise ValueError("配置中必须包含有效的 dim_inputs")

        self.dim_inputs = config.dim_inputs
        self.permuted = config.mot_opt

    def forward(
        self,
        hidden_states: torch.Tensor,
        experts_indices: torch.Tensor,
        start_indices: torch.Tensor,
        end_indices: torch.Tensor,
    ) -> torch.Tensor:

        if self.permuted:
            permuted_inputs = hidden_states
        else:
            batch_size, seq_length, hidden_dim = hidden_states.shape

            flat_hidden = hidden_states.reshape(-1, hidden_dim)
            experts_indices = experts_indices.reshape(-1)
            probs = torch.ones_like(experts_indices, dtype=torch.float32).reshape(-1, 1)
            permuted_inputs, row_id_map = ops.permute(flat_hidden, experts_indices)

        # buffer
        final_output = torch.zeros_like(permuted_inputs)

        # Expert forward（contain selective recompute）
        for expert_idx, expert in enumerate(self.experts):
            start, end = start_indices[expert_idx], end_indices[expert_idx]
            if start == end:
                continue

            dim_input = self.dim_inputs[expert_idx]
            expert_input = permuted_inputs[start:end, :dim_input]

            partial_output = expert(expert_input)
            final_output[start:end, :dim_input] = partial_output[:, :dim_input]

        if self.permuted:
            return final_output
        else:
            final_output = ops.unpermute(final_output, row_id_map, probs)
            return final_output.reshape(batch_size, seq_length, hidden_dim)


class ActionModelMixMin:
    # config: Qwen2_5_VLConfig
    action_preprocessor: ActionProcessor
    router: TokenTypeRouter
    moe: SparseMoeBlock

    def __init__(self, config, action_preprocessor, router, moe):
        self.config = config
        self.action_preprocessor = action_preprocessor
        self.router = router
        self.moe = moe
        self._mot_opt_warned = False

    def set_normalizer(self, normalizer_action, normalizer_propri):
        if hasattr(self, "action_preprocessor"):
            self.action_preprocessor.set_normalizer(normalizer_action, normalizer_propri)
        else:
            logger.warning("ActionModelMixMin.set_normalizer is called but action_preprocessor is not set")

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
        if (
            proprioception is not None
            and not self.config.use_state_string_representation
        ):
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
        self, input_ids, inputs_embeds, action_chunk, dataset_names, dof_mask
    ):
        if not self.config.use_flow_action_expert:
            return inputs_embeds, None, None
        adarms_cond, flow = None, None
        if action_chunk is not None:
            action_chunk = action_chunk.to(inputs_embeds.device)
            dof_mask = dof_mask.to(inputs_embeds.device)
            noisy_action_emb, flow, adarms_cond = self.action_preprocessor(
                action_chunk, dataset_names, dof_mask
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
    def _update_position_ids(
        position_ids,
        moe_token_types,
        positional_masks,
    ):
        if (
            positional_masks is None
            or "ar_predict_token_positions" not in positional_masks
        ):
            return position_ids

        new_position_ids = position_ids.clone()
        ar_predict_token_positions = positional_masks["ar_predict_token_positions"]
        flow_mask = moe_token_types == 1

        start_ar_pos, end_ar_pos = find_first_last_ones(ar_predict_token_positions)
        start_flow_pos, end_flow_pos = find_first_last_ones(flow_mask)

        for bs_i in range(position_ids.shape[1]):
            if start_ar_pos[bs_i] != -1 and end_ar_pos[bs_i] != -1:
                start_ar_ids = new_position_ids[:, bs_i, start_ar_pos[bs_i]]
                start_flow_ids = new_position_ids[:, bs_i, start_flow_pos[bs_i]]
                diff = start_flow_ids - start_ar_ids
                new_position_ids[:, bs_i, start_flow_pos[bs_i] :] = position_ids[
                    :, bs_i, start_flow_pos[bs_i] :
                ] - diff.unsqueeze(-1)

        return new_position_ids

    def _update_joint_attention_mask_2d(
        self,
        attention_mask,
        moe_token_types,
        positional_masks,
    ):
        if attention_mask.dim() == 3:  # bs, seq_len, seq_len
            return attention_mask

        bs, seq_len = moe_token_types.shape[0], moe_token_types.shape[1]
        # Create a lower triangular matrix as a causal mask.
        causal_mask = torch.tril(
            torch.ones(
                (seq_len, seq_len), dtype=torch.bfloat16, device=moe_token_types.device
            )
        )
        # Extended to the batch dimension.
        attention_mask = causal_mask.unsqueeze(0).expand(bs, -1, -1)

        if positional_masks is not None and "padding_positions" in positional_masks:
            padding_positions = positional_masks["padding_positions"]
            # The padding is set to zero.
            attention_mask = torch.where(
                padding_positions[:, None, :],
                torch.zeros_like(attention_mask),
                attention_mask,
            )
            # The padding is set to zero.
            attention_mask = torch.where(
                padding_positions[:, :, None],
                torch.zeros_like(attention_mask),
                attention_mask,
            )

        # Set all values ​​in the moe1 section to 1, and disable the fast section.
        moe1_mask = (moe_token_types[:, :, None]) & (moe_token_types[:, None, :])

        if (
            not self.config.causal_action_attention_mask
        ):  # If a causal action attention mask is not used, then all elements in the moe1 section are set to 1.
            attention_mask = torch.where(
                moe1_mask, torch.ones_like(attention_mask), attention_mask
            )

        if (
            positional_masks is not None
            and "ar_predict_token_positions" in positional_masks
        ):
            ar_predict_token_positions = positional_masks["ar_predict_token_positions"]
            moe1_mask = (moe_token_types[:, :, None]) & (
                ar_predict_token_positions[:, None, :]
            )
            attention_mask = torch.where(
                moe1_mask, torch.zeros_like(attention_mask), attention_mask
            )

        if (
            positional_masks is not None
            and "valid_flow_action_positions" in positional_masks
        ):
            # true in moe_token_types but false in valid_flow_action_positions
            nonvalid_flow_action_positions = (
                moe_token_types & ~positional_masks["valid_flow_action_positions"]
            )
            attention_mask = torch.where(
                nonvalid_flow_action_positions[:, None, :],
                torch.zeros_like(attention_mask),
                attention_mask,
            )
            attention_mask = torch.where(
                nonvalid_flow_action_positions[:, :, None],
                torch.zeros_like(attention_mask),
                attention_mask,
            )

        return attention_mask

    def _update_joint_attention_flash_mask(
        self,
        attention_mask,
        moe_token_types,
        positional_masks,
        debug=False,
    ):
        device = moe_token_types.device
        B, S = moe_token_types.shape
        i32 = torch.int32

        # ---- Return vector initialization ----
        LTS = torch.ones((B, S), device=device, dtype=i32) * S
        UTE = (
            torch.arange(S, device=device, dtype=i32).unsqueeze(0).expand(B, S).clone()
        )

        # Handling padding positions
        if positional_masks is not None and "padding_positions" in positional_masks:
            padding_positions = positional_masks["padding_positions"]
            LTS[padding_positions] = 0
            UTE[padding_positions] = S

        # Handling ar predict tokens
        if (
            positional_masks is not None
            and "ar_predict_token_positions" in positional_masks
        ):
            start_ar_pos, end_ar_pos = find_first_last_ones(
                positional_masks["ar_predict_token_positions"]
            )
            for bs_i in range(B):
                if end_ar_pos[bs_i] != -1:
                    LTS[bs_i, positional_masks["ar_predict_token_positions"][bs_i]] = (
                        end_ar_pos[bs_i].to(i32) + 1
                    )

        # Handling flow action bidirectional mask
        flow_mask = moe_token_types == 1
        if not self.config.causal_action_attention_mask:
            start_flow_pos, end_flow_pos = find_first_last_ones(flow_mask)
            for bs_i in range(B):
                if start_flow_pos[bs_i] != -1:
                    UTE[bs_i, flow_mask[bs_i]] = start_flow_pos[bs_i].to(i32)

        # Handling validate flow
        if (
            positional_masks is not None
            and "valid_flow_action_positions" in positional_masks
        ):
            flow_mask = moe_token_types == 1
            nonvalid_flow_action_positions = (
                flow_mask & ~positional_masks["valid_flow_action_positions"]
            )
            if nonvalid_flow_action_positions.any():
                LTS[nonvalid_flow_action_positions] = 0
                UTE[nonvalid_flow_action_positions] = S

        LTS = LTS.unsqueeze(-1)
        UTE = UTE.unsqueeze(-1)

        startend_row_indices = torch.cat([LTS, UTE], dim=-1)
        # startend_row_indices = LTS

        # add num_heads dimension
        startend_row_indices = startend_row_indices.unsqueeze(1)

        return startend_row_indices


class ActionGenerationMixin(GenerationMixin):
    action_preprocessor: ActionProcessor

    def to_bfloat16_for_selected_params(self, fsdp_plugin=None, accelerator=None):
        """
        Keep some model parameters as float32, and convert others to bfloat16.
        - If `fsdp_plugin` exists, use FSDP v1's `mixed_precision` wrapper.
        - Otherwise, directly modify the parameter dtype.
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
                raise RuntimeError("仅支持 FSDP v1（fsdp_version=1）。")

            device = getattr(
                accelerator, "device", torch.device("cuda", torch.cuda.current_device())
            )
            if isinstance(device, torch.device) and device.type == "cuda":
                if device.index is not None:
                    torch.cuda.set_device(device.index)
            device_id = device.index

            # move model to device
            self = self.to(device)

            # Define the mixed-precision strategy.
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

            # Step 1️⃣: Identify the top-level ActionProcessor module and wrap it separately with FSDP (FP32).
            for name, module in list(self.named_modules()):
                if isinstance(module, nn.Module) and any(
                    k in name.lower() for k in ["action_preprocessor"]
                ):
                    if any(True for _ in module.children()): 
                        continue
                    if getattr(module, "_fsdp_wrapped", False):
                        continue

                    print(f"[FSDP v1] wrapping module in FP32: {name}")
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

            # Step 2️⃣: The outermost layer uses unified FSDP (BF16 strategy).
            print("[FSDP v1] wrapping root model with bf16 mixed precision...")
            self = FSDP(
                self,
                mixed_precision=bf16_policy,
                sharding_strategy=torch.distributed.fsdp.ShardingStrategy.SHARD_GRAD_OP,
                backward_prefetch="BACKWARD_PRE",
                device_id=device_id,
                use_orig_params=True,
            )

            return self

        # ----------------- Non-FSDP scenarios -----------------
        else:
            print("[INFO] Running manual dtype conversion (no FSDP).")
            self.to(dtype=torch.float32)

            params_to_keep_float32 = []
            for name, _ in self.named_parameters():
                if any(
                    k in name
                    for k in [
                        "input_layernorm",
                        "post_attention_layernorm",
                        "model.norm",
                        "action_preprocessor",
                    ]
                ):
                    params_to_keep_float32.append(name)

            for name, param in self.named_parameters():
                if name not in params_to_keep_float32:
                    param.data = param.data.to(torch.bfloat16)

            return self

    def define_action_token_id(self):
        action_token_list = []
        if self.action_tokenizer_type:
            for i in range(self.action_tokenizer.vocab_size):
                action_token_id = self.processor.tokenizer.convert_tokens_to_ids(
                    f"<|action_token_{i}|>"
                )
                action_token_list.append(action_token_id)

        action_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|action|>")
        propri_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|propri|>")
        self.action_token_id_set = {
            "action_token_list": action_token_list,
            "propri_token_id": propri_token_id,
            "action_token_id": action_token_id,
        }

    def add_lora(
        self, r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.1
    ):
        """Add LoRA adapter"""
        config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, config)
        # Print trainable parameter information.
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
        **kwargs,
    ):
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape

        loss = 0
        cross_entropy_loss, flow_loss = None, None

        # if dataset_names is not None:
        #     unique_datasets_name = list(set(dataset_names))
        #     channel_loss_dict = {
        #         dataset_name: torch.tensor(0.0, device=logits.device)
        #         for dataset_name in _ACTION_DATASET_NAMES + _MULTIMODAL_DATASET_NAMES
        #     }
        #     channel_loss_count_dict = {
        #         dataset_name: torch.tensor(0, device=logits.device)
        #         for dataset_name in _ACTION_DATASET_NAMES + _MULTIMODAL_DATASET_NAMES
        #     }
        # else:
        unique_datasets_name, channel_loss_dict, channel_loss_count_dict = (
            None,
            None,
            None,
            )

        if labels is not None:
            action_accuracy = 0

            shift_logits = logits[..., :-1, :].contiguous()
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
            _cross_entropy_loss = _cross_entropy_loss.view(batch_size, seq_length - 1)
            non_ignored_mask = non_ignored_mask.view(batch_size, seq_length - 1)
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
                loss += cross_entropy_loss
            else:
                with torch.no_grad():
                    cross_entropy_loss.detach()

            # compute action token accuracy
            if len(self.action_token_id_set["action_token_list"]) > 0:
                shift_logits = logits[..., :-1, :].contiguous()
                action_preds = shift_logits.argmax(dim=-1)
                shift_labels = labels[..., 1:].contiguous()
                action_mask = (
                    shift_labels > self.action_token_id_set["action_token_list"][0]
                )
                correct_preds = (action_preds == shift_labels) & action_mask
                action_accuracy = (
                    correct_preds.sum().float() / action_mask.sum().float()
                )
                channel_loss_dict["action_accuracy"] = action_accuracy

        if action_chunk is not None:
            action_mask = input_ids == self.action_token_id_set["action_token_id"]
            if action_mask.any():
                action_hidden_states = hidden_states[action_mask].to(torch.float32)
                flow = flow.reshape(-1, flow.shape[-1])
                _flow_loss = self.action_preprocessor.flow_loss(
                    action_hidden_states, flow, action_chunk, dof_mask, flow_loss_mask
                )
                if isinstance(_flow_loss, torch.Tensor):
                    flow_loss = _flow_loss.mean()
                loss += flow_loss * self.config.flow_loss_weight
                _flow_loss = _flow_loss.view(
                    dof_mask.shape[0], dof_mask.shape[1], dof_mask.shape[2]
                )

        return (
            loss,
            cross_entropy_loss,
            flow_loss,
            channel_loss_dict,
            channel_loss_count_dict,
        )


class AttentionsSelectorMixin:

    @classmethod
    def _autoset_attn_implementation(
        cls,
        config,
        use_flash_attention_2: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        check_device_map: bool = True,
    ):
        """
        Automatically checks and dispatches to a default attention implementation. In order of priority:
            1. An implementation specified in `config._attn_implementation` (due for example to the argument attn_implementation="sdpa" in from_pretrained).
            2. DEPRECATED: if use_flash_attention_2 is set to `True` and `flash_attn` is available, flash attention. (`LlamaFlashAttention` for example)
            3. SDPA implementation, if available and supported by the model type. (`LlamaSdpaAttention` for example)
            4. The default model's implementation otherwise (`LlamaAttention` for example) .
        """
        # Here we use config._attn_implementation_internal to check whether the attention implementation was explicitly set by the user.
        # The property `PretrainedConfig._attn_implementation` is never `None`, for backward compatibility (always fall back on "eager").
        # The `hasattr` here is used as some Transformers tests for some reason do not call PretrainedConfig __init__ (e.g. test_no_super_init_config_and_model)
        requested_attn_implementation = None
        if (
            hasattr(config, "_attn_implementation_internal")
            and config._attn_implementation_internal is not None
        ):
            if (
                config._attn_implementation != "flash_attention_2"
                and use_flash_attention_2
            ):
                raise ValueError(
                    f'Both attn_implementation="{config._attn_implementation}" and `use_flash_attention_2=True` were used when loading the model, which are not compatible.'
                    ' We recommend to just use `attn_implementation="flash_attention_2"` when loading the model.'
                )

            if (
                not isinstance(config._attn_implementation, dict)
                and config._attn_implementation
                not in ["eager"]
                + ALL_ATTENTION_FUNCTIONS.valid_keys()
                + X2ROBOT_ATTENTION_FUNCTIONS
            ):
                message = f'Specified `attn_implementation="{config._attn_implementation}"` is not supported. The only possible arguments are `attn_implementation="eager"` (manual attention implementation)'
                if cls._supports_flash_attn_2:
                    message += ', `"attn_implementation=flash_attention_2"` (implementation using flash attention 2)'
                if cls._supports_sdpa:
                    message += ', `"attn_implementation=sdpa"` (implementation using torch.nn.functional.scaled_dot_product_attention)'
                if cls._supports_flex_attn:
                    message += ', `"attn_implementation=flex_attention"` (implementation using torch\'s flex_attention)'
                raise ValueError(message + ".")

            # If a config is passed with a preset attn_implementation, we skip the automatic dispatch and use the user-provided config, with hard checks that the requested attention implementation is available.
            requested_attn_implementation = config._attn_implementation_internal

        if use_flash_attention_2:
            logger.warning_once(
                'The model was loaded with use_flash_attention_2=True, which is deprecated and may be removed in a future release. Please use `attn_implementation="flash_attention_2"` instead.'
            )
            config._attn_implementation = "flash_attention_2"

        if config._attn_implementation == "flash_attention_2":
            cls._check_and_enable_flash_attn_2(
                config,
                torch_dtype=torch_dtype,
                device_map=device_map,
                hard_check_only=False,
                check_device_map=check_device_map,
            )
        elif requested_attn_implementation == "flex_attention":
            config = cls._check_and_enable_flex_attn(config, hard_check_only=True)
        elif (
            requested_attn_implementation in [None, "sdpa"]
            and not is_torch_xla_available()
        ):
            # use_flash_attention_2 takes priority over SDPA, hence SDPA treated in this elif.
            config = cls._check_and_enable_sdpa(
                config,
                hard_check_only=(
                    False if requested_attn_implementation is None else True
                ),
            )

            if (
                torch.version.hip is not None
                and config._attn_implementation == "sdpa"
                and torch.cuda.device_count() > 1
                and version.parse(torch.__version__) < version.parse("2.4.1")
            ):
                logger.warning_once(
                    "Using the `SDPA` attention implementation on multi-gpu setup with ROCM may lead to performance issues due to the FA backend. Disabling it to use alternative backends."
                )
                torch.backends.cuda.enable_flash_sdp(False)
        elif requested_attn_implementation in ALL_ATTENTION_FUNCTIONS.valid_keys():
            config._attn_implementation = requested_attn_implementation
        elif isinstance(requested_attn_implementation, dict):
            config._attn_implementation = None
        elif config._attn_implementation in X2ROBOT_ATTENTION_FUNCTIONS:
            pass
        else:
            config._attn_implementation = "eager"

        config._attn_implementation_autoset = True
        return config

    def _check_and_adjust_attn_implementation(
        self, attn_implementation: Optional[str], is_init_check: bool = False
    ) -> str:
        assert (
            attn_implementation
            in ["eager", "flash_attention_2", "sdpa"] + X2ROBOT_ATTENTION_FUNCTIONS
        )
        return attn_implementation
