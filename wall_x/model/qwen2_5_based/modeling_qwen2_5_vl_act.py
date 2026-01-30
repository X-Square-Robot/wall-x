import os
import torch
import yaml
import numpy as np
import glob
import torch.nn as nn
from torchdiffeq import odeint
from dataclasses import dataclass
from torch.nn import CrossEntropyLoss
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model
from typing import Optional, List, Tuple, Any, Dict, Union
import time
from transformers import AutoConfig, AutoProcessor
from transformers.activations import ACT2FN
from transformers.utils import logging, is_torchdynamo_compiling
from transformers.cache_utils import (
    Cache,
    DynamicCache,
    SlidingWindowCache,
    StaticCache,
)
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl import (
    Qwen2_5_VLMLP,
    Qwen2_5_VLRotaryEmbedding,
    Qwen2_5_VLPreTrainedModel,
    Qwen2RMSNorm,
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.modeling_outputs import (
    ModelOutput,
    BaseModelOutputWithPast,
)

from wall_x.fusions import ops
from wall_x.model.action_head import ActionProcessor
from wall_x.model.qwen2_5_based.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from wall_x.model.model_utils import load_wallx_processors, update_model_config
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLAttention,
    Qwen2_5_VLFlashAttention2,
    Qwen2_5_VLSdpaAttention,
)
from wall_x.model.vla_mixin import ActionGenerationMixin, ActionModelMixMin
from wall_x.model.vla_mixin import TokenTypeRouter, SparseMoeBlock
from wall_x.model.vla_mixin import (
    ATTENTION_TYPES_WITH_2D_MASK,
)
from wall_x.model.joint_attention import JOINT_QWEN_ATTENTION_CLASSES

from wall_x.data.config import ACTION_DATASET_NAMES, MULTIMODAL_DATASET_NAMES
from wall_x.data.utils import update_action_statistics
from wall_x.utils.constant import action_statistic_dof
from pprint import pprint

logger = logging.get_logger(__name__)


@dataclass
class Qwen2_5_VLACausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    flow_loss: Optional[torch.FloatTensor] = None
    cross_entropy_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None

    channel_loss_dict: Optional[dict[torch.FloatTensor]] = None
    channel_loss_count_dict: Optional[dict[torch.FloatTensor]] = None

QWEN2_5_VL_ATTENTION_CLASSES = {
    "eager": Qwen2_5_VLAttention,
    "flash_attention_2": Qwen2_5_VLFlashAttention2,
    "sdpa": Qwen2_5_VLSdpaAttention,
}



class Qwen2_5_VLDecoderLayer_with_MoE(nn.Module, ActionModelMixMin):
    def __init__(
        self,
        config: Qwen2_5_VLConfig,
        layer_idx: int,
        num_experts: int,
        use_selective_recompute: bool = True,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.use_selective_recompute = use_selective_recompute

        if (
            config.use_sliding_window
            and config._attn_implementation != "flash_attention_2"
        ):
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        if config.attention_moe:
            self.self_attn = JOINT_QWEN_ATTENTION_CLASSES[config._attn_implementation](
                config, layer_idx
            )
        else:
            self.self_attn = QWEN2_5_VL_ATTENTION_CLASSES[config._attn_implementation](
                config, layer_idx
            )

        if config.use_adarms:
            adarms_cond_dims = [None, config.adarms_cond_dim]
        else:
            adarms_cond_dims = [None, None]

        if config.norm_moe:
            self.input_layernorms = nn.ModuleList(
                [
                    Qwen2RMSNorm(
                        config.dim_inputs[i],
                        eps=config.rms_norm_eps,
                        cond_dim=adarms_cond_dims[i],
                    )
                    for i in range(num_experts)
                ]
            )
            self.post_attention_layernorms = nn.ModuleList(
                [
                    Qwen2RMSNorm(
                        config.dim_inputs[i],
                        eps=config.rms_norm_eps,
                        cond_dim=adarms_cond_dims[i],
                    )
                    for i in range(num_experts)
                ]
            )
            self.input_layernorm, self.post_attention_layernorm = None, None
        else:
            self.input_layernorm = Qwen2RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_attention_layernorm = Qwen2RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.input_layernorms, self.post_attention_layernorms = None, None

        if config.mlp_moe:
            self.router = TokenTypeRouter(num_experts=num_experts)
            self.moe = SparseMoeBlock(
                config,
                num_experts=num_experts,
                use_selective_recompute=use_selective_recompute,
            )
            self.mlp = None
        else:
            self.mlp = Qwen2_5_VLMLP(config)
            self.moe, self.router = None, None

        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        # for vla
        token_types: Optional[torch.LongTensor] = None,
        start_indices: Optional[torch.Tensor] = None,
        end_indices: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        row_id_map: Optional[torch.Tensor] = None,
        orig_shape: Optional[Tuple[int, int, int]] = None,
        adarms_conds: Optional[List[torch.Tensor]] = [None, None],
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states, gate, _ = self._apply_norm_moe(
            hidden_states,
            token_types,
            adarms_conds,
            self.input_layernorms,
            self.input_layernorm,
            start_indices,
            end_indices,
            self.use_selective_recompute,
        )

        # Self Attention
        if self.config.attention_moe:
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                token_types=token_types,
                start_indices=start_indices,
                end_indices=end_indices,
                probs=probs,
                row_id_map=row_id_map,
                orig_shape=orig_shape,
                position_embeddings=position_embeddings,
            )
        else:
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = self._gated_residual(
            residual, hidden_states, gate, start_indices, end_indices
        )

        # Fully Connected
        residual = hidden_states

        hidden_states, gate, gate_mask = self._apply_norm_moe(
            hidden_states,
            token_types,
            adarms_conds,
            self.post_attention_layernorms,
            self.post_attention_layernorm,
            start_indices,
            end_indices,
            self.use_selective_recompute,
        )

        hidden_states = self._apply_mlp_moe(
            hidden_states, token_types, start_indices, end_indices
        )

        hidden_states = self._gated_residual(
            residual, hidden_states, gate, start_indices, end_indices
        )

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class Qwen2_5_VLMoEModel(Qwen2_5_VLPreTrainedModel, ActionModelMixMin):
    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, num_experts=None, *args, **kwargs
    ):
        # If `num_experts` is provided, ensure it is added to the config.
        config = kwargs.get("config", None)
        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

        if num_experts is not None:
            config.num_experts = num_experts

        kwargs["config"] = config
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    def __init__(self, config: Qwen2_5_VLConfig, use_selective_recompute=False):
        super().__init__(config)
        self.config = config
        self.use_selective_recompute = use_selective_recompute
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                Qwen2_5_VLDecoderLayer_with_MoE(
                    config,
                    layer_idx,
                    config.num_experts,
                    use_selective_recompute=use_selective_recompute,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._attn_implementation = config._attn_implementation

        if config.use_adarms:
            adarms_cond_dims = [None, config.adarms_cond_dim]
        else:
            adarms_cond_dims = [None, None]

        if config.norm_moe:
            self.norms = nn.ModuleList(
                [
                    Qwen2RMSNorm(
                        config.dim_inputs[i],
                        eps=config.rms_norm_eps,
                        cond_dim=adarms_cond_dims[i],
                    )
                    for i in range(config.num_experts)
                ]
            )
            self.norm = None
        else:
            self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.norms = None

        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        moe_token_types: Optional[torch.LongTensor] = None,
        start_indices: Optional[torch.Tensor] = None,
        end_indices: Optional[torch.Tensor] = None,
        positional_masks: Optional[dict] = None, 
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        adarms_conds: Optional[List[torch.Tensor]] = [None, None],
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        if moe_token_types is None:
            raise ValueError("moe_token_types must be provided for MoE routing.")
        if start_indices is None or end_indices is None:
            raise ValueError("start_indices and end_indices must be provided for MoE routing")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(
                3, inputs_embeds.shape[0], -1
            )
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if not self.config.attention_moe:
            causal_mask = self._update_causal_mask(
                attention_mask,
                inputs_embeds,
                cache_position,
                past_key_values,
                output_attentions,
                moe_token_types,
            )
        else:
            causal_mask = attention_mask

        hidden_states = inputs_embeds

        if (
            self.config._attn_implementation != "flash_attention_2"
            and self.config.attention_moe is True
        ):
            position_ids = self._update_position_ids(
                position_ids, moe_token_types, positional_masks
            )

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # If `mot_opt` is enabled, the tokens from different experts will be permuted first, resulting in a dimension of [Tokens, HiddenSize].
        orig_shape = hidden_states.shape
        if self.config.mot_opt:
            hidden_states = hidden_states.view(-1, hidden_states.size(-1))
            hidden_states, row_id_map = ops.permute(hidden_states, moe_token_types.view(-1))
        else:
            row_id_map = None
        probs = torch.ones_like(moe_token_types.view(-1), dtype=torch.float32).view(
            -1, 1
        )

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # generate 2d attention mask if needed
        if (
            self.config._attn_implementation in ATTENTION_TYPES_WITH_2D_MASK
            and self.config.attention_moe is True
        ):
            if causal_mask is not None and inputs_embeds.shape[1] > 1:
                causal_mask = self._update_joint_attention_mask_2d(
                    attention_mask=causal_mask,
                    moe_token_types=moe_token_types,
                    positional_masks=positional_masks,
                )


        for decoder_layer in self.layers:
            if output_hidden_states:
                assert (
                    self.config.mot_opt is False
                ), "When using mot_opt, output_hidden_states is not supported yet."
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    # for vla
                    moe_token_types,
                    start_indices,
                    end_indices,
                    probs,
                    row_id_map,
                    orig_shape,
                    adarms_conds,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    # for vla
                    token_types=moe_token_types,
                    start_indices=start_indices,
                    end_indices=end_indices,
                    probs=probs,
                    row_id_map=row_id_map,
                    orig_shape=orig_shape,
                    adarms_conds=adarms_conds,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                assert (
                    self.config.mot_opt is False
                ), "When using mot_opt, output_hidden_states is not supported yet."
                all_self_attns += (layer_outputs[1],)

        hidden_states, _, _ = self._apply_norm_moe(
            hidden_states,
            moe_token_types,
            adarms_conds,
            self.norms,
            self.norm,
            start_indices,
            end_indices,
            self.use_selective_recompute,
        )

        # add hidden states from the last decoder layer
        if output_hidden_states:
            assert (
                self.config.mot_opt is False
            ), "When using mot_opt, output_hidden_states is not supported yet."
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if self.config.mot_opt:
            hidden_states = ops.unpermute(hidden_states, row_id_map, probs)
            hidden_states = hidden_states.view(orig_shape)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
        moe_token_types: Optional[torch.LongTensor] = None,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = (
                    attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                )
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen2_5_VL. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if attention_mask.ndim == 2:
                if AttentionMaskConverter._ignore_causal_mask_sdpa(
                    attention_mask,
                    inputs_embeds=input_tensor,
                    past_key_values_length=past_seen_tokens,
                    sliding_window=self.config.sliding_window,
                    is_training=self.training,
                ):
                    return None
            elif attention_mask.ndim == 3:
                return attention_mask

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )
        # Modify the mask to support bidirectional attention.
        if moe_token_types is not None:
            # Find the positions of all tokens of type 1.
            type1_tokens = (
                (moe_token_types == 1).unsqueeze(1).unsqueeze(2)
            )  # [B, 1, 1, S]

            # Create a square mask for the type1 region.
            type1_mask = torch.zeros_like(causal_mask)  # [B, num_heads, S, S]
            type1_region = type1_tokens & type1_tokens.transpose(-1, -2)  # [B, 1, S, S]
            type1_mask = type1_mask.masked_fill(type1_region, 1.0).to(torch.bool)
            # Set the original causal_mask to zero in the type1 region, and then add the type1_mask.
            causal_mask = torch.where(
                type1_mask, 
                torch.zeros_like(causal_mask),
                causal_mask,
            )
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype
            )

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen2_5_VLConfig,
        past_key_values: Cache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen2_5_VLConfig`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            diagonal_attend_mask = torch.arange(
                target_length, device=device
            ) > cache_position.reshape(-1, 1)
            if config.sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if (
                    not isinstance(past_key_values, SlidingWindowCache)
                    or sequence_length > target_length
                ):
                    sliding_attend_mask = torch.arange(
                        target_length, device=device
                    ) <= (cache_position.reshape(-1, 1) - config.sliding_window)
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = (
                    causal_mask.clone()
                )  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[
                    :, None, None, :
                ].to(causal_mask.device)
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length
                ].masked_fill(padding_mask, min_dtype)
        return causal_mask


class Qwen2_5_VLMoEForAction(Qwen2_5_VLForConditionalGeneration, ActionGenerationMixin, ActionModelMixMin):
    """
    Qwen2.5 Vision-Language Mixture of Experts model for action processing.

    This model extends the base Qwen2.5 VL model with action token processing capabilities
    and optional LoRA fine-tuning support.
    """

    _tied_weights_keys = ["lm_head.weight"]
    config_class = Qwen2_5_VLConfig
    _no_split_modules = ["Qwen2_5_VLDecoderLayer_with_MoE", "Qwen2_5_VLVisionBlock"]

    @classmethod
    def _set_customized_config(cls, config):
        """
        Processing norm_stats.json and reconstruct the DoF mapping
        """
        dataload_config = config["data"]
        if not dataload_config.get("use_lerobot", False):
            raise NotImplementedError(
                "Not implemented for non-lerobot dataset currently"
            )

        enable_customized_robot_config = config.get(
            "enable_customized_robot_config", False
        )
        assert (
            enable_customized_robot_config
        ), "enable_customized_robot_config must be true when use lerobot dataset"

        customized_dof_config = config["customized_robot_config"][
            "customized_dof_config"
        ]
        customized_agent_pos_config = config["customized_robot_config"][
            "customized_agent_pos_config"
        ]
        norm_stats_path = config["norm_stats_path"]

        # Use the compute_action_statistics function from utils

        name = config["customized_robot_config"]["name"]

        update_action_statistics(
            action_statistic_dof=action_statistic_dof,  # Assuming this is a global variable
            norm_stats_path=norm_stats_path,
            repo_id=config["data"]["lerobot_config"]["repo_id"],
            robot_name=name,
            customized_dof_config=customized_dof_config,
            customized_agent_pos_config=customized_agent_pos_config,
        )

        print("Customized robot config added")
        pprint(action_statistic_dof)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path,
        train_config=None,
        config_path=None,
        processor_path=None,
        action_tokenizer_path=None,
        is_train=False,
        **kwargs,
    ):
        """
        Load model from pretrained model path.

        Args:
            pretrained_model_path (str): Model directory path containing model.safetensors file
            config_path (str, optional): Configuration file path, if None will look for qwen25_config.json in pretrained_model_path
            processor_path (str, optional): Processor path, if None will load from default config
            action_tokenizer_path (str, optional): Action tokenizer path, if None will load from default config
            **kwargs: Additional arguments

        Returns:
            Qwen2_5_VLMoEForAction: Loaded model instance
        """
        # Load model components from pretrained path

        if train_config is None:
            try:
                with open(os.path.join(pretrained_model_path, "config.yml"), "r") as f:
                    train_config = yaml.load(f, Loader=yaml.FullLoader)
            except Exception as e:
                print(f"load train_config.yml fail: {e}")
                train_config = None
        
        model_config_path = os.path.join(pretrained_model_path, "config.json")
        model_config = cls.config_class.from_pretrained(model_config_path)

        model_config = update_model_config(train_config, model_config)
        if not is_train:
            model_config._attn_implementation = "sdpa"
   
        if train_config is not None:
            processors_dict = load_wallx_processors(train_config)
            processor = processors_dict["processor"]
        else:
            processor = AutoProcessor.from_pretrained(model_config, use_fast=True)

        if action_tokenizer_path is not None:
            processor.action_processor = AutoProcessor.from_pretrained(
                action_tokenizer_path, trust_remote_code=True
            )

        # Set the customized robot configuration to ensure consistency between cross-embodiment
        # representations and the Wall-X action dimensionality.
        if not train_config:
            cls._set_customized_config(train_config)
            customized_dof_config = train_config["customized_robot_config"][
                "customized_dof_config"
            ]
            customized_agent_pos_config = train_config["customized_robot_config"][
                "customized_agent_pos_config"
            ]
            setattr(model_config, "customized_dof_config", customized_dof_config)
            setattr(model_config, "customized_agent_pos_config", customized_agent_pos_config)

        # Initialize model with configuration and processor
        model = cls(model_config, processor=processor, **kwargs)

        # Resize token embeddings to match processor tokenizer vocabulary size
        model.resize_token_embeddings(len(processor.tokenizer))

        # Load model state dict from safetensors file
        safetensor_files = glob.glob(
            os.path.join(pretrained_model_path, "*.safetensors")
        )
        state_dict = {}
        for file in safetensor_files:
            sd = load_file(file, device="cpu")
            # filter normalizer statistic params
            del_keys = []
            for key in sd.keys():
                if "action_preprocessor.normalizer" in key:
                    print(f"filter load model weight {key}")
                    del_keys.append(key)
            for key in del_keys:
                del sd[key]
            state_dict.update(sd)

        model.load_state_dict(state_dict, strict=False)

        return model

    def __init__(
        self,
        config,
        use_fast_tokenizer=False,
        processor=None,
        action_tokenizer=None,
        action_mapper=None,
        flow_loss_weight=1.0,
        use_selective_recompute=False,
    ):
        """
        Initialize the Qwen2.5 VLMoE model for action processing.

        Args:
            config: Model configuration
            use_fast_tokenizer (bool): Whether to use fast tokenizer
            processor: Text and image processor
            action_tokenizer: Action-specific tokenizer
            action_mapper: Action mapping utility
            flow_loss_weight (float): Weight for flow loss computation
        """
        super().__init__(config)

        # Initialize vision transformer and language model components
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(
            config.vision_config
        )
        self.model = Qwen2_5_VLMoEModel(config, use_selective_recompute=use_selective_recompute)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize loss function without reduction for channel-wise loss computation
        self.loss_fct = CrossEntropyLoss(reduction="none")
        self.flow_loss_weight = flow_loss_weight
        self.use_fast_tokenizer = use_fast_tokenizer
        self.processor = processor

        # Define action token IDs
        self.define_action_token_id()
        self.times_cache = {}  # cache times linspace for each num_inference_timesteps

        # Cache for rope deltas
        self.rope_deltas = None

        # Initialize action preprocessor
        self.action_preprocessor = ActionProcessor(config)

        # Apply LoRA if specified in configuration
        if hasattr(config, "use_lora") and config.use_lora:
            self.add_lora(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                lora_dropout=config.lora_dropout,
            )

        # Initialize weights and apply final processing
        self.post_init()

    def define_action_token_id(self):
        """
        Define action token IDs based on tokenizer configuration.

        Creates mappings for fast action tokens, proprioception tokens, and general action tokens.
        """
        # Create list of fast action token IDs
        fast_action_token_list = []
        if self.use_fast_tokenizer:
            for i in range(
                self.processor.tokenizer.init_kwargs["action_token_vocab_size"]
            ):
                action_token_id = self.processor.tokenizer.convert_tokens_to_ids(
                    f"<|action_token_{i}|>"
                )
                fast_action_token_list.append(action_token_id)

        # Get special action token IDs
        action_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|action|>")
        propri_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|propri|>")

        # Store action token ID mappings
        self.action_token_id_set = {
            "fast_action_token_list": fast_action_token_list,
            "propri_token_id": propri_token_id,
            "action_token_id": action_token_id,
        }

    def add_lora(
        self, r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.1
    ):
        """
        Add LoRA (Low-Rank Adaptation) adapters to the model.

        Args:
            r (int): Rank of adaptation
            lora_alpha (int): LoRA scaling parameter
            target_modules (list): List of module names to apply LoRA to
            lora_dropout (float): Dropout probability for LoRA layers
        """
        config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, config)

        # Print information about trainable parameters
        self.model.print_trainable_parameters()

    def get_input_embeddings(self):
        """Get input embeddings layer."""
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """Set input embeddings layer."""
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        """Get output embeddings layer."""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """Set output embeddings layer."""
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        """Set the decoder model."""
        self.model = decoder

    def get_decoder(self):
        """Get the decoder model."""
        return self.model
    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embeddin for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None
        ):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(
                    input_ids == vision_start_token_id
                ).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                    time_tensor = (
                        expanded_range
                        * second_per_grid_t
                        * self.config.vision_config.tokens_per_second
                    )

                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()

                    h_index = (
                        torch.arange(llm_grid_h)
                        .view(1, -1, 1)
                        .expand(llm_grid_t, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w)
                        .view(1, 1, -1)
                        .expand(llm_grid_t, llm_grid_h, -1)
                        .flatten()
                    )
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                    position_ids.device
                )
                mrope_position_deltas.append(
                    llm_positions.max() + 1 - len(total_input_ids[i])
                )
            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = (
                    position_ids.unsqueeze(0)
                    .expand(3, -1, -1)
                    .to(attention_mask.device)
                )
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                    -1, keepdim=True
                )[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas
    
    def train_step_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        # for vla
        moe_token_types: Optional[torch.LongTensor] = None,
        start_indices: Optional[torch.Tensor] = None,
        end_indices: Optional[torch.Tensor] = None,
        positional_masks: Optional[dict] = None,
        action_chunk: Optional[torch.FloatTensor] = None,
        proprioception: Optional[torch.FloatTensor] = None,
        dataset_names: Optional[str] = None,
        dof_mask: Optional[torch.FloatTensor] = None,
        agent_pos_mask: Optional[torch.FloatTensor] = None,
        flow_loss_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[Tuple, Qwen2_5_VLACausalLMOutputWithPast]:
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if start_indices is None or end_indices is None:
            # Calculate the start and end positions of each expert group's tokens after permutation
            group_size = torch.zeros(
                self.config.num_experts, dtype=torch.long, device="cpu"
            )
            for i in range(self.config.num_experts):
                group_size[i] = (moe_token_types == i).sum()

            # Calculate start and end indices for each expert group
            start_indices = torch.cumsum(group_size, dim=0) - group_size
            end_indices = torch.cumsum(group_size, dim=0)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (
            attention_mask is None or attention_mask.ndim == 2
        ):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                # batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(cache_position.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=cache_position.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            inputs_embeds = self.scatter_proprioception_embeddings(
                input_ids, inputs_embeds, proprioception, dataset_names, agent_pos_mask
            )

            inputs_embeds, flow, adarms_cond = self.scatter_flow_action_embeddings(
                input_ids, inputs_embeds, action_chunk, dataset_names, dof_mask
            )

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            moe_token_types=moe_token_types,
            start_indices=start_indices,
            end_indices=end_indices,
            positional_masks=positional_masks,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            adarms_conds=[None, adarms_cond],
            # cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        (
            loss,
            cross_entropy_loss,
            flow_loss,
            channel_loss_dict,
            channel_loss_count_dict,
        ) = self.compute_loss(
            hidden_states=hidden_states,
            logits=logits,
            input_ids=input_ids,
            dataset_names=dataset_names,
            labels=labels,
            action_chunk=action_chunk,
            dof_mask=dof_mask,
            flow=flow,
            flow_loss_mask=flow_loss_mask,
        )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLACausalLMOutputWithPast(
            loss=loss,
            cross_entropy_loss=(
                cross_entropy_loss.clone() if cross_entropy_loss is not None else None
            ),
            flow_loss=flow_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
            channel_loss_dict=channel_loss_dict,
            channel_loss_count_dict=channel_loss_count_dict,
        )

    def predict_action(self, predict_mode: str, **kwargs):
        """
        Predict actions using specified prediction mode.

        Args:
            predict_mode (str): Prediction mode, either "fast" or "diffusion"
            **kwargs: Additional arguments passed to the predict method

        Returns:
            tuple: (predicted_action, ground_truth_action) where ground_truth_action may be None
        """
        assert predict_mode in ["fast", "diffusion"]

        output = self.predict(predict_mode=predict_mode, **kwargs)

        return output["predict_action"], output.get("gt_action", None)

    @torch.no_grad()
    def predict(
        self,
        predict_mode: str,
        pred_horizon: Optional[int] = None,
        action_dim: Optional[int] = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        moe_token_types: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        action_chunk: Optional[torch.FloatTensor] = None,
        proprioception: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        num_inference_timesteps: Optional[int] = 10,
        dataset_names: Optional[str] = None,
        dof_mask: Optional[torch.FloatTensor] = None,
        agent_pos_mask: Optional[torch.FloatTensor] = None,
        re_generate: bool = False,
        **kwargs,
    ):
        """
        Multi-modal prediction method supporting text generation, fast action prediction, and diffusion-based action prediction.

        This method handles three prediction modes:
        1. "text": Pure text generation using autoregressive decoding
        2. "fast": Fast action prediction using discrete action tokens
        3. "diffusion": Continuous action prediction using diffusion/flow matching

        Args:
            predict_mode (str): Prediction mode ("text", "fast", or "diffusion")
            pred_horizon (int, optional): Prediction horizon for action sequences
            action_dim (int, optional): Dimensionality of action space
            input_ids (torch.LongTensor, optional): Input token IDs
            attention_mask (torch.Tensor, optional): Attention mask for input tokens
            position_ids (torch.LongTensor, optional): Position IDs for tokens
            past_key_values (List[torch.FloatTensor], optional): Cached key-value pairs
            inputs_embeds (torch.FloatTensor, optional): Pre-computed input embeddings
            moe_token_types (torch.LongTensor, optional): Token type assignments for MoE routing
            labels (torch.LongTensor, optional): Target labels for evaluation
            use_cache (bool, optional): Whether to use key-value caching
            output_attentions (bool, optional): Whether to return attention weights
            output_hidden_states (bool, optional): Whether to return hidden states
            return_dict (bool, optional): Whether to return structured output
            pixel_values (torch.Tensor, optional): Image pixel values
            pixel_values_videos (torch.FloatTensor, optional): Video pixel values
            image_grid_thw (torch.LongTensor, optional): Image grid dimensions
            video_grid_thw (torch.LongTensor, optional): Video grid dimensions
            action_chunk (torch.FloatTensor, optional): Ground truth action sequences
            proprioception (torch.FloatTensor, optional): Proprioceptive sensor data
            rope_deltas (torch.LongTensor, optional): RoPE position deltas
            cache_position (torch.LongTensor, optional): Cache position indices
            second_per_grid_ts (torch.Tensor, optional): Time interval per temporal grid
            num_inference_timesteps (int, optional): Number of diffusion inference steps
            dataset_names (str, optional): Dataset names for normalization
            dof_mask (torch.FloatTensor, optional): Degrees of freedom mask
            agent_pos_mask (torch.FloatTensor, optional): Agent position mask
            re_generate (bool, optional): Whether to use sampling for regeneration
            **kwargs: Additional keyword arguments

        Returns:
            dict: Dictionary containing prediction results with keys like:
                - 'predict_action': Predicted action sequences
                - 'gt_action': Ground truth actions (if available)
                - 'input_text': Input text (for text/fast modes)
                - 'predict_output_text': Generated text (for text/fast modes)
                - 'gt_output_text': Ground truth text (for text/fast modes)
        """
        batch_size = (
            input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        )

        # Text and fast modes require batch size 1 for autoregressive generation
        if predict_mode in ["text", "fast"]:
            assert (
                batch_size == 1
            ), "predict only support batch size 1 for ar generation"

        # Set output configuration from model config if not specified
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Process input embeddings with multi-modal data
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

            # Process image embeddings
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]

                # Validate image token and feature count match
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            # Process video embeddings
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]

                # Validate video token and feature count match
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            # Process proprioceptive data
            if proprioception is not None:
                proprioception = proprioception.to(inputs_embeds.device).to(
                    inputs_embeds.dtype
                )
                agent_pos_mask = agent_pos_mask.to(inputs_embeds.device).to(
                    inputs_embeds.dtype
                )
                proprio_embed = self.action_preprocessor.proprioception_proj(
                    proprioception,
                    dataset_names,
                    agent_pos_mask,
                    use_history=proprioception.shape[1] > 1,
                )
                proprioception_mask = (
                    input_ids == self.action_token_id_set["propri_token_id"]
                )
                inputs_embeds[proprioception_mask] = proprio_embed.reshape(
                    -1, inputs_embeds.shape[-1]
                ).to(inputs_embeds.dtype)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # Calculate RoPE position IDs if not provided
        # Note: Cannot calculate rope deltas with 4D attention mask. TODO: Fix this limitation
        if position_ids is None and (
            attention_mask is None or attention_mask.ndim == 2
        ):
            # Calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = ops.get_rope_index(
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                    spatial_merge_size=self.config.vision_config.spatial_merge_size,
                    image_token_id=self.config.image_token_id,
                    video_token_id=self.config.video_token_id,
                    vision_start_token_id=self.config.vision_start_token_id,
                    tokens_per_second=self.config.vision_config.tokens_per_second,
                )
                self.rope_deltas = rope_deltas
            # Use previously calculated rope deltas to get correct position IDs
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # Prepare action chunk data if provided
        if action_chunk is not None:
            action_chunk = action_chunk.to(inputs_embeds.device).to(inputs_embeds.dtype)

        output = {}

        # Split input sequence for text and fast modes (not needed for diffusion)
        if predict_mode == "text" or predict_mode == "fast":
            # Look for generation prompt tokens: <|im_start|>assistant
            generation_prompt_ids = torch.tensor(
                [151644, 77091], device=input_ids.device, dtype=input_ids.dtype
            )
            matches = (input_ids[0, :-1] == generation_prompt_ids[0]) & (
                input_ids[0, 1:] == generation_prompt_ids[1]
            )

            if matches.any():
                split_pos = torch.nonzero(matches, as_tuple=True)[0][0].item()
                # Extract ground truth output tokens (including newline)
                gt_output_ids = input_ids[:, split_pos + 3 :]
                # Remove output part from input, keeping prompt
                input_ids = input_ids[:, : split_pos + 3]
                inputs_embeds = inputs_embeds[:, : split_pos + 3, :]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, : split_pos + 3]
                if labels is not None:
                    labels = labels[:, split_pos + 3 :]
            else:
                raise Warning(
                    "input_ids does not contain the generation prompt tokens <|im_start|>assistant"
                )

            # Decode input text for output
            input_text = self.processor.batch_decode(
                input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True
            )
            output["input_text"] = input_text

        # Handle text and fast prediction modes using autoregressive generation
        if predict_mode == "text" or predict_mode == "fast":
            # Initialize MoE token types for generation
            moe_token_types = torch.zeros_like(input_ids)
            batch = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "moe_token_types": moe_token_types,
                "image_grid_thw": image_grid_thw,
                "dof_mask": dof_mask,
                "agent_pos_mask": agent_pos_mask,
                "proprioception": proprioception,
                "dataset_names": dataset_names,
            }

            # Generate output tokens
            predict_output_ids = self.generate(
                **batch,
                max_new_tokens=100,
                eos_token_id=[self.processor.tokenizer.eos_token_id],
                use_cache=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                temperature=(
                    1.0 if not re_generate else 0.7
                ),  # Higher temperature for regeneration
                do_sample=(
                    False if not re_generate else True
                ),  # Enable sampling for regeneration
            )

            # Decode generated and ground truth text
            gt_output_text = self.processor.batch_decode(
                gt_output_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            predict_output_text = self.processor.batch_decode(
                predict_output_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            output["gt_output_text"] = gt_output_text
            output["predict_output_text"] = predict_output_text

        # Convert tokens to actions for fast prediction mode
        if predict_mode == "fast":
            action_id = []
            # Extract action tokens from generated sequence
            for token_id_i in predict_output_ids[0]:
                if (
                    token_id_i.item()
                    >= self.processor.tokenizer.init_kwargs["action_token_start_index"]
                ):
                    action_id.append(
                        token_id_i.item()
                        - self.processor.tokenizer.init_kwargs[
                            "action_token_start_index"
                        ]
                    )

            predict_action = self.processor.action_processor.decode(
                [action_id], time_horizon=pred_horizon, action_dim=action_dim
            )
            # Handle action decoding errors
            if np.sum(predict_action) == 0:
                print("Error in decoding action, predict_action is None")
                output["predict_action"] = None
            else:
                # Convert discrete tokens to continuous actions
                predict_action = torch.tensor(predict_action, device=self.device)
                dof_mask = dof_mask.to(self.device).to(pixel_values.dtype)
                predict_action = (
                    self.action_preprocessor.normalizer_action.unnormalize_data(
                        predict_action, dataset_names, dof_mask
                    )
                )
                output["predict_action"] = predict_action

            # Process ground truth actions if available
            if action_chunk is not None:
                # Apply DOF mask and unnormalize action chunk to get ground truth actions
                action_chunk = action_chunk[:, :, dof_mask[0, 0, :].bool()]
                output["gt_action"] = (
                    self.action_preprocessor.normalizer_action.unnormalize_data(
                        action_chunk, dataset_names, dof_mask
                    )
                )
            else:
                output["gt_action"] = None

        # Handle diffusion-based action prediction
        if predict_mode == "diffusion":
            # Initialize with random noise
            noisy_action = torch.randn(
                size=(batch_size, pred_horizon, action_dim),
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )
            dof_mask = dof_mask.to(inputs_embeds.device).to(inputs_embeds.dtype)

            # Calculate token distribution across MoE expert groups
            group_size = torch.zeros(
                self.config.num_experts, dtype=torch.long, device="cpu"
            )
            for i in range(self.config.num_experts):
                group_size[i] = (moe_token_types == i).sum()

            # Calculate start and end indices for each expert group
            start_indices = torch.cumsum(group_size, dim=0) - group_size
            end_indices = torch.cumsum(group_size, dim=0)

            def step(timestep, noisy_action):
                """
                Single denoising step for diffusion process.

                Args:
                    timestep: Current diffusion timestep
                    noisy_action: Current noisy action estimate

                Returns:
                    torch.Tensor: Predicted clean action
                """
                action_mask = input_ids == self.action_token_id_set["action_token_id"]
                assert action_mask.any(), "No action token found in input_ids"

                # Prepare timestep for batch processing
                timestep = timestep.unsqueeze(0).repeat(noisy_action.shape[0])
                action_embed,_ = self.action_preprocessor.step(
                    timestep=timestep, noisy_action=noisy_action, dof_mask=dof_mask
                )
                action_embed = action_embed.reshape(-1, inputs_embeds.shape[-1])

                # Create temporary copy of embeddings for thread safety
                temp_inputs_embeds = inputs_embeds.clone()
                temp_inputs_embeds[action_mask] = action_embed.to(temp_inputs_embeds.dtype)

                # Forward pass through transformer
                transformer_outputs = self.model(
                    input_ids=None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=temp_inputs_embeds,
                    moe_token_types=moe_token_types,
                    start_indices=start_indices,
                    end_indices=end_indices,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )

                # Extract action predictions from hidden states
                hidden_states = transformer_outputs.last_hidden_state
                action_mask = input_ids == self.action_token_id_set["action_token_id"]
                action_hidden_states = hidden_states[action_mask]
                pred = self.action_preprocessor.action_proj_back(action_hidden_states[:, : self.action_preprocessor.action_hidden_size])
                return pred.reshape(batch_size, pred_horizon, action_dim)

            # Perform ODE integration for diffusion sampling
            times = torch.linspace(
                0,
                1,
                num_inference_timesteps + 1,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            action_trajectory = odeint(step, noisy_action.to(torch.float32), times.to(torch.float32), method="euler")

            # Extract final predicted action and unnormalize
            predict_action = action_trajectory[-1]
            predict_action = (
                self.action_preprocessor.normalizer_action.unnormalize_data(
                    predict_action, dataset_names
                )
            )
            output["predict_action"] = predict_action

            # Process ground truth actions if available
            if action_chunk is not None:
                output["gt_action"] = (
                    self.action_preprocessor.normalizer_action.unnormalize_data(
                        action_chunk, dataset_names
                    )
                )

        return output

    @torch.no_grad()
    def generate_flow_action(
        self,
        input_ids,
        action_horizon,
        action_dim,
        num_inference_timesteps: int = 10,
        padding_action: Optional[torch.Tensor] = None,
        prefix_length: Optional[int] = None,  # 前缀长度
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        moe_token_types: Optional[torch.LongTensor] = None,
        start_indices: Optional[torch.Tensor] = None,
        end_indices: Optional[torch.Tensor] = None,
        positional_masks: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        action_chunk: Optional[torch.FloatTensor] = None,
        proprioception: Optional[torch.FloatTensor] = None,
        unnorm_proprioception: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        dataset_names: Optional[str] = None,
        dof_mask: Optional[torch.FloatTensor] = None,
        agent_pos_mask: Optional[torch.FloatTensor] = None,
        unnorm: Optional[bool] = True,
        **kwargs,
    ):

        total_start_time = time.time()
        timing_results = {}

        batch_size = (
            input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        embed_start_time = time.time()
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if (
                proprioception is not None
                and not self.config.use_state_string_representation
            ):
                proprioception = proprioception.to(inputs_embeds.device)
                agent_pos_mask = agent_pos_mask.to(inputs_embeds.device)
                proprio_embed = self.action_preprocessor.proprioception_proj(
                    proprioception,
                    dataset_names,
                    agent_pos_mask,
                    use_history=proprioception.shape[1] > 1,
                )
                proprioception_mask = (
                    input_ids == self.action_token_id_set["propri_token_id"]
                )
                inputs_embeds[proprioception_mask] = proprio_embed.reshape(
                    -1, inputs_embeds.shape[-1]
                ).to(inputs_embeds.dtype)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        timing_results["embed_processing"] = time.time() - embed_start_time

        position_start_time = time.time()
        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (
            attention_mask is None or attention_mask.ndim == 2
        ):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if start_indices is None or end_indices is None:
            # Calculate the start and end positions of each expert group's tokens after permutation (the dataset does not contain `num_expert` information, so this calculation must be done here).
            group_size = torch.zeros(
                self.config.num_experts, dtype=torch.long, device="cpu"
            )
            for i in range(self.config.num_experts):
                group_size[i] = (moe_token_types == i).sum()

            # Calculate start and end indices for each expert group
            start_indices = torch.cumsum(group_size, dim=0) - group_size
            end_indices = torch.cumsum(group_size, dim=0)

        timing_results["position_encoding"] = time.time() - position_start_time

        action_init_start_time = time.time()
        if action_chunk is not None:
            action_chunk = action_chunk.to(inputs_embeds.device).to(torch.float32)

        output = {}
        # Reproduce
        # torch.manual_seed(0)
        noise = torch.randn(
            size=(batch_size, action_horizon, action_dim),
            dtype=torch.float32,
            device=inputs_embeds.device,
        )
        noisy_action = noise.clone()
        dof_mask = dof_mask.to(inputs_embeds.device).to(torch.float32)

        if num_inference_timesteps not in self.times_cache:
            self.times_cache[num_inference_timesteps] = torch.linspace(
                0.0,
                1.0,
                num_inference_timesteps + 1,
                device=inputs_embeds.device,
                dtype=torch.float32,
            )
        times = self.times_cache[num_inference_timesteps]
        dt = times[1] - times[0]
        time_0 = times[0].unsqueeze(0).repeat(noisy_action.shape[0])
        action_embed, adarms_cond = self.action_preprocessor.step(
            timestep=time_0, noisy_action=noisy_action, dof_mask=dof_mask
        )
        action_embed = action_embed.reshape(-1, inputs_embeds.shape[-1]).to(
            inputs_embeds.dtype
        )
        flow_action_mask = input_ids == self.action_token_id_set["action_token_id"]


        inputs_embeds[flow_action_mask] = action_embed

        timing_results["action_initialization"] = time.time() - action_init_start_time

        prefetch_start_time = time.time()
        prefetch_output = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            moe_token_types=moe_token_types,
            start_indices=start_indices,
            end_indices=end_indices,
            positional_masks=positional_masks,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            adarms_conds=[None, adarms_cond],
        )
        hidden_states = prefetch_output.last_hidden_state
        prefix_kv_cache = prefetch_output.past_key_values

        action_hidden_states = hidden_states[flow_action_mask].to(torch.float32)
        action_pred = self.action_preprocessor.action_proj_back(
            action_hidden_states[:, : self.action_preprocessor.action_hidden_size]
        )
        if getattr(self.config, "use_x_pred", False): 
            v_0 = action_pred - noise.reshape(-1, noise.shape[-1])
        else:
            v_0 = action_pred

        if (not dof_mask.all()) and (padding_action is not None):
            print("use padding action", flush=True)
            v_padding = padding_action - noisy_action
            v_0 = (v_padding) * (1 - dof_mask) + v_0 * dof_mask

        noisy_action = noisy_action + dt * v_0.reshape(
            batch_size, action_horizon, action_dim
        )

        timing_results["prefetch_forward"] = time.time() - prefetch_start_time

        cache_prep_start_time = time.time()

        if prefix_length is None:
            has_true = flow_action_mask.any(dim=1)
            prefix_length = torch.argmax(flow_action_mask.float(), dim=1, keepdim=True)
            prefix_length[~has_true] = flow_action_mask.shape[1]
            prefix_length = prefix_length[0]

        # support different transformers version （ryan: 1018）
        if hasattr(prefix_kv_cache, "key_cache"):
            for layer_i in range(len(prefix_kv_cache.key_cache)):
                prefix_kv_cache.key_cache[layer_i] = prefix_kv_cache.key_cache[layer_i][
                    :, :, :prefix_length, :
                ]
                prefix_kv_cache.value_cache[layer_i] = prefix_kv_cache.value_cache[
                    layer_i
                ][:, :, :prefix_length, :]
        else:
            for layer_i in range(len(prefix_kv_cache.layers)):
                prefix_kv_cache.layers[layer_i].keys = prefix_kv_cache.layers[
                    layer_i
                ].keys[:, :, :prefix_length, :]
                prefix_kv_cache.layers[layer_i].values = prefix_kv_cache.layers[
                    layer_i
                ].values[:, :, :prefix_length, :]

        postfix_position_ids = position_ids[:, :, prefix_length:]
        postfix_inputs_embeds = inputs_embeds[:, prefix_length:, :]
        postfix_attention_mask = attention_mask[:, prefix_length:]
        postfix_moe_token_types = moe_token_types[:, prefix_length:]
        postfix_input_ids = input_ids[:, prefix_length:]

        group_size = torch.zeros(
            self.config.num_experts, dtype=torch.long, device="cpu"
        )
        for i in range(self.config.num_experts):
            group_size[i] = (postfix_moe_token_types == i).sum()

        # Calculate start and end indices for each expert group
        postfix_start_indices = torch.cumsum(group_size, dim=0) - group_size
        postfix_end_indices = torch.cumsum(group_size, dim=0)

        pad_token_id = self.processor.tokenizer.pad_token_id
        padding_mask = input_ids == pad_token_id

        # prefix_length, postfix_length = prefix_indices.shape[0], postfix_indices.shape[0]

        postfix_length = input_ids.shape[-1] - prefix_length
        _postfix_attention_mask = torch.ones(
            (batch_size, postfix_length, prefix_length + postfix_length),
            dtype=torch.bool,
            device=postfix_attention_mask.device,
        )

        # Use a padding mask to set the corresponding rows and columns to false.
        # Get the padding mask for the postfix portion.
        postfix_padding_mask = padding_mask[
            :, prefix_length:
        ]  # [batch_size, postfix_length]
        full_padding_mask = padding_mask  # [batch_size, prefix_length + postfix_length]

        # causal mask for postfix attention
        if self.config.causal_action_attention_mask:
            _postfix_attention_mask[:, :, prefix_length:] = torch.tril(
                torch.ones(
                    (postfix_length, postfix_length),
                    dtype=torch.bool,
                    device=postfix_attention_mask.device,
                )
            )

        for batch_idx in range(padding_mask.shape[0]):
            # Set the rows corresponding to the padding positions to False (where the query position is padding).
            _postfix_attention_mask[batch_idx, postfix_padding_mask[batch_idx], :] = (
                False
            )
            # Set the columns corresponding to the padding positions to False (the key position is the padding).
            _postfix_attention_mask[batch_idx, :, full_padding_mask[batch_idx]] = False

        timing_results["cache_preprocessing"] = time.time() - cache_prep_start_time

        ode_start_time = time.time()

        def step_with_kvcache(timestep, noisy_action):
            action_mask = (
                postfix_input_ids == self.action_token_id_set["action_token_id"]
            )
            assert action_mask.any(), "No action token found in input_ids"
            timestep = timestep.unsqueeze(0).repeat(noisy_action.shape[0])
            action_embed, adarms_cond = self.action_preprocessor.step(
                timestep=timestep, noisy_action=noisy_action, dof_mask=dof_mask
            )
            action_embed = action_embed.reshape(-1, postfix_inputs_embeds.shape[-1])

            temp_inputs_embeds = postfix_inputs_embeds.clone()
            temp_inputs_embeds[action_mask] = action_embed.to(temp_inputs_embeds.dtype)
            transformer_outputs = self.model(
                input_ids=None,
                attention_mask=_postfix_attention_mask,
                position_ids=postfix_position_ids,
                past_key_values=prefix_kv_cache,
                inputs_embeds=temp_inputs_embeds,
                moe_token_types=postfix_moe_token_types,
                start_indices=postfix_start_indices,
                end_indices=postfix_end_indices,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                adarms_conds=[None, adarms_cond],
            )

            hidden_states = transformer_outputs.last_hidden_state
            action_hidden_states = hidden_states[action_mask].to(torch.float32)
            action_pred = self.action_preprocessor.action_proj_back(
                action_hidden_states[:, : self.action_preprocessor.action_hidden_size]
            )
            if getattr(self.config, "use_x_pred", False):
                v_t = action_pred - noise.reshape(-1, noise.shape[-1])
            else:
                v_t = action_pred
            return v_t.reshape(batch_size, action_horizon, action_dim)


        action_trajectory = odeint(
            step_with_kvcache, noisy_action, times[1:], method="euler"
        )

        timing_results["ode_integration"] = time.time() - ode_start_time

        postprocess_start_time = time.time()
        predict_action = action_trajectory[-1]
        if unnorm:
            predict_action = self.action_preprocessor.normalizer_action.unnormalize_data(
                predict_action, dataset_names
            )
        output["predict_action"] = predict_action
        # normalize action chunk to get gt_action
        if action_chunk is not None:
            output["gt_action"] = (
                self.action_preprocessor.normalizer_action.unnormalize_data(
                    action_chunk, dataset_names
                )
            )

        timing_results["postprocessing"] = time.time() - postprocess_start_time
        timing_results["total_time"] = time.time() - total_start_time

        output["timing_results"] = timing_results

        return output

    def forward(
        self, mode: Optional[str] = None, predict_mode: Optional[str] = "text", **kwargs
    ):
        """
        Main forward pass dispatcher for different execution modes.

        This method routes execution to appropriate forward functions based on the specified mode:
        - No mode (None): Training step with gradient disabled
        - 'predict': Prediction/inference mode
        - 'train': Training mode with gradients enabled
        - 'validate': Validation mode with gradients disabled

        Args:
            mode (str, optional): Execution mode. If None, defaults to training step without gradients
            predict_mode (str, optional): Prediction mode for 'predict' mode ("text", "fast", or "diffusion")
            **kwargs: Additional arguments passed to the selected forward function

        Returns:
            Model outputs appropriate for the selected mode

        Todo:
            - Add support for distinguishing multi-modal data types in prediction mode
        """
        if not mode:
            with torch.no_grad():
                return self.train_step_forward(**kwargs)
        elif mode == "predict":
            return self.generate_flow_action(predict_mode=predict_mode, **kwargs)
        elif mode == "train":
            return self.train_step_forward(use_cache=False, **kwargs)
        elif mode == "validate":
            with torch.no_grad():
                return self.train_step_forward(use_cache=False, **kwargs)
        else:
            raise NotImplementedError("invalid key")

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        moe_token_types=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        dataset_names=None,
        proprioception=None,
        dof_mask=None,
        agent_pos_mask=None,
        **kwargs,
    ):
        """
        Prepare inputs for autoregressive generation with multi-modal support.

        This method handles input preparation for generation, including proper slicing of inputs
        based on cache position, MoE token type management, and multi-modal data handling.
        Vision inputs are selectively forwarded only when needed during generation.

        Args:
            input_ids: Input token IDs
            past_key_values: Cached key-value pairs from previous generation steps
            attention_mask: Attention mask for input tokens
            inputs_embeds: Pre-computed input embeddings
            moe_token_types: Token type assignments for MoE routing
            cache_position: Current cache position for generation
            position_ids: Position IDs for tokens
            use_cache: Whether to use key-value caching
            pixel_values: Image pixel values
            pixel_values_videos: Video pixel values
            image_grid_thw: Image grid dimensions
            video_grid_thw: Video grid dimensions
            second_per_grid_ts: Time interval per temporal grid
            dataset_names: Dataset names for processing
            proprioception: Proprioceptive sensor data
            dof_mask: Degrees of freedom mask
            agent_pos_mask: Agent position mask
            **kwargs: Additional arguments

        Returns:
            dict: Prepared model inputs for generation step

        Todo:
            - Test this function thoroughly with various input configurations

        Note:
            This is an overridden method that handles specific cases for multi-modal generation:
            - Slices input_ids through cache_position to keep only unprocessed tokens
            - Handles special cases for input_embeds, generation methods, and GPU synchronization
            - Manages vision inputs to avoid unnecessary forward passes
        """
        # Initialize MoE token types if not provided
        if moe_token_types is None:
            moe_token_types = torch.zeros_like(
                input_ids
            )  # FIXME: Handle case when input_embeds is used instead
        else:
            # Ensure moe_token_types length matches input_ids
            if moe_token_types.shape[1] < input_ids.shape[1]:
                # Calculate required padding length
                pad_length = input_ids.shape[1] - moe_token_types.shape[1]
                # Create padding tensor with default token type (0)
                pad_tensor = torch.zeros(
                    (moe_token_types.shape[0], pad_length),
                    dtype=moe_token_types.dtype,
                    device=moe_token_types.device,
                )
                # Concatenate padding to existing moe_token_types
                moe_token_types = torch.cat([moe_token_types, pad_tensor], dim=1)

        # Handle input slicing based on cache state and special cases
        if past_key_values is not None:
            if (
                inputs_embeds is not None and input_ids.shape[1] == 0
            ):  # Exception 4: input_embeds case
                inputs_embeds = inputs_embeds[:, -cache_position.shape[0] :]
                moe_token_types = moe_token_types[:, -cache_position.shape[0] :]
            elif inputs_embeds is not None or (  # Exception 1: input_embeds provided
                is_torchdynamo_compiling() or cache_position[-1] >= input_ids.shape[1]
            ):  # Exception 3: GPU sync edge case
                input_ids = input_ids[:, -cache_position.shape[0] :]
                moe_token_types = moe_token_types[:, -cache_position.shape[0] :]
            elif (
                input_ids.shape[1] != cache_position.shape[0]
            ):  # Default case (Exception 2 is no-op)
                cache_pos = cache_position.clone()
                input_ids = input_ids[:, cache_pos]
                moe_token_types = moe_token_types[:, cache_pos]

        # Skip vision inputs for continuation steps (not initial generation)
        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None

        # Determine whether to use inputs_embeds or input_ids for this generation step
        if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        # Prepare 4D causal attention mask for static cache
        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            attention_mask = (
                self.model._prepare_4d_causal_attention_mask_with_cache_position(
                    attention_mask,
                    sequence_length=sequence_length,
                    target_length=past_key_values.get_max_cache_shape(),
                    dtype=self.lm_head.weight.dtype,
                    device=device,
                    cache_position=cache_position,
                    batch_size=batch_size,
                    config=self.config,
                    past_key_values=past_key_values,
                )
            )

        # Assemble all model inputs for generation
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "moe_token_types": moe_token_types,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "cache_position": cache_position,
                "second_per_grid_ts": second_per_grid_ts,
                "proprioception": proprioception,
                "dataset_names": dataset_names,
                "dof_mask": dof_mask,
                "agent_pos_mask": agent_pos_mask,
            }
        )
        return model_inputs

    def _get_image_nums_and_video_nums(
        self,
        input_ids: Optional[torch.LongTensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the number of images and videos for each sample to calculate tensor separation lengths.

        These parameters are computed directly from input_ids rather than being passed through
        the processor to avoid unpredictable impacts from interface modifications.

        Args:
            input_ids (torch.LongTensor): Input token IDs of shape (batch_size, sequence_length)

        Returns:
            tuple:
                - image_nums (torch.LongTensor): Number of images per sample
                - video_nums (torch.LongTensor): Number of videos per sample
        """
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        # Find vision start tokens and their following tokens
        vision_start_mask = input_ids == vision_start_token_id
        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_mask = input_ids == image_token_id
        video_mask = input_ids == video_token_id

        # Count images and videos following vision start tokens
        image_nums = torch.sum(vision_first_mask & image_mask, dim=1)
        video_nums = torch.sum(vision_first_mask & video_mask, dim=1)

        return image_nums, video_nums

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """
        Expand inputs for generation with support for multi-modal tensors.

        This is an overridden method that supports expanding tensors without a standard batch
        size dimension, specifically for vision-related tensors:
        - pixel_values.shape[0] = sum(sequence_lengths for all image samples)
        - image_grid_thw.shape[0] = sum(num_images for all samples)
        - Similar patterns for video tensors

        Args:
            expand_size (int): Factor by which to expand inputs (for beam search, etc.)
            is_encoder_decoder (bool): Whether using encoder-decoder architecture
            input_ids (torch.LongTensor, optional): Input token IDs
            **model_kwargs: Additional model arguments to expand

        Returns:
            tuple: (expanded_input_ids, expanded_model_kwargs)
        """
        if expand_size == 1:
            return input_ids, model_kwargs

        # Define keys for vision-related tensors that need special handling
        visual_keys = [
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "second_per_grid_ts",
        ]

        def _expand_dict_for_generation_visual(dict_to_expand):
            """Expand vision-related tensors based on image/video counts per sample."""
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(input_ids)

            def _repeat_interleave_samples(x, lengths, repeat_times):
                """Split tensor by lengths and repeat each sample."""
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat(
                    [sample.repeat(*repeat_args) for sample in samples], dim=0
                )
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # Split images into samples and compute sequence lengths
                    samples = torch.split(image_grid_thw, list(image_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # Expand based on number of images per sample
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    # Split videos into samples and compute sequence lengths
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    # Expand based on number of videos per sample
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "second_per_grid_ts":
                    # Handle list-type temporal grid data
                    if not isinstance(dict_to_expand[key], list):
                        raise TypeError(
                            f"Expected value for key '{key}' to be a list, but got {type(dict_to_expand[key])} instead."
                        )
                    tensor = torch.tensor(dict_to_expand[key])
                    lengths = list(video_nums)
                    tensor = _repeat_interleave_samples(
                        tensor, lengths=lengths, repeat_times=expand_size
                    )
                    dict_to_expand[key] = tensor.tolist()
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            """Expand standard tensors using repeat_interleave."""
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(
                        expand_size, dim=0
                    )
            return dict_to_expand

        # Expand visual inputs only if input_ids is available for counting images/videos
        # If input_ids is unavailable, visual inputs won't be used, so no expansion needed
        if input_ids is not None and input_ids.numel() != 0:
            model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        # Expand input_ids using standard repeat_interleave
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        # Expand all other model arguments
        model_kwargs = _expand_dict_for_generation(model_kwargs)

        # Handle encoder-decoder specific expansion
        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError(
                    "If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined."
                )
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(
                model_kwargs["encoder_outputs"]
            )

        return input_ids, model_kwargs
