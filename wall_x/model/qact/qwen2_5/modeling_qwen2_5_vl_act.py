import re
import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.nn import CrossEntropyLoss
from typing import Optional, List, Tuple
from torchdiffeq import odeint
import numpy as np
from typing import Any, Dict, Union
from wall_x.model.qact.qwen2_5.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from wall_x.model.qact.qwen2_5.modeling_qwen2_5_vl import (
    QWEN2_5_VL_ATTENTION_CLASSES,
    Qwen2_5_VLMLP,
    Qwen2_5_VLRotaryEmbedding,
    Qwen2_5_VLPreTrainedModel,
    Qwen2RMSNorm,
    Qwen2_5_VLPatchMerger,
    Qwen2_5_VLVisionBlock,
    Qwen2_5_VisionTransformerPretrainedModel,
)

from transformers import AutoConfig

from transformers.cache_utils import (
    Cache,
    DynamicCache,
    StaticCache,
)

try:
    from transformers.cache_utils import SlidingWindowCache
except ImportError:
    # Compatibility with newer transformers versions
    SlidingWindowCache = StaticCache

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    ModelOutput,
)
from transformers.utils import (
    logging,
    is_torchdynamo_compiling,
)
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
)
from wall_x.model.core.vla_mixin import ActionGenerationMixin, ActionModelMixMin
from wall_x.model.core.action.normalizer import (
    normalize_data_with_virtual_tail,
    unnormalize_data_with_virtual_tail,
)
from wall_x.model.core.action.moe import TokenTypeRouter, SparseMoeBlock
from wall_x.model.core.attention.selector import (
    ATTENTION_TYPES_WITH_2D_MASK,
    ATTENTION_TYPES_WITH_FLASH_MASK,
)
from wall_x.model.core.attention.joint import JOINT_QWEN_ATTENTION_CLASSES
from wall_x.model.qact.qwen2_5.inference_mixin import VLAInferenceMixin

from wall_x.model.core.ops import unpermute, permute, get_rope_index

logger = logging.get_logger(__name__)

_QWEN25_DMUON_BLOCKED_NAME_PARTS = (
    "embed_tokens",
    "pos_embed",
    "lm_head",
    "norm",
)


def _is_qwen25_dmuon_target_param(name: str, param: nn.Parameter) -> bool:
    """Select trainable Qwen2.5 VLA matrices for DMuon."""
    if not param.requires_grad or param.ndim != 2 or not name.endswith(".weight"):
        return False
    return not any(part in name for part in _QWEN25_DMUON_BLOCKED_NAME_PARTS)


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
        hidden_states, gate, gate_mask = self._apply_norm_moe(
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
        # If num_experts is provided, make sure it is added to config
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
        moe_token_types: Optional[torch.LongTensor] = None,  # new parameter
        start_indices: Optional[torch.Tensor] = None,
        end_indices: Optional[torch.Tensor] = None,
        positional_masks: Optional[
            dict
        ] = None,  # stores token position masks needed by each category
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
            raise ValueError("moe_token_types must be provided for MoE routing")
        if start_indices is None or end_indices is None:
            raise ValueError(
                "start_indices and end_indices must be provided for MoE routing"
            )

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

        # When mot_opt is enabled, permute tokens from different experts first; shape becomes [Tokens, HiddenSize]
        orig_shape = hidden_states.shape
        moe_token_types = moe_token_types.contiguous()
        if self.config.mot_opt:
            hidden_states = hidden_states.view(-1, hidden_states.size(-1))
            hidden_states, row_id_map = permute(hidden_states, moe_token_types.view(-1))
        else:
            row_id_map = None
        # Use reshape instead of view to support non-contiguous tensors
        probs = torch.ones_like(moe_token_types, dtype=torch.float32).reshape(-1, 1)

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

        if (
            self.config._attn_implementation in ATTENTION_TYPES_WITH_FLASH_MASK
            and self.config.attention_moe is True
        ):
            causal_mask = self._update_joint_attention_flash_mask(
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
            hidden_states = unpermute(hidden_states, row_id_map, probs)
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
        # Modify the mask to support bidirectional attention
        if moe_token_types is not None:
            # Find all token positions with type 1
            type1_tokens = (
                (moe_token_types == 1).unsqueeze(1).unsqueeze(2)
            )  # [B, 1, 1, S]

            # Create a square mask for the type-1 region
            type1_mask = torch.zeros_like(causal_mask)  # [B, num_heads, S, S]
            type1_region = type1_tokens & type1_tokens.transpose(-1, -2)  # [B, 1, S, S]
            type1_mask = type1_mask.masked_fill(type1_region, 1.0).to(torch.bool)
            # Zero the original causal_mask in the type-1 region, then add type1_mask
            causal_mask = torch.where(
                type1_mask,  # Expand dimensions to match causal_mask
                torch.zeros_like(causal_mask),  # zero the type-1 region
                causal_mask,  # keep other regions unchanged
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


class Qwen2_5_VLMoEForAction(
    Qwen2_5_VLPreTrainedModel,
    ActionGenerationMixin,
    ActionModelMixMin,
    VLAInferenceMixin,
):
    # Compatibility with newer transformers versions (5.x):_tied_weights_keys changed from list[str] to dict[str, str] (target -> source mapping).
    # Older versions (4.x) iterate over dict keys, so the behavior is equivalent and compatible.
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    config_class = Qwen2_5_VLConfig
    _no_split_modules = ["Qwen2_5_VLDecoderLayer_with_MoE", "Qwen2_5_VLVisionBlock"]

    def __init__(
        self,
        config: Qwen2_5_VLConfig,
        processor=None,
        tokenizer_mixin=None,
        use_selective_recompute=False,
    ):
        super().__init__(config)
        self.visual = self._build_visual(config, use_selective_recompute)
        self.model = Qwen2_5_VLMoEModel(
            config, use_selective_recompute=use_selective_recompute
        )
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.loss_fct = CrossEntropyLoss(
            reduction="none"
        )  # do not do reduction to compute channel loss

        # Manage through tokenizer_mixin; in pure flow mode the mixin is None
        self.processor = processor
        self.tokenizer_mixin = tokenizer_mixin
        if self.tokenizer_mixin is not None:
            self.action_tokenizer_type = self.tokenizer_mixin.tokenizer_type
            self.action_tokenizer = self.tokenizer_mixin.tokenizer
            self.action_mapper = self.tokenizer_mixin.build_action_mapper(processor)
        else:
            self.action_tokenizer_type = None
            self.action_tokenizer = None
            self.action_mapper = None
        self.define_action_token_id()

        self.times_cache = {}  # cache times linspace for each num_inference_timesteps
        self._infer_stable_cache: dict = {}  # cache stable-state inference structures

        self.rope_deltas = None  # cache rope_deltas here

        from wall_x.model.core.action.processor import ActionProcessor

        self.action_preprocessor = ActionProcessor(config)  # action processing

        # Apply LoRA if the configuration contains LoRA settings
        if hasattr(config, "use_lora") and config.use_lora:
            self.add_lora(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                lora_dropout=config.lora_dropout,
            )
        # Initialize weights and apply final processing
        self.post_init()
        self._post_init_engine(config)

    def _build_visual(self, config, use_selective_recompute):
        """Factory method for visual encoder. Subclasses may override to swap implementation."""
        return Qwen2_5_VisionTransformerPretrainedModel(
            config=config.vision_config,
            use_selective_recompute=use_selective_recompute,
        )

    def _post_init_engine(self, config):
        """Hook called at the end of __init__. Subclasses may override to add engine-specific state."""
        pass

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
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

    def get_attention_maps(
        self,
        input_ids,
        action_horizon,
        action_dim,
        num_inference_timesteps: int = 10,
        prefix_length: Optional[int] = None,  # prefix length
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
        **kwargs,
    ):

        if input_ids is not None:
            batch_size, seq_length = input_ids.shape

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if start_indices is None or end_indices is None:
            # Compute start and end positions for each expert token group after permutation; the dataset has no num_expert metadata, so this is computed here
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
                position_ids, rope_deltas = get_rope_index(
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
                img_mask = input_ids == self.config.image_token_id
                mask_unsqueezed = img_mask.unsqueeze(-1)
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
            moe_token_types=moe_token_types,  # pass token_types
            start_indices=start_indices,
            end_indices=end_indices,
            positional_masks=positional_masks,  # pass masks for each category
            use_cache=False,
            output_attentions=True,
            output_hidden_states=False,
            return_dict=return_dict,
            adarms_conds=[None, adarms_cond],
            # cache_position=cache_position,
        )
        attention_maps = torch.stack(outputs.attentions, dim=0)[
            None, :, -action_horizon:
        ]  # [flow steps, layer depths, all tokens, all tokens]
        attention_maps = attention_maps.mean(2)
        image_mask_indices = img_mask[0].nonzero(as_tuple=True)[0]
        attention_maps = attention_maps[:, :, image_mask_indices]
        return attention_maps

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
        sample_time: Optional[torch.FloatTensor] = None,
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
            # Compute start and end positions for each expert token group after permutation; the dataset has no num_expert metadata, so this is computed here
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
                position_ids, rope_deltas = get_rope_index(
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
                input_ids,
                inputs_embeds,
                action_chunk,
                dataset_names,
                sample_time,
                dof_mask,
            )

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            moe_token_types=moe_token_types,  # pass token_types
            start_indices=start_indices,
            end_indices=end_indices,
            positional_masks=positional_masks,  # pass masks for each category
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            adarms_conds=[None, adarms_cond],
            # cache_position=cache_position,
        )

        hidden_states = outputs[0]

        # --- optimization: loss-token-only LM Head ---
        # During training, only compute lm_head for tokens that contribute to CE loss,
        # reducing GEMM size from [B*S, V] to [N_loss, V].
        # FSDP fp32 layernorms emit fp32 hidden_states; lm_head runs in bf16.
        _lm_dtype = self.lm_head.weight.dtype
        _lm_loss_mask = None
        if labels is not None and self.training:
            shift_labels = labels[..., 1:].contiguous()
            _lm_loss_mask = shift_labels != -100  # [B, S-1]
            if _lm_loss_mask.any():
                lm_hidden = hidden_states[:, :-1, :][_lm_loss_mask]  # [N_loss, H]
                logits = self.lm_head(lm_hidden.to(_lm_dtype))  # [N_loss, V]
            else:
                logits = None
        else:
            logits = self.lm_head(hidden_states.to(_lm_dtype))

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
            _lm_loss_mask=_lm_loss_mask,
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
        assert predict_mode in ["fast", "spatial_token", "flow", "diffusion", "dllm"]

        if predict_mode in ["fast", "spatial_token"]:
            output = self.generate_ar_action(**kwargs)
        elif predict_mode in ["diffusion", "flow"]:
            output = self.generate_flow_action(**kwargs)
        elif predict_mode == "dllm":
            output = self.generate_dllm_action(**kwargs)

        return output["predict_action"], output.get("gt_action", None)

    @torch.no_grad()
    def generate_text(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        moe_token_types: Optional[torch.LongTensor] = None,  # new parameter
        start_indices: Optional[torch.Tensor] = None,
        end_indices: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_side_image: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        side_image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        action_chunk: Optional[torch.FloatTensor] = None,  # action chunk
        proprioception: Optional[torch.FloatTensor] = None,  # joint positions
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        dataset_names: Optional[str] = None,
        dof_mask: Optional[torch.FloatTensor] = None,
        agent_pos_mask: Optional[torch.FloatTensor] = None,
        gt_output_ids: Optional[torch.LongTensor] = None,
        prefix_length: Optional[int] = None,  # prefix length
        positional_masks: Optional[dict] = None,  # position masks for each modality
        re_generate: bool = False,
        **kwargs,
    ):
        # assert self.config._attn_implementation == "flash_attention_2", "generate_text only support flash_attention_2 attn implementation"

        # generate text only need prefix part
        if prefix_length is not None:
            input_ids, _gt_output_ids = (
                input_ids[:, :prefix_length],
                input_ids[:, prefix_length:],
            )
            gt_output_ids = (
                gt_output_ids if gt_output_ids is not None else _gt_output_ids
            )
            attention_mask = (
                attention_mask[:, :prefix_length]
                if attention_mask is not None
                else None
            )
            moe_token_types = (
                moe_token_types[:, :prefix_length]
                if moe_token_types is not None
                else None
            )
        else:
            prefix_length = input_ids.shape[1]

        if start_indices is None or end_indices is None:
            # Compute start and end positions for each expert token group after permutation; the dataset has no num_expert metadata, so this is computed here
            group_size = torch.zeros(
                self.config.num_experts, dtype=torch.long, device="cpu"
            )
            for i in range(self.config.num_experts):
                group_size[i] = (moe_token_types == i).sum()

            # Calculate start and end indices for each expert group
            start_indices = torch.cumsum(group_size, dim=0) - group_size
            end_indices = torch.cumsum(group_size, dim=0)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "moe_token_types": moe_token_types,
            "start_indices": start_indices,
            "end_indices": end_indices,
            "image_grid_thw": image_grid_thw,
            "dof_mask": dof_mask,
            "agent_pos_mask": agent_pos_mask,
            "proprioception": proprioception,
            "dataset_names": dataset_names,
            # "prefix_length": prefix_length,
        }
        predict_output_ids = self.generate(
            **batch,
            max_new_tokens=100,
            eos_token_id=[
                self.processor.tokenizer.eos_token_id
            ],  # set multiple end markers
            use_cache=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,  # explicitly set pad_token_id
            temperature=(
                1.0 if not re_generate else 0.7
            ),  # use a higher temperature when regenerating
            do_sample=(
                False if not re_generate else True
            ),  # use sampling when regenerating
        )
        input_text = self.processor.batch_decode(
            input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        if gt_output_ids is not None and gt_output_ids.shape[-1] != 0:
            gt_output_text = self.processor.batch_decode(
                gt_output_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
        else:
            gt_output_text = None
        predict_output_text = self.processor.batch_decode(
            predict_output_ids[:, prefix_length:],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        output = {
            "input_text": input_text,
            "gt_output_text": gt_output_text,
            "predict_output_text": predict_output_text,
            "predict_output_ids": predict_output_ids,
        }
        return output

    @torch.no_grad()
    def generate_ar_action(
        self,
        input_ids,
        action_horizon,
        action_dim,
        state=None,
        dataset_names=None,
        action_chunk=None,
        dof_mask=None,
        unnorm=True,
        max_retries=3,  # maximum retry count
        proprioception=None,  # proprioception for v3.1_delta decoding
        robot_type_id=None,  # robot type ID for v3.1_delta decoding
        **kwargs,
    ):
        # Regeneration mechanism
        predict_action = None
        output = None
        decode_success = False
        if dataset_names is None and (unnorm or action_chunk is not None):
            raise KeyError(
                "generate_ar_action requires dataset_names when normalizing or "
                "unnormalizing actions."
            )

        for retry_attempt in range(max_retries):
            use_re_generate = retry_attempt > 0

            if retry_attempt > 0:
                logger.info(
                    "Retrying action generation (attempt %s/%s)...",
                    retry_attempt + 1,
                    max_retries,
                )

            output = self.generate_text(
                input_ids=input_ids, re_generate=use_re_generate, **kwargs
            )
            predict_output_ids = output["predict_output_ids"]

            # Decode uniformly through tokenizer_mixin
            predict_action, decode_success = self.tokenizer_mixin.decode_action(
                output_ids=predict_output_ids,
                action_mapper=self.action_mapper,
                action_horizon=action_horizon,
                action_dim=action_dim,
                device=input_ids.device,
                proprioception=proprioception,
                dof_mask=dof_mask,
                robot_type_id=robot_type_id,
                state=state,
            )

            if decode_success:
                if retry_attempt > 0:
                    logger.info(
                        "Action generation succeeded on attempt %s",
                        retry_attempt + 1,
                    )
                break
            else:
                if retry_attempt < max_retries - 1:
                    logger.warning(
                        "Error in decoding action (attempt %s/%s), retrying with re_generate mode...",
                        retry_attempt + 1,
                        max_retries,
                    )
                else:
                    logger.warning(
                        "Error in decoding action after %s attempts, returning None",
                        max_retries,
                    )

        # Check whether dof_mask is required through mixin attributes
        uses_dof_mask = (
            self.tokenizer_mixin.uses_dof_mask_for_unnorm
            if self.tokenizer_mixin is not None
            else True
        )

        if action_chunk is not None:
            action_chunk = action_chunk.to(input_ids.device).to(torch.bfloat16)
            if uses_dof_mask:
                action_chunk = action_chunk[:, :, dof_mask[0, 0, :].bool()]
                output["gt_action"] = (
                    self.action_preprocessor.normalizer_action.unnormalize_data(
                        action_chunk, dataset_names, dof_mask
                    )
                )
            else:
                # Output full dimensions without mask filtering
                output["gt_action"] = (
                    self.action_preprocessor.normalizer_action.unnormalize_data(
                        action_chunk, dataset_names, None
                    )
                )

        if not decode_success or predict_action is None:
            logger.warning("Error in decoding action, predict_action is None")
            output["predict_action"] = None
        else:
            if unnorm:
                if isinstance(predict_action, np.ndarray):
                    predict_action = torch.tensor(
                        predict_action, device=input_ids.device
                    )
                elif predict_action.device != input_ids.device:
                    predict_action = predict_action.to(input_ids.device)

                # Add the batch dimension
                if predict_action.dim() == 2:
                    predict_action = predict_action.unsqueeze(0)

                if uses_dof_mask:
                    predict_action = unnormalize_data_with_virtual_tail(
                        self.action_preprocessor.normalizer_action,
                        predict_action,
                        dataset_names,
                        dof_mask,
                    )
                else:
                    # Output full dimensions without mask filtering
                    predict_action = (
                        self.action_preprocessor.normalizer_action.unnormalize_data(
                            predict_action, dataset_names, None
                        )
                    )
            output["predict_action"] = predict_action

        return output

    @torch.no_grad()
    def generate_flow_action(
        self,
        input_ids,
        action_horizon,
        action_dim,
        num_inference_timesteps: int = 10,
        prefix_length: Optional[int] = None,  # prefix length
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
        **kwargs,
    ):

        # assert self.config._attn_implementation == "sdpa", "generate_flow_action only support sdpa attn implementation"
        ctx = self._prepare_flow_action_inputs(
            input_ids=input_ids,
            action_horizon=action_horizon,
            action_dim=action_dim,
            num_inference_timesteps=num_inference_timesteps,
            prefix_length=prefix_length,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            moe_token_types=moe_token_types,
            start_indices=start_indices,
            end_indices=end_indices,
            positional_masks=positional_masks,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            action_chunk=action_chunk,
            proprioception=proprioception,
            unnorm_proprioception=unnorm_proprioception,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            second_per_grid_ts=second_per_grid_ts,
            dataset_names=dataset_names,
            dof_mask=dof_mask,
            agent_pos_mask=agent_pos_mask,
        )

        action_trajectory, attention_maps = self._execute_flow_action(ctx)

        output = self._finalize_flow_action_output(action_trajectory, ctx)
        if output_attentions:
            output["attention_maps"] = attention_maps
        return output

    def _execute_flow_action(self, ctx):
        """Strategy hook: selects which flow-action backend to use.

        Returns:
            action_trajectory: list of action tensors
            attention_maps: attention maps or None
        """
        return self._generate_flow_action_vanilla(ctx)

    def _prepare_flow_action_inputs(
        self,
        input_ids,
        action_horizon,
        action_dim,
        num_inference_timesteps,
        prefix_length,
        attention_mask,
        position_ids,
        past_key_values,
        inputs_embeds,
        moe_token_types,
        start_indices,
        end_indices,
        positional_masks,
        labels,
        use_cache,
        output_attentions,
        output_hidden_states,
        return_dict,
        pixel_values,
        pixel_values_videos,
        image_grid_thw,
        video_grid_thw,
        action_chunk,
        proprioception,
        unnorm_proprioception,
        rope_deltas,
        cache_position,
        second_per_grid_ts,
        dataset_names,
        dof_mask,
        agent_pos_mask,
    ):
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

        # Timing: input embedding processing
        img_mask = None
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                # from wall_x.utils.timers import ScopeTimer
                # with ScopeTimer("pixel_values.visual"):
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                img_mask = input_ids == self.config.image_token_id
                mask_unsqueezed = img_mask.unsqueeze(-1)
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
                position_ids, rope_deltas = get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                    self.config.vision_config.spatial_merge_size,
                    self.config.image_token_id,
                    self.config.video_token_id,
                    self.config.vision_start_token_id,
                    self.config.vision_config.tokens_per_second,
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

        flow_action_mask = input_ids == self.action_token_id_set["action_token_id"]

        if prefix_length is None:
            has_true = flow_action_mask.any(dim=1)
            prefix_length = torch.argmax(flow_action_mask.float(), dim=1, keepdim=True)
            prefix_length[~has_true] = flow_action_mask.shape[1]
            # check if prefix_length is the same for all batch
            if not torch.all(prefix_length == prefix_length[0]):
                raise ValueError(
                    "prefix_length differs across batch; batch prompts must align"
                )
            prefix_length = int(prefix_length[0].item())

        if start_indices is None or end_indices is None:
            # Compute start and end positions for each expert token group after permutation; the dataset has no num_expert metadata, so this is computed here
            # Cache key: shape + sum is sufficient for stable-state (moe_token_types structure doesn't change)
            _moe_key = (moe_token_types.shape, int(moe_token_types.sum().item()))
            _moe_cache = self._infer_stable_cache.get("moe_indices")
            if _moe_cache is None or _moe_cache[0] != _moe_key:
                group_size = torch.zeros(
                    self.config.num_experts, dtype=torch.long, device="cpu"
                )
                for i in range(self.config.num_experts):
                    group_size[i] = (moe_token_types == i).sum()

                # Calculate start and end indices for each expert group
                start_indices = torch.cumsum(group_size, dim=0) - group_size
                end_indices = torch.cumsum(group_size, dim=0)
                self._infer_stable_cache["moe_indices"] = (
                    _moe_key,
                    start_indices,
                    end_indices,
                )
            else:
                _, start_indices, end_indices = _moe_cache

        # Timing: action initialization
        if action_chunk is not None:
            action_chunk = action_chunk.to(inputs_embeds.device).to(torch.float32)

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
            self.times_cache[num_inference_timesteps] = (
                self.action_preprocessor.get_inference_times(
                    num_inference_timesteps, inputs_embeds.device, torch.float32
                )
            )
        times = self.times_cache[num_inference_timesteps]
        time_0 = times[0].unsqueeze(0).repeat(noisy_action.shape[0])
        action_embed, adarms_cond = self.action_preprocessor.step(
            timestep=time_0, noisy_action=noisy_action, dof_mask=dof_mask
        )
        action_embed = action_embed.reshape(-1, inputs_embeds.shape[-1]).to(
            inputs_embeds.dtype
        )
        inputs_embeds[flow_action_mask] = action_embed

        # Automatically generate padding actions from dof_mask
        if not dof_mask.all():
            padding_action = (
                torch.zeros((1, dof_mask.shape[-1]))
                .to(dof_mask.device)
                .to(torch.float32)
            )
            padding_action = normalize_data_with_virtual_tail(
                self.action_preprocessor.normalizer_action,
                padding_action,
                dataset_names,
                dof_mask,
            )
        else:
            padding_action = None

        return {
            "batch_size": batch_size,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "prefix_length": prefix_length,
            "flow_action_mask": flow_action_mask,
            "start_indices": start_indices,
            "end_indices": end_indices,
            "dof_mask": dof_mask,
            "noise": noise,
            "noisy_action": noisy_action,
            "times": times,
            "adarms_cond": adarms_cond,
            "action_chunk": action_chunk,
            "padding_action": padding_action,
            "action_horizon": action_horizon,
            "action_dim": action_dim,
            "input_ids": input_ids,
            "moe_token_types": moe_token_types,
            "positional_masks": positional_masks,
            "dataset_names": dataset_names,
            "output_attentions": output_attentions,
            "img_mask": img_mask,
        }

    def _generate_flow_action_vanilla(self, ctx):
        # Timing: prefill forward pass
        all_attention_maps = []
        output_attentions = True if ctx["output_attentions"] else False
        prefetch_output = self.model(
            input_ids=None,
            attention_mask=ctx["attention_mask"],
            position_ids=ctx["position_ids"],
            past_key_values=None,
            inputs_embeds=ctx["inputs_embeds"],
            moe_token_types=ctx["moe_token_types"],
            start_indices=ctx["start_indices"],
            end_indices=ctx["end_indices"],
            positional_masks=ctx["positional_masks"],
            use_cache=True,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=True,
            adarms_conds=[None, ctx["adarms_cond"]],
        )
        hidden_states = prefetch_output.last_hidden_state
        prefix_kv_cache = prefetch_output.past_key_values
        if output_attentions:
            all_attention_maps.append(prefetch_output.attentions)
        action_hidden_states = hidden_states[ctx["flow_action_mask"]].to(torch.float32)
        action_pred = self.action_preprocessor.action_proj_back(
            action_hidden_states[:, : self.action_preprocessor.action_hidden_size]
        )
        if getattr(self.config, "use_x_pred", False):
            v_0 = action_pred - ctx["noise"].reshape(-1, ctx["noise"].shape[-1])
        else:
            v_0 = action_pred
        v_0 = v_0.reshape(ctx["batch_size"], ctx["action_horizon"], ctx["action_dim"])

        v_padding = None
        if (not ctx["dof_mask"].all()) and (ctx["padding_action"] is not None):
            v_padding = ctx["padding_action"] - ctx["noisy_action"]
            v_0 = (v_padding) * (1 - ctx["dof_mask"]) + v_0 * ctx["dof_mask"]
        dt = ctx["times"][1] - ctx["times"][0]
        ctx["noisy_action"] = ctx["noisy_action"] + dt * v_0

        (
            postfix_position_ids,
            postfix_inputs_embeds,
            _postfix_attention_mask,
            postfix_moe_token_types,
            postfix_input_ids,
            postfix_start_indices,
            postfix_end_indices,
            padding_mask,
        ) = self._prepare_flow_postfix(ctx, prefix_kv_cache)

        def step_with_kvcache(timestep, noisy_action):
            action_mask = (
                postfix_input_ids == self.action_token_id_set["action_token_id"]
            )
            assert action_mask.any(), "No action token found in input_ids"
            timestep = timestep.unsqueeze(0).repeat(noisy_action.shape[0])
            action_embed, adarms_cond = self.action_preprocessor.step(
                timestep=timestep, noisy_action=noisy_action, dof_mask=ctx["dof_mask"]
            )
            action_embed = action_embed.reshape(-1, postfix_inputs_embeds.shape[-1])

            # Create a temporary inputs_embeds copy for thread safety
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
                output_attentions=output_attentions,
                output_hidden_states=False,
                return_dict=True,
                adarms_conds=[None, adarms_cond],
            )
            if output_attentions:
                all_attention_maps.append(transformer_outputs.attentions)
            hidden_states = transformer_outputs.last_hidden_state
            action_hidden_states = hidden_states[action_mask].to(torch.float32)
            action_pred = self.action_preprocessor.action_proj_back(
                action_hidden_states[:, : self.action_preprocessor.action_hidden_size]
            )
            if getattr(self.config, "use_x_pred", False):
                # Align noisy_action (and timestep) to action_pred's shape
                B, action_horizon, action_dim = noisy_action.shape
                noisy_action_flat = noisy_action.reshape(
                    -1, action_dim
                )  # [B * action_horizon, action_dim]
                timestep_expand = timestep.view(-1, 1).repeat_interleave(
                    action_horizon, dim=0
                )  # [B * action_horizon, 1]
                v_t = (action_pred - noisy_action_flat) / torch.clamp(
                    1 - timestep_expand, min=0.05
                )
            else:
                v_t = action_pred
            v_t = v_t.reshape(
                ctx["batch_size"], ctx["action_horizon"], ctx["action_dim"]
            )

            if (not ctx["dof_mask"].all()) and (ctx["padding_action"] is not None):
                v_t = (v_padding) * (1 - ctx["dof_mask"]) + v_t * ctx["dof_mask"]

            return v_t

        action_trajectory = odeint(
            step_with_kvcache, ctx["noisy_action"], ctx["times"][1:], method="euler"
        )

        attention_maps = None
        if output_attentions and ctx["img_mask"] is not None:
            attention_maps = [torch.stack(map, dim=0) for map in all_attention_maps]
            attention_maps = torch.stack(
                attention_maps, dim=0
            )  # [flow steps, layer depths, action tokens, all tokens]
            attention_maps = attention_maps.mean(2)
            image_mask_indices = ctx["img_mask"][0].nonzero(as_tuple=True)[0]
            attention_maps = attention_maps[:, :, image_mask_indices]
        return action_trajectory, attention_maps

    def _prepare_flow_postfix(self, ctx, prefix_kv_cache=None):
        postfix_position_ids = ctx["position_ids"][:, :, ctx["prefix_length"] :]
        postfix_inputs_embeds = ctx["inputs_embeds"][:, ctx["prefix_length"] :, :]
        postfix_attention_mask = ctx["attention_mask"][:, ctx["prefix_length"] :]
        postfix_moe_token_types = ctx["moe_token_types"][:, ctx["prefix_length"] :]
        postfix_input_ids = ctx["input_ids"][:, ctx["prefix_length"] :]

        postfix_start_indices = None
        postfix_end_indices = None
        padding_mask = None

        if prefix_kv_cache is not None:
            if ctx["prefix_length"] is None:
                has_true = ctx["flow_action_mask"].any(dim=1)
                prefix_length = torch.argmax(
                    ctx["flow_action_mask"].float(), dim=1, keepdim=True
                )
                prefix_length[~has_true] = ctx["flow_action_mask"].shape[1]
                # check if prefix_length is the same for all batch
                if not torch.all(prefix_length == prefix_length[0]):
                    raise ValueError(
                        "prefix_length differs across batch; batch prompts must align"
                    )
                prefix_length = int(prefix_length[0].item())
                ctx["prefix_length"] = prefix_length

            if hasattr(prefix_kv_cache, "key_cache"):
                for layer_i in range(len(prefix_kv_cache.key_cache)):
                    prefix_kv_cache.key_cache[layer_i] = prefix_kv_cache.key_cache[
                        layer_i
                    ][:, :, : ctx["prefix_length"], :]
                    prefix_kv_cache.value_cache[layer_i] = prefix_kv_cache.value_cache[
                        layer_i
                    ][:, :, : ctx["prefix_length"], :]
            else:
                for layer_i in range(len(prefix_kv_cache.layers)):
                    prefix_kv_cache.layers[layer_i].keys = prefix_kv_cache.layers[
                        layer_i
                    ].keys[:, :, : ctx["prefix_length"], :]
                    prefix_kv_cache.layers[layer_i].values = prefix_kv_cache.layers[
                        layer_i
                    ].values[:, :, : ctx["prefix_length"], :]

            group_size = torch.zeros(
                self.config.num_experts, dtype=torch.long, device="cpu"
            )
            for i in range(self.config.num_experts):
                group_size[i] = (postfix_moe_token_types == i).sum()

            # Calculate start and end indices for each expert group
            postfix_start_indices = torch.cumsum(group_size, dim=0) - group_size
            postfix_end_indices = torch.cumsum(group_size, dim=0)

            pad_token_id = self.processor.tokenizer.pad_token_id
            padding_mask = ctx["input_ids"] == pad_token_id
        else:
            pad_token_id = self.processor.tokenizer.pad_token_id
            padding_mask = ctx["input_ids"] == pad_token_id

        postfix_length = ctx["input_ids"].shape[-1] - ctx["prefix_length"]
        _postfix_attention_mask = torch.ones(
            (ctx["batch_size"], postfix_length, ctx["prefix_length"] + postfix_length),
            dtype=torch.bool,
            device=postfix_attention_mask.device,
        )

        postfix_padding_mask = padding_mask[:, ctx["prefix_length"] :]
        full_padding_mask = padding_mask

        if self.config.causal_action_attention_mask:
            _postfix_attention_mask[:, :, ctx["prefix_length"] :] = torch.tril(
                torch.ones(
                    (postfix_length, postfix_length),
                    dtype=torch.bool,
                    device=postfix_attention_mask.device,
                )
            )

        # postfix_padding_mask: [B, postfix_length], True where padding (query rows)
        # full_padding_mask: [B, prefix_length+postfix_length], True where padding (key cols)
        _postfix_attention_mask.masked_fill_(postfix_padding_mask.unsqueeze(2), False)
        _postfix_attention_mask.masked_fill_(full_padding_mask.unsqueeze(1), False)

        return (
            postfix_position_ids,
            postfix_inputs_embeds,
            _postfix_attention_mask,
            postfix_moe_token_types,
            postfix_input_ids,
            postfix_start_indices,
            postfix_end_indices,
            padding_mask,
        )

    def _finalize_flow_action_output(self, action_trajectory, ctx):
        predict_action = action_trajectory[-1]
        predict_action = unnormalize_data_with_virtual_tail(
            self.action_preprocessor.normalizer_action,
            predict_action,
            ctx["dataset_names"],
            ctx.get("dof_mask"),
        )
        output = {"predict_action": predict_action}
        if ctx["action_chunk"] is not None:
            output["gt_action"] = unnormalize_data_with_virtual_tail(
                self.action_preprocessor.normalizer_action,
                ctx["action_chunk"],
                ctx["dataset_names"],
                ctx.get("dof_mask"),
            )

        return output

    def forward(self, mode: Optional[str] = None, **kwargs):
        if not mode:
            with torch.no_grad():
                return self.train_step_forward(**kwargs)
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
        start_indices=None,
        end_indices=None,
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
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        # Exception 3: with synced GPUs cache_position may go out of bounds, but we only want dummy token in that case.
        #              (we can't check exception 3 while compiling)
        # Exception 4: If input_embeds are passed then slice it through `cache_position`, to keep only the unprocessed tokens and
        # generate the first token for each sequence. Later use the generated Input ids for continuation.

        if moe_token_types is None:
            moe_token_types = torch.zeros_like(
                input_ids
            )  # FIXME  input_ids or input_embeds
        else:
            # Ensure moe_token_types has the same length as input_ids
            if moe_token_types.shape[1] < input_ids.shape[1]:
                # Compute the required padding length
                pad_length = input_ids.shape[1] - moe_token_types.shape[1]
                # Create the padding tensor using 0 as the default token type
                pad_tensor = torch.zeros(
                    (moe_token_types.shape[0], pad_length),
                    dtype=moe_token_types.dtype,
                    device=moe_token_types.device,
                )
                # Append padding after the existing moe_token_types
                moe_token_types = torch.cat([moe_token_types, pad_tensor], dim=1)

        if past_key_values is not None:
            if inputs_embeds is not None and input_ids.shape[1] == 0:  # Exception 4
                inputs_embeds = inputs_embeds[:, -cache_position.shape[0] :]
                moe_token_types = moe_token_types[:, -cache_position.shape[0] :]
            elif inputs_embeds is not None or (  # Exception 1
                is_torchdynamo_compiling() or cache_position[-1] >= input_ids.shape[1]
            ):  # Exception 3
                input_ids = input_ids[:, -cache_position.shape[0] :]
                moe_token_types = moe_token_types[:, -cache_position.shape[0] :]
            elif (
                input_ids.shape[1] != cache_position.shape[0]
            ):  # Default case (the "else", a no op, is Exception 2)
                cache_pos = cache_position.clone()
                input_ids = input_ids[:, cache_pos]
                moe_token_types = moe_token_types[:, cache_pos]

        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

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

        if (
            self.config._attn_implementation == "flash_attention_2_ki"
            and self.config.attention_moe is True
        ):
            attention_mask = None  # Pass None as the mask to use the official flash attention for autoregressive prediction

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "moe_token_types": moe_token_types,
                "start_indices": start_indices,
                "end_indices": end_indices,
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
        Get the number of images and videos for each sample to calculate the separation length of the sample tensor.
        These parameters are not passed through the processor to avoid unpredictable impacts from interface modifications.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

        Returns:
            image_nums (`torch.LongTensor` of shape `(batch_size, num_images_sample)`)
            video_nums (`torch.LongTensor` of shape `(batch_size, num_videos_sample)`)
        """
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        vision_start_mask = input_ids == vision_start_token_id
        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_mask = input_ids == image_token_id
        video_mask = input_ids == video_token_id
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
        # Overwritten -- Support for expanding tensors without a batch size dimension
        # e.g., pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_t
        # pixel_values.shape[0] is sum(seqlen_images for samples)
        # image_grid_thw.shape[0] is sum(num_images for samples)

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = [
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "second_per_grid_ts",
        ]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(input_ids)

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat(
                    [sample.repeat(*repeat_args) for sample in samples], dim=0
                )
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # split images into samples
                    samples = torch.split(image_grid_thw, list(image_nums))
                    # compute the sequence length of images for each sample
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # get the num of images for each sample
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "second_per_grid_ts":
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

        # input_ids is required for expanding visual inputs
        # If input_ids is unavailable, visual inputs will not be used; therefore, there is no need to expand visual inputs.
        if input_ids is not None and input_ids.numel() != 0:
            model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError(
                    "If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined."
                )
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(
                model_kwargs["encoder_outputs"]
            )

        return input_ids, model_kwargs

    def rename_vlm_weights_for_vla(self, merged_weights):
        renamed = {}

        for key, value in merged_weights.items():

            if key.startswith("model.layers") and "mlp." in key and self.config.mlp_moe:
                layer_num = key.split(".layers.")[1].split(".mlp")[0]
                new_key = key.replace(
                    f"layers.{layer_num}.mlp.", f"layers.{layer_num}.moe.experts.0."
                )
                renamed[new_key] = value
                continue

            if (
                key.startswith("model.layers")
                and "self_attn." in key
                and self.config.attention_moe
            ):
                layer_num = key.split(".layers.")[1].split(".self_attn")[0]
                proj_types = ["q_proj", "k_proj", "v_proj", "o_proj"]
                for proj in proj_types:
                    if proj in key:
                        new_key = key.replace(
                            f"layers.{layer_num}.self_attn.{proj}",
                            f"layers.{layer_num}.self_attn.{proj}_experts.0",
                        )
                        renamed[new_key] = value
                        break
                continue

            if self.config.norm_moe and ".input_layernorm." in key:
                renamed[key.replace("input_layernorm", "input_layernorms.0")] = value
                continue
            if self.config.norm_moe and ".post_attention_layernorm." in key:
                renamed[
                    key.replace(
                        "post_attention_layernorm", "post_attention_layernorms.0"
                    )
                ] = value
                continue
            if self.config.norm_moe and ".norm." in key:
                renamed[key.replace("norm", "norms.0")] = value
                continue

            renamed[key] = value

        fused = Qwen2_5_VLMoEForAction.convert_to_fused(renamed)

        return fused

    @staticmethod
    def fuse_gate_up(
        fused, prefix, suffix_gate="gate_proj", suffix_up="up_proj", out="gate_up_proj"
    ):
        gate_w = fused.get(prefix + f"{suffix_gate}.weight")
        up_w = fused.get(prefix + f"{suffix_up}.weight")
        gate_b = fused.get(prefix + f"{suffix_gate}.bias")
        up_b = fused.get(prefix + f"{suffix_up}.bias")

        # Skip fusion if gate_up_proj already exists
        if prefix + f"{out}.weight" in fused:
            return

        if gate_w is not None and up_w is not None:
            fused[prefix + f"{out}.weight"] = torch.cat([gate_w, up_w], dim=0)
            if gate_b is not None and up_b is not None:
                fused[prefix + f"{out}.bias"] = torch.cat([gate_b, up_b], dim=0)

            # Remove the old modules
            for n in [
                f"{suffix_gate}.weight",
                f"{suffix_up}.weight",
                f"{suffix_gate}.bias",
                f"{suffix_up}.bias",
            ]:
                full = prefix + n
                if full in fused:
                    del fused[full]

    @staticmethod
    def fuse_qkv(fused, prefix, *, experts=False, expert_id=None):
        """
        prefix:
        - non-MoE: model.layers.X.self_attn.
        - MoE:     model.layers.X.self_attn.

        When experts=True, expert_id must be provided
        """

        if experts:
            assert expert_id is not None, "expert_id must be provided for MoE attention"

            eid = expert_id

            q_w = fused.get(prefix + f"q_proj_experts.{eid}.weight")
            k_w = fused.get(prefix + f"k_proj_experts.{eid}.weight")
            v_w = fused.get(prefix + f"v_proj_experts.{eid}.weight")

            q_b = fused.get(prefix + f"q_proj_experts.{eid}.bias")
            k_b = fused.get(prefix + f"k_proj_experts.{eid}.bias")
            v_b = fused.get(prefix + f"v_proj_experts.{eid}.bias")

            out_w = f"qkv_proj_experts.{eid}.weight"
            out_b = f"qkv_proj_experts.{eid}.bias"

            remove_list = [
                f"q_proj_experts.{eid}.weight",
                f"k_proj_experts.{eid}.weight",
                f"v_proj_experts.{eid}.weight",
                f"q_proj_experts.{eid}.bias",
                f"k_proj_experts.{eid}.bias",
                f"v_proj_experts.{eid}.bias",
            ]

        else:
            q_w = fused.get(prefix + "q_proj.weight")
            k_w = fused.get(prefix + "k_proj.weight")
            v_w = fused.get(prefix + "v_proj.weight")

            q_b = fused.get(prefix + "q_proj.bias")
            k_b = fused.get(prefix + "k_proj.bias")
            v_b = fused.get(prefix + "v_proj.bias")

            out_w = "qkv_proj.weight"
            out_b = "qkv_proj.bias"

            remove_list = [
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "q_proj.bias",
                "k_proj.bias",
                "v_proj.bias",
            ]

        if q_w is None or k_w is None or v_w is None:
            return

        fused[prefix + out_w] = torch.cat([q_w, k_w, v_w], dim=0)

        if q_b is not None and k_b is not None and v_b is not None:
            fused[prefix + out_b] = torch.cat([q_b, k_b, v_b], dim=0)

        for n in remove_list:
            full = prefix + n
            if full in fused:
                del fused[full]

    @staticmethod
    def is_fused(state_dict):
        # Check whether the model is already fused by looking for fused weight markers
        return any(".moe.experts.0.gate_up_proj" in key for key in state_dict.keys())

    @staticmethod
    def convert_to_fused(state_dict):
        """
        Convert an unfused checkpoint to a fused checkpoint
        """
        fused = {}

        # =========================
        # Phase 1: copy everything first
        # =========================
        for k, v in state_dict.items():
            fused[k] = v

        # =========================
        # Phase 2: fuse gate + up (MoE aware)
        # =========================
        for key in list(fused.keys()):
            # language MoE mlp (ANY expert)
            m = re.match(
                r"(model\.layers\.\d+\.moe\.experts\.\d+\.)gate_proj\.weight",
                key,
            )
            if m:
                prefix = m.group(1)
                Qwen2_5_VLMoEForAction.fuse_gate_up(fused, prefix)
                continue

            # language non-MoE mlp
            if re.match(r"(model\.layers\.\d+\.mlp\.)gate_proj\.weight", key):
                prefix = key.replace("gate_proj.weight", "")
                Qwen2_5_VLMoEForAction.fuse_gate_up(fused, prefix)
                continue

            # visual mlp
            if re.match(r"(visual\.blocks\.\d+\.mlp\.)gate_proj\.weight", key):
                prefix = key.replace("gate_proj.weight", "")
                Qwen2_5_VLMoEForAction.fuse_gate_up(fused, prefix)

        # =========================
        # Phase 3: fuse attention qkv (MoE aware)
        # =========================
        for key in list(fused.keys()):
            # language MoE attention: ANY expert
            m = re.match(
                r"(model\.layers\.\d+\.self_attn\.)q_proj_experts\.(\d+)\.weight",
                key,
            )
            if m:
                prefix = m.group(1)
                expert_id = int(m.group(2))
                Qwen2_5_VLMoEForAction.fuse_qkv(
                    fused,
                    prefix,
                    experts=True,
                    expert_id=expert_id,
                )
                continue

            # language non-MoE attention
            if re.match(r"(model\.layers\.\d+\.self_attn\.)q_proj\.weight", key):
                prefix = key.replace("q_proj.weight", "")
                Qwen2_5_VLMoEForAction.fuse_qkv(
                    fused,
                    prefix,
                    experts=False,
                )
                continue

            # visual attention
            if re.match(r"(visual\.blocks\.\d+\.attn\.)q_proj\.weight", key):
                prefix = key.replace("q_proj.weight", "")
                Qwen2_5_VLMoEForAction.fuse_qkv(
                    fused,
                    prefix,
                    experts=False,
                )

        # =========================
        # Phase 4: sanity check(optional)
        # =========================
        for k in fused.keys():
            assert not any(
                x in k
                for x in [
                    ".gate_proj.",
                    ".up_proj.",
                    ".q_proj.",
                    ".k_proj.",
                    ".v_proj.",
                ]
            ), f"Unfused key still exists: {k}"

        return fused

    def convert_to_mix_precision(self):
        # Mix Precision
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
            if name in params_to_keep_float32:
                param.data = param.data.to(torch.float32)

    def convert_to_fsdp(
        self,
        *,
        mesh,
        mp_policy,
        offload_policy=None,
        reshard_after_forward: bool = True,
        use_dmuon: bool = False,
    ):
        """Wrap with FSDP2: per-decoder-layer + per-vision-block fully_shard
        with bf16 ``mp_policy``; layernorms and ``action_preprocessor`` leaves
        get their own nested ``fully_shard`` with an fp32 policy so their
        forward stays in fp32 (matching the old FSDP1 NO_SHARD + fp32_policy
        layout).

        ``mesh`` and ``mp_policy`` are required and produced by
        ``FSDPStrategy._build_fsdp2_layout``. The pre-migration FSDP1 wrap
        used ``ShardingStrategy.SHARD_GRAD_OP`` regardless of the yaml
        config; the new path honors ``fsdp_sharding_strategy`` (default
        ``full_shard`` -> ``reshard_after_forward=True``). Active runs that
        relied on the implicit SHARD_GRAD_OP throughput should set
        ``distributed.fsdp_sharding_strategy: shard_grad_op`` explicitly.

        When ``use_dmuon=True``, DMuon owns selected trainable
        matrix weights before the remaining parameters are wrapped by FSDP2.
        """
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

        if use_dmuon:
            import dmuon

            # DMuon keeps fp32 master weights for dedicated params; symmetric
            # FSDP2 params still use MixedPrecisionPolicy for bf16 compute.
            for param in self.parameters():
                if param.requires_grad and param.dtype != torch.float32:
                    param.data = param.data.float()

            action_preprocessor_linear_ids = {
                id(module)
                for module in self.action_preprocessor.modules()
                if isinstance(module, nn.Linear)
            }

            def hook_boundary(module: nn.Module) -> bool:
                # ActionProcessor helper methods call internal Linear layers
                # directly, bypassing ActionProcessor.forward(). Those internal
                # Linear layers need their own hooks.
                if id(module) in action_preprocessor_linear_ids:
                    return True
                return isinstance(
                    module,
                    (
                        Qwen2_5_VLDecoderLayer_with_MoE,
                        Qwen2_5_VLVisionBlock,
                        Qwen2_5_VLPatchMerger,
                        type(self.action_preprocessor),
                    ),
                )

            assignment = dmuon.dedicate_params(
                self,
                mesh,
                predicate=_is_qwen25_dmuon_target_param,
                hook_boundary_predicate=hook_boundary,
                hook_boundary_strict=True,
                compute_dtype=torch.bfloat16,
                reshard_after_forward=reshard_after_forward,
            )
            logger.info(
                "[Qwen2.5 FSDP2 + DMuon] dedicate_params: %d params assigned",
                len(assignment),
            )

        fp32_mp = MixedPrecisionPolicy(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
        )
        shard_kwargs = dict(
            mesh=mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )
        if offload_policy is not None:
            shard_kwargs["offload_policy"] = offload_policy
        fp32_shard_kwargs = dict(shard_kwargs, mp_policy=fp32_mp)

        def fully_shard_fp32_leaf_or_container(name: str, module: nn.Module) -> None:
            # FSDP2 cannot shard containers without forward(), e.g. ModuleList.
            # When a norm match resolves to a container, shard only norm leaves.
            def is_norm_leaf(leaf_name: str, leaf: nn.Module) -> bool:
                return (
                    "norm" in leaf_name.lower() or "norm" in type(leaf).__name__.lower()
                )

            if isinstance(module, (nn.ModuleList, nn.ModuleDict)):
                for leaf_name, leaf in module.named_modules():
                    if not leaf_name:
                        continue
                    if any(True for _ in leaf.children()):
                        continue
                    if not is_norm_leaf(leaf_name, leaf):
                        continue
                    logger.info(
                        "[FSDP2] fully_shard fp32 layernorm: %s.%s",
                        name,
                        leaf_name,
                    )
                    fully_shard(leaf, **fp32_shard_kwargs)
                return

            logger.info("[FSDP2] fully_shard fp32 layernorm: %s", name)
            fully_shard(module, **fp32_shard_kwargs)

        # Layernorm leaves inside decoder layers / vision blocks / model.norm
        # - wrap with fp32 mp_policy BEFORE wrapping their parents so the
        # nested policy survives. Match-by-FQN mirrors the FSDP1 logic.
        for module_name, module in list(self.named_modules()):
            for child_name, child in list(module.named_children()):
                if any(
                    k in child_name.lower()
                    for k in ("input_layernorm", "post_attention_layernorm")
                ) or ("norm" in child_name.lower() and module_name.endswith("model")):
                    fully_shard_fp32_leaf_or_container(
                        f"{module_name}.{child_name}", child
                    )

        # action_preprocessor leaves: same fp32 treatment as layernorms.
        for name, module in list(self.named_modules()):
            if "action_preprocessor" not in name.lower():
                continue
            if any(True for _ in module.children()):
                continue  # only leaves
            logger.info(
                "[FSDP2] fully_shard fp32 action_preprocessor leaf: %s",
                name,
            )
            fully_shard(module, **fp32_shard_kwargs)

        # Each transformer decoder block: bf16 mp_policy from strategy.
        for idx, layer in enumerate(self.model.layers):
            fully_shard(layer, **shard_kwargs)
            if idx == 0:
                logger.info(
                    "[FSDP2] fully_shard model.layers.* (bf16, "
                    "reshard_after_forward=%s)",
                    reshard_after_forward,
                )

        # Each vision tower block: same.
        if hasattr(self, "visual") and hasattr(self.visual, "blocks"):
            for idx, block in enumerate(self.visual.blocks):
                fully_shard(block, **shard_kwargs)
                if idx == 0:
                    logger.info("[FSDP2] fully_shard visual.blocks.* (bf16)")

        # Root: covers lm_head, embed_tokens, and any params not in a
        # nested fully_shard. Inner fp32 wraps maintain their own policy.
        logger.info("[FSDP2] fully_shard root (bf16)")
        fully_shard(self, **shard_kwargs)
        return self
