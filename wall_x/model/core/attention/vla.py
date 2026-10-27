from transformers.integrations import use_kernelized_func
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

from transformers.activations import ACT2FN
from transformers.utils import logging
from transformers.utils.import_utils import (
    is_causal_conv1d_available,
    is_flash_linear_attention_available,
)
from wall_x.model.qact.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from wall_x.model.qact.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5RMSNorm,
    Qwen3_5RMSNormGated,
    apply_rotary_pos_emb,
    Qwen3_5DynamicCache,
    apply_mask_to_padding_states,
)
from wall_x.model.qact.qwen3_5.modeling_qwen3_5 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    Cache,
    Unpack,
    eager_attention_forward,
    is_fast_path_available,
)
from wall_x.model.qact.qwen3_5.modeling_qwen3_5 import (
    torch_causal_conv1d_update,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)
from wall_x.model.core.action.layout import TwoExpertLayout, prepare_two_expert_layout
from wall_x.model.core.ops import two_expert_pack, two_expert_unpack
from wall_x.model.core.ops.gated_delta import preprocess_gated_delta_inputs


if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None

if is_flash_linear_attention_available():
    from fla.modules import FusedRMSNormGated
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
else:
    fused_recurrent_gated_delta_rule = None
    FusedRMSNormGated = None

# Chunk kernel is backend-resolved (FlashQLA / fla) by the ops module.
from wall_x.model.core.ops.gated_delta import chunk_gated_delta_rule

try:
    from flash_mask.flash_mask_interface import flash_mask_attn_func
except ImportError:
    flash_mask_attn_func = None

try:
    from flash_mask.flash_mask_interface import flashmask_attn_func_stop_gradient
except ImportError:
    flashmask_attn_func_stop_gradient = None


logger = logging.get_logger(__name__)


def _mot_two_expert_layout(
    action_mask: torch.Tensor | None,
    action_embeds: torch.Tensor | None,
    two_expert_layout: TwoExpertLayout | None,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> TwoExpertLayout | None:
    if action_mask is None or action_embeds is None:
        return None
    if two_expert_layout is None:
        two_expert_layout = prepare_two_expert_layout(
            action_mask.to(dtype=torch.int64),
            batch_size=batch_size,
            seq_length=seq_len,
            device=device,
        )

    _, action_segment = two_expert_layout.expert_segments
    total_action = action_segment[1] - action_segment[0]
    expected_action = int(action_embeds.shape[0]) * int(action_embeds.shape[1])
    if total_action != expected_action:
        raise RuntimeError(
            "qwen3.5 MoT two-expert layout/action mismatch: "
            f"layout has {total_action} action tokens, "
            f"action_embeds has {expected_action}"
        )
    return two_expert_layout


def _replace_action_tokens(
    tokens: torch.Tensor,
    action_tokens: torch.Tensor,
    two_expert_layout: TwoExpertLayout,
) -> torch.Tensor:
    batch_size, seq_len = tokens.shape[:2]
    token_features = tokens.reshape(batch_size, seq_len, -1)
    action_features = action_tokens.reshape(-1, token_features.shape[-1])
    _, action_segment = two_expert_layout.expert_segments
    action_start, action_end = action_segment
    if action_features.shape[0] != action_end - action_start:
        raise RuntimeError(
            "qwen3.5 MoT action feature count mismatch: "
            f"got {action_features.shape[0]}, expected {action_end - action_start}"
        )

    packed_tokens = two_expert_pack(token_features, two_expert_layout)
    packed_tokens = torch.cat((packed_tokens[:action_start], action_features), dim=0)
    return two_expert_unpack(
        packed_tokens, batch_size, seq_len, two_expert_layout
    ).view_as(tokens)


def _packed_action_tokens(
    tokens: torch.Tensor,
    two_expert_layout: TwoExpertLayout,
) -> torch.Tensor:
    batch_size, seq_len = tokens.shape[:2]
    packed_tokens = two_expert_pack(
        tokens.reshape(batch_size, seq_len, -1), two_expert_layout
    )
    action_start, action_end = two_expert_layout.expert_segments[1]
    return packed_tokens[action_start:action_end]


def flash_mask_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs,
):
    """
    Flash Mask attention interface wrapper compatible with ALL_ATTENTION_FUNCTIONS.

    Args:
        module: attention module (for accessing num_key_value_groups, num_heads, etc.)
        query:   [B, num_heads, seq_len, head_dim]  (post-transpose)
        key:     [B, num_kv_heads, seq_len, head_dim]
        value:   [B, num_kv_heads, seq_len, head_dim]
        attention_mask: startend_row_indices, shape [B, num_heads, seq_len, 2] or [B, 1, seq_len, 2]
        dropout: dropout probability (unused by flash_mask, kept for interface compatibility)
        scaling: softmax scaling factor (unused by flash_mask, kept for interface compatibility)

    Returns:
        attn_output: [B, seq_len, num_heads, head_dim]
        attn_weights: None  (flash_mask does not return attention weights)
    """
    if flash_mask_attn_func is None:
        raise ImportError(
            "flash_mask is required for 'flash_mask' attention implementation. "
            "Please install it first."
        )

    # Native GQA: K/V stay at num_kv_heads — flash_mask accepts num_heads_k !=
    # num_heads, so no repeat_kv materialization is needed. This is a hard
    # requirement on the pinned internal flashmask build; an older build fails
    # loudly here with a head-count kernel error (upgrade flashmask if so).
    # See docs qwen35_mfu_report (FlashMask native GQA).

    # flash_mask_attn_func expects [B, seq_len, num_heads, head_dim]
    # but query/key/value arrive as [B, num_heads, seq_len, head_dim]
    query = query.transpose(1, 2)  # [B, seq_len, num_heads, head_dim]
    key = key.transpose(1, 2)  # [B, seq_len, num_heads, head_dim]
    value = value.transpose(1, 2)  # [B, seq_len, num_heads, head_dim]

    # Expand the (head-independent) attention_mask to the KV head count. With
    # native GQA, flash_mask requires startend_row_indices' head dim to equal
    # num_heads_k (= key's head count), not the query head count. When native
    # GQA is disabled, key was repeated to num_heads so key.shape[2] == num_heads
    # and this is unchanged. See docs qwen35_mfu_report (mask at KV head count).
    if attention_mask is not None and attention_mask.shape[1] == 1:
        num_heads_k = key.shape[2]
        attention_mask = attention_mask.expand(-1, num_heads_k, -1, -1).contiguous()

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # FlashAttention (and FlashMask) CUDA kernel only supports bf16/fp16.
    # When training with fp32 master weights + autocast, q/k/v may still be
    # fp32 because custom CUDA kernels are not in PyTorch's autocast dispatch
    # table. Cast explicitly here so the kernel always receives bf16.
    _orig_dtype = query.dtype
    if _orig_dtype == torch.float32:
        query = query.to(torch.bfloat16)
        key = key.to(torch.bfloat16)
        value = value.to(torch.bfloat16)

    attn_output = flash_mask_attn_func(
        query,
        key,
        value,
        startend_row_indices=attention_mask,
        causal=False,
    )

    # Cast output back to original dtype so the rest of the model stays in fp32
    if _orig_dtype == torch.float32:
        attn_output = attn_output.to(_orig_dtype)

    # attn_output: [B, seq_len, num_heads, head_dim]
    return attn_output, None


# Register flash_mask interface into ALL_ATTENTION_FUNCTIONS so that
# Qwen3_5 attention modules can dispatch to it via _attn_implementation = "flash_mask"
ALL_ATTENTION_FUNCTIONS["flash_mask"] = flash_mask_attention_forward


@use_kernelized_func(apply_rotary_pos_emb)
class Qwen3_5_VLA_Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim * 2,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        if self.config.vla_config.use_mot:
            self.q_proj_action = nn.Linear(
                config.vla_config.dim_inputs[1],
                config.num_attention_heads * self.head_dim * 2,
                bias=config.attention_bias,
            )
            self.k_proj_action = nn.Linear(
                config.vla_config.dim_inputs[1],
                config.num_key_value_heads * self.head_dim,
                bias=config.attention_bias,
            )
            self.v_proj_action = nn.Linear(
                config.vla_config.dim_inputs[1],
                config.num_key_value_heads * self.head_dim,
                bias=config.attention_bias,
            )
            self.o_proj_action = nn.Linear(
                config.num_attention_heads * self.head_dim,
                config.vla_config.dim_inputs[1],
                bias=config.attention_bias,
            )

        self.q_norm = Qwen3_5RMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3_5RMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape

        if self.config.vla_config.use_mot:
            self.q_norm_action = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm_action = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        # for vla
        action_embeds: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        two_expert_layout: TwoExpertLayout | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        batch_size, seq_len = input_shape

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
            2,
            dim=-1,
        )
        query_states = self.q_norm(query_states.view(hidden_shape))
        gate = gate.reshape(*input_shape, -1)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        action_layout = None
        if self.config.vla_config.use_mot:
            action_layout = _mot_two_expert_layout(
                action_mask,
                action_embeds,
                two_expert_layout,
                batch_size=batch_size,
                seq_len=seq_len,
                device=hidden_states.device,
            )

        if action_layout is not None:
            action_input_shape = action_embeds.shape[:-1]
            action_hidden_shape = (*action_input_shape, -1, self.head_dim)
            query_states_action, gate_action = torch.chunk(
                self.q_proj_action(action_embeds).view(
                    *action_input_shape, -1, self.head_dim * 2
                ),
                2,
                dim=-1,
            )
            query_states_action = self.q_norm_action(
                query_states_action.view(action_hidden_shape)
            )
            gate_action = gate_action.reshape(*action_input_shape, -1)

            query_states = _replace_action_tokens(
                query_states, query_states_action, action_layout
            )
            gate = _replace_action_tokens(gate, gate_action, action_layout)

            key_states_action = self.k_norm_action(
                self.k_proj_action(action_embeds).view(action_hidden_shape)
            )
            value_states_action = self.v_proj_action(action_embeds).view(
                action_hidden_shape
            )
            key_states = _replace_action_tokens(
                key_states, key_states_action, action_layout
            )
            value_states = _replace_action_tokens(
                value_states, value_states_action, action_layout
            )

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)

        output = self.o_proj(attn_output)
        output_action = None
        if action_layout is not None:
            output_action = self.o_proj_action(
                _packed_action_tokens(attn_output.contiguous(), action_layout)
            )
            output_action = output_action.reshape(
                action_embeds.shape[0], action_embeds.shape[1], -1
            )

        return output, output_action, attn_weights


class Qwen3_5_VLA_GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

        A = torch.empty(self.num_v_heads).uniform_(0, 16).clamp_(min=1e-4)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = (
            Qwen3_5RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
            if FusedRMSNormGated is None
            else FusedRMSNormGated(
                self.head_v_dim,
                eps=self.layer_norm_epsilon,
                activation=self.activation,
                device=torch.cuda.current_device(),
                dtype=(
                    config.dtype
                    if config.dtype is not None
                    else torch.get_default_dtype()
                ),
            )
        )

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.causal_conv1d_fn = causal_conv1d_fn
        self.causal_conv1d_update = causal_conv1d_update or torch_causal_conv1d_update
        self.chunk_gated_delta_rule = (
            chunk_gated_delta_rule or torch_chunk_gated_delta_rule
        )
        self.recurrent_gated_delta_rule = (
            fused_recurrent_gated_delta_rule or torch_recurrent_gated_delta_rule
        )

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because one of the required library is not installed. Falling back to "
                "torch implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation and"
                " https://github.com/Dao-AILab/causal-conv1d"
            )

        self.in_proj_qkv = nn.Linear(
            self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False
        )
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        # MoT: action-specific projections
        self.action_linear_attention = (
            config.vla_config.action_linear_attention and self.config.vla_config.use_mot
        )
        if self.action_linear_attention:
            action_dim = config.vla_config.dim_inputs[1]
            self.in_proj_qkv_action = nn.Linear(
                action_dim, self.key_dim * 2 + self.value_dim, bias=False
            )
            self.in_proj_z_action = nn.Linear(action_dim, self.value_dim, bias=False)
            self.in_proj_b_action = nn.Linear(action_dim, self.num_v_heads, bias=False)
            self.in_proj_a_action = nn.Linear(action_dim, self.num_v_heads, bias=False)
            self.out_proj_action = nn.Linear(self.value_dim, action_dim, bias=False)
            self.norm_action = (
                Qwen3_5RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
                if FusedRMSNormGated is None
                else FusedRMSNormGated(
                    self.head_v_dim,
                    eps=self.layer_norm_epsilon,
                    activation=self.activation,
                    device=torch.cuda.current_device(),
                    dtype=(
                        config.dtype
                        if config.dtype is not None
                        else torch.get_default_dtype()
                    ),
                )
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Qwen3_5DynamicCache | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        # for vla
        action_embeds: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        two_expert_layout: TwoExpertLayout | None = None,
    ):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        if not self.action_linear_attention and action_mask is not None:
            # 将action token的位置视为padding token进行处理
            hidden_states = apply_mask_to_padding_states(hidden_states, ~action_mask)

        # Set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape
        action_layout = None
        if self.action_linear_attention:
            action_layout = _mot_two_expert_layout(
                action_mask,
                action_embeds,
                two_expert_layout,
                batch_size=batch_size,
                seq_len=seq_len,
                device=hidden_states.device,
            )

        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state
            and seq_len == 1
            and cache_position is not None
        )

        # getting projected states from cache if it exists
        if cache_params is not None:
            conv_state = cache_params.conv_states[self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]

        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        # MoT: replace action token projections with action-specific projections
        # TODO: optimize
        output_action = None
        if action_embeds is not None and action_mask is not None:
            if action_layout is not None:
                mixed_qkv = _replace_action_tokens(
                    mixed_qkv.transpose(1, 2).contiguous(),
                    self.in_proj_qkv_action(action_embeds),
                    action_layout,
                ).transpose(1, 2)

                z_action = self.in_proj_z_action(action_embeds).reshape(
                    batch_size, action_embeds.shape[1], -1, self.head_v_dim
                )
                z = _replace_action_tokens(z, z_action, action_layout)
                b = _replace_action_tokens(
                    b, self.in_proj_b_action(action_embeds), action_layout
                )
                a = _replace_action_tokens(
                    a, self.in_proj_a_action(action_embeds), action_layout
                )
            else:
                output_action = action_embeds

        if use_precomputed_states:
            # 2. Convolution sequence transformation
            # NOTE: the conv state is updated in `causal_conv1d_update`
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            if cache_params is not None:
                conv_state = F.pad(
                    mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0)
                )
                cache_params.conv_states[self.layer_idx] = conv_state
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        query, key, value, beta, g, use_qk_l2norm_in_kernel = (
            preprocess_gated_delta_inputs(self, mixed_qkv, b, a, batch_size, seq_len)
        )

        if not use_precomputed_states:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )

        else:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )

        # Update cache
        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1, self.head_v_dim)
        if action_layout is not None:
            core_tokens = core_attn_out.reshape(batch_size, seq_len, -1)
            z_tokens = z.reshape(batch_size, seq_len, -1)
            packed_core = two_expert_pack(core_tokens, action_layout)
            packed_z = two_expert_pack(z_tokens, action_layout)
            packed_norm = torch.empty_like(packed_core)

            (llm_start, llm_end), (action_start, action_end) = (
                action_layout.expert_segments
            )
            if llm_end > llm_start:
                packed_norm[llm_start:llm_end] = self.norm(
                    packed_core[llm_start:llm_end].reshape(-1, self.head_v_dim),
                    packed_z[llm_start:llm_end].reshape(-1, self.head_v_dim),
                ).reshape(llm_end - llm_start, -1)
            if action_end > action_start:
                packed_norm[action_start:action_end] = self.norm_action(
                    packed_core[action_start:action_end].reshape(-1, self.head_v_dim),
                    packed_z[action_start:action_end].reshape(-1, self.head_v_dim),
                ).reshape(action_end - action_start, -1)
            core_attn_out = two_expert_unpack(
                packed_norm, batch_size, seq_len, action_layout
            )
        else:
            core_attn_out = self.norm(
                core_attn_out.reshape(-1, self.head_v_dim),
                z.reshape(-1, self.head_v_dim),
            ).reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        if action_layout is not None:
            output_action = self.out_proj_action(
                _packed_action_tokens(core_attn_out, action_layout)
            )
            output_action = output_action.reshape(
                action_embeds.shape[0], action_embeds.shape[1], -1
            )
        return output, output_action
