import torch
import torch.nn as nn
from typing import Optional, Tuple

from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from transformers.cache_utils import Cache
from transformers.utils import logging
from wall_x.fusions import ops
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
from flash_attn import flash_attn_func
from transformers.modeling_flash_attention_utils import (
    is_flash_attn_greater_or_equal_2_10,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLRotaryEmbedding,
    repeat_kv,
)

logger = logging.get_logger(__name__)


# def rotate_half(x):
#     x1 = x[..., : x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2 :]
#     return torch.cat((-x2, x1), dim=-1)


# def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=2):
#     mrope_section = mrope_section * 2
#     cos_split = torch.cat(
#         [m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1
#     ).unsqueeze(unsqueeze_dim)
#     sin_split = torch.cat(
#         [m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1
#     ).unsqueeze(unsqueeze_dim)
#     q_embed = (q * cos_split) + (rotate_half(q) * sin_split)
#     k_embed = (k * cos_split) + (rotate_half(k) * sin_split)
#     return q_embed, k_embed



class JointQwen2VLAttention(nn.Module):
    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        if not hasattr(config, "dim_inputs") or not config.dim_inputs:
            raise ValueError("配置中必须包含有效的 dim_inputs")

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling

        self.dim_inputs = config.dim_inputs  # Tuple[int, ...]

        if config.model_type == "qwen2_5_vl":
            bias_qkv = True
        else:
            bias_qkv = False

        self.q_proj_experts = nn.ModuleList(
            [
                nn.Linear(dim_input, self.num_heads * self.head_dim, bias=bias_qkv)
                for dim_input in self.dim_inputs
            ]
        )
        self.k_proj_experts = nn.ModuleList(
            [
                nn.Linear(
                    dim_input, self.num_key_value_heads * self.head_dim, bias=bias_qkv
                )
                for dim_input in self.dim_inputs
            ]
        )
        self.v_proj_experts = nn.ModuleList(
            [
                nn.Linear(
                    dim_input, self.num_key_value_heads * self.head_dim, bias=bias_qkv
                )
                for dim_input in self.dim_inputs
            ]
        )
        self.o_proj_experts = nn.ModuleList(
            [
                nn.Linear(self.num_heads * self.head_dim, dim_input, bias=False)
                for dim_input in self.dim_inputs
            ]
        )

        # Rotary embedding init
        if config.model_type == "qwen2_5_vl":
            self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)
        else:
            raise NotImplementedError(f"不支持的模型类型: {config.model_type}")

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeat key/value heads along the num_key_value_heads dimension (which is dim=2).
        Input shape:  (batch, seqlen, num_key_value_heads, head_dim)
        Output shape: (batch, seqlen, num_key_value_heads * n_rep, head_dim)
        Equivalent to torch.repeat_interleave(x, dim=2, repeats=n_rep)
        """
        if n_rep == 1:
            return hidden_states

        batch, slen, num_key_value_heads, head_dim = hidden_states.shape

        hidden_states = hidden_states.unsqueeze(3)

        hidden_states = hidden_states.expand(
            batch, slen, num_key_value_heads, n_rep, head_dim
        )

        return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        token_types: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_indices: Optional[torch.Tensor] = None,
        end_indices: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        row_id_map: Optional[torch.Tensor] = None,
        orig_shape: Optional[Tuple[int]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if token_types is None:
            raise ValueError("token_types can not be None")
        if token_types.max() >= len(self.dim_inputs):
            raise ValueError(f"token_types contains invalid expert indices: {token_types.max()}")

        if self.config.mot_opt:
            bsz, q_len, _ = orig_shape
            query_states, key_states, value_states = self._generate_qkv_mot_opt(
                hidden_states,
                token_types,
                start_indices,
                end_indices,
                probs,
                row_id_map,
                bsz,
                q_len,
            )
        else:
            bsz, q_len, _ = hidden_states.size()
            masks = [
                (token_types == expert_idx)
                for expert_idx in range(len(self.dim_inputs))
            ]
            query_states, key_states, value_states = self._generate_qkv(
                hidden_states, masks
            )

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        cos, sin = position_embeddings
        query_states, key_states = self._apply_rotary_pos_embed(
            query_states, key_states, cos, sin, unsqueeze_dim=2
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }  # Specific to RoPE models
            if use_cache:
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
            else:
                past_key_states, past_value_states = past_key_value[self.layer_idx]
                key_states = torch.cat([past_key_states, key_states], dim=-2)
                value_states = torch.cat([past_value_states, value_states], dim=-2)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            # Ensure that the attention_mask correctly matches across the head dimension.
            if len(attention_mask.shape) == 2:  # [batch_size, seq_len]
                # Expanded to a causal mask format of [batch_size, 1, seq_len, seq_len]
                bsz, seq_len = attention_mask.shape
                causal_mask = attention_mask.view(bsz, 1, 1, seq_len).expand(
                    bsz, 1, seq_len, seq_len
                )
            elif len(attention_mask.shape) == 3:  # [batch_size, seq_len, seq_len]
                # add head dimension：[batch_size, 1, seq_len, seq_len]
                causal_mask = attention_mask.unsqueeze(1)
            elif (
                len(attention_mask.shape) == 4
            ):  # [batch_size, num_heads, seq_len, seq_len]
                causal_mask = attention_mask
            else:
                raise ValueError(f"不支持的attention_mask维度: {attention_mask.shape}")

            # convert the attention mask to bool type
            causal_mask = causal_mask.to(torch.bool)

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal = True if causal_mask is None and q_len > 1 else False

        if q_len == 1:
            is_causal = False
            causal_mask = torch.ones(
                bsz,
                1,
                1,
                key_states.shape[2],
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            ).contiguous()
            causal_mask = causal_mask.to(torch.bool)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        if self.config.mot_opt:
            output = self._generate_output_mot_opt(
                attn_output, token_types, start_indices, end_indices
            )
        else:
            output = self._generate_output(attn_output, masks)

        return output, None, past_key_value

    def _generate_qkv(self, hidden_states, masks):
        bsz, q_len, _ = hidden_states.size()

        query_states = torch.zeros(
            bsz,
            q_len,
            self.num_heads,
            self.head_dim,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        key_states = torch.zeros(
            bsz,
            q_len,
            self.num_key_value_heads,
            self.head_dim,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        value_states = torch.zeros(
            bsz,
            q_len,
            self.num_key_value_heads,
            self.head_dim,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        # for expert_idx in range(len(self.dim_inputs)):
        for expert_idx, (q_proj, k_proj, v_proj, mask) in enumerate(
            zip(self.q_proj_experts, self.k_proj_experts, self.v_proj_experts, masks)
        ):
            if not mask.any():
                continue
            dim_input = self.dim_inputs[expert_idx]

            selected_hidden = hidden_states[mask].clone()

            q_out = q_proj(selected_hidden[:, :dim_input]).view(
                -1, self.num_heads, self.head_dim
            )
            k_out = k_proj(selected_hidden[:, :dim_input]).view(
                -1, self.num_key_value_heads, self.head_dim
            )
            v_out = v_proj(selected_hidden[:, :dim_input]).view(
                -1, self.num_key_value_heads, self.head_dim
            )

            if self.config.model_type == "qwen3_vl_text":
                q_out = self.q_norms[expert_idx](q_out)[0]
                k_out = self.k_norms[expert_idx](k_out)[0]

            query_states[mask] = q_out
            key_states[mask] = k_out
            value_states[mask] = v_out

        return query_states, key_states, value_states

    def _generate_qkv_mot_opt(
        self,
        hidden_states: torch.Tensor,
        experts_indices: torch.Tensor,
        start_indices: torch.Tensor,
        end_indices: torch.Tensor,
        probs: torch.Tensor,
        row_id_map: torch.Tensor,
        batch_size: int,
        seq_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate Q, K, V based on expert-sharded segments (start_indices / end_indices),
        then restore them to the original sequence order.

        Args:
            hidden_states: [total_tokens, hidden_dim], tokens already permuted and grouped by experts
            experts_indices: [B, S], expert index for each token
            start_indices: start token index for each expert (in the permuted token space)
            end_indices: end token index for each expert (in the permuted token space)
            probs: probability vector for each token (used for unpermute)
            batch_size, seq_length: original batch size and sequence length

        Returns:
            query_states: [B, num_heads, S, head_dim]
            key_states:   [B, num_key_value_heads, S, head_dim]
            value_states: [B, num_key_value_heads, S, head_dim]
        """

        total_tokens, hidden_dim = hidden_states.shape
        device, dtype = hidden_states.device, hidden_states.dtype

        # Initialize Q/K/V buffers in the permuted token space
        q_buffer = torch.zeros(total_tokens, hidden_dim, device=device, dtype=dtype)
        k_buffer = torch.zeros(
            total_tokens,
            self.num_key_value_heads * self.head_dim,
            device=device,
            dtype=dtype,
        )
        v_buffer = torch.zeros(
            total_tokens,
            self.num_key_value_heads * self.head_dim,
            device=device,
            dtype=dtype,
        )

        # === Each expert processes its own token slice ===
        for expert_idx, (q_proj, k_proj, v_proj) in enumerate(
            zip(self.q_proj_experts, self.k_proj_experts, self.v_proj_experts)
        ):
            start, end = start_indices[expert_idx], end_indices[expert_idx]
            if start == end:
                continue

            dim_input = self.dim_inputs[expert_idx]
            expert_input = hidden_states[start:end, :dim_input]

            # Compute Q/K/V
            q_out = q_proj(expert_input)
            k_out = k_proj(expert_input)
            v_out = v_proj(expert_input)

            if getattr(self.config, "model_type", None) == "qwen3_vl_text":
                q_out = self.q_norms[expert_idx](q_out)
                q_out = q_out[0] if isinstance(q_out, (tuple, list)) else q_out
                k_out = self.k_norms[expert_idx](k_out)
                k_out = k_out[0] if isinstance(k_out, (tuple, list)) else k_out

            q_buffer[start:end] = q_out
            k_buffer[start:end] = k_out
            v_buffer[start:end] = v_out

        # === Restore tokens to the original order ===
        #  unpermute (using the same unpermute operation)
        q_unpermuted = ops.unpermute(q_buffer, row_id_map, probs)
        k_unpermuted = ops.unpermute(k_buffer, row_id_map, probs)
        v_unpermuted = ops.unpermute(v_buffer, row_id_map, probs)

        # === Reshape to final form ===
        query_states = q_unpermuted.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        )
        key_states = k_unpermuted.view(
            batch_size, seq_length, self.num_key_value_heads, self.head_dim
        )
        value_states = v_unpermuted.view(
            batch_size, seq_length, self.num_key_value_heads, self.head_dim
        )

        return query_states, key_states, value_states

    def _apply_rotary_pos_embed(
        self, query_states, key_states, cos, sin, unsqueeze_dim=1
    ):
        if self.config.model_type == "qwen2_5_vl":
            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states.contiguous(),
                key_states.contiguous(),
                cos.contiguous(),
                sin.contiguous(),
                self.rope_scaling["mrope_section"],
                unsqueeze_dim
            )
        else:
            raise NotImplementedError(f"不支持的模型类型: {self.config.model_type}")
        return query_states, key_states

    def _generate_output(self, attn_output, masks):
        output = torch.zeros(
            *attn_output.shape[:2],
            self.hidden_size,
            device=attn_output.device,
            dtype=attn_output.dtype,
        )
        for expert_idx, (o_proj, mask) in enumerate(zip(self.o_proj_experts, masks)):
            if not mask.any():
                continue
            dim_input = self.dim_inputs[expert_idx]

            # Obtain all necessary indexes in a single operation.
            mask_indices = mask.nonzero(as_tuple=False) 
            if mask_indices.numel() == 0:
                continue

            batch_indices = mask_indices[:, 0]
            seq_indices = mask_indices[:, 1]

            # Use advanced indexing directly to avoid intermediate tensors.
            selected_attn_output = attn_output[batch_indices, seq_indices]
            projected_output = o_proj(selected_attn_output)

            output[batch_indices, seq_indices, :dim_input] = projected_output

        return output

    def _generate_output_mot_opt(
        self,
        attn_output: torch.Tensor,
        experts_indices: torch.Tensor,
        start_indices: torch.Tensor,
        end_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Expert-sharded version of attn_output processing based on start_indices / end_indices.
        Rearranges the [B, S, H] attn_output according to expert order (permute),
        applies the o_proj projection for each expert individually,
        and keeps the final output in expert order ([Tokens, Hidden])
        instead of restoring it back to [B, S, H].

        Args:
            attn_output: [B, S, hidden_dim]
            experts_indices: [B, S], expert index for each token
            start_indices, end_indices: start and end token indices for each expert
                                        (in the permuted token space)

        Returns:
            output_buffer: [TotalTokens, hidden_dim], arranged in expert order
        """

        _, _, hidden_dim = attn_output.shape
        device, dtype = attn_output.device, attn_output.dtype

        # === 1. Flatten and reorder by expert assignment ===
        flat_attn_output = attn_output.view(-1, hidden_dim)  # [B*S, H]
        flat_expert_indices = experts_indices.reshape(-1)  # [B*S]
        permuted_inputs, _ = ops.permute(flat_attn_output, flat_expert_indices)
        total_tokens = permuted_inputs.shape[0]

        # === 2. Initialize output buffer (still in permuted token space) ===
        output_buffer = torch.zeros(
            total_tokens, hidden_dim, device=device, dtype=dtype
        )

        # === 3. Each expert processes its own token segment independently ===
        for expert_idx, o_proj in enumerate(self.o_proj_experts):
            start, end = start_indices[expert_idx], end_indices[expert_idx]
            if start == end:
                continue

            dim_input = self.dim_inputs[expert_idx]
            expert_input = permuted_inputs[start:end]  # [N_e, dim_input]
            expert_output = o_proj(expert_input)  # [N_e, hidden_dim]

            # Write results into the buffer (overwrite only valid dimension region)
            output_buffer[start:end, :dim_input] = expert_output[:, :dim_input]

        # === 4. Return the output ordered by expert sequence ===
        return output_buffer


class JointQwen2VLFlashAttention(JointQwen2VLAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        token_types: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        start_indices: Optional[torch.Tensor] = None,
        end_indices: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        row_id_map: Optional[torch.Tensor] = None,
        orig_shape: Optional[Tuple[int]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        if token_types is None:
            raise ValueError("token_types 不能为空")

        if self.config.mot_opt:
            bsz, q_len, _ = orig_shape
            query_states, key_states, value_states = self._generate_qkv_mot_opt(
                hidden_states,
                token_types,
                start_indices,
                end_indices,
                probs,
                row_id_map,
                bsz,
                q_len,
            )
        else:
            bsz, q_len, _ = hidden_states.size()
            masks = [
                (token_types == expert_idx)
                for expert_idx in range(len(self.dim_inputs))
            ]
            query_states, key_states, value_states = self._generate_qkv(
                hidden_states, masks
            )

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        cos, sin = position_embeddings
        query_states, key_states = self._apply_rotary_pos_embed(
            query_states, key_states, cos, sin, unsqueeze_dim=2
        )

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                self.layer_idx,
                cache_kwargs,
            )
            key_states, value_states = key_states.transpose(
                1, 2
            ), value_states.transpose(1, 2)

        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout_rate,
            softmax_scale=None,
            causal=self.is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()

        if self.config.mot_opt:
            output = self._generate_output_mot_opt(
                attn_output, token_types, start_indices, end_indices
            )
        else:
            output = self._generate_output(attn_output, masks)

        return output, None, past_key_value

JOINT_QWEN_ATTENTION_CLASSES = {
    "eager": JointQwen2VLAttention,
    "flash_attention_2": JointQwen2VLFlashAttention,
    "sdpa": JointQwen2VLAttention,
}
