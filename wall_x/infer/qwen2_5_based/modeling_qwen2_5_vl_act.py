import os
import torch.nn as nn
import torch
from transformers.activations import ACT2FN
from typing import Optional
from flash_attn import (
    flash_attn_with_kvcache,
)

from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import (
    Qwen2_5_VLPreTrainedModel,
    ActionModelMixMin,
    Qwen2_5_VLConfig,
    Qwen2RMSNorm,
)
import flashinfer
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

DUMP_PATH = "/path/to/dump/"
ENABLE_BITWISE_ALIGN = os.environ.get("ENABLE_BITWISE_ALIGN", "False").lower() == "true"
if ENABLE_BITWISE_ALIGN:
    USE_FLASHINFER = False
    USE_MERGED_QKV = False
    USE_MERGED_GATE_UP = False
    VITOPT_INFER = False
else:
    USE_FLASHINFER = os.environ.get("USE_FLASHINFER", "False").lower() == "true"
    USE_MERGED_QKV = os.environ.get("USE_MERGED_QKV", "False").lower() == "true"
    USE_MERGED_GATE_UP = os.environ.get("USE_MERGED_GATE_UP", "False").lower() == "true"
    VITOPT_INFER = os.getenv("VITOPT_INFER", "False").lower() == "true"
ENABLE_CUDA_GRAPH = os.environ.get("ENABLE_CUDA_GRAPH", "True").lower() == "true"

try:
    from xcompute import m_rope
except ImportError:
    raise ImportError("Please install xcompute from gitlab")

# use green
GREEN = "\033[92m"
RESET = "\033[0m"
print(f"{GREEN}USE_FLASHINFER: {USE_FLASHINFER}{RESET}")
print(f"{GREEN}USE_MERGED_QKV: {USE_MERGED_QKV}{RESET}")
print(f"{GREEN}USE_MERGED_GATE_UP: {USE_MERGED_GATE_UP}{RESET}")
print(f"{GREEN}ENABLE_CUDA_GRAPH: {ENABLE_CUDA_GRAPH}{RESET}")
print(f"{GREEN}VITOPT_INFER: {VITOPT_INFER}{RESET}")


def save_tensor(tensor, name, layer_idx):
    # Bypass
    return None
    if layer_idx == 0 or layer_idx == 1:
        torch.save(tensor, os.path.join(DUMP_PATH, f"{name}_{layer_idx}.pt"))


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class QKV_Merged_Linear(nn.Module):
    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.vocab_size = config.vocab_size
        self.qkv_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size + self.num_kv_heads * self.head_dim * 2,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor):
        # TODO: support fused linear kernel in Xcompute
        qkv = self.qkv_proj(hidden_states)
        return qkv


class OutProjection(nn.Module):
    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        # return gemm(hidden_states, self.o_proj.weight)
        return self.o_proj(hidden_states)


class Qwen2_5_Attention(nn.Module):
    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.rope_scaling = config.rope_scaling
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.qkv_proj = QKV_Merged_Linear(config)
        self.o_proj = OutProjection(config)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        kvcache: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cache_leftpad: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
    ):
        save_tensor(hidden_states, "attn_hidden_states", self.layer_idx)
        bz, seq_len, _ = hidden_states.shape
        if USE_MERGED_QKV:
            qkv = self.qkv_proj(hidden_states)
            q, k, v = qkv.split(
                [
                    self.hidden_size,
                    self.num_kv_heads * self.head_dim,
                    self.num_kv_heads * self.head_dim,
                ],
                dim=-1,
            )
        else:
            if not hasattr(self, "wq"):
                self.wq, self.wk, self.wv = self.qkv_proj.qkv_proj.weight.split(
                    [
                        self.hidden_size,
                        self.num_kv_heads * self.head_dim,
                        self.num_kv_heads * self.head_dim,
                    ],
                    dim=0,
                )
                self.wq_bias, self.wk_bias, self.wv_bias = (
                    self.qkv_proj.qkv_proj.bias.split(
                        [
                            self.hidden_size,
                            self.num_kv_heads * self.head_dim,
                            self.num_kv_heads * self.head_dim,
                        ],
                        dim=0,
                    )
                )
            q = torch.nn.functional.linear(hidden_states, self.wq, bias=self.wq_bias)
            k = torch.nn.functional.linear(hidden_states, self.wk, bias=self.wk_bias)
            v = torch.nn.functional.linear(hidden_states, self.wv, bias=self.wv_bias)
        cos, sin = position_embeddings

        q = q.reshape(q.shape[0], q.shape[1], -1, self.head_dim)
        k = k.reshape(k.shape[0], k.shape[1], -1, self.head_dim)
        v = v.reshape(v.shape[0], v.shape[1], -1, self.head_dim)

        # TODO(Dance): high performance mrope
        q, k = m_rope(q, k, cos, sin, self.rope_scaling["mrope_section"])

        # TODO: optimize attn with kvcache system
        # 1. use remove padding logic(varlen) in vlm phase
        # 2. use block wise kvcache system in flow phase
        if USE_FLASHINFER:
            out_attn = flash_attn_with_kvcache(
                q,
                kvcache[0],
                kvcache[1],
                k,
                v,
                rotary_cos=None,
                rotary_sin=None,
                cache_seqlens=cache_seqlens,
                cache_batch_idx=None,
                cache_leftpad=cache_leftpad,
                block_table=None,
                causal=True,
                window_size=(-1, -1),
                rotary_interleaved=False,
                alibi_slopes=None,
                num_splits=1,
            )
        else:
            q = q.transpose(1, 2)
            kcache = kvcache[0]
            vcache = kvcache[1]
            kcache = kcache.transpose(1, 2)
            vcache = vcache.transpose(1, 2)
            # 根据 attention_mask 的 L_k 维度来切片 kcache 和 vcache
            # attention_mask 形状: [B, L_q, L_k]
            if attention_mask is not None:
                actual_cache_len = attention_mask.shape[-1]  # L_k
                kcache = kcache[:, :, :actual_cache_len, :]
                vcache = vcache[:, :, :actual_cache_len, :]
            attention_mask = (
                attention_mask.unsqueeze(1) if attention_mask is not None else None
            )  # [B, L_q, L_k] -> [B, 1, L_q, L_k]
            kcache = repeat_kv(kcache, self.num_key_value_groups)
            vcache = repeat_kv(vcache, self.num_key_value_groups)
            q = q.contiguous()
            kcache = kcache.contiguous()
            vcache = vcache.contiguous()
            out_attn = torch.nn.functional.scaled_dot_product_attention(
                q,
                kcache,
                vcache,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
            out_attn = out_attn.transpose(1, 2).contiguous()

        save_tensor(out_attn, "attn_out", self.layer_idx)
        out_attn = out_attn.reshape(bz, seq_len, -1)
        # output = gemm(out_attn, self.o_proj.o_proj.weight)
        output = self.o_proj(out_attn)
        save_tensor(output, "attn_o_proj", self.layer_idx)
        return output


class Qwen2_5_ACT_Expert_MLP(nn.Module):
    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: int, expert_index: int):
        # TODO: unify vlm phase and dcd phase
        super().__init__()
        config = config.experts[expert_index]
        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config["intermediate_size"]
        self.hidden_act = config["hidden_act"]
        self.gate_up_proj = nn.Linear(
            self.hidden_size, 2 * self.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[self.hidden_act]
        self.layer_idx = layer_idx

    def forward(self, hidden_states: torch.Tensor):
        # TODO: fix merge gate up proj kernel
        if USE_MERGED_GATE_UP:
            # gate_up = gemm(hidden_states, self.gate_up_proj.weight)
            gate_up = self.gate_up_proj(hidden_states)
            gate, up = gate_up.split(
                [self.intermediate_size, self.intermediate_size], dim=-1
            )
        else:
            if not hasattr(self, "w_gate"):
                self.w_gate, self.w_up = self.gate_up_proj.weight.split(
                    [self.intermediate_size, self.intermediate_size], dim=0
                )
            gate = torch.nn.functional.linear(hidden_states, self.w_gate)
            up = torch.nn.functional.linear(hidden_states, self.w_up)
        # TODO: fix fused activ multiply kernel
        # act = fused_activ_multiply(gate, up, self.hidden_act)
        act = self.act_fn(gate) * up
        # output = gemm(act, self.down_proj.weight)
        output = self.down_proj(act)
        return output


class Qwen2_5_VL_ACT_DecodeLayer(nn.Module):
    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: int, expert_index: int):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.layer_idx = layer_idx

        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.self_attn = Qwen2_5_Attention(config, layer_idx)
        self.act_expert_mlp = Qwen2_5_ACT_Expert_MLP(config, layer_idx, expert_index)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        kvcache: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cache_leftpad: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
    ):
        residual = hidden_states
        save_tensor(residual, "residual", self.layer_idx)
        # TODO: Add qwen2 rmsnorm kernel
        hidden_states = flashinfer.norm.rmsnorm(
            hidden_states,
            self.input_layernorm.weight,
            self.input_layernorm.variance_epsilon,
        )
        save_tensor(
            self.input_layernorm.weight, "input_layernorm_weight", self.layer_idx
        )
        hidden_states = self.self_attn(
            hidden_states,
            position_embeddings,
            kvcache,
            cache_seqlens,
            cache_leftpad,
            attention_mask,
        )
        # TODO: fuse residual add kernel to last gemm
        hidden_states = residual + hidden_states

        residual = hidden_states
        # TODO: Add qwen2 rmsnorm kernel 此处不融合性能更高
        hidden_states = flashinfer.norm.rmsnorm(
            hidden_states,
            self.post_attention_layernorm.weight,
            self.input_layernorm.variance_epsilon,
        )
        hidden_states = self.act_expert_mlp(hidden_states)
        # TODO: fuse residual add kernel to last gemm
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen2_5_VLRotaryEmbedding_Decode(nn.Module):
    def __init__(self, config: Qwen2_5_VLConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer(
                "inv_freq", inv_freq, persistent=False
            )  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if (
            seq_len < self.original_max_seq_len
            and self.max_seq_len_cached > self.original_max_seq_len
        ):  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block. In contrast to other models, Qwen2_5_VL has different position ids for thw grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None]
            .float()
            .expand(3, position_ids.shape[1], -1, 1)
        )
        position_ids_expanded = position_ids[
            :, :, None, :
        ].float()  # shape (3, bs, 1, positions)
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(2, 3)
            cos = freqs.cos()
            sin = freqs.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen2_5_VL_ACT_Decode(Qwen2_5_VLPreTrainedModel, ActionModelMixMin):
    def __init__(
        self, config: Qwen2_5_VLConfig, expert_index: int, kvcache: torch.Tensor
    ):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.layers = nn.ModuleList(
            [
                Qwen2_5_VL_ACT_DecodeLayer(config, layer_idx, expert_index)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.rotary_emb = Qwen2_5_VLRotaryEmbedding_Decode(config=config)
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.max_cache_len = config.max_position_embeddings
        self.kvcache = kvcache
        self.kv_len = 0
        self.attention_mask = None

    def forward(
        self,
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cache_leftpad: Optional[torch.Tensor],
        attention_mask: torch.Tensor = None,
    ):
        hidden_states = input_embeds
        # TODO: support cache
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, layer in enumerate(self.layers):
            # TODO: optimize kvcache system
            hidden_states = layer(
                hidden_states,
                position_embeddings,
                self.kvcache[:, layer_idx, :],
                cache_seqlens,
                cache_leftpad,
                self.attention_mask,
            )
        save_tensor(hidden_states, "hidden_states_after_decoder_layers", 0)
        hidden_states = flashinfer.norm.rmsnorm(
            hidden_states, self.norm.weight, self.norm.variance_epsilon
        )
        return hidden_states

    def preprocess_kvcache(self, seq_len: int):
        self.kv_len = seq_len

    def preprocess_attn_mask(self, attn_mask: torch.Tensor):
        bz, pred_horizon, seq_len = attn_mask.shape
        need_alloc = (
            not hasattr(self, "attention_mask_tensor")
            or self.attention_mask_tensor.shape[0] != bz
            or self.attention_mask_tensor.shape[1] != pred_horizon
            or self.attention_mask_tensor.shape[2] < seq_len
        )
        if need_alloc:
            cap_len = max(self.max_cache_len, seq_len)
            self.attention_mask_tensor = torch.ones(
                (bz, pred_horizon, cap_len),
                device=attn_mask.device,
                dtype=attn_mask.dtype,
            ).contiguous()
        self.attention_mask = self.attention_mask_tensor[:bz, :, :seq_len]
        self.attention_mask.copy_(attn_mask)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def to_bfloat16_for_selected_params(self):
        self.to(dtype=torch.float32)

        params_to_keep_float32 = []

        # for name, param in self.named_parameters():
        #     if (
        #         "input_layernorm" in name
        #         or "post_attention_layernorm" in name
        #         or "norm" in name
        #     ):
        #         params_to_keep_float32.append(name)
        #     if "action_preprocessor" in name:
        #         params_to_keep_float32.append(name)

        for name, param in self.named_parameters():
            if name not in params_to_keep_float32:
                param.data = param.data.to(torch.bfloat16)
