import torch
import os
import numpy as np
from transformers import AutoProcessor
from wall_x.model.action_head import Normalizer
from wall_x.data.load_lerobot_dataset import load_test_dataset, get_data_configs
from wall_x.utils.constant import action_statistic_dof as default_action_statistic_dof
import json
def update_model_config(train_config, model_config):
    model_config.use_state_string_representation = train_config["data"].get(
        "use_state_string_representation", False
    )
    model_config.flow_loss_weight = train_config.get("flow_loss_weight", 1.0)

    model_config.dof_config = train_config["dof_config"]
    model_config.agent_pos_config = train_config["agent_pos_config"]

    model_config.action_horizon_flow = train_config["data"].get(
        "action_horizon_flow", 32
    )

    if train_config.get("_attn_implementation", None) is not None:
        model_config._attn_implementation = train_config["_attn_implementation"]

    return model_config

def load_wallx_processors(config):
    processor = AutoProcessor.from_pretrained(config["processor_path"], use_fast=True)
    # pad side = left
    processor.tokenizer.padding_side = "left"

    new_tokens = ["<|propri|>", "<|action|>"]
    # special_tokens = []
    action_tokenizer_type = config.get("action_tokenizer_type", None)
    if action_tokenizer_type == "fast":
        train_action_tokenizer = AutoProcessor.from_pretrained(
            config["action_tokenizer_path"], trust_remote_code=True
        )
        val_action_tokenizer = AutoProcessor.from_pretrained(
            config["action_tokenizer_path"], trust_remote_code=True
        )
        new_tokens += [
            f"<|action_token_{i}|>" for i in range(train_action_tokenizer.vocab_size)
        ]
    elif action_tokenizer_type == "spatialvla":
        raise NotImplementedError("SpatialActionTokenizer is not implemented")
    else:
        train_action_tokenizer = None
        val_action_tokenizer = None

    num_added_tokens = processor.tokenizer.add_tokens(new_tokens)

    if action_tokenizer_type and train_action_tokenizer.vocab_size > 0:
        action_mapper = {}
        for i in range(train_action_tokenizer.vocab_size):
            token = f"<|action_token_{i}|>"
            token_id = processor.tokenizer.convert_tokens_to_ids(token)
            action_mapper[token_id] = i
    else:
        action_mapper = None

    return {
        "processor": processor,
        "train_action_tokenizer": train_action_tokenizer,
        "val_action_tokenizer": val_action_tokenizer,
        "action_mapper": action_mapper,
        "num_added_tokens": num_added_tokens,
    }


def register_normalizers(config, model_path):
    # if config.get("customized_action_statistic_dof", None):
    #     action_statistic_dof = json.load(open(config["customized_action_statistic_dof"], "r"))
    # else:
    #     action_statistic_dof = default_action_statistic_dof

    action_statistic_dof = None

    if os.path.exists(model_path+"/normalizer_action.pth"):
        print("Loading normalizer_action from checkpoint", model_path+"/normalizer_action.pth", flush=True)
        normalizer_action = Normalizer.from_ckpt(model_path+"/normalizer_action.pth")
    else:
        normalizer_action = Normalizer(
            action_statistic_dof, config["dof_config"],
            min_key=config.get("min_key", "min"), 
            delta_key=config.get("delta_key", "delta")
        )

    # print("action_statistic_dof",action_statistic_dof)

    if os.path.exists(model_path+"/normalizer_propri.pth"):
        print("Loading normalizer_propri from checkpoint", model_path+"/normalizer_propri.pth", flush=True)
        normalizer_propri = Normalizer.from_ckpt(model_path+"/normalizer_propri.pth")
    else:
        normalizer_propri = Normalizer(
            action_statistic_dof, config["agent_pos_config"],
            min_key=config.get("min_key", "min"),
            delta_key=config.get("delta_key", "delta")
        )

    return normalizer_action, normalizer_propri

def find_first_last_ones(tensor):
    """
    Input: a tensor of shape (bs, seq_len) containing 0s and 1s
    Output: (first_indices, last_indices), each of shape (bs,)
    where first_indices[i] is the index of the first 1 in the i-th batch, or -1 if none exists.
    last_indices[i] is the index of the last 1 in the i-th batch, or -1 if none exists.
    """
    bs, seq_len = tensor.shape
    masks = tensor == 1
    has_ones = masks.any(dim=1)

    first = torch.full((bs,), -1, dtype=torch.long, device=tensor.device)
    last = first.clone()

    first[has_ones] = torch.argmax(masks[has_ones].float(), dim=1)

    flipped_masks = masks.flip(dims=[1])
    last_argmax = torch.argmax(flipped_masks[has_ones].float(), dim=1)
    last[has_ones] = seq_len - 1 - last_argmax

    return first, last


def flashmask_to_densemask(startend_row_indices, dtype, causal=True):
    if startend_row_indices is None:
        return None
    bz, num_head, seq_len, bound_num = startend_row_indices.shape
    m = np.ones((bz, num_head, seq_len, seq_len), dtype=dtype)
    has_end = (causal and bound_num == 2) or ((not causal) and bound_num == 4)
    for bi in range(bz):
        for hi in range(num_head):
            for j in range(seq_len):
                downstart = startend_row_indices[bi, hi, j, 0]
                if has_end:
                    downend = startend_row_indices[bi, hi, j, 1]
                    m[bi, hi, downstart:downend, j] = 0
                else:
                    m[bi, hi, downstart:, j] = 0
                if causal:
                    m[bi, hi, :j, j] = 0
                else:
                    if has_end:
                        upstart = startend_row_indices[bi, hi, j, 2]
                        upend = startend_row_indices[bi, hi, j, 3]
                        m[bi, hi, upstart:upend, j] = 0
                    else:
                        upend = startend_row_indices[bi, hi, j, 1]
                        m[bi, hi, :upend, j] = 0
    return m


def num_floating_point_operations(
    args,
    batch_size: int,
    num_lang_tokens: int,
    num_action_tokens: int,
    vision_seq_length: int = 756,
):
    """
    Accurately estimate the training FLOPs of Transformer + MoE + MoT + Vision. 

    Supported:
      - expert0 = language tokens
      - expert1 = action tokens
      - MoE MLP (2 experts)
      - MoT Attention (2 experts)
      - GQA
      - Vision Transformer (full+window attention)
    """
    assert args.num_experts == 2, "The current model only supports 2 experts."

    dim_lang, dim_act = args.dim_inputs

    # 每层 token 数量（batch 展开）
    N_lang = batch_size * num_lang_tokens
    N_action = batch_size * num_action_tokens
    N_total = N_lang + N_action  # 用于 non-MoT attention

    # ================================================================
    # 文本 MLP FLOPs
    # ================================================================
    hidden_size = args.hidden_size
    ffn_hidden_size = args.intermediate_size
    num_layers = args.num_hidden_layers

    use_moe_mlp = getattr(args, "mlp_moe", False)

    # ---------- Forward-only MLP FLOPs ----------
    def forward_mlp_flops(N, d_in, d_ff):
        """
        SwiGLU forward:
            gate = x @ W1   (2*N*d_in*d_ff)
            up   = x @ W2   (2*N*d_in*d_ff)
            act  = silu + mul   (~2*N*d_ff)
            down = h @ W3   (2*N*d_ff*d_in)

        Forward ≈ 4*N*d_in*d_ff + 2*N*d_ff*d_in = 6*N*d_in*d_ff + 2*N*d_ff
        """
        return 6 * N * d_in * d_ff + 2 * N * d_ff

    if not use_moe_mlp:
        # Dense MLP
        F_fwd = forward_mlp_flops(N_total, hidden_size, ffn_hidden_size)
        total_mlp_flops_text = 3 * num_layers * F_fwd  # <-- training FLOPs
    else:
        # MoE: expert0(language) + expert1(action)
        hid_lang = args.experts[0]["intermediate_size"]
        hid_act = args.experts[1]["intermediate_size"]

        F_lang_fwd = forward_mlp_flops(N_lang, dim_lang, hid_lang)
        F_act_fwd = forward_mlp_flops(N_action, dim_act, hid_act)

        total_mlp_flops_text = 3 * num_layers * (F_lang_fwd + F_act_fwd)

    # ================================================================
    # Text Attention FLOPs
    # ================================================================
    num_heads = args.num_attention_heads
    num_kv = args.num_key_value_heads
    H = hidden_size
    B = batch_size
    S = num_lang_tokens + num_action_tokens
    N = B * S

    use_mot = getattr(args, "attention_moe", False)

    # ---------- attention matmul ----------
    F_matmul_fwd = 4 * B * (S**2) * H

    if not use_mot:
        # -------- GQA + QKV / O --------
        F_q_fwd = 2 * N * H * H
        F_kv_fwd = 4 * N * H * H * (num_kv / num_heads)
        F_o_fwd = 2 * N * H * H

        F_attn_fwd = F_q_fwd + F_kv_fwd + F_o_fwd + F_matmul_fwd

    else:
        # -------- MoT：expert0 + expert1  QKV --------
        F_lang_qkv = N_lang * dim_lang * H * (2 + 4 * num_kv / num_heads)
        F_act_qkv = N_action * dim_act * H * (2 + 4 * num_kv / num_heads)
        F_attn_fwd = F_lang_qkv + F_act_qkv + F_matmul_fwd

    # Training FLOPs
    total_attn_flops_text = 3 * num_layers * F_attn_fwd

    # ================================================================
    # Logits projection FLOPs
    # ================================================================
    vocab_size = getattr(args, "padded_vocab_size", args.vocab_size)

    F_logits_fwd = 2 * N * H * vocab_size
    total_logits_flops = 3 * F_logits_fwd

    total_text_flops = total_mlp_flops_text + total_attn_flops_text + total_logits_flops

    # ================================================================
    # Vision Transformer FLOPs
    # ================================================================
    total_vision_flops = 0

    if hasattr(args, "vision_config") and vision_seq_length is not None:
        vcfg = args.vision_config

        Bv = batch_size
        Sv = vision_seq_length
        Nv = Bv * Sv

        Hv = vcfg.hidden_size
        Iv = vcfg.intermediate_size
        num_heads_v = vcfg.num_heads
        window_size = vcfg.window_size
        out_hidden = vcfg.out_hidden_size

        depth_v = vcfg.depth
        fullatt = set(vcfg.fullatt_block_indexes)
        num_full = len(fullatt)
        num_local = depth_v - num_full

        # ---------- forward FLOPs ----------
        def forward_vit_mlp(N, H, Inner):
            return 6 * N * H * Inner + 2 * N * Inner

        F_mlp_v = forward_vit_mlp(Nv, Hv, Iv)
        F_qkv_v = 6 * Nv * Hv * Hv
        F_o_v = 2 * Nv * Hv * Hv

        F_full = 4 * Bv * (Sv**2) * Hv
        num_windows = Sv / window_size
        F_win = 4 * Bv * num_windows * (window_size**2) * (Hv / num_heads_v)

        F_block_full_fwd = F_mlp_v + F_qkv_v + F_o_v + F_full
        F_block_local_fwd = F_mlp_v + F_qkv_v + F_o_v + F_win

        # ---------- train FLOPs ----------
        total_vision_flops = 3 * (
            num_full * F_block_full_fwd + num_local * F_block_local_fwd
        )

        # merger
        total_vision_flops += 3 * (2 * Nv * Hv * out_hidden)

    # ================================================================
    # TOTAL TRAIN FLOPs
    # ================================================================
    return total_text_flops + total_vision_flops
