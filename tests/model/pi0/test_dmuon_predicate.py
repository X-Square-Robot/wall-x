import dataclasses

import torch

from wall_x.config.hyperparams_config import DMuonConfig
from wall_x.config.loader import _build_optimizer_config
from wall_x.model.pi0.dmuon_utils import is_pi0_dmuon_target_param

PALIGEMMA = "paligemma_with_expert.paligemma"
GEMMA_EXPERT = "paligemma_with_expert.gemma_expert"
VISION_ENCODER = f"{PALIGEMMA}.model.vision_tower.vision_model.encoder"
VISION_EMBEDDINGS = f"{PALIGEMMA}.model.vision_tower.vision_model.embeddings"


def _param(shape, *, requires_grad=True):
    return torch.nn.Parameter(torch.empty(*shape), requires_grad=requires_grad)


def test_pi0_dmuon_predicate_selects_trainable_projection_weights():
    selected = [
        f"{PALIGEMMA}.model.language_model.layers.0.self_attn.q_proj.weight",
        f"{PALIGEMMA}.model.language_model.layers.0.self_attn.k_proj.weight",
        f"{PALIGEMMA}.model.language_model.layers.0.self_attn.v_proj.weight",
        f"{PALIGEMMA}.model.language_model.layers.0.self_attn.o_proj.weight",
        f"{GEMMA_EXPERT}.model.layers.0.mlp.gate_proj.weight",
        f"{GEMMA_EXPERT}.model.layers.0.mlp.up_proj.weight",
        f"{GEMMA_EXPERT}.model.layers.0.mlp.down_proj.weight",
        f"{VISION_ENCODER}.layers.0.self_attn.out_proj.weight",
        f"{VISION_ENCODER}.layers.0.mlp.fc1.weight",
        f"{VISION_ENCODER}.layers.0.mlp.fc2.weight",
        f"{PALIGEMMA}.model.multi_modal_projector.linear.weight",
        "action_in_proj.weight",
        "action_out_proj.weight",
        "state_proj.weight",
        "time_mlp_in.weight",
        "time_mlp_out.weight",
        "action_time_mlp_in.weight",
        "action_time_mlp_out.weight",
    ]

    for name in selected:
        assert is_pi0_dmuon_target_param(name, _param((128, 128))), name


def test_pi0_dmuon_predicate_excludes_non_targets():
    rejected = [
        (f"{PALIGEMMA}.model.language_model.embed_tokens.weight", (128, 128)),
        (f"{PALIGEMMA}.lm_head.weight", (128, 128)),
        (f"{VISION_EMBEDDINGS}.position_embedding.weight", (128, 128)),
        (f"{VISION_EMBEDDINGS}.patch_embedding.weight", (64, 3, 14, 14)),
        (f"{PALIGEMMA}.model.vision_tower.vision_model.head.probe", (1, 1, 128)),
        (f"{GEMMA_EXPERT}.model.layers.0.input_layernorm.weight", (128,)),
        (f"{GEMMA_EXPERT}.model.layers.0.self_attn.q_proj.bias", (128,)),
    ]

    for name, shape in rejected:
        assert not is_pi0_dmuon_target_param(name, _param(shape)), name

    assert not is_pi0_dmuon_target_param(
        f"{GEMMA_EXPERT}.model.layers.0.self_attn.q_proj.weight",
        _param((128, 128), requires_grad=False),
    )


def test_dmuon_config_exposes_only_current_public_api_fields():
    fields = {field.name for field in dataclasses.fields(DMuonConfig)}

    assert "per_head_ns" not in fields
    assert "block_diagonal_ns" not in fields
    assert {"ns_steps", "ns_backend", "nesterov"} <= fields


def test_legacy_removed_dmuon_fields_are_ignored_by_loader():
    opt_cfg = _build_optimizer_config(
        {
            "optimizer_type": "dmuon",
            "per_head_ns": True,
            "block_diagonal_ns": False,
        }
    )

    assert isinstance(opt_cfg, DMuonConfig)
    assert not hasattr(opt_cfg, "per_head_ns")
    assert not hasattr(opt_cfg, "block_diagonal_ns")
