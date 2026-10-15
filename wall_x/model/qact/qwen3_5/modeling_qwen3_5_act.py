import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass
from typing import Optional


from . import initialization as init
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    ModelOutput,
)
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.modeling_layers import GradientCheckpointingLayer

from wall_x.model.qact.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5TextConfig,
    Qwen3_5Config,
)
from wall_x.model.qact.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5RMSNorm,
    Qwen3_5AdaptiveRMSNorm,
)
from wall_x.model.qact.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
from wall_x.model.qact.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5VisionModel,
    Qwen3_5TextModel,
    Qwen3_5VisionBlock,
    Qwen3_5VisionPatchEmbed,
    Qwen3_5VisionPatchMerger,
    create_causal_mask,
    Qwen3_5DynamicCache,
)
from wall_x.model.qact.qwen3_5.qwen3_5_vla_vision import (
    Qwen3_5VLAVisionConfig,
    Qwen3_5VLAVisionModel,
    Qwen3_5VLAVisionBlock,
)
from wall_x.model.qact.qwen3_5.dino_vision import Qwen3_5DinoVisionModel
from wall_x.model.qact.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Model,
    Qwen3_5MLP,
    Qwen3_5MLP_VLA,
    Qwen3_5ForConditionalGeneration,
)
from wall_x.model.qact.qwen3_5.qwen3_5_vla_attention import (
    Qwen3_5_VLA_Attention,
    Qwen3_5_VLA_GatedDeltaNet,
    _flash_mask_ki_prefix_lens,
)
from wall_x.model.core.vla_mixin import (
    ActionModelMixMin,
    ActionGenerationMixin,
)
from wall_x.model.core.attention.selector import (
    ATTENTION_TYPES_WITH_2D_MASK,
    ATTENTION_TYPES_WITH_FLASH_MASK,
)
from wall_x.model.core.action.layout import (
    TwoExpertLayout,
    prepare_two_expert_layout,
    prepare_two_expert_layout_packed,
)
from wall_x.model.core.ops.packed_position_ids import packed_position_ids
from wall_x.model.core.action.processor import ActionProcessor
from wall_x.model.core.refiner import RefinerConfig, install_twins
from wall_x.model.qact.qwen3_5.qwen3_5_inference_mixin import Qwen3_5VLAInferenceMixin


_QWEN35_DMUON_BLOCKED_NAME_PARTS = (
    "embed_tokens",
    "pos_embed",
    "lm_head",
    "norm",
)

_QWEN35_DMUON_SHARDED_ADAMW_NAME_PARTS = (
    "embed_tokens",
    "lm_head",
)


def _resolve_pad_token_id(config: Qwen3_5Config) -> int:
    """Resolve the token id used by fully masked packed-DP dummy spans."""
    pad_token_id = getattr(config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(config.text_config, "pad_token_id", None)
    return int(pad_token_id or 0)


def _is_qwen35_dmuon_target_param(name: str, param: nn.Parameter) -> bool:
    """Select trainable Qwen3.5 VLA matrices for DMuon."""
    if not param.requires_grad or param.ndim != 2 or not name.endswith(".weight"):
        return False
    return not any(part in name for part in _QWEN35_DMUON_BLOCKED_NAME_PARTS)


def _is_qwen35_dmuon_sharded_adamw_param(name: str, param: nn.Parameter) -> bool:
    if not param.requires_grad or param.ndim != 2 or not name.endswith(".weight"):
        return False
    return any(part in name for part in _QWEN35_DMUON_SHARDED_ADAMW_NAME_PARTS)


def _is_qwen35_dmuon_trainable_param(name: str, param: nn.Parameter) -> bool:
    del name
    return bool(param.requires_grad)


def _qwen35_pure_dmuon_route_hint(name: str, param: nn.Parameter) -> Optional[str]:
    if not param.requires_grad:
        return None
    if _is_qwen35_dmuon_target_param(name, param):
        return "muon"
    if _is_qwen35_dmuon_sharded_adamw_param(name, param):
        return "sharded_adamw"
    return "adamw"


def _format_qwen35_route_examples(examples: dict[str, list[str]]) -> str:
    return ", ".join(
        f"{route}={names}" for route, names in sorted(examples.items()) if names
    )


def _validate_qwen35_pure_dmuon_route_plan(
    named_parameters,
) -> tuple[dict[str, int], dict[str, list[str]]]:
    route_counts: dict[str, int] = {}
    route_examples: dict[str, list[str]] = {
        "muon": [],
        "adamw": [],
        "sharded_adamw": [],
    }
    bad_routes: list[tuple[str, str]] = []

    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        route = _qwen35_pure_dmuon_route_hint(name, param)
        route_key = str(route)
        route_counts[route_key] = route_counts.get(route_key, 0) + 1
        route_examples.setdefault(route_key, [])
        if len(route_examples[route_key]) < 8:
            route_examples[route_key].append(name)

        if route == "sharded_adamw" and not _is_qwen35_dmuon_sharded_adamw_param(
            name, param
        ):
            bad_routes.append((name, "unexpected sharded_adamw"))
        if route == "muon" and not _is_qwen35_dmuon_target_param(name, param):
            bad_routes.append((name, "unexpected muon"))
        if route == "adamw" and (
            _is_qwen35_dmuon_target_param(name, param)
            or _is_qwen35_dmuon_sharded_adamw_param(name, param)
        ):
            bad_routes.append((name, "unexpected adamw"))

    if bad_routes:
        preview = bad_routes[:16]
        raise RuntimeError(f"Invalid Qwen3.5 Pure DMuon route plan: {preview}")
    return route_counts, route_examples


@dataclass
class Qwen3_5VLAModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    rope_deltas: torch.LongTensor | None = None
    # loss related
    loss: Optional[torch.FloatTensor] = None
    flow_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    cross_entropy_loss: Optional[torch.FloatTensor] = None
    channel_loss_dict: Optional[dict[torch.FloatTensor]] = None
    channel_loss_count_dict: Optional[dict[torch.FloatTensor]] = None
    flow: Optional[torch.FloatTensor] = None
    # for vla
    action_embeds: torch.FloatTensor | None = None
    ar_action_loss_weight: Optional[torch.FloatTensor] = None


class Qwen3_5VLADecoderLayer(GradientCheckpointingLayer):
    def __init__(
        self,
        config: Qwen3_5TextConfig,
        layer_idx: int,
        use_selective_recompute: bool = False,
    ):
        super().__init__()
        self.config = config
        self.use_selective_recompute = use_selective_recompute
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5_VLA_GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5_VLA_Attention(config, layer_idx)

        # mlp moe
        if self.config.vla_config.mlp_moe:
            self.mlp = Qwen3_5MLP_VLA(
                config.vla_config.dim_inputs[0],
                config.intermediate_size,
                config.hidden_act,
                use_selective_recompute=use_selective_recompute,
            )
            self.mlp_action_expert = Qwen3_5MLP_VLA(
                config.vla_config.dim_inputs[1],
                config.vla_config.action_mlp_intermediate_size,
                config.hidden_act,
                use_selective_recompute=use_selective_recompute,
            )
        else:
            self.mlp = Qwen3_5MLP(
                config,
                config.intermediate_size,
                use_selective_recompute=use_selective_recompute,
            )

        # norm moe
        if self.config.vla_config.norm_moe:
            self.input_layernorm = Qwen3_5RMSNorm(
                config.vla_config.dim_inputs[0], eps=config.rms_norm_eps
            )
            self.post_attention_layernorm = Qwen3_5RMSNorm(
                config.vla_config.dim_inputs[0], eps=config.rms_norm_eps
            )
            self.input_layernorm_action = Qwen3_5AdaptiveRMSNorm(
                config.vla_config.dim_inputs[1],
                eps=config.rms_norm_eps,
                cond_dim=config.vla_config.adarms_cond_dim,
            )
            self.post_attention_layernorm_action = Qwen3_5AdaptiveRMSNorm(
                config.vla_config.dim_inputs[1],
                eps=config.rms_norm_eps,
                cond_dim=config.vla_config.adarms_cond_dim,
            )
        else:
            self.input_layernorm = Qwen3_5RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_attention_layernorm = Qwen3_5RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

    def _apply_mlp(self, hidden_states, action_embeds=None):
        if self.config.vla_config.mlp_moe:
            hidden_states = self.mlp(hidden_states)
            if action_embeds is not None:
                action_embeds = self.mlp_action_expert(action_embeds)
        else:
            hidden_states = self.mlp(hidden_states)
            if action_embeds is not None:
                action_embeds = self.mlp(action_embeds)
        return hidden_states, action_embeds

    def _apply_input_layernorm(
        self, hidden_states, action_embeds=None, adarms_conds=None
    ):
        if self.use_selective_recompute:
            normed_hidden_states = cp.checkpoint(
                self.input_layernorm, hidden_states, use_reentrant=False
            )
        else:
            normed_hidden_states = self.input_layernorm(hidden_states)
        if self.config.vla_config.norm_moe and action_embeds is not None:
            normed_action_embeds, gate = self._apply_action_adaptive_norm(
                self.input_layernorm_action, action_embeds, adarms_conds
            )
            return normed_hidden_states, normed_action_embeds, gate
        return normed_hidden_states, None, None

    def _apply_action_adaptive_norm(self, norm_module, action_embeds, adarms_conds):
        if self.use_selective_recompute:
            return cp.checkpoint(
                norm_module, action_embeds, adarms_conds, use_reentrant=False
            )
        return norm_module(action_embeds, adarms_conds)

    def _apply_post_attention_layernorm(
        self, hidden_states, action_embeds=None, adarms_conds=None
    ):
        if self.use_selective_recompute:
            normed_hidden_states = cp.checkpoint(
                self.post_attention_layernorm, hidden_states, use_reentrant=False
            )
        else:
            normed_hidden_states = self.post_attention_layernorm(hidden_states)
        if self.config.vla_config.norm_moe and action_embeds is not None:
            normed_action_embeds, gate = self._apply_action_adaptive_norm(
                self.post_attention_layernorm_action, action_embeds, adarms_conds
            )
            return normed_hidden_states, normed_action_embeds, gate
        return normed_hidden_states, None, None

    def _apply_gated_residual(self, action_residual, action_embeds, gate):
        if action_embeds is None:
            return None
        return action_residual + action_embeds * gate

    def _attn_section(
        self,
        hidden_states,
        action_embeds,
        adarms_conds,
        position_embeddings,
        attention_mask,
        position_ids,
        past_key_values,
        cache_position,
        action_mask,
        two_expert_layout,
        **kwargs,
    ):
        residual = hidden_states
        action_residual = action_embeds

        hidden_states, action_embeds, gate = self._apply_input_layernorm(
            hidden_states, action_embeds, adarms_conds
        )

        # Token Mixer
        if self.layer_type == "linear_attention":
            hidden_states, action_embeds = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
                # for vla
                action_embeds=action_embeds,
                action_mask=action_mask,
                two_expert_layout=two_expert_layout,
                pack_cu_seqlens=kwargs.get("pack_cu_seqlens"),
                # **kwargs,
            )
        elif self.layer_type == "full_attention":
            # Self Attention
            hidden_states, action_embeds, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                # for vla
                action_embeds=action_embeds,
                action_mask=action_mask,
                two_expert_layout=two_expert_layout,
                **kwargs,
            )

        hidden_states = residual + hidden_states
        action_embeds = self._apply_gated_residual(action_residual, action_embeds, gate)
        return hidden_states, action_embeds

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        # for vla
        action_embeds: torch.FloatTensor | None = None,
        action_mask: torch.Tensor | None = None,
        adarms_conds: torch.FloatTensor | None = None,
        two_expert_layout: TwoExpertLayout | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor:
        # Attention section (input LN -> token mixer -> residual + gated
        # residual) optionally recomputes on its own while the MLP keeps its
        # activations: the ckpt cost/benefit table (MEM_CKPT_TUNING.md) shows
        # the GDN attention half is a poor memory-for-time trade (~3 ms/GB)
        # while its MLP half is good (~5.5) -- this mode buys the good half
        # without paying for the bad one. Selected per layer by the
        # distributed.recompute config (mode "attn_only", applied in
        # gradient_checkpointing_enable); requires the layer-level flag OFF
        # (set by the same code path).
        if (
            getattr(self, "_ckpt_attn_only", False)
            and self.training
            and torch.is_grad_enabled()
        ):
            hidden_states, action_embeds = cp.checkpoint(
                self._attn_section,
                hidden_states,
                action_embeds,
                adarms_conds,
                position_embeddings,
                attention_mask,
                position_ids,
                past_key_values,
                cache_position,
                action_mask,
                two_expert_layout,
                use_reentrant=False,
                **kwargs,
            )
        else:
            hidden_states, action_embeds = self._attn_section(
                hidden_states,
                action_embeds,
                adarms_conds,
                position_embeddings,
                attention_mask,
                position_ids,
                past_key_values,
                cache_position,
                action_mask,
                two_expert_layout,
                **kwargs,
            )

        # Fully Connected
        residual = hidden_states
        action_residual = action_embeds
        hidden_states, action_embeds, gate = self._apply_post_attention_layernorm(
            hidden_states, action_embeds, adarms_conds
        )
        hidden_states, action_embeds = self._apply_mlp(hidden_states, action_embeds)
        hidden_states = residual + hidden_states
        action_embeds = self._apply_gated_residual(action_residual, action_embeds, gate)

        return hidden_states, action_embeds


class Qwen3_5_VLATextModel(Qwen3_5TextModel, ActionModelMixMin):
    def __init__(
        self, config: Qwen3_5TextConfig, use_selective_recompute: bool = False
    ):
        super(Qwen3_5TextModel, self).__init__(config)
        self.config = config
        self.use_selective_recompute = use_selective_recompute
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [
                Qwen3_5VLADecoderLayer(
                    config,
                    layer_idx,
                    use_selective_recompute=use_selective_recompute,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.vla_config.norm_moe:
            self.norm_action = Qwen3_5AdaptiveRMSNorm(
                config.vla_config.dim_inputs[1],
                eps=config.rms_norm_eps,
                cond_dim=config.vla_config.adarms_cond_dim,
            )

        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

    def _apply_norm(self, hidden_states, action_embeds=None, adarms_conds=None):
        if self.use_selective_recompute:
            normed_hidden_states = cp.checkpoint(
                self.norm, hidden_states, use_reentrant=False
            )
        else:
            normed_hidden_states = self.norm(hidden_states)
        if self.config.vla_config.norm_moe and action_embeds is not None:
            if self.use_selective_recompute:
                normed_action_embeds, _ = cp.checkpoint(
                    self.norm_action,
                    action_embeds,
                    adarms_conds,
                    use_reentrant=False,
                )
            else:
                normed_action_embeds, _ = self.norm_action(action_embeds, adarms_conds)
            return normed_hidden_states, normed_action_embeds
        return normed_hidden_states, None

    # @merge_with_config_defaults
    # @capture_outputs
    # @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        # for vla
        action_embeds: torch.FloatTensor | None = None,
        action_mask: torch.Tensor | None = None,
        adarms_conds: torch.FloatTensor | None = None,
        two_expert_layout: TwoExpertLayout | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        position_ids_were_none = position_ids is None
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = Qwen3_5DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
            cache_position_start = past_seen_tokens
        else:
            cache_position_start = int(cache_position[0].item())

        # mrope: the hard coded `4` is for text, temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(
                4, inputs_embeds.shape[0], -1
            )
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        positional_masks = kwargs.get("positional_masks")
        pack_seq_lens = (
            positional_masks.get("pack_seq_lens")
            if isinstance(positional_masks, dict)
            else None
        )
        if pack_seq_lens is not None:
            if inputs_embeds.shape[0] != 1:
                raise NotImplementedError(
                    "Qwen3.5 packing currently requires one packed row per batch"
                )
            if use_cache or past_key_values is not None:
                raise RuntimeError("Qwen3.5 packing is training-only and has no cache support")
            if position_ids_were_none or position_ids.ndim != 3:
                raise RuntimeError(
                    "Packed Qwen3.5 requires Wall-X-built 3-D mRoPE position_ids"
                )
            if (
                "full_attention" in self.config.layer_types
                and self.config._attn_implementation
                not in ATTENTION_TYPES_WITH_2D_MASK
                and self.config._attn_implementation
                not in ATTENTION_TYPES_WITH_FLASH_MASK
            ):
                raise RuntimeError(
                    "Packed Qwen3.5 requires a packing-safe 2-D or FlashMask backend"
                )
        if self.config._attn_implementation in ATTENTION_TYPES_WITH_2D_MASK:
            if (
                attention_mask is not None
                and inputs_embeds.shape[1] > 1
                and (action_mask is not None or pack_seq_lens is not None)
            ):
                causal_mask = self._update_joint_attention_mask_2d(
                    attention_mask=attention_mask,
                    moe_token_types=action_mask,
                    positional_masks=positional_masks,
                    pack_seq_lens=pack_seq_lens,
                )
            else:
                causal_mask = create_causal_mask(
                    config=self.config,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    position_ids=text_position_ids,
                )
        elif self.config._attn_implementation in ATTENTION_TYPES_WITH_FLASH_MASK:
            causal_mask = self._update_joint_attention_flash_mask(
                attention_mask=attention_mask,
                moe_token_types=action_mask,
                positional_masks=positional_masks,
                batch_size=inputs_embeds.shape[0],
                seq_len=inputs_embeds.shape[1],
                device=inputs_embeds.device,
                pack_seq_lens=pack_seq_lens,
            )
        else:
            causal_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=text_position_ids,
            )
        linear_attn_mask = self._update_linear_attn_mask(
            attention_mask, cache_position_start=cache_position_start
        )

        pack_cu_seqlens = None
        if pack_seq_lens is not None:
            if torch.is_tensor(pack_seq_lens):
                normalized_lens = pack_seq_lens.detach().cpu().tolist()
            else:
                normalized_lens = pack_seq_lens
            if normalized_lens and isinstance(normalized_lens[0], (int, float)):
                normalized_lens = [normalized_lens]
            cumulative = [0]
            for length in normalized_lens[0]:
                cumulative.append(cumulative[-1] + int(length))
            if cumulative[-1] > inputs_embeds.shape[1]:
                raise ValueError("pack_seq_lens exceeds the packed row length")
            if cumulative[-1] < inputs_embeds.shape[1]:
                cumulative.append(inputs_embeds.shape[1])
            pack_cu_seqlens = torch.tensor(
                cumulative, dtype=torch.int32, device=inputs_embeds.device
            )

        hidden_states = inputs_embeds

        if (
            self.config._attn_implementation != "flash_attention_2"
            and action_mask is not None
        ):
            position_ids = self._update_position_ids(
                position_ids, action_mask, positional_masks, pack_seq_lens
            )

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        if (
            two_expert_layout is None
            and action_embeds is not None
            and action_mask is not None
            and (
                self.config.vla_config.use_mot
                or self.config.vla_config.mlp_moe
                or self.config.vla_config.norm_moe
            )
        ):
            if pack_seq_lens is not None:
                two_expert_layout = prepare_two_expert_layout_packed(
                    action_mask.to(dtype=torch.int64), device=inputs_embeds.device
                )
            else:
                two_expert_layout = prepare_two_expert_layout(
                    action_mask.to(dtype=torch.int64),
                    batch_size=inputs_embeds.shape[0],
                    seq_length=inputs_embeds.shape[1],
                    device=inputs_embeds.device,
                )

        # flash_mask_ki: hoist the stop-gradient prefix boundary out of the
        # layer loop — every attention layer used to recompute the same
        # find_first_last_ones over action_mask. Computed once here and passed
        # down via layer kwargs; Qwen3_5_VLA_Attention.forward still recomputes
        # it when absent, so other call paths are unaffected.
        if (
            self.config._attn_implementation == "flash_mask_ki"
            and action_mask is not None
            and kwargs.get("prefix_lens") is None
        ):
            kwargs["prefix_lens"] = _flash_mask_ki_prefix_lens(
                action_mask,
                batch_size=inputs_embeds.shape[0],
                seq_len=inputs_embeds.shape[1],
            )

        for layer_idx, decoder_layer in enumerate(
            self.layers[: self.config.num_hidden_layers]
        ):
            kwargs["pack_cu_seqlens"] = pack_cu_seqlens
            layer_mask = (
                linear_attn_mask
                if decoder_layer.layer_type == "linear_attention"
                else causal_mask
            )

            hidden_states, action_embeds = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                # for vla
                action_embeds=action_embeds,
                action_mask=action_mask,
                adarms_conds=adarms_conds,
                two_expert_layout=two_expert_layout,
                **kwargs,
            )

        hidden_states, action_embeds = self._apply_norm(
            hidden_states, action_embeds, adarms_conds
        )

        return Qwen3_5VLAModelOutputWithPast(
            last_hidden_state=hidden_states,
            action_embeds=action_embeds,
            past_key_values=past_key_values,
        )


class Qwen3_5_VLAModel(Qwen3_5Model, ActionModelMixMin):
    def __init__(self, config, use_selective_recompute: bool = False):
        super(Qwen3_5Model, self).__init__(config)
        if config.vision_config.model_type == "qwen3_5_dino_vision":
            self.visual = Qwen3_5DinoVisionModel(config.vision_config)
        elif isinstance(config.vision_config, Qwen3_5VLAVisionConfig):
            self.visual = Qwen3_5VLAVisionModel._from_config(
                config.vision_config,
                use_selective_recompute=use_selective_recompute,
            )
        else:
            self.visual = Qwen3_5VisionModel._from_config(
                config.vision_config,
                use_selective_recompute=use_selective_recompute,
            )
        self.language_model = Qwen3_5_VLATextModel._from_config(
            config.text_config,
            use_selective_recompute=use_selective_recompute,
        )
        self.action_processor = ActionProcessor(config.vla_config)
        self.rope_deltas = None  # cache rope_deltas here

    @property
    def action_preprocessor(self):
        # ActionModelMixMin references ``action_preprocessor``; the inner
        # model stores it as ``action_processor``. Expose both names.
        return self.action_processor

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        # Only handle VLA-specific types not covered by the parent class
        if isinstance(module, Qwen3_5_VLA_GatedDeltaNet):
            init.ones_(module.dt_bias)
            init.copy_(
                module.A_log,
                # uniform_ can return exactly 0 (1/256 odds in bf16); log(0) = -inf.
                torch.empty_like(module.A_log, dtype=torch.float32)
                .uniform_(0, 16)
                .clamp_(min=1e-4)
                .log_(),
            )
        # We initialize with 0s to be 1 centered as the RMSNorm here does (1 + weight)
        elif isinstance(module, Qwen3_5AdaptiveRMSNorm):
            init.zeros_(module.weight)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        pixel_values_history: torch.Tensor | None = None,
        image_grid_thw_history: torch.LongTensor | None = None,
        history_image_mask: torch.Tensor | None = None,
        num_history_per_image: list[int] | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        pixel_values = pixel_values.type(self.visual.dtype)
        if pixel_values_history is not None:
            pixel_values_history = pixel_values_history.type(self.visual.dtype)

        token_counts = image_grid_thw.prod(-1)
        split_sizes = (token_counts // self.visual.spatial_merge_size**2).tolist()
        total_output_tokens = int(token_counts.sum().item())
        if image_grid_thw_history is not None:
            total_output_tokens += int(image_grid_thw_history.prod(-1).sum().item())

        if isinstance(self.visual, Qwen3_5VLAVisionModel):
            if (
                num_history_per_image is None
                and pixel_values_history is None
                and image_grid_thw_history is None
            ):
                num_history_per_image = [0] * int(image_grid_thw.shape[0])
            vision_output: BaseModelOutputWithPooling = self.visual(
                pixel_values,
                grid_thw=image_grid_thw,
                pixel_values_history=pixel_values_history,
                image_grid_thw_history=image_grid_thw_history,
                history_image_mask=history_image_mask,
                total_output_tokens=total_output_tokens,
                num_history_per_image=num_history_per_image,
                **kwargs,
            )
        else:
            vision_output: BaseModelOutputWithPooling = self.visual(
                pixel_values,
                grid_thw=image_grid_thw,
                total_output_tokens=total_output_tokens,
                **kwargs,
            )

        image_embeds = vision_output.pooler_output
        image_embeds = torch.split(image_embeds, split_sizes)
        vision_output.pooler_output = image_embeds
        return vision_output

    # @auto_docstring
    # @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        # for vla
        action_chunk: torch.FloatTensor | None = None,
        dof_mask: Optional[torch.FloatTensor] = None,
        proprioception: Optional[torch.FloatTensor] = None,
        agent_pos_mask: Optional[torch.FloatTensor] = None,
        dataset_names: Optional[str] = None,
        sample_time: Optional[torch.Tensor] = None,
        # for history observation
        pixel_values_history: torch.Tensor | None = None,
        image_grid_thw_history: torch.LongTensor | None = None,
        history_image_mask: torch.Tensor | None = None,
        num_history_per_image: list[int] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Qwen3_5VLAModelOutputWithPast:

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        positional_masks = kwargs.get("positional_masks")
        pack_seq_lens = (
            positional_masks.get("pack_seq_lens")
            if isinstance(positional_masks, dict)
            else None
        )
        packing_active = pack_seq_lens is not None
        if packing_active and not self.training:
            raise RuntimeError("Qwen3.5 sequence packing is training-only")
        if packing_active and inputs_embeds is not None:
            raise NotImplementedError(
                "Qwen3.5 packing requires input_ids; precomputed inputs_embeds "
                "cannot be extended safely for packed DP synchronization"
            )
        if packing_active and (past_key_values is not None or cache_position is not None):
            raise RuntimeError("Qwen3.5 sequence packing does not support cache")
        if packing_active and input_ids.shape[0] != 1:
            raise NotImplementedError("Qwen3.5 packing requires batch size 1")
        if packing_active and position_ids is not None and position_ids.ndim != 3:
            raise ValueError(
                "Packed Qwen3.5 rejects dataset-side 1-D position_ids; "
                "Wall-X must build packing-aware 3-D mRoPE ids"
            )

        packing_dp_dummy = bool(
            packing_active
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )
        if packing_dp_dummy and self.config.vla_config.action_linear_attention:
            raise NotImplementedError(
                "Packed mixed Action/VQA DP requires action_linear_attention=false"
            )

        # Every DP rank appends the same padding-only action span. This keeps
        # action expert module calls and gradient presence symmetric even when
        # a local packed bin contains only VQA samples. The span is removed
        # from model outputs below, and all of its masks/targets are zero, so it
        # has no numeric contribution. This branch is packing-only.
        original_seq_len = input_ids.shape[1]
        dummy_action_horizon = 0
        dummy_span_len = 0
        if packing_dp_dummy:
            # Adaptive packing expands every real action row to K*T.  Match the
            # local compact action width so concatenating the synchronization
            # row remains valid.  A VQA-only rank has no local K and only needs
            # one base block to execute the same action modules/collectives.
            dummy_action_horizon = (
                int(action_chunk.shape[1])
                if torch.is_tensor(action_chunk) and action_chunk.ndim >= 2
                else int(self.config.vla_config.action_horizon_flow)
            )
            dummy_span_len = dummy_action_horizon + 1
            dummy_ids = torch.full(
                (1, dummy_span_len),
                _resolve_pad_token_id(self.config),
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            dummy_ids[:, 0] = int(self.action_token_id_set["propri_token_id"])
            input_ids = torch.cat((input_ids, dummy_ids), dim=1)
            if attention_mask is None:
                attention_mask = torch.ones(
                    (1, original_seq_len), dtype=torch.long, device=input_ids.device
                )
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.zeros(
                        (1, dummy_span_len),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ),
                dim=1,
            )
            if position_ids is not None:
                position_ids = torch.cat(
                    (
                        position_ids,
                        position_ids.new_zeros(
                            (position_ids.shape[0], 1, dummy_span_len)
                        ),
                    ),
                    dim=2,
                )
            if mm_token_type_ids is not None:
                mm_token_type_ids = torch.cat(
                    (
                        mm_token_type_ids,
                        mm_token_type_ids.new_zeros((1, dummy_span_len)),
                    ),
                    dim=1,
                )
            extended_masks = {}
            for key, value in positional_masks.items():
                if key == "pack_seq_lens" or not torch.is_tensor(value):
                    extended_masks[key] = value
                    continue
                if value.ndim < 2 or value.shape[1] != original_seq_len:
                    extended_masks[key] = value
                    continue
                if key == "padding_positions":
                    fill = True
                elif key == "flow_block_id":
                    fill = -1
                else:
                    fill = False
                tail = torch.full(
                    (value.shape[0], dummy_span_len),
                    fill,
                    dtype=value.dtype,
                    device=value.device,
                )
                extended_masks[key] = torch.cat((value, tail), dim=1)
            positional_masks = extended_masks
            kwargs["positional_masks"] = positional_masks

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # Build the action-encoder embeds BEFORE the vision tower so ViT gets a
        # HIGHER autograd sequence number and runs its backward FIRST, overlapping
        # the large LLM reduce (action_processor.* otherwise wedges between LLM and
        # ViT in backward and forces ViT to wait on the ~25ms LLM reduce).
        noisy_action_embeds = None
        action_mask = None
        adarms_cond = None
        flow = None

        processor_action_chunk = action_chunk
        processor_sample_time = sample_time
        processor_dof_mask = dof_mask
        if packing_dp_dummy:
            action_dim = int(self.action_processor.action_dim)
            dummy_chunk = inputs_embeds.new_zeros(
                (1, dummy_action_horizon, action_dim), dtype=torch.float32
            )
            if processor_action_chunk is None:
                processor_action_chunk = dummy_chunk
                processor_sample_time = inputs_embeds.new_zeros((1,))
            else:
                processor_action_chunk = torch.cat(
                    (processor_action_chunk, dummy_chunk.to(processor_action_chunk)),
                    dim=0,
                )
                if processor_sample_time.ndim == 1:
                    processor_sample_time = torch.cat(
                        (processor_sample_time, processor_sample_time.new_zeros((1,)))
                    )
                else:
                    processor_sample_time = torch.cat(
                        (
                            processor_sample_time,
                            processor_sample_time.new_zeros(
                                (1, dummy_action_horizon)
                            ),
                        ),
                        dim=0,
                    )
            if self.config.vla_config.proj_with_mask:
                dummy_dof = dummy_chunk
                if processor_dof_mask is None:
                    processor_dof_mask = dummy_dof
                else:
                    processor_dof_mask = torch.cat(
                        (processor_dof_mask, dummy_dof.to(processor_dof_mask)), dim=0
                    )

        if (
            processor_action_chunk is not None
            and self.config.vla_config.use_flow_action_expert
        ):
            noisy_action_embeds, flow, adarms_cond = self.action_processor(
                processor_action_chunk,
                dataset_names,
                processor_sample_time,
                processor_dof_mask,
            )
            noisy_action_embeds = noisy_action_embeds[
                :, :, : self.config.vla_config.dim_inputs[1]
            ].to(inputs_embeds.dtype)
            action_mask = input_ids == self.action_token_id_set["action_token_id"]
            if packing_dp_dummy:
                action_mask[:, -dummy_action_horizon:] = True

        # Build the MoT two-expert layout BEFORE the vision tower: its
        # construction reads token counts back to the host (cudaStreamSync).
        # Placed after the ViT it stalls the CPU until the whole tower
        # finishes; here it only waits on the tiny embedding/action-encoder
        # ops, and the ViT + decoder launches proceed unblocked.
        two_expert_layout = None
        _vla_cfg = getattr(self.config, "vla_config", None) or getattr(
            getattr(self.config, "text_config", None), "vla_config", None
        )
        if (
            noisy_action_embeds is not None
            and action_mask is not None
            and _vla_cfg is not None
            and (_vla_cfg.use_mot or _vla_cfg.mlp_moe or _vla_cfg.norm_moe)
        ):
            if packing_active:
                two_expert_layout = prepare_two_expert_layout_packed(
                    action_mask.to(dtype=torch.int64), device=inputs_embeds.device
                )
            else:
                two_expert_layout = prepare_two_expert_layout(
                    action_mask.to(dtype=torch.int64),
                    batch_size=inputs_embeds.shape[0],
                    seq_length=inputs_embeds.shape[1],
                    device=inputs_embeds.device,
                )

        # Always run the vision tower — real images or FSDP-sync dummy — so
        # every DP rank executes the same collective sequence.
        _has_real_images = pixel_values is not None
        # Precompute flat image-placeholder positions BEFORE the ViT so the single
        # nonzero D2H (count readback) overlaps the tower forward; index_copy at the
        # merge below then avoids masked_scatter backward sync (masked_select ->
        # dynamic-size D2H, a 40-70ms exposed bubble). index_copy backward is
        # index_select(grad, idx) with static size -> no sync.
        _image_flat_idx = None
        if _has_real_images:
            _image_flat_idx = (
                input_ids.reshape(-1) == self.config.image_token_id
            ).nonzero(as_tuple=True)[0]
            _vis_pixels, _vis_grid = pixel_values, image_grid_thw
        else:
            vc = self.config.vision_config
            m = vc.spatial_merge_size
            _vis_pixels = torch.zeros(
                m * m,
                vc.in_channels * vc.temporal_patch_size * vc.patch_size**2,
                device=inputs_embeds.device,
                dtype=self.visual.dtype,
            )
            _vis_grid = torch.tensor(
                [[1, m, m]], device=inputs_embeds.device, dtype=torch.long
            )
        image_outputs: BaseModelOutputWithPooling = self.get_image_features(
            _vis_pixels,
            _vis_grid,
            pixel_values_history=pixel_values_history,
            image_grid_thw_history=image_grid_thw_history,
            history_image_mask=history_image_mask,
            num_history_per_image=num_history_per_image,
        )
        image_embeds = image_outputs.pooler_output
        image_embeds = torch.cat(image_embeds, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        if _has_real_images:
            _H = inputs_embeds.shape[-1]
            inputs_embeds = (
                inputs_embeds.reshape(-1, _H)
                .index_copy(0, _image_flat_idx, image_embeds)
                .view_as(inputs_embeds)
            )
        else:
            # No image tokens to scatter into; keep the tower's output in the graph
            # at zero weight so its backward reduce-scatter fires here too (same
            # position as the real masked_scatter above).
            inputs_embeds = inputs_embeds + 0.0 * image_embeds.sum()

        # NOTE: video shares this same FSDP-wrapped tower (get_video_features ->
        # self.visual), so a joint video + text-only batch has the same
        # collective-order desync risk as images above: a rank with no videos
        # calls self.visual one fewer time than a rank with videos -> NCCL
        # _ALLGATHER_BASE hang. NOT guarded here because no current training mixes
        # real video frames (pixel_values_videos is always None). When video +
        # text-only training is introduced, mirror the always-run + zero-weight
        # pattern used for images above (and smoke-test it).
        if pixel_values_videos is not None:
            video_outputs: BaseModelOutputWithPooling = self.get_video_features(
                pixel_values_videos, video_grid_thw, return_dict=True
            )
            video_embeds = video_outputs.pooler_output
            video_embeds = torch.cat(video_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if packing_dp_dummy and not self.config.vla_config.use_state_string_representation:
            proprio_dim = int(self.action_processor.propri_dim)
            dummy_proprio = inputs_embeds.new_zeros(
                (1, 1, proprio_dim), dtype=torch.float32
            )
            if proprioception is None:
                proprioception = dummy_proprio
                agent_pos_mask = dummy_proprio.clone()
            else:
                proprioception = torch.cat(
                    (proprioception, dummy_proprio.to(proprioception)), dim=0
                )
                if agent_pos_mask is None:
                    agent_pos_mask = torch.ones_like(proprioception)
                    agent_pos_mask[-1].zero_()
                else:
                    agent_pos_mask = torch.cat(
                        (agent_pos_mask, dummy_proprio.to(agent_pos_mask)), dim=0
                    )

        inputs_embeds = self.scatter_proprioception_embeddings(
            input_ids, inputs_embeds, proprioception, dataset_names, agent_pos_mask
        )

        if position_ids is None:
            if packing_active:
                position_ids = packed_position_ids(
                    input_ids,
                    attention_mask,
                    pack_seq_lens,
                    image_token_id=self.config.image_token_id,
                    video_token_id=self.config.video_token_id,
                    spatial_merge_size=self.config.vision_config.spatial_merge_size,
                    mm_token_type_ids=mm_token_type_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                )
            else:
                position_ids = self.compute_3d_position_ids(
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    mm_token_type_ids=mm_token_type_ids,
                )
        position_ids = self.apply_rvq_training_positions(input_ids, position_ids)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            # for vla
            action_embeds=noisy_action_embeds,
            action_mask=action_mask,
            adarms_conds=adarms_cond,
            two_expert_layout=two_expert_layout,
            **kwargs,
        )

        if packing_dp_dummy:
            outputs.last_hidden_state = outputs.last_hidden_state[:, :original_seq_len]

        return Qwen3_5VLAModelOutputWithPast(
            **outputs,
            rope_deltas=self.rope_deltas,
            flow=flow,
        )


class Qwen3_5_VLAForConditionalGeneration(
    Qwen3_5ForConditionalGeneration, ActionGenerationMixin, Qwen3_5VLAInferenceMixin
):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*"]
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: Qwen3_5Config

    def __init__(
        self, config, processor, tokenizer_mixin, use_selective_recompute=False
    ):
        super(Qwen3_5ForConditionalGeneration, self).__init__(config)
        self.model = Qwen3_5_VLAModel(
            config, use_selective_recompute=use_selective_recompute
        )
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )

        self.processor = processor
        self.tokenizer_mixin = tokenizer_mixin
        if self.tokenizer_mixin is not None:
            self.action_tokenizer_type = self.tokenizer_mixin.tokenizer_type
            self.action_tokenizer = self.tokenizer_mixin.tokenizer
            self.action_mapper = self.tokenizer_mixin.build_action_mapper(processor)
            build_position_mapper = getattr(
                self.tokenizer_mixin, "build_position_mapper", None
            )
            if callable(build_position_mapper):
                self.model._rvq_position_mapper = build_position_mapper(processor)
                self.config.rvq_token_protocol = "wall_vlm_start_v1"
                self.config.rvq_position_encoding = "time_level_2d"
                self.config.rvq_tokenizer_geometry = self.action_mapper[
                    "rvq_tokenizer_geometry"
                ]
                self.config.rvq_token_id_manifest_hash = self.action_mapper[
                    "rvq_token_id_manifest_hash"
                ]
        else:
            self.action_tokenizer_type = None
            self.action_tokenizer = None
            self.action_mapper = None

        self.define_action_token_id()
        self.model.action_token_id_set = self.action_token_id_set

        self.post_init()
        if RefinerConfig.from_vla_config(config.vla_config).enabled:
            # Built with the model so that checkpoint twin weights load into
            # them like any other parameter.
            install_twins(self.model)

    def set_normalizer(self, normalizer_action, normalizer_propri):
        self.action_preprocessor.set_normalizer(normalizer_action, normalizer_propri)

    @property
    def action_preprocessor(self):
        return self.model.action_processor

    @property
    def loss_fct(self):
        return CrossEntropyLoss(
            reduction="none"
        )  # do not do reduction to compute channel loss

    # @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        # for vla
        action_chunk: torch.FloatTensor | None = None,
        dof_mask: Optional[torch.FloatTensor] = None,
        proprioception: Optional[torch.FloatTensor] = None,
        agent_pos_mask: Optional[torch.FloatTensor] = None,
        dataset_names: Optional[str] = None,
        sample_time: Optional[torch.Tensor] = None,
        flow_loss_mask: Optional[torch.FloatTensor] = None,
        ar_action_loss_weight: Optional[float] = None,
        # for history observation
        pixel_values_history: torch.Tensor | None = None,
        image_grid_thw_history: torch.LongTensor | None = None,
        history_image_mask: torch.Tensor | None = None,
        num_history_per_image: list[int] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Qwen3_5VLAModelOutputWithPast:

        positional_masks = kwargs.get("positional_masks")
        packing_active = bool(
            isinstance(positional_masks, dict)
            and positional_masks.get("pack_seq_lens") is not None
        )
        packing_dp_dummy = bool(
            packing_active
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            cache_position=cache_position,
            # for vla
            action_chunk=action_chunk,
            dof_mask=dof_mask,
            proprioception=proprioception,
            agent_pos_mask=agent_pos_mask,
            dataset_names=dataset_names,
            sample_time=sample_time,
            # for history observation
            pixel_values_history=pixel_values_history,
            image_grid_thw_history=image_grid_thw_history,
            history_image_mask=history_image_mask,
            num_history_per_image=num_history_per_image,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        flow = outputs.flow

        loss_action_chunk = action_chunk
        loss_dof_mask = dof_mask
        loss_flow_loss_mask = flow_loss_mask
        if packing_dp_dummy:
            horizon = (
                int(action_chunk.shape[1])
                if torch.is_tensor(action_chunk) and action_chunk.ndim >= 2
                else int(self.config.vla_config.action_horizon_flow)
            )
            action_dim = int(self.action_preprocessor.action_dim)
            dummy_chunk = hidden_states.new_zeros(
                (1, horizon, action_dim), dtype=torch.float32
            )
            if loss_action_chunk is None:
                loss_action_chunk = dummy_chunk
            else:
                loss_action_chunk = torch.cat(
                    (loss_action_chunk, dummy_chunk.to(loss_action_chunk)), dim=0
                )
            dummy_mask = dummy_chunk
            if loss_dof_mask is None:
                loss_dof_mask = dummy_mask
            else:
                loss_dof_mask = torch.cat(
                    (loss_dof_mask, dummy_mask.to(loss_dof_mask)), dim=0
                )
            dummy_flow_mask = hidden_states.new_zeros((1, horizon))
            if loss_flow_loss_mask is None:
                real_rows = loss_action_chunk.shape[0] - 1
                loss_flow_loss_mask = hidden_states.new_ones((real_rows, horizon))
                loss_flow_loss_mask = torch.cat(
                    (loss_flow_loss_mask, dummy_flow_mask), dim=0
                )
            else:
                loss_flow_loss_mask = torch.cat(
                    (
                        loss_flow_loss_mask,
                        dummy_flow_mask.to(loss_flow_loss_mask),
                    ),
                    dim=0,
                )

        # --- optimization: loss-token-only LM Head ---
        # Whenever a CE loss is computed (train OR val), only run lm_head on the
        # tokens that contribute to it, shrinking [B*S, V] -> [N_loss, V]. Gating
        # this on `self.training` left validation on the full-logits path, which
        # materializes [B, S, V] and overflows 32-bit CUDA indexing at large vocab
        # (e.g. 16*768*252177 = 3.1e9 > 2^31 -> "CUDA error: invalid argument").
        _lm_loss_mask = None
        compute_ar_loss = self.should_compute_ar_loss(labels)
        if labels is not None:
            if compute_ar_loss:
                shift_labels = labels[..., 1:].contiguous()
                _lm_loss_mask = shift_labels != -100  # [B, S-1]
                if _lm_loss_mask.any():
                    lm_hidden = hidden_states[:, :-1, :][_lm_loss_mask]  # [N_loss, H]
                    logits = self.lm_head(lm_hidden)  # [N_loss, V]
                else:
                    logits = None
            else:
                # Pure flow training: do not materialize the large-vocab LM
                # logits when the configured AR contribution is exactly zero.
                logits = None
        else:
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            slice_indices = (
                slice(-logits_to_keep, None)
                if isinstance(logits_to_keep, int)
                else logits_to_keep
            )
            logits = self.lm_head(hidden_states[:, slice_indices, :])

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
            action_chunk=loss_action_chunk,
            dof_mask=loss_dof_mask,
            flow=flow,
            flow_loss_mask=loss_flow_loss_mask,
            _lm_loss_mask=_lm_loss_mask,
            action_hidden_states=outputs.action_embeds,
            ar_action_loss_weight=ar_action_loss_weight,
            packing_active=packing_active,
            packing_dummy_active=packing_dp_dummy,
        )

        # When there is no cross_entropy_loss (e.g. pure action batches without text labels),
        # hidden_states would not participate in the loss computation, causing parameters in
        # the last layers (e.g. layers.23.*, norm.*) to receive no gradient.
        # Add a dummy loss (0-weighted) to keep those parameters in the computation graph.
        if cross_entropy_loss is None:
            loss = loss + 0.0 * hidden_states.sum()
        if self.training and logits is None:
            loss = loss + 0.0 * self.lm_head(hidden_states[:, :1, :]).sum()

        return Qwen3_5VLAModelOutputWithPast(
            loss=loss,
            cross_entropy_loss=(
                cross_entropy_loss.clone() if cross_entropy_loss is not None else None
            ),
            ar_action_loss_weight=(
                torch.tensor(ar_action_loss_weight, device=hidden_states.device)
                if ar_action_loss_weight is not None
                else None
            ),
            flow_loss=flow_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
            channel_loss_dict=channel_loss_dict,
            channel_loss_count_dict=channel_loss_count_dict,
        )

    def rename_vlm_weights_for_vla(self, weights):
        return weights

    @staticmethod
    def is_fused(state_dict):
        return True

    def gradient_checkpointing_enable(self, recompute=None):
        """Enable gradient checkpointing (a.k.a. activation recomputation)
        for all decoder layers and vision blocks. Trades ~30-40% extra
        compute for significant memory savings. Method name follows the
        HuggingFace convention so ``DistributionStrategy`` can dispatch
        uniformly across model families.

        ``recompute`` (Qwen35RecomputeConfig | None) refines the plan per
        decoder layer: mode "full" recomputes the whole layer (default),
        "attn_only" recomputes just the attention section (MLP activations
        stay resident), "none" keeps the whole layer's activations. None or
        an empty plan means full recompute everywhere.
        """
        # Text (language) model. Setting the MODEL attribute alone is a
        # silent no-op: GradientCheckpointingLayer gates on the
        # ``gradient_checkpointing`` flag of EACH LAYER (plus a per-layer
        # ``_gradient_checkpointing_func``), and the text forward loop has no
        # checkpoint dispatch of its own -- so decoder recompute never
        # actually engaged (verified 2026-08-05: layer flags stayed False and
        # decoder fwd kernel counts were identical with "recompute" on/off;
        # only the vision tower, which already used the official API below,
        # was ever recomputed). Route through the official HF API, which
        # walks submodules and arms every GradientCheckpointingLayer.
        self.model.language_model.gradient_checkpointing_enable()
        # Vision encoder — covers all Qwen3_5VisionBlock instances
        if hasattr(self.model.visual, "gradient_checkpointing_enable"):
            self.model.visual.gradient_checkpointing_enable()
        else:
            self.model.visual.gradient_checkpointing = True
        print(
            "[GRAD CKPT] Gradient checkpointing enabled for language_model and visual encoder."
        )
        if recompute is None:
            return
        if isinstance(recompute, dict):
            # dataclasses.asdict flattened the plan on its way through
            # build_strategy; rebuild the typed config (revalidates too).
            from wall_x.config.infra_config import Qwen35RecomputeConfig

            recompute = Qwen35RecomputeConfig.build(
                recompute, where="distributed.recompute"
            )
        if not recompute.enable:
            return
        decoder = getattr(recompute, "decoder", None)
        if decoder is None:
            return
        layers = self.model.language_model.layers
        num_layers = len(layers)
        assigned = {}
        for override in decoder.overrides:
            if override.layers == "gdn":
                idxs = [
                    i
                    for i, layer in enumerate(layers)
                    if getattr(layer, "layer_type", "") == "linear_attention"
                ]
            else:
                idxs = list(override.layers)
            for i in idxs:
                if not 0 <= i < num_layers:
                    raise ValueError(
                        f"recompute override layer {i} out of range "
                        f"[0, {num_layers})"
                    )
                if i in assigned:
                    raise ValueError(
                        f"decoder layer {i} matched by more than one "
                        "recompute override"
                    )
                assigned[i] = override.mode

        def _apply(index, mode):
            # "full" is the armed state set above; the other modes disarm the
            # whole-layer flag, and attn_only re-arms just the attention
            # section (see Qwen3_5DecoderLayer.forward).
            if mode == "full":
                return
            layers[index].gradient_checkpointing = False
            if mode == "attn_only":
                layers[index]._ckpt_attn_only = True

        if decoder.default != "full":
            for i in range(num_layers):
                if i not in assigned:
                    _apply(i, decoder.default)
        for i, mode in assigned.items():
            _apply(i, mode)
        by_mode = {}
        for i, mode in sorted(assigned.items()):
            by_mode.setdefault(mode, []).append(i)
        print(
            f"[GRAD CKPT] decoder plan: default={decoder.default}, "
            f"overrides={by_mode or 'none'}"
        )

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
                    "pos_embed",
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

    def convert_to_pure_dmuon_mix_precision(self):
        """Prepare live parameter dtypes before pure DMuon wrapping.

        Pure DMuon keeps fp32 master/storage for trainable dedicated params after
        ``convert_to_dmuon`` runs. This pre-wrap pass only defines live forward
        dtypes for ordinary module parameters that are not covered by FSDP2
        MixedPrecision hooks.
        """

        params_to_keep_float32 = []
        for name, _ in self.named_parameters():
            if any(k in name for k in ["action_preprocessor", "action_processor"]):
                params_to_keep_float32.append(name)
        for name, param in self.named_parameters():
            if name in params_to_keep_float32:
                param.data = param.data.to(torch.float32)
            else:
                param.data = param.data.to(torch.bfloat16)

    def convert_to_fsdp(
        self,
        *,
        mesh,
        mp_policy,
        offload_policy=None,
        reshard_after_forward: bool = True,
        use_dmuon: bool = False,
        dmuon_owner_kwargs=None,
    ):
        """Wrap with FSDP2: per-decoder-layer + single vision-tower fully_shard
        with bf16 ``mp_policy``; layernorms and ``action_processor`` leaves get
        nested fully_shard with fp32 mp_policy to preserve fp32 forward
        (matching the prior FSDP1 NO_SHARD + fp32_policy layout).

        Inner fp32 wraps must precede outer wraps so the nested policy
        survives. ``mesh`` and ``mp_policy`` are required (produced by
        ``FSDPStrategy._build_fsdp2_layout``). Pre-migration FSDP1 hardcoded
        ``SHARD_GRAD_OP``; the new path honors
        ``distributed.fsdp.sharding_strategy``, so active runs wanting that
        throughput should set it to ``shard_grad_op`` explicitly.
        """
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

        if use_dmuon:
            import dmuon
            from wall_x.trainer.optimizer.dmuon.utils import dmuon_dedicate_mesh_args

            # DMuon keeps fp32 master weights for dedicated params; symmetric
            # FSDP2 params still use MixedPrecisionPolicy for bf16 compute.
            for param in self.parameters():
                if param.requires_grad and param.dtype != torch.float32:
                    param.data = param.data.float()

            action_processor_linear_ids = {
                id(module)
                for module in self.model.action_processor.modules()
                if isinstance(module, nn.Linear)
            }

            def hook_boundary(module: nn.Module) -> bool:
                # ActionProcessor exposes helper methods that call internal
                # Linear layers directly, bypassing ActionProcessor.forward().
                # Those internal Linears need their own hooks, while language
                # and vision params stay grouped at layer/block granularity.
                if id(module) in action_processor_linear_ids:
                    return True
                return isinstance(
                    module,
                    (
                        Qwen3_5VLADecoderLayer,
                        Qwen3_5VisionModel,
                        Qwen3_5VLAVisionModel,
                        Qwen3_5DinoVisionModel,
                        Qwen3_5VisionPatchEmbed,
                        Qwen3_5VisionBlock,
                        Qwen3_5VLAVisionBlock,
                        Qwen3_5VisionPatchMerger,
                        ActionProcessor,
                    ),
                )

            dmuon_mesh, dmuon_mesh_kwargs = dmuon_dedicate_mesh_args(mesh)
            assignment = dmuon.dedicate_params(
                self,
                dmuon_mesh,
                predicate=_is_qwen35_dmuon_target_param,
                hook_boundary_predicate=hook_boundary,
                hook_boundary_strict=True,
                compute_dtype=torch.bfloat16,
                reshard_after_forward=reshard_after_forward,
                **dmuon_mesh_kwargs,
                **(dmuon_owner_kwargs or {}),
            )
            print(
                "[Qwen3.5 FSDP2 + DMuon] dedicate_params: "
                f"{len(assignment)} params assigned"
            )

        fp32_mp = MixedPrecisionPolicy(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            cast_forward_inputs=False,
        )
        shard_kwargs = dict(
            mesh=mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )
        if offload_policy is not None:
            shard_kwargs["offload_policy"] = offload_policy
        fp32_shard_kwargs = dict(shard_kwargs, mp_policy=fp32_mp)

        # Layernorm leaves inside decoder layers / vision blocks /
        # text model.norm — wrap with fp32 BEFORE wrapping their parents.
        # AdaRMSNorm stays in its decoder layer's bf16 group: its cond is cast to
        # the activation dtype, matching the pure-DMuon compute layout.
        for module_name, module in list(self.named_modules()):
            for child_name, child in list(module.named_children()):
                if isinstance(child, Qwen3_5AdaptiveRMSNorm):
                    continue
                if any(
                    k in child_name.lower()
                    for k in ("input_layernorm", "post_attention_layernorm")
                ) or ("norm" in child_name.lower() and module_name.endswith("model")):
                    print(
                        f"[FSDP2] fully_shard fp32 layernorm: {module_name}.{child_name}"
                    )
                    fully_shard(child, **fp32_shard_kwargs)

        # action_processor leaves: same fp32 treatment as layernorms.
        for name, module in list(self.named_modules()):
            if not any(
                k in name.lower() for k in ("action_preprocessor", "action_processor")
            ):
                continue
            if any(True for _ in module.children()):
                continue  # only leaves
            print(f"[FSDP2] fully_shard fp32 action_processor leaf: {name}")
            fully_shard(module, **fp32_shard_kwargs)

        # Each text decoder layer: bf16 mp_policy from strategy.
        if hasattr(self, "model") and hasattr(self.model, "language_model"):
            for idx, layer in enumerate(self.model.language_model.layers):
                fully_shard(layer, **shard_kwargs)
                if idx == 0:
                    print(
                        f"[FSDP2] fully_shard model.language_model.layers.* "
                        f"(bf16, reshard_after_forward={reshard_after_forward})"
                    )

        # The vision tower is sharded as one unit to avoid per-block all-gather
        # launch overhead and rank drift observed in the old per-block layout.
        if hasattr(self, "model") and hasattr(self.model, "visual"):
            fully_shard(self.model.visual, **shard_kwargs)
            print("[FSDP2] fully_shard model.visual (bf16)")

        tie_word_embeddings = getattr(self.config, "tie_word_embeddings", False)
        if not tie_word_embeddings:
            embed_tokens = getattr(self.model.language_model, "embed_tokens", None)
            if embed_tokens is not None:
                fully_shard(embed_tokens, **shard_kwargs)
                print("[FSDP2] fully_shard model.language_model.embed_tokens (bf16)")
            if getattr(self, "lm_head", None) is not None:
                fully_shard(self.lm_head, **shard_kwargs)
                print("[FSDP2] fully_shard lm_head (bf16)")

        # Root: covers any params not in a nested fully_shard.
        print("[FSDP2] fully_shard root (bf16)")
        fully_shard(self, **shard_kwargs)
        return self

    def convert_to_dmuon(
        self, *, mesh, reshard_after_forward: bool = False, dmuon_owner_kwargs=None
    ):
        """Install pure DMuon ownership without calling FSDP2 ``fully_shard``."""
        import dmuon
        from wall_x.trainer.optimizer.dmuon.policy import (
            make_wallx_action_fp32_dmuon_param_policy_fn,
            summarize_dmuon_param_policy,
        )
        from wall_x.trainer.optimizer.dmuon.utils import dmuon_dedicate_mesh_args

        trainable_count = sum(1 for _, p in self.named_parameters() if p.requires_grad)
        route_counts, route_examples = _validate_qwen35_pure_dmuon_route_plan(
            self.named_parameters()
        )
        print(
            "[Qwen3.5 Pure DMuon] route plan: "
            f"counts={route_counts} "
            f"examples={_format_qwen35_route_examples(route_examples)}"
        )

        # DMuon keeps fp32 master weights for dedicated params.
        for param in self.parameters():
            if param.requires_grad and param.dtype != torch.float32:
                param.data = param.data.float()

        action_processor_linear_ids = {
            id(module)
            for module in self.model.action_processor.modules()
            if isinstance(module, nn.Linear)
        }
        dmuon_sharded_adamw_module_ids = {
            id(module)
            for module in (
                getattr(self, "lm_head", None),
                getattr(
                    getattr(self.model, "language_model", None),
                    "embed_tokens",
                    None,
                ),
            )
            if isinstance(module, (nn.Embedding, nn.Linear))
        }

        def hook_boundary(module: nn.Module) -> bool:
            if (
                id(module) in action_processor_linear_ids
                or id(module) in dmuon_sharded_adamw_module_ids
            ):
                return True
            return isinstance(
                module,
                (
                    Qwen3_5VLADecoderLayer,
                    # Wrap the whole vision tower as ONE DMuon comm unit instead of
                    # per-block: per-rank vision-work imbalance (different Nv) was
                    # desyncing ranks at each vision block's grad reduce-scatter.
                    # Qwen3_5VisionBlock / Qwen3_5VLAVisionBlock removed on purpose
                    # so the block stack folds into the VisionModel boundary below.
                    Qwen3_5VisionModel,
                    Qwen3_5VLAVisionModel,
                    Qwen3_5DinoVisionModel,
                    Qwen3_5VisionPatchEmbed,
                    Qwen3_5VisionPatchMerger,
                    ActionProcessor,
                ),
            )

        dmuon_mesh, dmuon_mesh_kwargs = dmuon_dedicate_mesh_args(mesh)
        param_policy_fn = make_wallx_action_fp32_dmuon_param_policy_fn(
            _qwen35_pure_dmuon_route_hint
        )
        policy_counts, policy_examples = summarize_dmuon_param_policy(
            self.named_parameters(), param_policy_fn
        )
        print(
            "[Qwen3.5 Pure DMuon] param policy: "
            f"counts={policy_counts} "
            f"examples={_format_qwen35_route_examples(policy_examples)}"
        )
        assignment = dmuon.dedicate_params(
            self,
            dmuon_mesh,
            predicate=_is_qwen35_dmuon_trainable_param,
            param_policy_fn=param_policy_fn,
            hook_boundary_predicate=hook_boundary,
            hook_boundary_strict=False,
            reshard_after_forward=reshard_after_forward,
            **dmuon_mesh_kwargs,
            **(dmuon_owner_kwargs or {}),
        )
        print(
            "[Qwen3.5 Pure DMuon] dedicate_params: "
            f"{len(assignment)}/{trainable_count} trainable params assigned "
            f"(reshard_after_forward={reshard_after_forward})"
        )
        if len(assignment) != trainable_count:
            print(
                "[Qwen3.5 Pure DMuon] WARNING: only "
                f"{len(assignment)}/{trainable_count} trainable params were assigned"
            )
        return self
