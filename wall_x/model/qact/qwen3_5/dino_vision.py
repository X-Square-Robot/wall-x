"""DINOv3 visual tower compatible with Qwen3.5's multimodal interface."""

from __future__ import annotations


import torch
import torch.nn as nn
from transformers import AutoModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.dinov3_vit import modeling_dinov3_vit as _dinov3

from wall_x.model.core.ops import rope, scaled_add_layernorm, silu_and_mul
from wall_x.model.core.ops.slim_flash import SLIM_FA3_AVAILABLE as _SLIM_FA3
from wall_x.model.core.ops.slim_flash import SlimFA3, SlimFA3Varlen

# Direct flash-attention library binding for the DINO tower. The transformers
# ALL_ATTENTION_FUNCTIONS "flash_attention_2" route goes through
# integrations/flash_attention.py + modeling_flash_attention_utils.py, whose
# per-call python plumbing (kwargs/dtype/mask handling and a [B,H,S,D]
# round-trip transpose) is pure CPU overhead in a launch-bound profile. We
# call the kernel library directly on the [B, S, H, D] layout that both our
# rope output and flash natively use. FA3 (flash_attn_interface, Hopper) is
# picked up automatically when installed; FA3 has no dropout support, so it
# is only used when attention dropout is off.
try:
    from flash_attn_interface import flash_attn_func as _flash_attn_func
    from flash_attn_interface import flash_attn_varlen_func as _flash_attn_varlen_func

    _FLASH_IS_V3 = True
except Exception as _fa3_exc:  # noqa: BLE001
    print(
        "[DINO] flash_attn_interface (FA3) unavailable, using FA2: "
        f"{type(_fa3_exc).__name__}: {_fa3_exc}"
    )
    try:
        from flash_attn import flash_attn_func as _flash_attn_func
        from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func

        _FLASH_IS_V3 = False
    except ImportError:
        _flash_attn_func = None
        _flash_attn_varlen_func = None
        _FLASH_IS_V3 = False

from .configuration_qwen3_5 import Qwen3_5DinoVisionConfig
from .modeling_qwen3_5 import _fuse_split_weights_for_load


def patchify_image(
    image: torch.Tensor, patch_size: int
) -> tuple[torch.Tensor, tuple[int, int]]:
    channels, height, width = image.shape
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"image size {(height, width)} must be divisible by patch_size={patch_size}"
        )
    grid_h, grid_w = height // patch_size, width // patch_size
    patches = image.reshape(channels, grid_h, patch_size, grid_w, patch_size).permute(
        1, 3, 0, 2, 4
    )
    return patches.contiguous().reshape(
        grid_h * grid_w, channels * patch_size * patch_size
    ), (grid_h, grid_w)


class PerPatchMLPProjector(nn.Module):
    def __init__(
        self,
        encoder_hidden_size: int,
        output_hidden_size: int,
        *,
        layer_norm_eps: float = 1e-6,
        hidden_act: str = "gelu",
    ):
        super().__init__()
        if hidden_act != "gelu":
            raise ValueError(f"unsupported DINO projector activation: {hidden_act!r}")
        self.norm = nn.LayerNorm(encoder_hidden_size, eps=layer_norm_eps)
        self.fc1 = nn.Linear(encoder_hidden_size, output_hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(output_hidden_size, output_hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(self.norm(hidden_states))))


class SpatialMergeMLPProjector(nn.Module):
    """Project each raster-ordered ``merge x merge`` patch group into one LM token.

    Qwen-VL-style 4x token downsampling for the DINO tower: LN on per-patch
    features, group each frame's (merge x merge) neighborhood, then a 2-layer
    MLP on the concatenated group. Grid H/W must be divisible by ``merge``
    (the qwen3_5 vision processor's factor-32 smart_resize guarantees even
    grids for patch16).

    The pretrained per-patch projector weights (fc1 [out, E]) cannot be
    reshaped into the merged fc1 [4E, 4E] -- ``_load_from_state_dict`` drops
    the incompatible keys LOUDLY so the tower fresh-initializes the projector
    instead of crashing torch's size-mismatch check (strict=False does not
    forgive shape mismatches).
    """

    def __init__(
        self,
        encoder_hidden_size: int,
        output_hidden_size: int,
        *,
        spatial_merge_size: int = 2,
        layer_norm_eps: float = 1e-6,
        hidden_act: str = "gelu",
    ):
        super().__init__()
        if hidden_act != "gelu":
            raise ValueError(f"unsupported DINO projector activation: {hidden_act!r}")
        self.spatial_merge_size = int(spatial_merge_size)
        self.encoder_hidden_size = int(encoder_hidden_size)
        merged_hidden = self.encoder_hidden_size * self.spatial_merge_size**2
        self.norm = nn.LayerNorm(encoder_hidden_size, eps=layer_norm_eps)
        self.fc1 = nn.Linear(merged_hidden, merged_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(merged_hidden, output_hidden_size)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        own = dict(self.named_parameters(recurse=True))
        for key in list(state_dict.keys()):
            if not key.startswith(prefix):
                continue
            local = key[len(prefix) :]
            if local in own and tuple(state_dict[key].shape) != tuple(own[local].shape):
                print(
                    f"[DINO merger] dropping pretrained {key} "
                    f"{tuple(state_dict[key].shape)} != {tuple(own[local].shape)} "
                    "(per-patch -> spatial-merge projector swap; fresh init)"
                )
                del state_dict[key]
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def merge_spatial_features(
        self, x: torch.Tensor, grids: list[tuple[int, int, int]]
    ) -> torch.Tensor:
        merge = self.spatial_merge_size
        blocks = []
        offset = 0
        for temporal, grid_h, grid_w in grids:
            if grid_h % merge or grid_w % merge:
                raise ValueError(
                    f"spatial merge requires grids divisible by {merge}, got "
                    f"{(temporal, grid_h, grid_w)}"
                )
            count = temporal * grid_h * grid_w
            block = x[offset : offset + count]
            offset += count
            block = block.reshape(
                temporal, grid_h // merge, merge, grid_w // merge, merge, -1
            )
            block = block.permute(0, 1, 3, 2, 4, 5).contiguous()
            blocks.append(block.reshape(-1, self.encoder_hidden_size * merge**2))
        if offset != x.shape[0]:
            raise ValueError(
                f"grids describe {offset} patches, projector got {x.shape[0]}"
            )
        return torch.cat(blocks, dim=0)

    def forward(
        self, x: torch.Tensor, grids: list[tuple[int, int, int]]
    ) -> torch.Tensor:
        merged = self.merge_spatial_features(self.norm(x), grids)
        return self.fc2(self.act(self.fc1(merged)))


class _DinoRopeHalfPadded(nn.Module):
    """Wraps DINOv3ViTRotaryEmbedding for the fused-rope attention path.

    Emits fp32 HALF-dim cos/sin (the external_accel rope convention: the stock
    DINOv3 tables are ``angles.tile(2)`` = [freqs, freqs], so the first half
    is the compact form) with ``num_prefix_tokens`` identity rows (cos=1,
    sin=0) prepended: the fused kernel then rotates every token, and
    cls/register tokens get an exact identity rotation. Numerically equal to
    the stock split->rotate->cat, but computed once per encoder call instead
    of re-sliced in every layer. No parameters/persistent buffers, so
    checkpoint key layout is unchanged.
    """

    def __init__(self, inner: nn.Module, num_prefix_tokens: int):
        super().__init__()
        self.inner = inner
        self.num_prefix_tokens = num_prefix_tokens

    def forward(self, pixel_values: torch.Tensor):
        cos, sin = self.inner(pixel_values)
        half = cos.shape[-1] // 2
        cos_half = cos[..., :half].float()
        sin_half = sin[..., :half].float()
        ones = cos_half.new_ones((self.num_prefix_tokens, half))
        zeros = sin_half.new_zeros((self.num_prefix_tokens, half))
        cos_half = torch.cat((ones, cos_half), dim=0)
        sin_half = torch.cat((zeros, sin_half), dim=0)
        batch = pixel_values.shape[0]
        cos_half = cos_half.unsqueeze(0).expand(batch, -1, -1).contiguous()
        sin_half = sin_half.unsqueeze(0).expand(batch, -1, -1).contiguous()
        return cos_half, sin_half


class DinoFusedRopeAttention(_dinov3.DINOv3ViTAttention):
    """DINOv3 attention with a fused qkv GEMM and the fused external_accel rope.

    Structural merges (math unchanged vs stock DINOv3ViTAttention):
      * q/k/v projections collapse into one ``qkv_proj`` GEMM emitting
        [q | k | v]; checkpoint loading fuses the stock split keys on the fly
        (same convention as Qwen3_5MLP.gate_up_proj).
      * DINOv3-H+ has query/value biases but no key bias. The fused GEMM adds
        a bias assembled per call from the trainable q/v segments and a
        constant zero k segment (non-persistent buffer), so the trainable
        parametrization is exactly the stock one.
      * ``rope.pack`` rotates q/k inside the fused qkv in one in-place kernel
        (the wall-x ViT consumption pattern; autograd handled in the op).
        cos/sin arrive half-dim prefix-padded from ``_DinoRopeHalfPadded`` --
        cls/register rows are identity rotations, replacing the stock
        split->rotate_half->cat chain.
    """

    def __init__(self, config):
        super().__init__(config)
        del self.q_proj, self.k_proj, self.v_proj
        self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)
        self.q_bias = (
            nn.Parameter(torch.zeros(self.embed_dim)) if config.query_bias else None
        )
        self.k_bias = (
            nn.Parameter(torch.zeros(self.embed_dim)) if config.key_bias else None
        )
        self.v_bias = (
            nn.Parameter(torch.zeros(self.embed_dim)) if config.value_bias else None
        )
        self._any_qkv_bias = bool(
            config.query_bias or config.key_bias or config.value_bias
        )
        if self._any_qkv_bias:
            self.register_buffer(
                "_qkv_zero_bias", torch.zeros(self.embed_dim), persistent=False
            )

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        _fuse_split_weights_for_load(
            state_dict,
            prefix,
            "qkv_proj.weight",
            ["q_proj.weight", "k_proj.weight", "v_proj.weight"],
        )
        for old_key, new_key in (
            ("q_proj.bias", "q_bias"),
            ("k_proj.bias", "k_bias"),
            ("v_proj.bias", "v_bias"),
        ):
            if prefix + old_key in state_dict and prefix + new_key not in state_dict:
                state_dict[prefix + new_key] = state_dict.pop(prefix + old_key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        **kwargs,
    ):
        batch_size, patches, _ = hidden_states.size()

        if self._any_qkv_bias:
            # dmuon installs bf16 compute copies for the dedicated bias params
            # but never touches buffers, so the zero k segment must follow the
            # LIVE dtype of its siblings or torch.cat type-promotes the fused
            # bias to fp32 and the addmm dtype-mismatches (seen on DLC).
            ref = next(
                b for b in (self.q_bias, self.k_bias, self.v_bias) if b is not None
            )
            zero = self._qkv_zero_bias
            if zero.dtype != ref.dtype:
                zero = zero.to(ref.dtype)
            qkv_bias = torch.cat(
                (
                    self.q_bias if self.q_bias is not None else zero,
                    self.k_bias if self.k_bias is not None else zero,
                    self.v_bias if self.v_bias is not None else zero,
                )
            )
            if qkv_bias.dtype != hidden_states.dtype:
                qkv_bias = qkv_bias.to(hidden_states.dtype)
        else:
            qkv_bias = None
        qkv = nn.functional.linear(hidden_states, self.qkv_proj.weight, qkv_bias)

        cos, sin = position_embeddings  # [B, S, head_dim/2] fp32, prefix-padded
        tokens = batch_size * patches
        qkv = rope.pack(
            qkv.view(tokens, 3 * self.embed_dim),
            self.num_heads,
            self.num_heads,
            cos.reshape(1, tokens, -1),
            sin.reshape(1, tokens, -1),
        )

        if cu_seqlens is not None:
            # Varlen multi-image path: hidden is [1, total_tokens, E] with
            # per-image attention boundaries in cu_seqlens -- flash varlen is
            # the ONLY implementation that expresses this, so failing its
            # preconditions must be loud, never a silent dense fallback that
            # would attend across image boundaries.
            dropout_p = 0.0 if not self.training else self.dropout
            if (
                _flash_attn_varlen_func is None
                or attention_mask is not None
                or (_FLASH_IS_V3 and dropout_p > 0.0)
            ):
                raise RuntimeError(
                    "DINO varlen attention requires flash-attn "
                    f"(varlen_func={_flash_attn_varlen_func is not None}, "
                    f"mask={attention_mask is not None}, dropout={dropout_p})"
                )
            query_states, key_states, value_states = qkv.view(
                tokens, 3, self.num_heads, self.head_dim
            ).unbind(1)
            if _FLASH_IS_V3 and _SLIM_FA3 and dropout_p == 0.0:
                attn_output = SlimFA3Varlen.apply(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens,
                    max_seqlen,
                    self.scaling,
                )
            elif _FLASH_IS_V3:
                attn_output = _flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens,
                    cu_seqlens,
                    max_seqlen,
                    max_seqlen,
                    softmax_scale=self.scaling,
                    causal=False,
                )
            else:
                attn_output = _flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens,
                    cu_seqlens,
                    max_seqlen,
                    max_seqlen,
                    dropout_p=dropout_p,
                    softmax_scale=self.scaling,
                    causal=False,
                )
            if isinstance(attn_output, tuple):  # FA3 may return (out, lse)
                attn_output = attn_output[0]
            attn_output = attn_output.reshape(batch_size, patches, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, None

        query_states, key_states, value_states = qkv.view(
            batch_size, patches, 3, self.num_heads, self.head_dim
        ).unbind(2)

        dropout_p = 0.0 if not self.training else self.dropout
        use_direct_flash = (
            _flash_attn_func is not None
            and self.config._attn_implementation.startswith("flash_attention")
            and attention_mask is None
            and query_states.dtype in (torch.bfloat16, torch.float16)
            and not (_FLASH_IS_V3 and dropout_p > 0.0)
        )
        if use_direct_flash:
            # q/k/v are already [B, S, H, D] -- flash's native layout; no
            # transposes, no transformers integration layer.
            if _FLASH_IS_V3 and _SLIM_FA3 and dropout_p == 0.0:
                # raw-op path: skips the flash_attn_interface python
                # Function + custom_op wrapper (~165us host per pair, see
                # WRAPPER_SLIMMING_GUIDE.md). Verified bitwise (fwd) on the
                # real fused-qkv strided views.
                attn_output = SlimFA3.apply(
                    query_states, key_states, value_states, self.scaling
                )
            elif _FLASH_IS_V3:
                attn_output = _flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    softmax_scale=self.scaling,
                    causal=False,
                )
            else:
                attn_output = _flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout_p=dropout_p,
                    softmax_scale=self.scaling,
                    causal=False,
                )
            if isinstance(attn_output, tuple):  # FA3 may return (out, lse)
                attn_output = attn_output[0]
            attn_weights = None
        else:
            attention_interface = _dinov3.ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, _dinov3.eager_attention_forward
            )
            attn_output, attn_weights = attention_interface(
                self,
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                attention_mask,
                dropout=dropout_p,
                scaling=self.scaling,
                **kwargs,
            )

        attn_output = attn_output.reshape(batch_size, patches, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class DinoFusedMLP(nn.Module):
    """DINOv3 gated MLP with the fused gate_up GEMM + silu_and_mul kernel.

    Mirrors Qwen3_5MLP: one GEMM emits [gate | up] and the fused kernel
    computes silu(gate)*up. Checkpoint loading fuses the stock
    gate_proj/up_proj keys -- weights AND biases (DINO has mlp_bias=True; the
    fused [2I] bias is the same trainable DOF, and adamw is elementwise, so
    training dynamics are unchanged). Only swapped in for silu/swish gated
    MLPs; other activations keep the stock module.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = nn.Linear(
            self.hidden_size, 2 * self.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=config.mlp_bias
        )

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        _fuse_split_weights_for_load(
            state_dict,
            prefix,
            "gate_up_proj.weight",
            ["gate_proj.weight", "up_proj.weight"],
        )
        _fuse_split_weights_for_load(
            state_dict,
            prefix,
            "gate_up_proj.bias",
            ["gate_proj.bias", "up_proj.bias"],
        )
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x):
        # inplace=True selects the split API; under autograd the op routes to
        # the out-of-place, backward-safe path (checkpoint recompute must not
        # mutate the saved gate_up) -- same call as Qwen3_5MLP.
        return self.down_proj(silu_and_mul(self.gate_up_proj(x), inplace=True))


class DinoFusedLayer(_dinov3.DINOv3ViTLayer):
    """DINOv3ViTLayer with layer_scale-mul + residual-add folded into one
    ``torch.addcmul`` kernel per sub-block.

    Stock chain is ``drop_path(hidden * lambda) + residual`` = two full
    [B, S, E] elementwise passes (LN itself is already one fused aten
    kernel; the unfused mul/add around it are what the LLM's add_rmsnorm
    absorbed on the decoder side). addcmul is the same math with one pass
    and fma rounding. Falls back to the stock composition whenever
    drop_path is not Identity -- the fused form has no slot for stochastic
    depth.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        fuse = isinstance(self.drop_path, nn.Identity)

        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states, _ = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        if fuse:
            # Intra-layer pair fused into ONE kernel: the attention branch's
            # layerscale-mul + residual-add AND the following norm2. (norm1
            # pairs with the PREVIOUS layer's add across the layer boundary
            # -- fusing that needs an encoder-loop interface change; left as
            # a follow-up.)
            residual, hidden_states = scaled_add_layernorm(
                residual,
                hidden_states,
                self.layer_scale1.lambda1,
                self.norm2.weight,
                self.norm2.bias,
                self.norm2.eps,
            )
        else:
            residual = self.drop_path(self.layer_scale1(hidden_states)) + residual
            hidden_states = self.norm2(residual)

        hidden_states = self.mlp(hidden_states)
        if fuse:
            hidden_states = torch.addcmul(
                residual, hidden_states, self.layer_scale2.lambda1
            )
        else:
            hidden_states = self.drop_path(self.layer_scale2(hidden_states)) + residual

        return hidden_states


class Qwen3_5DinoVisionModel(nn.Module):
    """A DINOv3 encoder and projector with Qwen-compatible flattened inputs."""

    config_class = Qwen3_5DinoVisionConfig

    def __init__(self, config: Qwen3_5DinoVisionConfig):
        super().__init__()
        self.config = config
        self.encoder = AutoModel.from_config(config.encoder_config)
        encoder_hidden = int(config.encoder_config.hidden_size)
        projector_config = config.projector_config
        if int(config.spatial_merge_size) > 1:
            self.merger = SpatialMergeMLPProjector(
                encoder_hidden,
                config.out_hidden_size,
                spatial_merge_size=int(config.spatial_merge_size),
                layer_norm_eps=float(projector_config.get("layer_norm_eps", 1e-6)),
                hidden_act=projector_config.get("hidden_act", "gelu"),
            )
        else:
            self.merger = PerPatchMLPProjector(
                encoder_hidden,
                config.out_hidden_size,
                layer_norm_eps=float(projector_config.get("layer_norm_eps", 1e-6)),
                hidden_act=projector_config.get("hidden_act", "gelu"),
            )
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.num_prefix_tokens = 1 + int(
            getattr(config.encoder_config, "num_register_tokens", 0)
        )

        # Fused-rope surgery: give every layer the external_accel rope.full attention
        # and make the rotary module emit the half-dim prefix-padded tables it
        # consumes. The original projection Linears are reused by reference
        # (identical parameter names/objects), so external weight loading and
        # the dmuon param routing see an unchanged module tree.
        enc_cfg = self.encoder.config
        fuse_mlp = enc_cfg.use_gated_mlp and enc_cfg.hidden_act in ("silu", "swish")
        for layer_idx, layer in enumerate(self.encoder.layer):
            stock_attn = layer.attention
            fused_attn = DinoFusedRopeAttention(enc_cfg)
            with torch.no_grad():
                fused_attn.qkv_proj.weight.copy_(
                    torch.cat(
                        (
                            stock_attn.q_proj.weight,
                            stock_attn.k_proj.weight,
                            stock_attn.v_proj.weight,
                        )
                    )
                )
                if fused_attn.q_bias is not None:
                    fused_attn.q_bias.copy_(stock_attn.q_proj.bias)
                if fused_attn.k_bias is not None:
                    fused_attn.k_bias.copy_(stock_attn.k_proj.bias)
                if fused_attn.v_bias is not None:
                    fused_attn.v_bias.copy_(stock_attn.v_proj.bias)
            fused_attn.o_proj = stock_attn.o_proj
            layer.attention = fused_attn
            if fuse_mlp:
                stock_mlp = layer.mlp
                fused_mlp = DinoFusedMLP(enc_cfg)
                with torch.no_grad():
                    fused_mlp.gate_up_proj.weight.copy_(
                        torch.cat(
                            (stock_mlp.gate_proj.weight, stock_mlp.up_proj.weight)
                        )
                    )
                    if fused_mlp.gate_up_proj.bias is not None:
                        fused_mlp.gate_up_proj.bias.copy_(
                            torch.cat(
                                (stock_mlp.gate_proj.bias, stock_mlp.up_proj.bias)
                            )
                        )
                fused_mlp.down_proj = stock_mlp.down_proj
                layer.mlp = fused_mlp
            # Rehost the (now fused) submodules on a DinoFusedLayer so the
            # layer-level mul+add chain fuses too. Submodules move by
            # reference -- attribute names, parameters, and checkpoint keys
            # are unchanged.
            fused_layer = DinoFusedLayer(enc_cfg)
            fused_layer.norm1 = layer.norm1
            fused_layer.attention = layer.attention
            fused_layer.layer_scale1 = layer.layer_scale1
            fused_layer.drop_path = layer.drop_path
            fused_layer.norm2 = layer.norm2
            fused_layer.mlp = layer.mlp
            fused_layer.layer_scale2 = layer.layer_scale2
            self.encoder.layer[layer_idx] = fused_layer
        self.encoder.rope_embeddings = _DinoRopeHalfPadded(
            self.encoder.rope_embeddings, self.num_prefix_tokens
        )

    @property
    def dtype(self) -> torch.dtype:
        return next(self.encoder.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    def gradient_checkpointing_enable(self, **kwargs):
        self.encoder.gradient_checkpointing_enable(**kwargs)

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor | None = None,
        pixel_values_history: torch.Tensor | None = None,
        image_grid_thw_history: torch.Tensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        if grid_thw is None:
            raise ValueError("DINO vision requires grid_thw")
        if pixel_values_history is not None or image_grid_thw_history is not None:
            raise NotImplementedError("DINO vision does not support MEM history inputs")

        grids = [(int(t), int(gh), int(gw)) for t, gh, gw in grid_thw.tolist()]
        total_patches = sum(t * gh * gw for t, gh, gw in grids)
        if total_patches != pixel_values.shape[0]:
            raise ValueError(
                f"grid_thw describes {total_patches} patches, got {pixel_values.shape[0]}"
            )

        # Flash-varlen single forward: ALL frames (mixed grids included) in
        # one varlen sequence -- one encoder pass per step instead of one per
        # distinct grid, and the conv patch-embed becomes a single Linear over
        # the already-flattened input rows. This is the only DINO forward
        # path; requirements are validated loudly instead of silently falling
        # back to a slower implementation.
        if total_patches == 0:
            encoded = None
        else:
            if _flash_attn_varlen_func is None:
                raise RuntimeError(
                    "DINO vision requires flash-attn varlen "
                    "(flash_attn_varlen_func unavailable)"
                )
            attn_impl = getattr(self.encoder.config, "_attn_implementation", "")
            if not attn_impl.startswith("flash_attention"):
                raise RuntimeError(
                    "DINO vision requires _attn_implementation="
                    f"'flash_attention_*', got {attn_impl!r}"
                )
            if self.dtype not in (torch.bfloat16, torch.float16):
                raise RuntimeError(f"DINO vision requires bf16/fp16, got {self.dtype}")
            dropout = float(getattr(self.encoder.layer[0].attention, "dropout", 0.0))
            if _FLASH_IS_V3 and self.training and dropout > 0.0:
                raise RuntimeError(
                    "FA3 varlen has no attention dropout; set the encoder "
                    "attention dropout to 0 or install FA2"
                )
            encoded = self._forward_varlen(pixel_values, grids)
        if encoded is not None and encoded.shape[0] > 0:
            if isinstance(self.merger, SpatialMergeMLPProjector):
                projected = self.merger(encoded, grids)
            else:
                projected = self.merger(encoded)
        else:
            projected = pixel_values.new_zeros((0, self.config.out_hidden_size))
        return BaseModelOutputWithPooling(
            last_hidden_state=projected,
            pooler_output=projected,
        )

    def _forward_varlen(
        self, pixel_values: torch.Tensor, grids: list[tuple[int, int, int]]
    ) -> torch.Tensor:
        """One flash-varlen encoder pass over every frame of the batch.

        The varlen sequence is frame-major in the original grid_thw order,
        each frame contributing ``[cls, register*R, patches...]`` -- so the
        stripped patch rows come out in per-frame input order.
        Equivalences relied on:
          * Conv2d(kernel=p, stride=p) == Linear over patchify rows (the
            [C, p, p] row flatten matches the conv weight layout);
          * rope tables depend only on (gh, gw), so they are built once per
            distinct grid -- frames sharing a grid share one train-time aug
            draw -- and concatenated per frame with identity prefix rows;
          * per-image full self-attention == varlen attention with cu_seqlens
            boundaries.
        """
        emb = self.encoder.embeddings
        conv = emb.patch_embeddings
        prefix_len = self.num_prefix_tokens

        frame_grids: list[tuple[int, int]] = []
        for temporal, grid_h, grid_w in grids:
            frame_grids.extend([(grid_h, grid_w)] * temporal)
        n_frames = len(frame_grids)

        # Host-side plan (a handful of small CPU tensor ops per call).
        seqlens = torch.tensor(
            [prefix_len + gh * gw for gh, gw in frame_grids], dtype=torch.int64
        )
        starts = torch.cat([seqlens.new_zeros(1), seqlens.cumsum(0)])
        total = int(starts[-1])
        n_prefix = n_frames * prefix_len
        prefix_pos = (starts[:n_frames, None] + torch.arange(prefix_len)).reshape(-1)
        perm = torch.empty(total, dtype=torch.int64)
        perm[prefix_pos] = torch.arange(n_prefix)
        patch_mask = torch.ones(total, dtype=torch.bool)
        patch_mask[prefix_pos] = False
        patch_pos = patch_mask.nonzero(as_tuple=False).squeeze(1)
        perm[patch_pos] = n_prefix + torch.arange(total - n_prefix)
        device = pixel_values.device
        perm = perm.to(device, non_blocking=True)
        patch_pos = patch_pos.to(device, non_blocking=True)
        cu_seqlens = starts.to(torch.int32).to(device, non_blocking=True)
        max_seqlen = int(seqlens.max())
        # Patch embed as one Linear over the flattened patch rows.
        weight = conv.weight
        flat_weight = weight.reshape(weight.shape[0], -1)
        x = pixel_values.to(flat_weight.dtype)
        if conv.bias is not None:
            x = torch.addmm(conv.bias, x, flat_weight.t())
        else:
            x = x @ flat_weight.t()

        prefix = torch.cat([emb.cls_token, emb.register_tokens], dim=1)[0]
        source = torch.cat(
            [
                prefix.unsqueeze(0)
                .expand(n_frames, prefix_len, prefix.shape[-1])
                .reshape(n_prefix, -1),
                x,
            ],
            dim=0,
        )
        hidden = source.index_select(0, perm).unsqueeze(0)  # [1, total, E]

        # Rope tables: one build (and one train-time aug draw) per distinct
        # grid; frames sharing a grid share the draw. The dummy carries only
        # shape/device/dtype (zero storage).
        patch = self.patch_size
        tables: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        for grid in frame_grids:
            if grid not in tables:
                grid_h, grid_w = grid
                dummy = x.new_zeros(1).expand(
                    1, self.config.in_channels, grid_h * patch, grid_w * patch
                )
                cos_g, sin_g = self.encoder.rope_embeddings(dummy)
                tables[grid] = (cos_g[0], sin_g[0])
        cos = torch.cat([tables[grid][0] for grid in frame_grids], dim=0)
        sin = torch.cat([tables[grid][1] for grid in frame_grids], dim=0)

        for layer in self.encoder.layer:
            hidden = layer(
                hidden,
                attention_mask=None,
                position_embeddings=(cos, sin),
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        hidden = self.encoder.norm(hidden)
        return hidden[0].index_select(0, patch_pos)
