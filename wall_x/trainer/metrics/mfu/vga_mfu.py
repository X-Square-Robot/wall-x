"""Analytical FLOPs model for VGA(wan) training MFU.

Pure functions of (pipe structure, typed cfg, batch geometry) -- no adapter
state. The adapter owns runtime concerns (recording the live batch geometry,
CUDA-event timers, diagnostics logging) and calls in here for the math.

Formula source of truth: mirrors the module structure of
``wall_x/model/vga/modeling`` (WanTextEncoder / CLIP ViT / WanVideoVAE
encoder / DiT / ActionTransformer).
"""

from __future__ import annotations

import torch


def text_encoder_flops(text_encoder, batch_size, seq_len):
    """Compute one forward-pass FLOPs for WanTextEncoder (T5-style, frozen).

    Architecture per layer (T5SelfAttention):
      - T5Attention: q/k/v (dim→dim_attn) + o (dim_attn→dim) projections
                     + QK^T and AV matmuls
      - T5FeedForward (gated SwiGLU): gate Linear(dim,dim_ffn) + fc1 + fc2

    Args:
        text_encoder: WanTextEncoder with .dim / .dim_attn / .dim_ffn attrs.
        batch_size:   per-GPU batch size (text prompt shared across views).
        seq_len:      text sequence length (e.g. 512 for umt5-xxl).

    Returns:
        int: total FLOPs for one forward pass.
    """
    n_layers = len(text_encoder.blocks)
    if n_layers == 0:
        return 0

    dim = text_encoder.dim
    dim_attn = text_encoder.dim_attn
    dim_ffn = text_encoder.dim_ffn
    B, S = batch_size, seq_len

    flops_proj = 2 * B * S * (3 * dim * dim_attn + dim_attn * dim)
    flops_attn = 4 * B * S * S * dim_attn
    flops_ffn = 2 * B * S * (2 * dim * dim_ffn + dim_ffn * dim)

    return n_layers * (flops_proj + flops_attn + flops_ffn)


def clip_vit_flops(vit, batch_size):
    """Compute one forward-pass FLOPs for the CLIP ViT (frozen).

    Architecture per layer:
      - SelfAttention: combined qkv (dim->dim*3) + o (dim->dim) projections
                       + QK^T and AV matmuls
      - MLP: fc1 (dim->dim*mlp_ratio) + fc2 (dim*mlp_ratio->dim)

    Plus patch embedding Conv2d at the front.

    Args:
        vit:        VisionTransformer with .dim/.num_layers/.num_heads/.mlp_ratio/
                    .patch_size/.image_size/.pos_embedding attrs.
        batch_size: per-GPU batch size (one image per sample).

    Returns:
        int: total FLOPs for one forward pass.
    """
    dim = vit.dim
    num_layers = vit.num_layers
    num_heads = vit.num_heads
    ffn_dim = int(dim * vit.mlp_ratio)
    patch_size = vit.patch_size
    image_size = vit.image_size
    S = vit.pos_embedding.shape[1]
    B = batch_size
    head_dim = dim // num_heads

    n_patches_h = image_size // patch_size
    n_patches_w = image_size // patch_size
    flops_patch = 2 * B * dim * n_patches_h * n_patches_w * 3 * patch_size * patch_size

    flops_proj = 2 * B * S * dim * (3 * dim + dim)
    flops_attn = 4 * B * num_heads * S * S * head_dim
    flops_mlp = 4 * B * S * dim * ffn_dim

    return flops_patch + num_layers * (flops_proj + flops_attn + flops_mlp)


def vae_encoder_flops(vae, batch_size, num_frames, height, width, num_views):
    """Compute one forward-pass FLOPs for the WanVideoVAE encoder (frozen).

    All views are batched together in a single call (B_eff = batch_size × num_views).
    There are 2 such calls per training step:
      1) InputVideoEmbedder  — encode noisy video   [B*V, C, T, H, W]
      2) ImageEmbedderVAE    — encode observation    [V*B, C, T, H, W]

    Handles both Encoder3d (standard) and Encoder3d_38 (38-channel variant).
    """
    try:
        from wall_x.model.vga.modeling.wan_vae import Encoder3d_38
    except ImportError:
        Encoder3d_38 = None

    vae_model = getattr(vae, "model", vae)
    encoder = getattr(vae_model, "encoder", None)
    if encoder is None:
        return 0

    dim = encoder.dim
    z_dim = encoder.z_dim
    dim_mult = list(encoder.dim_mult)
    num_res_blocks = encoder.num_res_blocks
    attn_scales = list(encoder.attn_scales)
    temperal_downsample = list(encoder.temperal_downsample)
    is_38 = Encoder3d_38 is not None and isinstance(encoder, Encoder3d_38)

    B = batch_size * num_views
    T_c, H_c, W_c = num_frames, height, width
    if is_38:
        H_c //= 2
        W_c //= 2
    dims = [dim * u for u in [1] + dim_mult]

    def c3d(C_in, C_out, kT, kH, kW, T_o, H_o, W_o):
        return 2 * B * C_out * T_o * H_o * W_o * C_in * kT * kH * kW

    def c2d(C_in, C_out, kH, kW, H_o, W_o, T_f):
        return 2 * B * T_f * C_out * H_o * W_o * C_in * kH * kW

    total = 0

    in_ch = 12 if is_38 else 3
    total += c3d(in_ch, dims[0], 3, 3, 3, T_c, H_c, W_c)

    scale = 1.0
    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        cur_in = in_dim

        for _ in range(num_res_blocks):
            total += c3d(cur_in, out_dim, 3, 3, 3, T_c, H_c, W_c)
            total += c3d(out_dim, out_dim, 3, 3, 3, T_c, H_c, W_c)
            if cur_in != out_dim:
                total += c3d(cur_in, out_dim, 1, 1, 1, T_c, H_c, W_c)
            cur_in = out_dim

            if scale in attn_scales:
                total += c2d(out_dim, out_dim * 3, 1, 1, H_c, W_c, T_c)
                total += c2d(out_dim, out_dim, 1, 1, H_c, W_c, T_c)
                total += 4 * B * T_c * (H_c * W_c) ** 2 * out_dim

        if i != len(dim_mult) - 1:
            use_temp = temperal_downsample[i] if i < len(temperal_downsample) else False
            H_out, W_out = H_c // 2, W_c // 2
            total += c2d(out_dim, out_dim, 3, 3, H_out, W_out, T_c)
            H_c, W_c = H_out, W_out
            if use_temp:
                T_out = (T_c - 1) // 2 + 1
                total += c3d(out_dim, out_dim, 3, 1, 1, T_out, H_c, W_c)
                T_c = T_out
            scale /= 2.0

    out_dim = dims[-1]

    total += c3d(out_dim, out_dim, 3, 3, 3, T_c, H_c, W_c) * 2
    total += c2d(out_dim, out_dim * 3, 1, 1, H_c, W_c, T_c)
    total += c2d(out_dim, out_dim, 1, 1, H_c, W_c, T_c)
    total += 4 * B * T_c * (H_c * W_c) ** 2 * out_dim
    total += c3d(out_dim, out_dim, 3, 3, 3, T_c, H_c, W_c) * 2

    total += c3d(out_dim, z_dim, 3, 3, 3, T_c, H_c, W_c)

    if hasattr(vae_model, "conv1"):
        try:
            c1w = vae_model.conv1.weight
            total += c3d(c1w.shape[1], c1w.shape[0], 1, 1, 1, T_c, H_c, W_c)
        except Exception:
            total += c3d(z_dim, z_dim, 1, 1, 1, T_c, H_c, W_c)

    return total


# --------------------------------------------------------------------------- #
# Batch geometry (what the forward actually saw / what the config implies)
# --------------------------------------------------------------------------- #


def batch_info_from_videos(videos):
    if not isinstance(videos, (list, tuple)) or not videos:
        return None
    first = videos[0]
    if not torch.is_tensor(first) or first.dim() < 5:
        return None
    return {
        "batch_size": int(first.shape[0]),
        "num_frames": int(first.shape[1]),
        "height": int(first.shape[2]),
        "width": int(first.shape[3]),
        "num_views": int(len(videos)),
    }


def batch_info_from_raw_batch(batch, cfg):
    obs = batch.get("obs")
    next_obs = batch.get("next_obs")
    if not isinstance(obs, dict) or not isinstance(next_obs, dict):
        return None

    camera_names = batch.get("camera_names")
    num_views = None
    if isinstance(camera_names, (list, tuple)) and camera_names:
        first_names = camera_names[0]
        if isinstance(first_names, (list, tuple)) and first_names:
            num_views = len(first_names)

    if num_views is not None:
        view_keys = [f"view{i}" for i in range(1, num_views + 1)]
    else:
        view_keys = sorted(
            key
            for key in obs.keys()
            if isinstance(key, str) and key.startswith("view") and key[4:].isdigit()
        )
        num_views = len([key for key in view_keys if key in next_obs])

    first_key = None
    for key in view_keys:
        if torch.is_tensor(obs.get(key)) and torch.is_tensor(next_obs.get(key)):
            first_key = key
            break
    if first_key is None:
        return None

    obs_frames = obs[first_key]
    next_frames = next_obs[first_key]
    if obs_frames.dim() < 5 or next_frames.dim() < 5:
        return None

    history_frames = int(getattr(cfg.data, "image_history_length", 1) or 1)
    num_frames = history_frames + int(next_frames.shape[1])
    return {
        "batch_size": int(obs_frames.shape[0]),
        "num_frames": int(num_frames),
        "height": int(obs_frames.shape[2]),
        "width": int(obs_frames.shape[3]),
        "num_views": int(num_views),
    }


def fallback_batch_info(cfg):
    m = cfg.model
    d = cfg.data
    val = m.validation or {}

    num_frames = val.get("num_frames")
    future_image_length = getattr(d, "future_image_length", None)
    image_history_length = getattr(d, "image_history_length", None)
    if future_image_length is not None and image_history_length is not None:
        num_frames = int(image_history_length) + int(future_image_length)
    if num_frames is None:
        num_frames = 21

    return {
        "batch_size": int(cfg.hyperparams.batch_size_per_gpu),
        "num_frames": int(num_frames),
        "height": int(m.height),
        "width": int(m.width),
        "num_views": int(m.num_views),
    }


# --------------------------------------------------------------------------- #
# Per-step FLOPs (analytical, with per-component breakdown + GEMM table)
# --------------------------------------------------------------------------- #


def compute_flops_per_step(model, cfg, runtime_batch_info=None, logger=None):
    """Compute theoretical FLOPs per training step with per-component breakdown.

    Component multipliers (fwd + bwd cost):
      - TextEncoder:       frozen, 1× forward
      - VAE encoder:       frozen, 1× forward
      - DiT:               trainable, 3× forward  (fwd + 2× bwd ≈ 3× fwd)
      - ActionTransformer: frozen, 1× forward

    Args:
        model: the (unwrapped) training module exposing ``.pipe``.
        cfg:   typed TrainConfig.
        runtime_batch_info: geometry dict recorded from the live forward
            (see :func:`batch_info_from_videos`); None falls back to the
            config-implied geometry.
        logger: optional logger for warnings.

    Returns:
        dict with per-component and total FLOPs (incl. ``gemm_table`` whose
        ``timer_key``/``timer_phase`` entries the adapter's CUDA-event GEMM
        timers key into).
    """
    pipe = model.pipe if hasattr(model, "pipe") else model.module.pipe
    dit = pipe.dit

    dim = dit.dim
    num_layers = len(dit.blocks)
    ffn_dim = dit.blocks[0].ffn[0].out_features
    patch_size = dit.patch_size
    has_image_input = dit.has_image_input
    has_view_attn = hasattr(dit.blocks[0], "view_attn")

    m = cfg.model
    using_runtime_batch = runtime_batch_info is not None
    batch_info = runtime_batch_info or fallback_batch_info(cfg)
    height = batch_info["height"]
    width = batch_info["width"]
    num_views = batch_info["num_views"]
    batch_size = batch_info["batch_size"]
    num_frames = batch_info["num_frames"]
    try:
        height, width, num_frames = pipe.check_resize_height_width(
            height, width, num_frames
        )
    except Exception:
        pass
    enable_action = m.enable_action
    action_horizon = cfg.task.action_horizon
    dit_train_multiplier = 3.0

    abm = m.action_video_block_mapping or ""
    if isinstance(abm, str):
        num_action_blocks = len([x for x in abm.split(",") if x.strip()])
    elif isinstance(abm, list):
        num_action_blocks = len(abm)
    else:
        num_action_blocks = 0

    pt, ph, pw = patch_size
    vae = getattr(pipe, "vae", None)
    vae_spatial_factor = int(getattr(vae, "upsampling_factor", 8) or 8)
    latent_frames = batch_info.get("latent_frames")
    latent_height = batch_info.get("latent_height")
    latent_width = batch_info.get("latent_width")
    seq_frames = batch_info.get("seq_frames")
    seq_height = batch_info.get("seq_height")
    seq_width = batch_info.get("seq_width")
    if (
        latent_frames is not None
        and latent_height is not None
        and latent_width is not None
    ):
        # Prefer the real VAE/DiT latent grid recorded from the forward path.
        f_latent = int(latent_frames)
        h_latent = int(latent_height)
        w_latent = int(latent_width)
        latent_source = batch_info.get("latent_source", "forward")
    else:
        f_latent = (num_frames - 1) // 4 + 1
        h_latent = height // vae_spatial_factor
        w_latent = width // vae_spatial_factor
        source_prefix = "batch_video" if using_runtime_batch else "config"
        latent_source = f"{source_prefix}_vae_factor_{vae_spatial_factor}"
    if seq_frames is not None and seq_height is not None and seq_width is not None:
        f = int(seq_frames)
        h = int(seq_height)
        w = int(seq_width)
    else:
        f = (f_latent - pt) // pt + 1
        h = (h_latent - ph) // ph + 1
        w = (w_latent - pw) // pw + 1
    if f <= 0 or h <= 0 or w <= 0:
        raise ValueError(
            "Invalid DiT latent grid for MFU: "
            f"latent={(f_latent, h_latent, w_latent)}, "
            f"patch_size={tuple(patch_size)}, source={latent_source}"
        )
    S = f * h * w
    # DiT processes all camera views as an effective batch.
    B = int(batch_info.get("effective_batch", batch_size * num_views))
    logical_batch_size = max(B // max(num_views, 1), 1)
    S_text = (
        pipe.prompter.tokenizer.seq_len
        if hasattr(pipe, "prompter")
        and getattr(pipe.prompter, "tokenizer", None) is not None
        else 512
    )
    if (
        has_image_input
        and hasattr(pipe, "image_encoder")
        and pipe.image_encoder is not None
    ):
        S_img = pipe.image_encoder.model.visual.pos_embedding.shape[1]
    else:
        S_img = 0

    # 1. TextEncoder (T5-style, frozen, 1×)
    flops_text = 0
    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        try:
            flops_text = text_encoder_flops(pipe.text_encoder, batch_size, S_text)
        except Exception:
            flops_text = 0

    # 2. VAE encoder (3-D conv, frozen, fwd only)
    flops_vae = 0
    num_vae_calls = 2
    if hasattr(pipe, "vae") and pipe.vae is not None:
        try:
            flops_vae = vae_encoder_flops(
                pipe.vae, batch_size, num_frames, height, width, num_views
            )
        except Exception as e:
            if logger:
                logger.info(f"[MFU] WARNING: vae_encoder_flops failed: {e}")
            flops_vae = 0

    # 2b. CLIP ViT (frozen, 1× forward)
    flops_clip = 0
    if (
        has_image_input
        and hasattr(pipe, "image_encoder")
        and pipe.image_encoder is not None
    ):
        try:
            vit = pipe.image_encoder.model.visual
            flops_clip = clip_vit_flops(vit, batch_size)
        except Exception:
            flops_clip = 0

    # 3. DiT (trainable, 3× forward)
    flops_self_linear = 8 * B * S * dim**2
    flops_self_attn = 4 * B * S**2 * dim
    flops_self = flops_self_linear + flops_self_attn

    if has_view_attn:
        S_view = num_views * h * w
        B_view = logical_batch_size * f
        flops_view_linear = 10 * B * S * dim**2
        flops_view_attn = 4 * B_view * S_view**2 * dim
        flops_view = flops_view_linear + flops_view_attn
    else:
        flops_view_linear = 0
        flops_view_attn = 0
        flops_view = 0

    n_cross = 4 * S + 4 * S_text
    if has_image_input:
        n_cross += 4 * S_img
    flops_cross_linear = n_cross * B * dim**2
    flops_cross_attn = 4 * B * S * (S_text + S_img) * dim
    flops_cross = flops_cross_linear + flops_cross_attn

    flops_ffn = 4 * B * S * dim * ffn_dim

    flops_dit_block_linear_fwd = num_layers * (
        flops_self_linear + flops_view_linear + flops_cross_linear + flops_ffn
    )
    flops_dit_block_attn_fwd = num_layers * (
        flops_self_attn + flops_view_attn + flops_cross_attn
    )
    flops_dit_fwd = num_layers * (flops_self + flops_view + flops_cross + flops_ffn)
    flops_dit = dit_train_multiplier * flops_dit_fwd

    # 4. ActionTransformer (frozen, 1×)
    flops_action = 0
    if enable_action and num_action_blocks > 0 and hasattr(pipe, "action_transformer"):
        S_a = action_horizon + 1
        B_a = batch_size
        at = pipe.action_transformer
        act_ffn_dim = at.blocks[0].ffn[0].out_features

        f_a_self = 8 * B_a * S_a * dim**2 + 4 * B_a * S_a**2 * dim
        f_a_cross = (
            2 * B_a * S_a * dim**2
            + 2 * B_a * S * 3 * dim * dim
            + 2 * B_a * S * 3 * dim * dim
            + 2 * B_a * S_a * dim**2
            + 4 * B_a * S_a * S * dim
        )
        f_a_ffn = 4 * B_a * S_a * dim * act_ffn_dim

        flops_action = num_action_blocks * (f_a_self + f_a_cross + f_a_ffn)

    flops_forward_total = (
        flops_text
        + flops_vae * num_vae_calls
        + flops_clip
        + flops_dit_fwd
        + flops_action
    )
    flops_total = flops_forward_total + (flops_dit - flops_dit_fwd)

    num_heads = dit.num_heads
    head_dim = dim // num_heads
    gemm_table = []

    def _g(
        name,
        M,
        N,
        K,
        num_calls,
        timer_key=None,
        timer_phase="fwd",
        timer_frac=1.0,
        total_flops=None,
    ):
        flops_1 = 2 * M * N * K
        t_flops = total_flops if total_flops is not None else flops_1 * num_calls
        mem_bytes = 2 * (M * K + K * N + M * N)  # BF16
        gemm_table.append(
            {
                "name": name,
                "M": M,
                "N": N,
                "K": K,
                "flops_per_call": flops_1,
                "num_calls": num_calls,
                "total_flops": t_flops,
                "arith_intensity": flops_1 / mem_bytes if mem_bytes else 0,
                "timer_key": timer_key,
                "timer_phase": timer_phase,
                "timer_frac": timer_frac,
            }
        )

    def _linear(prefix, M, N_out, N_in, n_layers, trainable=True):
        _g(
            f"{prefix}.fwd",
            M,
            N_out,
            N_in,
            n_layers,
            timer_key=prefix,
            timer_phase="fwd",
        )
        if trainable:
            _g(
                f"{prefix}.bwd_dgrad",
                M,
                N_in,
                N_out,
                n_layers,
                timer_key=prefix,
                timer_phase="bwd",
                timer_frac=0.5,
            )
            _g(
                f"{prefix}.bwd_wgrad",
                N_out,
                N_in,
                M,
                n_layers,
                timer_key=prefix,
                timer_phase="bwd",
                timer_frac=0.5,
            )

    def _flash_attn(
        prefix,
        B_outer,
        S_q,
        S_kv,
        nh,
        hd,
        n_layers,
        trainable=True,
        kernel_key=None,
        timer_frac_fwd=1.0,
        timer_frac_bwd=1.0,
    ):
        kk = kernel_key if kernel_key else f"{prefix}._kernel"
        fwd_flops = 4 * B_outer * nh * S_q * S_kv * hd * n_layers
        _g(
            f"{prefix}.attn.fwd",
            S_q,
            S_kv,
            hd,
            n_layers,
            timer_key=kk,
            timer_phase="fwd",
            timer_frac=timer_frac_fwd,
            total_flops=fwd_flops,
        )
        if trainable:
            bwd_flops = 10 * B_outer * nh * S_q * S_kv * hd * n_layers
            _g(
                f"{prefix}.attn.bwd",
                S_q,
                S_kv,
                hd,
                n_layers,
                timer_key=kk,
                timer_phase="bwd",
                timer_frac=timer_frac_bwd,
                total_flops=bwd_flops,
            )

    def _conv3d(prefix, B_c, C_in, C_out, kT, kH, kW, T_o, H_o, W_o):
        M_c = B_c * T_o * H_o * W_o
        K_c = C_in * kT * kH * kW
        _g(prefix, M_c, C_out, K_c, 1, timer_key=prefix, timer_phase="fwd")

    def _conv2d(prefix, B_c, T_f, C_in, C_out, kH, kW, H_o, W_o):
        M_c = B_c * T_f * H_o * W_o
        K_c = C_in * kH * kW
        _g(prefix, M_c, C_out, K_c, 1, timer_key=prefix, timer_phase="fwd")

    # 5a. TextEncoder GEMMs
    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        try:
            te = pipe.text_encoder
            te_n = len(te.blocks)
            te_dim = te.dim
            te_dim_attn = te.dim_attn
            te_dim_ffn = te.dim_ffn
            te_nh = te.num_heads
            te_hd = te_dim_attn // te_nh if te_nh else te_dim_attn
            B_te = batch_size
            S_te = S_text
            BT_te = B_te * S_te
            for proj in ("q", "k", "v"):
                _linear(
                    f"text.attn.{proj}",
                    BT_te,
                    te_dim_attn,
                    te_dim,
                    te_n,
                    trainable=False,
                )
            _linear("text.attn.o", BT_te, te_dim, te_dim_attn, te_n, trainable=False)
            _flash_attn(
                "text.attn", B_te, S_te, S_te, te_nh, te_hd, te_n, trainable=False
            )
            _linear("text.ffn.gate", BT_te, te_dim_ffn, te_dim, te_n, trainable=False)
            _linear("text.ffn.fc1", BT_te, te_dim_ffn, te_dim, te_n, trainable=False)
            _linear("text.ffn.fc2", BT_te, te_dim, te_dim_ffn, te_n, trainable=False)
        except Exception:
            pass

    # 5b. Image Encoder GEMMs (CLIP ViT, frozen)
    if (
        has_image_input
        and hasattr(pipe, "image_encoder")
        and pipe.image_encoder is not None
    ):
        try:
            vit = pipe.image_encoder.model.visual
            vit_dim = vit.dim
            vit_nl = vit.num_layers
            vit_nh = vit.num_heads
            vit_hd = vit_dim // vit_nh
            vit_ffn = int(vit_dim * vit.mlp_ratio)
            vit_ps = vit.patch_size
            vit_S = vit.pos_embedding.shape[1]
            B_vit = batch_size
            BT_vit = B_vit * vit_S

            n_patches_h = vit.image_size // vit_ps
            n_patches_w = vit.image_size // vit_ps
            _conv2d(
                "clip.patch_embed",
                B_vit,
                1,
                3,
                vit_dim,
                vit_ps,
                vit_ps,
                n_patches_h,
                n_patches_w,
            )

            _linear(
                "clip.attn.qkv",
                BT_vit,
                vit_dim * 3,
                vit_dim,
                vit_nl,
                trainable=False,
            )
            _linear("clip.attn.o", BT_vit, vit_dim, vit_dim, vit_nl, trainable=False)
            _flash_attn(
                "clip.attn",
                B_vit,
                vit_S,
                vit_S,
                vit_nh,
                vit_hd,
                vit_nl,
                trainable=False,
            )
            _linear("clip.ffn.fc1", BT_vit, vit_ffn, vit_dim, vit_nl, trainable=False)
            _linear("clip.ffn.fc2", BT_vit, vit_dim, vit_ffn, vit_nl, trainable=False)
        except Exception:
            pass

    # 5c. VAE Encoder GEMMs
    if hasattr(pipe, "vae") and pipe.vae is not None:
        try:
            from wall_x.model.vga.modeling.wan_vae import Encoder3d_38
        except ImportError:
            Encoder3d_38 = None

        try:
            vae_model = getattr(pipe.vae, "model", pipe.vae)
            encoder = getattr(vae_model, "encoder", None)
            if encoder is not None:
                v_dim = encoder.dim
                v_z = encoder.z_dim
                v_mult = list(encoder.dim_mult)
                v_nres = encoder.num_res_blocks
                v_attn = list(encoder.attn_scales)
                v_td = list(encoder.temperal_downsample)
                v_is38 = Encoder3d_38 is not None and isinstance(encoder, Encoder3d_38)

                B_v = batch_size * num_views
                T_v, H_v, W_v = num_frames, height, width
                v_dims = [v_dim * u for u in [1] + v_mult]

                v_inch = 12 if v_is38 else 3
                _conv3d("vae.init_conv", B_v, v_inch, v_dims[0], 3, 3, 3, T_v, H_v, W_v)

                v_scale = 1.0
                for si, (v_cin, v_cout) in enumerate(zip(v_dims[:-1], v_dims[1:])):
                    cur = v_cin
                    for ri in range(v_nres):
                        _conv3d(
                            f"vae.s{si}.r{ri}.c1",
                            B_v,
                            cur,
                            v_cout,
                            3,
                            3,
                            3,
                            T_v,
                            H_v,
                            W_v,
                        )
                        _conv3d(
                            f"vae.s{si}.r{ri}.c2",
                            B_v,
                            v_cout,
                            v_cout,
                            3,
                            3,
                            3,
                            T_v,
                            H_v,
                            W_v,
                        )
                        if cur != v_cout:
                            _conv3d(
                                f"vae.s{si}.r{ri}.sc",
                                B_v,
                                cur,
                                v_cout,
                                1,
                                1,
                                1,
                                T_v,
                                H_v,
                                W_v,
                            )
                        cur = v_cout
                        if v_scale in v_attn:
                            _conv2d(
                                f"vae.s{si}.r{ri}.aqkv",
                                B_v,
                                T_v,
                                v_cout,
                                v_cout * 3,
                                1,
                                1,
                                H_v,
                                W_v,
                            )
                            _conv2d(
                                f"vae.s{si}.r{ri}.aproj",
                                B_v,
                                T_v,
                                v_cout,
                                v_cout,
                                1,
                                1,
                                H_v,
                                W_v,
                            )
                            _g(
                                f"vae.s{si}.r{ri}.asdpa",
                                H_v * W_v,
                                H_v * W_v,
                                v_cout,
                                2 * B_v * T_v,
                            )
                    if si != len(v_mult) - 1:
                        v_use_t = v_td[si] if si < len(v_td) else False
                        Hd, Wd = H_v // 2, W_v // 2
                        _conv2d(
                            f"vae.s{si}.down_s",
                            B_v,
                            T_v,
                            v_cout,
                            v_cout,
                            3,
                            3,
                            Hd,
                            Wd,
                        )
                        H_v, W_v = Hd, Wd
                        if v_use_t:
                            Td = (T_v - 1) // 2 + 1
                            _conv3d(
                                f"vae.s{si}.down_t",
                                B_v,
                                v_cout,
                                v_cout,
                                3,
                                1,
                                1,
                                Td,
                                H_v,
                                W_v,
                            )
                            T_v = Td
                        v_scale /= 2.0

                v_out = v_dims[-1]
                for mi in range(2):
                    _conv3d(
                        f"vae.mid.r{mi}.c1",
                        B_v,
                        v_out,
                        v_out,
                        3,
                        3,
                        3,
                        T_v,
                        H_v,
                        W_v,
                    )
                    _conv3d(
                        f"vae.mid.r{mi}.c2",
                        B_v,
                        v_out,
                        v_out,
                        3,
                        3,
                        3,
                        T_v,
                        H_v,
                        W_v,
                    )
                _conv2d("vae.mid.aqkv", B_v, T_v, v_out, v_out * 3, 1, 1, H_v, W_v)
                _conv2d("vae.mid.aproj", B_v, T_v, v_out, v_out, 1, 1, H_v, W_v)
                _g(
                    "vae.mid.asdpa",
                    H_v * W_v,
                    H_v * W_v,
                    v_out,
                    2 * B_v * T_v,
                )

                _conv3d("vae.head", B_v, v_out, v_z, 3, 3, 3, T_v, H_v, W_v)

                if hasattr(vae_model, "conv1"):
                    _conv3d("vae.post_conv", B_v, v_z, v_z, 1, 1, 1, T_v, H_v, W_v)
        except Exception:
            pass

    # 5d. DiT GEMMs (trainable, fwd + bwd)
    BT = B * S
    L = num_layers

    for proj in ("q", "k", "v", "o"):
        _linear(f"dit.self_attn.{proj}", BT, dim, dim, L)
    _flash_attn("dit.self_attn", B, S, S, num_heads, head_dim, L)

    if has_view_attn:
        S_view_g = num_views * h * w
        B_view_g = batch_size * f
        for proj in ("q", "k", "v", "o"):
            _linear(f"dit.view_attn.{proj}", BT, dim, dim, L)
        _linear("dit.view_attn.proj", BT, dim, dim, L)
        _flash_attn(
            "dit.view_attn",
            B_view_g,
            S_view_g,
            S_view_g,
            num_heads,
            head_dim,
            L,
        )

    _linear("dit.cross_attn.q", BT, dim, dim, L)
    _linear("dit.cross_attn.k_text", B * S_text, dim, dim, L)
    _linear("dit.cross_attn.v_text", B * S_text, dim, dim, L)
    _linear("dit.cross_attn.o", BT, dim, dim, L)
    _cross_total = S_text + (S_img if has_image_input and S_img > 0 else 0)
    _text_frac = S_text / _cross_total if _cross_total else 1.0
    _img_frac = 1.0 - _text_frac
    _flash_attn(
        "dit.cross_attn.text",
        B,
        S,
        S_text,
        num_heads,
        head_dim,
        L,
        kernel_key="dit.cross_attn._kernel",
        timer_frac_fwd=_text_frac,
        timer_frac_bwd=_text_frac,
    )
    if has_image_input and S_img > 0:
        _linear("dit.cross_attn.k_img", B * S_img, dim, dim, L)
        _linear("dit.cross_attn.v_img", B * S_img, dim, dim, L)
        _flash_attn(
            "dit.cross_attn.img",
            B,
            S,
            S_img,
            num_heads,
            head_dim,
            L,
            kernel_key="dit.cross_attn._kernel",
            timer_frac_fwd=_img_frac,
            timer_frac_bwd=_img_frac,
        )

    _linear("dit.ffn.up", BT, ffn_dim, dim, L)
    _linear("dit.ffn.down", BT, dim, ffn_dim, L)

    # 5e. ActionTransformer GEMMs (frozen, fwd only)
    if enable_action and num_action_blocks > 0 and hasattr(pipe, "action_transformer"):
        at = pipe.action_transformer
        act_ffn_d = at.blocks[0].ffn[0].out_features
        S_a_g = action_horizon + 1
        BT_a = batch_size * S_a_g
        La = num_action_blocks

        for proj in ("q", "k", "v", "o"):
            _linear(
                f"act.self_attn.{proj}",
                BT_a,
                dim,
                dim,
                La,
                trainable=False,
            )
        _flash_attn(
            "act.self_attn",
            batch_size,
            S_a_g,
            S_a_g,
            num_heads,
            head_dim,
            La,
            trainable=False,
        )

        _linear("act.cross_attn.q", BT_a, dim, dim, La, trainable=False)
        _linear(
            "act.cross_attn.k",
            batch_size * S,
            dim,
            dim * num_views,
            La,
            trainable=False,
        )
        _linear(
            "act.cross_attn.v",
            batch_size * S,
            dim,
            dim * num_views,
            La,
            trainable=False,
        )
        _linear("act.cross_attn.o", BT_a, dim, dim, La, trainable=False)
        _flash_attn(
            "act.cross_attn",
            batch_size,
            S_a_g,
            S,
            num_heads,
            head_dim,
            La,
            trainable=False,
        )

        _linear(
            "act.ffn.up",
            BT_a,
            act_ffn_d,
            dim,
            La,
            trainable=False,
        )
        _linear(
            "act.ffn.down",
            BT_a,
            dim,
            act_ffn_d,
            La,
            trainable=False,
        )

    return {
        "flops_per_step": flops_total,
        "flops_forward_total": flops_forward_total,
        "flops_text": flops_text,
        "flops_vae": flops_vae,
        "num_vae_calls": num_vae_calls,
        "flops_clip": flops_clip,
        "flops_dit_fwd": flops_dit_fwd,
        "flops_dit_total": flops_dit,
        "flops_dit_block_linear_fwd": flops_dit_block_linear_fwd,
        "flops_dit_block_attn_fwd": flops_dit_block_attn_fwd,
        "flops_action": flops_action,
        "seq_len": S,
        "seq_dims": (f, h, w),
        "batch_eff": B,
        "dim": dim,
        "ffn_dim": ffn_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_views": num_views,
        "num_frames": num_frames,
        "latent_dims": (f_latent, h_latent, w_latent),
        "latent_source": latent_source,
        "vae_spatial_factor": vae_spatial_factor,
        "height": height,
        "width": width,
        "dit_train_multiplier": dit_train_multiplier,
        "gemm_table": gemm_table,
    }
