"""Checkpoint save/load helpers for distributed training."""

from __future__ import annotations

import gc
import logging
import os
import random
import shutil
import time
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import yaml
from safetensors.torch import load_file, save_file
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.nn.parallel import DistributedDataParallel as DDP

from wall_x.trainer.optimizer.dmuon import is_dmuon_model


def _noop_log(_msg: str, **_kw) -> None:
    pass


def _dict_section(config: dict, key: str) -> dict:
    section = config.get(key, {})
    return section if isinstance(section, dict) else {}


# ----------------------------------------------------------------------
# Detectors
# ----------------------------------------------------------------------


def _is_fsdp2_model(model: torch.nn.Module) -> bool:
    """Detect FSDP2 models by the presence of DTensor parameters.

    FSDP2's ``fully_shard`` converts parameters in-place to DTensors without
    wrapping the module in an outer class. DDP wraps in
    ``DistributedDataParallel``; unwrapped models have plain
    ``torch.Tensor`` parameters.
    """
    if isinstance(model, DDP):
        return False
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        return False
    for p in model.parameters():
        if isinstance(p, DTensor):
            return True
    return False


def _detect_legacy_fsdp1_format(checkpoint_path: str) -> bool:
    """True when the ckpt directory looks like a pre-migration FSDP1 save.

    Legacy: ``model.safetensors`` is present (rank-0 full state), but
    optimizer state lives in per-rank ``optimizer_rank{N}.pt`` files
    instead of a single ``optimizer.pt``. After the FSDP1 -> FSDP2
    migration we cannot reshard those flat_param-keyed optim files into
    the new DTensor layout, so the legacy loader cold-starts the
    optimizer and warns.
    """
    if not os.path.isdir(checkpoint_path):
        return False
    if os.path.exists(os.path.join(checkpoint_path, "optimizer.pt")):
        return False
    try:
        return any(
            f.startswith("optimizer_rank") and f.endswith(".pt")
            for f in os.listdir(checkpoint_path)
        )
    except OSError:
        return False


# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------


def save_checkpoint(
    *,
    ckpt_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    config: dict,
    rank: int,
    is_main: bool,
    epoch: int,
    global_step: int,
    seed: int,
    normalizer_action,
    normalizer_propri,
    dataset=None,
    grad_scaler=None,
    log_fn: Optional[Callable] = None,
    frozen_prefixes: Optional[Tuple[str, ...]] = None,
) -> None:
    """Save model + optimizer + scheduler + metadata at *ckpt_path*.

    ``frozen_prefixes``: when set, model state-dict keys starting with any
    of these prefixes are excluded from the saved file. This is used for
    frozen-by-design submodules that should not be duplicated in every
    checkpoint. Loaders already use ``strict=False``.
    """
    log_fn = log_fn or _noop_log
    os.makedirs(ckpt_path, exist_ok=True)

    if is_main:
        _save_training_checkpoint_metadata(
            ckpt_path=ckpt_path,
            config=config,
            epoch=epoch,
            global_step=global_step,
            seed=seed,
            world_size=_world_size_for_metadata(),
            normalizer_action=normalizer_action,
            normalizer_propri=normalizer_propri,
            grad_scaler=grad_scaler,
            log_fn=log_fn,
        )

    if is_dmuon_model(model):
        _save_dmuon_state_dict(
            ckpt_path=ckpt_path,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            is_main=is_main,
            log_fn=log_fn,
            frozen_prefixes=frozen_prefixes,
        )
    elif _is_fsdp2_model(model):
        _save_fsdp2_full_state_dict(
            ckpt_path=ckpt_path,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            is_main=is_main,
            log_fn=log_fn,
            frozen_prefixes=frozen_prefixes,
        )
    else:
        # DDP or unwrapped (fallback).
        _save_ddp_state_dict(
            ckpt_path=ckpt_path,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            is_main=is_main,
            log_fn=log_fn,
            frozen_prefixes=frozen_prefixes,
        )

    # Per-rank dataset state (only when saving mid-epoch, step != 0).
    if dataset is not None and global_step != 0:
        _save_dataset_state(
            ckpt_path=ckpt_path, dataset=dataset, rank=rank, log_fn=log_fn
        )


def _save_training_checkpoint_metadata(
    *,
    ckpt_path: str,
    config: dict,
    epoch: int,
    global_step: int,
    seed: int,
    world_size: int,
    normalizer_action,
    normalizer_propri,
    grad_scaler=None,
    log_fn: Callable,
) -> None:
    torch.save({"seed": seed}, os.path.join(ckpt_path, "seed.pth"))
    torch.save({"global_step": global_step}, os.path.join(ckpt_path, "global_step.pth"))
    torch.save({"current_epoch": epoch}, os.path.join(ckpt_path, "current_epoch.pth"))

    # world_size: public metadata, used at resume to detect reshard.
    # Written unconditionally (previously sharded-mode only).
    torch.save({"world_size": world_size}, os.path.join(ckpt_path, "world_size.pth"))

    # RNG state (rank-0 snapshot; every rank restores the same state on
    # resume, matching the existing seed_all(seed) convention where every
    # rank is seeded identically).
    rng_state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state()
    torch.save(rng_state, os.path.join(ckpt_path, "rng_state.pt"))

    # GradScaler state (fp16 AMP only).
    if grad_scaler is not None:
        torch.save(grad_scaler.state_dict(), os.path.join(ckpt_path, "grad_scaler.pt"))

    with open(os.path.join(ckpt_path, "config.yml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    model_cfg = _dict_section(config, "model")
    data_cfg = _dict_section(config, "data")

    # Copy processor files to checkpoint directory.
    processor_dir = model_cfg.get("processor_path") or config.get("processor_path")
    if processor_dir is None:
        # Backward compatibility: fall back to pretrained_qwen_vl_path.
        processor_dir = config.get("pretrained_qwen_vl_path")
        if processor_dir is not None:
            log_fn(
                "WARNING: 'pretrained_qwen_vl_path' is deprecated for processor "
                "file copying, please use 'processor_path' instead.",
                level=logging.WARNING,
            )

    if processor_dir is not None:
        for filename in (
            "preprocessor_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "tokenizer.model",
            "special_tokens_map.json",
            "vocab.json",
        ):
            src = os.path.join(processor_dir, filename)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(ckpt_path, filename))

    act_config_path = model_cfg.get("config_path") or config.get(
        "qwen_vl_act_config_path"
    )
    if act_config_path is not None:
        if os.path.exists(act_config_path):
            shutil.copy(act_config_path, os.path.join(ckpt_path, "config.json"))
            log_fn(f"[Checkpoint] Copied act config to {ckpt_path}/config.json")
        else:
            log_fn(f"[Checkpoint] WARNING: {act_config_path} not found, skipping.")

    norm_stats_path = data_cfg.get("norm_stats_path") or config.get("norm_stats_path")
    if norm_stats_path is not None:
        if os.path.exists(norm_stats_path):
            shutil.copy(norm_stats_path, os.path.join(ckpt_path, "norm_stats.json"))
            log_fn(f"[Checkpoint] Copied norm stats to {ckpt_path}/norm_stats.json")
        elif (data_cfg.get("dataset_type") or config.get("dataset_type")) == "lerobot":
            log_fn(f"[Checkpoint] WARNING: {norm_stats_path} not found, skipping.")

    torch.save(
        normalizer_action.state_dict(),
        os.path.join(ckpt_path, "normalizer_action.pth"),
    )
    torch.save(
        normalizer_propri.state_dict(),
        os.path.join(ckpt_path, "normalizer_propri.pth"),
    )


def _save_fsdp2_full_state_dict(
    *,
    ckpt_path: str,
    model,
    optimizer,
    lr_scheduler,
    is_main: bool,
    log_fn: Callable,
    frozen_prefixes: Optional[Tuple[str, ...]] = None,
) -> None:
    """Save FSDP2 model + optimizer as rank-0 full state dict.

    ``get_state_dict`` with ``full_state_dict=True, cpu_offload=True``
    gathers DTensors to full CPU tensors on rank 0 (other ranks get empty
    dicts). This is the cross-world-size-compatible format: on resume,
    ``set_state_dict`` with ``broadcast_from_rank0=True`` re-shards from
    rank 0's copy to whatever mesh the new run has.
    """
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    model_sd, optim_sd = get_state_dict(model, optimizer, options=options)

    if is_main:
        model_sd = _filter_frozen_prefixes(model_sd, frozen_prefixes, log_fn)
        model_sd_out = _make_contiguous_and_clone_shared(model_sd)
        save_file(model_sd_out, os.path.join(ckpt_path, "model.safetensors"))
        torch.save(optim_sd, os.path.join(ckpt_path, "optimizer.pt"))
        torch.save(lr_scheduler.state_dict(), os.path.join(ckpt_path, "scheduler.pt"))
        log_fn("[Checkpoint] Saved FSDP2 full state dict (rank 0)")

    # Release the consolidated copy immediately; non-rank-0 already held {}.
    del model_sd, optim_sd
    gc.collect()


def _save_dmuon_state_dict(
    *,
    ckpt_path: str,
    model,
    optimizer,
    lr_scheduler,
    is_main: bool,
    log_fn: Callable,
    frozen_prefixes: Optional[Tuple[str, ...]] = None,
) -> None:
    """Save via DMuon's state-dict helpers (full tensors, HF-compatible keys)."""
    import dmuon

    model_sd = dmuon.get_model_state_dict(model, cpu_offload=True, rank0_only=True)
    if is_main:
        model_sd = _filter_frozen_prefixes(model_sd, frozen_prefixes, log_fn)
        model_sd = _make_contiguous_and_clone_shared(model_sd)
        save_file(model_sd, os.path.join(ckpt_path, "model.safetensors"))
        log_fn("[Checkpoint] Saved DMuon model state dict (full, rank0)")

    optim_sd = dmuon.get_optimizer_state_dict(
        model, optimizer, cpu_offload=True, rank0_only=True
    )
    if is_main:
        torch.save(optim_sd, os.path.join(ckpt_path, "optimizer.pt"))
        torch.save(lr_scheduler.state_dict(), os.path.join(ckpt_path, "scheduler.pt"))

    del model_sd, optim_sd
    gc.collect()


def _save_ddp_state_dict(
    *,
    ckpt_path: str,
    model,
    optimizer,
    lr_scheduler,
    is_main: bool,
    log_fn: Callable = _noop_log,
    frozen_prefixes: Optional[Tuple[str, ...]] = None,
) -> None:
    if not is_main:
        return
    model_state = (
        model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    )
    model_state = _filter_frozen_prefixes(model_state, frozen_prefixes, log_fn)
    model_state_contiguous = _make_contiguous_and_clone_shared(model_state)
    save_file(model_state_contiguous, os.path.join(ckpt_path, "model.safetensors"))
    torch.save(optimizer.state_dict(), os.path.join(ckpt_path, "optimizer.pt"))
    torch.save(lr_scheduler.state_dict(), os.path.join(ckpt_path, "scheduler.pt"))


def _save_dataset_state(
    *, ckpt_path: str, dataset, rank: int, log_fn: Callable
) -> None:
    """Save per-rank dataset resume state if the dataset supports it."""
    if not hasattr(dataset, "save_episode_containers"):
        return
    ec_path = os.path.join(ckpt_path, f"episode_containers_rank_{rank}.pkl")
    dataset.save_episode_containers(ec_path)

    _t0 = time.time()
    while not (os.path.exists(ec_path) and os.path.getsize(ec_path) > 0):
        time.sleep(0.5)
        if time.time() - _t0 > 120:
            log_fn(
                f"WARNING: episode container checkpoint save timeout: {ec_path}",
            )
            break


def _filter_frozen_prefixes(
    state_dict: dict,
    frozen_prefixes: Optional[Tuple[str, ...]],
    log_fn: Callable,
) -> dict:
    """Drop entries whose key starts with any frozen prefix.

    Used to exclude frozen-by-design submodules from checkpoints. Returns
    the input unchanged when no prefixes are configured or the dict is
    already empty (non-rank-0 case).
    """
    if not frozen_prefixes or not state_dict:
        return state_dict
    original = len(state_dict)
    filtered = {
        k: v for k, v in state_dict.items() if not k.startswith(frozen_prefixes)
    }
    log_fn(
        f"[Checkpoint] Filtered state dict: {len(filtered)}/{original} entries "
        f"(excluded {original - len(filtered)} frozen entries)"
    )
    return filtered


def _make_contiguous_and_clone_shared(state_dict: dict) -> dict:
    """Make tensors contiguous for safetensors, cloning any that share storage."""
    seen_data_ptrs = {}
    out = {}
    for k, v in state_dict.items():
        ptr = v.data_ptr()
        if ptr in seen_data_ptrs:
            v = v.clone()
        else:
            seen_data_ptrs[ptr] = k
        out[k] = v.contiguous() if v.is_floating_point() or v.is_complex() else v
    return out


def _world_size_for_metadata() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------

_EMBED_WEIGHT_KEYS = (
    "model.embed_tokens.weight",
    "model.language_model.embed_tokens.weight",
)


def _maybe_resize_token_embeddings_for_load(
    model: torch.nn.Module,
    state_dict: dict,
    log_fn: Optional[Callable] = None,
) -> None:
    """Resize model embeddings when checkpoint vocab size differs from the model."""
    log_fn = log_fn or _noop_log
    for key in _EMBED_WEIGHT_KEYS:
        if key not in state_dict:
            continue
        ckpt_vocab = state_dict[key].shape[0]
        embed = (
            model.get_input_embeddings()
            if hasattr(model, "get_input_embeddings")
            else None
        )
        if embed is None:
            return
        cur_vocab = embed.weight.shape[0]
        if cur_vocab == ckpt_vocab:
            return
        log_fn(
            f"resize_token_embeddings from {cur_vocab} to {ckpt_vocab} "
            f"to match checkpoint ({key})"
        )
        if hasattr(model, "resize_token_embeddings"):
            model.resize_token_embeddings(ckpt_vocab)
        elif hasattr(model, "model") and hasattr(
            model.model, "resize_token_embeddings"
        ):
            model.model.resize_token_embeddings(ckpt_vocab)
        return


def load_weights(
    *,
    model: torch.nn.Module,
    resume_config: dict,
    model_class=None,
    log_fn: Optional[Callable] = None,
):
    """Load ``model`` weights from ``resume_config['ckpt']`` (file-level).

    Supports .safetensors / .pth sources and optional ``try_harder`` shape
    matching for action-preprocessor weights when action dims changed.
    """
    log_fn = log_fn or _noop_log
    src = resume_config["ckpt"]
    if src.endswith(".safetensors"):
        log_fn(f"Loading model from safetensors: {src}")
        state_dict = load_file(src)
    elif src.endswith(".pth"):
        checkpoint = torch.load(src, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model_state_dict"]
    else:
        raise ValueError(f"Unsupported checkpoint format: {src}")

    if model_class is not None and hasattr(model_class, "is_fused"):
        if not model_class.is_fused(state_dict):
            log_fn("Converting non-fused weights to fused format...")
            state_dict = model_class.convert_to_fused(state_dict)
        else:
            log_fn("The weights is fused, skipping conversion.")

    filtered_state_dict = _drop_checkpoint_normalizer_state(state_dict, log_fn)
    _maybe_resize_token_embeddings_for_load(model, filtered_state_dict, log_fn=log_fn)

    if resume_config.get("try_harder", False):
        log_fn("### try harder to squeeze checkpoint weights into new model ###")
        new_state_dict = reshape_compatible_state_dict(
            filtered_state_dict, model.state_dict(), log_fn=log_fn
        )
        err = model.load_state_dict(new_state_dict, strict=False)
    else:
        err = model.load_state_dict(filtered_state_dict, strict=False)

    log_fn(f"err in load model: {err}")
    return model


def reshape_compatible_state_dict(
    state_dict: dict, model_sd: dict, log_fn: Optional[Callable] = None
) -> dict:
    """Pad / slice action-preprocessor weights to match target model shape."""
    log_fn = log_fn or _noop_log
    out = {}
    for name, param in state_dict.items():
        if name not in model_sd:
            log_fn(f"Not used parameter: {name}")
            continue
        if "action_preprocessor" in name or "action_processor" in name:
            if param.size() == model_sd[name].size():
                out[name] = param
                continue
            size_0 = param.size()
            size_1 = model_sd[name].size()
            if any(old_dim > new_dim for old_dim, new_dim in zip(size_0, size_1)):
                raise ValueError(
                    f"Shape mismatch for '{name}': checkpoint shape {tuple(size_0)} is "
                    f"larger than model shape {tuple(size_1)} in at least one dimension. "
                    "Loading a larger checkpoint into a smaller model is not supported here. "
                    "If the action dimension has changed, please configure action padding "
                    "in the dataset processor so checkpoint actions match the new action "
                    "size."
                )
            out[name] = model_sd[name].clone()
            slices = [
                slice(0, min(old_dim, new_dim))
                for old_dim, new_dim in zip(size_0, size_1)
            ]
            out[name][slices] = param[slices]
            log_fn(
                f"Not match key: {name}, checkpoint shape: {tuple(size_0)}, "
                f"model shape: {tuple(size_1)}. Filled checkpoint weights into the first "
                f"{[s.stop for s in slices]} dims, remaining dims keep model init weights."
            )
        else:
            if param.size() == model_sd[name].size():
                out[name] = param
            else:
                log_fn(
                    f"Skipping '{name}': checkpoint shape {tuple(param.size())} "
                    f"!= model shape {tuple(model_sd[name].size())}"
                )
    return out


def resume_from_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    resume_config: dict,
    rank: int,
    grad_scaler=None,
    model_class=None,
    log_fn: Optional[Callable] = None,
) -> None:
    """Restore model / optimizer / scheduler / RNG / grad_scaler from ckpt."""
    log_fn = log_fn or _noop_log
    checkpoint_path = resume_config["ckpt"]

    is_fsdp2 = _is_fsdp2_model(model)
    is_dmuon = is_dmuon_model(model)

    # --- world_size diagnostic ---------------------------------------
    ckpt_world_size = _read_ckpt_world_size(checkpoint_path, log_fn)
    ws_mismatch = (
        ckpt_world_size is not None and ckpt_world_size != _world_size_for_metadata()
    )
    if ws_mismatch:
        log_fn(
            f"Cross-world-size resume: ckpt_world_size={ckpt_world_size}, "
            f"current={_world_size_for_metadata()}."
        )

    # --- single-file path (.safetensors / .pth) ----------------------
    if checkpoint_path.endswith(".safetensors") or checkpoint_path.endswith(".pth"):
        _load_weights_into_model(
            model=model,
            resume_config=resume_config,
            is_fsdp2=is_fsdp2,
            is_dmuon=is_dmuon,
            model_class=model_class,
            log_fn=log_fn,
        )
        log_fn(f"Resumed weights from single-file checkpoint: {checkpoint_path}")
        return

    # --- directory path ----------------------------------------------
    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")

    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    rank_shard_path = os.path.join(checkpoint_path, f"model_rank{rank}.pt")

    if os.path.exists(safetensors_path):
        if _detect_legacy_fsdp1_format(checkpoint_path):
            # Pre-migration FSDP1 ckpt: per-rank optimizer files cannot be
            # resharded into the new DTensor layout. Load model only and
            # cold-start the optimizer.
            _load_legacy_fsdp1_full(
                checkpoint_path=checkpoint_path,
                model=model,
                is_fsdp2=is_fsdp2,
                is_dmuon=is_dmuon,
                model_class=model_class,
                try_harder=resume_config.get("try_harder", False),
                log_fn=log_fn,
            )
        elif is_fsdp2 or is_dmuon:
            # New-format path: model + optimizer via state_dict helpers,
            # with automatic reshard on load via broadcast_from_rank0.
            _load_fsdp2_or_dmuon_full(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                is_dmuon=is_dmuon,
                ws_mismatch=ws_mismatch,
                log_fn=log_fn,
            )
        else:
            # DDP / unwrapped: single-file model + single-file optimizer.
            inner_resume = {
                "ckpt": safetensors_path,
                "try_harder": resume_config.get("try_harder", False),
            }
            _load_weights_into_model(
                model=model,
                resume_config=inner_resume,
                is_fsdp2=False,
                is_dmuon=False,
                model_class=model_class,
                log_fn=log_fn,
            )
            _resume_ddp_optimizer_scheduler(
                checkpoint_path=checkpoint_path,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                log_fn=log_fn,
            )
    elif os.path.exists(rank_shard_path):
        # FSDP1 sharded checkpoints are rank-layout-bound; dropping support.
        raise RuntimeError(
            f"Legacy FSDP1 sharded checkpoint at {checkpoint_path} is no "
            f"longer supported. Convert to a single model.safetensors first."
        )
    else:
        raise FileNotFoundError(
            f"No model.safetensors or model_rank*.pt found under {checkpoint_path}"
        )

    # --- auxiliary state (all paths) ---------------------------------
    _resume_rng(checkpoint_path, log_fn)
    if grad_scaler is not None:
        _resume_grad_scaler(checkpoint_path, grad_scaler, log_fn)
    log_fn(f"Resumed from checkpoint: {checkpoint_path}")


def _read_ckpt_world_size(checkpoint_path: str, log_fn: Callable):
    if not os.path.isdir(checkpoint_path):
        return None
    ws_path = os.path.join(checkpoint_path, "world_size.pth")
    if not os.path.exists(ws_path):
        return None
    try:
        return int(torch.load(ws_path, map_location="cpu")["world_size"])
    except (KeyError, ValueError, RuntimeError, TypeError) as e:
        log_fn(
            f"world_size.pth unreadable ({e}); assuming same ws.",
            level=logging.WARNING,
        )
        return None


def _strip_fused_flags(osd: dict) -> None:
    """In-place pop of fused / foreach flags from an optimizer state dict's
    param_groups. Old checkpoints may have fused=True which breaks dtype
    matching when the optimizer is reconstructed under a different precision."""
    for pg in osd.get("param_groups", []) or []:
        pg.pop("fused", None)
        pg.pop("foreach", None)


def _drop_checkpoint_normalizer_state(state_dict: dict, log_fn: Callable) -> dict:
    """Drop saved normalizer buffers so current-run stats stay authoritative."""
    normalizer_state_prefixes = (
        "action_preprocessor.normalizer",
        "action_processor.normalizer",
    )
    filtered = {
        k: v
        for k, v in state_dict.items()
        if not k.startswith(normalizer_state_prefixes)
    }
    dropped = len(state_dict) - len(filtered)
    if dropped:
        log_fn(
            f"[Checkpoint] Dropped {dropped} checkpoint normalizer entries; "
            "keeping current-run normalizers."
        )
    return filtered


def _load_fsdp2_or_dmuon_full(
    *,
    checkpoint_path: str,
    model,
    optimizer,
    lr_scheduler,
    is_dmuon: bool,
    ws_mismatch: bool,
    log_fn: Callable,
) -> None:
    """Load FSDP2 (or DMuon) model + optimizer from a full-state-dict ckpt.

    FSDP2 path: rank 0 deserializes, ``broadcast_from_rank0=True`` in
    ``set_*_state_dict`` reshards to the current mesh.

    DMuon path: every rank MUST load the full state dict from disk.
    ``dmuon.set_model_state_dict`` / ``set_optimizer_state_dict`` iterate
    ``fqn_to_dp`` and skip any FQN not present in the provided dict - so
    if only rank 0 has the data, new owners on other ranks (after a
    cross-ws resume) silently miss their assigned params and
    ``_owned_data`` stays at fresh-init (random weights), producing a
    massive post-resume loss spike. The ckpt files live on CPFS and are
    shared, so per-rank reads are essentially free.
    """
    is_main = dist.get_rank() == 0 if dist.is_initialized() else True
    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    optim_path = os.path.join(checkpoint_path, "optimizer.pt")
    sched_path = os.path.join(checkpoint_path, "scheduler.pt")

    # --- model ---
    if is_dmuon:
        if not os.path.exists(safetensors_path):
            raise FileNotFoundError(
                f"DMuon resume requires {safetensors_path} on every rank."
            )
        # Every rank loads the full state dict so new owners (after a
        # cross-ws reshard, which re-runs dedicate_params and may assign
        # FQNs to different ranks) each see their own FQN in the dict.
        # dmuon.set_model_state_dict is FQN-keyed and only the new owner
        # writes into _owned_data, so passing the full dict on every
        # rank is correct (and required) for the cross-ws case.
        model_sd = _drop_checkpoint_normalizer_state(
            load_file(safetensors_path), log_fn
        )
        import dmuon

        dmuon.set_model_state_dict(model, model_sd)
        log_fn("[Checkpoint] DMuon model state loaded.")
    else:
        if is_main and os.path.exists(safetensors_path):
            model_sd = _drop_checkpoint_normalizer_state(
                load_file(safetensors_path), log_fn
            )
        else:
            model_sd = {}
        options = StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
            broadcast_from_rank0=True,
            strict=False,
        )
        set_model_state_dict(
            model,
            model_state_dict=model_sd,
            options=options,
        )
        log_fn("[Checkpoint] FSDP2 model state loaded (reshard via broadcast).")
    del model_sd
    gc.collect()

    # --- optimizer ---
    if os.path.exists(optim_path):
        if is_dmuon:
            # Same rationale as model_sd: every rank needs full optim state.
            optim_sd = torch.load(optim_path, map_location="cpu", weights_only=False)
            _strip_fused_flags(optim_sd)
        elif is_main:
            optim_sd = torch.load(optim_path, map_location="cpu", weights_only=False)
            _strip_fused_flags(optim_sd)
        else:
            optim_sd = {}

        try:
            if is_dmuon:
                import dmuon

                dmuon.set_optimizer_state_dict(model, optimizer, optim_sd)
                log_fn("[Checkpoint] DMuon optimizer state loaded.")
            else:
                options = StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=True,
                    broadcast_from_rank0=True,
                )
                set_optimizer_state_dict(
                    model,
                    optimizers=optimizer,
                    optim_state_dict=optim_sd,
                    options=options,
                )
                log_fn(
                    "[Checkpoint] FSDP2 optimizer state loaded "
                    "(reshard via broadcast)."
                )
        except (ValueError, RuntimeError, TypeError, KeyError) as e:
            # Most likely a legacy FSDP1 consolidated OSD (flat_param-specific).
            log_fn(
                f"[Checkpoint] Optimizer state incompatible with current "
                f"layout ({e!r}); cold-starting optimizer. Model weights are "
                f"loaded; momentum resets to zero.",
                level=logging.WARNING,
            )
        del optim_sd
        gc.collect()
    else:
        log_fn(
            "[Checkpoint] optimizer.pt not found; cold-starting optimizer.",
            level=logging.WARNING,
        )

    # --- scheduler ---
    if os.path.exists(sched_path):
        try:
            lr_scheduler.load_state_dict(
                torch.load(sched_path, map_location="cpu", weights_only=False)
            )
            log_fn("[Checkpoint] Scheduler state loaded.")
        except (ValueError, RuntimeError, TypeError, KeyError) as e:
            log_fn(
                f"Scheduler load failed ({e}); keeping fresh state.",
                level=logging.WARNING,
            )


def _resume_ddp_optimizer_scheduler(
    *,
    checkpoint_path: str,
    optimizer,
    lr_scheduler,
    log_fn: Callable,
) -> None:
    """Restore optimizer / scheduler for DDP layouts (single-file)."""
    optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
    if os.path.exists(optimizer_path):
        try:
            optim_sd = torch.load(
                optimizer_path, map_location="cpu", weights_only=False
            )
            _strip_fused_flags(optim_sd)
            optimizer.load_state_dict(optim_sd)
            log_fn("[Checkpoint] Optimizer state loaded (single file).")
        except (ValueError, RuntimeError) as e:
            log_fn(
                f"Failed to load optimizer state dict, "
                f"optimizer will be re-initialized. Error: {e}",
                level=logging.WARNING,
            )

    sched_path = os.path.join(checkpoint_path, "scheduler.pt")
    if os.path.exists(sched_path):
        try:
            lr_scheduler.load_state_dict(
                torch.load(sched_path, map_location="cpu", weights_only=False)
            )
            log_fn(f"[Checkpoint] Scheduler state loaded ({sched_path}).")
        except (ValueError, RuntimeError, KeyError, TypeError) as e:
            log_fn(
                f"Scheduler load failed ({e}); keeping fresh state.",
                level=logging.WARNING,
            )


def _load_legacy_fsdp1_full(
    *,
    checkpoint_path: str,
    model,
    is_fsdp2: bool,
    is_dmuon: bool,
    model_class,
    try_harder: bool,
    log_fn: Callable,
) -> None:
    """Load a pre-migration FSDP1 ckpt: weights only, optimizer cold-starts.

    The per-rank ``optimizer_rank{N}.pt`` files use FSDP1's flat_param
    layout, which cannot be resharded into FSDP2's DTensor layout. The
    only safe action is to load the rank-0 ``model.safetensors`` (which
    is layout-agnostic) and let the optimizer warm up from scratch.
    """
    inner_resume = {
        "ckpt": os.path.join(checkpoint_path, "model.safetensors"),
        "try_harder": try_harder,
    }
    _load_weights_into_model(
        model=model,
        resume_config=inner_resume,
        is_fsdp2=is_fsdp2,
        is_dmuon=is_dmuon,
        model_class=model_class,
        log_fn=log_fn,
    )
    log_fn(
        f"Legacy FSDP1 ckpt detected at {checkpoint_path}: model weights "
        f"loaded, but per-rank optimizer files cannot reshard into FSDP2. "
        f"Optimizer state DROPPED - it will warm up from scratch. For long "
        f"resume runs prefer a fresh FSDP2 ckpt; for short runs the loss "
        f"bump is usually negligible.",
        level=logging.WARNING,
    )


def _resume_rng(checkpoint_path: str, log_fn: Callable) -> None:
    path = os.path.join(checkpoint_path, "rng_state.pt")
    if not os.path.exists(path):
        log_fn(
            "rng_state.pt not found; RNG stays at seed_all() state.",
            level=logging.WARNING,
        )
        return
    try:
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if "torch" in sd:
            torch.set_rng_state(sd["torch"])
        if "cuda" in sd and torch.cuda.is_available():
            torch.cuda.set_rng_state(sd["cuda"])
        if "numpy" in sd:
            np.random.set_state(sd["numpy"])
        if "python" in sd:
            random.setstate(sd["python"])
        log_fn("[Checkpoint] RNG state restored.")
    except (RuntimeError, ValueError, TypeError, KeyError) as e:
        log_fn(
            f"RNG load failed ({e}); keeping current RNG state.",
            level=logging.WARNING,
        )


def _resume_grad_scaler(checkpoint_path: str, grad_scaler, log_fn: Callable) -> None:
    path = os.path.join(checkpoint_path, "grad_scaler.pt")
    if not os.path.exists(path):
        log_fn(
            "grad_scaler.pt not found; GradScaler stays fresh.",
            level=logging.WARNING,
        )
        return
    try:
        grad_scaler.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=False)
        )
        log_fn("[Checkpoint] GradScaler state restored.")
    except (RuntimeError, ValueError, TypeError, KeyError) as e:
        log_fn(
            f"GradScaler load failed ({e}); keeping fresh state.",
            level=logging.WARNING,
        )


def _load_weights_into_model(
    *,
    model: torch.nn.Module,
    resume_config: dict,
    is_fsdp2: bool = False,
    is_dmuon: bool = False,
    model_class,
    log_fn: Callable,
) -> None:
    """Weights-only load for single .safetensors / .pth sources."""
    if is_dmuon or is_fsdp2:
        src = resume_config["ckpt"]
        is_main = dist.get_rank() == 0 if dist.is_initialized() else True
        # DMuon needs the full state dict on every rank (FQN-keyed lookup;
        # see _load_fsdp2_or_dmuon_full above). FSDP2 with
        # broadcast_from_rank0=True only needs rank 0 to read.
        if is_dmuon or is_main:
            if src.endswith(".safetensors"):
                state_dict = load_file(src)
            else:
                state_dict = torch.load(src, map_location="cpu", weights_only=False)
                if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
            state_dict = _drop_checkpoint_normalizer_state(state_dict, log_fn)
        else:
            state_dict = {}
        if is_dmuon:
            import dmuon

            dmuon.set_model_state_dict(model, state_dict)
            log_fn(f"[DMuon] Loaded model weights from {src}")
        else:
            options = StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
                broadcast_from_rank0=True,
                strict=False,
            )
            set_model_state_dict(
                model,
                model_state_dict=state_dict,
                options=options,
            )
            log_fn(f"[FSDP2] Loaded model weights from {src}")
    else:
        # DDP / unwrapped.
        unwrapped_model = model.module if isinstance(model, DDP) else model
        load_weights(
            model=unwrapped_model,
            resume_config=resume_config,
            model_class=model_class,
            log_fn=log_fn,
        )


def finalize_save(log_fn: Optional[Callable] = None) -> None:
    """Barrier + GC hygiene after a checkpoint write. Trainer calls this."""
    if dist.is_initialized():
        dist.barrier()
    gc.collect()
    torch.cuda.empty_cache()
    if log_fn:
        pass  # trainer already logs the "Saved checkpoint to X" message
