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
from wall_x.trainer.fsdp_trainer import checkpoint_transaction

_module_logger = logging.getLogger(__name__)


def _noop_log(_msg: str, **_kw) -> None:
    pass


def _log(log_fn: Callable, message: str, *, level: int = logging.INFO) -> None:
    if level >= logging.WARNING:
        # log_fn callables (rank0 info/print wrappers) flatten the record level;
        # serious messages (cold-started optimizer, dropped state) must also go
        # out as real WARNING records so level-based filters/alerting see them.
        _module_logger.log(level, message)
    prefix = "WARNING: " if level >= logging.WARNING else ""
    log_fn(f"{prefix}{message}")


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
    instead of a single ``optimizer.pt``. After the FSDP1 → FSDP2
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
    epoch_complete: bool,
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
            epoch_complete=epoch_complete,
            seed=seed,
            world_size=_world_size_for_metadata(),
            normalizer_action=normalizer_action,
            normalizer_propri=normalizer_propri,
            grad_scaler=grad_scaler,
            log_fn=log_fn,
        )
    _save_rank_rng_state(ckpt_path, rank, write_legacy=is_main)

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
    epoch_complete: bool,
    seed: int,
    world_size: int,
    normalizer_action,
    normalizer_propri,
    grad_scaler=None,
    log_fn: Callable,
) -> None:
    # ``global_step`` is the number of optimizer steps fully completed.  Keep
    # the old scalar files for older readers, but make all new readers consume
    # this explicit, versioned state instead of guessing whether to add one.
    torch.save(
        {
            "format_version": 2,
            "completed_steps": int(global_step),
            "current_epoch": int(epoch),
            "epoch_complete": bool(epoch_complete),
        },
        os.path.join(ckpt_path, checkpoint_transaction.TRAINER_STATE_FILE),
    )
    torch.save({"seed": seed}, os.path.join(ckpt_path, "seed.pth"))
    torch.save({"global_step": global_step}, os.path.join(ckpt_path, "global_step.pth"))
    torch.save({"current_epoch": epoch}, os.path.join(ckpt_path, "current_epoch.pth"))

    # world_size: public metadata, used at resume to detect reshard.
    # Written unconditionally (previously sharded-mode only).
    torch.save({"world_size": world_size}, os.path.join(ckpt_path, "world_size.pth"))

    # GradScaler state (fp16 AMP only).
    if grad_scaler is not None:
        torch.save(grad_scaler.state_dict(), os.path.join(ckpt_path, "grad_scaler.pt"))

    with open(os.path.join(ckpt_path, "config.yml"), "w") as f:
        # safe_dump (no !!python/* tags) so the saved config loads with a plain
        # yaml.safe_load; sanitize first to coerce tuples/objects to safe types.
        from wall_x.config.loader import _sanitize_for_yaml

        yaml.safe_dump(
            _sanitize_for_yaml(config),
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )

    model_cfg = _dict_section(config, "model")
    data_cfg = _dict_section(config, "data")
    is_portable = bool(config.get("portable_bundle_version"))

    # Copy processor files to checkpoint directory.
    processor_dir = (
        None
        if is_portable
        else (model_cfg.get("processor_path") or config.get("processor_path"))
    )
    if processor_dir is None and not is_portable:
        # Backward compatibility: fall back to pretrained_qwen_vl_path.
        processor_dir = config.get("pretrained_qwen_vl_path")
        if processor_dir is not None:
            _log(
                log_fn,
                "'pretrained_qwen_vl_path' is deprecated for processor "
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

    act_config_path = (
        None
        if is_portable
        else (model_cfg.get("config_path") or config.get("qwen_vl_act_config_path"))
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


def _collect_rng_state() -> dict:
    rng_state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state()
    return rng_state


def _save_rank_rng_state(ckpt_path: str, rank: int, *, write_legacy: bool) -> None:
    rng_state = _collect_rng_state()
    torch.save(rng_state, os.path.join(ckpt_path, f"rng_state_rank_{rank}.pt"))
    if write_legacy:
        torch.save(rng_state, os.path.join(ckpt_path, "rng_state.pt"))


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
        model_sd = _strip_gemm_padding(model_sd, model, log_fn)
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
    optim_sd = dmuon.get_optimizer_state_dict(
        model, optimizer, cpu_offload=True, rank0_only=True
    )

    if is_main:
        model_sd = _filter_frozen_prefixes(model_sd, frozen_prefixes, log_fn)
        model_sd = _strip_gemm_padding(model_sd, model, log_fn)
        model_sd = _make_contiguous_and_clone_shared(model_sd)
        save_file(model_sd, os.path.join(ckpt_path, "model.safetensors"))
        log_fn("[Checkpoint] Saved DMuon model state dict (full, rank0)")
        torch.save(optim_sd, os.path.join(ckpt_path, "optimizer.pt"))
        torch.save(lr_scheduler.state_dict(), os.path.join(ckpt_path, "scheduler.pt"))
        log_fn("[Checkpoint] Saved DMuon optimizer and scheduler state dict (rank0)")

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
    base_model = model.module if isinstance(model, DDP) else model
    model_state = base_model.state_dict()
    model_state = _filter_frozen_prefixes(model_state, frozen_prefixes, log_fn)
    # DDP's state_dict() already triggers Qwen2_5_VLMLP._save_to_state_dict, so
    # this is a no-op here (shape-guarded/idempotent); kept for parity/safety.
    model_state = _strip_gemm_padding(model_state, base_model, log_fn)
    model_state_contiguous = _make_contiguous_and_clone_shared(model_state)
    save_file(model_state_contiguous, os.path.join(ckpt_path, "model.safetensors"))
    torch.save(optimizer.state_dict(), os.path.join(ckpt_path, "optimizer.pt"))
    torch.save(lr_scheduler.state_dict(), os.path.join(ckpt_path, "scheduler.pt"))


def _save_dataset_state(
    *, ckpt_path: str, dataset, rank: int, log_fn: Callable
) -> None:
    """Save per-rank dataset resume state if the dataset supports it."""
    if hasattr(dataset, "dataset_state_dict"):
        state = dataset.dataset_state_dict()
        if state:
            state_path = os.path.join(ckpt_path, f"dataset_state_rank_{rank}.pt")
            torch.save(state, state_path)
            log_fn(f"[Checkpoint] Saved dataset state: {state_path}")
            return

    if hasattr(dataset, "save_checkpoint_state"):
        dataset.save_checkpoint_state(ckpt_path, rank)
        return

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


def load_dataset_state(
    *,
    checkpoint_path: str,
    rank: int,
    log_fn: Optional[Callable] = None,
) -> Optional[dict]:
    """Load an opaque per-rank dataset resume payload.

    Compatibility and fingerprint checks belong to the dataset library and
    run when Wall-X gives this object back to its checkpoint controller.
    """
    log_fn = log_fn or _noop_log
    state_path = os.path.join(checkpoint_path, f"dataset_state_rank_{rank}.pt")
    if not os.path.exists(state_path):
        return None

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    log_fn(f"[Checkpoint] Loaded dataset state: {state_path}")
    return state


def _legacy_step_checkpoint(checkpoint_path: str, current_epoch: int) -> bool:
    """Classify a legacy checkpoint from Wall-X's canonical directory name.

    Legacy epoch checkpoints were written as ``<epoch>`` and step checkpoints
    as ``<epoch>_<step>``. Dataset cursor artifacts are deliberately not used
    here: some backends could not save them, but the model/optimizer counters
    still have step-checkpoint semantics.
    """

    # Resolve a conventional ``latest`` symlink before applying the naming
    # contract; copied/renamed legacy directories remain intentionally
    # ambiguous and are rejected rather than guessed.
    name = os.path.basename(os.path.normpath(os.path.realpath(checkpoint_path)))
    if name == str(current_epoch):
        return False
    prefix = f"{current_epoch}_"
    if name.startswith(prefix):
        step = name[len(prefix) :]
        if step.isdigit() and int(step) > 0:
            return True
    raise ValueError(
        "Cannot classify legacy checkpoint as epoch- or step-level from its "
        f"directory name {name!r}; expected {current_epoch!r} or "
        f"'{current_epoch}_<positive-step>'"
    )


def validate_resume_data_cursor(
    trainer_state: dict, *, has_data_cursor: bool
) -> None:
    """Reject an incomplete checkpoint that cannot restore data progress."""

    if not bool(trainer_state["epoch_complete"]) and not has_data_cursor:
        raise RuntimeError(
            "Cannot resume an incomplete-epoch checkpoint without a resumable "
            "per-rank dataset cursor. Restart from an epoch checkpoint or use "
            "a data backend that implements checkpoint state."
        )


def load_trainer_state(checkpoint_path: str) -> dict:
    """Load completed-step/epoch state, translating old checkpoints once.

    Pre-v2 mid-epoch checkpoints wrote the zero-based step being saved and
    advanced it after saving, while epoch-boundary checkpoints already held
    the completed count. Legacy step/epoch kind is derived from the checkpoint
    directory naming contract, independently of dataset cursor availability.
    The v2 format records both meanings directly.
    """
    checkpoint_transaction.validate_for_resume(checkpoint_path)
    state_path = os.path.join(
        checkpoint_path, checkpoint_transaction.TRAINER_STATE_FILE
    )
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if int(state.get("format_version", -1)) != 2:
            raise ValueError(
                "Unsupported trainer checkpoint format_version: "
                f"{state.get('format_version')!r}"
            )
        completed_steps = int(state["completed_steps"])
        current_epoch = int(state["current_epoch"])
        epoch_complete = bool(state["epoch_complete"])
        if completed_steps < 0 or current_epoch < 0:
            raise ValueError("Trainer checkpoint counters must be non-negative")
        return {
            "global_step": completed_steps,
            "start_epoch": current_epoch + int(epoch_complete),
            "epoch_complete": epoch_complete,
            "format_version": 2,
        }

    # Legacy directory: accept it without a completion marker and normalize
    # its two ambiguous scalar files into completed-step semantics.
    global_step = 0
    global_step_path = os.path.join(checkpoint_path, "global_step.pth")
    if os.path.exists(global_step_path):
        stored_step = int(
            torch.load(global_step_path, map_location="cpu", weights_only=False)[
                "global_step"
            ]
        )
        global_step = stored_step

    start_epoch = 0
    current_epoch_path = os.path.join(checkpoint_path, "current_epoch.pth")
    if os.path.exists(current_epoch_path):
        current_epoch = int(
            torch.load(current_epoch_path, map_location="cpu", weights_only=False)[
                "current_epoch"
            ]
        )
        legacy_step_checkpoint = _legacy_step_checkpoint(
            checkpoint_path, current_epoch
        )
        start_epoch = current_epoch + int(not legacy_step_checkpoint)
    else:
        raise ValueError(
            "Legacy checkpoint is missing current_epoch.pth and cannot be "
            "classified as an epoch or step checkpoint"
        )

    # Old step checkpoints saved update N+1 under the zero-based scalar N.
    global_step += int(legacy_step_checkpoint)

    return {
        "global_step": global_step,
        "start_epoch": start_epoch,
        "epoch_complete": not legacy_step_checkpoint,
        "format_version": 1,
    }


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


def _strip_gemm_padding(
    state_dict: dict, model: torch.nn.Module, log_fn: Callable = _noop_log
) -> dict:
    """Slice GEMM-alignment padding out of a gathered full state dict in place.

    Modules like ``Qwen2_5_VLMLP`` pad ``intermediate_size`` to a multiple of
    128 (e.g. 3420 -> 3456) for cuBLAS efficiency and rely on their own
    ``_save_to_state_dict`` hook to strip the padding on save. That hook only
    fires for ``nn.Module.state_dict()`` (the DDP path); the FSDP2
    (``get_state_dict``) and DMuon (``get_model_state_dict``) paths gather
    DTensors directly and bypass it, so the padded rows leak into the saved
    checkpoint. This restores parity by stripping post-gather.

    The strip spec is derived from the model itself: any module exposing the
    padding contract (``padded_intermediate_size`` != ``intermediate_size``).
    This targets only the padded class and naturally excludes the non-padded
    LLM ``Qwen2MLP`` and Qwen3.5 ``Qwen3_5VisionMLP``, which share the
    ``gate_up_proj``/``down_proj`` names but carry no padding. The slice is
    shape-guarded (only acts when a dim still equals the padded size), so it is
    idempotent and safe to run even after the DDP hook already stripped.
    """
    if not state_dict:
        return state_dict

    def _resolve_key(suffix: str) -> Optional[str]:
        if suffix in state_dict:
            return suffix
        # Tolerate a wrapper prefix on the gathered FQN (defensive; FSDP2/DMuon
        # produce clean keys, so this rarely fires).
        matches = [k for k in state_dict if k.endswith("." + suffix)]
        return matches[0] if len(matches) == 1 else None

    stripped = 0
    for module_name, module in model.named_modules():
        orig = getattr(module, "intermediate_size", None)
        padded = getattr(module, "padded_intermediate_size", None)
        if orig is None or padded is None or orig == padded:
            continue

        # gate_up_proj is a fused [gate; up] concat along dim 0; split, drop each
        # half's padding rows, then re-concat. weight: [2*padded, h] -> [2*orig, h];
        # bias: [2*padded] -> [2*orig].
        for sub in ("gate_up_proj.weight", "gate_up_proj.bias"):
            key = _resolve_key(f"{module_name}.{sub}")
            if key is None:
                continue
            t = state_dict[key]
            if t.shape[0] == 2 * padded:
                gate, up = t.split([padded, padded], dim=0)
                state_dict[key] = torch.cat([gate[:orig], up[:orig]], dim=0)
                stripped += 1
            elif t.shape[0] != 2 * orig:
                log_fn(
                    f"[Checkpoint] WARNING: unexpected shape for '{key}': "
                    f"{tuple(t.shape)} (orig={orig}, padded={padded}); not stripped."
                )

        # down_proj.weight: [h, padded] -> [h, orig].
        key = _resolve_key(f"{module_name}.down_proj.weight")
        if key is not None:
            t = state_dict[key]
            if t.shape[1] == padded:
                state_dict[key] = t[:, :orig]
                stripped += 1
            elif t.shape[1] != orig:
                log_fn(
                    f"[Checkpoint] WARNING: unexpected shape for '{key}': "
                    f"{tuple(t.shape)} (orig={orig}, padded={padded}); not stripped."
                )

    if stripped:
        log_fn(f"[Checkpoint] Stripped GEMM padding from {stripped} tensor(s).")
    return state_dict


def _world_size_for_metadata() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------


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
    passed_through_unknown = 0
    for name, param in state_dict.items():
        if name not in model_sd:
            # Pass unknown keys through instead of dropping them: modules with
            # name-changing _load_from_state_dict hooks (e.g. qwen3_5's fused
            # qkv_proj/gate_up_proj/in_proj_zba consuming legacy split
            # q_proj/gate_proj/... keys) only see the checkpoint tensor if it
            # survives to load_state_dict. Dropping here would leave those
            # fused projections at random init while strict=False training
            # continues silently. Truly-unused keys are still tolerated by
            # strict=False downstream.
            log_fn(
                f"Key not in model state_dict (passing through for "
                f"_load_from_state_dict hooks / strict=False): {name}"
            )
            out[name] = param
            passed_through_unknown += 1
            continue
        # action_preprocessor/action_processor: action-dim changes (pad only).
        # embed_tokens/lm_head: vocab grow/shrink (append/remove-at-end), so the
        # same first-N slice handles both. These are bare nn.Embedding/nn.Linear
        # with no _load_from_state_dict hook, so a mismatch here would otherwise
        # crash load_state_dict(strict=False) (size mismatch still raises).
        if (
            "action_preprocessor" in name
            or "action_processor" in name
            or "embed_tokens" in name
            or "lm_head" in name
        ):
            if param.size() == model_sd[name].size():
                out[name] = param
                continue
            size_0 = param.size()
            size_1 = model_sd[name].size()
            # Action dims are concrete joints/channels and are NOT prefix-aligned,
            # so a larger action checkpoint must be padded explicitly, not sliced
            # (blind slicing would mismatch joints). Vocab is append/remove-at-end,
            # so a larger vocab checkpoint is safely clipped to the model's first
            # rows by the slice logic below.
            is_action = "action_preprocessor" in name or "action_processor" in name
            if is_action and any(
                old_dim > new_dim for old_dim, new_dim in zip(size_0, size_1)
            ):
                raise ValueError(
                    f"Shape mismatch for '{name}': checkpoint shape {tuple(size_0)} is "
                    f"larger than model shape {tuple(size_1)} in at least one dimension. "
                    "Loading a larger checkpoint into a smaller model is not supported here. "
                    "If the action dimension has changed, please configure action padding "
                    "in the dataset processor so checkpoint actions match the new action "
                    "size."
                )
            # Keep model init for the new dims of: normalizer buffers (delta must
            # stay 1.0, else div-by-zero) and vocab embeddings (proper resize init
            # — a zero embedding row is not meaningful). Zero-init projection
            # weights to avoid polluting existing hidden features with noise.
            keep_model_init = (
                "normalizer" in name or "embed_tokens" in name or "lm_head" in name
            )
            if keep_model_init:
                out[name] = model_sd[name].clone()
            else:
                out[name] = torch.zeros_like(model_sd[name])
            slices = [
                slice(0, min(old_dim, new_dim))
                for old_dim, new_dim in zip(size_0, size_1)
            ]
            out[name][slices] = param[slices]
            clipped = any(old_dim > new_dim for old_dim, new_dim in zip(size_0, size_1))
            init_desc = "model init" if keep_model_init else "zero-initialized"
            if clipped:
                log_fn(
                    f"Not match key: {name}, checkpoint {tuple(size_0)} -> model "
                    f"{tuple(size_1)}. Clipped checkpoint to first "
                    f"{[s.stop for s in slices]} dims (rows beyond dropped)."
                )
            else:
                log_fn(
                    f"Not match key: {name}, checkpoint {tuple(size_0)} -> model "
                    f"{tuple(size_1)}. Filled first {[s.stop for s in slices]} dims, "
                    f"remaining dims {init_desc}."
                )
        else:
            # Always pass through, even on shape mismatch.
            # Modules with a custom _load_from_state_dict hook (e.g.
            # Qwen2_5_VLMLP pads ViT MLP 3420->3456 for cuBLAS alignment)
            # must receive the original checkpoint tensor to do their own
            # adaptation.  Skipping here would bypass that hook entirely.
            out[name] = param
            if param.size() != model_sd[name].size():
                log_fn(
                    f"Shape mismatch '{name}': checkpoint {tuple(param.size())} "
                    f"vs model {tuple(model_sd[name].size())}. "
                    f"Passing through to _load_from_state_dict hook."
                )
    if passed_through_unknown:
        log_fn(
            f"Passed through {passed_through_unknown} key(s) not in model "
            f"state_dict (for _load_from_state_dict hooks / strict=False)"
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
    resume_optimizer_state = resume_config.get("resume_optimizer_state", True)

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
    checkpoint_transaction.validate_for_resume(checkpoint_path)

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
                resume_optimizer_state=resume_optimizer_state,
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
                resume_optimizer_state=resume_optimizer_state,
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
        _log(
            log_fn,
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


def _distributed_int_sum_batch(*values: int) -> list:
    """Sum several ints across ranks in ONE all_reduce (fewer collective rounds)."""
    if not dist.is_initialized():
        return [int(v) for v in values]
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    tensor = torch.tensor([int(v) for v in values], dtype=torch.int64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.tolist()


def _format_name_preview(names, limit: int = 8) -> str:
    names = sorted(str(name) for name in names)
    if not names:
        return "[]"
    suffix = "" if len(names) <= limit else f", ... (+{len(names) - limit})"
    return "[" + ", ".join(names[:limit]) + suffix + "]"


def _log_dmuon_optimizer_resume_diagnostics(
    *,
    model,
    optimizer,
    optim_sd: dict,
    log_fn: Callable,
) -> None:
    """Audit DMuon optimizer restore coverage without changing optimizer state."""
    try:
        from dmuon.checkpoint import (
            _compute_dedicated_fqns,
            _owns_dedicated_adamw_optimizer_state,
            _owns_dedicated_optimizer_state,
        )
    except Exception as exc:
        _log(
            log_fn,
            f"[Checkpoint][DMuon] Resume diagnostics unavailable: {exc!r}",
            level=logging.WARNING,
        )
        return

    rank = dist.get_rank() if dist.is_initialized() else 0
    is_main = rank == 0

    dp_fqns = _compute_dedicated_fqns(model)
    dedicated_fqn_set = set(dp_fqns.values())
    dp_to_muon_group = getattr(optimizer, "_dp_to_muon_group_idx", {}) or {}
    dp_to_adamw_group = getattr(optimizer, "_dp_to_adamw_group_idx", {}) or {}

    current_muon_fqns = {
        fqn for dp, fqn in dp_fqns.items() if id(dp) in dp_to_muon_group
    }
    current_dedicated_adamw_fqns = {
        fqn for dp, fqn in dp_fqns.items() if id(dp) in dp_to_adamw_group
    }
    current_fsdp_fqns = {
        name
        for name, _param in model.named_parameters()
        if name not in dedicated_fqn_set
    }

    saved_dedicated_fqns = set((optim_sd.get("dedicated") or {}).keys())
    saved_dedicated_adamw_fqns = set((optim_sd.get("dedicated_adamw") or {}).keys())
    saved_fsdp_fqns = set((optim_sd.get("fsdp") or {}).keys())

    fqn_to_dp = {fqn: dp for dp, fqn in dp_fqns.items()}
    local_muon_expected = 0
    local_muon_loaded = 0
    local_muon_missing = []
    for fqn in current_muon_fqns & saved_dedicated_fqns:
        dp = fqn_to_dp[fqn]
        if not _owns_dedicated_optimizer_state(dp):
            continue
        local_muon_expected += 1
        state = optimizer.state.get(id(dp), {})
        if "momentum_buffer" in state:
            local_muon_loaded += 1
        else:
            local_muon_missing.append(fqn)

    local_dedicated_adamw_expected = 0
    local_dedicated_adamw_loaded = 0
    local_dedicated_adamw_missing = []
    for fqn in current_dedicated_adamw_fqns & saved_dedicated_adamw_fqns:
        dp = fqn_to_dp[fqn]
        if not _owns_dedicated_adamw_optimizer_state(dp):
            continue
        local_dedicated_adamw_expected += 1
        state = optimizer.state.get(id(dp), {})
        if state:
            local_dedicated_adamw_loaded += 1
        else:
            local_dedicated_adamw_missing.append(fqn)

    fsdp_param_by_fqn = {
        name: param
        for name, param in model.named_parameters()
        if name not in dedicated_fqn_set
    }
    local_fsdp_expected = len(current_fsdp_fqns & saved_fsdp_fqns)
    local_fsdp_loaded = 0
    local_fsdp_missing = []
    for fqn in current_fsdp_fqns & saved_fsdp_fqns:
        param = fsdp_param_by_fqn[fqn]
        state = optimizer.state.get(param, {})
        if state:
            local_fsdp_loaded += 1
        else:
            local_fsdp_missing.append(fqn)

    # One batched all_reduce instead of six separate collectives.
    (
        global_muon_expected,
        global_muon_loaded,
        global_dedicated_adamw_expected,
        global_dedicated_adamw_loaded,
        global_fsdp_expected,
        global_fsdp_loaded,
    ) = _distributed_int_sum_batch(
        local_muon_expected,
        local_muon_loaded,
        local_dedicated_adamw_expected,
        local_dedicated_adamw_loaded,
        local_fsdp_expected,
        local_fsdp_loaded,
    )

    if is_main:
        log_fn(
            "[Checkpoint][DMuon] optimizer resume coverage: "
            f"muon_saved={len(saved_dedicated_fqns)} "
            f"muon_current={len(current_muon_fqns)} "
            f"muon_owner_loaded={global_muon_loaded}/{global_muon_expected}; "
            f"dedicated_adamw_saved={len(saved_dedicated_adamw_fqns)} "
            f"dedicated_adamw_current={len(current_dedicated_adamw_fqns)} "
            f"dedicated_adamw_owner_loaded={global_dedicated_adamw_loaded}/"
            f"{global_dedicated_adamw_expected}; "
            f"fsdp_saved={len(saved_fsdp_fqns)} "
            f"fsdp_current={len(current_fsdp_fqns)} "
            f"fsdp_local_loaded_sum={global_fsdp_loaded}/{global_fsdp_expected}"
        )
        log_fn(
            "[Checkpoint][DMuon] optimizer resume FQN diff: "
            f"muon_missing_current={_format_name_preview(current_muon_fqns - saved_dedicated_fqns)} "
            f"muon_unexpected_saved={_format_name_preview(saved_dedicated_fqns - current_muon_fqns)} "
            f"dedicated_adamw_missing_current="
            f"{_format_name_preview(current_dedicated_adamw_fqns - saved_dedicated_adamw_fqns)} "
            f"dedicated_adamw_unexpected_saved="
            f"{_format_name_preview(saved_dedicated_adamw_fqns - current_dedicated_adamw_fqns)} "
            f"fsdp_missing_current={_format_name_preview(current_fsdp_fqns - saved_fsdp_fqns)} "
            f"fsdp_unexpected_saved={_format_name_preview(saved_fsdp_fqns - current_fsdp_fqns)}"
        )

    if local_muon_missing or local_dedicated_adamw_missing or local_fsdp_missing:
        _log(
            log_fn,
            "[Checkpoint][DMuon] optimizer resume local missing state: "
            f"rank={rank} "
            f"muon={_format_name_preview(local_muon_missing)} "
            f"dedicated_adamw={_format_name_preview(local_dedicated_adamw_missing)} "
            f"fsdp={_format_name_preview(local_fsdp_missing)}",
            level=logging.WARNING,
        )


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
    resume_optimizer_state: bool = True,
    log_fn: Callable,
) -> None:
    """Load FSDP2 (or DMuon) model + optimizer from a full-state-dict ckpt.

    FSDP2 path: rank 0 deserializes, ``broadcast_from_rank0=True`` in
    ``set_*_state_dict`` reshards to the current mesh.

    DMuon path: every rank MUST load the full state dict from disk.
    ``dmuon.set_model_state_dict`` / ``set_optimizer_state_dict`` iterate
    ``fqn_to_dp`` and skip any FQN not present in the provided dict — so
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
    if not resume_optimizer_state:
        log_fn(
            "[Checkpoint] Optimizer and scheduler state resume disabled; "
            "both will start fresh."
        )
    elif os.path.exists(optim_path):
        if is_dmuon:
            # Same rationale as model_sd: every rank needs full optim state.
            optim_sd = torch.load(optim_path, map_location="cpu", weights_only=False)
            _strip_fused_flags(optim_sd)
        elif is_main:
            # FSDP2 optimizer restore reshards the full optimizer state via the
            # distributed backend. Loading rank-0 state on CUDA avoids a CPU
            # DTensor/backend path that fails under NCCL-only process groups.
            map_location = (
                torch.device("cuda", torch.cuda.current_device())
                if torch.cuda.is_available()
                else "cpu"
            )
            optim_sd = torch.load(
                optim_path, map_location=map_location, weights_only=False
            )
            _strip_fused_flags(optim_sd)
        else:
            optim_sd = {}

        try:
            if is_dmuon:
                import dmuon

                dmuon.set_optimizer_state_dict(model, optimizer, optim_sd)
                log_fn("[Checkpoint] DMuon optimizer state loaded.")
                _log_dmuon_optimizer_resume_diagnostics(
                    model=model,
                    optimizer=optimizer,
                    optim_sd=optim_sd,
                    log_fn=log_fn,
                )
            else:
                options = StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=False,
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
            # CUDA OOM is not a layout incompatibility: rank 0 materializes the
            # full unsharded optimizer state on GPU here, and swallowing an OOM
            # would silently cold-start momentum mid-run. Fail loudly instead.
            if isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
                raise
            # Most likely a legacy FSDP1 consolidated OSD (flat_param-specific).
            _log(
                log_fn,
                f"[Checkpoint] Optimizer state incompatible with current "
                f"layout ({e!r}); cold-starting optimizer. Model weights are "
                f"loaded; momentum resets to zero.",
                level=logging.WARNING,
            )
        del optim_sd
        gc.collect()
    else:
        _log(
            log_fn,
            "[Checkpoint] optimizer.pt not found; cold-starting optimizer.",
            level=logging.WARNING,
        )

    # --- scheduler ---
    if resume_optimizer_state and os.path.exists(sched_path):
        try:
            lr_scheduler.load_state_dict(
                torch.load(sched_path, map_location="cpu", weights_only=False)
            )
            log_fn("[Checkpoint] Scheduler state loaded.")
        except (ValueError, RuntimeError, TypeError, KeyError) as e:
            _log(
                log_fn,
                f"Scheduler load failed ({e}); keeping fresh state.",
                level=logging.WARNING,
            )


def _resume_ddp_optimizer_scheduler(
    *,
    checkpoint_path: str,
    optimizer,
    lr_scheduler,
    resume_optimizer_state: bool = True,
    log_fn: Callable,
) -> None:
    """Restore optimizer / scheduler for DDP layouts (single-file)."""
    if not resume_optimizer_state:
        log_fn(
            "[Checkpoint] Optimizer and scheduler state resume disabled; "
            "both will start fresh."
        )
        return

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
            _log(
                log_fn,
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
            _log(
                log_fn,
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
    _log(
        log_fn,
        f"Legacy FSDP1 ckpt detected at {checkpoint_path}: model weights "
        f"loaded, but per-rank optimizer files cannot reshard into FSDP2. "
        f"Optimizer state DROPPED — it will warm up from scratch. For long "
        f"resume runs prefer a fresh FSDP2 ckpt; for short runs the loss "
        f"bump is usually negligible.",
        level=logging.WARNING,
    )


def _resume_rng(checkpoint_path: str, log_fn: Callable) -> None:
    rank = dist.get_rank() if dist.is_initialized() else 0
    rank_path = os.path.join(checkpoint_path, f"rng_state_rank_{rank}.pt")
    legacy_path = os.path.join(checkpoint_path, "rng_state.pt")
    path = rank_path if os.path.exists(rank_path) else legacy_path
    if not os.path.exists(path):
        _log(
            log_fn,
            f"{os.path.basename(path)} not found; RNG stays at seed_all() state.",
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
        log_fn(f"[Checkpoint] RNG state restored from {os.path.basename(path)}.")
    except (RuntimeError, ValueError, TypeError, KeyError) as e:
        _log(
            log_fn,
            f"RNG load failed ({e}); keeping current RNG state.",
            level=logging.WARNING,
        )


def _resume_grad_scaler(checkpoint_path: str, grad_scaler, log_fn: Callable) -> None:
    path = os.path.join(checkpoint_path, "grad_scaler.pt")
    if not os.path.exists(path):
        _log(
            log_fn,
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
        _log(
            log_fn,
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
