import hashlib
import json
import logging
import math
import os
import random
import threading

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoProcessor

from wall_x.model.core.action.normalizer import Normalizer
from wall_x.model.qact.tokenizer_mixin import get_action_tokenizer_mixin
from wall_x.utils.metrics import get_action_accuracy, dtw_distance, frechet_distance

_logger = logging.getLogger(__name__)

_WALL_VLM_RVQ_TOKENIZERS = {
    "wall_vlm_rvq_v3_2",
    "wall_vlm_rvq_v3_3_ar",
}


def _stamp_rvq_token_manifest(processor, action_mapper):
    """Record the exact RVQ token-to-ID contract for bootstrap and strict runs."""
    geometry = action_mapper["rvq_tokenizer_geometry"]
    tokens = ["<rvq_start>"] + [
        f"<rvq_r{q}_{idx}>"
        for q in range(int(geometry["num_quantizers"]))
        for idx in range(int(geometry["codebook_size"]))
    ]
    ids = [int(processor.tokenizer.convert_tokens_to_ids(token)) for token in tokens]
    if len(ids) != len(set(ids)):
        raise ValueError("RVQ manifest tokens must map to distinct token IDs")
    manifest = "\n".join(f"{token}\t{token_id}" for token, token_id in zip(tokens, ids))
    manifest_hash = hashlib.sha256(manifest.encode("utf-8")).hexdigest()
    action_mapper["rvq_token_id_manifest_hash"] = manifest_hash
    return manifest_hash


def _validate_wall_vlm_checkpoint_contract(processor_path, processor, action_mapper):
    from wall_x.model.qact.rvq_position import validate_rvq_tokenizer_geometry

    config_path = os.path.join(os.fspath(processor_path), "config.json")
    if not os.path.isfile(config_path):
        raise ValueError(
            "wall_vlm_rvq_v3_2 requires the Wall-VLM checkpoint config at "
            f"{config_path}"
        )
    with open(config_path) as stream:
        saved = json.load(stream)
    if saved.get("rvq_position_encoding") != "time_level_2d":
        raise ValueError(
            "Wall-VLM checkpoint must declare " "rvq_position_encoding='time_level_2d'"
        )
    geometry = action_mapper["rvq_tokenizer_geometry"]
    validate_rvq_tokenizer_geometry(
        saved.get("rvq_tokenizer_geometry"),
        geometry,
        context="the Wall-VLM checkpoint",
    )
    current_manifest_hash = _stamp_rvq_token_manifest(processor, action_mapper)
    saved_manifest_hash = saved.get("rvq_token_id_manifest_hash")
    if saved_manifest_hash is None:
        raise ValueError(
            "Wall-VLM checkpoint has no rvq_token_id_manifest_hash metadata"
        )
    if saved_manifest_hash != current_manifest_hash:
        raise ValueError(
            "Wall-VLM checkpoint RVQ token IDs differ from the current tokenizer: "
            f"saved={saved_manifest_hash}, current={current_manifest_hash}"
        )


def load_wallx_processors(
    config,
    normalizer=None,
    action_statistic_dof=None,
    device: str = "cpu",
):
    """
    Load Wall-X processors, including tokenizer and action mapper.

    Args:
        config: Configuration dictionary.
        device: Tokenizer device. Training usually uses "cpu"; inference
            usually uses "cuda".

    Returns:
        Dictionary containing:
            - processor: HuggingFace processor
            - train_action_tokenizer: action tokenizer for training
            - val_action_tokenizer: action tokenizer for validation
            - action_mapper: action mapper dictionary
            - num_added_tokens: number of added tokens
            - tokenizer_mixin: ActionTokenizerMixin instance
    """
    processor_path = config["processor_path"]
    from wall_x.model.qact.qwen3_5.dino_processor import (
        is_dino_processor_checkpoint,
        load_dino_qwen35_processor,
    )

    if is_dino_processor_checkpoint(processor_path):
        processor = load_dino_qwen35_processor(processor_path)
    else:
        processor = AutoProcessor.from_pretrained(processor_path, use_fast=True)
    # pad side = left
    processor.tokenizer.padding_side = "left"

    new_tokens = ["<|propri|>", "<|action|>"]
    if config.get("new_special_tokens", None) is not None:
        new_tokens.extend(config.get("new_special_tokens"))

    action_tokenizer_type = config.get("action_tokenizer_type", None)

    train_action_tokenizer = None
    val_action_tokenizer = None
    action_mapper = None
    tokenizer_mixin = None

    if action_tokenizer_type:
        # Use tokenizer_mixin as the single action-tokenizer interface.
        action_tokenizer_config = config.get("action_tokenizer", {})
        # Backward compatibility: read top-level keys as fallback values.
        action_tokenizer_config.setdefault(
            "action_tokenizer_path", config.get("action_tokenizer_path")
        )
        action_tokenizer_config.setdefault(
            "action_tokenizer_checkpoint_path",
            config.get("action_tokenizer_checkpoint_path"),
        )
        action_tokenizer_config.setdefault(
            "action_tokenizer_config_dir", config.get("action_tokenizer_config_dir")
        )
        action_tokenizer_config.setdefault(
            "rvq_token_layout", config.get("rvq_token_layout", "layer_major")
        )
        action_tokenizer_config.setdefault(
            "rvq_position_encoding",
            config.get("rvq_position_encoding", "time_level_2d"),
        )
        action_tokenizer_config.setdefault(
            "rvq_checkpoint_contract",
            config.get("rvq_checkpoint_contract", "strict"),
        )
        # Pass action_horizon_ar to the tokenizer for DLLM.
        data_config = config.get("data", {})
        action_tokenizer_config.setdefault(
            "action_horizon_ar",
            config.get("rvq_action_horizon")
            or data_config.get("action_horizon_ar", 32),
        )
        # Fall back to dof_config when ar_dof_config is not provided.
        ar_dof_config = config.get("ar_dof_config") or config.get("dof_config")
        assert ar_dof_config is not None, "Missing ar_dof_config and dof_config"
        # v3.3-ar mixin needs wall-x layout keys (not tokenizer profile roles/forms).
        action_tokenizer_config.setdefault("ar_dof_config", ar_dof_config)
        action_tokenizer_config.setdefault(
            "agent_pos_config", config.get("agent_pos_config")
        )
        if normalizer is None:
            if action_statistic_dof is None:
                raise ValueError(
                    "Action tokenizer setup requires an explicit normalizer or "
                    "action statistics. Public Wall-X builds do not bundle "
                    "default action statistics."
                )
            ar_normalizer = Normalizer(action_statistic_dof, ar_dof_config)
        else:
            ar_normalizer = normalizer
        tokenizer_mixin = get_action_tokenizer_mixin(action_tokenizer_type)
        tokenizer_mixin.load_tokenizer(
            action_tokenizer_config, ar_normalizer, device=device
        )

        # Collect special tokens.
        _new_tokens, special_tokens = tokenizer_mixin.get_all_special_tokens()
        if action_tokenizer_type in _WALL_VLM_RVQ_TOKENIZERS:
            if config.get("rvq_token_protocol") != "wall_vlm_start_v1":
                raise ValueError(
                    f"{action_tokenizer_type} requires "
                    "rvq_token_protocol='wall_vlm_start_v1'"
                )
            known_vocab = processor.tokenizer.get_vocab()
            missing = [token for token in _new_tokens if token not in known_vocab]
            checkpoint_contract = config.get("rvq_checkpoint_contract", "strict")
            if checkpoint_contract == "strict" and missing:
                raise ValueError(
                    "Wall-VLM processor/tokenizer is missing pretrained RVQ tokens: "
                    f"{missing[:8]} (missing={len(missing)}). Set processor_path to "
                    "the Wall-VLM checkpoint processor."
                )
            if checkpoint_contract == "bootstrap" and not missing:
                _logger.warning(
                    "RVQ bootstrap was requested, but the processor already contains "
                    "all action tokens; use rvq_checkpoint_contract='strict' when "
                    "continuing an action-trained checkpoint"
                )
        new_tokens += _new_tokens

        # Add tokens to the vocabulary.
        num_added_tokens = processor.tokenizer.add_tokens(new_tokens)

        # Set placeholder_seq for discrete diffusion.
        if special_tokens and action_tokenizer_config.get(
            "input_placeholder_flag", False
        ):
            processor.placeholder_seq = [
                processor.tokenizer.convert_tokens_to_ids(token)
                for token in special_tokens
            ]

        # Backward compatibility: use the first added token when
        # <|action_token_0|> does not exist.
        ar_first_token_id = processor.tokenizer.convert_tokens_to_ids(
            "<|action_token_0|>"
        )
        if (
            ar_first_token_id is None
            or ar_first_token_id == processor.tokenizer.unk_token_id
        ):
            ar_first_token_id = processor.tokenizer.convert_tokens_to_ids(
                _new_tokens[0]
            )
        processor.ar_first_token = ar_first_token_id

        # Build action_mapper.
        action_mapper = tokenizer_mixin.build_action_mapper(processor)
        if action_tokenizer_type in _WALL_VLM_RVQ_TOKENIZERS:
            # Bootstrap checkpoints have no saved RVQ metadata to validate, but
            # the newly-created vocabulary still needs a stable manifest hash.
            _stamp_rvq_token_manifest(processor, action_mapper)
        if (
            action_tokenizer_type in _WALL_VLM_RVQ_TOKENIZERS
            and config.get("rvq_checkpoint_contract", "strict") == "strict"
        ):
            _validate_wall_vlm_checkpoint_contract(
                processor_path, processor, action_mapper
            )

        # Fetch the underlying tokenizer.
        train_action_tokenizer = tokenizer_mixin.tokenizer
        # Fast tokenizers need a separate validation instance; others share one.
        val_action_tokenizer = tokenizer_mixin.get_val_tokenizer(config)
    else:
        num_added_tokens = processor.tokenizer.add_tokens(new_tokens)

    return {
        "processor": processor,
        "train_action_tokenizer": train_action_tokenizer,
        "val_action_tokenizer": val_action_tokenizer,
        "action_mapper": action_mapper,
        "num_added_tokens": num_added_tokens,
        "tokenizer_mixin": tokenizer_mixin,
    }


def load_wallx_processors_from_cfg(
    cfg,
    normalizer=None,
    action_statistic_dof=None,
    device: str = "cpu",
):
    """Typed convenience wrapper around ``load_wallx_processors``.

    Builds the flat dict that the legacy function expects from typed
    TrainConfig sub-configs, then delegates. Callers using TrainConfig
    can use this directly instead of hand-flattening.
    """
    import dataclasses

    flat = dataclasses.asdict(cfg.model)
    flat["model_type"] = cfg.model_type
    flat["data"] = dict(cfg._raw_data or {})
    flat["dof_config"] = cfg.task.dof_config
    flat["agent_pos_config"] = cfg.task.agent_pos_config
    if cfg.task.ar_dof_config is not None:
        flat["ar_dof_config"] = cfg.task.ar_dof_config
    flat["batch_size_per_gpu"] = cfg.hyperparams.batch_size_per_gpu
    return load_wallx_processors(
        flat,
        normalizer=normalizer,
        action_statistic_dof=action_statistic_dof,
        device=device,
    )


def load_qwen_pretrain_weight(model, pretrain_weight_path):
    weight_files = sorted(
        [f for f in os.listdir(pretrain_weight_path) if f.endswith(".safetensors")]
    )
    # Initialize empty dictionary to store merged weights
    merged_weights = {}

    # Load and merge each file sequentially
    for weight_file in weight_files:
        file_path = os.path.join(pretrain_weight_path, weight_file)
        weights = load_file(file_path)
        merged_weights.update(weights)

    renamed_weights = model.rename_vlm_weights_for_vla(merged_weights)
    renamed_weights = {
        k: v
        for k, v in renamed_weights.items()
        if "action_preprocessor.normalizer_" not in k
    }  # remove normalizer weights
    if (
        model.config.model_type == "qwen2_5_vl"
        and model.model.embed_tokens.weight.shape[0]
        != renamed_weights["model.embed_tokens.weight"].shape[0]
    ):
        _logger.info(
            "resize_token_embeddings from %d to %d",
            model.model.embed_tokens.weight.shape[0],
            renamed_weights["model.embed_tokens.weight"].shape[0],
        )
        model.model.resize_token_embeddings(
            renamed_weights["model.embed_tokens.weight"].shape[0]
        )

    err = model.load_state_dict(renamed_weights, strict=False)

    return model, err


def update_model_config(train_config, model_config):
    model_config.use_state_string_representation = train_config["data"].get(
        "use_state_string_representation", False
    )
    model_config.ar_loss_weight = train_config.get("ar_loss_weight", 1.0)

    model_config.dof_config = train_config["dof_config"]
    model_config.agent_pos_config = train_config["agent_pos_config"]

    model_config.action_horizon_flow = train_config["data"].get(
        "action_horizon_flow", 32
    )

    attn_impl = train_config.get(
        "_attn_implementation", train_config.get("attn_implementation")
    )
    if attn_impl is not None:
        model_config._attn_implementation = attn_impl

    # attn_deterministic is no longer a standalone knob — it is derived from the
    # single ``hyperparams.deterministic_algorithms`` switch so "turn determinism
    # on" enables all paths (incl. attention kernels) at once (MR !556 review).
    _det = train_config.get("hyperparams", {}).get("deterministic_algorithms", False)
    model_config.attn_deterministic = _det
    model_config.vision_config.attn_deterministic = _det
    if _det:
        _logger.info("Attention is using deterministic kernel for this run")

    if train_config.get("noise_scheduler", None) is not None:
        model_config.noise_scheduler = train_config["noise_scheduler"]

    return model_config


def update_data_config(config):
    """Keep the top-level model type aligned with the nested data config."""
    config["data"]["model_type"] = config.get("model_type")

    if config.get("use_state_string_representation", None) is not None:
        config["data"]["use_state_string_representation"] = config[
            "use_state_string_representation"
        ]

    return config


def get_detailed_memory_usage():
    """Return process memory and thread-count diagnostics."""
    process = psutil.Process()
    memory_info = process.memory_info()
    current_threads = threading.active_count()
    return {
        "rss": f"{memory_info.rss / 1024 / 1024:.2f}MB ",
        "vms": f"{memory_info.vms / 1024 / 1024:.2f}MB ",
        "threads_count": current_threads,
    }


def is_last_rank():
    return torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1)


def print_rank_last(message):
    """If distributed is initialized, log only on last rank."""
    if torch.distributed.is_initialized():
        if is_last_rank():
            _logger.info(message)
    else:
        _logger.info(message)


def seed_all(seed, deterministic_algorithms: bool = False):
    """Seed RNGs; opt into strong determinism via ``deterministic_algorithms``.

    Strong determinism (deterministic algorithms + cudnn.deterministic + the
    cublas workspace pin) inserts fill/elementwise kernels and disables the cudnn
    autotuner, measurably lowering training MFU. Default is performance mode;
    enable it only for bitwise-reproducible experiments via the yaml switch
    ``hyperparams.deterministic_algorithms`` — the single determinism knob, with
    no environment-variable override. See docs qwen35_mfu_report.
    """
    # RNG seed 不算「确定性算法」——任何 run 都需要 seed，否则 random op 不可复现。
    # 这部分跟 flag 无关，永远调。
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic_algorithms:
        # 严格确定性套餐：cudnn 锁 deterministic kernel + benchmark 关 + torch 全 op 走
        # 确定性实现。same-seed + same-yaml 两次 run 的 step 0 forward
        # byte-equal、grad 单步 drift < 1e-3。唯一开关是 yaml 的
        # hyperparams.deterministic_algorithms（无环境变量旁路）。
        # CUBLAS_WORKSPACE_CONFIG 只 setdefault 兜底——launch 脚本显式 export 的值
        # 优先；不设则 PyTorch 第一个 cuBLAS 调用就 raise RuntimeError。
        # warn_only=True 内置：bf16 训练里总有几个 op 没确定实现（如某些 scatter_add
        # 路径），warn 比 crash 实用——拿大部分 deterministic 收益即可。
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        # perf 套餐：cudnn 自由选最快 algo（含 nondet variant），但 benchmark 不默认开：
        # benchmark=True 对每个新 conv 配置做一轮算法竞速，动态 shape 负载（如 Qwen3.5
        # vision Conv3d patch_embed）几乎每步撞新 shape、反复付竞速成本，step time 3.2x
        # 回退（实测 MFU 37.7%→11.7%）。固定 shape 负载（如 LIBERO 224
        # 输入）warm-up 一次后收益为正——已实测过收益的固定 shape 场景可用
        # WALLX_CUDNN_BENCHMARK=1 显式开启，不走全局默认。
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = os.environ.get(
            "WALLX_CUDNN_BENCHMARK", "0"
        ).lower() in ("1", "true")
        torch.use_deterministic_algorithms(False)


def save_text_results_to_file(
    current_step, all_input_texts, all_gt_texts, all_pred_texts, save_dir
):
    """Save text-generation results to a local JSON file."""
    import json

    os.makedirs(save_dir, exist_ok=True)

    results = []
    for i, (input_text, gt_text, pred_text) in enumerate(
        zip(all_input_texts, all_gt_texts, all_pred_texts)
    ):
        # Some callers pass one-item lists instead of plain strings.
        input_clean = input_text[0] if isinstance(input_text, list) else input_text
        gt_clean = gt_text[0] if isinstance(gt_text, list) else gt_text
        pred_clean = pred_text[0] if isinstance(pred_text, list) else pred_text

        results.append(
            {
                "sample_id": i,
                "input": input_clean,
                "ground_truth": gt_clean,
                "prediction": pred_clean,
                "step": current_step,
            }
        )

    filename = os.path.join(save_dir, f"text_predictions_step_{current_step}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# Add accuracy metrics.
# Add ade, fde, dtw, and frechet metrics from Ryan.
def compute_action_metrics(all_preds, all_actions, config, step_log={}):
    metrics_settings = config.get("metrics_settings", "default")
    metrics_available = ["l1", "mse", "accuracy", "ade", "fde", "dtw", "frechet"]
    metrics_default = ["l1", "mse", "accuracy"]
    if metrics_settings == "all":
        metrics_enabled = metrics_available
    elif metrics_settings == "default":
        metrics_enabled = metrics_default
    elif isinstance(metrics_settings, str):
        metrics_enabled = [
            m.strip()
            for m in metrics_settings.split(" ")
            if m.strip() in metrics_available
        ]
    elif isinstance(metrics_settings, list):
        metrics_enabled = [m for m in metrics_settings if m in metrics_available]
    else:
        metrics_enabled = metrics_default
        _logger.warning(
            'Unknown metrics_settings "%s". Using default metrics: %s',
            metrics_settings,
            metrics_enabled,
        )

    overall_l1 = F.l1_loss(all_preds, all_actions)
    overall_mse = F.mse_loss(all_preds, all_actions)

    step_log["val_action_l1"] = overall_l1.item()
    step_log["val_action_mse"] = overall_mse.item()

    if "accuracy" in metrics_enabled:
        accuracy_thresholds = [0.05, 0.1, 0.2, 0.4]
        # The accuracy that all predicted action dimensions are within a certain range of the ground truth.
        accuracies = get_action_accuracy(
            all_preds, all_actions, thresholds=accuracy_thresholds
        )
        for th_idx, threshold in enumerate(accuracy_thresholds):
            step_log[f"val_action_acc_thr{threshold}"] = accuracies[th_idx].item()

    start_idx = 0
    dof_config = config["dof_config"]
    for dof_key, dof_dim in dof_config.items():
        end_idx = start_idx + dof_dim
        # all_preds.shape = (B, T, action_dim)
        pred_dof = all_preds[..., start_idx:end_idx]
        action_dof = all_actions[..., start_idx:end_idx]
        dof_l1 = F.l1_loss(pred_dof, action_dof)
        dof_mse = F.mse_loss(pred_dof, action_dof)
        step_log[f"val_l1/{dof_key}"] = dof_l1.item()
        step_log[f"val_mse/{dof_key}"] = dof_mse.item()

        if "accuracy" in metrics_enabled:
            accuracies = get_action_accuracy(
                action_dof, pred_dof, thresholds=accuracy_thresholds
            )
            for th_idx, threshold in enumerate(accuracy_thresholds):
                step_log[f"val_acc/{dof_key}_thr{threshold}"] = accuracies[
                    th_idx
                ].item()

        if "ee_cartesian_pos" in dof_key:
            if "ade" in metrics_enabled:
                displacement_error = torch.norm(pred_dof - action_dof, dim=-1)
                ade = torch.mean(displacement_error)
                step_log[f"val_ade/{dof_key}"] = ade.item()

            if "fde" in metrics_enabled:
                final_pred = pred_dof[:, -1, :]
                final_gt = action_dof[:, -1, :]
                fde = torch.mean(torch.norm(final_pred - final_gt, dim=-1))
                step_log[f"val_fde/{dof_key}"] = fde.item()

            if "dtw" in metrics_enabled or "frechet" in metrics_enabled:
                dtw_distances = []
                frechet_distances = []
                batch_size = pred_dof.shape[0]
                for i in range(batch_size):
                    pred_seq = pred_dof[i]  # shape: (T, D)
                    gt_seq = action_dof[i]  # shape: (T, D)
                    dtw_dist = dtw_distance(pred_seq, gt_seq)
                    dtw_distances.append(dtw_dist)

                    frechet_dist = frechet_distance(pred_seq, gt_seq)
                    frechet_distances.append(frechet_dist)

                avg_dtw = torch.mean(torch.stack(dtw_distances))
                if "dtw" in metrics_enabled:
                    step_log[f"val_dtw/{dof_key}"] = avg_dtw.item()

                avg_frechet = torch.mean(torch.stack(frechet_distances))
                if "frechet" in metrics_enabled:
                    step_log[f"val_frechet/{dof_key}"] = avg_frechet.item()

        start_idx = end_idx

    return step_log


def get_openpi_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    peak_lr: float = None,
    end_lr: float = None,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that follows OpenPI style:
    - Warmup: linearly increases from peak_lr/(warmup_steps+1) to peak_lr
    - Decay: cosine decay from peak_lr to end_lr

    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        num_warmup_steps: The number of steps for the warmup phase.
        num_training_steps: The total number of training steps.
        peak_lr: The peak learning rate. If None, uses optimizer's initial lr.
        end_lr: The minimum learning rate at the end. If None, defaults to peak_lr * 0.1.
        last_epoch: The index of the last epoch when resuming training.

    Return:
        torch.optim.lr_scheduler.LambdaLR with the appropriate schedule.
    """
    if peak_lr is None:
        peak_lr = optimizer.defaults["lr"]
    if end_lr is None:
        end_lr = peak_lr * 0.1

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            # OpenPI warmup: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (num_warmup_steps + 1)
            current_lr = init_lr + (peak_lr - init_lr) * current_step / num_warmup_steps
            return current_lr / peak_lr  # LambdaLR multiplies by base_lr
        else:
            # Cosine decay
            decay_steps = num_training_steps - num_warmup_steps
            progress = min(1.0, (current_step - num_warmup_steps) / max(1, decay_steps))
            cos = 0.5 * (1 + math.cos(math.pi * progress))
            current_lr = end_lr + (peak_lr - end_lr) * cos
            return current_lr / peak_lr

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def plot_openloop(
    action_pred_list,
    action_gt_list,
    l1_loss,
    episode_index,
    save_path,
    is_static_list=None,
    dim_labels=None,
    skip_all_nan=False,
):
    """
    Plot openloop action comparison visualization.

    Args:
        action_pred_list: List of predicted actions, each with shape (horizon, action_dim)
        action_gt_list: List of ground truth actions, each with shape (horizon, action_dim)
        l1_loss: L1 loss array with shape (total_frames, action_dim)
        episode_index: Index of the episode being visualized
        save_path: Directory path to save the plot
        is_static_list: Optional list of booleans indicating static frames
        dim_labels: Optional per-dim subplot names (e.g. ``left_x`` /
            ``left_gripper`` / ``left_thumb_cmc_abduction``). Each subplot title
            is this name; falls back to ``Action Dimension i`` when not given.
        skip_all_nan: When True, drop dims whose L1 is all-NaN (padded DOF such
            as a dexterous hand's gripper or vendor-absent joints) so they don't
            add empty subplots.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assert len(action_pred_list) == len(
        action_gt_list
    ), "Predicted action and ground truth action must have the same shape."

    dim = action_pred_list[0].shape[1]
    dims = list(range(dim))
    if skip_all_nan:
        dims = [i for i in dims if np.any(~np.isnan(l1_loss[:, i]))]
    if not dims:
        return
    n = len(dims)
    plt.figure(figsize=(12, 4 * n))

    for sp, i in enumerate(dims):
        plt.subplot(n, 1, sp + 1)

        has_labeled_static = False
        for j in range(len(action_gt_list)):
            gt_action = action_gt_list[j]
            predict_action = action_pred_list[j]

            x_vals_gt = np.linspace(j, j + 1, len(gt_action))
            x_vals_pred = np.linspace(j, j + 1, len(predict_action))

            if is_static_list is not None and is_static_list[j]:
                label = None
                if not has_labeled_static:
                    label = "Static GT"
                    has_labeled_static = True
                plt.axvspan(j, j + 1, color="gray", alpha=0.2, label=label)

            if j == 0:
                plt.plot(
                    x_vals_gt,
                    gt_action[:, i],
                    label="Ground Truth",
                    color="blue",
                    linewidth=2,
                    linestyle="-",
                    marker="o",
                    markersize=3,
                )
                plt.plot(
                    x_vals_pred,
                    predict_action[:, i],
                    label="Model Output",
                    color="orange",
                    linewidth=2,
                    linestyle="--",
                    marker="x",
                    markersize=4,
                )
            else:
                plt.plot(
                    x_vals_gt,
                    gt_action[:, i],
                    color="blue",
                    linewidth=2,
                    linestyle="-",
                    marker="o",
                    markersize=3,
                )
                plt.plot(
                    x_vals_pred,
                    predict_action[:, i],
                    color="orange",
                    linewidth=2,
                    linestyle="--",
                    marker="x",
                    markersize=4,
                )

        # Mark chunk boundaries: each chunk is an INDEPENDENT open-loop
        # re-prediction (new observation + freshly sampled flow noise), so the
        # curve legitimately jumps at integer x. The line within [j, j+1] is one
        # chunk's full action_horizon-step prediction, not overlaid per-frame.
        for b in range(1, len(action_gt_list)):
            plt.axvline(
                b,
                color="green",
                linestyle=":",
                alpha=0.35,
                linewidth=1,
                label="Chunk boundary" if b == 1 else None,
            )

        name = (
            dim_labels[i]
            if dim_labels is not None and i < len(dim_labels)
            else f"Action Dimension {i + 1}"
        )
        plt.title(f"{name}, L1 Loss: {np.nanmean(l1_loss[:, i]):.6f}")
        plt.xlabel("Chunk index (each unit = one action_horizon prediction)")
        plt.ylabel("Action Value")
        plt.legend()

    plt.suptitle(
        f"Openloop Action Comparison for Episode {episode_index}, "
        f"L1 Loss: {np.nanmean(l1_loss):.6f}"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/{episode_index}.png")
    plt.close()
    _logger.info("Saved openloop plot to %s/%s.png", save_path, episode_index)
