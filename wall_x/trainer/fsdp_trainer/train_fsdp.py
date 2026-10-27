#!/usr/bin/env python3
"""
FSDP Training Entry Point
Launch with torchrun:
    torchrun --nproc_per_node=8 --master_port=29500 train_fsdp.py --config config.yaml
"""
import argparse
import dataclasses
import json
import logging
import os
import sys
import torch
from datetime import datetime
from wall_x.config.loader import load_config as load_typed_config
from wall_x.trainer.fsdp_trainer import (
    FSDPTrainer,
    cleanup_distributed,
    is_main_process,
)
import wandb

logger = logging.getLogger(__name__)


def _log_failure_diagnostics(trainer, device):
    """Best-effort per-rank CUDA and last-batch diagnostics."""
    global_rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    diagnostics = {
        "global_rank": global_rank,
        "local_rank": local_rank,
        "pid": os.getpid(),
        "hostname": os.environ.get("HOSTNAME", "<unknown>"),
        "device": str(device),
        "last_batch": getattr(trainer, "_last_batch_debug_context", None),
    }
    try:
        properties = torch.cuda.get_device_properties(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        diagnostics["gpu"] = {
            "name": properties.name,
            "total_gib": total_bytes / (1024**3),
            "free_gib": free_bytes / (1024**3),
            "allocated_gib": torch.cuda.memory_allocated(device) / (1024**3),
            "reserved_gib": torch.cuda.memory_reserved(device) / (1024**3),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device)
            / (1024**3),
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device)
            / (1024**3),
        }
    except Exception as diagnostic_error:
        diagnostics["gpu_diagnostics_error"] = (
            f"{type(diagnostic_error).__name__}: {diagnostic_error}"
        )
    logger.error(
        "[FailureDiagnostics] %s",
        json.dumps(diagnostics, ensure_ascii=False, sort_keys=True),
    )


class TeeOutput:
    """Write output to both the terminal and a file."""

    def __init__(self, file_path, mode="w"):
        self.terminal = sys.stdout
        self.log = open(file_path, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="FSDP Training Script")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML file",
    )
    # FSDP specific overrides
    parser.add_argument(
        "--fsdp_sharding_strategy",
        type=str,
        default=None,
        choices=[
            "full_shard",
            "shard_grad_op",
            "hybrid_shard",
            "_hybrid_shard_zero2",
        ],
        help="Legacy sharding strategy override for the active distributed backend",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode: reduces buffer size, sets log/save path to debug.",
    )
    parser.add_argument(
        "--wandb_offline",
        type=str,
        default=None,
        help="Whether to run wandb in offline mode (overrides config).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Whether to visualize samples during training.",
    )
    parser.add_argument(
        "--log_to_file",
        action="store_true",
        default=False,
        help="Whether to redirect stdout and stderr to log files.",
    )
    return parser.parse_args()


def load_config(config_path: str, cli_args=None):
    """Load configuration from YAML file into TrainConfig."""
    return load_typed_config(config_path, cli_args=cli_args)


def setup_logger(cfg):
    """Setup wandb logger if enabled"""
    log = cfg.logging
    if log.use_wandb and is_main_process():
        logger.info(
            "rank %s is initializing wandb , is main process %s",
            torch.distributed.get_rank(),
            is_main_process(),
        )
        wandb_run = wandb.init(
            project=log.log_project,
            name=log.log_name,
            entity=log.log_entity,
            config=dataclasses.asdict(cfg),
            save_code=False,
            force=False,
            mode="offline" if log.wandb_offline else "online",
        )
        logger.info("Wandb Initialized")
        return wandb_run
    return None


def print_fsdp_config(cfg):
    """Print FSDP configuration"""
    if is_main_process():
        dist = cfg.distributed
        logger.info("%s", "=" * 60)
        logger.info("Distributed Configuration:")
        logger.info("  backend: %s", dist.backend)
        active = dist.dmuon if dist.backend == "dmuon" else dist.fsdp
        logger.info("  sharding_strategy: %s", active.sharding_strategy)
        logger.info("  hsdp_replicate_size: %s", active.hsdp_replicate_size)
        logger.info("%s", "=" * 60)


def main():
    args = parse_args()

    # Redirect each process to rank-qualified files to avoid cross-rank write races.
    if args.log_to_file:
        import yaml as _yaml

        with open(args.config, "r") as f:
            _tmp = _yaml.safe_load(f)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        global_rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        # support both old (save_path) and new (checkpoint.save_path) schema
        log_dir = _tmp.get("save_path") or _tmp.get("checkpoint", {}).get(
            "save_path", "./ckpt"
        )
        if args.debug:
            log_dir = "./ckpt/debug"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(
            log_dir,
            f"training_log_{timestamp}_rank{global_rank}_local{local_rank}_pid{os.getpid()}.log",
        )
        sys.stdout = TeeOutput(log_file, mode="w")
        sys.stderr = TeeOutput(log_file.replace(".log", "_stderr.log"), mode="w")
        logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
        logger.info("\n%s", "=" * 80)
        logger.info("LOG TO FILE MODE: All output will be saved to:")
        logger.info("  STDOUT: %s", log_file)
        logger.info("  STDERR: %s", log_file.replace(".log", "_stderr.log"))
        logger.info("%s\n", "=" * 80)
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    # Determinism (incl. the cublas workspace pin) is config-driven and applied
    # in seed_all() from hyperparams.deterministic_algorithms.
    torch.cuda.init()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group("nccl", device_id=device)
    cfg = load_config(args.config, cli_args=args)
    wandb_run = setup_logger(cfg)
    print_fsdp_config(cfg)

    trainer = None
    try:
        trainer = FSDPTrainer(
            train_config=cfg,
            wandb_run=wandb_run,
        )
        trainer.fit()

    except Exception as e:
        logger.exception("Training failed with error: %s", e)
        _log_failure_diagnostics(trainer, device)
        raise

    finally:
        cleanup_distributed()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
