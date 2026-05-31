#!/usr/bin/env python3
"""
FSDP Training Entry Point
Launch with torchrun:
    torchrun --nproc_per_node=8 --master_port=29500 train_fsdp.py --config config.yaml
"""
import argparse
import dataclasses
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
        choices=["full_shard", "shard_grad_op", "no_shard", "hybrid_shard"],
        help="FSDP sharding strategy (overrides config)",
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
        logger.info("FSDP Configuration:")
        logger.info("  use_fsdp: %s", dist.use_fsdp)
        logger.info("%s", "=" * 60)


def main():
    args = parse_args()

    # Redirect logs to files only on the main process to avoid write races.
    if args.log_to_file:
        import yaml as _yaml

        with open(args.config, "r") as f:
            _tmp = _yaml.safe_load(f)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # support both old (save_path) and new (checkpoint.save_path) schema
        log_dir = _tmp.get("save_path") or _tmp.get("checkpoint", {}).get(
            "save_path", "./ckpt"
        )
        if args.debug:
            log_dir = "./ckpt/debug"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"training_log_{timestamp}.log")
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

    torch.cuda.init()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group("nccl", device_id=device)
    cfg = load_config(args.config, cli_args=args)
    wandb_run = setup_logger(cfg)
    print_fsdp_config(cfg)

    try:
        trainer = FSDPTrainer(
            train_config=cfg,
            wandb_run=wandb_run,
        )
        trainer.fit()

    except Exception as e:
        logger.exception("Training failed with error: %s", e)
        raise

    finally:
        cleanup_distributed()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
