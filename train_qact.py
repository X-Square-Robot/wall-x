import os
import json
import time
import yaml
import wandb
import accelerate
from argparse import ArgumentParser
from accelerate import (
    Accelerator,
    DistributedDataParallelKwargs,
    DataLoaderConfiguration,
)

from wall_x.trainer.qwen_vl_act_trainer import QwenVlAct_Trainer


def setup_environment():
    """Set up environment variables for training."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Set model_type in data config if not already set
    config["data"]["model_type"] = config.get("model_type")

    return config


def setup_accelerator(config):
    """Initialize and configure the accelerator for distributed training."""
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] Preparing accelerator"
    )

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator_dataloader_config = DataLoaderConfiguration(dispatch_batches=False)

    if config.get("FSDP2", False):
        # Use Fully Sharded Data Parallel (FSDP) version 2
        fsdp_plugin = accelerate.utils.dataclasses.FullyShardedDataParallelPlugin(
            fsdp_version=2, reshard_after_forward=True
        )
        print("[INFO] Using FSDP version 2 for distributed training")
    else:
        fsdp_plugin = None

    if config.get("torch_compile", False):
        # Use Torch Dynamo for compilation
        dynamo_plugin = accelerate.utils.TorchDynamoPlugin(
            backend="inductor",
            mode="default",
            fullgraph=False,
            dynamic=False,
        )
        print("[INFO] Using Torch Dynamo for compilation")
    else:
        dynamo_plugin = None

    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        mixed_precision="bf16",
        fsdp_plugin=fsdp_plugin,
        dynamo_plugin=dynamo_plugin,
        dataloader_config=accelerator_dataloader_config,
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
    )

    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] Accelerator initialization complete"
    )

    return accelerator


def setup_logging(config, accelerator):
    """Set up logging with wandb for the main process."""
    if not accelerator.is_main_process:
        return None

    # Create save directory if it doesn't exist
    save_path = config["save_path"]
    if not os.path.exists(save_path):
        print(f"Save path {save_path} does not exist, creating directory.")
        os.makedirs(save_path, exist_ok=True)

    print("Configuration:")
    print("=" * 50)
    print(json.dumps(config, indent=2, ensure_ascii=False))
    print("=" * 50)

    # Initialize wandb logger
    logger = wandb.init(
        project=config["log_project"],
        name=config["log_name"],
        save_code=False,
        force=False,
    )

    return logger


def main(args):
    """Main training function."""
    setup_environment()

    # Load configuration
    config = load_config(args.config)

    # Set up accelerator
    accelerator = setup_accelerator(config)

    # Set up logging
    logger = setup_logging(config, accelerator)

    # Initialize trainer
    trainer = QwenVlAct_Trainer(
        config=config,
        logger=logger,
        accelerator=accelerator,
        seed=args.seed,
        data_config_path=args.config,
    )

    # Start training
    trainer.fit()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script for Wall-X model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()
    main(args)
