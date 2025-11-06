import os
import json
import time
import yaml
import wandb
import torch
import accelerate
from argparse import ArgumentParser
from accelerate import (
    Accelerator,
    DistributedDataParallelKwargs,
    DataLoaderConfiguration,
)
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial

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
        # Import model classes for auto wrap policy
        from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import (
            Qwen2_5_VLDecoderLayer_with_MoE
        )
        from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl import (
            Qwen2_5_VLVisionBlock
        )
        
        # IMPORTANT: MixedPrecisionPolicy configuration for selective mixed precision
        # 
        # The strategy:
        # 1. param_dtype: The dtype for model parameters (weights, biases)
        # 2. reduce_dtype: The dtype for gradient all-reduce operations
        #
        # With FSDP2, MixedPrecisionPolicy is applied to ALL wrapped modules.
        # To keep specific layers (LayerNorm, ActionProcessor) in fp32:
        # - We apply bf16 policy globally here
        # - Then model.to_bfloat16_for_selected_params() converts specific params back to fp32
        # - This happens BEFORE FSDP wrapping in the trainer
        mixed_precision_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,  # Store parameters in bf16 (after selective conversion)
            reduce_dtype=torch.bfloat16,  # Reduce gradients in bf16 for efficiency
        )
        
        # Auto wrap policy: wrap transformer layers as separate FSDP units
        # This enables efficient sharding and memory usage
        # Note: We DON'T wrap LayerNorm or ActionProcessor here - they stay with their parent
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                Qwen2_5_VLDecoderLayer_with_MoE,  # Wrap decoder layers (will be bf16 except LayerNorms)
                Qwen2_5_VLVisionBlock,             # Wrap vision blocks (will be bf16 except LayerNorms)
            },
        )
        
        # Use Fully Sharded Data Parallel (FSDP) version 2 with mixed precision
        fsdp_plugin = accelerate.utils.dataclasses.FullyShardedDataParallelPlugin(
            fsdp_version=2,
            reshard_after_forward=True,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision_policy=mixed_precision_policy,
        )
        print("[INFO] Using FSDP version 2 with selective mixed precision")
        print("[INFO] Wrapping as separate FSDP units: Qwen2_5_VLDecoderLayer_with_MoE, Qwen2_5_VLVisionBlock")
        print("[INFO] MixedPrecisionPolicy: param_dtype=bf16, reduce_dtype=bf16")
        print("[INFO] LayerNorms and action_preprocessor will be kept in fp32 via model.to_bfloat16_for_selected_params()")
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
