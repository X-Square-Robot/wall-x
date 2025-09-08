import os
import gc
import time
import torch
import random
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from functools import wraps
from datetime import datetime
from torch.optim import AdamW
from accelerate import Accelerator
from safetensors.torch import load_file
from accelerate.utils import DistributedType
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup

from wall_x.utils.timers import Timers
from wall_x.model.qwen2_5_based import Qwen2_5_VLMoEForAction
from wall_x.data.config import ACTION_DATASET_NAMES, MULTIMODAL_DATASET_NAMES
from wall_x.data.load_lerobot_dataset import PreprocessedDataset, get_data_configs, load_lerobot_data


def timer(func):
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to be timed
        
    Returns:
        Wrapped function with timing functionality
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(
            f"\033[92m[current time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] Function {func.__name__} took {end_time - start_time:.2f} seconds to execute\033[0m"
        )
        return result
    return wrapper


def print_rank_last(message):
    """
    Print message only on the last rank in distributed training.
    
    Args:
        message (str): Message to print
    """
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1):
            print(message, flush=True)
    else:
        print(message, flush=True)


def seed_all(seed):
    """
    Set random seeds for reproducible training.
    
    Args:
        seed (int): Random seed value
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


class QwenVlAct_Trainer:
    """
    Vision-Language-Action trainer for Qwen-VL models with robotic action prediction.
    
    This trainer handles multi-modal learning combining vision, language, and action data
    for robotic control applications. It supports distributed training, mixed precision,
    gradient accumulation, and various optimization strategies including MoE (Mixture of Experts).
    
    Features:
    - Multi-modal data processing (vision + language + actions)
    - Distributed training with Accelerate
    - Gradient accumulation and clipping
    - Learning rate scheduling with warmup
    - Checkpoint saving and resuming
    - Comprehensive logging and monitoring
    """

    @timer
    def __init__(self, config, logger, accelerator: Accelerator = None, seed=42, data_config_path=None):
        """
        Initialize the Vision-Language-Action trainer.
        
        Args:
            config (dict): Training configuration dictionary containing:
                - processor_path (str): Path to data preprocessing processor
                - qwen_vl_act_config_path (str): Path to model configuration file
                - learning_rate (float): Base learning rate for training
                - num_epoch (int): Number of training epochs
                - pretrained_qwen_vl_path (str): Path to pretrained model
                - And other training hyperparameters
            logger: Logger instance for tracking metrics
            accelerator (Accelerator, optional): Hugging Face Accelerate instance for distributed training
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
            data_config_path (str, optional): Path to data configuration file
            
        Raises:
            ValueError: If required configuration keys are missing
        """
        # Validate required configuration keys
        required_keys = ["learning_rate", "num_epoch"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")

        self.config = config
        self.logger = logger
        self.accelerator = accelerator
        self.seed = seed
        
        # Initialize random seeds for reproducibility
        seed_all(self.seed)
        
        # Training state variables
        self.start_epoch = 0
        self.global_step = 0
        self.num_epoch = self.config["num_epoch"]
        self.initial_step = 0
        
        # Data and model configuration
        self.dataload_config = get_data_configs(self.config["data"])
        self.data_config_path = data_config_path
        self.use_fast_tokenizer = self.config.get("use_fast_tokenizer", False)
        
        # Load model and initialize training components
        self.load_model()
        self.action_dim = sum(self.config["dof_config"].values())
        
        # Distributed training setup
        self.rank = self.accelerator.process_index
        self.world_size = self.accelerator.num_processes
        print(f"rank {self.accelerator.process_index} after load model memory usage: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB", flush=True)
        
        # Load training data
        self.load_qact_data()
        print(f"rank {self.accelerator.process_index} after load qact data usage: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB", flush=True)

        # Resume from checkpoint if specified
        if "resume" in self.config:
            self.resume_from_checkpoint()

        # Initialize special token IDs
        self.propri_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|propri|>")
        self.action_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|action|>")

        # Initialize evaluation metrics
        self.base_l1_loss = None
        self.base_l1_loss_detail = {}

        # Performance monitoring
        self.timers = Timers(log_level=0, log_option="minmax")

        # Adjust global step if resuming from checkpoint
        if self.initial_step != 0:
            self.global_step = self.initial_step // self.config.get("gradient_accumulation_steps", 1)

    def print_rank0(self, msg, flush=True):
        """
        Print message only on rank 0 (main process).
        
        Args:
            msg: Message to print
            flush (bool): Whether to flush output buffer
        """
        if self.accelerator.is_main_process:
            print(msg, flush=flush)

    def fit(self):
        """
        Main training loop executing multiple epochs with validation.
        
        Handles the complete training process including:
        - Training loop execution
        - Validation after each epoch
        - Process synchronization
        - Memory cleanup
        """
        self.accelerator.wait_for_everyone()
        
        # Optional validation before training starts
        if self.config.get("resume", None) is not None and self.config["resume"].get("validate_first", False):
            self.val_loop()
            self.accelerator.wait_for_everyone()

        # Main training loop
        for epoch in range(self.start_epoch, self.num_epoch):
            self.train_loop(epoch)
            self.accelerator.wait_for_everyone()
            
            if (epoch + 1) % self.config.get("epoch_save_interval", 10) == 0:
                self.save_checkpoint(epoch)
            
            # Validation after each epoch
            self.val_loop()
            self.accelerator.wait_for_everyone()
            
            # Memory cleanup
            gc.collect()

    def train_loop(self, epoch):
        """
        Execute training for a single epoch.
        
        Args:
            epoch (int): Current epoch number
            
        Handles:
        - Data loading and batching
        - Forward/backward passes
        - Gradient accumulation and clipping
        - Learning rate scheduling
        - Loss logging and monitoring
        - Performance profiling (optional)
        """
        # Initialize training dataloader for current epoch
        if isinstance(self.dataset, PreprocessedDataset):
            if getattr(self, "train_dataloader", None) is not None:
                self.train_sampler.set_epoch(epoch)
            else:
                self.train_dataloader, self.train_sampler = self.dataset.get_train_dataloader()
                self.train_sampler.set_epoch(epoch)
        else:
            self.train_dataloader = self.dataset.get_train_dataloader()

        self.model.train()
        grad_accum_steps = self.config.get("gradient_accumulation_steps", 1)
        total = len(self.train_dataloader)
        t0 = time.time()
        enable_profiling = self.config['profile']

        # Optional PyTorch profiler for performance analysis
        if enable_profiling:
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=self.config['profile_wait_iters'],
                                                 warmup=self.config['profile_warmup_iters'],
                                                 active=self.config['profile_active_iters']),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.config['profile_save_path'], worker_name="worker0"),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            profiler.__enter__()

        try:
            for i, batch in enumerate(self.train_dataloader, self.initial_step):
                # Move batch to device
                if isinstance(self.dataset, PreprocessedDataset):
                    batch = {k: v.to(self.accelerator.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                self.timers("data-load").stop()
                
                with self.accelerator.accumulate(self.model):
                    # Forward pass
                    self.timers("forward-compute", log_level=0).start(barrier=False)
                    outputs = self.model(**batch, mode="train")
                    self.timers("forward-compute").stop()
                    
                    loss = outputs.loss
                    
                    # Check for NaN loss
                    if torch.isnan(loss):
                        print(f"Warning: NaN loss detected in epoch: {epoch}, step: {i}", flush=True)
                        continue
                    
                    # Backward pass
                    self.timers("backward-compute", log_level=0).start(barrier=False)
                    self.accelerator.backward(loss)
                    self.timers("backward-compute").stop()
                    
                    # Gradient clipping
                    total_norm = self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.config.get("max_grad_norm", 1.0)
                    )

                    # Optimizer step
                    self.timers("optimizer", log_level=0).start(barrier=False)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.timers("optimizer").stop()

                    # Update global step and learning rate after gradient accumulation
                    if (i + 1) % grad_accum_steps == 0:
                        self.lr_scheduler.step()
                        self.global_step += 1
                        lr = self.lr_scheduler.get_last_lr()[0]
                        
                        # Gather loss across all processes for logging
                        train_loss = self.accelerator.gather(loss.detach()).mean().item()
                        _log_dict = {
                            "lr": lr,
                            "train_loss": train_loss,
                        }

                        # Log component losses
                        if "cross_entropy_loss" in outputs and outputs.cross_entropy_loss is not None:
                            _log_dict["cross_entropy_loss"] = self.accelerator.gather(outputs.cross_entropy_loss.detach()).mean().item()
                        
                        if "flow_loss" in outputs and outputs.flow_loss is not None:
                            _log_dict["flow_loss"] = self.accelerator.gather(outputs.flow_loss.detach()).mean().item()
                        
                        # Log per-dataset channel losses
                        if "channel_loss_dict" in outputs and outputs.channel_loss_dict is not None:
                            for dataset_name_i in ACTION_DATASET_NAMES + MULTIMODAL_DATASET_NAMES:
                                count_sum = self.accelerator.gather(outputs.channel_loss_count_dict[dataset_name_i]).sum().item()
                                if count_sum > 0:
                                    channel_loss = self.accelerator.gather(outputs.channel_loss_dict[dataset_name_i].detach()).sum().item() / count_sum
                                    _log_dict[f"channel_loss_{dataset_name_i}"] = channel_loss
                            
                            # Log action accuracy for fast tokenizer
                            if "action_accuracy" in outputs.channel_loss_dict and self.use_fast_tokenizer:
                                _log_dict["action_accuracy"] = (
                                    self.accelerator.gather(outputs.channel_loss_dict["action_accuracy"].detach()).mean().item()
                                )

                        # Log metrics
                        if self.logger is not None:
                            self.logger.log(_log_dict, step=self.global_step)

                        # Log gradient norm
                        if self.logger is not None and self.accelerator.sync_gradients:
                            self.logger.log({"total_norm": total_norm}, step=self.global_step)

                self.timers("interval-time").stop()
                
                # Setup timers for next iteration
                if i < len(self.train_dataloader) - 1:
                    self.timers("interval-time", log_level=0).start(barrier=False)
                    self.timers("data-load", log_level=0).start(barrier=False)
                

                # Periodic logging
                t1 = time.time()
                if i % 1 == 0:
                    lr = self.lr_scheduler.get_last_lr()[0]
                    self.training_log(epoch, self.num_epoch, i, total, loss, lr, t1 - t0)
                    t0 = time.time()
                
                if enable_profiling:
                    profiler.step()

        finally:
            if enable_profiling:
                profiler.__exit__(None, None, None)

    @torch.no_grad()
    def val_loop(self):
        """
        Execute validation loop with gradient computation disabled.
        
        Evaluates model performance on validation set and logs validation loss.
        """
        # Initialize validation dataloader
        if getattr(self, "val_dataloader", None) is not None:
            self.val_sampler.set_epoch(0)
        else:
            self.val_dataloader, self.val_sampler = self.dataset.get_val_dataloader()
            self.val_sampler.set_epoch(0)

        self.model.eval()
        self.val_loss = 0
        
        # Validation loop
        for i, batch in enumerate(
            tqdm(self.val_dataloader, desc="Validating", total=len(self.val_dataloader), 
                 disable=not self.accelerator.is_main_process)
        ):
            if isinstance(self.dataset, PreprocessedDataset):
                batch = {k: v.to(self.accelerator.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            with torch.no_grad():
                outputs = self.model(**batch, mode="train")
                loss = outputs.loss
                self.val_loss += self.accelerator.gather(loss.detach()).mean().item()
        
        # Calculate average validation loss
        self.val_loss /= len(self.val_dataloader)
        
        # Log validation metrics
        if self.logger is not None:
            self.logger.log({"val_loss": self.val_loss}, step=self.global_step)
        
        self.model.train()

    @timer
    def load_model(self):
        """
        Load and configure the Vision-Language-Action model.
        
        Handles:
        - Model loading from pretrained weights
        - Processor initialization
        - Optimizer configuration (with support for different learning rates for different components)
        - Learning rate scheduler setup
        - Model preparation for distributed training
        """
        # Load pretrained model
        model = Qwen2_5_VLMoEForAction.from_pretrained(
            self.config["pretrained_qwen_vl_path"], 
            **{"use_fast_tokenizer": self.use_fast_tokenizer}
        )
        self.processor = model.processor
        model = model.to(torch.bfloat16)

        # Configure optimizer based on training strategy
        if "freeze_vlm" in self.config and self.config["freeze_vlm"]:
            print("Freezing VLM parameters, training only MoE experts", flush=True)
            moe_params = []
            for name, param in model.named_parameters():
                if "moe.experts.1." not in name:
                    param.requires_grad = False
                else:
                    moe_params.append(param)
            param_groups = [{"params": moe_params, "lr": self.config["learning_rate"]}]
            self.optimizer = AdamW(param_groups, weight_decay=0.1)
            
        elif "action_expert_learning_rate" in self.config:
            # Separate learning rates for VLM and action expert parameters
            moe_params = []
            vlm_params = []
            for name, param in model.named_parameters():
                if "moe.experts.1." in name:
                    moe_params.append(param)
                else:
                    vlm_params.append(param)

            # Configure parameter groups
            if self.config.get("train_action_expert_only", False):
                self.print_rank0("Training action expert only", flush=True)
                param_groups = [{"params": moe_params, "lr": self.config["action_expert_learning_rate"]}]
            else:
                param_groups = [
                    {"params": vlm_params, "lr": self.config["learning_rate"]},
                    {"params": moe_params, "lr": self.config["action_expert_learning_rate"]},
                ]

            self.optimizer = AdamW(param_groups, weight_decay=0.1)
            self.print_rank0(
                f"Setting MoE learning rate to {self.config['action_expert_learning_rate']}, "
                f"VLM learning rate to {self.config['learning_rate']}", flush=True
            )
        else:
            # Standard optimizer configuration
            self.optimizer = AdamW(
                model.parameters(),
                lr=self.config["learning_rate"],
                weight_decay=0.1,
            )

        # Configure learning rate scheduler
        warmup_steps = self.config.get("num_warmup_steps", 0)
        num_training_steps = self.config.get("num_training_steps", 1000000000)
        min_lr = self.config.get("min_lr", 0.1 * self.config["learning_rate"])
        self.lr_scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            min_lr=min_lr,
        )

        self.model = model

        # Enable gradient computation for embeddings
        if hasattr(model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            self.model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # Prepare model, optimizer, and scheduler for distributed training
        self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler
        )

    @timer
    def load_qact_data(self):
        """
        Load and configure training data for Vision-Language-Action learning.
        
        Supports LeRobot dataset format and handles distributed data loading
        across multiple processes.
        """
        print(f"Loading Vision-Language-Action data from {__file__}")
        self.accelerator.wait_for_everyone()

        # Load LeRobot dataset
        self.dataset, self.train_num = load_lerobot_data(
            self.config,
            self.dataload_config.get("lerobot_config", {}),
            rank=self.rank,
            world_size=self.world_size,
        )

    @timer
    def load_qwen_pretrain_weight(self, model, pretrain_weight_path):
        """
        Load pretrained Qwen weights with MoE adaptation.
        
        Args:
            model: Model instance to load weights into
            pretrain_weight_path (str): Path to pretrained weight files
            
        Returns:
            Model with loaded pretrained weights
            
        Handles weight key renaming for MoE architecture compatibility.
        """
        # Load all safetensors files
        weight_files = sorted([f for f in os.listdir(pretrain_weight_path) if f.endswith(".safetensors")])
        merged_weights = {}

        # Merge weights from all files
        for weight_file in weight_files:
            file_path = os.path.join(pretrain_weight_path, weight_file)
            weights = load_file(file_path)
            merged_weights.update(weights)

        # Rename weights for MoE compatibility
        renamed_weights = {}
        for key, value in merged_weights.items():
            if key.startswith("model.layers") and "mlp." in key and model.config.mlp_moe:
                # Rename MLP weights for MoE structure
                layer_num = key.split(".layers.")[1].split(".mlp")[0]
                new_key = key.replace(f"layers.{layer_num}.mlp.", f"layers.{layer_num}.moe.experts.0.")
                renamed_weights[new_key] = value
            elif key.startswith("model.layers") and "self_attn." in key and model.config.attention_moe:
                # Rename attention weights for MoE structure
                layer_num = key.split(".layers.")[1].split(".self_attn")[0]
                proj_types = ["q_proj", "k_proj", "v_proj", "o_proj"]
                for proj in proj_types:
                    if proj in key:
                        new_key = key.replace(f"layers.{layer_num}.self_attn.{proj}", 
                                            f"layers.{layer_num}.self_attn.{proj}_experts.0")
                        renamed_weights[new_key] = value
                        break
            else:
                renamed_weights[key] = value

        # Load weights into model
        err = model.load_state_dict(renamed_weights, strict=False)
        self.print_rank0(f"Weight loading report: {err}", flush=True)
        if self.accelerator.is_main_process:
            self.print_rank0(f"Loaded pretrained weights from: {pretrain_weight_path}")

        return model

    def training_log(self, current_epoch, total_epoch, current_train_iter, total_train_iter, loss, lr, time_per_step):
        """
        Log training progress and performance metrics.
        
        Args:
            current_epoch (int): Current epoch number
            total_epoch (int): Total number of epochs
            current_train_iter (int): Current training iteration
            total_train_iter (int): Total iterations in epoch
            loss (torch.Tensor): Current loss value
            lr (float): Current learning rate
            time_per_step (float): Time taken for current step
        """
        timers_to_log = ["interval-time", "data-load", "forward-compute", "backward-compute", "optimizer"]
        
        log_string = f" [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
        log_string += " epoch {:3d}/{:3d} |".format(current_epoch, total_epoch)
        log_string += " iter {:6d}/{:6d} |".format(current_train_iter, total_train_iter)
        log_string += " loss {:.6f} |".format(loss)
        log_string += " lr {:.6f} |".format(lr)
        log_string += " time_per_step_avg {:.6f}s |".format(time_per_step)
        
        print_rank_last(log_string)
        self.timers.log(timers_to_log, normalizer=1)

    def save_checkpoint(self, epoch, step=0):
        """
        Save training checkpoint.
        
        Args:
            epoch (int): Current epoch number
            step (int, optional): Current step number. Defaults to 0.
            
        Saves model state, optimizer state, and training progress information.
        """
        save_path = self.config["save_path"]
        if step == 0:
            ckpt_path = f"{save_path}/{epoch}"
        else:
            ckpt_path = f"{save_path}/{epoch}_{step}"
        
        self.accelerator.save_state(ckpt_path)

        # Save current iteration steps for dataset resuming
        if step != 0:
            _rank = self.accelerator.process_index
            if isinstance(self.dataset, PreprocessedDataset):
                torch.save(
                    {"epoch": epoch, "step": step}, 
                    os.path.join(ckpt_path, f"epoch_{epoch}_step_{step}_rank_{_rank}.pth")
                )

    def resume_from_checkpoint(self):
        """
        Resume training from a saved checkpoint.
        
        Handles both full checkpoint loading and model-only loading based on configuration.
        """
        checkpoint_path = self.config["resume"]["ckpt"]

        if self.config.get("resume", {}).get("load_ckpt_only", False):
            # Load only model weights
            ckpt_path = self.config["resume"]["ckpt"] + "/model.safetensors"
            state_dict = load_file(ckpt_path, device="cpu")
            
            # Add module prefix if needed for distributed training
            new_state_dict = {}
            for key in state_dict:
                if not key.startswith("module."):
                    new_key = "module." + key
                new_state_dict[new_key] = state_dict[key]
            
            err = self.model.load_state_dict(new_state_dict, strict=False)
        else:
            # Load full checkpoint including optimizer and scheduler states
            self.accelerator.load_state(checkpoint_path)
        
        self.print_rank0(f"Resumed from checkpoint: {checkpoint_path}")

    def log_l1_details(self, all_label, all_pred, all_task, all_dof_mask):
        """
        Log detailed L1 loss metrics by degrees of freedom.
        
        Args:
            all_label (torch.Tensor): Ground truth action labels
            all_pred (torch.Tensor): Predicted actions
            all_task (list): Task identifiers
            all_dof_mask (torch.Tensor): Degrees of freedom mask
            
        Computes and logs L1 loss for each DOF component separately for detailed analysis.
        """
        all_task = all_task[:len(all_label)]

        # Apply DOF mask
        all_label = all_label * all_dof_mask
        all_pred = all_pred * all_dof_mask
        
        # Compute baseline L1 loss (predict mean action)
        if self.base_l1_loss is None:
            mean_action = all_label.mean(dim=0)
            self.base_l1_loss = nn.functional.l1_loss(all_label, mean_action)

        self.logger.log({"base_l1_loss": self.base_l1_loss.item()}, step=self.global_step)

        # Log L1 loss for each DOF component
        start_idx = 0
        dof_config = self.config.get("dof_config", {})
        for dof in dof_config:
            end_idx = start_idx + dof_config[dof]
            dof_label = all_label[:, :, start_idx:end_idx]
            dof_pred = all_pred[:, :, start_idx:end_idx]
            dof_l1 = nn.functional.l1_loss(dof_pred, dof_label)
            
            self.print_rank0(f"DOF {dof}, L1 loss: {dof_l1.item()}", flush=True)
            self.logger.log({f"detail/l1_loss_{dof}": dof_l1.item()}, step=self.global_step)
            
            start_idx = end_idx