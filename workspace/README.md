# Training Guide

This document explains the key configuration parameters and memory requirements for Wall-X training.

## Quick Start Checklist

### 🚀 **Step 1: Download Pre-trained Model**
Choose one of the available models:
- **WALL-OSS-FLOW**: https://huggingface.co/x-square-robot/wall-oss-flow
- **WALL-OSS-FAST**: https://huggingface.co/x-square-robot/wall-oss-fast

### ⚙️ **Step 2: Configure Environment**
- Update `run.sh`: Set `code_dir` and `config_path` to your actual paths
- Set `CUDA_VISIBLE_DEVICES` for your available GPUs

### 📝 **Step 3: Update Configuration Files**
- Replace all `/path/to/` placeholders in `config_qact.yml` with actual paths
- Configure robot settings: `dof_config` and `agent_pos_config`
- Set dataset: Choose appropriate `repo_id`
- Adjust `batch_size_per_gpu` based on your GPU memory

### ▶️ **Step 4: Start Training**
```bash
bash ./workspace/lerobot_example/run.sh
```

## Enable FAST tokenizer
To fine-tune using the FAST tokenizer, please download the repository and update the `action_tokenizer_path`. Make sure to set `use_fast_tokenizer` to `true`:
```bash
git clone https://huggingface.co/physical-intelligence/fast
```

## Required Paths (Must Modify)
```yaml
pretrained_wallx_path: "/path/to/wallx_model/"      # Path to pretrained wallx model
save_path: "/path/to/workspace/"                    # Path to save training outputs
use_fast_tokenizer: False                           # True: train FAST, False: train Flow
action_tokenizer_path: "/path/to/fast/"             # Must set if use_fast_tokenizer is True
```

## Training Parameters (Commonly Modified)

### Learning Rate Settings
- `learning_rate`: Initial learning rate (default: 0.00009)
- `min_lr`: Minimum learning rate for scheduler (default: 0.00005)
- `num_warmup_steps`: Number of warmup steps (default: 100)

### Batch Size and Memory
- `batch_size_per_gpu`: Batch size per GPU - adjust based on GPU memory
- `gradient_accumulation_steps`: Gradient accumulation steps
- `num_training_steps`: Total training steps
- `num_epoch`: Number of training epochs

### Training Optimization Settings
- `FSDP2`: Enable FSDP2 for distributed training (default: True) - **Recommended for multi-GPU**
- `torch_compile`: Enable PyTorch compilation optimization (default: False)

**⚠️ Important Note on torch_compile:**
- **Benefits**: Enabling `torch_compile` can significantly improve training efficiency
- **Requirements**: Requires that the data input shape is always consistent throughout training
- **Caution**: If you don't have sufficient understanding of torch compile, please **DO NOT** enable it as it may cause unexpected issues with dynamic input shapes

## Robot Configuration (Modify for Your Robot)

### DOF Configuration
Modify `dof_config` to match your robot's action space:
- Add/remove action keys based on your robot's capabilities
- Ensure DOF numbers match your robot's action dimensions

### Agent Position Configuration
Keep `agent_pos_config` consistent with `dof_config`.

### Action Keys
- `obs_action_keys`: Actions used as observation context
- `predict_action_keys`: Actions to predict/control

## Data Configuration

### Dataset
- `repo_id`: LeRobot dataset identifier
- `train_test_split`: Training/validation split ratio (default: 0.95)
- `action_horizon`: Number of future actions to predict (default: 32)

### Image Settings
- `resolution`: Image resolution for different camera views
- `download_videos`: Whether to download video files (true/false)

## Resume Training (Optional)
- `resume.ckpt`: Path to checkpoint for resuming training
- `resume.load_ckpt_only`: Only load model weights, not optimizer state

## Memory Usage

Below are the memory consumption benchmarks for different training configurations using the `lerobot/aloha_mobile_cabinet` dataset:

| Dataset | Batch Size | FSDP2 | Torch Compile | Num GPUs | Max Allocated Memory |
|---------|------------|--------|---------------|----------|---------------------|
| lerobot/aloha_mobile_cabinet | 1 | ❌ | ❌ | 1 | 40.11G |
| lerobot/aloha_mobile_cabinet | 1 | ❌ | ❌ | 8 | 48.02G |
| lerobot/aloha_mobile_cabinet | 1 | ✅ | ❌ | 2 | 43.70G |
| lerobot/aloha_mobile_cabinet | 1 | ✅ | ❌ | 8 | 24.96G |
| lerobot/aloha_mobile_cabinet | 1 | ✅ | ✅ | 8 | 24.21G |


**Hardware Recommendations:**

- For single GPU training: Ensure at least 48GB VRAM (e.g., RTX 6000 Ada, A6000)
- For multi-GPU training: Enable FSDP2 for optimal memory distribution
