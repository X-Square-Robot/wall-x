# Wall-X-OSS Usage Guide

This document explains how to fine-tune, run inference, and perform open-loop evaluation with the **Wall-OSS-0.5** pretrained model on LeRobot-format datasets. The workflow has five steps: download the model → write a config → start training → start inference → plot open-loop results.

> All commands below assume you are in the **repository root** (`wall-x-oss/`).

> **Note:** This open-source release targets **Wall-OSS-0.5**. If you are using **Wall-OSS-FLOW** or **Wall-OSS-FAST** instead, switch back to the previous codebase version:
>
> ```bash
> git checkout 97406f2ab5de414c79b091873f946c112d105c72
> ```

---

## Environment Setup

```bash
conda create --name wallx python=3.10
conda activate wallx

pip install torch torchvision transformers
pip install huggingface_hub lerobot pyyaml wandb websockets msgpack msgpack-numpy opencv-python

# Install wall-x-oss (from repo root)
pip install -e .
```

---

## Step 1: Download the Wall-OSS-0.5 Model

Wall-OSS-0.5 is a VLA foundation model built on Qwen2.5-VL-3B. Fine-tuning requires both the **pretrained weights** and the **VLM processor**.

### 1.1 Download Wall-OSS-0.5

```bash
# Option A: huggingface-cli
huggingface-cli download X-Square-Robot/wall-oss-0.5 \
  --local-dir /path/to/wall-oss-0.5

# Option B: Python
python -c "
from huggingface_hub import snapshot_download
snapshot_download('X-Square-Robot/wall-oss-0.5', local_dir='/path/to/wall-oss-0.5')
"
```

After download, the directory should contain at least:

| File | Purpose |
|------|---------|
| `config.json` | Model architecture; maps to `model.config_path` in training YAML |
| `model.safetensors` | Pretrained weights; maps to `checkpoint.resume_from` |
| tokenizer / processor files | Used at inference time |

HuggingFace: <https://huggingface.co/X-Square-Robot/wall-oss-0.5>

### 1.2 Download Qwen2.5-VL-3B-Instruct (processor)

Set `model.processor_path` and `model.pretrained_path` to the Qwen2.5-VL processor directory:

```bash
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir /path/to/Qwen2.5-VL-3B-Instruct
```

### 1.3 Prepare a LeRobot Dataset

Convert your target dataset to **LeRobot v3** format and note the local root path (used for `data.lerobot_config.repo_id`).

Reference configs in this repo:

- `workspace/example/fintune_bus2602/libero.yml` — LIBERO single-arm, 256px, 2 cameras
- `workspace/example/fintune_bus2602/maniparena_example.yml` — ManipArena dual-arm, 448px, 3 cameras

---

## Step 2: Write config.yml

Copy an example YAML and replace every `/path/to/*` placeholder with real paths:

```bash
cp workspace/example/fintune_bus2602/libero.yml /path/to/my_config.yml
```

### 2.1 Required Paths (Must Modify)

```yaml
model:
  config_path: /path/to/wall-oss-0.5/config.json
  processor_path: /path/to/Qwen2.5-VL-3B-Instruct
  pretrained_path: /path/to/Qwen2.5-VL-3B-Instruct

data:
  lerobot_config:
    repo_id: /path/to/libero_all          # local LeRobot dataset root
  norm_stats_path: /path/to/norm_stats.json
  key_mappings:                           # match your dataset keys
    camera:
      observation.images.faceImg: face_view
      observation.images.rightImg: right_wrist_view
    state: observation.state
    action: action

checkpoint:
  save_path: /path/to/training_output
  resume_from: /path/to/wall-oss-0.5/model.safetensors
```

### 2.2 Compute Normalization Stats

Generate `norm_stats.json` from your dataset before training:

```bash
python scripts/compute_norm_stats.py \
  --train_config /path/to/my_config.yml \
  --data_root /path/to/libero_all \
  --output_path /path/to/norm_stats.json
```

Then set `data.norm_stats_path` to the generated JSON file.

### 2.3 Robot DOF Configuration

`task.dof_config` defines **predicted action dimensions**; `task.agent_pos_config` defines **observation state dimensions**. Keys must be unique, and the sum of each config's dimensions must match `norm_stats.json`.

Wall-OSS-0.5 pretraining uses a **26-dimensional** action/state space. If your robot has fewer dimensions, pad with `action_padding` (see comments in `libero.yml`):

```yaml
task:
  dof_config:
    master_right_ee_cartesian_pos: 3
    master_right_ee_rotation: 3
    master_right_gripper: 1
    action_padding: 19                    # 3+3+1+19 = 26
  agent_pos_config:
    follow_right_ee_cartesian_pos: 3
    follow_right_ee_rotation: 3
    follow_right_gripper: 1
    action_padding: 19
  action_horizon: 10
```

### 2.4 Common Training Hyperparameters

| Parameter | Description | Default (libero example) |
|-----------|-------------|--------------------------|
| `hyperparams.batch_size_per_gpu` | Batch size per GPU; tune for VRAM | 4 |
| `hyperparams.gradient_accumulation_steps` | Gradient accumulation steps | 4 |
| `hyperparams.optimizer.learning_rate` | Learning rate | 5e-5 |
| `hyperparams.num_epoch` | Number of training epochs | 100 |
| `distributed.use_fsdp` | Multi-GPU FSDP training | true |
| `logging.save_interval` | Checkpoint save interval (steps) | 2000 |

**Memory benchmarks** (`lerobot/aloha_mobile_cabinet` dataset):

| Batch Size | FSDP2 | Num GPUs | Peak VRAM |
|------------|-------|----------|-----------|
| 1 | ❌ | 1 | ~40 GB |
| 1 | ✅ | 8 | ~25 GB |

Single-GPU training needs at least **48 GB** VRAM. For multi-GPU runs, enable `distributed.use_fsdp: true`.

---

## Step 3: Start Training

### 3.1 Single-GPU / Multi-GPU Training

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 \
  python wall_x/trainer/fsdp_trainer/train_fsdp.py \
    --config /path/to/my_config.yml

# Multi-GPU (recommended)
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  torchrun --nproc_per_node=4 \
    wall_x/trainer/fsdp_trainer/train_fsdp.py \
    --config /path/to/my_config.yml
```

Logs and checkpoints are written to `checkpoint.save_path`.

### 3.2 Resume Training

Point `checkpoint.resume_from` to a checkpoint **directory** (not a single `.safetensors` file) to restore optimizer and scheduler state:

```yaml
checkpoint:
  resume_from: /path/to/training_output/checkpoint_step_10000
```

### 3.3 Merge FSDP Sharded Weights (Optional)

If training saved FSDP-sharded checkpoints, merge them into a single safetensors file before inference:

```bash
# Option A: accelerate
accelerate merge-weights /path/to/sharded_tensors /path/to/merged_model.safetensors

# Option B: repo script (handles older PyTorch versions)
python scripts/merge_sharded_weights.py \
  /path/to/sharded_checkpoint \
  /path/to/merged_checkpoint
```

Copy processor / tokenizer files into the merged checkpoint directory afterward.

---

## Step 4: Start Inference

Ensure a training config (`config.yml` or `config.yaml`) sits next to the checkpoint, or pass it explicitly via `--train-config-path`.

### 4.1 Quick Smoke Test

Verify the checkpoint loads and produces an action chunk from synthetic observations:

```bash
python scripts/fake_inference.py \
  --checkpoint-path /path/to/checkpoint \
  --train-config-path /path/to/my_config.yml
```

### 4.2 LIBERO Simulation Evaluation

```bash
CHECKPOINT_PATH=/path/to/checkpoint \
TRAIN_CONFIG_PATH=/path/to/my_config.yml \
TASK_SUITE_NAME=libero_spatial \
NUM_TRIALS_PER_TASK=50 \
bash scripts/run_libero.sh

# Quick smoke test (1 trial per task)
SMOKE=1 CHECKPOINT_PATH=/path/to/checkpoint bash scripts/run_libero.sh
```

Useful environment variables:

| Variable | Description |
|----------|-------------|
| `CHECKPOINT_PATH` | Checkpoint directory |
| `TRAIN_CONFIG_PATH` | Training YAML |
| `TASK_SUITE_NAME` | `libero_spatial` / `libero_object` / `libero_goal` / `libero_10` |
| `TASK_INDICES` | Task indices, e.g. `0,1,2` |
| `CUDA_ID` | GPU index |

### 4.3 WebSocket Inference Server

Suitable for real-robot deployment or open-loop evaluation clients:

```bash
export CUDA_VISIBLE_DEVICES=0

python -m wall_x._vendor.harrix.serving.launch_serving \
  --env X2ROBOT \
  --port 32194 \
  --no-serialize-actions \
  model-config:server-model-config \
  --model-config.checkpoint-path /path/to/checkpoint \
  --model-config.train-config-path /path/to/my_config.yml \
  --model-config.action-horizon 10 \
  --model-config.robot-type desktop
```

Clients connect at `ws://127.0.0.1:32194` to send observations and receive predicted actions.

---

## Step 5: Plot Open-Loop Results

Open-loop evaluation compares model-predicted action trajectories against dataset ground truth **without executing actions or feeding back state**.

### 5.1 Start the Inference Server (Terminal 1)

```bash
python -m wall_x._vendor.harrix.serving.launch_serving \
  --env X2ROBOT \
  --port 32194 \
  --no-serialize-actions \
  model-config:server-model-config \
  --model-config.checkpoint-path /path/to/checkpoint \
  --model-config.train-config-path /path/to/my_config.yml \
  --model-config.action-horizon 10 \
  --model-config.robot-type desktop
```

### 5.2 Run the Open-Loop Script (Terminal 2)

```bash
python scripts/draw_openloop_plot.py \
  --uri ws://127.0.0.1:32194 \
  --dataset-root /path/to/libero_all \
  --train-config /path/to/my_config.yml \
  --episode-indices 0,1,2 \
  --save-dir ./openloop_plots
```

The script loads episodes from the LeRobot dataset, queries the WebSocket server frame by frame, and saves **predicted vs. ground-truth** action comparison plots (PNG) under `--save-dir`.

### 5.3 Common Options

| Option | Description |
|--------|-------------|
| `--episode-indices` | Comma-separated episode indices to evaluate |
| `--start-ratio` | Start position as a fraction of episode length (0.0 = from the beginning) |
| `--stride` | Frames between inference calls (default: `action_horizon`) |
| `--max-inferences` | Maximum inference requests per episode |

---

## Appendix

### Example Config Files

| File | Scenario |
|------|----------|
| `workspace/example/fintune_bus2602/libero.yml` | LIBERO single-arm fine-tuning |
| `workspace/example/fintune_bus2602/maniparena_example.yml` | ManipArena dual-arm fine-tuning |

### Helper Scripts

| Script | Purpose |
|--------|---------|
| `scripts/compute_norm_stats.py` | Compute dataset normalization statistics |
| `scripts/fake_inference.py` | Inference smoke test |
| `scripts/run_libero.sh` | Batch LIBERO simulation evaluation |
| `scripts/draw_openloop_plot.py` | LeRobot open-loop evaluation and plotting |
| `scripts/merge_sharded_weights.py` | Merge FSDP sharded checkpoints |

See [`scripts/README.md`](../scripts/README.md) for more script details.

### Configuration Notes

- `dof_config` maps to `predict_action_keys` (action dimensions the model predicts)
- `agent_pos_config` maps to `obs_action_keys` (state dimensions fed as observation)
- State and action DOF sets may differ, but each must sum to the same total as in norm stats
- `action_horizon`: number of future action steps per inference (flow-matching chunk length)

### Initialize from Qwen-2.5-VL (Optional)

To train from the raw VLM instead of Wall-OSS-0.5, download <https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct> and follow a `config_qact_from_vlm.yml`-style setup (see `README.md.old`).

---

## Quick Start Checklist

- [ ] Downloaded `wall-oss-0.5` and `Qwen2.5-VL-3B-Instruct`
- [ ] Prepared a LeRobot v3 dataset
- [ ] Wrote `config.yml` and replaced all `/path/to/*` placeholders
- [ ] Ran `compute_norm_stats.py` to generate norm stats
- [ ] Launched `train_fsdp.py` and completed fine-tuning
- [ ] Verified inference with `fake_inference.py` or `run_libero.sh`
- [ ] Generated open-loop comparison plots with `draw_openloop_plot.py`
