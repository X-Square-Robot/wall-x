# Wall-X-OSS Usage Guide

This guide explains how to fine-tune, evaluate in simulation, and deploy on real robots with the **Wall-OSS-0.5** pretrained model on LeRobot-format datasets.

> All commands below assume you are in the **repository root** (`wall-x/`).

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

pip install -r requirements.txt
pip install "dmuon @ git+https://github.com/X-Square-Robot/dmuon.git"

git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout c66cd401767e60baece16e1cf68da2824227e076
pip install --no-deps -e .
cd -

# Optional: only needed for LIBERO simulator evaluation.
pip install -r requirements-libero.txt
mkdir -p third_party
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO

# Install wall-x (from repo root)
MAX_JOBS=8 pip install --no-build-isolation -e .
```

---

## Part 1: Download Wall-OSS-0.5 Weights

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

---

## Part 2: Simulation Tasks (LIBERO)

### 2.1 Download the LIBERO Dataset

Download the LeRobot-format LIBERO dataset from HuggingFace:

```bash
huggingface-cli download lerobot/libero \
  --repo-type dataset \
  --local-dir /path/to/libero_all
```

Dataset page: <https://huggingface.co/datasets/lerobot/libero>

### 2.2 Edit Your Training Config (`config.yml`)

Copy an example config and replace every `/path/to/*` placeholder:

```bash
cp workspace/example/libero.yml /path/to/my_libero_config.yml
```

**Required paths:**

```yaml
model:
  config_path: /path/to/wall-oss-0.5/config.json
  processor_path: /path/to/Qwen2.5-VL-3B-Instruct
  pretrained_path: /path/to/Qwen2.5-VL-3B-Instruct

data:
  lerobot_config:
    repo_id: /path/to/libero_all          # local LeRobot dataset root
  norm_stats_path: /path/to/libero_all_norm_stats.json
  key_mappings:                           # must match your dataset keys
    camera:
      observation.images.faceImg: face_view
      observation.images.rightImg: right_wrist_view
    state: observation.state
    action: action

checkpoint:
  save_path: /path/to/libero_training_output
  resume_from: /path/to/wall-oss-0.5/model.safetensors
```

See `workspace/example/libero.yml` for the full example. LIBERO uses a 7-dim single-arm action; pad to 26 dims with `action_padding` to match the Wall-OSS-0.5 pretraining space (see comments in the YAML).

### 2.3 Compute LIBERO Normalization Stats

Generate `norm_stats.json` from the dataset before training:

```bash
python scripts/compute_norm_stats.py \
  --train_config /path/to/my_libero_config.yml \
  --data_root /path/to/libero_all \
  --output_path /path/to/libero_all_norm_stats.json
```

Then set `data.norm_stats_path` to the generated JSON file.

### 2.4 Start Training

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 \
  python wall_x/trainer/fsdp_trainer/train_fsdp.py \
    --config /path/to/my_libero_config.yml

# Multi-GPU (recommended)
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  torchrun --nproc_per_node=4 \
    wall_x/trainer/fsdp_trainer/train_fsdp.py \
    --config /path/to/my_libero_config.yml
```

Logs and checkpoints are written to `checkpoint.save_path`. If training saved FSDP-sharded checkpoints, merge them before inference:

```bash
python scripts/merge_sharded_weights.py \
  /path/to/sharded_checkpoint \
  /path/to/merged_checkpoint
```

Single-GPU training needs at least **48 GB** VRAM. For multi-GPU runs, enable `distributed.use_fsdp: true`.

### 2.5 Run Inference (LIBERO Simulation)

`scripts/run_libero.sh` requires the optional LIBERO simulator stack:

```bash
pip install -r requirements-libero.txt
mkdir -p third_party
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
```

The script checks for LIBERO, robosuite, MuJoCo, PyOpenGL, BDDL, Gym, and h5py
before loading the model, so missing simulator dependencies fail fast with
installation instructions.

Use `scripts/run_libero.sh` for batch evaluation in the LIBERO simulator:

```bash
CHECKPOINT_PATH=/path/to/checkpoint \
TRAIN_CONFIG_PATH=/path/to/my_libero_config.yml \
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
| `ALL_SUITES=1` | Run all 4 standard suites sequentially |
| `TASK_INDICES` | Task indices, e.g. `0,1,2` |
| `CUDA_ID` | GPU index |

---

## Part 3: Real-Robot Tasks

### 3.1 Prepare Your Real-Robot Training Dataset

Convert your robot data to **LeRobot v3** format and note the local dataset root path (used for `data.lerobot_config.repo_id`).

Reference config: `workspace/example/maniparena_example.yml` (dual-arm, 448px, 3 cameras).

### 3.2 Edit Your Training Config (`config.yml`)

```bash
cp workspace/example/maniparena_example.yml /path/to/my_robot_config.yml
```

**Required paths:**

```yaml
model:
  config_path: /path/to/wall-oss-0.5/config.json
  processor_path: /path/to/Qwen2.5-VL-3B-Instruct
  pretrained_path: /path/to/Qwen2.5-VL-3B-Instruct

data:
  lerobot_config:
    repo_id: /path/to/your_robot_dataset
  norm_stats_path: /path/to/your_robot_norm_stats.json
  key_mappings:                           # match your dataset camera / state / action keys
    camera:
      observation.images.faceImg: face_view
      # ...
    state: observation.state
    action: action

checkpoint:
  save_path: /path/to/robot_training_output
  resume_from: /path/to/wall-oss-0.5/model.safetensors
```

`task.dof_config` defines **predicted action dimensions**; `task.agent_pos_config` defines **observation state dimensions**. The sum of each config must match `norm_stats.json`. If your robot has fewer than 26 DOF, pad with `action_padding` (see comments in the example YAML).

### 3.3 Compute Dataset Normalization Stats

```bash
python scripts/compute_norm_stats.py \
  --train_config /path/to/my_robot_config.yml \
  --data_root /path/to/your_robot_dataset \
  --output_path /path/to/your_robot_norm_stats.json
```

### 3.4 Start Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  torchrun --nproc_per_node=4 \
    wall_x/trainer/fsdp_trainer/train_fsdp.py \
    --config /path/to/my_robot_config.yml
```

To resume training, point `checkpoint.resume_from` to a checkpoint **directory** (not a single `.safetensors` file).

### 3.5 Start Inference (WebSocket Server)

Use `scripts/run_serving.sh` to launch the inference server for real-robot clients or open-loop evaluation:

```bash
bash scripts/run_serving.sh \
  --checkpoint-path /path/to/checkpoint \
  --train-config-path /path/to/my_robot_config.yml \
  --port 32195
```

By default this wrapper returns raw model action chunks, which is the expected
mode for open-loop evaluation. Pass `--serialize-actions` if your client expects
robot-serialized actions.

Or invoke the serving module directly (adjust parameters as needed):

```bash
export ENABLE_CUDA_GRAPH=True
export ENABLE_EXPERIMENTAL_INFERENCE_ENGINE=True

CKPT_PATH=/path/to/checkpoint

python -m wall_x._vendor.harrix.serving.launch_serving \
  --env X2ROBOT \
  --port 32195 \
  --no-serialize-actions \
  model-config:server-model-config \
  --model-config.checkpoint-path "$CKPT_PATH" \
  --model-config.train-config-path /path/to/my_robot_config.yml \
  --model-config.action-horizon 32 \
  --model-config.robot-action-interpolate-multiplier 1 \
  --model-config.robot-action-end-ratio 1.0 \
  --model-config.robot-type desktop
```

Clients connect at `ws://127.0.0.1:32195` to send observations and receive predicted actions.

### 3.6 Plot Open-Loop Results to Verify the Server

Open-loop evaluation compares model-predicted action trajectories against dataset ground truth **without executing actions or feeding back state**.

**Terminal 1:** Keep the inference server from section3.5 running.

**Terminal 2:** Run the open-loop plotting script:

```bash
python scripts/draw_openloop_plot.py \
  --uri ws://127.0.0.1:32195 \
  --dataset-root /path/to/your_robot_dataset \
  --train-config /path/to/my_robot_config.yml \
  --episode-indices 0,1,2 \
  --save-dir ./openloop_plots
```

The script loads episodes from the LeRobot dataset, queries the WebSocket server frame by frame, and saves **predicted vs. ground-truth** comparison plots (PNG) under `--save-dir`.
`--dataset-root` and `--train-config` are both required; pass the same training
config used for the checkpoint so dataset layouts and action dimensions match.

Common options:

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
| `workspace/example/libero.yml` | LIBERO single-arm fine-tuning |
| `workspace/example/maniparena_example.yml` | Real-robot dual-arm fine-tuning |

### Helper Scripts

| Script | Purpose |
|--------|---------|
| `scripts/compute_norm_stats.py` | Compute dataset normalization statistics |
| `scripts/fake_inference.py` | Inference smoke test |
| `scripts/run_libero.sh` | Batch LIBERO simulation evaluation |
| `scripts/run_serving.sh` | Launch WebSocket inference server |
| `scripts/draw_openloop_plot.py` | Open-loop evaluation and plotting |
| `scripts/merge_sharded_weights.py` | Merge FSDP sharded checkpoints |

See [`scripts/README.md`](../scripts/README.md) for more script details.

### Training Hyperparameters

| Parameter | Description | Default (libero example) |
|-----------|-------------|--------------------------|
| `hyperparams.batch_size_per_gpu` | Batch size per GPU | 4 |
| `hyperparams.gradient_accumulation_steps` | Gradient accumulation steps | 4 |
| `hyperparams.optimizer.learning_rate` | Learning rate | 5e-5 |
| `hyperparams.num_epoch` | Number of training epochs | 100 |
| `distributed.use_fsdp` | Multi-GPU FSDP training | true |
| `logging.save_interval` | Checkpoint save interval (steps) | 2000 |

### Quick Start Checklist

**Simulation (LIBERO)**

- [ ] Downloaded `wall-oss-0.5` and `Qwen2.5-VL-3B-Instruct`
- [ ] Downloaded the `lerobot/libero` dataset
- [ ] Wrote `config.yml` and replaced all `/path/to/*` placeholders
- [ ] Ran `compute_norm_stats.py` to generate norm stats
- [ ] Launched `train_fsdp.py` and completed fine-tuning
- [ ] Ran simulation evaluation with `run_libero.sh`

**Real Robot**

- [ ] Prepared a LeRobot v3 real-robot dataset
- [ ] Wrote `config.yml` and configured `key_mappings` / DOF
- [ ] Ran `compute_norm_stats.py` to generate norm stats
- [ ] Launched `train_fsdp.py` and completed fine-tuning
- [ ] Started the inference server with `run_serving.sh`
- [ ] Verified server output with `draw_openloop_plot.py`
