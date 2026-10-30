# Flower-arrangement task (arrange_3_flowers) training guide

This guide fine-tunes **Wall-OSS-0.5** pretrained weights on an existing **LeRobot v3** dataset for a dual-arm flower-arrangement task.

Task: a dual-arm robot **arranges three flowers**, with flow action prediction. Example inference instruction: `arrange flower`.

---

## 1. Task and action space

### 1.1 Robot and cameras

| Item | Setting |
|------|---------|
| Robot type | Dual arm (ex001) |
| Camera keys | `observation.images.faceImg`, `leftImg`, `rightImg` |
| Training resolution | 448×448 (three views) |
| Suggested frame rate | 20 fps |

### 1.2 Action dimensions (26-D, aligned with Wall-OSS-0.5)

Training **predicts the arms only**. Base, lift, and head are not predicted. `action_padding` fills the vector out to 26 dimensions:

| Component | Dims | Notes |
|-----------|------|-------|
| Left-arm relative pose + gripper | 3 + 6 + 1 = 10 | `follow_left_ee_*_relative` + gripper |
| Right-arm relative pose + gripper | 3 + 6 + 1 = 10 | `follow_right_ee_*_relative` + gripper |
| **Effective action total** | **20** | The loader keeps the first 20 dims of the dataset vector |
| `action_padding` | 6 | Dummy padding; no gradient is propagated through it |
| **Model input total** | **26** | Matches the Wall-OSS-0.5 pretraining space |

> If `action` / `observation.state` in the LeRobot dataset is **26-D** (including extra DOFs), the training config truncates it through `dof_config` and the loader, and uses only the first 20 effective dual-arm dims.

The example training config is under `workspace/example/` (copy it and edit the paths).

---

## 2. Prerequisites

### 2.1 Environment

See `workspace/rtx5090/DEPLOY.md` or the official README. You need:

- Python 3.10 + wall-x + dmuon + **lerobot v0.4.4**
- flash-attn (required by the vision module for training and inference)
- A CUDA 12.8 toolchain (if you compile the wall-x operators)

### 2.2 Pretrained weights

Download these locally first:

| Resource | Use | Config field |
|----------|-----|--------------|
| [wall-oss-0.5](https://huggingface.co/X-Square-Robot/wall-oss-0.5) | Fine-tune starting point | `checkpoint.resume_from`, `model.config_path` |
| [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | Processor | `model.processor_path`, `model.pretrained_path` |

```bash
huggingface-cli download X-Square-Robot/wall-oss-0.5 --local-dir /path/to/wall-oss-0.5
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir /path/to/Qwen2.5-VL-3B-Instruct
```

### 2.3 Path conventions (edit as needed)

```bash
export REPO_ROOT=/path/to/wall-x
export DATA_ROOT=/path/to/data
export MODEL_ROOT=/path/to/model
export CKPT_ROOT=/path/to/ckpt
```

| Path | Description |
|------|-------------|
| `${DATA_ROOT}/arrange_3_flowers_lerobot` | LeRobot v3 dataset root |
| `${DATA_ROOT}/arrange_3_flowers_norm_stats.json` | Normalization stats |
| `${CKPT_ROOT}/arrange_3_flowers` | Training checkpoint output |
| `${MODEL_ROOT}/wall-oss-0.5` | Pretrained weights |
| `${MODEL_ROOT}/Qwen2.5-VL-3B-Instruct` | VLM processor |

---

## 2.4 Convert raw data (x2robot → LeRobot v3)

If the data is in the x2robot collection format, convert it first with `x2robot2lerobot.py` at the repository root:

```bash
cd "${REPO_ROOT}"

# 1. Copy and edit the conversion config (fill in src_path_list / output_path)
cp workspace/data_config/arrange_3_flowers_wrc_red_x2robot2lerobot.json /tmp/my_conversion.json

# 2. Run the conversion (requires lerobot==0.4.4)
python x2robot2lerobot.py --config /tmp/my_conversion.json
```

Templates and notes:

- `workspace/data_config/arrange_3_flowers_wrc_red_x2robot2lerobot.json`
- `workspace/data_config/arrange_3_flowers_wrc_red.yml` (example list of raw collection paths)

The conversion output directory is `data.lerobot_config.repo_id` in the training YAML.

---

## 3. LeRobot dataset requirements

Training data from outside this repo should be a local **LeRobot v3** dataset at the path in `data.lerobot_config.repo_id`.

### 3.1 Required fields

| Key | Description |
|-----|-------------|
| `observation.state` | Proprioceptive state vector (float32) |
| `action` | Action vector (float32) |
| `observation.images.faceImg` | Front camera |
| `observation.images.leftImg` | Left-wrist camera |
| `observation.images.rightImg` | Right-wrist camera |

`key_mappings` in the training YAML maps those keys to `face_view` / `left_wrist_view` / `right_wrist_view`:

```yaml
data:
  key_mappings:
    camera:
      observation.images.faceImg: face_view
      observation.images.leftImg: left_wrist_view
      observation.images.rightImg: right_wrist_view
    state: observation.state
    action: action
```

### 3.2 Shape and content

- `observation.state` / `action` should be **26-D**. If extra DOFs are present, training keeps the first 20 dual-arm dims according to `dof_config`.
- Camera resolution is unconstrained; training resizes every view to 448×448.
- The dataset must follow the LeRobot v3 layout (`meta/`, `data/`, and so on).

### 3.3 Compute normalization stats

Generate `norm_stats.json` before training:

```bash
cd "${REPO_ROOT}"

# Copy the flower-arrangement example under workspace/example/ to my_flowers_train.yml, edit the paths, then run:
python scripts/compute_norm_stats.py \
  --train_config /path/to/my_flowers_train.yml \
  --data_root "${DATA_ROOT}/arrange_3_flowers_lerobot" \
  --output_path "${DATA_ROOT}/arrange_3_flowers_norm_stats.json"
```

The script:

- Reads `dof_config` / `agent_pos_config` / `action_horizon` from the YAML
- Computes relative-pose stats for `_relative` keys the same way the training loader does
- Writes mean/std/q01/q99 for the 20 effective DOFs (one set for state, one for action)

---

## 4. Training config

Copy the flower-arrangement example YAML under `workspace/example/`, edit the paths, and use that copy.

**Required paths:**

```yaml
model:
  config_path: ${MODEL_ROOT}/wall-oss-0.5/config.json
  processor_path: ${MODEL_ROOT}/Qwen2.5-VL-3B-Instruct
  pretrained_path: ${MODEL_ROOT}/Qwen2.5-VL-3B-Instruct

data:
  dataset_type: lerobot
  lerobot_config:
    repo_id: ${DATA_ROOT}/arrange_3_flowers_lerobot
    root: null
  norm_stats_path: ${DATA_ROOT}/arrange_3_flowers_norm_stats.json

checkpoint:
  save_path: ${CKPT_ROOT}/arrange_3_flowers
  resume_from: ${MODEL_ROOT}/wall-oss-0.5/model.safetensors
```

**Task DOF config (the important part; usually leave it unchanged):**

```yaml
task:
  dof_config:
    follow_left_ee_cartesian_pos_relative: 3
    follow_left_ee_rotation_6D_relative: 6
    follow_left_gripper: 1
    follow_right_ee_cartesian_pos_relative: 3
    follow_right_ee_rotation_6D_relative: 6
    follow_right_gripper: 1
    action_padding: 6
  action_horizon: 32
  action_horizon_flow: 32
```

### 4.1 Key hyperparameters (reference values)

| Group | Parameter | Value |
|-------|-----------|-------|
| Optimizer | AdamW, lr | 5e-5 |
| Scheduler | cosine, warmup | 1000 steps |
| Max training steps | `num_training_steps` | 200000 |
| Per-GPU batch | `batch_size_per_gpu` | 4 |
| Gradient accumulation | `gradient_accumulation_steps` | 4 |
| Effective batch (4 GPUs) | 4 × 4 × 4 | **64** |
| Action horizon | `action_horizon` / `action_horizon_flow` | 32 |
| Mixed precision | bf16 + FSDP | on |
| Save interval | `save_interval` | every 2000 steps |
| W&B | `log_project` / `log_name` | `wall_oss_flowers` / `arrange_3_flowers` |

### 4.2 Data-loading notes

- `data.num_workers: 0`: a multiprocess DataLoader on a network filesystem can raise `Errno 95`. Keep this at 0.
- `data.train_test_split: 0.95`
- `data.max_length: 1024`
- `data.resolution`: 448 for all three views

---

## 5. Launch training

### 5.1 Smoke test (do this first)

Copy the example config and shorten the step count or resolution, or use `DEBUG=1` on the launch script:

```bash
cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES=0 \
  torchrun --nproc_per_node=1 \
    wall_x/trainer/fsdp_trainer/train_fsdp.py \
    --config /path/to/my_flowers_smoke.yml
```

```bash
DEBUG=1 CONFIG=/path/to/my_flowers_train.yml \
  bash workspace/example/run_oss_wandb_local.sh
```

`DEBUG=1` writes a temporary config (about 30 steps, with a shorter save interval).

### 5.2 Full training (single node, multiple GPUs)

```bash
cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
  torchrun --nproc_per_node=4 \
    wall_x/trainer/fsdp_trainer/train_fsdp.py \
    --config /path/to/my_flowers_train.yml \
    --log_to_file
```

### 5.3 Cluster launch script

```bash
CONFIG=/path/to/my_flowers_train.yml \
  bash workspace/example/run_oss_wandb_local.sh
```

The script:

- Detects `NPROC_PER_NODE` / `WORLD_SIZE` (multi-node is supported)
- Can optionally install the RDMA library (`SKIP_RDMA_INSTALL=1` skips it)
- Runs wandb offline when `WANDB_OFFLINE=1`
- Requires `CONDA_HOME`, `CONDA_ENV`, and `CUDA_HOME`

```bash
export CONDA_HOME=/path/to/miniconda3
export CONDA_ENV=wallx
export CUDA_HOME=/path/to/cuda-12.8
export WANDB_API_KEY=your_key
export WANDB_ENTITY=your_entity
```

---

## 6. Checkpoints and resume

### 6.1 Output layout

```
${CKPT_ROOT}/arrange_3_flowers/
├── 3_50000/
│   ├── model.safetensors      # ~17 GB full weights
│   ├── config.json
│   ├── config.yml             # snapshot of the training config
│   ├── norm_stats.json
│   ├── preprocessor_config.json
│   └── tokenizer*.json
├── 3_48000/
└── ...
```

### 6.2 Merge FSDP shards

If the save is an FSDP shard, merge it before inference:

```bash
python scripts/merge_sharded_weights.py \
  /path/to/sharded_checkpoint \
  /path/to/merged_checkpoint
```

### 6.3 Resume from a checkpoint

```yaml
checkpoint:
  resume_from: ${CKPT_ROOT}/arrange_3_flowers/3_50000/model.safetensors
```

---

## 7. Checks after training

### 7.1 Open-loop evaluation (optional)

```bash
bash scripts/run_serving.sh \
  --checkpoint-path ${CKPT_ROOT}/arrange_3_flowers/3_50000 \
  --train-config-path ${CKPT_ROOT}/arrange_3_flowers/3_50000/config.yml \
  --port 44660 \
  --robot-type ex001 \
  --serialize-actions

python scripts/draw_openloop_plot.py \
  --uri ws://127.0.0.1:44660 \
  --dataset-root ${DATA_ROOT}/arrange_3_flowers_lerobot \
  --train-config /path/to/my_flowers_train.yml \
  --episode-indices 0,1,2 \
  --save-dir ./openloop_plots
```

### 7.2 Real-robot deployment

See `workspace/rtx5090/DEPLOY.md`. Copy the checkpoint to the inference machine and start the WebSocket service.

---

## 8. Troubleshooting

### Q1: `num_workers > 0` raises Errno 95

**Fix:** Keep `data.num_workers: 0`.

### Q2: Norm-stats rank does not match training

**Symptom:** State or action stats are not 20-D.

**Fix:** Run `compute_norm_stats.py` with the same YAML as training, and keep `dof_config` identical to the example.

### Q3: The dataset is 26-D but training only wants 20-D

**Explanation:** This is expected. The loader keeps the first 20 dual-arm dims according to `dof_config`, and `action_padding(6)` pads the model input back to 26-D.

### Q4: Out of GPU memory

- Lower `batch_size_per_gpu`, or raise `gradient_accumulation_steps`
- For a smoke run, drop `resolution` to 256
- Use multi-GPU FSDP for the full run. A single GPU needs at least **48 GB** at 448 resolution.

### Q5: Wrong lerobot version

Pin the **v0.4.4** tag and install with `pip install --no-deps -e .`.

---

## 9. Related files

| File | Purpose |
|------|---------|
| `workspace/example/` | Example training / smoke YAML for the flower-arrangement task |
| `workspace/example/run_oss_wandb_local.sh` | Cluster or local training launcher |
| `scripts/compute_norm_stats.py` | Normalization stats |
| `wall_x/trainer/fsdp_trainer/train_fsdp.py` | FSDP training entry point |
| `scripts/merge_sharded_weights.py` | Merge an FSDP checkpoint |
| `scripts/draw_openloop_plot.py` | Open-loop evaluation |
| `workspace/rtx5090/DEPLOY.md` | Inference-machine deployment |

---

## 10. Minimum checklist

- [ ] wall-oss-0.5 and Qwen2.5-VL-3B-Instruct are downloaded
- [ ] The LeRobot v3 dataset is in place, and its keys match `key_mappings`
- [ ] `norm_stats.json` exists (20-D state and action)
- [ ] Every path in the training YAML is a real path
- [ ] The smoke run (smoke config or `DEBUG=1`) passed
- [ ] Full training has started, and the logs / wandb look healthy
- [ ] The checkpoint contains `model.safetensors` (about 17 GB) and `config.yml`
- [ ] The inference service can load the checkpoint and return actions
