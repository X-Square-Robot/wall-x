# 插花任务（arrange_3_flowers）训练指南

本文档描述在已有 **LeRobot v3 格式数据集** 上，基于 **Wall-OSS-0.5** 预训练权重进行双臂插花任务微调的流程。

任务：双臂机器人 **三花插花**，flow 动作预测，推理指令示例：`arrange flower`。

---

## 1. 任务与动作空间

### 1.1 机器人与相机

| 项目 | 配置 |
|------|------|
| 机器人类型 | 双臂（ex001） |
| 相机键名 | `observation.images.faceImg`、`leftImg`、`rightImg` |
| 训练分辨率 | 448×448（三视角） |
| 建议帧率 | 20 fps |

### 1.2 动作维度（26 维，对齐 Wall-OSS-0.5）

训练**只预测双臂**，底盘/升降/头部不预测，用 `action_padding` 补齐到 26 维：

| 分量 | 维度 | 说明 |
|------|------|------|
| 左臂相对位姿 + 夹爪 | 3 + 6 + 1 = 10 | `follow_left_ee_*_relative` + gripper |
| 右臂相对位姿 + 夹爪 | 3 + 6 + 1 = 10 | `follow_right_ee_*_relative` + gripper |
| **有效动作合计** | **20** | loader 从数据集向量中取前 20 维 |
| `action_padding` | 6 | 虚拟填充，loss 不反传 |
| **模型输入合计** | **26** | 与 Wall-OSS-0.5 预训练空间一致 |

> 若 LeRobot 数据集中 `action` / `observation.state` 为 **26 维**（含额外 DOF），训练配置通过 `dof_config` + loader 截断，只使用前 20 维有效双臂数据。

示例训练配置位于 `workspace/example/`（复制后按实际路径修改）。

---

## 2. 前置准备

### 2.1 环境

参考 `workspace/rtx5090/DEPLOY.md` 或官方 README，需安装：

- Python 3.10 + wall-x + dmuon + **lerobot v0.4.4**
- flash-attn（训练/推理 vision 模块需要）
- CUDA 12.8 编译工具链（若需编译 wall-x 算子）

### 2.2 预训练权重

需提前下载到本地：

| 资源 | 用途 | 配置项 |
|------|------|--------|
| [wall-oss-0.5](https://huggingface.co/X-Square-Robot/wall-oss-0.5) | 微调起点 | `checkpoint.resume_from`、`model.config_path` |
| [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | Processor | `model.processor_path`、`model.pretrained_path` |

```bash
huggingface-cli download X-Square-Robot/wall-oss-0.5 --local-dir /path/to/wall-oss-0.5
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir /path/to/Qwen2.5-VL-3B-Instruct
```

### 2.3 路径约定（按需修改）

```bash
export REPO_ROOT=/path/to/wall-x
export DATA_ROOT=/path/to/data
export MODEL_ROOT=/path/to/model
export CKPT_ROOT=/path/to/ckpt
```

| 路径 | 说明 |
|------|------|
| `${DATA_ROOT}/arrange_3_flowers_lerobot` | LeRobot v3 数据集根目录 |
| `${DATA_ROOT}/arrange_3_flowers_norm_stats.json` | 归一化统计 |
| `${CKPT_ROOT}/arrange_3_flowers` | 训练 checkpoint 输出 |
| `${MODEL_ROOT}/wall-oss-0.5` | 预训练权重 |
| `${MODEL_ROOT}/Qwen2.5-VL-3B-Instruct` | VLM processor |

---

## 2.4 原始数据转换（x2robot → LeRobot v3）

若数据为 x2robot 采集格式，先用仓库根目录的 `x2robot2lerobot.py` 转换：

```bash
cd "${REPO_ROOT}"

# 1. 复制并编辑转换配置（填入 src_path_list / output_path）
cp workspace/data_config/arrange_3_flowers_wrc_red_x2robot2lerobot.json /tmp/my_conversion.json

# 2. 运行转换（需 lerobot==0.4.4）
python x2robot2lerobot.py --config /tmp/my_conversion.json
```

配置模板与说明见：

- `workspace/data_config/arrange_3_flowers_wrc_red_x2robot2lerobot.json`
- `workspace/data_config/arrange_3_flowers_wrc_red.yml`（原始采集路径列表示例）

转换输出目录即训练 YAML 中的 `data.lerobot_config.repo_id`。

---

## 3. LeRobot 数据集要求

外部提供的训练数据应为 **LeRobot v3** 本地数据集，放置于 `data.lerobot_config.repo_id` 所指路径。

### 3.1 必需字段

| 键 | 说明 |
|----|------|
| `observation.state` | 本体状态向量（float32） |
| `action` | 动作向量（float32） |
| `observation.images.faceImg` | 正面相机 |
| `observation.images.leftImg` | 左腕相机 |
| `observation.images.rightImg` | 右腕相机 |

训练 YAML 中的 `key_mappings` 将上述键映射为 `face_view` / `left_wrist_view` / `right_wrist_view`：

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

### 3.2 维度与内容

- `observation.state` / `action` 建议为 **26 维**；若含额外 DOF，训练时按 `dof_config` 自动截取前 20 维双臂分量
- 相机原始分辨率不限，训练时统一 resize 到 448×448
- 数据集需符合 LeRobot v3 目录结构（含 `meta/`、`data/` 等）

### 3.3 计算归一化统计

在训练前必须生成 `norm_stats.json`：

```bash
cd "${REPO_ROOT}"

# 复制 workspace/example/ 下插花示例配置为 my_flowers_train.yml 并改好路径后执行
python scripts/compute_norm_stats.py \
  --train_config /path/to/my_flowers_train.yml \
  --data_root "${DATA_ROOT}/arrange_3_flowers_lerobot" \
  --output_path "${DATA_ROOT}/arrange_3_flowers_norm_stats.json"
```

脚本会：

- 读取 YAML 中的 `dof_config` / `agent_pos_config` / `action_horizon`
- 对 `_relative` 键做与训练 loader 一致的相对位姿统计
- 输出 20 维有效 DOF 的 mean/std/q01/q99（state 与 action 各一份）

---

## 4. 训练配置

复制 `workspace/example/` 下插花任务示例 YAML，修改路径后使用。

**必填路径示例：**

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

**任务 DOF 配置（核心，一般无需改动）：**

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

### 4.1 关键超参（参考值）

| 类别 | 参数 | 值 |
|------|------|-----|
| 优化器 | AdamW, lr | 5e-5 |
| 调度器 | cosine, warmup | 1000 steps |
| 训练步数上限 | `num_training_steps` | 200000 |
| 每卡 batch | `batch_size_per_gpu` | 4 |
| 梯度累积 | `gradient_accumulation_steps` | 4 |
| 有效 batch（4 卡） | 4 × 4 × 4 | **64** |
| 动作 horizon | `action_horizon` / `action_horizon_flow` | 32 |
| 混合精度 | bf16 + FSDP | 开启 |
| 保存间隔 | `save_interval` | 每 2000 step |
| W&B | `log_project` / `log_name` | `wall_oss_flowers` / `arrange_3_flowers` |

### 4.2 数据加载注意

- `data.num_workers: 0`：网络文件系统 + 多进程 DataLoader 可能触发 `Errno 95`，建议保持 0
- `data.train_test_split: 0.95`
- `data.max_length: 1024`
- `data.resolution`：三视角均为 448

---

## 5. 启动训练

### 5.1 冒烟测试（推荐先做）

复制示例配置并减小步数/分辨率，或使用启动脚本的 `DEBUG=1` 模式：

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

`DEBUG=1` 会自动生成临时配置（约 30 step、缩短 save 间隔）。

### 5.2 正式训练（单机多卡）

```bash
cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
  torchrun --nproc_per_node=4 \
    wall_x/trainer/fsdp_trainer/train_fsdp.py \
    --config /path/to/my_flowers_train.yml \
    --log_to_file
```

### 5.3 集群启动脚本

```bash
CONFIG=/path/to/my_flowers_train.yml \
  bash workspace/example/run_oss_wandb_local.sh
```

脚本特性：

- 自动检测 `NPROC_PER_NODE` / `WORLD_SIZE`（支持多机）
- 可选 RDMA 库安装（`SKIP_RDMA_INSTALL=1` 跳过）
- wandb 可通过 `WANDB_OFFLINE=1` 离线运行
- 需设置 `CONDA_HOME`、`CONDA_ENV`、`CUDA_HOME`

```bash
export CONDA_HOME=/path/to/miniconda3
export CONDA_ENV=wallx
export CUDA_HOME=/path/to/cuda-12.8
export WANDB_API_KEY=your_key
export WANDB_ENTITY=your_entity
```

---

## 6. Checkpoint 与续训

### 6.1 输出目录结构

```
${CKPT_ROOT}/arrange_3_flowers/
├── 3_50000/
│   ├── model.safetensors      # ~17GB 全量权重
│   ├── config.json
│   ├── config.yml             # 训练配置快照
│   ├── norm_stats.json
│   ├── preprocessor_config.json
│   └── tokenizer*.json
├── 3_48000/
└── ...
```

### 6.2 FSDP 分片合并

若保存为 FSDP shard，推理前需合并：

```bash
python scripts/merge_sharded_weights.py \
  /path/to/sharded_checkpoint \
  /path/to/merged_checkpoint
```

### 6.3 从 checkpoint 续训

```yaml
checkpoint:
  resume_from: ${CKPT_ROOT}/arrange_3_flowers/3_50000/model.safetensors
```

---

## 7. 训练后验证

### 7.1 开环评估（可选）

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

### 7.2 真机部署

参考 `workspace/rtx5090/DEPLOY.md`，将 checkpoint 部署到推理机并启动 WebSocket 服务。

---

## 8. 常见问题

### Q1: `num_workers > 0` 报 Errno 95

**处理**：保持 `data.num_workers: 0`。

### Q2: norm stats 维度与训练不匹配

**现象**：state/action 统计维度不是 20。

**处理**：确认 `compute_norm_stats.py` 与训练使用同一份 YAML，且 `dof_config` 与示例一致。

### Q3: 数据集 26 维但训练只要 20 维

**说明**：预期行为。loader 按 `dof_config` 截取前 20 维双臂分量，`action_padding(6)` 在模型侧补齐到 26 维。

### Q4: 显存不足

- 减小 `batch_size_per_gpu` 或增大 `gradient_accumulation_steps`
- 冒烟阶段可将 `resolution` 降至 256
- 正式训练建议多卡 FSDP；单卡至少 **48GB** 显存（448 分辨率）

### Q5: lerobot 版本不对

固定使用 **v0.4.4** tag，`pip install --no-deps -e .` 安装。

---

## 9. 相关文件索引

| 文件 | 用途 |
|------|------|
| `workspace/example/` | 插花任务示例训练 / 冒烟 YAML |
| `workspace/example/run_oss_wandb_local.sh` | 集群/本地训练启动脚本 |
| `scripts/compute_norm_stats.py` | 归一化统计 |
| `wall_x/trainer/fsdp_trainer/train_fsdp.py` | FSDP 训练入口 |
| `scripts/merge_sharded_weights.py` | FSDP checkpoint 合并 |
| `scripts/draw_openloop_plot.py` | 开环评估 |
| `workspace/rtx5090/DEPLOY.md` | 推理机部署 |

---

## 10. 最小检查清单

- [ ] wall-oss-0.5 与 Qwen2.5-VL-3B-Instruct 已下载
- [ ] LeRobot v3 数据集已就位，键名与 `key_mappings` 一致
- [ ] `norm_stats.json` 已生成（20 维 state/action）
- [ ] 训练 YAML 中所有路径已替换为实际路径
- [ ] 冒烟训练（smoke / `DEBUG=1`）通过
- [ ] 正式训练启动，日志 / wandb 正常
- [ ] checkpoint 含 `model.safetensors`（约 17GB）及 `config.yml`
- [ ] 推理服务可加载 checkpoint 并返回动作
