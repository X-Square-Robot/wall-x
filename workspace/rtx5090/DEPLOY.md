# Wall-X 在 RTX 5090 上的模型部署指南

本文档适用于在 **NVIDIA RTX 5090（Blackwell, sm_120）** 推理机上部署 Wall-X 模型服务。内容基于 Wall-X 1.1.0 + torch 2.10 (cu128) 的实际验证经验整理，不绑定某一台机器的路径或账号。

相关文件位于 `workspace/rtx5090/`：

| 文件 | 说明 |
|------|------|
| [README.md](./README.md) | 目录说明与快速开始 |
| [install.sh](./install.sh) | 一键安装脚本 |
| [run_server.sh](./run_server.sh) | 一键启动服务 |
| [env.example](./env.example) | 路径配置模板 → 复制为 `local/env.sh` |
| [local/NOTES.md](./local/NOTES.md) | 某台机器的实测记录（示例） |

## 1. 适用范围

| 项目 | 要求 |
|------|------|
| GPU | NVIDIA RTX 5090（Compute Capability 12.0） |
| 操作系统 | Ubuntu 22.04 / 24.04（推荐） |
| Python | 3.10 |
| 驱动 | 支持 CUDA 12.x 的 NVIDIA 驱动（建议 ≥ 550） |
| 代码仓 | [X-Square-Robot/wall-x](https://github.com/X-Square-Robot/wall-x) |

部署前请自行准备：

- Wall-X 代码（clone 或 fork 后的本地副本）
- 微调/推理用 checkpoint 目录
- 足够的磁盘空间（环境约 15–20GB；Wall-OSS 类全量 checkpoint 约 **17GB/个**，`model.safetensors` 占绝大部分）

## 2. 路径与环境变量约定

推荐将机器相关路径写入 `workspace/rtx5090/local/env.sh`（从 `env.example` 复制）：

```bash
export CONDA_SH=/path/to/miniforge3/etc/profile.d/conda.sh
export CONDA_ENV=wallx                              # 或 /path/to/conda/envs/wallx
export LEROBOT_DIR=/path/to/lerobot
export CHECKPOINT_PATH=/path/to/checkpoint          # 含 model.safetensors
export PORT=44660
export CUDA_ID=0
```

下文手动步骤中还会用到：

```bash
export REPO_ROOT=/path/to/wall-x   # Wall-X 仓库根目录
```

激活环境后，`$CONDA_PREFIX` 会自动指向当前 conda 环境路径，下文 pip/python 均基于此变量。

## 3. 创建 Conda 环境

```bash
conda create -n wallx python=3.10 -y
conda activate wallx
```

> **备选**：若 `conda create -n wallx` 报 `Environment paths cannot be immediately nested...`，
> 说明本机 conda 目录结构异常，可改用独立路径创建：
> `conda create --prefix /path/to/conda/envs/wallx python=3.10 -y`，
> 再用 `conda activate /path/to/conda/envs/wallx` 激活。

**务必使用 conda 环境内的 pip/python**，避免被 pyenv 等工具劫持：

```bash
export PIP="${CONDA_PREFIX}/bin/pip"
export PY="${CONDA_PREFIX}/bin/python"
which pip    # 应指向 ${CONDA_PREFIX}/bin/pip
which python # 应指向 ${CONDA_PREFIX}/bin/python
```

### 一键安装（推荐）

配置好 `local/env.sh` 后，在仓库根目录执行：

```bash
bash workspace/rtx5090/install.sh
```

## 4. 安装依赖（手动步骤）

### 4.1 基础 Python 依赖

```bash
cd "${REPO_ROOT}"
${PIP} install -r requirements.txt
```

### 4.2 dmuon

```bash
${PIP} install "dmuon @ git+https://github.com/X-Square-Robot/dmuon.git"
```

### 4.3 LeRobot（固定 v0.4.4）

Wall-X 1.1.0 与 **lerobot 0.4.4** 配套。README 中的 `c66cd401` commit 与当前 Python 3.10 环境不兼容，请使用 tag：

```bash
git clone https://github.com/huggingface/lerobot.git "${LEROBOT_DIR}"
cd "${LEROBOT_DIR}"
git checkout v0.4.4
${PIP} install --no-deps -e .
cd "${REPO_ROOT}"
```

### 4.4 CUDA 编译工具链（RTX 5090 关键步骤）

RTX 5090 机器通常预装 **CUDA 13.x** 工具链，而 `requirements.txt` 中的 torch 2.10 为 **cu128**。直接编译 wall-x / flash-attn 会报错：

```
CUDA version (13.x) mismatches ... (12.8)
```

在 conda 环境中安装与 torch 匹配的 nvcc：

```bash
conda install -p "${CONDA_PREFIX}" -y -c conda-forge cuda-nvcc=12.8 cuda-cudart-dev=12.8

export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export CPATH="${CUDA_HOME}/targets/x86_64-linux/include:${CPATH:-}"
export FLASH_ATTN_CUDA_ARCHS=120   # RTX 5090 = sm_120
```

### 4.5 flash-attn（推理必需）

Vision 模块默认使用 `flash_attention_2`，未安装 flash-attn 时无法加载 `qwen2_5` 模型。

```bash
MAX_JOBS=4 ${PIP} install flash-attn==2.8.3 --no-build-isolation
```

若编译报 `cuda_runtime.h: No such file`，确认已设置 `CPATH`（见 4.4 节）。

### 4.6 安装 wall-x（编译 CUDA 算子）

```bash
cd "${REPO_ROOT}"
MAX_JOBS=8 ${PIP} install --no-build-isolation -e .
```

### 4.7 验证安装

```bash
${PY} -c "
import torch, flash_attn, lerobot, wall_x
import wall_x.model.core.ops._cuda_ext_bin as cuda_ext
print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())
print('lerobot:', lerobot.__version__)
print('flash_attn: ok')
print('cuda_ext:', cuda_ext.__file__)
"
```

期望输出中 `torch` 版本为 `2.10.0+cu128`，`cuda: True`，且无 import 报错。

## 5. 准备 Checkpoint

推理用的 checkpoint 目录至少应包含：

| 文件 | 说明 |
|------|------|
| `model.safetensors` | 模型权重 |
| `config.json` | 模型结构配置 |
| `config.yml` | 训练配置（serving 读取 DOF / 相机映射等） |
| `preprocessor_config.json` | 图像预处理器配置 |
| `norm_stats.json` | 归一化统计（或通过 `--model-config.norm-key` 指定） |
| `tokenizer.json` 等 | 分词器相关文件 |

将 checkpoint 路径写入 `local/env.sh` 的 `CHECKPOINT_PATH`。

## 6. 启动模型服务

### 6.1 推荐环境变量（RTX 5090）

| 变量 | 推荐值 | 说明 |
|------|--------|------|
| `HOST` | `0.0.0.0` | 激活 conda 后 `HOST` 可能被覆盖为编译器三元组，必须显式设置 |
| `WALLX_VISION_ATTN_IMPLEMENTATION` | `flash_attention_2` | 依赖 flash-attn |
| `ENABLE_FAST_PREPROCESS` | `false` | 建议关闭；开启后可能出现 image token 数量不匹配 |
| `ENABLE_CUDA_GRAPH` | `true`（`--enable-cuda-graph`） | CUDA Graph 加速，5090 上已验证可用；首次推理较慢，warmup 后延迟显著下降 |
| `ENABLE_EXPERIMENTAL_INFERENCE_ENGINE` | `true`（`--enable-experimental-engine`） | 实验性推理引擎，5090 上已验证可与 CUDA Graph 同时开启 |

### 6.2 启动命令

**方式 A：一键脚本**（需已配置 `local/env.sh`）

```bash
bash workspace/rtx5090/run_server.sh
```

**方式 B：手动启动**

```bash
conda activate wallx

export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export PYTHON_BIN="${CONDA_PREFIX}/bin/python"
export HOST="0.0.0.0"
export ENABLE_FAST_PREPROCESS=false
export WALLX_VISION_ATTN_IMPLEMENTATION=flash_attention_2
export ENABLE_CUDA_GRAPH=1
export ENABLE_EXPERIMENTAL_ENGINE=1

cd "${REPO_ROOT}"
bash scripts/run_serving.sh \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --train-config-path "${CHECKPOINT_PATH}/config.yml" \
  --port "${PORT}" \
  --cuda-id "${CUDA_ID}" \
  --robot-type ex001 \
  --serialize-actions \
  --enable-cuda-graph \
  --enable-experimental-engine \
  -- \
  --model-config.norm-key ex_normal
```

参数说明：

- `--robot-type`：按机器人平台选择（`ex001` / `desktop` / `turtle`）
- `--serialize-actions`：返回机器人可直接执行的序列化动作；调试时可改为 `--no-serialize-actions`
- `--model-config.norm-key`：与 checkpoint 中 `norm_stats.json` 的 key 对应

### 6.3 健康检查

模型冷启动约需 1–2 分钟。日志出现 `Server started on 0.0.0.0:<PORT>` 后：

```bash
curl http://127.0.0.1:${PORT}/healthz
# 期望输出: OK
```

WebSocket 地址：`ws://<推理机IP>:<PORT>`

### 6.4 后台运行

```bash
nohup bash workspace/rtx5090/run_server.sh > /tmp/wallx_serve.log 2>&1 &
tail -f /tmp/wallx_serve.log
```

停止服务：

```bash
pkill -f "launch_serving.*${PORT}"
```

## 7. 推理验证

### 7.1 客户端 payload 格式（ex001 + base64 图像）

```python
{
    "state": {
        "follow1_pos": [7维 float],  # 左臂位姿 + 夹爪
        "follow2_pos": [7维 float],  # 右臂位姿 + 夹爪
    },
    "views": {
        "camera_front": "<base64 JPEG>",
        "camera_left": "<base64 JPEG>",
        "camera_right": "<base64 JPEG>",
    },
    "instruction": "任务描述文本",
}
```

连接后服务端会先发送 metadata（msgpack），客户端再发送上述 observation，服务端返回序列化动作。

### 7.2 期望返回（`--serialize-actions` + ex001）

常见字段：`follow1_pos`、`follow2_pos`、`head_pos`、`lift`、`velocity_decomposed_odom`。

### 7.3 参考性能（RTX 5090, 32GB）

| 指标 | 参考值 |
|------|--------|
| 冷启动（含模型加载） | ~1–2 min |
| 单次 flow 推理（首次，含 warmup） | ~1.5 s |
| 单次 flow 推理（warmup 后） | ~0.2–0.3 s |
| 显存占用 | ~8–12 GB（视 checkpoint 与配置而定） |

> 以上延迟为开启 `ENABLE_CUDA_GRAPH` + `ENABLE_EXPERIMENTAL_INFERENCE_ENGINE` 后在 RTX 5090 上的实测参考值；关闭加速时稳态延迟约 ~1.5 s。

## 8. 常见问题

### Q1: CUDA 版本不匹配

**现象**：编译 wall-x 或 flash-attn 时报 `CUDA version (13.x) mismatches ... (12.8)`。

**处理**：在 conda 环境内安装 `cuda-nvcc=12.8`，设置 `CUDA_HOME` 指向该环境，不要用系统 `/usr/local/cuda`（多为 13.x）。

### Q2: flash-attn 编译失败，缺少头文件

**现象**：`fatal error: cuda_runtime.h: No such file or directory`。

**处理**：

```bash
export CPATH="${CUDA_HOME}/targets/x86_64-linux/include:${CPATH:-}"
```

### Q3: 推理报 image token 不匹配

**现象**：

```
ValueError: Image features and image tokens do not match: tokens: 730, features 768
```

**处理**：设置 `ENABLE_FAST_PREPROCESS=false` 并重启服务。

### Q4: 服务绑定到错误 Host

**现象**：日志显示 `Host: x86_64-conda-linux-gnu`，外部无法连接。

**处理**：启动前 `export HOST=0.0.0.0`。

### Q5: `ModuleNotFoundError: No module named 'flash_attn'`

**处理**：按 4.5 节安装 flash-attn；不要用 `WALLX_VISION_ATTN_IMPLEMENTATION=sdpa` 绕过，当前 transformers 版本下 sdpa 路径存在兼容问题。

### Q6: pip 装到了错误环境

**现象**：包装入 pyenv 或系统 Python，conda 环境中 import 失败。

**处理**：始终使用 `${CONDA_PREFIX}/bin/pip` 或 `${CONDA_PREFIX}/bin/python -m pip`（需先 `conda activate wallx`）。

### Q7: LeRobot 版本不对

**现象**：数据加载或 norm stats 脚本报 API 不兼容。

**处理**：固定使用 `lerobot v0.4.4`，`pip install --no-deps -e .` 安装。

## 9. 与官方 README 的差异摘要

| 官方 README | RTX 5090 推理部署建议 |
|-------------|----------------------|
| lerobot commit `c66cd401` | 使用 **v0.4.4** tag |
| 未强调 flash-attn | **必须安装** flash-attn==2.8.3 |
| 未说明 CUDA 工具链 | 需 conda 安装 **cuda-nvcc 12.8** 匹配 torch cu128 |
| 未说明 serving 环境变量 | 5090 上建议 `ENABLE_FAST_PREPROCESS=false`；加速项建议开启 CUDA Graph + 实验性引擎 |
| `conda create -n wallx` | 若失败，改用 `conda create --prefix <path>` |

## 10. 最小检查清单

部署完成后，按顺序确认：

- [ ] `nvidia-smi` 可见 RTX 5090
- [ ] conda 环境中 `torch` / `flash_attn` / `wall_x` / `lerobot` 均可 import
- [ ] checkpoint 目录文件齐全，`model.safetensors` 大小正常（Wall-OSS 全量权重约 17GB，视模型规模而定）
- [ ] 服务启动日志无 ERROR，`/healthz` 返回 `OK`
- [ ] WebSocket 发送测试 observation 能收到动作响应
- [ ] 真机/仿真客户端能连上 `ws://<IP>:<PORT>`
