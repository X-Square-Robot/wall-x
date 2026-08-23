# RTX 5090 推理机部署实测记录

本文档记录 **2026-08-23** 在某台 **NVIDIA RTX 5090 / Ubuntu 24.04** 推理机上的实际安装与推理验证。路径配置见同目录 [`env.sh`](./env.sh)。

通用部署流程见 [`../DEPLOY.md`](../DEPLOY.md)。

## 环境信息

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA GeForce RTX 5090 (32GB) |
| 代码仓 | `/home/x2eng-agent/Documents/zane/code/wall-oss-05/wall-x` |
| 分支 | `fix/lerobot-dof-layout` |
| 模型 checkpoint | `/home/x2eng-agent/Documents/zane/model/arrange_3_flowers_wrc_red/3_50000` |
| Conda 环境 | `/home/x2eng-agent/Documents/zane/conda_envs/wallx` |
| 服务端口 | `44660` |

## 一、创建 Conda 环境

> **注意**：本机 `miniforge3/envs/` 目录结构异常，无法直接 `conda create -n wallx`。
> 在 `local/env.sh` 中使用 `--prefix` 路径。

```bash
conda create --prefix /home/x2eng-agent/Documents/zane/conda_envs/wallx python=3.10 -y
conda activate /home/x2eng-agent/Documents/zane/conda_envs/wallx
```

或使用一键脚本：

```bash
bash workspace/rtx5090/install.sh
```

## 二、安装依赖（手动步骤）

**务必使用 conda 环境内的 pip**，避免 pyenv 劫持：

```bash
PIP=/home/x2eng-agent/Documents/zane/conda_envs/wallx/bin/pip
REPO=/home/x2eng-agent/Documents/zane/code/wall-oss-05/wall-x

# 1. 基础依赖
$PIP install -r $REPO/requirements.txt

# 2. dmuon
$PIP install "dmuon @ git+https://github.com/X-Square-Robot/dmuon.git"

# 3. LeRobot 0.4.4（不要用 README 里的 c66cd401 commit）
git clone https://github.com/huggingface/lerobot.git /home/x2eng-agent/Documents/zane/code/lerobot
cd /home/x2eng-agent/Documents/zane/code/lerobot
git checkout v0.4.4
$PIP install --no-deps -e .
cd -

# 4. CUDA 12.8 编译工具链（系统默认是 CUDA 13.2，与 torch cu128 不匹配）
conda install -p /home/x2eng-agent/Documents/zane/conda_envs/wallx \
  -y -c conda-forge cuda-nvcc=12.8 cuda-cudart-dev=12.8

export CUDA_HOME=/home/x2eng-agent/Documents/zane/conda_envs/wallx
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CPATH
export FLASH_ATTN_CUDA_ARCHS=120   # RTX 5090 = sm_120

# 5. flash-attn（推理必需，否则无法加载 qwen2_5 模型）
MAX_JOBS=4 $PIP install flash-attn==2.8.3 --no-build-isolation

# 6. wall-x 本体（编译 CUDA 算子）
cd $REPO
MAX_JOBS=8 $PIP install --no-build-isolation -e .
```

### 验证安装

```bash
/home/x2eng-agent/Documents/zane/conda_envs/wallx/bin/python -c "
import torch, flash_attn, lerobot, wall_x
import wall_x.model.core.ops._cuda_ext_bin as cuda_ext
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('lerobot', lerobot.__version__)
print('cuda_ext', cuda_ext.__file__)
"
```

## 三、部署模型服务

模型目录需包含至少：

- `model.safetensors`
- `config.json` / `config.yml`
- `preprocessor_config.json`
- `norm_stats.json`
- tokenizer 相关文件

启动脚本：

```bash
bash workspace/rtx5090/run_server.sh
```

关键环境变量（`run_server.sh` 内已设置）：

| 变量 | 推荐值 | 说明 |
|------|--------|------|
| `HOST` | `0.0.0.0` | conda 激活后 `HOST` 会被覆盖为编译器三元组，必须显式设置 |
| `WALLX_VISION_ATTN_IMPLEMENTATION` | `flash_attention_2` | 需要 flash-attn |
| `ENABLE_FAST_PREPROCESS` | `false` | **必须为 false**，否则 image token 数量不匹配 |
| `ENABLE_CUDA_GRAPH` | `true` | 5090 上已验证可用，warmup 后延迟显著下降 |
| `ENABLE_EXPERIMENTAL_INFERENCE_ENGINE` | `true` | 可与 CUDA Graph 同时开启 |

### 健康检查

```bash
curl http://127.0.0.1:44660/healthz
# 期望输出: OK
```

WebSocket 地址：`ws://<推理机IP>:44660`

## 四、推理验证结果

| 检查项 | 结果 |
|--------|------|
| 模型加载 | 通过（~1 分钟冷启动） |
| `/healthz` | `OK` |
| WebSocket 推理（ex001 + 三相机） | 通过，首次 ~1.5s，warmup 后 ~0.22s |
| 返回字段 | `follow1_pos`, `follow2_pos`, `head_pos`, `lift`, `velocity_decomposed_odom` |

客户端 payload 示例（base64 图像模式）：

```python
{
  "state": {
    "follow1_pos": [7维 float],
    "follow2_pos": [7维 float],
  },
  "views": {
    "camera_front": "<base64 JPEG>",
    "camera_left": "<base64 JPEG>",
    "camera_right": "<base64 JPEG>",
  },
  "instruction": "arrange flower",
}
```

## 五、本机踩坑记录

### 1. `conda create -n wallx` 失败

```
CondaValueError: Environment paths cannot be immediately nested...
```

**解决**：在 `local/env.sh` 中使用 `--prefix` 路径。

### 2. `pip install` 装到了 pyenv

**现象**：包装到了 `~/.pyenv/versions/3.10.20` 而非 conda 环境。

**解决**：始终用 `$CONDA_ENV/bin/pip` 或 `$CONDA_ENV/bin/python -m pip`。

### 3. wall-x / flash-attn 编译报 CUDA 版本不匹配

```
CUDA version (13.2) mismatches ... (12.8)
```

**解决**：安装 conda-forge 的 `cuda-nvcc=12.8`，并设置 `CUDA_HOME` 指向 conda 环境。

### 4. flash-attn 编译报 `cuda_runtime.h: No such file`

**解决**：

```bash
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CPATH
```

### 5. 推理报 `Image features and image tokens do not match`

**解决**：设置 `export ENABLE_FAST_PREPROCESS=false` 后重启服务。

### 6. 服务绑定到错误 host

**现象**：日志显示 `Host: x86_64-conda-linux-gnu`。

**解决**：启动前 `export HOST=0.0.0.0`。

## 六、与原始文档的差异

| 原始文档 | 本机实际做法 |
|----------|-------------|
| `conda create -n wallx` | `conda create --prefix .../conda_envs/wallx` |
| lerobot 未指定版本 | `git checkout v0.4.4` |
| 未包含 flash-attn | **必须安装** flash-attn==2.8.3 |
| 未包含 CUDA 工具链 | 需 conda 安装 cuda-nvcc 12.8 |
| 未包含 serving 配置 | 使用 `workspace/rtx5090/run_server.sh` |

## 七、后台运行（可选）

```bash
nohup bash workspace/rtx5090/run_server.sh > /tmp/wallx_serve_5090.log 2>&1 &
tail -f /tmp/wallx_serve_5090.log
```

停止服务：

```bash
pkill -f 'launch_serving.*44660'
```
