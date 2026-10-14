# Wall-X

Wall-X 开源仓库提供 WALL 系列开源具身基础模型的训练与推理代码，当前公开范围包含 LeRobot 数据链路、Qwen2.5 模型、Wall-OSS-0.5 训练配置、最小 fake inference、LIBERO 仿真评测、open-loop WebSocket 评测，以及安装时编译的导出 CUDA 算子源码。

## 最新动态

- 2026 年 6 月：Wall-X 1.1.0 更新 Wall-OSS-0.5 训练与推理栈，包含公开 serving/evaluation runtime、DMuon 训练支持，以及安装期 CUDA 算子编译。
- 2026 年 5 月：发布 [WALL-WM: Carving World Action Modeling at the Event Joints](https://arxiv.org/pdf/2606.01955)。
- 2026 年 5 月：发布 [Wall-OSS-0.5: A Deployment-Ready VLA with Gradient-Bridged Pretraining](https://arxiv.org/pdf/2605.30877v2)。
- 2025 年 9 月：发布 [WALL-OSS: Igniting VLMs toward the Embodied Space](https://arxiv.org/pdf/2509.11766)。

## 模型

- WALL-OSS-0.5: https://huggingface.co/x-square-robot/wall-oss-0.5
- WALL-OSS-FLOW-0.1: https://huggingface.co/x-square-robot/wall-oss-flow-0.1
- WALL-OSS-FLOW: https://huggingface.co/x-square-robot/wall-oss-flow
- WALL-OSS-FAST: https://huggingface.co/x-square-robot/wall-oss-fast

## 环境安装

创建 Python 环境：

```bash
conda create --name wallx python=3.10
conda activate wallx
```

安装依赖：

```bash
pip install -r requirements.txt
export FLASH_ATTN_CUDA_ARCHS=$(python -c 'import torch; print(f"{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}")')
MAX_JOBS=4 pip install flash-attn==2.8.3 --no-build-isolation
```

Qwen3.5 模型的线性注意力层还需要 `causal-conv1d`（`flash-linear-attention` 已包含在 `requirements.txt` 中）：

```bash
MAX_JOBS=8 pip install causal-conv1d==1.6.2.post1 --no-build-isolation
```

没有匹配的预编译 wheel 时，这两个包都会现场编译 CUDA 代码。`PATH` 上的 `nvcc` 版本必须与已安装 torch 的 CUDA 版本一致（`torch==2.10.0` 对应 12.8）。如果系统 CUDA 版本不同，先在 conda 环境里安装匹配的编译器：

```bash
conda install -c nvidia -c conda-forge cuda-nvcc-impl=12.8 cuda-nvcc-tools=12.8 cuda-cudart-dev=12.8 cuda-version=12.8
export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$(python -c 'import glob, nvidia, os; print(":".join(glob.glob(os.path.join(nvidia.__path__[0], "*", "include"))))')
```

安装默认训练配置使用的 DMuon：

```bash
pip install "dmuon @ git+https://github.com/X-Square-Robot/dmuon.git"
```

安装 LeRobot（0.4.4 支持 LeRobot v3.0 格式数据集）：

```bash
pip install --no-deps lerobot==0.4.4
```

LeRobot 使用 `--no-deps` 安装，避免覆盖 `requirements.txt` 中固定的 Wall-X 依赖版本；Wall-X 需要的 LeRobot 运行时依赖已经列在 `requirements.txt` 里。

安装 Wall-X：

```bash
MAX_JOBS=8 pip install --no-build-isolation -e .
```

公开辅助脚本位于 `scripts/` 下，文档中的命令保持仓库根目录形式，例如 `python scripts/fake_inference.py` 或 `bash scripts/run_libero.sh`。

导出的 CUDA 算子源码位于 `wall_x/model/core/ops/csrc/`。安装 Wall-X 时，`setup.py` 会通过 PyTorch `CUDAExtension` 编译这些算子。`requirements.txt` 已包含 `ninja` 用于并行编译，`MAX_JOBS` 控制编译并发。这里需要 `--no-build-isolation`，这样构建过程会复用当前环境里已经安装好的 torch。

## 使用 LeRobot 训练

仓库内提供了 Qwen2.5 + LeRobot 训练模板：

```text
workspace/example/lerobot/qwen2_5_lerobot_template.yml
```

复制该模板，替换数据集、归一化统计、模型和 checkpoint 输出路径后运行：

```bash
python -m wall_x.trainer.fsdp_trainer.train_fsdp --config <path/to/config.yml>
```

完整 Wall-OSS-0.5 微调、归一化统计、LIBERO 评测和 open-loop WebSocket 评测说明见 `workspace/README.md`。

## Fake Inference

`scripts/fake_inference.py` 提供最小推理链路检查，用于确认 checkpoint、训练配置、processor、normalizer 和模型调用可以连通：

```bash
python scripts/fake_inference.py --checkpoint-path <path/to/checkpoint>
```

如果 checkpoint 旁边没有 `config.yml` 或 `config.yaml`，可以显式指定训练配置：

```bash
python scripts/fake_inference.py \
  --checkpoint-path <path/to/checkpoint> \
  --train-config-path <path/to/config.yml>
```

## LIBERO 评测

使用封装脚本运行 LIBERO 评测：

```bash
bash scripts/run_libero.sh <path/to/checkpoint>
```

常用环境变量：

```bash
CHECKPOINT_PATH=<path/to/checkpoint>
TRAIN_CONFIG_PATH=<path/to/config.yml>
TASK_SUITE_NAME=libero_spatial
TASK_INDICES=0,1,2
NUM_TRIALS_PER_TASK=50
CUDA_ID=0
SMOKE=1
MAX_INFER_TIMES=52
```

`MAX_INFER_TIMES` 可以不传；默认会按 suite 使用与 LIBERO 评测对齐的 action chunk 数：spatial 22、object 28、goal 30、libero_10 52、libero_90 40。

也可以直接调用 Python 入口：

```bash
python scripts/infer_libero.py \
  --checkpoint-path <path/to/checkpoint> \
  --task-suite-name libero_spatial \
  --num-trials-per-task 50 \
  --driver-mode in_process
```

## WebSocket 服务

```bash
bash scripts/run_serving.sh \
  --checkpoint-path <path/to/checkpoint> \
  --train-config-path <path/to/config.yml> \
  --port 32195
```

默认返回模型原始 action chunk，适合 open-loop 评测；需要按机器人格式序列化的动作时加 `--serialize-actions`。

双臂 desktop 机器人使用 `--robot-type desktop --serialize-actions`。每个 msgpack 请求包含：

- `state`：`follow1_pos` / `follow2_pos`（左 / 右臂末端位姿，各 7 维：xyz、roll-pitch-yaw、夹爪）
- `views`：`camera_front`、`camera_left`、`camera_right` 三路 base64 编码的 JPEG 图像
- `instruction`：任务指令

返回的 `follow1_pos` / `follow2_pos` 为绝对位姿，形状为 `(action_horizon + 1, 7)`，第 0 行即输入状态。若训练时末端增量定义在当前夹爪坐标系下，在训练配置中设置 `data.relative_action_frame: body`（默认 `base` 表示在机器人基座坐标系下相加）。

`workspace/example/desktop/qwen3_5_2B_desktop.yml` 是 Qwen3.5-2B checkpoint 的完整双臂 desktop 配置。提示词可通过 `data.system_prompt`（系统消息）和 `data.instruction_label`（任务文本前的标签，默认 `Instruction`）配置；LeRobot 训练和推理读取同一组字段，需与 checkpoint 训练时保持一致。

## Open-Loop WebSocket 评测

```bash
python scripts/draw_openloop_plot.py \
  --uri ws://127.0.0.1:32195 \
  --dataset-root <path/to/lerobot_dataset> \
  --train-config <path/to/config.yml> \
  --episode-indices 0,1,2
```

## 社区

欢迎扫码加入社区讨论群。

<img src="assets/QRcode_community.jpg" alt="QR Code" width="400">

## 引用

如果 WALL-OSS 对你的研究或项目有帮助，请引用：

```bibtex
@article{zhai2025igniting,
  title   = {Igniting VLMs Toward the Embodied Space},
  author  = {Zhai, Andy and Liu, Brae and Fang, Bruno and Cai, Chalse and Ma, Ellie and Yin, Ethan and Wang, Hao and Zhou, Hugo and Wang, James and Shi, Lights and Liang, Lucy and Wang, Make and Wang, Qian and Gan, Roy and Yu, Ryan and Li, Shalfun and Liu, Starrick and Chen, Sylas and Chen, Vincent and Xu, Zach},
  journal = {arXiv preprint arXiv:2509.11766},
  year    = {2025}
}
```