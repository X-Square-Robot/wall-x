# Wall-X

<div align="left">

<a href="https://huggingface.co/x-square-robot">
  <img src="https://img.shields.io/badge/Hugging%20Face-x--square--robot-FFB000?style=for-the-badge&logo=huggingface&logoColor=000" alt="Hugging Face">
</a>
<a href="https://x2robot.com/en/research/68bc2cde8497d7f238dde690">
  <img src="https://img.shields.io/badge/Project-1E90FF?style=for-the-badge&logo=google-chrome&logoColor=fff" alt="Project Page">
</a>

<br/>
<img src="https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=fff" alt="Python 3.10">
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=fff" alt="PyTorch">
<img src="https://img.shields.io/badge/FlashAttention-0F9D58?style=flat&logo=nvidia&logoColor=fff" alt="FlashAttention">
<img src="https://img.shields.io/badge/LeRobot-222?style=flat&logo=huggingface&logoColor=ffd21e" alt="LeRobot">
<img src="https://img.shields.io/badge/CUDA-12.x-76B900?style=flat&logo=nvidia&logoColor=fff" alt="CUDA">
<img src="https://img.shields.io/badge/OS-Ubuntu%2022.04-E95420?style=flat&logo=ubuntu&logoColor=fff" alt="Ubuntu 22.04">

</div>

## Building General-Purpose Robots Based on Embodied Foundation Model

We are building embodied foundation models to capture and compress the world's
most valuable data: continuous, high-fidelity physical interaction.

By creating a direct feedback loop between model decisions and the body's lived
experience, we enable generalizable intelligence that understands not just how
the world works, but how to act effectively within it.

## Repository

This repository provides the training and inference code for the WALL series
open-source embodied foundation models. It includes LeRobot data preparation,
model configuration, flow-matching and FAST action branches, public serving and
evaluation utilities, and exported CUDA operator sources that compile during
package installation.

## News

- [June 2026] Wall-X 1.1.0 updates the open-source training and inference
  stack for Wall-OSS-0.5, including the public serving/evaluation runtime,
  DMuon training support, and install-time CUDA operator builds.
- [May 2026] We introduce [**WALL-WM: Carving World Action Modeling at the Event Joints**](https://x2robot.com/api/files/file/WALL-WM.pdf), a World Action Model that couples future-video imagination with action prediction at semantic event boundaries.
- [May 2026] We introduce [**Wall-OSS-0.5: A Deployment-Ready VLA with Gradient-Bridged Pretraining**](https://x2robot.com/api/files/file/wall_oss_05.pdf), an open-source model for directly deployable real-robot manipulation and downstream adaptation.
- [Sept 2025] We introduce [**WALL-OSS: Igniting VLMs toward the Embodied Space**](https://x2robot.com/en/research/68bc2cde8497d7f238dde690), an end-to-end embodied foundation model that leverages large-scale multimodal pretraining to achieve embodiment-aware vision-language understanding, language-action association, and robust manipulation capability.

## Models

- WALL-OSS-0.5: https://huggingface.co/x-square-robot/wall-oss-0.5
- WALL-OSS-FLOW-0.1: https://huggingface.co/x-square-robot/wall-oss-flow-0.1
- WALL-OSS-FLOW: https://huggingface.co/x-square-robot/wall-oss-flow
- WALL-OSS-FAST: https://huggingface.co/x-square-robot/wall-oss-fast

## Environment Setup

Create and activate a conda environment:

```bash
conda create --name wallx python=3.10
conda activate wallx
```

Install requirements:

```bash
pip install -r requirements.txt
MAX_JOBS=4 pip install flash-attn==2.7.4.post1 --no-build-isolation
```

Install DMuon, which is used by the default training configs:

```bash
pip install "dmuon @ git+https://github.com/X-Square-Robot/dmuon.git"
```

Install LeRobot:

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout c66cd401767e60baece16e1cf68da2824227e076
pip install --no-deps -e .
```

Use `--no-deps` for LeRobot so it does not override the Wall-X dependency
versions installed from `requirements.txt`.

Install Wall-X:

```bash
MAX_JOBS=8 pip install --no-build-isolation -e .
```

Public helper scripts live under `scripts/`; the examples below use the
repository-root form, such as `python scripts/fake_inference.py`.

The exported CUDA operator sources are included in
`wall_x/model/core/ops/csrc/`. `setup.py` builds them with PyTorch
`CUDAExtension` when Wall-X is installed. `ninja` is included in
`requirements.txt` for parallel builds, and `MAX_JOBS` controls compile
parallelism. `--no-build-isolation` is required so the build can use the torch
package already installed in the active environment.

## Training

### Finetune on LeRobot Datasets

Before training, see `workspace/README.md` for configuration details,
including:

- Training script configuration
- GPU setup
- Model and data paths
- Robot DOF configuration
- Training hyperparameters

Download the pretrained checkpoint, copy
`workspace/example/lerobot/qwen2_5_lerobot_template.yml`, replace the
placeholder paths, and launch training with:

```bash
python -m wall_x.trainer.fsdp_trainer.train_fsdp --config <path/to/config.yml>
```

For Wall-OSS-0.5 fine-tuning, normalization, LIBERO evaluation, and open-loop
WebSocket evaluation instructions, see `workspace/README.md`.

## Inference

### Basic Action Inference

For a minimal end-to-end example, run:

```bash
python scripts/fake_inference.py --checkpoint-path <path/to/checkpoint>
```

This script demonstrates how to:

- Load the Wall-OSS model with `Qwen2_5_VLMoEForAction.from_pretrained()`
- Prepare proprioceptive inputs, attention masks, and dataset specs
- Run inference in `validate` mode at bfloat16
- Validate output shape and check numerical stability

### Simulator Evaluation

Convenience launchers for closed-loop simulator evaluation live under
`scripts/`. LIBERO simulator setup is optional and documented with the helper
scripts; see `scripts/README.md`.

```bash
bash scripts/run_libero.sh <path/to/checkpoint>
```

### WebSocket Serving

Start a Wall-X WebSocket server with:

```bash
bash scripts/run_serving.sh \
  --checkpoint-path <path/to/checkpoint> \
  --train-config-path <path/to/config.yml> \
  --port 32195
```

The wrapper has no built-in checkpoint path. It returns raw model action chunks
by default, which is suitable for open-loop evaluation. Pass
`--serialize-actions` for clients that expect robot-serialized actions.

### Open-Loop WebSocket Evaluation

To compare predictions from a running Wall-X WebSocket server against LeRobot
ground truth, run:

```bash
python scripts/draw_openloop_plot.py \
  --uri ws://127.0.0.1:32195 \
  --dataset-root <path/to/lerobot_dataset> \
  --train-config <path/to/config.yml> \
  --episode-indices 0,1,2
```

## Join Our Community

Scan the QR code on WeChat to join the discussion group.

<img src="assets/QRcode_community.jpg" alt="QR Code" width="400">

## Cite Us

If you find WALL-OSS models useful, please cite:

```bibtex
@article{zhai2025igniting,
  title   = {Igniting VLMs Toward the Embodied Space},
  author  = {Zhai, Andy and Liu, Brae and Fang, Bruno and Cai, Chalse and Ma, Ellie and Yin, Ethan and Wang, Hao and Zhou, Hugo and Wang, James and Shi, Lights and Liang, Lucy and Wang, Make and Wang, Qian and Gan, Roy and Yu, Ryan and Li, Shalfun and Liu, Starrick and Chen, Sylas and Chen, Vincent and Xu, Zach},
  journal = {arXiv preprint arXiv:2509.11766},
  year    = {2025}
}
```
