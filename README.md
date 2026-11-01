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
- [May 2026] We introduce [**WALL-WM: Carving World Action Modeling at the Event Joints**](https://arxiv.org/pdf/2606.01955), a World Action Model that couples future-video imagination with action prediction at semantic event boundaries.
- [May 2026] We introduce [**Wall-OSS-0.5: A Deployment-Ready VLA with Gradient-Bridged Pretraining**](https://arxiv.org/pdf/2605.30877v2), an open-source model for directly deployable real-robot manipulation and downstream adaptation.
- [Sept 2025] We introduce [**WALL-OSS: Igniting VLMs toward the Embodied Space**](https://arxiv.org/pdf/2509.11766), an end-to-end embodied foundation model that leverages large-scale multimodal pretraining to achieve embodiment-aware vision-language understanding, language-action association, and robust manipulation capability.

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
export FLASH_ATTN_CUDA_ARCHS=$(python -c 'import torch; print(f"{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}")')
MAX_JOBS=4 pip install flash-attn==2.8.3 --no-build-isolation
```

Qwen3.5 models also use `causal-conv1d` for the linear-attention layers
(`flash-linear-attention` is already in `requirements.txt`):

```bash
MAX_JOBS=8 pip install causal-conv1d==1.6.2.post1 --no-build-isolation
```

Both packages compile CUDA code when no matching prebuilt wheel is available.
The `nvcc` on `PATH` must match the CUDA version of the installed torch
(12.8 for `torch==2.10.0`). If the system CUDA toolkit differs, install a
matching compiler into the conda environment first:

```bash
conda install -c nvidia -c conda-forge cuda-nvcc-impl=12.8 cuda-nvcc-tools=12.8 cuda-cudart-dev=12.8 cuda-version=12.8
export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$(python -c 'import glob, nvidia, os; print(":".join(glob.glob(os.path.join(nvidia.__path__[0], "*", "include"))))')
```

Install DMuon, which is used by the default training configs:

```bash
pip install "dmuon @ git+https://github.com/X-Square-Robot/dmuon.git"
```

Install LeRobot (0.4.4 reads LeRobot v3.0 datasets):

```bash
pip install --no-deps lerobot==0.4.4
```

Use `--no-deps` for LeRobot so it does not override the Wall-X dependency
versions installed from `requirements.txt`; the LeRobot runtime packages Wall-X
needs are already listed there.

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

For a dual-arm desktop robot, add `--robot-type desktop --serialize-actions`.
Each msgpack request carries:

- `state`: `follow1_pos` / `follow2_pos` (left / right arm end-effector pose,
  7 values each: xyz, roll-pitch-yaw, gripper)
- `views`: base64-encoded JPEG images for `camera_front`, `camera_left`, and
  `camera_right`
- `instruction`: the task prompt

The response contains `follow1_pos` / `follow2_pos` as absolute poses with
shape `(action_horizon + 1, 7)`; row 0 is the input state. If the checkpoint
was trained on end-effector deltas expressed in the current gripper frame, set
`data.relative_action_frame: body` in the training config (the default `base`
adds deltas in the robot base frame).

`workspace/example/desktop/qwen3_5_2B_desktop.yml` is a complete dual-arm
desktop config for a Qwen3.5-2B checkpoint. The prompt text is configurable
through `data.system_prompt` (the system message) and `data.instruction_label`
(the label before the task text, default `Instruction`). LeRobot training and
inference read the same keys, so set them to whatever the checkpoint was
trained with.

### RoboDojo Serving

`wall_x.serving.robodojo` serves a checkpoint to
[RoboDojo](https://robodojo-benchmark.com) through the XPolicyLab WebSocket
protocol; the XPolicyLab `WallX` policy adapter drives it. Keyframe-memory
checkpoints are detected from their config and served with client-held memory
(see `wall_x/serving/robodojo/memory_protocol.py` for the request format):
each request carries its episode's memory state, so the server keeps nothing
between requests and `infer_batch` can answer several simulator environments
in one forward pass.

```bash
python -m wall_x.serving.robodojo.serve \
  --checkpoint-path <path/to/checkpoint> --norm-key ex_normal --port 43865 \
  --image-resize serving
```

- `--replan-steps` is how far the client advances per request (default 16);
  `--action-chunk-length` is how many actions a request returns (defaults to
  the full horizon for keyframe-memory checkpoints, to `--replan-steps`
  otherwise).
- `--image-resize serving` reproduces the two-stage resize the released
  RoboDojo numbers were produced with; `training` is the single resize of the
  data pipeline.
- The checkpoint's refiner settings (`vla_config.refiner.refine_loops`,
  `refine_step_scale`) are read from its config.
- Pose conversions follow the reference stack (float32 state 6D and returned
  quaternion); set `WALLX_POSE_FP64=1` for the float64 path.
- Reproducing the released numbers exactly also needs
  `flash-linear-attention==0.4.2` (pinned in `requirements.txt`) and the CUDA
  operators built at install time (`wall_x/model/core/ops/csrc`, see
  Installation). Without the build the PyTorch fallbacks run; they agree within
  floating-point tolerance but are not bit-identical.

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