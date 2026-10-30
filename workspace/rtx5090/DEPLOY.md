# Deploying Wall-X on an RTX 5090

This guide deploys the Wall-X model service on an **NVIDIA RTX 5090 (Blackwell, sm_120)** inference machine. It is based on a verified Wall-X 1.1.0 + torch 2.10 (cu128) setup and does not depend on any one machine's paths or accounts.

Related files live under `workspace/rtx5090/`:

| File | Description |
|------|-------------|
| [README.md](./README.md) | Directory overview and quick start |
| [install.sh](./install.sh) | One-shot install script |
| [run_server.sh](./run_server.sh) | One-shot server launch |
| [env.example](./env.example) | Path-config template; copy to `local/env.sh` |
| [local/NOTES.md](./local/NOTES.md) | Example of machine-specific bring-up notes |

## 1. Scope

| Item | Requirement |
|------|-------------|
| GPU | NVIDIA RTX 5090 (compute capability 12.0) |
| OS | Ubuntu 22.04 / 24.04 (recommended) |
| Python | 3.10 |
| Driver | NVIDIA driver with CUDA 12.x support (550 or newer recommended) |
| Repository | [X-Square-Robot/wall-x](https://github.com/X-Square-Robot/wall-x) |

Prepare the following before you start:

- A local copy of the Wall-X code (clone or fork)
- A fine-tuned or inference checkpoint directory
- Enough disk space (the environment is about 15–20 GB; a full Wall-OSS checkpoint is about **17 GB each**, mostly `model.safetensors`)

## 2. Paths and environment variables

Put machine-specific paths in `workspace/rtx5090/local/env.sh` (copy from `env.example`):

```bash
export CONDA_SH=/path/to/miniforge3/etc/profile.d/conda.sh
export CONDA_ENV=wallx                              # or /path/to/conda/envs/wallx
export LEROBOT_DIR=/path/to/lerobot
export CHECKPOINT_PATH=/path/to/checkpoint          # contains model.safetensors
export PORT=44660
export CUDA_ID=0
```

The manual steps below also use:

```bash
export REPO_ROOT=/path/to/wall-x   # Wall-X repository root
```

After the environment is activated, `$CONDA_PREFIX` points at the current conda environment. The pip and python commands below use that variable.

## 3. Create the conda environment

```bash
conda create -n wallx python=3.10 -y
conda activate wallx
```

> **Alternative:** If `conda create -n wallx` reports `Environment paths cannot be immediately nested...`,
> the local conda directory layout is unusual. Create the environment at an explicit prefix instead:
> `conda create --prefix /path/to/conda/envs/wallx python=3.10 -y`,
> then activate it with `conda activate /path/to/conda/envs/wallx`.

**Use the pip and python inside the conda environment** so pyenv or another tool does not intercept them:

```bash
export PIP="${CONDA_PREFIX}/bin/pip"
export PY="${CONDA_PREFIX}/bin/python"
which pip    # should be ${CONDA_PREFIX}/bin/pip
which python # should be ${CONDA_PREFIX}/bin/python
```

### One-shot install (recommended)

After `local/env.sh` is filled in, run this from the repository root:

```bash
bash workspace/rtx5090/install.sh
```

## 4. Install dependencies (manual steps)

### 4.1 Base Python dependencies

```bash
cd "${REPO_ROOT}"
${PIP} install -r requirements.txt
```

### 4.2 dmuon

```bash
${PIP} install "dmuon @ git+https://github.com/X-Square-Robot/dmuon.git"
```

### 4.3 LeRobot (pin v0.4.4)

Wall-X 1.1.0 is paired with **lerobot 0.4.4**. The `c66cd401` commit mentioned in the README is not compatible with this Python 3.10 environment. Use the tag:

```bash
git clone https://github.com/huggingface/lerobot.git "${LEROBOT_DIR}"
cd "${LEROBOT_DIR}"
git checkout v0.4.4
${PIP} install --no-deps -e .
cd "${REPO_ROOT}"
```

### 4.4 CUDA toolchain (required on RTX 5090)

RTX 5090 machines often ship a **CUDA 13.x** toolchain, while torch 2.10 in `requirements.txt` is **cu128**. Compiling wall-x or flash-attn directly then fails with:

```
CUDA version (13.x) mismatches ... (12.8)
```

Install an nvcc that matches torch, inside the conda environment:

```bash
conda install -p "${CONDA_PREFIX}" -y -c conda-forge cuda-nvcc=12.8 cuda-cudart-dev=12.8

export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export CPATH="${CUDA_HOME}/targets/x86_64-linux/include:${CPATH:-}"
export FLASH_ATTN_CUDA_ARCHS=120   # RTX 5090 = sm_120
```

### 4.5 flash-attn (required for inference)

The vision module defaults to `flash_attention_2`. Without flash-attn, the `qwen2_5` model cannot be loaded.

```bash
MAX_JOBS=4 ${PIP} install flash-attn==2.8.3 --no-build-isolation
```

If the build reports `cuda_runtime.h: No such file`, confirm that `CPATH` is set (section 4.4).

### 4.6 Install wall-x (compile CUDA operators)

```bash
cd "${REPO_ROOT}"
MAX_JOBS=8 ${PIP} install --no-build-isolation -e .
```

### 4.7 Verify the install

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

Expect torch `2.10.0+cu128`, `cuda: True`, and no import errors.

## 5. Prepare the checkpoint

An inference checkpoint directory should contain at least:

| File | Description |
|------|-------------|
| `model.safetensors` | Model weights |
| `config.json` | Model architecture config |
| `config.yml` | Training config (serving reads DOF and camera mappings from it) |
| `preprocessor_config.json` | Image preprocessor config |
| `norm_stats.json` | Normalization stats (or pass them with `--model-config.norm-key`) |
| `tokenizer.json` and related files | Tokenizer files |

Set that directory as `CHECKPOINT_PATH` in `local/env.sh`.

## 6. Start the model service

### 6.1 Recommended environment variables (RTX 5090)

| Variable | Recommended value | Notes |
|----------|-------------------|-------|
| `HOST` | `0.0.0.0` | After conda activation, `HOST` may be overwritten with a compiler triple. Set it explicitly. |
| `WALLX_VISION_ATTN_IMPLEMENTATION` | `flash_attention_2` | Requires flash-attn |
| `ENABLE_FAST_PREPROCESS` | `false` | Keep this off. Turning it on can mismatch the image-token count. |
| `ENABLE_CUDA_GRAPH` | `true` (`--enable-cuda-graph`) | CUDA Graph acceleration, verified on the 5090. The first inference is slow; latency drops after warmup. |
| `ENABLE_EXPERIMENTAL_INFERENCE_ENGINE` | `true` (`--enable-experimental-engine`) | Experimental inference engine. Verified on the 5090 together with CUDA Graph. |

### 6.2 Launch command

**Option A: helper script** (requires `local/env.sh`)

```bash
bash workspace/rtx5090/run_server.sh
```

**Option B: manual launch**

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

Arguments:

- `--robot-type`: robot platform (`ex001` / `desktop` / `turtle`)
- `--serialize-actions`: return actions the robot can execute directly. For debugging, use `--no-serialize-actions` instead.
- `--model-config.norm-key`: key inside the checkpoint's `norm_stats.json`

### 6.3 Health check

A cold start takes about 1–2 minutes. After the log prints `Server started on 0.0.0.0:<PORT>`:

```bash
curl http://127.0.0.1:${PORT}/healthz
# expected: OK
```

WebSocket URL: `ws://<inference-machine-ip>:<PORT>`

### 6.4 Run in the background

```bash
nohup bash workspace/rtx5090/run_server.sh > /tmp/wallx_serve.log 2>&1 &
tail -f /tmp/wallx_serve.log
```

Stop the service:

```bash
pkill -f "launch_serving.*${PORT}"
```

## 7. Inference check

### 7.1 Client payload (ex001 + base64 images)

```python
{
    "state": {
        "follow1_pos": [7 floats],  # left-arm pose + gripper
        "follow2_pos": [7 floats],  # right-arm pose + gripper
    },
    "views": {
        "camera_front": "<base64 JPEG>",
        "camera_left": "<base64 JPEG>",
        "camera_right": "<base64 JPEG>",
    },
    "instruction": "task instruction text",
}
```

After the connection opens, the server first sends metadata (msgpack). The client then sends the observation above, and the server returns serialized actions.

### 7.2 Expected response (`--serialize-actions` + ex001)

Common fields: `follow1_pos`, `follow2_pos`, `head_pos`, `lift`, `velocity_decomposed_odom`.

### 7.3 Reference performance (RTX 5090, 32 GB)

| Metric | Reference |
|--------|-----------|
| Cold start (including model load) | ~1–2 min |
| One flow inference (first call, including warmup) | ~1.5 s |
| One flow inference (after warmup) | ~0.2–0.3 s |
| GPU memory | ~8–12 GB (depends on the checkpoint and config) |

> These latencies were measured on an RTX 5090 with `ENABLE_CUDA_GRAPH` and `ENABLE_EXPERIMENTAL_INFERENCE_ENGINE` enabled. With acceleration off, steady-state latency is about 1.5 s.

## 8. Troubleshooting

### Q1: CUDA version mismatch

**Symptom:** Building wall-x or flash-attn reports `CUDA version (13.x) mismatches ... (12.8)`.

**Fix:** Install `cuda-nvcc=12.8` inside the conda environment and point `CUDA_HOME` at that environment. Do not use the system `/usr/local/cuda`, which is often 13.x.

### Q2: flash-attn build cannot find headers

**Symptom:** `fatal error: cuda_runtime.h: No such file or directory`.

**Fix:**

```bash
export CPATH="${CUDA_HOME}/targets/x86_64-linux/include:${CPATH:-}"
```

### Q3: Image-token mismatch at inference

**Symptom:**

```
ValueError: Image features and image tokens do not match: tokens: 730, features 768
```

**Fix:** Set `ENABLE_FAST_PREPROCESS=false` and restart the service.

### Q4: Server binds the wrong host

**Symptom:** The log shows `Host: x86_64-conda-linux-gnu`, and clients outside the machine cannot connect.

**Fix:** `export HOST=0.0.0.0` before launch.

### Q5: `ModuleNotFoundError: No module named 'flash_attn'`

**Fix:** Install flash-attn as in section 4.5. Do not work around this with `WALLX_VISION_ATTN_IMPLEMENTATION=sdpa`; the sdpa path has compatibility issues on the current transformers version.

### Q6: pip installed into the wrong environment

**Symptom:** Packages land in pyenv or the system Python, and imports fail inside the conda environment.

**Fix:** Always use `${CONDA_PREFIX}/bin/pip` or `${CONDA_PREFIX}/bin/python -m pip` after `conda activate wallx`.

### Q7: Wrong LeRobot version

**Symptom:** Data loading or the norm-stats script fails with an incompatible API.

**Fix:** Pin **lerobot v0.4.4** and install it with `pip install --no-deps -e .`.

## 9. Differences from the official README

| Official README | RTX 5090 inference recommendation |
|-----------------|-----------------------------------|
| lerobot commit `c66cd401` | Use the **v0.4.4** tag |
| flash-attn not emphasized | **Must install** flash-attn==2.8.3 |
| CUDA toolchain not described | Install **cuda-nvcc 12.8** from conda to match torch cu128 |
| Serving environment variables not described | On the 5090, set `ENABLE_FAST_PREPROCESS=false` and enable CUDA Graph plus the experimental engine |
| `conda create -n wallx` | If that fails, use `conda create --prefix <path>` |

## 10. Minimum checklist

After deployment, confirm in order:

- [ ] `nvidia-smi` shows the RTX 5090
- [ ] `torch`, `flash_attn`, `wall_x`, and `lerobot` import inside the conda environment
- [ ] The checkpoint directory is complete, and `model.safetensors` has a plausible size (a full Wall-OSS checkpoint is about 17 GB, depending on the model)
- [ ] The server log has no ERROR, and `/healthz` returns `OK`
- [ ] A WebSocket test observation receives an action response
- [ ] The real-robot or simulator client can connect to `ws://<IP>:<PORT>`
