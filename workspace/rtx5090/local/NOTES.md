# RTX 5090 inference-machine bring-up notes

These notes record the install and inference check run on **2026-08-23** on one **NVIDIA RTX 5090 / Ubuntu 24.04** inference machine. Copy [`../env.example`](../env.example) to `local/env.sh` and fill in this machine's paths (`env.sh` is not tracked).

The general procedure is in [`../DEPLOY.md`](../DEPLOY.md).

## Environment

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 5090 (32 GB) |
| Repository | `<path/to/wall-x>` |
| Branch | `feat/arrange-3-flowers` |
| Model checkpoint | `<path/to/checkpoint>` |
| Conda environment | `<path/to/conda/envs/wallx>` |
| Service port | `44660` |

## 1. Create the conda environment

> **Note:** On this machine the `miniforge3/envs/` layout is unusual, so `conda create -n wallx` does not work.
> Use a `--prefix` path in `local/env.sh`.

```bash
conda create --prefix <path/to/conda/envs/wallx> python=3.10 -y
conda activate <path/to/conda/envs/wallx>
```

Or use the helper script:

```bash
bash workspace/rtx5090/install.sh
```

## 2. Install dependencies (manual steps)

**Use the pip inside the conda environment** so pyenv does not intercept it:

```bash
PIP=<path/to/conda/envs/wallx>/bin/pip
REPO=<path/to/wall-x>

# 1. Base dependencies
$PIP install -r $REPO/requirements.txt

# 2. dmuon
$PIP install "dmuon @ git+https://github.com/X-Square-Robot/dmuon.git"

# 3. LeRobot 0.4.4 (do not use the c66cd401 commit from the README)
git clone https://github.com/huggingface/lerobot.git <path/to/lerobot>
cd <path/to/lerobot>
git checkout v0.4.4
$PIP install --no-deps -e .
cd -

# 4. CUDA 12.8 toolchain (the system default is CUDA 13.2, which does not match torch cu128)
conda install -p <path/to/conda/envs/wallx> \
  -y -c conda-forge cuda-nvcc=12.8 cuda-cudart-dev=12.8

export CUDA_HOME=<path/to/conda/envs/wallx>
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CPATH
export FLASH_ATTN_CUDA_ARCHS=120   # RTX 5090 = sm_120

# 5. flash-attn (required for inference; otherwise the qwen2_5 model cannot load)
MAX_JOBS=4 $PIP install flash-attn==2.8.3 --no-build-isolation

# 6. wall-x itself (compile CUDA operators)
cd $REPO
MAX_JOBS=8 $PIP install --no-build-isolation -e .
```

### Verify the install

```bash
<path/to/conda/envs/wallx>/bin/python -c "
import torch, flash_attn, lerobot, wall_x
import wall_x.model.core.ops._cuda_ext_bin as cuda_ext
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('lerobot', lerobot.__version__)
print('cuda_ext', cuda_ext.__file__)
"
```

## 3. Deploy the model service

The model directory must contain at least:

- `model.safetensors`
- `config.json` / `config.yml`
- `preprocessor_config.json`
- `norm_stats.json`
- tokenizer files

Launch script:

```bash
bash workspace/rtx5090/run_server.sh
```

Important environment variables (already set inside `run_server.sh`):

| Variable | Recommended value | Notes |
|----------|-------------------|-------|
| `HOST` | `0.0.0.0` | After conda activation, `HOST` is overwritten with a compiler triple. Set it explicitly. |
| `WALLX_VISION_ATTN_IMPLEMENTATION` | `flash_attention_2` | Requires flash-attn |
| `ENABLE_FAST_PREPROCESS` | `false` | **Must be false**, or the image-token count will not match |
| `ENABLE_CUDA_GRAPH` | `true` | Verified on the 5090. Latency drops sharply after warmup. |
| `ENABLE_EXPERIMENTAL_INFERENCE_ENGINE` | `true` | Can be enabled together with CUDA Graph |

### Health check

```bash
curl http://127.0.0.1:44660/healthz
# expected: OK
```

WebSocket URL: `ws://<inference-machine-ip>:44660`

## 4. Inference check

| Check | Result |
|-------|--------|
| Model load | Passed (~1 minute cold start) |
| `/healthz` | `OK` |
| WebSocket inference (ex001 + three cameras) | Passed; first call ~1.5 s, ~0.22 s after warmup |
| Returned fields | `follow1_pos`, `follow2_pos`, `head_pos`, `lift`, `velocity_decomposed_odom` |

Example client payload (base64 images):

```python
{
  "state": {
    "follow1_pos": [7 floats],
    "follow2_pos": [7 floats],
  },
  "views": {
    "camera_front": "<base64 JPEG>",
    "camera_left": "<base64 JPEG>",
    "camera_right": "<base64 JPEG>",
  },
  "instruction": "arrange flower",
}
```

## 5. Pitfalls on this machine

### 1. `conda create -n wallx` failed

```
CondaValueError: Environment paths cannot be immediately nested...
```

**Fix:** Use a `--prefix` path in `local/env.sh`.

### 2. `pip install` landed in pyenv

**Symptom:** Packages went to `~/.pyenv/versions/3.10.20` instead of the conda environment.

**Fix:** Always use `$CONDA_ENV/bin/pip` or `$CONDA_ENV/bin/python -m pip`.

### 3. wall-x / flash-attn build reported a CUDA version mismatch

```
CUDA version (13.2) mismatches ... (12.8)
```

**Fix:** Install `cuda-nvcc=12.8` from conda-forge and point `CUDA_HOME` at the conda environment.

### 4. flash-attn build reported `cuda_runtime.h: No such file`

**Fix:**

```bash
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CPATH
```

### 5. Inference reported `Image features and image tokens do not match`

**Fix:** `export ENABLE_FAST_PREPROCESS=false`, then restart the service.

### 6. The service bound the wrong host

**Symptom:** The log showed `Host: x86_64-conda-linux-gnu`.

**Fix:** `export HOST=0.0.0.0` before launch.

## 6. Differences from the original docs

| Original docs | What this machine actually did |
|---------------|--------------------------------|
| `conda create -n wallx` | `conda create --prefix .../conda_envs/wallx` |
| lerobot version unspecified | `git checkout v0.4.4` |
| flash-attn not included | **Must install** flash-attn==2.8.3 |
| CUDA toolchain not included | Install cuda-nvcc 12.8 from conda |
| Serving config not included | Use `workspace/rtx5090/run_server.sh` |

## 7. Background run (optional)

```bash
nohup bash workspace/rtx5090/run_server.sh > /tmp/wallx_serve_5090.log 2>&1 &
tail -f /tmp/wallx_serve_5090.log
```

Stop the service:

```bash
pkill -f 'launch_serving.*44660'
```
