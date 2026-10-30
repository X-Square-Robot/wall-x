# RTX 5090 inference deployment

Scripts and docs for deploying the Wall-X inference service on an **NVIDIA RTX 5090**.

## Directory layout

```
workspace/rtx5090/
├── README.md          # this file
├── DEPLOY.md          # general deployment guide (any 5090 machine)
├── install.sh         # environment install script
├── run_server.sh      # start the inference service
├── env.example        # path-config template
└── local/             # machine-local config (edit as needed; do not commit sensitive paths)
    ├── env.sh         # local paths
    └── NOTES.md       # local bring-up notes
```

## Quick start

**1. Set paths**

```bash
cp workspace/rtx5090/env.example workspace/rtx5090/local/env.sh
# Edit local/env.sh and fill in CONDA_SH, CONDA_ENV, LEROBOT_DIR, CHECKPOINT_PATH
```

**2. Install the environment** (from the repository root)

```bash
bash workspace/rtx5090/install.sh
```

**3. Start the service**

```bash
bash workspace/rtx5090/run_server.sh
```

**4. Health check**

```bash
curl http://127.0.0.1:44660/healthz   # expected: OK
```

## Documents

| File | Purpose |
|------|---------|
| [DEPLOY.md](./DEPLOY.md) | General deployment flow with placeholder paths, for any 5090 inference machine |
| [local/NOTES.md](./local/NOTES.md) | Machine-specific notes (environment, pitfalls, verification) |

## Environment variables

Scripts load these variables from `local/env.sh`:

| Variable | Description |
|----------|-------------|
| `CONDA_SH` | Path to the conda init script |
| `CONDA_ENV` | Conda environment name, or a full `--prefix` path |
| `LEROBOT_DIR` | LeRobot source directory |
| `CHECKPOINT_PATH` | Model checkpoint directory |
| `PORT` | Service port (default 44660) |
| `CUDA_ID` | GPU index (default 0) |

The server also sets `ENABLE_FAST_PREPROCESS=false`, `ENABLE_CUDA_GRAPH=1`, and related flags. See [DEPLOY.md](./DEPLOY.md).
