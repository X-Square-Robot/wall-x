# Scripts

This directory contains the public Wall-X command-line helpers. Run the examples
below from the repository root, using `python scripts/...` and `bash scripts/...`.
Pass file and directory paths explicitly.

## Inference smoke test

Use `fake_inference.py` to verify that a checkpoint can be loaded and can
produce one action chunk from a synthetic LIBERO-style observation.

```bash
python scripts/fake_inference.py --checkpoint-path /path/to/checkpoint
```

If the training config is not stored next to the checkpoint as `config.yml` or
`config.yaml`, pass it explicitly:

```bash
python scripts/fake_inference.py \
  --checkpoint-path /path/to/checkpoint \
  --train-config-path /path/to/config.yml
```

## LIBERO evaluation

`run_libero.sh` is a small shell wrapper around `infer_libero.py`. It requires
the optional LIBERO simulator stack:

```bash
pip install -r requirements-libero.txt
mkdir -p third_party
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
```

The launcher checks for LIBERO, robosuite, MuJoCo, PyOpenGL, BDDL, Gym, and
h5py before loading the model. If LIBERO is cloned elsewhere, pass
`LIBERO_PATH=/path/to/LIBERO`.

```bash
bash scripts/run_libero.sh /path/to/checkpoint
```

Useful environment variables:

```bash
CHECKPOINT_PATH=/path/to/checkpoint
TRAIN_CONFIG_PATH=/path/to/config.yml
TASK_SUITE_NAME=libero_spatial
TASK_INDICES=0,1,2
NUM_TRIALS_PER_TASK=50
CUDA_ID=0
SMOKE=1
MAX_INFER_TIMES=52
```

`MAX_INFER_TIMES` is optional. When omitted, the launcher uses suite-specific
defaults aligned with the LIBERO evaluator: spatial 22, object 28, goal 30,
libero_10 52, and libero_90 40 action chunks.

For full control, call the Python entry directly:

```bash
python scripts/infer_libero.py \
  --checkpoint-path /path/to/checkpoint \
  --task-suite-name libero_spatial \
  --num-trials-per-task 50 \
  --driver-mode in_process
```

You can also pass a complete eval config:

```bash
python scripts/infer_libero.py --config /path/to/eval_config.yml
```

## WebSocket serving

`run_serving.sh` launches the Wall-X WebSocket server through the public
vendored serving runtime. Pass paths explicitly; the script has no built-in
checkpoint path.

```bash
bash scripts/run_serving.sh \
  --checkpoint-path /path/to/checkpoint \
  --train-config-path /path/to/config.yml \
  --port 32195
```

By default the script returns raw model action chunks, which is the expected
mode for open-loop plotting. Pass `--serialize-actions` when your client expects
robot-serialized actions.

Useful options:

```bash
CUDA_ID=0
ACTION_HORIZON=32
IMAGE_PASSING_MODE=base64
MAX_BATCH_SIZE=1
```

Additional `launch_serving.py` arguments can be forwarded after `--`:

```bash
bash scripts/run_serving.sh --checkpoint-path /path/to/checkpoint -- \
  --model-config.norm-key libero_all
```

## Open-loop WebSocket evaluation

`draw_openloop_plot.py` compares predicted action chunks from a running
WebSocket server against LeRobot dataset ground truth. `--dataset-root` and
`--train-config` are required and have no built-in default.

```bash
python scripts/draw_openloop_plot.py \
  --uri ws://127.0.0.1:32195 \
  --dataset-root /path/to/lerobot_dataset \
  --train-config /path/to/train_config.yml \
  --episode-indices 0,1,2 \
  --save-dir ./openloop_plots
```

## Dataset and checkpoint utilities

- `compute_norm_stats.py`: compute action normalization statistics for a
  local LeRobot v3 dataset. The script reads state/action parquet columns
  directly when available, so image and video columns are not decoded.
- `merge_sharded_weights.py`: merge FSDP sharded checkpoint files into a single
  checkpoint directory.
- `merge_tokenizer.py`: merge FAST action tokens into a Qwen2.5-VL processor
  tokenizer.

```bash
python scripts/merge_tokenizer.py \
  --processor-path /path/to/Qwen2.5-VL-3B-Instruct \
  --action-tokenizer-path /path/to/fast_tokenizer \
  --output-dir /path/to/merged_processor
```

Most scripts support `--help` for their command-line options.
