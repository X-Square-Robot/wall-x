# Scripts

This directory contains the public Wall-X command-line helpers. Run commands
from the repository root unless noted otherwise.

## Inference smoke test

Use `fake_inference.py` to verify that a checkpoint can be loaded and can
produce one action chunk from a synthetic LIBERO-style observation.

```bash
python scripts/fake_inference.py \
  --checkpoint-path /path/to/checkpoint
```

If the training config is not stored next to the checkpoint as `config.yml` or
`config.yaml`, pass it explicitly:

```bash
python scripts/fake_inference.py \
  --checkpoint-path /path/to/checkpoint \
  --train-config-path /path/to/config.yml
```

## LIBERO evaluation

`run_libero.sh` is a small shell wrapper around `infer_libero.py`.

```bash
CHECKPOINT_PATH=/path/to/checkpoint bash scripts/run_libero.sh
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
```

For full control, call the Python entry directly:

```bash
python scripts/infer_libero.py \
  --checkpoint-path /path/to/checkpoint \
  --task-suite-name libero_spatial \
  --num-trials-per-task 50 \
  --driver-mode in_process
```

You can also pass a complete harrix eval config:

```bash
python scripts/infer_libero.py --config /path/to/eval_config.yml
```

## Dataset and checkpoint utilities

- `compute_norm_stats.py`: compute action normalization statistics for a
  LeRobot dataset.
- `merge_sharded_weights.py`: merge FSDP sharded checkpoint files into a single
  checkpoint directory.
- `normalize.py`: helper for generating normalization files from dataset
  statistics.

Most scripts support `--help` for their command-line options.
