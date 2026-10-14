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

You can also pass a complete harrix eval config:

```bash
python scripts/infer_libero.py --config /path/to/eval_config.yml
```

## RoboDojo serving

`run_robodojo_serving.sh` serves a flow checkpoint over the RoboDojo
(XPolicyLab) WebSocket protocol, including batched requests from parallel
simulator environments.

```bash
CUDA_ID=0 bash scripts/run_robodojo_serving.sh \
  --checkpoint-path /path/to/checkpoint \
  --norm-key ex_normal \
  --port 19000 \
  --replan-steps 16
```

Each request returns `--replan-steps` actions as
`{left,right}_ee_pose` (`xyz + quaternion wxyz`) and
`{left,right}_ee_joint_state`. Relative actions are composed with the observed
end-effector pose in the frame recorded in the checkpoint config. The first
request for each new batch size is slower while kernels autotune.

## Dataset and checkpoint utilities

- `compute_norm_stats.py`: compute action normalization statistics for a
  LeRobot dataset.
- `merge_sharded_weights.py`: merge FSDP sharded checkpoint files into a single
  checkpoint directory.
- `normalize.py`: helper for generating normalization files from dataset
  statistics.

Most scripts support `--help` for their command-line options.
