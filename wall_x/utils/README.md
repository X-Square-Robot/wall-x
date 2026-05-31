# wall_x.utils

Utility modules used during Wall-X training and inference.

## Activation Inspector

`activation_inspector.py` provides a **drop-in activation statistics tool** for analyzing activation memory during forward and backward passes. Typical uses:

- Find modules with the largest activation footprint
- Debug VRAM spikes during training
- Choose candidates for activation checkpointing / recompute
- Compare activation cost across configs (resolution, architecture, FA1 vs FA2, etc.)

---

## Basic usage

```python
from wall_x.utils.activation_inspector import ActivationInspector

inspector = ActivationInspector(
    model,
    only_requires_grad=True,   # only activations that participate in backward
    min_size_mb=5.0,           # ignore smaller activations (MB)
    rank0_only=True,           # rank-0 only; suitable for DDP / FSDP
)

with inspector:
    loss = model(inputs).mean()
    loss.backward()

inspector.print_top_modules(topk=20)
```
