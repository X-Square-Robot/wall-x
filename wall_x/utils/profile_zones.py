"""Named regions for torch.profiler — central registry of `record_function` zones.

`torch.profiler.record_function(name)` writes a `user_annotation` slice into the
profiler trace. We use it to mark phase boundaries that would otherwise be
invisible when `with_stack=False` (the lightweight profiling mode).

Call sites use the `zone(...)` alias and the `Z` namespace:

    from wall_x.utils.profile_zones import zone, Z

    with zone(Z.FORWARD):
        outputs = self.adapter.forward(...)

Keeping the names here makes them grep-able and prevents string-typo drift
across the files that emit them.
"""

import torch

# Alias — same call site ergonomics as `torch.no_grad()`.
zone = torch.profiler.record_function


class Z:
    # All zone names use a dotted prefix so trace tooling can disambiguate
    # them with exact / prefix LIKE patterns.

    # ── Training phase (fsdp_trainer.train_loop) ──
    TRAIN_STEP = "train.step"
    DATA_TO_DEVICE = "train.data_to_device"
    FORWARD = "train.forward"
    BACKWARD = "train.backward"
    GRAD_NORMS = "train.grad_norms"
    GRAD_CLIP = "train.grad_clip"
    OPTIMIZER_STEP = "train.optimizer_step"
    ZERO_GRAD = "train.zero_grad"
    LOGGING = "train.logging"
    SAVE_CHECKPOINT = "train.save_checkpoint"
