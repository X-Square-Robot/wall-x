from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
import math


def create_cosine_scheduler(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    peak_lr: float | None = None,
    end_lr: float | None = None,
    last_epoch: int = -1,
):
    if peak_lr is None:
        peak_lr = float(optimizer.defaults["lr"])
    if end_lr is None:
        end_lr = peak_lr * 0.1

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            # OpenPI warmup: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (num_warmup_steps + 1)
            current_lr = init_lr + (peak_lr - init_lr) * current_step / num_warmup_steps
            return current_lr / peak_lr  # LambdaLR multiplies by base_lr
        else:
            # Cosine decay
            decay_steps = num_training_steps - num_warmup_steps
            progress = min(1.0, (current_step - num_warmup_steps) / max(1, decay_steps))
            cos = 0.5 * (1 + math.cos(math.pi * progress))
            current_lr = end_lr + (peak_lr - end_lr) * cos
            return current_lr / peak_lr

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def create_step_scheduler(
    optimizer: Optimizer,
    lr_decay_steps: str,
    lr_gamma: float = 0.1,
):
    decay_steps = [int(s.strip()) for s in lr_decay_steps.split(",")]

    def lr_lambda(current_step):
        factor = 1.0
        for step in decay_steps:
            if current_step >= step:
                factor *= lr_gamma
        return factor

    return LambdaLR(optimizer, lr_lambda)


def create_constant_scheduler(
    optimizer: Optimizer,
    last_epoch: int = -1,
    num_warmup_steps: int = 0,
):
    """Constant LR at the target value, with optional linear warmup.

    With ``num_warmup_steps <= 0`` the LR is constant at the target from step 0
    (the previous behaviour). With ``num_warmup_steps = n > 0`` the LR ramps
    linearly from ``1/(n+1)`` of the target up to the full target over the first
    ``n`` steps, then stays constant::

        factor(step) = 1/(n+1) + (1 - 1/(n+1)) * step / n   for step < n
        factor(step) = 1.0                                   for step >= n

    e.g. n=4: step 0..4 -> 0.2, 0.4, 0.6, 0.8, 1.0 (x target). The initial LR is
    non-zero by design. Covered by test_constant_scheduler_supports_warmup_per_lr_group.
    """

    def lr_lambda(current_step: int):
        if num_warmup_steps <= 0 or current_step >= num_warmup_steps:
            return 1.0

        init_lr_factor = 1.0 / (num_warmup_steps + 1)
        return init_lr_factor + (1.0 - init_lr_factor) * current_step / num_warmup_steps

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_scheduler(optimizer: Optimizer, lr_scheduler_type: str, **kwargs):
    if lr_scheduler_type == "cosine":
        return create_cosine_scheduler(optimizer, **kwargs)
    elif lr_scheduler_type == "step":
        return create_step_scheduler(optimizer, **kwargs)
    elif lr_scheduler_type == "constant":
        return create_constant_scheduler(optimizer, **kwargs)
    else:
        raise ValueError(f"Unsupported lr_scheduler: {lr_scheduler_type}")
