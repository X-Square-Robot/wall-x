import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ConstantLR, LambdaLR


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
            # Start from peak_lr / (warmup_steps + 1).
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
    factor: float = 1 / 3,
    total_iters: int = 5,
    last_epoch: int = -1,
):
    return ConstantLR(optimizer, factor, total_iters, last_epoch)


def get_scheduler(optimizer: Optimizer, lr_scheduler_type: str, **kwargs):
    if lr_scheduler_type == "cosine":
        return create_cosine_scheduler(optimizer, **kwargs)
    elif lr_scheduler_type == "step":
        return create_step_scheduler(optimizer, **kwargs)
    elif lr_scheduler_type == "constant":
        return create_constant_scheduler(optimizer, **kwargs)
    else:
        raise ValueError(f"Unsupported lr_scheduler: {lr_scheduler_type}")
