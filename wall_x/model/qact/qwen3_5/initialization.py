import torch


def ones_(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nn.init.ones_(tensor)


def zeros_(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nn.init.zeros_(tensor)


def copy_(tensor: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return tensor.copy_(source)
