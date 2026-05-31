from __future__ import annotations

from collections import defaultdict
from typing import Callable, DefaultDict, Optional

import torch
from torch.distributed.tensor import DTensor, Replicate


class DistributedNormAccumulator:
    """Accumulate L2 norms without leaking parallel-layout branches to adapters."""

    def __init__(
        self,
        *,
        device: torch.device,
        reduce_tensor_fn: Optional[Callable[[torch.Tensor, bool], torch.Tensor]],
    ):
        self.device = device
        self.reduce_tensor_fn = reduce_tensor_fn
        self._regular: DefaultDict[str, list[torch.Tensor]] = defaultdict(list)
        self._dtensor: DefaultDict[str, list[torch.Tensor]] = defaultdict(list)
        self._owner_only: DefaultDict[str, list[torch.Tensor]] = defaultdict(list)

    @staticmethod
    def _is_dtensor(tensor: torch.Tensor) -> bool:
        return isinstance(tensor, DTensor)

    def add(self, group: Optional[str], tensor: Optional[torch.Tensor]) -> None:
        if group is None or tensor is None or tensor.numel() == 0:
            return
        tensor = tensor.detach()
        if self._is_dtensor(tensor):
            self._dtensor[group].append(tensor)
        else:
            self._regular[group].append(tensor)

    def add_owner_only(
        self, group: Optional[str], tensor: Optional[torch.Tensor]
    ) -> None:
        if group is None or tensor is None or tensor.numel() == 0:
            return
        self._owner_only[group].append(tensor.detach())

    def squared_norm(self, group: str) -> torch.Tensor:
        with torch.no_grad():
            total = torch.zeros((), dtype=torch.float32, device=self.device)
            total = total + self._regular_squared_norm(self._regular[group])
            total = total + self._owner_only_squared_norm(self._owner_only[group])
            total = total + self._dtensor_squared_norm(self._dtensor[group])
            return total

    def norm(self, group: str) -> torch.Tensor:
        return self.squared_norm(group).sqrt()

    def _regular_squared_norm(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        if not tensors:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        per = self._foreach_norm(tensors)
        return self._sum_norms_squared(per)

    def _owner_only_squared_norm(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        local_sq = self._regular_squared_norm(tensors)
        if self.reduce_tensor_fn is None:
            return local_sq
        return self.reduce_tensor_fn(local_sq, average=False)

    def _dtensor_squared_norm(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        if not tensors:
            return torch.zeros((), dtype=torch.float32, device=self.device)

        per = self._foreach_norm(tensors)

        # Group by (mesh, placements) so each unique layout costs only one
        # collective (redistribute).
        groups: DefaultDict[object, list[torch.Tensor]] = defaultdict(list)
        for norm in per:
            if self._is_dtensor(norm):
                key = (id(norm.device_mesh), tuple(norm.placements))
            else:
                key = None
            groups[key].append(norm)

        parts: list[torch.Tensor] = []
        for group_tensors in groups.values():
            try:
                stacked = torch.stack(group_tensors)
                local = self._to_local_replicated(stacked).to(
                    device=self.device, dtype=torch.float32
                )
                parts.append(local.pow(2).sum())
            except (RuntimeError, NotImplementedError):
                # Stack/redistribute not supported for this layout in the
                # current PyTorch version; fall back to per-tensor for safety.
                for norm in group_tensors:
                    local = self._to_local_replicated(norm).to(
                        device=self.device, dtype=torch.float32
                    )
                    parts.append(local.pow(2).sum())

        return torch.stack(parts).sum()

    def _foreach_norm(self, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        try:
            return torch._foreach_norm(tensors, 2.0)
        except (RuntimeError, NotImplementedError):
            return [torch.linalg.vector_norm(t, 2.0) for t in tensors]

    def _sum_norms_squared(self, norms: list[torch.Tensor]) -> torch.Tensor:
        if not norms:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        local_norms = [
            self._to_local_replicated(norm).to(device=self.device, dtype=torch.float32)
            for norm in norms
        ]
        return torch.stack(local_norms).pow(2).sum()

    def _to_local_replicated(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self._is_dtensor(tensor):
            return tensor.to(device=self.device)
        mesh = tensor.device_mesh
        tensor = tensor.redistribute(placements=[Replicate()] * mesh.ndim)
        return tensor.to_local().to(device=self.device)
