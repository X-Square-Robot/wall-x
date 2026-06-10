import os

import torch

ENABLE_CUDA_GRAPH = os.environ.get("ENABLE_CUDA_GRAPH", "True").lower() == "true"
# When using a single master buffer + multi-bucket graph, max batch must be pre-allocated;
# otherwise resize/reallocation after capture invalidates addresses recorded by the graph.
CUDA_GRAPH_MAX_BS = int(os.environ.get("CUDA_GRAPH_MAX_BS", "128"))
_SHAPE_GUARD_FASTPATH = (
    os.environ.get("CUDAGRAPH_DISABLE_SHAPE_GUARD_FASTPATH", "0") != "1"
)


class CUDAGraph_Wrapper:
    """
    Bucketed CUDA Graph wrapper:
    - When batch size changes, select/reuse a CUDAGraph by bucket_size
    - For batch_size < bucket_size: copy inputs into static buffer [:B], return outputs [:B] after replay
    """

    def __init__(
        self,
        model,
        warm_up_times: int = 3,
        enable: bool | None = None,
        batch_size_key: str | None = None,
    ):
        self.model = model
        self.warm_up_times = warm_up_times
        self._enable = ENABLE_CUDA_GRAPH if enable is None else bool(enable)
        self._batch_size_key = batch_size_key or "suffix_inputs_embeds"

        # graph cache: bucket_bs -> CUDAGraph
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        # Per bucket, store views into the master buffer (graph capture depends on these views' address/shape/stride)
        self.static_inputs_map: dict[int, dict[str, torch.Tensor]] = {}
        self.static_output_tensor: dict[int, torch.Tensor] = {}
        self.graph_pool = None

        # Master buffers (shared across all buckets)
        self.graph_vars: dict[str, torch.Tensor] = {}
        self._max_bs: int | None = None

        # Record non-batch dimension shape signature (globally consistent) to avoid silent errors
        self._shape_signature: dict[str, tuple] = {}
        # Batch dimension index per input (default 0; e.g. suffix_position_ids batch is at dim=1)
        self._batch_dim: dict[str, int] = {}
        self._shape_guard_verified: dict[int, bool] = {}

    def _get_batch_size(self, **kwargs) -> int:
        key = self._batch_size_key
        if key in kwargs:
            v = kwargs[key]
            if isinstance(v, torch.Tensor) and v.dim() >= 1:
                return int(v.shape[0])
        raise ValueError(f"CUDAGraph_Wrapper.forward requires {key} argument")

    def _select_bucket_bs(self, bs: int) -> int:
        """
        Bucket selection:
        - 1, 2, 4, 8 use pow2 buckets
        - >8 align to 16 (16, 32, 48, ...)
        """
        if bs <= 1:
            return 1
        if bs <= 2:
            return 2
        if bs <= 4:
            return 4
        if bs <= 8:
            return 8
        return int(((bs + 15) // 16) * 16)

    def _batch_dim_for_key(self, key: str, tensor: torch.Tensor) -> int:
        # Minimal special-case handling for known inputs only
        # suffix_position_ids: shape [3, B, T], batch at dim=1
        if key == "suffix_position_ids" and tensor.dim() >= 2:
            return 1
        return 0

    def _view_for_bucket(self, tensor: torch.Tensor, bucket_bs: int, batch_dim: int):
        if tensor.dim() == 0:
            return tensor
        if batch_dim == 0:
            return tensor[:bucket_bs]
        if batch_dim == 1:
            return tensor[:, :bucket_bs]
        raise ValueError(f"Unsupported batch_dim={batch_dim} for cudagraph wrapper")

    def _copy_and_pad(
        self,
        master: torch.Tensor,
        tensor: torch.Tensor,
        bs: int,
        bucket_bs: int,
        batch_dim: int,
    ):
        if tensor.dim() == 0:
            master.copy_(tensor)
            return
        if batch_dim == 0:
            master[:bs].copy_(tensor)
            if bs < bucket_bs:
                master[bs:bucket_bs].zero_()
            return
        if batch_dim == 1:
            master[:, :bs].copy_(tensor)
            if bs < bucket_bs:
                master[:, bs:bucket_bs].zero_()
            return
        raise ValueError(f"Unsupported batch_dim={batch_dim} for cudagraph wrapper")

    def _allocate_master_tensor(
        self, key: str, tensor: torch.Tensor, max_bs: int, bs: int
    ) -> torch.Tensor:
        """
        Allocate a master buffer for one input (single large buffer):
        - Extend only the batch dimension to max_bs; other dims unchanged
        - Copy the current bs valid range first; zero the rest (avoid stale values on replay)
        """
        if tensor.dim() == 0:
            return tensor.clone()
        batch_dim = self._batch_dim_for_key(key, tensor)
        shape = list(tensor.shape)
        if batch_dim >= len(shape):
            raise ValueError(
                f"Invalid batch_dim={batch_dim} for key={key}, tensor.shape={tuple(tensor.shape)}"
            )
        shape[batch_dim] = max_bs
        buf = torch.zeros(tuple(shape), device=tensor.device, dtype=tensor.dtype)
        # Initial copy: pad to current bucket (at least cover the valid bs range)
        self._copy_and_pad(buf, tensor, bs=bs, bucket_bs=bs, batch_dim=batch_dim)
        return buf

    def _ensure_master_buffers(self, bs: int, **kwargs):
        """
        Ensure master buffers are allocated once.
        - max_bs from CUDA_GRAPH_MAX_BS; if unset, use the first bucket_bs as max_bs
          and disallow exceeding it later (resize would invalidate captured graphs).
        """
        if self._max_bs is not None:
            return

        bucket_bs = self._select_bucket_bs(bs)
        max_bs = CUDA_GRAPH_MAX_BS if CUDA_GRAPH_MAX_BS > 0 else bucket_bs
        self._max_bs = int(max_bs)

        for k, v in kwargs.items():
            if not isinstance(v, torch.Tensor):
                raise TypeError(
                    f"CUDAGraph_Wrapper only supports Tensor kwargs, got {k}={type(v)}"
                )
            batch_dim = self._batch_dim_for_key(k, v) if v.dim() > 0 else 0
            self._batch_dim[k] = batch_dim
            # Record non-batch dimensions (globally consistent)
            if v.dim() == 0:
                self._shape_signature[k] = tuple()
            else:
                sig = list(v.shape)
                sig.pop(batch_dim)
                self._shape_signature[k] = tuple(sig)
            self.graph_vars[k] = self._allocate_master_tensor(
                k, v, max_bs=self._max_bs, bs=bs
            )

    def _initialize_bucket(self, bucket_bs: int, bs: int, **kwargs):
        assert self._max_bs is not None
        self._shape_guard_verified[bucket_bs] = False
        if bucket_bs > self._max_bs:
            raise ValueError(
                f"CUDAGraph bucket_bs({bucket_bs}) exceeds master max_bs({self._max_bs}). "
                f"Set CUDA_GRAPH_MAX_BS >= {bucket_bs}."
            )

        print(f"[CUDAGraph] Initializing bucket_bs={bucket_bs} (current bs={bs}) ...")

        # Build view map for this bucket (fixed address/shape/stride)
        static_map: dict[str, torch.Tensor] = {}
        for k, v in kwargs.items():
            # Shape guard: non-batch dims must match
            expected = self._shape_signature.get(k)
            if v.dim() == 0:
                got = tuple()
            else:
                batch_dim = self._batch_dim.get(k, self._batch_dim_for_key(k, v))
                got_list = list(v.shape)
                got_list.pop(batch_dim)
                got = tuple(got_list)
            if expected is not None and got != expected:
                raise ValueError(
                    f"CUDAGraph bucket init: non-batch dim changed: key={k}, expected={expected}, got={got}"
                )
            master = self.graph_vars[k]
            batch_dim = self._batch_dim.get(k, 0)
            static_map[k] = self._view_for_bucket(
                master, bucket_bs=bucket_bs, batch_dim=batch_dim
            )

        # Warmup (stabilize kernels/caches)
        out = None
        for _ in range(self.warm_up_times):
            out = self.forward_naive(**static_map)
        assert isinstance(out, torch.Tensor)

        # Output master buffer (allocate once)
        if "_outputs" not in self.graph_vars:
            out_shape = (self._max_bs,) + tuple(out.shape[1:])
            self.graph_vars["_outputs"] = torch.empty(
                out_shape, device=out.device, dtype=out.dtype
            )

        self.static_inputs_map[bucket_bs] = static_map
        self.static_output_tensor[bucket_bs] = self.graph_vars["_outputs"][:bucket_bs]

        # capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.graph_pool):
            self.static_output_tensor[bucket_bs].copy_(self.forward_naive(**static_map))
        if self.graph_pool is None:
            self.graph_pool = graph.pool()
        self.graphs[bucket_bs] = graph
        torch.cuda.synchronize()
        print(f"[CUDAGraph] Bucket {bucket_bs} captured.")

    def forward_naive(self, **kwargs):
        return self.model(**kwargs)

    def forward(self, **kwargs):
        if not self._enable:
            return self.forward_naive(**kwargs)
        bs = self._get_batch_size(**kwargs)
        bucket_bs = self._select_bucket_bs(bs)

        # Initialize master buffers (first call)
        self._ensure_master_buffers(bs=bs, **kwargs)
        assert self._max_bs is not None
        if bucket_bs > self._max_bs:
            # Do not exceed master max_bs at runtime (resize would invalidate captured graphs)
            return self.forward_naive(**kwargs)

        # lazy capture for this bucket
        if bucket_bs not in self.graphs:
            self._initialize_bucket(bucket_bs=bucket_bs, bs=bs, **kwargs)

        # Shape guard: non-batch dim change is unsafe for cudagraph; fallback
        # fast-path: skip guard if already verified for this bucket (stable-state optimization)
        if not (
            _SHAPE_GUARD_FASTPATH and self._shape_guard_verified.get(bucket_bs, False)
        ):
            for k, v in kwargs.items():
                if not isinstance(v, torch.Tensor):
                    raise TypeError(
                        f"CUDAGraph_Wrapper only supports Tensor kwargs, got {k}={type(v)}"
                    )
                expected = self._shape_signature.get(k)
                if v.dim() == 0:
                    got = tuple()
                else:
                    batch_dim = self._batch_dim.get(k, self._batch_dim_for_key(k, v))
                    got_list = list(v.shape)
                    got_list.pop(batch_dim)
                    got = tuple(got_list)
                if expected is not None and got != expected:
                    return self.forward_naive(**kwargs)
            # Mark verified for fast-path on subsequent calls
            self._shape_guard_verified[bucket_bs] = True

        # Copy inputs into master buffer (per batch_dim, zero padding)
        for k, tensor in kwargs.items():
            master = self.graph_vars[k]
            batch_dim = self._batch_dim.get(k, 0)
            self._copy_and_pad(
                master, tensor, bs=bs, bucket_bs=bucket_bs, batch_dim=batch_dim
            )

        graph = self.graphs[bucket_bs]
        graph.replay()
        return self.graph_vars["_outputs"][:bs]
