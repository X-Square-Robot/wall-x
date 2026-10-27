"""Root-cause workaround for the torch-2.13 + CPython 3.12.5..3.13.x profiler
"FunctionMeta.apply tower" (thousands of nested ghost C frames in with_stack
chrome traces, clamped to trace end).

Root cause (CPython bug gh-122029, fixed only in 3.14 by python/cpython#130898):
  A CALL_FUNCTION_EX (star-args) call of a *bound method wrapping a C function*
  emits `c_call` but never `c_return` to sys.setprofile / sys.monitoring.
  torch._ops.OpOverload.redispatch does exactly that:
      return self._handle.redispatch_boxed(keyset, *args, **kwargs)
  (`_handle.redispatch_boxed` is a `method` wrapping a pybind11 PyCFunction).
  Every torch.library custom-op invocation (flash-attn-3, causal_conv1d, ...)
  therefore leaks >=1 unmatched c_call into the torch profiler's python tracer;
  replay then never closes the enclosing `FunctionMeta.apply` C frame -> tower.

Fix: star-calls of a *plain PyCFunction* are paired correctly, so call the
underlying function explicitly with the handle as first argument.

Usage (before creating torch.profiler.profile, e.g. at trainer startup):
    import fix_profiler_tower
    fix_profiler_tower.apply_patch()

Validated on torch 2.13.0+cu130 / python 3.12.13 (H100, 2026-08-05):
  custom-op-in-autograd.Function workload, 300 iters, with_stack=True:
    before patch: 300/300 apply C frames clamped to trace end (tower)
    after  patch:   0/300 leaked; apply median dur 50us; outputs bit-identical.
Run `python fix_profiler_tower.py` for the self-test.
"""

import torch

_orig = {}


def apply_patch():
    """Idempotent. Patches torch._ops.OpOverload.redispatch (the sole leak
    site) to star-call the raw PyCFunction so gh-122029 pairs call/return.

    Best-effort: it touches torch internals, so any layout change in a
    future torch raises here and the patch is skipped with a warning --
    phantom frame towers in a debug trace are better than a run that
    cannot start.
    """
    if _orig:
        return

    try:
        import torch._ops as _ops

        _raw_rdb = torch._C._DispatchOperatorHandle.redispatch_boxed.__func__
        original = _ops.OpOverload.redispatch

        def redispatch(self, /, keyset, *args, **kwargs):
            return _raw_rdb(self._handle, keyset, *args, **kwargs)

        redispatch.__name__ = "redispatch"
        redispatch.__qualname__ = "OpOverload.redispatch"
        _ops.OpOverload.redispatch = redispatch
        _orig["redispatch"] = original
    except Exception as exc:  # noqa: BLE001 -- debug aid must never kill a run
        _orig.clear()
        print(
            "[WallX] WARNING: profiler tower patch not applied "
            f"({type(exc).__name__}: {exc}); with_stack traces may show "
            "phantom frame towers."
        )


def remove_patch():
    if _orig:
        import torch._ops as _ops

        _ops.OpOverload.redispatch = _orig.pop("redispatch")


def _self_test(device=None):
    import json
    import tempfile

    from torch.library import custom_op

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    @custom_op("fixtest::op", mutates_args=())
    def _op(t: torch.Tensor) -> torch.Tensor:
        return t * 2

    @_op.register_fake
    def _(t):
        return torch.empty_like(t)

    _op.register_autograd(lambda ctx, g: g * 2, setup_context=lambda ctx, i, o: None)

    class W(torch.autograd.Function):
        @staticmethod
        def forward(ctx, t):
            return _op(t)

        @staticmethod
        def backward(ctx, g):
            return g * 2

    x = torch.randn(64, 64, device=device, requires_grad=True)
    W.apply(x).sum().backward()

    def leak_count():
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], with_stack=True
        ) as p:
            for _ in range(300):
                W.apply(x).sum().backward()
        f = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        p.export_chrome_trace(f.name)
        d = json.load(open(f.name))
        pf = [e for e in d["traceEvents"] if e.get("cat") == "python_function"]
        tmax = max(e["ts"] + e.get("dur", 0) for e in pf)
        ap = [e for e in pf if "apply of FunctionMeta" in e.get("name", "")]
        return sum(1 for e in ap if abs(e["ts"] + e.get("dur", 0) - tmax) < 5), len(ap)

    before = leak_count()
    apply_patch()
    y = W.apply(x)
    assert torch.allclose(y, x * 2)
    y.sum().backward()
    after = leak_count()
    print(
        f"[fix_profiler_tower] device={device} "
        f"before: {before[0]}/{before[1]} apply frames leaked; "
        f"after: {after[0]}/{after[1]} leaked"
    )
    assert after[0] == 0, "patch failed to stop the leak"
    print("[fix_profiler_tower] PASS")


if __name__ == "__main__":
    _self_test()
