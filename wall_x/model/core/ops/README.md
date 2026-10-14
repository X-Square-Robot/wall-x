# `wall_x.model.core.ops` — operator boundary contract

This package is the **single boundary** between wall-x and the internal
`external_accel` kernel library. It is also a **public (open-source) module**: the
open-source export ships `wall_x/model/core/ops/` but **not** `external_accel`, so the
PyTorch fallback in each operator is the code path open-source users actually
run. That makes the contract below load-bearing, not stylistic.

## The contract

Every operator that can be accelerated by an external_accel kernel MUST:

1. **Touch external_accel only here.** No module outside `core/ops/` may
   `import external_accel` (directly or via `from external_accel import ...`). Higher layers
   import the operator from `wall_x.model.core.ops`.

2. **Import cleanly without external_accel.** Kernel access is wrapped so the module
   imports on a machine with no external_accel:
   - `OpsProxy` subclasses resolve lazily via `importlib.import_module`.
   - Plain-function ops use `try: from external_accel import <sym> except Exception: <sym> = None`.
   The `from external_accel import` line must be indented under `try` (never column-0),
   so the open-source export grep guard passes.

3. **Own a self-contained PyTorch fallback.** The fallback lives **in this
   package**, computes the same result as the kernel, and is a **first-class,
   independently callable** entry — never buried in a caller module. Rationale:
   - open-source runs the fallback (no external_accel), so it must be complete here;
   - the fallback must be selectable in isolation for the bitexact test below.

4. **Select backend by capability — no env switches.** An op uses its kernel
   iff the loaded external_accel build provides it (and inputs are on CUDA); otherwise
   it runs the fallback. There are **no per-op switches and no global disable
   flag** — capability detection ("kernel present ⇒ use it") is the only
   mechanism. To exercise the fallback, run without external_accel (the open-source
   env) or, in a test, call the fallback directly / null the module's kernel
   handle.

5. **Be covered by `tests/test_ops_fallback.py`** with two assertions:
   - **fallback vs reference** — the PyTorch fallback matches an independent
     reference (this runs in the open-source overlay, guaranteeing OSS
     correctness with no external_accel present);
   - **kernel vs fallback** (`test_vs_external_accel`) — when external_accel is present, the
     kernel and the fallback agree. Use an explicit tolerance for
     native-dtype/fused rounding (fp32 ~2e-6, fp16 ~2e-2, bf16 ~3e-2); a
     divergence far outside tolerance is a **convention/layout bug**, not a
     rounding artifact — fix the operator, do not widen the tolerance.

## Two conforming shapes

- **`OpsProxy` subclass** (rmsnorm, rope, m_rope, silu_and_mul, …): define
  `_external_accel_name` and `_pytorch_fallback`; get `__call__` dispatch,
  `call_with_backend("pytorch"|"external_accel")`, and `available_backends()` for free.
  Preferred for single tensor→tensor ops.

- **Plain-function module** (gated_delta, qkv_norm_replace, attention_mask):
  when the op is not a single callable (multiple entry points, or a fused op
  with a structured signature), expose `<op>(...)` (capability-dispatched) and a
  directly-callable `<op>_pytorch(...)` fallback in the same module, plus a
  `<op>_available(...)` probe. The fallback must be a pure tensor function (it
  may take a `module` for parameters but must not embed caller-only control
  flow), so the bitexact test can call kernel and fallback side by side.

## Anti-patterns (do not ship)

- Fallback logic living in a caller module (e.g. an attention forward) while the
  ops module only wraps the kernel — the op then has no OSS-runnable, testable
  fallback.
- Any env switch (per-op or global) to force the fallback — capability
  detection covers every real need; nulling the kernel handle covers tests.
- Widening a `test_vs_external_accel` tolerance to hide a large divergence.
