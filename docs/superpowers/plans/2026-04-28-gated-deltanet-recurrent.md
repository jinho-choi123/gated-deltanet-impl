# GatedDeltaNet Triton Implementation Plan (Phase 1: Recurrent Forward)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a minimal Triton kernel for the GatedDeltaNet recurrent forward pass (single-token, decode-style), validated against a PyTorch reference and `fla-org/flash-linear-attention`.

**Architecture:** Greenfield Python package. One naive PyTorch reference (ground truth, math-mapped 1:1 with the paper), one Triton kernel (fused recurrent forward, one program per `(batch, head)` pair), thin `torch.autograd.Function` wrapper that exposes the kernel and explicitly raises on backward. Tests assert numerical equivalence vs the reference and against `fla.ops.gated_delta_rule` naive output.

**Tech Stack:** Python 3.14, PyTorch ≥2.11, Triton ≥3.6, pytest, einops, `flash-linear-attention` (test/benchmark dependency only). NVIDIA GPU with CUDA required.

---

## Scope decision

This plan covers **Phase 1 only** from the BAL-28 spec: the recurrent forward kernel. The rest of the spec is intentionally deferred so this plan produces a working, testable slice end-to-end:

| BAL-28 AC item | This plan | Follow-up |
|---|---|---|
| Recurrent kernel forward + naive equivalence test | ✅ | — |
| Chunkwise kernel forward + backward (WY representation) | ❌ | New ticket (BAL-28b) |
| Fused kernels (RMSNorm gated, causal conv1d) | ❌ | New ticket (BAL-28c) |
| Performance benchmark vs fla-core | Partial: recurrent only, fused_recurrent comparison | Chunk benchmark in BAL-28b |
| Code documentation + math-code mapping | ✅ | — |

The user requested "정말 간단하게" (really simple). Recurrent forward alone:
- Unlocks decode-time inference (single-token at a time).
- Validates the math and dimensional conventions before any chunk work.
- Provides a slow-but-correct kernel that the chunkwise kernel will later be tested against.

**Recommended next action after this plan lands:** ask the user to create sub-tickets BAL-28b (chunkwise fwd+bwd) and BAL-28c (fused kernels), each with its own writing-plans pass.

---

## Math reference

Convention used throughout: state matrix `S` has shape `(D_K, D_V)`. This matches the explicit pseudo-code in the BAL-28 spec ("Delta rule: `delta = β * (v - S^T @ k)`" / "State update: `S = α * S + outer(k, delta)`"). It is also the convention used by `fla.ops.gated_delta_rule`.

Per-timestep recurrence (one program will execute this loop body `L` times):

```
inputs at step t:
    q_t, k_t       shape (D_K,)
    v_t            shape (D_V,)
    alpha_t, beta_t scalars in (0, 1)

intermediates:
    pred_v = S^T @ k_t                         shape (D_V,)
    delta  = beta_t * (v_t - pred_v)           shape (D_V,)
    S      = alpha_t * S + outer(k_t, delta)   shape (D_K, D_V)

output at step t:
    o_t    = S^T @ q_t                         shape (D_V,)
```

Two normalizations are applied **outside** the kernel for v1 (kept in the wrapper for simplicity, can be fused later):
- `q`, `k` are L2-normalized along the last dim.
- `alpha`, `beta` are passed already-activated (sigmoid). Caller's responsibility.

The exact convention vs `fla` (specifically: whether the spec's Form-1 matrix update `S = α S (I − β k k^T) + β v k^T` matches the pseudo-code above; they differ by a factor of `α` on the prediction term) is **resolved in Task 4 by direct numerical comparison**. The reference implementation is then frozen to match `fla`.

---

## File structure

```
gated-deltanet-triton/
├── pyproject.toml                          (modify: add dev deps)
├── README.md                               (modify: usage + math)
├── main.py                                 (modify: tiny demo)
├── gated_deltanet/
│   ├── __init__.py                         (create: exports)
│   ├── reference.py                        (create: naive PyTorch)
│   └── recurrent.py                        (create: Triton kernel + Function)
├── tests/
│   ├── __init__.py                         (create: empty)
│   ├── conftest.py                         (create: cuda fixture, seed)
│   ├── test_reference.py                   (create)
│   └── test_recurrent.py                   (create)
└── benchmarks/
    └── bench_recurrent.py                  (create: vs fla fused_recurrent)
```

Top-level package layout (no `src/`) so `uv run pytest` and `uv run python main.py` work without an editable install.

Each file responsibility:
- `gated_deltanet/reference.py` — pure-PyTorch naive recurrent loop. Math 1:1 with the spec. Ground truth.
- `gated_deltanet/recurrent.py` — one `@triton.jit` kernel (`gated_delta_recurrent_fwd_kernel`), one Python launcher (`gated_delta_recurrent_fwd`), one `torch.autograd.Function` (`GatedDeltaRecurrentFn`) that wraps the launcher and raises `NotImplementedError` on `.backward`.
- `gated_deltanet/__init__.py` — re-exports `gated_delta_recurrent_naive` and `gated_delta_recurrent`.
- `tests/test_reference.py` — sanity checks for the reference: shapes, NaN-freeness, edge cases (`alpha=0` ⇒ pure write, `alpha=1, beta=1` ⇒ DeltaNet).
- `tests/test_recurrent.py` — Triton kernel vs reference (allclose), then both vs `fla.ops.gated_delta_rule`.
- `benchmarks/bench_recurrent.py` — wall-clock decode-step throughput (us/token) vs fla.

---

## Tasks

### Task 1: Project skeleton + dependencies

**Files:**
- Modify: `pyproject.toml`
- Create: `gated_deltanet/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `benchmarks/.gitkeep`

- [ ] **Step 1: Add development and reference dependencies to pyproject.toml**

Replace the existing `[project]` block with:

```toml
[project]
name = "gated-deltanet-triton"
version = "0.1.0"
description = "Minimal Triton kernel for the GatedDeltaNet recurrent forward pass"
readme = "README.md"
requires-python = ">=3.14"
dependencies = [
    "torch>=2.11.0",
    "triton>=3.6.0",
    "einops>=0.8.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-xdist>=3.6",
    "fla-core>=0.3.0",
]
```

The `fla-core` dependency is the published distribution name of `flash-linear-attention` and is used only in tests/benchmarks. Pinning is loose because we just need any recent release that ships `fla.ops.gated_delta_rule`.

- [ ] **Step 2: Sync the lock and verify install**

Run: `uv sync --extra dev`
Expected: succeeds, `uv pip show triton` and `uv pip show fla-core` both print versions.

- [ ] **Step 3: Create the empty package files**

Create `gated_deltanet/__init__.py` with:

```python
"""Minimal Triton kernels for the GatedDeltaNet recurrent forward pass."""

from gated_deltanet.reference import gated_delta_recurrent_naive
from gated_deltanet.recurrent import gated_delta_recurrent

__all__ = ["gated_delta_recurrent_naive", "gated_delta_recurrent"]
```

Create `tests/__init__.py` as a single empty line (just `\n`) so pytest collects the package.

- [ ] **Step 4: Create the test fixtures**

Create `tests/conftest.py`:

```python
import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton kernels")
    return torch.device("cuda")
```

- [ ] **Step 5: Create benchmarks placeholder**

Create `benchmarks/.gitkeep` as an empty file (placeholder so the directory is committed).

- [ ] **Step 6: Verify pytest collects nothing yet but does not error**

Run: `uv run pytest -q`
Expected: `no tests ran` (exit 5) — directory exists, no test files yet.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock gated_deltanet tests benchmarks
git commit -m "chore: scaffold gated_deltanet package with deps and pytest config"
```

---

### Task 2: Naive PyTorch reference implementation

**Files:**
- Create: `gated_deltanet/reference.py`
- Create: `tests/test_reference.py`

- [ ] **Step 1: Write the failing shape test**

Create `tests/test_reference.py`:

```python
import pytest
import torch

from gated_deltanet.reference import gated_delta_recurrent_naive


def _make_inputs(B=2, H=4, L=8, DK=16, DV=16, dtype=torch.float32, device="cpu"):
    q = torch.randn(B, H, L, DK, dtype=dtype, device=device)
    k = torch.randn(B, H, L, DK, dtype=dtype, device=device)
    k = k / k.norm(dim=-1, keepdim=True)
    q = q / q.norm(dim=-1, keepdim=True)
    v = torch.randn(B, H, L, DV, dtype=dtype, device=device)
    alpha = torch.sigmoid(torch.randn(B, H, L, dtype=dtype, device=device))
    beta = torch.sigmoid(torch.randn(B, H, L, dtype=dtype, device=device))
    return q, k, v, alpha, beta


def test_naive_shapes():
    q, k, v, alpha, beta = _make_inputs()
    o, S = gated_delta_recurrent_naive(q, k, v, alpha, beta)
    assert o.shape == (2, 4, 8, 16)
    assert S.shape == (2, 4, 16, 16)
    assert torch.isfinite(o).all()
    assert torch.isfinite(S).all()
```

- [ ] **Step 2: Run test, verify it fails**

Run: `uv run pytest tests/test_reference.py::test_naive_shapes -v`
Expected: `ImportError` or `ModuleNotFoundError: gated_delta_recurrent_naive` (file doesn't exist yet).

- [ ] **Step 3: Implement the naive reference**

Create `gated_deltanet/reference.py`:

```python
"""Naive PyTorch reference for the GatedDeltaNet recurrent rule.

Convention: state S has shape (D_K, D_V).
For each timestep t (per (batch, head)):

    pred_v = S^T @ k_t                       # (D_V,)
    delta  = beta_t * (v_t - pred_v)         # (D_V,)
    S      = alpha_t * S + outer(k_t, delta) # (D_K, D_V)
    o_t    = S^T @ q_t                       # (D_V,)
"""

from __future__ import annotations

import torch


def gated_delta_recurrent_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference recurrent forward pass.

    Args:
        q, k: (B, H, L, D_K) — queries, keys (caller should pre-L2-normalize).
        v:    (B, H, L, D_V) — values.
        alpha, beta: (B, H, L) — gate values, expected in (0, 1).
        initial_state: (B, H, D_K, D_V) or None.

    Returns:
        o: (B, H, L, D_V)
        final_state: (B, H, D_K, D_V)
    """
    B, H, L, DK = q.shape
    DV = v.shape[-1]
    out_dtype = q.dtype
    device = q.device

    if initial_state is None:
        S = torch.zeros(B, H, DK, DV, device=device, dtype=torch.float32)
    else:
        S = initial_state.to(torch.float32).clone()

    q32 = q.to(torch.float32)
    k32 = k.to(torch.float32)
    v32 = v.to(torch.float32)
    a32 = alpha.to(torch.float32)
    b32 = beta.to(torch.float32)

    o = torch.empty(B, H, L, DV, device=device, dtype=out_dtype)

    for t in range(L):
        q_t = q32[:, :, t]                      # (B, H, DK)
        k_t = k32[:, :, t]                      # (B, H, DK)
        v_t = v32[:, :, t]                      # (B, H, DV)
        a_t = a32[:, :, t, None, None]          # (B, H, 1, 1)
        b_t = b32[:, :, t, None, None]          # (B, H, 1, 1)

        # pred_v = S^T @ k_t   shape (B, H, DV)
        pred_v = torch.einsum("bhkv,bhk->bhv", S, k_t)
        delta = b_t.squeeze(-1) * (v_t - pred_v)             # (B, H, DV)
        # S = alpha * S + outer(k, delta)
        S = a_t * S + torch.einsum("bhk,bhv->bhkv", k_t, delta)
        # o = S^T @ q
        o_t = torch.einsum("bhkv,bhk->bhv", S, q_t)
        o[:, :, t] = o_t.to(out_dtype)

    return o, S.to(out_dtype)
```

- [ ] **Step 4: Run test, verify it passes**

Run: `uv run pytest tests/test_reference.py::test_naive_shapes -v`
Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add gated_deltanet/reference.py tests/test_reference.py
git commit -m "feat: add naive PyTorch reference for gated delta recurrent"
```

---

### Task 3: Reference math edge cases

Two cases pin down the math: `alpha=0` should fully reset state every step (so output only sees the current write), `alpha=1, beta=1, k orthonormal` should reduce to DeltaNet's perfect-recall behavior. These tests are the spec for the kernel.

**Files:**
- Modify: `tests/test_reference.py`

- [ ] **Step 1: Add the alpha=0 reset test**

Append to `tests/test_reference.py`:

```python
def test_naive_alpha_zero_resets_state():
    """alpha=0 means S_t = beta_t * outer(k_t, v_t); o_t = beta_t * <q_t, k_t> * v_t."""
    B, H, L, DK, DV = 1, 1, 4, 8, 8
    q, k, v, _, _ = _make_inputs(B, H, L, DK, DV)
    alpha = torch.zeros(B, H, L)
    beta = torch.ones(B, H, L)

    o, _ = gated_delta_recurrent_naive(q, k, v, alpha, beta)

    # With alpha=0, beta=1: S_t = outer(k_t, v_t - 0) = outer(k_t, v_t)
    # o_t = S_t^T q_t = <k_t, q_t> * v_t
    expected = (q * k).sum(dim=-1, keepdim=True) * v
    torch.testing.assert_close(o, expected, rtol=1e-5, atol=1e-5)
```

- [ ] **Step 2: Add the orthonormal DeltaNet test**

Append to `tests/test_reference.py`:

```python
def test_naive_orthonormal_keys_recall_v():
    """alpha=1, beta=1, orthonormal keys: querying with k_t recovers v_t exactly."""
    B, H, DK, DV = 1, 1, 4, 4
    L = DK  # at most DK orthonormal keys

    # Construct orthonormal keys via QR
    k_raw = torch.randn(L, DK)
    Q_orth, _ = torch.linalg.qr(k_raw)  # (L, DK), orthonormal rows
    k = Q_orth.view(1, 1, L, DK)
    q = k.clone()  # query each key after writing it
    v = torch.randn(1, 1, L, DV)
    alpha = torch.ones(1, 1, L)
    beta = torch.ones(1, 1, L)

    o, _ = gated_delta_recurrent_naive(q, k, v, alpha, beta)
    torch.testing.assert_close(o, v, rtol=1e-4, atol=1e-4)
```

- [ ] **Step 3: Run the new tests, verify both pass**

Run: `uv run pytest tests/test_reference.py -v`
Expected: 3 passed.

If `test_naive_orthonormal_keys_recall_v` fails by a factor of `alpha` on the prediction term, the math convention used by the reference does not match the standard DeltaNet. That is the cue to inspect the formula and reconcile (likely by changing `pred_v` to `alpha_t * pred_v` inside the inner loop). Do not silently weaken the test.

- [ ] **Step 4: Commit**

```bash
git add tests/test_reference.py
git commit -m "test: pin reference math via alpha=0 reset and orthonormal recall"
```

---

### Task 4: Cross-check reference against fla-org naive

Goal: confirm our `gated_delta_recurrent_naive` matches the fla-core naive implementation up to fp32 numerical noise. If it does not, fla wins — adjust our reference until it does. After this task, the reference is frozen.

**Files:**
- Modify: `tests/test_reference.py`

- [ ] **Step 1: Discover the fla naive entry point**

Run: `uv run python -c "from fla.ops.gated_delta_rule import naive_recurrent_gated_delta_rule; help(naive_recurrent_gated_delta_rule)"`
Expected: prints the docstring, including the parameter signature.

If the symbol name differs in the installed version (`fla` rename, layout change), use:

```bash
uv run python -c "import fla.ops.gated_delta_rule as m; print([n for n in dir(m) if 'naive' in n.lower() or 'recurrent' in n.lower()])"
```

Pick the most-naive (slowest) entry point. Record the actual symbol name in a comment at the top of the new test (so the reader can re-find it).

- [ ] **Step 2: Add the fla equivalence test**

Append to `tests/test_reference.py`:

```python
# Symbol name confirmed in Step 1 of Task 4. Update if fla renames it.
try:
    from fla.ops.gated_delta_rule import naive_recurrent_gated_delta_rule as _fla_naive
    HAS_FLA = True
except ImportError:
    HAS_FLA = False


@pytest.mark.skipif(not HAS_FLA, reason="fla-core not installed")
def test_naive_matches_fla(device):
    """Our reference must match fla's naive to fp32 tolerance."""
    B, H, L, DK, DV = 2, 4, 16, 32, 32
    q, k, v, alpha, beta = _make_inputs(B, H, L, DK, DV, device=device)

    o_ours, S_ours = gated_delta_recurrent_naive(q, k, v, alpha, beta)
    # fla signature varies; the common shape contract is (B, H, L, D) for q/k/v.
    # If fla expects (B, L, H, D) instead, transpose before/after.
    o_fla, S_fla = _fla_naive(q, k, v, alpha, beta)

    torch.testing.assert_close(o_ours, o_fla, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(S_ours, S_fla, rtol=1e-4, atol=1e-4)
```

- [ ] **Step 3: Run the test**

Run: `uv run pytest tests/test_reference.py::test_naive_matches_fla -v`

Expected outcomes (in order of likelihood):
1. **Passes.** Reference convention matches fla. Frozen. Move on.
2. **Fails on tensor layout** (e.g. fla expects `(B, L, H, D)`). Fix by adding `permute` calls inside the test, **not** by changing the reference.
3. **Fails on signature** (extra kwargs like `scale`, `output_final_state`). Read fla's docstring, pass through the right defaults, retest.
4. **Fails on values.** This is the convention-divergence case. The reference formula needs adjusting; the most likely fix is the `α` factor on the prediction term. Edit `gated_deltanet/reference.py` so:

   ```python
   pred_v = a_t.squeeze(-1) * torch.einsum("bhkv,bhk->bhv", S, k_t)
   ```

   Re-run Tasks 2 and 3 tests after the change to confirm they still pass (the orthonormal test in Task 3 with `alpha=1` is invariant to this change). Then re-run this test.

- [ ] **Step 4: Commit (after all reference tests green)**

```bash
git add tests/test_reference.py gated_deltanet/reference.py
git commit -m "test: pin reference to fla naive; freeze convention"
```

---

### Task 5: Triton recurrent forward kernel

The kernel is the smallest possible implementation: one program per `(batch, head)`, the entire `(D_K, D_V)` state tile lives in registers, the kernel loops over `L` sequentially. This caps single-token-at-a-time throughput by SM count divided by `B*H`, which is fine for typical decode-time shapes (`B*H ≥ 32`). Multi-tile state (so we can scale to larger D_K, D_V) is explicitly out of scope — Phase 2 will introduce that.

**Files:**
- Create: `gated_deltanet/recurrent.py`

- [ ] **Step 1: Write the kernel and Python launcher (no autograd wrapper yet)**

Create `gated_deltanet/recurrent.py`:

```python
"""Triton recurrent forward kernel for the GatedDeltaNet rule.

One program per (batch, head). Full (D_K, D_V) state held in registers.
Math 1:1 with `gated_delta_recurrent_naive`:

    pred_v = S^T @ k_t
    delta  = beta_t * (v_t - pred_v)
    S      = alpha_t * S + outer(k_t, delta)
    o_t    = S^T @ q_t
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def gated_delta_recurrent_fwd_kernel(
    Q, K, V, A, B, O, S_INIT, S_FINAL,
    L,
    sQ_b, sQ_h, sQ_l,
    sV_b, sV_h, sV_l,
    sA_b, sA_h,
    sS_b, sS_h,
    DK: tl.constexpr,
    DV: tl.constexpr,
    USE_INIT: tl.constexpr,
    STORE_FINAL: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_dk = tl.arange(0, DK)
    offs_dv = tl.arange(0, DV)

    if USE_INIT:
        s_base = S_INIT + pid_b * sS_b + pid_h * sS_h
        S = tl.load(s_base + offs_dk[:, None] * DV + offs_dv[None, :]).to(tl.float32)
    else:
        S = tl.zeros((DK, DV), dtype=tl.float32)

    q_base = Q + pid_b * sQ_b + pid_h * sQ_h
    k_base = K + pid_b * sQ_b + pid_h * sQ_h  # q and k share strides
    v_base = V + pid_b * sV_b + pid_h * sV_h
    a_base = A + pid_b * sA_b + pid_h * sA_h
    o_base = O + pid_b * sV_b + pid_h * sV_h

    for t in range(L):
        q = tl.load(q_base + t * sQ_l + offs_dk).to(tl.float32)
        k = tl.load(k_base + t * sQ_l + offs_dk).to(tl.float32)
        v = tl.load(v_base + t * sV_l + offs_dv).to(tl.float32)
        alpha = tl.load(a_base + t).to(tl.float32)
        # beta is in tensor B with the same strides as A
        beta = tl.load(B + pid_b * sA_b + pid_h * sA_h + t).to(tl.float32)

        # pred_v = S^T @ k    shape (DV,)
        pred_v = tl.sum(S * k[:, None], axis=0)
        delta = beta * (v - pred_v)
        # outer(k, delta): (DK, 1) * (1, DV) = (DK, DV)
        S = alpha * S + k[:, None] * delta[None, :]
        # o = S^T @ q
        o = tl.sum(S * q[:, None], axis=0)
        tl.store(o_base + t * sV_l + offs_dv, o.to(O.dtype.element_ty))

    if STORE_FINAL:
        s_base = S_FINAL + pid_b * sS_b + pid_h * sS_h
        tl.store(s_base + offs_dk[:, None] * DV + offs_dv[None, :], S.to(S_FINAL.dtype.element_ty))


def gated_delta_recurrent_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton-backed recurrent forward.

    Same contract as `gated_delta_recurrent_naive`. Requires CUDA.
    """
    assert q.is_cuda, "Triton kernel requires CUDA tensors"
    B, H, L, DK = q.shape
    DV = v.shape[-1]
    assert k.shape == (B, H, L, DK)
    assert v.shape == (B, H, L, DV)
    assert alpha.shape == (B, H, L)
    assert beta.shape == (B, H, L)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    alpha = alpha.contiguous()
    beta = beta.contiguous()

    o = torch.empty(B, H, L, DV, device=q.device, dtype=q.dtype)
    S_final = torch.empty(B, H, DK, DV, device=q.device, dtype=torch.float32)

    if initial_state is None:
        S_init = torch.empty(1, device=q.device, dtype=torch.float32)  # dummy
        use_init = False
    else:
        S_init = initial_state.contiguous().to(torch.float32)
        use_init = True

    grid = (B, H)
    gated_delta_recurrent_fwd_kernel[grid](
        q, k, v, alpha, beta, o, S_init, S_final,
        L,
        q.stride(0), q.stride(1), q.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        alpha.stride(0), alpha.stride(1),
        S_final.stride(0), S_final.stride(1),
        DK=DK, DV=DV,
        USE_INIT=use_init,
        STORE_FINAL=True,
    )
    return o, S_final
```

- [ ] **Step 2: Smoke-test the kernel compiles and runs**

Run: `uv run python -c "import torch; from gated_deltanet.recurrent import gated_delta_recurrent_fwd; q=torch.randn(1,1,4,8,device='cuda'); k=q.clone(); v=torch.randn(1,1,4,8,device='cuda'); a=torch.full((1,1,4),0.5,device='cuda'); b=torch.full((1,1,4),0.5,device='cuda'); o,s=gated_delta_recurrent_fwd(q,k,v,a,b); print(o.shape, s.shape)"`
Expected: `torch.Size([1, 1, 4, 8]) torch.Size([1, 1, 8, 8])`. No Triton compilation errors.

If you hit a compile error about `tl.sum` axis or `[:, None]` broadcasting on this Triton version, simplify the inner pred/output computation to use `tl.dot` on a `(1, DK) x (DK, DV)` shape, e.g.:

```python
pred_v = tl.sum(S * k[:, None], axis=0)
# alternative if needed:
# pred_v = tl.sum(S * tl.reshape(k, (DK, 1)), axis=0)
```

- [ ] **Step 3: Commit**

```bash
git add gated_deltanet/recurrent.py
git commit -m "feat: add Triton recurrent forward kernel"
```

---

### Task 6: Autograd Function wrapper (forward-only)

A thin `torch.autograd.Function` so the kernel can plug into PyTorch's graph and so the public API is symmetric with `gated_delta_recurrent_naive`. Backward is explicitly out of scope: the wrapper raises if called.

**Files:**
- Modify: `gated_deltanet/recurrent.py`

- [ ] **Step 1: Append the Function class and public entry point**

Append to `gated_deltanet/recurrent.py`:

```python
class GatedDeltaRecurrentFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, alpha, beta, initial_state):
        o, S_final = gated_delta_recurrent_fwd(q, k, v, alpha, beta, initial_state)
        return o, S_final

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError(
            "Backward not implemented in v1. See follow-up ticket for chunkwise "
            "fwd+bwd kernel that supports training."
        )


def gated_delta_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Public entry point: Triton recurrent forward, no_grad enforced."""
    return GatedDeltaRecurrentFn.apply(q, k, v, alpha, beta, initial_state)
```

- [ ] **Step 2: Verify import still works**

Run: `uv run python -c "from gated_deltanet import gated_delta_recurrent; print(gated_delta_recurrent)"`
Expected: prints the function object, no errors.

- [ ] **Step 3: Commit**

```bash
git add gated_deltanet/recurrent.py
git commit -m "feat: add autograd Function wrapper for recurrent kernel"
```

---

### Task 7: Triton kernel correctness vs reference

Numerical equivalence test on small but non-trivial shapes. fp32 throughout to avoid mixing up correctness with precision. Subsequent task adds dtype/edge coverage.

**Files:**
- Create: `tests/test_recurrent.py`

- [ ] **Step 1: Write the failing equivalence test**

Create `tests/test_recurrent.py`:

```python
import pytest
import torch

from gated_deltanet.reference import gated_delta_recurrent_naive
from gated_deltanet.recurrent import gated_delta_recurrent


def _inputs(B, H, L, DK, DV, device, dtype=torch.float32):
    q = torch.randn(B, H, L, DK, device=device, dtype=dtype)
    k = torch.randn(B, H, L, DK, device=device, dtype=dtype)
    q = q / q.norm(dim=-1, keepdim=True)
    k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn(B, H, L, DV, device=device, dtype=dtype)
    alpha = torch.sigmoid(torch.randn(B, H, L, device=device, dtype=dtype))
    beta = torch.sigmoid(torch.randn(B, H, L, device=device, dtype=dtype))
    return q, k, v, alpha, beta


def test_triton_matches_reference_small(device):
    q, k, v, alpha, beta = _inputs(2, 4, 8, 16, 16, device)
    o_ref, S_ref = gated_delta_recurrent_naive(q, k, v, alpha, beta)
    o_tri, S_tri = gated_delta_recurrent(q, k, v, alpha, beta)
    torch.testing.assert_close(o_tri, o_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(S_tri.to(torch.float32), S_ref.to(torch.float32),
                               rtol=1e-4, atol=1e-4)
```

- [ ] **Step 2: Run test, expect it to either pass or surface a real bug**

Run: `uv run pytest tests/test_recurrent.py::test_triton_matches_reference_small -v`
Expected: passes. If it fails, the kernel math is wrong — do **not** loosen tolerances. Walk through the kernel against the reference math line-by-line. Most common causes:
1. Stride mismatch (passed wrong stride to a base pointer).
2. `tl.sum` axis confusion (axis=0 collapses `DK` rows; axis=1 collapses `DV` cols).
3. Forgot to cast to fp32 internally before the matmul-style ops.

- [ ] **Step 3: Add a medium-shape test**

Append to `tests/test_recurrent.py`:

```python
def test_triton_matches_reference_medium(device):
    q, k, v, alpha, beta = _inputs(1, 4, 64, 64, 64, device)
    o_ref, S_ref = gated_delta_recurrent_naive(q, k, v, alpha, beta)
    o_tri, S_tri = gated_delta_recurrent(q, k, v, alpha, beta)
    torch.testing.assert_close(o_tri, o_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(S_tri.to(torch.float32), S_ref.to(torch.float32),
                               rtol=1e-3, atol=1e-3)
```

The looser tolerance reflects the longer accumulation chain (L=64 vs 8); the kernel is bit-different from PyTorch's einsum but algebraically identical.

- [ ] **Step 4: Run both, verify they pass**

Run: `uv run pytest tests/test_recurrent.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_recurrent.py
git commit -m "test: triton recurrent kernel matches naive reference"
```

---

### Task 8: Edge cases, dtypes, initial state

Cover the surface area: bf16 input, non-zero initial state, the alpha=0 / orthonormal cases that nail the math (already tested for the reference; now confirm the kernel honors them too).

**Files:**
- Modify: `tests/test_recurrent.py`

- [ ] **Step 1: Add the bf16 test**

Append to `tests/test_recurrent.py`:

```python
def test_triton_bf16(device):
    q, k, v, alpha, beta = _inputs(2, 4, 32, 32, 32, device, dtype=torch.bfloat16)
    # alpha/beta stay fp32 — common convention for gates
    alpha = alpha.float()
    beta = beta.float()
    o_ref, S_ref = gated_delta_recurrent_naive(q, k, v, alpha, beta)
    o_tri, S_tri = gated_delta_recurrent(q, k, v, alpha, beta)
    # bf16 has ~7-bit mantissa; loosen tolerance accordingly
    torch.testing.assert_close(o_tri.float(), o_ref.float(), rtol=1e-2, atol=1e-2)
```

If the kernel fails because `alpha` and `beta` strides assume contiguous-along-L (and dtype mismatch with `q`), fix the kernel to load alpha/beta from `tl.float32`-strided pointers regardless of `q.dtype`. The launcher already enforces `.contiguous()`; if dtype is the issue, cast `alpha`/`beta` to fp32 inside the launcher before passing pointers.

- [ ] **Step 2: Add the initial-state test**

Append to `tests/test_recurrent.py`:

```python
def test_triton_with_initial_state(device):
    B, H, L, DK, DV = 2, 4, 16, 16, 16
    q, k, v, alpha, beta = _inputs(B, H, L, DK, DV, device)
    S0 = torch.randn(B, H, DK, DV, device=device, dtype=torch.float32) * 0.1

    o_ref, S_ref = gated_delta_recurrent_naive(q, k, v, alpha, beta, initial_state=S0)
    o_tri, S_tri = gated_delta_recurrent(q, k, v, alpha, beta, initial_state=S0)
    torch.testing.assert_close(o_tri, o_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(S_tri.to(torch.float32), S_ref.to(torch.float32),
                               rtol=1e-4, atol=1e-4)
```

- [ ] **Step 3: Add the alpha=0 sanity test**

Append to `tests/test_recurrent.py`:

```python
def test_triton_alpha_zero_resets(device):
    B, H, L, DK, DV = 1, 1, 4, 8, 8
    q, k, v, _, _ = _inputs(B, H, L, DK, DV, device)
    alpha = torch.zeros(B, H, L, device=device)
    beta = torch.ones(B, H, L, device=device)
    o, _ = gated_delta_recurrent(q, k, v, alpha, beta)
    expected = (q * k).sum(dim=-1, keepdim=True) * v
    torch.testing.assert_close(o, expected, rtol=1e-4, atol=1e-4)
```

- [ ] **Step 4: Add a backward-raises test**

Append to `tests/test_recurrent.py`:

```python
def test_backward_raises(device):
    q, k, v, alpha, beta = _inputs(1, 1, 4, 8, 8, device)
    q.requires_grad_(True)
    o, _ = gated_delta_recurrent(q, k, v, alpha, beta)
    with pytest.raises(NotImplementedError, match="Backward not implemented"):
        o.sum().backward()
```

- [ ] **Step 5: Run the full kernel test suite**

Run: `uv run pytest tests/test_recurrent.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add tests/test_recurrent.py gated_deltanet/recurrent.py
git commit -m "test: bf16, initial-state, alpha=0, and no-backward coverage"
```

---

### Task 9: Benchmark vs fla fused_recurrent

Single decode-step throughput. We are slower than fla's tuned kernel — that is fine and expected for v1. The benchmark exists so future kernel tuning has a baseline number to beat.

**Files:**
- Create: `benchmarks/bench_recurrent.py`

- [ ] **Step 1: Write the benchmark script**

Create `benchmarks/bench_recurrent.py`:

```python
"""Decode-step throughput: ours vs fla fused_recurrent.

Run: uv run python benchmarks/bench_recurrent.py
"""

from __future__ import annotations

import time

import torch

from gated_deltanet.recurrent import gated_delta_recurrent


def _bench(fn, *args, warmup=10, iters=50):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    device = torch.device("cuda")
    B, H, L, DK, DV = 4, 16, 1024, 128, 128
    dtype = torch.bfloat16
    q = torch.randn(B, H, L, DK, device=device, dtype=dtype)
    k = torch.randn(B, H, L, DK, device=device, dtype=dtype)
    q = q / q.norm(dim=-1, keepdim=True)
    k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn(B, H, L, DV, device=device, dtype=dtype)
    alpha = torch.sigmoid(torch.randn(B, H, L, device=device, dtype=torch.float32))
    beta = torch.sigmoid(torch.randn(B, H, L, device=device, dtype=torch.float32))

    t_ours = _bench(gated_delta_recurrent, q, k, v, alpha, beta)
    print(f"ours:    {t_ours * 1e3:7.3f} ms/call  ({B*H*L / t_ours / 1e6:7.2f} M tok/s)")

    try:
        from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
    except ImportError:
        print("fla-core not installed; skipping fla comparison")
        return

    t_fla = _bench(fused_recurrent_gated_delta_rule, q, k, v, alpha, beta)
    print(f"fla:     {t_fla * 1e3:7.3f} ms/call  ({B*H*L / t_fla / 1e6:7.2f} M tok/s)")
    print(f"speedup of fla over ours: {t_ours / t_fla:5.2f}x")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the benchmark**

Run: `uv run python benchmarks/bench_recurrent.py`
Expected: prints two lines (ours, fla) and a speedup ratio. Ours will likely be 5-20x slower than fla — note the number, that is the v1 baseline.

If `fused_recurrent_gated_delta_rule` is not the right symbol in the installed fla version, look it up the same way as in Task 4 step 1.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/bench_recurrent.py
git commit -m "bench: decode throughput vs fla fused_recurrent"
```

---

### Task 10: Demo script + README

Final polish. The README should be enough that a fresh reader knows what this is, how the math maps onto the code, and what is intentionally missing.

**Files:**
- Modify: `main.py`
- Modify: `README.md`

- [ ] **Step 1: Replace main.py with a usage demo**

Replace the entire contents of `main.py`:

```python
"""End-to-end demo: random GatedDeltaNet recurrent forward pass on GPU."""

import torch

from gated_deltanet import gated_delta_recurrent


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; this demo requires a GPU.")
        return

    device = torch.device("cuda")
    B, H, L, DK, DV = 1, 4, 32, 64, 64
    dtype = torch.bfloat16

    q = torch.randn(B, H, L, DK, device=device, dtype=dtype)
    k = torch.randn(B, H, L, DK, device=device, dtype=dtype)
    q = q / q.norm(dim=-1, keepdim=True)
    k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn(B, H, L, DV, device=device, dtype=dtype)
    alpha = torch.sigmoid(torch.randn(B, H, L, device=device))
    beta = torch.sigmoid(torch.randn(B, H, L, device=device))

    o, S_final = gated_delta_recurrent(q, k, v, alpha, beta)
    print(f"output: {tuple(o.shape)}  dtype={o.dtype}")
    print(f"final state: {tuple(S_final.shape)}  dtype={S_final.dtype}")
    print(f"output mean={o.float().mean().item():.4f}  std={o.float().std().item():.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the README**

Replace the entire contents of `README.md`:

````markdown
# gated-deltanet-triton

Minimal Triton kernel for the GatedDeltaNet recurrent forward pass. v1 covers
inference (decode-time, single-token) only.

## What this is

A small, deliberately simple implementation of the gated delta rule from
[Gated Delta Networks (Yang, Kautz, Hatamizadeh, ICLR 2025)](https://arxiv.org/abs/2412.06464),
faithful to the convention in
[fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention).

The state $S \in \mathbb{R}^{D_K \times D_V}$ updates as:

```
pred_v = S^T @ k_t                       # shape (D_V,)
delta  = beta_t * (v_t - pred_v)         # shape (D_V,)
S      = alpha_t * S + outer(k_t, delta) # shape (D_K, D_V)
o_t    = S^T @ q_t                       # shape (D_V,)
```

`alpha_t ∈ (0, 1)` is the decay gate (forget gate); `alpha → 0` resets memory,
`alpha → 1` recovers DeltaNet. `beta_t ∈ (0, 1)` is the write strength.

## Install

```
uv sync --extra dev
```

## Usage

```python
import torch
from gated_deltanet import gated_delta_recurrent

q = torch.randn(1, 4, 32, 64, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 4, 32, 64, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, 4, 32, 64, device="cuda", dtype=torch.bfloat16)
q, k = q / q.norm(dim=-1, keepdim=True), k / k.norm(dim=-1, keepdim=True)
alpha = torch.sigmoid(torch.randn(1, 4, 32, device="cuda"))
beta  = torch.sigmoid(torch.randn(1, 4, 32, device="cuda"))

o, S_final = gated_delta_recurrent(q, k, v, alpha, beta)
```

## What is *not* here

- **Backward pass.** Forward only. Calling `.backward()` raises `NotImplementedError`.
- **Chunkwise kernel.** No WY representation, no chunked parallel form. Decode-only.
- **Fused kernels.** No RMSNorm-gated, no causal conv1d, no Q/K projection fusion.
- **Auto-tuning.** No `triton.autotune`. Single block configuration.

These live in follow-up tickets (BAL-28b for chunkwise, BAL-28c for fused).

## Testing

```
uv run pytest -v
```

`tests/test_reference.py` pins the math via the fla naive cross-check and two
hand-derived edge cases. `tests/test_recurrent.py` checks the Triton kernel
matches the reference at fp32, bf16, with and without an initial state.

## Benchmarking

```
uv run python benchmarks/bench_recurrent.py
```

Prints decode throughput for our kernel and `fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule`.
Ours is slower; this is expected and is the v1 baseline.
````

- [ ] **Step 3: Run the demo to confirm it works end-to-end**

Run: `uv run python main.py`
Expected: prints output shape, final-state shape, and a finite mean/std. No errors.

- [ ] **Step 4: Run the entire test suite as a final smoke check**

Run: `uv run pytest -v`
Expected: all tests pass (3 reference + 5 recurrent + 1 fla cross-check = 9).

- [ ] **Step 5: Commit**

```bash
git add main.py README.md
git commit -m "docs: README, math mapping, and end-to-end demo"
```

---

## Self-review

**Spec coverage (BAL-28 acceptance criteria):**
- "Recurrent kernel: forward pass + unit test (fla naive numerical equivalence)" → Tasks 5–8 (Triton kernel, Function wrapper, equivalence vs reference, fla cross-check).
- "Chunkwise kernel: forward + backward, WY representation" → **deferred to BAL-28b** by the scope decision at the top.
- "Fused kernels (RMSNorm gated, causal conv1d)" → **deferred to BAL-28c**.
- "Performance benchmark vs fla-core" → Task 9 (recurrent only).
- "Code documentation + math-code mapping comments" → reference.py docstring + recurrent.py docstring + README math block.

The deferred items are explicitly called out in the README's "What is not here" section so future readers do not assume they exist.

**No-placeholder check:** every code step contains the actual code. Every test step contains the actual test. The Task 4 fix-when-it-fails branch shows the concrete edit (the `α` factor). The Task 5 troubleshooting branch shows the concrete `tl.reshape` alternative. Task 8 step 1 has a concrete fix (cast alpha/beta to fp32 in the launcher) for the most likely failure.

**Type / signature consistency:** the function and tensor names line up across tasks:
- `gated_delta_recurrent_naive(q, k, v, alpha, beta, initial_state=None) -> (o, S_final)` — Task 2, used in Tasks 3, 4, 7, 8.
- `gated_delta_recurrent_fwd(...)` — same signature — Task 5, called by Task 6.
- `gated_delta_recurrent(...)` — same signature — Task 6, used in Tasks 7, 8, 9, 10.
- All three return `(o, final_state)` with shapes `(B,H,L,DV)` and `(B,H,DK,DV)`.
- The kernel name `gated_delta_recurrent_fwd_kernel` is consistent in Tasks 5 and 6.

No drift detected.
