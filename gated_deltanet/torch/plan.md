# Gated DeltaNet (PyTorch Reference Kernel) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained, two-file PyTorch reference implementation of the
Gated DeltaNet kernel: a naive O(L) recurrent forward in
`gated_delta_recurrent_torch.py` and a chunkwise O(L · C) forward via the WY
(UT-transform) representation in `gated_delta_chunkwise_torch.py`. Both share the
same public signature; autograd handles backward.

**Architecture:**
- Two sibling modules under `gated_deltanet/torch/`, each owning one public
  function with an identical signature. The recurrent file is the golden,
  unconditional reference; the chunkwise file is the optimized variant validated
  against it.
- A small private helper module `_common.py` holds shared pieces (effective-gate
  computation from `(g, A_log, dt_bias)`, optional QK-L2-norm preprocessing) so
  the two implementations cannot drift on those conventions.
- The WY triangular-inverse helper is private to the chunkwise module
  (`_wy_inv_triangular`).
- No nn.Module wrappers, no projections, no convs. Backward is autograd; no
  hand-written backward.

**Tech Stack:** Python 3.14, PyTorch 2.11 (cu128), pytest, einops (already in deps).
Tests use `torch.float64` on CPU for numerical equivalence checks plus a CUDA
fp32 smoke pass when `torch.cuda.is_available()` is true.

---

## Public API (frozen — match the existing stubs exactly)

Both files expose one function with this signature:

```python
def gated_delta_recurrent_torch(   # or gated_delta_chunkwise_torch
    q: torch.Tensor,                # (B, T, H, K)
    k: torch.Tensor,                # (B, T, H, K)
    v: torch.Tensor,                # (B, T, H, V)
    g: torch.Tensor,                # (B, T, H) — raw "dt"; see effective-gate math below
    beta: torch.Tensor,             # (B, T, H) — write strength in [0, 1]
    A_log: torch.Tensor,            # (H,)      — per-head log of state-space scalar A
    initial_state: torch.Tensor | None = None,   # (B, H, K, V)
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,      # variable-length packing (NOT IMPLEMENTED in this plan)
    use_qk_l2norm_in_kernel: bool = False,       # L2-normalize q and k along K before recurrence
    use_gate_in_kernel: bool = False,            # NOT IMPLEMENTED in this plan (raises)
    dt_bias: float = 0.0,                        # scalar bias added to g before softplus
) -> tuple[torch.Tensor, torch.Tensor | None]:
    ...
```

Returns `(o, final_state)` where `o.shape == (B, T, H, V)` and `final_state` is
`(B, H, K, V)` if `output_final_state=True`, else `None`.

### Effective gate math (single source of truth)

`g` is treated as raw `dt`. Per-head effective per-step **log-decay** is:

    g_eff_{b,t,h} = -softplus(g_{b,t,h} + dt_bias) * exp(A_log_h)

i.e. `decay_{b,t,h} = exp(g_eff_{b,t,h}) ∈ (0, 1]`. This is the only place where
`A_log` and `dt_bias` enter the kernel; both files MUST go through `_common.effective_gate`.

### Recurrence (single source of truth)

For each `t = 0..T-1`, per `(b, h)`, with `scale = K ** -0.5`:

    S_t_pre  = exp(g_eff_t) * S_{t-1}                     # gated decay
    pred_v_t = S_t_pre^T @ k_t                            # (V,)
    S_t      = S_t_pre + beta_t * (v_t - pred_v_t) outer k_t
    o_t      = (scale * q_t)^T @ S_t                      # (V,)

`S_0 = initial_state` (zeros if not provided). `final_state = S_{T-1}` if
`output_final_state=True`. If `use_qk_l2norm_in_kernel=True`, replace `q, k` with
`F.normalize(q, dim=-1)`, `F.normalize(k, dim=-1)` BEFORE the recurrence (same
preprocessing for both files).

### Unsupported flags (in this plan)

The following raise `NotImplementedError` with a clear message at the top of
each public function:
- `cu_seqlens is not None` — variable-length packing.
- `use_gate_in_kernel=True` — output-gate path requires an extra tensor not in
  this signature; defer to the layer-wrapper plan.

---

## File Structure

| Path | Status | Responsibility |
|---|---|---|
| `gated_deltanet/torch/_common.py` | create | `effective_gate(g, A_log, dt_bias) -> (B,T,H)`; `maybe_l2norm_qk(q, k, enabled) -> (q,k)`; `validate_unsupported(cu_seqlens, use_gate_in_kernel)`. |
| `gated_deltanet/torch/gated_delta_recurrent_torch.py` | modify | Implement the naive O(L) loop. |
| `gated_deltanet/torch/gated_delta_chunkwise_torch.py` | modify | Implement chunkwise O(L · C) with WY; chunk size constant `C = 64` defined as module-level `_CHUNK_SIZE`. |
| `gated_deltanet/torch/__init__.py` | modify | Re-export both `gated_delta_recurrent_torch`, `gated_delta_chunkwise_torch`. |
| `gated_deltanet/__init__.py` | modify | Re-export top-level: `from .torch import gated_delta_recurrent_torch, gated_delta_chunkwise_torch`. |
| `tests/test_gated_delta_recurrent_torch.py` | create | Shape, decay, beta=0/1, initial-state, A_log/dt_bias sanity, l2norm flag, gradcheck, unsupported-flag errors. |
| `tests/test_gated_delta_chunkwise_torch.py` | create | Parity vs recurrent across multiple T (incl. non-multiple of `_CHUNK_SIZE`), gradcheck, CUDA smoke, unsupported-flag errors. |
| `tests/test_common.py` | create | `effective_gate` math, `maybe_l2norm_qk` no-op vs enabled. |

Each task is bite-sized (≤ 5 minutes) and follows red → green → commit.

---

### Task 1: Common helpers + scaffold exports

**Files:**
- Create: `gated_deltanet/torch/_common.py`
- Modify: `gated_deltanet/torch/__init__.py`
- Modify: `gated_deltanet/__init__.py`

- [ ] **Step 1: Write `_common.py`**

```python
"""Shared helpers for the PyTorch reference kernels."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def effective_gate(
    g: torch.Tensor,        # (B, T, H)
    A_log: torch.Tensor,    # (H,)
    dt_bias: float,
) -> torch.Tensor:
    """Return per-step log-decay g_eff with shape (B, T, H).

    g_eff_{b,t,h} = -softplus(g_{b,t,h} + dt_bias) * exp(A_log_h)
    """
    if A_log.ndim != 1 or A_log.shape[0] != g.shape[-1]:
        raise ValueError(
            f"A_log must have shape (H={g.shape[-1]},), got {tuple(A_log.shape)}"
        )
    softplus_dt = F.softplus(g + dt_bias)               # (B, T, H)
    return -softplus_dt * torch.exp(A_log)              # broadcast (H,) over (B,T,H)


def maybe_l2norm_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not enabled:
        return q, k
    return F.normalize(q, dim=-1), F.normalize(k, dim=-1)


def validate_unsupported(
    cu_seqlens: torch.Tensor | None,
    use_gate_in_kernel: bool,
) -> None:
    if cu_seqlens is not None:
        raise NotImplementedError("cu_seqlens (variable-length packing) is not supported in this kernel.")
    if use_gate_in_kernel:
        raise NotImplementedError("use_gate_in_kernel=True requires an output-gate tensor not in this signature.")
```

- [ ] **Step 2: `gated_deltanet/torch/__init__.py`**

```python
from gated_deltanet.torch.gated_delta_chunkwise_torch import gated_delta_chunkwise_torch
from gated_deltanet.torch.gated_delta_recurrent_torch import gated_delta_recurrent_torch

__all__ = ["gated_delta_recurrent_torch", "gated_delta_chunkwise_torch"]
```

- [ ] **Step 3: `gated_deltanet/__init__.py`**

```python
from gated_deltanet.torch import gated_delta_chunkwise_torch, gated_delta_recurrent_torch

__all__ = ["gated_delta_recurrent_torch", "gated_delta_chunkwise_torch"]
```

- [ ] **Step 4: Smoke import**

Run: `uv run python -c "from gated_deltanet import gated_delta_recurrent_torch, gated_delta_chunkwise_torch; print('ok')"`
Expected: `ok` (functions resolve, both still raise `NotImplementedError` if called).

- [ ] **Step 5: Tests for `_common`**

Create `tests/test_common.py`:

```python
import math

import torch

from gated_deltanet.torch._common import (
    effective_gate,
    maybe_l2norm_qk,
    validate_unsupported,
)


def test_effective_gate_zero_dt_bias_zero_g_zero_A():
    # g=0, dt_bias=0, A_log=0 => softplus(0)=ln(2), exp(0)=1 => g_eff = -ln(2)
    B, T, H = 2, 3, 4
    g = torch.zeros(B, T, H)
    A_log = torch.zeros(H)
    out = effective_gate(g, A_log, dt_bias=0.0)
    expected = torch.full_like(out, -math.log(2.0))
    torch.testing.assert_close(out, expected, atol=1e-7, rtol=1e-7)


def test_effective_gate_per_head_A_broadcast():
    g = torch.zeros(1, 1, 3)
    A_log = torch.tensor([0.0, 1.0, -1.0])
    out = effective_gate(g, A_log, dt_bias=0.0)[0, 0]   # (3,)
    expected = -math.log(2.0) * torch.exp(A_log)
    torch.testing.assert_close(out, expected, atol=1e-7, rtol=1e-7)


def test_maybe_l2norm_disabled_is_noop():
    q = torch.randn(2, 3, 4, 5)
    k = torch.randn(2, 3, 4, 5)
    q2, k2 = maybe_l2norm_qk(q, k, enabled=False)
    assert q2 is q and k2 is k


def test_maybe_l2norm_enabled_unit_norm():
    q = torch.randn(2, 3, 4, 5)
    k = torch.randn(2, 3, 4, 5)
    q2, k2 = maybe_l2norm_qk(q, k, enabled=True)
    torch.testing.assert_close(q2.norm(dim=-1), torch.ones_like(q2.norm(dim=-1)), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(k2.norm(dim=-1), torch.ones_like(k2.norm(dim=-1)), atol=1e-6, rtol=1e-6)


def test_validate_unsupported_raises():
    import pytest

    with pytest.raises(NotImplementedError):
        validate_unsupported(cu_seqlens=torch.tensor([0, 4]), use_gate_in_kernel=False)
    with pytest.raises(NotImplementedError):
        validate_unsupported(cu_seqlens=None, use_gate_in_kernel=True)
    validate_unsupported(cu_seqlens=None, use_gate_in_kernel=False)  # no-op
```

- [ ] **Step 6: Run common tests**

Run: `uv run pytest tests/test_common.py -v`
Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add gated_deltanet/torch/_common.py gated_deltanet/torch/__init__.py gated_deltanet/__init__.py tests/test_common.py
git commit -m "feat(torch): add common helpers and package re-exports"
```

---

### Task 2: Failing shape test for `gated_delta_recurrent_torch`

**Files:**
- Create: `tests/test_gated_delta_recurrent_torch.py`

- [ ] **Step 1: Write the failing test**

```python
import torch

from gated_deltanet import gated_delta_recurrent_torch


def _make_inputs(B=2, T=8, H=2, K=4, V=4, dtype=torch.float64, device="cpu", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, T, H, K, generator=g, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, generator=g, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, generator=g, dtype=dtype, device=device)
    dt = torch.randn(B, T, H, generator=g, dtype=dtype, device=device)            # raw g
    beta = torch.sigmoid(torch.randn(B, T, H, generator=g, dtype=dtype, device=device))
    A_log = torch.randn(H, generator=g, dtype=dtype, device=device) * 0.1          # small
    return q, k, v, dt, beta, A_log


def test_recurrent_output_shapes():
    q, k, v, g, beta, A_log = _make_inputs()
    o, final = gated_delta_recurrent_torch(
        q, k, v, g, beta, A_log, output_final_state=True,
    )
    assert o.shape == (2, 8, 2, 4)
    assert final is not None and final.shape == (2, 2, 4, 4)
```

- [ ] **Step 2: Run; expect RED**

Run: `uv run pytest tests/test_gated_delta_recurrent_torch.py::test_recurrent_output_shapes -v`
Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_gated_delta_recurrent_torch.py
git commit -m "test(torch): failing shape test for gated_delta_recurrent_torch"
```

---

### Task 3: Implement `gated_delta_recurrent_torch` (naive O(L) loop)

**Files:**
- Modify: `gated_deltanet/torch/gated_delta_recurrent_torch.py`

- [ ] **Step 1: Replace the stub**

```python
"""API for Gated DeltaNet Recurrent implementation."""

from __future__ import annotations

import torch

from gated_deltanet.torch._common import (
    effective_gate,
    maybe_l2norm_qk,
    validate_unsupported,
)


def gated_delta_recurrent_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    dt_bias: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    validate_unsupported(cu_seqlens, use_gate_in_kernel)
    q, k = maybe_l2norm_qk(q, k, enabled=use_qk_l2norm_in_kernel)

    B, T, H, K = q.shape
    V = v.shape[-1]
    scale = K ** -0.5
    dtype, device = q.dtype, q.device

    if initial_state is None:
        S = torch.zeros(B, H, K, V, dtype=dtype, device=device)
    else:
        S = initial_state.clone()

    g_eff = effective_gate(g, A_log, dt_bias)              # (B, T, H), <= 0
    decay = torch.exp(g_eff)                               # (B, T, H), in (0, 1]

    out = torch.empty(B, T, H, V, dtype=dtype, device=device)

    for t in range(T):
        S = S * decay[:, t].unsqueeze(-1).unsqueeze(-1)    # (B, H, K, V)
        k_t = k[:, t]                                      # (B, H, K)
        v_t = v[:, t]                                      # (B, H, V)
        q_t = q[:, t] * scale                              # (B, H, K)
        beta_t = beta[:, t].unsqueeze(-1)                  # (B, H, 1)

        pred_v = torch.einsum("bhkv,bhk->bhv", S, k_t)     # (B, H, V)
        delta = beta_t * (v_t - pred_v)                    # (B, H, V)
        S = S + torch.einsum("bhk,bhv->bhkv", k_t, delta)

        out[:, t] = torch.einsum("bhkv,bhk->bhv", S, q_t)

    final = S if output_final_state else None
    return out, final
```

- [ ] **Step 2: Run shape test; expect GREEN**

Run: `uv run pytest tests/test_gated_delta_recurrent_torch.py::test_recurrent_output_shapes -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add gated_deltanet/torch/gated_delta_recurrent_torch.py
git commit -m "feat(torch): implement naive recurrent forward for gated delta rule"
```

---

### Task 4: Math edge-case tests for the recurrent path

**Files:**
- Modify: `tests/test_gated_delta_recurrent_torch.py`

- [ ] **Step 1: Add five targeted tests**

```python
import math


def test_recurrent_g_to_minus_inf_collapses_state():
    # softplus(50) ~= 50; exp(A_log)=exp(0)=1; g_eff ~= -50; decay ~= 0
    # => state collapses each step => output at t=0 is from a fresh write only.
    q, k, v, _g, beta, _A = _make_inputs()
    g = torch.full_like(_g, 50.0)              # so softplus(50) is huge
    A_log = torch.zeros_like(_A)
    o, _ = gated_delta_recurrent_torch(q, k, v, g, beta, A_log)
    K = q.shape[-1]
    scale = K ** -0.5
    qk0 = (q[:, 0] * k[:, 0]).sum(-1, keepdim=True) * scale
    expected_0 = qk0 * beta[:, 0].unsqueeze(-1) * v[:, 0]
    torch.testing.assert_close(o[:, 0], expected_0, atol=1e-5, rtol=1e-5)


def test_recurrent_beta_zero_with_zero_initial_state_is_zero_output():
    q, k, v, g, _, A_log = _make_inputs()
    beta = torch.zeros_like(_make_inputs()[4])
    o, _ = gated_delta_recurrent_torch(q, k, v, g, beta, A_log)
    torch.testing.assert_close(o, torch.zeros_like(o), atol=1e-9, rtol=1e-9)


def test_recurrent_initial_state_passthrough_when_beta_zero_decay_one():
    # decay=1 needs g_eff=0; pick A_log=-inf-equivalent (very negative) so exp(A_log)~=0
    # Easier: set softplus(g+dt_bias) = 0 by taking g -> -inf. We'll just bypass
    # by constructing inputs that yield decay=1 numerically.
    B, T, H, K, V = 2, 5, 2, 4, 4
    q = torch.randn(B, T, H, K, dtype=torch.float64)
    k = torch.randn(B, T, H, K, dtype=torch.float64)
    v = torch.randn(B, T, H, V, dtype=torch.float64)
    # Achieve g_eff ~= 0 via A_log very negative.
    g = torch.zeros(B, T, H, dtype=torch.float64)
    A_log = torch.full((H,), -50.0, dtype=torch.float64)
    beta = torch.zeros(B, T, H, dtype=torch.float64)
    S0 = torch.randn(B, H, K, V, dtype=torch.float64)

    o, final = gated_delta_recurrent_torch(
        q, k, v, g, beta, A_log,
        initial_state=S0, output_final_state=True,
    )
    scale = K ** -0.5
    expected = scale * torch.einsum("bthk,bhkv->bthv", q, S0)
    torch.testing.assert_close(o, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(final, S0, atol=1e-9, rtol=1e-9)


def test_recurrent_beta_one_overwrites_pred_at_same_key():
    B, T, H, K, V = 1, 1, 1, 4, 4
    q = torch.randn(B, T, H, K, dtype=torch.float64)
    k = torch.randn(B, T, H, K, dtype=torch.float64)
    v = torch.randn(B, T, H, V, dtype=torch.float64)
    g = torch.full((B, T, H), -50.0, dtype=torch.float64)   # any decay; pred_prev=0 since S0=0
    A_log = torch.zeros(H, dtype=torch.float64)
    beta = torch.ones(B, T, H, dtype=torch.float64)
    _, final = gated_delta_recurrent_torch(
        q, k, v, g, beta, A_log, output_final_state=True,
    )
    pred_v = torch.einsum("bhkv,bhk->bhv", final, k[:, 0])
    torch.testing.assert_close(pred_v, v[:, 0], atol=1e-5, rtol=1e-5)


def test_recurrent_qk_l2norm_flag_changes_output():
    q, k, v, g, beta, A_log = _make_inputs()
    o_off, _ = gated_delta_recurrent_torch(
        q, k, v, g, beta, A_log, use_qk_l2norm_in_kernel=False,
    )
    o_on, _ = gated_delta_recurrent_torch(
        q, k, v, g, beta, A_log, use_qk_l2norm_in_kernel=True,
    )
    # Inputs are random and not unit-norm => outputs must differ.
    assert not torch.allclose(o_off, o_on, atol=1e-3, rtol=1e-3)


def test_recurrent_unsupported_flags_raise():
    import pytest

    q, k, v, g, beta, A_log = _make_inputs(B=1, T=2, H=1)
    with pytest.raises(NotImplementedError):
        gated_delta_recurrent_torch(q, k, v, g, beta, A_log, cu_seqlens=torch.tensor([0, 2]))
    with pytest.raises(NotImplementedError):
        gated_delta_recurrent_torch(q, k, v, g, beta, A_log, use_gate_in_kernel=True)
```

- [ ] **Step 2: Run; expect all GREEN**

Run: `uv run pytest tests/test_gated_delta_recurrent_torch.py -v`
Expected: 7 passed (1 from Task 2 + 6 here).

- [ ] **Step 3: Commit**

```bash
git add tests/test_gated_delta_recurrent_torch.py
git commit -m "test(torch): math edge cases and flag tests for recurrent forward"
```

---

### Task 5: Autograd grad-check on the recurrent path

**Files:**
- Modify: `tests/test_gated_delta_recurrent_torch.py`

- [ ] **Step 1: Add gradcheck**

```python
def test_recurrent_gradcheck():
    torch.manual_seed(0)
    B, T, H, K, V = 1, 3, 1, 3, 3
    q = torch.randn(B, T, H, K, dtype=torch.float64, requires_grad=True)
    k = torch.randn(B, T, H, K, dtype=torch.float64, requires_grad=True)
    v = torch.randn(B, T, H, V, dtype=torch.float64, requires_grad=True)
    g = torch.randn(B, T, H, dtype=torch.float64, requires_grad=True)
    beta = torch.sigmoid(torch.randn(B, T, H, dtype=torch.float64)).detach().requires_grad_(True)
    A_log = (torch.randn(H, dtype=torch.float64) * 0.1).detach().requires_grad_(True)

    def f(q, k, v, g, beta, A_log):
        o, _ = gated_delta_recurrent_torch(q, k, v, g, beta, A_log)
        return o

    assert torch.autograd.gradcheck(f, (q, k, v, g, beta, A_log), eps=1e-6, atol=1e-4)
```

- [ ] **Step 2: Run gradcheck**

Run: `uv run pytest tests/test_gated_delta_recurrent_torch.py::test_recurrent_gradcheck -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_gated_delta_recurrent_torch.py
git commit -m "test(torch): autograd gradcheck for recurrent forward"
```

---

### Task 6: Failing parity test — chunkwise vs recurrent

**Files:**
- Create: `tests/test_gated_delta_chunkwise_torch.py`

- [ ] **Step 1: Write parity tests against the recurrent reference**

```python
import pytest
import torch

from gated_deltanet import gated_delta_chunkwise_torch, gated_delta_recurrent_torch


def _inputs(B, T, H, K, V, seed=0, dtype=torch.float64):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(B, T, H, K, generator=gen, dtype=dtype)
    k = torch.randn(B, T, H, K, generator=gen, dtype=dtype)
    v = torch.randn(B, T, H, V, generator=gen, dtype=dtype)
    g = torch.randn(B, T, H, generator=gen, dtype=dtype)
    beta = torch.sigmoid(torch.randn(B, T, H, generator=gen, dtype=dtype))
    A_log = (torch.randn(H, generator=gen, dtype=dtype) * 0.1)
    return q, k, v, g, beta, A_log


@pytest.mark.parametrize("T", [16, 64, 128])
def test_chunkwise_matches_recurrent_full_block(T):
    # T is a multiple of the internal _CHUNK_SIZE=64 (and below it).
    q, k, v, g, beta, A_log = _inputs(B=2, T=T, H=2, K=8, V=8)
    o_ref, s_ref = gated_delta_recurrent_torch(
        q, k, v, g, beta, A_log, output_final_state=True,
    )
    o_chk, s_chk = gated_delta_chunkwise_torch(
        q, k, v, g, beta, A_log, output_final_state=True,
    )
    torch.testing.assert_close(o_chk, o_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(s_chk, s_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("T", [1, 7, 13, 65, 130])
def test_chunkwise_matches_recurrent_partial_last_chunk(T):
    # T not a multiple of 64 — last chunk is short.
    q, k, v, g, beta, A_log = _inputs(B=1, T=T, H=2, K=4, V=4, seed=T)
    o_ref, _ = gated_delta_recurrent_torch(q, k, v, g, beta, A_log)
    o_chk, _ = gated_delta_chunkwise_torch(q, k, v, g, beta, A_log)
    torch.testing.assert_close(o_chk, o_ref, atol=1e-5, rtol=1e-5)


def test_chunkwise_initial_state_parity():
    q, k, v, g, beta, A_log = _inputs(B=2, T=33, H=2, K=4, V=4, seed=42)
    S0 = torch.randn(2, 2, 4, 4, dtype=torch.float64)
    o_ref, sf_ref = gated_delta_recurrent_torch(
        q, k, v, g, beta, A_log, initial_state=S0, output_final_state=True,
    )
    o_chk, sf_chk = gated_delta_chunkwise_torch(
        q, k, v, g, beta, A_log, initial_state=S0, output_final_state=True,
    )
    torch.testing.assert_close(o_chk, o_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(sf_chk, sf_ref, atol=1e-5, rtol=1e-5)


def test_chunkwise_unsupported_flags_raise():
    q, k, v, g, beta, A_log = _inputs(B=1, T=4, H=1, K=2, V=2)
    with pytest.raises(NotImplementedError):
        gated_delta_chunkwise_torch(q, k, v, g, beta, A_log, cu_seqlens=torch.tensor([0, 4]))
    with pytest.raises(NotImplementedError):
        gated_delta_chunkwise_torch(q, k, v, g, beta, A_log, use_gate_in_kernel=True)
```

- [ ] **Step 2: Run; expect RED (NotImplementedError)**

Run: `uv run pytest tests/test_gated_delta_chunkwise_torch.py -v`
Expected: FAIL on every test except the unsupported-flag one (which will fail
because the function raises before reaching the flag check — that's fine, this
test is just to ensure the file is wired up).

- [ ] **Step 3: Commit**

```bash
git add tests/test_gated_delta_chunkwise_torch.py
git commit -m "test(torch): failing parity tests for chunkwise vs recurrent"
```

---

### Task 7: WY (UT-transform) helper inside the chunkwise file

**Files:**
- Modify: `gated_deltanet/torch/gated_delta_chunkwise_torch.py`

**Background:** Within a chunk of length `C`, the in-chunk recurrence stays linear
in `S` but each step subtracts the *current* `S @ k_t`. Solving the chunk in
closed form requires inverting a unit-lower-triangular system. We compute, per
chunk, `T = inv(I + tril(diag(beta) K K^T, -1))` so the in-chunk update reduces
to two matmuls.

- [ ] **Step 1: Add helper near the top of the chunkwise file**

```python
"""API for Gated DeltaNet Chunkwise implementation."""

from __future__ import annotations

import torch

from gated_deltanet.torch._common import (
    effective_gate,
    maybe_l2norm_qk,
    validate_unsupported,
)

_CHUNK_SIZE = 64


def _wy_inv_triangular(k_chunk: torch.Tensor, beta_chunk: torch.Tensor) -> torch.Tensor:
    """Compute T = inv(I + tril(diag(beta) @ K @ K^T, -1)) for one chunk.

    Args:
        k_chunk:    (B, H, C, K)
        beta_chunk: (B, H, C)

    Returns:
        T: (B, H, C, C) lower-triangular with 1s on diagonal.
    """
    C = k_chunk.shape[-2]
    device, dtype = k_chunk.device, k_chunk.dtype
    A = torch.einsum("bhck,bhdk->bhcd", k_chunk, k_chunk)
    A = beta_chunk.unsqueeze(-1) * A
    A = torch.tril(A, diagonal=-1)
    eye = torch.eye(C, dtype=dtype, device=device).expand_as(A)
    return torch.linalg.solve_triangular(eye + A, eye, upper=False, unitriangular=True)
```

- [ ] **Step 2: Sanity tests for the helper**

Append to `tests/test_gated_delta_chunkwise_torch.py`:

```python
from gated_deltanet.torch.gated_delta_chunkwise_torch import _wy_inv_triangular


def test_wy_inv_triangular_identity_when_beta_zero():
    B, H, C, K = 2, 2, 4, 3
    k_chunk = torch.randn(B, H, C, K, dtype=torch.float64)
    beta_chunk = torch.zeros(B, H, C, dtype=torch.float64)
    T = _wy_inv_triangular(k_chunk, beta_chunk)
    torch.testing.assert_close(
        T, torch.eye(C, dtype=torch.float64).expand(B, H, C, C), atol=0, rtol=0,
    )


def test_wy_inv_triangular_left_inverse_property():
    torch.manual_seed(0)
    B, H, C, K = 1, 1, 5, 3
    k_chunk = torch.randn(B, H, C, K, dtype=torch.float64)
    beta_chunk = torch.sigmoid(torch.randn(B, H, C, dtype=torch.float64))
    T = _wy_inv_triangular(k_chunk, beta_chunk)
    A = beta_chunk.unsqueeze(-1) * torch.einsum("bhck,bhdk->bhcd", k_chunk, k_chunk)
    A = torch.tril(A, diagonal=-1)
    eye = torch.eye(C, dtype=torch.float64).expand_as(A)
    torch.testing.assert_close(T @ (eye + A), eye, atol=1e-9, rtol=1e-9)
```

- [ ] **Step 3: Run helper tests**

Run: `uv run pytest tests/test_gated_delta_chunkwise_torch.py -v -k wy_inv`
Expected: 2 passed.

- [ ] **Step 4: Commit**

```bash
git add gated_deltanet/torch/gated_delta_chunkwise_torch.py tests/test_gated_delta_chunkwise_torch.py
git commit -m "feat(torch): add WY triangular-inverse helper for chunked update"
```

---

### Task 8: Implement `gated_delta_chunkwise_torch`

**Files:**
- Modify: `gated_deltanet/torch/gated_delta_chunkwise_torch.py`

**Per-chunk math (let `C = _CHUNK_SIZE`, `S_prev` = state before chunk, `g_eff`
already computed via `_common.effective_gate`):**

1. Cumulative chunk-local log-decay:
   `g_cum_t = sum_{s<=t} g_eff_s` (inclusive),
   `g_cum_excl_t = g_cum_t - g_eff_t` (exclusive).
2. Decayed q and k for dense matmuls:
   `k_dec_s = exp(-g_cum_excl_s) * k_s`,
   `q_dec_t = exp( g_cum_t)      * q_t * scale`.
   Then `attn(t,s) = q_dec_t · k_dec_s = exp(g_eff_s + ... + g_eff_t) * (q_t · k_s)`.
3. Effective in-chunk values (subtract what `S_prev` already encodes for `k_s`):
   `pred_prev_s = exp(g_cum_excl_s) * (S_prev^T @ k_s)`,
   `u_s        = v_s - pred_prev_s`.
4. WY solve for the in-chunk write magnitudes:
   `w = T_wy @ (beta * u)`,    `T_wy = _wy_inv_triangular(k_c, beta_c)`.
5. Output:
   `o_t = exp(g_cum_t) * (q_t * scale) @ S_prev + sum_{s<=t} attn(t,s) * w_s`.
6. State update at chunk end (`L = C - 1`):
   `S_next = exp(g_cum_L) * S_prev + sum_s exp(g_cum_L - g_cum_excl_s) * k_s outer w_s`.

**Mask convention:** `mask = tril(diagonal=0)` — the write at step `s` IS visible
to the read at step `s` (because the recurrence updates `S_t` before reading `o_t`).

- [ ] **Step 1: Implement**

```python
def gated_delta_chunkwise_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    dt_bias: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    validate_unsupported(cu_seqlens, use_gate_in_kernel)
    q, k = maybe_l2norm_qk(q, k, enabled=use_qk_l2norm_in_kernel)

    B, T, H, K = q.shape
    V = v.shape[-1]
    scale = K ** -0.5
    dtype, device = q.dtype, q.device

    if initial_state is None:
        S = torch.zeros(B, H, K, V, dtype=dtype, device=device)
    else:
        S = initial_state.clone()

    g_eff = effective_gate(g, A_log, dt_bias)              # (B, T, H)

    # Move time to position -2 for cleaner einsums: (B, H, T, *)
    q_h = q.transpose(1, 2)
    k_h = k.transpose(1, 2)
    v_h = v.transpose(1, 2)
    g_h = g_eff.transpose(1, 2)
    b_h = beta.transpose(1, 2)

    out_h = torch.empty(B, H, T, V, dtype=dtype, device=device)

    for start in range(0, T, _CHUNK_SIZE):
        end = min(start + _CHUNK_SIZE, T)
        C = end - start

        q_c = q_h[:, :, start:end] * scale                 # (B, H, C, K)
        k_c = k_h[:, :, start:end]
        v_c = v_h[:, :, start:end]                         # (B, H, C, V)
        g_c = g_h[:, :, start:end]                         # (B, H, C)
        beta_c = b_h[:, :, start:end]

        g_cum = torch.cumsum(g_c, dim=-1)                  # (B, H, C) inclusive
        g_cum_excl = g_cum - g_c                           # exclusive

        k_dec = k_c * torch.exp(-g_cum_excl).unsqueeze(-1) # (B, H, C, K)
        q_dec = q_c * torch.exp(g_cum).unsqueeze(-1)       # (B, H, C, K)

        pred_prev = torch.exp(g_cum_excl).unsqueeze(-1) * torch.einsum(
            "bhkv,bhck->bhcv", S, k_c,
        )                                                  # (B, H, C, V)
        u = v_c - pred_prev

        T_wy = _wy_inv_triangular(k_c, beta_c)             # (B, H, C, C)
        w = torch.einsum("bhcd,bhdv->bhcv", T_wy, beta_c.unsqueeze(-1) * u)

        o_prev = torch.exp(g_cum).unsqueeze(-1) * torch.einsum(
            "bhck,bhkv->bhcv", q_c, S,
        )

        attn = torch.einsum("bhck,bhdk->bhcd", q_dec, k_dec)
        mask = torch.ones(C, C, dtype=torch.bool, device=device).tril(diagonal=0)
        attn = attn.masked_fill(~mask, 0.0)

        out_h[:, :, start:end] = o_prev + torch.einsum("bhcd,bhdv->bhcv", attn, w)

        cum_last = g_cum[:, :, -1]                         # (B, H)
        S = torch.exp(cum_last).unsqueeze(-1).unsqueeze(-1) * S
        decay_to_end = torch.exp(cum_last.unsqueeze(-1) - g_cum_excl).unsqueeze(-1)
        S = S + torch.einsum("bhck,bhcv->bhkv", k_c * decay_to_end, w)

    out = out_h.transpose(1, 2).contiguous()
    final = S if output_final_state else None
    return out, final
```

- [ ] **Step 2: Run all parity tests**

Run: `uv run pytest tests/test_gated_delta_chunkwise_torch.py -v`
Expected: all PASS (parametrized full-block, partial last chunk, initial-state,
unsupported flags, WY helper).

- [ ] **Step 3: Debug recipe if a parity case fails**

Print divergence per timestep:

```python
diff = (o_chk - o_ref).abs().reshape(o_chk.shape[0], o_chk.shape[1], -1).max(-1).values
print(diff)
```

Most likely causes (check in order):
1. Inclusive vs exclusive cumsum mix-up in `q_dec`/`k_dec` factors.
2. Mask diagonal off-by-one — must be `tril(diagonal=0)`.
3. `S_prev` decay applied twice (it should appear exactly once, in `o_prev` via
   the `exp(g_cum)` factor).
4. `effective_gate` sign — `g_eff` must be ≤ 0; if it's positive the state
   explodes. Verify `(g_eff <= 0).all()` on a sample input.

- [ ] **Step 4: Commit**

```bash
git add gated_deltanet/torch/gated_delta_chunkwise_torch.py
git commit -m "feat(torch): chunkwise forward via WY representation, parity-checked vs recurrent"
```

---

### Task 9: Autograd grad-check on the chunkwise path

**Files:**
- Modify: `tests/test_gated_delta_chunkwise_torch.py`

- [ ] **Step 1: Add gradcheck spanning a partial last chunk**

```python
@pytest.mark.parametrize("T", [3, 65])      # well below and just over one full chunk
def test_chunkwise_gradcheck(T):
    torch.manual_seed(0)
    B, H, K, V = 1, 1, 3, 3
    q = torch.randn(B, T, H, K, dtype=torch.float64, requires_grad=True)
    k = torch.randn(B, T, H, K, dtype=torch.float64, requires_grad=True)
    v = torch.randn(B, T, H, V, dtype=torch.float64, requires_grad=True)
    g = torch.randn(B, T, H, dtype=torch.float64, requires_grad=True)
    beta = torch.sigmoid(torch.randn(B, T, H, dtype=torch.float64)).detach().requires_grad_(True)
    A_log = (torch.randn(H, dtype=torch.float64) * 0.1).detach().requires_grad_(True)

    def f(q, k, v, g, beta, A_log):
        o, _ = gated_delta_chunkwise_torch(q, k, v, g, beta, A_log)
        return o

    assert torch.autograd.gradcheck(f, (q, k, v, g, beta, A_log), eps=1e-6, atol=1e-4)
```

- [ ] **Step 2: Run**

Run: `uv run pytest tests/test_gated_delta_chunkwise_torch.py::test_chunkwise_gradcheck -v`
Expected: PASS for both `T=3` and `T=65`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_gated_delta_chunkwise_torch.py
git commit -m "test(torch): autograd gradcheck for chunkwise forward"
```

---

### Task 10: CUDA fp32 smoke parity

**Files:**
- Modify: `tests/test_gated_delta_chunkwise_torch.py`

- [ ] **Step 1: Add CUDA smoke**

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_recurrent_chunkwise_parity_fp32():
    torch.manual_seed(0)
    B, T, H, K, V = 2, 130, 4, 16, 16
    dev = "cuda"
    q = torch.randn(B, T, H, K, device=dev)
    k = torch.randn(B, T, H, K, device=dev)
    v = torch.randn(B, T, H, V, device=dev)
    g = torch.randn(B, T, H, device=dev)
    beta = torch.sigmoid(torch.randn(B, T, H, device=dev))
    A_log = (torch.randn(H, device=dev) * 0.1)

    o_ref, _ = gated_delta_recurrent_torch(q, k, v, g, beta, A_log)
    o_chk, _ = gated_delta_chunkwise_torch(q, k, v, g, beta, A_log)
    torch.testing.assert_close(o_chk, o_ref, atol=1e-3, rtol=1e-3)
```

- [ ] **Step 2: Run on CUDA host**

Run: `uv run pytest tests/test_gated_delta_chunkwise_torch.py::test_cuda_recurrent_chunkwise_parity_fp32 -v`
Expected: PASS on RTX 3090; SKIP on CPU-only.

- [ ] **Step 3: Final full-suite green**

Run: `uv run pytest tests/ -v`
Expected: every test PASS or SKIP — no FAIL.

- [ ] **Step 4: Commit**

```bash
git add tests/test_gated_delta_chunkwise_torch.py
git commit -m "test(torch): cuda smoke parity for recurrent vs chunkwise"
```

---

## Acceptance Criteria

- `gated_delta_recurrent_torch` passes shape, decay-collapse, beta=0/1,
  initial-state passthrough, l2norm flag-changes-output, and unsupported-flag
  error tests.
- `gated_delta_chunkwise_torch` matches `gated_delta_recurrent_torch` to
  `1e-5` (fp64) for every `T ∈ {1, 7, 13, 16, 64, 65, 128, 130}`, with and
  without `initial_state`.
- `torch.autograd.gradcheck` succeeds on both APIs with fp64 inputs.
- `from gated_deltanet import gated_delta_recurrent_torch, gated_delta_chunkwise_torch`
  works at the package root.
- `cu_seqlens != None` and `use_gate_in_kernel=True` raise
  `NotImplementedError` from both functions.
- No Triton, no nn.Module, no manual backward.
