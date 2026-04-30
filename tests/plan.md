# Test Plan: `gated_delta_recurrent_torch` Equivalence Suite

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pytest suite at `tests/test_gated_delta_recurrent_torch.py` that
proves the user's `gated_delta_recurrent_torch` matches fla's
`fused_recurrent_gated_delta_rule` on every code path the simplified API
exercises.

**Architecture:** A small reference-wrapper helper (`tests/_fla_reference.py`)
maps the user's narrow signature onto fla's broad signature, freezing every fla
parameter that the user's simplification implies. Tests then call both
implementations on the same inputs and compare outputs and final states with
`torch.testing.assert_close`.

**Tech Stack:** Python 3.14, PyTorch 2.11+cu126, triton, `fla-core` (new dep),
pytest, pytest-xdist (already in `pyproject.toml`).

---

## Reference contract (single source of truth)

User's function under test:

```python
gated_delta_recurrent_torch(
    q, k, v, g, beta,            # q,k,v: (1, L, H, D); g, beta: (1, L, H)
    cu_seqlens,                  # (N+1,)
    initial_state,               # (N, H, D, D)
    use_qk_l2norm_in_kernel: bool = False,
) -> (output, final_state)       # output: (1, L, H, D); final_state: (N, H, D, D)
```

User's simplifications relative to fla (frozen for the entire test suite):
- `gk = None`, `gv = None` (no per-channel gates).
- `A_log = None`, `dt_bias = None` (state decay is exactly `exp(g)`).
- `use_gate_in_kernel = False` (no fused output gate; `g` is log-space decay).
- `use_exp2 = False` (use natural-base `exp`, matching `exp(g)` semantics).
- `transpose_state_layout = False` (state stored as `(K, V) == (D, D)` per slice).
- `scale = None` → fla defaults to `1 / sqrt(K) = 1 / sqrt(D)`. User's impl
  must match this scaling convention (the existing impl multiplies by `sqrt(D)`,
  which is wrong; tests will surface that defect — that is intended).
- `output_final_state = True` (user's wrapper always returns `final_state`).

Reference call (the only call shape any test uses):

```python
o_ref, s_ref = fused_recurrent_gated_delta_rule(
    q=q, k=k, v=v,
    g=g, gk=None, gv=None,
    beta=beta,
    scale=None,                  # fla -> 1/sqrt(D)
    initial_state=initial_state,
    output_final_state=True,
    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    use_gate_in_kernel=False,
    A_log=None, dt_bias=None,
    cu_seqlens=cu_seqlens,
    use_exp2=False,
    transpose_state_layout=False,
)
```

## Device & dtype policy

- fla's `fused_recurrent_gated_delta_rule` is a Triton kernel — **CUDA only**.
- All equivalence tests therefore live behind `pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")`.
- Equivalence tests use `dtype=torch.float32` on `cuda` (Triton kernels typically
  do not support fp64). Tolerance: `atol=1e-4, rtol=1e-4`.
- Inputs are constructed on CPU with a fixed `torch.Generator(device='cpu').manual_seed(...)`,
  then `.to('cuda')`'d. Both functions read the same CUDA tensor objects.

---

## File Structure

| Path | Status | Responsibility |
|---|---|---|
| `pyproject.toml` | modify | Add `fla-core` to `dependencies` (or `dependency-groups.dev`). |
| `tests/conftest.py` | create | Shared `_make_inputs(...)` factory + `_pack(seqs)` helper for `cu_seqlens`-style packing. |
| `tests/_fla_reference.py` | create | One function `fla_reference(q, k, v, g, beta, cu_seqlens, initial_state, use_qk_l2norm_in_kernel)` that calls fla with frozen kwargs. |
| `tests/test_gated_delta_recurrent_torch.py` | modify (currently empty) | All equivalence tests. |

---

### Task 1: Add `fla-core` dependency and verify import

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `fla-core` to dependencies**

In `pyproject.toml`, append to the `[project] dependencies` list:

```toml
"fla-core>=0.5.0",
```

Resulting `dependencies` block:

```toml
dependencies = [
    "fla-core>=0.5.0",
    "loguru>=0.7.3",
    "pytest>=9.0.3",
    "pytest-xdist>=3.8.0",
    "ruff>=0.15.12",
    "torch==2.11.0+cu126",
    "transformers>=5.7.0",
    "triton>=3.6.0",
]
```

- [ ] **Step 2: Sync the environment**

Run: `uv sync`
Expected: lockfile resolves, fla-core installs.

- [ ] **Step 3: Smoke-import the reference function**

Run:
```bash
/workspace/gated-deltanet-triton/.venv/bin/python -c "from fla.ops.gated_delta_rule.fused_recurrent import fused_recurrent_gated_delta_rule; print('ok')"
```
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add fla-core for reference equivalence tests"
```

---

### Task 2: Reference wrapper

**Files:**
- Create: `tests/_fla_reference.py`

- [ ] **Step 1: Write the wrapper**

```python
"""Thin adapter that maps the simplified user API onto fla's full signature."""

from __future__ import annotations

import torch
from fla.ops.gated_delta_rule.fused_recurrent import fused_recurrent_gated_delta_rule


def fla_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Call fla with the user's simplifications frozen.

    Frozen: gk=gv=None, A_log=dt_bias=None, scale=None (=> 1/sqrt(D)),
    use_gate_in_kernel=False, use_exp2=False, transpose_state_layout=False,
    output_final_state=True.
    """
    o, s = fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        gk=None,
        gv=None,
        beta=beta,
        scale=None,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=False,
        A_log=None,
        dt_bias=None,
        cu_seqlens=cu_seqlens,
        use_exp2=False,
        transpose_state_layout=False,
    )
    return o, s
```

- [ ] **Step 2: Smoke import**

Run:
```bash
/workspace/gated-deltanet-triton/.venv/bin/python -c "from tests._fla_reference import fla_reference; print('ok')"
```
Expected: `ok` (or fails import with a clear message — fix the path before
proceeding).

- [ ] **Step 3: Commit**

```bash
git add tests/_fla_reference.py
git commit -m "test: add fla reference wrapper for gated_delta_recurrent equivalence"
```

---

### Task 3: Shared input factories in `conftest.py`

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Write factories**

```python
"""Shared fixtures and input factories for gated-delta-rule tests."""

from __future__ import annotations

import torch

CUDA = torch.device("cuda")
DTYPE = torch.float32


def make_packed_inputs(
    seq_lens: list[int],
    H: int,
    D: int,
    *,
    use_initial_state: bool = True,
    seed: int = 0,
    device: torch.device = CUDA,
    dtype: torch.dtype = DTYPE,
) -> dict:
    """Build B=1 packed inputs for variable-length sequences.

    Returns a dict with keys: q, k, v, g, beta, cu_seqlens, initial_state.
    Tensor shapes:
      q, k, v: (1, L, H, D) where L = sum(seq_lens)
      g, beta: (1, L, H)
      cu_seqlens: (N+1,) on `device`, dtype=torch.int32
      initial_state: (N, H, D, D) (zeros if not `use_initial_state`)
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    L = sum(seq_lens)
    N = len(seq_lens)

    q = torch.randn(1, L, H, D, generator=gen, dtype=dtype)
    k = torch.randn(1, L, H, D, generator=gen, dtype=dtype)
    v = torch.randn(1, L, H, D, generator=gen, dtype=dtype)
    # g is log-decay; sample in (-1, 0] so decay = exp(g) is in (e^-1, 1].
    g = -torch.rand(1, L, H, generator=gen, dtype=dtype)
    beta = torch.sigmoid(torch.randn(1, L, H, generator=gen, dtype=dtype))

    offsets = torch.tensor([0, *list(_cumsum(seq_lens))], dtype=torch.int32)

    if use_initial_state:
        s0 = torch.randn(N, H, D, D, generator=gen, dtype=dtype) * 0.1
    else:
        s0 = torch.zeros(N, H, D, D, dtype=dtype)

    return {
        "q": q.to(device),
        "k": k.to(device),
        "v": v.to(device),
        "g": g.to(device),
        "beta": beta.to(device),
        "cu_seqlens": offsets.to(device),
        "initial_state": s0.to(device),
    }


def _cumsum(xs: list[int]) -> list[int]:
    out, total = [], 0
    for x in xs:
        total += x
        out.append(total)
    return out
```

- [ ] **Step 2: Smoke**

Run:
```bash
/workspace/gated-deltanet-triton/.venv/bin/python -c "
from tests.conftest import make_packed_inputs
import torch
if torch.cuda.is_available():
    d = make_packed_inputs([3, 5], H=2, D=4)
    print({k: tuple(v.shape) for k, v in d.items()})
else:
    print('no cuda — skipping smoke')
"
```
Expected (CUDA host): a dict of shapes with `q.shape == (1, 8, 2, 4)`,
`cu_seqlens.shape == (3,)`, `initial_state.shape == (2, 2, 4, 4)`.

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add shared input factory for packed gated-delta-rule tests"
```

---

### Task 4: Single-sequence equivalence (the canonical case)

**Files:**
- Modify: `tests/test_gated_delta_recurrent_torch.py`

- [ ] **Step 1: Write the test**

```python
import pytest
import torch

from gated_deltanet import gated_delta_recurrent_torch
from tests._fla_reference import fla_reference
from tests.conftest import make_packed_inputs


CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fla kernels need CUDA",
)


@CUDA_REQUIRED
def test_matches_fla_single_sequence():
    inputs = make_packed_inputs(seq_lens=[16], H=2, D=8, seed=0)
    o_ref, s_ref = fla_reference(**inputs)
    o_user, s_user = gated_delta_recurrent_torch(
        inputs["q"], inputs["k"], inputs["v"],
        inputs["g"], inputs["beta"],
        inputs["cu_seqlens"], inputs["initial_state"],
    )
    torch.testing.assert_close(o_user, o_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(s_user, s_ref, atol=1e-4, rtol=1e-4)
```

- [ ] **Step 2: Run; expect FAIL on broken impl**

Run: `/workspace/gated-deltanet-triton/.venv/bin/python -m pytest tests/test_gated_delta_recurrent_torch.py::test_matches_fla_single_sequence -v`
Expected: FAIL — the user's impl currently has 3 critical bugs (raises
`NotImplementedError`, scale inverted, state slice corruption). This is the
TDD red layer; do **not** fix the impl in this plan, only flag the failure
mode in the commit message.

- [ ] **Step 3: Commit**

```bash
git add tests/test_gated_delta_recurrent_torch.py
git commit -m "test(recurrent): equivalence vs fla on single-sequence input (RED)"
```

---

### Task 5: Multi-sequence packed equivalence

**Files:**
- Modify: `tests/test_gated_delta_recurrent_torch.py`

- [ ] **Step 1: Add the test**

```python
@CUDA_REQUIRED
@pytest.mark.parametrize("seq_lens", [[1, 1, 1], [3, 5, 7], [13, 4, 8, 11]])
def test_matches_fla_multi_sequence_packed(seq_lens):
    inputs = make_packed_inputs(seq_lens=seq_lens, H=2, D=8, seed=1)
    o_ref, s_ref = fla_reference(**inputs)
    o_user, s_user = gated_delta_recurrent_torch(
        inputs["q"], inputs["k"], inputs["v"],
        inputs["g"], inputs["beta"],
        inputs["cu_seqlens"], inputs["initial_state"],
    )
    torch.testing.assert_close(o_user, o_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(s_user, s_ref, atol=1e-4, rtol=1e-4)
```

- [ ] **Step 2: Run**

Run: `/workspace/gated-deltanet-triton/.venv/bin/python -m pytest tests/test_gated_delta_recurrent_torch.py -v -k multi_sequence`
Expected: 3 FAIL (RED) — the state-slice corruption bug means each sequence's
final_state contains the cross-product of all previous sequences' updates.
This test surfaces that defect explicitly.

- [ ] **Step 3: Commit**

```bash
git add tests/test_gated_delta_recurrent_torch.py
git commit -m "test(recurrent): equivalence on multi-sequence packed inputs (RED)"
```

---

### Task 6: Initial-state passthrough equivalence

**Files:**
- Modify: `tests/test_gated_delta_recurrent_torch.py`

- [ ] **Step 1: Add the test**

```python
@CUDA_REQUIRED
def test_matches_fla_with_random_initial_state():
    inputs = make_packed_inputs(
        seq_lens=[6, 9], H=3, D=4, seed=2, use_initial_state=True,
    )
    o_ref, s_ref = fla_reference(**inputs)
    o_user, s_user = gated_delta_recurrent_torch(
        inputs["q"], inputs["k"], inputs["v"],
        inputs["g"], inputs["beta"],
        inputs["cu_seqlens"], inputs["initial_state"],
    )
    torch.testing.assert_close(o_user, o_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(s_user, s_ref, atol=1e-4, rtol=1e-4)


@CUDA_REQUIRED
def test_matches_fla_with_zero_initial_state():
    inputs = make_packed_inputs(
        seq_lens=[7], H=2, D=4, seed=3, use_initial_state=False,
    )
    o_ref, s_ref = fla_reference(**inputs)
    o_user, s_user = gated_delta_recurrent_torch(
        inputs["q"], inputs["k"], inputs["v"],
        inputs["g"], inputs["beta"],
        inputs["cu_seqlens"], inputs["initial_state"],
    )
    torch.testing.assert_close(o_user, o_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(s_user, s_ref, atol=1e-4, rtol=1e-4)
```

- [ ] **Step 2: Run**

Run: `/workspace/gated-deltanet-triton/.venv/bin/python -m pytest tests/test_gated_delta_recurrent_torch.py -v -k initial_state`
Expected: 2 FAIL (RED).

- [ ] **Step 3: Commit**

```bash
git add tests/test_gated_delta_recurrent_torch.py
git commit -m "test(recurrent): equivalence with non-zero/zero initial_state (RED)"
```

---

### Task 7: L2-norm flag equivalence

**Files:**
- Modify: `tests/test_gated_delta_recurrent_torch.py`

- [ ] **Step 1: Add the test**

```python
@CUDA_REQUIRED
@pytest.mark.parametrize("flag", [False, True])
def test_matches_fla_use_qk_l2norm_in_kernel(flag):
    inputs = make_packed_inputs(seq_lens=[5, 11], H=2, D=8, seed=4)
    o_ref, s_ref = fla_reference(**inputs, use_qk_l2norm_in_kernel=flag)
    o_user, s_user = gated_delta_recurrent_torch(
        inputs["q"], inputs["k"], inputs["v"],
        inputs["g"], inputs["beta"],
        inputs["cu_seqlens"], inputs["initial_state"],
        use_qk_l2norm_in_kernel=flag,
    )
    torch.testing.assert_close(o_user, o_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(s_user, s_ref, atol=1e-4, rtol=1e-4)
```

- [ ] **Step 2: Run**

Run: `/workspace/gated-deltanet-triton/.venv/bin/python -m pytest tests/test_gated_delta_recurrent_torch.py -v -k l2norm`
Expected: 2 FAIL (RED).

- [ ] **Step 3: Commit**

```bash
git add tests/test_gated_delta_recurrent_torch.py
git commit -m "test(recurrent): equivalence with use_qk_l2norm_in_kernel on/off (RED)"
```

---

### Task 8: Boundary cases (T=1, single head, large dim)

**Files:**
- Modify: `tests/test_gated_delta_recurrent_torch.py`

- [ ] **Step 1: Add tests**

```python
@CUDA_REQUIRED
def test_matches_fla_single_token_per_sequence():
    # Each sequence is exactly 1 token long.
    inputs = make_packed_inputs(seq_lens=[1, 1, 1, 1], H=2, D=4, seed=5)
    o_ref, s_ref = fla_reference(**inputs)
    o_user, s_user = gated_delta_recurrent_torch(
        inputs["q"], inputs["k"], inputs["v"],
        inputs["g"], inputs["beta"],
        inputs["cu_seqlens"], inputs["initial_state"],
    )
    torch.testing.assert_close(o_user, o_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(s_user, s_ref, atol=1e-4, rtol=1e-4)


@CUDA_REQUIRED
def test_matches_fla_single_head():
    inputs = make_packed_inputs(seq_lens=[12], H=1, D=4, seed=6)
    o_ref, s_ref = fla_reference(**inputs)
    o_user, s_user = gated_delta_recurrent_torch(
        inputs["q"], inputs["k"], inputs["v"],
        inputs["g"], inputs["beta"],
        inputs["cu_seqlens"], inputs["initial_state"],
    )
    torch.testing.assert_close(o_user, o_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(s_user, s_ref, atol=1e-4, rtol=1e-4)


@CUDA_REQUIRED
def test_matches_fla_realistic_dims():
    # H=4, D=64 — closer to a real layer config.
    inputs = make_packed_inputs(seq_lens=[64, 64], H=4, D=64, seed=7)
    o_ref, s_ref = fla_reference(**inputs)
    o_user, s_user = gated_delta_recurrent_torch(
        inputs["q"], inputs["k"], inputs["v"],
        inputs["g"], inputs["beta"],
        inputs["cu_seqlens"], inputs["initial_state"],
    )
    torch.testing.assert_close(o_user, o_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(s_user, s_ref, atol=1e-4, rtol=1e-4)
```

- [ ] **Step 2: Run**

Run: `/workspace/gated-deltanet-triton/.venv/bin/python -m pytest tests/test_gated_delta_recurrent_torch.py -v -k "single_token or single_head or realistic_dims"`
Expected: 3 FAIL (RED).

- [ ] **Step 3: Commit**

```bash
git add tests/test_gated_delta_recurrent_torch.py
git commit -m "test(recurrent): boundary-case equivalence vs fla (RED)"
```

---

### Task 9: Final-state-only assertion when full output not needed

**Files:**
- Modify: `tests/test_gated_delta_recurrent_torch.py`

This task pins the per-sequence `final_state` semantics independently of the
output trajectory — so a regression that breaks state slicing but happens to
get `output` right (or vice versa) still gets caught.

- [ ] **Step 1: Add the test**

```python
@CUDA_REQUIRED
def test_final_state_per_sequence_independent():
    # Run sequence A alone vs. sequence A packed with sequence B.
    # final_state[A] must be IDENTICAL in both runs.
    inputs_pair = make_packed_inputs(seq_lens=[7, 5], H=2, D=4, seed=8)
    inputs_a_only = {
        "q": inputs_pair["q"][:, :7],
        "k": inputs_pair["k"][:, :7],
        "v": inputs_pair["v"][:, :7],
        "g": inputs_pair["g"][:, :7],
        "beta": inputs_pair["beta"][:, :7],
        "cu_seqlens": torch.tensor([0, 7], dtype=torch.int32, device="cuda"),
        "initial_state": inputs_pair["initial_state"][:1],
    }
    _, s_pair = gated_delta_recurrent_torch(
        inputs_pair["q"], inputs_pair["k"], inputs_pair["v"],
        inputs_pair["g"], inputs_pair["beta"],
        inputs_pair["cu_seqlens"], inputs_pair["initial_state"],
    )
    _, s_solo = gated_delta_recurrent_torch(
        inputs_a_only["q"], inputs_a_only["k"], inputs_a_only["v"],
        inputs_a_only["g"], inputs_a_only["beta"],
        inputs_a_only["cu_seqlens"], inputs_a_only["initial_state"],
    )
    torch.testing.assert_close(s_pair[0], s_solo[0], atol=1e-4, rtol=1e-4)
```

- [ ] **Step 2: Run**

Run: `/workspace/gated-deltanet-triton/.venv/bin/python -m pytest tests/test_gated_delta_recurrent_torch.py::test_final_state_per_sequence_independent -v`
Expected: FAIL (RED) — the current impl applies decay/writes to the full
`(N, H, D, D)` tensor, so `s_pair[0] != s_solo[0]`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_gated_delta_recurrent_torch.py
git commit -m "test(recurrent): final_state must be per-sequence independent (RED)"
```

---

### Task 10: Wire suite into pytest defaults & sanity-check non-CUDA hosts

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Confirm `[tool.pytest]` block has `testpaths = ["tests"]`**

Read `pyproject.toml` and verify `[tool.pytest]` has `testpaths = ["tests"]`.
If not present, add:

```toml
[tool.pytest]
testpaths = ["tests"]
python_files = ["test_*"]
python_functions = ["test_*"]
```

- [ ] **Step 2: Run full suite once**

Run: `/workspace/gated-deltanet-triton/.venv/bin/python -m pytest tests/ -v`
Expected on a CUDA host: every equivalence test FAILs (RED) — exactly the
defects the impl review flagged. Expected on a CPU-only host: every test
SKIPs with reason "fla kernels need CUDA".

- [ ] **Step 3: Commit (only if `pyproject.toml` changed)**

```bash
git add pyproject.toml
git commit -m "test: ensure pytest discovers tests/ for gated-delta-rule suite"
```

---

## Acceptance Criteria

- `tests/test_gated_delta_recurrent_torch.py` contains all tasks 4–9 above.
- Every equivalence test calls fla via `tests._fla_reference.fla_reference`
  with the frozen kwargs listed in the contract section.
- All tests are CUDA-gated and skip cleanly on non-CUDA hosts.
- Suite passes once the impl bugs (scale, state slicing, `NotImplementedError`)
  are fixed in a separate plan; no impl changes are made here.
- No test asserts on internal state mutation — only on the public return tuple
  `(output, final_state)` — so the test suite remains valid if the impl is
  rewritten.

## Out of scope (will not be tested here)

- Backward / gradcheck — fla's kernel and the user's loop diverge in autograd
  surface; covered by a separate plan once both go through `torch.autograd.Function`
  or a shared math contract.
- Performance benchmarking.
- Chunkwise variant (separate plan).
