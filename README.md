# gated-deltanet-impl

Reference and Triton implementations of the [Gated DeltaNet](https://arxiv.org/abs/2412.06464) recurrent kernel.

The PyTorch reference is validated for numerical equivalence against
[`flash-linear-attention`](https://github.com/fla-org/flash-linear-attention)
(`fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule`).

## Status

| Component | Path | State |
|---|---|---|
| Recurrent forward (PyTorch reference) | `gated_deltanet/torch/gated_delta_recurrent_torch.py` | Implemented, 12 equivalence tests passing |
| Chunkwise forward (PyTorch reference) | `gated_deltanet/torch/gated_delta_chunkwise_torch.py` | Stub |
| Triton recurrent forward | — | Planned |
| Triton chunkwise forward / backward | — | Planned |

## Algorithm

For each `(sequence, head)` slice independently, with `S_0 = initial_state`,
`scale = D ** -0.5`:

```
S_t = exp(g_t) * S_{t-1}                                # gated decay
v_new_t = beta_t * (v_t - k_t @ S_t)                    # delta-rule write
S_t = S_t + outer(k_t, v_new_t)
o_t = (q_t * scale) @ S_t                               # scaled read
```

Simplifications relative to the full fla kernel surface:

- No per-channel gates (`gk = gv = None`)
- State decay is exactly `exp(g)` (no `A_log`, no `dt_bias`, no `softplus`)
- No fused output gate (`use_gate_in_kernel = False`)
- Exponential base is `e`, not `2` (`use_exp2 = False`)
- State layout is `(K, V) == (D, D)` (no transpose)
- `K == V == D` (single head dimension)

## Public API

```python
from gated_deltanet import gated_delta_recurrent_torch

output, final_state = gated_delta_recurrent_torch(
    q,                # (1, L, H, D)
    k,                # (1, L, H, D)
    v,                # (1, L, H, D)
    g,                # (1, L, H), log-decay (sample in (-1, 0])
    beta,             # (1, L, H), in (0, 1)
    cu_seqlens,       # (N+1,) int64, packed-variable-length offsets
    initial_state,    # (N, H, D, D)
    use_qk_l2norm_in_kernel=False,
)
# output:      (1, L, H, D)
# final_state: (N, H, D, D), per-sequence-per-head
```

`B=1` packed-variable-length layout only. `N` sequences are concatenated along
the time axis; `cu_seqlens[i]` is the start offset of sequence `i`.

## Install

Requires Python 3.14, CUDA 12.6+, [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Test

The full equivalence suite calls fla on CUDA; CPU-only hosts skip every test.

```bash
uv run pytest
```

Expected on a CUDA host: `12 passed`. Tests live in
`tests/test_gated_delta_recurrent_torch.py` and cover:

- Single-sequence equivalence
- Multi-sequence packed equivalence (3 seq-length parametrizations)
- Random and zero initial state
- `use_qk_l2norm_in_kernel` on and off
- Boundary cases: T=1 per sequence, H=1, realistic dims (H=4, D=64)
- Per-sequence final-state independence (solo run vs packed run produces
  identical `final_state[A]`)

Tolerance is `atol=rtol=1e-4` for small dims; `1e-3` for the realistic-dims
test where fp32 reduction-order drift between Triton and Python loops produces
~6e-4 relative round-off on a small fraction of elements.

## Project structure

```
gated_deltanet/
├── __init__.py                          # re-exports gated_delta_recurrent_torch
└── torch/
    ├── __init__.py
    ├── gated_delta_recurrent_torch.py   # naive O(L) reference
    ├── gated_delta_chunkwise_torch.py   # stub
    └── plan.md                          # implementation plan (recurrent+chunkwise)
tests/
├── helper.py                            # make_packed_inputs, fla_reference
├── test_gated_delta_recurrent_torch.py  # 12 equivalence tests
├── conftest.py
└── plan.md                              # test-suite plan
```

## References

- Yang et al., *Gated Delta Networks: Improving Mamba2 with Delta Rule*, 2024 — [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)
- `flash-linear-attention` — https://github.com/fla-org/flash-linear-attention
