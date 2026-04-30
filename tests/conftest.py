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
