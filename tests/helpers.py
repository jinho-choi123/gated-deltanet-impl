"""Shared test helpers: input factory, fla reference, equivalence assertion.

Single source of truth for everything the test files need beyond pytest itself.
Add new helpers here rather than re-deriving them in test files. When adding a
sibling kernel (e.g. chunkwise), reuse `make_packed_inputs` and
`assert_matches_fla` directly — only the impl callable changes.
"""

from __future__ import annotations

from itertools import accumulate
from collections.abc import Callable

import torch
from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

__all__ = [
    "CUDA",
    "DTYPE",
    "TIGHT_TOLERANCE",
    "LOOSE_TOLERANCE",
    "GatedDeltaImpl",
    "make_packed_inputs",
    "fla_reference",
    "assert_matches_fla",
]

CUDA = torch.device("cuda")
DTYPE = torch.float32

# Default fp32 equivalence tolerance for small dims (T <= 17, D <= 8).
TIGHT_TOLERANCE: dict[str, float] = {"atol": 1e-4, "rtol": 1e-4}
# Looser tolerance for realistic dims (T >= 64, D >= 64) where fla's Triton
# kernel and a Python loop accumulate reductions in different orders, producing
# round-off drift up to ~6e-4 relative on a small fraction of elements.
LOOSE_TOLERANCE: dict[str, float] = {"atol": 1e-3, "rtol": 1e-3}

# A gated-delta-rule kernel implementation. Any callable matching this protocol
# can be slotted into `assert_matches_fla` without further wiring.
GatedDeltaImpl = Callable[..., tuple[torch.Tensor, torch.Tensor]]


# --- Input factory ----------------------------------------------------------


def make_packed_inputs(
    seq_lens: list[int],
    H: int,
    D: int,
    *,
    use_initial_state: bool = True,
    seed: int = 0,
    device: torch.device = CUDA,
    dtype: torch.dtype = DTYPE,
) -> dict[str, torch.Tensor]:
    """Build B=1 packed inputs for variable-length sequences.

    Returns a dict with keys: q, k, v, g, beta, cu_seqlens, initial_state.

    Tensor shapes:
        q, k, v:        (1, L, H, D)   where L = sum(seq_lens)
        g, beta:        (1, L, H)
        cu_seqlens:     (N+1,)         dtype=torch.int64 (fla expects LongTensor)
        initial_state:  (N, H, D, D)   zeros if `use_initial_state` is False
    """
    # Build inputs on CPU so the seeded CPU generator matches the factory's
    # device, regardless of any test-time default device override (e.g. the
    # autouse CUDA-default fixture in conftest). Final tensors are moved to
    # ``device`` below.
    gen = torch.Generator(device="cpu").manual_seed(seed)
    L = sum(seq_lens)
    N = len(seq_lens)

    q = torch.randn(1, L, H, D, generator=gen, dtype=dtype, device="cpu")
    k = torch.randn(1, L, H, D, generator=gen, dtype=dtype, device="cpu")
    v = torch.randn(1, L, H, D, generator=gen, dtype=dtype, device="cpu")
    # g is log-decay; sample in (-1, 0] so decay = exp(g) is in (e^-1, 1].
    g = -torch.rand(1, L, H, generator=gen, dtype=dtype, device="cpu")
    beta = torch.sigmoid(torch.randn(1, L, H, generator=gen, dtype=dtype, device="cpu"))
    cu_seqlens = torch.tensor([0, *accumulate(seq_lens)], dtype=torch.int64, device="cpu")

    if use_initial_state:
        initial_state = torch.randn(N, H, D, D, generator=gen, dtype=dtype, device="cpu") * 0.1
    else:
        initial_state = torch.zeros(N, H, D, D, dtype=dtype, device="cpu")

    return {
        "q": q.to(device),
        "k": k.to(device),
        "v": v.to(device),
        "g": g.to(device),
        "beta": beta.to(device),
        "cu_seqlens": cu_seqlens.to(device),
        "initial_state": initial_state.to(device),
    }


# --- fla reference ----------------------------------------------------------


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

    Frozen kwargs (representing features the simplified API does not expose):
        gk = gv = None             # no per-channel gates
        A_log = dt_bias = None     # state decay is exactly exp(g)
        scale = None               # fla -> 1 / sqrt(D)
        use_gate_in_kernel = False # no fused output gate
        use_exp2 = False           # natural-base exp
        transpose_state_layout = False
        output_final_state = True  # always return final_state
    """
    return fused_recurrent_gated_delta_rule(
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


# --- Equivalence assertion --------------------------------------------------


def assert_matches_fla(
    impl: GatedDeltaImpl,
    inputs: dict[str, torch.Tensor],
    *,
    tolerance: dict[str, float] = TIGHT_TOLERANCE,
    use_qk_l2norm_in_kernel: bool = False,
) -> None:
    """Run `impl` and `fla_reference` on the same inputs; assert outputs match.

    Args:
        impl: A callable with signature
            ``(q, k, v, g, beta, cu_seqlens, initial_state, use_qk_l2norm_in_kernel) -> (output, final_state)``.
        inputs: Dict produced by :func:`make_packed_inputs`.
        tolerance: ``atol``/``rtol`` mapping; pass ``LOOSE_TOLERANCE`` for fp32
            stress configurations where reduction-order drift exceeds ``1e-4``.
        use_qk_l2norm_in_kernel: Forwarded identically to both `impl` and fla.
    """
    o_ref, s_ref = fla_reference(
        **inputs,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    o_user, s_user = impl(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        inputs["cu_seqlens"],
        inputs["initial_state"],
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    torch.testing.assert_close(o_user, o_ref, **tolerance)
    torch.testing.assert_close(s_user, s_ref, **tolerance)
