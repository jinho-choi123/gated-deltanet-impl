import torch
from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

from itertools import accumulate


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
) -> dict[str, torch.Tensor]:
    """Build B=1 packed inputs for variable-length sequences.

    Returns a dict with keys: q, k, v, g, beta, cu_seqlens, initial_state.
    Tensor shapes:
      q, k, v: (1, L, H, D) where L = sum(seq_lens)
      g, beta: (1, L, H)
      cu_seqlens: (N+1,) on `device`, dtype=torch.int64 (fla expects LongTensor)
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

    offsets = torch.tensor([0, *accumulate(seq_lens)], dtype=torch.int64)

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
