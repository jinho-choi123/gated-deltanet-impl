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
