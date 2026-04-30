"""API for Gated DeltaNet Chunkwise implementation."""

import math

import torch


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
    raise NotImplementedError("Gated DeltaNet Recurrent is not implemented yet.")
