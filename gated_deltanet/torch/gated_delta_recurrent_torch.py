"""API for Gated DeltaNet Recurrent implementation."""

from typing import final

import torch


def gated_delta_recurrent_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gated DeltaNet Recurrent implementation in PyTorch.

    Always use `B=1`, which assumes packed variable-length sequences.
    `L` is the total sequence length after packing, and `N` is the number of sequences in the batch.
    `H` is the number of heads, and `D` is the head dimension.

    Args:
        q: Query tensor of shape (B=1, L, H, D).

        k: Key tensor of shape (B=1, L, H, D).

        v: Value tensor of shape (B=1, L, H, D).

        g: Gate tensor of shape (B=1, L, H).

        beta: Beta tensor of shape (B=1, L, H).

        cu_seqlens: Cumulative sequence lengths of shape (N+1). \
            This is used to identify the start and end positions of each sequence in the packed input.

        initial_state: Initial state tensor of shape (N, H, D, D).

        use_qk_l2norm_in_kernel: Whether to apply L2 normalization to q and k before computing attention scores in the kernel.

    Returns:
        output: Output tensor of shape (B=1, L, H, D).

        final_state: Final state tensor of shape (N, H, D, D). This is the state after processing the entire sequence for each batch item.
    """

    # Get `B`, `L`, `H`, `D`, `N`
    B, L, H, D = q.shape
    N = cu_seqlens.shape[0] - 1
    assert B == 1, "Only batch size of 1 is supported for packed variable-length sequences."
    assert k.shape == (B, L, H, D), "Key tensor shape must match query tensor shape."
    assert v.shape == (B, L, H, D), "Value tensor shape must match query tensor shape."
    assert g.shape == (B, L, H), "Gate tensor shape must match (B, L, H)."
    assert beta.shape == (B, L, H), "Beta tensor shape must match (B, L, H)."
    assert initial_state.shape == (N, H, D, D), "Initial state tensor shape must match (N, H, D, D)."

    # Reserve output tensor and final state tensor
    output = torch.empty_like(v)
    final_state = torch.clone(initial_state)

    grid = H * N

    # For loop the grids and call the fused_recurrent_gated_delta_rule_fwd_kernel_torch
    for i in range(grid):
        pid = i
        seq_idx = pid // H
        head_idx = pid % H
        # iteration loop에서 처리할 시퀀스의 시작과 끝 인덱스를 구함
        bos_idx = cu_seqlens[seq_idx]
        eos_idx = cu_seqlens[seq_idx + 1]

        # iteration loop에서 차리할 시퀀스 길이
        seq_len = eos_idx - bos_idx

        # iteration loop에서 처리할 q, k, v, g, beta 시퀀스 슬라이스
        q_seq = q[0, bos_idx:eos_idx, head_idx, :]  # shape (seq_len,D)
        k_seq = k[0, bos_idx:eos_idx, head_idx, :]  # shape (seq_len, D)
        v_seq = v[0, bos_idx:eos_idx, head_idx, :]  # shape (seq_len, D)
        out_seq = output[0, bos_idx:eos_idx, head_idx, :]  # shape (seq_len, D)
        g_seq = g[0, bos_idx:eos_idx, head_idx]  # shape (seq_len,)
        beta_seq = beta[0, bos_idx:eos_idx, head_idx]  # shape (seq_len,)
        final_state_seq = final_state[seq_idx, head_idx]  # shape (D, D)

        # Recurrent loop
        # Processing token by token in the sequence, updating the final state.
        for t in range(seq_len):
            q_t = torch.clone(q_seq[t, :])  # shape (D,)
            k_t = torch.clone(k_seq[t, :])  # shape (D,)
            v_t = torch.clone(v_seq[t, :])  # shape (D,)
            if use_qk_l2norm_in_kernel:
                # L2 normalization of q and k
                q_t = torch.nn.functional.normalize(q_t, p=2, dim=-1)
                k_t = torch.nn.functional.normalize(k_t, p=2, dim=-1)

            # q에 scale 곱 (1/√D) — output magnitude 안정화
            q_t = q_t * (D**-0.5)

            # beta loading
            beta_t = beta_seq[t]  # shape (1,)

            #### Main Recurrent Update ####

            # 1. State decay (gating) 적용 — Sₜ ← αₜ·Sₜ₋₁
            alpha_t = torch.exp(g_seq[t])  # shape (1,)
            final_state_seq *= alpha_t

            # 2. Delta rule update
            v_new = beta_t * (v_t - k_t @ final_state_seq)  # shape (D,)
            final_state_seq += torch.outer(k_t, v_new)  # shape (D, D)

            # 3. Output 계산
            out_seq[t, :] = q_t @ final_state_seq  # shape (D,)

    return output, final_state
