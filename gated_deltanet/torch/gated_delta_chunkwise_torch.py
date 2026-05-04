"""API for Gated DeltaNet Chunkwise implementation."""

import math

import torch


def gated_delta_chunkwise_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Gated DeltaNet Chunkwise implementation in PyTorch.

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

    CHUNK_SIZE = 32  # This is a placeholder value. The optimal chunk size may depend on the hardware and sequence lengths.

    # Reserve output tensor and final state tensor
    output = torch.empty_like(v)
    final_state = torch.clone(initial_state)

    # Cache the original output dtype so the fp64 chunk computation can be cast back
    # to fp32 (or whatever the input dtype is) at the very end of each chunk. The
    # fp64 promotion below is what makes chunk_size=64 with D=64 numerically stable
    # for the (I + A) inverse — fp32 alone hits 7-digit precision limits when A entries
    # near the diagonal reach magnitude ~β·sqrt(D)·γ_i/γ_j (cond ≫ 1e7 for chunk=64).
    out_dtype = q.dtype

    # Split the input into chunks and process each chunk sequentially. The state is updated after each chunk.
    for seq_idx in range(N):
        for head_idx in range(H):
            seq_bos_idx = cu_seqlens[seq_idx]
            seq_eos_idx = cu_seqlens[seq_idx + 1]

            seq_len = seq_eos_idx - seq_bos_idx
            num_chunks = math.ceil(seq_len / CHUNK_SIZE)

            for chunk_idx in range(num_chunks):
                chunk_bos_idx = seq_bos_idx + chunk_idx * CHUNK_SIZE
                chunk_eos_idx = min(seq_bos_idx + (chunk_idx + 1) * CHUNK_SIZE, seq_eos_idx)
                chunk_len = int((chunk_eos_idx - chunk_bos_idx).item())

                # Slice once in the input dtype, then promote to fp64 for stable matmul/inv.
                q_chunk = q[0, chunk_bos_idx:chunk_eos_idx, head_idx, :].double()
                k_chunk = k[0, chunk_bos_idx:chunk_eos_idx, head_idx, :].double()
                v_chunk = v[0, chunk_bos_idx:chunk_eos_idx, head_idx, :].double()
                g_chunk = g[0, chunk_bos_idx:chunk_eos_idx, head_idx].double()
                beta_chunk = beta[0, chunk_bos_idx:chunk_eos_idx, head_idx].double()

                # State carries between chunks in the original dtype; promote on read,
                # demote on write back.
                out_chunk = output[0, chunk_bos_idx:chunk_eos_idx, head_idx, :]  # fp32 view (write target)
                final_state_chunk = final_state[seq_idx, head_idx]  # fp32 view (read + write target)
                S_prev = final_state_chunk.double()

                if use_qk_l2norm_in_kernel:
                    # L2 normalization of q and k
                    q_chunk = torch.nn.functional.normalize(q_chunk, p=2, dim=-1)
                    k_chunk = torch.nn.functional.normalize(k_chunk, p=2, dim=-1)

                # scale q_chunk
                q_chunk = q_chunk * (D**-0.5)

                # Get degatified gamma_chunk
                cum_g = torch.cumsum(g_chunk, dim=0)
                gamma_chunk = torch.exp(cum_g)  # shape (chunk_len,)
                Gamma_chunk = torch.tril(torch.exp(cum_g[:, None] - cum_g[None, :]))  # shape (chunk_len, chunk_len)
                kkT_chunk = k_chunk @ k_chunk.transpose(0, 1)  # shape (chunk_len, chunk_len)
                A_chunk = torch.tril(-torch.diag(beta_chunk) @ (Gamma_chunk * kkT_chunk), -1)
                I_sub_A_chunk = torch.eye(chunk_len, dtype=q_chunk.dtype, device=q_chunk.device) - A_chunk

                W_chunk = torch.linalg.solve(I_sub_A_chunk, torch.diag(beta_chunk * gamma_chunk)) @ k_chunk
                U_chunk = torch.linalg.solve(I_sub_A_chunk, torch.diag(beta_chunk) @ v_chunk)

                q_decay_to_start_chunk = torch.diag(gamma_chunk) @ q_chunk
                k_decay_to_end_chunk = gamma_chunk[-1] * torch.diag(1 / gamma_chunk) @ k_chunk

                out_fp64 = q_decay_to_start_chunk @ S_prev + ((q_chunk @ k_chunk.transpose(0, 1)) * Gamma_chunk) @ (U_chunk - W_chunk @ S_prev)
                state_fp64 = gamma_chunk[-1] * S_prev + k_decay_to_end_chunk.transpose(0, 1) @ (U_chunk - W_chunk @ S_prev)

                out_chunk.copy_(out_fp64.to(out_dtype))
                final_state_chunk.copy_(state_fp64.to(out_dtype))

    return output, final_state
