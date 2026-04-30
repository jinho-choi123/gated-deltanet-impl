import pytest
import torch

from gated_deltanet import gated_delta_recurrent_torch
from tests.helper import fla_reference, make_packed_inputs


CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fla kernels need CUDA",
)


@CUDA_REQUIRED
def test_matches_fla_single_sequence():
    inputs = make_packed_inputs(seq_lens=[16], H=2, D=8, seed=0)
    o_ref, s_ref = fla_reference(**inputs)
    o_user, s_user = gated_delta_recurrent_torch(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        inputs["cu_seqlens"],
        inputs["initial_state"],
    )
    torch.testing.assert_close(o_user, o_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(s_user, s_ref, atol=1e-4, rtol=1e-4)


@CUDA_REQUIRED
@pytest.mark.parametrize("seq_lens", [[1, 1, 1], [3, 5, 7], [13, 4, 8, 11]])
def test_matches_fla_multi_sequence_packed(seq_lens):
    inputs = make_packed_inputs(seq_lens=seq_lens, H=2, D=8, seed=1)
    o_ref, s_ref = fla_reference(**inputs)
    o_user, s_user = gated_delta_recurrent_torch(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        inputs["cu_seqlens"],
        inputs["initial_state"],
    )
    torch.testing.assert_close(o_user, o_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(s_user, s_ref, atol=1e-4, rtol=1e-4)


@CUDA_REQUIRED
def test_matches_fla_with_random_initial_state():
    inputs = make_packed_inputs(
        seq_lens=[6, 9],
        H=3,
        D=4,
        seed=2,
        use_initial_state=True,
    )
    o_ref, s_ref = fla_reference(**inputs)
    o_user, s_user = gated_delta_recurrent_torch(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        inputs["cu_seqlens"],
        inputs["initial_state"],
    )
    torch.testing.assert_close(o_user, o_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(s_user, s_ref, atol=1e-4, rtol=1e-4)


@CUDA_REQUIRED
def test_matches_fla_with_zero_initial_state():
    inputs = make_packed_inputs(
        seq_lens=[7],
        H=2,
        D=4,
        seed=3,
        use_initial_state=False,
    )
    o_ref, s_ref = fla_reference(**inputs)
    o_user, s_user = gated_delta_recurrent_torch(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        inputs["cu_seqlens"],
        inputs["initial_state"],
    )
    torch.testing.assert_close(o_user, o_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(s_user, s_ref, atol=1e-4, rtol=1e-4)


@CUDA_REQUIRED
@pytest.mark.parametrize("flag", [False, True])
def test_matches_fla_use_qk_l2norm_in_kernel(flag):
    inputs = make_packed_inputs(seq_lens=[5, 11], H=2, D=8, seed=4)
    o_ref, s_ref = fla_reference(**inputs, use_qk_l2norm_in_kernel=flag)
    o_user, s_user = gated_delta_recurrent_torch(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        inputs["cu_seqlens"],
        inputs["initial_state"],
        use_qk_l2norm_in_kernel=flag,
    )
    torch.testing.assert_close(o_user, o_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(s_user, s_ref, atol=1e-4, rtol=1e-4)


@CUDA_REQUIRED
def test_matches_fla_single_token_per_sequence():
    # Each sequence is exactly 1 token long.
    inputs = make_packed_inputs(seq_lens=[1, 1, 1, 1], H=2, D=4, seed=5)
    o_ref, s_ref = fla_reference(**inputs)
    o_user, s_user = gated_delta_recurrent_torch(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        inputs["cu_seqlens"],
        inputs["initial_state"],
    )
    torch.testing.assert_close(o_user, o_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(s_user, s_ref, atol=1e-4, rtol=1e-4)


@CUDA_REQUIRED
def test_matches_fla_single_head():
    inputs = make_packed_inputs(seq_lens=[12], H=1, D=4, seed=6)
    o_ref, s_ref = fla_reference(**inputs)
    o_user, s_user = gated_delta_recurrent_torch(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        inputs["cu_seqlens"],
        inputs["initial_state"],
    )
    torch.testing.assert_close(o_user, o_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(s_user, s_ref, atol=1e-4, rtol=1e-4)


@CUDA_REQUIRED
def test_matches_fla_realistic_dims():
    # H=4, D=64 — closer to a real layer config.
    # Looser fp32 tolerance: at T=128 with D=64, fla's Triton kernel and the
    # naive Python loop accumulate reductions in different orders, producing
    # round-off drift up to ~6e-4 relative on a small fraction of elements.
    inputs = make_packed_inputs(seq_lens=[64, 64], H=4, D=64, seed=7)
    o_ref, s_ref = fla_reference(**inputs)
    o_user, s_user = gated_delta_recurrent_torch(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        inputs["cu_seqlens"],
        inputs["initial_state"],
    )
    torch.testing.assert_close(o_user, o_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(s_user, s_ref, atol=1e-3, rtol=1e-3)


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
        "cu_seqlens": torch.tensor([0, 7], dtype=torch.int64, device="cuda"),
        "initial_state": inputs_pair["initial_state"][:1],
    }
    _, s_pair = gated_delta_recurrent_torch(
        inputs_pair["q"],
        inputs_pair["k"],
        inputs_pair["v"],
        inputs_pair["g"],
        inputs_pair["beta"],
        inputs_pair["cu_seqlens"],
        inputs_pair["initial_state"],
    )
    _, s_solo = gated_delta_recurrent_torch(
        inputs_a_only["q"],
        inputs_a_only["k"],
        inputs_a_only["v"],
        inputs_a_only["g"],
        inputs_a_only["beta"],
        inputs_a_only["cu_seqlens"],
        inputs_a_only["initial_state"],
    )
    torch.testing.assert_close(s_pair[0], s_solo[0], atol=1e-4, rtol=1e-4)
