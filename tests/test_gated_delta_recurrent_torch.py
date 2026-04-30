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
