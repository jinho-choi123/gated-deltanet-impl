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
