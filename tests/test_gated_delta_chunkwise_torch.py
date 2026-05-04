"""Equivalence tests for `gated_delta_chunkwise_torch` against fla.

Mirror of ``test_gated_delta_recurrent_torch`` with ``IMPL`` swapped to the
chunkwise kernel. Both impls are equivalence-tested against the same fla
reference, so a shared signature lets us reuse every helper unchanged.

Chunkwise involves an extra ``(I - A)^{-1}`` solve and two more matmul stages
than the token-by-token recurrence, so fp32 reduction-order drift is larger:
all tests use ``LOOSE_TOLERANCE`` to avoid spurious failures from non-bug
round-off. Tests are auto-skipped on non-CUDA hosts via
``conftest.pytest_collection_modifyitems``.
"""

from __future__ import annotations

import pytest
import torch

from gated_deltanet import gated_delta_chunkwise_torch
from tests.helpers import (
    LOOSE_TOLERANCE,
    GatedDeltaImpl,
    assert_matches_fla,
    make_packed_inputs,
)

IMPL: GatedDeltaImpl = gated_delta_chunkwise_torch


def test_single_sequence() -> None:
    inputs = make_packed_inputs(seq_lens=[16], H=2, D=8, seed=0)
    assert_matches_fla(IMPL, inputs, tolerance=LOOSE_TOLERANCE)


@pytest.mark.parametrize("seq_lens", [[1, 1, 1], [3, 5, 7], [13, 4, 8, 11]])
def test_multi_sequence_packed(seq_lens: list[int]) -> None:
    inputs = make_packed_inputs(seq_lens=seq_lens, H=2, D=8, seed=1)
    assert_matches_fla(IMPL, inputs, tolerance=LOOSE_TOLERANCE)


def test_random_initial_state() -> None:
    inputs = make_packed_inputs(seq_lens=[6, 9], H=3, D=4, seed=2, use_initial_state=True)
    assert_matches_fla(IMPL, inputs, tolerance=LOOSE_TOLERANCE)


def test_zero_initial_state() -> None:
    inputs = make_packed_inputs(seq_lens=[7], H=2, D=4, seed=3, use_initial_state=False)
    assert_matches_fla(IMPL, inputs, tolerance=LOOSE_TOLERANCE)


@pytest.mark.parametrize("flag", [False, True])
def test_use_qk_l2norm_in_kernel(flag: bool) -> None:
    inputs = make_packed_inputs(seq_lens=[5, 11], H=2, D=8, seed=4)
    assert_matches_fla(IMPL, inputs, tolerance=LOOSE_TOLERANCE, use_qk_l2norm_in_kernel=flag)


def test_single_token_per_sequence() -> None:
    inputs = make_packed_inputs(seq_lens=[1, 1, 1, 1], H=2, D=4, seed=5)
    assert_matches_fla(IMPL, inputs, tolerance=LOOSE_TOLERANCE)


def test_single_head() -> None:
    inputs = make_packed_inputs(seq_lens=[12], H=1, D=4, seed=6)
    assert_matches_fla(IMPL, inputs, tolerance=LOOSE_TOLERANCE)


def test_realistic_dims() -> None:
    """H=4, D=64 — closer to a real layer config; LOOSE_TOLERANCE for fp32 drift."""
    inputs = make_packed_inputs(seq_lens=[64, 64], H=4, D=64, seed=7)
    assert_matches_fla(IMPL, inputs, tolerance=LOOSE_TOLERANCE)


def test_multi_chunk_sequence() -> None:
    """Sequence longer than CHUNK_SIZE (64) exercises inter-chunk state propagation.

    Single-chunk paths can pass even when ``final_state.copy_`` write-back is
    broken, because the next chunk never reads the updated state. Use a length
    that forces ≥ 2 chunks to catch that class of bug.
    """
    inputs = make_packed_inputs(seq_lens=[150], H=2, D=8, seed=9)
    assert_matches_fla(IMPL, inputs, tolerance=LOOSE_TOLERANCE)


def test_partial_last_chunk() -> None:
    """Sequence length not a multiple of CHUNK_SIZE forces a partial trailing chunk.

    Catches dimensioning bugs where intermediate matrices are sized to
    ``CHUNK_SIZE`` instead of the actual ``chunk_len``.
    """
    inputs = make_packed_inputs(seq_lens=[70], H=2, D=8, seed=10)
    assert_matches_fla(IMPL, inputs, tolerance=LOOSE_TOLERANCE)


def test_final_state_per_sequence_independent() -> None:
    """Solo run vs. packed run must produce identical ``final_state[A]``.

    Catches state-slice corruption directly, regardless of fla's behavior, by
    comparing the impl against itself on overlapping inputs.
    """
    inputs_pair = make_packed_inputs(seq_lens=[7, 5], H=2, D=4, seed=8)
    inputs_solo = {
        "q": inputs_pair["q"][:, :7],
        "k": inputs_pair["k"][:, :7],
        "v": inputs_pair["v"][:, :7],
        "g": inputs_pair["g"][:, :7],
        "beta": inputs_pair["beta"][:, :7],
        "cu_seqlens": torch.tensor([0, 7], dtype=torch.int64, device=inputs_pair["cu_seqlens"].device),
        "initial_state": inputs_pair["initial_state"][:1],
    }

    def _final_state(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        _, final_state = IMPL(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["g"],
            inputs["beta"],
            inputs["cu_seqlens"],
            inputs["initial_state"],
        )
        return final_state

    s_pair = _final_state(inputs_pair)
    s_solo = _final_state(inputs_solo)
    torch.testing.assert_close(s_pair[0], s_solo[0], **LOOSE_TOLERANCE)
