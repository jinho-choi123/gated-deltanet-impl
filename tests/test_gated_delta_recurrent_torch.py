"""Equivalence tests for `gated_delta_recurrent_torch` against fla.

Each test builds packed inputs via :func:`make_packed_inputs` and asserts
both ``output`` and ``final_state`` match fla's `fused_recurrent_gated_delta_rule`
within the configured tolerance. Tests are auto-skipped on non-CUDA hosts via
``conftest.pytest_collection_modifyitems``.

To validate a different implementation (e.g. chunkwise) against fla, copy this
file and change `IMPL`. The helper signature is shared.
"""

from __future__ import annotations

import pytest
import torch

from gated_deltanet import gated_delta_recurrent_torch
from tests.helpers import (
    LOOSE_TOLERANCE,
    TIGHT_TOLERANCE,
    GatedDeltaImpl,
    assert_matches_fla,
    make_packed_inputs,
)

IMPL: GatedDeltaImpl = gated_delta_recurrent_torch


def test_single_sequence() -> None:
    inputs = make_packed_inputs(seq_lens=[16], H=2, D=8, seed=0)
    assert_matches_fla(IMPL, inputs)


@pytest.mark.parametrize("seq_lens", [[1, 1, 1], [3, 5, 7], [13, 4, 8, 11]])
def test_multi_sequence_packed(seq_lens: list[int]) -> None:
    inputs = make_packed_inputs(seq_lens=seq_lens, H=2, D=8, seed=1)
    assert_matches_fla(IMPL, inputs)


def test_random_initial_state() -> None:
    inputs = make_packed_inputs(seq_lens=[6, 9], H=3, D=4, seed=2, use_initial_state=True)
    assert_matches_fla(IMPL, inputs)


def test_zero_initial_state() -> None:
    inputs = make_packed_inputs(seq_lens=[7], H=2, D=4, seed=3, use_initial_state=False)
    assert_matches_fla(IMPL, inputs)


@pytest.mark.parametrize("flag", [False, True])
def test_use_qk_l2norm_in_kernel(flag: bool) -> None:
    inputs = make_packed_inputs(seq_lens=[5, 11], H=2, D=8, seed=4)
    assert_matches_fla(IMPL, inputs, use_qk_l2norm_in_kernel=flag)


def test_single_token_per_sequence() -> None:
    inputs = make_packed_inputs(seq_lens=[1, 1, 1, 1], H=2, D=4, seed=5)
    assert_matches_fla(IMPL, inputs)


def test_single_head() -> None:
    inputs = make_packed_inputs(seq_lens=[12], H=1, D=4, seed=6)
    assert_matches_fla(IMPL, inputs)


def test_realistic_dims() -> None:
    """H=4, D=64 — closer to a real layer config; uses LOOSE_TOLERANCE for fp32 drift."""
    inputs = make_packed_inputs(seq_lens=[64, 64], H=4, D=64, seed=7)
    assert_matches_fla(IMPL, inputs, tolerance=LOOSE_TOLERANCE)


def test_final_state_per_sequence_independent() -> None:
    """Solo run vs. packed run must produce identical `final_state[A]`.

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
        "cu_seqlens": torch.tensor([0, 7], dtype=torch.int64, device="cuda"),
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
    torch.testing.assert_close(s_pair[0], s_solo[0], **TIGHT_TOLERANCE)
