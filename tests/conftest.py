"""pytest configuration for the gated-delta-rule equivalence suite.

Auto-skips every test in `tests/` when CUDA is unavailable. fla's Triton kernels
require a CUDA device, so equivalence tests cannot run on CPU-only hosts.
Individual tests therefore do NOT need a `@pytest.mark.skipif` decorator —
this hook adds it during collection.

Also forces `torch.device("cuda")` as the default device for every test, so
helper code (e.g. ``torch.eye``, ``torch.ones``) inside the impl can omit
``device=`` and still allocate on CUDA. This matches the expected runtime
environment and avoids spurious CPU/CUDA mismatch errors during development.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if torch.cuda.is_available():
        return
    skip_no_cuda = pytest.mark.skip(reason="fla kernels need CUDA")
    for item in items:
        item.add_marker(skip_no_cuda)


@pytest.fixture(autouse=True)
def _default_device_cuda() -> Iterator[None]:
    """Force CUDA as the default device for tensor factories during each test.

    The skip hook above guarantees CUDA is available whenever this fixture
    actually runs a test body, so the unguarded ``torch.device("cuda")`` is safe.
    """
    with torch.device("cuda"):
        yield
