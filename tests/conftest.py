"""Shared pytest configuration."""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``@pytest.mark.gpu`` tests unless ``DIFFUSERS_MM_RUN_GPU_TESTS=1``.

    GPU tests load real diffusers pipelines and apply
    ``torch.cuda.set_per_process_memory_fraction`` caps. They require a
    CUDA device, downloaded model weights, and significant wall-time, so
    they're opt-in. The default ``make test`` run leaves them skipped.
    """
    if os.environ.get("DIFFUSERS_MM_RUN_GPU_TESTS") == "1":
        return
    skip_gpu = pytest.mark.skip(reason="set DIFFUSERS_MM_RUN_GPU_TESTS=1 to enable GPU tests")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
