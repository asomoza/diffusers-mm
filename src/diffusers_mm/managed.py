"""The ``managed()`` wrapper — drop-in replacement for diffusers' built-in offload methods."""

from __future__ import annotations

import logging
from functools import wraps
from typing import Any

import torch

from diffusers_mm.manager import ModelManager


logger = logging.getLogger(__name__)


def managed(
    pipe: Any,
    *,
    mm: ModelManager | None = None,
    strategy: str = "auto",
    device: torch.device | str = "cuda",
    dtype: torch.dtype | None = None,
    group_offload_use_stream: bool = False,
    group_offload_low_cpu_mem: bool = False,
) -> Any:
    """Wrap a diffusers pipeline with smart model management.

    Discovers every ``nn.Module`` on the pipeline, registers it with a
    ``ModelManager``, and applies the chosen offload strategy.

    Pass an existing ``mm`` to share a manager across multiple pipelines —
    components shared by identity (the same ``nn.Module`` registered under
    the same name) are recognised as no-ops, so a pipeline can be recreated
    without re-hooking already-managed weights.

    Usage::

        from diffusers import StableDiffusionXLPipeline
        from diffusers_mm import managed, ModelManager

        # Single-pipeline case (default behaviour)
        pipe = StableDiffusionXLPipeline.from_pretrained(...)
        pipe = managed(pipe)
        image = pipe(prompt="...").images[0]

        # Multi-pipeline / long-lived manager
        mm = ModelManager(strategy="model_offload")
        pipe1 = managed(pipe1, mm=mm)
        pipe2 = managed(pipe2, mm=mm)  # shared components are deduped

    Args:
        pipe: A diffusers ``DiffusionPipeline`` instance.
        mm: Optional existing ``ModelManager`` to register against. When
            provided, the ``strategy`` and ``group_offload_*`` arguments
            are ignored — the manager owns its own configuration.
        strategy: Offload strategy when *mm* is not provided. One of
            ``"auto"``, ``"no_offload"``, ``"model_offload"``,
            ``"sequential_group_offload"``, ``"group_offload"``.
        device: Target device for inference.
        dtype: Optional dtype override for device scoping.
        group_offload_use_stream: Use CUDA streams for group offload
            transfers (only honoured when *mm* is not provided).
        group_offload_low_cpu_mem: Low CPU memory mode, only effective
            with streams (only honoured when *mm* is not provided).

    Returns:
        The same pipeline object, augmented with a ``.mm`` attribute and
        a wrapped ``__call__``.
    """
    if isinstance(device, str):
        device = torch.device(device)

    if mm is None:
        mm = ModelManager(
            strategy=strategy,
            group_offload_use_stream=group_offload_use_stream,
            group_offload_low_cpu_mem=group_offload_low_cpu_mem,
        )

    if not hasattr(pipe, "components") or not isinstance(pipe.components, dict):
        raise TypeError(
            f"{type(pipe).__name__} has no 'components' property. managed() requires a DiffusionPipeline instance."
        )

    registered = mm.register_components(pipe)
    for name in registered:
        logger.info("Registered component: %s (%s)", name, type(pipe.components[name]).__name__)

    if not registered:
        logger.warning("No nn.Module components found on pipeline %s", type(pipe).__name__)

    mm.apply_offload_strategy(device)

    original_call = pipe.__call__

    @wraps(original_call)
    def wrapped_call(*args: Any, **kwargs: Any) -> Any:
        with mm.device_scope(device=device, dtype=dtype):
            return original_call(*args, **kwargs)

    pipe.__call__ = wrapped_call  # type: ignore[method-assign]
    pipe.mm = mm  # type: ignore[attr-defined]

    logger.info(
        "Pipeline managed: applied_strategy=%s, device=%s, components=%s",
        mm.applied_strategy,
        device,
        registered,
    )

    return pipe
