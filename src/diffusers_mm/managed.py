"""The ``managed()`` wrapper — drop-in replacement for diffusers' built-in offload methods."""

from __future__ import annotations

import logging
from functools import wraps
from typing import Any

import torch
from torch import nn

from diffusers_mm.manager import ModelManager


logger = logging.getLogger(__name__)


def managed(
    pipe: Any,
    *,
    strategy: str = "auto",
    device: torch.device | str = "cuda",
    dtype: torch.dtype | None = None,
    group_offload_use_stream: bool = False,
    group_offload_low_cpu_mem: bool = False,
) -> Any:
    """Wrap a diffusers pipeline with smart model management.

    Automatically discovers all ``nn.Module`` components from the pipeline,
    registers them with a ``ModelManager``, and applies the chosen offload
    strategy.

    Usage::

        from diffusers import LTX2Pipeline
        from diffusers_mm import managed

        pipe = LTX2Pipeline.from_pretrained("Lightricks/LTX-Video-0.9.7", torch_dtype=torch.bfloat16)
        pipe = managed(pipe)  # auto strategy, just works
        video = pipe(prompt="A cat")

        # Or with explicit options:
        pipe = managed(pipe, strategy="group_offload", group_offload_use_stream=True)

    Args:
        pipe: A diffusers ``DiffusionPipeline`` instance.
        strategy: Offload strategy. One of ``"auto"``, ``"no_offload"``,
            ``"model_offload"``, ``"sequential_group_offload"``, ``"group_offload"``.
        device: Target device for inference.
        dtype: Optional dtype override for device scoping.
        group_offload_use_stream: Use CUDA streams for group offload transfers.
        group_offload_low_cpu_mem: Low CPU memory mode (only effective with streams).

    Returns:
        The same pipeline object, augmented with a ``.mm`` attribute and
        wrapped ``__call__``.
    """
    if isinstance(device, str):
        device = torch.device(device)

    mm = ModelManager(
        strategy=strategy,
        group_offload_use_stream=group_offload_use_stream,
        group_offload_low_cpu_mem=group_offload_low_cpu_mem,
    )

    # Discover and register all nn.Module components from the pipeline
    components = getattr(pipe, "components", None)
    if components is None:
        raise TypeError(
            f"{type(pipe).__name__} has no 'components' property. managed() requires a DiffusionPipeline instance."
        )

    registered = []
    for name, component in components.items():
        if isinstance(component, nn.Module):
            mm.register_component(name, component)
            registered.append(name)
            logger.info("Registered component: %s (%s)", name, type(component).__name__)

    if not registered:
        logger.warning("No nn.Module components found on pipeline %s", type(pipe).__name__)

    # Apply the offload strategy
    mm.apply_offload_strategy(device)

    # Wrap __call__ with device scoping
    original_call = pipe.__call__

    @wraps(original_call)
    def wrapped_call(*args: Any, **kwargs: Any) -> Any:
        with mm.device_scope(device=device, dtype=dtype):
            return original_call(*args, **kwargs)

    pipe.__call__ = wrapped_call  # type: ignore[method-assign]
    pipe.mm = mm  # type: ignore[attr-defined]

    logger.info(
        "Pipeline managed: strategy=%s (resolved=%s), device=%s, components=%s",
        strategy,
        mm.applied_strategy,
        device,
        registered,
    )

    return pipe
