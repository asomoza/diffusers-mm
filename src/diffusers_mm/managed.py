"""The ``managed()`` wrapper — drop-in replacement for diffusers' built-in offload methods."""

from __future__ import annotations

import logging
from typing import Any

import torch

from diffusers_mm.manager import ModelManager


logger = logging.getLogger(__name__)


# Sentinel for "user didn't pass this kwarg" — distinguishes "I passed the
# default value explicitly" from "I didn't pass anything." Used so that
# managed() can warn if the caller mixes ``mm=`` with configuration kwargs,
# without false-positiving when those kwargs equal their defaults.
_UNSET: Any = object()


def managed(
    pipe: Any,
    *,
    mm: ModelManager | None = None,
    strategy: str = _UNSET,
    device: torch.device | str = "cuda",
    dtype: torch.dtype | None = None,
    group_offload_use_stream: bool = _UNSET,
    group_offload_low_cpu_mem: bool = _UNSET,
    block_pin_auto_evict: bool = _UNSET,
    denoiser_concurrency: str = _UNSET,
    block_pin_spill_aware: bool = _UNSET,
    block_pin_spill_margin_gb: float = _UNSET,
    auto_no_offload_factor: float = _UNSET,
    auto_model_offload_factor: float = _UNSET,
    auto_ram_headroom: float = _UNSET,
    auto_low_cpu_mem_ram_headroom_gb: float = _UNSET,
    auto_block_pin_working_set_gb: float = _UNSET,
    auto_block_pin_working_set_windows_gb: float = _UNSET,
    auto_block_pin_min_blocks: int = _UNSET,
    auto_block_pin_ram_evict_headroom_gb: float = _UNSET,
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
            ``"group_offload"``. Default ``"auto"``.
        device: Target device for inference.
        dtype: Optional dtype override for device scoping.
        group_offload_use_stream: Use CUDA streams for group offload
            transfers — overlaps transfers with compute (~1.5–3× faster
            on hardware that supports it). Default True.
        group_offload_low_cpu_mem: Low CPU memory mode for group offload —
            avoids pinning a full copy of every weight upfront (which
            would ~double host RAM). Only honored when ``use_stream=True``.
            Default True.
        block_pin_auto_evict: For the ``block_pin`` strategy, evict the
            pinned subset to CPU when a neighbor component (text encoder,
            VAE, etc.) runs, then repin on demand when the pinned
            component's next forward fires. Frees several GiB of VRAM
            during VAE decode at the cost of two extra CPU↔GPU transfers
            per inference (typically 1–2 s on PCIe 4). Default True.
        denoiser_concurrency: How to budget pipelines with multiple denoisers
            (DiTs). ``"co_resident"`` (default) sums their sizes — correct when
            both run every step (e.g. Ideogram4 conditional + unconditional
            under True-CFG). ``"sequential"`` takes the largest single one —
            correct when only one is active at a time (e.g. Wan2.2 high/low-noise
            experts split by timestep). Wrong-way ``sequential`` on a
            co-resident pipeline under-budgets and can spill to RAM.
        block_pin_spill_aware: When ``block_pin`` is active, check after each
            managed generation whether the caching allocator reserved more than
            the card's VRAM (oversubscription / Windows sysmem fallback) and, if
            so, evict pinned blocks until it fits. Self-tunes the pin count to
            just under the memory ceiling for the actual workload. Default True.
        block_pin_spill_margin_gb: Headroom (GiB) kept below total VRAM when
            deciding whether the last run spilled and how much to evict.
            Default ``0.5``.
        auto_no_offload_factor: Activation margin for the ``no_offload``
            auto tier (``pipeline_weights × this`` must fit in VRAM).
            Default ``1.5``.
        auto_model_offload_factor: Activation margin for the
            ``model_offload`` auto tier (``largest_component × this``
            must fit in VRAM). Default ``1.5``.
        auto_ram_headroom: Fraction of RAM treated as "usable" before
            logging a 'workload won't fit' warning. Default ``0.85``.
        auto_low_cpu_mem_ram_headroom_gb: RAM headroom (GiB) required to
            flip ``group_offload``'s ``low_cpu_mem_usage=False`` when
            ``auto`` picks it. Default ``16.0``.
        auto_block_pin_working_set_gb: VRAM (GiB) reserved per
            ``block_pin`` component for streaming working set on
            Linux/macOS. Default ``6.5``. Bump for long-video workloads
            (10–14 GiB measured at 768×512×121f).
        auto_block_pin_working_set_windows_gb: Same as above on Windows
            (no ``expandable_segments``, ~2 GiB structural overhead).
            Default ``8.5``.
        auto_block_pin_min_blocks: Minimum block count required before
            ``auto`` will pick ``block_pin`` over ``group_offload``.
            Default ``8``.
        auto_block_pin_ram_evict_headroom_gb: RAM safety margin (GiB)
            for the ``block_pin`` auto-evict RAM-absorb check. Default
            ``4.0``.

    Returns:
        The same pipeline object, augmented with a ``.mm`` attribute and
        a wrapped ``__call__``.
    """
    if isinstance(device, str):
        device = torch.device(device)

    # All caller-supplied configuration knobs, in the same shape we'd
    # forward them to ``ModelManager``. Used twice below: once to build
    # the new-manager kwargs, once to warn when the caller mixed an
    # existing manager with config kwargs.
    config_kwargs = {
        "strategy": strategy,
        "group_offload_use_stream": group_offload_use_stream,
        "group_offload_low_cpu_mem": group_offload_low_cpu_mem,
        "block_pin_auto_evict": block_pin_auto_evict,
        "denoiser_concurrency": denoiser_concurrency,
        "block_pin_spill_aware": block_pin_spill_aware,
        "block_pin_spill_margin_gb": block_pin_spill_margin_gb,
        "auto_no_offload_factor": auto_no_offload_factor,
        "auto_model_offload_factor": auto_model_offload_factor,
        "auto_ram_headroom": auto_ram_headroom,
        "auto_low_cpu_mem_ram_headroom_gb": auto_low_cpu_mem_ram_headroom_gb,
        "auto_block_pin_working_set_gb": auto_block_pin_working_set_gb,
        "auto_block_pin_working_set_windows_gb": auto_block_pin_working_set_windows_gb,
        "auto_block_pin_min_blocks": auto_block_pin_min_blocks,
        "auto_block_pin_ram_evict_headroom_gb": auto_block_pin_ram_evict_headroom_gb,
    }
    explicit = {k: v for k, v in config_kwargs.items() if v is not _UNSET}

    if mm is None:
        # Build kwargs only for explicitly-passed values so ModelManager's
        # own defaults govern unset ones.
        mm = ModelManager(**explicit)
    elif explicit:
        # Detect a likely-confused caller: passing both an existing manager
        # AND configuration kwargs. The kwargs would be ignored, so surface
        # the mismatch instead of silently dropping intent.
        logger.warning(
            "managed(): an existing ModelManager was supplied along with "
            "configuration kwargs %s. These are ignored — the manager's "
            "existing configuration is used. Configure the manager directly "
            "if you want to change them.",
            explicit,
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

    # Wrap __call__ at the TYPE level, not the instance. `pipe(...)` resolves
    # `__call__` on type(pipe), so an instance attribute (`pipe.__call__ = ...`)
    # is silently bypassed by call syntax. We install the wrapper on a per-pipe
    # dynamic subclass instead. The wrapper scopes device/dtype for the call and,
    # afterward, lets the manager evict pinned blocks if the run oversubscribed
    # VRAM (spill-aware block_pin).
    cls = type(pipe)
    if not getattr(cls, "_diffusers_mm_wrapped", False):
        base_call = cls.__call__

        def _managed_call(self, *args: Any, __base_call=base_call, **kwargs: Any) -> Any:
            _mm = getattr(self, "mm", None)
            dev = getattr(self, "_diffusers_mm_device", None)
            if _mm is None or dev is None:
                return __base_call(self, *args, **kwargs)
            with _mm.device_scope(device=dev, dtype=getattr(self, "_diffusers_mm_dtype", None)):
                result = __base_call(self, *args, **kwargs)
            _mm._maybe_recalibrate_block_pin_spill(dev)
            return result

        # Reuse the original class name so repr / save_pretrained are unaffected.
        managed_cls = type(cls.__name__, (cls,), {"__call__": _managed_call, "_diffusers_mm_wrapped": True})
        pipe.__class__ = managed_cls  # type: ignore[assignment]

    pipe._diffusers_mm_device = device  # type: ignore[attr-defined]
    pipe._diffusers_mm_dtype = dtype  # type: ignore[attr-defined]
    pipe.mm = mm  # type: ignore[attr-defined]

    logger.info(
        "Pipeline managed: applied_strategy=%s, device=%s, components=%s",
        mm.applied_strategy,
        device,
        registered,
    )

    return pipe
