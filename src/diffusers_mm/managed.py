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
    block_pin_workload_probe: bool = _UNSET,
    block_pin_call_workload: bool = _UNSET,
    unload_text_encoders: bool = _UNSET,
    auto_no_offload_factor: float = _UNSET,
    auto_model_offload_factor: float = _UNSET,
    auto_ram_headroom: float = _UNSET,
    auto_low_cpu_mem_ram_headroom_gb: float = _UNSET,
    auto_block_pin_working_set_gb: float = _UNSET,
    auto_block_pin_working_set_windows_gb: float = _UNSET,
    auto_block_pin_min_blocks: int = _UNSET,
    auto_block_pin_ram_evict_headroom_gb: float = _UNSET,
    auto_block_pin_act_intercept_gb: float = _UNSET,
    auto_block_pin_act_slope_gb_per_ktoken: float = _UNSET,
    auto_block_pin_act_safety_factor: float = _UNSET,
    auto_block_pin_act_safety_factor_measured: float = _UNSET,
    auto_block_pin_act_fallback_gb: float = _UNSET,
    auto_block_pin_allocator_inflation: float = _UNSET,
    auto_block_pin_allocator_inflation_windows: float = _UNSET,
    auto_block_pin_allocator_pool_overhead_gb: float = _UNSET,
    auto_block_pin_allocator_pool_overhead_windows_gb: float = _UNSET,
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
            avoids pinning a full copy of every weight upfront, which would
            roughly double host RAM. Only honored when ``use_stream=True``.
            Default True.
        block_pin_auto_evict: For the ``block_pin`` strategy, evict the
            pinned subset to CPU when a neighbor component (text encoder,
            VAE, etc.) runs, then repin on demand when the pinned
            component's next forward fires. Frees the pinned subset's VRAM
            during VAE decode at the cost of two extra CPU↔GPU transfers
            per inference. Default True.
        denoiser_concurrency: How to budget pipelines with multiple denoisers
            (DiTs). ``"co_resident"`` sums their sizes — correct when both run
            every step (e.g. Ideogram4 conditional + unconditional under
            True-CFG). ``"sequential"`` takes the largest single one — correct
            when only one is active at a time (e.g. Wan2.2 high/low-noise
            experts split by timestep). Wrong-way ``sequential`` on a
            co-resident pipeline under-budgets and can spill to RAM.

            Usually you don't need this: when left unset, recognised pipeline
            architectures get the right value from their
            :class:`~diffusers_mm.model_profiles.ModelProfile`, falling back to
            ``"co_resident"`` for unknown ones. Pass it explicitly to override
            the profile — an explicit value always wins. Teach the resolver about
            a new architecture with
            :func:`~diffusers_mm.model_profiles.register_model_profile`.
        block_pin_spill_aware: When ``block_pin`` is active, check after each
            managed generation whether the caching allocator reserved more than
            the card's VRAM (oversubscription / Windows sysmem fallback) and, if
            so, evict pinned blocks until it fits. Self-tunes the pin count to
            just under the memory ceiling for the actual workload. Default True.
        block_pin_spill_margin_gb: Headroom (GiB) kept below total VRAM when
            deciding whether the last run spilled and how much to evict.
            Default ``0.5``.
        block_pin_workload_probe: When ``block_pin`` is active, read the true
            denoise sequence length off the denoiser's own input on its first
            forward — before any activation is allocated — and unpin blocks if
            the pin count doesn't leave enough room for it. This is what makes
            ``strategy="auto"`` safe on long/large video without the caller
            having to call :meth:`ModelManager.set_block_pin_workload`; the
            apply-time budget otherwise falls back to an image-scale activation
            estimate and over-pins. Only ever lowers the pin count. Default True.
        block_pin_call_workload: When ``block_pin`` is active and the pipeline's
            architecture has a profiled ``workload_fn``, compute the denoise
            workload from each call's own ``height`` / ``width`` / ``num_frames``
            and rebalance the pin count to match, before the pipeline runs. This
            is what makes ``set_block_pin_workload`` unnecessary: the budget is
            exact from the first step rather than corrected after it. Unlike the
            probe it rebalances in **both** directions, so a small generation
            following a large one recovers the pins the large one shed. Default
            True. Unprofiled architectures are unaffected.
        unload_text_encoders: Drop the text encoder(s) as soon as the first
            denoiser forward begins — prompt encoding is finished by then, and
            they are dead weight for the rest of the generation. Frees their
            weights *and* the pinned host copy group offload holds for them,
            which on a text-encoder-dominated pipeline is the largest single
            block of reclaimable memory. Covers multi-encoder pipelines
            (SDXL, SD3) by role, not by name.

            Destructive and therefore opt-in: the pipeline's own attribute is
            cleared, since that reference is what keeps the weights alive. A
            following generation reloads them if the pipeline can
            (``load_components``), otherwise it raises. Best suited to
            one-shot or memory-starved runs — a long-lived process doing many
            generations pays a full reload each time. Default False.
        auto_no_offload_factor: Activation margin for the ``no_offload``
            auto tier (``pipeline_weights × this`` must fit in VRAM).
        auto_model_offload_factor: Activation margin for the
            ``model_offload`` auto tier (``largest_component × this``
            must fit in VRAM).
        auto_ram_headroom: Fraction of RAM treated as "usable" before
            logging a 'workload won't fit' warning.
        auto_low_cpu_mem_ram_headroom_gb: RAM headroom (GiB) required to
            flip ``group_offload``'s ``low_cpu_mem_usage=False`` when
            ``auto`` picks it.
        auto_block_pin_working_set_gb: Platform safety headroom (GiB)
            added on top of the workload-aware activation estimate for
            the ``block_pin`` working set. Only the headroom — the bulk of
            the reserve scales with the recorded workload, see
            ``ModelManager.set_block_pin_workload``.
        auto_block_pin_working_set_windows_gb: The same on Windows, which
            has no ``expandable_segments`` and larger allocator overhead.
        auto_block_pin_min_blocks: Minimum block count required before
            ``auto`` will pick ``block_pin`` over ``group_offload``.
        auto_block_pin_ram_evict_headroom_gb: RAM safety margin (GiB)
            for the ``block_pin`` auto-evict RAM-absorb check.
        auto_block_pin_act_intercept_gb: Intercept (GiB) of the
            workload-aware activation fit.
        auto_block_pin_act_slope_gb_per_ktoken: Slope (GiB per 1000
            ``batch × seq_len`` tokens) of that fit.
        auto_block_pin_act_safety_factor: Multiplier on the activation
            estimate before the platform headroom is added, used when the
            slope is the generic default — most of it is cushion for not
            knowing the architecture's real activation cost.
        auto_block_pin_act_safety_factor_measured: The same multiplier, used
            instead when the slope came from a
            :class:`~diffusers_mm.model_profiles.ModelProfile`, which needs
            far less cushion.
        auto_block_pin_act_fallback_gb: Activation estimate (GiB) used
            when no workload has been recorded.
        auto_block_pin_allocator_inflation: Multiplier turning the
            sequence-length part of the activation estimate into a
            reserved-pool figure for the pin budget. Neutral off Windows.
        auto_block_pin_allocator_inflation_windows: The same, on Windows.
        auto_block_pin_allocator_pool_overhead_gb: Fixed pool overhead (GiB)
            added to the pin budget. Neutral off Windows.
        auto_block_pin_allocator_pool_overhead_windows_gb: The same, on
            Windows.

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
        "block_pin_workload_probe": block_pin_workload_probe,
        "block_pin_call_workload": block_pin_call_workload,
        "unload_text_encoders": unload_text_encoders,
        "auto_no_offload_factor": auto_no_offload_factor,
        "auto_model_offload_factor": auto_model_offload_factor,
        "auto_ram_headroom": auto_ram_headroom,
        "auto_low_cpu_mem_ram_headroom_gb": auto_low_cpu_mem_ram_headroom_gb,
        "auto_block_pin_working_set_gb": auto_block_pin_working_set_gb,
        "auto_block_pin_working_set_windows_gb": auto_block_pin_working_set_windows_gb,
        "auto_block_pin_min_blocks": auto_block_pin_min_blocks,
        "auto_block_pin_ram_evict_headroom_gb": auto_block_pin_ram_evict_headroom_gb,
        "auto_block_pin_act_intercept_gb": auto_block_pin_act_intercept_gb,
        "auto_block_pin_act_slope_gb_per_ktoken": auto_block_pin_act_slope_gb_per_ktoken,
        "auto_block_pin_act_safety_factor": auto_block_pin_act_safety_factor,
        "auto_block_pin_act_safety_factor_measured": auto_block_pin_act_safety_factor_measured,
        "auto_block_pin_act_fallback_gb": auto_block_pin_act_fallback_gb,
        "auto_block_pin_allocator_inflation": auto_block_pin_allocator_inflation,
        "auto_block_pin_allocator_inflation_windows": auto_block_pin_allocator_inflation_windows,
        "auto_block_pin_allocator_pool_overhead_gb": auto_block_pin_allocator_pool_overhead_gb,
        "auto_block_pin_allocator_pool_overhead_windows_gb": auto_block_pin_allocator_pool_overhead_windows_gb,
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
            "configuration kwargs %s. These are ignored - the manager's "
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
            # Size the block_pin budget from this call's own arguments before
            # the pipeline allocates anything. Never fatal: a failure here means
            # the previous budget stands and the forward-time probe still guards.
            # Reload anything a previous generation's unload_text_encoders()
            # dropped; raises with guidance if this pipeline can't reload.
            _mm.restore_dropped_components(self, dev)
            try:
                _mm._prepare_block_pin_for_call(self, kwargs, dev)
            except Exception as e:
                logger.warning("block_pin call-workload preparation failed: %s", e)
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
