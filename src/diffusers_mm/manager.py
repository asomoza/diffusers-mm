"""Core ModelManager — thread-safe model lifecycle and offload strategy management."""

from __future__ import annotations

import contextlib
import contextvars
import gc
import hashlib
import logging
import math
import sys
import threading
import weakref
from collections.abc import Callable, Generator
from typing import Any

import torch
from torch import nn

from diffusers_mm import offload_defaults
from diffusers_mm.block_pin import (
    BlockPinState,
    apply_block_pin,
    evict_pinned_subset,
    find_largest_block_list,
    non_block_size_bytes,
    per_block_size_bytes,
    pin_blocks,
    repin_pinned_subset,
    unpin_blocks,
)
from diffusers_mm.hooks import find_legacy_weight_norm, remove_offload_hooks
from diffusers_mm.inventory import ComponentInfo, build_inventory, module_size_gb
from diffusers_mm.model_profiles import (
    DENOISER_CONCURRENCY_MODES,
    ModelProfile,
    get_model_profile,
    resolve_call_workload,
)
from diffusers_mm.modular_compat import ensure_modular_compat


__all__ = ["DENOISER_CONCURRENCY_MODES", "ModelManager", "get_device", "get_dtype"]


logger = logging.getLogger(__name__)

OFFLOAD_STRATEGIES = ("auto", "no_offload", "model_offload", "group_offload", "block_pin")

# Sentinel for "the wrapped method was not on the instance __dict__; restore
# by deleting the instance attribute so the class-level method is reachable
# again." See ``_wrap_neighbor_method`` for the rationale.
_INSTANCE_ATTR_ABSENT: Any = object()

# Methods on neighbor components (the non-pinned ones) we wrap with an
# evict-first shim. Bound forward is handled by register_forward_pre_hook;
# decode and encode are the standard diffusers VAE-style entry points that
# bypass ``__call__``. Extend cautiously — every name here costs one
# instance-attribute write per component on apply and one delete on
# strategy transition.
_BLOCK_PIN_NEIGHBOR_WRAP_METHODS: tuple[str, ...] = ("decode", "encode")

_SCOPED_DEVICE: contextvars.ContextVar[torch.device | None] = contextvars.ContextVar(
    "diffusers_mm_scoped_device", default=None
)
_SCOPED_DTYPE: contextvars.ContextVar[torch.dtype | None] = contextvars.ContextVar(
    "diffusers_mm_scoped_dtype", default=None
)


def _is_rocm() -> bool:
    """True when torch is a ROCm/HIP build rather than a CUDA one.

    Both report themselves as ``"cuda"`` devices, so ``torch.version.hip``
    is the only reliable discriminator. Wrapped because the attribute is
    absent on some builds.
    """
    try:
        return torch.version.hip is not None
    except Exception:
        return False


def _allocator_conf() -> str:
    """Return the caching-allocator config string torch will actually parse.

    Mirrors torch's own precedence (``c10/cuda/CUDAAllocatorConfig.h``):
    ``PYTORCH_CUDA_ALLOC_CONF`` first, then ``PYTORCH_HIP_ALLOC_CONF`` on
    ROCm builds, then the unified ``PYTORCH_ALLOC_CONF``. Presence wins over
    content there, so a var that is set but empty stops the fallback here
    too. Reading only the CUDA-named var would miss a user who configured
    the allocator through either of the other two.
    """
    import os

    names = ["PYTORCH_CUDA_ALLOC_CONF"]
    if _is_rocm():
        names.append("PYTORCH_HIP_ALLOC_CONF")
    names.append("PYTORCH_ALLOC_CONF")
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value
    return ""


def get_device() -> torch.device | None:
    """Return the device set by the nearest enclosing ``device_scope``."""
    return _SCOPED_DEVICE.get()


def get_dtype() -> torch.dtype | None:
    """Return the dtype set by the nearest enclosing ``device_scope``."""
    return _SCOPED_DTYPE.get()


class ModelManager:
    """Thread-safe manager for model component lifecycle and offload strategies.

    Handles component registration, hash-keyed caching, device scoping via
    context vars, and automatic offload strategy resolution/application.
    """

    # Heuristic factors used by the ``auto`` strategy resolver: weights occupy
    # GPU memory steadily, activations come and go on top. Tweak via subclass /
    # attribute set for an unusually large or small activation footprint.
    AUTO_NO_OFFLOAD_FACTOR = 1.5  # DEPRECATED: superseded by the additive
    # ``weights + working_set`` no_offload gate (see resolve_offload_strategy).
    # Retained so existing ctor/attribute usage keeps working; no longer read
    # by the default resolver path.
    AUTO_MODEL_OFFLOAD_FACTOR = 1.5  # largest single component must fit × this
    # If pipeline weights exceed RAM × this, log a loud warning that the
    # workload likely won't fit on host memory at all.
    AUTO_RAM_HEADROOM = 0.85
    # When ``auto`` picks ``group_offload``, whether to flip
    # ``low_cpu_mem_usage`` off: ``False`` pre-pins a full host copy of every
    # weight once at apply time (faster transfers, higher peak host RAM),
    # ``True`` pins per-transfer. Flips when ``RAM >= weights + headroom``; the
    # headroom covers OS, activations and transient buffers, but not the original
    # weights, since safetensors is mmap'd and those pages get evicted as needed.
    AUTO_LOW_CPU_MEM_RAM_HEADROOM_GB = 16.0
    # Platform safety headroom added *on top of* the workload-aware activation
    # estimate — allocator fragmentation, the group-offload stream double-buffer,
    # attention overhead — not the whole working set as in releases <= 0.2.x. See
    # :meth:`_resolve_working_set_gb`. Higher on Windows, which has no
    # ``expandable_segments`` and so reserves more under the same load.
    AUTO_BLOCK_PIN_WORKING_SET_GB = offload_defaults.BLOCK_PIN_WORKING_SET_HEADROOM_GB
    AUTO_BLOCK_PIN_WORKING_SET_WINDOWS_GB = offload_defaults.BLOCK_PIN_WORKING_SET_HEADROOM_WINDOWS_GB
    # VRAM withheld from every budget so the Windows sysmem fallback is never reached. See
    # :meth:`_resolve_vram_reserve_gb`; zero off Windows, where the allocator raises instead of spilling.
    VRAM_RESERVE_GB = offload_defaults.VRAM_RESERVE_GB
    VRAM_RESERVE_WINDOWS_GB = offload_defaults.VRAM_RESERVE_WINDOWS_GB
    VRAM_RESERVE_WINDOWS_LARGE_CARD_EXTRA_GB = offload_defaults.VRAM_RESERVE_WINDOWS_LARGE_CARD_EXTRA_GB
    VRAM_RESERVE_LARGE_CARD_THRESHOLD_GB = offload_defaults.VRAM_RESERVE_LARGE_CARD_THRESHOLD_GB
    # Activation fit ``intercept + slope × ktokens``, where
    # ``ktokens = batch × seq_len / 1000`` from :meth:`set_block_pin_workload`,
    # so the reserve scales with video size / length instead of being flat. The
    # safety factor lifts the bare fit to a safe ceiling; the fallback covers an
    # unknown workload.
    AUTO_BLOCK_PIN_ACT_INTERCEPT_GB = offload_defaults.BLOCK_PIN_ACT_INTERCEPT_GB
    AUTO_BLOCK_PIN_ACT_SLOPE_GB_PER_KTOKEN = offload_defaults.BLOCK_PIN_ACT_SLOPE_GB_PER_KTOKEN
    AUTO_BLOCK_PIN_ACT_SAFETY_FACTOR = offload_defaults.BLOCK_PIN_ACT_SAFETY_FACTOR
    AUTO_BLOCK_PIN_ACT_SAFETY_FACTOR_MEASURED = offload_defaults.BLOCK_PIN_ACT_SAFETY_FACTOR_MEASURED
    AUTO_BLOCK_PIN_ACT_FALLBACK_GB = offload_defaults.BLOCK_PIN_ACT_FALLBACK_GB
    # Turns the activation estimate (peak *live* bytes) into the caching
    # allocator's peak *reserved* pool, which is what competes with pinned
    # weights for driver pages: a multiplier on the sequence-length term plus a
    # fixed overhead. Both neutral off Windows — see
    # ``offload_defaults.BLOCK_PIN_ALLOCATOR_INFLATION``.
    AUTO_BLOCK_PIN_ALLOCATOR_INFLATION = offload_defaults.BLOCK_PIN_ALLOCATOR_INFLATION
    AUTO_BLOCK_PIN_ALLOCATOR_INFLATION_WINDOWS = offload_defaults.BLOCK_PIN_ALLOCATOR_INFLATION_WINDOWS
    AUTO_BLOCK_PIN_ALLOCATOR_POOL_OVERHEAD_GB = offload_defaults.BLOCK_PIN_ALLOCATOR_POOL_OVERHEAD_GB
    AUTO_BLOCK_PIN_ALLOCATOR_POOL_OVERHEAD_WINDOWS_GB = offload_defaults.BLOCK_PIN_ALLOCATOR_POOL_OVERHEAD_WINDOWS_GB
    # Don't bother with block_pin if the discoverable block list is
    # smaller than this — the overhead of per-block apply_group_offloading
    # outweighs the benefit when there are only a handful of blocks.
    AUTO_BLOCK_PIN_MIN_BLOCKS = 8
    # Safety headroom for the auto-evict RAM check: eviction fires only when
    # ``ram_available >= evicted_subset + this``, so pushing pinned blocks back
    # to the host cannot itself OOM. Without it an eviction can "succeed" via
    # swap and then starve the next ``cudaHostAlloc``; refusing it instead leaves
    # the neighbor to fit in whatever VRAM is free, which is not worse. The
    # headroom covers the neighbor's own ``pin_memory`` staging and OS slack.
    AUTO_BLOCK_PIN_RAM_EVICT_HEADROOM_GB = 4.0

    def __init__(
        self,
        strategy: str = "auto",
        group_offload_use_stream: bool = True,
        group_offload_low_cpu_mem: bool = True,
        block_pin_auto_evict: bool = True,
        denoiser_concurrency: str | None = None,
        block_pin_spill_aware: bool = True,
        block_pin_spill_margin_gb: float = 0.5,
        block_pin_workload_probe: bool = True,
        block_pin_call_workload: bool = True,
        unload_text_encoders: bool = False,
        *,
        auto_no_offload_factor: float | None = None,
        auto_model_offload_factor: float | None = None,
        auto_ram_headroom: float | None = None,
        auto_low_cpu_mem_ram_headroom_gb: float | None = None,
        auto_block_pin_working_set_gb: float | None = None,
        auto_block_pin_working_set_windows_gb: float | None = None,
        auto_block_pin_min_blocks: int | None = None,
        auto_block_pin_ram_evict_headroom_gb: float | None = None,
        auto_block_pin_act_intercept_gb: float | None = None,
        auto_block_pin_act_slope_gb_per_ktoken: float | None = None,
        auto_block_pin_act_safety_factor: float | None = None,
        auto_block_pin_act_safety_factor_measured: float | None = None,
        auto_block_pin_act_fallback_gb: float | None = None,
        auto_block_pin_allocator_inflation: float | None = None,
        auto_block_pin_allocator_inflation_windows: float | None = None,
        auto_block_pin_allocator_pool_overhead_gb: float | None = None,
        auto_block_pin_allocator_pool_overhead_windows_gb: float | None = None,
    ) -> None:
        """Construct a manager.

        The first four arguments configure runtime behaviour. The
        ``auto_*`` keyword-only arguments tune the size-aware ``"auto"``
        resolver — each one defaults to the class constant of the same
        name (uppercased) when left as ``None``. Set the class constant
        on a subclass for a global default; pass the ctor arg for a
        per-instance override; assign ``mm.AUTO_FOO = value`` for live
        mutation after construction.

        Args:
            strategy: Offload strategy. ``"auto"`` (default) resolves
                against VRAM / RAM / component sizes; explicit values
                bypass the resolver. One of ``"auto"``, ``"no_offload"``,
                ``"model_offload"``, ``"group_offload"``, ``"block_pin"``.
            group_offload_use_stream: Use CUDA streams for group offload
                transfers, overlapping CPU↔GPU copies with compute.
            group_offload_low_cpu_mem: Defer pinned-host-buffer allocation
                per transfer instead of pre-pinning a full copy of every
                streamed weight. Lower host RAM, slightly slower
                steady-state. Only honored when ``use_stream=True``.
            block_pin_auto_evict: When the ``block_pin`` strategy is
                active, evict the pinned subset back to CPU before a
                neighbor (text encoder, VAE, ...) runs, then repin on
                demand. Frees VRAM during VAE decode at the cost of two
                extra CPU↔GPU transfers per inference.
            auto_no_offload_factor: Activation margin for the
                ``no_offload`` tier — ``pipeline_weights × this`` must fit
                in available VRAM. Raise when activations are large
                relative to weights.
            auto_model_offload_factor: The same for the ``model_offload``
                tier, against the largest component.
            auto_ram_headroom: Fraction of RAM the resolver considers
                "usable" before logging a 'workload won't fit on host'
                warning.
            auto_low_cpu_mem_ram_headroom_gb: When ``auto`` picks
                ``group_offload``, flip ``low_cpu_mem_usage`` to ``False``
                (faster transfers, higher host RAM) if
                ``RAM ≥ pipeline_weights + this``. Lower it on RAM-rich
                systems to bias toward speed, raise it to stay in low-RAM
                mode.
            auto_block_pin_working_set_gb: Platform safety headroom (GiB)
                added on top of the workload-aware activation estimate for
                the ``block_pin`` working set. Only the headroom — the bulk
                of the reserve scales with the recorded workload, see
                :meth:`set_block_pin_workload`.
            auto_block_pin_working_set_windows_gb: The same, on Windows,
                whose allocator reserves more under the same load.
            auto_block_pin_min_blocks: Minimum block count for ``auto`` to
                pick ``block_pin`` over plain ``group_offload``; below it
                per-block hook overhead outweighs pinning.
            auto_block_pin_ram_evict_headroom_gb: Safety margin for the
                ``block_pin`` auto-evict RAM-absorb check. Eviction fires
                only when ``ram_available ≥ evicted_subset + this``. Raise
                it for neighbors with large host-side staging needs.
            auto_block_pin_act_intercept_gb: Intercept (GiB) of the
                workload-aware activation fit.
            auto_block_pin_act_slope_gb_per_ktoken: Slope (GiB per 1000
                ``batch × seq_len`` tokens) of that fit. Raise it for
                pipelines whose activations grow faster with sequence
                length.
            auto_block_pin_act_safety_factor: Multiplier on the activation
                estimate before the platform headroom is added.
            auto_block_pin_act_safety_factor_measured: The same multiplier,
                used instead when the slope came from a
                :class:`~diffusers_mm.model_profiles.ModelProfile`, which
                needs far less cushion than the generic default.
            auto_block_pin_act_fallback_gb: Activation estimate used when
                no workload has been recorded via
                :meth:`set_block_pin_workload`.
            auto_block_pin_allocator_inflation: Multiplier turning the
                sequence-length part of the activation estimate into a
                reserved-pool figure for the pin budget. Neutral off
                Windows.
            auto_block_pin_allocator_inflation_windows: The same, on Windows.
            auto_block_pin_allocator_pool_overhead_gb: Fixed pool overhead
                (GiB) added to the pin budget on top of the platform
                headroom. Neutral off Windows.
            auto_block_pin_allocator_pool_overhead_windows_gb: The same, on
                Windows. Follows the streaming configuration rather than the
                model, so measure it for unusual block geometry.
        """
        self._lock = threading.RLock()
        self._component_cache: dict[str, Any] = {}
        self._managed_components: dict[str, Any] = {}
        # Per-component applied strategy. Names absent from this dict are
        # "pending" — they were registered after a strategy was last applied
        # and need to be hooked/placed on the next apply_offload_strategy
        # call. Cleared wholesale on a strategy *transition* so every
        # component gets re-applied under the new regime.
        self._component_strategies: dict[str, str] = {}
        # Refcount keyed by id(module). Every register_component call
        # increments; every unregister_component call decrements. Cleanup
        # (hooks, cache, slot deletion) only happens when a module's
        # refcount hits 0 — this is what lets shared modules across
        # multiple pipelines stay alive while any consumer still needs them.
        self._refcount: dict[int, int] = {}
        # Per-source registration record, keyed by id(source). Stores the
        # exact (name, module) pairs each source registered via
        # register_components, so unregister_components can decrement
        # precisely without needing the user to re-list. Idempotency for
        # bulk register/unregister also lives here: a second
        # register_components on the same source is a no-op.
        self._source_registrations: dict[int, dict[str, Any]] = {}
        # weakref.finalize handles per source, keyed by id(source). When a
        # source object is garbage-collected, its finalizer fires and
        # auto-unregisters its components — preventing the leak where a
        # user lets a pipeline go out of scope without calling
        # unregister_components. Dict sources aren't weakref-able, so the
        # entry may be missing for those.
        self._source_finalizers: dict[int, weakref.finalize] = {}
        # Weak handles on the registered sources, parallel to
        # ``_source_registrations``. Needed by :meth:`unload_text_encoders`,
        # which must clear ``pipe.text_encoder`` as well as its own registry —
        # the pipeline's attribute is a strong reference, so leaving it in place
        # would free nothing.
        self._source_refs: dict[int, weakref.ref] = {}
        # Components dropped by :meth:`unload_text_encoders`, as
        # ``name -> source_id``, so a later call can reload them (or explain
        # why it can't).
        self._dropped_components: dict[str, int] = {}
        # When model_offload installs accelerate's chained
        # ``cpu_offload_with_hook``, the last hook in the chain is kept here.
        # It can be used to manually offload the trailing component (the one
        # that stays on GPU after the chain's final forward) — diffusers
        # exposes this on the pipeline as ``final_offload_hook``.
        self._model_offload_final_hook: Any = None
        # Per-component overrides for ``block_pin``: name → number of blocks
        # to pin on GPU. Names absent from this dict get auto-computed from
        # the available VRAM at apply time.
        self._block_pin_counts: dict[str, int] = {}
        # Which of those counts the *caller* set via ``set_block_pin_count``,
        # as opposed to ones the manager calibrated itself. Both live in
        # ``_block_pin_counts`` (so a later re-apply reproduces either), but
        # only the manager's own are up for automatic rebalancing — an
        # explicit count is an instruction, not a starting point.
        self._block_pin_user_counts: set[str] = set()
        # Expected denoise workload for the workload-aware block_pin working
        # set. ``seq_len = latent_frames × latent_h × latent_w`` and ``batch``
        # is 2 under CFG; ``activation_scale`` (>= 1.0) inflates the base
        # estimate for LoRAs / conditioning (see ``block_pin_activation_scale``).
        # Recorded via :meth:`set_block_pin_workload`; ``seq_len = 0`` means
        # "unknown" and falls back to ``AUTO_BLOCK_PIN_ACT_FALLBACK_GB``.
        self._block_pin_seq_len: int = 0
        self._block_pin_batch: int = 1
        self._block_pin_activation_scale: float = 1.0
        # Per-component pinned-subset state, populated when block_pin is
        # applied. Drives the auto-evict / repin pre-forward hooks so the
        # pinned weights don't squat on VRAM while a sibling component
        # (e.g. the VAE during decode) needs the space.
        self._block_pin_states: dict[str, BlockPinState] = {}
        # All pre-forward hook handles installed by block_pin's auto-evict
        # machinery — repin hooks on block-pinned components AND evict
        # hooks on neighbor components. Wholesale-removed on strategy
        # transition / clear.
        self._block_pin_hook_handles: list[Any] = []
        # Wrapped methods on non-pinned components (e.g. ``vae.decode``) —
        # tracked so we can unwrap on strategy transition. Each entry:
        # ``(component, method_name, restore_value)``. ``restore_value``
        # is a saved bound method if the original was an instance attribute
        # (rare); otherwise the sentinel ``_INSTANCE_ATTR_ABSENT`` which
        # tells the unwrap step to ``delattr`` instead.
        self._block_pin_wrapped_methods: list[tuple[Any, str, Any]] = []
        # Forward-hook handles for the step-1 spill calibration (Windows).
        self._spill_calib_handles: list[Any] = []
        # Pre-forward-hook handles for the one-shot workload probe, which reads
        # the true sequence length off the denoiser's own input.
        self._workload_probe_handles: list[Any] = []
        # Per-neighbor explicit override for the auto-evict decision.
        # ``True`` forces eviction on every call (paranoid mode); ``False``
        # disables it entirely for that component (e.g. text encoders that
        # don't need the headroom). A missing entry means "use the runtime
        # free-VRAM check" — see ``_should_evict_for_neighbor``.
        self._evict_on_neighbor: dict[str, bool] = {}
        self._applied_strategy: str | None = None

        self._offload_strategy: str = "auto"
        self.offload_strategy = strategy  # validate through setter
        self._group_offload_use_stream: bool = bool(group_offload_use_stream)
        self._group_offload_low_cpu_mem: bool = bool(group_offload_low_cpu_mem)
        self._block_pin_auto_evict: bool = bool(block_pin_auto_evict)

        # How to budget multiple denoisers (DiTs): "co_resident" sums them (both
        # used every step, e.g. Ideogram4 True-CFG); "sequential" takes the max
        # (one active at a time, e.g. Wan2.2 high/low-noise experts).
        # ``None`` means "use the registered pipeline's ModelProfile if it has one,
        # else the co_resident default" — an explicit value always wins over the
        # profile (see the ``denoiser_concurrency`` property).
        if denoiser_concurrency is not None and denoiser_concurrency not in DENOISER_CONCURRENCY_MODES:
            raise ValueError(
                f"denoiser_concurrency must be one of {DENOISER_CONCURRENCY_MODES}, got {denoiser_concurrency!r}"
            )
        self._denoiser_concurrency: str | None = denoiser_concurrency
        # ModelProfile of the registered pipeline architecture, if recognised.
        self._model_profile: ModelProfile | None = None

        # Spill-aware block_pin: after a managed generation, if the caching
        # allocator reserved more than the card's VRAM (Windows sysmem fallback
        # / oversubscription), evict pinned blocks until it fits — self-tuning
        # the pin count to just under the ceiling for the actual workload.
        self._block_pin_spill_aware: bool = bool(block_pin_spill_aware)
        self._block_pin_spill_margin_gb: float = float(block_pin_spill_margin_gb)
        self._block_pin_workload_probe: bool = bool(block_pin_workload_probe)
        # Read the denoise workload off each managed ``pipe(...)`` call's own
        # arguments (via the architecture's ``ModelProfile.workload_fn``) and
        # rebalance the pin count to match, before the pipeline body runs.
        self._block_pin_call_workload: bool = bool(block_pin_call_workload)
        # Opt-in: drop the text encoder(s) once denoising starts, reclaiming
        # both their weights and their pinned host copy for the rest of the run.
        self._unload_text_encoders: bool = bool(unload_text_encoders)
        self._text_encoder_unload_handles: list[Any] = []
        self._block_pin_spill_recalibrations: int = 0
        self._workload_fit_warned: set[tuple[str, int]] = set()

        # Per-instance overrides for the auto-resolver constants. We
        # shadow the class attribute with an instance attribute only
        # when the user explicitly passed a value, so subclassing /
        # ``mm.AUTO_X = value`` mutation continue to work unchanged.
        if auto_no_offload_factor is not None:
            self.AUTO_NO_OFFLOAD_FACTOR = float(auto_no_offload_factor)
        if auto_model_offload_factor is not None:
            self.AUTO_MODEL_OFFLOAD_FACTOR = float(auto_model_offload_factor)
        if auto_ram_headroom is not None:
            self.AUTO_RAM_HEADROOM = float(auto_ram_headroom)
        if auto_low_cpu_mem_ram_headroom_gb is not None:
            self.AUTO_LOW_CPU_MEM_RAM_HEADROOM_GB = float(auto_low_cpu_mem_ram_headroom_gb)
        if auto_block_pin_working_set_gb is not None:
            self.AUTO_BLOCK_PIN_WORKING_SET_GB = float(auto_block_pin_working_set_gb)
        if auto_block_pin_working_set_windows_gb is not None:
            self.AUTO_BLOCK_PIN_WORKING_SET_WINDOWS_GB = float(auto_block_pin_working_set_windows_gb)
        if auto_block_pin_min_blocks is not None:
            self.AUTO_BLOCK_PIN_MIN_BLOCKS = int(auto_block_pin_min_blocks)
        if auto_block_pin_ram_evict_headroom_gb is not None:
            self.AUTO_BLOCK_PIN_RAM_EVICT_HEADROOM_GB = float(auto_block_pin_ram_evict_headroom_gb)
        # Tracked so a ModelProfile's measured slope/intercept can slot in
        # *between* an explicit caller value and the class default.
        self._explicit_act_slope = auto_block_pin_act_slope_gb_per_ktoken is not None
        self._explicit_act_intercept = auto_block_pin_act_intercept_gb is not None
        self._explicit_act_safety_factor = auto_block_pin_act_safety_factor is not None
        if auto_block_pin_act_intercept_gb is not None:
            self.AUTO_BLOCK_PIN_ACT_INTERCEPT_GB = float(auto_block_pin_act_intercept_gb)
        if auto_block_pin_act_slope_gb_per_ktoken is not None:
            self.AUTO_BLOCK_PIN_ACT_SLOPE_GB_PER_KTOKEN = float(auto_block_pin_act_slope_gb_per_ktoken)
        if auto_block_pin_act_safety_factor is not None:
            self.AUTO_BLOCK_PIN_ACT_SAFETY_FACTOR = float(auto_block_pin_act_safety_factor)
        if auto_block_pin_act_safety_factor_measured is not None:
            self.AUTO_BLOCK_PIN_ACT_SAFETY_FACTOR_MEASURED = float(auto_block_pin_act_safety_factor_measured)
        if auto_block_pin_act_fallback_gb is not None:
            self.AUTO_BLOCK_PIN_ACT_FALLBACK_GB = float(auto_block_pin_act_fallback_gb)
        if auto_block_pin_allocator_inflation is not None:
            self.AUTO_BLOCK_PIN_ALLOCATOR_INFLATION = float(auto_block_pin_allocator_inflation)
        if auto_block_pin_allocator_inflation_windows is not None:
            self.AUTO_BLOCK_PIN_ALLOCATOR_INFLATION_WINDOWS = float(auto_block_pin_allocator_inflation_windows)
        if auto_block_pin_allocator_pool_overhead_gb is not None:
            self.AUTO_BLOCK_PIN_ALLOCATOR_POOL_OVERHEAD_GB = float(auto_block_pin_allocator_pool_overhead_gb)
        if auto_block_pin_allocator_pool_overhead_windows_gb is not None:
            self.AUTO_BLOCK_PIN_ALLOCATOR_POOL_OVERHEAD_WINDOWS_GB = float(
                auto_block_pin_allocator_pool_overhead_windows_gb
            )

    # ------------------------------------------------------------------
    # Strategy properties
    # ------------------------------------------------------------------

    @property
    def offload_strategy(self) -> str:
        with self._lock:
            return self._offload_strategy

    @offload_strategy.setter
    def offload_strategy(self, value: str) -> None:
        if value not in OFFLOAD_STRATEGIES:
            raise ValueError(f"Unknown offload strategy {value!r}. Must be one of {OFFLOAD_STRATEGIES}")
        with self._lock:
            self._offload_strategy = value

    @property
    def group_offload_use_stream(self) -> bool:
        with self._lock:
            return self._group_offload_use_stream

    @group_offload_use_stream.setter
    def group_offload_use_stream(self, value: bool) -> None:
        with self._lock:
            self._group_offload_use_stream = bool(value)

    @property
    def group_offload_low_cpu_mem(self) -> bool:
        with self._lock:
            return self._group_offload_low_cpu_mem

    @group_offload_low_cpu_mem.setter
    def group_offload_low_cpu_mem(self, value: bool) -> None:
        with self._lock:
            self._group_offload_low_cpu_mem = bool(value)

    @property
    def block_pin_auto_evict(self) -> bool:
        """If True, ``block_pin`` evicts pinned blocks when a neighbor runs.

        Without this, pinned blocks stay on GPU forever — fast for the
        denoise loop (no per-step transfers) but wasteful during the
        trailing VAE decode, which then has to share VRAM with the dead
        transformer subset. With this on, every non-pinned component
        installs a pre-forward / pre-decode / pre-encode hook that pushes
        every resident pinned subset back to CPU before its own work
        starts; the next pinned-component forward repins on demand.

        The trade-off is one extra ~``pinned_size`` CPU↔GPU transfer per
        eviction/repin cycle. For the common single-stage video flow
        (encode → denoise loop → decode), that's two extra transfers in
        exchange for freeing the pinned subset's VRAM during decode.
        """
        with self._lock:
            return self._block_pin_auto_evict

    @block_pin_auto_evict.setter
    def block_pin_auto_evict(self, value: bool) -> None:
        with self._lock:
            self._block_pin_auto_evict = bool(value)

    @property
    def applied_strategy(self) -> str | None:
        with self._lock:
            return self._applied_strategy

    # ------------------------------------------------------------------
    # Component registration
    # ------------------------------------------------------------------

    def _decrement_module_refcount(self, module: Any, *, slot_name_to_skip: str | None = None) -> None:
        """Decrement *module*'s refcount. If it reaches zero, fully clean up.

        Caller must hold ``self._lock``. "Fully clean up" means: remove
        offload hooks (idempotent — safe even if no hooks were installed),
        evict any cache entries pointing at the module, and delete every
        slot in ``_managed_components`` that still points to the module.

        *slot_name_to_skip* lets the displacement path in
        ``register_component`` reuse the slot it's about to reassign — we
        don't want to delete the slot only to immediately set it to the
        new module.
        """
        rc = self._refcount.get(id(module), 0) - 1
        if rc > 0:
            self._refcount[id(module)] = rc
            return
        # Last reference — clean up.
        self._refcount.pop(id(module), None)
        try:
            remove_offload_hooks(module)
        except Exception as e:
            logger.warning("Failed to remove offload hooks during refcount cleanup: %s", e)
        # Delete every slot that still points at this module (covers
        # aliases). Skip the slot the caller is about to reassign.
        orphan_names = [n for n, m in self._managed_components.items() if m is module and n != slot_name_to_skip]
        for n in orphan_names:
            del self._managed_components[n]
            self._component_strategies.pop(n, None)
        # Evict cache entries pointing at this module.
        cache_keys = [k for k, v in self._component_cache.items() if v is module]
        for k in cache_keys:
            del self._component_cache[k]
        if orphan_names or cache_keys:
            logger.info(
                "Refcount cleanup: removed %d orphan slot(s), evicted %d cache entr%s",
                len(orphan_names),
                len(cache_keys),
                "y" if len(cache_keys) == 1 else "ies",
            )

    def register_component(self, name: str, module: Any) -> None:
        """Register a named ``nn.Module`` component for lifecycle management.

        **Each call increments the module's refcount** — register_component
        is *not* idempotent. Two registrations of the same module (whether
        under the same name from two sources, or under different names from
        one source) raise the refcount to 2; the module won't be cleaned
        up until both registrations are matched by ``unregister_component``
        calls (or via ``unregister_components(source)``).

        For pipeline-level idempotency — i.e. calling ``managed(pipe)``
        twice on the same pipe should not double-count — use the bulk
        :meth:`register_components` API which dedupes by source identity.

        If a *different* module is registered under an existing name, the
        previous module's refcount is decremented (potentially triggering
        full cleanup if it was that module's last registration). The slot
        is then reassigned to the new module and its per-component strategy
        state is reset so the new module gets re-hooked on the next
        ``apply_offload_strategy`` call. Other components are left alone.
        """
        with self._lock:
            existing = self._managed_components.get(name)
            if existing is not module:
                if existing is not None:
                    self._decrement_module_refcount(existing, slot_name_to_skip=name)
                self._managed_components[name] = module
                self._component_strategies.pop(name, None)
            self._refcount[id(module)] = self._refcount.get(id(module), 0) + 1

    def unregister_component(self, name: str) -> bool:
        """Decrement the refcount of the module at *name*.

        - If the resulting refcount is still > 0, the slot **stays** in the
          registry and no cleanup runs (some other consumer — another
          source, another alias — still needs the module). This is what
          makes shared components across multiple pipelines work: pipe1
          unregistering its T5 doesn't yank T5 out from under pipe2.
        - If the refcount hits 0, the module is fully cleaned up (hooks
          removed, cache evicted, all slots pointing to it deleted).

        Returns ``True`` if a registration was found and decremented,
        ``False`` if nothing was registered under *name*.
        """
        with self._lock:
            existing = self._managed_components.get(name)
            if existing is None:
                return False
            self._decrement_module_refcount(existing)
            return True

    def unload_component(self, name: str) -> bool:
        """Symmetric counterpart to :meth:`load_component`.

        With refcount-based lifecycle management, this is equivalent to
        :meth:`unregister_component` — cache eviction is automatic when
        the module's refcount hits 0. Kept as a distinct entry point for
        symmetry with ``load_component``.
        """
        return self.unregister_component(name)

    def register_components(self, source: Any) -> list[str]:
        """Bulk-register every ``nn.Module`` exposed by *source*.

        *source* may be a ``DiffusionPipeline``-like object exposing a
        ``components`` dict, or a plain ``dict[str, nn.Module]``.

        Idempotent per source: if the same *source* is registered twice,
        the second call is a no-op (returns the names from the prior
        registration without incrementing refcounts again). This is what
        makes ``managed(pipe)`` safe to call repeatedly on the same pipe.

        Sources are tracked by ``id(source)``. If the user lets *source*
        go out of scope without calling :meth:`unregister_components`, the
        per-source record sticks around (small leak — there's no weakref
        auto-cleanup yet). Modules registered through that source stay
        with refcount > 0 and won't be released. Best practice: call
        ``unregister_components(source)`` (or ``unload_components``) when
        you're done with a pipeline.

        Returns the list of names that were registered (or, on a repeat
        call, the names from the prior registration). Non-modules in the
        components dict are silently skipped.
        """
        if isinstance(source, dict):
            components = source
        elif hasattr(source, "components") and isinstance(source.components, dict):
            components = source.components
            # Modular pipelines' _execution_device doesn't detect group-offload
            # onload devices; patch it so group_offload/block_pin work on them.
            ensure_modular_compat(source)
            # Pick up per-architecture budgeting facts (denoiser concurrency,
            # role hints) that can't be inferred from the module tree.
            self._adopt_model_profile(source)
        else:
            raise TypeError(
                f"register_components expected a pipeline (with .components) or a dict, got {type(source).__name__}"
            )

        source_id = id(source)
        with self._lock:
            existing_record = self._source_registrations.get(source_id)
            if existing_record is not None:
                return list(existing_record.keys())

            registered: dict[str, Any] = {}
            for name, comp in components.items():
                if isinstance(comp, nn.Module):
                    self.register_component(name, comp)
                    registered[name] = comp
            self._source_registrations[source_id] = registered

            # Auto-cleanup if the source is GC'd before the user calls
            # unregister_components. Dicts and a few other types can't be
            # weakref'd — for those we just skip and rely on explicit
            # unregister.
            try:
                self._source_finalizers[source_id] = weakref.finalize(source, self._on_source_gc, source_id)
                # Keep a weak handle on the source itself too: dropping a
                # component has to clear the pipeline's own attribute, or its
                # reference keeps the weights alive and nothing is freed.
                self._source_refs[source_id] = weakref.ref(source)
            except TypeError:
                logger.debug(
                    "register_components: source %s is not weakref-able; "
                    "auto-cleanup on GC won't fire - caller must call "
                    "unregister_components explicitly.",
                    type(source).__name__,
                )

            return list(registered.keys())

    def _on_source_gc(self, source_id: int) -> None:
        """Finalizer callback fired when a registered source is garbage-collected.

        Performs the same teardown as :meth:`unregister_components` but
        keyed by id directly (the source object no longer exists). May
        run in any thread; acquires the lock before touching state.
        """
        with self._lock:
            record = self._source_registrations.pop(source_id, None)
            self._source_finalizers.pop(source_id, None)
            if record is None:
                return
            cleaned = 0
            for name, module in record.items():
                current = self._managed_components.get(name)
                if current is module:
                    self.unregister_component(name)
                    cleaned += 1
            if cleaned:
                logger.info("Auto-cleanup: source GC released %d component(s)", cleaned)

    def unregister_components(self, source: Any) -> list[str]:
        """Bulk-unregister using the per-source record from
        :meth:`register_components`.

        For each ``(name, module)`` pair this *source* originally
        registered, decrements that module's refcount **if** the slot
        still holds the same instance. Slots that were displaced by a
        later registration from another source are skipped — the
        displacement already decremented our refcount on the old module
        at displacement time.

        Modules shared with other still-registered sources survive
        (refcount > 0); modules unique to this source are fully released
        (refcount → 0 triggers hook cleanup, cache eviction, and slot
        deletion).

        Returns the list of names actually processed (skipping stale
        entries). Empty list if *source* wasn't registered.
        """
        source_id = id(source)
        with self._lock:
            # Detach the GC finalizer (if any) so it doesn't fire later
            # and try to clean up state we just removed.
            finalizer = self._source_finalizers.pop(source_id, None)
            if finalizer is not None:
                finalizer.detach()

            record = self._source_registrations.pop(source_id, None)
            if record is None:
                return []
            processed: list[str] = []
            for name, module in record.items():
                current = self._managed_components.get(name)
                if current is module:
                    self.unregister_component(name)
                    processed.append(name)
                else:
                    logger.debug(
                        "unregister_components: skipping %r - slot was displaced by another source",
                        name,
                    )
            return processed

    def unload_components(self, source: Any) -> list[str]:
        """Symmetric counterpart to bulk loading.

        Equivalent to :meth:`unregister_components` under refcount-based
        cleanup (cache eviction happens automatically when a module's
        refcount hits 0). Kept for symmetry with the load/unload naming
        of the singular methods.
        """
        return self.unregister_components(source)

    def get_component(self, name: str) -> Any | None:
        """Retrieve a managed component by name."""
        with self._lock:
            return self._managed_components.get(name)

    @property
    def component_names(self) -> list[str]:
        """Return names of all registered components."""
        with self._lock:
            return list(self._managed_components.keys())

    # ------------------------------------------------------------------
    # Hash-keyed cache
    # ------------------------------------------------------------------

    @staticmethod
    def component_hash(identifier: str) -> str:
        """Deterministic 16-char hex hash for cache keys."""
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]

    def get_cached(self, hash_key: str) -> Any | None:
        with self._lock:
            return self._component_cache.get(hash_key)

    def set_cached(self, hash_key: str, obj: Any) -> None:
        with self._lock:
            self._component_cache[hash_key] = obj

    def load_component(
        self,
        name: str,
        identifier: str,
        factory: Callable[[], nn.Module],
    ) -> nn.Module:
        """Load a component, reusing a cached instance keyed by *identifier*.

        On first call with a given *identifier*, invokes ``factory()`` to
        produce the module, caches it under ``component_hash(identifier)``,
        and registers it under *name*. On subsequent calls with the same
        identifier the cached module is reused — the factory is **not**
        invoked. The cached module is registered under whatever *name*
        the caller passes, so the same module can end up aliased under
        multiple names (which the registry handles correctly via id-dedup).

        This is the right tool for sharing heavy components across several
        pipelines (e.g. a T5 text encoder used by both a base and a refiner
        pipeline) when you don't have an existing pipeline to share *from*.
        When you *do* have one, just pass its components into the second
        pipeline's ``from_pretrained`` kwargs — diffusers-mm's identity
        dedup handles that case without any new API.

        The factory is called **outside** the manager's lock (loads can be
        slow). If two threads race for the same uncached identifier, both
        will run the factory; the first cache write wins and the second
        thread will use the winning module (the loser's module is
        discarded — wasteful but not incorrect).
        """
        cache_key = self.component_hash(identifier)

        with self._lock:
            cached = self._component_cache.get(cache_key)

        if cached is not None and isinstance(cached, nn.Module):
            self.register_component(name, cached)
            logger.info("load_component: cache hit for identifier %r -> %r", identifier, name)
            return cached

        module = factory()
        if not isinstance(module, nn.Module):
            raise TypeError(f"load_component factory must return an nn.Module, got {type(module).__name__}")

        with self._lock:
            winner = self._component_cache.get(cache_key)
            if winner is not None and isinstance(winner, nn.Module):
                module = winner
                logger.info("load_component: lost cache race for identifier %r, using winner", identifier)
            else:
                self._component_cache[cache_key] = module
                logger.info("load_component: cache miss for identifier %r, loaded -> %r", identifier, name)

        self.register_component(name, module)
        return module

    # ------------------------------------------------------------------
    # Device / dtype scope
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def device_scope(self, *, device: torch.device | str, dtype: torch.dtype | None = None) -> Generator[None]:
        """Context manager that sets scoped device/dtype via context vars."""
        if isinstance(device, str):
            device = torch.device(device)
        dev_token = _SCOPED_DEVICE.set(device)
        dtype_token = _SCOPED_DTYPE.set(dtype)
        try:
            yield
        finally:
            _SCOPED_DEVICE.reset(dev_token)
            _SCOPED_DTYPE.reset(dtype_token)

    # ------------------------------------------------------------------
    # Offload strategy resolution & application
    # ------------------------------------------------------------------

    def _detect_available_ram_gb(self) -> tuple[float, float]:
        """Return ``(available_gb, total_gb)`` of system RAM.

        On Linux/macOS: returns ``psutil.virtual_memory().available`` —
        what's actually free for new allocations right now (free +
        reclaimable cache).

        On Windows: applies ComfyUI-style adjusted accounting because
        ``psutil.available`` is too conservative there. WDDM inflates the
        system-wide commit charge by every VRAM allocation as a worst-
        case page-out reserve, even though that "committed" memory isn't
        actually competing for physical RAM. The adjusted figure is:

            adjusted = physical_total - (committed - vram_in_use)

        and we return ``max(psutil.available, adjusted)`` so the manager's
        downstream decisions (strategy resolution, low_cpu_mem auto-tune,
        block_pin tier-4 RAM-absorb check) operate on a realistic number.
        Borrowed from ``comfy/windows.py``; see ``_windows.py`` for the
        ctypes wrapping.

        Returns ``(0.0, 0.0)`` if psutil can't be queried at all. On
        Windows, if the PSAPI call fails we silently fall back to
        psutil's value — the warning is logged inside ``_windows.py``.
        """
        try:
            import psutil

            vm = psutil.virtual_memory()
            psutil_avail_gb = vm.available / (1024**3)
            total_gb = vm.total / (1024**3)
        except Exception:
            return 0.0, 0.0

        if sys.platform != "win32":
            return psutil_avail_gb, total_gb

        from diffusers_mm._windows import query_performance_info_bytes

        result = query_performance_info_bytes()
        if result is None:
            return psutil_avail_gb, total_gb

        committed_bytes, physical_total_bytes = result
        vram_in_use_bytes = self._total_vram_in_use_bytes()
        adjusted_bytes = physical_total_bytes - (committed_bytes - vram_in_use_bytes)
        adjusted_gb = adjusted_bytes / (1024**3)
        # Use the more generous of psutil and the adjusted figure. On a
        # healthy system the two agree closely; the adjustment only
        # matters when VRAM is heavily allocated and WDDM has inflated
        # the commit charge.
        return max(psutil_avail_gb, adjusted_gb), total_gb

    def _total_vram_in_use_bytes(self) -> int:
        """Return total VRAM currently allocated across all CUDA devices.

        Used by the Windows free-RAM adjustment to back out WDDM's
        speculative commit charge for VRAM. Sums ``(total - free)`` from
        ``torch.cuda.mem_get_info`` over every visible CUDA device.
        Returns 0 if CUDA isn't available or any device query fails —
        the caller treats that as "no adjustment", which is safer than
        over-subtracting and reporting more RAM than actually exists.
        """
        try:
            if not torch.cuda.is_available():
                return 0
            total = 0
            for i in range(torch.cuda.device_count()):
                free, dev_total = torch.cuda.mem_get_info(i)
                total += dev_total - free
            return total
        except Exception:
            return 0

    def _resolve_vram_reserve_gb(self, total_gb: float) -> float:
        """VRAM withheld from the free reading, so no budget can reach the card's ceiling.

        Windows serves allocations past the dedicated limit out of host RAM rather than raising, which never
        fails and so never triggers any of the guards here — it just quietly takes the RAM the offloaded weights
        were living in. The only way to not spill is to not get there, hence a reserve rather than a check.

        Zero off Windows: the allocator raises there, which the existing degradation paths already handle.
        """
        if sys.platform != "win32":
            return self.VRAM_RESERVE_GB

        reserve = self.VRAM_RESERVE_WINDOWS_GB
        if total_gb > self.VRAM_RESERVE_LARGE_CARD_THRESHOLD_GB:
            reserve += self.VRAM_RESERVE_WINDOWS_LARGE_CARD_EXTRA_GB
        return reserve

    def _detect_available_vram_gb(self, device: torch.device | str) -> tuple[float, float]:
        """Return ``(available_gb, total_gb)`` of VRAM on *device*.

        Uses ``torch.cuda.mem_get_info`` so the answer reflects whatever
        else is already allocated on the GPU — the CUDA context, other
        PyTorch tensors, other processes sharing the device. Returns
        ``(0.0, 0.0)`` on failure (non-CUDA, driver issue, etc.).

        ``available_gb`` has :meth:`_resolve_vram_reserve_gb` already taken off it, so every budget downstream
        inherits the reserve from one place. ``total_gb`` stays the true card size — the reserve limits what we
        are willing to use, not what exists.
        """
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        except Exception as e:
            logger.warning("auto: VRAM detection failed (%s)", e)
            return 0.0, 0.0

        free_gb = free_bytes / (1024**3)
        total_gb = total_bytes / (1024**3)
        return max(0.0, free_gb - self._resolve_vram_reserve_gb(total_gb)), total_gb

    def _effective_free_vram_gb(self, device: torch.device | str) -> float:
        """Free VRAM the *workload* can actually use, in GB.

        ``mem_get_info`` reports the driver's free pages, which counts PyTorch's
        caching-allocator reserved-but-unallocated pool as "used" even though
        PyTorch will hand that memory back out without touching the driver. Once
        the allocator's reserved pool grows over the first runs, driver-free
        drops and the block_pin eviction check fires spuriously, thrashing the
        pinned blocks CPU<->GPU every neighbor forward. Adding the reclaimable
        pool (``reserved - allocated``) back gives the honest "how much can the
        next op grab" figure and keeps eviction from triggering on warmed-up runs.
        """
        free_gb, _ = self._detect_available_vram_gb(device)
        try:
            reserved = torch.cuda.memory_reserved(device) / (1024**3)
            allocated = torch.cuda.memory_allocated(device) / (1024**3)
            reclaimable = max(0.0, reserved - allocated)
        except Exception:
            reclaimable = 0.0
        return free_gb + reclaimable

    def _estimate_components_size_gb(self) -> tuple[float, float]:
        """Return ``(total_size_gb, max_component_size_gb)`` of registered components.

        Sums param + buffer bytes for every registered ``nn.Module``,
        deduplicating by ``id(module)`` so aliased registrations count once.
        Returns ``(0.0, 0.0)`` if no components are registered (or the
        traversal fails for every component).
        """
        with self._lock:
            components = list(self._managed_components.values())
        seen_ids: set[int] = set()
        total_bytes = 0
        max_bytes = 0
        for mod in components:
            if id(mod) in seen_ids:
                continue
            seen_ids.add(id(mod))
            try:
                size = sum(p.numel() * p.element_size() for p in mod.parameters())
                size += sum(b.numel() * b.element_size() for b in mod.buffers())
            except Exception:
                continue
            total_bytes += size
            if size > max_bytes:
                max_bytes = size
        return total_bytes / (1024**3), max_bytes / (1024**3)

    @property
    def denoiser_concurrency(self) -> str:
        """How to budget multiple denoisers: ``"co_resident"`` or ``"sequential"``.

        Resolution order, most specific first:

        1. An explicit value passed to the constructor / ``managed()`` — the
           caller knows their pipeline better than any table.
        2. The :class:`~diffusers_mm.model_profiles.ModelProfile` of the
           registered pipeline architecture, if it is a recognised one.
        3. ``"co_resident"`` — the safe default. Wrong-way ``co_resident`` on a
           sequential pipeline merely over-reserves; wrong-way ``sequential`` on
           a co-resident one under-budgets and spills.
        """
        if self._denoiser_concurrency is not None:
            return self._denoiser_concurrency
        if self._model_profile is not None and self._model_profile.denoiser_concurrency is not None:
            return self._model_profile.denoiser_concurrency
        return "co_resident"

    @denoiser_concurrency.setter
    def denoiser_concurrency(self, value: str | None) -> None:
        if value is not None and value not in DENOISER_CONCURRENCY_MODES:
            raise ValueError(f"denoiser_concurrency must be one of {DENOISER_CONCURRENCY_MODES}, got {value!r}")
        self._denoiser_concurrency = value

    def _adopt_model_profile(self, source: Any) -> None:
        """Look up and record the :class:`ModelProfile` for a registered *source*.

        Called from ``register_components``. The first recognised architecture
        wins — with a shared manager across several pipelines they are normally
        the same family, and a later differing profile would silently re-budget
        components already placed under the first.
        """
        profile = get_model_profile(source)
        if profile is None:
            return
        with self._lock:
            if self._model_profile is profile:
                return
            if self._model_profile is not None:
                logger.debug(
                    "model profile for %s ignored - already using one from an earlier source",
                    type(source).__name__,
                )
                return
            self._model_profile = profile
        logger.info(
            "model profile: %s -> denoiser_concurrency=%s%s (%s)",
            type(source).__name__,
            profile.denoiser_concurrency or "unset",
            f", role overrides {dict(profile.roles)}" if profile.roles else "",
            profile.note or "no note",
        )

    def classify_components(self) -> list[ComponentInfo]:
        """Classify registered components by role (denoiser / text_encoder / vae / other).

        Detects how many denoisers and text encoders a pipeline has and their
        sizes — the inputs the size-aware resolver and block_pin budgeting need
        for multi-DiT / multi-text-encoder pipelines. See ``inventory.py``.

        Role overrides from the registered pipeline's
        :class:`~diffusers_mm.model_profiles.ModelProfile` win over the
        name/structure heuristics.
        """
        with self._lock:
            components = dict(self._managed_components)
            overrides = self._model_profile.roles if self._model_profile is not None else None
        return build_inventory(components, self.AUTO_BLOCK_PIN_MIN_BLOCKS, role_overrides=overrides)

    def _concurrent_working_set_gb(self) -> float:
        """Peak concurrently-resident weight footprint (GiB), by role.

        - Denoisers: summed if ``denoiser_concurrency == "co_resident"`` (both
          DiTs live every step), else the largest single one ("sequential").
        - Text encoders: largest single one (they run sequentially pre-denoise).
        - Other (VAE, etc.): largest single one.

        The peak is the max across roles. Replaces the old "largest single
        component" figure in the ``model_offload`` tier, which under-budgeted
        co-resident dual-DiT pipelines and let them spill to RAM.
        """
        inventory = self.classify_components()
        denoisers = [c.size_gb for c in inventory if c.role == "denoiser"]
        text_encoders = [c.size_gb for c in inventory if c.role == "text_encoder"]
        others = [c.size_gb for c in inventory if c.role in ("vae", "other")]

        if self.denoiser_concurrency == "co_resident":
            denoiser_ws = sum(denoisers)
        else:
            denoiser_ws = max(denoisers, default=0.0)
        return max(denoiser_ws, max(text_encoders, default=0.0), max(others, default=0.0), 0.0)

    def _coresident_denoiser_count(self) -> int:
        """Number of denoisers that must be resident together every denoise step.

        ``model_offload``'s accelerate chain keeps at most **one** chained
        component on the GPU at a time — each component's forward offloads the
        previous. So it structurally cannot co-reside two denoisers that both
        run every step (e.g. Ideogram4 True-CFG's conditional +
        unconditional transformers): it would bulk-swap a multi-GB DiT
        CPU↔GPU on *every* step. Returns the denoiser count only under
        ``denoiser_concurrency == "co_resident"``; ``"sequential"`` denoisers
        (e.g. Wan2.2 high/low-noise experts split by timestep) are swapped one
        at a time, which model_offload handles fine, so this returns 0 there.
        """
        if self.denoiser_concurrency != "co_resident":
            return 0
        return sum(1 for c in self.classify_components() if c.role == "denoiser")

    def resolve_offload_strategy(self, device: torch.device | str) -> str:
        """Resolve ``"auto"`` to a concrete strategy based on hardware + workload.

        Uses **available** VRAM and RAM at decision time (not total) so
        the answer reflects whatever else is on the system or GPU when
        ``managed()`` is called: another process holding GPU memory, the
        CUDA context overhead, the pipeline weights already mmap'd into
        the page cache, etc. Component sizes come from registered
        ``nn.Module`` parameters/buffers.

        Decision rule:

        - If pipeline weights + the workload-aware working set ≤ available VRAM
          → ``no_offload`` (everything fits on GPU with activation headroom).
        - Else if block_pin would pin the *entire* largest component on the GPU
          (``_block_pin_would_fully_pin_largest``) → ``block_pin``. Same VRAM
          peak as model_offload, but the transformer stays resident across runs
          instead of being re-cycled each generation — faster for repeated use.
        - Else if ≥2 denoisers are co-resident every step (True-CFG's
          conditional + unconditional DiTs, ``denoiser_concurrency ==
          "co_resident"``) → ``block_pin`` (or ``group_offload`` if no block
          list). ``model_offload`` is skipped here: its chain holds only one
          component on the GPU at a time, so it can't co-reside them and would
          swap a multi-GB DiT CPU↔GPU every step.
        - Else if concurrent working set × ``AUTO_MODEL_OFFLOAD_FACTOR`` ≤
          available VRAM → ``model_offload`` (swap the largest on/off the GPU;
          chosen when block_pin could only partially pin it).
        - Else if the largest component has a usable repeated-block list →
          ``block_pin`` (partial pin + stream the overflow).
        - Otherwise → ``group_offload`` (leaf-level streaming).

        If no components are registered yet, falls back to a tier table on
        available VRAM: ``≥ 20 GB`` → no_offload; ``≥ 12 GB`` →
        model_offload; else group_offload.

        A warning is logged if pipeline weights exceed available RAM ×
        ``AUTO_RAM_HEADROOM`` — that workload won't fit on host memory.
        """
        strategy = self.offload_strategy
        if strategy != "auto":
            return strategy

        device = torch.device(device) if isinstance(device, str) else device
        if device.type != "cuda":
            return "group_offload"

        vram_avail_gb, vram_total_gb = self._detect_available_vram_gb(device)
        if vram_avail_gb <= 0:
            return "group_offload"

        ram_avail_gb, ram_total_gb = self._detect_available_ram_gb()
        weights_gb, max_component_gb = self._estimate_components_size_gb()
        # Peak concurrently-resident weights (sums co-resident denoisers), which
        # is what model_offload actually holds on the GPU — not the single
        # largest component. Falls back to max_component if classification finds
        # nothing (e.g. a single-component pipeline).
        concurrent_ws_gb = max(self._concurrent_working_set_gb(), max_component_gb)
        # model_offload can't co-reside ≥2 denoisers that run every step (its
        # chain holds one component at a time); picking it there thrashes a DiT
        # CPU↔GPU per step. Count them so the model_offload tier can be skipped.
        coresident_denoisers = self._coresident_denoiser_count()

        if weights_gb == 0:
            # No components registered yet — fall back to a tier table on
            # *available* VRAM (still better than nothing).
            if vram_avail_gb >= 20:
                chosen = "no_offload"
            elif vram_avail_gb >= 12:
                chosen = "model_offload"
            else:
                chosen = "group_offload"
            logger.info(
                "auto: vram=%.1f / %.1f GB (no components yet) -> %s",
                vram_avail_gb,
                vram_total_gb,
                chosen,
            )
            return chosen

        if ram_avail_gb > 0 and weights_gb > ram_avail_gb * self.AUTO_RAM_HEADROOM:
            logger.warning(
                "auto: pipeline weights (%.1f GB) likely exceed available RAM (%.1f GB). "
                "Loading and offloading may fail.",
                weights_gb,
                ram_avail_gb,
            )

        # no_offload must leave room for the denoise activations too, not
        # just the resident weights — a long/large video needs several GB of
        # working set on top. The additive reserve (weights + working_set) is
        # workload-aware via set_block_pin_workload, replacing the old flat
        # ``weights × AUTO_NO_OFFLOAD_FACTOR`` multiplier (retained as a
        # deprecated ctor arg / constant for backward compatibility only).
        if weights_gb + self._resolve_working_set_gb() <= vram_avail_gb:
            chosen = "no_offload"
        elif self._block_pin_would_fully_pin_largest(device):
            # The largest component (typically the transformer) fits entirely as
            # pinned-resident blocks. block_pin then has the same VRAM peak as
            # model_offload but keeps the transformer on the GPU across runs
            # instead of bulk-cycling it every generation — strictly faster for
            # repeated inference. Prefer it over model_offload here.
            chosen = "block_pin"
        elif coresident_denoisers >= 2:
            # ≥2 denoisers run every step (True-CFG conditional + unconditional).
            # model_offload's chain can only hold one component resident at a
            # time, so it cannot co-reside them — it would bulk-swap a multi-GB
            # DiT CPU↔GPU on every step. Skip the model_offload tier entirely and
            # fall through to block_pin (pin what fits, stream the rest) or
            # group_offload (leaf-level streaming), both of which keep the DiT
            # weights resident/streamed without a per-step full-component swap.
            chosen = "block_pin" if self._largest_component_has_block_list() else "group_offload"
        elif concurrent_ws_gb * self.AUTO_MODEL_OFFLOAD_FACTOR <= vram_avail_gb:
            # The concurrent working set (all co-resident denoisers) fits on the
            # GPU but block_pin couldn't pin it all (overflow would stream).
            # Cycle it rather than risk a partial-pin that under-budgets
            # activations on video. Using the summed working set here (not the
            # single largest) stops dual-DiT True-CFG pipelines from picking
            # model_offload and then spilling both DiTs + activations to RAM.
            chosen = "model_offload"
        elif self._largest_component_has_block_list():
            # Largest component won't fit under model_offload, but it has a long
            # enough repeated-block list to beat plain leaf-level streaming: pin
            # as many blocks as VRAM allows, stream the rest. Components without
            # a block list fall back to group_offload at apply time.
            chosen = "block_pin"
        else:
            chosen = "group_offload"

        logger.info(
            "auto: vram=%.1f / %.1f GB, ram=%.1f / %.1f GB, pipeline=%.1f GB "
            "(largest %.1f GB, concurrent working set %.1f GB, "
            "denoiser_concurrency=%s, co-resident denoisers=%d) -> %s",
            vram_avail_gb,
            vram_total_gb,
            ram_avail_gb,
            ram_total_gb,
            weights_gb,
            max_component_gb,
            concurrent_ws_gb,
            self.denoiser_concurrency,
            coresident_denoisers,
            chosen,
        )

        # Once the strategy is picked, also auto-tune any of its knobs
        # we can sensibly derive from the same hardware picture. The user
        # opted into "auto" → they want the manager to manage all of it.
        # block_pin uses the same group_offload kwargs for its overflow
        # streaming, so the same RAM-headroom heuristic applies.
        if chosen in ("group_offload", "block_pin"):
            self._auto_tune_group_offload(weights_gb, ram_avail_gb)

        return chosen

    def _auto_tune_group_offload(self, weights_gb: float, ram_gb: float) -> None:
        """Auto-tune ``group_offload`` knobs based on RAM headroom.

        *ram_gb* is the **available** RAM at decision time (not total),
        so the budget reflects whatever else the system is currently
        using. Flips ``low_cpu_mem_usage`` off when available RAM is
        enough to absorb a full pinned copy of the pipeline weights plus
        a fixed headroom for OS / activations / transient buffers:
        ``low_cpu_mem=False`` if ``ram_gb >= weights_gb +
        AUTO_LOW_CPU_MEM_RAM_HEADROOM_GB``.

        We don't budget for the *original* weights staying resident
        because modern safetensors mmaps them — pages get evicted as
        needed. The pinned copy is the dominant memory cost.

        Mutates instance state — only called when ``auto`` resolved to
        ``group_offload``, so the user has already opted into the manager
        making this kind of decision.
        """
        if ram_gb <= 0 or weights_gb <= 0:
            return  # No size info — leave defaults alone.

        required_gb = weights_gb + self.AUTO_LOW_CPU_MEM_RAM_HEADROOM_GB
        new_low_cpu_mem = ram_gb < required_gb

        with self._lock:
            old = self._group_offload_low_cpu_mem
            self._group_offload_low_cpu_mem = new_low_cpu_mem

        if old != new_low_cpu_mem:
            slack_gb = ram_gb - required_gb
            logger.info(
                "auto: tuned group_offload low_cpu_mem_usage=%s "
                "(ram=%.1f GB available, required=%.1f GB = weights %.1f + headroom %.1f, slack=%+.1f GB)",
                new_low_cpu_mem,
                ram_gb,
                required_gb,
                weights_gb,
                self.AUTO_LOW_CPU_MEM_RAM_HEADROOM_GB,
                slack_gb,
            )

    # ------------------------------------------------------------------
    # Block-pin (selective offload) — public override + auto budget
    # ------------------------------------------------------------------

    # Cap on how many times spill-aware recalibration will evict-and-repin in a
    # session. Eviction is monotonic (only reduces pins) and converges in 1–2
    # rounds; the cap is a backstop against oscillation.
    _BLOCK_PIN_MAX_SPILL_RECALIBRATIONS = 3

    def _maybe_recalibrate_block_pin_spill(self, device: torch.device | str) -> None:
        """Evict pinned blocks if the last generation oversubscribed VRAM.

        Called after a managed ``pipe(...)`` under ``block_pin``. If the caching
        allocator reserved more than the card's total VRAM (minus a margin), the
        workload spilled into shared/host memory (Windows sysmem fallback) — slow
        and OOM-prone. We free that overage by unpinning blocks (which then
        stream instead), landing the pin count just under the ceiling for the
        real activation footprint. Self-tunes to any resolution/model/card,
        replacing the fragile static working-set reserve as the spill guard.
        """
        if not self._block_pin_spill_aware or self._applied_strategy != "block_pin":
            return
        # VRAM oversubscription (reserved > total, backed by shared host memory) is
        # a Windows sysmem-fallback behaviour. On Linux the allocator hard-OOMs
        # instead, and the auto budget (which this library was tuned on) already
        # fits — so recalibration is a no-op there.
        if sys.platform != "win32":
            return
        if not torch.cuda.is_available():
            return
        dev = torch.device(device) if isinstance(device, str) else device
        if dev.type != "cuda":
            return
        if self._block_pin_spill_recalibrations >= self._BLOCK_PIN_MAX_SPILL_RECALIBRATIONS:
            return

        try:
            total_gb = torch.cuda.get_device_properties(dev).total_memory / (1024**3)
            reserved_gb = torch.cuda.memory_reserved(dev) / (1024**3)
        except Exception:
            return

        target_gb = total_gb - self._block_pin_spill_margin_gb
        if reserved_gb <= target_gb:
            return  # No spill — nothing to do.

        with self._lock:
            pinned = [(name, st) for name, st in self._block_pin_states.items() if st.n_pinned > 0]
        if not pinned:
            logger.warning(
                "block_pin: reserved %.1f GB > %.1f GB target but no pinned blocks to evict "
                "(spill is from activations alone). Consider a lower resolution or group_offload.",
                reserved_gb,
                target_gb,
            )
            self._block_pin_spill_recalibrations += 1  # don't retry a lost cause every call
            return

        # Free the overage plus a little slack so we land under the ceiling.
        need_gb = (reserved_gb - target_gb) + self._block_pin_spill_margin_gb
        evicted_total, freed_gb = self._unpin_to_free_gb(need_gb, dev, reason="spill")
        if evicted_total == 0:
            return

        logger.info(
            "block_pin spill-aware recalibration #%d: reserved %.1f GB > %.1f GB target, "
            "unpinned ~%d blocks (~%.1f GB).",
            self._block_pin_spill_recalibrations + 1,
            reserved_gb,
            target_gb,
            evicted_total,
            freed_gb,
        )
        self._block_pin_spill_recalibrations += 1

    def _unpin_to_free_gb(self, need_gb: float, device: torch.device, *, reason: str) -> tuple[int, float]:
        """Unpin pinned blocks until ~*need_gb* of VRAM has been freed.

        Shared by the Windows spill recalibration and the workload probe.
        Walks pinned components most-pinned-first and calls
        :func:`unpin_blocks` — a targeted, mid-generation-safe conversion of
        pinned blocks into streamed ones (it only touches the overflow block
        submodules, never the top-level component's own hook dict, so it is
        safe from inside a hook on that component).

        The calibrated count is persisted into ``_block_pin_counts`` so a later
        full re-apply reproduces it. Releases the caching allocator's pool
        afterwards, because the freed VRAM is only useful to the workload once
        it is back with the driver.

        Returns ``(blocks_unpinned, gb_freed)``.
        """
        with self._lock:
            pinned = [(name, st) for name, st in self._block_pin_states.items() if st.n_pinned > 0]
        if not pinned:
            return 0, 0.0
        pinned.sort(key=lambda item: item[1].n_pinned, reverse=True)

        offload_kwargs = self._group_offload_kwargs(device)
        freed_gb = 0.0
        unpinned_total = 0
        for name, st in pinned:
            if freed_gb >= need_gb:
                break
            blocks = getattr(st.component, st.block_attr, None)
            if blocks is None or len(blocks) == 0:
                continue
            per_block_gb = per_block_size_bytes(blocks) / (1024**3)
            if per_block_gb <= 0:
                continue
            k = math.ceil((need_gb - freed_gb) / per_block_gb)
            old_n = st.n_pinned
            actually = unpin_blocks(st, k, offload_kwargs)
            if actually == 0:
                continue
            # Persist the calibrated count so any later full re-apply reproduces it.
            with self._lock:
                self._block_pin_counts[name] = st.n_pinned
            freed_gb += actually * per_block_gb
            unpinned_total += actually
            logger.info("block_pin %s: %s %d -> %d pinned blocks", reason, name, old_n, st.n_pinned)

        if unpinned_total:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return unpinned_total, freed_gb

    def _install_spill_calibration_hook(self, device: torch.device | str) -> None:
        """Register forward hooks that recalibrate after the first denoise step.

        The activation peak is reached on the first denoising step (every step
        runs the same-shaped forward), so we don't need a whole generation to
        know whether the pin count spills — one full step is enough. This is
        pipeline-agnostic (a module forward hook fires on standard *and* modular
        pipelines, unlike ``callback_on_step_end`` which modular lacks).

        Windows-only, and a no-op unless ``block_pin`` is active with pinned
        denoisers. Fires the (targeted, mid-generation-safe) recalibration once,
        after each pinned denoiser has run one forward. The end-of-call check
        remains as a safety net for any residual spill.
        """
        if not self._block_pin_spill_aware or sys.platform != "win32":
            return
        if self._applied_strategy != "block_pin":
            return

        with self._lock:
            denoisers = [st.component for st in self._block_pin_states.values() if st.n_pinned > 0]
            # Remove any stale calibration hooks from a prior apply.
            for handle in self._spill_calib_handles:
                handle.remove()
            self._spill_calib_handles = []
        if not denoisers:
            return

        n_expected = len(denoisers)  # ~one forward per pinned denoiser per step
        fired = {"count": 0, "done": False}

        def _calib_hook(module, args, output):
            if fired["done"]:
                return
            fired["count"] += 1
            if fired["count"] < n_expected:
                return
            fired["done"] = True
            # Do NOT remove hooks here — mutating the module's hook dict while it
            # is being iterated (we're inside a forward hook) is unsafe. The
            # ``done`` flag makes further fires no-ops; handles are cleared on the
            # next apply / strategy transition. The recalibration itself only
            # touches overflow block submodules, so it is safe from here.
            try:
                self._maybe_recalibrate_block_pin_spill(device)
            except Exception as e:  # never let calibration break a generation
                logger.warning("block_pin step-1 spill calibration failed: %s", e)

        handles = [comp.register_forward_hook(_calib_hook) for comp in denoisers]
        with self._lock:
            self._spill_calib_handles = handles

    # A denoiser input tensor has to carry at least this many tokens before the
    # probe treats it as the packed sequence. Filters out conditioning inputs
    # (pooled embeds, short text embeds) that share the (B, S, C) layout.
    _BLOCK_PIN_PROBE_MIN_SEQ_LEN = 256

    @classmethod
    def _infer_workload_from_inputs(cls, args: Any, kwargs: Any) -> tuple[int, int] | None:
        """Infer ``(seq_len, batch)`` from a denoiser forward's inputs, or ``None``.

        Looks for the token-sequence hidden state every modern DiT takes as
        ``(batch, seq_len, channels)`` and returns the largest such tensor's
        geometry — that is the sequence the activations scale with. Deliberately
        narrow:

        - ``ndim == 3`` only. UNet-style ``(B, C, H, W)`` / video ``(B, C, F, H,
          W)`` inputs are skipped: their token count depends on model-specific
          patchification we can't read off the tensor, and a wrong guess is worse
          than no guess.
        - ``seq_len`` must clear :attr:`_BLOCK_PIN_PROBE_MIN_SEQ_LEN` and exceed
          the channel dim, which rejects text/pooled conditioning tensors.

        Returning ``None`` leaves the existing budget untouched, so a model whose
        inputs we can't read simply keeps today's behaviour.
        """
        candidates: list[Any] = list(args or ())
        if isinstance(kwargs, dict):
            candidates += list(kwargs.values())

        best: tuple[int, int] | None = None
        best_tokens = 0
        for t in candidates:
            if not isinstance(t, torch.Tensor) or t.ndim != 3 or not t.is_floating_point():
                continue
            batch, seq, channels = (int(d) for d in t.shape)
            if seq < cls._BLOCK_PIN_PROBE_MIN_SEQ_LEN or seq <= channels:
                continue
            if batch * seq > best_tokens:
                best_tokens = batch * seq
                best = (seq, batch)
        return best

    def _recalibrate_for_observed_workload(self, args: Any, kwargs: Any, device: torch.device) -> None:
        """Re-budget the pin count against the *actual* denoise workload.

        Runs from a pre-forward hook on a block-pinned denoiser, i.e. after the
        text encoder has had its turn but before a single activation of the
        denoise step has been allocated — the one moment where both the true
        sequence length and the true free VRAM are known.

        Strictly a safety valve: it only ever *lowers* the pin count, and only
        when the observed workload needs a bigger working set than whatever was
        budgeted. A workload smaller than the recorded one is left alone so an
        explicit :meth:`set_block_pin_workload` is never silently downgraded.
        """
        observed = self._infer_workload_from_inputs(args, kwargs)
        if observed is None:
            return
        seq_len, batch = observed

        with self._lock:
            prev_seq = self._block_pin_seq_len
            prev_batch = self._block_pin_batch
            scale = self._block_pin_activation_scale
        if (seq_len, batch) == (prev_seq, prev_batch):
            return  # Already budgeted for exactly this workload.

        current_need_gb = self._resolve_pin_budget_working_set_gb()
        observed_need_gb = self._working_set_for_workload_gb(seq_len, batch, pool_aware=True)
        if observed_need_gb <= current_need_gb:
            # The budget already reserves at least this much — don't downgrade it.
            return

        self.set_block_pin_workload(seq_len, batch, activation_scale=scale)
        free_gb = self._effective_free_vram_gb(device)
        logger.info(
            "block_pin workload probe: observed seq_len=%d batch=%d (budgeted for %d/%d) -> "
            "working set %.2f -> %.2f GiB, effective free %.2f GiB",
            seq_len,
            batch,
            prev_seq,
            prev_batch,
            current_need_gb,
            observed_need_gb,
            free_gb,
        )
        if free_gb <= 0 or free_gb >= observed_need_gb:
            return  # Detection failed, or the pin count already leaves enough room.

        unpinned, freed_gb = self._unpin_to_free_gb(observed_need_gb - free_gb, device, reason="workload probe")
        if unpinned == 0:
            logger.warning(
                "block_pin workload probe: seq_len=%d needs ~%.1f GiB working set but only %.1f GiB is free "
                "and there are no pinned blocks left to unpin. This workload may OOM - "
                "consider a smaller resolution/duration or strategy='group_offload'.",
                seq_len,
                observed_need_gb,
                free_gb,
            )
            return
        logger.info(
            "block_pin workload probe: unpinned %d block(s) (~%.1f GiB) -> effective free %.2f GiB",
            unpinned,
            freed_gb,
            self._effective_free_vram_gb(device),
        )

    # Minimum blocks a rebalance must be able to regain before it bothers
    # re-pinning. Effectively "any", and deliberately so: re-pinning costs one
    # host-to-device transfer, while leaving a block streamed costs a transfer on
    # *every* denoise step, so a single block repays itself on the first step.
    # Exposed as a knob for anyone who would rather have zero churn.
    _BLOCK_PIN_REPIN_MIN_BLOCKS = 1

    def _rebalance_block_pin(self, device: torch.device | str, *, reason: str) -> tuple[int, int]:
        """Re-fit every auto-managed component's pin count to the current workload.

        Bidirectional, unlike the forward-time probe and the Windows spill
        recalibration, which can only ever shed blocks. Those are safety valves
        firing mid-run, where growing the pinned set would be reckless; this one
        runs *between* calls, where it is merely a transfer.

        The direction that only this method can supply matters in a long-lived
        process: a big generation shrinks the pin count to fit its activations,
        and without a way back every later small generation inherits that count
        and keeps paying the per-block streaming cost. The ratchet has to turn
        both ways or the first big job silently taxes the rest of the session.

        Budgets against *effective* free VRAM plus what the component's own
        pinned blocks currently hold — i.e. "if I released everything I pinned,
        what would fit?" — so it converges to the same answer whether it is
        growing or shrinking. Components with a caller-set
        :meth:`set_block_pin_count` are left alone entirely, as are evicted
        subsets (those belong to the auto-evict path, which will repin them).

        Returns ``(blocks_pinned, blocks_unpinned)``.
        """
        if self._applied_strategy != "block_pin":
            return 0, 0
        dev = torch.device(device) if isinstance(device, str) else device
        # No device-type gate: on anything without a VRAM reading
        # ``_effective_free_vram_gb`` returns 0.0 and every component is skipped
        # below, which is the same no-op by a shorter route — and keeps the
        # whole path exercisable on CPU in tests.
        with self._lock:
            states = [(n, st) for n, st in self._block_pin_states.items() if n not in self._block_pin_user_counts]
        if not states:
            return 0, 0

        working_set_gb = self._resolve_pin_budget_working_set_gb()
        offload_kwargs = self._group_offload_kwargs(dev)
        pinned_total = unpinned_total = 0

        for name, st in states:
            blocks = getattr(st.component, st.block_attr, None)
            if blocks is None or len(blocks) == 0:
                continue
            per_block_gb = per_block_size_bytes(blocks) / (1024**3)
            if per_block_gb <= 0:
                continue

            # Re-read free VRAM per component: an earlier component's rebalance
            # in this same pass has already moved weights on or off the device.
            free_gb = self._effective_free_vram_gb(dev)
            if free_gb <= 0:
                continue  # Detection failed — leave the existing count alone.
            # How much could be resident during the step: what is free now, plus
            # what this component's own pinned blocks would give back if
            # released, less the activations the step needs and the one block
            # apply_group_offloading keeps prefetched. An evicted subset is
            # already off the device, so it adds nothing back — but it is about
            # to return, which is precisely why it still gets rebalanced here.
            resident_gb = st.n_pinned * per_block_gb if st.resident else 0.0
            capacity_gb = free_gb + resident_gb - working_set_gb - per_block_gb
            target = max(0, min(int(capacity_gb / per_block_gb), len(blocks)))
            # A profiled architecture's call-time workload lands here rather than
            # in the apply-time budget, so this path needs the warning too.
            self._warn_workload_does_not_fit(name, working_set_gb, free_gb + resident_gb - per_block_gb)

            if target < st.n_pinned:
                moved = unpin_blocks(st, st.n_pinned - target, offload_kwargs)
                unpinned_total += moved
            elif target - st.n_pinned >= self._BLOCK_PIN_REPIN_MIN_BLOCKS:
                moved = pin_blocks(st, target - st.n_pinned, dev)
                pinned_total += moved
            else:
                continue
            if moved == 0:
                continue
            with self._lock:
                self._block_pin_counts[name] = st.n_pinned
            logger.info(
                "block_pin %s: %s -> %d/%d pinned blocks (working set %.2f GiB)",
                reason,
                name,
                st.n_pinned,
                len(blocks),
                working_set_gb,
            )

        if unpinned_total:
            # Only useful to the workload once the pages are back with the driver.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return pinned_total, unpinned_total

    def _prepare_block_pin_for_call(self, pipe: Any, kwargs: Any, device: torch.device | str) -> None:
        """Size the block_pin budget from the call's own arguments, before it runs.

        The workload is the one budgeting input that belongs to the *request*
        rather than the model, so ``managed()`` cannot know it when it applies
        the strategy. The architecture's :attr:`ModelProfile.workload_fn` can
        compute it exactly from the height / width / frame count the caller
        just passed, which makes an explicit
        :meth:`set_block_pin_workload` unnecessary for profiled pipelines.

        Runs before the pipeline body, so any resulting change to the pin count
        is in place for the first denoise step. A no-op when the architecture is
        unprofiled or the workload is unchanged; the forward-time probe stays
        installed either way as the backstop for both cases.
        """
        if not self._block_pin_call_workload or self._applied_strategy != "block_pin":
            return
        workload = resolve_call_workload(pipe, kwargs)
        if workload is None:
            return
        seq_len, batch = workload

        with self._lock:
            prev = (self._block_pin_seq_len, self._block_pin_batch)
            scale = self._block_pin_activation_scale
        if (seq_len, batch) == prev:
            return  # Already budgeted for exactly this job.

        # The pin-budget figure, since that is what the rebalance below spends.
        before_gb = self._resolve_pin_budget_working_set_gb()
        self.set_block_pin_workload(seq_len, batch, activation_scale=scale)
        logger.info(
            "block_pin: workload from call args seq_len=%d batch=%d (was %d/%d) -> working set %.2f -> %.2f GiB",
            seq_len,
            batch,
            prev[0],
            prev[1],
            before_gb,
            self._resolve_pin_budget_working_set_gb(),
        )
        self._rebalance_block_pin(device, reason="call workload")

    @staticmethod
    def release_host_cache() -> bool:
        """Return PyTorch's pooled **pinned host** memory to the OS.

        Group offloading with ``low_cpu_mem_usage=False`` pins a full host copy
        of every streamed weight, and that pool outlives the tensors: dropping
        the module and running ``gc.collect()`` frees nothing, because the
        caching host allocator keeps the pages for reuse. This is the only call
        that hands them back.

        Returns ``True`` if the release was attempted, ``False`` on a torch
        build that doesn't expose it (the private API arrived in torch 2.5).
        """
        release = getattr(torch._C, "_host_emptyCache", None)
        if release is None:
            logger.debug("torch._C._host_emptyCache() unavailable; pinned host memory stays pooled")
            return False
        try:
            release()
        except Exception as e:
            logger.warning("Failed to release pinned host memory: %s", e)
            return False
        return True

    def unload_text_encoders(self, *, release_host_cache: bool = True) -> list[str]:
        """Drop every text-encoder component and reclaim its memory.

        Prompt encoding happens once per generation, before the denoise loop,
        after which the text encoder is dead weight for the rest of the call —
        yet its weights stay resident and, under group offload, keep a pinned
        host copy the same size as the weights themselves. On a pipeline whose
        text encoder is the largest component that is the single biggest block of
        reclaimable memory in the process.

        Role-based, so it covers the older multi-encoder pipelines (SDXL's
        ``text_encoder`` + ``text_encoder_2``, SD3's three) with no per-model
        knowledge — see :func:`~diffusers_mm.inventory.classify_role`.

        Destructive by design: the components are unregistered, their offload
        hooks removed, **and the pipeline's own attribute is set to None**,
        because that reference is what keeps the weights alive. A later
        generation needs them back; :meth:`restore_dropped_components` reloads
        them for sources that support it, and raises a clear error for those
        that don't.

        Returns the names actually dropped.
        """
        names = [c.name for c in self.classify_components() if c.role == "text_encoder"]
        if not names:
            return []

        freed_gb = 0.0
        dropped: list[str] = []
        for name in names:
            with self._lock:
                module = self._managed_components.get(name)
            if module is None:
                continue
            freed_gb += module_size_gb(module)
            # Clear the attribute on every source that registered this name, or
            # the pipeline keeps the module alive and nothing is reclaimed.
            with self._lock:
                sources = [(sid, rec) for sid, rec in self._source_registrations.items() if rec.get(name) is module]
            for source_id, _rec in sources:
                ref = self._source_refs.get(source_id)
                source = ref() if ref is not None else None
                if source is None:
                    continue
                try:
                    setattr(source, name, None)
                except Exception as e:
                    logger.warning("Could not clear %s on %s: %s", name, type(source).__name__, e)
                    continue
                with self._lock:
                    self._dropped_components[name] = source_id
            self.unregister_component(name)
            self._purge_component_references(name, module)
            dropped.append(name)

        if not dropped:
            return []

        del module, sources
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        released = self.release_host_cache() if release_host_cache else False
        logger.info(
            "Unloaded text encoder(s) %s (~%.2f GiB of weights); pinned host cache %s.",
            dropped,
            freed_gb,
            "released" if released else "left pooled",
        )
        return dropped

    def _purge_component_references(self, name: str, module: Any) -> None:
        """Drop every *manager-side* strong reference to *module*.

        Unregistering removes it from the active registry, but the manager keeps
        it alive in two other places, and a single surviving reference makes the
        whole unload pointless — the weights stay resident and the pinned host
        pool has nothing free to release:

        * ``_source_registrations``, the per-source record used for idempotent
          registration and precise teardown, maps name → module strongly.
        * ``_block_pin_wrapped_methods``, which holds the component whose
          ``decode``/``encode`` were wrapped for auto-evict. A text encoder is a
          *neighbor* under ``block_pin``, so it lands in that list.
        """
        with self._lock:
            for record in self._source_registrations.values():
                if record.get(name) is module:
                    record.pop(name, None)

            remaining: list[tuple[Any, str, Any]] = []
            for component, method_name, restore_value in self._block_pin_wrapped_methods:
                if component is not module:
                    remaining.append((component, method_name, restore_value))
                    continue
                # Restore before dropping the entry so the module isn't left
                # carrying a wrapper that closes over this manager.
                try:
                    if restore_value is _INSTANCE_ATTR_ABSENT:
                        component.__dict__.pop(method_name, None)
                    else:
                        setattr(component, method_name, restore_value)
                except Exception as e:
                    logger.debug("Could not unwrap %s.%s during unload: %s", name, method_name, e)
            self._block_pin_wrapped_methods = remaining

    def restore_dropped_components(self, source: Any, device: torch.device | str | None = None) -> list[str]:
        """Reload components dropped by :meth:`unload_text_encoders`, if possible.

        Called before a managed generation so a second ``pipe(...)`` works after
        an unload. Modular pipelines can rebuild a component from its
        ``ComponentSpec`` via ``load_components(names=[...])``; a standard
        ``DiffusionPipeline`` has no equivalent, so this raises rather than
        letting diffusers fail later with a confusing ``NoneType`` error.

        Pass *device* so the reloaded weights are re-hooked under the active
        strategy; without it they are registered but left unplaced.

        Returns the names restored (empty when there was nothing to do).
        """
        with self._lock:
            pending = [name for name, sid in self._dropped_components.items() if sid == id(source)]
        if not pending:
            return []

        loader = getattr(source, "load_components", None)
        if not callable(loader):
            raise RuntimeError(
                f"{', '.join(pending)} was unloaded by unload_text_encoders() and "
                f"{type(source).__name__} cannot reload it (no `load_components`). Rebuild the "
                f"pipeline, or disable `unload_text_encoders` if you need more than one generation."
            )
        logger.info("Reloading unloaded component(s) %s before this generation.", pending)
        loader(names=pending)

        # Register the fresh modules directly: ``register_components`` is
        # idempotent per source, so it would return early on a source it has
        # already seen and leave the reloaded weights unmanaged and unhooked.
        restored: list[str] = []
        for name in pending:
            module = getattr(source, name, None)
            if not isinstance(module, nn.Module):
                logger.warning("Reload of %s produced %s, not a module; leaving it unmanaged.", name, type(module))
                continue
            self.register_component(name, module)
            with self._lock:
                record = self._source_registrations.get(id(source))
                if record is not None:
                    record[name] = module
                self._dropped_components.pop(name, None)
            restored.append(name)

        if restored and device is not None:
            # Incremental: only the freshly-registered slots are processed.
            self.apply_offload_strategy(device)
        return restored

    def _install_text_encoder_unload_hook(self, device: torch.device | str) -> None:
        """Drop the text encoders once the first denoiser forward begins.

        That moment is the natural boundary: prompt encoding is complete for the
        whole generation, and not a single denoise activation has been allocated
        yet, so the memory comes back exactly when the step is about to need it.
        Reuses the same pre-forward-hook shape as the workload probe, but is
        installed for every strategy rather than only ``block_pin``.
        """
        if not self._unload_text_encoders:
            return
        with self._lock:
            for handle in self._text_encoder_unload_handles:
                handle.remove()
            self._text_encoder_unload_handles = []
            components = dict(self._managed_components)
        denoisers = [
            components[c.name] for c in self.classify_components() if c.role == "denoiser" and c.name in components
        ]
        if not denoisers:
            return

        fired = {"done": False}

        def _unload_hook(module, args):
            if fired["done"]:
                return
            fired["done"] = True
            # Do NOT remove hooks from inside the hook — the dict is mid-iteration.
            try:
                self.unload_text_encoders()
            except Exception as e:  # never let this break a generation
                logger.warning("Text-encoder unload failed: %s", e)

        handles = [comp.register_forward_pre_hook(_unload_hook) for comp in denoisers]
        with self._lock:
            self._text_encoder_unload_handles = handles

    def _keep_resident_instead_of_offload(self, name: str, mod: Any, device: torch.device | str) -> bool:
        """Move *mod* fully onto *device* instead of group-offloading it, if it must be.

        Returns True when the component was made resident and the caller should
        skip ``apply_group_offloading`` for it.

        Today the only such case is legacy ``torch.nn.utils.weight_norm``, which
        group offloading cannot serve at all — see
        :func:`diffusers_mm.hooks.find_legacy_weight_norm` for the mechanism.
        Streaming those weights produces a CPU/CUDA device mismatch on the first
        forward, so residency is the only working placement; the affected
        components in practice are small audio autoencoders / vocoders.

        Logged at warning level because it silently raises the VRAM floor: the
        caller asked for this component to be offloaded and it won't be.
        """
        wn = find_legacy_weight_norm(mod)
        if wn is None:
            return False
        size_gb = 0.0
        try:
            size_gb = sum(p.numel() * p.element_size() for p in mod.parameters()) / (1024**3)
            size_gb += sum(b.numel() * b.element_size() for b in mod.buffers()) / (1024**3)
        except Exception:
            pass
        try:
            mod.to(device)
        except Exception as e:
            # ``.to()`` walks submodules in order and mutates as it goes, so a
            # failure part-way (OOM, typically) leaves the component split
            # across devices. Put it back on the CPU before returning: the
            # caller's fallback path assumes a uniform placement, and holding
            # half a component on the GPU after an OOM helps nobody.
            logger.warning("Failed to make %s resident despite legacy weight_norm: %s", name, e)
            self._reset_component_to_cpu(name, mod)
            return False
        logger.warning(
            "%s uses legacy torch.nn.utils.weight_norm (at %r), which is incompatible with "
            "group offloading - its recomputed `weight` would stay on the CPU. Keeping it "
            "resident on %s instead (%.2f GiB of VRAM).",
            name,
            wn,
            device,
            size_gb,
        )
        return True

    @staticmethod
    def _reset_component_to_cpu(name: str, mod: Any) -> bool:
        """Best-effort: strip offload hooks from *mod*, move it to CPU, free the VRAM.

        Used on the failure paths where a component was left half-moved. Never
        raises; returns True only if the reset actually went through, since the
        callers' fallbacks are only safe on a uniformly placed component.
        """
        try:
            remove_offload_hooks(mod)
            mod.to("cpu")
        except Exception as e:
            logger.warning("block_pin: could not reset %s to CPU after a failure: %s", name, e)
            return False
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()
        return True

    def _rollback_pin_to_group_offload(self, name: str, mod: Any, offload_kwargs: dict[str, Any]) -> bool:
        """Undo a partially applied block_pin on *mod* and group-offload it instead.

        :func:`~diffusers_mm.block_pin.apply_block_pin` moves the non-block
        parts and then each pinned block onto the GPU *before* hooking the
        overflow blocks, so a failure part-way (OOM on one of those moves, on a
        card that raises instead of spilling) leaves the component split across
        devices with no hooks on the blocks that never got them. The next
        forward then dies on a device mismatch, which reads as a bug in the
        model rather than a budget that did not fit.

        So reset the component and fall back to plain group offload, the same
        degradation block_pin already applies to a component with no usable
        block list. Slower than pinning, but it runs.

        Returns True when the component ends up group-offloaded, so the caller
        can register it as a neighbor.
        """
        from diffusers.hooks.group_offloading import apply_group_offloading

        if not self._reset_component_to_cpu(name, mod):
            return False
        try:
            apply_group_offloading(mod, **offload_kwargs)
        except Exception as e:
            logger.warning("block_pin: group_offload rollback failed for %s: %s", name, e)
            return False
        logger.warning(
            "block_pin: %s could not be pinned, rolled back to group_offload for the whole "
            "component. Expect it to run slower than a successful pin.",
            name,
        )
        return True

    def _install_block_pin_workload_probe(self, device: torch.device | str) -> None:
        """Register a one-shot pre-forward probe on each block-pinned denoiser.

        The apply-time block_pin budget has to guess the activation footprint:
        unless the caller recorded the job with :meth:`set_block_pin_workload`,
        it falls back to :attr:`AUTO_BLOCK_PIN_ACT_FALLBACK_GB` — an image-scale
        figure. A long/large video sequence needs several times that, so the
        budget over-pins and the first denoise step OOMs.

        The probe closes that gap without asking the caller for anything: the
        denoiser's own input tensor states the true sequence length, and a
        *pre*-forward hook reads it before any activation is allocated, so the
        pin count can still be lowered in time. Model-agnostic (no per-pipeline
        geometry maths) and works on standard and modular pipelines alike, since
        it hooks the module rather than the pipeline's callback protocol.

        Complements the Windows spill recalibration, which is reactive
        (it measures oversubscription *after* a step) and can only help on a
        platform that spills instead of OOMing.
        """
        if not self._block_pin_workload_probe or self._applied_strategy != "block_pin":
            return
        if not torch.cuda.is_available():
            return
        dev = torch.device(device) if isinstance(device, str) else device
        if dev.type != "cuda":
            return

        with self._lock:
            pinned = [st.component for st in self._block_pin_states.values() if st.n_pinned > 0]
            for handle in self._workload_probe_handles:
                handle.remove()
            self._workload_probe_handles = []
        if not pinned:
            return

        fired = {"done": False}

        def _probe_hook(module, args, kwargs):
            if fired["done"]:
                return
            fired["done"] = True
            # Do NOT remove hooks from here — we're inside the iteration of this
            # module's pre-hook dict. The ``done`` flag makes later fires
            # no-ops; handles are cleared on the next apply / transition.
            try:
                self._recalibrate_for_observed_workload(args, kwargs, dev)
            except Exception as e:  # never let calibration break a generation
                logger.warning("block_pin workload probe failed: %s", e)

        handles = [comp.register_forward_pre_hook(_probe_hook, with_kwargs=True) for comp in pinned]
        with self._lock:
            self._workload_probe_handles = handles

    def set_block_pin_count(self, component_name: str, count: int) -> None:
        """Override the number of blocks to pin on GPU for *component_name*.

        Used only when the active strategy is ``"block_pin"``. Names without
        an override get an auto-computed value from available VRAM at apply
        time. ``count=0`` is valid — it means "pin nothing for this
        component, just stream it" (effectively per-block ``group_offload``).

        An explicit count also opts the component **out** of per-call
        rebalancing (see :meth:`_rebalance_block_pin`): the manager will not
        raise or lower a number the caller chose. The safety valves still
        apply — a workload that would OOM can still force blocks out — but
        nothing will quietly grow the count back afterwards.
        """
        if int(count) < 0:
            raise ValueError("block_pin count must be >= 0")
        with self._lock:
            self._block_pin_counts[component_name] = int(count)
            self._block_pin_user_counts.add(component_name)

    def set_evict_on_neighbor(self, component_name: str, value: bool | None) -> None:
        """Override the auto-evict decision for a specific neighbor component.

        ``True`` — *always* evict pinned subsets when this component runs
        (forward / decode / encode), regardless of how much VRAM is free.
        Useful when the neighbor has a known large but inconsistent
        activation footprint (e.g. video VAE decode at varying resolutions)
        and you don't want to rely on the runtime check guessing right.

        ``False`` — *never* evict pinned subsets when this component runs.
        Useful for small-activation neighbors like text encoders, where
        the eviction + repin transfer cost is pure overhead. The wrap is
        still installed; it just no-ops.

        ``None`` (default for any unset name) — let the runtime check
        decide based on currently-free VRAM compared to the working-set
        margin reserved at apply time. See
        :meth:`_should_evict_for_neighbor`.
        """
        with self._lock:
            if value is None:
                self._evict_on_neighbor.pop(component_name, None)
            else:
                self._evict_on_neighbor[component_name] = bool(value)

    def set_block_pin_workload(self, seq_len: int, batch: int = 1, *, activation_scale: float = 1.0) -> None:
        """Record the expected denoise workload for block_pin budgeting.

        Makes the block_pin working set (and the ``no_offload`` activation
        reserve) scale with the actual job instead of a flat constant. Call
        it before ``managed()`` / ``apply_offload_strategy`` runs the
        resolver — e.g. once the latent dimensions are known.

        Args:
            seq_len: Denoise sequence length, ``latent_frames × latent_h ×
                latent_w`` (for images ``latent_frames`` is 1). ``0`` clears
                the workload and falls back to ``AUTO_BLOCK_PIN_ACT_FALLBACK_GB``.
            batch: Forward batch size — ``2`` when classifier-free guidance
                doubles the batch, else ``1``.
            activation_scale: ``>= 1.0`` multiplier inflating the base
                activation estimate for LoRAs / conditioning that make the
                forward allocate more than a plain text-to-X pass. See
                :func:`diffusers_mm.offload_defaults.block_pin_activation_scale`.
        """
        with self._lock:
            self._block_pin_seq_len = max(0, int(seq_len))
            self._block_pin_batch = max(1, int(batch))
            self._block_pin_activation_scale = max(1.0, float(activation_scale))

    def _resolve_working_set_headroom_gb(self) -> float:
        """Platform-appropriate safety headroom added on top of the activation estimate.

        Windows uses the higher constant because ``expandable_segments`` is
        Linux-only and the Windows allocator reserves more under the same load.
        """
        if sys.platform == "win32":
            return self.AUTO_BLOCK_PIN_WORKING_SET_WINDOWS_GB
        return self.AUTO_BLOCK_PIN_WORKING_SET_GB

    def _warn_workload_does_not_fit(self, component_name: str, working_set_gb: float, room_gb: float) -> None:
        """Warn when the working set alone leaves no room, so no pin count helps.

        Deduplicated per (component, workload): the budget is recomputed every
        call, the news is only new once.
        """
        if working_set_gb <= room_gb:
            return
        with self._lock:
            seq_len = self._block_pin_seq_len
            key = (component_name, seq_len)
            if key in self._workload_fit_warned:
                return
            self._workload_fit_warned.add(key)
        # Windows absorbs the overflow into shared memory and only gets slow;
        # elsewhere it raises.
        consequence = (
            "Expect the overflow to spill into shared memory and the run to get several times slower"
            if sys.platform == "win32"
            else "Expect a CUDA out-of-memory error"
        )
        logger.warning(
            "block_pin: this workload does not fit %s regardless of pinning - the denoise working "
            "set alone needs ~%.1f GiB but only ~%.1f GiB is available for it (short by ~%.1f GiB%s). "
            "%s. The working set scales with the denoise sequence length, so fewer frames or a "
            "smaller resolution is what reduces it.",
            component_name,
            working_set_gb,
            room_gb,
            working_set_gb - room_gb,
            f", seq_len={seq_len}" if seq_len > 0 else "",
            consequence,
        )

    def _resolve_allocator_inflation(self) -> float:
        """Multiplier on the activation term of the pin budget, per platform."""
        if sys.platform == "win32":
            return self.AUTO_BLOCK_PIN_ALLOCATOR_INFLATION_WINDOWS
        return self.AUTO_BLOCK_PIN_ALLOCATOR_INFLATION

    def _resolve_allocator_pool_overhead_gb(self) -> float:
        """Fixed reserved-pool overhead (GiB) for the pin budget, per platform."""
        if sys.platform == "win32":
            return self.AUTO_BLOCK_PIN_ALLOCATOR_POOL_OVERHEAD_WINDOWS_GB
        return self.AUTO_BLOCK_PIN_ALLOCATOR_POOL_OVERHEAD_GB

    def _activation_fit(self) -> tuple[float, float, bool]:
        """``(intercept_gb, slope_gb_per_ktoken, is_measured)`` for the architecture.

        Resolution order, most specific first: an explicit ctor value → the
        registered pipeline's :class:`ModelProfile` measurement → the class
        default. The spread across architectures is wide enough that the default
        cannot serve them all, so prefer a profile measurement where one exists.
        """
        intercept = self.AUTO_BLOCK_PIN_ACT_INTERCEPT_GB
        slope = self.AUTO_BLOCK_PIN_ACT_SLOPE_GB_PER_KTOKEN
        measured = False
        profile = self._model_profile
        if profile is not None:
            if not self._explicit_act_slope and profile.act_slope_gb_per_ktoken is not None:
                slope = profile.act_slope_gb_per_ktoken
                measured = True
            if not self._explicit_act_intercept and profile.act_intercept_gb is not None:
                intercept = profile.act_intercept_gb
        return intercept, slope, measured

    def _act_safety_factor(self) -> float:
        """Multiplier applied to the activation estimate before platform headroom.

        A **measured** slope needs far less cushion than the generic default:
        most of :attr:`AUTO_BLOCK_PIN_ACT_SAFETY_FACTOR` exists to cover not
        knowing the architecture's real activation cost. Keeping 1.5x on top of a
        measurement double-counts badly enough to invert the fix — on MiniMax-H3
        at 104k tokens it reserves more than the card has and pins zero blocks.
        An explicit ``auto_block_pin_act_safety_factor=`` still wins.
        """
        if self._explicit_act_safety_factor:
            return self.AUTO_BLOCK_PIN_ACT_SAFETY_FACTOR
        return (
            self.AUTO_BLOCK_PIN_ACT_SAFETY_FACTOR_MEASURED
            if self._activation_fit()[2]
            else (self.AUTO_BLOCK_PIN_ACT_SAFETY_FACTOR)
        )

    def _activation_estimate_gb(self, seq_len: int, batch: int) -> float:
        """Estimated transformer activation VRAM (GB) for a denoise forward.

        Linear fit ``intercept + slope × ktokens`` (``ktokens = batch ×
        seq_len / 1000``), scaled by the recorded ``activation_scale``, using the
        registered architecture's measured fit where one is known (see
        :meth:`_activation_fit`). Activations are bf16 regardless of weight
        quantization, so the fit generalises across dtypes. Returns the fixed
        fallback when ``seq_len`` is unknown.
        """
        with self._lock:
            scale = self._block_pin_activation_scale
        if seq_len <= 0:
            return self.AUTO_BLOCK_PIN_ACT_FALLBACK_GB * scale
        ktokens = (batch * seq_len) / 1000.0
        intercept, slope, _ = self._activation_fit()
        return (intercept + slope * ktokens) * scale

    def _working_set_for_workload_gb(self, seq_len: int, batch: int, *, pool_aware: bool = False) -> float:
        """Working-set VRAM (GB) a denoise forward of this shape needs.

        ``activation_estimate × SAFETY_FACTOR + platform_headroom``, plus the
        reserved-pool correction when *pool_aware*.

        Only the pin budget wants ``pool_aware=True``, because pinned blocks are
        what the pool competes with. Eviction must not: it compares against
        :meth:`_effective_free_vram_gb`, which already adds the reclaimable pool
        back on the other side, so correcting both would double-count it and
        re-introduce the warm-up thrash that method prevents. Strategy choice
        must not either: it compares options that run the same forward, so the
        pool cancels.

        Split out from :meth:`_resolve_working_set_gb` so the forward-time probe
        prices an observed workload with the same formula the budget was built
        with.
        """
        act = self._activation_estimate_gb(seq_len, batch) * self._act_safety_factor()
        if pool_aware:
            return (
                act * self._resolve_allocator_inflation()
                + self._resolve_allocator_pool_overhead_gb()
                + self._resolve_working_set_headroom_gb()
            )
        return act + self._resolve_working_set_headroom_gb()

    def _resolve_working_set_gb(self) -> float:
        """Peak *live* working-set VRAM (GB) for the recorded workload.

        The workload recorded via :meth:`set_block_pin_workload` priced by
        :meth:`_working_set_for_workload_gb`. Scales with video size / length
        instead of a flat constant; falls back to a fixed activation estimate
        when the workload is unknown. Serves the eviction threshold (a neighbor
        triggers pinned-block eviction before its onload would OOM), the
        ``no_offload`` activation reserve and the strategy-choice checks.

        For the block_pin **pin budget**, use
        :meth:`_resolve_pin_budget_working_set_gb` instead.
        """
        with self._lock:
            seq_len = self._block_pin_seq_len
            batch = self._block_pin_batch
        return self._working_set_for_workload_gb(seq_len, batch)

    def _resolve_pin_budget_working_set_gb(self) -> float:
        """VRAM (GB) that must stay free for the forward, as a *reserved pool*."""
        with self._lock:
            seq_len = self._block_pin_seq_len
            batch = self._block_pin_batch
        return self._working_set_for_workload_gb(seq_len, batch, pool_aware=True)

    def _compute_block_pin_count(
        self,
        component_name: str,
        component: Any,
        block_attr: str,
        blocks: Any,
        device: torch.device | str,
    ) -> int:
        """Auto-compute the number of blocks to pin for *component*.

        If the user has set an override via ``set_block_pin_count``, that
        wins (clamped to ``[0, len(blocks)]``). Otherwise: take available
        VRAM, subtract the component's non-block size, the working-set
        safety margin, and one per-block worth of VRAM for the streamed-
        in-flight block (``apply_group_offloading(use_stream=True)`` keeps
        the next overflow block prefetched on GPU while the current one
        computes). Divide by per-block size.
        """
        with self._lock:
            override = self._block_pin_counts.get(component_name)
        if override is not None:
            return max(0, min(override, len(blocks)))

        per_block = per_block_size_bytes(blocks)
        if per_block <= 0:
            return 0
        per_block_gb = per_block / (1024**3)

        vram_avail_gb, _ = self._detect_available_vram_gb(device)
        non_block_gb = non_block_size_bytes(component, block_attr) / (1024**3)
        working_set_gb = self._resolve_pin_budget_working_set_gb()

        budget_gb = vram_avail_gb - non_block_gb - working_set_gb - per_block_gb
        if budget_gb <= 0:
            logger.warning(
                "block_pin: %s - no VRAM budget for pinning (avail=%.1f, non_block=%.1f, "
                "working_set=%.1f, streamed_in_flight=%.2f) -> 0 pinned, all blocks stream",
                component_name,
                vram_avail_gb,
                non_block_gb,
                working_set_gb,
                per_block_gb,
            )
            self._warn_workload_does_not_fit(
                component_name, working_set_gb, vram_avail_gb - non_block_gb - per_block_gb
            )
            return 0

        n = int(budget_gb / per_block_gb)
        n = max(0, min(n, len(blocks)))
        return n

    @staticmethod
    def _maybe_warn_expandable_segments() -> None:
        """Hint about ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``.

        Block-pin tightly budgets VRAM (pinned blocks + non-block parts +
        a working-set margin). Without ``expandable_segments``, allocator
        fragmentation eats into that budget and can turn it into an OOM.
        Logged as a one-time hint when the strategy is applied.

        Windows is skipped: ``expandable_segments`` depends on the CUDA
        virtual memory management API not exposed on the Windows driver,
        so the env var is a silent no-op there. The Windows working-set
        constant already accounts for the larger allocator overhead.

        ROCm is skipped too, but for the opposite reason: the flag *is*
        honoured on HIP builds (torch reads the CUDA-named var first), and
        it swaps ``hipMalloc`` for the HIP virtual-memory path. That path
        has been reported to hard-fail on small allocations while the
        driver still reports many GiB free, which is not the fragmentation
        this flag exists to fix. Recommending it there can break a working
        run, so the hint points at the budget knobs instead.
        """
        if sys.platform == "win32":
            return
        conf = _allocator_conf()
        enabled = "expandable_segments:True" in conf
        if _is_rocm():
            if enabled:
                logger.info(
                    "block_pin: 'expandable_segments:True' is set on a ROCm build, where it routes "
                    "allocation through the HIP virtual-memory path instead of hipMalloc. If this "
                    "run raises an out-of-memory error on a small allocation while the message "
                    "still reports several GiB free, unset it: that failure is not the "
                    "fragmentation this flag addresses."
                )
            else:
                logger.info(
                    "block_pin: ROCm build, so 'expandable_segments:True' is not recommended. The "
                    "pin budget assumes an allocator whose reserved pool tracks live bytes; if this "
                    "run OOMs, raise auto_block_pin_allocator_inflation (try 1.25) and "
                    "auto_block_pin_allocator_pool_overhead_gb (try 1.0) instead."
                )
            return
        if not enabled:
            logger.warning(
                "block_pin: PYTORCH_CUDA_ALLOC_CONF does not include "
                "'expandable_segments:True'. Allocator fragmentation eats into "
                "the pin budget and can cause OOM. Recommended: set "
                "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before "
                "starting Python."
            )

    def _largest_block_list_info(self) -> tuple[nn.Module, str, Any] | None:
        """Return ``(component, block_attr, blocks)`` for the largest *block-bearing*
        component, or ``None`` if no component has a usable block list.

        block_pin operates **per component** — it pins the repeated blocks of
        each block-bearing component and group-offloads the rest — so the
        decision that matters is "which is the heaviest component we could
        actually pin", NOT "does the single largest component overall happen to
        have a block list". Those differ whenever the largest component has no
        top-level block list: e.g. a dual-DiT pipeline where the text encoder
        (its blocks nested under ``.language_model.layers``, invisible to the
        top-level-only :func:`find_largest_block_list`) is marginally larger
        than each transformer. Selecting the largest component *among those with
        a block list* finds the transformer's ``layers`` there instead of
        bailing to ``group_offload``.

        Shared by the block-list check and the "would fully pin" check so they
        reason about the same component.
        """
        with self._lock:
            components = list(self._managed_components.values())

        seen_ids: set[int] = set()
        best: tuple[nn.Module, str, Any] | None = None
        best_size = -1
        for mod in components:
            if id(mod) in seen_ids:
                continue
            seen_ids.add(id(mod))
            result = find_largest_block_list(mod)
            if result is None:
                continue
            try:
                size = sum(p.numel() * p.element_size() for p in mod.parameters())
            except Exception:
                continue
            if size > best_size:
                best_size = size
                block_attr, blocks = result
                best = (mod, block_attr, blocks)
        return best

    def _largest_component_has_block_list(self) -> bool:
        """True if any registered component has a usable block list.

        Reports on the largest *block-bearing* component (see
        :meth:`_largest_block_list_info`), not the largest component overall —
        so a pipeline whose heaviest component lacks a top-level block list
        (e.g. a text encoder) still picks ``block_pin`` when a denoiser is
        pinnable.

        "Usable" = at least :attr:`AUTO_BLOCK_PIN_MIN_BLOCKS` entries.
        Below that threshold, per-block ``apply_group_offloading``
        overhead outweighs the benefit and plain ``group_offload`` is
        a better default.
        """
        info = self._largest_block_list_info()
        return info is not None and len(info[2]) >= self.AUTO_BLOCK_PIN_MIN_BLOCKS

    def _block_pin_would_fully_pin_largest(self, device: torch.device | str) -> bool:
        """True if block_pin would pin the *entire* largest component on the GPU.

        This is the condition under which block_pin should be preferred over
        model_offload: when every block fits resident, block_pin's VRAM peak
        equals model_offload's (the whole transformer on GPU during its forward),
        but it stays resident across runs instead of being re-transferred each
        generation — strictly faster for repeated inference.

        When it would only *partially* pin (overflow streamed), we deliberately
        do NOT prefer it over a viable model_offload: the working-set margin is
        calibrated for image diffusion and under-budgets long-video activations,
        so a partial-pin auto-choice could overflow where model_offload wouldn't.

        Mirrors the apply-time full-pin condition in ``_compute_block_pin_count``
        (``avail - non_block - working_set - per_block >= block_total``), i.e.
        ``avail >= max_component + working_set + per_block``. Uses the resolver's
        injected/estimated sizes so the decision matches the size accounting the
        rest of resolve_offload_strategy uses.
        """
        info = self._largest_block_list_info()
        if info is None:
            return False
        _, _, blocks = info
        n = len(blocks)
        if n < self.AUTO_BLOCK_PIN_MIN_BLOCKS:
            return False

        _, max_component_gb = self._estimate_components_size_gb()
        if max_component_gb <= 0:
            return False
        # Approximate per-block from the largest component / block count. This
        # slightly over-estimates per_block (it folds in the non-block parts),
        # making the threshold conservative: if we say "fully pins", the precise
        # apply-time budget will too.
        per_block_gb = max_component_gb / n
        avail_gb, _ = self._detect_available_vram_gb(device)
        working_set_gb = self._resolve_working_set_gb()
        return avail_gb >= max_component_gb + working_set_gb + per_block_gb

    # ------------------------------------------------------------------
    # Block-pin auto-evict (cross-component coordination)
    # ------------------------------------------------------------------

    def _should_evict_for_neighbor(self, neighbor_name: str | None) -> bool:
        """Decide whether to evict pinned subsets before *neighbor_name* runs.

        Four-tier policy:

        1. **Per-component override** (``set_evict_on_neighbor``) — if the
           user explicitly set ``True`` / ``False`` for this name, follow
           that. This always wins.
        2. **No states** — if no pinned subset exists, there's nothing to
           evict; short-circuit before bothering with the VRAM query.
        3. **Runtime VRAM check** — query *effective* free VRAM
           (``_effective_free_vram_gb``: driver-free + PyTorch's reclaimable
           reserved pool). If it's at or above the working-set margin the
           auto-budget reserved (``_resolve_working_set_gb``), the neighbor
           fits without evicting pinned: skip. Using effective (not driver)
           free is what stops the warm-up thrash — the grown allocator pool
           is reusable, not consumed. If detection fails (``0.0``) we evict.
        4. **Runtime RAM-absorb check** — if step 3 wants to evict, also
           verify the host can absorb the evicted subset without itself
           running out. Compare ``ram_available`` against
           ``evicted_subset + AUTO_BLOCK_PIN_RAM_EVICT_HEADROOM_GB``. If
           the host can't absorb, refuse to evict — pushing 10+ GiB of
           weights into a host with no room left would just trigger an
           ``cudaHostAlloc`` failure on the neighbor's next ``pin_memory``
           call, which is strictly worse than letting the neighbor try to
           fit in whatever VRAM is free.

        *neighbor_name* may be ``None`` for callers without identity info
        (defensive — every install path threads the name through, so this
        is only the fallback in case a future wrap forgets).
        """
        if neighbor_name is not None:
            with self._lock:
                override = self._evict_on_neighbor.get(neighbor_name)
            if override is not None:
                return override

        with self._lock:
            states = list(self._block_pin_states.values())
        if not states:
            return False

        sample_device = states[0].device
        free_gb = self._effective_free_vram_gb(sample_device)
        threshold_gb = self._resolve_working_set_gb()
        if free_gb >= threshold_gb:
            return False

        resident = [s for s in states if s.resident]
        if not resident:
            return False

        evicted_gb = sum(s.pinned_size_bytes for s in resident) / (1024**3)
        ram_avail_gb, _ = self._detect_available_ram_gb()
        required_ram_gb = evicted_gb + self.AUTO_BLOCK_PIN_RAM_EVICT_HEADROOM_GB
        if ram_avail_gb > 0 and ram_avail_gb < required_ram_gb:
            return False

        return True

    def _evict_all_pinned(self, neighbor_name: str | None = None) -> None:
        """Pre-forward callback for neighbor components.

        Pushes every currently-resident pinned subset back to CPU so the
        about-to-run neighbor has the VRAM. Held subsets are repinned on
        demand by :meth:`_repin_one_pinned` when the owning component's
        own forward fires next.

        Skips entirely if :meth:`_should_evict_for_neighbor` says the
        neighbor has enough headroom (or the user disabled it for this
        component). That's the cheap path most of the time on a healthy
        system where the auto-budget's reservation hasn't been blown.

        Logs the decision (with the inputs that drove it) so users can
        diagnose unexpected behavior without instrumenting their own code
        — without this, the install-time "auto-evict installed" line is
        the only signal and you can't tell "eviction fired and helped"
        apart from "eviction was correctly skipped" apart from "the hook
        never fired at all" in a real inference run.
        """
        with self._lock:
            states_snapshot = list(self._block_pin_states.values())
        if not states_snapshot:
            return

        sample_device = states_snapshot[0].device
        free_gb = self._effective_free_vram_gb(sample_device)
        threshold_gb = self._resolve_working_set_gb()
        with self._lock:
            override = self._evict_on_neighbor.get(neighbor_name) if neighbor_name else None

        if not self._should_evict_for_neighbor(neighbor_name):
            # Pick the most informative log line based on *why* the
            # decision was skip. Mirrors the policy in
            # :meth:`_should_evict_for_neighbor` so the message reflects
            # the binding constraint.
            if override is False:
                logger.info(
                    "block_pin: skip evict for %r (override=False)",
                    neighbor_name,
                )
            elif free_gb >= threshold_gb:
                logger.info(
                    "block_pin: skip evict for %r (free=%.2f GiB >= threshold=%.2f GiB, override=%r)",
                    neighbor_name,
                    free_gb,
                    threshold_gb,
                    override,
                )
            else:
                resident = [s for s in states_snapshot if s.resident]
                evicted_gb = sum(s.pinned_size_bytes for s in resident) / (1024**3)
                ram_avail_gb, _ = self._detect_available_ram_gb()
                required_ram_gb = evicted_gb + self.AUTO_BLOCK_PIN_RAM_EVICT_HEADROOM_GB
                logger.warning(
                    "block_pin: eviction needed for %r (vram free=%.2f GiB < threshold=%.2f GiB) "
                    "but host RAM cannot absorb evicted subset (avail=%.2f GiB < %.2f GiB = "
                    "%.2f evicted + %.2f headroom). Keeping pinned subset on GPU; "
                    "neighbor may OOM if it doesn't fit in free VRAM.",
                    neighbor_name,
                    free_gb,
                    threshold_gb,
                    ram_avail_gb,
                    required_ram_gb,
                    evicted_gb,
                    self.AUTO_BLOCK_PIN_RAM_EVICT_HEADROOM_GB,
                )
            return

        evicted = 0
        for state in states_snapshot:
            if state.resident:
                evict_pinned_subset(state)
                evicted += 1
        if evicted:
            # Hand the freed pages back to the driver. Eviction only returns them
            # to PyTorch's caching allocator, and the neighbor we just made room
            # for may need memory PyTorch's pool cannot serve: cuDNN convolution
            # workspaces, Triton scratch and every other non-PyTorch allocator go
            # straight to the driver. A video VAE decode hits exactly that path
            # and fails with an opaque CUDNN_STATUS_INTERNAL_ERROR (not an OOM)
            # when the pool is still holding the card.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        logger.info(
            "block_pin: evicted %d subset(s) for %r (free=%.2f GiB, threshold=%.2f GiB, override=%r)",
            evicted,
            neighbor_name,
            free_gb,
            threshold_gb,
            override,
        )

    def _repin_one_pinned(self, component_name: str) -> None:
        """Pre-forward callback for a block-pinned component.

        If the pinned subset is currently on CPU (a neighbor evicted it),
        move it back to GPU before forward runs.
        """
        with self._lock:
            state = self._block_pin_states.get(component_name)
        if state is None:
            return
        if not state.resident:
            repin_pinned_subset(state)

    def _wrap_neighbor_method(self, component: Any, method_name: str, component_name: str) -> None:
        """Wrap ``component.<method_name>`` to evict pinned subsets first.

        Catches entry points that bypass ``__call__`` / ``forward``,
        notably ``vae.decode`` and ``vae.encode`` — these never trigger
        ``register_forward_pre_hook`` so the auto-evict logic would miss
        them otherwise.

        *component_name* is captured into the wrapper closure so the
        runtime evict decision can check this component's
        ``set_evict_on_neighbor`` override. Without it, the wrap would
        always evict unconditionally, undermining the runtime check that
        the forward-pre-hook path already respects.

        Idempotent via a marker attribute on the wrapper: re-wrapping is a
        no-op. The original is restored on strategy transition by either
        deleting the instance attribute (when the original lived on the
        class — the common case) or by re-assigning the saved bound
        method (when the instance already had its own).
        """
        original = getattr(component, method_name, None)
        if original is None or not callable(original):
            return
        if getattr(original, "_diffusers_mm_block_pin_wrap", False):
            return  # Already wrapped (e.g. incremental apply).

        # Was the attribute on the instance __dict__ (rare), or inherited
        # from the class (common)? Restore needs to know which to do.
        instance_already_had_attr = method_name in component.__dict__

        manager = self
        captured_name = component_name

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            manager._evict_all_pinned(captured_name)
            return original(*args, **kwargs)

        wrapper._diffusers_mm_block_pin_wrap = True  # type: ignore[attr-defined]
        try:
            wrapper.__wrapped__ = original  # type: ignore[attr-defined]
        except Exception:
            pass

        setattr(component, method_name, wrapper)
        restore_value = original if instance_already_had_attr else _INSTANCE_ATTR_ABSENT
        with self._lock:
            self._block_pin_wrapped_methods.append((component, method_name, restore_value))

    def _install_block_pin_auto_evict(
        self,
        new_pinned: list[tuple[str, Any, str, int]],
        new_neighbors: list[tuple[str, Any]],
        device: torch.device,
    ) -> None:
        """Record per-component pinned state and install the eviction hooks.

        *new_pinned* are the components that got ``apply_block_pin`` this
        call — they each get a pre-forward hook that repins their subset
        on demand. *new_neighbors* are the fallback group_offload ones —
        they each get a pre-forward hook (catches ``__call__``) plus
        ``decode``/``encode`` method wraps that evict every resident
        pinned subset before their own work starts.

        Called per-apply so incremental registrations are handled: a new
        component registered after the initial apply walks through here
        and gets its hook(s) installed without disturbing existing ones.
        Skips entirely when :attr:`block_pin_auto_evict` is False so the
        opt-out is genuinely zero-cost.
        """
        # Record pinned-subset states unconditionally — the
        # ``set_block_pin_count`` / inspection paths benefit from them
        # even when the auto-evict feature is off. Cache the eviction
        # footprint (pinned blocks + non-block top-level parts that
        # ``evict_pinned_subset`` also moves) so the auto-evict
        # pre-forward hook can do the RAM-absorb check without
        # re-walking module parameters every call.
        for name, mod, block_attr, n in new_pinned:
            blocks = getattr(mod, block_attr)
            per_block = per_block_size_bytes(blocks)
            non_block = non_block_size_bytes(mod, block_attr)
            pinned_size = n * per_block + non_block
            with self._lock:
                self._block_pin_states[name] = BlockPinState(
                    component=mod,
                    block_attr=block_attr,
                    n_pinned=n,
                    device=device,
                    resident=True,
                    pinned_size_bytes=pinned_size,
                )

        if not self._block_pin_auto_evict:
            return

        for name, mod, _block_attr, _n in new_pinned:
            handle = mod.register_forward_pre_hook(lambda _module, _inputs, _name=name: self._repin_one_pinned(_name))
            with self._lock:
                self._block_pin_hook_handles.append(handle)

        for name, mod in new_neighbors:
            handle = mod.register_forward_pre_hook(lambda _module, _inputs, _name=name: self._evict_all_pinned(_name))
            with self._lock:
                self._block_pin_hook_handles.append(handle)
            for method_name in _BLOCK_PIN_NEIGHBOR_WRAP_METHODS:
                self._wrap_neighbor_method(mod, method_name, name)

        if new_pinned or new_neighbors:
            logger.info(
                "block_pin: auto-evict installed (pinned=%d, neighbors=%d)",
                len(new_pinned),
                len(new_neighbors),
            )

    def _teardown_block_pin_auto_evict(self) -> None:
        """Remove all auto-evict hooks and restore wrapped methods.

        Called on strategy transition away from ``block_pin`` and on
        :meth:`clear`. Before tearing down, makes sure every pinned subset
        is moved back to GPU — the surrounding transition path will then
        move everything to CPU via plain ``.to('cpu')``, but it relies on
        the modules being in a consistent device state first.
        """
        with self._lock:
            handles = list(self._block_pin_hook_handles)
            self._block_pin_hook_handles.clear()
            handles += list(self._spill_calib_handles)
            self._spill_calib_handles = []
            handles += list(self._workload_probe_handles)
            self._workload_probe_handles = []
            wrapped = list(self._block_pin_wrapped_methods)
            self._block_pin_wrapped_methods.clear()
            states = list(self._block_pin_states.values())
            self._block_pin_states.clear()

        for handle in handles:
            try:
                handle.remove()
            except Exception as e:
                logger.warning("block_pin: failed to remove auto-evict hook: %s", e)

        for component, method_name, restore_value in wrapped:
            try:
                if restore_value is _INSTANCE_ATTR_ABSENT:
                    if method_name in component.__dict__:
                        delattr(component, method_name)
                else:
                    setattr(component, method_name, restore_value)
            except Exception as e:
                logger.warning("block_pin: failed to unwrap %s.%s: %s", type(component).__name__, method_name, e)

        # If a neighbor ran last, pinned subsets may have been evicted to
        # CPU — repin them so the subsequent transition's bulk ``.to('cpu')``
        # operates on a consistent starting state, and so debugging tools
        # see the residency the strategy promised.
        for state in states:
            if not state.resident:
                try:
                    repin_pinned_subset(state)
                except Exception as e:
                    logger.warning(
                        "block_pin: failed to repin %s during teardown: %s",
                        type(state.component).__name__,
                        e,
                    )

    def prepare_strategy_transition(self, new_strategy: str, device: torch.device | str) -> None:
        """Clean up the old offload strategy before applying *new_strategy*.

        Iterates components deduplicated by ``id(module)`` so that aliased
        names (the same module registered under multiple names) only get
        cleaned up once. Clears per-component strategy state so every
        component will be re-applied under the new strategy.

        Hook-based strategies (``group_offload`` via diffusers hooks,
        ``model_offload`` via accelerate hooks, ``block_pin`` via per-block
        diffusers hooks on the overflow blocks) all have their hooks
        stripped via ``remove_offload_hooks`` (which handles both flavors).
        Then every component is moved back to CPU so the next strategy
        starts from a clean state.
        """
        with self._lock:
            old = self._applied_strategy
            if old == new_strategy:
                return

            hook_based = old in ("group_offload", "model_offload", "block_pin")
            was_block_pin = old == "block_pin"

        # Done outside the lock so the teardown (which may take its own
        # passes through ``self._lock``) doesn't deadlock against itself.
        if was_block_pin:
            self._teardown_block_pin_auto_evict()

        with self._lock:
            seen_ids: set[int] = set()
            for name, mod in self._managed_components.items():
                if id(mod) in seen_ids:
                    continue
                seen_ids.add(id(mod))
                if hook_based:
                    remove_offload_hooks(mod)
                if hasattr(mod, "to"):
                    mod.to("cpu")
                    logger.debug("Strategy transition: cleaned %s (was %s)", name, old)

            self._component_strategies.clear()
            self._model_offload_final_hook = None
            self._applied_strategy = new_strategy

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _group_offload_kwargs(self, device: torch.device | str) -> dict[str, Any]:
        """Build kwargs for ``apply_group_offloading`` (leaf-level only).

        Configurable knobs come from instance state, settable via the
        constructor or matching properties:

        - ``group_offload_use_stream`` — overlap transfers with compute
          (default True; ~1.5–3× faster on hardware that supports streams).
        - ``group_offload_low_cpu_mem`` — defer pinned host buffer
          allocation to per-transfer; without this, ``apply_group_offloading``
          pins a full copy of every weight upfront and holds it for the
          whole inference (~2× host RAM). Only honored when
          ``use_stream=True``. Default True.

        ``record_stream`` is always passed as ``False`` — some models
        produce numerical noise with ``record_stream=True`` and the
        speed-up is negligible, so it's not worth exposing as a knob.
        """
        use_stream = self._group_offload_use_stream
        kwargs: dict[str, Any] = {
            "onload_device": torch.device(device) if isinstance(device, str) else device,
            "offload_device": torch.device("cpu"),
            "offload_type": "leaf_level",
            "use_stream": use_stream,
            "record_stream": False,
        }
        if use_stream and self._group_offload_low_cpu_mem:
            kwargs["low_cpu_mem_usage"] = True
        return kwargs

    def apply_offload_strategy(self, device: torch.device | str) -> str:
        """Resolve and apply the current offload strategy to managed components.

        Incremental: components that already have the resolved strategy
        applied are skipped. Components registered after a previous apply
        call are picked up automatically. On a strategy *transition*,
        every component is re-applied under the new strategy.

        Returns the resolved (concrete) strategy name.
        """
        strategy = self.resolve_offload_strategy(device)

        with self._lock:
            if not self._managed_components:
                self._applied_strategy = strategy
                return strategy

        # Strategy transition: clean up old hooks/placement first. After
        # this, every component is "pending" and will be re-applied below.
        if self._applied_strategy is not None and self._applied_strategy != strategy:
            self.prepare_strategy_transition(strategy, device)
        else:
            with self._lock:
                self._applied_strategy = strategy

        with self._lock:
            pending: list[tuple[str, Any]] = [
                (name, mod)
                for name, mod in self._managed_components.items()
                if self._component_strategies.get(name) != strategy
            ]

        if not pending:
            return strategy

        # Deduplicate by module identity: a single nn.Module registered
        # under multiple names should only be hooked/moved once. Aliased
        # names still get their per-component strategy state recorded so a
        # subsequent apply doesn't try to re-process them.
        seen_ids: set[int] = set()
        unique: list[tuple[str, Any]] = []
        aliases: list[str] = []
        for name, mod in pending:
            if id(mod) in seen_ids:
                aliases.append(name)
                continue
            seen_ids.add(id(mod))
            unique.append((name, mod))

        if strategy == "no_offload":
            for name, mod in unique:
                if hasattr(mod, "to"):
                    mod.to(device)
                    logger.info("no_offload: moved %s to %s", name, device)

        elif strategy == "group_offload":
            from diffusers.hooks.group_offloading import apply_group_offloading

            offload_kwargs = self._group_offload_kwargs(device)
            # Log the resolved kwargs once so the actual config (especially
            # whether low_cpu_mem_usage is being passed) is visible at run
            # time without having to read source. low_cpu_mem_usage is
            # absent from kwargs when use_stream=False (diffusers ignores it
            # in that mode); show "absent" explicitly there.
            logger.info(
                "group_offload kwargs: use_stream=%s, low_cpu_mem_usage=%s",
                offload_kwargs.get("use_stream"),
                offload_kwargs.get("low_cpu_mem_usage", "absent"),
            )
            for name, mod in unique:
                try:
                    remove_offload_hooks(mod)
                    if self._keep_resident_instead_of_offload(name, mod, device):
                        continue
                    apply_group_offloading(mod, **offload_kwargs)
                    logger.info("group_offload (leaf_level) enabled for %s", name)
                except Exception as e:
                    logger.warning("Failed to enable group_offload for %s: %s", name, e)

        elif strategy == "model_offload":
            self._install_model_offload_chain(device)

        elif strategy == "block_pin":
            from diffusers.hooks.group_offloading import apply_group_offloading

            self._maybe_warn_expandable_segments()
            offload_kwargs = self._group_offload_kwargs(device)
            logger.info(
                "block_pin: overflow streaming kwargs use_stream=%s, low_cpu_mem_usage=%s",
                offload_kwargs.get("use_stream"),
                offload_kwargs.get("low_cpu_mem_usage", "absent"),
            )
            device_obj = torch.device(device) if isinstance(device, str) else device
            # Track per-component outcomes so the auto-evict installer
            # below knows which got a pinned subset (gets a repin hook)
            # and which fell back to group_offload (gets evict hooks).
            new_pinned: list[tuple[str, Any, str, int]] = []
            new_neighbors: list[tuple[str, Any]] = []
            try:
                roles = {info.name: info.role for info in self.classify_components()}
            except Exception:  # classification is best-effort; empty roles disables the guard below
                roles = {}
            for name, mod in unique:
                # Pre-clean any prior hooks so re-applying is safe (idempotent).
                remove_offload_hooks(mod)
                # Checked before block-list discovery: legacy `weight_norm` cannot be
                # offloaded *or* partially pinned, so it must win over any block list the
                # component happens to expose (audio VAEs have both).
                if self._keep_resident_instead_of_offload(name, mod, device):
                    continue
                # Only denoisers are eligible for pinning. `_resolve_working_set_gb()` models the
                # *denoiser's* activation footprint, so pinning another component's blocks spends
                # VRAM that component's own activations still need — a transformer-decoder video
                # VAE has a decode footprint of its own and would OOM against the denoiser's
                # reserve.
                #
                # This is defense-in-depth rather than a behaviour change: `find_largest_block_list`
                # only walks top-level children, and across the installed diffusers every one of the
                # 65 denoisers (transformers/ + unets/) exposes its block list there while all 28
                # VAEs and every transformers text encoder nest theirs (`decoder.blocks`,
                # `text_model.encoder.layers`, `model.language_model.layers`). So non-denoisers
                # already fell through to group_offload; this makes the intent explicit and covers a
                # future model that does put a repeated-block list at the top level of a VAE.
                #
                # Must stay *after* the legacy-weight_norm check above: such components have to be
                # kept resident, not group_offloaded by this branch.
                if roles and roles.get(name) != "denoiser":
                    try:
                        apply_group_offloading(mod, **offload_kwargs)
                        logger.info("block_pin: %s is not a denoiser, using group_offload", name)
                        new_neighbors.append((name, mod))
                    except Exception as e:
                        logger.warning("Failed to enable group_offload for %s: %s", name, e)
                    continue
                result = find_largest_block_list(mod)
                if result is None:
                    # No discoverable repeated-block list — fall back to
                    # plain leaf-level group offload for this component.
                    try:
                        apply_group_offloading(mod, **offload_kwargs)
                        logger.info(
                            "block_pin: %s has no block list, fell back to group_offload",
                            name,
                        )
                        new_neighbors.append((name, mod))
                    except Exception as e:
                        logger.warning("block_pin: group_offload fallback failed for %s: %s", name, e)
                    continue
                block_attr, blocks = result
                n = self._compute_block_pin_count(name, mod, block_attr, blocks, device)
                try:
                    applied_n = apply_block_pin(
                        mod,
                        block_attr,
                        blocks,
                        n,
                        device_obj,
                        offload_kwargs=offload_kwargs,
                    )
                    logger.info(
                        "block_pin: %s - pinned %d/%d %s blocks, streaming %d",
                        name,
                        applied_n,
                        len(blocks),
                        block_attr,
                        len(blocks) - applied_n,
                    )
                    new_pinned.append((name, mod, block_attr, applied_n))
                except Exception as e:
                    logger.warning("block_pin: failed for %s: %s", name, e)
                    if self._rollback_pin_to_group_offload(name, mod, offload_kwargs):
                        new_neighbors.append((name, mod))

            self._install_block_pin_auto_evict(new_pinned, new_neighbors, device_obj)
            # Read the true sequence length off the denoiser's first input and
            # lower the pin count before any activation is allocated. This is
            # what keeps an unrecorded (or under-recorded) workload from
            # over-pinning and OOMing on the first step.
            self._install_block_pin_workload_probe(device_obj)
            # Windows: probe VRAM after the first denoise step and unpin if it
            # oversubscribes, instead of waiting for the whole (slow) generation.
            self._install_spill_calibration_hook(device_obj)

        # Strategy-independent: the text encoders are dead weight once denoising
        # starts no matter how the weights are placed.
        self._install_text_encoder_unload_hook(device)

        with self._lock:
            for name, _ in unique:
                self._component_strategies[name] = strategy
            for name in aliases:
                self._component_strategies[name] = strategy

        return strategy

    def _install_model_offload_chain(self, device: torch.device | str) -> None:
        """Install accelerate ``cpu_offload_with_hook`` chained across all managed components.

        Replaces the previous "do nothing at apply time, defer to use_components"
        behavior with a real drop-in replacement for diffusers'
        ``enable_model_cpu_offload``: each component auto-moves to GPU at its
        own forward call, and the chained ``prev_module_hook`` ensures the
        previous component is offloaded before the next one loads — which
        is what keeps the transformer resident on GPU across multi-step
        denoising loops without thrashing.

        Always processes ALL managed components (not just pending ones)
        because the chain order has to be consistent with registration
        order; existing accelerate hooks on those components are removed
        first so re-applying is safe.
        """
        from accelerate import cpu_offload_with_hook

        device_obj = torch.device(device) if isinstance(device, str) else device

        with self._lock:
            chain_components: list[tuple[str, Any]] = []
            seen_ids: set[int] = set()
            for name, mod in self._managed_components.items():
                if id(mod) in seen_ids:
                    continue
                seen_ids.add(id(mod))
                chain_components.append((name, mod))

        # Strip any prior hooks (idempotent no-op if there are none).
        for _, mod in chain_components:
            remove_offload_hooks(mod)

        prev_hook = None
        for name, mod in chain_components:
            try:
                _, prev_hook = cpu_offload_with_hook(mod, device_obj, prev_module_hook=prev_hook)
                logger.info("model_offload: chained accelerate hook on %s", name)
            except Exception as e:
                logger.warning("Failed to install model_offload hook on %s: %s", name, e)

        self._model_offload_final_hook = prev_hook

    def reapply_group_offload(self, component_name: str, device: torch.device | str) -> None:
        """Re-apply group offload hooks to a single component.

        Useful after modifying a component's module structure (e.g. loading
        LoRA adapters) so that new submodules get offload hooks.
        """
        if self._applied_strategy != "group_offload":
            return

        from diffusers.hooks.group_offloading import apply_group_offloading

        with self._lock:
            mod = self._managed_components.get(component_name)
        if mod is None:
            return

        offload_kwargs = self._group_offload_kwargs(device)
        remove_offload_hooks(mod)
        apply_group_offloading(mod, **offload_kwargs)
        logger.info("Re-applied group offload hooks for %s", component_name)

    # ------------------------------------------------------------------
    # use_components context manager
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def use_components(
        self,
        *names: str,
        device: torch.device | str,
        strategy_override: str | None = None,
    ) -> Generator[None]:
        """Context manager that places named components on *device*.

        Behaviour depends on the active offload strategy:

        - ``no_offload`` / ``group_offload`` / ``block_pin``: no-op yield
          (placement is already correct — components are on GPU permanently
          for no_offload, have hooks for group_offload, or have a mix of
          pinned blocks + per-block hooks for block_pin).
        - ``model_offload``: bulk CPU → GPU move on enter, back to CPU on
          exit. Mostly relevant for **decomposed** pipeline workflows;
          monolithic ``pipe(...)`` calls already work correctly under
          model_offload via the accelerate chain installed at apply time.

        *strategy_override* lets callers force a different strategy for
        these components for the duration of the block.
        """
        strategy = self._applied_strategy
        if strategy is None:
            strategy = self.apply_offload_strategy(device)
        if strategy_override is not None:
            strategy = strategy_override

        if strategy in ("no_offload", "group_offload", "block_pin"):
            yield
            return

        modules: list[tuple[str, Any]] = []
        with self._lock:
            for n in names:
                mod = self._managed_components.get(n)
                if mod is not None and hasattr(mod, "to"):
                    modules.append((n, mod))

        if strategy == "model_offload":
            actual_strategy = self._applied_strategy
            for name, mod in modules:
                remove_offload_hooks(mod)
                mod.to(device)
                logger.debug("model_offload: moved %s to %s", name, device)
            try:
                yield
            finally:
                for name, mod in modules:
                    mod.to("cpu")
                    logger.debug("model_offload: moved %s back to CPU", name)
                # Restore group_offload hooks if the real strategy is group_offload
                # and we only used model_offload as a temporary override.
                if actual_strategy == "group_offload":
                    from diffusers.hooks.group_offloading import apply_group_offloading

                    restore_kwargs = self._group_offload_kwargs(device)
                    for name, mod in modules:
                        try:
                            apply_group_offloading(mod, **restore_kwargs)
                            logger.debug("model_offload: restored group_offload hooks on %s", name)
                        except Exception as e:
                            logger.warning("model_offload: failed to restore hooks on %s: %s", name, e)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        else:
            yield

    # ------------------------------------------------------------------
    # Debugging
    # ------------------------------------------------------------------

    def debug_vram_breakdown(self, *, device: torch.device | str | None = None) -> dict[str, float]:
        """Print and return a dedicated-VRAM accounting breakdown.

        Designed for the recurring "PyTorch says X but Task Manager says Y"
        confusion. Three numbers matter on the dedicated-VRAM side:

        - **Driver used** — ``cudaMemGetInfo``'s ``total - free``. The
          authoritative "how much physical VRAM is committed right now"
          number. Includes every allocation on the device: PyTorch's
          caching allocator pool, CUDA context overhead, cuDNN/cuBLAS
          workspaces, kernels that use ``cudaMalloc`` directly (e.g.
          Triton scratch buffers), other processes.
        - **PyTorch reserved** — what PyTorch's caching allocator holds in
          its pool. Allocated + free-but-cached. This is what shrinks
          back when you call :func:`torch.cuda.empty_cache`.
        - **External** — driver-used minus PyTorch-reserved. Everything
          on the device that PyTorch doesn't manage: normally the CUDA
          context plus cuDNN workspaces. If it is much larger, suspect a
          non-PyTorch allocator (Triton, cuFFT plans, etc.).

        Crucially, this is the *dedicated* side only. On Windows, Task
        Manager's "GPU Memory" line is dedicated + shared. Shared GPU
        memory is CUDA-pinned host memory (system RAM mapped to GPU
        address space) — created by ``apply_group_offloading`` when
        ``low_cpu_mem_usage=True`` so streamed weights can transfer
        without per-call pinning. CUDA cannot report it, so it is absent
        here — but it is the usual reason Task Manager shows far more than
        this breakdown accounts for.

        Returns the same numbers as a dict for programmatic use.
        """
        gb = 1024**3
        if not torch.cuda.is_available():
            print("[debug_vram] CUDA not available")
            return {}

        target_device = device if device is not None else torch.cuda.current_device()
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(target_device)
        except Exception as e:
            print(f"[debug_vram] mem_get_info failed: {e}")
            return {}
        used_bytes = total_bytes - free_bytes

        allocated = torch.cuda.memory_allocated(target_device)
        reserved = torch.cuda.memory_reserved(target_device)
        max_allocated = torch.cuda.max_memory_allocated(target_device)
        max_reserved = torch.cuda.max_memory_reserved(target_device)

        external_bytes = max(0, used_bytes - reserved)

        print("[debug_vram] === Dedicated VRAM breakdown ===")
        print(
            f"[debug_vram] Driver used:        {used_bytes / gb:6.2f} / {total_bytes / gb:6.2f} GiB "
            f"({free_bytes / gb:.2f} GiB free)"
        )
        print(f"[debug_vram] PyTorch allocated:  {allocated / gb:6.2f} GiB (peak {max_allocated / gb:6.2f} GiB)")
        print(f"[debug_vram] PyTorch reserved:   {reserved / gb:6.2f} GiB (peak {max_reserved / gb:6.2f} GiB)")
        print(f"[debug_vram] External (driver_used - pytorch_reserved): {external_bytes / gb:6.2f} GiB")
        print("[debug_vram]   = CUDA context + cuDNN/cuBLAS workspaces + non-PyTorch allocators (e.g. Triton)")
        print("[debug_vram] (Task Manager 'GPU Memory' on Windows also includes shared GPU memory,")
        print("[debug_vram]  i.e. CUDA-pinned host buffers from group_offload — not shown here.)")

        with self._lock:
            states = dict(self._block_pin_states)
        if states:
            print("[debug_vram] === block_pin state ===")
            for name, state in states.items():
                print(
                    f"[debug_vram]   {name}: n_pinned={state.n_pinned}, "
                    f"resident={state.resident}, device={state.device}"
                )

        return {
            "driver_used_gb": used_bytes / gb,
            "driver_total_gb": total_bytes / gb,
            "driver_free_gb": free_bytes / gb,
            "pytorch_allocated_gb": allocated / gb,
            "pytorch_reserved_gb": reserved / gb,
            "pytorch_max_allocated_gb": max_allocated / gb,
            "pytorch_max_reserved_gb": max_reserved / gb,
            "external_gb": external_bytes / gb,
        }

    @contextlib.contextmanager
    def record_memory_history(
        self,
        output_path: str,
        *,
        max_entries: int = 100_000,
    ) -> Generator[None]:
        """Record CUDA allocations during the context and dump a snapshot on exit.

        Wraps ``torch.cuda.memory._record_memory_history`` and writes the
        snapshot to *output_path*. Visualize locally with::

            python -m torch.cuda._memory_viz trace_plot snapshot.pickle -o trace.html

        (or ``segment_plot`` / ``stats`` / ``compare`` — see ``--help``).
        The hosted viewer at https://docs.pytorch.org/memory_viz reads the
        same pickle if you'd rather drag-and-drop. Useful when a user
        reports an unexpected OOM and you need to see where the spike
        came from.

        No-op when CUDA is unavailable so debug calls are safe to leave in
        place across CPU-only test runs.
        """
        if not torch.cuda.is_available():
            yield
            return

        torch.cuda.memory._record_memory_history(max_entries=max_entries)
        logger.info("Recording CUDA memory history (max_entries=%d)", max_entries)
        try:
            yield
        finally:
            try:
                torch.cuda.memory._dump_snapshot(output_path)
                logger.info("Dumped CUDA memory snapshot to %s", output_path)
            except Exception as e:
                logger.warning("Failed to dump CUDA memory snapshot: %s", e)
            finally:
                torch.cuda.memory._record_memory_history(enabled=None)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Free all components, caches, and CUDA memory.

        Removes offload hooks from every managed module before dropping
        references — calls ``remove_offload_hooks`` (idempotent and safe
        even if the module had no hooks installed).
        """
        # Done before taking the lock — teardown takes its own locks and
        # the helper relies on ``_managed_components`` still being populated
        # to find components to unwrap.
        self._teardown_block_pin_auto_evict()
        with self._lock:
            seen: set[int] = set()
            for module in self._managed_components.values():
                if id(module) in seen:
                    continue
                seen.add(id(module))
                try:
                    remove_offload_hooks(module)
                except Exception as e:
                    logger.warning("clear: failed to remove hooks: %s", e)
            # Detach any active source finalizers so they don't fire
            # later trying to clean up state we just wiped.
            for finalizer in self._source_finalizers.values():
                finalizer.detach()
            self._source_finalizers.clear()
            self._component_cache.clear()
            self._managed_components.clear()
            self._component_strategies.clear()
            self._refcount.clear()
            self._source_registrations.clear()
            self._model_offload_final_hook = None
            self._applied_strategy = None
            self._group_offload_use_stream = True
            self._group_offload_low_cpu_mem = True
            self._block_pin_auto_evict = True
            self._block_pin_counts.clear()
            self._block_pin_user_counts.clear()
            self._dropped_components.clear()
            self._source_refs.clear()
            self._evict_on_neighbor.clear()
            self._block_pin_seq_len = 0
            self._block_pin_batch = 1
            self._block_pin_activation_scale = 1.0
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass
