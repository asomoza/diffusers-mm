"""Core ModelManager — thread-safe model lifecycle and offload strategy management."""

from __future__ import annotations

import contextlib
import contextvars
import gc
import hashlib
import logging
import sys
import threading
import weakref
from collections.abc import Callable, Generator
from typing import Any

import torch
from torch import nn

from diffusers_mm.block_pin import (
    BlockPinState,
    apply_block_pin,
    evict_pinned_subset,
    find_largest_block_list,
    non_block_size_bytes,
    per_block_size_bytes,
    repin_pinned_subset,
)
from diffusers_mm.hooks import remove_offload_hooks


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

    # Heuristic factors used by the ``auto`` strategy resolver. The
    # rationale: weights occupy GPU memory steadily; activations come and
    # go on top. 1.5× is a reasonable middle-ground for typical diffusion
    # pipelines (SDXL, Flux, Wan-class). Tweak via subclass / attribute set
    # if your workload has unusually large or small activation footprint.
    AUTO_NO_OFFLOAD_FACTOR = 1.5  # full pipeline must fit in VRAM × this margin
    AUTO_MODEL_OFFLOAD_FACTOR = 1.5  # largest single component must fit × this
    # If pipeline weights exceed RAM × this, log a loud warning that the
    # workload likely won't fit on host memory at all.
    AUTO_RAM_HEADROOM = 0.85
    # When ``auto`` picks ``group_offload``, decide whether to flip
    # ``low_cpu_mem_usage`` off. With ``low_cpu_mem_usage=False``
    # diffusers pre-pins a full copy of every weight at apply time for
    # faster transfers (pinning happens once, not per-transfer). With
    # ``True``, pinning is per-transfer — slower steady-state but lower
    # peak host RAM.
    #
    # The flip condition is "RAM ≥ pipeline_weights + headroom". The
    # headroom covers OS + pipeline activations + transient buffers; we
    # don't budget for the original weights because modern safetensors
    # is mmap'd (originals may or may not be resident, and pages get
    # evicted as needed). Default 16 GB is comfortable for most setups.
    AUTO_LOW_CPU_MEM_RAM_HEADROOM_GB = 16.0
    # Block-pin auto-budget: when computing the optimal ``num_to_pin`` per
    # component, reserve this much VRAM for the streaming overflow's
    # working set (pinned host buffers in flight + activations + per-step
    # peaks).
    #
    # Empirically calibrated for image-diffusion workloads. **For long
    # video at meaningful resolution (e.g. LTX-2.3 at 768×512×121f) the
    # actual working set is 10–14 GiB**, far above this constant — the
    # auto-budget will over-pin on small GPUs and overflow. Bump this on
    # the instance/subclass for those workloads, or override the per-
    # component pin count via :meth:`set_block_pin_count`.
    #
    # Measured on LTX-2.3 distilled int4 (per-block 0.223 GiB,
    # non-block 0.71 GiB, 48 blocks) at 768×512×121f, 8 steps:
    #   - Linux RTX 5090 32 GiB:    WS ≈ 10.3 GiB (n=28) → 12.2 GiB (n=0)
    #   - Windows RTX 4090L 16 GiB: WS ≈ 13.1 GiB (n=28) → 14.3 GiB (n=0)
    # Working set scales with the number of streamed (non-pinned) blocks;
    # the constant cannot capture that, so we pick a conservative value
    # that's safe for image diffusion and document the video gap.
    AUTO_BLOCK_PIN_WORKING_SET_GB = 6.5
    # Windows pays a structural ~2 GiB penalty on top of the Linux value:
    # ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` is Linux-only
    # (relies on the CUDA virtual memory management API not exposed on
    # the Windows driver), so the Windows allocator runs in fixed-segment
    # mode and reserves more under the same load. Measured on the same
    # LTX-2.3 int4 sweep above, ``peak_reserved`` was 2.0–2.8 GiB higher
    # on Windows 16 GiB than Linux 32 GiB at every pin count. Splitting
    # the constant by OS keeps Linux users from paying for an allocator
    # regime they don't have. Override the same way as the Linux value.
    AUTO_BLOCK_PIN_WORKING_SET_WINDOWS_GB = 8.5
    # Don't bother with block_pin if the discoverable block list is
    # smaller than this — the overhead of per-block apply_group_offloading
    # outweighs the benefit when there are only a handful of blocks.
    AUTO_BLOCK_PIN_MIN_BLOCKS = 8
    # Safety headroom for the auto-evict RAM check. When the pre-forward
    # hook decides "free VRAM is below the working-set margin, I should
    # evict the pinned subset", we also check that the host can absorb
    # the evicted weights without itself OOMing. The check is:
    #
    #   ram_available_gb >= evicted_subset_gb + RAM_EVICT_HEADROOM_GB
    #
    # The headroom covers the about-to-run neighbor's own ``pin_memory``
    # allocations for streaming + activations + OS slack. Default 4 GiB
    # is comfortable for image/video diffusion neighbors; bump it if your
    # neighbors have unusually large host-side staging needs.
    #
    # Real-world trigger: a Windows user with 24 GiB VRAM and 32 GiB RAM
    # hit ``pin_memory()`` OOM on the connectors forward after auto-evict
    # pushed ~12 GiB of int8 transformer blocks from VRAM to a host that
    # had 1.8 GiB free. The eviction "succeeded" (via swap) but starved
    # the next ``cudaHostAlloc`` call. With this check, the eviction is
    # refused, the pinned subset stays on GPU, and the neighbor is left
    # to fit in whatever VRAM is free — not great, but not strictly worse.
    AUTO_BLOCK_PIN_RAM_EVICT_HEADROOM_GB = 4.0

    def __init__(
        self,
        strategy: str = "auto",
        group_offload_use_stream: bool = True,
        group_offload_low_cpu_mem: bool = True,
        block_pin_auto_evict: bool = True,
    ) -> None:
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
        (encode → denoise loop → decode), that's two extra transfers
        total — typically 1–2 s on PCIe 4 — in exchange for freeing
        several GiB of VRAM during decode.
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
            except TypeError:
                logger.debug(
                    "register_components: source %s is not weakref-able; "
                    "auto-cleanup on GC won't fire — caller must call "
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
                        "unregister_components: skipping %r — slot was displaced by another source",
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
            logger.info("load_component: cache hit for identifier %r → %r", identifier, name)
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
                logger.info("load_component: cache miss for identifier %r, loaded → %r", identifier, name)

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

    def _detect_available_vram_gb(self, device: torch.device | str) -> tuple[float, float]:
        """Return ``(available_gb, total_gb)`` of VRAM on *device*.

        Uses ``torch.cuda.mem_get_info`` so the answer reflects whatever
        else is already allocated on the GPU — the CUDA context, other
        PyTorch tensors, other processes sharing the device. Returns
        ``(0.0, 0.0)`` on failure (non-CUDA, driver issue, etc.).
        """
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            return free_bytes / (1024**3), total_bytes / (1024**3)
        except Exception as e:
            logger.warning("auto: VRAM detection failed (%s)", e)
            return 0.0, 0.0

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

    def resolve_offload_strategy(self, device: torch.device | str) -> str:
        """Resolve ``"auto"`` to a concrete strategy based on hardware + workload.

        Uses **available** VRAM and RAM at decision time (not total) so
        the answer reflects whatever else is on the system or GPU when
        ``managed()`` is called: another process holding GPU memory, the
        CUDA context overhead, the pipeline weights already mmap'd into
        the page cache, etc. Component sizes come from registered
        ``nn.Module`` parameters/buffers.

        Decision rule:

        - If pipeline weights × ``AUTO_NO_OFFLOAD_FACTOR`` ≤ available VRAM →
          ``no_offload`` (everything fits on GPU with activation headroom).
        - Else if largest component × ``AUTO_MODEL_OFFLOAD_FACTOR`` ≤
          available VRAM → ``model_offload`` (one component swaps onto GPU
          at a time).
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
                "auto: vram=%.1f / %.1f GB (no components yet) → %s",
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

        if weights_gb * self.AUTO_NO_OFFLOAD_FACTOR <= vram_avail_gb:
            chosen = "no_offload"
        elif max_component_gb * self.AUTO_MODEL_OFFLOAD_FACTOR <= vram_avail_gb:
            chosen = "model_offload"
        elif self._largest_component_has_block_list():
            # Largest component (typically the transformer) won't fit
            # under model_offload, but it has a long enough repeated-block
            # list to do better than plain leaf-level streaming: pin as
            # many blocks as VRAM allows, stream the rest. Components
            # without a block list fall back to plain group_offload at
            # apply time, so this is safe even in mixed pipelines.
            chosen = "block_pin"
        else:
            chosen = "group_offload"

        logger.info(
            "auto: vram=%.1f / %.1f GB, ram=%.1f / %.1f GB, pipeline=%.1f GB (largest component %.1f GB) → %s",
            vram_avail_gb,
            vram_total_gb,
            ram_avail_gb,
            ram_total_gb,
            weights_gb,
            max_component_gb,
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

    def set_block_pin_count(self, component_name: str, count: int) -> None:
        """Override the number of blocks to pin on GPU for *component_name*.

        Used only when the active strategy is ``"block_pin"``. Names without
        an override get an auto-computed value from available VRAM at apply
        time. ``count=0`` is valid — it means "pin nothing for this
        component, just stream it" (effectively per-block ``group_offload``).
        """
        if int(count) < 0:
            raise ValueError("block_pin count must be >= 0")
        with self._lock:
            self._block_pin_counts[component_name] = int(count)

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

    def _resolve_working_set_gb(self) -> float:
        """Return the platform-appropriate working-set margin.

        Windows uses the higher constant because ``expandable_segments``
        is Linux-only and the Windows allocator reserves ~2 GiB more
        under the same load. See the class-level constants for the full
        rationale.
        """
        if sys.platform == "win32":
            return self.AUTO_BLOCK_PIN_WORKING_SET_WINDOWS_GB
        return self.AUTO_BLOCK_PIN_WORKING_SET_GB

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
        working_set_gb = self._resolve_working_set_gb()

        budget_gb = vram_avail_gb - non_block_gb - working_set_gb - per_block_gb
        if budget_gb <= 0:
            logger.warning(
                "block_pin: %s — no VRAM budget for pinning (avail=%.1f, non_block=%.1f, "
                "working_set=%.1f, streamed_in_flight=%.2f) → 0 pinned, all blocks stream",
                component_name,
                vram_avail_gb,
                non_block_gb,
                working_set_gb,
                per_block_gb,
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
        fragmentation can swallow 1–2 GiB and turn a careful budget into
        an OOM. Logged as a one-time hint when the strategy is applied.

        Windows is skipped — ``expandable_segments`` depends on the CUDA
        virtual memory management API not exposed on the Windows driver,
        so the env var is a silent no-op there. The Windows working-set
        constant already accounts for the larger allocator overhead.
        """
        import os

        if sys.platform == "win32":
            return
        conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments:True" not in conf:
            logger.warning(
                "block_pin: PYTORCH_CUDA_ALLOC_CONF does not include "
                "'expandable_segments:True'. Allocator fragmentation may "
                "consume ~1-2 GiB and cause OOM. Recommended: set "
                "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before "
                "starting Python."
            )

    def _largest_component_has_block_list(self) -> bool:
        """True if the largest registered component has a usable block list.

        "Usable" = at least :attr:`AUTO_BLOCK_PIN_MIN_BLOCKS` entries.
        Below that threshold, per-block ``apply_group_offloading``
        overhead outweighs the benefit and plain ``group_offload`` is
        a better default.
        """
        with self._lock:
            components = list(self._managed_components.values())
        if not components:
            return False

        seen_ids: set[int] = set()
        largest: nn.Module | None = None
        largest_size = 0
        for mod in components:
            if id(mod) in seen_ids:
                continue
            seen_ids.add(id(mod))
            try:
                size = sum(p.numel() * p.element_size() for p in mod.parameters())
            except Exception:
                continue
            if size > largest_size:
                largest_size = size
                largest = mod
        if largest is None:
            return False

        result = find_largest_block_list(largest)
        if result is None:
            return False
        _, blocks = result
        return len(blocks) >= self.AUTO_BLOCK_PIN_MIN_BLOCKS

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
        3. **Runtime VRAM check** — query ``mem_get_info`` for currently
           free VRAM. If it's at or above the working-set margin the
           auto-budget reserved (``_resolve_working_set_gb``), the neighbor
           fits without evicting pinned: skip. Otherwise something has
           consumed more than expected and we'd *want* to evict.
           If VRAM detection fails (``free=0.0``) we fail safe and evict.
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
        free_gb, _ = self._detect_available_vram_gb(sample_device)
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
        free_gb, _ = self._detect_available_vram_gb(sample_device)
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
            for name, mod in unique:
                # Pre-clean any prior hooks so re-applying is safe (idempotent).
                remove_offload_hooks(mod)
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
                        "block_pin: %s — pinned %d/%d %s blocks, streaming %d",
                        name,
                        applied_n,
                        len(blocks),
                        block_attr,
                        len(blocks) - applied_n,
                    )
                    new_pinned.append((name, mod, block_attr, applied_n))
                except Exception as e:
                    logger.warning("block_pin: failed for %s: %s", name, e)

            self._install_block_pin_auto_evict(new_pinned, new_neighbors, device_obj)

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
          on the device that PyTorch doesn't manage. On a typical run
          this is CUDA context (~1 GiB) plus cuDNN workspaces. If it's
          way bigger than that, suspect a non-PyTorch allocator (Triton,
          cuFFT plans, etc.).

        Crucially, this is the *dedicated* side only. On Windows, Task
        Manager's "GPU Memory" line is dedicated + shared. Shared GPU
        memory is CUDA-pinned host memory (system RAM mapped to GPU
        address space) — created by ``apply_group_offloading`` when
        ``low_cpu_mem_usage=True`` so streamed weights can transfer
        without per-call pinning. That's not directly queryable from
        CUDA, so it's not in this breakdown — but it's the most common
        explanation when this method says "PyTorch peak is 9 GiB" while
        Task Manager says "21 GiB".

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
            self._evict_on_neighbor.clear()
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass
