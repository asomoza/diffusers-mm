"""Core ModelManager — thread-safe model lifecycle and offload strategy management."""

from __future__ import annotations

import contextlib
import contextvars
import gc
import hashlib
import logging
import threading
from collections.abc import Callable, Generator
from typing import Any

import torch
from torch import nn

from diffusers_mm.hooks import remove_offload_hooks


logger = logging.getLogger(__name__)

OFFLOAD_STRATEGIES = ("auto", "no_offload", "model_offload", "sequential_group_offload", "group_offload")

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

    def __init__(
        self,
        strategy: str = "auto",
        group_offload_use_stream: bool = False,
        group_offload_low_cpu_mem: bool = False,
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
        self._applied_strategy: str | None = None

        self._offload_strategy: str = "auto"
        self.offload_strategy = strategy  # validate through setter
        self._group_offload_use_stream: bool = group_offload_use_stream
        self._group_offload_low_cpu_mem: bool = group_offload_low_cpu_mem

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
            return list(registered.keys())

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

    def resolve_offload_strategy(self, device: torch.device | str) -> str:
        """Resolve ``"auto"`` to a concrete strategy based on available VRAM."""
        strategy = self.offload_strategy
        if strategy != "auto":
            return strategy

        device = torch.device(device) if isinstance(device, str) else device
        if device.type != "cuda":
            return "group_offload"

        try:
            total_mem = torch.cuda.get_device_properties(device).total_mem
            total_gb = total_mem / (1024**3)
        except Exception:
            return "group_offload"

        if total_gb >= 20:
            return "no_offload"
        if total_gb >= 12:
            return "model_offload"
        if total_gb >= 8:
            return "sequential_group_offload"
        return "group_offload"

    def prepare_strategy_transition(self, new_strategy: str, device: torch.device | str) -> None:
        """Clean up the old offload strategy before applying *new_strategy*.

        Iterates components deduplicated by ``id(module)`` so that aliased
        names (the same module registered under multiple names) only get
        cleaned up once. Clears per-component strategy state so every
        component will be re-applied under the new strategy.
        """
        with self._lock:
            old = self._applied_strategy
            if old == new_strategy:
                return

            seen_ids: set[int] = set()
            for name, mod in self._managed_components.items():
                if id(mod) in seen_ids:
                    continue
                seen_ids.add(id(mod))
                if old in ("group_offload", "sequential_group_offload"):
                    remove_offload_hooks(mod)
                    if hasattr(mod, "to"):
                        mod.to("cpu")
                    logger.debug("Removed offload hooks from %s, moved to CPU", name)
                elif old in ("no_offload", "model_offload"):
                    if hasattr(mod, "to"):
                        mod.to("cpu")
                        logger.debug("Moved %s to CPU (leaving %s)", name, old)

            self._component_strategies.clear()
            self._applied_strategy = new_strategy

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _group_offload_kwargs(self, device: torch.device | str) -> dict[str, Any]:
        """Build kwargs for ``apply_group_offloading``."""
        use_stream = self._group_offload_use_stream
        kwargs: dict[str, Any] = {
            "onload_device": torch.device(device) if isinstance(device, str) else device,
            "offload_device": torch.device("cpu"),
            "offload_type": "leaf_level",
            "use_stream": use_stream,
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
            for name, mod in unique:
                try:
                    remove_offload_hooks(mod)
                    apply_group_offloading(mod, **offload_kwargs)
                    logger.info("Group offload enabled for %s", name)
                except Exception as e:
                    logger.warning("Failed to enable group offload for %s: %s", name, e)

        elif strategy == "sequential_group_offload":
            for name, mod in unique:
                if hasattr(mod, "to"):
                    mod.to("cpu")
            logger.info("sequential_group_offload: %d component(s) on CPU, hooks deferred", len(unique))

        elif strategy == "model_offload":
            logger.info("model_offload: %d component(s) remain on CPU", len(unique))

        with self._lock:
            for name, _ in unique:
                self._component_strategies[name] = strategy
            for name in aliases:
                self._component_strategies[name] = strategy

        return strategy

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

        - ``no_offload`` / ``group_offload``: no-op yield.
        - ``model_offload``: bulk CPU <-> GPU move.
        - ``sequential_group_offload``: apply group-offload hooks on enter,
          remove hooks + move to CPU on exit.

        *strategy_override* lets callers force a different strategy for these
        components (e.g. ``"model_offload"`` for small models like VAE that
        are too granular for leaf-level hook offloading).
        """
        strategy = self._applied_strategy
        if strategy is None:
            strategy = self.apply_offload_strategy(device)
        if strategy_override is not None:
            strategy = strategy_override

        if strategy in ("no_offload", "group_offload"):
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

        elif strategy == "sequential_group_offload":
            from diffusers.hooks.group_offloading import apply_group_offloading

            offload_kwargs = self._group_offload_kwargs(device)
            for name, mod in modules:
                try:
                    apply_group_offloading(mod, **offload_kwargs)
                    logger.debug("sequential_group_offload: hooks applied to %s", name)
                except Exception as e:
                    logger.warning("sequential_group_offload: failed to hook %s: %s", name, e)
            try:
                yield
            finally:
                for name, mod in modules:
                    remove_offload_hooks(mod)
                    if hasattr(mod, "to"):
                        mod.to("cpu")
                    logger.debug("sequential_group_offload: cleaned up %s", name)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        else:
            yield

    # ------------------------------------------------------------------
    # Debugging
    # ------------------------------------------------------------------

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
            self._component_cache.clear()
            self._managed_components.clear()
            self._component_strategies.clear()
            self._refcount.clear()
            self._source_registrations.clear()
            self._applied_strategy = None
            self._group_offload_use_stream = False
            self._group_offload_low_cpu_mem = False
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass
