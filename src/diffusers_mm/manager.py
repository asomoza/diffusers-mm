"""Core ModelManager — thread-safe model lifecycle and offload strategy management."""

from __future__ import annotations

import contextlib
import contextvars
import gc
import hashlib
import logging
import threading
from collections.abc import Generator
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

    def register_component(self, name: str, module: Any) -> None:
        """Register a named nn.Module component for lifecycle management.

        Re-registering the same ``(name, module)`` pair is a no-op — useful
        when a pipeline is recreated against the same long-lived manager
        and re-declares its components.

        Re-registering a *different* module under an existing name displaces
        the previous module. If that displaced module had hook-based
        offloading applied (``group_offload`` / ``sequential_group_offload``)
        and isn't aliased under another name in the registry, its hooks are
        removed before the reference is dropped — otherwise the registry
        would silently leak hooks attached to a module it can no longer
        clean up. Per-component strategy state for *name* is reset so the
        new module gets re-hooked on the next ``apply_offload_strategy``
        call. The global applied-strategy state is left untouched so other
        components keep their hooks.

        Adding a brand-new name leaves the slot pending: only the new
        component will be touched on the next apply call.
        """
        with self._lock:
            existing = self._managed_components.get(name)
            if existing is module:
                return

            if existing is not None:
                old_strategy = self._component_strategies.get(name)
                if old_strategy in ("group_offload", "sequential_group_offload"):
                    still_aliased = any(
                        other_name != name and other_module is existing
                        for other_name, other_module in self._managed_components.items()
                    )
                    if not still_aliased:
                        try:
                            remove_offload_hooks(existing)
                            logger.info(
                                "Cleaned offload hooks on displaced module under name %r (was %s)",
                                name,
                                old_strategy,
                            )
                        except Exception as e:
                            logger.warning("Failed to clean hooks on displaced module under %r: %s", name, e)

            self._managed_components[name] = module
            self._component_strategies.pop(name, None)

    def register_components(self, source: Any) -> list[str]:
        """Register every ``nn.Module`` exposed by *source*.

        *source* may be a ``DiffusionPipeline``-like object that exposes a
        ``components`` dict, or a plain ``dict[str, nn.Module]``. Returns
        the list of names that were registered (skipping non-modules).
        """
        if isinstance(source, dict):
            components = source
        elif hasattr(source, "components") and isinstance(source.components, dict):
            components = source.components
        else:
            raise TypeError(
                f"register_components expected a pipeline (with .components) or a dict, got {type(source).__name__}"
            )

        registered: list[str] = []
        for name, comp in components.items():
            if isinstance(comp, nn.Module):
                self.register_component(name, comp)
                registered.append(name)
        return registered

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
        """Free all components, caches, and CUDA memory."""
        with self._lock:
            self._component_cache.clear()
            self._managed_components.clear()
            self._component_strategies.clear()
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
