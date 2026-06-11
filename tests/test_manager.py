"""Tests for ModelManager core functionality."""

from __future__ import annotations

import gc
import threading

import pytest
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
from diffusers_mm.manager import ModelManager, get_device, get_dtype


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)


class TestModelManagerInit:
    def test_default_strategy(self):
        mm = ModelManager()
        assert mm.offload_strategy == "auto"
        assert mm.applied_strategy is None

    def test_explicit_strategy(self):
        mm = ModelManager(strategy="no_offload")
        assert mm.offload_strategy == "no_offload"

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown offload strategy"):
            ModelManager(strategy="bogus")

    def test_group_offload_default_options(self):
        # Defaults: use_stream + low_cpu_mem ON. These match the "fast,
        # RAM-conservative" leaf-level group offload that diffusers
        # recommends — and avoid the 2× host-RAM trap of use_stream=True
        # without low_cpu_mem_usage.
        mm = ModelManager()
        assert mm.group_offload_use_stream is True
        assert mm.group_offload_low_cpu_mem is True

    def test_group_offload_options_overrides(self):
        mm = ModelManager(
            group_offload_use_stream=False,
            group_offload_low_cpu_mem=False,
        )
        assert mm.group_offload_use_stream is False
        assert mm.group_offload_low_cpu_mem is False


class TestAutoTuningCtorArgs:
    """The ``auto_*`` keyword-only args shadow the class constants per instance.

    Reads in the resolver go through ``self.AUTO_X``, so a non-``None``
    ctor arg sets the instance attribute and Python's normal lookup
    picks it up. ``None`` leaves the class default in place.
    """

    def test_defaults_unset_so_class_constants_apply(self):
        mm = ModelManager()
        # Reading ``mm.AUTO_X`` should reach the class constant — confirm
        # by checking it equals the documented default.
        assert mm.AUTO_NO_OFFLOAD_FACTOR == 1.5
        assert mm.AUTO_MODEL_OFFLOAD_FACTOR == 1.5
        assert mm.AUTO_RAM_HEADROOM == 0.85
        assert mm.AUTO_LOW_CPU_MEM_RAM_HEADROOM_GB == 16.0
        assert mm.AUTO_BLOCK_PIN_WORKING_SET_GB == 6.5
        assert mm.AUTO_BLOCK_PIN_WORKING_SET_WINDOWS_GB == 8.5
        assert mm.AUTO_BLOCK_PIN_MIN_BLOCKS == 8
        assert mm.AUTO_BLOCK_PIN_RAM_EVICT_HEADROOM_GB == 4.0
        # No instance attribute shadow exists when defaults were used.
        assert "AUTO_NO_OFFLOAD_FACTOR" not in mm.__dict__
        assert "AUTO_BLOCK_PIN_WORKING_SET_GB" not in mm.__dict__

    def test_ctor_args_shadow_class_constants(self):
        mm = ModelManager(
            auto_no_offload_factor=2.0,
            auto_model_offload_factor=1.8,
            auto_ram_headroom=0.9,
            auto_low_cpu_mem_ram_headroom_gb=24.0,
            auto_block_pin_working_set_gb=12.0,
            auto_block_pin_working_set_windows_gb=14.0,
            auto_block_pin_min_blocks=12,
            auto_block_pin_ram_evict_headroom_gb=6.0,
        )
        assert mm.AUTO_NO_OFFLOAD_FACTOR == 2.0
        assert mm.AUTO_MODEL_OFFLOAD_FACTOR == 1.8
        assert mm.AUTO_RAM_HEADROOM == 0.9
        assert mm.AUTO_LOW_CPU_MEM_RAM_HEADROOM_GB == 24.0
        assert mm.AUTO_BLOCK_PIN_WORKING_SET_GB == 12.0
        assert mm.AUTO_BLOCK_PIN_WORKING_SET_WINDOWS_GB == 14.0
        assert mm.AUTO_BLOCK_PIN_MIN_BLOCKS == 12
        assert mm.AUTO_BLOCK_PIN_RAM_EVICT_HEADROOM_GB == 6.0
        # Each override should produce an instance attribute (verifies
        # we're shadowing, not mutating the class).
        assert "AUTO_NO_OFFLOAD_FACTOR" in mm.__dict__
        assert ModelManager.AUTO_NO_OFFLOAD_FACTOR == 1.5  # class untouched

    def test_min_blocks_coerced_to_int(self):
        # AUTO_BLOCK_PIN_MIN_BLOCKS is the only int constant — confirm
        # we coerce floats to int rather than silently leaving them.
        mm = ModelManager(auto_block_pin_min_blocks=10.7)
        assert mm.AUTO_BLOCK_PIN_MIN_BLOCKS == 10
        assert isinstance(mm.AUTO_BLOCK_PIN_MIN_BLOCKS, int)

    def test_live_mutation_via_attribute_assignment_still_works(self):
        # Documented escape hatch: mutate the constant after construction.
        # Must continue working regardless of whether the ctor arg was used.
        mm = ModelManager()
        mm.AUTO_BLOCK_PIN_WORKING_SET_GB = 10.0
        assert mm.AUTO_BLOCK_PIN_WORKING_SET_GB == 10.0
        # And on top of an existing ctor override:
        mm2 = ModelManager(auto_block_pin_working_set_gb=12.0)
        mm2.AUTO_BLOCK_PIN_WORKING_SET_GB = 15.0
        assert mm2.AUTO_BLOCK_PIN_WORKING_SET_GB == 15.0

    def test_ctor_arg_affects_resolver_decision(self):
        # End-to-end check: raising AUTO_NO_OFFLOAD_FACTOR should push the
        # auto-resolver past the no_offload threshold even when the
        # default factor would have allowed it.
        mm = ModelManager(strategy="auto", auto_no_offload_factor=4.0)
        # No components registered → resolver falls back to the simple
        # VRAM-only tier table, which doesn't use the factor. Register a
        # tiny component so the size-aware path runs instead.
        mm.register_component("tiny", DummyModel())

        # Patch the env so the size-aware path is reached with plenty
        # of VRAM/RAM, but the factored threshold is the binding rule.
        mm._detect_available_vram_gb = lambda device: (10.0, 10.0)
        mm._detect_available_ram_gb = lambda: (64.0, 64.0)
        mm._estimate_components_size_gb = lambda: (3.0, 3.0)
        # weights=3.0 GiB, factor=4.0 → required 12.0 GiB; VRAM=10.0 → fail tier 1.
        # Without our override (factor=1.5 → required 4.5 GiB) tier 1 would pass.
        resolved = mm.resolve_offload_strategy("cuda")
        assert resolved != "no_offload"


class TestGroupOffloadKwargs:
    """Validate the kwargs builder's output for each setting combination."""

    def test_leaf_level_with_defaults(self):
        mm = ModelManager()
        kwargs = mm._group_offload_kwargs("cpu")
        assert kwargs["offload_type"] == "leaf_level"
        assert kwargs["use_stream"] is True
        assert kwargs["record_stream"] is False  # always hardcoded False
        assert kwargs["low_cpu_mem_usage"] is True  # gated on use_stream

    def test_low_cpu_mem_dropped_when_streams_off(self):
        mm = ModelManager(group_offload_use_stream=False, group_offload_low_cpu_mem=True)
        kwargs = mm._group_offload_kwargs("cpu")
        assert kwargs["use_stream"] is False
        # low_cpu_mem is documented as only honoured when use_stream=True.
        assert "low_cpu_mem_usage" not in kwargs

    def test_record_stream_always_false(self):
        # record_stream is hardcoded — there's no constructor param for it
        # because some models produce numerical noise with it on, and the
        # speed gain isn't worth the configurability.
        mm = ModelManager()
        assert mm._group_offload_kwargs("cpu")["record_stream"] is False


class TestComponentRegistration:
    def test_register_and_get(self):
        mm = ModelManager()
        model = DummyModel()
        mm.register_component("test_model", model)
        assert mm.get_component("test_model") is model

    def test_get_missing_returns_none(self):
        mm = ModelManager()
        assert mm.get_component("nonexistent") is None

    def test_component_names(self):
        mm = ModelManager()
        mm.register_component("a", DummyModel())
        mm.register_component("b", DummyModel())
        assert sorted(mm.component_names) == ["a", "b"]

    def test_register_new_component_keeps_global_strategy(self):
        # New behaviour: adding a component does not invalidate already-applied
        # strategy on existing components. Only the new slot is marked pending.
        mm = ModelManager(strategy="no_offload")
        existing = DummyModel()
        mm.register_component("existing", existing)
        mm.apply_offload_strategy("cpu")
        assert mm.applied_strategy == "no_offload"
        assert mm._component_strategies["existing"] == "no_offload"

        new = DummyModel()
        mm.register_component("new", new)
        # Global state untouched; only the new component is pending.
        assert mm.applied_strategy == "no_offload"
        assert mm._component_strategies.get("existing") == "no_offload"
        assert "new" not in mm._component_strategies

    def test_register_same_component_is_noop(self):
        mm = ModelManager(strategy="no_offload")
        model = DummyModel()
        mm.register_component("model", model)
        mm.apply_offload_strategy("cpu")
        before = dict(mm._component_strategies)
        mm.register_component("model", model)  # same name + same module
        assert mm._component_strategies == before

    def test_register_different_module_under_existing_name_resets_only_that_slot(self):
        mm = ModelManager(strategy="no_offload")
        a = DummyModel()
        b = DummyModel()
        mm.register_component("a", a)
        mm.register_component("b", b)
        mm.apply_offload_strategy("cpu")
        assert mm._component_strategies == {"a": "no_offload", "b": "no_offload"}

        replacement = DummyModel()
        mm.register_component("a", replacement)
        # Only "a" is now pending; "b" keeps its applied state.
        assert "a" not in mm._component_strategies
        assert mm._component_strategies["b"] == "no_offload"
        assert mm.applied_strategy == "no_offload"


class TestDisplacedModuleHookCleanup:
    """Replacing a module under an existing name shouldn't orphan its hooks."""

    def test_displacing_hooked_module_removes_hooks(self, monkeypatch):
        cleaned: list = []
        monkeypatch.setattr("diffusers_mm.manager.remove_offload_hooks", lambda m: cleaned.append(m))

        mm = ModelManager(strategy="group_offload")
        old = DummyModel()
        mm.register_component("transformer", old)
        # Simulate post-apply state without going through real diffusers hooks
        # (which need CUDA): we just claim the strategy was applied.
        mm._component_strategies["transformer"] = "group_offload"

        new = DummyModel()
        mm.register_component("transformer", new)

        assert cleaned == [old]

    def test_displacing_aliased_module_leaves_hooks_intact(self, monkeypatch):
        # The displaced module is still reachable via another name in the
        # registry, so its hooks are still meaningful — don't clean them up.
        cleaned: list = []
        monkeypatch.setattr("diffusers_mm.manager.remove_offload_hooks", lambda m: cleaned.append(m))

        mm = ModelManager(strategy="group_offload")
        shared = DummyModel()
        mm.register_component("primary", shared)
        mm.register_component("alias", shared)
        mm._component_strategies = {"primary": "group_offload", "alias": "group_offload"}

        new = DummyModel()
        mm.register_component("primary", new)

        assert shared not in cleaned

    def test_displacing_calls_remove_at_refcount_zero(self, monkeypatch):
        # Under refcount-based cleanup, remove_offload_hooks is called
        # whenever a module's refcount hits 0 — regardless of which
        # strategy was previously applied. remove_offload_hooks is
        # documented as idempotent so this is safe; the call is
        # unconditional in the cleanup path.
        # Flush any pending weakref.finalize callbacks from prior tests
        # before installing the monkeypatch — apply_offload_strategy()
        # below runs gc.collect() inside prepare_strategy_transition,
        # which would otherwise fire those finalizers and route their
        # cleanup calls into our patched recorder.
        gc.collect()
        cleaned: list = []
        monkeypatch.setattr("diffusers_mm.manager.remove_offload_hooks", lambda m: cleaned.append(m))

        mm = ModelManager(strategy="no_offload")
        old = DummyModel()
        mm.register_component("model", old)
        mm.apply_offload_strategy("cpu")

        new = DummyModel()
        mm.register_component("model", new)

        # old's refcount went 1 → 0 on displacement, so cleanup ran.
        # Use membership rather than equality so any unrelated finalizer
        # cleanups that happen to slip through don't make the assertion
        # brittle to test ordering.
        assert old in cleaned
        assert new not in cleaned


class TestUnregisterComponent:
    def test_returns_true_when_present(self):
        mm = ModelManager()
        mm.register_component("model", DummyModel())
        assert mm.unregister_component("model") is True
        assert mm.get_component("model") is None
        assert mm.component_names == []

    def test_returns_false_when_missing(self):
        mm = ModelManager()
        assert mm.unregister_component("nonexistent") is False

    def test_drops_per_component_strategy_state(self):
        mm = ModelManager(strategy="no_offload")
        mm.register_component("model", DummyModel())
        mm.apply_offload_strategy("cpu")
        assert mm._component_strategies["model"] == "no_offload"
        mm.unregister_component("model")
        assert "model" not in mm._component_strategies

    def test_leaves_global_applied_strategy_untouched(self):
        # Other components keep their state; the manager's global strategy
        # remains so future applies don't trigger transitions.
        mm = ModelManager(strategy="no_offload")
        mm.register_component("a", DummyModel())
        mm.register_component("b", DummyModel())
        mm.apply_offload_strategy("cpu")

        mm.unregister_component("a")
        assert mm.applied_strategy == "no_offload"
        assert mm._component_strategies == {"b": "no_offload"}

    def test_cleans_hooks_for_hooked_strategy(self, monkeypatch):
        cleaned: list = []
        monkeypatch.setattr("diffusers_mm.manager.remove_offload_hooks", lambda m: cleaned.append(m))

        mm = ModelManager(strategy="group_offload")
        m = DummyModel()
        mm.register_component("transformer", m)
        # Simulate post-apply state without going through real diffusers hooks.
        mm._component_strategies["transformer"] = "group_offload"

        mm.unregister_component("transformer")
        assert cleaned == [m]

    def test_skips_cleanup_for_aliased_module(self, monkeypatch):
        cleaned: list = []
        monkeypatch.setattr("diffusers_mm.manager.remove_offload_hooks", lambda m: cleaned.append(m))

        mm = ModelManager(strategy="group_offload")
        shared = DummyModel()
        mm.register_component("primary", shared)
        mm.register_component("alias", shared)
        mm._component_strategies = {"primary": "group_offload", "alias": "group_offload"}

        mm.unregister_component("primary")
        # The alias still exposes the module — its hooks are still meaningful.
        assert shared not in cleaned
        assert mm.get_component("alias") is shared

    def test_unregister_calls_remove_at_refcount_zero_regardless_of_strategy(self, monkeypatch):
        # Under refcount-based cleanup, remove_offload_hooks is called
        # whenever refcount hits 0 — even for strategies that didn't
        # install hooks. The call is idempotent so this is safe.
        cleaned: list = []
        monkeypatch.setattr("diffusers_mm.manager.remove_offload_hooks", lambda m: cleaned.append(m))

        mm = ModelManager(strategy="no_offload")
        m = DummyModel()
        mm.register_component("model", m)
        mm.apply_offload_strategy("cpu")

        mm.unregister_component("model")
        assert cleaned == [m]


class TestRefcount:
    """Refcount-based lifecycle: shared modules survive partial unregister."""

    def test_register_component_always_increments(self):
        mm = ModelManager()
        m = DummyModel()
        mm.register_component("a", m)
        assert mm._refcount[id(m)] == 1
        mm.register_component("a", m)  # same name + same module
        assert mm._refcount[id(m)] == 2

    def test_unregister_decrements_keeps_slot_alive_above_zero(self):
        mm = ModelManager(strategy="no_offload")
        m = DummyModel()
        mm.register_component("a", m)
        mm.register_component("a", m)  # refcount = 2
        mm.apply_offload_strategy("cpu")

        mm.unregister_component("a")
        # refcount = 1 → slot stays
        assert mm.get_component("a") is m
        assert mm._refcount[id(m)] == 1

        mm.unregister_component("a")
        # refcount = 0 → cleanup
        assert mm.get_component("a") is None
        assert id(m) not in mm._refcount

    def test_aliases_in_one_source_clean_up_both_slots(self, monkeypatch):
        cleaned: list = []
        monkeypatch.setattr("diffusers_mm.manager.remove_offload_hooks", lambda m: cleaned.append(m))

        mm = ModelManager()
        m = DummyModel()
        mm.register_component("primary", m)
        mm.register_component("alias", m)  # refcount = 2

        mm.unregister_component("primary")  # refcount = 1, slot "primary" stays
        assert mm.get_component("primary") is m
        assert mm.get_component("alias") is m

        mm.unregister_component("alias")  # refcount = 0, full cleanup
        # Both slots gone, hooks cleaned (idempotent — no actual hooks here).
        assert mm.get_component("primary") is None
        assert mm.get_component("alias") is None
        assert cleaned == [m]

    def test_displacement_of_last_reference_evicts_cache(self):
        mm = ModelManager()
        m1 = DummyModel()
        mm.load_component("transformer", "id-1", lambda: m1)
        cache_key_1 = ModelManager.component_hash("id-1")
        assert mm.get_cached(cache_key_1) is m1

        m2 = DummyModel()
        mm.register_component("transformer", m2)  # displaces m1, refcount(m1) → 0
        # m1's cache entry evicted as part of refcount-zero cleanup.
        assert mm.get_cached(cache_key_1) is None
        assert mm.get_component("transformer") is m2


class TestRegisterComponentsSource:
    """Per-source idempotency and bulk lifecycle."""

    def _make_pipe(self, **comps):
        class FakePipe:
            pass

        p = FakePipe()
        p.components = comps
        return p

    def test_register_components_is_idempotent_per_source(self):
        mm = ModelManager()
        m = DummyModel()
        pipe = self._make_pipe(transformer=m)

        mm.register_components(pipe)
        assert mm._refcount[id(m)] == 1

        mm.register_components(pipe)  # same pipe — no double counting
        assert mm._refcount[id(m)] == 1

    def test_two_sources_sharing_a_module_both_count(self):
        mm = ModelManager()
        shared = DummyModel()
        pipe1 = self._make_pipe(text_encoder=shared)
        pipe2 = self._make_pipe(text_encoder=shared)

        mm.register_components(pipe1)
        mm.register_components(pipe2)
        assert mm._refcount[id(shared)] == 2

    def test_unregister_components_only_decrements_its_source(self):
        # pipe1 and pipe2 both use shared T5. pipe1 going away leaves
        # T5 in the registry for pipe2.
        mm = ModelManager()
        shared = DummyModel()
        pipe1 = self._make_pipe(text_encoder=shared)
        pipe2 = self._make_pipe(text_encoder=shared)
        mm.register_components(pipe1)
        mm.register_components(pipe2)

        mm.unregister_components(pipe1)
        assert mm.get_component("text_encoder") is shared
        assert mm._refcount[id(shared)] == 1

        mm.unregister_components(pipe2)
        assert mm.get_component("text_encoder") is None
        assert id(shared) not in mm._refcount

    def test_unregister_components_skips_displaced_slots(self):
        # pipe1 registers Tx1; pipe2 displaces with Tx2.
        # When pipe1 later unregisters, the stale "transformer → Tx1"
        # entry in pipe1's record must be skipped (Tx1 was already
        # cleaned up at displacement time).
        mm = ModelManager()
        Tx1 = DummyModel()
        Tx2 = DummyModel()
        pipe1 = self._make_pipe(transformer=Tx1)
        pipe2 = self._make_pipe(transformer=Tx2)

        mm.register_components(pipe1)
        mm.register_components(pipe2)
        # Tx1 was displaced → refcount(Tx1) hit 0 → cleaned up.
        assert id(Tx1) not in mm._refcount
        assert mm._refcount[id(Tx2)] == 1

        # pipe1's stale unregister must NOT touch Tx2.
        processed = mm.unregister_components(pipe1)
        assert processed == []
        assert mm._refcount[id(Tx2)] == 1
        assert mm.get_component("transformer") is Tx2

    def test_unregister_components_returns_empty_for_unknown_source(self):
        mm = ModelManager()
        pipe = self._make_pipe(transformer=DummyModel())
        # Never registered.
        assert mm.unregister_components(pipe) == []

    def test_unload_components_delegates_to_unregister(self):
        mm = ModelManager()
        m = DummyModel()
        pipe = self._make_pipe(transformer=m)
        mm.register_components(pipe)

        names = mm.unload_components(pipe)
        assert names == ["transformer"]
        assert mm.get_component("transformer") is None


class TestSourceWeakrefCleanup:
    """When a source object is GC'd, its components should auto-unregister
    so the user doesn't leak modules by forgetting unregister_components."""

    def _make_pipe(self, **comps):
        class FakePipe:
            pass

        p = FakePipe()
        p.components = comps
        return p

    def test_gc_of_source_releases_unique_modules(self):
        import gc as _gc

        mm = ModelManager()
        m = DummyModel()
        pipe = self._make_pipe(transformer=m)
        mm.register_components(pipe)
        assert mm._refcount[id(m)] == 1

        # Drop the only strong reference to pipe and force GC.
        del pipe
        _gc.collect()

        # The finalizer fired → record removed, refcount decremented to 0,
        # module fully cleaned up.
        assert mm._refcount == {}
        assert mm.get_component("transformer") is None
        assert mm._source_registrations == {}
        assert mm._source_finalizers == {}

    def test_gc_of_one_source_keeps_shared_module_alive(self):
        import gc as _gc

        mm = ModelManager()
        shared = DummyModel()
        pipe1 = self._make_pipe(text_encoder=shared)
        pipe2 = self._make_pipe(text_encoder=shared)
        mm.register_components(pipe1)
        mm.register_components(pipe2)
        assert mm._refcount[id(shared)] == 2

        # GC pipe1 only — pipe2 still keeps shared alive.
        del pipe1
        _gc.collect()

        assert mm.get_component("text_encoder") is shared
        assert mm._refcount[id(shared)] == 1

    def test_explicit_unregister_detaches_finalizer(self):
        # After explicit unregister_components, GCing the source must NOT
        # try to clean up again — the finalizer should have been detached.
        import gc as _gc

        mm = ModelManager()
        m = DummyModel()
        pipe = self._make_pipe(transformer=m)
        mm.register_components(pipe)
        mm.unregister_components(pipe)

        # Re-register a different module under the same name to set up
        # detection: if the (detached) finalizer fired, it would try to
        # touch state and we'd notice.
        m2 = DummyModel()
        pipe2 = self._make_pipe(transformer=m2)
        mm.register_components(pipe2)
        assert mm._refcount[id(m2)] == 1

        del pipe
        _gc.collect()

        # pipe's finalizer was detached at unregister, so this GC is a no-op.
        assert mm._refcount[id(m2)] == 1
        assert mm.get_component("transformer") is m2

    def test_clear_detaches_finalizers(self):
        import gc as _gc

        mm = ModelManager()
        pipe = self._make_pipe(transformer=DummyModel())
        mm.register_components(pipe)
        assert mm._source_finalizers != {}

        mm.clear()
        # Finalizers detached as part of clear.
        assert mm._source_finalizers == {}

        # GCing the source post-clear is a no-op (no callback to fire).
        del pipe
        _gc.collect()
        assert mm._source_registrations == {}

    def test_dict_source_skips_finalizer_silently(self):
        # Dicts can't be weakref'd. register_components should still work
        # — just no auto-cleanup.
        mm = ModelManager()
        m = DummyModel()
        d = {"transformer": m}
        mm.register_components(d)
        assert mm.get_component("transformer") is m
        # No finalizer registered for dict source.
        assert id(d) not in mm._source_finalizers
        # Explicit unregister still works.
        mm.unregister_components(d)
        assert mm.get_component("transformer") is None


class TestUnloadComponent:
    def test_returns_false_when_missing(self):
        mm = ModelManager()
        assert mm.unload_component("nonexistent") is False

    def test_removes_from_registry_and_cache(self):
        mm = ModelManager()
        m = DummyModel()
        mm.load_component("text_encoder", "id-1", lambda: m)
        cache_key = ModelManager.component_hash("id-1")
        assert mm.get_cached(cache_key) is m

        assert mm.unload_component("text_encoder") is True
        assert mm.get_component("text_encoder") is None
        assert mm.get_cached(cache_key) is None

    def test_unload_then_load_reruns_factory(self):
        # Symmetry test: after unload, the next load_component for the
        # same identifier must call the factory again (cache miss).
        mm = ModelManager()
        calls: list[bool] = []

        def factory():
            calls.append(True)
            return DummyModel()

        mm.load_component("text_encoder", "id-1", factory)
        assert len(calls) == 1
        mm.unload_component("text_encoder")
        mm.load_component("text_encoder", "id-1", factory)
        assert len(calls) == 2

    def test_unload_keeps_module_alive_when_aliased(self):
        # Under refcount semantics, the slot for the unloaded name stays
        # alive while another reference exists (the alias). Cache is
        # preserved alongside. Both slots will only go away when the
        # second alias is unloaded.
        mm = ModelManager()
        m = DummyModel()
        mm.load_component("primary", "id-1", lambda: m)
        mm.load_component("alias", "id-1", lambda: m)
        cache_key = ModelManager.component_hash("id-1")
        assert mm.get_cached(cache_key) is m

        mm.unload_component("primary")
        # refcount(m) went 2 → 1; slot "primary" stays, module survives,
        # cache survives.
        assert mm.get_component("primary") is m
        assert mm.get_component("alias") is m
        assert mm.get_cached(cache_key) is m

        # Unloading the second name drops refcount to 0 → full cleanup.
        mm.unload_component("alias")
        assert mm.get_component("primary") is None
        assert mm.get_component("alias") is None
        assert mm.get_cached(cache_key) is None

    def test_unload_cleans_hooks_for_hooked_strategy(self, monkeypatch):
        cleaned: list = []
        monkeypatch.setattr("diffusers_mm.manager.remove_offload_hooks", lambda m: cleaned.append(m))

        mm = ModelManager(strategy="group_offload")
        m = DummyModel()
        mm.load_component("transformer", "id-1", lambda: m)
        mm._component_strategies["transformer"] = "group_offload"

        mm.unload_component("transformer")
        assert cleaned == [m]


class TestRegisterComponents:
    def test_from_dict(self):
        mm = ModelManager()
        a, b = DummyModel(), DummyModel()
        registered = mm.register_components({"a": a, "b": b})
        assert sorted(registered) == ["a", "b"]
        assert mm.get_component("a") is a
        assert mm.get_component("b") is b

    def test_from_pipeline_like(self):
        class FakePipe:
            def __init__(self):
                self.components = {"transformer": DummyModel(), "vae": DummyModel(), "scheduler": "not_a_module"}

        mm = ModelManager()
        pipe = FakePipe()
        registered = mm.register_components(pipe)
        # Non-modules are silently skipped.
        assert sorted(registered) == ["transformer", "vae"]
        assert mm.get_component("scheduler") is None

    def test_invalid_source_raises(self):
        mm = ModelManager()
        with pytest.raises(TypeError, match="register_components expected"):
            mm.register_components(["not", "a", "dict"])

    def test_shared_module_across_pipelines(self):
        # Same nn.Module passed under the same name from two "pipelines"
        # should not cause a strategy reset on the second pass.
        mm = ModelManager(strategy="no_offload")
        shared = DummyModel()
        mm.register_components({"text_encoder": shared})
        mm.apply_offload_strategy("cpu")
        before = dict(mm._component_strategies)

        # Second pipeline registers the same shared module under the same name.
        mm.register_components({"text_encoder": shared})
        assert mm._component_strategies == before


class TestCache:
    def test_hash_deterministic(self):
        assert ModelManager.component_hash("foo") == ModelManager.component_hash("foo")

    def test_hash_differs(self):
        assert ModelManager.component_hash("foo") != ModelManager.component_hash("bar")

    def test_hash_length(self):
        assert len(ModelManager.component_hash("anything")) == 16

    def test_cache_miss(self):
        mm = ModelManager()
        assert mm.get_cached("missing") is None

    def test_cache_hit(self):
        mm = ModelManager()
        obj = {"data": 42}
        mm.set_cached("key", obj)
        assert mm.get_cached("key") is obj


class TestLoadComponent:
    def test_first_call_invokes_factory_and_caches(self):
        mm = ModelManager()
        calls: list[bool] = []
        m = DummyModel()

        def factory():
            calls.append(True)
            return m

        result = mm.load_component("text_encoder", "id-1", factory)
        assert result is m
        assert mm.get_component("text_encoder") is m
        assert mm.get_cached(ModelManager.component_hash("id-1")) is m
        assert len(calls) == 1

    def test_second_call_with_same_identifier_skips_factory(self):
        mm = ModelManager()
        calls: list[bool] = []
        m = DummyModel()

        def factory():
            calls.append(True)
            return m

        first = mm.load_component("text_encoder", "id-1", factory)
        second = mm.load_component("text_encoder", "id-1", factory)
        assert first is second is m
        assert len(calls) == 1

    def test_cache_hit_under_different_name_aliases_the_module(self):
        # Same identifier, different name → module is aliased under both
        # names. The factory must NOT be called the second time.
        mm = ModelManager()
        m = DummyModel()
        first = mm.load_component("primary", "id-1", lambda: m)

        def trap_factory():
            raise AssertionError("factory should not be called on cache hit")

        second = mm.load_component("alias", "id-1", trap_factory)
        assert first is second is m
        assert mm.get_component("primary") is m
        assert mm.get_component("alias") is m

    def test_factory_returning_non_module_raises(self):
        mm = ModelManager()
        with pytest.raises(TypeError, match="factory must return"):
            mm.load_component("text_encoder", "id-1", lambda: "not a module")

    def test_different_identifiers_are_kept_separate(self):
        mm = ModelManager()
        m1, m2 = DummyModel(), DummyModel()
        first = mm.load_component("a", "id-1", lambda: m1)
        second = mm.load_component("b", "id-2", lambda: m2)
        assert first is m1
        assert second is m2
        assert mm.get_component("a") is m1
        assert mm.get_component("b") is m2

    def test_cache_hit_treats_aliasing_correctly_under_apply_strategy(self):
        # The cached-module-via-alias case must integrate with the
        # apply-strategy / id-dedup machinery from earlier iterations.
        mm = ModelManager(strategy="no_offload")
        m = DummyModel()
        mm.load_component("primary", "id-1", lambda: m)
        mm.load_component("alias", "id-1", lambda: m)  # cache hit
        mm.apply_offload_strategy("cpu")
        assert mm._component_strategies == {"primary": "no_offload", "alias": "no_offload"}


class TestDeviceScope:
    def test_sets_and_resets_device(self):
        mm = ModelManager()
        assert get_device() is None
        with mm.device_scope(device="cpu"):
            assert get_device() == torch.device("cpu")
        assert get_device() is None

    def test_sets_and_resets_dtype(self):
        mm = ModelManager()
        assert get_dtype() is None
        with mm.device_scope(device="cpu", dtype=torch.bfloat16):
            assert get_dtype() == torch.bfloat16
        assert get_dtype() is None

    def test_nested_scopes(self):
        mm = ModelManager()
        with mm.device_scope(device="cpu", dtype=torch.float32):
            assert get_device() == torch.device("cpu")
            with mm.device_scope(device="meta", dtype=torch.bfloat16):
                assert get_device() == torch.device("meta")
                assert get_dtype() == torch.bfloat16
            assert get_device() == torch.device("cpu")
            assert get_dtype() == torch.float32


class TestResolveStrategy:
    def test_explicit_strategy_passes_through(self):
        mm = ModelManager(strategy="model_offload")
        assert mm.resolve_offload_strategy("cuda") == "model_offload"

    def test_auto_non_cuda_returns_group_offload(self):
        mm = ModelManager(strategy="auto")
        assert mm.resolve_offload_strategy("cpu") == "group_offload"


def _patch_vram(monkeypatch, available_gib: float, total_gib: float | None = None) -> None:
    """Pretend any cuda device has *available_gib* free / *total_gib* total VRAM.

    Patches ``torch.cuda.mem_get_info`` (the API the resolver actually uses
    now). If *total_gib* is omitted, total = available.
    """
    total = total_gib if total_gib is not None else available_gib
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda d=None: (int(available_gib * 1024**3), int(total * 1024**3)),
    )


class TestComponentSizeEstimation:
    def test_empty_registry(self):
        mm = ModelManager()
        total, max_size = mm._estimate_components_size_gb()
        assert total == 0.0
        assert max_size == 0.0

    def test_single_component(self):
        mm = ModelManager()
        m = DummyModel()
        mm.register_component("a", m)
        total, max_size = mm._estimate_components_size_gb()
        # DummyModel is small but non-zero.
        assert total > 0.0
        assert max_size == total  # only one component

    def test_aliases_counted_once(self):
        # Same module under two names — should count once toward total.
        mm = ModelManager()
        m = DummyModel()
        mm.register_component("primary", m)
        mm.register_component("alias", m)
        total, max_size = mm._estimate_components_size_gb()

        mm2 = ModelManager()
        mm2.register_component("only", m)
        total2, max_size2 = mm2._estimate_components_size_gb()

        assert total == total2
        assert max_size == max_size2

    def test_two_distinct_components_sum(self):
        mm = ModelManager()
        m1 = DummyModel()
        m2 = DummyModel()
        mm.register_component("a", m1)
        mm.register_component("b", m2)
        total, max_size = mm._estimate_components_size_gb()
        # Two equal-sized components → total = 2× max_size.
        assert total == pytest.approx(max_size * 2, rel=0.01)


class TestAutoResolutionSized:
    """Auto resolution should consider component sizes, not just VRAM tiers."""

    def test_huge_vram_picks_no_offload(self, monkeypatch):
        _patch_vram(monkeypatch, 100.0)
        mm = ModelManager(strategy="auto")
        mm.register_component("model", DummyModel())
        assert mm.resolve_offload_strategy("cuda") == "no_offload"

    def test_pipeline_too_large_for_no_offload_picks_model_offload(self, monkeypatch):
        # Simulate: 24 GB VRAM, but pipeline weights estimated at 20 GB
        # (so 20 × 1.5 = 30 > 24 → can't fit fully). One component alone
        # at ~10 GB fits (10 × 1.5 = 15 ≤ 24) → model_offload.
        _patch_vram(monkeypatch, 24.0)
        mm = ModelManager(strategy="auto")

        # Stub the size estimation to return values that exercise the
        # decision logic without actually allocating 30 GB of params.
        mm._estimate_components_size_gb = lambda: (20.0, 10.0)
        assert mm.resolve_offload_strategy("cuda") == "model_offload"

    def test_largest_component_exceeds_vram_picks_group_offload(self, monkeypatch):
        # 12 GB VRAM, largest component 10 GB → 10 × 1.5 = 15 > 12 → group_offload.
        _patch_vram(monkeypatch, 12.0)
        mm = ModelManager(strategy="auto")
        mm._estimate_components_size_gb = lambda: (15.0, 10.0)
        assert mm.resolve_offload_strategy("cuda") == "group_offload"

    def test_empty_registry_falls_back_to_vram_tier(self, monkeypatch):
        _patch_vram(monkeypatch, 16.0)
        mm = ModelManager(strategy="auto")
        # No components → 16 GB falls in the [12, 20) tier → model_offload.
        assert mm.resolve_offload_strategy("cuda") == "model_offload"

    def test_vram_detection_failure_falls_back_to_group_offload(self, monkeypatch):
        def boom(d=None):
            raise RuntimeError("no cuda here")

        monkeypatch.setattr(torch.cuda, "mem_get_info", boom)
        mm = ModelManager(strategy="auto")
        mm.register_component("model", DummyModel())
        assert mm.resolve_offload_strategy("cuda") == "group_offload"


class TestRamDetection:
    def test_psutil_returns_positive(self):
        mm = ModelManager()
        # psutil is a hard dep — should always return positive numbers.
        avail, total = mm._detect_available_ram_gb()
        assert avail > 0.0
        assert total >= avail

    def test_warns_when_pipeline_exceeds_ram(self, monkeypatch, caplog):
        _patch_vram(monkeypatch, 24.0)
        mm = ModelManager(strategy="auto")
        # Pretend 16 GB available RAM, pipeline 30 GB.
        mm._detect_available_ram_gb = lambda: (16.0, 16.0)
        mm._estimate_components_size_gb = lambda: (30.0, 12.0)

        with caplog.at_level("WARNING"):
            mm.resolve_offload_strategy("cuda")
        assert any("exceed available RAM" in rec.message for rec in caplog.records)


class TestWindowsRamAccounting:
    """The Windows path of ``_detect_available_ram_gb`` adjusts for WDDM
    commit-charge inflation. Borrowed from ComfyUI; see ``_windows.py``.
    Tests monkeypatch ``sys.platform`` + the PSAPI helper so they run on
    Linux."""

    def test_non_windows_skips_adjustment(self, monkeypatch):
        # Non-Windows path: psutil is the source of truth, no PSAPI call.
        import sys as _sys

        monkeypatch.setattr(_sys, "platform", "linux")
        # If anything reaches the Windows helper on this path it's a bug.
        monkeypatch.setattr(
            "diffusers_mm._windows.query_performance_info_bytes",
            lambda: (_ for _ in ()).throw(AssertionError("should not be called on linux")),
        )
        mm = ModelManager()
        avail, total = mm._detect_available_ram_gb()
        assert avail > 0.0
        assert total >= avail

    def test_windows_uses_adjusted_when_more_generous(self, monkeypatch):
        # WDDM commit-inflated case: psutil says 2 GiB free, but the
        # adjusted formula (total − (committed − vram_in_use)) yields
        # 10 GiB. We should return 10, not 2.
        import sys as _sys

        monkeypatch.setattr(_sys, "platform", "win32")

        # Fake psutil values: 32 GiB total, 2 GiB available.
        class _FakeVM:
            total = 32 * (1024**3)
            available = 2 * (1024**3)

        monkeypatch.setattr("psutil.virtual_memory", lambda: _FakeVM())

        # Fake PSAPI: 32 GiB physical, 30 GiB committed.
        monkeypatch.setattr(
            "diffusers_mm._windows.query_performance_info_bytes",
            lambda: (30 * (1024**3), 32 * (1024**3)),
        )

        mm = ModelManager()
        # Fake VRAM-in-use: 10 GiB. Adjusted = 32 − (30 − 10) = 12 GiB.
        monkeypatch.setattr(mm, "_total_vram_in_use_bytes", lambda: 10 * (1024**3))

        avail, total = mm._detect_available_ram_gb()
        assert total == pytest.approx(32.0, abs=0.01)
        # max(2, 12) = 12 → we report the adjusted figure.
        assert avail == pytest.approx(12.0, abs=0.01)

    def test_windows_uses_psutil_when_more_generous(self, monkeypatch):
        # Healthy case: psutil's number already exceeds the adjusted
        # figure (e.g. fresh system, no VRAM held). We should not
        # *lower* the reported available by switching to adjusted.
        import sys as _sys

        monkeypatch.setattr(_sys, "platform", "win32")

        class _FakeVM:
            total = 32 * (1024**3)
            available = 20 * (1024**3)

        monkeypatch.setattr("psutil.virtual_memory", lambda: _FakeVM())
        # Committed equals physical → adjusted = 0 + vram_in_use = small.
        monkeypatch.setattr(
            "diffusers_mm._windows.query_performance_info_bytes",
            lambda: (32 * (1024**3), 32 * (1024**3)),
        )

        mm = ModelManager()
        monkeypatch.setattr(mm, "_total_vram_in_use_bytes", lambda: 1 * (1024**3))

        avail, _total = mm._detect_available_ram_gb()
        # max(20, 1) → keep psutil's 20.
        assert avail == pytest.approx(20.0, abs=0.01)

    def test_windows_psapi_failure_falls_back_to_psutil(self, monkeypatch):
        # If GetPerformanceInfo fails, the helper returns None and we
        # quietly use psutil's number.
        import sys as _sys

        monkeypatch.setattr(_sys, "platform", "win32")

        class _FakeVM:
            total = 32 * (1024**3)
            available = 5 * (1024**3)

        monkeypatch.setattr("psutil.virtual_memory", lambda: _FakeVM())
        monkeypatch.setattr(
            "diffusers_mm._windows.query_performance_info_bytes",
            lambda: None,
        )

        mm = ModelManager()
        avail, total = mm._detect_available_ram_gb()
        assert avail == pytest.approx(5.0, abs=0.01)
        assert total == pytest.approx(32.0, abs=0.01)

    def test_user_reported_scenario(self, monkeypatch):
        # Real-world log: 24 GiB VRAM, 32 GiB RAM, psutil reported
        # 1.8 GiB available while the system was loading the int8
        # pipeline. WDDM had committed ~22 GiB worth of VRAM-backed
        # reserve on top of the actual host commitments. With the
        # adjustment, we should see a more honest "usable" number.
        import sys as _sys

        monkeypatch.setattr(_sys, "platform", "win32")

        class _FakeVM:
            total = 32 * (1024**3)
            available = int(1.8 * (1024**3))

        monkeypatch.setattr("psutil.virtual_memory", lambda: _FakeVM())

        # Suppose committed = 47 GiB (overcommit on a 32 GiB box is
        # possible because pagefile + WDDM padding); VRAM in use is
        # ~22 GiB out of 24.
        monkeypatch.setattr(
            "diffusers_mm._windows.query_performance_info_bytes",
            lambda: (47 * (1024**3), 32 * (1024**3)),
        )
        mm = ModelManager()
        monkeypatch.setattr(mm, "_total_vram_in_use_bytes", lambda: 22 * (1024**3))

        avail, _ = mm._detect_available_ram_gb()
        # Adjusted = 32 − (47 − 22) = 7 GiB. max(1.8, 7) = 7.
        # That's the more honest "usable" figure the strategy
        # resolver should see.
        assert avail == pytest.approx(7.0, abs=0.01)


class TestTotalVramInUse:
    def test_returns_zero_when_no_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        mm = ModelManager()
        assert mm._total_vram_in_use_bytes() == 0

    def test_sums_across_devices(self, monkeypatch):
        # Fake two devices, each with (total - free) = 4 GiB. Sum = 8.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        monkeypatch.setattr(
            torch.cuda,
            "mem_get_info",
            lambda i: (4 * (1024**3), 8 * (1024**3)),  # free=4, total=8 → in_use=4
        )
        mm = ModelManager()
        assert mm._total_vram_in_use_bytes() == 8 * (1024**3)

    def test_failure_returns_zero(self, monkeypatch):
        # Defensive: any exception inside the loop should be swallowed
        # and the caller's adjustment falls back gracefully.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

        def boom(_i):
            raise RuntimeError("driver hiccup")

        monkeypatch.setattr(torch.cuda, "mem_get_info", boom)
        mm = ModelManager()
        assert mm._total_vram_in_use_bytes() == 0


class TestAutoTuneGroupOffload:
    """When auto picks group_offload, knobs should be tuned to the hardware."""

    def test_abundant_ram_flips_low_cpu_mem_off(self, monkeypatch):
        # 64 GB available RAM, 8 GB pipeline → required = 8 + 16 = 24 ≤ 64 → flip off.
        _patch_vram(monkeypatch, 4.0)
        mm = ModelManager(strategy="auto")
        mm._detect_available_ram_gb = lambda: (64.0, 64.0)
        mm._estimate_components_size_gb = lambda: (8.0, 5.0)
        assert mm.group_offload_low_cpu_mem is True

        chosen = mm.resolve_offload_strategy("cuda")
        assert chosen == "group_offload"
        assert mm.group_offload_low_cpu_mem is False

    def test_users_real_setup_flips_low_cpu_mem_off(self, monkeypatch):
        # 123.4 GB available RAM / 65.7 GB pipeline: required = 81.7 ≤ 123.4 → flip off.
        _patch_vram(monkeypatch, 4.0)
        mm = ModelManager(strategy="auto")
        mm._detect_available_ram_gb = lambda: (123.4, 128.0)
        mm._estimate_components_size_gb = lambda: (65.7, 35.4)

        mm.resolve_offload_strategy("cuda")
        assert mm.group_offload_low_cpu_mem is False

    def test_tight_ram_keeps_low_cpu_mem_on(self, monkeypatch):
        # 32 GB available RAM, 24 GB pipeline → required = 40 > 32 → keep True.
        _patch_vram(monkeypatch, 4.0)
        mm = ModelManager(strategy="auto")
        mm._detect_available_ram_gb = lambda: (32.0, 32.0)
        mm._estimate_components_size_gb = lambda: (24.0, 12.0)
        assert mm.group_offload_low_cpu_mem is True

        chosen = mm.resolve_offload_strategy("cuda")
        assert chosen == "group_offload"
        assert mm.group_offload_low_cpu_mem is True

    def test_loaded_system_keeps_low_cpu_mem_on_despite_high_total(self, monkeypatch):
        # 128 GB total RAM but only 20 GB *available* (something else is hogging it),
        # 30 GB pipeline → required = 46 > 20 → keep True. Demonstrates that
        # using available rather than total catches loaded systems.
        _patch_vram(monkeypatch, 4.0)
        mm = ModelManager(strategy="auto")
        mm._detect_available_ram_gb = lambda: (20.0, 128.0)
        mm._estimate_components_size_gb = lambda: (30.0, 15.0)

        mm.resolve_offload_strategy("cuda")
        assert mm.group_offload_low_cpu_mem is True

    def test_explicit_strategy_does_not_auto_tune(self, monkeypatch):
        # User picked group_offload directly → manager should NOT touch
        # their knob choice, even if RAM is abundant.
        _patch_vram(monkeypatch, 4.0)
        mm = ModelManager(strategy="group_offload", group_offload_low_cpu_mem=True)
        mm._detect_available_ram_gb = lambda: (64.0, 64.0)
        mm._estimate_components_size_gb = lambda: (8.0, 5.0)

        mm.resolve_offload_strategy("cuda")
        # Explicit strategy → resolve returns immediately without auto-tuning.
        assert mm.group_offload_low_cpu_mem is True

    def test_no_size_info_skips_tuning(self, monkeypatch):
        # Auto with no components → falls back to VRAM tier table; even if
        # that picks group_offload, we have no weights estimate to base
        # tuning on, so leave knobs alone.
        _patch_vram(monkeypatch, 4.0)
        mm = ModelManager(strategy="auto")
        # No components registered.
        mm.resolve_offload_strategy("cuda")
        # Default unchanged.
        assert mm.group_offload_low_cpu_mem is True


class TestApplyStrategyNoOffload:
    def test_no_offload_moves_to_device(self):
        mm = ModelManager(strategy="no_offload")
        model = DummyModel()
        mm.register_component("model", model)
        mm.apply_offload_strategy("cpu")
        assert mm.applied_strategy == "no_offload"
        # Model should be on CPU (which it already was, but strategy is applied)
        assert next(model.parameters()).device == torch.device("cpu")

    def test_model_offload_keeps_on_cpu(self):
        mm = ModelManager(strategy="model_offload")
        model = DummyModel()
        mm.register_component("model", model)
        mm.apply_offload_strategy("cpu")
        assert mm.applied_strategy == "model_offload"
        assert next(model.parameters()).device == torch.device("cpu")

    def test_idempotent(self):
        mm = ModelManager(strategy="no_offload")
        model = DummyModel()
        mm.register_component("model", model)
        mm.apply_offload_strategy("cpu")
        # Second call should be a no-op
        result = mm.apply_offload_strategy("cpu")
        assert result == "no_offload"

    def test_empty_components(self):
        mm = ModelManager(strategy="no_offload")
        result = mm.apply_offload_strategy("cpu")
        assert result == "no_offload"

    def test_incremental_apply_only_touches_pending(self):
        # Applying twice in a row with no new components is a true no-op:
        # already-marked components are skipped.
        mm = ModelManager(strategy="no_offload")
        a, b = DummyModel(), DummyModel()
        mm.register_component("a", a)
        mm.apply_offload_strategy("cpu")
        assert mm._component_strategies == {"a": "no_offload"}

        mm.register_component("b", b)
        # "a" is already marked; only "b" is pending.
        mm.apply_offload_strategy("cpu")
        assert mm._component_strategies == {"a": "no_offload", "b": "no_offload"}

    def test_aliased_module_registered_under_two_names_dedupes(self):
        # The same nn.Module registered under two names should only be
        # processed once, but both names should track the applied strategy
        # so subsequent applies are no-ops.
        mm = ModelManager(strategy="no_offload")
        shared = DummyModel()
        mm.register_component("primary", shared)
        mm.register_component("alias", shared)
        mm.apply_offload_strategy("cpu")
        assert mm._component_strategies == {"primary": "no_offload", "alias": "no_offload"}


class TestUseComponentsModelOffload:
    def test_moves_to_device_and_back(self):
        mm = ModelManager(strategy="model_offload")
        model = DummyModel()
        mm.register_component("model", model)
        mm.apply_offload_strategy("cpu")

        with mm.use_components("model", device="cpu"):
            assert next(model.parameters()).device == torch.device("cpu")
        assert next(model.parameters()).device == torch.device("cpu")

    def test_no_offload_is_noop(self):
        mm = ModelManager(strategy="no_offload")
        model = DummyModel()
        mm.register_component("model", model)
        mm.apply_offload_strategy("cpu")

        with mm.use_components("model", device="cpu"):
            pass  # Should not raise


class TestClear:
    def test_clears_everything(self):
        # Override the defaults to non-default values to verify clear()
        # resets back to the documented defaults (use_stream=True,
        # low_cpu_mem=True).
        mm = ModelManager(
            group_offload_use_stream=False,
            group_offload_low_cpu_mem=False,
        )
        mm.register_component("model", DummyModel())
        mm.set_cached("key", "value")
        mm._applied_strategy = "no_offload"

        mm.clear()

        assert mm.get_component("model") is None
        assert mm.get_cached("key") is None
        assert mm.applied_strategy is None
        assert mm.group_offload_use_stream is True
        assert mm.group_offload_low_cpu_mem is True
        assert mm.component_names == []


class TestRecordMemoryHistory:
    def test_yields_cleanly(self, tmp_path):
        mm = ModelManager()
        out = tmp_path / "snapshot.pickle"
        # Should not raise whether CUDA is available or not. If CUDA is
        # available the snapshot is written; otherwise it's a no-op.
        with mm.record_memory_history(str(out)):
            pass

    def test_no_op_without_cuda(self, monkeypatch, tmp_path):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        mm = ModelManager()
        out = tmp_path / "snapshot.pickle"
        with mm.record_memory_history(str(out)):
            pass
        assert not out.exists()


class TestDebugVramBreakdown:
    def test_returns_empty_dict_without_cuda(self, monkeypatch, capsys):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        mm = ModelManager()
        result = mm.debug_vram_breakdown()
        assert result == {}
        assert "CUDA not available" in capsys.readouterr().out

    def test_returns_breakdown_dict_with_mocked_cuda(self, monkeypatch, capsys):
        # Patch the CUDA queries so we can run the breakdown on a CPU-only
        # box. Verifies the formula (driver_used - pytorch_reserved =
        # external) and the dict keys downstream code might depend on.
        gb = 1024**3
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d=None: (4 * gb, 16 * gb))
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d=None: 7 * gb)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d=None: 9 * gb)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda d=None: 8 * gb)
        monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda d=None: 10 * gb)

        mm = ModelManager()
        result = mm.debug_vram_breakdown()

        assert result["driver_used_gb"] == 12.0  # 16 - 4
        assert result["driver_free_gb"] == 4.0
        assert result["driver_total_gb"] == 16.0
        assert result["pytorch_allocated_gb"] == 7.0
        assert result["pytorch_reserved_gb"] == 9.0
        assert result["pytorch_max_allocated_gb"] == 8.0
        assert result["pytorch_max_reserved_gb"] == 10.0
        assert result["external_gb"] == 3.0  # driver_used (12) - pytorch_reserved (9)
        out = capsys.readouterr().out
        assert "Driver used" in out
        assert "External" in out

    def test_includes_block_pin_state_when_present(self, monkeypatch, capsys):
        gb = 1024**3
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d=None: (gb, 16 * gb))
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d=None: gb)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d=None: gb)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda d=None: gb)
        monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda d=None: gb)

        mm = ModelManager()
        mm._block_pin_states["transformer"] = BlockPinState(
            component=nn.Linear(4, 4),  # any module — we only display metadata
            block_attr="blocks",
            n_pinned=23,
            device=torch.device("cuda"),
            resident=True,
        )
        mm.debug_vram_breakdown()
        out = capsys.readouterr().out
        assert "block_pin state" in out
        assert "transformer: n_pinned=23" in out

    def test_external_clamps_to_zero_when_pytorch_overreports(self, monkeypatch, capsys):
        # Sanity: PyTorch's reserved counter can momentarily exceed
        # mem_get_info's reported usage on some drivers (rounding,
        # asynchronous allocator updates). External should never be
        # reported as a negative number.
        gb = 1024**3
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d=None: (10 * gb, 16 * gb))
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d=None: 5 * gb)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d=None: 7 * gb)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda d=None: 5 * gb)
        monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda d=None: 7 * gb)

        mm = ModelManager()
        result = mm.debug_vram_breakdown()
        # driver_used (6) - pytorch_reserved (7) = -1 → clamp to 0
        assert result["external_gb"] == 0.0


class TestBlockPinHelpers:
    """Internals of the ``block_pin`` module."""

    def test_find_largest_block_list_picks_largest_by_param_count(self):
        m = nn.Module()
        m.small_blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
        m.big_blocks = nn.ModuleList([nn.Linear(64, 64) for _ in range(3)])
        result = find_largest_block_list(m)
        assert result is not None
        name, blocks = result
        assert name == "big_blocks"
        assert len(blocks) == 3

    def test_find_largest_block_list_skips_short_lists(self):
        m = nn.Module()
        m.solo = nn.ModuleList([nn.Linear(4, 4)])
        assert find_largest_block_list(m) is None

    def test_find_largest_block_list_skips_mixed_types(self):
        m = nn.Module()
        m.mixed = nn.ModuleList([nn.Linear(4, 4), nn.Conv2d(3, 3, 3)])
        assert find_largest_block_list(m) is None

    def test_find_largest_block_list_returns_none_when_no_lists(self):
        m = nn.Module()
        m.head = nn.Linear(4, 4)
        assert find_largest_block_list(m) is None

    def test_per_block_size_bytes(self):
        # nn.Linear(8, 8) → 8*8 + 8 = 72 params × 4 bytes (float32) = 288 bytes.
        blocks = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])
        assert per_block_size_bytes(blocks) == 288

    def test_per_block_size_bytes_empty_list_returns_zero(self):
        assert per_block_size_bytes(nn.ModuleList()) == 0

    def test_non_block_size_bytes_includes_top_level_param(self):
        # Per the LTX-2 ``scale_shift_table`` case: a direct nn.Parameter
        # on the component (not inside any child) must be counted.
        m = nn.Module()
        m.head = nn.Linear(4, 4)  # 4*4 + 4 = 20 params, 80 bytes
        m.blocks = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])
        m.scale = nn.Parameter(torch.zeros(10))  # 10 params, 40 bytes
        assert non_block_size_bytes(m, "blocks") == 80 + 40

    def test_apply_block_pin_clamps_num_to_pin_and_runs_cpu_to_cpu(self):
        # CPU-only smoke test: pinning is a no-op move on CPU but the
        # function should still walk every code path (children, top-level
        # params/buffers, blocks) without raising. ``num_to_pin > len`` is
        # clamped to len(blocks). num_to_pin = len so no overflow → we
        # don't need diffusers in this test.
        m = nn.Module()
        m.head = nn.Linear(4, 4)
        m.blocks = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])
        m.scale = nn.Parameter(torch.zeros(10))
        m.register_buffer("buf", torch.zeros(5))

        applied = apply_block_pin(
            m,
            "blocks",
            m.blocks,
            num_to_pin=99,  # over-large → clamps to 3
            device=torch.device("cpu"),
            offload_kwargs={},  # not consulted when no overflow
        )
        assert applied == 3


class TestBlockPinCount:
    """``_compute_block_pin_count`` budget logic."""

    def test_override_clamped_to_block_count(self):
        mm = ModelManager()
        mm.set_block_pin_count("transformer", 100)
        m = nn.Module()
        m.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(8)])
        n = mm._compute_block_pin_count("transformer", m, "blocks", m.blocks, "cpu")
        assert n == 8  # clamped to len(blocks)

    def test_override_zero_pins_nothing(self):
        mm = ModelManager()
        mm.set_block_pin_count("transformer", 0)
        m = nn.Module()
        m.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(8)])
        n = mm._compute_block_pin_count("transformer", m, "blocks", m.blocks, "cpu")
        assert n == 0

    def test_set_block_pin_count_negative_raises(self):
        mm = ModelManager()
        with pytest.raises(ValueError, match=">= 0"):
            mm.set_block_pin_count("transformer", -1)

    def test_auto_count_uses_vram_budget(self, monkeypatch):
        # 12 GB VRAM, 0.5 GB non-block, 6.5 GB Linux working set, 1 GB
        # streamed in flight = 4.0 GB budget. Per-block size = 1 GB → 4
        # blocks fit. Pin sys.platform to "linux" so the test is stable
        # regardless of host OS.
        monkeypatch.setattr("diffusers_mm.manager.sys.platform", "linux")
        _patch_vram(monkeypatch, 12.0)
        mm = ModelManager()
        # Stub the size helpers — actually allocating GB of params is not
        # an option here.
        monkeypatch.setattr("diffusers_mm.manager.per_block_size_bytes", lambda b: int(1.0 * 1024**3))
        monkeypatch.setattr("diffusers_mm.manager.non_block_size_bytes", lambda c, a: int(0.5 * 1024**3))
        m = nn.Module()
        m.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(20)])
        n = mm._compute_block_pin_count("transformer", m, "blocks", m.blocks, "cuda")
        assert n == 4

    def test_auto_count_uses_windows_working_set(self, monkeypatch):
        # Same 12 GB VRAM / 0.5 GB non-block / 1 GB per-block as the Linux
        # test, but on Windows the working set is 8.5 GiB → budget = 12 -
        # 0.5 - 8.5 - 1 = 2.0 GB → 2 blocks fit (2 fewer than on Linux).
        # This is the whole point of the OS split: don't make Linux pay
        # for Windows' allocator overhead.
        monkeypatch.setattr("diffusers_mm.manager.sys.platform", "win32")
        _patch_vram(monkeypatch, 12.0)
        mm = ModelManager()
        monkeypatch.setattr("diffusers_mm.manager.per_block_size_bytes", lambda b: int(1.0 * 1024**3))
        monkeypatch.setattr("diffusers_mm.manager.non_block_size_bytes", lambda c, a: int(0.5 * 1024**3))
        m = nn.Module()
        m.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(20)])
        n = mm._compute_block_pin_count("transformer", m, "blocks", m.blocks, "cuda")
        assert n == 2

    def test_auto_count_returns_zero_when_no_budget(self, monkeypatch, caplog):
        # 6 GB VRAM, 6 GB non-block → budget = 6 - 6 - 6.5 - 1 = -7.5 → 0 pinned.
        monkeypatch.setattr("diffusers_mm.manager.sys.platform", "linux")
        _patch_vram(monkeypatch, 6.0)
        mm = ModelManager()
        monkeypatch.setattr("diffusers_mm.manager.per_block_size_bytes", lambda b: int(1.0 * 1024**3))
        monkeypatch.setattr("diffusers_mm.manager.non_block_size_bytes", lambda c, a: int(6.0 * 1024**3))
        m = nn.Module()
        m.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(20)])
        with caplog.at_level("WARNING"):
            n = mm._compute_block_pin_count("transformer", m, "blocks", m.blocks, "cuda")
        assert n == 0
        assert any("no VRAM budget" in r.message for r in caplog.records)

    def test_auto_count_clamped_to_block_count(self, monkeypatch):
        # Huge budget but only 4 blocks exist.
        _patch_vram(monkeypatch, 1000.0)
        mm = ModelManager()
        monkeypatch.setattr("diffusers_mm.manager.per_block_size_bytes", lambda b: int(0.001 * 1024**3))
        monkeypatch.setattr("diffusers_mm.manager.non_block_size_bytes", lambda c, a: 0)
        m = nn.Module()
        m.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])
        n = mm._compute_block_pin_count("transformer", m, "blocks", m.blocks, "cuda")
        assert n == 4


class TestAutoResolutionPicksBlockPin:
    """Auto resolver should prefer block_pin over group_offload when
    the largest component has a long enough block list."""

    def _make_with_blocks(self, num_blocks: int = 16) -> nn.Module:
        m = nn.Module()
        m.head = nn.Linear(4, 4)
        m.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(num_blocks)])
        return m

    def test_picks_block_pin_when_largest_has_block_list(self, monkeypatch):
        # 12 GB VRAM, largest component 10 GB (×1.5 = 15 > 12 → can't
        # model_offload), but it has 16 blocks → block_pin.
        _patch_vram(monkeypatch, 12.0)
        mm = ModelManager(strategy="auto")
        mm._estimate_components_size_gb = lambda: (15.0, 10.0)
        mm.register_component("transformer", self._make_with_blocks(num_blocks=16))
        assert mm.resolve_offload_strategy("cuda") == "block_pin"

    def test_falls_back_to_group_offload_when_no_block_list(self, monkeypatch):
        _patch_vram(monkeypatch, 12.0)
        mm = ModelManager(strategy="auto")
        mm._estimate_components_size_gb = lambda: (15.0, 10.0)
        # Plain DummyModel — no block list discoverable.
        mm.register_component("transformer", DummyModel())
        assert mm.resolve_offload_strategy("cuda") == "group_offload"

    def test_falls_back_when_block_list_too_short(self, monkeypatch):
        # AUTO_BLOCK_PIN_MIN_BLOCKS = 8 by default. 4 blocks → too few.
        _patch_vram(monkeypatch, 12.0)
        mm = ModelManager(strategy="auto")
        mm._estimate_components_size_gb = lambda: (15.0, 10.0)
        mm.register_component("transformer", self._make_with_blocks(num_blocks=4))
        assert mm.resolve_offload_strategy("cuda") == "group_offload"

    def test_prefers_block_pin_over_model_offload_when_fully_pinnable(self, monkeypatch):
        # 24 GB VRAM, largest 10 GB with 16 blocks. model_offload would fit
        # (10 × 1.5 = 15 ≤ 24), but block_pin can pin the WHOLE component
        # (10 + working_set + per_block ≈ 17 ≤ 24) and keep it resident across
        # runs — same VRAM peak, faster. block_pin wins.
        _patch_vram(monkeypatch, 24.0)
        mm = ModelManager(strategy="auto")
        mm._estimate_components_size_gb = lambda: (20.0, 10.0)
        mm.register_component("transformer", self._make_with_blocks(num_blocks=16))
        assert mm.resolve_offload_strategy("cuda") == "block_pin"

    def test_keeps_model_offload_when_only_partial_pin(self, monkeypatch):
        # 17 GB VRAM, largest 10 GB with 16 blocks. model_offload fits
        # (10 × 1.5 = 15 ≤ 17), but block_pin could NOT fully pin it
        # (10 + working_set(6.5/8.5) + per_block > 17), so the protective
        # guard keeps model_offload rather than risk a partial pin that
        # under-budgets activations.
        _patch_vram(monkeypatch, 17.0)
        mm = ModelManager(strategy="auto")
        mm._estimate_components_size_gb = lambda: (20.0, 10.0)
        mm.register_component("transformer", self._make_with_blocks(num_blocks=16))
        assert mm.resolve_offload_strategy("cuda") == "model_offload"

    def test_block_pin_triggers_low_cpu_mem_auto_tune(self, monkeypatch):
        # Confirm the auto-tune of low_cpu_mem_usage runs for block_pin
        # (same RAM-headroom heuristic as group_offload).
        _patch_vram(monkeypatch, 12.0)
        mm = ModelManager(strategy="auto")
        mm._estimate_components_size_gb = lambda: (15.0, 10.0)
        mm._detect_available_ram_gb = lambda: (128.0, 128.0)
        mm.register_component("transformer", self._make_with_blocks(num_blocks=16))
        assert mm.group_offload_low_cpu_mem is True

        chosen = mm.resolve_offload_strategy("cuda")
        assert chosen == "block_pin"
        assert mm.group_offload_low_cpu_mem is False


class TestApplyBlockPinStrategy:
    """End-to-end: apply_offload_strategy with block_pin and a fake apply_group_offloading."""

    def test_apply_block_pin_pins_and_streams_overflow(self, monkeypatch):
        # Stub apply_group_offloading so we don't need a real CUDA setup,
        # and capture which blocks it gets called on. With num_to_pin=3
        # out of 5 blocks, we should see exactly 2 streaming calls.
        streamed: list = []

        def fake_apply(mod, **kwargs):
            streamed.append(mod)

        monkeypatch.setattr("diffusers.hooks.group_offloading.apply_group_offloading", fake_apply)

        # Force the override so we don't depend on VRAM detection.
        mm = ModelManager(strategy="block_pin")
        m = nn.Module()
        m.head = nn.Linear(4, 4)
        m.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(5)])
        mm.register_component("transformer", m)
        mm.set_block_pin_count("transformer", 3)

        mm.apply_offload_strategy("cpu")

        assert mm.applied_strategy == "block_pin"
        # Two overflow blocks streamed, and the per-block call gets the
        # block module (not the whole transformer).
        assert len(streamed) == 2
        assert streamed[0] is m.blocks[3]
        assert streamed[1] is m.blocks[4]

    def test_apply_block_pin_falls_back_to_group_offload_without_block_list(self, monkeypatch):
        # A component with no discoverable block list should get plain
        # group_offload at apply time — we expect exactly one call on
        # the whole component.
        called_with: list = []

        def fake_apply(mod, **kwargs):
            called_with.append(mod)

        monkeypatch.setattr("diffusers.hooks.group_offloading.apply_group_offloading", fake_apply)

        mm = ModelManager(strategy="block_pin")
        m = DummyModel()  # no nn.ModuleList anywhere
        mm.register_component("text_encoder", m)

        mm.apply_offload_strategy("cpu")
        assert mm.applied_strategy == "block_pin"
        assert called_with == [m]


class TestBlockPinTransition:
    """``prepare_strategy_transition`` must clean up block_pin's hooks
    on its way out, the same way it does for group_offload."""

    def test_transition_from_block_pin_strips_hooks(self, monkeypatch):
        cleaned: list = []
        monkeypatch.setattr("diffusers_mm.manager.remove_offload_hooks", lambda m: cleaned.append(m))

        mm = ModelManager(strategy="block_pin")
        m = DummyModel()
        mm.register_component("transformer", m)
        # Pretend block_pin was applied.
        mm._applied_strategy = "block_pin"
        mm._component_strategies["transformer"] = "block_pin"

        mm.prepare_strategy_transition("no_offload", "cpu")
        assert m in cleaned
        assert mm.applied_strategy == "no_offload"


class _FakeTransformer(nn.Module):
    """Block-pinnable component: has a real ``forward`` plus a block list."""

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 4)
        self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

    def forward(self, x):
        return self.head(x)


class _FakeVAE(nn.Module):
    """Neighbor with explicit ``decode`` / ``encode`` entry points (the
    typical diffusers VAE shape that bypasses ``__call__``)."""

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 4)

    def forward(self, x):
        return self.layer(x)

    def decode(self, x):
        return self.layer(x)

    def encode(self, x):
        return self.layer(x)


class _FakeTextEncoder(nn.Module):
    """Neighbor without decode/encode — forward-only entry point."""

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 4)

    def forward(self, x):
        return self.layer(x)


def _stub_group_offload(monkeypatch):
    """Make ``apply_group_offloading`` a no-op so block_pin tests don't
    need a real CUDA / accelerate setup."""
    monkeypatch.setattr(
        "diffusers.hooks.group_offloading.apply_group_offloading",
        lambda mod, **kwargs: None,
    )


class TestBlockPinAutoEvict:
    """End-to-end checks for the cross-component auto-evict / repin
    behavior introduced to keep VAE decode from sharing VRAM with the
    permanently-resident pinned transformer."""

    def test_state_recorded_after_apply(self, monkeypatch):
        _stub_group_offload(monkeypatch)
        mm = ModelManager(strategy="block_pin")
        transformer = _FakeTransformer()
        mm.register_component("transformer", transformer)
        mm.set_block_pin_count("transformer", 3)
        mm.apply_offload_strategy("cpu")

        state = mm._block_pin_states["transformer"]
        assert state.component is transformer
        assert state.block_attr == "blocks"
        assert state.n_pinned == 3
        assert state.resident is True

    def test_neighbor_forward_evicts_pinned(self, monkeypatch):
        _stub_group_offload(monkeypatch)
        mm = ModelManager(strategy="block_pin")
        transformer = _FakeTransformer()
        text_encoder = _FakeTextEncoder()
        mm.register_component("transformer", transformer)
        mm.register_component("text_encoder", text_encoder)
        mm.set_block_pin_count("transformer", 3)
        mm.apply_offload_strategy("cpu")

        state = mm._block_pin_states["transformer"]
        assert state.resident is True
        text_encoder(torch.zeros(4))
        assert state.resident is False

    def test_pinned_forward_repins_after_evict(self, monkeypatch):
        _stub_group_offload(monkeypatch)
        mm = ModelManager(strategy="block_pin")
        transformer = _FakeTransformer()
        text_encoder = _FakeTextEncoder()
        mm.register_component("transformer", transformer)
        mm.register_component("text_encoder", text_encoder)
        mm.set_block_pin_count("transformer", 3)
        mm.apply_offload_strategy("cpu")

        state = mm._block_pin_states["transformer"]
        text_encoder(torch.zeros(4))
        assert state.resident is False

        transformer(torch.zeros(4))
        assert state.resident is True

    def test_decode_wrap_evicts_pinned(self, monkeypatch):
        # The headline reason this feature exists: ``vae.decode`` bypasses
        # ``__call__``, so without method-level wrapping the pinned subset
        # would stay resident through the entire video VAE decode pass.
        _stub_group_offload(monkeypatch)
        mm = ModelManager(strategy="block_pin")
        transformer = _FakeTransformer()
        vae = _FakeVAE()
        mm.register_component("transformer", transformer)
        mm.register_component("vae", vae)
        mm.set_block_pin_count("transformer", 3)
        mm.apply_offload_strategy("cpu")

        state = mm._block_pin_states["transformer"]
        assert state.resident is True
        vae.decode(torch.zeros(4))
        assert state.resident is False

    def test_encode_wrap_evicts_pinned(self, monkeypatch):
        _stub_group_offload(monkeypatch)
        mm = ModelManager(strategy="block_pin")
        transformer = _FakeTransformer()
        vae = _FakeVAE()
        mm.register_component("transformer", transformer)
        mm.register_component("vae", vae)
        mm.set_block_pin_count("transformer", 3)
        mm.apply_offload_strategy("cpu")

        state = mm._block_pin_states["transformer"]
        vae.encode(torch.zeros(4))
        assert state.resident is False

    def test_auto_evict_disabled_skips_all_hooks(self, monkeypatch):
        # Opt-out path: pinned state is still recorded (useful for the
        # ``set_block_pin_count`` / inspection paths) but neither
        # pre-forward hooks nor method wraps are installed, so behavior
        # matches the pre-feature baseline.
        _stub_group_offload(monkeypatch)
        mm = ModelManager(strategy="block_pin", block_pin_auto_evict=False)
        transformer = _FakeTransformer()
        vae = _FakeVAE()
        mm.register_component("transformer", transformer)
        mm.register_component("vae", vae)
        mm.set_block_pin_count("transformer", 3)
        mm.apply_offload_strategy("cpu")

        assert "transformer" in mm._block_pin_states
        assert mm._block_pin_hook_handles == []
        assert mm._block_pin_wrapped_methods == []
        assert "decode" not in vae.__dict__

        state = mm._block_pin_states["transformer"]
        vae.decode(torch.zeros(4))
        assert state.resident is True  # neighbor call did not flip residency

    def test_transition_removes_hooks_and_restores_methods(self, monkeypatch):
        _stub_group_offload(monkeypatch)
        mm = ModelManager(strategy="block_pin")
        transformer = _FakeTransformer()
        vae = _FakeVAE()
        mm.register_component("transformer", transformer)
        mm.register_component("vae", vae)
        mm.set_block_pin_count("transformer", 3)
        mm.apply_offload_strategy("cpu")

        # The wrap installed a shadowing instance attribute.
        assert "decode" in vae.__dict__
        assert "encode" in vae.__dict__
        assert len(mm._block_pin_hook_handles) >= 2  # repin + neighbor evict

        mm.prepare_strategy_transition("no_offload", "cpu")

        # Wrapper instance attributes are gone; class-level methods reachable again.
        assert "decode" not in vae.__dict__
        assert "encode" not in vae.__dict__
        assert mm._block_pin_states == {}
        assert mm._block_pin_hook_handles == []
        assert mm._block_pin_wrapped_methods == []

        # Calling the restored decode does not affect any (now-empty) state.
        vae.decode(torch.zeros(4))

    def test_reapply_does_not_double_wrap(self, monkeypatch):
        _stub_group_offload(monkeypatch)
        mm = ModelManager(strategy="block_pin")
        transformer = _FakeTransformer()
        vae = _FakeVAE()
        mm.register_component("transformer", transformer)
        mm.register_component("vae", vae)
        mm.set_block_pin_count("transformer", 3)
        mm.apply_offload_strategy("cpu")

        wrapper_once = vae.decode
        decode_entries_before = [e for e in mm._block_pin_wrapped_methods if e[1] == "decode" and e[0] is vae]
        assert len(decode_entries_before) == 1

        # Idempotent re-apply: nothing pending, no new wraps.
        mm.apply_offload_strategy("cpu")
        assert vae.decode is wrapper_once
        decode_entries_after = [e for e in mm._block_pin_wrapped_methods if e[1] == "decode" and e[0] is vae]
        assert len(decode_entries_after) == 1

    def test_incremental_neighbor_gets_evict_hook(self, monkeypatch):
        # Registering a brand-new neighbor after the initial apply should
        # pick up the same auto-evict wiring as the originals.
        _stub_group_offload(monkeypatch)
        mm = ModelManager(strategy="block_pin")
        transformer = _FakeTransformer()
        mm.register_component("transformer", transformer)
        mm.set_block_pin_count("transformer", 3)
        mm.apply_offload_strategy("cpu")

        state = mm._block_pin_states["transformer"]
        # Manually flip residency on so we can observe an eviction.
        state.resident = True

        vae = _FakeVAE()
        mm.register_component("vae", vae)
        mm.apply_offload_strategy("cpu")

        assert "decode" in vae.__dict__
        vae.decode(torch.zeros(4))
        assert state.resident is False

    def test_clear_tears_down_auto_evict(self, monkeypatch):
        _stub_group_offload(monkeypatch)
        mm = ModelManager(strategy="block_pin")
        transformer = _FakeTransformer()
        vae = _FakeVAE()
        mm.register_component("transformer", transformer)
        mm.register_component("vae", vae)
        mm.set_block_pin_count("transformer", 3)
        mm.apply_offload_strategy("cpu")

        assert "decode" in vae.__dict__
        mm.clear()
        assert "decode" not in vae.__dict__
        assert mm._block_pin_states == {}
        assert mm._block_pin_hook_handles == []
        assert mm._block_pin_wrapped_methods == []


class TestBlockPinAutoEvictConditional:
    """Eviction shouldn't fire unconditionally — only when the runtime
    check says we're tight on VRAM or the user explicitly forced it via
    ``set_evict_on_neighbor``. The cpu-device default in the simpler
    tests above happens to satisfy "tight" (free=0 < margin), which is
    why they observe eviction; these tests pin down the conditional
    branches explicitly."""

    def test_abundant_vram_skips_eviction(self, monkeypatch):
        _stub_group_offload(monkeypatch)
        mm = ModelManager(strategy="block_pin")
        # Plenty of free VRAM, way above the 6.5 GiB working-set margin.
        monkeypatch.setattr(mm, "_detect_available_vram_gb", lambda d: (50.0, 64.0))
        transformer = _FakeTransformer()
        vae = _FakeVAE()
        mm.register_component("transformer", transformer)
        mm.register_component("vae", vae)
        mm.set_block_pin_count("transformer", 3)
        mm.apply_offload_strategy("cpu")

        state = mm._block_pin_states["transformer"]
        # Pinned starts resident.
        assert state.resident is True
        vae.decode(torch.zeros(4))
        # Runtime check saw 50 GiB free ≥ 6.5 GiB margin → skip eviction.
        assert state.resident is True

    def test_tight_vram_triggers_eviction(self, monkeypatch):
        _stub_group_offload(monkeypatch)
        mm = ModelManager(strategy="block_pin")
        monkeypatch.setattr(mm, "_detect_available_vram_gb", lambda d: (1.0, 16.0))
        transformer = _FakeTransformer()
        vae = _FakeVAE()
        mm.register_component("transformer", transformer)
        mm.register_component("vae", vae)
        mm.set_block_pin_count("transformer", 3)
        mm.apply_offload_strategy("cpu")

        state = mm._block_pin_states["transformer"]
        vae.decode(torch.zeros(4))
        # 1 GiB free < 6.5 GiB margin → evict.
        assert state.resident is False

    def test_reclaimable_reserved_pool_prevents_warmup_thrash(self, monkeypatch):
        # Driver reports only 1 GiB free (below the 6.5 GiB margin), but
        # PyTorch holds an 8 GiB reserved-but-unallocated pool it will reuse
        # without touching the driver. EFFECTIVE free = 1 + 8 = 9 ≥ margin, so
        # eviction must be SKIPPED — this is the warm-up thrash the fix prevents.
        _stub_group_offload(monkeypatch)
        mm = ModelManager(strategy="block_pin")
        monkeypatch.setattr(mm, "_detect_available_vram_gb", lambda d: (1.0, 16.0))
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d=None: 8 * 1024**3)
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d=None: 0)
        transformer = _FakeTransformer()
        vae = _FakeVAE()
        mm.register_component("transformer", transformer)
        mm.register_component("vae", vae)
        mm.set_block_pin_count("transformer", 3)
        mm.apply_offload_strategy("cpu")

        state = mm._block_pin_states["transformer"]
        assert state.resident is True
        vae.decode(torch.zeros(4))
        assert state.resident is True  # reclaimable pool covered the neighbor → no thrash

    def test_tight_vram_with_no_reclaimable_pool_still_evicts(self, monkeypatch):
        # Driver free low AND the reserved pool is fully allocated (nothing
        # reclaimable), so effective free == driver free (1 GiB) < margin →
        # eviction still fires. Confirms the fix doesn't suppress genuine pressure.
        _stub_group_offload(monkeypatch)
        mm = ModelManager(strategy="block_pin")
        monkeypatch.setattr(mm, "_detect_available_vram_gb", lambda d: (1.0, 16.0))
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d=None: 4 * 1024**3)
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d=None: 4 * 1024**3)
        transformer = _FakeTransformer()
        vae = _FakeVAE()
        mm.register_component("transformer", transformer)
        mm.register_component("vae", vae)
        mm.set_block_pin_count("transformer", 3)
        mm.apply_offload_strategy("cpu")

        state = mm._block_pin_states["transformer"]
        vae.decode(torch.zeros(4))
        assert state.resident is False

    def test_set_evict_on_neighbor_true_forces_eviction(self, monkeypatch):
        _stub_group_offload(monkeypatch)
        mm = ModelManager(strategy="block_pin")
        # Plenty of free VRAM — runtime check would say "skip".
        monkeypatch.setattr(mm, "_detect_available_vram_gb", lambda d: (50.0, 64.0))
        transformer = _FakeTransformer()
        vae = _FakeVAE()
        mm.register_component("transformer", transformer)
        mm.register_component("vae", vae)
        mm.set_block_pin_count("transformer", 3)
        mm.apply_offload_strategy("cpu")
        # Override beats the runtime check.
        mm.set_evict_on_neighbor("vae", True)

        state = mm._block_pin_states["transformer"]
        vae.decode(torch.zeros(4))
        assert state.resident is False

    def test_set_evict_on_neighbor_false_disables_eviction(self, monkeypatch):
        _stub_group_offload(monkeypatch)
        mm = ModelManager(strategy="block_pin")
        # Tight free VRAM — runtime check would say "evict".
        monkeypatch.setattr(mm, "_detect_available_vram_gb", lambda d: (1.0, 16.0))
        transformer = _FakeTransformer()
        text_encoder = _FakeTextEncoder()
        mm.register_component("transformer", transformer)
        mm.register_component("text_encoder", text_encoder)
        mm.set_block_pin_count("transformer", 3)
        mm.apply_offload_strategy("cpu")
        # Even though VRAM is tight, the user knows text_encoder doesn't
        # need the headroom.
        mm.set_evict_on_neighbor("text_encoder", False)

        state = mm._block_pin_states["transformer"]
        text_encoder(torch.zeros(4))
        assert state.resident is True

    def test_set_evict_on_neighbor_none_restores_runtime_check(self, monkeypatch):
        _stub_group_offload(monkeypatch)
        mm = ModelManager(strategy="block_pin")
        monkeypatch.setattr(mm, "_detect_available_vram_gb", lambda d: (50.0, 64.0))
        transformer = _FakeTransformer()
        vae = _FakeVAE()
        mm.register_component("transformer", transformer)
        mm.register_component("vae", vae)
        mm.set_block_pin_count("transformer", 3)
        mm.apply_offload_strategy("cpu")

        mm.set_evict_on_neighbor("vae", True)
        state = mm._block_pin_states["transformer"]
        vae.decode(torch.zeros(4))
        assert state.resident is False

        # Restore residency to verify the override clears.
        state.resident = True
        mm.set_evict_on_neighbor("vae", None)
        vae.decode(torch.zeros(4))
        # Runtime check sees abundant VRAM → skip.
        assert state.resident is True

    def test_override_applies_to_forward_path_too(self, monkeypatch):
        # The override should govern the register_forward_pre_hook path
        # as well as the decode/encode wraps, not just one of them.
        _stub_group_offload(monkeypatch)
        mm = ModelManager(strategy="block_pin")
        monkeypatch.setattr(mm, "_detect_available_vram_gb", lambda d: (50.0, 64.0))
        transformer = _FakeTransformer()
        vae = _FakeVAE()
        mm.register_component("transformer", transformer)
        mm.register_component("vae", vae)
        mm.set_block_pin_count("transformer", 3)
        mm.apply_offload_strategy("cpu")
        mm.set_evict_on_neighbor("vae", True)

        state = mm._block_pin_states["transformer"]
        vae(torch.zeros(4))  # exercises forward, not decode
        assert state.resident is False


class TestBlockPinAutoEvictRamAware:
    """Eviction should refuse to push the pinned subset to a host that
    can't absorb it. Without this guard, evicting ~12 GiB of int8 weights
    into a host with 1.8 GiB free succeeds via swap but starves the
    neighbor's next ``pin_memory`` call (real-world Windows OOM logged
    by a user, May 2026)."""

    def test_low_ram_refuses_eviction(self, monkeypatch):
        _stub_group_offload(monkeypatch)
        mm = ModelManager(strategy="block_pin")
        # Tight VRAM → runtime check wants to evict.
        monkeypatch.setattr(mm, "_detect_available_vram_gb", lambda d: (1.0, 16.0))
        # Critically low RAM → cannot absorb the evicted subset.
        monkeypatch.setattr(mm, "_detect_available_ram_gb", lambda: (1.8, 32.0))
        transformer = _FakeTransformer()
        vae = _FakeVAE()
        mm.register_component("transformer", transformer)
        mm.register_component("vae", vae)
        mm.set_block_pin_count("transformer", 3)
        mm.apply_offload_strategy("cpu")

        state = mm._block_pin_states["transformer"]
        # Force the state to claim a pinned-subset size that exceeds
        # available RAM + headroom. _FakeTransformer is tiny so the
        # real cached size is bytes, not GiB; spoof it.
        state.pinned_size_bytes = int(10 * (1024**3))  # 10 GiB pinned

        vae.decode(torch.zeros(4))
        # 1.8 GiB available < 10 GiB evicted + 4 GiB headroom → refuse.
        assert state.resident is True

    def test_abundant_ram_allows_eviction(self, monkeypatch):
        _stub_group_offload(monkeypatch)
        mm = ModelManager(strategy="block_pin")
        monkeypatch.setattr(mm, "_detect_available_vram_gb", lambda d: (1.0, 16.0))
        # Plenty of RAM to absorb whatever we evict.
        monkeypatch.setattr(mm, "_detect_available_ram_gb", lambda: (128.0, 128.0))
        transformer = _FakeTransformer()
        vae = _FakeVAE()
        mm.register_component("transformer", transformer)
        mm.register_component("vae", vae)
        mm.set_block_pin_count("transformer", 3)
        mm.apply_offload_strategy("cpu")

        state = mm._block_pin_states["transformer"]
        state.pinned_size_bytes = int(10 * (1024**3))

        vae.decode(torch.zeros(4))
        # 128 GiB ≥ 10 + 4 → evict proceeds.
        assert state.resident is False

    def test_ram_detection_failure_does_not_block_eviction(self, monkeypatch):
        # When psutil is unavailable, ``_detect_available_ram_gb`` returns
        # (0.0, 0.0). We treat 0.0 as "unknown, don't gate" rather than
        # "no RAM" — otherwise systems without psutil would never evict.
        _stub_group_offload(monkeypatch)
        mm = ModelManager(strategy="block_pin")
        monkeypatch.setattr(mm, "_detect_available_vram_gb", lambda d: (1.0, 16.0))
        monkeypatch.setattr(mm, "_detect_available_ram_gb", lambda: (0.0, 0.0))
        transformer = _FakeTransformer()
        vae = _FakeVAE()
        mm.register_component("transformer", transformer)
        mm.register_component("vae", vae)
        mm.set_block_pin_count("transformer", 3)
        mm.apply_offload_strategy("cpu")

        state = mm._block_pin_states["transformer"]
        state.pinned_size_bytes = int(10 * (1024**3))

        vae.decode(torch.zeros(4))
        assert state.resident is False

    def test_override_true_beats_ram_check(self, monkeypatch):
        # Per-component override is tier 1, above the RAM-absorb check.
        _stub_group_offload(monkeypatch)
        mm = ModelManager(strategy="block_pin")
        monkeypatch.setattr(mm, "_detect_available_vram_gb", lambda d: (50.0, 64.0))
        monkeypatch.setattr(mm, "_detect_available_ram_gb", lambda: (1.0, 32.0))
        transformer = _FakeTransformer()
        vae = _FakeVAE()
        mm.register_component("transformer", transformer)
        mm.register_component("vae", vae)
        mm.set_block_pin_count("transformer", 3)
        mm.apply_offload_strategy("cpu")
        mm.set_evict_on_neighbor("vae", True)

        state = mm._block_pin_states["transformer"]
        state.pinned_size_bytes = int(50 * (1024**3))  # way bigger than RAM

        vae.decode(torch.zeros(4))
        # Override forces evict regardless of RAM.
        assert state.resident is False

    def test_pinned_size_cached_at_apply(self, monkeypatch):
        # Verify the cached ``pinned_size_bytes`` reflects pinned blocks +
        # non-block parts (the same set ``evict_pinned_subset`` moves).
        _stub_group_offload(monkeypatch)
        mm = ModelManager(strategy="block_pin")
        monkeypatch.setattr(mm, "_detect_available_vram_gb", lambda d: (50.0, 64.0))
        transformer = _FakeTransformer()
        mm.register_component("transformer", transformer)
        mm.set_block_pin_count("transformer", 2)
        mm.apply_offload_strategy("cpu")

        state = mm._block_pin_states["transformer"]

        # 2 blocks (each nn.Linear(4,4) = 16 weights + 4 bias = 20 floats)
        # + non-block head (same shape) = 60 floats × 4 bytes = 240 bytes.
        from diffusers_mm.block_pin import non_block_size_bytes, per_block_size_bytes

        expected = 2 * per_block_size_bytes(transformer.blocks) + non_block_size_bytes(transformer, "blocks")
        assert state.pinned_size_bytes == expected
        assert state.pinned_size_bytes > 0


class TestTransformerOffloadCycle:
    """End-to-end verification that the *transformer* (the block-pinned
    component) actually goes off-device when a neighbor runs, then comes
    back when its own forward fires next. The existing ``state.resident``
    flag checks are necessary but not sufficient — these tests prove the
    underlying tensor moves happen on the right targets.
    """

    def test_evict_repin_cycle_dispatches_to_calls(self):
        # CPU-only spy test: replace ``.to`` on each candidate module with
        # a recorder so we can verify (a) which modules were moved, (b)
        # what device they were moved to, and (c) overflow blocks were
        # not touched — without needing a real GPU.
        transformer = _FakeTransformer()  # head + 3 blocks
        # Add a direct top-level param to mirror LTX-2's scale_shift_table
        # so the param-walk in evict/repin has something to chew on.
        transformer.scale = nn.Parameter(torch.zeros(4))
        fake_gpu = torch.device("cuda")

        state = BlockPinState(
            component=transformer,
            block_attr="blocks",
            n_pinned=2,
            device=fake_gpu,
            resident=True,
        )

        calls: dict[int, list[torch.device]] = {}
        members = [transformer.head, transformer.blocks[0], transformer.blocks[1], transformer.blocks[2]]
        for mod in members:
            calls[id(mod)] = []

        def make_stub(mod):
            def stub(target, *args, **kwargs):
                calls[id(mod)].append(torch.device(target) if isinstance(target, str) else target)
                return mod

            return stub

        for mod in members:
            mod.to = make_stub(mod)  # type: ignore[method-assign]

        # Evict: pinned subset → cpu, overflow untouched.
        evict_pinned_subset(state)
        cpu = torch.device("cpu")

        assert state.resident is False
        assert calls[id(transformer.head)] == [cpu]
        assert calls[id(transformer.blocks[0])] == [cpu]
        assert calls[id(transformer.blocks[1])] == [cpu]
        assert calls[id(transformer.blocks[2])] == []  # overflow not touched
        # Direct top-level scale param was moved too (data reassigned to cpu).
        assert transformer.scale.data.device.type == "cpu"

        # Repin: back to state.device.
        repin_pinned_subset(state)

        assert state.resident is True
        assert calls[id(transformer.head)] == [cpu, fake_gpu]
        assert calls[id(transformer.blocks[0])] == [cpu, fake_gpu]
        assert calls[id(transformer.blocks[1])] == [cpu, fake_gpu]
        assert calls[id(transformer.blocks[2])] == []

    def test_evict_is_idempotent(self):
        # Second call on an already-evicted state must not call .to() again.
        transformer = _FakeTransformer()
        state = BlockPinState(
            component=transformer,
            block_attr="blocks",
            n_pinned=2,
            device=torch.device("cuda"),
            resident=False,  # already evicted
        )
        calls: list[torch.device] = []
        transformer.head.to = lambda target, *a, **kw: calls.append(target) or transformer.head  # type: ignore[method-assign]
        evict_pinned_subset(state)
        assert calls == []
        assert state.resident is False

    def test_repin_is_idempotent(self):
        transformer = _FakeTransformer()
        state = BlockPinState(
            component=transformer,
            block_attr="blocks",
            n_pinned=2,
            device=torch.device("cuda"),
            resident=True,  # already resident
        )
        calls: list[torch.device] = []
        transformer.head.to = lambda target, *a, **kw: calls.append(target) or transformer.head  # type: ignore[method-assign]
        repin_pinned_subset(state)
        assert calls == []
        assert state.resident is True

    @pytest.mark.gpu
    def test_evict_repin_cycle_changes_cuda_memory(self):
        # Real-device cycle: build a transformer on CUDA, evict it, and
        # verify the GPU memory allocator actually reports the drop.
        # Catches the case where the dispatch is correct but the actual
        # ``.to('cpu')`` call somehow doesn't free VRAM (e.g. a hidden
        # alias keeping the tensor on GPU).
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")

        class _BigTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                # Sized so the move is comfortably above allocator noise
                # (a 2048x2048 float32 linear ≈ 16 MiB params + 8 KiB bias).
                self.head = nn.Linear(2048, 2048)
                self.blocks = nn.ModuleList([nn.Linear(2048, 2048) for _ in range(3)])

            def forward(self, x):
                return self.head(x)

        transformer = _BigTransformer().to(device)
        # Mirror block_pin's residency: 2 pinned blocks on GPU, overflow on CPU.
        transformer.blocks[2].to("cpu")

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Bytes that should leave GPU after evict (head + 2 pinned blocks).
        pinned_bytes = 0
        for mod in (transformer.head, transformer.blocks[0], transformer.blocks[1]):
            pinned_bytes += sum(p.numel() * p.element_size() for p in mod.parameters())
        pinned_mib = pinned_bytes / (1024 * 1024)

        before = torch.cuda.memory_allocated()

        state = BlockPinState(
            component=transformer,
            block_attr="blocks",
            n_pinned=2,
            device=device,
            resident=True,
        )

        evict_pinned_subset(state)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        after_evict = torch.cuda.memory_allocated()
        freed = before - after_evict
        # Allow some tolerance for allocator bookkeeping; the bulk of the
        # pinned bytes must be off-GPU.
        assert state.resident is False
        assert freed >= pinned_bytes * 0.9, (
            f"Expected ~{pinned_mib:.1f} MiB freed, got {freed / (1024 * 1024):.1f} MiB"
        )
        # The pinned subset's params should now report CPU device.
        assert transformer.head.weight.device.type == "cpu"
        assert transformer.blocks[0].weight.device.type == "cpu"
        assert transformer.blocks[1].weight.device.type == "cpu"
        # Overflow block must be untouched.
        assert transformer.blocks[2].weight.device.type == "cpu"

        # Repin and verify VRAM climbs back.
        repin_pinned_subset(state)
        torch.cuda.synchronize()

        after_repin = torch.cuda.memory_allocated()
        assert state.resident is True
        assert after_repin >= before * 0.9, (
            f"Expected VRAM to recover to ~{before / (1024 * 1024):.1f} MiB, got {after_repin / (1024 * 1024):.1f} MiB"
        )
        assert transformer.head.weight.device.type == "cuda"
        assert transformer.blocks[0].weight.device.type == "cuda"
        assert transformer.blocks[1].weight.device.type == "cuda"

    @pytest.mark.gpu
    def test_neighbor_call_evicts_transformer_from_cuda(self):
        # Top-level integration check: with the strategy applied for real
        # and the runtime check forced into "tight" mode, a neighbor's
        # call actually moves the pinned transformer subset off CUDA.
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # apply_group_offloading on a leaf-only neighbor still works on
        # CUDA, but we stub it here so the test stays self-contained and
        # doesn't depend on diffusers internals.
        from unittest import mock

        device = torch.device("cuda")

        class _BigTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Linear(2048, 2048)
                self.blocks = nn.ModuleList([nn.Linear(2048, 2048) for _ in range(3)])

            def forward(self, x):
                return self.head(x)

        transformer = _BigTransformer()
        text_encoder = _FakeTextEncoder()

        with mock.patch("diffusers.hooks.group_offloading.apply_group_offloading", lambda mod, **kwargs: None):
            mm = ModelManager(strategy="block_pin")
            # Force "tight VRAM" so the runtime check decides to evict.
            mm._detect_available_vram_gb = lambda d: (0.5, 16.0)  # type: ignore[assignment]
            mm.register_component("transformer", transformer)
            mm.register_component("text_encoder", text_encoder)
            mm.set_block_pin_count("transformer", 2)
            mm.apply_offload_strategy(device)

            state = mm._block_pin_states["transformer"]
            assert state.resident is True
            assert transformer.head.weight.device.type == "cuda"
            assert transformer.blocks[0].weight.device.type == "cuda"
            assert transformer.blocks[1].weight.device.type == "cuda"

            # Neighbor runs → eviction triggered via the forward_pre_hook.
            text_encoder.to(device)
            text_encoder(torch.zeros(4, device=device))

            torch.cuda.synchronize()
            assert state.resident is False
            assert transformer.head.weight.device.type == "cpu"
            assert transformer.blocks[0].weight.device.type == "cpu"
            assert transformer.blocks[1].weight.device.type == "cpu"

            # Transformer's own forward fires → repin.
            transformer(torch.zeros(2048, device=device))

            torch.cuda.synchronize()
            assert state.resident is True
            assert transformer.head.weight.device.type == "cuda"
            assert transformer.blocks[0].weight.device.type == "cuda"
            assert transformer.blocks[1].weight.device.type == "cuda"


class TestThreadSafety:
    def test_concurrent_register(self):
        mm = ModelManager()
        errors = []

        def register(name):
            try:
                for _ in range(100):
                    mm.register_component(name, DummyModel())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register, args=(f"model_{i}",)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(mm.component_names) == 4
