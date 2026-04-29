"""Tests for ModelManager core functionality."""

from __future__ import annotations

import threading

import pytest
import torch
from torch import nn

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
        # Defaults: use_stream + low_cpu_mem ON, record_stream OFF. These
        # match the "fast, RAM-conservative" leaf-level group offload that
        # diffusers itself recommends — and avoid the 2× host-RAM trap of
        # use_stream=True without low_cpu_mem_usage.
        mm = ModelManager()
        assert mm.group_offload_use_stream is True
        assert mm.group_offload_low_cpu_mem is True
        assert mm.group_offload_record_stream is False

    def test_group_offload_options_overrides(self):
        mm = ModelManager(
            group_offload_use_stream=False,
            group_offload_low_cpu_mem=False,
            group_offload_record_stream=True,
        )
        assert mm.group_offload_use_stream is False
        assert mm.group_offload_low_cpu_mem is False
        assert mm.group_offload_record_stream is True


class TestGroupOffloadKwargs:
    """Validate the kwargs builder's output for each setting combination."""

    def test_leaf_level_with_defaults(self):
        mm = ModelManager()
        kwargs = mm._group_offload_kwargs("cpu")
        assert kwargs["offload_type"] == "leaf_level"
        assert kwargs["use_stream"] is True
        assert kwargs["record_stream"] is False
        assert kwargs["low_cpu_mem_usage"] is True  # gated on use_stream

    def test_low_cpu_mem_dropped_when_streams_off(self):
        mm = ModelManager(group_offload_use_stream=False, group_offload_low_cpu_mem=True)
        kwargs = mm._group_offload_kwargs("cpu")
        assert kwargs["use_stream"] is False
        # low_cpu_mem is documented as only honoured when use_stream=True.
        assert "low_cpu_mem_usage" not in kwargs

    def test_record_stream_passed_through(self):
        mm = ModelManager(group_offload_record_stream=True)
        kwargs = mm._group_offload_kwargs("cpu")
        assert kwargs["record_stream"] is True


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
        cleaned: list = []
        monkeypatch.setattr("diffusers_mm.manager.remove_offload_hooks", lambda m: cleaned.append(m))

        mm = ModelManager(strategy="no_offload")
        old = DummyModel()
        mm.register_component("model", old)
        mm.apply_offload_strategy("cpu")

        new = DummyModel()
        mm.register_component("model", new)

        # old's refcount went 1 → 0 on displacement, so cleanup ran.
        assert cleaned == [old]


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
        # low_cpu_mem=True, record_stream=False).
        mm = ModelManager(
            group_offload_use_stream=False,
            group_offload_low_cpu_mem=False,
            group_offload_record_stream=True,
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
        assert mm.group_offload_record_stream is False
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
