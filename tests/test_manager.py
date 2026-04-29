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

    def test_group_offload_options(self):
        mm = ModelManager(group_offload_use_stream=True, group_offload_low_cpu_mem=True)
        assert mm.group_offload_use_stream is True
        assert mm.group_offload_low_cpu_mem is True


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

    def test_register_new_component_resets_applied_strategy(self):
        mm = ModelManager()
        mm._applied_strategy = "no_offload"
        mm.register_component("model", DummyModel())
        assert mm.applied_strategy is None

    def test_register_same_component_keeps_strategy(self):
        mm = ModelManager()
        model = DummyModel()
        mm.register_component("model", model)
        mm._applied_strategy = "no_offload"
        mm.register_component("model", model)  # same object
        assert mm.applied_strategy == "no_offload"


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
        mm = ModelManager(group_offload_use_stream=True, group_offload_low_cpu_mem=True)
        mm.register_component("model", DummyModel())
        mm.set_cached("key", "value")
        mm._applied_strategy = "no_offload"

        mm.clear()

        assert mm.get_component("model") is None
        assert mm.get_cached("key") is None
        assert mm.applied_strategy is None
        assert mm.group_offload_use_stream is False
        assert mm.group_offload_low_cpu_mem is False
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
