"""Tests for the managed() pipeline wrapper."""

from __future__ import annotations

import pytest
from torch import nn

from diffusers_mm.managed import managed
from diffusers_mm.manager import ModelManager


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)


class FakePipeline:
    """Mimics a DiffusionPipeline with a components property."""

    def __init__(self, **components):
        self._components = components
        for name, comp in components.items():
            setattr(self, name, comp)

    @property
    def components(self):
        return dict(self._components)

    def __call__(self, prompt: str = "", **kwargs):
        return {"prompt": prompt, **kwargs}


class TestManagedBasic:
    def test_returns_same_pipeline(self):
        pipe = FakePipeline(transformer=DummyModel())
        result = managed(pipe, device="cpu")
        assert result is pipe

    def test_attaches_mm(self):
        pipe = FakePipeline(transformer=DummyModel())
        managed(pipe, device="cpu")
        assert isinstance(pipe.mm, ModelManager)

    def test_registers_nn_modules(self):
        transformer = DummyModel()
        vae = DummyModel()
        pipe = FakePipeline(transformer=transformer, vae=vae, tokenizer="not_a_module")
        managed(pipe, device="cpu")
        assert pipe.mm.get_component("transformer") is transformer
        assert pipe.mm.get_component("vae") is vae
        assert pipe.mm.get_component("tokenizer") is None

    def test_call_still_works(self):
        pipe = FakePipeline(transformer=DummyModel())
        managed(pipe, device="cpu")
        result = pipe(prompt="hello")
        assert result["prompt"] == "hello"

    def test_strategy_applied(self):
        pipe = FakePipeline(transformer=DummyModel())
        managed(pipe, strategy="model_offload", device="cpu")
        assert pipe.mm.applied_strategy == "model_offload"

    def test_no_offload_strategy(self):
        pipe = FakePipeline(transformer=DummyModel())
        managed(pipe, strategy="no_offload", device="cpu")
        assert pipe.mm.applied_strategy == "no_offload"


class TestManagedErrors:
    def test_no_components_property_raises(self):
        with pytest.raises(TypeError, match="no 'components' property"):
            managed(object(), device="cpu")

    def test_empty_components_warns(self, caplog):
        pipe = FakePipeline()  # no nn.Module components
        managed(pipe, device="cpu")
        assert "No nn.Module components found" in caplog.text


class TestManagedGroupOffloadOptions:
    def test_stream_options_forwarded(self):
        pipe = FakePipeline(transformer=DummyModel())
        managed(
            pipe,
            device="cpu",
            strategy="model_offload",
            group_offload_use_stream=True,
            group_offload_low_cpu_mem=True,
        )
        assert pipe.mm.group_offload_use_stream is True
        assert pipe.mm.group_offload_low_cpu_mem is True
