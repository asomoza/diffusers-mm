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


class TestManagedSharedManager:
    def test_two_pipelines_share_manager(self):
        mm = ModelManager(strategy="model_offload")
        pipe1 = FakePipeline(transformer=DummyModel(), vae=DummyModel())
        pipe2 = FakePipeline(transformer=DummyModel(), text_encoder=DummyModel())

        managed(pipe1, mm=mm, device="cpu")
        managed(pipe2, mm=mm, device="cpu")

        assert pipe1.mm is mm
        assert pipe2.mm is mm
        # All four components live in the same registry. pipe2's transformer
        # replaces pipe1's under the same name (this is the documented limit
        # — different pipelines with name collisions overwrite).
        assert sorted(mm.component_names) == ["text_encoder", "transformer", "vae"]

    def test_shared_module_across_pipelines_is_idempotent(self):
        mm = ModelManager(strategy="no_offload")
        shared = DummyModel()
        pipe1 = FakePipeline(text_encoder=shared)
        managed(pipe1, mm=mm, device="cpu")
        before = dict(mm._component_strategies)

        # A second pipeline declares the same shared module under the same
        # name — should be a no-op for strategy state.
        pipe2 = FakePipeline(text_encoder=shared)
        managed(pipe2, mm=mm, device="cpu")
        assert mm._component_strategies == before

    def test_external_mm_ignores_strategy_kwargs(self):
        # When mm is provided, strategy/use_stream kwargs are not used to
        # override the manager's existing configuration.
        mm = ModelManager(strategy="model_offload")
        pipe = FakePipeline(transformer=DummyModel())
        managed(pipe, mm=mm, strategy="no_offload", device="cpu")
        assert mm.offload_strategy == "model_offload"
