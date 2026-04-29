"""Tests for hook management utilities."""

from __future__ import annotations

from torch import nn

from diffusers_mm.hooks import remove_offload_hooks


class FakeHookRegistry:
    """Mimics diffusers' HookRegistry on a module."""

    def __init__(self):
        self._hooks = {}

    def add_hook(self, name: str, hook: object) -> None:
        self._hooks[name] = hook

    def remove_hook(self, name: str, recurse: bool = False) -> None:
        self._hooks.pop(name, None)


class TestRemoveOffloadHooks:
    def test_removes_hooks_from_root(self):
        model = nn.Linear(4, 4)
        registry = FakeHookRegistry()
        registry.add_hook("group_offloading", object())
        registry.add_hook("layer_execution_tracker", object())
        model._diffusers_hook = registry

        remove_offload_hooks(model)
        assert len(registry._hooks) == 0

    def test_removes_hooks_from_submodules(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        for submodule in model.modules():
            registry = FakeHookRegistry()
            registry.add_hook("group_offloading", object())
            submodule._diffusers_hook = registry

        remove_offload_hooks(model)
        for submodule in model.modules():
            assert len(submodule._diffusers_hook._hooks) == 0

    def test_noop_without_hooks(self):
        model = nn.Linear(4, 4)
        remove_offload_hooks(model)  # should not raise

    def test_ignores_non_offload_hooks(self):
        model = nn.Linear(4, 4)
        registry = FakeHookRegistry()
        registry.add_hook("some_other_hook", object())
        registry.add_hook("group_offloading", object())
        model._diffusers_hook = registry

        remove_offload_hooks(model)
        assert "some_other_hook" in registry._hooks
        assert "group_offloading" not in registry._hooks
