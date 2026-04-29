"""Hook management utilities for diffusers offloading."""

from __future__ import annotations

from typing import Any


_GROUP_OFFLOAD_HOOK_NAMES = (
    "group_offloading",
    "layer_execution_tracker",
    "lazy_prefetch_group_offloading",
)


def remove_offload_hooks(module: Any) -> None:
    """Remove diffusers group-offloading hooks from *module* and all submodules.

    Diffusers' own ``remove_hook(recurse=True)`` only recurses from the root's
    registry, which may not reach submodules whose parent lacks a
    ``_diffusers_hook`` attribute. This function iterates ``module.modules()``
    to ensure no nested hooks are missed.
    """
    for submodule in module.modules():
        if hasattr(submodule, "_diffusers_hook"):
            registry = submodule._diffusers_hook
            for hook_name in _GROUP_OFFLOAD_HOOK_NAMES:
                try:
                    registry.remove_hook(hook_name, recurse=False)
                except Exception:
                    pass
