"""Hook management utilities for diffusers and accelerate offloading."""

from __future__ import annotations

from typing import Any


_GROUP_OFFLOAD_HOOK_NAMES = (
    "group_offloading",
    "layer_execution_tracker",
    "lazy_prefetch_group_offloading",
)


def remove_offload_hooks(module: Any) -> None:
    """Remove diffusers and accelerate offload hooks from *module* and submodules.

    Handles two hook systems:

    - Diffusers' group-offloading hooks (stored in per-module
      ``_diffusers_hook`` registries). Diffusers' own
      ``remove_hook(recurse=True)`` only recurses from the root's registry
      and may miss submodules whose parent lacks a ``_diffusers_hook``
      attribute, so we iterate ``module.modules()`` manually.
    - Accelerate's ``AlignDevicesHook`` and friends (installed by
      ``cpu_offload`` / ``cpu_offload_with_hook``, stored as ``_hf_hook``).
      Accelerate's own ``remove_hook_from_module(..., recurse=True)`` does
      handle nested cases correctly, so a single top-level call suffices.

    Idempotent and safe to call on a module with no hooks.
    """
    # Accelerate hooks — one top-level call handles the whole tree.
    try:
        from accelerate.hooks import remove_hook_from_module

        try:
            remove_hook_from_module(module, recurse=True)
        except Exception:
            pass
    except ImportError:
        pass

    # Diffusers group_offloading hooks — walk submodules manually because of
    # the diffusers recurse-traversal bug.
    for submodule in module.modules():
        if hasattr(submodule, "_diffusers_hook"):
            registry = submodule._diffusers_hook
            for hook_name in _GROUP_OFFLOAD_HOOK_NAMES:
                try:
                    registry.remove_hook(hook_name, recurse=False)
                except Exception:
                    pass
