"""Hook management utilities for diffusers and accelerate offloading."""

from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger(__name__)

_GROUP_OFFLOAD_HOOK_NAMES = (
    "group_offloading",
    "layer_execution_tracker",
    "lazy_prefetch_group_offloading",
)


def find_legacy_weight_norm(module: Any) -> str | None:
    """Return the name of a submodule using legacy ``weight_norm``, or ``None``.

    ``torch.nn.utils.weight_norm`` (the deprecated ``weight_g`` / ``weight_v``
    spelling) cannot be combined with diffusers' group offloading, and the
    failure is silent until it isn't:

    - ``weight_norm`` installs a ``_forward_pre_hook`` that recomputes
      ``module.weight`` from ``weight_g`` / ``weight_v`` and assigns it as a
      plain (non-Parameter) attribute.
    - Group offloading onloads a module's *parameters* from inside a wrapper
      around ``module.forward``, which runs **after** every
      ``_forward_pre_hook``.

    So ``weight`` is recomputed while ``weight_g`` / ``weight_v`` are still on
    the CPU, the parameters are then onloaded, and the op sees a CPU ``weight``
    against a CUDA input — ``"Input type (torch.cuda.FloatTensor) and weight
    type (torch.FloatTensor) should be the same"``. Since ``weight`` is not a
    Parameter, no amount of parameter shuffling fixes it; such a component has
    to stay resident.

    Affects diffusers audio autoencoders / vocoders, which keep the legacy
    spelling to match their checkpoints (e.g. MiniMax-H3's ``audio_vae``). The
    modern ``torch.nn.utils.parametrizations.weight_norm`` computes ``weight``
    on attribute access inside forward and is unaffected, so it is not flagged.
    """
    try:
        from torch.nn.utils.weight_norm import WeightNorm
    except Exception:  # pragma: no cover - torch layout change
        WeightNorm = None  # type: ignore[assignment]

    for name, submodule in module.named_modules():
        hooks = getattr(submodule, "_forward_pre_hooks", None)
        if not hooks:
            continue
        if WeightNorm is not None:
            if any(isinstance(h, WeightNorm) for h in hooks.values()):
                return name or type(submodule).__name__
        else:
            # Fallback: legacy weight_norm always leaves a ``<name>_g`` /
            # ``<name>_v`` parameter pair behind.
            params = getattr(submodule, "_parameters", {})
            if any(f"{p[:-2]}_v" in params for p in params if p.endswith("_g")):
                return name or type(submodule).__name__
    return None


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
