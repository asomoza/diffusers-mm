"""Compatibility shims for diffusers' (experimental) modular pipelines.

Modular pipelines expose a ``.components`` dict and a ``_execution_device``
property, so ``managed()`` attaches to them fine. But their base-class
``_execution_device`` (``ModularPipeline`` in
``diffusers.modular_pipelines.modular_pipeline``) only inspects accelerate
``_hf_hook``s — it never learned the group-offloading branch that
``DiffusionPipeline._execution_device`` has. So under ``group_offload`` /
``block_pin`` (which use diffusers' own group-offloading hooks, not accelerate
hooks) it falls back to ``self.device`` = CPU. Blocks that create intermediates
on ``_execution_device`` (e.g. Ideogram4's text-encoder mask) then land on CPU
while the streamed submodules compute on CUDA → device-mismatch crash.

This patches ``ModularPipeline._execution_device`` to detect the group-offload
onload device first, mirroring ``DiffusionPipeline._execution_device``. It's
idempotent, guarded (no-op if modular pipelines aren't available), and strictly
more permissive — with no group-offload hooks present it behaves exactly as
before.
"""

from __future__ import annotations

import logging

import torch


logger = logging.getLogger(__name__)

_PATCH_TAG = "_dmm_group_offload_aware"


def _group_offload_aware_execution_device(self):
    """Drop-in for modular ``_execution_device`` that also detects group offload.

    Mirrors ``diffusers.pipelines.pipeline_utils.DiffusionPipeline._execution_device``.
    """
    from diffusers.hooks.group_offloading import _get_group_onload_device

    # Group offloading (leaf_level / block_pin overflow) hooks carry the onload
    # device; return it so intermediates are created on the compute device.
    for _name, model in self.components.items():
        if not isinstance(model, torch.nn.Module):
            continue
        try:
            return _get_group_onload_device(model)
        except ValueError:
            pass

    # Fall back to the original accelerate-hook logic.
    for _name, model in self.components.items():
        if not isinstance(model, torch.nn.Module):
            continue
        if not hasattr(model, "_hf_hook"):
            return self.device
        for module in model.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
    return self.device


def is_modular_pipeline(obj: object) -> bool:
    """True if *obj* is a diffusers ``ModularPipeline`` instance (guarded import)."""
    try:
        from diffusers.modular_pipelines.modular_pipeline import ModularPipeline
    except Exception:
        return False
    return isinstance(obj, ModularPipeline)


def patch_modular_execution_device() -> bool:
    """Make ``ModularPipeline._execution_device`` group-offload-aware. Idempotent.

    Returns True if the patch is in place (applied now or already present),
    False if modular pipelines aren't importable.
    """
    try:
        from diffusers.modular_pipelines.modular_pipeline import ModularPipeline
    except Exception:
        return False

    current = ModularPipeline.__dict__.get("_execution_device")
    if current is not None and getattr(current.fget, _PATCH_TAG, False):
        return True

    setattr(_group_offload_aware_execution_device, _PATCH_TAG, True)
    ModularPipeline._execution_device = property(_group_offload_aware_execution_device)
    logger.info(
        "diffusers-mm: patched ModularPipeline._execution_device to be group-offload-aware "
        "(enables group_offload/block_pin on modular pipelines)."
    )
    return True


def ensure_modular_compat(source: object) -> None:
    """If *source* is a modular pipeline, apply the ``_execution_device`` shim."""
    if is_modular_pipeline(source):
        patch_modular_execution_device()
