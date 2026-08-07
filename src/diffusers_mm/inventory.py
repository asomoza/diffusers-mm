"""Component classification for size-aware budgeting.

The auto-resolver and block_pin need to know *what* each registered component is,
not just its size. In particular:

- A pipeline can have **multiple denoisers** (Ideogram4: ``transformer`` +
  ``unconditional_transformer`` used together every step under True-CFG; Wan2.2:
  ``transformer`` + ``transformer_2`` high/low-noise experts used in different
  step ranges). The concurrent working set is the *sum* of co-resident denoisers,
  not the largest single one — using the max under-budgets and spills to RAM.
- A pipeline can have **multiple text encoders** (HiDream: four). They run
  sequentially before denoising, so their peak is the *largest single* one.

Roles are detected by component name (diffusers' naming is stable) with a
structural fallback. Name checks for text-encoder / VAE come **before** the
block-list heuristic, because text encoders (e.g. Qwen3-VL ``language_model.layers``)
also expose repeated-block lists and would otherwise be misread as denoisers.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

import torch.nn as nn

from diffusers_mm.block_pin import find_largest_block_list


# Denoiser component names in diffusers: unet, transformer, transformer_2 (Wan2.2),
# unconditional_transformer (Ideogram4), plus a generic *denoiser*/*prior* catch.
_DENOISER_NAME_RE = re.compile(r"^(unconditional_)?transformer(_\d+)?$|^unet(_\d+)?$|denoiser|^prior$")
_TEXT_ENCODER_NAME_RE = re.compile(r"^text_encoder(_\d+)?$")
_VAE_NAME_RE = re.compile(r"^(vae|vqvae|movq)(_\d+)?$")


@dataclass
class ComponentInfo:
    """One registered nn.Module component, classified for budgeting."""

    name: str
    role: str  # "denoiser" | "text_encoder" | "vae" | "other"
    size_gb: float
    n_blocks: int  # length of the largest repeated-block list (0 if none)
    block_attr: str | None  # attribute holding that block list


def module_size_gb(module: nn.Module) -> float:
    """Param + buffer bytes of *module*, in GiB."""
    try:
        size = sum(p.numel() * p.element_size() for p in module.parameters())
        size += sum(b.numel() * b.element_size() for b in module.buffers())
    except Exception:
        return 0.0
    return size / (1024**3)


def classify_role(name: str, module: nn.Module, n_blocks: int, min_blocks: int) -> str:
    """Classify a component by name, with a block-list structural fallback.

    Order matters: VAE and text-encoder name checks run before the block-list
    heuristic so block-list-bearing text encoders aren't misread as denoisers.
    """
    lname = name.lower()
    if _VAE_NAME_RE.match(lname) or "vae" in lname:
        return "vae"
    if _TEXT_ENCODER_NAME_RE.match(lname) or lname.startswith("text_encoder"):
        return "text_encoder"
    if _DENOISER_NAME_RE.search(lname):
        return "denoiser"
    # Structural fallback: an unrecognised component with a substantial repeated
    # block list is almost certainly a denoiser (novel transformer/unet naming).
    if n_blocks >= min_blocks:
        return "denoiser"
    return "other"


def build_inventory(
    components: dict[str, nn.Module],
    min_blocks: int,
    role_overrides: Mapping[str, str] | None = None,
) -> list[ComponentInfo]:
    """Classify every nn.Module in *components* (deduped by identity).

    *role_overrides* comes from the pipeline's :class:`ModelProfile` and wins over
    the name/structure heuristics — it exists for components whose role the
    generic rules get wrong (see :mod:`diffusers_mm.model_profiles`).
    """
    inventory: list[ComponentInfo] = []
    seen_ids: set[int] = set()
    for name, module in components.items():
        if not isinstance(module, nn.Module) or id(module) in seen_ids:
            continue
        seen_ids.add(id(module))
        result = find_largest_block_list(module)
        block_attr, blocks = (result[0], result[1]) if result is not None else (None, None)
        n_blocks = len(blocks) if blocks is not None else 0
        if role_overrides and name in role_overrides:
            role = role_overrides[name]
        else:
            role = classify_role(name, module, n_blocks, min_blocks)
        inventory.append(
            ComponentInfo(
                name=name,
                role=role,
                size_gb=module_size_gb(module),
                n_blocks=n_blocks,
                block_attr=block_attr,
            )
        )
    return inventory
