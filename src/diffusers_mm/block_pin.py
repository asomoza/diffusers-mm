"""Selective transformer-block pinning — internals for the ``block_pin`` strategy.

Pattern (validated empirically on LTX-2.3, see demo_scripts/test_demo_ltx23_block_pin.py):

1. Find the largest ``nn.ModuleList`` of repeated same-type children — this is
   the "block list" (e.g. ``transformer_blocks``).
2. Move every non-block top-level child to GPU (``proj_in``, ``proj_out``,
   ``time_embed``, norms, etc.).
3. Move direct top-level params/buffers of the component to GPU
   (``named_*(recurse=False)``) — some models have these (LTX-2 has
   ``scale_shift_table``).
4. Pin the first ``num_to_pin`` blocks on GPU permanently.
5. Apply ``apply_group_offloading`` per-block to the remaining (overflow) blocks.

Critical: the pinned blocks **never see** ``apply_group_offloading``, so no
``cpu_param_dict`` is ever allocated for them. PyTorch's host caching
allocator never holds those buffers, which is the only way to actually
recover the host RAM (``remove_offload_hooks`` after the fact does NOT
release pinned memory because the cache pools survive).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn


logger = logging.getLogger(__name__)


def find_largest_block_list(module: nn.Module) -> tuple[str, nn.ModuleList] | None:
    """Return ``(attr_name, module_list)`` for the largest repeated-block list, or ``None``.

    Walks top-level children. A "block list" is a ``nn.ModuleList`` with at
    least 2 entries where every entry is the same Python type (the standard
    pattern for transformer blocks, decoder layers, etc.). Among candidates,
    picks the one with the largest total parameter count — typically the
    transformer's ``transformer_blocks`` rather than something incidental.
    """
    candidates: list[tuple[str, nn.ModuleList, int]] = []
    for name, child in module.named_children():
        if not isinstance(child, nn.ModuleList) or len(child) < 2:
            continue
        first_type = type(child[0])
        if not all(type(c) is first_type for c in child):
            continue
        size = sum(p.numel() * p.element_size() for p in child.parameters())
        candidates.append((name, child, size))
    if not candidates:
        return None
    name, child, _ = max(candidates, key=lambda x: x[2])
    return name, child


def per_block_size_bytes(blocks: nn.ModuleList) -> int:
    """Bytes per block (params + buffers, dedup-by-id)."""
    if len(blocks) == 0:
        return 0
    seen: set[int] = set()
    total = 0
    for p in blocks[0].parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel() * p.element_size()
    for b in blocks[0].buffers():
        if id(b) in seen:
            continue
        seen.add(id(b))
        total += b.numel() * b.element_size()
    return total


def non_block_size_bytes(component: nn.Module, block_attr: str) -> int:
    """Bytes occupied by the component's non-block parts.

    Sum of params + buffers in every top-level child *except* the named block
    list, plus direct params/buffers of the component itself.
    """
    total = 0
    seen: set[int] = set()
    for name, child in component.named_children():
        if name == block_attr:
            continue
        for p in child.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel() * p.element_size()
        for b in child.buffers():
            if id(b) in seen:
                continue
            seen.add(id(b))
            total += b.numel() * b.element_size()
    for _, p in component.named_parameters(recurse=False):
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel() * p.element_size()
    for _, b in component.named_buffers(recurse=False):
        if id(b) in seen:
            continue
        seen.add(id(b))
        total += b.numel() * b.element_size()
    return total


def apply_block_pin(
    component: nn.Module,
    block_attr: str,
    blocks: nn.ModuleList,
    num_to_pin: int,
    device: torch.device,
    *,
    offload_kwargs: dict[str, Any],
) -> int:
    """Apply the block-pin pattern to *component*.

    1. Move every non-block top-level child to *device*.
    2. Move direct top-level params/buffers of *component* to *device*
       (catches things like LTX-2's ``scale_shift_table`` that live as
       direct attributes, not inside a child).
    3. Move the first ``num_to_pin`` blocks to *device*.
    4. Call ``apply_group_offloading`` per-block on the remaining blocks
       with the supplied ``offload_kwargs``.

    Returns the actual number of blocks pinned (clamped to ``[0, len(blocks)]``).
    """
    from diffusers.hooks.group_offloading import apply_group_offloading

    n = max(0, min(num_to_pin, len(blocks)))

    # 1: non-block children.
    for child_name, child in component.named_children():
        if child_name == block_attr:
            continue
        child.to(device)

    # 2: direct top-level params/buffers of the component.
    for _, p in component.named_parameters(recurse=False):
        p.data = p.data.to(device)
    for _, b in component.named_buffers(recurse=False):
        b.data = b.data.to(device)

    # 3: pinned blocks.
    for i in range(n):
        blocks[i].to(device)

    # 4: group_offload per overflow block.
    for i in range(n, len(blocks)):
        apply_group_offloading(blocks[i], **offload_kwargs)

    return n
