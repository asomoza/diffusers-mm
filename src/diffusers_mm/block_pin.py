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
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


logger = logging.getLogger(__name__)


@dataclass
class BlockPinState:
    """Snapshot of the pinned subset of one block_pin'd component.

    Carries everything :func:`evict_pinned_subset` / :func:`repin_pinned_subset`
    need to move the pinned subset between CPU and GPU after the fact —
    without re-running the discovery that :func:`apply_block_pin` did at
    apply time.

    The overflow (non-pinned) blocks are deliberately *not* tracked here:
    they have their own per-block ``apply_group_offloading`` hooks that
    manage their placement, and touching them from this state object would
    fight those hooks.

    Attributes:
        component: The top-level component (e.g. the transformer).
        block_attr: Attribute name of the block list on the component
            (e.g. ``"transformer_blocks"``).
        n_pinned: Number of leading blocks that were pinned. The pinned
            range is ``component.<block_attr>[:n_pinned]``.
        device: The GPU device the pinned subset lives on when resident.
        resident: ``True`` iff the pinned subset is currently on *device*.
            Flipped by :func:`evict_pinned_subset` / :func:`repin_pinned_subset`.
        pinned_size_bytes: Total bytes the pinned subset occupies — pinned
            blocks plus the non-block top-level parts that
            :func:`evict_pinned_subset` also moves. Cached at apply time so
            the auto-evict pre-forward hook (on the hot path) can decide
            "does host RAM have room for this eviction?" without re-walking
            module parameters every call.
    """

    component: nn.Module
    block_attr: str
    n_pinned: int
    device: torch.device
    resident: bool = True
    pinned_size_bytes: int = 0


def _iter_pinned_targets(state: BlockPinState):
    """Yield every ``nn.Module`` whose ``.to()`` should be called.

    Walks the same set :func:`apply_block_pin` originally moved to *device*:
    non-block top-level children + the first *n_pinned* blocks. Direct
    top-level params/buffers of the component are handled separately
    because they're not modules.
    """
    blocks = getattr(state.component, state.block_attr)
    for child_name, child in state.component.named_children():
        if child_name == state.block_attr:
            continue
        yield child
    for i in range(min(state.n_pinned, len(blocks))):
        yield blocks[i]


def evict_pinned_subset(state: BlockPinState) -> None:
    """Move *state*'s pinned subset to CPU and flip ``resident`` off.

    Idempotent — calling on an already-evicted state is a no-op. Mirrors
    the residency moves that :func:`apply_block_pin` did at apply time,
    just in the reverse direction.
    """
    if not state.resident:
        return
    cpu = torch.device("cpu")
    for mod in _iter_pinned_targets(state):
        mod.to(cpu)
    for _, p in state.component.named_parameters(recurse=False):
        p.data = p.data.to(cpu)
    for _, b in state.component.named_buffers(recurse=False):
        b.data = b.data.to(cpu)
    state.resident = False


def repin_pinned_subset(state: BlockPinState) -> None:
    """Move *state*'s pinned subset back to ``state.device`` and flip ``resident`` on.

    Idempotent — calling on an already-resident state is a no-op.
    """
    if state.resident:
        return
    for mod in _iter_pinned_targets(state):
        mod.to(state.device)
    for _, p in state.component.named_parameters(recurse=False):
        p.data = p.data.to(state.device)
    for _, b in state.component.named_buffers(recurse=False):
        b.data = b.data.to(state.device)
    state.resident = True


def unpin_blocks(state: BlockPinState, k: int, offload_kwargs: dict[str, Any]) -> int:
    """Unpin the last *k* currently-pinned blocks of *state* — they start streaming.

    Attaches per-block ``apply_group_offloading`` hooks (which offload them to
    CPU on init) to blocks ``[n_pinned - k : n_pinned]`` and lowers
    ``state.n_pinned``. Deliberately touches **only** those overflow block
    submodules — not the top-level component, not the blocks that stay pinned —
    so it is safe to call *between* denoising steps, including from a forward
    hook on the component itself: it never mutates the top-level module's own
    hook dict while that dict is being iterated.

    Returns the number of blocks actually unpinned.
    """
    from diffusers.hooks.group_offloading import apply_group_offloading

    blocks = getattr(state.component, state.block_attr, None)
    if blocks is None:
        return 0
    n = state.n_pinned
    k = max(0, min(k, n))
    if k == 0:
        return 0
    for i in range(n - k, n):
        apply_group_offloading(blocks[i], **offload_kwargs)
    state.n_pinned = n - k
    # Refresh the cached eviction footprint the auto-evict RAM check reads.
    per_block = per_block_size_bytes(blocks)
    non_block = non_block_size_bytes(state.component, state.block_attr)
    state.pinned_size_bytes = state.n_pinned * per_block + non_block
    return k


def pin_blocks(state: BlockPinState, k: int, device: torch.device) -> int:
    """Pin the next *k* streamed blocks of *state* — they stop streaming.

    The inverse of :func:`unpin_blocks`: strips the per-block group-offload
    hooks from blocks ``[n_pinned : n_pinned + k]``, moves them onto *device*,
    and raises ``state.n_pinned``. Lets the budget grow again when a later
    generation needs a smaller working set than the one that shrank it —
    without this, every pin count is a one-way ratchet downward and a small job
    following a large one keeps paying the streaming cost forever.

    **Not** safe mid-generation, unlike its counterpart: removing hooks walks
    the blocks' own hook dicts, so call it between calls (the managed
    ``__call__`` wrapper does, before the pipeline body runs), never from
    inside a forward hook.

    One caveat worth knowing: re-pinning restores the *speed* (no per-step
    transfer for those blocks) but not the *host RAM*. Once
    ``apply_group_offloading`` has run on a block, PyTorch's host caching
    allocator keeps its pinned staging buffers pooled for the life of the
    process, and hook removal does not return them — the same asymmetry that
    made ``apply_block_pin`` skip the pinned blocks entirely rather than
    hooking-then-unhooking them. Budgeting correctly the first time is
    therefore still worth more than rebalancing after the fact.

    Returns the number of blocks actually pinned.
    """
    from diffusers_mm.hooks import remove_offload_hooks

    blocks = getattr(state.component, state.block_attr, None)
    if blocks is None:
        return 0
    n = state.n_pinned
    k = max(0, min(k, len(blocks) - n))
    if k == 0:
        return 0
    # An evicted subset lives on the CPU and is brought back wholesale by
    # ``repin_pinned_subset`` on the component's next forward. Newly-pinned
    # blocks have to join it *there*, not on the GPU, or the subset would be
    # split across two devices and the repin would move only part of it.
    target = device if state.resident else torch.device("cpu")
    for i in range(n, n + k):
        remove_offload_hooks(blocks[i])
        blocks[i].to(target)
    state.n_pinned = n + k
    state.device = device
    # Refresh the cached eviction footprint the auto-evict RAM check reads.
    per_block = per_block_size_bytes(blocks)
    non_block = non_block_size_bytes(state.component, state.block_attr)
    state.pinned_size_bytes = state.n_pinned * per_block + non_block
    return k


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
