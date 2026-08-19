# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Reserved-pool budgeting for `block_pin`** (`auto_block_pin_allocator_inflation` /
  `..._windows`, `auto_block_pin_allocator_pool_overhead_gb` / `..._windows_gb`). The pin
  budget prices the caching allocator's *reserved pool* instead of peak *live* bytes,
  since the pool is what competes with pinned blocks for driver pages. Two terms: a fixed
  overhead plus a multiplier on the activation estimate. Neutral off Windows, where
  `expandable_segments` keeps pool close to live. Scoped to the pin budget — eviction and
  strategy choice keep live bytes, the former because `_effective_free_vram_gb` already
  adds the reclaimable pool back in.
- `block_pin` warns when the working set alone leaves no room, so no pin count can make
  the workload fit.

### Fixed
- **A failed pin no longer strands a component across two devices.** `apply_block_pin`
  moves the non-block parts and the pinned blocks onto the GPU before hooking the overflow
  blocks, so a failure part-way (OOM on a card that raises instead of spilling) left the
  front of the model resident with unhooked blocks on the CPU, and the next forward died on
  a device mismatch that reads as a model bug. The component is now reset to the CPU and
  rolled back to plain `group_offload`, the same degradation `block_pin` already applies to
  a component with no usable block list. The legacy-`weight_norm` residency path got the
  same reset for its own half-moved case.
- **No `expandable_segments` recommendation on ROCm.** The hint fired on any non-Windows
  build, but on HIP builds the flag is honoured (torch reads the CUDA-named variable first)
  and swaps `hipMalloc` for the HIP virtual-memory path, which has been reported to
  hard-fail on small allocations while the driver still reports many GiB free. That is not
  the fragmentation the flag addresses, so following the hint can break a working run. ROCm
  now gets a pointer to `auto_block_pin_allocator_inflation` /
  `auto_block_pin_allocator_pool_overhead_gb` instead, plus a note to unset the flag if it
  is already set.
- **The allocator config is read the way torch reads it**: `PYTORCH_CUDA_ALLOC_CONF`, then
  `PYTORCH_HIP_ALLOC_CONF` on ROCm, then the current unified `PYTORCH_ALLOC_CONF`. Users
  who had configured `expandable_segments` through either of the latter two were told to
  set it again.
- **Log messages are ASCII-only.** `logging` encodes with the stream's encoding, so a
  non-ASCII character dropped the whole record on a legacy-code-page console — including
  every `block_pin` rebalance, workload-probe and spill-recalibration line.
  `tests/test_log_messages.py` fails on any non-ASCII string reaching a logger, warning
  or exception.
- `managed()` was missing the allocator pool-overhead kwargs that `ModelManager` accepts.
- `test_ctor_arg_affects_resolver_decision` tuned the Linux working-set headroom without
  pinning `sys.platform`, so it only passed on Linux hosts.

## [0.4.0] - 2026-08-14

### Added
- **Model profiles**: `ModelProfile` and `register_model_profile`, a per-architecture
  table keyed by pipeline class holding the two facts the resolver cannot read off the
  module tree: denoiser concurrency and measured activation cost. Ships entries for the
  Wan 2.2 family, Ideogram4, MiniMax-H3, LTX-2 and Krea 2.
- **Call-time workload inference** (`ModelProfile.workload_fn`). The block_pin budget is
  computed from each call's `height` / `width` / `num_frames` before the pipeline runs, so
  `set_block_pin_workload` is no longer needed on a profiled architecture. Toggle with
  `block_pin_call_workload=`.
- **Bidirectional pin rebalancing**: the pin count can grow again, not only shrink, so a
  small generation following a large one recovers the blocks the large one shed.
- **`block_pin_workload_probe`** (default on) reads the real sequence length off the
  denoiser's first input and lowers the pin count before any activation is allocated. Makes
  `auto` safe on long video with nothing recorded by the caller.
- **`unload_text_encoders`** (opt-in), with `unload_text_encoders()`,
  `restore_dropped_components()` and `release_host_cache()` on `ModelManager`. Drops text
  encoders by role once denoising starts and releases the pinned host pool, which `del` plus
  `gc.collect()` does not: 44.3 GiB resident down to 7.9 on MiniMax-H3.
- **Legacy `weight_norm` guard**: components using the deprecated `weight_g`/`weight_v`
  spelling cannot be group offloaded, and are now kept resident with a warning instead of
  failing on a CPU/CUDA mismatch. Hits diffusers audio autoencoders and vocoders.
- `demo_scripts/measure_activation_slope.py` measures an architecture's activation slope
  from `config.json` alone, with no checkpoint download.

### Changed
- Default activation slope raised 0.118 to 0.16 GiB/ktoken; every model measured sat above
  the old value, LTX-2.3 (0.136) included. Expect slightly more conservative pin counts.
- A **measured** slope now takes a 1.2x safety factor rather than 1.5x. Keeping 1.5x on top
  of a measurement double-counts, badly enough to pin zero blocks on MiniMax-H3.
- `denoiser_concurrency` defaults to `None`, meaning "use the pipeline's profile, else
  `co_resident`". An explicit value still wins.
- `block_pin` pins denoisers only; other components with an incidental block list fall back
  to group offload instead of spending the pin budget.

### Fixed
- block_pin could pin more blocks than the workload left room for and OOM inside the first
  transformer forward.
- Auto-evict now releases the caching allocator's pool, so allocators outside PyTorch (cuDNN
  workspaces, Triton scratch) can actually use the freed VRAM.

## [0.3.1] - 2026-07-20

### Fixed
- `auto` no longer picks `model_offload` for pipelines with two or more **co-resident
  denoisers** (e.g. Ideogram4 True-CFG). The accelerate chain holds one component on the GPU
  at a time, so it cannot co-reside them and would bulk-swap a multi-GB DiT every step; the
  resolver now uses `block_pin`, or `group_offload` when there is no block list.
- block_pin auto-resolution targets the largest **block-bearing** component rather than the
  largest overall, so a pipeline whose heaviest component has no top-level block list still
  resolves to `block_pin` and pins the denoiser.

## [0.3.0] - 2026-07-20

### Changed
- **`auto` now prefers `block_pin` over `model_offload` when block_pin would fully pin the
  largest component**. Same VRAM peak, but the transformer stays resident across runs
  instead of being re-cycled every generation. WARNING: this changes which strategy `auto`
  resolves to on some hardware/model combinations. Pass one explicitly if you depend on it.
- The `model_offload` tier budgets against the **concurrent working set** (co-resident
  denoisers summed) rather than the largest single component, so dual-DiT True-CFG pipelines
  no longer pick it and then spill both denoisers to RAM.
- block_pin working-set headroom lowered to 2.0 GiB (Linux/macOS) and 3.0 GiB (Windows). It
  is now headroom *on top of* the workload-aware activation estimate rather than the whole
  flat margin (was 6.5 / 8.5 GiB).
- `managed()` wraps `__call__` at the **type level** via a per-pipe dynamic subclass instead
  of an instance attribute, so the device/dtype scope applies to every `pipe(...)` call.

### Added
- **Workload-aware block_pin working set**: `set_block_pin_workload(seq_len, batch, *,
  activation_scale)` scales the reserved margin with the actual job, with the exported
  `block_pin_activation_scale(...)` helper for LoRA / image / video conditioning and
  `auto_block_pin_act_*` knobs.
- **Multi-denoiser budgeting**: components are classified by role, and `denoiser_concurrency`
  (`"co_resident"` / `"sequential"`) decides whether denoisers are summed or maxed.
- **Windows spill-aware recalibration** (`block_pin_spill_aware`,
  `block_pin_spill_margin_gb`) unpins blocks when the allocator reserves more than the
  card's VRAM (sysmem fallback), self-tuning the pin count. No-op on Linux/macOS.
- **Modular-pipeline support**: `_execution_device` is patched to be group-offload-aware, so
  `group_offload` / `block_pin` no longer strand intermediates on the CPU.

### Fixed
- block_pin auto-evict warm-up thrash: the eviction check now uses *effective* free VRAM
  (driver-free plus PyTorch's reclaimable reserved pool), so it stops firing spuriously once
  the allocator's pool warms up (~1.8x slowdown seen on Klein 9B).

### Deprecated
- `AUTO_NO_OFFLOAD_FACTOR` / `auto_no_offload_factor`: the `no_offload` tier is additive now
  (`weights + working_set <= VRAM`). Unused, retained for backward compatibility.

## [0.2.1] - 2026-05-27

### Added
- All 12 auto-tuning constants exposed as keyword-only `ModelManager` ctor args
  and `managed()` kwargs (previously class-attribute overrides only).

### Fixed
- Windows free-RAM accounting now applies a ComfyUI-borrowed correction
  (`max(psutil.available, physical_total - (committed - vram_in_use))`, via
  `GetPerformanceInfo`) so WDDM commit-charge inflation no longer makes the
  resolver under-report available host RAM.

## [0.2.0] - 2026-05-14

### Added
- block_pin **cross-component auto-evict** (`block_pin_auto_evict`) with a
  four-tier eviction policy (per-component override, then no-states, then runtime
  free-VRAM check, then runtime RAM-absorb check), including VAE `decode`/`encode`
  method wrapping so eviction triggers on the calls that bypass `__call__`.
- `debug_vram_breakdown()` and `record_memory_history()` debugging helpers.

## [0.1.0] - 2026-05-14

### Added
- Initial release: `managed()`, `ModelManager`, and `remove_offload_hooks()`.
- Five offload strategies: `auto`, `no_offload`, `model_offload`, `group_offload`,
  `block_pin`.
- Size-aware `auto` resolution (VRAM + RAM + component sizes).
- Refcount-based, multi-pipeline-safe component registration.
- `block_pin` selective transformer-block pinning with per-block overflow streaming.
- `device_scope` / `get_device` / `get_dtype` ContextVars.
- Hash-keyed component cache and `remove_offload_hooks()` fix for diffusers'
  submodule hook-traversal bug.

[0.4.0]: https://github.com/asomoza/diffusers-mm/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/asomoza/diffusers-mm/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/asomoza/diffusers-mm/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/asomoza/diffusers-mm/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/asomoza/diffusers-mm/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/asomoza/diffusers-mm/releases/tag/v0.1.0
