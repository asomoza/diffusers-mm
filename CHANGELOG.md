# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2026-07-20

### Fixed
- `auto` no longer picks `model_offload` for pipelines with **two or more
  co-resident denoisers** (e.g. Ideogram4 True-CFG's conditional +
  unconditional transformers). `model_offload`'s accelerate chain holds only one
  component on the GPU at a time, so it cannot co-reside them and would bulk-swap
  a multi-GB DiT CPU↔GPU on every denoise step. The resolver now skips the
  `model_offload` tier there and uses `block_pin` (or `group_offload` when there
  is no block list). Only applies under `denoiser_concurrency="co_resident"`.
- block_pin auto-resolution now targets the largest **block-bearing** component
  instead of the largest component overall. Pipelines whose heaviest component
  has no top-level block list (e.g. a text encoder marginally larger than each
  transformer, or a denoiser whose blocks are nested rather than top-level) now
  correctly resolve to `block_pin` and pin the denoiser's blocks, instead of
  bailing to `group_offload`.

## [0.3.0] - 2026-07-20

### Changed
- **`auto` now prefers `block_pin` over `model_offload` when block_pin would fully
  pin the largest component.** Same VRAM peak as `model_offload`, but the
  transformer stays resident across runs instead of being re-cycled every
  generation — strictly faster for repeated inference. ⚠️ **This changes which
  strategy `auto` resolves to on some hardware/model combinations** (previously
  `model_offload`, now `block_pin`). If you rely on a specific strategy, pass it
  explicitly.
- The `model_offload` auto tier now budgets against the **concurrent working set**
  (all co-resident denoisers summed) instead of the single largest component, so
  dual-DiT True-CFG pipelines no longer pick `model_offload` and then spill both
  denoisers plus activations to RAM.
- block_pin working-set headroom lowered to **2.0 GiB (Linux/macOS) / 3.0 GiB
  (Windows)** — it is now safety headroom added *on top of* the workload-aware
  activation estimate rather than the entire flat margin (was 6.5 / 8.5 GiB).
- `managed()` now wraps the pipeline's `__call__` at the **type level** (via a
  per-pipe dynamic subclass) instead of as an instance attribute, so the
  device/dtype scope reliably applies to every `pipe(...)` call.

### Added
- **Workload-aware block_pin working set.** `ModelManager.set_block_pin_workload(seq_len, batch, *, activation_scale)`
  records the denoise job so the reserved activation margin scales with it. The
  new module-level `block_pin_activation_scale(lora_count=, image_cond=,
  video_cond=, video_mode=)` helper (exported from `diffusers_mm`) computes the
  `activation_scale` for LoRA / image / video conditioning. Activation-fit
  constants are tunable via `auto_block_pin_act_*` ctor/`managed()` kwargs.
- **Multi-denoiser budgeting.** Registered components are now classified by role
  (denoiser / text_encoder / vae / other). The new `denoiser_concurrency`
  ctor/`managed()` kwarg (`"co_resident"` default, or `"sequential"`) controls
  whether multiple denoisers are budgeted as summed (both run every step —
  Ideogram4 True-CFG) or as the largest single one (one active at a time —
  Wan2.2 high/low-noise experts).
- **Windows spill-aware block_pin recalibration** (`block_pin_spill_aware`,
  default `True`; `block_pin_spill_margin_gb`, default `0.5`). After the first
  denoise step (and again after each generation), if the caching allocator
  reserved more than the card's VRAM (Windows sysmem fallback), pinned blocks are
  unpinned until the workload fits — self-tuning the pin count to the real
  activation footprint. No-op on Linux/macOS.
- **Modular-pipeline support.** `group_offload` / `block_pin` now work on
  diffusers experimental `ModularPipeline`s: `_execution_device` is patched to be
  group-offload-aware so intermediates are created on the compute device instead
  of CPU.

### Fixed
- block_pin auto-evict warm-up thrash: the runtime eviction check now uses
  *effective* free VRAM (driver-free **plus** PyTorch's reclaimable
  reserved-but-unallocated pool), so it no longer fires spuriously once the
  allocator's pool warms up over the first few runs (~1.8× slowdown observed
  previously on Klein 9B).

### Deprecated
- `AUTO_NO_OFFLOAD_FACTOR` / `auto_no_offload_factor` — the `no_offload` auto tier
  is now additive (`weights + working_set ≤ VRAM`) rather than a multiplier. The
  constant is unused but retained for backward compatibility.

## [0.2.1] - 2026-05-27

### Added
- All 12 auto-tuning constants exposed as keyword-only `ModelManager` ctor args
  and `managed()` kwargs (previously class-attribute overrides only).

### Fixed
- Windows free-RAM accounting now applies a ComfyUI-borrowed correction
  (`max(psutil.available, physical_total − (committed − vram_in_use))`, via
  `GetPerformanceInfo`) so WDDM commit-charge inflation no longer makes the
  resolver under-report available host RAM.

## [0.2.0] - 2026-05-14

### Added
- block_pin **cross-component auto-evict** (`block_pin_auto_evict`) with a
  four-tier eviction policy (per-component override → no-states → runtime
  free-VRAM check → runtime RAM-absorb check), including VAE `decode`/`encode`
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

[0.3.1]: https://github.com/asomoza/diffusers-mm/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/asomoza/diffusers-mm/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/asomoza/diffusers-mm/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/asomoza/diffusers-mm/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/asomoza/diffusers-mm/releases/tag/v0.1.0
