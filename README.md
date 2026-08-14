# diffusers-mm

Smart model management for Hugging Face Diffusers pipelines. A drop-in replacement for `enable_model_cpu_offload()` and `enable_group_offloading()` that's size-aware, more configurable, and handles the edge cases diffusers doesn't.

## Installation

```bash
uv add diffusers-mm
```

`diffusers` and `accelerate` are required (the library rides whatever versions are already installed rather than pinning specific ones).

## Quick Start

```python
import torch
from diffusers import LTX2Pipeline
from diffusers_mm import managed

pipe = LTX2Pipeline.from_pretrained("OzzyGT/LTX-2.3-Distilled", torch_dtype=torch.bfloat16)
pipe = managed(pipe)  # auto strategy based on VRAM + component sizes
video, audio = pipe(prompt="A cat walking on a beach")
```

`managed()` mutates the pipeline in place (registers components, installs hooks, wraps `__call__` with a device scope) and returns the same object with a `.mm` attribute exposing the underlying `ModelManager`.

## Offload Strategies

`managed()` supports five strategies via the `strategy=` argument:

| Strategy | Description | When auto picks it |
|----------|-------------|--------------------|
| `"auto"` | Resolves to one of the below based on VRAM, RAM, and component sizes | Default |
| `"no_offload"` | All components stay on GPU | Pipeline weights × 1.5 fit in VRAM |
| `"model_offload"` | Components stream onto GPU one at a time via an accelerate hook chain | Largest component × 1.5 fits in VRAM |
| `"block_pin"` | Pins as many transformer blocks on GPU as VRAM allows; streams the rest via leaf-level group_offload | Largest component is too big for `model_offload` but has ≥ 8 repeated blocks |
| `"group_offload"` | Leaf-level streaming on every component (diffusers' `apply_group_offloading` with the fast defaults) | Fallback when nothing else fits |

### Auto Resolution

When `strategy="auto"` (the default), the resolver looks at *available* VRAM and RAM (not total — so other processes on the GPU and host are accounted for) and the size of the registered components. The decision rule:

1. If `pipeline_weights × 1.5 ≤ available VRAM` → `no_offload`.
2. Else if `largest_component × 1.5 ≤ available VRAM` → `model_offload`.
3. Else if the largest component has a discoverable `nn.ModuleList` of ≥ 8 repeated same-type blocks → `block_pin`.
4. Otherwise → `group_offload`.

The `1.5×` factor (`AUTO_NO_OFFLOAD_FACTOR` / `AUTO_MODEL_OFFLOAD_FACTOR`) is the activation budget — empirically validated for SDNQ int8 LTX-2.3 to give ~0.3 GiB margin above peak.

If no components are registered yet, the resolver falls back to a VRAM-only tier table:

| Available VRAM | Strategy |
|----------------|----------|
| ≥ 20 GB | `no_offload` |
| ≥ 12 GB | `model_offload` |
| < 12 GB / non-CUDA | `group_offload` |

If pipeline weights exceed `RAM × 0.85`, a warning is logged — the workload likely won't fit on host memory regardless of strategy.

### Block-pin tuning

The `block_pin` strategy fills the gap between `model_offload` (largest component must fit) and `group_offload` (everything streams, transformer pays transfer cost on every step). It pins as many transformer blocks as VRAM allows on the GPU permanently, and streams the rest via `apply_group_offloading(offload_type="leaf_level")`.

The pin count is auto-budgeted per component:

```python
pipe = managed(pipe)  # auto-budget block_pin on whatever component is biggest
```

Override the per-component count explicitly when the auto budget is wrong for your workload (e.g. very high activation cost):

```python
pipe.mm.set_block_pin_count("transformer", 30)
```

**Workload-aware working set.** The VRAM reserved per block_pin component for denoise activations scales with the actual job rather than a flat constant. Record the workload before the strategy is resolved so the auto-budget pins the right number of blocks:

```python
from diffusers_mm import block_pin_activation_scale

mm = ModelManager()
# seq_len = latent_frames * latent_h * latent_w; batch = 2 under CFG.
mm.set_block_pin_workload(
    seq_len=24 * 16 * 24, batch=2,
    activation_scale=block_pin_activation_scale(lora_count=1, video_cond=True),
)
pipe = managed(pipe, mm=mm)  # resolver now reserves a workload-sized working set
```

The reserve is `activation_estimate(seq_len, batch) × safety + platform_headroom`, where the activation estimate is a linear fit measured on a video DiT (bf16 activations, so it generalises across int4/int8/bf16). When no workload is recorded it falls back to a fixed estimate. The platform-headroom knobs (`auto_block_pin_working_set_gb`, default `2.0` GiB Linux / `3.0` Windows) sit on top of this and rarely need changing — in releases ≤ 0.2.x they *were* the entire flat margin (`6.5`/`8.5`).

For `block_pin` to budget tightly, set the env var before starting Python:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Without it, allocator fragmentation can eat ~1-2 GiB and a careful budget can OOM. The strategy logs a warning if it's missing on apply.

**Windows: the reserved pool, not the live bytes.** `expandable_segments` is inert on Windows — torch parses the flag, warns `expandable_segments not supported on this platform`, and ignores it, because the CUDA virtual-memory API it needs isn't exposed by the Windows driver. The allocator therefore accumulates fixed segments, and its *reserved pool* settles above the peak *live* bytes a forward holds: partly by a fixed amount (streamed block weights it holds without reusing), partly in proportion to the activations. The pool is what competes with pinned blocks for driver pages, so on Windows the pin budget corrects for both before subtracting:

```python
pipe = managed(
    pipe,
    auto_block_pin_allocator_inflation_windows=1.25,        # multiplier on the seq_len term
    auto_block_pin_allocator_pool_overhead_windows_gb=3.0,  # fixed, on top of the headroom
)
```

Set both to `1.0` / `0.0` to opt out. To calibrate for your own setup, log `torch.cuda.max_memory_reserved()` against `max_memory_allocated()` after one denoise step at two different sequence lengths and fit the line. The defaults come from one video DiT on one card and the fixed term follows your block count and per-block size rather than the model, so treat them as a starting point and measure — reserved also grows in segment-sized steps rather than smoothly.

If the working set alone exceeds available VRAM, no pin count can help — streaming every block still spills — and `block_pin` says so explicitly rather than leaving you with a run that is quietly several times slower. The fix there is a smaller job: the working set scales with the denoise sequence length, so fewer frames or a smaller canvas is what moves it.

### Tuning the auto resolver

Every threshold the `"auto"` strategy uses internally is also a constructor argument on `ModelManager` (and a `managed()` kwarg). The defaults are tuned for image diffusion on consumer hardware — bump them for unusual workloads (high-activation video, very RAM-tight machines, etc.). All are keyword-only:

```python
pipe = managed(
    pipe,
    # --- Strategy thresholds (1.5× ≈ activation budget) ---
    auto_no_offload_factor=1.5,                # pipeline_weights × this ≤ VRAM → no_offload
    auto_model_offload_factor=1.5,             # largest_component × this ≤ VRAM → model_offload
    auto_block_pin_min_blocks=8,               # min ModuleList size to pick block_pin over group_offload

    # --- RAM safety margins ---
    auto_ram_headroom=0.85,                    # warn if pipeline_weights > RAM × this
    auto_low_cpu_mem_ram_headroom_gb=16.0,     # flip group_offload's low_cpu_mem=False if RAM ≥ weights + this

    # --- block_pin VRAM budget ---
    # The working set is now workload-aware (see set_block_pin_workload). These
    # two are the platform safety headroom added on top of the activation estimate.
    auto_block_pin_working_set_gb=2.0,         # headroom per component (Linux/macOS)
    auto_block_pin_working_set_windows_gb=3.0, # same, Windows (allocator runs in fixed-segment mode)
    auto_block_pin_ram_evict_headroom_gb=4.0,  # auto-evict only if RAM ≥ evicted_subset + this

    # --- block_pin activation fit (workload-aware working set) ---
    auto_block_pin_act_intercept_gb=0.30,         # base activation at seq_len→0
    auto_block_pin_act_slope_gb_per_ktoken=0.118, # +GiB per 1000 (batch × seq_len) tokens
    auto_block_pin_act_safety_factor=1.5,         # multiplier before adding headroom
    auto_block_pin_act_fallback_gb=4.0,           # estimate used when no workload recorded

    # --- allocator reserved-pool model (pin budget only) ---
    auto_block_pin_allocator_inflation=1.0,             # multiplier on the seq_len term
    auto_block_pin_allocator_inflation_windows=1.25,    # same, Windows (no expandable_segments)
    auto_block_pin_allocator_pool_overhead_gb=0.0,      # fixed pool overhead
    auto_block_pin_allocator_pool_overhead_windows_gb=3.0,
)
```

Each knob in detail:

| Knob | Default | Raise it when… | Lower it when… |
|---|---|---|---|
| `auto_no_offload_factor` | 1.5 | Activations are large relative to weights — pick a more conservative `no_offload`. | Activations are small, you want to stay GPU-resident more aggressively. |
| `auto_model_offload_factor` | 1.5 | Largest component runs into VRAM ceiling at peak — push toward `block_pin` / `group_offload`. | Same direction as above, just for the per-component tier. |
| `auto_ram_headroom` | 0.85 | RAM is tight and you want the "won't fit" warning to fire earlier. | RAM is plentiful and the warning is noisy. |
| `auto_low_cpu_mem_ram_headroom_gb` | 16.0 | RAM-constrained system, prefer staying in low-RAM mode (slower steady-state but stable). | RAM-rich system, bias toward `low_cpu_mem=False` for faster transfers. |
| `auto_block_pin_working_set_gb` | 2.0 | Allocator fragmentation / attention overhead is unusually high (the workload-aware estimate already covers normal activations). | Rarely — it's just platform headroom now. |
| `auto_block_pin_working_set_windows_gb` | 3.0 | Same as above, on Windows. | Same. |
| `auto_block_pin_min_blocks` | 8 | Your block list is small enough that per-block hook overhead dominates. | You want `block_pin` even for shallow transformers (rarely useful). |
| `auto_block_pin_ram_evict_headroom_gb` | 4.0 | Neighbors have unusually large host-side staging (large pinned buffers, big activations). | Neighbors are lightweight and you want eviction to fire more readily. |
| `auto_block_pin_act_slope_gb_per_ktoken` | 0.118 | Your pipeline's activations grow faster with sequence length than the measured video-DiT fit. | Activations are flatter (e.g. heavily windowed attention). |
| `auto_block_pin_act_fallback_gb` | 4.0 | You can't call `set_block_pin_workload` and your typical job has large activations. | Typical job is small and you want more blocks pinned by default. |
| `auto_block_pin_allocator_inflation` | 1.0 | `max_memory_reserved()` grows faster than `max_memory_allocated()` as you raise the sequence length. | You verified the pool tracks live bytes closely. |
| `auto_block_pin_allocator_inflation_windows` | 1.25 | Your measured slope ratio is steeper. | Shallower. Set to `1.0` with the overhead at `0.0` to opt out entirely. |
| `auto_block_pin_allocator_pool_overhead_gb` | 0.0 | `reserved - allocated` has a floor independent of sequence length. | It doesn't. |
| `auto_block_pin_allocator_pool_overhead_windows_gb` | 3.0 | More or larger streamed blocks than the default was calibrated on. | Fewer or smaller streamed blocks, or the whole denoiser is pinned. |

These can also be set after construction by assigning the matching ALL_CAPS attribute on the manager — e.g. `pipe.mm.AUTO_BLOCK_PIN_WORKING_SET_GB = 12.0`. The ctor arg is shorthand for "do that at construction time."

## Usage Examples

### Explicit strategy

```python
pipe = managed(pipe, strategy="group_offload")
```

### Group offload tuning

The two main knobs (defaults match the recommended fast config):

```python
pipe = managed(
    pipe,
    strategy="group_offload",
    group_offload_use_stream=True,    # overlap transfers with compute
    group_offload_low_cpu_mem=True,   # defer pinning per-transfer (saves RAM)
)
```

Without `low_cpu_mem_usage=True`, a full pinned host copy of every weight is held for the entire inference (~2× host RAM). This pairing is enforced — `low_cpu_mem` is dropped from kwargs when `use_stream=False`.

### Shared manager (multiple pipelines)

When you have multiple pipelines sharing components — e.g. an LTX-2 base and refiner sharing the same T5 and VAE — pass a single `ModelManager` to both `managed()` calls. The manager refcounts shared modules so they aren't re-hooked, and unregistering one pipeline doesn't pull components out from under the other:

```python
from diffusers_mm import ModelManager, managed

mm = ModelManager(strategy="auto")
pipe1 = managed(pipe1, mm=mm, device="cuda")
pipe2 = managed(pipe2, mm=mm, device="cuda")  # T5 + VAE shared, transformer separate

# Later, just unregister one — the other keeps working
mm.unregister_components(pipe1)
```

When `mm=` is passed, the strategy/group_offload kwargs are ignored (the manager owns its own configuration).

### Per-step strategy override

For decomposed pipelines (calling components individually) where the global strategy doesn't fit a specific step:

```python
pipe = managed(pipe, strategy="group_offload")

# VAE is too granular for leaf-level hooks — temporarily switch to model_offload
with pipe.mm.use_components("vae", device="cuda", strategy_override="model_offload"):
    decoded = pipe.vae.decode(latents)
# Original group_offload hooks are restored automatically on exit
```

### Standalone `ModelManager`

If you're not using a standard `DiffusionPipeline` (custom inference loop, decomposed graph), drive `ModelManager` directly:

```python
import torch
from diffusers_mm import ModelManager

mm = ModelManager(strategy="auto")
mm.register_component("transformer", transformer)
mm.register_component("vae", vae)
mm.apply_offload_strategy("cuda")

with mm.use_components("transformer", device="cuda"):
    output = transformer(latents)

# Cross-pipeline component caching (load-or-reuse)
def load_my_transformer():
    return MyTransformer.from_pretrained(...)

transformer = mm.load_component(
    "transformer",
    identifier="/models/my-transformer",
    factory=load_my_transformer,
)
# A second call with the same identifier returns the cached module
# without invoking the factory.

mm.clear()  # remove hooks, drop components, gc + empty_cache
```

### Re-apply hooks after LoRA

Loading LoRA adapters adds new submodules to the transformer; those new submodules won't have offload hooks unless re-applied:

```python
transformer.load_lora_adapter(state_dict, adapter_name="my_lora")
pipe.mm.reapply_group_offload("transformer", device="cuda")
```

### Standalone hook cleanup

Sometimes you need to strip diffusers' group-offload hooks from a module tree without going through the manager — e.g. before serializing or transferring weights. The library exposes a submodule-walking cleanup that fixes diffusers' `remove_hook(recurse=True)` bug (it misses submodules whose parent lacks a `_diffusers_hook` attribute):

```python
from diffusers_mm import remove_offload_hooks

remove_offload_hooks(module)  # idempotent; safe if no hooks installed
```

### Debugging memory

`record_memory_history` is a context manager around `torch.cuda.memory._record_memory_history` that dumps a snapshot pickle on exit. No-op when CUDA is unavailable so it's safe to leave in CPU-only test runs:

```python
with pipe.mm.record_memory_history("trace.pickle"):
    pipe(prompt="...")
# Visualize with:
#   python -m torch.cuda._memory_viz trace_plot trace.pickle -o trace.html
# or upload to https://docs.pytorch.org/memory_viz
```

## Comparison with Diffusers built-ins

| Feature | Diffusers | diffusers-mm |
|---------|-----------|--------------|
| Model CPU offload | `pipe.enable_model_cpu_offload()` | `managed(pipe, strategy="model_offload")` |
| Group offload | `pipe.enable_group_offloading(...)` | `managed(pipe, strategy="group_offload")` (defaults match the fast config) |
| Block-level pinning | Not available | `managed(pipe, strategy="block_pin")` |
| Auto strategy | No | Yes — size-aware (looks at VRAM, RAM, and component sizes) |
| Per-step strategy override | No | `mm.use_components(..., strategy_override=...)` |
| Hook cleanup | `remove_hook(recurse=True)` misses nested submodules | `remove_offload_hooks(module)` walks all submodules |
| Hook restore after override | No | Automatic in `use_components` `finally` block |
| Re-apply after LoRA | Manual | `mm.reapply_group_offload(name, device)` |
| Shared components across pipelines | No tracking | Refcount + per-source registration |
| Thread safety | No | RLock-guarded |
| Component caching | No | Hash-keyed cache + `load_component(identifier, factory)` |

## Development

```bash
make format       # auto-format with ruff
make lint-fix     # auto-fix lint issues
make check        # CI-friendly: format-check + lint (no modifications)
make test         # CPU-only unit tests (~2s)
make cov          # coverage report (terminal)
make cov-html     # coverage report (HTML, in htmlcov/)
```

Real-model GPU tests are opt-in (require a CUDA device + downloaded weights):

```bash
make test-envs-fast   # strategy-decision tests with synthetic modules (fast)
make test-envs-real   # real LTX-2.3 distilled inference under a 24 GiB VRAM cap
```

The real-env tests cap VRAM via a held dummy tensor (genuine `cudaMalloc` OOM if exceeded). For an additional kernel-enforced RAM cap, wrap the invocation in a cgroup:

```bash
systemd-run --user --scope -p MemoryMax=32G -p MemorySwapMax=0 make test-envs-real
```

## License

Apache 2.0
