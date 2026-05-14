# diffusers-mm

Smart model management for Hugging Face Diffusers pipelines. A drop-in replacement for `enable_model_cpu_offload()` and `enable_group_offloading()` that's smarter, more configurable, and handles the edge cases diffusers doesn't.

## Installation

```bash
uv add diffusers-mm
```

## Quick Start

```python
import torch
from diffusers import LTX2Pipeline
from diffusers_mm import managed

pipe = LTX2Pipeline.from_pretrained("Lightricks/LTX-Video-0.9.7", torch_dtype=torch.bfloat16)
pipe = managed(pipe)  # auto strategy based on VRAM — just works
video = pipe(prompt="A cat walking on a beach")
```

## Offload Strategies

The `managed()` wrapper supports five offload strategies:

| Strategy | Description | When to use |
|----------|-------------|-------------|
| `"auto"` | Picks the best strategy based on available VRAM | Default — recommended for most users |
| `"no_offload"` | All components stay on GPU | 20+ GB VRAM |
| `"model_offload"` | Components moved to GPU one at a time | 12-20 GB VRAM |
| `"sequential_group_offload"` | Hooks applied per component on demand | 8-12 GB VRAM |
| `"group_offload"` | Leaf-level hooks on all components | < 8 GB VRAM |

### Auto Resolution

When `strategy="auto"` (the default), VRAM is checked and the strategy is resolved:

- >= 20 GB: `no_offload`
- >= 12 GB: `model_offload`
- >= 8 GB: `sequential_group_offload`
- < 8 GB or non-CUDA: `group_offload`

## Usage Examples

### Explicit Strategy

```python
pipe = managed(pipe, strategy="group_offload")
```

### Group Offload with CUDA Streams

```python
pipe = managed(
    pipe,
    strategy="group_offload",
    group_offload_use_stream=True,
    group_offload_low_cpu_mem=True,
)
```

### Per-Step Strategy Override (Advanced)

For decomposed pipelines or custom inference loops, use the `ModelManager` directly:

```python
pipe = managed(pipe, strategy="group_offload")

# Override strategy for specific components (e.g. VAE is too granular for leaf-level hooks)
with pipe.mm.use_components("vae", device="cuda", strategy_override="model_offload"):
    decoded = pipe.vae.decode(latents)
```

### Standalone ModelManager

If you're not using a standard pipeline (e.g. decomposed inference), use `ModelManager` directly:

```python
import torch
from diffusers_mm import ModelManager

mm = ModelManager(strategy="auto")

# Register components manually
mm.register_component("transformer", transformer)
mm.register_component("vae", vae)

# Apply strategy
mm.apply_offload_strategy("cuda")

# Use components with automatic placement
with mm.use_components("transformer", device="cuda"):
    output = transformer(latents)

# Hash-keyed caching for heavy models
key = mm.component_hash("/models/my-transformer")
cached = mm.get_cached(key)
if cached is None:
    model = load_model(...)
    mm.set_cached(key, model)

# Cleanup
mm.clear()
```

### Re-apply Hooks After LoRA

After loading LoRA adapters (which modify module structure), group offload hooks need to be refreshed:

```python
transformer.load_lora_adapter(state_dict, adapter_name="my_lora")
pipe.mm.reapply_group_offload("transformer", device="cuda")
```

## Comparison with Diffusers Built-in

| Feature | Diffusers | diffusers-mm |
|---------|-----------|-------------|
| Model CPU offload | `pipe.enable_model_cpu_offload()` | `managed(pipe, strategy="model_offload")` |
| Group offload | `pipe.enable_group_offloading(...)` | `managed(pipe, strategy="group_offload")` |
| Auto strategy | No | Yes — picks best strategy based on VRAM |
| CUDA streams + low CPU mem | Manual kwargs | `group_offload_use_stream=True, group_offload_low_cpu_mem=True` |
| Per-step strategy override | No | `use_components(..., strategy_override=...)` |
| Hook cleanup | Buggy (misses submodules) | Proper submodule-walking cleanup |
| Hook restore after override | No | Automatic in `use_components` finally block |
| Re-apply after LoRA | Manual | `mm.reapply_group_offload(name, device)` |
| Thread safety | No | RLock-guarded |
| Component caching | No | Hash-keyed cache |

## License

Apache 2.0
