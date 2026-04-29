"""diffusers-mm — Smart model management for Hugging Face Diffusers pipelines."""

from diffusers_mm.hooks import remove_offload_hooks
from diffusers_mm.managed import managed
from diffusers_mm.manager import ModelManager


__all__ = [
    "ModelManager",
    "managed",
    "remove_offload_hooks",
]
