"""Test helpers for simulating different VRAM/RAM environments.

Shared between ``test_envs_fast.py`` (pure decision tests with synthetic
modules) and ``test_envs_real.py`` (real models under
``torch.cuda.set_per_process_memory_fraction``). The fake manager
overrides the resource detectors so the strategy resolver sees whatever
VRAM/RAM/weights tuple the test injects, without requiring those values
to match the host hardware.

Extending the matrix is a one-place edit per axis: add to ``ENVS`` for a
new hardware preset, add to ``PROFILES`` for a new model size profile,
and add the expected resolution to ``EXPECTED_STRATEGY``.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from diffusers_mm.manager import ModelManager


# (vram_avail_gb, ram_avail_gb) — what each preset env reports.
ENVS: dict[str, tuple[float, float]] = {
    # Starting case: 24 GiB GPU on a 32 GiB system. Common consumer
    # workstation before Threadripper / 64 GiB kits became cheap.
    "vram24_ram32": (24.0, 32.0),
    # Big card: the transformer fits fully pinned, so block_pin (resident
    # across runs) should win over model_offload (re-cycles it each run).
    "vram32_ram64": (32.0, 64.0),
    # Tight card where the int4 transformer fits under model_offload's 1.5×
    # check but block_pin could only PARTIALLY pin it (avail < weights +
    # working_set). Locks in the protective guard: stay on model_offload here
    # rather than risk a partial pin that under-budgets video activations.
    "vram17_ram32": (17.0, 32.0),
}


@dataclass(frozen=True)
class Profile:
    """A synthetic model-component size profile.

    ``total_weights_gb`` and ``max_component_gb`` drive the auto
    resolver's main branches; ``n_transformer_blocks`` controls the
    size of the structural ``ModuleList`` registered on the fake
    transformer so the block-list check has something real to walk.
    Set it to 0 if the test will register its own real modules.
    """

    total_weights_gb: float
    max_component_gb: float
    n_transformer_blocks: int


PROFILES: dict[str, Profile] = {
    # LTX-2.3 distilled with SDNQ dynamic int8 quantization. Numbers
    # measured on the demo script (auto_budget_findings.md + skill doc):
    # transformer = 18.535 GiB int8, total pipeline weights = 38.5 GiB,
    # 48 transformer blocks.
    "ltx23_distilled_sdnq_int8": Profile(
        total_weights_gb=38.5,
        max_component_gb=18.535,
        n_transformer_blocks=48,
    ),
    # Same pipeline but with SDNQ dynamic int4 quantization. Numbers
    # from auto_budget_findings.md (per-component probe): transformer =
    # 10.769 GiB, total pipeline weights = 26.10 GiB. Same 48 blocks.
    # The int4 transformer fits under model_offload's 1.5× factor on
    # 24 GiB VRAM (10.769 × 1.5 = 16.15 ≤ 24), so this profile resolves
    # to a different strategy than int8 on the same env.
    "ltx23_distilled_sdnq_int4": Profile(
        total_weights_gb=26.10,
        max_component_gb=10.769,
        n_transformer_blocks=48,
    ),
}


# (env_name, profile_name) → expected resolved strategy. Keeping this
# as a table (rather than computing it from the rule) makes regressions
# in the resolver loud — when a constant moves, every affected cell
# fails individually instead of silently shifting.
EXPECTED_STRATEGY: dict[tuple[str, str], str] = {
    # 24 VRAM / 32 RAM + LTX-2.3 distilled int8:
    #   total × 1.5 = 57.75 > 24                       → not no_offload
    #   fully-pin needs 18.535 + 6.5 + 0.39 = 25.4 > 24 → not fully-pin block_pin
    #   max × 1.5 = 27.80 > 24                          → not model_offload
    #   48 blocks ≥ 8                                   → block_pin (partial)
    ("vram24_ram32", "ltx23_distilled_sdnq_int8"): "block_pin",
    # 24 VRAM / 32 RAM + LTX-2.3 distilled int4:
    #   total × 1.5 = 39.15 > 24                        → not no_offload
    #   fully-pin needs 10.769 + 6.5 + 0.22 = 17.5 ≤ 24 → block_pin (fully pinned).
    #   Previously model_offload (max × 1.5 = 16.15 ≤ 24), but block_pin keeps
    #   the transformer resident across runs, same VRAM peak, faster.
    ("vram24_ram32", "ltx23_distilled_sdnq_int4"): "block_pin",
    # 32 VRAM / 64 RAM: both quants fully pin → block_pin over model_offload.
    #   int8 fully-pin needs 25.4 ≤ 32 (was model_offload: 27.8 ≤ 32).
    #   int4 fully-pin needs 17.5 ≤ 32.
    ("vram32_ram64", "ltx23_distilled_sdnq_int8"): "block_pin",
    ("vram32_ram64", "ltx23_distilled_sdnq_int4"): "block_pin",
    # 17 VRAM / 32 RAM + int4: model_offload viable (16.15 ≤ 17) but NOT
    # fully-pinnable (17.5 > 17) → protective guard keeps model_offload.
    ("vram17_ram32", "ltx23_distilled_sdnq_int4"): "model_offload",
}


class _BlockyModule(nn.Module):
    """Stand-in for a transformer with N repeated blocks.

    The structure (``transformer_blocks = nn.ModuleList(...)``) matches
    what ``find_largest_block_list`` looks for, so the resolver's
    block-list check traverses something real. Each block is a 2×2
    ``nn.Linear`` so the module's actual CPU footprint stays
    negligible — the injected size from ``_estimate_components_size_gb``
    is what the resolver branches on for VRAM/RAM math.
    """

    def __init__(self, n_blocks: int) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList(nn.Linear(2, 2) for _ in range(n_blocks))


def make_blocky_module(n_blocks: int) -> nn.Module:
    return _BlockyModule(n_blocks)


class FakeEnvManager(ModelManager):
    """ModelManager subclass with injected resource values.

    Overrides the VRAM/RAM detectors and the components-size estimator
    so the strategy resolver sees the (vram, ram, total_weights,
    max_component) tuple the test specifies. This lets us test the
    resolver's branches deterministically without real hardware
    constraints or full-size model tensors.

    Pass ``n_transformer_blocks > 0`` to also auto-register a
    structurally correct fake transformer so the block-list branch can
    fire. Pass 0 when the test will register real pipeline components
    separately (the real-model test does this).
    """

    def __init__(
        self,
        *,
        vram_avail_gb: float,
        ram_avail_gb: float,
        total_weights_gb: float,
        max_component_gb: float,
        n_transformer_blocks: int = 0,
        **mm_kwargs,
    ) -> None:
        super().__init__(**mm_kwargs)
        self._fake_vram = vram_avail_gb
        self._fake_ram = ram_avail_gb
        self._fake_total = total_weights_gb
        self._fake_max = max_component_gb
        if n_transformer_blocks > 0:
            self.register_component("transformer", make_blocky_module(n_transformer_blocks))

    def _detect_available_vram_gb(self, device):
        return self._fake_vram, self._fake_vram

    def _detect_available_ram_gb(self):
        return self._fake_ram, self._fake_ram

    def _estimate_components_size_gb(self):
        return self._fake_total, self._fake_max
