"""Strategy-decision tests across simulated (VRAM, RAM, workload) envs.

Pure unit tests: no real GPU, no real model weights. Each parametrized
case constructs a ``FakeEnvManager`` with injected resource detectors
and checks that the auto resolver picks the strategy we expect. The
matrix grows by editing ``ENVS`` / ``PROFILES`` / ``EXPECTED_STRATEGY``
in ``_env_fakes.py`` — no test code changes needed for new presets.

For testing actual behavior (peak VRAM, completion) under a real
constraint, see ``test_envs_real.py``.
"""

from __future__ import annotations

import pytest
from _env_fakes import ENVS, EXPECTED_STRATEGY, PROFILES, FakeEnvManager


@pytest.mark.parametrize("env_name", list(ENVS))
@pytest.mark.parametrize("profile_name", list(PROFILES))
def test_resolved_strategy_matches_expected(env_name: str, profile_name: str) -> None:
    """For every (env, profile) with a recorded expectation, the auto
    resolver must pick the recorded strategy.

    Cases without an entry in ``EXPECTED_STRATEGY`` skip — keeps the
    matrix sparse-by-design so an unaudited combo can't masquerade as
    a passing test.
    """
    expected = EXPECTED_STRATEGY.get((env_name, profile_name))
    if expected is None:
        pytest.skip(f"no expected strategy recorded for ({env_name}, {profile_name})")

    vram, ram = ENVS[env_name]
    profile = PROFILES[profile_name]
    mm = FakeEnvManager(
        vram_avail_gb=vram,
        ram_avail_gb=ram,
        total_weights_gb=profile.total_weights_gb,
        max_component_gb=profile.max_component_gb,
        n_transformer_blocks=profile.n_transformer_blocks,
    )
    resolved = mm.resolve_offload_strategy("cuda")
    assert resolved == expected, f"env={env_name} profile={profile_name}: expected {expected}, got {resolved}"


def test_low_cpu_mem_stays_true_when_ram_tight() -> None:
    # 24 / 32 box + 38.5 GiB pipeline weights → ram (32) is not >=
    # weights+16 (54.5), so auto-tune keeps low_cpu_mem=True (defer
    # pinning, save RAM). This is the right move for a RAM-constrained
    # host even though it costs steady-state speed.
    profile = PROFILES["ltx23_distilled_sdnq_int8"]
    mm = FakeEnvManager(
        vram_avail_gb=24.0,
        ram_avail_gb=32.0,
        total_weights_gb=profile.total_weights_gb,
        max_component_gb=profile.max_component_gb,
        n_transformer_blocks=profile.n_transformer_blocks,
    )
    mm.resolve_offload_strategy("cuda")
    assert mm.group_offload_low_cpu_mem is True


def test_low_cpu_mem_flips_false_when_ram_plentiful() -> None:
    # Same VRAM + workload but 128 GiB RAM: 128 >= 38.5 + 16 holds, so
    # auto-tune flips low_cpu_mem=False. The user trades RAM for
    # steady-state speed because they have it to spare.
    profile = PROFILES["ltx23_distilled_sdnq_int8"]
    mm = FakeEnvManager(
        vram_avail_gb=24.0,
        ram_avail_gb=128.0,
        total_weights_gb=profile.total_weights_gb,
        max_component_gb=profile.max_component_gb,
        n_transformer_blocks=profile.n_transformer_blocks,
    )
    mm.resolve_offload_strategy("cuda")
    assert mm.group_offload_low_cpu_mem is False


def test_explicit_strategy_skips_auto_tune() -> None:
    # When the user picks a strategy explicitly, auto-tune of the
    # group_offload knobs must NOT fire — the user's chosen values stay.
    # We pick the (24, 128) env that would otherwise flip low_cpu_mem
    # to False and verify the explicit True survives.
    profile = PROFILES["ltx23_distilled_sdnq_int8"]
    mm = FakeEnvManager(
        vram_avail_gb=24.0,
        ram_avail_gb=128.0,
        total_weights_gb=profile.total_weights_gb,
        max_component_gb=profile.max_component_gb,
        n_transformer_blocks=profile.n_transformer_blocks,
        strategy="group_offload",
        group_offload_low_cpu_mem=True,
    )
    resolved = mm.resolve_offload_strategy("cuda")
    assert resolved == "group_offload"
    assert mm.group_offload_low_cpu_mem is True
