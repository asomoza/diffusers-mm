"""Default tuning values for the ``auto`` / ``block_pin`` resolver.

Plain floats/ints with no ``torch`` import, so ``ModelManager`` and any
caller computing an activation scale can share one source of truth without
pulling heavy dependencies or drifting out of sync.

Every value here is a starting point calibrated on a video DiT, not a constant of
nature. Each has a matching ``AUTO_*`` attribute on ``ModelManager`` (subclass,
ctor arg, or instance assignment) for pipelines that behave differently.
"""

# --- block_pin working set (activation reserve) -----------------------------
# The working set reserved per block_pin component is
#   ``activation_estimate(seq_len, batch) * SAFETY_FACTOR + platform_headroom``
# where the activation estimate is a linear fit in the denoise sequence length,
#   seq_len = (W / spatial) * (H / spatial) * ((frames - 1) / temporal + 1)
# and activations are bf16 whatever the weights are, so the fit carries across
# quantization. Fallbacks only: an architecture with a measured
# ``ModelProfile.act_slope_gb_per_ktoken`` uses that instead, which is the better
# route for anything whose activations scale unusually.
BLOCK_PIN_ACT_INTERCEPT_GB = 0.30
BLOCK_PIN_ACT_SLOPE_GB_PER_KTOKEN = 0.16
# Lifts the bare fit to a safe ceiling, covering cross-model variance and
# neighbor-onload transients.
BLOCK_PIN_ACT_SAFETY_FACTOR = 1.5
# The same, for a profile-supplied slope: most of the factor above is cushion for
# not knowing the architecture's real slope, so keeping it on a measurement
# double-counts and can reserve enough to pin nothing at all.
BLOCK_PIN_ACT_SAFETY_FACTOR_MEASURED = 1.2
# Activation estimate used when the denoise seq_len is unknown at pin time
# (i.e. the caller never called ``set_block_pin_workload``).
BLOCK_PIN_ACT_FALLBACK_GB = 4.0

# --- caching-allocator reserved-pool model -----------------------------------
# Corrects the pin budget from peak *live* bytes to the allocator's *reserved
# pool*, which is what pinned blocks actually compete with: a fixed overhead plus
# a multiplier on the activation estimate. Neutral off Windows, where
# ``expandable_segments`` keeps pool close to live. Raise if
# ``max_memory_reserved`` runs above ``max_memory_allocated`` on your setup.
BLOCK_PIN_ALLOCATOR_INFLATION = 1.0
BLOCK_PIN_ALLOCATOR_INFLATION_WINDOWS = 1.25
BLOCK_PIN_ALLOCATOR_POOL_OVERHEAD_GB = 0.0
BLOCK_PIN_ALLOCATOR_POOL_OVERHEAD_WINDOWS_GB = 3.0
# Platform safety headroom on top of the activation estimate: allocator
# fragmentation, the group-offload stream double-buffer, attention overhead.
# Higher on Windows, which has no ``expandable_segments``.
BLOCK_PIN_WORKING_SET_HEADROOM_GB = 2.0
BLOCK_PIN_WORKING_SET_HEADROOM_WINDOWS_GB = 3.0

# --- VRAM held back from every budget ---------------------------------------
# Windows in WDDM mode satisfies allocations past the dedicated limit out of host RAM instead of raising, so
# nothing fails and every error-driven guard here stays unreachable — the run just gets slower as the borrowed
# RAM stops holding the offloaded weights. Reaching the ceiling is the thing to avoid rather than to detect, so
# a fixed slice is withheld from the free reading every budget derives from. Zero off Windows, where the
# allocator raises and the existing guards handle it. ComfyUI reserves the same way for the same reason
# (``model_management.py``: ``EXTRA_RESERVED_VRAM``, 600MB on Windows "because of the shared vram issue", plus
# 100MB on cards above 15GB); these values follow theirs.
VRAM_RESERVE_GB = 0.0
VRAM_RESERVE_WINDOWS_GB = 0.6
# Bigger cards get a bigger slice: the fallback is driven by the allocator's reserved pool, which scales with
# the card rather than with the workload.
VRAM_RESERVE_WINDOWS_LARGE_CARD_EXTRA_GB = 0.1
VRAM_RESERVE_LARGE_CARD_THRESHOLD_GB = 15.0

# --- conditioning / LoRA activation scaling ---------------------------------
# Inflate the base activation estimate so block_pin pins fewer blocks up front
# when the forward will allocate more than a plain text-to-X pass. They multiply
# the base activation, so they read as additive contributions to a multiplier.
#
# LoRA: each active adapter adds ``lora_B(lora_A(x))`` forward temporaries.
BLOCK_PIN_LORA_ACT_FACTOR = 0.5
# Image conditioning (img2img / I2V): clean-latent blending + the image-encode
# VAE peak coexisting with the pinned transformer.
BLOCK_PIN_IMAGE_COND_ACT_FACTOR = 0.65
# Video conditioning (V2V replace): VAE-encodes the whole source clip while the
# transformer is pinned, plus multi-frame clean latents.
BLOCK_PIN_VIDEO_COND_ACT_FACTOR = 1.5
# Keyframe / concat conditioning: appends tokens (can double the token count)
# and may disable flash attention (O(seq^2)). Kept >= replace as a guard.
BLOCK_PIN_KEYFRAME_COND_ACT_FACTOR = 2.0


def block_pin_activation_scale(
    *,
    lora_count: int = 0,
    image_cond: bool = False,
    video_cond: bool = False,
    video_mode: str = "replace",
) -> float:
    """Multiplier on the base denoise activation estimate for the active workload.

    Returns ``1.0`` for a plain text-to-X pass and grows with LoRAs +
    conditioning so block_pin reserves more working set (pins fewer blocks)
    before it would OOM. Conservative by design — pass the result as
    ``activation_scale`` to :meth:`ModelManager.set_block_pin_workload`.
    """
    scale = 1.0 + BLOCK_PIN_LORA_ACT_FACTOR * max(0, lora_count)
    if image_cond:
        scale += BLOCK_PIN_IMAGE_COND_ACT_FACTOR
    if video_cond:
        if video_mode in ("keyframe", "concat"):
            scale += BLOCK_PIN_KEYFRAME_COND_ACT_FACTOR
        else:
            scale += BLOCK_PIN_VIDEO_COND_ACT_FACTOR
    return scale
