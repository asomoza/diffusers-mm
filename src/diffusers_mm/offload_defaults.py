"""Default tuning values for the ``auto`` / ``block_pin`` resolver.

Plain floats/ints with no ``torch`` import, so ``ModelManager`` and any
caller computing an activation scale can share one source of truth without
pulling heavy dependencies or drifting out of sync.

The activation-fit constants below are calibrated on LTX-2.3 distilled (a
video DiT) — see :meth:`ModelManager.set_block_pin_workload`. The denoise
working set is bf16 regardless of weight quantization, so the fit generalises
across int4 / int8 / bf16. For pipelines with a very different
activation-vs-sequence-length profile, override the matching ``AUTO_*``
attribute on the manager (subclass, ctor arg, or instance assignment).
"""

# --- block_pin working set (activation reserve) -----------------------------
# The working set reserved per block_pin component is
#   ``activation_estimate(seq_len, batch) * SAFETY_FACTOR + platform_headroom``.
# The activation estimate is a linear fit measured on a resolution/duration
# sweep: the denoise-loop peak minus the resident transformer weights is
# cleanly linear in ``seq_len`` with a small slope and near-zero intercept.
#   seq_len = (W / spatial) * (H / spatial) * ((frames - 1) / temporal + 1)
#
# These are the fallback for an architecture with no measured
# ``ModelProfile.act_slope_gb_per_ktoken``. The slope was raised from 0.118 to
# 0.16 in 2026-08 after direct measurement (see
# ``demo_scripts/measure_activation_slope.py``) found *every* model tested sat
# above the old value — including LTX-2.3, the model it was originally derived
# from, which measures 0.136. 0.16 covers the measured cluster (LTX-2.3 0.136,
# Krea 2 0.134, MiniMax-H3 0.158) with a little margin. It deliberately does
# *not* try to cover outliers like Ideogram4 (0.60): stretching the default that
# far would over-reserve for every ordinary model, so outliers get a profile.
# The intercept stays at 0.30 despite measuring near zero on most models — it is
# a cheap fixed cushion that matters only for small workloads, where being
# generous is free.
BLOCK_PIN_ACT_INTERCEPT_GB = 0.30
BLOCK_PIN_ACT_SLOPE_GB_PER_KTOKEN = 0.16
# Multiplier on the activation estimate before adding platform headroom
# (measured denoise peak -> safe ceiling, covering cross-model variance and
# neighbor-onload transients).
BLOCK_PIN_ACT_SAFETY_FACTOR = 1.5
# Safety factor used when the slope came from a **measured**
# ``ModelProfile.act_slope_gb_per_ktoken`` rather than the generic default.
# Most of the 1.5x above is cushion for not knowing the architecture's real
# slope; once it *is* known, keeping 1.5x double-counts. On MiniMax-H3 at
# 104k tokens, 1.5x on the measured slope reserves 27.1 GiB of a 27.7 GiB card
# and pins **zero** blocks — strictly worse than the under-budget it replaced.
# 1.2x reproduces roughly the pin count that measurably worked, and still
# covers the ~7% the probe misses (latents and text embeds held across the
# step, stream staging, allocator slack).
BLOCK_PIN_ACT_SAFETY_FACTOR_MEASURED = 1.2
# Activation estimate used when the denoise seq_len is unknown at pin time
# (i.e. the caller never called ``set_block_pin_workload``).
BLOCK_PIN_ACT_FALLBACK_GB = 4.0
# Platform safety headroom added on top of the activation estimate. Covers
# allocator fragmentation, the group-offload stream double-buffer, and modest
# attention overhead. Windows is higher because ``expandable_segments`` is
# Linux-only and the Windows allocator reserves more under the same load.
BLOCK_PIN_WORKING_SET_HEADROOM_GB = 2.0
BLOCK_PIN_WORKING_SET_HEADROOM_WINDOWS_GB = 3.0

# --- conditioning / LoRA activation scaling ---------------------------------
# These inflate the base activation estimate so block_pin pins fewer blocks up
# front when the denoise forward will allocate more than a plain text-to-X
# pass — avoiding reactive OOM. All scale with seq_len (they multiply the base
# activation), so they're expressed as additive contributions to a multiplier.
#
# LoRA: each active adapter adds ``lora_B(lora_A(x))`` forward temporaries.
BLOCK_PIN_LORA_ACT_FACTOR = 0.5
# Image conditioning (img2img / I2V): clean-latent blending + the image-encode
# VAE peak coexisting with the pinned transformer.
BLOCK_PIN_IMAGE_COND_ACT_FACTOR = 0.65
# Video conditioning (V2V replace): VAE-encodes the whole source clip while the
# transformer is pinned, plus multi-frame clean latents — the heaviest.
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

    This is a heuristic starting point calibrated on a video DiT; tune the
    ``BLOCK_PIN_*_ACT_FACTOR`` constants for your pipeline if needed.
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
