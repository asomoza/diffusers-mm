"""Per-architecture budgeting facts that cannot be read off the module tree.

Most of what the resolver needs it can measure: component sizes, block lists,
free VRAM. Two things it cannot:

1. **Whether multiple denoisers run together.** A dual-DiT pipeline gives no
   structural hint about *when* each DiT runs. Wan2.2 (``transformer`` +
   ``transformer_2``) and Ideogram4 (``transformer`` +
   ``unconditional_transformer``) look identical to
   :func:`~diffusers_mm.inventory.build_inventory` — two same-shaped denoisers —
   yet Wan splits them by timestep (only one resident at a time) while Ideogram4
   runs both every step under True-CFG. Budgeting them the same way is wrong for
   one of them: ``sequential`` on a co-resident pipeline under-budgets and spills
   to RAM.
2. **Roles that defy the naming conventions.** ``inventory``'s name patterns plus
   its block-list fallback cover diffusers' stable names, but new architectures
   keep arriving with names that need a hint.

This module is the lookup table for both, keyed by **pipeline class name**, so a
recognised pipeline is budgeted correctly with no arguments from the caller. It
is deliberately dependency-free (no torch) — it is data, plus a registry the
caller can extend:

    from diffusers_mm.model_profiles import ModelProfile, register_model_profile

    register_model_profile("MyCustomPipeline", ModelProfile(denoiser_concurrency="sequential"))

A profile is only ever a **default**: an explicit ``denoiser_concurrency=`` on
``ModelManager`` / ``managed()`` always wins, so a profile can never override a
caller who knows better.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


logger = logging.getLogger(__name__)

# MiniMax-H3's audio VAE latent rate, in latents per second of video.
_MINIMAX_H3_AUDIO_LATENTS_PER_SECOND = 40

# The two ways a pipeline's denoisers can share the step loop.
DENOISER_CONCURRENCY_MODES = ("co_resident", "sequential")

# The roles ``inventory.classify_role`` assigns, and therefore the only values a
# profile's ``roles`` mapping may use.
COMPONENT_ROLES = ("denoiser", "text_encoder", "vae", "other")


@dataclass(frozen=True)
class ModelProfile:
    """Budgeting facts for one pipeline architecture.

    Args:
        denoiser_concurrency: ``"co_resident"`` if every denoiser runs on every
            step (their weights must be summed), ``"sequential"`` if only one is
            active at a time (take the largest). ``None`` leaves the manager's
            own setting alone.
        roles: Component-name → role overrides, applied after
            :func:`~diffusers_mm.inventory.classify_role` and before budgeting.
            For names the generic patterns get wrong.
        act_slope_gb_per_ktoken: Measured denoise activation cost, GiB per 1000
            tokens, overriding ``AUTO_BLOCK_PIN_ACT_SLOPE_GB_PER_KTOKEN``. Fill
            from ``demo_scripts/measure_activation_slope.py``; the spread across
            architectures is wide enough (0.13 to 0.60) that a single default
            cannot serve them all.
        act_intercept_gb: Measured fixed activation cost, overriding
            ``AUTO_BLOCK_PIN_ACT_INTERCEPT_GB``. Usually near zero; worth setting
            only for a model with a real fixed overhead.
        workload_fn: ``(pipe, call_kwargs) -> (seq_len, batch) | None``, the
            architecture's own geometry maths. Called from the managed
            ``__call__`` wrapper on every generation, *before* the pipeline
            runs, so the block_pin budget is sized against the job actually
            being asked for instead of a fallback constant. This is the one
            budgeting input that depends on the request rather than the model,
            so it cannot be read off the module tree at ``managed()`` time —
            see :meth:`ModelManager.set_block_pin_workload`, which this
            automates. Return ``None`` for anything unrecognised (missing
            kwargs, an unexpected signature): the caller then keeps the
            previous budget and the forward-time workload probe still guards
            the run, so a wrong-shaped call can never be worse than no
            function at all.
        note: Why this profile says what it says — shown in the resolver log so a
            surprising budget can be traced back to its justification.
    """

    denoiser_concurrency: str | None = None
    roles: Mapping[str, str] = field(default_factory=dict)
    act_slope_gb_per_ktoken: float | None = None
    act_intercept_gb: float | None = None
    workload_fn: Callable[[Any, Mapping[str, Any]], tuple[int, int] | None] | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.denoiser_concurrency is not None and self.denoiser_concurrency not in DENOISER_CONCURRENCY_MODES:
            raise ValueError(
                f"denoiser_concurrency must be one of {DENOISER_CONCURRENCY_MODES} or None, "
                f"got {self.denoiser_concurrency!r}"
            )
        for name, role in self.roles.items():
            if role not in COMPONENT_ROLES:
                raise ValueError(f"role for {name!r} must be one of {COMPONENT_ROLES}, got {role!r}")
        if self.act_slope_gb_per_ktoken is not None and self.act_slope_gb_per_ktoken <= 0:
            raise ValueError(f"act_slope_gb_per_ktoken must be > 0, got {self.act_slope_gb_per_ktoken!r}")
        if self.act_intercept_gb is not None and self.act_intercept_gb < 0:
            raise ValueError(f"act_intercept_gb must be >= 0, got {self.act_intercept_gb!r}")


# Wan2.2-style high/low-noise experts: ``boundary_ratio`` splits the schedule so
# ``transformer`` handles timesteps >= the boundary and ``transformer_2`` handles
# the rest. Only one is ever active, so the working set is the larger of the two.
_TIMESTEP_SPLIT_EXPERTS = ModelProfile(
    denoiser_concurrency="sequential",
    note="transformer/transformer_2 are high/low-noise experts split by boundary_ratio",
)

# True-CFG: the step evaluates the conditional and unconditional transformers on
# the same latents and blends them, so both are resident for the whole step.
_TRUE_CFG_DUAL_DIT = ModelProfile(
    denoiser_concurrency="co_resident",
    note="transformer + unconditional_transformer both run every step under True-CFG",
)


def _minimax_h3_workload(pipe: Any, kwargs: Mapping[str, Any]) -> tuple[int, int] | None:
    """``(seq_len, batch)`` for a MiniMax-H3 request, from its own geometry.

    H3 denoises **one packed sequence** of text + audio + video rows, so the
    activation cost tracks the row count, not the pixel count. Every constant
    needed is on the pipeline: the video VAE's ``17``-pixel-frames-to-``5``-latent
    chunking and its spatial compression, and the transformer's patch. Mirrors
    ``MiniMaxH3PrepareLatentsStep``: ``sequence_length = num_text_tokens +
    num_condition_rows + num_audio_rows + num_video_rows``.

    Text and conditioning rows are deliberately **not** counted — the prompt
    length is not knowable here, and both are small next to the video rows (a
    few hundred against tens of thousands). Under-counting slightly is safe:
    the forward-time probe reads the true packed length off the denoiser's own
    input and raises the reserve if it matters.

    ``batch`` is 1: the released checkpoints are guidance-distilled and denoise a
    single stream. Were a CFG-batched variant to appear, the probe would observe
    ``batch=2`` on the first forward and re-budget upward.
    """
    height, width, num_frames = kwargs.get("height"), kwargs.get("width"), kwargs.get("num_frames")
    if not height or not width or not num_frames:
        return None

    frames_per_chunk = int(getattr(pipe, "vae_frames_per_chunk", 17))
    latents_per_chunk = int(getattr(pipe, "vae_latents_per_chunk", 5))
    ratio = int(getattr(pipe, "vae_spatial_compression_ratio", 16))
    patch = tuple(getattr(pipe, "patch_size", (1, 2, 2)))
    if frames_per_chunk <= 0 or latents_per_chunk <= 0 or ratio <= 0 or len(patch) != 3:
        return None
    _, patch_h, patch_w = (int(p) for p in patch)
    if patch_h <= 0 or patch_w <= 0:
        return None

    # Snap up to the next `frames_per_chunk * n + latents_per_chunk` the VAE can
    # encode, exactly as the pipeline's own `align_num_frames` does.
    aligned = int(num_frames)
    while aligned % frames_per_chunk != latents_per_chunk:
        aligned += 1
    latent_frames = (aligned - latents_per_chunk) // frames_per_chunk * latents_per_chunk + 2

    rows_per_frame = (int(height) // ratio // patch_h) * (int(width) // ratio // patch_w)
    video_rows = latent_frames * rows_per_frame

    # Audio rows: one latent per 1/40 s of video, packed channel-major.
    fps = float(getattr(pipe, "fps", 24.0)) or 24.0
    channels = int(getattr(pipe, "audio_channels", 2))
    audio_latents = math.ceil(aligned / fps * _MINIMAX_H3_AUDIO_LATENTS_PER_SECOND)
    audio_rows = audio_latents * max(1, channels)

    seq_len = video_rows + audio_rows
    return (seq_len, 1) if seq_len > 0 else None


# MiniMax-H3 ships both checkpoint partitions in one repository — ``transformer``
# for the t2va/fl2va workflows, ``transformer_ref`` for ref2va — but a workflow
# loads only its own, so they are never co-resident. The role hint covers
# ``transformer_ref``, which today only reaches "denoiser" via the block-list
# fallback rather than by name.
_MINIMAX_H3 = ModelProfile(
    denoiser_concurrency="sequential",
    roles={"transformer_ref": "denoiser"},
    act_slope_gb_per_ktoken=0.158,
    workload_fn=_minimax_h3_workload,
    note="transformer (t2va/fl2va) and transformer_ref (ref2va) are alternative partitions; a workflow loads one",
)

# Measured activation slopes (demo_scripts/measure_activation_slope.py, 2026-08,
# bf16, RTX 5090). Each is a least-squares fit over four sequence lengths and was
# linear to within 0.1%.
_LTX2_3 = ModelProfile(
    act_slope_gb_per_ktoken=0.136,
    note="measured 0.1364 GiB/ktoken over 4k-94k tokens (intercept ~0)",
)

_KREA2 = ModelProfile(
    act_slope_gb_per_ktoken=0.134,
    act_intercept_gb=0.10,
    note="measured 0.1336 GiB/ktoken over 4k-16k tokens; has a real ~0.1 GiB fixed cost, unlike the others",
)

# The outlier that justifies per-model slopes existing at all: Ideogram4 carries
# per-token LLM features at llm_features_dim=53248 across the *full* packed
# sequence (the pipeline pads the image positions with zeros rather than slicing),
# so one input tensor alone is ~104 KiB/token. 4.5x the other three, and it is a
# co-resident dual DiT on top, so under-budgeting it is doubly expensive.
_IDEOGRAM4 = ModelProfile(
    denoiser_concurrency="co_resident",
    act_slope_gb_per_ktoken=0.603,
    note=(
        "transformer + unconditional_transformer both run every step; "
        "measured 0.6025 GiB/ktoken (llm_features_dim=53248)"
    ),
)


MODEL_PROFILES: dict[str, ModelProfile] = {
    # Wan 2.2 family — verified against `boundary_ratio` in each pipeline.
    "WanPipeline": _TIMESTEP_SPLIT_EXPERTS,
    "WanImageToVideoPipeline": _TIMESTEP_SPLIT_EXPERTS,
    "WanVACEPipeline": _TIMESTEP_SPLIT_EXPERTS,
    "LucyEditPipeline": _TIMESTEP_SPLIT_EXPERTS,
    # Wan modular — same boundary_timestep switch, in
    # `modular_pipelines/wan/denoise.py`. The Wan22* classes subclass
    # WanModularPipeline, so MRO lookup would cover them from the base entry
    # alone; they are listed anyway so the dual-expert models are greppable by
    # name. Harmless on Wan 2.1 (one denoiser makes sum == max).
    "WanModularPipeline": _TIMESTEP_SPLIT_EXPERTS,
    "WanImage2VideoModularPipeline": _TIMESTEP_SPLIT_EXPERTS,
    "Wan22ModularPipeline": _TIMESTEP_SPLIT_EXPERTS,
    "Wan22Image2VideoModularPipeline": _TIMESTEP_SPLIT_EXPERTS,
    # Ideogram4 — verified: denoise.py calls both transformers per step.
    "Ideogram4Pipeline": _IDEOGRAM4,
    "Ideogram4ModularPipeline": _IDEOGRAM4,
    # MiniMax-H3.
    "MiniMaxH3ModularPipeline": _MINIMAX_H3,
    # LTX-2 family. NOTE: the slope was measured on an **LTX-2.3** config, and
    # LTX2Pipeline also serves LTX-2.0 checkpoints — the class name cannot tell
    # them apart. The two share the transformer class and hidden width, so the
    # slope should carry over; re-measure with an LTX-2.0 config before relying
    # on it there.
    "LTX2Pipeline": _LTX2_3,
    "LTX2ImageToVideoPipeline": _LTX2_3,
    "LTX2ConditionPipeline": _LTX2_3,
    "LTX2InContextPipeline": _LTX2_3,
    "LTX2HDRPipeline": _LTX2_3,
    "LTXModularPipeline": _LTX2_3,
    # Krea 2.
    "Krea2Pipeline": _KREA2,
    "Krea2ModularPipeline": _KREA2,
    "Krea2TurboModularPipeline": _KREA2,
}


def register_model_profile(class_name: str, profile: ModelProfile) -> None:
    """Register (or replace) the profile for a pipeline class name.

    Lets a caller teach the resolver about an architecture this module doesn't
    ship — including their own custom pipelines. Call before ``managed()`` /
    ``register_components`` so the profile is picked up at registration.
    """
    if not isinstance(profile, ModelProfile):
        raise TypeError(f"profile must be a ModelProfile, got {type(profile).__name__}")
    MODEL_PROFILES[class_name] = profile


def resolve_call_workload(source: Any, kwargs: Mapping[str, Any] | None) -> tuple[int, int] | None:
    """``(seq_len, batch)`` for this generation, via *source*'s profile, or ``None``.

    The bridge between a pipeline call and the block_pin budget: looks up the
    architecture's :attr:`ModelProfile.workload_fn` and runs it against the
    call's own keyword arguments. Returns ``None`` whenever the architecture is
    unprofiled, has no workload function, or that function declines — in every
    such case the caller keeps whatever budget it already had.

    Never raises. A profile's geometry maths is third-party-ish code running on
    the hot path of somebody's generation; a bug in it must degrade to "no
    estimate", not to a failed run.
    """
    profile = get_model_profile(source)
    if profile is None or profile.workload_fn is None:
        return None
    try:
        result = profile.workload_fn(source, kwargs or {})
    except Exception as e:
        logger.debug("workload_fn for %s failed: %s", type(source).__name__, e)
        return None
    if result is None:
        return None
    try:
        seq_len, batch = (int(v) for v in result)
    except (TypeError, ValueError):
        logger.debug("workload_fn for %s returned %r, expected (seq_len, batch)", type(source).__name__, result)
        return None
    if seq_len <= 0 or batch <= 0:
        return None
    return seq_len, batch


def get_model_profile(source: Any) -> ModelProfile | None:
    """Return the profile for *source*, or ``None`` if the architecture is unknown.

    *source* may be a pipeline instance, a class, or a class name. Instances and
    classes are matched along the MRO, so a subclass of a known pipeline inherits
    its profile. ``managed()``'s dynamic ``__call__`` subclass reuses the original
    class name, so a managed pipeline still resolves to the same profile.
    """
    if isinstance(source, str):
        return MODEL_PROFILES.get(source)
    klass = source if isinstance(source, type) else type(source)
    for base in getattr(klass, "__mro__", (klass,)):
        profile = MODEL_PROFILES.get(getattr(base, "__name__", ""))
        if profile is not None:
            return profile
    return None
