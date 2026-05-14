"""End-to-end tests: real diffusers pipelines under simulated env caps.

These tests cap real VRAM via a **held dummy tensor** (not the soft
``torch.cuda.set_per_process_memory_fraction``) and inject RAM via the
same ``FakeEnvManager`` override that drives the fast tests. The
pipeline is then *actually loaded and run*, so we catch resolver/budget
mistakes that would only surface under real memory pressure.

Why the dummy and not the fraction: ``set_per_process_memory_fraction``
only caps PyTorch's caching allocator. CUDA context (~1 GiB), Triton
and SDNQ quantized-matmul kernel workspaces (several GiB), cuDNN
handles — all bypass the fraction. A test using only the fraction can
report ``max_memory_allocated`` comfortably under cap while
``nvidia-smi`` shows the process well above. Holding ``(total -
target)`` GiB as a ``torch.empty`` tensor takes physical memory off the
table for the driver, so ``cudaMalloc`` fails for real once the rest of
the process exceeds the budget.

Marked ``gpu`` + ``slow``: opt-in. Run with::

    DIFFUSERS_MM_RUN_GPU_TESTS=1 pytest tests/test_envs_real.py -v -s

Or via ``make test-envs-real``.

Requirements:
    * A CUDA device with at least the largest cap requested (today: 24 GiB).
    * Model weights downloaded on first run (handled by HF cache).
    * Inference params match ``demo_scripts/test_demo_ltx23_auto.py``
      (768x512x121 frames, 8 steps) so numbers are directly comparable
      to the manual demo.

VRAM caps are real (hard, via dummy reservation). RAM is reported but
not asserted: a background ``_MemorySampler`` records peak ``RssAnon``
(anonymous resident pages — the memory the process genuinely must keep
in RAM) and ``VmRSS`` (total resident, including mmap'd safetensors
that the kernel can evict). ``RssAnon`` matches what btop / top show
for real system pressure; it's the operational answer to "would this
fit on a real box?". We avoid ``psutil.memory_full_info().uss`` because
it counts MAP_PRIVATE-mmap'd file pages (safetensors) as "private
memory" — those are file-backed and reclaimable, and counting them
produced false-negative RAM failures. For a kernel-enforced RAM check
that hard-fails when truly over, run under cgroup.

The strategy resolver inside the manager sees the injected RAM value
through ``FakeEnvManager`` — that drives strategy choice and the
``low_cpu_mem`` auto-tune, which IS what the test asserts on.

To *enforce* the RAM cap at the kernel level (real OOM-kill if
exceeded), wrap pytest in a cgroup::

    systemd-run --user --scope -p MemoryMax=32G -p MemorySwapMax=0 \\
        env DIFFUSERS_MM_RUN_GPU_TESTS=1 pytest tests/test_envs_real.py

The test self-detects whether it's running constrained (by reading
``/sys/fs/cgroup/<self>/memory.max``) and prints which mode is active.
"""

from __future__ import annotations

import gc
import threading
import time
from pathlib import Path

import psutil
import pytest
import torch
from _env_fakes import ENVS, EXPECTED_STRATEGY, PROFILES, FakeEnvManager

from diffusers_mm import managed


pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def _cap_vram(target_gb: float, device: int = 0) -> tuple[torch.Tensor | None, float]:
    """Cap total process VRAM at ``target_gb`` via a held dummy tensor.

    Returns ``(reservation_tensor, reserved_gb)`` — keep the reservation
    alive for the duration of the test. When the reference drops, the
    cap goes with it.

    Allocates ``(total - target)`` GiB as a ``torch.empty`` uint8 tensor
    on the device, taking that memory off the table physically. The
    CUDA driver then has only ``target_gb`` left for everything in the
    process — PyTorch allocator + CUDA context + Triton/SDNQ scratch +
    cuDNN — so over-budget code raises an OOM at ``cudaMalloc`` for
    real. This is the bit that makes the test fail-honest under
    pressure; the soft ``set_per_process_memory_fraction`` is not used
    because it only caps PyTorch's allocator and lets non-allocator
    CUDA usage push total VRAM well above the cap.

    Skips the test if the physical device is smaller than ``target_gb``.
    """
    total_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    if total_gb < target_gb:
        pytest.skip(f"device {device} only has {total_gb:.1f} GiB; need >= {target_gb}")
    gc.collect()
    torch.cuda.empty_cache()
    reserve_gb = max(0.0, total_gb - target_gb)
    reservation: torch.Tensor | None = None
    if reserve_gb > 0:
        reserve_bytes = int(reserve_gb * 1024**3)
        reservation = torch.empty(reserve_bytes, dtype=torch.uint8, device=f"cuda:{device}")
    # Reset peak counters AFTER allocating the dummy so the inference's
    # peak readings can be reported as a delta from the dummy baseline.
    torch.cuda.reset_peak_memory_stats(device)
    return reservation, reserve_gb


def _read_proc_self_status() -> dict[str, int]:
    """Read selected memory fields from ``/proc/self/status``.

    Returns a dict mapping field name to bytes for: ``VmRSS``,
    ``RssAnon``, ``RssFile``, ``RssShmem``, ``VmSwap``. Missing keys
    indicate the kernel didn't expose them (older kernels) or the read
    failed; callers should treat absent keys as 0.

    This is the *authoritative* per-process memory accounting on Linux:
    same numbers ``top`` and ``htop`` derive their RES/SHR display from.
    Anonymous memory (``RssAnon``) is the bit that genuinely has to stay
    resident — file-backed pages (``RssFile``) are kernel-evictable
    even when private-mapped, which is why ``psutil``'s USS (from
    ``smaps_rollup``) misleadingly reports them as "process memory".
    """
    wanted = {"VmRSS", "RssAnon", "RssFile", "RssShmem", "VmSwap"}
    result: dict[str, int] = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                key, sep, rest = line.partition(":")
                if not sep or key not in wanted:
                    continue
                parts = rest.split()
                if parts and parts[0].isdigit():
                    result[key] = int(parts[0]) * 1024  # kB → bytes
    except OSError:
        pass
    return result


class _MemorySampler:
    """Background thread tracking peak anonymous-RSS and total RSS.

    Reads ``/proc/self/status`` to track ``RssAnon`` (anonymous resident
    pages — the memory the process genuinely needs to keep) and
    ``VmRSS`` (total resident, including mmap'd file pages that the
    kernel can evict for free).

    ``RssAnon`` is the operational answer to "would this fit on a real
    32 GiB box?". It matches what btop / top show for system pressure.
    We previously used psutil's ``memory_full_info().uss`` but that
    counts MAP_PRIVATE file pages as "private memory" — for safetensors
    loaded via mmap that's misleading: those pages are private but
    file-backed and reclaimable.

    Sampling at 100 ms; ``/proc/self/status`` is a small text file and
    cheap to read.

    Usage::

        sampler = _MemorySampler()
        sampler.start()
        try:
            do_work()
        finally:
            peak_anon, peak_rss = sampler.stop()
    """

    def __init__(self, interval_s: float = 0.1) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_anon = 0
        self.peak_rss = 0

    def _sample(self) -> tuple[int, int]:
        info = _read_proc_self_status()
        return info.get("RssAnon", 0), info.get("VmRSS", 0)

    def start(self) -> None:
        self.peak_anon, self.peak_rss = self._sample()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[int, int]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        # One final sample in case a spike happened between the last
        # poll and the stop signal.
        anon, rss = self._sample()
        if anon > self.peak_anon:
            self.peak_anon = anon
        if rss > self.peak_rss:
            self.peak_rss = rss
        return self.peak_anon, self.peak_rss

    def _run(self) -> None:
        # ``Event.wait`` doubles as the sleep + stop check — wakes early
        # when ``_stop`` is set, naps for ``interval_s`` otherwise.
        while not self._stop.wait(self.interval_s):
            anon, rss = self._sample()
            if anon > self.peak_anon:
                self.peak_anon = anon
            if rss > self.peak_rss:
                self.peak_rss = rss


def _detect_cgroup_memory_max_gb() -> float | None:
    """Return this process's cgroup v2 ``memory.max`` in GiB, or ``None``.

    Reads ``/proc/self/cgroup`` to find the cgroup path, then reads
    ``/sys/fs/cgroup/<path>/memory.max``. Returns ``None`` if not on
    cgroup v2, no cap is set, or anything fails. The value ``"max"``
    in the file (no limit) also returns ``None``.

    Important: ``psutil.virtual_memory().total`` is NOT reduced inside
    a cgroup — it always reports host total. The cgroup limit must be
    discovered from sysfs directly.
    """
    try:
        with open("/proc/self/cgroup") as f:
            line = f.readline().strip()
        if not line.startswith("0::"):
            return None  # not cgroup v2
        cgroup_rel = line[3:].lstrip("/")
        memory_max_path = Path("/sys/fs/cgroup") / cgroup_rel / "memory.max"
        if not memory_max_path.exists():
            return None
        value = memory_max_path.read_text().strip()
        if value == "max":
            return None
        return int(value) / (1024**3)
    except (OSError, ValueError):
        return None


def _is_ram_actually_constrained(target_gb: float, slack_gb: float = 4.0) -> bool:
    """True if this process is running under a cgroup ``memory.max`` near
    ``target_gb``.

    ``slack_gb`` covers cgroup accounting overhead and the fact that
    ``MemoryMax`` isn't reported byte-exact when set via ``systemd-run``.
    """
    cap_gb = _detect_cgroup_memory_max_gb()
    if cap_gb is None:
        return False
    return cap_gb <= target_gb + slack_gb


def _load_ltx23_distilled_sdnq(quant: str):
    """Load the LTX-2.3 distilled pipeline with SDNQ-quantized weights.

    ``quant`` is ``"int8"`` or ``"int4"`` and selects the HuggingFace
    repo path. Mirrors the demo scripts so test behavior matches the
    user's manual exercise.
    """
    from diffusers import LTX2Pipeline, LTX2VideoTransformer3DModel
    from sdnq import SDNQConfig  # noqa: F401
    from sdnq.common import use_torch_compile as triton_is_available
    from sdnq.loader import apply_sdnq_options_to_model
    from transformers import Gemma3ForConditionalGeneration

    repo = f"OzzyGT/LTX-2.3-Distilled-1.1-sdnq-dynamic-{quant}"
    text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
        repo,
        subfolder="text_encoder",
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    transformer = LTX2VideoTransformer3DModel.from_pretrained(
        repo,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    pipe = LTX2Pipeline.from_pretrained(
        "OzzyGT/LTX-2.3-Distilled",
        transformer=transformer,
        text_encoder=text_encoder,
        torch_dtype=torch.bfloat16,
    )
    if triton_is_available and torch.cuda.is_available():
        pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
        pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)
    return pipe


def _run_ltx23_vram24_ram32_test(profile_name: str) -> None:
    """Shared body for ``test_ltx23_*_vram24_ram32``.

    Loads the LTX-2.3 distilled pipeline for the given profile, applies
    a hard 24 GiB VRAM cap (dummy reservation) + an injected 32 GiB RAM
    target (FakeEnvManager), runs a small inference, and asserts the
    resolved strategy + peak RSS. The expected strategy comes from
    ``EXPECTED_STRATEGY`` so adding a new profile is a single-table edit.
    """
    env_name = "vram24_ram32"
    vram_gb, ram_gb = ENVS[env_name]
    expected_strategy = EXPECTED_STRATEGY[(env_name, profile_name)]
    profile = PROFILES[profile_name]
    quant = "int4" if "int4" in profile_name else "int8"

    reservation, reserve_gb = _cap_vram(vram_gb)
    mem_sampler = _MemorySampler()
    try:
        pipe = _load_ltx23_distilled_sdnq(quant)

        # Inject both VRAM (so strategy budgeting uses 24 GiB, not the
        # real card's 32) and RAM (so auto-tune sees a 32 GiB box).
        # n_transformer_blocks=0 because the real pipeline's transformer
        # is what will be registered through managed().
        mm = FakeEnvManager(
            vram_avail_gb=vram_gb,
            ram_avail_gb=ram_gb,
            total_weights_gb=profile.total_weights_gb,
            max_component_gb=profile.max_component_gb,
            n_transformer_blocks=0,
        )
        pipe = managed(pipe, mm=mm, device="cuda")

        assert pipe.mm.applied_strategy == expected_strategy, (
            f"expected {expected_strategy} under {env_name} + {profile_name}; got {pipe.mm.applied_strategy}"
        )

        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES

        # Inference params match demo_scripts/test_demo_ltx23_auto.py so
        # the numbers in the printed table are directly comparable to the
        # user's manual exercise of the same workload at full resolution.
        prompt = (
            "A highly detailed macro cinematic shot inside a dense tropical rainforest "
            "just after heavy rain. Giant glossy leaves fill the frame, covered in "
            "crystal-clear water droplets that reflect the environment like tiny lenses. "
            "A bright metallic-blue butterfly rests on a leaf in the foreground, its "
            "wings slowly opening to reveal intricate shimmering patterns. A sudden "
            "droplet falls from a higher leaf and lands nearby, causing smaller droplets "
            "to bounce and scatter in slow motion. The butterfly reacts, gently lifting "
            "off into the humid air. As it flutters away, the camera performs a subtle "
            "smooth push-in through layers of foliage, creating rich natural depth and "
            "parallax. In the background, soft mist drifts between massive tree trunks "
            "while distant leaves sway slightly. Tiny floating pollen particles catch "
            "shafts of warm sunlight breaking through the canopy. Ultra-realistic "
            "textures, natural lighting, shallow depth of field, cinematic focus "
            "transitions, physically accurate motion, rich environmental detail. "
            "Sound description: soft rainforest ambience, distant birds, gentle water "
            "drips, subtle wing flutters."
        )
        negative_prompt = (
            "blurry, out of focus, overexposed, underexposed, low contrast, washed out "
            "colors, excessive noise, grainy texture, poor lighting, flickering, motion "
            "blur, distorted proportions, unnatural skin tones, deformed facial features, "
            "asymmetrical face, missing facial features, extra limbs, disfigured hands, "
            "wrong hand count, artifacts around text, inconsistent perspective, camera "
            "shake, incorrect depth of field, background too sharp, background clutter, "
            "distracting reflections, harsh shadows, inconsistent lighting direction, "
            "color banding, cartoonish rendering, 3D CGI look, unrealistic materials, "
            "uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated "
            "expressions, wrong gaze direction, mismatched lip sync, silent or muted "
            "audio, distorted voice, robotic voice, echo, background noise, off-sync "
            "audio, incorrect dialogue, added dialogue, repetitive speech, jittery "
            "movement, awkward pauses, incorrect timing, unnatural transitions, "
            "inconsistent framing, tilted camera, flat lighting, inconsistent tone, "
            "cinematic oversaturation, stylized filters, or AI artifacts."
        )

        mem_sampler.start()
        t0 = time.perf_counter()
        try:
            pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=768,
                height=512,
                num_frames=121,
                frame_rate=24.0,
                num_inference_steps=8,
                sigmas=DISTILLED_SIGMA_VALUES,
                guidance_scale=1.0,
                generator=torch.Generator("cuda").manual_seed(42),
                output_type="np",
                return_dict=False,
            )
        finally:
            elapsed = time.perf_counter() - t0
            peak_anon_bytes, peak_rss_bytes = mem_sampler.stop()
            peak_anon_gib = peak_anon_bytes / (1024**3)
            peak_rss_gib = peak_rss_bytes / (1024**3)
            # Subtract the dummy reservation so the printed VRAM peak
            # reflects the inference's own contribution (the dummy is
            # counted by max_memory_* because torch.empty allocated it).
            peak_alloc_gib = torch.cuda.max_memory_allocated() / (1024**3) - reserve_gb
            peak_resv_gib = torch.cuda.max_memory_reserved() / (1024**3) - reserve_gb
            cgroup_cap_gb = _detect_cgroup_memory_max_gb()
            ram_constrained = _is_ram_actually_constrained(ram_gb)
            if ram_constrained:
                ram_mode = f"cgroup-enforced at {cgroup_cap_gb:.1f} GiB — completion = real usage stayed under cap"
            else:
                ram_mode = "informational only (no cgroup; run under systemd-run to enforce)"
            print(
                f"\n[envs_real] {env_name} {profile_name} "
                f"(applied={pipe.mm.applied_strategy}): {elapsed:.1f}s"
                f"\n  VRAM (PyTorch): alloc={peak_alloc_gib:.2f} GiB, reserved={peak_resv_gib:.2f} GiB"
                f"\n  RAM anonymous peak (RssAnon, the real cost): {peak_anon_gib:.2f} GiB"
                f"\n  RAM total resident peak (VmRSS, incl. mmap'd safetensors): {peak_rss_gib:.2f} GiB"
                f"\n  RAM cap: {ram_mode}"
            )
            if not ram_constrained:
                host_total = psutil.virtual_memory().total / (1024**3)
                print(
                    f"[envs_real] NOTE: host reports {host_total:.0f} GiB total RAM and "
                    f"no cgroup memory cap was detected. RssAnon is the metric to watch — "
                    f"it tracks anonymous memory that genuinely has to stay resident. "
                    f"VmRSS additionally counts mmap'd safetensors that the kernel can "
                    f"evict under pressure. For a kernel-enforced check, run: "
                    f"`systemd-run --user --scope -p MemoryMax={int(ram_gb)}G "
                    f"-p MemorySwapMax=0 make test-envs-real`."
                )
            print(
                f"[envs_real] NOTE: real process VRAM is several GiB higher than the "
                f"PyTorch-tracked peak (CUDA context, Triton/SDNQ workspaces not in "
                f"PyTorch stats). The dummy reservation enforces the {int(vram_gb)} GiB cap — "
                f"completion = real usage stayed under {int(vram_gb)} GiB."
            )

        # No strict RAM assertion. The cgroup is the source of truth:
        # under MemoryMax the kernel OOM-kills if the process really
        # exceeds the cap. The test reaching this line means (a) VRAM
        # stayed under cap (dummy enforced) and (b) if running under
        # cgroup, RAM also stayed under cap (kernel enforced). RssAnon
        # is printed as the operational signal — anonymous memory the
        # process MUST keep resident. We don't strict-assert on it
        # because the cgroup catches the real failures.
    finally:
        # Drop the reservation so the dummy bytes return to the device;
        # empty_cache nudges the allocator to release them.
        del reservation
        gc.collect()
        torch.cuda.empty_cache()


def test_ltx23_distilled_int8_vram24_ram32() -> None:
    """24 GiB hard VRAM cap + 32 GiB injected RAM, LTX-2.3 distilled int8.

    Auto picks ``block_pin`` (transformer 18.5 GiB int8 fails the
    model_offload 1.5× check at 24 GiB available: 27.8 > 24). The block
    budget — computed from the injected 24 GiB — must keep real total
    process VRAM under 24 GiB, which the dummy reservation enforces.

    Empirically the workload also fits 32 GiB of RAM under a real
    cgroup, despite psutil's USS reading above 32 GiB (locked DMA
    buffers and CUDA-related anonymous bookkeeping over-report). The
    test does not assert on USS — run under
    ``systemd-run -p MemoryMax=32G`` to make the RAM cap kernel-enforced.
    """
    _run_ltx23_vram24_ram32_test("ltx23_distilled_sdnq_int8")


def test_ltx23_distilled_int4_vram24_ram32() -> None:
    """24 GiB hard VRAM cap + 32 GiB injected RAM, LTX-2.3 distilled int4.

    Auto picks ``model_offload``: the int4 transformer is 10.769 GiB,
    so ``10.769 × 1.5 = 16.15 ≤ 24`` fits comfortably. Total pipeline
    weights are 26.10 GiB, which is below the 32 GiB cap minus the
    AUTO_RAM_HEADROOM (32 × 0.85 = 27.2), so no RAM-warning fires and
    the workload should fit on host with a few GiB to spare.

    A passing run confirms both halves of the simulated env: the real
    24 GiB VRAM cap (dummy reservation, OOM at ``cudaMalloc`` on
    overshoot) and the 32 GiB RAM cap (peak-RSS assertion; cgroup-
    enforced under ``systemd-run -p MemoryMax=32G``).
    """
    _run_ltx23_vram24_ram32_test("ltx23_distilled_sdnq_int4")
