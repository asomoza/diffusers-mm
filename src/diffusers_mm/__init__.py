"""diffusers-mm — Smart model management for Hugging Face Diffusers pipelines."""

import importlib.util as _importlib_util


def _require_companions() -> None:
    """Hard-fail at import if diffusers / accelerate aren't installed.

    diffusers-mm isn't pinned to specific versions of either (we ride the
    user's existing diffusers stack rather than risk forcing downgrades),
    but both are genuinely required: ``group_offload`` calls into
    ``diffusers.hooks.group_offloading`` and ``model_offload`` calls
    accelerate's ``cpu_offload_with_hook``. Without them the library can't
    do anything its name promises, so failing at import gives a clearer
    error than a confusing ImportError mid-strategy-apply.
    """
    missing = [name for name in ("diffusers", "accelerate") if _importlib_util.find_spec(name) is None]
    if missing:
        raise ImportError(
            "diffusers-mm requires "
            + " and ".join(missing)
            + " to be installed. Install with: pip install "
            + " ".join(missing)
        )


_require_companions()
del _require_companions


from diffusers_mm.hooks import remove_offload_hooks  # noqa: E402
from diffusers_mm.managed import managed  # noqa: E402
from diffusers_mm.manager import ModelManager  # noqa: E402


try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("diffusers-mm")
    del _pkg_version
except Exception:  # noqa: BLE001
    __version__ = "0.0.0+unknown"


__all__ = [
    "ModelManager",
    "__version__",
    "managed",
    "remove_offload_hooks",
]
