"""Windows-only helpers for memory accounting via the Win32 PSAPI.

The Windows commit charge inflates the apparent "in-use" RAM figure
because WDDM eagerly reserves commit for every VRAM allocation (a
worst-case page-out budget that never gets used in practice). Reading
``psutil.virtual_memory().available`` on Windows therefore reports
*less* usable RAM than is actually available for new allocations,
which makes strategy resolution overly conservative — e.g. it can
pick ``low_cpu_mem_usage=True`` on a system that has plenty of room
to hold a pinned host copy, just because psutil under-reports.

This module wraps ``GetPerformanceInfo`` (PSAPI) so the manager's
free-RAM calculation can subtract VRAM-backed commit from the total
and arrive at a more honest number. Borrowed from ComfyUI's
``comfy/windows.py``; the formula is theirs, the wrapping has been
restructured for our test harness (the OS query is broken out into
a single function the tests can monkeypatch).

Import gated by ``sys.platform == "win32"`` at every call site —
this file is safe to import on non-Windows but the ctypes WinDLL
loads at module import time would fail there, so the imports below
happen lazily inside the function.
"""

from __future__ import annotations

import logging


logger = logging.getLogger(__name__)


def query_performance_info_bytes() -> tuple[int, int] | None:
    """Return ``(commit_total_bytes, physical_total_bytes)`` from PSAPI.

    Returns ``None`` if anything fails — including being called on a
    non-Windows platform (defensive; every call site already gates on
    ``sys.platform``, but a stray import shouldn't crash). The caller
    is responsible for falling back to ``psutil``.

    ``commit_total_bytes`` is the system-wide commit charge — the
    total memory that processes have *promised* the system they might
    use. WDDM inflates this by the size of every VRAM allocation, which
    is what we'll subtract back out at the call site.

    ``physical_total_bytes`` is the total physical RAM as the kernel
    sees it (``PerformanceInformation.PhysicalTotal × PageSize``).
    Equivalent to ``psutil.virtual_memory().total`` but read from the
    same query so the numbers are coherent.
    """
    import sys

    if sys.platform != "win32":
        return None

    try:
        import ctypes
        from ctypes import wintypes
    except Exception as e:
        logger.warning("windows RAM accounting: ctypes import failed (%s)", e)
        return None

    class PERFORMANCE_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("CommitTotal", ctypes.c_size_t),
            ("CommitLimit", ctypes.c_size_t),
            ("CommitPeak", ctypes.c_size_t),
            ("PhysicalTotal", ctypes.c_size_t),
            ("PhysicalAvailable", ctypes.c_size_t),
            ("SystemCache", ctypes.c_size_t),
            ("KernelTotal", ctypes.c_size_t),
            ("KernelPaged", ctypes.c_size_t),
            ("KernelNonpaged", ctypes.c_size_t),
            ("PageSize", ctypes.c_size_t),
            ("HandleCount", wintypes.DWORD),
            ("ProcessCount", wintypes.DWORD),
            ("ThreadCount", wintypes.DWORD),
        ]

    try:
        psapi = ctypes.WinDLL("psapi")
    except Exception as e:
        logger.warning("windows RAM accounting: psapi.dll load failed (%s)", e)
        return None

    pi = PERFORMANCE_INFORMATION()
    pi.cb = ctypes.sizeof(pi)
    if not psapi.GetPerformanceInfo(ctypes.byref(pi), pi.cb):
        logger.warning("windows RAM accounting: GetPerformanceInfo returned 0")
        return None

    commit_total_bytes = pi.CommitTotal * pi.PageSize
    physical_total_bytes = pi.PhysicalTotal * pi.PageSize
    return commit_total_bytes, physical_total_bytes
