"""Resource guards for the learner process (RAM and VRAM ceilings)."""

from __future__ import annotations

import ctypes
import sys

from tm_bot.runs.config import ResourceLimits
from tm_bot.runs.errors import RunResourceError


def current_rss_bytes() -> int | None:
    """Return the current process resident set size in bytes, or None if unavailable."""
    if sys.platform == "win32":
        try:
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            )
            if ok:
                return int(counters.WorkingSetSize)
        except Exception:
            return None
        return None
    try:
        with open("/proc/self/statm", "r", encoding="ascii") as fh:
            pages = int(fh.read().split()[1])
        return pages * 4096
    except Exception:
        return None


def check_learner_memory(limits: ResourceLimits) -> dict[str, float | None]:
    """Raise `RunResourceError` if learner RAM or VRAM exceeds its configured ceiling."""
    usage: dict[str, float | None] = {"rss_gb": None, "vram_gb": None}
    rss = current_rss_bytes()
    if rss is not None:
        usage["rss_gb"] = rss / 1024**3
        if usage["rss_gb"] > limits.learner_ram_gb:
            raise RunResourceError(
                f"learner RAM {usage['rss_gb']:.2f} GB exceeds "
                f"{limits.learner_ram_gb:.2f} GB"
            )
    try:
        import torch

        if torch.cuda.is_available():
            usage["vram_gb"] = torch.cuda.memory_allocated() / 1024**3
            if usage["vram_gb"] > limits.vram_gb:
                raise RunResourceError(
                    f"learner VRAM {usage['vram_gb']:.2f} GB exceeds "
                    f"{limits.vram_gb:.2f} GB"
                )
    except RunResourceError:
        raise
    except Exception:
        pass
    return usage
