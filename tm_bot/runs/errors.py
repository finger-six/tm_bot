"""Run-level errors surfaced by orchestration and resource guards."""

from __future__ import annotations


class RunError(RuntimeError):
    """Base class for self-play run failures."""


class RunResourceError(RunError):
    """A configured resource ceiling (disk, RAM, VRAM) was exceeded."""


class RunResumeError(RunError):
    """A run could not be resumed from its saved state."""
