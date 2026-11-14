"""Exception hierarchy for the Python-Node adapter bridge."""

from __future__ import annotations

from typing import Any


class BridgeError(RuntimeError):
    """Base error raised by the tm_bot bridge."""


class AdapterBugError(BridgeError):
    """Error raised when the adapter or engine reports a game bug/rejection."""

    def __init__(
        self,
        message: str,
        *,
        seed_key: str = "",
        map_name: str | None = None,
        seat: int | None = None,
        history: list[dict[str, Any]] | None = None,
        error_type: str = "AdapterBugError",
    ) -> None:
        self.message = message
        self.seed_key = seed_key
        self.map = map_name
        self.seat = seat
        self.history: list[dict[str, Any]] = list(history) if history is not None else []
        self.error_type = error_type
        detail = f"{error_type} (seed_key={seed_key!r}, history_len={len(self.history)}): {message}"
        super().__init__(detail)


class WorkerCrashError(BridgeError):
    """Error raised when a Node worker subprocess terminates unexpectedly or fails I/O."""

    def __init__(
        self,
        message: str,
        *,
        returncode: int | None = None,
        stderr_tail: str = "",
        command: tuple[str, ...] = (),
        game_id: str | None = None,
        seed_key: str | None = None,
    ) -> None:
        self.message = message
        self.returncode = returncode
        self.stderr_tail = stderr_tail
        self.command = command
        self.game_id = game_id
        self.seed_key = seed_key
        parts = [message]
        if returncode is not None:
            parts.append(f"returncode={returncode}")
        if game_id is not None:
            parts.append(f"game_id={game_id!r}")
        if seed_key is not None:
            parts.append(f"seed_key={seed_key!r}")
        if stderr_tail.strip():
            parts.append(f"stderr_tail:\n{stderr_tail.rstrip()}")
        super().__init__(" | ".join(parts))


class BridgeProtocolError(BridgeError):
    """Error raised when a worker emits malformed or mismatched JSON protocol data."""

    def __init__(
        self,
        message: str,
        *,
        raw_response: str | None = None,
        stderr_tail: str = "",
    ) -> None:
        self.message = message
        self.raw_response = raw_response
        self.stderr_tail = stderr_tail
        detail = message
        if raw_response is not None:
            detail += f" | raw={raw_response[:300]!r}"
        if stderr_tail.strip():
            detail += f" | stderr_tail:\n{stderr_tail.rstrip()}"
        super().__init__(detail)
