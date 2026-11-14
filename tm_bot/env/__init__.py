"""Python environment bridge to the Node Terraforming Mars adapter."""

from __future__ import annotations

from tm_bot.env.client import GameSession, WorkerClient
from tm_bot.env.errors import AdapterBugError, BridgeError, BridgeProtocolError, WorkerCrashError
from tm_bot.env.pool import WorkerPool
from tm_bot.env.types import (
    SUPPORTED_MAPS,
    ActionRecord,
    ActorHistory,
    ActorObservation,
    BatchOp,
    BatchResult,
    Candidate,
    Choice,
    ChoiceKind,
    GameResult,
    Op,
    Seat,
    SupportedMap,
)

__all__ = [
    "SUPPORTED_MAPS",
    "ActionRecord",
    "ActorHistory",
    "ActorObservation",
    "AdapterBugError",
    "BatchOp",
    "BatchResult",
    "BridgeError",
    "BridgeProtocolError",
    "Candidate",
    "Choice",
    "ChoiceKind",
    "GameResult",
    "GameSession",
    "Op",
    "Seat",
    "SupportedMap",
    "WorkerClient",
    "WorkerCrashError",
    "WorkerPool",
]
