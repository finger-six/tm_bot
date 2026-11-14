"""Common agent protocol and game outcome models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from tm_bot.env.types import ActorObservation, Choice, GameResult


@runtime_checkable
class Agent(Protocol):
    """Interface implemented by all Terraforming Mars agents."""

    def reset(self, seed: int | None = None) -> None:
        """Reset per-game state and optionally re-seed the agent's RNG."""

    def act(self, observation: ActorObservation, choice: Choice) -> int:
        """Return a legal candidate index in [0, len(choice.candidates)) using only observation and choice."""


@dataclass(frozen=True, slots=True)
class GameRunResult:
    """Outcome and diagnostics for a single played game."""

    seed_key: str
    map: str
    result: GameResult | None
    generations: int | None
    decisions: int
    agent_versions: tuple[str | None, str | None]
    ok: bool
    error_type: str | None = None
    error: str | None = None
    decision_latencies_ms: tuple[tuple[float, ...], tuple[float, ...]] = field(
        default=((), ()),
        compare=False,
        repr=False,
    )
