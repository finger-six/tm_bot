"""Common agent protocol, optional search extensions, and game outcome models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

from tm_bot.env.types import ActorObservation, Choice, GameResult, Op, Seat


@runtime_checkable
class Agent(Protocol):
    """Interface implemented by all Terraforming Mars agents."""

    def reset(self, seed: int | None = None) -> None:
        """Reset per-game state and optionally re-seed the agent's RNG."""

    def act(self, observation: ActorObservation, choice: Choice) -> int:
        """Return a legal candidate index in [0, len(choice.candidates)) using only observation and choice."""


@runtime_checkable
class WorldSampler(Protocol):
    """Restricted simulation handle passed to search agents for the acting seat only.

    A `WorldSampler` lets an agent sample `k` actor-consistent determinized clones
    of the decision point and interact with those clones (`clone`, `close`,
    `pending`, `observe`, `choice`, `choose`, `result`, `batch_ops`). It never
    exposes the true live game, the live game ID, or the live opponent's
    observation.
    """

    @property
    def seat(self) -> Seat:
        """Seat of the acting agent this sampler is bound to."""

    def sample(self, k: int, seed: int) -> list[str]:
        """Sample `k` determinized clone game IDs from the actor's own observation."""

    def clone(self, clone_id: str) -> str:
        """Clone an existing sampled clone owned by this sampler."""

    def close(self, clone_ids: Sequence[str] | str) -> None:
        """Close one or more clone game IDs owned by this sampler."""

    def close_all(self) -> None:
        """Close all clones still owned by this sampler."""

    def pending(self, clone_id: str) -> list[Seat]:
        """Return pending seats for an owned clone."""

    def observe(self, clone_id: str, seat: Seat) -> ActorObservation:
        """Return observation for `seat` inside an owned determinized clone."""

    def choice(self, clone_id: str, seat: Seat) -> Choice:
        """Return choice for `seat` inside an owned determinized clone."""

    def choose(self, clone_id: str, seat: Seat, candidate_idx: int) -> None:
        """Apply a candidate index on an owned determinized clone."""

    def result(self, clone_id: str) -> GameResult | None:
        """Return game result if finished inside an owned determinized clone."""

    def batch_ops(
        self, ops: Sequence[Op], *, raise_on_error: bool = True
    ) -> list[Any]:
        """Execute a batch of operations restricted strictly to owned clones."""


@runtime_checkable
class SearchAgent(Agent, Protocol):
    """Optional protocol extension detected by the runner for search-capable agents."""

    def act_with_sampler(
        self,
        observation: ActorObservation,
        choice: Choice,
        sampler: WorldSampler,
    ) -> int:
        """Choose a legal candidate index using `(observation, choice)` and `sampler`."""


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
    searched_decision_latencies_ms: tuple[tuple[float, ...], tuple[float, ...]] = field(
        default=((), ()),
        compare=False,
        repr=False,
    )
