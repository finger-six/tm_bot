"""Per-decision recording wrapper used to collect self-play training targets."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from tm_bot.agents.protocol import Agent, SearchAgent
from tm_bot.env.types import ActorObservation, Choice, Seat


@dataclass
class DecisionRecord:
    """One decision seen by an agent during a game, with its candidate distribution."""

    observation: ActorObservation
    choice: Choice
    candidate_index: int
    policy: np.ndarray
    was_searched: bool
    seat: Seat
    generation: int


@dataclass
class GameRecorder:
    """Ordered action history and (optionally) policy records for one game."""

    seed_key: str
    map_name: str
    capture_seat: Seat | None = None
    decisions: list[DecisionRecord] = field(default_factory=list)
    actions: list[tuple[int, int]] = field(default_factory=list)

    def record(
        self,
        observation: ActorObservation,
        choice: Choice,
        candidate_index: int,
        policy: Sequence[float] | None,
        *,
        was_searched: bool,
    ) -> None:
        seat = int(observation["actor"])  # type: ignore[index]
        generation = int(observation["public"]["generation"])  # type: ignore[index]
        self.actions.append((seat, int(candidate_index)))
        if self.capture_seat is None or seat != self.capture_seat:
            return
        n = len(choice.candidates)
        if n <= 1 or policy is None:
            arr = np.zeros((n,), dtype=np.float64)
            arr[int(candidate_index)] = 1.0
        else:
            arr = np.asarray(policy[:n], dtype=np.float64)
            arr = np.where(np.isfinite(arr) & (arr > 0.0), arr, 0.0)
            total = float(arr.sum())
            arr = arr / total if total > 0.0 else np.full((n,), 1.0 / n)
        self.decisions.append(
            DecisionRecord(
                observation=observation,
                choice=choice,
                candidate_index=int(candidate_index),
                policy=arr,
                was_searched=was_searched,
                seat=seat,  # type: ignore[arg-type]
                generation=generation,
            )
        )

    @property
    def num_searched(self) -> int:
        return sum(1 for d in self.decisions if d.was_searched)


class RecordingAgent:
    """Plain `Agent` wrapper that records decisions into a shared `GameRecorder`.

    The runner deep-copies each `GameSpec` agent once per game; `__deepcopy__` shares
    the recorder (and therefore the action history) with the copy so the orchestrator
    can read it after `play_games` returns. The wrapped agent is copied normally.
    """

    def __init__(
        self,
        inner: Agent,
        recorder: GameRecorder,
        *,
        capture_seat: Seat | None = None,
        capture_policy: bool = True,
    ) -> None:
        self.inner = inner
        self.recorder = recorder
        self.capture_seat = capture_seat
        self.capture_policy = capture_policy
        if recorder.capture_seat is None and capture_seat is not None:
            recorder.capture_seat = capture_seat
        self.version = f"{getattr(inner, 'version', 'agent')}-rec"

    def __deepcopy__(self, memo: dict[int, Any]) -> RecordingAgent:
        clone = self.__class__(
            copy.deepcopy(self.inner, memo),
            self.recorder,
            capture_seat=self.capture_seat,
            capture_policy=self.capture_policy,
        )
        memo[id(self)] = clone
        return clone

    def reset(self, seed: int | None = None) -> None:
        self.inner.reset(seed)

    @property
    def last_decision_was_searched(self) -> bool:
        return bool(getattr(self.inner, "last_decision_was_searched", False))

    @property
    def last_decision_latency_ms(self) -> float:
        return float(getattr(self.inner, "last_decision_latency_ms", 0.0))

    @property
    def last_decision_policy(self):
        return getattr(self.inner, "last_decision_policy", None)

    def _record(
        self, observation: ActorObservation, choice: Choice, index: int
    ) -> None:
        policy = None
        if self.capture_policy and self.capture_seat is not None:
            if int(observation["actor"]) == self.capture_seat:
                policy = getattr(self.inner, "last_decision_policy", None)
        self.recorder.record(
            observation,
            choice,
            index,
            policy,
            was_searched=bool(getattr(self.inner, "last_decision_was_searched", False)),
        )

    def act(self, observation: ActorObservation, choice: Choice) -> int:
        index = self.inner.act(observation, choice)
        self._record(observation, choice, index)
        return index


class RecordingSearchAgent(RecordingAgent):
    """Recording wrapper for agents implementing the `SearchAgent` extension."""

    inner: Any

    def act_with_sampler(
        self,
        observation: ActorObservation,
        choice: Choice,
        sampler: Any,
    ) -> int:
        index = self.inner.act_with_sampler(observation, choice, sampler)
        self._record(observation, choice, index)
        return index

    def search_batch_key(self) -> Any:
        key_fn = getattr(self.inner, "search_batch_key", None)
        return key_fn() if callable(key_fn) else None

    @classmethod
    def act_batch_with_samplers(
        cls,
        agents: Sequence[RecordingSearchAgent],
        observations: Sequence[ActorObservation],
        choices: Sequence[Choice],
        samplers: Sequence[Any],
    ) -> list[int]:
        inners = [agent.inner for agent in agents]
        batch_fn = getattr(type(inners[0]), "act_batch_with_samplers")
        indices = batch_fn(inners, observations, choices, samplers)
        if len(indices) != len(agents):
            raise RuntimeError(
                f"act_batch_with_samplers returned {len(indices)} actions for {len(agents)} requests"
            )
        for agent, observation, choice, index in zip(agents, observations, choices, indices):
            agent._record(observation, choice, int(index))
        return [int(i) for i in indices]


def capture_agent(
    inner: Agent,
    recorder: GameRecorder,
    *,
    capture_seat: Seat | None,
) -> RecordingAgent:
    """Wrap `inner` so its decisions land in `recorder` (search-aware wrapper)."""
    cls = RecordingSearchAgent if isinstance(inner, SearchAgent) else RecordingAgent
    return cls(inner, recorder, capture_seat=capture_seat)
