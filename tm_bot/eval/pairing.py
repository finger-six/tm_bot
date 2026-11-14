"""Seat-swapped pairing schedule construction over seed keys and supported maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from tm_bot.agents.protocol import Agent
from tm_bot.agents.runner import GameSpec
from tm_bot.env.types import SUPPORTED_MAPS, Seat
from tm_bot.eval.seeds import SeedKey, SeedNamespace, derive_agent_seed, generate_seed_keys

AgentFactory = Agent | Callable[[], Agent]

DEFAULT_EVAL_SEEDS = 50


@dataclass(frozen=True, slots=True)
class PairedGameMeta:
    """Metadata aligning a `GameSpec` / `GameRunResult` back to its seat-swapped pair."""

    seed_key: str
    map_name: str
    a_seat: Seat
    seed_a: int
    seed_b: int


def _instantiate_agent(agent_or_factory: AgentFactory) -> Agent:
    if callable(agent_or_factory) and not isinstance(agent_or_factory, Agent):
        return agent_or_factory()
    return agent_or_factory  # type: ignore[return-value]


def build_paired_specs(
    *,
    agent_a: AgentFactory,
    agent_b: AgentFactory,
    seed_keys: Sequence[SeedKey | str] | None = None,
    namespace: SeedNamespace = "eval",
    num_seeds: int = DEFAULT_EVAL_SEEDS,
    seed_start: int = 0,
    maps: Sequence[str] = SUPPORTED_MAPS,
) -> tuple[list[GameSpec], list[PairedGameMeta]]:
    """Build seat-swapped `GameSpec` pairs for every `(seed_key, map_name)`.

    Each `(seed_key, map_name)` is scheduled twice:
      1. Seat 0 = Agent A, Seat 1 = Agent B (`seat_seeds=(seed_a, seed_b)`)
      2. Seat 0 = Agent B, Seat 1 = Agent A (`seat_seeds=(seed_b, seed_a)`)
    """
    if not maps:
        raise ValueError("maps must be non-empty")

    resolved_keys: tuple[SeedKey, ...]
    if seed_keys is None:
        if num_seeds < 1:
            raise ValueError(f"num_seeds must be >= 1, got {num_seeds}")
        resolved_keys = generate_seed_keys(namespace, num_seeds, start=seed_start)
    else:
        resolved_keys = tuple(
            sk if isinstance(sk, SeedKey) else SeedKey.parse(sk) for sk in seed_keys
        )
        if not resolved_keys:
            raise ValueError("seed_keys must be non-empty")

    specs: list[GameSpec] = []
    metas: list[PairedGameMeta] = []

    for sk in resolved_keys:
        for map_name in maps:
            seed_a = derive_agent_seed(sk, map_name=map_name, role="a")
            seed_b = derive_agent_seed(sk, map_name=map_name, role="b")

            # Game 1: A in seat 0, B in seat 1
            specs.append(
                GameSpec(
                    seed_key=sk.key,
                    map_name=map_name,
                    agents=(_instantiate_agent(agent_a), _instantiate_agent(agent_b)),
                    seat_seeds=(seed_a, seed_b),
                )
            )
            metas.append(
                PairedGameMeta(
                    seed_key=sk.key,
                    map_name=map_name,
                    a_seat=0,
                    seed_a=seed_a,
                    seed_b=seed_b,
                )
            )

            # Game 2: B in seat 0, A in seat 1 (swapped seats, same per-agent seeds)
            specs.append(
                GameSpec(
                    seed_key=sk.key,
                    map_name=map_name,
                    agents=(_instantiate_agent(agent_b), _instantiate_agent(agent_a)),
                    seat_seeds=(seed_b, seed_a),
                )
            )
            metas.append(
                PairedGameMeta(
                    seed_key=sk.key,
                    map_name=map_name,
                    a_seat=1,
                    seed_a=seed_a,
                    seed_b=seed_b,
                )
            )

    return specs, metas
