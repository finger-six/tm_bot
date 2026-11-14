"""High-level evaluation runner reusing `tm_bot.agents.play_games`."""

from __future__ import annotations

import importlib
import time
from typing import Any, Callable, Sequence

from tm_bot.agents import B0Agent, B1Agent, load_card_data
from tm_bot.agents.protocol import Agent
from tm_bot.agents.runner import DEFAULT_MAX_DECISIONS, play_games
from tm_bot.env.types import SUPPORTED_MAPS
from tm_bot.eval.pairing import DEFAULT_EVAL_SEEDS, AgentFactory, build_paired_specs
from tm_bot.eval.report import summarize_evaluation
from tm_bot.eval.seeds import SeedKey, SeedNamespace
from tm_bot.eval.stats import DEFAULT_BOOTSTRAP_RESAMPLES, DEFAULT_BOOTSTRAP_SEED


def resolve_agent_factory(spec: str) -> Callable[[], Agent]:
    """Resolve a built-in agent name (`b0`, `b1`) or `module:callable` import spec."""
    norm = spec.strip().lower()
    if norm == "b0":
        return lambda: B0Agent(seed=0)
    if norm == "b1":
        card_data = load_card_data()
        return lambda: B1Agent(seed=0, card_data=card_data)
    if ":" in spec:
        mod_name, attr_name = spec.split(":", 1)
        mod = importlib.import_module(mod_name)
        factory = getattr(mod, attr_name)
        if not callable(factory):
            raise TypeError(f"Resolved object {spec!r} is not callable")
        return lambda: factory()
    raise ValueError(
        f"Unknown agent spec {spec!r}; expected 'b0', 'b1', or 'module:factory'"
    )


def evaluate_agents(
    *,
    agent_a: AgentFactory,
    agent_b: AgentFactory,
    agent_a_name: str = "agent_a",
    agent_b_name: str = "agent_b",
    agent_a_version: str | None = None,
    agent_b_version: str | None = None,
    a_training_seeds: int = 0,
    b_training_seeds: int = 0,
    a_compute: dict[str, Any] | None = None,
    b_compute: dict[str, Any] | None = None,
    seed_keys: Sequence[SeedKey | str] | None = None,
    namespace: SeedNamespace = "eval",
    num_seeds: int = DEFAULT_EVAL_SEEDS,
    seed_start: int = 0,
    maps: Sequence[str] = SUPPORTED_MAPS,
    num_workers: int = 8,
    batch_size: int = 32,
    max_decisions: int = DEFAULT_MAX_DECISIONS,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Run seat-swapped evaluation of `agent_a` vs `agent_b` via `play_games` and summarize."""
    specs, metas = build_paired_specs(
        agent_a=agent_a,
        agent_b=agent_b,
        seed_keys=seed_keys,
        namespace=namespace,
        num_seeds=num_seeds,
        seed_start=seed_start,
        maps=maps,
    )

    t0 = time.perf_counter()
    outcomes = play_games(
        specs,
        num_workers=num_workers,
        batch_size=batch_size,
        max_decisions=max_decisions,
    )
    wall_time_s = time.perf_counter() - t0

    return summarize_evaluation(
        outcomes=outcomes,
        metas=metas,
        namespace=namespace,
        agent_a_name=agent_a_name,
        agent_b_name=agent_b_name,
        agent_a_version=agent_a_version,
        agent_b_version=agent_b_version,
        a_training_seeds=a_training_seeds,
        b_training_seeds=b_training_seeds,
        a_compute=a_compute,
        b_compute=b_compute,
        workers=num_workers,
        batch_size=batch_size,
        wall_time_s=wall_time_s,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
