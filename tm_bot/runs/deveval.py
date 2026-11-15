"""Dev-seed checkpoint evaluation against B1 and the previous best checkpoint.

Dev games use the same seat-swapped pairing as `tm_bot.eval.pairing`: every
`(dev-seed, map)` is played twice with the candidate in opposite seats, so the
paired bootstrap CI is valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
from typing import Any, Callable, Sequence

from tm_bot.agents import B1Agent, load_card_data
from tm_bot.agents.protocol import Agent
from tm_bot.agents.runner import play_games
from tm_bot.env.types import SUPPORTED_MAPS
from tm_bot.eval.pairing import build_paired_specs
from tm_bot.eval.report import summarize_evaluation
from tm_bot.eval.seeds import generate_seed_keys
from tm_bot.models.agent import PolicyValueAgent
from tm_bot.runs.monitoring import CheckpointMetrics
from tm_bot.runs.selfplay import assert_namespace
from tm_bot.search.agent import ISMCTSAgent
from tm_bot.search.config import SearchConfig


@lru_cache(maxsize=2)
def _load_policy_value_agent(checkpoint: str, device: str) -> PolicyValueAgent:
    """Bounded cache: at most the candidate and opponent of one pairing."""
    return PolicyValueAgent.from_checkpoint(checkpoint, device=device, temperature=0.0)


def cached_policy_value_agent(checkpoint: str | Path, device: str) -> PolicyValueAgent:
    return _load_policy_value_agent(str(Path(checkpoint).resolve()), device)


def clear_dev_agent_cache() -> None:
    _load_policy_value_agent.cache_clear()


def dev_agent_factory(
    checkpoint: str | Path,
    *,
    device: str,
    search_config: SearchConfig,
) -> Callable[[], Agent]:
    """Fresh search agent per game sharing the checkpoint's cached network."""

    def factory() -> Agent:
        pv = cached_policy_value_agent(checkpoint, device)
        return ISMCTSAgent(pv, config=search_config, version=f"dev-{Path(checkpoint).name}")

    return factory


def b1_agent_factory() -> Callable[[], Agent]:
    card_data = load_card_data()
    return lambda: B1Agent(seed=0, card_data=card_data)


def dev_seed_count(num_games: int, maps: Sequence[str] = SUPPORTED_MAPS) -> int:
    """Smallest seed count whose seat-swapped schedule covers `num_games` games."""
    if num_games < 2:
        raise ValueError("num_games must be >= 2 for seat-swapped dev evaluation")
    if not maps:
        raise ValueError("maps must be non-empty")
    return max(1, math.ceil(num_games / (2 * len(maps))))


def build_dev_games(
    num_games: int,
    *,
    maps: Sequence[str] = SUPPORTED_MAPS,
    seed_start: int = 0,
) -> list[tuple[str, str, int]]:
    """Build the seat-swapped `dev-*` schedule (>= `num_games`, multiple of 2*|maps|).

    Each `(seed, map)` appears twice, once with the candidate in each seat. The
    returned total is rounded up to a whole number of seat-swapped pairs.
    """
    seeds = generate_seed_keys("dev", dev_seed_count(num_games, maps), start=seed_start)
    games: list[tuple[str, str, int]] = []
    for seed_key in seeds:
        for map_name in maps:
            games.append((seed_key.key, map_name, 0))
            games.append((seed_key.key, map_name, 1))
    assert_namespace([g[0] for g in games], "dev")
    return games


@dataclass(frozen=True, slots=True)
class DevResult:
    """Summary of one dev pairing."""

    games: int
    completed: int
    failures: int
    win_rate: float | None
    win_rate_ci_95: tuple[float, float] | None
    mean_vp_margin: float | None
    report: dict[str, Any]


def evaluate_dev_pairing(
    *,
    candidate_factory: Callable[[], Agent],
    opponent_factory: Callable[[], Agent],
    num_games: int,
    maps: Sequence[str] = SUPPORTED_MAPS,
    seed_start: int = 0,
    num_workers: int = 8,
    batch_size: int = 32,
    agent_a_name: str = "candidate",
    agent_b_name: str = "opponent",
) -> DevResult:
    """Play seat-swapped `dev-*` games and summarize with the paired bootstrap CI."""
    seeds = generate_seed_keys("dev", dev_seed_count(num_games, maps), start=seed_start)
    specs, metas = build_paired_specs(
        agent_a=candidate_factory,
        agent_b=opponent_factory,
        seed_keys=seeds,
        maps=maps,
    )
    assert_namespace([m.seed_key for m in metas], "dev")

    import time

    t0 = time.perf_counter()
    outcomes = play_games(specs, num_workers=num_workers, batch_size=batch_size)
    wall_s = time.perf_counter() - t0
    report = summarize_evaluation(
        outcomes=outcomes,
        metas=metas,
        namespace="dev",
        agent_a_name=agent_a_name,
        agent_b_name=agent_b_name,
        agent_a_version=None,
        agent_b_version=None,
        workers=num_workers,
        batch_size=batch_size,
        wall_time_s=wall_s,
    )
    det = report["deterministic"]
    ci = det["win_rate_ci_95"]
    return DevResult(
        games=det["total_games"],
        completed=det["completed_games"],
        failures=det["failure_count"],
        win_rate=det["win_rate"],
        win_rate_ci_95=(tuple(ci) if ci is not None else None),  # type: ignore[arg-type]
        mean_vp_margin=det["mean_vp_margin"],
        report=report,
    )


def build_checkpoint_metrics(
    *,
    cycle: int,
    elapsed_s: float,
    step: int,
    games: int,
    checkpoint: Path,
    vs_b1: DevResult,
    vs_previous_best: DevResult | None,
    new_best: bool,
    best_dev_win_rate: float | None,
    training_s: float | None = None,
    dev_eval_s: float | None = None,
) -> CheckpointMetrics:
    return CheckpointMetrics(
        cycle=cycle,
        elapsed_s=float(elapsed_s),
        step=int(step),
        games=int(games),
        checkpoint=str(checkpoint),
        dev_games=int(vs_b1.games),
        dev_win_rate_vs_b1=vs_b1.win_rate,
        dev_win_rate_vs_b1_ci_95=vs_b1.win_rate_ci_95,
        dev_win_rate_vs_previous_best=(
            None if vs_previous_best is None else vs_previous_best.win_rate
        ),
        dev_win_rate_vs_previous_best_ci_95=(
            None if vs_previous_best is None else vs_previous_best.win_rate_ci_95
        ),
        dev_mean_vp_margin_vs_b1=vs_b1.mean_vp_margin,
        new_best=bool(new_best),
        best_dev_win_rate_vs_b1=best_dev_win_rate,
        training_s=None if training_s is None else float(training_s),
        dev_eval_s=None if dev_eval_s is None else float(dev_eval_s),
    )
