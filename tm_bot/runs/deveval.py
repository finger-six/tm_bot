"""Dev-seed checkpoint evaluation against B1 and the previous best checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from tm_bot.agents import B1Agent, load_card_data
from tm_bot.agents.protocol import Agent
from tm_bot.agents.runner import GameSpec, play_games
from tm_bot.env.types import SUPPORTED_MAPS
from tm_bot.eval.pairing import PairedGameMeta
from tm_bot.eval.report import summarize_evaluation
from tm_bot.eval.seeds import derive_agent_seed, make_seed_key
from tm_bot.models.agent import PolicyValueAgent
from tm_bot.runs.monitoring import CheckpointMetrics
from tm_bot.runs.selfplay import assert_namespace
from tm_bot.search.agent import ISMCTSAgent
from tm_bot.search.config import SearchConfig

_DEV_AGENT_CACHE: dict[tuple[str, str], PolicyValueAgent] = {}


def cached_policy_value_agent(checkpoint: str | Path, device: str) -> PolicyValueAgent:
    key = (str(Path(checkpoint).resolve()), device)
    cached = _DEV_AGENT_CACHE.get(key)
    if cached is None:
        cached = PolicyValueAgent.from_checkpoint(
            checkpoint, device=device, temperature=0.0
        )
        _DEV_AGENT_CACHE[key] = cached
    return cached


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


def build_dev_games(
    num_games: int,
    *,
    maps: Sequence[str] = SUPPORTED_MAPS,
    seed_start: int = 0,
) -> list[tuple[str, str, int]]:
    """Build exactly `num_games` `dev-*` games with balanced seats.

    Each `(seed, map)` is played once; the candidate takes seat 0 on even game
    indices and seat 1 on odd ones, so seats are balanced across the schedule.
    """
    if num_games < 1:
        raise ValueError("num_games must be >= 1")
    if not maps:
        raise ValueError("maps must be non-empty")
    games: list[tuple[str, str, int]] = []
    for i in range(num_games):
        seed_index = seed_start + i // len(maps)
        seed_key = make_seed_key("dev", seed_index).key
        map_name = maps[i % len(maps)]
        a_seat = 0 if i % 2 == 0 else 1
        games.append((seed_key, map_name, a_seat))
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
    """Play exactly `num_games` seat-balanced `dev-*` games and summarize."""
    games = build_dev_games(num_games, maps=maps, seed_start=seed_start)
    specs: list[GameSpec] = []
    metas: list[PairedGameMeta] = []
    for seed_key, map_name, a_seat in games:
        seed_a = derive_agent_seed(seed_key, map_name=map_name, role="a")
        seed_b = derive_agent_seed(seed_key, map_name=map_name, role="b")
        if a_seat == 0:
            specs.append(
                GameSpec(
                    seed_key=seed_key,
                    map_name=map_name,
                    agents=(candidate_factory(), opponent_factory()),
                    seat_seeds=(seed_a, seed_b),
                )
            )
        else:
            specs.append(
                GameSpec(
                    seed_key=seed_key,
                    map_name=map_name,
                    agents=(opponent_factory(), candidate_factory()),
                    seat_seeds=(seed_b, seed_a),
                )
            )
        metas.append(
            PairedGameMeta(
                seed_key=seed_key, map_name=map_name, a_seat=a_seat, seed_a=seed_a, seed_b=seed_b
            )
        )

    import time

    t0 = time.perf_counter()
    outcomes = play_games(
        specs, num_workers=num_workers, batch_size=batch_size
    )
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
    vs_previous_best: DevResult,
    new_best: bool,
    best_dev_win_rate: float | None,
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
        dev_win_rate_vs_previous_best=vs_previous_best.win_rate,
        dev_win_rate_vs_previous_best_ci_95=vs_previous_best.win_rate_ci_95,
        dev_mean_vp_margin_vs_b1=vs_b1.mean_vp_margin,
        new_best=bool(new_best),
        best_dev_win_rate_vs_b1=best_dev_win_rate,
    )
