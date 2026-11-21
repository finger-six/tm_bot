"""Dev-seed checkpoint evaluation against `b1-v2`, the incumbent best, and the warm-start anchor.

Dev games use the same seat-swapped pairing as `tm_bot.eval.pairing`: every
`(dev-seed, map)` is played twice with the candidate in opposite seats, so the
paired bootstrap CI is valid. Per-game results are kept (`DevGame`) so a
candidate's `b1-v2` games pair with the incumbent's by `(seed, map, seat)` for
the dev score difference (#40 section 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tm_bot.agents import B1Agent, load_card_data
from tm_bot.agents.protocol import Agent, GameRunResult
from tm_bot.agents.runner import play_games
from tm_bot.env.types import SUPPORTED_MAPS
from tm_bot.eval.pairing import PairedGameMeta, build_paired_specs
from tm_bot.eval.report import summarize_evaluation
from tm_bot.eval.seeds import generate_seed_keys
from tm_bot.eval.stats import cluster_bootstrap_mean_ci
from tm_bot.models.agent import PolicyValueAgent
from tm_bot.models.network import vp_margin_target
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


# Weight of the normalised VP margin in the shared utility `U = o + 0.15 * m` (#39 section 1c).
DEV_SCORE_MARGIN_WEIGHT = 0.15


@dataclass(frozen=True, slots=True)
class DevGame:
    """One dev game from the candidate's side, kept so pairings can be matched by seed."""

    seed_key: str
    map_name: str
    candidate_seat: int
    ok: bool
    outcome: int | None = None  # +1 win, 0 tie, -1 loss for the candidate
    vp_margin: int | None = None  # candidate VP - opponent VP
    generations: int | None = None

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.seed_key, self.map_name, self.candidate_seat)

    @property
    def utility(self) -> float | None:
        """`U = o + 0.15 * clip(margin, -60, 60) / 50` (#39 section 1c); None if unfinished."""
        if not self.ok or self.outcome is None or self.vp_margin is None:
            return None
        return float(self.outcome) + DEV_SCORE_MARGIN_WEIGHT * vp_margin_target(
            float(self.vp_margin), 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_key": self.seed_key,
            "map_name": self.map_name,
            "candidate_seat": self.candidate_seat,
            "ok": self.ok,
            "outcome": self.outcome,
            "vp_margin": self.vp_margin,
            "generations": self.generations,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DevGame:
        def opt_int(key: str) -> int | None:
            value = payload.get(key)
            return None if value is None else int(value)

        return cls(
            seed_key=str(payload["seed_key"]),
            map_name=str(payload["map_name"]),
            candidate_seat=int(payload["candidate_seat"]),
            ok=bool(payload["ok"]),
            outcome=opt_int("outcome"),
            vp_margin=opt_int("vp_margin"),
            generations=opt_int("generations"),
        )


def dev_game_records(
    outcomes: Sequence[GameRunResult], metas: Sequence[PairedGameMeta]
) -> tuple[DevGame, ...]:
    """Per-game candidate-side records of a seat-swapped pairing (agent A is the candidate)."""
    if len(outcomes) != len(metas):
        raise ValueError(f"outcomes ({len(outcomes)}) and metas ({len(metas)}) differ")
    records: list[DevGame] = []
    for out, meta in zip(outcomes, metas):
        seat = int(meta.a_seat)
        if not out.ok or out.result is None:
            records.append(DevGame(meta.seed_key, meta.map_name, seat, ok=False))
            continue
        res = out.result
        outcome = 0 if res.tie else (1 if res.winner == seat else -1)
        records.append(
            DevGame(
                meta.seed_key,
                meta.map_name,
                seat,
                ok=True,
                outcome=outcome,
                vp_margin=int(res.scores[seat]) - int(res.scores[1 - seat]),
                generations=int(res.generations),
            )
        )
    return tuple(records)


def dev_score(games: Sequence[DevGame]) -> float | None:
    """Dev score `D`: mean utility over the completed games (#40 section 1)."""
    utilities = [u for u in (g.utility for g in games) if u is not None]
    return float(sum(utilities) / len(utilities)) if utilities else None


def dev_win_rate(games: Sequence[DevGame]) -> float | None:
    """Win rate over completed games, a tie counting half (as in the eval report)."""
    done = [g.outcome for g in games if g.ok and g.outcome is not None]
    if not done:
        return None
    return float(sum((o + 1) / 2.0 for o in done) / len(done))


def games_on_seeds(games: Sequence[DevGame], seed_keys: Sequence[str]) -> tuple[DevGame, ...]:
    keep = set(seed_keys)
    return tuple(g for g in games if g.seed_key in keep)


def paired_delta_d(
    candidate: Sequence[DevGame],
    incumbent: Sequence[DevGame],
    *,
    n_resamples: int = 10_000,
    rng_seed: int = 0,
) -> tuple[float | None, tuple[float, float] | None]:
    """`dD = D_C - D_best` over games paired by `(seed, map, seat)`, with a 95% bootstrap CI.

    Only keys both checkpoints completed are paired. Whole seeds are resampled,
    because one seed's deal is shared by all its maps and seats.
    """
    best = {g.key: g.utility for g in incumbent if g.utility is not None}
    clusters: dict[str, list[float]] = {}
    for game in candidate:
        u = game.utility
        if u is None or game.key not in best:
            continue
        clusters.setdefault(game.seed_key, []).append(u - best[game.key])
    return cluster_bootstrap_mean_ci(
        [(sum(diffs), len(diffs)) for diffs in clusters.values()],
        n_resamples=n_resamples,
        rng_seed=rng_seed,
    )


def dev_eval_interval(eval_s: float, *, base_interval_s: float, cost_fraction: float) -> float:
    """Dev-eval cadence after an evaluation that took `eval_s` of wall clock (#40 section 2).

    The base interval doubles (60 -> 120 -> 240 min) until one evaluation fits in
    `cost_fraction` of it, and never drops below the base.
    """
    if base_interval_s <= 0.0 or cost_fraction <= 0.0:
        raise ValueError("base_interval_s and cost_fraction must be > 0")
    interval = float(base_interval_s)
    while eval_s > cost_fraction * interval:
        interval *= 2.0
    return interval


@dataclass(frozen=True, slots=True)
class DevResult:
    """Summary of one dev pairing, with its per-game candidate-side records."""

    games: int
    completed: int
    failures: int
    win_rate: float | None
    win_rate_ci_95: tuple[float, float] | None
    mean_vp_margin: float | None
    report: dict[str, Any]
    records: tuple[DevGame, ...] = ()
    # Candidate value-head ECE over its recorded decisions (the `b1-v2` pairing only).
    ece: float | None = None


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
    record_values: str | None = None,
) -> DevResult:
    """Play seat-swapped `dev-*` games and summarize with the paired bootstrap CI.

    `record_values` (`"main"` or `"comparator"`) also records the candidate's win
    probability at each non-forced decision and reports its ECE.
    """
    seeds = generate_seed_keys("dev", dev_seed_count(num_games, maps), start=seed_start)
    recorders: list[list[tuple[int, float]]] = []
    candidate = candidate_factory
    if record_values is not None:
        from tm_bot.training.run_warmstart_gate import _recording

        candidate = _recording(record_values, candidate_factory, recorders)
    specs, metas = build_paired_specs(
        agent_a=candidate,
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
    ece: float | None = None
    if record_values is not None:
        from tm_bot.training.run_warmstart_gate import value_entries_with_outcomes
        from tm_bot.training.warmstart_gate import live_value_metrics

        values = live_value_metrics(value_entries_with_outcomes(outcomes, metas, recorders))
        ece = values["all"].get("ece")
    return DevResult(
        games=det["total_games"],
        completed=det["completed_games"],
        failures=det["failure_count"],
        win_rate=det["win_rate"],
        win_rate_ci_95=(tuple(ci) if ci is not None else None),  # type: ignore[arg-type]
        mean_vp_margin=det["mean_vp_margin"],
        report=report,
        records=dev_game_records(outcomes, metas),
        ece=None if ece is None else float(ece),
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
    **selection: Any,
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
        **selection,
    )
