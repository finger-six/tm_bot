"""The 1 h development gate before a 24 h run (#40 section 3, #52).

A gate run is an ordinary self-play run with `--gate-1h` (`dev_gate_config`):
1 h of training clock, dev evaluation at 30 and 60 min. This module reads the
finished run directory and checks it; it plays no games and needs no GPU.

```bash
python -m tm_bot.runs.dev_gate runs/<run-name>   # Markdown report; exit 1 on failure
```

All seven checks must pass:

1. Legality: 0 failures, 0 adapter rejections, 0 step-cap hits.
2. Game shape: self-play median <= 18 generations, p95 <= 25, no game > 40.
3. Throughput: self-play games per training hour, main >= 250, comparator >= 800.
4. Symmetry: learner-seat win share vs the pool and vs the anchor in [35%, 65%].
5. Cost: dev-eval wall time <= 40% of the training clock.
6. Positive signal at 60 min: `a >= 0.45`, `D` >= the warm-start anchor's own `D`
   vs `b1-v2` on the same 104 `dev-*` seeds, and no stop rule triggered.
7. Resources: learner RSS stays under the 20.0 GiB guard.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from tm_bot.runs.monitoring import MonitorLogger

# Thresholds (#40 section 3).
MAX_MEDIAN_GENERATIONS = 18.0
MAX_P95_GENERATIONS = 25.0
MAX_GENERATIONS = 40
MIN_GAMES_PER_TRAINING_HOUR = {"main": 250.0, "comparator": 800.0}
SYMMETRY_LOW = 0.35
SYMMETRY_HIGH = 0.65
SYMMETRY_OPPONENTS = ("pool", "anchor")
MAX_DEV_EVAL_FRACTION = 0.40
MIN_ANCHOR_WIN_RATE = 0.45
# The 60-min evaluation runs once the training clock reaches the budget; allow
# float noise in the logged clock.
_CLOCK_TOLERANCE_S = 1.0


@dataclass(frozen=True, slots=True)
class DevEvalPoint:
    """One dev evaluation as the gate sees it."""

    training_s: float
    anchor_win_rate: float | None
    dev_score: float | None
    wall_s: float | None = None


@dataclass(frozen=True, slots=True)
class GateSummary:
    """Everything the gate checks, aggregated from one finished run."""

    run_name: str
    agent_kind: str
    num_simulations: int
    num_worlds: int
    budget_s: float
    learner_ram_gb: float
    selfplay_games: int
    training_s: float
    dev_eval_s: float
    failures: int = 0
    adapter_rejections: int = 0
    step_cap_hits: int = 0
    # Completed self-play games per generation count.
    generation_counts: Mapping[int, int] = field(default_factory=dict)
    # Learner-seat {"wins", "losses", "ties"} per opponent kind.
    results_by_opponent: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    dev_evals: tuple[DevEvalPoint, ...] = ()
    anchor_dev_score: float | None = None
    stop_kind: str | None = None
    stop_reason: str = ""
    rss_gb_max: float | None = None
    dev_eval_baseline_s: float = 0.0
    dev_gate: bool = True


@dataclass(frozen=True, slots=True)
class GateCheck:
    number: int
    name: str
    passed: bool
    measured: str
    threshold: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GateReport:
    summary: GateSummary
    checks: tuple[GateCheck, ...]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        summary = asdict(self.summary)
        summary["generation_counts"] = {
            str(k): v for k, v in sorted(self.summary.generation_counts.items())
        }
        return {
            "schema_version": "dev-gate-v1",
            "passed": self.passed,
            "summary": summary,
            "checks": [asdict(c) for c in self.checks],
        }


# --- loading ----------------------------------------------------------------
def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: not a finished run directory")
    return json.loads(path.read_text(encoding="utf-8"))


def load_gate_summary(run_dir: str | Path) -> GateSummary:
    """Aggregate a finished run directory (config, metrics stream, run summary)."""
    run_dir = Path(run_dir)
    config = _read_json(run_dir / "config.json")
    run = _read_json(run_dir / "run_summary.json")
    logger = MonitorLogger(run_dir / "monitor" / "metrics.jsonl")
    logger.load_existing()
    cycles = logger.cycles
    evals = logger.checkpoints

    generation_counts: dict[int, int] = {}
    by_opponent: dict[str, dict[str, int]] = {}
    for cycle in cycles:
        for gen, count in cycle.generation_counts.items():
            generation_counts[int(gen)] = generation_counts.get(int(gen), 0) + int(count)
        for kind, results in cycle.results_by_opponent.items():
            total = by_opponent.setdefault(kind, {"wins": 0, "losses": 0, "ties": 0})
            for key in total:
                total[key] += int(results.get(key, 0))
    rss = [r.rss_gb for r in [*cycles, *evals] if r.rss_gb is not None]
    stop = run.get("stop") or {}
    limits = config.get("limits") or {}
    return GateSummary(
        run_name=str(config.get("run_name", run_dir.name)),
        agent_kind=str(config.get("agent_kind", "main")),
        num_simulations=int(config["num_simulations"]),
        num_worlds=int(config["num_worlds"]),
        budget_s=float(config["wall_clock_budget_s"]),
        learner_ram_gb=float(limits.get("learner_ram_gb", 20.0)),
        selfplay_games=int(run.get("selfplay_games", 0)),
        training_s=float(run.get("training_s", 0.0)),
        dev_eval_s=float(run.get("dev_eval_s", 0.0)),
        failures=sum(int(c.failures) for c in cycles) + sum(int(e.dev_failures) for e in evals),
        adapter_rejections=sum(int(c.rejections) for c in cycles),
        step_cap_hits=sum(1 for r in logger.rejections if r.get("error_type") == "StepCapError"),
        generation_counts=generation_counts,
        results_by_opponent=by_opponent,
        dev_evals=tuple(
            DevEvalPoint(
                training_s=float(e.training_s if e.training_s is not None else e.elapsed_s),
                anchor_win_rate=e.dev_win_rate_vs_anchor,
                dev_score=e.dev_score_vs_b1,
                wall_s=e.dev_eval_wall_s,
            )
            for e in evals
        ),
        anchor_dev_score=(
            None if run.get("anchor_dev_score") is None else float(run["anchor_dev_score"])
        ),
        stop_kind=stop.get("kind"),
        stop_reason=str(stop.get("reason") or ""),
        rss_gb_max=max(rss) if rss else None,
        dev_eval_baseline_s=float(sum(e.dev_eval_baseline_s for e in evals)),
        dev_gate=bool(config.get("dev_gate", False)),
    )


# --- checks -------------------------------------------------------------------
def _fmt(value: float | None, spec: str = ".3f") -> str:
    return "n/a" if value is None else format(value, spec)


def _play_setting(s: GateSummary) -> str:
    """The comparator scores candidates directly; only the main agent searches."""
    if s.agent_kind == "comparator":
        return "direct scoring (no search)"
    return f"search {s.num_simulations}/{s.num_worlds}"


def check_legality(s: GateSummary) -> GateCheck:
    passed = s.failures == 0 and s.adapter_rejections == 0 and s.step_cap_hits == 0
    return GateCheck(
        1,
        "Legality",
        passed,
        f"{s.failures} failures, {s.adapter_rejections} adapter rejections, "
        f"{s.step_cap_hits} step-cap hits",
        "all 0",
        {
            "failures": s.failures,
            "adapter_rejections": s.adapter_rejections,
            "step_cap_hits": s.step_cap_hits,
        },
    )


def generation_stats(counts: Mapping[int, int]) -> tuple[float, float, int] | None:
    """(median, p95, max) over completed self-play games, or None without games."""
    values = np.repeat(
        np.asarray(list(counts.keys()), dtype=np.float64),
        np.asarray(list(counts.values()), dtype=np.int64),
    )
    if values.size == 0:
        return None
    return float(np.median(values)), float(np.percentile(values, 95)), int(values.max())


def check_game_shape(s: GateSummary) -> GateCheck:
    threshold = (
        f"median <= {MAX_MEDIAN_GENERATIONS:g}, p95 <= {MAX_P95_GENERATIONS:g}, "
        f"max <= {MAX_GENERATIONS}"
    )
    stats = generation_stats(s.generation_counts)
    if stats is None:
        return GateCheck(2, "Game shape", False, "no completed games logged", threshold)
    median, p95, top = stats
    passed = (
        median <= MAX_MEDIAN_GENERATIONS and p95 <= MAX_P95_GENERATIONS and top <= MAX_GENERATIONS
    )
    return GateCheck(
        2,
        "Game shape",
        passed,
        f"median {median:g}, p95 {p95:g}, max {top} generations "
        f"({sum(s.generation_counts.values())} games)",
        threshold,
        {"median": median, "p95": p95, "max": top},
    )


def check_throughput(s: GateSummary) -> GateCheck:
    floor = MIN_GAMES_PER_TRAINING_HOUR.get(s.agent_kind)
    rate = s.selfplay_games / (s.training_s / 3600.0) if s.training_s > 0.0 else None
    passed = floor is not None and rate is not None and rate >= floor
    return GateCheck(
        3,
        "Throughput",
        passed,
        f"{_fmt(rate, '.0f')} games / training h, {_play_setting(s)} "
        f"({s.selfplay_games} games in {s.training_s / 60.0:.1f} min)",
        f"{s.agent_kind} >= {_fmt(floor, '.0f')}",
        {"games_per_training_hour": rate, "floor": floor},
    )


def win_share(results: Mapping[str, int] | None) -> tuple[float | None, int]:
    """Learner-seat win share (a tie counts half) and the number of games."""
    if not results:
        return None, 0
    wins, losses, ties = (int(results.get(k, 0)) for k in ("wins", "losses", "ties"))
    games = wins + losses + ties
    return ((wins + 0.5 * ties) / games if games else None), games


def check_symmetry(s: GateSummary) -> GateCheck:
    shares: dict[str, Any] = {}
    parts: list[str] = []
    passed = True
    for kind in SYMMETRY_OPPONENTS:
        share, games = win_share(s.results_by_opponent.get(kind))
        shares[kind] = {"win_share": share, "games": games}
        parts.append(f"vs {kind} {_fmt(None if share is None else 100 * share, '.1f')}% (n={games})")
        passed = passed and share is not None and SYMMETRY_LOW <= share <= SYMMETRY_HIGH
    return GateCheck(
        4,
        "Symmetry",
        passed,
        ", ".join(parts),
        f"each in [{SYMMETRY_LOW:.0%}, {SYMMETRY_HIGH:.0%}]",
        shares,
    )


def check_cost(s: GateSummary) -> GateCheck:
    fraction = s.dev_eval_s / s.training_s if s.training_s > 0.0 else None
    recurring = (
        (s.dev_eval_s - s.dev_eval_baseline_s) / s.training_s if s.training_s > 0.0 else None
    )
    passed = fraction is not None and fraction <= MAX_DEV_EVAL_FRACTION
    return GateCheck(
        5,
        "Cost",
        passed,
        f"dev eval {s.dev_eval_s / 60.0:.1f} min = {_fmt(None if fraction is None else 100 * fraction, '.1f')}% "
        f"of {s.training_s / 60.0:.1f} min training clock "
        f"({_fmt(None if recurring is None else 100 * recurring, '.1f')}% without the "
        f"one-time anchor baseline)",
        f"<= {MAX_DEV_EVAL_FRACTION:.0%}",
        {
            "dev_eval_fraction": fraction,
            "dev_eval_fraction_without_baseline": recurring,
            "dev_eval_s": s.dev_eval_s,
            "dev_eval_baseline_s": s.dev_eval_baseline_s,
        },
    )


def check_positive_signal(s: GateSummary) -> GateCheck:
    threshold = (
        f"dev evals at 30 and 60 min; a >= {MIN_ANCHOR_WIN_RATE:.2f}; D >= anchor D; "
        "stop = budget"
    )
    at_min = [round(e.training_s / 60.0, 1) for e in s.dev_evals]
    final = s.dev_evals[-1] if s.dev_evals else None
    schedule_ok = (
        len(s.dev_evals) >= 2
        and final is not None
        and final.training_s >= s.budget_s - _CLOCK_TOLERANCE_S
    )
    a = None if final is None else final.anchor_win_rate
    d = None if final is None else final.dev_score
    anchor_d = s.anchor_dev_score
    a_ok = a is not None and a >= MIN_ANCHOR_WIN_RATE
    d_ok = d is not None and anchor_d is not None and d >= anchor_d
    stop_ok = s.stop_kind == "budget"
    stop = s.stop_kind or "none"
    if s.stop_kind != "budget" and s.stop_reason:
        stop = f"{stop} ({s.stop_reason})"
    return GateCheck(
        6,
        "Positive signal at 60 min",
        schedule_ok and a_ok and d_ok and stop_ok,
        f"dev evals at {at_min} min; a = {_fmt(a)}; D = {_fmt(d, '.4f')} vs anchor D = "
        f"{_fmt(anchor_d, '.4f')}; stop: {stop}",
        threshold,
        {
            "dev_evals_at_min": at_min,
            "schedule_ok": schedule_ok,
            "anchor_win_rate": a,
            "dev_score": d,
            "anchor_dev_score": anchor_d,
            "stop_kind": s.stop_kind,
        },
    )


def check_resources(s: GateSummary) -> GateCheck:
    # Same comparison as the guard (`check_learner_memory` stops above the limit).
    guard_fired = s.stop_kind == "resource_limit"
    passed = s.rss_gb_max is not None and s.rss_gb_max <= s.learner_ram_gb and not guard_fired
    measured = (
        "no RSS readings logged (guard inert)"
        if s.rss_gb_max is None
        else f"max RSS {s.rss_gb_max:.2f} GiB"
    )
    if guard_fired:
        measured += f"; resource_limit stop: {s.stop_reason}"
    return GateCheck(
        7,
        "Resources",
        passed,
        measured,
        f"<= {s.learner_ram_gb:.1f} GiB",
        {"rss_gb_max": s.rss_gb_max, "learner_ram_gb": s.learner_ram_gb},
    )


CHECKS = (
    check_legality,
    check_game_shape,
    check_throughput,
    check_symmetry,
    check_cost,
    check_positive_signal,
    check_resources,
)


def evaluate_gate(summary: GateSummary) -> GateReport:
    return GateReport(summary=summary, checks=tuple(check(summary) for check in CHECKS))


# --- report -------------------------------------------------------------------
def render_markdown(report: GateReport) -> str:
    s = report.summary
    verdict = "PASS" if report.passed else "FAIL"
    lines = [
        f"## 1 h development gate: `{s.run_name}` ({s.agent_kind}): **{verdict}**",
        "",
        f"{_play_setting(s).capitalize()}; training clock "
        f"{s.training_s / 60.0:.1f} min (budget {s.budget_s / 60.0:.0f} min); "
        f"{s.selfplay_games} self-play games; dev evaluation {s.dev_eval_s / 60.0:.1f} min.",
    ]
    if not s.dev_gate:
        lines += ["", "Note: this run was not started with `--gate-1h`."]
    lines += [
        "",
        "| # | Check | Result | Measured | Threshold |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for c in report.checks:
        lines.append(
            f"| {c.number} | {c.name} | {'PASS' if c.passed else '**FAIL**'} | "
            f"{c.measured} | {c.threshold} |"
        )
    throughput = report.checks[2]
    if s.agent_kind == "main" and not throughput.passed:
        lines += [
            "",
            "Throughput missed: rerun the gate with `--search-fallback` (8/4); if it still "
            "misses, stop and diagnose (#40 section 3).",
        ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check a finished 1 h development gate run (#52)."
    )
    parser.add_argument("run_dir", type=Path, help="The run directory, e.g. runs/<run-name>.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of Markdown.")
    parser.add_argument("--out", type=Path, default=None, help="Also write the report here.")
    args = parser.parse_args(argv)

    report = evaluate_gate(load_gate_summary(args.run_dir))
    text = (
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
        if args.json
        else render_markdown(report)
    )
    sys.stdout.write(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
