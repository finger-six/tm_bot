"""Evaluation aggregation, JSON serialization (deterministic vs runtime), and Markdown reports."""

from __future__ import annotations

import json
import os
import platform
import statistics
from pathlib import Path
from typing import Any, Sequence

from tm_bot.agents.protocol import GameRunResult
from tm_bot.env.types import SUPPORTED_MAPS
from tm_bot.eval.pairing import PairedGameMeta
from tm_bot.eval.stats import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    paired_bootstrap_ci,
    summarize_latency,
)

AMAZONIS_MAP_NAME = "amazonis planitia"
FIVE_MAP_SCOPE: tuple[str, ...] = (*SUPPORTED_MAPS, AMAZONIS_MAP_NAME)
EVAL_REPORT_SCHEMA_VERSION = "eval-report-v1"


def collect_hardware_info() -> dict[str, Any]:
    """Collect host hardware and Python environment metadata."""
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count() or 1,
        "python_version": platform.python_version(),
    }


def _resolve_agent_version(
    outcomes: Sequence[GameRunResult],
    metas: Sequence[PairedGameMeta],
    *,
    is_agent_a: bool,
    explicit_version: str | None,
) -> str | None:
    if explicit_version is not None:
        return explicit_version
    for out, meta in zip(outcomes, metas):
        seat_idx = meta.a_seat if is_agent_a else (1 - meta.a_seat)
        ver = out.agent_versions[seat_idx]
        if ver is not None:
            return ver
    return None


def summarize_evaluation(
    *,
    outcomes: Sequence[GameRunResult],
    metas: Sequence[PairedGameMeta],
    namespace: str,
    agent_a_name: str,
    agent_b_name: str,
    agent_a_version: str | None = None,
    agent_b_version: str | None = None,
    a_training_seeds: int = 0,
    b_training_seeds: int = 0,
    a_compute: dict[str, Any] | None = None,
    b_compute: dict[str, Any] | None = None,
    workers: int = 1,
    batch_size: int = 1,
    wall_time_s: float = 0.0,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Aggregate game outcomes into separated `deterministic` and `runtime` report dicts."""
    if len(outcomes) != len(metas):
        raise ValueError(
            f"Length mismatch between outcomes ({len(outcomes)}) and metas ({len(metas)})"
        )
    if a_training_seeds < 0 or b_training_seeds < 0:
        raise ValueError("training_seeds must be >= 0")

    wins = 0
    losses = 0
    ties = 0
    vp_margins: list[int] = []
    vp_a_list: list[int] = []
    vp_b_list: list[int] = []
    generations_list: list[int] = []
    completed_decisions_list: list[int] = []
    total_decisions = 0

    pair_buckets: dict[tuple[str, str], list[float]] = {}
    map_pair_buckets: dict[str, dict[tuple[str, str], list[float]]] = {
        m: {} for m in SUPPORTED_MAPS
    }
    map_stats_raw: dict[str, dict[str, Any]] = {
        m: {
            "total_games": 0,
            "completed_games": 0,
            "failures": 0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "margins": [],
            "generations": [],
        }
        for m in SUPPORTED_MAPS
    }

    failures: list[dict[str, Any]] = []
    latencies_a: list[float] = []
    latencies_b: list[float] = []

    for out, meta in zip(outcomes, metas):
        total_decisions += int(out.decisions)
        lat0, lat1 = out.decision_latencies_ms
        if meta.a_seat == 0:
            latencies_a.extend(lat0)
            latencies_b.extend(lat1)
        else:
            latencies_a.extend(lat1)
            latencies_b.extend(lat0)

        map_bucket = map_stats_raw.setdefault(
            meta.map_name,
            {
                "total_games": 0,
                "completed_games": 0,
                "failures": 0,
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "margins": [],
                "generations": [],
            },
        )
        map_bucket["total_games"] += 1

        pair_key = (meta.seed_key, meta.map_name)
        pair_buckets.setdefault(pair_key, [])
        map_pair_buckets.setdefault(meta.map_name, {}).setdefault(pair_key, [])

        if not out.ok or out.result is None:
            map_bucket["failures"] += 1
            failures.append(
                {
                    "seed_key": meta.seed_key,
                    "map": meta.map_name,
                    "a_seat": meta.a_seat,
                    "decisions": int(out.decisions),
                    "error_type": out.error_type or "UnknownError",
                    "error": out.error or "",
                }
            )
            continue

        res = out.result
        a_vp = int(res.scores[meta.a_seat])
        b_vp = int(res.scores[1 - meta.a_seat])
        margin = a_vp - b_vp

        vp_a_list.append(a_vp)
        vp_b_list.append(b_vp)
        vp_margins.append(margin)
        generations_list.append(int(res.generations))
        completed_decisions_list.append(int(out.decisions))

        map_bucket["completed_games"] += 1
        map_bucket["margins"].append(margin)
        map_bucket["generations"].append(int(res.generations))

        if res.tie:
            ties += 1
            pts = 0.5
            map_bucket["ties"] += 1
        elif res.winner == meta.a_seat:
            wins += 1
            pts = 1.0
            map_bucket["wins"] += 1
        else:
            losses += 1
            pts = 0.0
            map_bucket["losses"] += 1

        pair_buckets[pair_key].append(pts)
        map_pair_buckets[meta.map_name][pair_key].append(pts)

    total_games = len(outcomes)
    completed_games = wins + losses + ties
    failure_count = len(failures)
    failure_rate = (failure_count / total_games) if total_games > 0 else 0.0

    overall_pair_totals = [
        (sum(pts_list), len(pts_list))
        for pts_list in pair_buckets.values()
        if pts_list
    ]
    win_rate, win_rate_ci = paired_bootstrap_ci(
        overall_pair_totals,
        n_resamples=bootstrap_resamples,
        confidence=0.95,
        rng_seed=bootstrap_seed,
    )

    per_map: dict[str, dict[str, Any]] = {}
    for map_name in FIVE_MAP_SCOPE:
        if map_name == AMAZONIS_MAP_NAME:
            per_map[map_name] = {
                "status": "not supported",
                "note": "not supported",
                "total_games": 0,
                "completed_games": 0,
                "failures": 0,
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "win_rate": None,
                "win_rate_ci_95": None,
                "mean_vp_margin": None,
                "mean_generations": None,
                "loses_to_opponent": None,
            }
            continue

        mb = map_stats_raw[map_name]
        m_pairs = [
            (sum(pts_list), len(pts_list))
            for pts_list in map_pair_buckets[map_name].values()
            if pts_list
        ]
        m_wr, m_ci = paired_bootstrap_ci(
            m_pairs,
            n_resamples=bootstrap_resamples,
            confidence=0.95,
            rng_seed=bootstrap_seed,
        )
        per_map[map_name] = {
            "status": "supported",
            "total_games": mb["total_games"],
            "completed_games": mb["completed_games"],
            "failures": mb["failures"],
            "wins": mb["wins"],
            "losses": mb["losses"],
            "ties": mb["ties"],
            "win_rate": m_wr,
            "win_rate_ci_95": [m_ci[0], m_ci[1]] if m_ci is not None else None,
            "mean_vp_margin": statistics.mean(mb["margins"]) if mb["margins"] else None,
            "mean_generations": statistics.mean(mb["generations"]) if mb["generations"] else None,
            "loses_to_opponent": bool(m_wr is not None and m_wr < 0.5),
        }

    resolved_a_ver = _resolve_agent_version(
        outcomes, metas, is_agent_a=True, explicit_version=agent_a_version
    )
    resolved_b_ver = _resolve_agent_version(
        outcomes, metas, is_agent_a=False, explicit_version=agent_b_version
    )

    deterministic: dict[str, Any] = {
        "schema_version": EVAL_REPORT_SCHEMA_VERSION,
        "namespace": namespace,
        "agent_a": {
            "name": agent_a_name,
            "version": resolved_a_ver,
            "training_seeds": int(a_training_seeds),
            "fewer_than_3_training_seeds": bool(0 < a_training_seeds < 3),
            "compute": dict(a_compute or {}),
        },
        "agent_b": {
            "name": agent_b_name,
            "version": resolved_b_ver,
            "training_seeds": int(b_training_seeds),
            "fewer_than_3_training_seeds": bool(0 < b_training_seeds < 3),
            "compute": dict(b_compute or {}),
        },
        "total_games": total_games,
        "completed_games": completed_games,
        "failure_count": failure_count,
        "failure_rate": failure_rate,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": win_rate,
        "win_rate_ci_95": [win_rate_ci[0], win_rate_ci[1]] if win_rate_ci is not None else None,
        "bootstrap_resamples": int(bootstrap_resamples),
        "bootstrap_seed": int(bootstrap_seed),
        "mean_vp_margin": statistics.mean(vp_margins) if vp_margins else None,
        "mean_vp_a": statistics.mean(vp_a_list) if vp_a_list else None,
        "mean_vp_b": statistics.mean(vp_b_list) if vp_b_list else None,
        "mean_generations": statistics.mean(generations_list) if generations_list else None,
        "min_generations": min(generations_list) if generations_list else None,
        "max_generations": max(generations_list) if generations_list else None,
        "total_decisions": total_decisions,
        "mean_decisions_per_game": (
            statistics.mean(completed_decisions_list) if completed_decisions_list else None
        ),
        "per_map": per_map,
        "failures": failures,
    }

    runtime: dict[str, Any] = {
        "workers": int(workers),
        "batch_size": int(batch_size),
        "wall_time_s": float(wall_time_s),
        "games_per_s": (float(total_games) / float(wall_time_s)) if wall_time_s > 0.0 else None,
        "hardware": collect_hardware_info(),
        "latency_ms": {
            "agent_a": summarize_latency(latencies_a),
            "agent_b": summarize_latency(latencies_b),
            "overall": summarize_latency([*latencies_a, *latencies_b]),
        },
    }

    return {
        "schema_version": EVAL_REPORT_SCHEMA_VERSION,
        "deterministic": deterministic,
        "runtime": runtime,
    }


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def _fmt_float(value: float | None, *, signed: bool = False, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    fmt = f"+.{digits}f" if signed else f".{digits}f"
    return format(value, fmt)


def format_markdown_report(
    report: dict[str, Any],
    *,
    omit_timing: bool = False,
) -> str:
    """Format an evaluation report dictionary as a concise Markdown summary."""
    det = report["deterministic"]
    a_info = det["agent_a"]
    b_info = det["agent_b"]
    a_label = f"{a_info['name']} ({a_info['version'] or 'unversioned'})"
    b_label = f"{b_info['name']} ({b_info['version'] or 'unversioned'})"

    ci = det["win_rate_ci_95"]
    ci_str = f"[{_fmt_pct(ci[0])}, {_fmt_pct(ci[1])}]" if ci is not None else "n/a"

    lines: list[str] = [
        f"# Evaluation Report: {a_label} vs {b_label}",
        "",
        f"- **Namespace**: `{det['namespace']}`",
        f"- **Games**: {det['total_games']} total | {det['completed_games']} completed | "
        f"{det['failure_count']} failures ({_fmt_pct(det['failure_rate'])})",
        f"- **Record ({a_info['name']})**: {det['wins']}W - {det['losses']}L - {det['ties']}T",
        f"- **Win Rate (95% paired bootstrap CI)**: **{_fmt_pct(det['win_rate'])}** {ci_str} "
        f"(resamples={det['bootstrap_resamples']}, seed={det['bootstrap_seed']})",
        f"- **Mean VP Margin ({a_info['name']} - {b_info['name']})**: "
        f"{_fmt_float(det['mean_vp_margin'], signed=True)} "
        f"(mean VP: {_fmt_float(det['mean_vp_a'])} vs {_fmt_float(det['mean_vp_b'])})",
        f"- **Generations / Game**: {_fmt_float(det['mean_generations'])} "
        f"(min={det['min_generations'] if det['min_generations'] is not None else 'n/a'}, "
        f"max={det['max_generations'] if det['max_generations'] is not None else 'n/a'})",
        f"- **Total Decisions**: {det['total_decisions']} "
        f"(mean/game={_fmt_float(det['mean_decisions_per_game'], digits=1)})",
        f"- **Training Seeds**: {a_info['name']}={a_info['training_seeds']}"
        + (" (WARNING: < 3 independent training seeds)" if a_info["fewer_than_3_training_seeds"] else "")
        + f", {b_info['name']}={b_info['training_seeds']}"
        + (" (WARNING: < 3 independent training seeds)" if b_info["fewer_than_3_training_seeds"] else ""),
        "",
        "## Per-Map Breakdown",
        "",
        "| Map | Status | Games | Completed | Failures | Win Rate | 95% CI | Mean VP Margin | Mean Gen |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]

    for map_name, row in det["per_map"].items():
        display_map = map_name.title()
        if row["status"] != "supported":
            lines.append(
                f"| {display_map} | not supported | 0 | 0 | 0 | not supported | not supported | not supported | not supported |"
            )
            continue
        m_ci = row["win_rate_ci_95"]
        m_ci_str = f"[{_fmt_pct(m_ci[0])}, {_fmt_pct(m_ci[1])}]" if m_ci is not None else "n/a"
        flag = " (loses)" if row.get("loses_to_opponent") else ""
        lines.append(
            f"| {display_map} | supported{flag} | {row['total_games']} | {row['completed_games']} | "
            f"{row['failures']} | {_fmt_pct(row['win_rate'])} | {m_ci_str} | "
            f"{_fmt_float(row['mean_vp_margin'], signed=True)} | {_fmt_float(row['mean_generations'])} |"
        )

    if det["failures"]:
        lines.extend(
            [
                "",
                "## Failures",
                "",
                f"Total failures excluded from win rate: {det['failure_count']}",
                "",
            ]
        )
        for fail in det["failures"][:20]:
            lines.append(
                f"- `seed={fail['seed_key']}` `map={fail['map']}` `a_seat={fail['a_seat']}` "
                f"`decisions={fail['decisions']}` `{fail['error_type']}`: {fail['error']}"
            )

    if not omit_timing and "runtime" in report:
        rt = report["runtime"]
        hw = rt["hardware"]
        lat_a = rt["latency_ms"]["agent_a"]
        lat_b = rt["latency_ms"]["agent_b"]
        lat_all = rt["latency_ms"]["overall"]
        lines.extend(
            [
                "",
                "## Compute & Decision Latency (Runtime)",
                "",
                f"- **Workers / Batch Size**: {rt['workers']} workers, batch_size={rt['batch_size']}",
                f"- **Wall Time**: {_fmt_float(rt['wall_time_s'])}s "
                f"({_fmt_float(rt['games_per_s'])} games/s)",
                f"- **Hardware**: {hw['platform']} | {hw['processor'] or hw['machine']} | "
                f"{hw['cpu_count']} CPUs | Python {hw['python_version']}",
                f"- **{a_info['name']} Decision Latency**: p50={_fmt_float(lat_a['p50_ms'], digits=4)} ms | "
                f"p95={_fmt_float(lat_a['p95_ms'], digits=4)} ms | mean={_fmt_float(lat_a['mean_ms'], digits=4)} ms "
                f"(n={lat_a['decisions']})",
                f"- **{b_info['name']} Decision Latency**: p50={_fmt_float(lat_b['p50_ms'], digits=4)} ms | "
                f"p95={_fmt_float(lat_b['p95_ms'], digits=4)} ms | mean={_fmt_float(lat_b['mean_ms'], digits=4)} ms "
                f"(n={lat_b['decisions']})",
                f"- **Overall Decision Latency**: p50={_fmt_float(lat_all['p50_ms'], digits=4)} ms | "
                f"p95={_fmt_float(lat_all['p95_ms'], digits=4)} ms | mean={_fmt_float(lat_all['mean_ms'], digits=4)} ms "
                f"(n={lat_all['decisions']})",
            ]
        )

    lines.append("")
    return "\n".join(lines)


def write_report_files(
    report: dict[str, Any],
    out_dir: Path | str,
    *,
    omit_timing: bool = False,
) -> dict[str, Path]:
    """Write `report.json`, `report_deterministic.json`, and `report.md` to `out_dir`."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    det_path = target / "report_deterministic.json"
    det_path.write_text(
        json.dumps(report["deterministic"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    full_payload: dict[str, Any] = {
        "schema_version": report["schema_version"],
        "deterministic": report["deterministic"],
    }
    if not omit_timing and "runtime" in report:
        full_payload["runtime"] = report["runtime"]

    json_path = target / "report.json"
    json_path.write_text(
        json.dumps(full_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    md_path = target / "report.md"
    md_path.write_text(
        format_markdown_report(report, omit_timing=omit_timing),
        encoding="utf-8",
    )

    return {
        "json": json_path,
        "deterministic_json": det_path,
        "markdown": md_path,
    }
