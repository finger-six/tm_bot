"""Pre-self-play gate for the v2 warm starts (#47; thresholds from #38 §5): checks and report.

Everything here is a pure function of results that already exist: the #46 validation metrics
(`metrics.json`), the frequency baselines, and one `summarize_evaluation` report per live pairing.
`tm_bot.training.run_warmstart_gate` produces those (metric loading, baselines, live games) and
calls `evaluate_gate` and `format_gate_report`, so every check is testable on synthetic results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from tm_bot.agents.protocol import GameRunResult
from tm_bot.env.types import SUPPORTED_MAPS
from tm_bot.eval.pairing import PairedGameMeta
from tm_bot.eval.report import summarize_evaluation
from tm_bot.eval.seeds import SeedKey, generate_seed_keys

GATE_SCHEMA_VERSION = "warmstart-gate-v1"

# --- #38 §5.1: held-out validation (val split) -------------------------------------------------

# (gate task, dotted key in the #46 `metrics.json` "val" section, label)
POLICY_TASKS: tuple[tuple[str, str, str], ...] = (
    ("corp", "corp", "Corporation"),
    ("prelude", "prelude_top1", "Prelude (step-0)"),
    ("draft", "draft", "Draft pick"),
    ("research_buy", "research_buy.step0", "Research buy (step-0)"),
    ("branch", "branch", "Root branch"),
    ("project_card", "project_play", "Project card play"),
    ("standard_project", "standard_project", "Standard project"),
    ("card_action", "card_action", "Card action"),
    ("space", "space", "Tile placement (space)"),  # #62 (decision 4 of #59)
)
# v1 warm-start val top-1 (docs/warmstart.md, quoted in #38 §5.1) and the allowed regression.
V1_VAL_TOP1: dict[str, float] = {"corp": 0.725, "prelude": 0.813, "draft": 0.702}
V1_TOLERANCE = 0.015
VALUE_MIN_ACCURACY = 0.72
VALUE_MAX_BRIER = 0.19
VALUE_MAX_ECE = 0.04
MONOTONE_BUCKETS: tuple[str, ...] = ("gen 2-5", "gen 6-9", "gen 10+")
# Both agents' value checks score the val-fitted temperature's report (#58 amendment, #60).
VALUE_LABEL = "post-temperature"

# --- #38 §5.2: live dev-* behavioral gate --------------------------------------------------


@dataclass(frozen=True)
class PairingSpec:
    """One live pairing of the gate: the candidate (seat A) against `opponent`."""

    key: str
    label: str
    opponent: str  # "b0" | "self" | "v1" | "b1"
    num_seeds: int
    candidate: str = "gated"  # "gated" | "raw" (main agent's no-search policy, diagnostic)
    gating: bool = True

    @property
    def expected_games(self) -> int:
        return self.num_seeds * len(SUPPORTED_MAPS) * 2


PAIRINGS: tuple[PairingSpec, ...] = (
    PairingSpec("vs_b0", "vs b0-v1", "b0", 13),
    PairingSpec("mirror", "self-mirror", "self", 5),
    PairingSpec("vs_v1", "vs archived v1 warm start", "v1", 13),
    PairingSpec("vs_b1", "vs b1-v2", "b1", 13),
    PairingSpec("vs_b1_raw", "raw policy vs b1-v2 (diagnostic)", "b1", 13, candidate="raw", gating=False),
)
PAIRINGS_BY_KEY: dict[str, PairingSpec] = {p.key: p for p in PAIRINGS}

B0_MIN_WIN_RATE = 0.90
B0_MIN_CI_LOW = 0.80  # strict: the CI lower bound must exceed it
B0_MIN_MARGIN = 35.0
MIRROR_MAX_MEDIAN_GENS = 16.0
MIRROR_MAX_P95_GENS = 22.0
V1_MIN_WIN_RATE = 0.85
B1_MIN_OWN_VP = 75.0
B1_MIN_MARGIN = -35.0

FAILURE_KINDS: tuple[tuple[str, str], ...] = (
    ("illegal_response", "illegal responses"),
    ("adapter_error", "adapter errors"),
    ("step_cap", "step-cap truncations"),
)


def pairings_for(agent: str) -> tuple[PairingSpec, ...]:
    """The main agent adds the raw-policy diagnostic; the comparator plays standalone only."""
    if agent not in ("main", "comparator"):
        raise ValueError(f"agent must be 'main' or 'comparator', got {agent!r}")
    return tuple(p for p in PAIRINGS if agent == "main" or p.candidate != "raw")


def gate_seed_keys(num_seeds: int, *, start: int = 0) -> tuple[SeedKey, ...]:
    """`num_seeds` consecutive `dev-*` seed keys; the gate never touches `eval-*`."""
    keys = generate_seed_keys("dev", num_seeds, start=start)
    if any(k.namespace != "dev" for k in keys):
        raise AssertionError("gate seeds must be dev-*")
    return keys


def classify_failure(error_type: str | None) -> str:
    """Runner failure -> gate failure kind (`_validate_candidate_index` raises `IndexError`)."""
    if error_type == "StepCapError":
        return "step_cap"
    if error_type == "IndexError":
        return "illegal_response"
    return "adapter_error"


# --- results in -------------------------------------------------------------------------------


def live_value_metrics(entries: Sequence[tuple[int, float, float]]) -> dict[str, Any]:
    """Brier / ECE / accuracy of `(generation, predicted win prob, outcome utility)` entries."""
    from tm_bot.comparator.calibration import calibration_metrics

    def metrics(rows: Sequence[tuple[int, float, float]]) -> dict[str, Any]:
        m = calibration_metrics([r[1] for r in rows], [r[2] for r in rows])
        m.pop("bins", None)
        return m

    return {"all": metrics(entries), "gen_2_plus": metrics([e for e in entries if e[0] >= 2])}


def summarize_pairing(
    spec: PairingSpec,
    outcomes: Sequence[GameRunResult],
    metas: Sequence[PairedGameMeta],
    *,
    candidate_name: str,
    opponent_name: str,
    value_entries: Sequence[tuple[int, float, float]] | None = None,
    wall_time_s: float = 0.0,
) -> dict[str, Any]:
    """One pairing's gate summary: the eval harness report plus generation percentiles and failures."""
    report = summarize_evaluation(
        outcomes=outcomes,
        metas=metas,
        namespace="dev",
        agent_a_name=candidate_name,
        agent_b_name=opponent_name,
        wall_time_s=wall_time_s,
    )
    det = report["deterministic"]
    gens = [int(o.result.generations) for o in outcomes if o.ok and o.result is not None]
    failure_counts = {kind: 0 for kind, _ in FAILURE_KINDS}
    for out in outcomes:
        if not out.ok or out.result is None:
            failure_counts[classify_failure(out.error_type)] += 1
    seed_keys = sorted({m.seed_key for m in metas})
    return {
        "key": spec.key,
        "label": spec.label,
        "gating": spec.gating,
        "expected_games": spec.expected_games,
        "seed_keys": seed_keys,
        "all_dev_seeds": all(SeedKey.parse(k).namespace == "dev" for k in seed_keys),
        "seat_swapped": _is_seat_swapped(metas),
        "report": det,
        "generations": {
            "median": float(np.median(gens)) if gens else None,
            "p95": float(np.percentile(gens, 95)) if gens else None,
            "max": max(gens) if gens else None,
        },
        "failure_counts": failure_counts,
        "live_value": live_value_metrics(value_entries) if value_entries is not None else None,
        "wall_time_s": float(wall_time_s),
    }


def _is_seat_swapped(metas: Sequence[PairedGameMeta]) -> bool:
    seats: dict[tuple[str, str], list[int]] = {}
    for m in metas:
        seats.setdefault((m.seed_key, m.map_name), []).append(int(m.a_seat))
    return bool(seats) and all(sorted(v) == [0, 1] for v in seats.values())


# --- checks -----------------------------------------------------------------------------------


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.2f}%"


def _vp(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.2f}"


def _num(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def _prob(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.4f}"


_OPS: dict[str, Callable[[float, float], bool]] = {
    ">=": lambda a, b: a >= b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
}


@dataclass(frozen=True)
class Check:
    """One gate check: the measured value against its threshold."""

    group: str
    name: str
    measured: Any
    threshold: str
    passed: bool
    measured_text: str
    gating: bool = True
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def threshold_check(
    group: str,
    name: str,
    measured: float | None,
    op: str,
    bound: float,
    fmt: Callable[[float | None], str],
    *,
    note: str = "",
) -> Check:
    """`measured op bound`; a missing measurement fails."""
    passed = measured is not None and bool(_OPS[op](float(measured), float(bound)))
    return Check(
        group=group,
        name=name,
        measured=measured,
        threshold=f"{op} {fmt(bound)}",
        passed=passed,
        measured_text=fmt(measured),
        note=note if measured is not None else (note or "missing"),
    )


def _lookup(tree: Mapping[str, Any] | None, dotted: str) -> Any:
    out: Any = tree
    for part in dotted.split("."):
        if not isinstance(out, Mapping) or part not in out:
            return None
        out = out[part]
    return out


def _overall_acc(entry: Any) -> float | None:
    acc = _lookup(entry, "overall.acc")
    total = _lookup(entry, "overall.total")
    return None if acc is None or not total else float(acc)


def validation_checks(
    val: Mapping[str, Any] | None,
    baselines: Mapping[str, Any] | None,
    value: Mapping[str, Any] | None,
    *,
    value_label: str = "raw",
) -> list[Check]:
    """#38 §5.1 on the #46 val metrics.

    `val` is `metrics.json["val"]`; `baselines` maps each gate task to a frequency-baseline entry
    (`{"overall": {"acc", "total"}, "per_map": ...}`); `value` is the value report the gate judges:
    the post-temperature val report for both agents (#58 amendment of check 14).
    """
    group = "validation"
    checks: list[Check] = []
    for task, key, label in POLICY_TASKS:
        model = _overall_acc(_lookup(val, key))
        base = _overall_acc((baselines or {}).get(task))
        if model is None or base is None:
            checks.append(
                Check(group, f"{label}: top-1 beats frequency baseline", model, "> baseline", False,
                      _pct(model), note="missing " + ("model metric" if model is None else "baseline"))
            )
            continue
        checks.append(
            Check(group, f"{label}: top-1 beats frequency baseline", model, f"> {_pct(base)}",
                  model > base, _pct(model), note=f"baseline {_pct(base)}, delta {(model - base) * 100:+.2f} pp")
        )
    for task, ref in V1_VAL_TOP1.items():
        key, label = next((k, lab) for t, k, lab in POLICY_TASKS if t == task)
        bound = round(ref - V1_TOLERANCE, 10)
        checks.append(
            threshold_check(group, f"{label}: within -1.5 pp of v1 ({_pct(ref)})",
                            _overall_acc(_lookup(val, key)), ">=", bound, _pct)
        )
    mid = _lookup(value, "midgame")
    checks.append(threshold_check(group, f"Gen >= 2 value accuracy ({value_label})",
                                  _lookup(mid, "accuracy"), ">=", VALUE_MIN_ACCURACY, _pct))
    checks.append(threshold_check(group, f"Gen >= 2 value Brier ({value_label})",
                                  _lookup(mid, "brier"), "<=", VALUE_MAX_BRIER, _prob))
    checks.append(threshold_check(group, f"Gen >= 2 value ECE ({value_label})",
                                  _lookup(mid, "ece"), "<=", VALUE_MAX_ECE, _prob))
    checks.append(monotone_check(group, value, value_label=value_label))
    return checks


def monotone_check(group: str, value: Mapping[str, Any] | None, *, value_label: str = "raw") -> Check:
    """Value accuracy strictly increases over gen 2-5 -> 6-9 -> 10+."""
    accs = [_lookup(value, f"per_gen.{b}.accuracy") for b in MONOTONE_BUCKETS]
    name = f"Value accuracy increases over {' -> '.join(MONOTONE_BUCKETS)} ({value_label})"
    if any(a is None for a in accs):
        return Check(group, name, accs, "strictly increasing", False, "n/a", note="missing bucket")
    passed = all(b > a for a, b in zip(accs, accs[1:]))
    return Check(group, name, accs, "strictly increasing", passed, " -> ".join(_pct(a) for a in accs))


def _det(summary: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    return None if summary is None else summary["report"]


def schedule_check(spec: PairingSpec, summary: Mapping[str, Any] | None) -> Check:
    """The pairing ran the full #38 schedule: all games, seat-swapped, `dev-*` seeds only."""
    name = f"{spec.label}: {spec.expected_games} seat-swapped dev-* games ({spec.num_seeds} seeds x {len(SUPPORTED_MAPS)} maps x 2 seats)"
    if summary is None:
        return Check("schedule", name, None, f"== {spec.expected_games}", False, "not run",
                     gating=spec.gating, note="pairing not run")
    games = int(summary["report"]["total_games"])
    ok = games == spec.expected_games and bool(summary["all_dev_seeds"]) and bool(summary["seat_swapped"])
    note = []
    if not summary["all_dev_seeds"]:
        note.append("non-dev seeds")
    if not summary["seat_swapped"]:
        note.append("not seat-swapped")
    return Check("schedule", name, games, f"== {spec.expected_games}", ok, str(games),
                 gating=spec.gating, note=", ".join(note))


def failure_checks(summaries: Mapping[str, Mapping[str, Any] | None]) -> list[Check]:
    """0 illegal responses, 0 adapter errors and 0 step-cap truncations across every pairing played."""
    played = [s for s in summaries.values() if s is not None]
    checks = []
    for kind, label in FAILURE_KINDS:
        total = sum(int(s["failure_counts"][kind]) for s in played)
        per = ", ".join(f"{s['key']}={s['failure_counts'][kind]}" for s in played if s["failure_counts"][kind])
        checks.append(
            Check("failures", f"0 {label} across all pairings", total, "== 0", bool(played) and total == 0,
                  str(total), note=per if played else "no pairings played")
        )
    return checks


def live_checks(summaries: Mapping[str, Mapping[str, Any] | None]) -> list[Check]:
    """#38 §5.2 behavioral thresholds (the zero-failure and schedule checks are separate)."""
    b0, mirror, v1, b1 = (_det(summaries.get(k)) for k in ("vs_b0", "mirror", "vs_v1", "vs_b1"))
    ci = _lookup(b0, "win_rate_ci_95")
    mg = None if summaries.get("mirror") is None else summaries["mirror"]["generations"]
    return [
        threshold_check("vs_b0", "Win rate vs b0-v1", _lookup(b0, "win_rate"), ">=", B0_MIN_WIN_RATE, _pct),
        threshold_check("vs_b0", "Win rate 95% CI lower bound vs b0-v1", None if ci is None else ci[0],
                        ">", B0_MIN_CI_LOW, _pct, note="paired bootstrap over seeds"),
        threshold_check("vs_b0", "Mean VP margin vs b0-v1", _lookup(b0, "mean_vp_margin"), ">=", B0_MIN_MARGIN, _vp),
        threshold_check("mirror", "Self-mirror median game length (generations)",
                        None if mg is None else mg["median"], "<=", MIRROR_MAX_MEDIAN_GENS, _num),
        threshold_check("mirror", "Self-mirror p95 game length (generations)",
                        None if mg is None else mg["p95"], "<=", MIRROR_MAX_P95_GENS, _num),
        threshold_check("vs_v1", "Win rate vs archived v1 warm start", _lookup(v1, "win_rate"), ">=",
                        V1_MIN_WIN_RATE, _pct),
        threshold_check("vs_b1", "Mean own VP vs b1-v2", _lookup(b1, "mean_vp_a"), ">=", B1_MIN_OWN_VP,
                        lambda x: _num(x, 2)),
        threshold_check("vs_b1", "Mean VP margin vs b1-v2", _lookup(b1, "mean_vp_margin"), ">=", B1_MIN_MARGIN, _vp),
    ]


def search_vs_raw(summaries: Mapping[str, Mapping[str, Any] | None]) -> dict[str, str] | None:
    """Gated light search minus the raw policy vs b1-v2 (#58): own VP and VP margin."""
    search, raw = _det(summaries.get("vs_b1")), _det(summaries.get("vs_b1_raw"))
    if search is None or raw is None:
        return None
    vals = [search["mean_vp_a"], raw["mean_vp_a"], search["mean_vp_margin"], raw["mean_vp_margin"]]
    if any(v is None for v in vals):
        return None
    s_vp, r_vp, s_m, r_m = (float(v) for v in vals)
    return {
        "pairing": "search vs raw policy (vs b1-v2)",
        "name": "Own VP / VP margin, search minus raw",
        "value": f"{_vp(s_vp - r_vp)} / {_vp(s_m - r_m)} (search {_num(s_vp, 2)} / {_vp(s_m)}, "
                 f"raw {_num(r_vp, 2)} / {_vp(r_m)})",
    }


def diagnostics(summaries: Mapping[str, Mapping[str, Any] | None]) -> list[dict[str, str]]:
    """Non-gating numbers #38 §5.2 and #58 ask to log: win rate vs b1-v2, the main agent's
    raw-policy (no-search) result vs b1-v2 and search minus raw, and live value calibration
    (reported, post-temperature P(win)) per opponent: b1-v2, self-mirror, v1 warm start."""
    rows: list[dict[str, str]] = []
    for key in ("vs_b1", "vs_b1_raw", "mirror", "vs_v1"):
        s = summaries.get(key)
        if s is None:
            continue
        det = s["report"]
        ci = det["win_rate_ci_95"]
        if key in ("vs_b1", "vs_b1_raw"):
            rows.append({
                "pairing": s["label"],
                "name": "Win rate",
                "value": f"{_pct(det['win_rate'])}" + (f" [{_pct(ci[0])}, {_pct(ci[1])}]" if ci else ""),
            })
        if key == "vs_b1_raw":
            rows.append({"pairing": s["label"], "name": "Mean own VP / VP margin",
                         "value": f"{_num(det['mean_vp_a'], 2)} / {_vp(det['mean_vp_margin'])}"})
            delta = search_vs_raw(summaries)
            if delta is not None:
                rows.append(delta)
        lv = s.get("live_value")
        if lv:
            for scope, label in (("gen_2_plus", "gen >= 2"), ("all", "all gens")):
                m = lv[scope]
                rows.append({
                    "pairing": s["label"],
                    "name": f"Live value Brier / ECE / accuracy ({label})",
                    "value": f"{_prob(m['brier'])} / {_prob(m['ece'])} / {_pct(m['accuracy'])} (n={m['n']:,})",
                })
    return rows


def evaluate_gate(
    *,
    agent: str,
    val: Mapping[str, Any] | None,
    baselines: Mapping[str, Any] | None,
    value: Mapping[str, Any] | None,
    summaries: Mapping[str, Mapping[str, Any] | None],
) -> dict[str, Any]:
    """All checks, diagnostics and the overall verdict (PASS only if every gating check passes).

    `value` is the post-temperature val value report for both agents (#58 amendment of check 14).
    """
    specs = pairings_for(agent)
    checks: list[Check] = validation_checks(val, baselines, value, value_label=VALUE_LABEL)
    checks += failure_checks({s.key: summaries.get(s.key) for s in specs})
    checks += live_checks(summaries)
    checks += [schedule_check(s, summaries.get(s.key)) for s in specs]
    gating = [c for c in checks if c.gating]
    passed = sum(c.passed for c in gating)
    return {
        "verdict": "PASS" if gating and passed == len(gating) else "FAIL",
        "gating_passed": passed,
        "gating_total": len(gating),
        "checks": [c.to_dict() for c in checks],
        "diagnostics": diagnostics(summaries),
    }


# --- report -----------------------------------------------------------------------------------

_GROUP_LABELS = {
    "validation": "Validation (val split)",
    "failures": "Zero failures",
    "vs_b0": "vs b0-v1",
    "mirror": "Self-mirror",
    "vs_v1": "vs v1 warm start",
    "vs_b1": "vs b1-v2",
    "schedule": "Schedule",
}


def _per_map_table(det: Mapping[str, Any]) -> list[str]:
    lines = [
        "| Map | Games | Failures | Win rate | 95% CI | Mean VP margin | Mean gens |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for map_name in SUPPORTED_MAPS:
        row = det["per_map"].get(map_name)
        if row is None or not row["total_games"]:
            continue
        ci = row["win_rate_ci_95"]
        ci_s = f"[{_pct(ci[0])}, {_pct(ci[1])}]" if ci else "n/a"
        lines.append(
            f"| {map_name.title()} | {row['total_games']} | {row['failures']} | {_pct(row['win_rate'])} | "
            f"{ci_s} | {_vp(row['mean_vp_margin'])} | {_num(row['mean_generations'], 2)} |"
        )
    return lines


def _validation_per_map(val: Mapping[str, Any] | None, baselines: Mapping[str, Any] | None) -> list[str]:
    header = "| Task | Scope | Model top-1 | Frequency baseline | Delta | n |"
    lines = [header, "| --- | --- | ---: | ---: | ---: | ---: |"]
    for task, key, label in POLICY_TASKS:
        model = _lookup(val, key)
        base = (baselines or {}).get(task)
        if model is None:
            lines.append(f"| {label} | overall | n/a | {_pct(_overall_acc(base))} | n/a | n/a |")
            continue
        scopes = [("overall", model.get("overall"), _lookup(base, "overall"))]
        scopes += [(m, _lookup(model, f"per_map.{m}"), _lookup(base, f"per_map.{m}")) for m in SUPPORTED_MAPS]
        for scope, me, be in scopes:
            if not me or not me.get("total"):
                continue
            ma = float(me["acc"])
            ba = None if not be or not be.get("total") else float(be["acc"])
            delta = "n/a" if ba is None else f"{(ma - ba) * 100:+.2f} pp"
            lines.append(f"| {label} | {scope} | {_pct(ma)} | {_pct(ba)} | {delta} | {int(me['total']):,} |")
    return lines


def _value_table(value: Mapping[str, Any] | None) -> list[str]:
    if not value:
        return ["Value report missing."]
    lines = [f"Temperature {value.get('temperature', 1.0):.4f}.", "",
             "| Scope | n | Accuracy | Brier | ECE |", "| --- | ---: | ---: | ---: | ---: |"]
    scopes = [(b, value["per_gen"].get(b)) for b in ("gen 1", *MONOTONE_BUCKETS)]
    scopes += [("gen 2+ (all)", value.get("midgame"))]
    scopes += [(f"gen 2+ {m}", v) for m, v in (value.get("midgame_per_map") or {}).items()]
    for name, m in scopes:
        if m:
            lines.append(f"| {name} | {m['n']:,} | {_pct(m['accuracy'])} | {_prob(m['brier'])} | {_prob(m['ece'])} |")
    return lines


def format_gate_report(result: Mapping[str, Any]) -> str:
    """Markdown gate report: verdict, every check, per-map live breakdowns, validation detail."""
    gate = result["gate"]
    cfg = result.get("config", {})
    lines = [
        f"# Pre-self-play gate: `{result.get('version') or 'unversioned'}` ({result['agent']} agent)",
        "",
        f"**Verdict: {gate['verdict']}** ({gate['gating_passed']}/{gate['gating_total']} gating checks pass). "
        "Thresholds: #38 §5; `val` split and `dev-*` seeds only.",
        "",
    ]
    for label, key in (("Checkpoint", "checkpoint"), ("Validation metrics", "metrics_path"),
                       ("v1 opponent", "v1_checkpoint"), ("Candidate setting", "candidate_setting"),
                       ("v1 opponent setting", "v1_setting"), ("Raw-policy diagnostic", "raw_setting"),
                       ("Value temperature (reported P(win), check 14)", "value_temperature"),
                       ("Max decisions per game", "max_decisions"),
                       ("Workers / batch size", "workers"), ("Code", "git_commit")):
        if cfg.get(key) is not None:
            lines.append(f"- {label}: `{cfg[key]}`")
    lines += ["", "## Checks", "", "| # | Group | Check | Measured | Threshold | Result | Note |",
              "| ---: | --- | --- | ---: | --- | --- | --- |"]
    for i, c in enumerate(gate["checks"], start=1):
        result_s = "PASS" if c["passed"] else "**FAIL**"
        if not c["gating"]:
            result_s += " (diagnostic)"
        lines.append(
            f"| {i} | {_GROUP_LABELS.get(c['group'], c['group'])} | {c['name']} | {c['measured_text']} | "
            f"{c['threshold']} | {result_s} | {c['note']} |"
        )
    if gate["diagnostics"]:
        lines += ["", "## Diagnostics (not gating)", "",
                  "Live value rows score the candidate's reported (post-temperature) P(win) against the "
                  "final result of its games.", "",
                  "| Pairing | Measure | Value |", "| --- | --- | --- |"]
        lines += [f"| {d['pairing']} | {d['name']} | {d['value']} |" for d in gate["diagnostics"]]

    lines += ["", "## Live pairings", ""]
    for key, s in result.get("pairings", {}).items():
        if s is None:
            lines += [f"### {PAIRINGS_BY_KEY[key].label}", "", "Not run.", ""]
            continue
        det = s["report"]
        ci = det["win_rate_ci_95"]
        g = s["generations"]
        lines += [
            f"### {s['label']}",
            "",
            f"- Games: {det['total_games']} ({det['completed_games']} completed, {det['failure_count']} failures: "
            + ", ".join(f"{k}={v}" for k, v in s["failure_counts"].items()) + ")",
            f"- Record: {det['wins']}W-{det['losses']}L-{det['ties']}T; win rate {_pct(det['win_rate'])}"
            + (f" [{_pct(ci[0])}, {_pct(ci[1])}]" if ci else ""),
            f"- Mean VP: {_num(det['mean_vp_a'], 2)} vs {_num(det['mean_vp_b'], 2)} (margin {_vp(det['mean_vp_margin'])})",
            f"- Generations: median {_num(g['median'])}, p95 {_num(g['p95'])}, max {g['max'] if g['max'] is not None else 'n/a'}",
            f"- Wall time: {s['wall_time_s']:.0f} s",
            "",
            *_per_map_table(det),
            "",
        ]
        for f in det["failures"][:10]:
            lines.append(f"- failure `{f['seed_key']}` `{f['map']}` a_seat={f['a_seat']} `{f['error_type']}`: {f['error']}")
        if det["failures"]:
            lines.append("")

    lines += ["## Validation detail (val split)", "", *_validation_per_map(result.get("val"), result.get("baselines")), ""]
    lines += [f"### Value head ({VALUE_LABEL})", "",
              *_value_table(result.get("value")), ""]
    return "\n".join(lines)
