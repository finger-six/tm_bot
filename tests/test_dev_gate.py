"""1 h development gate (#52): the seven checks on synthetic run summaries, and gate mode."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from tm_bot.runs.config import (
    DEV_GATE_BUDGET_S,
    DEV_GATE_DEV_EVAL_INTERVAL_S,
    RunConfig,
    dev_gate_config,
)
from tm_bot.runs.dev_gate import (
    DevEvalPoint,
    GateSummary,
    check_cost,
    check_game_shape,
    check_legality,
    check_positive_signal,
    check_resources,
    check_symmetry,
    check_throughput,
    evaluate_gate,
    render_markdown,
)
from tm_bot.search.config import FALLBACK_NUM_SIMULATIONS, V2_NUM_SIMULATIONS, V2_NUM_WORLDS


def _passing(**overrides: Any) -> GateSummary:
    """A gate run that passes every check, several of them exactly at the boundary."""
    data: dict[str, Any] = dict(
        run_name="gate-main",
        agent_kind="main",
        num_simulations=16,
        num_worlds=4,
        budget_s=3600.0,
        learner_ram_gb=20.0,
        selfplay_games=250,
        training_s=3600.0,
        dev_eval_s=0.40 * 3600.0,
        # median 14, p95 25, max 40: every game-shape bound is met exactly.
        generation_counts={14: 90, 25: 9, 40: 1},
        results_by_opponent={
            "pool": {"wins": 35, "losses": 65, "ties": 0},
            "anchor": {"wins": 65, "losses": 35, "ties": 0},
            "latest": {"wins": 99, "losses": 1, "ties": 0},
        },
        dev_evals=(
            DevEvalPoint(training_s=1800.0, anchor_win_rate=0.2, dev_score=-0.5),
            DevEvalPoint(training_s=3600.0, anchor_win_rate=0.45, dev_score=-0.30),
        ),
        anchor_dev_score=-0.30,
        stop_kind="budget",
        rss_gb_max=20.0,
    )
    data.update(overrides)
    return GateSummary(**data)


def test_a_passing_run_passes_every_check_and_reports_pass() -> None:
    report = evaluate_gate(_passing())
    assert [c.number for c in report.checks] == [1, 2, 3, 4, 5, 6, 7]
    assert all(c.passed for c in report.checks), [c for c in report.checks if not c.passed]
    assert report.passed
    text = render_markdown(report)
    assert "**PASS**" in text and "**FAIL**" not in text
    assert text.count("| PASS |") == 7


# --- 1. legality ----------------------------------------------------------------
@pytest.mark.parametrize("field", ["failures", "adapter_rejections", "step_cap_hits"])
def test_legality_fails_on_any_failure_rejection_or_step_cap(field: str) -> None:
    assert check_legality(_passing()).passed
    assert not check_legality(_passing(**{field: 1})).passed


# --- 2. game shape -------------------------------------------------------------------
@pytest.mark.parametrize(
    ("counts", "passed"),
    [
        ({14: 90, 25: 9, 40: 1}, True),  # median 14, p95 25, max 40
        ({18: 60, 20: 40}, True),  # median exactly 18
        ({19: 60, 20: 40}, False),  # median 19 > 18
        ({14: 90, 26: 9, 40: 1}, False),  # p95 26 > 25
        ({14: 90, 25: 9, 41: 1}, False),  # one game > 40
        ({}, False),  # no completed games
    ],
)
def test_game_shape_bounds(counts: dict[int, int], passed: bool) -> None:
    check = check_game_shape(_passing(generation_counts=counts))
    assert check.passed is passed, check.measured


def test_game_shape_reports_median_p95_and_max() -> None:
    check = check_game_shape(_passing())
    assert check.details == {"median": 14.0, "p95": 25.0, "max": 40}


# --- 3. throughput -------------------------------------------------------------------
@pytest.mark.parametrize(
    ("agent_kind", "games", "passed"),
    [
        ("main", 250, True),
        ("main", 249, False),
        ("comparator", 800, True),
        ("comparator", 799, False),
    ],
)
def test_throughput_per_training_hour(agent_kind: str, games: int, passed: bool) -> None:
    check = check_throughput(_passing(agent_kind=agent_kind, selfplay_games=games))
    assert check.passed is passed
    assert check.details["games_per_training_hour"] == pytest.approx(games)


def test_throughput_uses_the_training_clock_not_wall_clock() -> None:
    # 250 games in 1.1 training hours is below 250/h however long dev eval took.
    check = check_throughput(_passing(training_s=1.1 * 3600.0, dev_eval_s=0.0))
    assert not check.passed


def test_main_throughput_failure_points_at_the_8_4_fallback() -> None:
    text = render_markdown(evaluate_gate(_passing(selfplay_games=200)))
    assert "--search-fallback" in text


# --- 4. symmetry ----------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["pool", "anchor"])
@pytest.mark.parametrize(
    ("wins", "losses", "ties", "passed"),
    [
        (35, 65, 0, True),
        (34, 66, 0, False),
        (65, 35, 0, True),
        (66, 34, 0, False),
        (30, 60, 10, True),  # ties count half: 35%
        (0, 0, 0, False),  # no games against that opponent
    ],
)
def test_symmetry_bounds(kind: str, wins: int, losses: int, ties: int, passed: bool) -> None:
    results = {
        "pool": {"wins": 50, "losses": 50, "ties": 0},
        "anchor": {"wins": 50, "losses": 50, "ties": 0},
        kind: {"wins": wins, "losses": losses, "ties": ties},
    }
    assert check_symmetry(_passing(results_by_opponent=results)).passed is passed


def test_symmetry_fails_without_any_anchor_games() -> None:
    results = {"pool": {"wins": 50, "losses": 50, "ties": 0}}
    assert not check_symmetry(_passing(results_by_opponent=results)).passed


def test_symmetry_ignores_latest_and_b1() -> None:
    results = {
        "pool": {"wins": 50, "losses": 50, "ties": 0},
        "anchor": {"wins": 50, "losses": 50, "ties": 0},
        "b1": {"wins": 0, "losses": 100, "ties": 0},
    }
    assert check_symmetry(_passing(results_by_opponent=results)).passed


# --- 5. cost ----------------------------------------------------------------------------
@pytest.mark.parametrize(("dev_eval_s", "passed"), [(1440.0, True), (1441.0, False)])
def test_cost_at_most_40_percent_of_training_clock(dev_eval_s: float, passed: bool) -> None:
    check = check_cost(_passing(dev_eval_s=dev_eval_s, dev_eval_baseline_s=600.0))
    assert check.passed is passed
    assert check.details["dev_eval_fraction_without_baseline"] == pytest.approx(
        (dev_eval_s - 600.0) / 3600.0
    )


# --- 6. positive signal ---------------------------------------------------------------------
def _final(**overrides: Any) -> tuple[DevEvalPoint, ...]:
    data: dict[str, Any] = dict(training_s=3600.0, anchor_win_rate=0.45, dev_score=-0.30)
    data.update(overrides)
    return (DevEvalPoint(1800.0, 0.5, -0.4), DevEvalPoint(**data))


@pytest.mark.parametrize(
    ("overrides", "passed"),
    [
        (dict(dev_evals=_final()), True),
        (dict(dev_evals=_final(anchor_win_rate=0.4499)), False),
        (dict(dev_evals=_final(anchor_win_rate=None)), False),
        (dict(dev_evals=_final(dev_score=-0.3001)), False),  # D below the anchor's D
        (dict(dev_evals=_final(dev_score=-0.10)), True),
        (dict(anchor_dev_score=None), False),  # the anchor's D was never measured
        (dict(stop_kind="plateau", stop_reason="no new best"), False),
        (dict(stop_kind="regression_anchor"), False),
        (dict(stop_kind=None), False),
        (dict(dev_evals=_final()[1:]), False),  # no 30-min evaluation
        (dict(dev_evals=_final(training_s=3000.0)), False),  # no 60-min evaluation
        (dict(dev_evals=()), False),
    ],
)
def test_positive_signal_at_60_min(overrides: dict[str, Any], passed: bool) -> None:
    check = check_positive_signal(_passing(**overrides))
    assert check.passed is passed, check.measured


def test_positive_signal_reads_the_last_evaluation_not_the_first() -> None:
    # A good 30-min result does not rescue a regressed 60-min one.
    evals = (DevEvalPoint(1800.0, 0.9, 0.5), DevEvalPoint(3600.0, 0.3, -0.5))
    assert not check_positive_signal(_passing(dev_evals=evals)).passed


# --- 7. resources ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("overrides", "passed"),
    [
        (dict(rss_gb_max=20.0), True),  # the guard stops only above the limit
        (dict(rss_gb_max=20.01), False),
        (dict(rss_gb_max=None), False),  # no readings: the guard was inert (#35)
        (dict(rss_gb_max=5.0, stop_kind="resource_limit", stop_reason="learner RAM"), False),
    ],
)
def test_resources_under_the_20_gib_guard(overrides: dict[str, Any], passed: bool) -> None:
    assert check_resources(_passing(**overrides)).passed is passed


def test_one_failing_check_fails_the_gate_and_is_marked() -> None:
    report = evaluate_gate(_passing(rss_gb_max=25.0))
    assert not report.passed
    assert [c.number for c in report.checks if not c.passed] == [7]
    assert "**FAIL**" in render_markdown(report)
    payload = report.to_dict()
    assert payload["passed"] is False
    assert payload["summary"]["generation_counts"] == {"14": 90, "25": 9, "40": 1}


# --- gate mode ----------------------------------------------------------------------------------
def test_gate_config_sets_the_1h_budget_and_30_min_dev_cadence(tmp_path: Path) -> None:
    base = RunConfig(run_name="gate", output_root=tmp_path)
    # The 24 h runs keep the 24 h budget and the 60-min dev cadence.
    assert base.wall_clock_budget_s == 24 * 3600.0
    assert base.dev_eval_interval_s == 60 * 60.0
    assert base.dev_gate is False

    gate = dev_gate_config(base)
    assert gate.wall_clock_budget_s == DEV_GATE_BUDGET_S == 3600.0
    assert gate.dev_eval_interval_s == DEV_GATE_DEV_EVAL_INTERVAL_S == 30 * 60.0
    assert gate.dev_gate is True
    # Everything else, search settings included, comes from the run config unchanged.
    assert gate.checkpoint_interval_s == base.checkpoint_interval_s == 30 * 60.0
    assert gate.search_config() == base.search_config()
    assert RunConfig.from_dict(gate.to_dict()) == gate


def test_cli_gate_flag_overrides_budget_and_cadence(tmp_path: Path) -> None:
    from tm_bot.runs.__main__ import parse_config

    default, _ = parse_config(["--run-name", "r", "--out-root", str(tmp_path)])
    assert default.wall_clock_budget_s == 24 * 3600.0
    assert default.dev_eval_interval_s == 3600.0
    assert default.dev_gate is False
    assert (default.num_simulations, default.num_worlds) == (V2_NUM_SIMULATIONS, V2_NUM_WORLDS)

    gate, _ = parse_config(
        ["--run-name", "g", "--out-root", str(tmp_path), "--gate-1h", "--budget-hours", "24"]
    )
    assert gate.wall_clock_budget_s == 3600.0
    assert gate.dev_eval_interval_s == 1800.0
    assert gate.dev_gate is True


def test_cli_search_fallback_switches_the_main_gate_to_8_by_4(tmp_path: Path) -> None:
    from tm_bot.runs.__main__ import parse_config

    argv = ["--run-name", "g", "--out-root", str(tmp_path), "--gate-1h"]
    full, _ = parse_config(argv)
    fallback, _ = parse_config([*argv, "--search-fallback"])
    assert full.search_config().num_simulations == V2_NUM_SIMULATIONS == 16
    assert fallback.search_config().num_simulations == FALLBACK_NUM_SIMULATIONS == 8
    assert fallback.search_config().num_worlds == full.search_config().num_worlds == 4
    assert replace(fallback, num_simulations=16, search_fallback=False) == full
