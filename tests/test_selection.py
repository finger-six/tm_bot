"""Floor-aware checkpoint selection, v2 stop rules, and the dev-eval cost cap (#51, #40 sections 1-2)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tm_bot.eval.stats import cluster_bootstrap_mean_ci, paired_bootstrap_ci
from tm_bot.runs.monitoring import CheckpointMetrics
from tm_bot.runs.deveval import (
    DevGame,
    dev_eval_interval,
    dev_game_records,
    dev_score,
    dev_win_rate,
    games_on_seeds,
    paired_delta_d,
)
from tm_bot.runs.stopping import (
    ROUTE_DELTA_D,
    ROUTE_HEAD_TO_HEAD,
    ROUTE_NO_INCUMBENT,
    StopDecision,
    StopKind,
    StoppingMonitor,
    is_new_best,
    new_best_route,
)


def _game(seed: str, map_name: str, seat: int, outcome: int | None, margin: int | None) -> DevGame:
    return DevGame(seed, map_name, seat, ok=outcome is not None, outcome=outcome, vp_margin=margin)


def _out(ok: bool, scores=(0, 0), winner=None, tie=False, generations=14):
    result = None if not ok else SimpleNamespace(
        scores=scores, winner=winner, tie=tie, generations=generations
    )
    return SimpleNamespace(ok=ok, result=result)


# --- per-game records, D, paired dD -------------------------------------
def test_dev_game_records_keep_each_game_from_the_candidate_side() -> None:
    metas = [
        SimpleNamespace(seed_key="dev-000000", map_name="tharsis", a_seat=0),
        SimpleNamespace(seed_key="dev-000000", map_name="tharsis", a_seat=1),
        SimpleNamespace(seed_key="dev-000001", map_name="hellas", a_seat=0),
    ]
    outcomes = [
        _out(True, scores=(70, 80), winner=1),  # candidate seat 0 loses by 10
        _out(True, scores=(90, 75), winner=0),  # candidate seat 1 loses by 15
        _out(False),
    ]
    records = dev_game_records(outcomes, metas)  # type: ignore[arg-type]
    assert records == (
        DevGame("dev-000000", "tharsis", 0, ok=True, outcome=-1, vp_margin=-10, generations=14),
        DevGame("dev-000000", "tharsis", 1, ok=True, outcome=-1, vp_margin=-15, generations=14),
        DevGame("dev-000001", "hellas", 0, ok=False),
    )
    assert [r.key for r in records][:2] == [
        ("dev-000000", "tharsis", 0),
        ("dev-000000", "tharsis", 1),
    ]
    assert DevGame.from_dict(records[0].to_dict()) == records[0]
    assert DevGame.from_dict(records[2].to_dict()) == records[2]


def test_dev_score_is_mean_outcome_plus_clipped_margin_term() -> None:
    games = [
        _game("dev-000000", "tharsis", 0, 1, 10),  # 1 + 0.15 * 10/50 = 1.03
        _game("dev-000000", "tharsis", 1, -1, -80),  # clipped to -60: -1 - 0.18 = -1.18
        _game("dev-000001", "tharsis", 0, 0, 0),  # tie: 0
        _game("dev-000001", "tharsis", 1, None, None),  # failed: excluded
    ]
    assert games[0].utility == pytest.approx(1.03)
    assert games[1].utility == pytest.approx(-1.18)
    assert dev_score(games) == pytest.approx((1.03 - 1.18 + 0.0) / 3)
    assert dev_win_rate(games) == pytest.approx((1.0 + 0.0 + 0.5) / 3)
    assert dev_score([]) is None
    # At the floor (all losses) D is a margin measure: -1 + 0.15 * mean(m).
    floor = [_game("dev-000000", "tharsis", s, -1, -30) for s in (0, 1)]
    assert dev_score(floor) == pytest.approx(-1.0 + 0.15 * (-30 / 50))


def test_paired_delta_d_pairs_by_seed_map_and_seat() -> None:
    seeds = [f"dev-{i:06d}" for i in range(4)]
    incumbent = [_game(s, "tharsis", seat, -1, -40) for s in seeds for seat in (0, 1)]
    candidate = [_game(s, "tharsis", seat, -1, -30) for s in seeds for seat in (0, 1)]
    delta, ci = paired_delta_d(candidate, incumbent)
    # Every pair differs by 0.15 * 10/50 = 0.03, so the CI collapses onto it.
    assert delta == pytest.approx(0.03)
    assert ci == pytest.approx((0.03, 0.03))

    # Unpaired or unfinished games drop out of the comparison.
    extra = [_game("dev-000099", "tharsis", 0, 1, 50)]
    failed = [_game(seeds[0], "tharsis", 0, None, None)]
    delta2, _ = paired_delta_d(candidate[1:] + failed + extra, incumbent)
    assert delta2 == pytest.approx(0.03)

    # A win against the incumbent's loss on one pair: hand-computed mean.
    mixed = list(candidate)
    mixed[0] = _game(seeds[0], "tharsis", 0, 1, 5)  # 1.015 vs -1.12
    delta3, ci3 = paired_delta_d(mixed, incumbent)
    expected = (7 * 0.03 + (1.015 - (-1.12))) / 8
    assert delta3 == pytest.approx(expected)
    assert ci3 is not None and ci3[0] <= expected <= ci3[1]
    assert paired_delta_d([], incumbent) == (None, None)


def test_games_on_seeds_restricts_to_the_anchor_schedule() -> None:
    games = [_game(f"dev-{i:06d}", "tharsis", 0, 1, 1) for i in range(13)]
    subset = games_on_seeds(games, [f"dev-{i:06d}" for i in range(7)])
    assert [g.seed_key for g in subset] == [f"dev-{i:06d}" for i in range(7)]


def test_cluster_bootstrap_is_unclipped_and_paired_bootstrap_unchanged() -> None:
    clusters = [(-2.0, 2), (-1.0, 2), (-3.0, 2)]
    mean, ci = cluster_bootstrap_mean_ci(clusters)
    assert mean == pytest.approx(-1.0)
    assert ci is not None and ci[0] < 0.0 and ci[1] < 0.0
    wr, wr_ci = paired_bootstrap_ci([(1.0, 2), (0.0, 2), (2.0, 2)])
    assert wr == pytest.approx(0.5)
    assert wr_ci is not None and 0.0 <= wr_ci[0] <= wr_ci[1] <= 1.0


# --- cost cap -----------------------------------------------------------
def test_cost_cap_doubles_the_cadence_on_a_slow_evaluation() -> None:
    hour = 3600.0
    assert dev_eval_interval(20 * 60.0, base_interval_s=hour, cost_fraction=0.4) == hour
    assert dev_eval_interval(24 * 60.0, base_interval_s=hour, cost_fraction=0.4) == hour
    assert dev_eval_interval(25 * 60.0, base_interval_s=hour, cost_fraction=0.4) == 2 * hour
    assert dev_eval_interval(50 * 60.0, base_interval_s=hour, cost_fraction=0.4) == 4 * hour
    # A fast evaluation never takes the cadence below the base.
    assert dev_eval_interval(0.0, base_interval_s=hour, cost_fraction=0.4) == hour


# --- is_new_best: gates and routes --------------------------------------
_R1 = dict(
    failures=0,
    anchor_win_rate=0.50,
    head_to_head=0.50,
    head_to_head_ci_95=(0.40, 0.60),
    delta_d_ci_95=(0.01, 0.05),
)
_R2 = dict(
    failures=0,
    anchor_win_rate=0.50,
    head_to_head=0.70,
    head_to_head_ci_95=(0.55, 0.80),
    delta_d_ci_95=(-0.03, 0.01),
)


def test_is_new_best_accepts_via_the_delta_d_route() -> None:
    assert new_best_route(**_R1) == ROUTE_DELTA_D
    assert is_new_best(**_R1) is True
    # R1 needs h >= 0.50 as a point estimate: it must not lose head-to-head.
    assert new_best_route(**{**_R1, "head_to_head": 0.49}) is None
    # ...and a dD CI lower bound strictly above 0.
    assert new_best_route(**{**_R1, "delta_d_ci_95": (0.0, 0.05)}) is None


def test_is_new_best_accepts_via_the_head_to_head_route() -> None:
    assert new_best_route(**_R2) == ROUTE_HEAD_TO_HEAD
    # R2 needs the h CI lower bound strictly above 0.50...
    assert new_best_route(**{**_R2, "head_to_head_ci_95": (0.50, 0.80)}) is None
    # ...and D non-inferior: dD CI lower bound >= -0.03.
    assert new_best_route(**{**_R2, "delta_d_ci_95": (-0.031, 0.01)}) is None


def test_is_new_best_rejects_on_each_gate() -> None:
    for evidence in (_R1, _R2):
        # G1: any failure in any pairing.
        assert new_best_route(**{**evidence, "failures": 1}) is None
        # G2: regression against the warm-start anchor.
        assert new_best_route(**{**evidence, "anchor_win_rate": 0.44}) is None
        assert new_best_route(**{**evidence, "anchor_win_rate": 0.45}) is not None
    # Without the incumbent's paired b1-v2 games neither route can be shown.
    assert new_best_route(**{**_R2, "delta_d_ci_95": None}) is None


def test_is_new_best_with_no_incumbent_falls_back_to_the_gates() -> None:
    none = dict(head_to_head=None, head_to_head_ci_95=None, delta_d_ci_95=None)
    assert new_best_route(failures=0, anchor_win_rate=None, **none) == ROUTE_NO_INCUMBENT
    assert new_best_route(failures=2, anchor_win_rate=None, **none) is None


# --- v2 stop rules on synthetic metric streams ---------------------------
_H = 3600.0


def _eval_record(
    cycle: int,
    hours: float,
    *,
    new_best: bool = False,
    d: float = -1.10,
    best_d: float = -1.10,
    a: float | None = 0.50,
    ece: float | None = 0.05,
) -> CheckpointMetrics:
    return CheckpointMetrics(
        cycle=cycle,
        elapsed_s=hours * _H * 1.3,
        step=cycle,
        games=cycle * 10,
        checkpoint=f"ckpt-{cycle:09d}",
        dev_games=104,
        dev_win_rate_vs_b1=0.0,
        dev_win_rate_vs_b1_ci_95=(0.0, 0.0),
        dev_win_rate_vs_previous_best=0.5,
        dev_win_rate_vs_previous_best_ci_95=(0.4, 0.6),
        dev_mean_vp_margin_vs_b1=-30.0,
        new_best=new_best,
        best_dev_win_rate_vs_b1=0.0,
        training_s=hours * _H,
        dev_score_vs_b1=d,
        best_dev_score_vs_b1=d if new_best else best_d,
        dev_win_rate_vs_anchor=a,
        dev_ece_vs_b1=ece,
    )


def _feed(monitor: StoppingMonitor, records: list[CheckpointMetrics]) -> list[StopDecision]:
    return [monitor.observe_checkpoint(r) for r in records]


def test_plateau_fires_after_six_training_hours_without_a_new_best() -> None:
    monitor = StoppingMonitor()
    stream = [_eval_record(1, 1.0, new_best=True)]
    stream += [_eval_record(c, float(c)) for c in range(2, 7)]  # 2..6 h
    stream += [_eval_record(7, 6.99)]
    assert not any(d.stopped for d in _feed(monitor, stream))
    decision = monitor.observe_checkpoint(_eval_record(8, 7.0))
    assert decision.stopped and decision.kind is StopKind.PLATEAU
    assert decision.details["since_best_s"] == pytest.approx(6 * _H)


def test_plateau_counts_from_run_start_and_resets_on_a_new_best() -> None:
    monitor = StoppingMonitor()
    assert monitor.observe_checkpoint(_eval_record(1, 5.0)).stopped is False
    assert monitor.observe_checkpoint(_eval_record(2, 5.5, new_best=True)).stopped is False
    assert monitor.observe_checkpoint(_eval_record(3, 11.0)).stopped is False
    assert monitor.observe_checkpoint(_eval_record(4, 11.5)).kind is StopKind.PLATEAU
    fresh = StoppingMonitor()
    assert fresh.observe_checkpoint(_eval_record(1, 6.0)).kind is StopKind.PLATEAU


def test_regression_vs_anchor_fires_after_three_consecutive_evaluations() -> None:
    monitor = StoppingMonitor()
    decisions = _feed(monitor, [_eval_record(c, c * 0.5, a=0.34) for c in (1, 2, 3)])
    assert [d.stopped for d in decisions] == [False, False, True]
    assert decisions[-1].kind is StopKind.REGRESSION_ANCHOR
    # A break in the streak, or a = 0.35 exactly, stays silent.
    quiet = StoppingMonitor()
    a_values = [0.34, 0.34, 0.36, 0.34, 0.34, 0.35, 0.35, 0.35]
    stream = [_eval_record(i + 1, (i + 1) * 0.5, a=a) for i, a in enumerate(a_values)]
    assert not any(d.stopped for d in _feed(quiet, stream))


def test_margin_collapse_fires_when_d_stays_far_below_the_best() -> None:
    monitor = StoppingMonitor()
    decisions = _feed(
        monitor, [_eval_record(c, c * 0.5, d=-1.16, best_d=-1.10) for c in (1, 2, 3)]
    )
    assert [d.stopped for d in decisions] == [False, False, True]
    assert decisions[-1].kind is StopKind.MARGIN_COLLAPSE
    quiet = StoppingMonitor()
    stream = [_eval_record(c, c * 0.5, d=-1.149, best_d=-1.10) for c in range(1, 6)]
    assert not any(d.stopped for d in _feed(quiet, stream))
    # A new best in between resets the streak.
    reset = StoppingMonitor()
    stream = [
        _eval_record(1, 0.5, d=-1.16),
        _eval_record(2, 1.0, d=-1.16),
        _eval_record(3, 1.5, d=-1.0, new_best=True),
        _eval_record(4, 2.0, d=-1.06, best_d=-1.0),
        _eval_record(5, 2.5, d=-1.06, best_d=-1.0),
    ]
    assert not any(d.stopped for d in _feed(reset, stream))


def test_calibration_fires_for_the_comparator_and_is_diagnostic_for_main() -> None:
    comparator = StoppingMonitor(calibration_stops=True)
    stream = [_eval_record(c, c * 0.5, ece=0.16) for c in (1, 2, 3)]
    decisions = _feed(comparator, stream)
    assert [d.stopped for d in decisions] == [False, False, True]
    assert decisions[-1].kind is StopKind.CALIBRATION
    main = StoppingMonitor(calibration_stops=False)
    assert not any(d.stopped for d in _feed(main, stream))
    assert main.calibration_streak == 3  # still tracked as a diagnostic
    quiet = StoppingMonitor(calibration_stops=True)
    stream = [_eval_record(c, c * 0.5, ece=0.15) for c in range(1, 6)]
    assert not any(d.stopped for d in _feed(quiet, stream))


def test_game_shape_fires_after_two_training_hours_above_22_generations() -> None:
    monitor = StoppingMonitor()
    for hours in (0.5, 1.0, 1.5, 1.99):
        assert monitor.observe_game_shape(23.0, elapsed_s=hours * _H).stopped is False
    decision = monitor.observe_game_shape(23.0, elapsed_s=2.0 * _H)
    assert decision.stopped and decision.kind is StopKind.GAME_SHAPE

    quiet = StoppingMonitor()
    medians = [(0.5, 23.0), (1.0, 23.0), (1.9, 23.0), (2.0, 22.0), (2.5, 30.0), (3.9, 30.0)]
    for hours, median in medians:
        assert quiet.observe_game_shape(median, elapsed_s=hours * _H).stopped is False
    # Cycles with no completed game carry no shape evidence either way.
    assert quiet.observe_game_shape(None, elapsed_s=3.95 * _H).stopped is False
    assert quiet.observe_game_shape(30.0, elapsed_s=4.0 * _H).stopped is True


def test_stop_rule_state_survives_a_resume() -> None:
    monitor = StoppingMonitor()
    _feed(monitor, [_eval_record(1, 1.0, new_best=True), _eval_record(2, 2.0, a=0.3, d=-1.2)])
    monitor.observe_game_shape(25.0, elapsed_s=2.5 * _H)
    restored = StoppingMonitor()
    restored.load_state_dict(monitor.state_dict())
    assert restored.state_dict() == monitor.state_dict()
    assert restored.regression_streak == 1 and restored.margin_streak == 1
    assert restored.last_best_training_s == pytest.approx(_H)
