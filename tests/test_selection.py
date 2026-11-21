"""Floor-aware checkpoint selection, v2 stop rules, and the dev-eval cost cap (#51, #40 sections 1-2)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tm_bot.eval.stats import cluster_bootstrap_mean_ci, paired_bootstrap_ci
from tm_bot.runs.deveval import (
    DevGame,
    dev_eval_interval,
    dev_game_records,
    dev_score,
    dev_win_rate,
    games_on_seeds,
    paired_delta_d,
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
