from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tm_bot.agents import B0_VERSION, B1_VERSION, B0Agent, B1Agent, GameRunResult, play_game
from tm_bot.env import SUPPORTED_MAPS, GameResult, WorkerClient
from tm_bot.eval import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    SEED_NAMESPACES,
    SeedKey,
    build_paired_specs,
    derive_agent_seed,
    evaluate_agents,
    generate_seed_keys,
    make_seed_key,
    paired_bootstrap_ci,
    summarize_evaluation,
    write_report_files,
)


def _make_run_result(
    *,
    seed_key: str,
    map_name: str,
    a_seat: int,
    a_vp: int,
    b_vp: int,
    tie: bool = False,
    generations: int = 10,
    decisions: int = 100,
    ok: bool = True,
    error_type: str | None = None,
    error: str | None = None,
    latencies_ms: tuple[tuple[float, ...], tuple[float, ...]] = ((0.1, 0.2), (0.3, 0.4)),
) -> GameRunResult:
    if not ok:
        return GameRunResult(
            seed_key=seed_key,
            map=map_name,
            result=None,
            generations=None,
            decisions=decisions,
            agent_versions=("agent-a-v1", "agent-b-v1"),
            ok=False,
            error_type=error_type or "RuntimeError",
            error=error or "boom",
            decision_latencies_ms=latencies_ms,
        )
    if a_seat == 0:
        scores = (a_vp, b_vp)
        winner = None if tie else (0 if a_vp > b_vp else 1)
    else:
        scores = (b_vp, a_vp)
        winner = None if tie else (1 if a_vp > b_vp else 0)
    return GameRunResult(
        seed_key=seed_key,
        map=map_name,
        result=GameResult(scores=scores, winner=winner, tie=tie, generations=generations),
        generations=generations,
        decisions=decisions,
        agent_versions=("agent-a-v1", "agent-b-v1") if a_seat == 0 else ("agent-b-v1", "agent-a-v1"),
        ok=True,
        decision_latencies_ms=latencies_ms,
    )


def test_seed_namespaces_are_disjoint_by_construction() -> None:
    assert set(SEED_NAMESPACES) == {"train", "dev", "eval"}

    train_keys = {make_seed_key("train", i).key for i in range(100)}
    dev_keys = {make_seed_key("dev", i).key for i in range(100)}
    eval_keys = {make_seed_key("eval", i).key for i in range(100)}

    assert len(train_keys) == 100
    assert len(dev_keys) == 100
    assert len(eval_keys) == 100
    assert train_keys.isdisjoint(dev_keys)
    assert train_keys.isdisjoint(eval_keys)
    assert dev_keys.isdisjoint(eval_keys)

    for k in train_keys:
        assert k.startswith("train-")
        assert SeedKey.parse(k).namespace == "train"
    for k in dev_keys:
        assert k.startswith("dev-")
        assert SeedKey.parse(k).namespace == "dev"
    for k in eval_keys:
        assert k.startswith("eval-")
        assert SeedKey.parse(k).namespace == "eval"

    with pytest.raises(ValueError):
        make_seed_key("gate", 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SeedKey.parse("gate-0000")
    with pytest.raises(ValueError):
        SeedKey.parse("eval-")
    with pytest.raises(ValueError):
        make_seed_key("eval", "")

    # Derived per-game agent seeds are deterministic and namespace/role isolated
    s_train_a = derive_agent_seed(make_seed_key("train", 0), map_name="tharsis", role="a")
    s_dev_a = derive_agent_seed(make_seed_key("dev", 0), map_name="tharsis", role="a")
    s_eval_a = derive_agent_seed(make_seed_key("eval", 0), map_name="tharsis", role="a")
    s_eval_b = derive_agent_seed(make_seed_key("eval", 0), map_name="tharsis", role="b")

    assert s_eval_a == derive_agent_seed("eval-0000", map_name="tharsis", role="a")
    assert len({s_train_a, s_dev_a, s_eval_a, s_eval_b}) == 4


def test_pairing_structure_swaps_seats_for_every_seed_and_map() -> None:
    seed_keys = generate_seed_keys("eval", 3)
    specs, meta = build_paired_specs(
        agent_a=lambda: B1Agent(seed=0),
        agent_b=lambda: B0Agent(seed=0),
        seed_keys=seed_keys,
        maps=SUPPORTED_MAPS,
    )

    assert len(specs) == 3 * len(SUPPORTED_MAPS) * 2 == 24
    assert len(meta) == 24

    by_pair: dict[tuple[str, str], list[tuple[Any, Any]]] = {}
    for spec, m in zip(specs, meta):
        assert m.seed_key == spec.seed_key
        assert m.map_name == spec.map_name
        assert spec.seed_key.startswith("eval-")
        by_pair.setdefault((m.seed_key, m.map_name), []).append((spec, m))

    assert len(by_pair) == 3 * len(SUPPORTED_MAPS)
    for (sk, map_name), entries in by_pair.items():
        assert len(entries) == 2
        (spec0, m0), (spec1, m1) = entries
        assert {m0.a_seat, m1.a_seat} == {0, 1}

        # Seat 0 = A, Seat 1 = B vs Seat 0 = B, Seat 1 = A with swapped seeds
        if m0.a_seat == 0:
            s_a0, s_b0 = spec0, spec1
        else:
            s_a0, s_b0 = spec1, spec0

        assert isinstance(s_a0.agents[0], B1Agent)
        assert isinstance(s_a0.agents[1], B0Agent)
        assert isinstance(s_b0.agents[0], B0Agent)
        assert isinstance(s_b0.agents[1], B1Agent)
        assert s_a0.seat_seeds is not None and s_b0.seat_seeds is not None
        assert s_a0.seat_seeds == (s_b0.seat_seeds[1], s_b0.seat_seeds[0])
        assert s_a0.seat_seeds[0] != s_a0.seat_seeds[1]

    # Default 50 seeds x 4 maps x 2 seats = 400 games
    default_specs, default_meta = build_paired_specs(
        agent_a=lambda: B1Agent(seed=0),
        agent_b=lambda: B0Agent(seed=0),
    )
    assert len(default_specs) == 400
    assert len(default_meta) == 400
    assert {m.map_name for m in default_meta} == set(SUPPORTED_MAPS)


def test_paired_bootstrap_ci_on_synthetic_outcomes() -> None:
    # 1. All wins -> [1.0, 1.0]
    all_wins = [(2.0, 2) for _ in range(20)]
    mean_w, ci_w = paired_bootstrap_ci(all_wins, n_resamples=2000, rng_seed=DEFAULT_BOOTSTRAP_SEED)
    assert mean_w == pytest.approx(1.0)
    assert ci_w == (1.0, 1.0)

    # 2. All losses -> [0.0, 0.0]
    all_losses = [(0.0, 2) for _ in range(20)]
    mean_l, ci_l = paired_bootstrap_ci(all_losses, n_resamples=2000, rng_seed=DEFAULT_BOOTSTRAP_SEED)
    assert mean_l == pytest.approx(0.0)
    assert ci_l == (0.0, 0.0)

    # 3. Perfect seat-0 bias (each pair has 1W + 1L -> pair score = 1.0 / 2) -> [0.5, 0.5]
    seat_biased = [(1.0, 2) for _ in range(25)]
    mean_s, ci_s = paired_bootstrap_ci(seat_biased, n_resamples=2000, rng_seed=DEFAULT_BOOTSTRAP_SEED)
    assert mean_s == pytest.approx(0.5)
    assert ci_s == (0.5, 0.5)

    # 4. Mixed synthetic outcomes (including a tie): compare against direct numpy computation
    pairs = [(2.0, 2)] * 14 + [(1.5, 2)] * 2 + [(1.0, 2)] * 2 + [(0.0, 2)] * 2
    mean_m, ci_m = paired_bootstrap_ci(pairs, n_resamples=4000, rng_seed=123)
    expected_mean = sum(p for p, _ in pairs) / sum(c for _, c in pairs)
    assert mean_m == pytest.approx(expected_mean)
    assert ci_m is not None
    assert ci_m[0] <= mean_m <= ci_m[1]
    assert 0.0 <= ci_m[0] < ci_m[1] <= 1.0

    # Direct reference computation
    rng = np.random.default_rng(123)
    pts_arr = np.array([p for p, _ in pairs], dtype=np.float64)
    cnt_arr = np.array([c for _, c in pairs], dtype=np.float64)
    idx = rng.integers(0, len(pairs), size=(4000, len(pairs)))
    boot = pts_arr[idx].sum(axis=1) / cnt_arr[idx].sum(axis=1)
    ref_low, ref_high = np.quantile(boot, [0.025, 0.975])
    assert ci_m[0] == pytest.approx(float(ref_low))
    assert ci_m[1] == pytest.approx(float(ref_high))

    # Empty pairs -> None
    mean_e, ci_e = paired_bootstrap_ci([], n_resamples=1000, rng_seed=0)
    assert mean_e is None
    assert ci_e is None


def test_overall_bootstrap_resamples_whole_seeds_across_maps() -> None:
    from tm_bot.eval.pairing import PairedGameMeta

    # The deal depends on the seed key only, so a seed's games on all maps form one cluster.
    # Seed 0: A wins all 8 games; seed 1: A loses all 8. Resampling 2 seed clusters can only
    # give means {0, 0.5, 1}, so the 95% CI is [0, 1]; (seed, map) clustering would be narrower.
    metas: list[PairedGameMeta] = []
    outcomes: list[GameRunResult] = []
    for s_idx in range(2):
        sk = f"eval-{s_idx:04d}"
        for map_name in SUPPORTED_MAPS:
            for a_seat in (0, 1):
                metas.append(PairedGameMeta(seed_key=sk, map_name=map_name, a_seat=a_seat, seed_a=1, seed_b=2))
                a_vp, b_vp = (60, 40) if s_idx == 0 else (40, 60)
                outcomes.append(
                    _make_run_result(seed_key=sk, map_name=map_name, a_seat=a_seat, a_vp=a_vp, b_vp=b_vp)
                )

    report = summarize_evaluation(
        outcomes=outcomes,
        metas=metas,
        namespace="eval",
        agent_a_name="a",
        agent_b_name="b",
        a_training_seeds=0,
        b_training_seeds=0,
        workers=1,
        batch_size=1,
        wall_time_s=1.0,
        bootstrap_resamples=2000,
        bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
    )
    det = report["deterministic"]
    assert det["win_rate"] == pytest.approx(0.5)
    assert det["win_rate_ci_95"] == [0.0, 1.0]


def test_scoring_ties_and_failures_excluded_from_win_rate() -> None:
    from tm_bot.eval.pairing import PairedGameMeta

    # 2 seeds x 4 maps x 2 seats = 16 games:
    # - 10 wins (a_vp > b_vp)
    # - 2 ties (0.5 each)
    # - 2 losses (a_vp < b_vp)
    # - 2 failures (excluded from win rate, margin, generations)
    metas: list[PairedGameMeta] = []
    outcomes: list[GameRunResult] = []
    idx = 0
    for s_idx in range(2):
        sk = f"eval-{s_idx:04d}"
        for map_name in SUPPORTED_MAPS:
            for a_seat in (0, 1):
                metas.append(
                    PairedGameMeta(
                        seed_key=sk,
                        map_name=map_name,
                        a_seat=a_seat,
                        seed_a=100 + idx,
                        seed_b=200 + idx,
                    )
                )
                if idx < 10:
                    outcomes.append(
                        _make_run_result(
                            seed_key=sk,
                            map_name=map_name,
                            a_seat=a_seat,
                            a_vp=60,
                            b_vp=40,
                            generations=11,
                        )
                    )
                elif idx < 12:
                    outcomes.append(
                        _make_run_result(
                            seed_key=sk,
                            map_name=map_name,
                            a_seat=a_seat,
                            a_vp=50,
                            b_vp=50,
                            tie=True,
                            generations=12,
                        )
                    )
                elif idx < 14:
                    outcomes.append(
                        _make_run_result(
                            seed_key=sk,
                            map_name=map_name,
                            a_seat=a_seat,
                            a_vp=45,
                            b_vp=55,
                            generations=13,
                        )
                    )
                else:
                    outcomes.append(
                        _make_run_result(
                            seed_key=sk,
                            map_name=map_name,
                            a_seat=a_seat,
                            a_vp=0,
                            b_vp=0,
                            ok=False,
                            error_type="StepCapError",
                            error="Exceeded max_decisions=20000",
                        )
                    )
                idx += 1

    report = summarize_evaluation(
        outcomes=outcomes,
        metas=metas,
        namespace="eval",
        agent_a_name="candidate",
        agent_b_name="b1",
        a_training_seeds=2,
        b_training_seeds=0,
        workers=4,
        batch_size=16,
        wall_time_s=1.25,
        bootstrap_resamples=1000,
        bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
    )

    det = report["deterministic"]
    assert det["total_games"] == 16
    assert det["completed_games"] == 14
    assert det["failure_count"] == 2
    assert det["failure_rate"] == pytest.approx(2 / 16)
    assert det["wins"] == 10
    assert det["ties"] == 2
    assert det["losses"] == 2
    # Win rate = (10 + 0.5 * 2) / 14 = 11 / 14
    assert det["win_rate"] == pytest.approx(11 / 14)
    ci = det["win_rate_ci_95"]
    assert ci[0] <= det["win_rate"] <= ci[1]
    # Mean VP margin = (10*(+20) + 2*(0) + 2*(-10)) / 14 = 180 / 14
    assert det["mean_vp_margin"] == pytest.approx(180 / 14)
    # Mean generations = (10*11 + 2*12 + 2*13) / 14 = 160 / 14
    assert det["mean_generations"] == pytest.approx(160 / 14)
    assert len(det["failures"]) == 2
    assert det["failures"][0]["error_type"] == "StepCapError"

    # Training seeds & < 3 warning flag
    assert det["agent_a"]["training_seeds"] == 2
    assert det["agent_a"]["fewer_than_3_training_seeds"] is True
    assert det["agent_b"]["training_seeds"] == 0
    assert det["agent_b"]["fewer_than_3_training_seeds"] is False

    # Amazonis "not supported" row in per_map
    assert "amazonis planitia" in det["per_map"]
    amazonis = det["per_map"]["amazonis planitia"]
    assert amazonis["status"] == "not supported"


def test_injected_failing_agent_isolated_and_reported_end_to_end(tmp_path: Path) -> None:
    class _CrashingAgent:
        version = "crash-v1"

        def __init__(self, crash_on_call: int = 3) -> None:
            self._crash_on_call = crash_on_call
            self._calls = 0

        def reset(self, seed: int | None = None) -> None:
            self._calls = 0

        def act(self, observation: Any, choice: Any) -> int:
            self._calls += 1
            if self._calls >= self._crash_on_call:
                raise RuntimeError("injected agent failure")
            return 0

    report = evaluate_agents(
        agent_a=lambda: _CrashingAgent(crash_on_call=3),
        agent_b=lambda: B0Agent(seed=0),
        agent_a_name="crashing",
        agent_b_name="b0",
        namespace="dev",
        num_seeds=1,
        num_workers=1,
        batch_size=4,
    )

    det = report["deterministic"]
    assert det["total_games"] == 8
    assert det["completed_games"] == 0
    assert det["failure_count"] == 8
    assert det["failure_rate"] == pytest.approx(1.0)
    assert det["win_rate"] is None
    assert det["win_rate_ci_95"] is None
    assert all(f["error_type"] == "RuntimeError" for f in det["failures"])

    paths = write_report_files(report, tmp_path)
    md_text = paths["markdown"].read_text(encoding="utf-8")
    assert "not supported" in md_text.lower()
    assert "amazonis" in md_text.lower()
    assert "failure" in md_text.lower()


def test_runner_records_per_decision_latency_without_breaking_equality() -> None:
    class _SpyNoExtraArgs:
        version = "spy-v1"

        def __init__(self) -> None:
            self.calls = 0

        def reset(self, seed: int | None = None) -> None:
            self.calls = 0

        def act(self, observation: Any, choice: Any) -> int:
            assert set(observation.keys()) == {"actor", "public", "private", "history"}
            self.calls += 1
            return 0

    with WorkerClient() as client:
        r1 = play_game(
            (_SpyNoExtraArgs(), B0Agent(seed=1)),
            seed_key="eval-lat-check",
            map_name="tharsis",
            client=client,
        )
        r2 = play_game(
            (_SpyNoExtraArgs(), B0Agent(seed=1)),
            seed_key="eval-lat-check",
            map_name="tharsis",
            client=client,
        )

    assert r1.ok is True
    assert r1 == r2
    lat0, lat1 = r1.decision_latencies_ms
    assert len(lat0) + len(lat1) == r1.decisions
    assert all(t >= 0.0 for t in lat0)
    assert all(t >= 0.0 for t in lat1)


def test_evaluation_reproduces_byte_identical_deterministic_report_and_cli(tmp_path: Path) -> None:
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"

    for out_dir in (out1, out2):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "tm_bot.eval",
                "--a",
                "b1",
                "--b",
                "b0",
                "--games",
                "8",
                "--namespace",
                "eval",
                "--workers",
                "2",
                "--batch-size",
                "4",
                "--out",
                str(out_dir),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "win rate" in proc.stdout.lower()

    det_bytes_1 = (out1 / "report_deterministic.json").read_bytes()
    det_bytes_2 = (out2 / "report_deterministic.json").read_bytes()
    assert det_bytes_1 == det_bytes_2

    full1 = json.loads((out1 / "report.json").read_text(encoding="utf-8"))
    full2 = json.loads((out2 / "report.json").read_text(encoding="utf-8"))
    assert full1["deterministic"] == full2["deterministic"]
    assert full1["deterministic"]["agent_a"]["version"] == B1_VERSION
    assert full1["deterministic"]["agent_b"]["version"] == B0_VERSION
    assert full1["deterministic"]["agent_a"]["training_seeds"] == 0
    assert full1["deterministic"]["agent_b"]["training_seeds"] == 0
    assert full1["deterministic"]["per_map"]["amazonis planitia"]["status"] == "not supported"

    rt1 = full1["runtime"]
    assert rt1["workers"] == 2
    assert rt1["wall_time_s"] > 0.0
    assert rt1["hardware"]["cpu_count"] >= 1
    assert rt1["latency_ms"]["agent_a"]["p50_ms"] >= 0.0
    assert rt1["latency_ms"]["agent_a"]["p95_ms"] >= rt1["latency_ms"]["agent_a"]["p50_ms"]
    assert rt1["latency_ms"]["agent_b"]["p50_ms"] >= 0.0

    md1 = (out1 / "report.md").read_text(encoding="utf-8")
    assert "Amazonis" in md1
    assert "not supported" in md1
    assert B1_VERSION in md1
    assert B0_VERSION in md1


def test_cli_omit_timing_and_games_validation(tmp_path: Path) -> None:
    # Invalid --games (not a multiple of 8)
    bad_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tm_bot.eval",
            "--a",
            "b1",
            "--b",
            "b0",
            "--games",
            "10",
            "--namespace",
            "eval",
            "--out",
            str(tmp_path / "bad"),
        ],
        capture_output=True,
        text=True,
    )
    assert bad_proc.returncode != 0

    # --omit-timing makes report.json and report.md reproducible too
    out1 = tmp_path / "omit1"
    out2 = tmp_path / "omit2"
    for out_dir in (out1, out2):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tm_bot.eval",
                "--a",
                "b0",
                "--b",
                "b0",
                "--games",
                "8",
                "--namespace",
                "dev",
                "--workers",
                "2",
                "--batch-size",
                "4",
                "--omit-timing",
                "--out",
                str(out_dir),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

    assert (out1 / "report.json").read_bytes() == (out2 / "report.json").read_bytes()
    assert (out1 / "report.md").read_bytes() == (out2 / "report.md").read_bytes()
    payload = json.loads((out1 / "report.json").read_text(encoding="utf-8"))
    assert "runtime" not in payload

