"""Pre-self-play warm-start gate (#47, thresholds #38 §5): every check passes and fails on synthetic
results, the dev-* seat-swapped schedule, frequency baselines, value recording, and the report."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tm_bot.agents.protocol import GameRunResult
from tm_bot.env.types import SUPPORTED_MAPS, Candidate, Choice, GameResult
from tm_bot.eval.pairing import PairedGameMeta
from tm_bot.search.agent import ISMCTSAgent
from tm_bot.search.config import SearchConfig
from tm_bot.training import run_warmstart_gate as runner
from tm_bot.training import warmstart_gate as wg

# --- synthetic inputs ------------------------------------------------------------------------


def _acc(acc: float, total: int = 1000) -> dict[str, Any]:
    per_map = {m: {"correct": 0, "total": 0, "acc": 0.0} for m in SUPPORTED_MAPS}
    per_map["tharsis"] = {"correct": int(acc * total), "total": total, "acc": acc}
    return {"overall": {"correct": int(acc * total), "total": total, "acc": acc}, "per_map": per_map}


def _val(**overrides: float) -> dict[str, Any]:
    """#46 `metrics.json["val"]` shape; every task comfortably passes by default."""
    acc = {"corp": 0.74, "prelude": 0.82, "draft": 0.71, "research_buy": 0.6, "branch": 0.5,
           "project_card": 0.4, "standard_project": 0.6, "card_action": 0.5, **overrides}
    return {
        "corp": _acc(acc["corp"]),
        "prelude_top1": _acc(acc["prelude"]),
        "draft": _acc(acc["draft"]),
        "research_buy": {"step0": _acc(acc["research_buy"]), "done": _acc(0.5)},
        "branch": _acc(acc["branch"]),
        "project_play": _acc(acc["project_card"]),
        "standard_project": _acc(acc["standard_project"]),
        "card_action": _acc(acc["card_action"]),
    }


def _baselines(**overrides: float) -> dict[str, Any]:
    base = {t: 0.3 for t, _, _ in wg.POLICY_TASKS}
    base.update(overrides)
    return {t: _acc(a) for t, a in base.items()}


def _value(accuracy: float = 0.75, brier: float = 0.17, ece: float = 0.02,
           buckets: tuple[float | None, float | None, float | None] = (0.70, 0.76, 0.82)) -> dict[str, Any]:
    def m(a: float | None) -> dict[str, Any] | None:
        return None if a is None else {"n": 100, "accuracy": a, "brier": 0.18, "ece": 0.03, "nll": 0.5}

    per_gen = {"gen 1": m(0.6)}
    per_gen.update({b: m(a) for b, a in zip(wg.MONOTONE_BUCKETS, buckets) if a is not None})
    return {"temperature": 1.0, "per_gen": per_gen,
            "midgame": {"n": 300, "accuracy": accuracy, "brier": brier, "ece": ece, "nll": 0.5},
            "midgame_by_phase": {}, "midgame_per_map": {}}


def _summary(key: str, *, win_rate: float = 0.95, ci: tuple[float, float] = (0.9, 1.0), margin: float = 50.0,
             own_vp: float = 90.0, median: float = 12.0, p95: float = 18.0,
             failures: dict[str, int] | None = None, games: int | None = None) -> dict[str, Any]:
    spec = wg.PAIRINGS_BY_KEY[key]
    counts = {k: 0 for k, _ in wg.FAILURE_KINDS}
    counts.update(failures or {})
    per_map = {m: {"total_games": 2, "failures": 0, "win_rate": win_rate, "win_rate_ci_95": list(ci),
                   "mean_vp_margin": margin, "mean_generations": median} for m in SUPPORTED_MAPS}
    total = spec.expected_games if games is None else games
    return {
        "key": key, "label": spec.label, "gating": spec.gating, "expected_games": spec.expected_games,
        "seed_keys": [], "all_dev_seeds": True, "seat_swapped": True,
        "report": {"total_games": total, "completed_games": total, "failure_count": sum(counts.values()),
                   "wins": 0, "losses": 0, "ties": 0, "win_rate": win_rate, "win_rate_ci_95": list(ci),
                   "mean_vp_margin": margin, "mean_vp_a": own_vp, "mean_vp_b": own_vp - margin,
                   "per_map": per_map, "failures": [], "agent_a": {"version": "warmstart-v3"}},
        "generations": {"median": median, "p95": p95, "max": 25},
        "failure_counts": counts,
        "live_value": None,
        "wall_time_s": 1.0,
    }


def _summaries(agent: str = "main", **overrides: dict[str, Any]) -> dict[str, Any]:
    out = {s.key: _summary(s.key) for s in wg.pairings_for(agent)}
    for key, kw in overrides.items():
        out[key] = _summary(key, **kw)
    return out


def _by_name(checks: list[wg.Check], fragment: str) -> wg.Check:
    hits = [c for c in checks if fragment in c.name]
    assert len(hits) == 1, [c.name for c in checks]
    return hits[0]


# --- #38 §5.1 validation ---------------------------------------------------------------------


@pytest.mark.parametrize("task,key,label", wg.POLICY_TASKS)
def test_each_policy_task_must_strictly_beat_its_frequency_baseline(task: str, key: str, label: str) -> None:
    passing = wg.validation_checks(_val(), _baselines(), _value())
    assert _by_name(passing, f"{label}: top-1 beats").passed
    model_acc = _val()[key.split(".")[0]]
    model_acc = model_acc["step0"] if task == "research_buy" else model_acc
    tied = wg.validation_checks(_val(), _baselines(**{task: model_acc["overall"]["acc"]}), _value())
    assert not _by_name(tied, f"{label}: top-1 beats").passed
    others = [c for c in tied if c.group == "validation" and "beats" in c.name and label not in c.name]
    assert all(c.passed for c in others)
    missing = wg.validation_checks(_val(), {t: v for t, v in _baselines().items() if t != task}, _value())
    assert not _by_name(missing, f"{label}: top-1 beats").passed


@pytest.mark.parametrize("task,label,ok,bad", [
    ("corp", "Corporation", 0.710, 0.7099),
    ("prelude", "Prelude (step-0)", 0.798, 0.7979),
    ("draft", "Draft pick", 0.687, 0.6869),
])
def test_v1_non_regression_boundary_is_minus_1_5_pp(task: str, label: str, ok: float, bad: float) -> None:
    assert _by_name(wg.validation_checks(_val(**{task: ok}), _baselines(), _value()), f"{label}: within").passed
    assert not _by_name(wg.validation_checks(_val(**{task: bad}), _baselines(), _value()), f"{label}: within").passed


@pytest.mark.parametrize("field,ok,bad,fragment", [
    ("accuracy", 0.72, 0.7199, "value accuracy"),
    ("brier", 0.19, 0.1901, "value Brier"),
    ("ece", 0.04, 0.0401, "value ECE"),
])
def test_midgame_value_thresholds(field: str, ok: float, bad: float, fragment: str) -> None:
    assert _by_name(wg.validation_checks(_val(), _baselines(), _value(**{field: ok})), fragment).passed
    assert not _by_name(wg.validation_checks(_val(), _baselines(), _value(**{field: bad})), fragment).passed


@pytest.mark.parametrize("buckets,passed", [
    ((0.70, 0.75, 0.80), True),
    ((0.70, 0.75, 0.75), False),  # flat is not increasing
    ((0.75, 0.70, 0.80), False),
    ((0.70, 0.80, 0.78), False),
    ((0.70, None, 0.80), False),  # a missing bucket fails
])
def test_value_accuracy_must_increase_monotonically_over_generation_buckets(buckets: Any, passed: bool) -> None:
    check = _by_name(wg.validation_checks(_val(), _baselines(), _value(buckets=buckets)), "increases over")
    assert check.passed is passed


def test_missing_validation_metrics_fail_every_validation_check() -> None:
    checks = wg.validation_checks(None, None, None)
    assert checks and not any(c.passed for c in checks)


def test_load_validation_judges_both_agents_post_temperature(tmp_path: Path) -> None:
    """#58 amended check 14: the main agent is judged post-temperature too (sidecar: test_value_temperature)."""
    raw, cal = _value(accuracy=0.70), _value(accuracy=0.80)
    for agent in ("main", "comparator"):
        run = tmp_path / agent
        run.mkdir()
        payload = {"status": "completed", "version": "v", "val": {**_val(), "value": raw}, "value_calibrated": cal}
        (run / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
        _, value, _ = runner.load_validation(run, agent)
        assert value == cal
    (tmp_path / "main" / "metrics.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not completed"):
        runner.load_validation(tmp_path / "main", "main")


# --- #38 §5.2 live checks --------------------------------------------------------------------


@pytest.mark.parametrize("key,kw,fragment", [
    ("vs_b0", {"win_rate": 0.90}, "Win rate vs b0-v1"),
    ("vs_b0", {"ci": (0.8001, 0.97)}, "CI lower bound"),
    ("vs_b0", {"margin": 35.0}, "Mean VP margin vs b0-v1"),
    ("mirror", {"median": 16.0}, "median game length"),
    ("mirror", {"p95": 22.0}, "p95 game length"),
    ("vs_v1", {"win_rate": 0.85}, "archived v1"),
    ("vs_b1", {"own_vp": 75.0}, "own VP"),
    ("vs_b1", {"margin": -35.0}, "Mean VP margin vs b1-v2"),
])
def test_live_thresholds_pass_on_their_boundary(key: str, kw: dict[str, Any], fragment: str) -> None:
    assert _by_name(wg.live_checks(_summaries(**{key: kw})), fragment).passed


@pytest.mark.parametrize("key,kw,fragment", [
    ("vs_b0", {"win_rate": 0.8999}, "Win rate vs b0-v1"),
    ("vs_b0", {"ci": (0.80, 0.97)}, "CI lower bound"),  # strict: must exceed 80%
    ("vs_b0", {"margin": 34.99}, "Mean VP margin vs b0-v1"),
    ("mirror", {"median": 16.5}, "median game length"),
    ("mirror", {"p95": 22.05}, "p95 game length"),
    ("vs_v1", {"win_rate": 0.8499}, "archived v1"),
    ("vs_b1", {"own_vp": 74.99}, "own VP"),
    ("vs_b1", {"margin": -35.01}, "Mean VP margin vs b1-v2"),
])
def test_live_thresholds_fail_past_their_boundary(key: str, kw: dict[str, Any], fragment: str) -> None:
    checks = wg.live_checks(_summaries(**{key: kw}))
    assert not _by_name(checks, fragment).passed
    assert sum(not c.passed for c in checks) == 1


def test_high_win_rate_with_a_wide_ci_still_fails_vs_b0() -> None:
    checks = wg.live_checks(_summaries(vs_b0={"win_rate": 0.93, "ci": (0.78, 0.99)}))
    assert _by_name(checks, "Win rate vs b0-v1").passed
    assert not _by_name(checks, "CI lower bound").passed


@pytest.mark.parametrize("kind", [k for k, _ in wg.FAILURE_KINDS])
def test_any_single_failure_in_any_pairing_fails_its_zero_failure_check(kind: str) -> None:
    clean = wg.failure_checks(_summaries())
    assert all(c.passed for c in clean) and len(clean) == 3
    checks = wg.failure_checks(_summaries(vs_b1_raw={"failures": {kind: 1}}))  # the diagnostic pairing counts too
    assert [c.passed for c in checks] == [k != kind for k, _ in wg.FAILURE_KINDS]


def test_failure_classification() -> None:
    assert wg.classify_failure("StepCapError") == "step_cap"
    assert wg.classify_failure("IndexError") == "illegal_response"
    for other in ("BridgeError", "RuntimeError", "MissingResultError", None):
        assert wg.classify_failure(other) == "adapter_error"


def test_live_checks_fail_when_pairings_did_not_run() -> None:
    checks = wg.live_checks({})
    assert checks and not any(c.passed for c in checks)
    assert not any(c.passed for c in wg.failure_checks({}))


# --- schedule ---------------------------------------------------------------------------------


def test_pairing_protocol_counts_match_38_section_5() -> None:
    main = {p.key: p.expected_games for p in wg.pairings_for("main")}
    assert main == {"vs_b0": 104, "mirror": 40, "vs_v1": 104, "vs_b1": 104, "vs_b1_raw": 104}
    assert {p.key: p.num_seeds for p in wg.pairings_for("main")} == {
        "vs_b0": 13, "mirror": 5, "vs_v1": 13, "vs_b1": 13, "vs_b1_raw": 13}
    assert [p.key for p in wg.pairings_for("comparator")] == ["vs_b0", "mirror", "vs_v1", "vs_b1"]
    assert not wg.PAIRINGS_BY_KEY["vs_b1_raw"].gating
    assert len(SUPPORTED_MAPS) == 4


@pytest.mark.parametrize("key", [p.key for p in wg.PAIRINGS])
def test_run_pairing_plays_the_exact_seat_swapped_dev_schedule(key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = wg.PAIRINGS_BY_KEY[key]
    captured: dict[str, Any] = {}

    def fake_play(specs: Any, **kwargs: Any) -> list[GameRunResult]:
        captured["specs"] = specs
        return [_game(s.seed_key, s.map_name, (100, 50), 0, 12) for s in specs]

    monkeypatch.setattr(runner, "play_games", fake_play)
    outcomes, metas, values, _ = runner.run_pairing(
        spec, agent="main", candidate=lambda: SimpleNamespace(), opponent=lambda: SimpleNamespace(),
        record_values=False, num_seeds=spec.num_seeds, seed_start=0, maps=SUPPORTED_MAPS,
        workers=1, batch_size=1, max_decisions=10,
    )
    specs = captured["specs"]
    assert len(specs) == len(outcomes) == spec.expected_games
    seeds = sorted({m.seed_key for m in metas})
    assert seeds == [f"dev-{i:04d}" for i in range(spec.num_seeds)]
    assert not any(s.seed_key.startswith("eval") for s in specs)
    pairs: dict[tuple[str, str], list[int]] = {}
    for m in metas:
        pairs.setdefault((m.seed_key, m.map_name), []).append(m.a_seat)
    assert len(pairs) == spec.num_seeds * len(SUPPORTED_MAPS)
    assert all(sorted(v) == [0, 1] for v in pairs.values())
    assert values is None


def test_gate_seed_keys_are_dev_only() -> None:
    keys = wg.gate_seed_keys(13)
    assert [k.key for k in keys] == [f"dev-{i:04d}" for i in range(13)]
    assert all(k.namespace == "dev" for k in keys)


def test_schedule_check_fails_short_missing_or_non_dev_runs() -> None:
    spec = wg.PAIRINGS_BY_KEY["vs_b0"]
    assert wg.schedule_check(spec, _summary("vs_b0")).passed
    assert not wg.schedule_check(spec, _summary("vs_b0", games=8)).passed
    assert not wg.schedule_check(spec, None).passed
    bad = _summary("vs_b0")
    bad["all_dev_seeds"] = False
    assert not wg.schedule_check(spec, bad).passed


# --- aggregation from game results ----------------------------------------------------------


def _game(seed: str, map_name: str, scores: tuple[int, int], winner: int | None, gens: int,
          error_type: str | None = None) -> GameRunResult:
    if error_type is not None:
        return GameRunResult(seed_key=seed, map=map_name, result=None, generations=None, decisions=5,
                             agent_versions=("cand", "opp"), ok=False, error_type=error_type, error="x")
    res = GameResult(scores=scores, winner=winner, tie=winner is None, generations=gens)
    return GameRunResult(seed_key=seed, map=map_name, result=res, generations=gens, decisions=100,
                         agent_versions=("cand", "opp"), ok=True)


def _schedule(num_seeds: int) -> list[PairedGameMeta]:
    return [PairedGameMeta(seed_key=k.key, map_name=m, a_seat=s, seed_a=1, seed_b=2)
            for k in wg.gate_seed_keys(num_seeds) for m in SUPPORTED_MAPS for s in (0, 1)]


def test_summarize_pairing_aggregates_win_rate_generations_and_failure_kinds() -> None:
    spec = wg.PAIRINGS_BY_KEY["mirror"]
    metas = _schedule(spec.num_seeds)
    gens = list(range(10, 10 + len(metas)))  # 10..49
    outcomes = []
    for i, (m, g) in enumerate(zip(metas, gens)):
        a, b = (80, 40) if i % 4 else (40, 80)  # the candidate loses every 4th game
        scores = (a, b) if m.a_seat == 0 else (b, a)
        outcomes.append(_game(m.seed_key, m.map_name, scores, int(scores[1] > scores[0]), g))
    for i, err in ((1, "StepCapError"), (2, "IndexError"), (3, "BridgeError")):
        outcomes[i] = _game(metas[i].seed_key, metas[i].map_name, (0, 0), None, 0, error_type=err)
    s = wg.summarize_pairing(spec, outcomes, metas, candidate_name="c", opponent_name="self")
    done = [g for i, g in enumerate(gens) if i not in (1, 2, 3)]
    assert s["generations"]["median"] == pytest.approx(float(np.median(done)))
    assert s["generations"]["p95"] == pytest.approx(float(np.percentile(done, 95)))
    assert s["failure_counts"] == {"illegal_response": 1, "adapter_error": 1, "step_cap": 1}
    det = s["report"]
    assert det["total_games"] == 40 and det["completed_games"] == 37
    assert det["wins"] == 27 and det["losses"] == 10
    assert det["mean_vp_a"] == pytest.approx((27 * 80 + 10 * 40) / 37)
    assert s["all_dev_seeds"] and s["seat_swapped"]
    assert wg.schedule_check(spec, s).passed
    assert not all(c.passed for c in wg.failure_checks({"mirror": s}))


def test_live_value_entries_score_the_candidate_seat_outcome() -> None:
    metas = _schedule(1)[:3]
    outcomes = [
        _game(metas[0].seed_key, metas[0].map_name, (90, 50), 0, 12),  # a_seat 0 wins
        _game(metas[1].seed_key, metas[1].map_name, (90, 50), 0, 12),  # a_seat 1 loses
        _game(metas[2].seed_key, metas[2].map_name, (0, 0), None, 0, error_type="StepCapError"),
    ]
    rows = runner.value_entries_with_outcomes(outcomes, metas, [[(1, 0.6), (3, 0.9)], [(2, 0.2)], [(2, 0.5)]])
    assert rows == [(1, 0.6, 1.0), (3, 0.9, 1.0), (2, 0.2, 0.0)]
    m = wg.live_value_metrics(rows)
    assert m["all"]["n"] == 3 and m["gen_2_plus"]["n"] == 2
    assert m["gen_2_plus"]["brier"] == pytest.approx(((0.9 - 1) ** 2 + 0.2 ** 2) / 2)
    with pytest.raises(RuntimeError):
        runner.value_entries_with_outcomes(outcomes, metas, [[]])


class _StubPolicy:
    version = "stub"
    temperature = 0.0
    model = vocab = None
    device = "cpu"

    def __init__(self, value_temperature: float = 1.0) -> None:
        self.value_temperature = value_temperature

    def reset(self, seed: int | None = None) -> None:
        pass

    def __deepcopy__(self, memo: dict[int, Any]) -> _StubPolicy:
        return self

    def predict_batch(self, observations: Any, choices: Any, *, compute_value: bool = True) -> Any:
        probs = [np.array([0.2, 0.8][: len(c.candidates)]) for c in choices]
        return probs, np.full(len(choices), 0.5, dtype=np.float32)  # value 0.5 -> win prob 0.75

    def win_probability(self, values: Any) -> np.ndarray:
        from tm_bot.models.value_temperature import calibrated_win_prob

        return calibrated_win_prob(values, self.value_temperature)


@pytest.mark.parametrize("temperature", [1.0, 1.5])
def test_value_recording_search_agent_records_without_changing_actions(temperature: float) -> None:
    """It records the reported P(win): σ(logit / T) of the value head (0.75 raw at value 0.5)."""
    import copy

    entries: list[tuple[int, float]] = []
    agent = runner.ValueRecordingSearchAgent(
        ISMCTSAgent(_StubPolicy(temperature), config=SearchConfig()), entries  # type: ignore[arg-type]
    )
    clone = copy.deepcopy(agent)
    assert type(clone) is runner.ValueRecordingSearchAgent and clone.value_entries is entries
    obs = {"actor": 0, "public": {"phase": "preludes", "generation": 3}}
    two = Choice(seat=0, kind="space", prompt="Select space", candidates=(Candidate("a", "a"), Candidate("b", "b")))
    one = Choice(seat=0, kind="space", prompt="Select space", candidates=(Candidate("a", "a"),))
    out = runner.ValueRecordingSearchAgent.act_batch_with_samplers([clone, clone], [obs, obs], [two, one], [None, None])
    assert out == [1, 0]
    expected = 1.0 / (1.0 + np.exp(-np.log(3.0) / temperature))  # logit of 0.75 is ln 3
    assert len(entries) == 1 and entries[0][0] == 3 and entries[0][1] == pytest.approx(expected, rel=1e-6)


def test_value_recording_comparator_agent_records_calibrated_win_prob() -> None:
    import copy

    class Inner:
        version = "comparator-warmstart-v2"
        last_decision_value: float | None = None

        def reset(self, seed: int | None = None) -> None:
            pass

        def act(self, observation: Any, choice: Any) -> int:
            self.last_decision_value = 0.3 if len(choice.candidates) > 1 else None
            return 0

    entries: list[tuple[int, float]] = []
    agent = copy.deepcopy(runner.ValueRecordingComparatorAgent(Inner(), entries))
    two = Choice(seat=0, kind="card", prompt="p", candidates=(Candidate("a", "a"), Candidate("b", "b")))
    one = Choice(seat=0, kind="card", prompt="p", candidates=(Candidate("a", "a"),))
    obs = {"public": {"generation": 4}}
    assert agent.act(obs, two) == 0 and agent.act(obs, one) == 0
    assert entries == [(4, 0.3)] and agent.version == "comparator-warmstart-v2"


# --- frequency baselines ---------------------------------------------------------------------


class _FakeTable:
    """Duck-typed `EncodedTable` over in-memory arrays."""

    def __init__(self, name: str, kind: str, cands: list[list[int]], split: list[int], maps: list[int],
                 target: list[int] | None = None, kept: list[int] | None = None,
                 feats: list[list[list[float]]] | None = None) -> None:
        self.task = SimpleNamespace(name=name, kind=kind)
        counts = np.array([len(c) for c in cands], dtype=np.int32)
        self.cand_off = np.zeros(len(cands) + 1, dtype=np.int64)
        np.cumsum(counts, out=self.cand_off[1:])
        self.arrays: dict[str, np.ndarray] = {
            "cand_card": np.array([c for row in cands for c in row], dtype=np.uint16),
            "cand_count": counts,
            "split": np.array(split, dtype=np.uint8),
            "map_id": np.array(maps, dtype=np.uint8),
        }
        if target is not None:
            self.arrays["target"] = np.array(target, dtype=np.int16)
        if kept is not None:
            self.arrays["kept_mask"] = np.array(kept, dtype=np.uint8)
        width = 10 if name == "branch" else 4
        flat = [f for row in (feats or [[[0.0] * width for _ in r] for r in cands]) for f in row]
        self.arrays["cand_feats"] = np.array(flat, dtype=np.float32).reshape(-1, width)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.arrays[key]


@pytest.mark.parametrize("chunk_rows", [1, 2, 100])
def test_frequency_baseline_ranks_candidates_by_train_selection_counts(chunk_rows: int) -> None:
    # train: card 7 chosen twice, card 5 once. val rows: {5,7}->7 (hit), {5,9} -> 5 (miss: 9 chosen),
    # {9,7} target 7 (hit, map 1), tie {3,4} -> first candidate 3 (hit).
    table = _FakeTable(
        "project_play", "policy",
        cands=[[5, 7], [7, 9], [5, 7], [5, 7], [5, 9], [9, 7], [3, 4]],
        split=[0, 0, 0, 1, 1, 1, 1], maps=[0, 0, 0, 0, 0, 1, 0], target=[1, 0, 0, 1, 1, 1, 0],
    )
    out = runner.frequency_baseline(table, chunk_rows=chunk_rows)
    assert out["overall"] == {"correct": 3, "total": 4, "acc": 0.75}
    assert out["per_map"]["elysium"]["total"] == 1 and out["per_map"]["elysium"]["correct"] == 1
    with pytest.raises(ValueError):
        runner.frequency_baseline(table, eval_split="test")


def test_frequency_baseline_research_buy_scores_step_zero_targets() -> None:
    done = 2  # any id: the last candidate is `done`
    # train: card 11 bought twice, nothing bought once (done +1). val: {11,12,done} kept {12} -> pred 11 (miss);
    # {12,14,done} nothing kept -> pred done (hit: done has 1 count, 12 and 14 none).
    table = _FakeTable(
        "research_buy", "research",
        cands=[[11, 12, done], [11, 13, done], [12, 13, done], [11, 12, done], [12, 14, done]],
        split=[0, 0, 0, 1, 1], maps=[0, 0, 0, 0, 0], kept=[0b01, 0b11, 0b00, 0b10, 0b00],
    )
    out = runner.frequency_baseline(table)
    assert out["overall"] == {"correct": 1, "total": 2, "acc": 0.5}


def test_frequency_baseline_tells_branch_candidates_apart_by_their_bits() -> None:
    def bit(k: int) -> list[float]:
        v = [0.0] * 10
        v[k] = 1.0
        return v

    # all branch candidates share card id 0; only the branch bit identifies them.
    table = _FakeTable(
        "branch", "policy", cands=[[0, 0], [0, 0], [0, 0]], split=[0, 0, 1], maps=[0, 0, 0],
        target=[1, 1, 0], feats=[[bit(0), bit(3)], [bit(0), bit(3)], [bit(3), bit(0)]],
    )
    assert runner.frequency_baseline(table)["overall"]["correct"] == 1


# --- verdict and report ----------------------------------------------------------------------


def _result(agent: str, summaries: dict[str, Any], **val_overrides: float) -> dict[str, Any]:
    val, value = _val(**val_overrides), _value()
    gate = wg.evaluate_gate(agent=agent, val=val, baselines=_baselines(), value=value, summaries=summaries)
    return {"agent": agent, "version": "warmstart-v3", "gate": gate, "config": {"checkpoint": "runs/x/best"},
            "pairings": {s.key: summaries.get(s.key) for s in wg.pairings_for(agent)},
            "val": val, "value": value, "baselines": _baselines()}


@pytest.mark.parametrize("agent", ["main", "comparator"])
def test_verdict_passes_only_when_every_gating_check_passes(agent: str) -> None:
    ok = _result(agent, _summaries(agent))
    assert ok["gate"]["verdict"] == "PASS"
    assert ok["gate"]["gating_passed"] == ok["gate"]["gating_total"]
    assert _result(agent, _summaries(agent, mirror={"p95": 23.0}))["gate"]["verdict"] == "FAIL"
    assert _result(agent, _summaries(agent), corp=0.70)["gate"]["verdict"] == "FAIL"
    missing = _summaries(agent)
    missing.pop("vs_v1")
    assert _result(agent, missing)["gate"]["verdict"] == "FAIL"


def test_diagnostic_pairing_result_does_not_gate_but_its_failures_do() -> None:
    bad_raw = _summaries("main", vs_b1_raw={"win_rate": 0.0, "margin": -90.0, "own_vp": 40.0})
    assert _result("main", bad_raw)["gate"]["verdict"] == "PASS"
    crashed_raw = _summaries("main", vs_b1_raw={"failures": {"adapter_error": 1}})
    assert _result("main", crashed_raw)["gate"]["verdict"] == "FAIL"


def test_report_renders_verdict_checks_and_per_map_breakdowns() -> None:
    summaries = _summaries("main", vs_b1={"margin": -40.0})
    summaries["vs_b1"]["live_value"] = wg.live_value_metrics([(3, 0.7, 1.0), (1, 0.4, 0.0)])
    result = _result("main", summaries)
    md = wg.format_gate_report(result)
    assert "**Verdict: FAIL**" in md
    fail_rows = [line for line in md.splitlines() if "**FAIL**" in line]
    assert len(fail_rows) == 1 and "Mean VP margin vs b1-v2" in fail_rows[0] and "-40.00" in fail_rows[0]
    assert md.count("| Tharsis |") >= len(wg.pairings_for("main"))
    for spec in wg.pairings_for("main"):
        assert f"### {spec.label}" in md
    assert "Live value Brier / ECE / accuracy (gen >= 2)" in md
    assert "Diagnostics (not gating)" in md and "raw policy vs b1-v2" in md
    assert "| Corporation | overall |" in md and "| Corporation | tharsis |" in md
    passing = wg.format_gate_report(_result("main", _summaries("main")))
    assert "**Verdict: PASS**" in passing and "**FAIL**" not in passing
    json.dumps(result)  # the JSON report is serializable


def test_main_light_search_uses_completed_q_and_the_v1_opponent_keeps_visits() -> None:
    assert runner.LIGHT_SEARCH["root_rule"] == "completed_q"
    assert runner.RAW_POLICY == {**runner.LIGHT_SEARCH, "num_simulations": 0}
    assert runner.V1_LIGHT_SEARCH == {**runner.LIGHT_SEARCH, "root_rule": "visits"}
    assert set(runner.VALUE_RECORDING_OPPONENTS) == {"b1", "self", "v1"}
    for agent in ("main", "comparator"):
        gate = _result(agent, _summaries(agent))["gate"]
        names = [c["name"] for c in gate["checks"]]
        assert "Gen >= 2 value ECE (post-temperature)" in names


def test_report_shows_search_vs_raw_and_live_value_per_opponent() -> None:
    summaries = _summaries("main", vs_b1={"margin": -22.0, "own_vp": 76.8},
                           vs_b1_raw={"margin": -17.5, "own_vp": 80.4})
    for key, rows in (("vs_b1", [(3, 0.7, 0.0)]), ("mirror", [(4, 0.6, 1.0)]), ("vs_v1", [(5, 0.9, 1.0)])):
        summaries[key]["live_value"] = wg.live_value_metrics(rows)
    gate = _result("main", summaries)["gate"]
    delta = [d for d in gate["diagnostics"] if d["pairing"] == "search vs raw policy (vs b1-v2)"]
    assert len(delta) == 1 and delta[0]["value"].startswith("-3.60 / -4.50")
    live = {d["pairing"] for d in gate["diagnostics"] if d["name"].startswith("Live value")}
    assert live == {"vs b1-v2", "self-mirror", "vs archived v1 warm start"}
    md = wg.format_gate_report(_result("main", summaries))
    assert "search vs raw policy (vs b1-v2)" in md and "| self-mirror | Live value" in md
    assert "post-temperature" in md and "### Value head (post-temperature)" in md
    # The comparator has no raw-policy pairing, so no search-vs-raw row.
    cmp_gate = _result("comparator", _summaries("comparator"))["gate"]
    assert not any(d["pairing"].startswith("search vs raw") for d in cmp_gate["diagnostics"])


def test_compute_frequency_baselines_on_the_golden_v2_tables(tmp_path: Path) -> None:
    """End to end on real v2 tables: setup baselines from parquet, gen 2+ baselines from the cache."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from tm_bot.data.extract import run_extraction
    from tm_bot.data.inventory import run_inventory
    from tm_bot.training.encoded_cache import EncodedTable

    repo = Path(__file__).resolve().parents[1]
    golden, table_id = repo / "tests" / "fixtures" / "golden_v2", "900000001"
    archive = tmp_path / "archive"
    for pid in ("10000001", "10000002"):
        (archive / pid).mkdir(parents=True)
        name = f"game_{table_id}_{pid}.json"
        (archive / pid / name).write_bytes((golden / pid / name).read_bytes())
    run_inventory(archive, tmp_path / "index.sqlite", progress_interval=0)
    data = tmp_path / "warmstart"
    run_extraction(archive, tmp_path / "index.sqlite", data,
                   card_registry_path=repo / "data" / "card_registry.json", progress_interval=0)
    for path in data.glob("*.parquet"):  # alternate train / val so both splits are populated
        t = pq.read_table(path)
        split = pa.array(["train" if i % 2 == 0 else "val" for i in range(t.num_rows)], pa.string())
        pq.write_table(t.set_column(t.schema.get_field_index("split"), "split", split), path)

    out = runner.compute_frequency_baselines(data, tmp_path / "cache", workers=0)
    assert set(out) == {t for t, _, _ in wg.POLICY_TASKS}
    for task, name in runner.BASELINE_TASKS:
        table = EncodedTable(tmp_path / "cache" / name)
        assert out[task]["overall"]["total"] == len(table.split_indices["val"])
        assert sum(m["total"] for m in out[task]["per_map"].values()) == out[task]["overall"]["total"]
    for task in ("corp", "prelude"):
        assert out[task]["overall"]["total"] == pq.read_table(data / "setup_choices.parquet")["split"].to_pylist().count("val") > 0
