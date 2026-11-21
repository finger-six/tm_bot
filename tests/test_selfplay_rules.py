"""Acceptance tests for the shared v2 self-play rules (#48): τ(g), opponent mix, mirror
recording, utility U, recorded anchor logits, and the 16 x 4 search budget."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from tests.test_comparator import _tiny_comparator
from tests.test_runs import _VOCAB, _choice, _obs, _tiny_model
from tm_bot.agents import B0Agent
from tm_bot.agents.play_rules import (
    V2_PLAY_TEMPERATURE,
    PlayTemperatureSchedule,
    game_utility,
)
from tm_bot.comparator.agent import comparator_eval_factory
from tm_bot.comparator.ppo import PPOBuffer, PPOSample
from tm_bot.comparator.selfplay import ComparatorSelfPlayEngine
from tm_bot.models.encode import collate_encoded_batch, encode_choice, encode_observation
from tm_bot.models.network import save_checkpoint
from tm_bot.runs.buffer import DecisionSample, ReplayBuffer
from tm_bot.runs.config import RunConfig
from tm_bot.runs.deveval import dev_agent_factory
from tm_bot.runs.recorder import GameRecorder
from tm_bot.runs.selfplay import (
    OpponentChoice,
    OpponentSampler,
    SelfPlayEngine,
    build_selfplay_specs,
    make_train_seed_keys,
)
from tm_bot.search.agent import ISMCTSAgent, make_warmstart_search_agent
from tm_bot.search.config import SearchConfig

GENERATIONS = range(0, 31)


def _action_obs(generation: int, *, actor: int = 0) -> dict[str, Any]:
    obs = _obs(actor=actor)
    obs["public"]["generation"] = generation
    assert obs["public"]["phase"] == "action"
    return obs


def _outcome(scores: tuple[int, int], winner: int | None) -> SimpleNamespace:
    return SimpleNamespace(
        decisions=4,
        ok=True,
        result=SimpleNamespace(
            scores=scores, winner=winner, tie=winner is None, generations=12
        ),
        generations=12,
        error_type=None,
        error=None,
        decision_latencies_ms=((), ()),
        seed_key="train-0000",
        map="tharsis",
    )


# --- τ(g) ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("generation", "tau"), [(1, 1.0), (4, 1.0), (7, 0.65), (10, 0.3), (15, 0.3)]
)
def test_play_temperature_matches_the_schedule(generation: int, tau: float) -> None:
    assert V2_PLAY_TEMPERATURE(generation) == pytest.approx(tau)
    assert RunConfig(run_name="r").play_temperature(generation) == pytest.approx(tau)


def test_play_temperature_is_never_zero() -> None:
    assert all(V2_PLAY_TEMPERATURE(g) > 0.0 for g in GENERATIONS)
    assert V2_PLAY_TEMPERATURE.for_observation(_obs()) == pytest.approx(1.0)
    non_action = _obs()
    non_action["public"]["phase"] = "research"
    assert V2_PLAY_TEMPERATURE.for_observation(non_action) is None


def _assert_search_agent_samples_at_tau(agent: ISMCTSAgent) -> None:
    for g in GENERATIONS:
        obs = _action_obs(g)
        # Search roots use the same rule as the policy-head pick below.
        root_tau = agent.config.decision_temperature(obs, agent.config.temperature)  # type: ignore[arg-type]
        assert root_tau == pytest.approx(V2_PLAY_TEMPERATURE(g))
        # A nested action-phase choice goes through the policy head (no search).
        type(agent).act_batch_with_samplers([agent], [obs], [_choice()], [None])  # type: ignore[list-item]
        assert agent.last_decision_temperature == pytest.approx(V2_PLAY_TEMPERATURE(g))
        assert agent.last_decision_temperature > 0.0


def _assert_comparator_agent_samples_at_tau(agent: Any) -> None:
    for g in GENERATIONS:
        agent.act(_action_obs(g), _choice())
        assert agent.last_decision_temperature == pytest.approx(V2_PLAY_TEMPERATURE(g))
        assert agent.last_decision_temperature > 0.0


def test_no_learned_main_seat_runs_at_tau_zero(tmp_path: Path) -> None:
    ckpt = save_checkpoint(tmp_path / "pool.pt", _tiny_model(seed=2), _VOCAB)
    config = RunConfig(run_name="tau-main", output_root=tmp_path, workers=1)
    engine = SelfPlayEngine(
        config, model=_tiny_model(), vocab=_VOCAB, anchor_model=_tiny_model(seed=1)
    )
    seats = {
        "learner": engine._learner_factory(1),
        "latest": engine._opponent_factory(OpponentChoice("latest"), 2),
        "pool": engine._opponent_factory(OpponentChoice("pool", ckpt), 3),
        "anchor": engine._opponent_factory(OpponentChoice("anchor"), 4),
        "dev": dev_agent_factory(ckpt, device="cpu", search_config=config.search_config())(),
    }
    for name, agent in seats.items():
        assert isinstance(agent, ISMCTSAgent), name
        _assert_search_agent_samples_at_tau(agent)
    assert V2_PLAY_TEMPERATURE.spec in seats["dev"].version


def test_eval_search_agent_defaults_to_v2_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for var in (
        "TM_SEARCH_ACTION_TEMP",
        "TM_SEARCH_PLAY_TEMP",
        "TM_SEARCH_FALLBACK",
        "TM_SEARCH_SIMS",
        "TM_SEARCH_WORLDS",
    ):
        monkeypatch.delenv(var, raising=False)
    ckpt = save_checkpoint(tmp_path / "eval.pt", _tiny_model(seed=3), _VOCAB)
    agent = make_warmstart_search_agent(checkpoint=ckpt, device="cpu")
    assert (agent.config.num_simulations, agent.config.num_worlds) == (16, 4)
    assert agent.config.margin_weight == pytest.approx(0.15)
    _assert_search_agent_samples_at_tau(agent)
    fallback = make_warmstart_search_agent(checkpoint=ckpt, device="cpu", search_fallback=True)
    assert (fallback.config.num_simulations, fallback.config.num_worlds) == (8, 4)


def test_no_learned_comparator_seat_runs_at_tau_zero(tmp_path: Path) -> None:
    from tm_bot.comparator.model import save_comparator_checkpoint

    ckpt = save_comparator_checkpoint(tmp_path / "pool.pt", _tiny_comparator(seed=2), _VOCAB)
    config = RunConfig(run_name="tau-cmp", agent_kind="comparator", workers=1)
    engine = ComparatorSelfPlayEngine(
        config, model=_tiny_comparator(), vocab=_VOCAB, anchor_model=_tiny_comparator(seed=1)
    )
    seats = {
        "learner": engine._learner_factory(1),
        "latest": engine._opponent_factory(OpponentChoice("latest"), 2),
        "pool": engine._opponent_factory(OpponentChoice("pool", Path(ckpt)), 3),
        "anchor": engine._opponent_factory(OpponentChoice("anchor"), 4),
        "eval": comparator_eval_factory(_tiny_comparator(), _VOCAB)(),
    }
    for agent in seats.values():
        _assert_comparator_agent_samples_at_tau(agent)
    assert V2_PLAY_TEMPERATURE.spec in seats["eval"].version


def test_comparator_records_the_tau_one_policy() -> None:
    agent = comparator_eval_factory(_tiny_comparator(), _VOCAB)()
    obs, choice = _action_obs(12), _choice()
    probs, _ = agent.predict(obs, choice)
    expected = np.asarray(probs[: len(choice.candidates)], dtype=np.float64)
    expected /= expected.sum()
    agent.act(obs, choice)
    assert agent.last_decision_temperature == pytest.approx(0.3)
    np.testing.assert_allclose(agent.last_decision_policy, expected, rtol=1e-6)


def test_generation_zero_keys_its_own_temperature() -> None:
    schedule = PlayTemperatureSchedule(start=1.0, floor=0.3, decay_from=0, floor_at=10)
    assert schedule.for_observation(_action_obs(0)) == pytest.approx(1.0)
    assert schedule.for_observation(_action_obs(1)) == pytest.approx(0.93)


# --- opponent mix --------------------------------------------------------------------


def test_opponent_sampler_matches_the_mix() -> None:
    config = RunConfig(run_name="mix")
    sampler = OpponentSampler(
        latest_prob=config.opponent_latest_prob,
        pool_prob=config.opponent_pool_prob,
        anchor_prob=config.opponent_anchor_prob,
        b1_prob=config.opponent_b1_prob,
        rng=np.random.default_rng(123),
    )
    sampler.set_pool([Path("a.pt"), Path("b.pt")])
    draws = 20_000
    counts: dict[str, int] = {}
    for _ in range(draws):
        kind = sampler.sample().kind
        counts[kind] = counts.get(kind, 0) + 1
    expected = {"latest": 0.50, "pool": 0.25, "anchor": 0.10, "b1": 0.15}
    assert set(counts) == set(expected)
    for kind, p in expected.items():
        assert counts[kind] / draws == pytest.approx(p, abs=0.01), kind


def test_opponent_mix_is_identical_for_both_agents() -> None:
    engines = [
        SelfPlayEngine(RunConfig(run_name="m"), model=_tiny_model(), vocab=_VOCAB),
        ComparatorSelfPlayEngine(
            RunConfig(run_name="c", agent_kind="comparator"),
            model=_tiny_comparator(),
            vocab=_VOCAB,
        ),
    ]
    mixes = {
        (s.latest_prob, s.pool_prob, s.anchor_prob, s.b1_prob)
        for s in (e.opponents for e in engines)
    }
    assert mixes == {(0.50, 0.25, 0.10, 0.15)}


def test_anchor_opponent_requires_a_warm_start() -> None:
    engine = SelfPlayEngine(RunConfig(run_name="no-anchor"), model=_tiny_model(), vocab=_VOCAB)
    with pytest.raises(RuntimeError, match="warmstart_checkpoint"):
        engine._opponent_factory(OpponentChoice("anchor"), 1)


# --- mirror recording ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "both"), [("latest", True), ("pool", False), ("anchor", False), ("b1", False)]
)
def test_mirror_games_record_both_seats(kind: str, both: bool) -> None:
    probs = {"latest_prob": 0.0, "pool_prob": 0.0, "anchor_prob": 0.0, "b1_prob": 0.0}
    probs[f"{kind}_prob"] = 1.0
    sampler = OpponentSampler(**probs, rng=np.random.default_rng(0))
    sampler.set_pool([Path("p.pt")])
    _specs, recorders, seats, kinds = build_selfplay_specs(
        seed_keys=make_train_seed_keys(0, 2),
        maps=["tharsis"],
        learner_factory=lambda seed: B0Agent(seed=seed),
        sampler=sampler,
        opponent_factory=lambda choice, seed: B0Agent(seed=seed),
        vocab=_VOCAB,
    )
    assert kinds == [kind, kind]
    for spec, recorder, training_seat in zip(_specs, recorders, seats):
        for seat in (0, 1):
            spec.agents[seat].act(_obs(actor=seat), _choice(seat=seat))
        recorded = sorted(int(d.seat) for d in recorder.decisions)
        assert recorded == ([0, 1] if both else [training_seat])
        assert len(recorder.actions) == 2


def test_mirror_rows_take_targets_from_their_own_seat() -> None:
    engine = SelfPlayEngine(RunConfig(run_name="mirror"), model=_tiny_model(), vocab=_VOCAB)
    recorder = GameRecorder(
        "train-0000", "tharsis", capture_seat=0, vocab=_VOCAB, record_both_seats=True
    )
    for seat in (0, 1):
        recorder.record(
            _obs(actor=seat), _choice(seat=seat), 0, [0.5, 0.5],  # type: ignore[arg-type]
            was_searched=True, play_temperature=0.65,
        )
    batch = engine.assemble_batch([_outcome((70, 40), 0)], [recorder], [0])
    by_seat = {s.seat: s for s in batch.samples}
    assert by_seat[0].value_target == 1.0 and by_seat[1].value_target == -1.0
    assert by_seat[0].margin_target == pytest.approx(30 / 50)
    assert by_seat[1].margin_target == pytest.approx(-30 / 50)
    assert all(s.play_temperature == pytest.approx(0.65) for s in batch.samples)

    # A non-mirror game still rejects a foreign seat's record.
    recorder.record_both_seats = False
    with pytest.raises(RuntimeError, match="training seat"):
        engine.assemble_batch([_outcome((70, 40), 0)], [recorder], [0])


def test_comparator_mirror_runs_gae_per_seat() -> None:
    engine = ComparatorSelfPlayEngine(
        RunConfig(run_name="cmp-mirror", agent_kind="comparator"),
        model=_tiny_comparator(),
        vocab=_VOCAB,
    )
    recorder = GameRecorder(
        "train-0000", "tharsis", capture_seat=1, vocab=_VOCAB, record_both_seats=True
    )
    for _ in range(2):
        for seat in (0, 1):
            recorder.record(
                _obs(actor=seat), _choice(seat=seat), 1, [0.5, 0.5],  # type: ignore[arg-type]
                was_searched=False, play_temperature=1.0,
            )
    batch = engine.assemble_batch([_outcome((50, 62), 1)], [recorder], [1])
    samples = batch.ppo_samples
    assert samples is not None and len(samples) == 4
    for seat, won in ((1, True), (0, False)):
        rows = [s for s in samples if s.seat == seat]
        assert len(rows) == 2
        assert rows[-1].return_ == pytest.approx(1.0 if won else 0.0)
        assert all(r.value_target == (1.0 if won else -1.0) for r in rows)
        assert all(r.margin_target == pytest.approx((12 if won else -12) / 50) for r in rows)


# --- utility U -----------------------------------------------------------------------


def test_utility_ranks_any_win_above_any_loss() -> None:
    vps = range(0, 201, 5)
    pairs = [(a, b) for a in vps for b in vps]
    wins = [game_utility(1.0, a, b) for a, b in pairs]
    losses = [game_utility(-1.0, a, b) for a, b in pairs]
    ties = [game_utility(0.0, a, b) for a, b in pairs]
    assert min(wins) > max(ties) and min(ties) > max(losses)
    assert min(wins) > max(losses)
    # Margins in -200..200 clip to ±60 VP, so U spans exactly 1 ± 0.15·1.2.
    assert min(wins) == pytest.approx(1.0 - 0.15 * 1.2)
    assert max(losses) == pytest.approx(-1.0 + 0.15 * 1.2)


# --- anchor logits -------------------------------------------------------------------


def _anchor_forward(model: torch.nn.Module, obs: Any, choice: Any) -> np.ndarray:
    batch = collate_encoded_batch([obs], [choice])
    with torch.no_grad():
        logits = model(batch, compute_policy=True, compute_value=False)["policy_logits"]
    return logits[0].float().numpy()


@pytest.mark.parametrize("agent_kind", ["main", "comparator"])
def test_recorded_decisions_carry_aligned_anchor_logits(agent_kind: str) -> None:
    if agent_kind == "main":
        anchor = _tiny_model(seed=9)
        engine: Any = SelfPlayEngine(
            RunConfig(run_name="anchor"), model=_tiny_model(), vocab=_VOCAB, anchor_model=anchor
        )
    else:
        anchor = _tiny_comparator(seed=9)
        engine = ComparatorSelfPlayEngine(
            RunConfig(run_name="anchor", agent_kind="comparator"),
            model=_tiny_comparator(),
            vocab=_VOCAB,
            anchor_model=anchor,
        )
    assert not any(p.requires_grad for p in anchor.parameters())
    recorder = GameRecorder("train-0000", "tharsis", capture_seat=0, vocab=_VOCAB)
    choice = _choice()
    obs = _obs()
    recorder.record(obs, choice, 1, [0.25, 0.75], was_searched=True)  # type: ignore[arg-type]
    batch = engine.assemble_batch([_outcome((60, 50), 0)], [recorder], [0])
    sample = batch.training_samples[0]
    n = len(choice.candidates)
    assert sample.anchor_logits is not None
    assert sample.anchor_logits.dtype == np.float32
    assert sample.anchor_logits.shape == (n,) == sample.policy_target.shape
    expected = _anchor_forward(
        anchor,
        encode_observation(obs, _VOCAB),  # type: ignore[arg-type]
        encode_choice(choice, obs, _VOCAB),  # type: ignore[arg-type]
    )
    np.testing.assert_allclose(sample.anchor_logits, expected[:n], rtol=1e-5, atol=1e-6)


def test_no_anchor_means_no_anchor_logits() -> None:
    engine = SelfPlayEngine(RunConfig(run_name="plain"), model=_tiny_model(), vocab=_VOCAB)
    recorder = GameRecorder("train-0000", "tharsis", capture_seat=0, vocab=_VOCAB)
    recorder.record(_obs(), _choice(), 0, None, was_searched=False)  # type: ignore[arg-type]
    batch = engine.assemble_batch([_outcome((60, 50), 0)], [recorder], [0])
    assert batch.samples[0].anchor_logits is None


# --- search budget and run config ----------------------------------------------------


def test_search_defaults_to_16x4_with_an_8x4_fallback() -> None:
    assert (SearchConfig().num_simulations, SearchConfig().num_worlds) == (16, 4)
    fallback = SearchConfig.v2(fallback=True)
    assert (fallback.num_simulations, fallback.num_worlds) == (8, 4)
    run = RunConfig(run_name="r").search_config()
    assert (run.num_simulations, run.num_worlds) == (16, 4)
    assert run.play_temperature == V2_PLAY_TEMPERATURE
    assert run.margin_weight == pytest.approx(0.15)
    cut = RunConfig(run_name="r", search_fallback=True)
    assert cut.num_simulations == 8
    assert (cut.search_config().num_simulations, cut.search_config().num_worlds) == (8, 4)


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_simulations": 4},
        {"num_worlds": 2},
        {"search_fallback": True, "num_simulations": 32},
        {"selfplay_rules": "v1"},
        {"opponent_anchor_prob": 0.2},
    ],
)
def test_run_config_rejects_settings_outside_the_v2_rules(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        RunConfig(run_name="bad", **overrides)


def test_run_config_records_the_rules_and_rejects_v1_payloads() -> None:
    config = RunConfig(run_name="v2", search_fallback=True)
    payload = config.to_dict()
    assert payload["selfplay_rules"] == "v2"
    assert payload["opponent_anchor_prob"] == pytest.approx(0.10)
    assert RunConfig.from_dict(payload) == config
    v1 = dict(payload)
    del v1["selfplay_rules"]
    with pytest.raises(ValueError, match="v1 run cannot be resumed"):
        RunConfig.from_dict(v1)


# --- buffer round-trip ---------------------------------------------------------------


def _encoded() -> tuple[Any, Any]:
    obs, choice = _obs(), _choice()
    return encode_observation(obs, _VOCAB), encode_choice(choice, obs, _VOCAB)  # type: ignore[arg-type]


def test_replay_buffer_round_trips_the_v2_fields() -> None:
    obs, choice = _encoded()
    sample = DecisionSample(
        observation=obs,
        choice=choice,
        policy_target=np.array([0.4, 0.6], dtype=np.float32),
        value_target=1.0,
        seat=1,
        was_searched=True,
        seed_key="train-0000",
        map_name="tharsis",
        generation=7,
        play_temperature=0.65,
        margin_target=-0.3,
        anchor_logits=np.array([0.1, -0.2], dtype=np.float32),
    )
    buffer = ReplayBuffer(capacity=4)
    buffer.add(sample)
    restored = ReplayBuffer(capacity=4)
    restored.load_state_dict(copy.deepcopy(buffer.state_dict()))
    got = restored.samples[0]
    assert got.play_temperature == pytest.approx(0.65)
    assert got.margin_target == pytest.approx(-0.3)
    np.testing.assert_array_equal(got.anchor_logits, sample.anchor_logits)

    # A pre-#48 payload (no v2 keys) loads with the defaults.
    old = buffer.state_dict()
    for key in ("play_temperature", "margin_target", "anchor_logits"):
        del old[key]
    restored.load_state_dict(old)
    legacy = restored.samples[0]
    assert (legacy.play_temperature, legacy.margin_target, legacy.anchor_logits) == (
        None,
        0.0,
        None,
    )


def test_ppo_buffer_round_trips_the_v2_fields() -> None:
    obs, choice = _encoded()
    sample = PPOSample(
        observation=obs,
        choice=choice,
        policy_target=np.array([0.5, 0.5], dtype=np.float32),
        value_target=-1.0,
        seat=0,
        was_searched=False,
        seed_key="train-0000",
        map_name="tharsis",
        generation=3,
        action=1,
        old_log_prob=-0.69,
        behavior_value=0.4,
        play_temperature=1.0,
        margin_target=0.5,
        anchor_logits=np.array([1.5, 0.5], dtype=np.float32),
    )
    buffer = PPOBuffer(capacity=4)
    buffer.add(sample)
    restored = PPOBuffer(capacity=4)
    restored.load_state_dict(buffer.state_dict())
    got = restored.samples[0]
    assert got.play_temperature == pytest.approx(1.0)
    assert got.margin_target == pytest.approx(0.5)
    np.testing.assert_array_equal(got.anchor_logits, sample.anchor_logits)
