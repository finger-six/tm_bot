"""Tests for self-play run orchestration (#24)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from tm_bot.agents import B0Agent, play_game, play_games
from tm_bot.agents.card_data import load_card_data
from tm_bot.env.types import Candidate, Choice
from tm_bot.eval.seeds import SeedKey
from tm_bot.models.encode import (
    EncodedChoice,
    EncodedObservation,
    encode_choice,
    encode_observation,
)
from tm_bot.models.network import TMModelConfig, TMNetwork, save_checkpoint
from tm_bot.models.vocab import CardVocab
from tm_bot.runs.buffer import (
    DecisionSample,
    ReplayBuffer,
    collate_samples,
    normalize_policy_target,
)
from tm_bot.runs.checkpoints import CheckpointManager
from tm_bot.runs.config import ResourceLimits, RunConfig
from tm_bot.runs.deveval import (
    DevResult,
    _load_policy_value_agent,
    build_dev_games,
    cached_policy_value_agent,
    clear_dev_agent_cache,
)
from tm_bot.runs.errors import RunResumeError
from tm_bot.runs.learner import (
    LearnerStats,
    compute_value_calibration,
    probe_policy_entropy,
)
from tm_bot.runs.monitoring import (
    CheckpointMetrics,
    CycleMetrics,
    GameSummary,
    MonitorLogger,
    compare_replay,
    game_shape_flags,
)
from tm_bot.runs.orchestrator import SelfPlayRunner
from tm_bot.runs.recorder import GameRecorder, RecordingAgent, capture_agent
from tm_bot.runs.selfplay import (
    OpponentSampler,
    SelfPlayBatch,
    assert_training_seed_keys,
    build_selfplay_specs,
    make_train_seed_keys,
    replay_game,
    utility_for_seat,
)
from tm_bot.runs.stopping import StopKind, StoppingMonitor, is_new_best

_CARDS = load_card_data()
_VOCAB = CardVocab.from_card_data(_CARDS)


def _tiny_model(*, seed: int = 0) -> TMNetwork:
    cfg = TMModelConfig(
        vocab_size=_VOCAB.size,
        d_model=32,
        card_id_emb_dim=16,
        card_static_proj_dim=16,
        slot_emb_dim=8,
        card_dyn_proj_dim=8,
        space_hidden_dim=16,
        n_heads=2,
        n_obs_layers=1,
        dim_feedforward=64,
        policy_ff_dim=32,
        dropout=0.0,
    )
    torch.manual_seed(seed)
    return TMNetwork(cfg, torch.from_numpy(_VOCAB.static_features.copy()))


def _obs(*, actor: int = 0, opponent_hand: tuple[str, ...] = ()) -> dict[str, Any]:
    def player(seat: int) -> dict[str, Any]:
        return {
            "seat": seat,
            "color": "red" if seat == 0 else "blue",
            "terraformRating": 20,
            "megacredits": 25,
            "steel": 1,
            "titanium": 0,
            "plants": 0,
            "energy": 0,
            "heat": 0,
            "megacreditProduction": 0,
            "steelProduction": 0,
            "titaniumProduction": 0,
            "plantProduction": 0,
            "energyProduction": 0,
            "heatProduction": 0,
            "steelValue": 2,
            "titaniumValue": 3,
            "cardCost": 3,
            "cardDiscount": 0,
            "cardsInHandCount": 2,
            "citiesCount": 0,
            "noTagsCount": 0,
            "tags": {},
            "tableau": [],
            "usedActions": [],
            "actionsTakenThisRound": 0,
            "actionsTakenThisGame": 0,
            "availableBlueCardActionCount": 0,
            "lastCardPlayed": None,
            "isActive": seat == actor,
            "passed": False,
            "protectedResources": {},
            "protectedProduction": {},
        }

    return {
        "actor": actor,
        "public": {
            "map": "tharsis",
            "generation": 2,
            "phase": "action",
            "firstPlayer": 0,
            "activePlayer": actor,
            "passedPlayers": [],
            "globalParameters": {
                "temperature": -24,
                "oxygenLevel": 2,
                "oceans": 1,
                "isTerraformed": False,
            },
            "deckSize": 120,
            "discardPileSize": 0,
            "otherDeckSizes": {
                "corporations": {"drawPile": 8, "discardPile": 0},
                "preludes": None,
            },
            "spaces": [],
            "milestones": [],
            "awards": [],
            "players": [player(0), player(1)],
            "publicLog": [],
        },
        "private": {
            "cardsInHand": [] if actor == 1 else [{"name": "CrediCor", "calculatedCost": 0}],
            "dealtCorporationCards": [],
            "dealtPreludeCards": [],
            "dealtProjectCards": [],
            "pickedCorporationCard": None,
            "preludeCardsInHand": [],
            "draftedCards": [],
            "currentDraftPacket": [],
            # Decoy field: only differs when it is the acting seat's own private data.
            "pendingCardSelection": list(opponent_hand) if actor == 1 else [],
        },
        "history": {
            "initialBuy": None,
            "drafts": [],
            "researchBuys": [],
            "cardSelections": [],
        },
    }


def _choice(*, seat: int = 0) -> Choice:
    return Choice(
        seat=seat,  # type: ignore[arg-type]
        kind="card",
        prompt="Select a card to keep",
        candidates=(Candidate("Algae", "Algae"), Candidate("Birds", "Birds")),
    )


def _encoded_sample(*, value: float = 1.0) -> DecisionSample:
    observation = _obs(actor=0)
    choice = _choice()
    return DecisionSample(
        observation=encode_observation(observation, _VOCAB),  # type: ignore[arg-type]
        choice=encode_choice(choice, observation, _VOCAB),  # type: ignore[arg-type]
        policy_target=np.array([0.75, 0.25], dtype=np.float32),
        value_target=value,
        seat=0,
        was_searched=True,
        seed_key="train-0000",
        map_name="tharsis",
        generation=2,
    )


def _assert_encoded_observation_equal(
    actual: EncodedObservation, expected: EncodedObservation
) -> None:
    assert np.array_equal(actual.obs_scalars, expected.obs_scalars)
    assert np.array_equal(actual.card_ids, expected.card_ids)
    assert np.array_equal(actual.card_slots, expected.card_slots)
    assert np.array_equal(actual.card_dyn, expected.card_dyn)
    assert np.array_equal(actual.space_feats, expected.space_feats)


def _assert_encoded_choice_equal(
    actual: EncodedChoice, expected: EncodedChoice
) -> None:
    assert actual.prompt_type_id == expected.prompt_type_id
    assert np.array_equal(actual.choice_ctx, expected.choice_ctx)
    assert np.array_equal(actual.cand_card_ids, expected.cand_card_ids)
    assert np.array_equal(actual.cand_feats, expected.cand_feats)
    assert np.array_equal(actual.picked_card_ids, expected.picked_card_ids)


# --- 1. stopping rules -------------------------------------------------
def test_budget_rule_stops_at_wall_clock_limit() -> None:
    monitor = StoppingMonitor()
    assert monitor.check_budget(59.0, 60.0).stopped is False
    decision = monitor.check_budget(60.0, 60.0)
    assert decision.stopped is True
    assert decision.kind is StopKind.BUDGET


def test_adapter_rejection_and_engine_error_hard_stop() -> None:
    from tm_bot.runs.stopping import FailureEvent

    rejection = StoppingMonitor().observe_failures(
        [FailureEvent("train-0001", "tharsis", "AdapterBugError", "illegal candidate")]
    )
    assert rejection.stopped is True
    assert rejection.kind is StopKind.ADAPTER_REJECTION

    engine = StoppingMonitor().observe_failures(
        [FailureEvent("train-0002", "hellas", "WorkerCrashError", "EOF")]
    )
    assert engine.stopped is True
    assert engine.kind is StopKind.ENGINE_ERROR

    unknown = StoppingMonitor().observe_failures(
        [FailureEvent("train-0004", "hellas", "RuntimeError", "boom")]
    )
    assert unknown.stopped is True
    assert unknown.kind is StopKind.ENGINE_ERROR

    step_cap = StoppingMonitor().observe_failures(
        [FailureEvent("train-0003", "elysium", "StepCapError", "too many decisions")]
    )
    assert step_cap.stopped is True
    assert step_cap.kind is StopKind.STEP_CAP


def test_nonfinite_loss_rule_fires() -> None:
    monitor = StoppingMonitor()
    assert monitor.observe_loss(1.5).stopped is False
    for bad in (float("nan"), float("inf"), -float("inf")):
        decision = monitor.observe_loss(bad)
        assert decision.stopped is True
        assert decision.kind is StopKind.NONFINITE_LOSS


def _checkpoint_record(cycle: int, win_rate: float, *, elapsed_s: float = 0.0) -> CheckpointMetrics:
    return CheckpointMetrics(
        cycle=cycle,
        elapsed_s=elapsed_s,
        step=cycle,
        games=cycle * 10,
        checkpoint=f"ckpt-{cycle:09d}",
        dev_games=100,
        dev_win_rate_vs_b1=win_rate,
        dev_win_rate_vs_b1_ci_95=(win_rate - 0.05, win_rate + 0.05),
        dev_win_rate_vs_previous_best=0.5,
        dev_win_rate_vs_previous_best_ci_95=(0.45, 0.55),
        dev_mean_vp_margin_vs_b1=0.0,
        new_best=False,
        best_dev_win_rate_vs_b1=win_rate,
    )


def test_collapse_win_rate_rule_fires_after_three_checkpoints() -> None:
    monitor = StoppingMonitor(collapse_win_rate_drop=0.15, collapse_checkpoints=3)
    assert monitor.observe_checkpoint(_checkpoint_record(1, 0.80)).stopped is False
    assert monitor.observe_checkpoint(_checkpoint_record(2, 0.70)).stopped is False
    assert monitor.observe_checkpoint(_checkpoint_record(3, 0.60)).stopped is False
    assert monitor.observe_checkpoint(_checkpoint_record(4, 0.64)).stopped is False
    decision = monitor.observe_checkpoint(_checkpoint_record(5, 0.63))
    assert decision.stopped is True
    assert decision.kind is StopKind.COLLAPSE_WIN_RATE
    assert decision.details["streak"] == 3


def test_collapse_entropy_rule_only_within_two_hours() -> None:
    monitor = StoppingMonitor(entropy_collapse_fraction=0.10, entropy_collapse_window_s=7200.0)
    monitor.set_entropy_baseline(2.0)
    decision = monitor.observe_entropy(0.1, elapsed_s=300.0)
    assert decision.stopped is True
    assert decision.kind is StopKind.COLLAPSE_ENTROPY
    later = StoppingMonitor(entropy_collapse_fraction=0.10, entropy_collapse_window_s=7200.0)
    later.set_entropy_baseline(2.0)
    assert later.observe_entropy(0.1, elapsed_s=7201.0).stopped is False


def test_plateau_rule_fires_after_twelve_checkpoints_without_a_new_best() -> None:
    monitor = StoppingMonitor(plateau_checkpoints=12)
    assert monitor.observe_checkpoint(_checkpoint_record(1, 0.50)).stopped is False
    for cycle in range(2, 13):
        assert monitor.observe_checkpoint(_checkpoint_record(cycle, 0.49)).stopped is False
    decision = monitor.observe_checkpoint(_checkpoint_record(13, 0.49))
    assert decision.stopped is True
    assert decision.kind is StopKind.PLATEAU
    assert decision.details["checkpoints_without_improvement"] == 12


def _tied_record(cycle: int, win_rate: float, vs_best_ci: tuple[float, float]) -> CheckpointMetrics:
    return dataclasses.replace(
        _checkpoint_record(cycle, win_rate),
        dev_win_rate_vs_previous_best=sum(vs_best_ci) / 2.0,
        dev_win_rate_vs_previous_best_ci_95=vs_best_ci,
    )


def test_is_new_best_breaks_vs_b1_ties_on_significant_win_over_previous_best() -> None:
    assert is_new_best(0.0, None, None) is True
    assert is_new_best(0.1, 0.0, (0.0, 0.2)) is True
    assert is_new_best(0.0, 0.1, (0.9, 1.0)) is False
    assert is_new_best(0.0, 0.0, (0.51, 0.8)) is True
    assert is_new_best(0.0, 0.0, (0.5, 0.8)) is False
    assert is_new_best(0.0, 0.0, None) is False
    assert is_new_best(None, None, (0.9, 1.0)) is False


def test_plateau_does_not_fire_while_tied_checkpoints_beat_previous_best() -> None:
    monitor = StoppingMonitor(plateau_checkpoints=12)
    for cycle in range(1, 20):
        decision = monitor.observe_checkpoint(_tied_record(cycle, 0.0, (0.6, 0.8)))
        assert decision.stopped is False
    assert monitor.checkpoints_without_improvement == 0


def test_plateau_fires_when_tied_checkpoints_do_not_beat_previous_best() -> None:
    monitor = StoppingMonitor(plateau_checkpoints=12)
    for cycle in range(1, 13):
        assert monitor.observe_checkpoint(_tied_record(cycle, 0.0, (0.4, 0.6))).stopped is False
    decision = monitor.observe_checkpoint(_tied_record(13, 0.0, (0.4, 0.6)))
    assert decision.kind is StopKind.PLATEAU


def test_learner_step_cap_rule_fires() -> None:
    monitor = StoppingMonitor(step_cap=5)
    assert monitor.check_steps(4).stopped is False
    decision = monitor.check_steps(5)
    assert decision.stopped is True
    assert decision.kind is StopKind.MAX_STEPS


# --- 2. seed namespaces ------------------------------------------------
def test_training_seeds_never_fall_in_dev_or_eval_namespaces() -> None:
    keys = make_train_seed_keys(7, 5)
    assert [k.key for k in keys] == [
        "train-0007",
        "train-0008",
        "train-0009",
        "train-0010",
        "train-0011",
    ]
    assert_training_seed_keys([k.key for k in keys])
    for bad in ("dev-0001", "eval-0001"):
        with pytest.raises(ValueError, match="namespace"):
            assert_training_seed_keys([bad])
    with pytest.raises(ValueError, match="namespace"):
        build_selfplay_specs(
            seed_keys=[SeedKey.parse("eval-0001")],
            maps=["tharsis"],
            learner_factory=lambda seed: B0Agent(seed=seed),
            sampler=OpponentSampler(rng=np.random.default_rng(0)),
            opponent_factory=lambda choice, seed: B0Agent(seed=seed),
            vocab=_VOCAB,
        )


def test_selfplay_specs_use_train_seeds_and_one_capture_seat() -> None:
    keys = make_train_seed_keys(0, 3)
    sampler = OpponentSampler(b1_prob=1.0, latest_prob=0.0, pool_prob=0.0)
    specs, recorders, seats, kinds = build_selfplay_specs(
        seed_keys=keys,
        maps=["tharsis", "hellas"],
        learner_factory=lambda seed: B0Agent(seed=seed),
        sampler=sampler,
        opponent_factory=lambda choice, seed: B0Agent(seed=seed),
        vocab=_VOCAB,
    )
    assert all(spec.seed_key.startswith("train-") for spec in specs)
    assert seats == [0, 1, 0]
    assert kinds == ["b1", "b1", "b1"]
    assert [r.capture_seat for r in recorders] == [0, 1, 0]
    assert [r.seed_key for r in recorders] == [k.key for k in keys]


def test_dev_games_are_seat_swapped_pairs() -> None:
    games = build_dev_games(
        100, maps=["tharsis", "elysium", "hellas", "vastitas borealis nova"]
    )
    assert len(games) == 104  # rounded up to whole seat-swapped pairs
    assert all(seed.startswith("dev-") for seed, _map, _seat in games)
    pairs: dict[tuple[str, str], list[int]] = {}
    for seed, map_name, seat in games:
        pairs.setdefault((seed, map_name), []).append(seat)
    assert all(sorted(seats) == [0, 1] for seats in pairs.values())
    assert sum(1 for _s, _m, seat in games if seat == 0) == 52


# --- leakage -----------------------------------------------------------
def test_opponent_private_cards_never_enter_the_acting_seat_encoding() -> None:
    obs_a = _obs(actor=0, opponent_hand=("Algae",))
    obs_b = _obs(actor=0, opponent_hand=("Birds", "Debris"))
    enc_a = encode_observation(obs_a, _VOCAB)  # type: ignore[arg-type]
    enc_b = encode_observation(obs_b, _VOCAB)  # type: ignore[arg-type]
    assert np.array_equal(enc_a.obs_scalars, enc_b.obs_scalars)
    assert np.array_equal(enc_a.card_ids, enc_b.card_ids)
    assert np.array_equal(enc_a.card_dyn, enc_b.card_dyn)
    assert np.array_equal(enc_a.space_feats, enc_b.space_feats)

    own_a = encode_observation(_obs(actor=1, opponent_hand=("Algae",)), _VOCAB)  # type: ignore[arg-type]
    own_b = encode_observation(_obs(actor=1, opponent_hand=("Birds", "Debris")), _VOCAB)  # type: ignore[arg-type]
    assert not np.array_equal(own_a.obs_scalars, own_b.obs_scalars)


def test_recorder_captures_only_the_training_seat() -> None:
    recorder = GameRecorder("train-0000", "tharsis", capture_seat=0, vocab=_VOCAB)
    learner = RecordingAgent(B0Agent(seed=1), recorder, capture_seat=0)
    opponent = RecordingAgent(B0Agent(seed=2), recorder, capture_seat=None)
    learner.act(_obs(actor=0), _choice(seat=0))  # type: ignore[arg-type]
    opponent.act(_obs(actor=1), _choice(seat=1))  # type: ignore[arg-type]
    assert len(recorder.actions) == 2
    assert len(recorder.decisions) == 1
    assert recorder.decisions[0].seat == 0

    from tm_bot.runs.recorder import DecisionRecord
    from tm_bot.runs.selfplay import SelfPlayEngine

    config = RunConfig(run_name="leak", output_root=Path("unused"), workers=1)
    engine = SelfPlayEngine(config, model=_tiny_model(), vocab=_VOCAB)
    bad = GameRecorder("train-0001", "tharsis", capture_seat=0, vocab=_VOCAB)
    bad_obs = _obs(actor=1)
    bad_choice = _choice(seat=1)
    bad.decisions.append(
        DecisionRecord(
            observation=encode_observation(bad_obs, _VOCAB),  # type: ignore[arg-type]
            choice=encode_choice(bad_choice, bad_obs, _VOCAB),  # type: ignore[arg-type]
            candidate_index=0,
            policy=np.array([1.0, 0.0]),
            was_searched=False,
            seat=1,
            generation=2,
            num_candidates=2,
        )
    )
    outcome = SimpleNamespace(
        decisions=1,
        ok=True,
        result=SimpleNamespace(scores=(60, 50), winner=0, tie=False, generations=12),
        generations=12,
        error_type=None,
        error=None,
        decision_latencies_ms=((), ()),
        seed_key="train-0001",
        map="tharsis",
    )
    with pytest.raises(RuntimeError, match="does not match training seat"):
        engine.assemble_batch([outcome], [bad], [0])


def test_recorder_retains_encoded_decisions_not_raw_observations() -> None:
    recorder = GameRecorder("train-0000", "tharsis", capture_seat=0, vocab=_VOCAB)
    agent = RecordingAgent(B0Agent(seed=1), recorder, capture_seat=0)
    observation = _obs(actor=0)
    choice = _choice(seat=0)
    agent.act(observation, choice)  # type: ignore[arg-type]

    assert len(recorder.decisions) == 1
    record = recorder.decisions[0]
    assert isinstance(record.observation, EncodedObservation)
    assert isinstance(record.choice, EncodedChoice)
    assert record.seat == 0
    assert record.num_candidates == len(choice.candidates)
    assert record.generation == int(observation["public"]["generation"])
    stored = [getattr(record, f.name) for f in dataclasses.fields(record)]
    assert not any(isinstance(value, dict) for value in stored)
    assert not any(isinstance(value, Choice) for value in stored)


def test_recorder_snapshots_the_observation_at_record_time() -> None:
    """Mutating the observation after `record` must not change the stored row."""
    import copy

    recorder = GameRecorder("train-0000", "tharsis", capture_seat=0, vocab=_VOCAB)
    observation = _obs(actor=0)
    choice = _choice(seat=0)
    snapshot = copy.deepcopy(observation)
    recorder.record(observation, choice, 0, [0.5, 0.5], was_searched=True)  # type: ignore[arg-type]

    observation["public"]["generation"] = 99
    for player in observation["public"]["players"]:
        player["terraformRating"] = 63
        player["megacredits"] = 400

    record = recorder.decisions[0]
    _assert_encoded_observation_equal(
        record.observation,
        encode_observation(snapshot, _VOCAB),  # type: ignore[arg-type]
    )
    assert record.generation == int(snapshot["public"]["generation"])


def test_recorder_does_not_log_an_action_when_encoding_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed encode must not leave an unapplied action in the regression history."""

    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("encode failed")

    monkeypatch.setattr("tm_bot.runs.recorder.encode_observation", fail)
    recorder = GameRecorder("train-0000", "tharsis", capture_seat=0, vocab=_VOCAB)
    with pytest.raises(ValueError, match="encode failed"):
        recorder.record(_obs(actor=0), _choice(seat=0), 0, None, was_searched=False)  # type: ignore[arg-type]
    assert recorder.actions == []
    assert recorder.decisions == []


def test_batched_recording_rolls_back_when_a_later_entry_fails_to_encode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No batch entry may stay recorded when the batch raises (the runner retries or fails it)."""
    from tm_bot.runs import recorder as recorder_module
    from tm_bot.runs.recorder import RecordingSearchAgent

    class _BatchInner:
        version = "fake"

        @classmethod
        def act_batch_with_samplers(cls, agents, observations, choices, samplers):  # type: ignore[no-untyped-def]
            return [0 for _ in agents]

    real_encode = recorder_module.encode_observation
    calls = {"n": 0}

    def encode_second_fails(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("encode failed")
        return real_encode(*args, **kwargs)

    recorders = [
        GameRecorder(f"train-000{i}", "tharsis", capture_seat=0, vocab=_VOCAB) for i in range(2)
    ]
    # History recorded before the batch must survive the rollback.
    for r in recorders:
        r.record(_obs(actor=1), _choice(seat=1), 1, None, was_searched=False)  # type: ignore[arg-type]
    monkeypatch.setattr(recorder_module, "encode_observation", encode_second_fails)
    # Both seats of one game share a recorder in self-play; agents 0 and 1 do here too.
    agents = [
        RecordingSearchAgent(_BatchInner(), recorders[0], capture_seat=0),
        RecordingSearchAgent(_BatchInner(), recorders[0], capture_seat=0),
        RecordingSearchAgent(_BatchInner(), recorders[1], capture_seat=0),
    ]
    with pytest.raises(ValueError, match="encode failed"):
        RecordingSearchAgent.act_batch_with_samplers(
            agents,
            [_obs(actor=0), _obs(actor=0), _obs(actor=0)],  # type: ignore[list-item]
            [_choice(seat=0), _choice(seat=0), _choice(seat=0)],
            [None, None, None],
        )
    for r in recorders:
        assert r.actions == [(1, 1)]
        assert r.decisions == []


def test_decision_samples_match_encode_at_assembly_rows() -> None:
    """Rows assembled from encoded records equal the old encode-at-assembly rows."""
    from tm_bot.runs.selfplay import SelfPlayEngine

    config = RunConfig(run_name="encoded", output_root=Path("unused"), workers=1)
    engine = SelfPlayEngine(config, model=_tiny_model(), vocab=_VOCAB)
    recorder = GameRecorder("train-0000", "tharsis", capture_seat=0, vocab=_VOCAB)
    cases = [
        (_obs(actor=0), _choice(seat=0), [0.5, 0.5], True),
        (_obs(actor=0), _choice(seat=0), [0.25, 0.75], False),
        (_obs(actor=0), _choice(seat=0), None, False),
    ]
    for observation, choice, policy, was_searched in cases:
        recorder.record(
            observation,  # type: ignore[arg-type]
            choice,
            0,
            policy,
            was_searched=was_searched,
        )
    outcome = SimpleNamespace(
        decisions=3,
        ok=True,
        result=SimpleNamespace(scores=(60, 50), winner=0, tie=False, generations=12),
        generations=12,
        error_type=None,
        error=None,
        decision_latencies_ms=((), ()),
        seed_key="train-0000",
        map="tharsis",
    )
    batch = engine.assemble_batch([outcome], [recorder], [0])
    assert len(batch.samples) == 3
    for sample, (observation, choice, policy, was_searched) in zip(batch.samples, cases):
        _assert_encoded_observation_equal(
            sample.observation,
            encode_observation(observation, _VOCAB),  # type: ignore[arg-type]
        )
        _assert_encoded_choice_equal(
            sample.choice,
            encode_choice(choice, observation, _VOCAB),  # type: ignore[arg-type]
        )
        expected_policy = policy if policy is not None else [1.0, 0.0]
        assert np.array_equal(
            sample.policy_target, normalize_policy_target(expected_policy, 2)
        )
        assert sample.value_target == 1.0
        assert sample.seat == 0
        assert sample.was_searched is was_searched
        assert sample.generation == 2


# --- 3. resume ---------------------------------------------------------
class _FakeEngine:
    def __init__(self, *, value: float = 1.0) -> None:
        self.seed_cursor = 0
        self.opponents = SimpleNamespace(pool=())
        self.value = value

    def set_opponent_pool(self, checkpoints) -> None:
        self.opponents.pool = tuple(Path(p) for p in checkpoints)

    def play_batch(self, num_games: int, *, stop_when=None) -> SelfPlayBatch:
        games = []
        for _ in range(num_games):
            seed = f"train-{self.seed_cursor:04d}"
            self.seed_cursor += 1
            games.append(
                GameSummary(
                    seed_key=seed,
                    map_name="tharsis",
                    ok=True,
                    generations=12,
                    decisions=120,
                    scores=(60, 50),
                    winner=0,
                    tie=False,
                )
            )
        return SelfPlayBatch(
            games=games,
            samples=[_encoded_sample(value=self.value) for _ in games],
            training_seats=[0 for _ in games],
            opponent_kinds=["latest" for _ in games],
            decisions=120 * num_games,
            searched_decisions=20 * num_games,
        )


class _FakeLearner:
    def __init__(self, model: TMNetwork | None = None) -> None:
        self.step = 0
        self.model = model
        self.nonfinite = False
        self.set_win_probability = False

    def train(self, buffer, *, updates=None) -> LearnerStats:
        self.step += 1
        if self.model is not None and self.set_win_probability:
            with torch.no_grad():
                self.model.value_head[-1].bias.fill_(10.0)
        if self.model is not None and self.nonfinite:
            with torch.no_grad():
                self.model.obs_norm.weight.fill_(float("nan"))
            return LearnerStats(updates=0, nonfinite_loss=float("nan"))
        return LearnerStats(
            updates=1,
            policy_loss=1.2,
            value_loss=0.7,
            total_loss=1.9,
        )


class _SharpeningLearner(_FakeLearner):
    def __init__(self, model: TMNetwork) -> None:
        super().__init__(model=model)
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    def train(self, buffer, *, updates=None) -> LearnerStats:
        self.step += 1
        assert self.model is not None
        batch, _targets, _values = collate_samples([_encoded_sample()])
        target = torch.tensor([[1.0, 0.0]])
        for _ in range(20):
            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(batch, compute_policy=True, compute_value=False)
            log_probs = torch.log_softmax(out["policy_logits"], dim=-1)
            loss = -(target * log_probs).sum(dim=-1).mean()
            loss.backward()
            self.optimizer.step()
        return LearnerStats(updates=1, policy_loss=1.0, value_loss=0.5, total_loss=1.5)


def _fake_dev(_checkpoint: Path, _opponent: Path | None) -> DevResult:
    return DevResult(
        games=16,
        completed=16,
        failures=0,
        win_rate=0.5,
        win_rate_ci_95=(0.3, 0.7),
        mean_vp_margin=0.0,
        report={},
    )


def _run_config(tmp_path: Path, name: str, **overrides: Any) -> RunConfig:
    data: dict[str, Any] = dict(
        run_name=name,
        output_root=tmp_path,
        workers=1,
        games_per_cycle=1,
        updates_per_cycle=1,
        learner_batch_size=2,
        checkpoint_interval_s=1e-6,
        dev_games_per_opponent=4,
        keep_last_checkpoints=2,
        determinism_check_every_games=1_000,
        limits=ResourceLimits(disk_gb=50.0),
    )
    data.update(overrides)
    return RunConfig(**data)


def test_resume_continues_from_last_good_checkpoint(tmp_path: Path) -> None:
    config = _run_config(tmp_path, "resume-run")
    model = _tiny_model(seed=1)
    runner = SelfPlayRunner(
        config,
        engine=_FakeEngine(),
        learner=_FakeLearner(),
        evaluator=_fake_dev,
        model=model,
        vocab=_VOCAB,
    )
    first = runner.run(max_cycles=2)
    assert first.cycles == 2
    assert first.selfplay_games == 2
    assert first.last_checkpoint is not None
    assert first.last_checkpoint.name.startswith("ckpt-")
    assert len(runner.buffer) == 2
    assert first.best_dev_win_rate == pytest.approx(0.5)
    assert runner.stopping.best_dev_win_rate == pytest.approx(0.5)
    state = runner.manager.load_state()
    assert state is not None and state["cycle"] == 2
    assert state["seed_cursor"] == 2
    assert state["pool"]
    state["determinism_checks_done"] = 2
    runner.manager.state_path.write_text(json.dumps(state), encoding="utf-8")

    resumed_model = _tiny_model(seed=999)
    resumed_learner = _FakeLearner()
    resumed = SelfPlayRunner(
        config,
        engine=_FakeEngine(),
        learner=resumed_learner,
        evaluator=_fake_dev,
        model=resumed_model,
        vocab=_VOCAB,
    )
    second = resumed.run(max_cycles=3, resume=True)
    assert second.cycles == 3
    assert second.selfplay_games == 3
    assert resumed.cycle == 3
    assert resumed.learner.step == 3  # type: ignore[union-attr]
    assert len(resumed.buffer) == 3
    assert resumed.best_dev_win_rate == pytest.approx(0.5)
    assert resumed.stopping.best_dev_win_rate == pytest.approx(0.5)
    assert resumed._determinism_checks_done == 2
    assert resumed.engine.seed_cursor == 3  # type: ignore[attr-defined]
    assert resumed.engine.opponents.pool  # type: ignore[attr-defined]

    def weights(path: Path) -> dict[str, torch.Tensor]:
        payload = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=False)
        return payload["model_state_dict"]

    step2 = weights(first.last_checkpoint)
    step3 = weights(second.last_checkpoint)
    for key in step2:
        assert torch.equal(step2[key], step3[key]), key

    with pytest.raises(RunResumeError):
        SelfPlayRunner(
            config,
            engine=_FakeEngine(),
            learner=_FakeLearner(),
            evaluator=_fake_dev,
            model=_tiny_model(seed=3),
            vocab=_VOCAB,
        ).run(max_cycles=1)


# --- 5. run outputs stay out of git ------------------------------------
def test_run_outputs_are_gitignored(tmp_path: Path) -> None:
    import subprocess

    repo = Path(__file__).resolve().parents[1]
    assert "/runs/" in (repo / ".gitignore").read_text(encoding="utf-8").splitlines()
    result = subprocess.run(
        ["git", "check-ignore", "-q", "runs/selfplay-test/metrics.jsonl"],
        cwd=repo,
        check=False,
    )
    assert result.returncode == 0


# --- checkpoints / buffer / monitoring ---------------------------------
def test_checkpoint_retention_keeps_best_and_last_ten(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path / "run", keep_last=10, disk_limit_gb=50.0)
    for _ in range(14):
        target = manager.checkpoint_dir(manager._save_counter)
        manager._save_counter += 1
        target.mkdir(parents=True)
        (target / "checkpoint.pt").write_bytes(b"x" * 16)
    manager.best_dir.mkdir(parents=True, exist_ok=True)
    (manager.best_dir / "checkpoint.pt").write_bytes(b"best")
    manager.prune()
    remaining = [p.name for p in manager.list_checkpoints()]
    assert remaining == [f"ckpt-{counter:09d}" for counter in range(5, 15)]
    assert (manager.best_dir / "checkpoint.pt").read_bytes() == b"best"


def test_checkpoint_names_are_monotonic_even_with_same_step(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path / "run", keep_last=10, disk_limit_gb=50.0)
    model = _tiny_model()
    first = manager.save(step=0, model=model, vocab=_VOCAB, metadata={})
    second = manager.save(step=0, model=model, vocab=_VOCAB, metadata={})
    assert first.name == "ckpt-000000001"
    assert second.name == "ckpt-000000002"
    assert first.exists() and second.exists()


def test_disk_cap_prunes_oldest_then_raises(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path / "run", keep_last=10, disk_limit_gb=1e-9)
    for counter in range(1, 4):
        target = manager.checkpoint_dir(counter)
        target.mkdir(parents=True)
        (target / "checkpoint.pt").write_bytes(b"x" * 4096)
    removed = manager.enforce_disk_cap()
    assert removed
    assert manager.list_checkpoints() == []
    (manager.monitor_dir / "metrics.jsonl").write_bytes(b"x" * 4096)
    from tm_bot.runs.errors import RunResourceError

    with pytest.raises(RunResourceError):
        manager.enforce_disk_cap()


def test_replay_buffer_ring_eviction_and_collation() -> None:
    buffer = ReplayBuffer(capacity=2)
    for _ in range(3):
        buffer.add(_encoded_sample())
    assert len(buffer) == 2
    batch, targets, values = collate_samples(buffer.sample_batch(4, rng=np.random.default_rng(0)))
    assert batch["cand_mask"].shape[0] == 4
    assert targets.shape == (4, 2)
    assert values.shape == (4,)
    assert torch.allclose(targets.sum(dim=-1), torch.ones(4))


def test_monitor_logger_writes_jsonl_and_summary(tmp_path: Path) -> None:
    logger = MonitorLogger(tmp_path / "monitor" / "metrics.jsonl")
    logger.log_cycle(
        CycleMetrics(
            cycle=1,
            elapsed_s=10.0,
            wall_clock_s=10.0,
            selfplay_games=16,
            total_selfplay_games=16,
            decisions=320,
            searched_decisions=64,
            searched_share=0.2,
            rejections=0,
            failures=0,
            wins=8,
            losses=6,
            ties=2,
            mean_generations=12.0,
            opponent_kinds={"latest": 12, "b1": 4},
            policy_entropy=1.5,
            target_policy_entropy=1.1,
            policy_loss=1.2,
            value_loss=0.7,
            value_calibration={"n": 32, "brier": 0.2, "ece": 0.05, "wins": 16, "losses": 16, "ties": 0},
            throughput_games_per_hour=5760.0,
        )
    )
    logger.log_checkpoint(
        CheckpointMetrics(
            cycle=1,
            elapsed_s=10.0,
            step=1,
            games=16,
            checkpoint="ckpt-000000001",
            dev_games=100,
            dev_win_rate_vs_b1=0.55,
            dev_win_rate_vs_b1_ci_95=(0.45, 0.65),
            dev_win_rate_vs_previous_best=0.5,
            dev_win_rate_vs_previous_best_ci_95=(0.4, 0.6),
            dev_mean_vp_margin_vs_b1=3.0,
            new_best=True,
            best_dev_win_rate_vs_b1=0.55,
        )
    )
    summary_path = logger.write_summary(tmp_path / "monitor" / "summary.json")
    reloaded = MonitorLogger(tmp_path / "monitor" / "metrics.jsonl")
    reloaded.load_existing()
    summary = reloaded.summary()
    assert summary_path.exists()
    assert summary["cycles"] == 1
    assert summary["checkpoints"] == 1
    assert summary["selfplay_games"] == 16
    assert summary["wins"] == 8 and summary["losses"] == 6 and summary["ties"] == 2
    assert summary["opponent_kinds"] == {"latest": 12, "b1": 4}
    assert summary["policy_entropy"]["mean"] == pytest.approx(1.5)
    assert summary["value_calibration"]["mean_brier"] == pytest.approx(0.2)
    lines = (tmp_path / "monitor" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_game_shape_flags_outside_six_to_twenty_five_generations() -> None:
    games = [
        GameSummary("train-0", "tharsis", True, 4, 100),
        GameSummary("train-1", "tharsis", True, 12, 100),
        GameSummary("train-2", "tharsis", True, 30, 100),
        GameSummary("train-3", "tharsis", False, None, 5, error_type="AdapterBugError"),
    ]
    flags = game_shape_flags(games, min_generations=6, max_generations=25)
    assert [flag["seed_key"] for flag in flags] == ["train-0", "train-2"]


def test_replay_from_seed_and_actions_matches_finished_game() -> None:
    recorder = GameRecorder("train-0042", "tharsis", capture_seat=None)
    agents = (
        capture_agent(B0Agent(seed=11), recorder, capture_seat=None),
        capture_agent(B0Agent(seed=12), recorder, capture_seat=None),
    )
    outcome = play_game(agents, "train-0042", "tharsis")  # type: ignore[arg-type]
    assert outcome.ok and outcome.result is not None
    assert len(recorder.actions) == outcome.decisions
    original = GameSummary(
        seed_key=outcome.seed_key,
        map_name=outcome.map,
        ok=True,
        generations=outcome.generations,
        decisions=outcome.decisions,
        scores=(outcome.result.scores[0], outcome.result.scores[1]),
        winner=outcome.result.winner,
        tie=outcome.result.tie,
    )
    replay = replay_game(original.seed_key, original.map_name, recorder.actions)
    assert compare_replay(original, replay).ok is True


def test_play_games_deadline_reports_partial_games() -> None:
    specs = [
        SimpleNamespace(
            seed_key="train-0099",
            map_name="tharsis",
            agents=(B0Agent(seed=1), B0Agent(seed=2)),
            seat_seeds=None,
        )
    ]
    outcomes = play_games(specs, num_workers=1, stop_when=lambda: True)  # type: ignore[arg-type]
    assert len(outcomes) == 1
    assert outcomes[0].ok is False
    assert outcomes[0].error_type == "DeadlineReached"


# --- fix-specific orchestrator tests ----------------------------------
def test_cycle_calibration_uses_frozen_pre_update_weights(tmp_path: Path) -> None:
    config = _run_config(tmp_path, "calibration-run")
    model = _tiny_model(seed=2)
    engine = _FakeEngine(value=-1.0)
    expected = compute_value_calibration(model, engine.play_batch(1).samples, device="cpu")

    learner = _FakeLearner(model=model)
    learner.set_win_probability = True
    runner = SelfPlayRunner(
        config,
        engine=engine,
        learner=learner,
        evaluator=None,
        model=model,
        vocab=_VOCAB,
    )
    runner.run(max_cycles=1)
    record = runner.logger.cycles[-1]
    calib = record.value_calibration
    assert calib["n"] == expected["n"]
    assert calib["losses"] == 1
    assert calib["brier"] == pytest.approx(expected["brier"], rel=1e-6)
    assert calib["brier"] < 0.99  # the post-update all-win model would score ~1.0


def test_tied_vs_b1_checkpoint_that_beats_previous_best_becomes_best(tmp_path: Path) -> None:
    def evaluator(candidate: Path, opponent: Path | None) -> DevResult:
        if opponent is None:
            return DevResult(16, 16, 0, 0.0, (0.0, 0.0), -20.0, {})
        return DevResult(16, 16, 0, 0.9, (0.75, 1.0), 10.0, {})

    ticks = iter(float(i) for i in range(10_000))
    config = _run_config(tmp_path, "tie-break-run", checkpoint_interval_s=0.5)
    runner = SelfPlayRunner(
        config,
        engine=_FakeEngine(),
        learner=_FakeLearner(),
        evaluator=evaluator,
        model=_tiny_model(seed=6),
        vocab=_VOCAB,
        clock=lambda: next(ticks),
    )
    runner.run(max_cycles=3)
    records = runner.logger.checkpoints
    assert len(records) == 3
    assert all(r.new_best for r in records)
    assert runner.stopping.checkpoints_without_improvement == 0
    best = runner.manager.best_dir / "checkpoint.pt"
    assert torch.load(best, map_location="cpu", weights_only=False)["metadata"]["cycle"] == records[-1].cycle


def test_soft_stop_does_not_recheckpoint_an_unchanged_model(tmp_path: Path) -> None:
    # Each clock read advances 100 s, so the dev eval of the last periodic
    # checkpoint ends > 60 s after it started; no training happens afterwards.
    ticks = iter(float(i) * 100.0 for i in range(10_000))
    config = _run_config(tmp_path, "no-dup-final", checkpoint_interval_s=1e9)
    runner = SelfPlayRunner(
        config,
        engine=_FakeEngine(),
        learner=_FakeLearner(),
        evaluator=_fake_dev,
        model=_tiny_model(seed=7),
        vocab=_VOCAB,
        clock=lambda: next(ticks),
    )
    runner.run(max_cycles=2)
    steps = [r.step for r in runner.logger.checkpoints]
    assert steps == sorted(set(steps))
    assert len(runner.manager.list_checkpoints()) == len(steps)


def test_first_checkpoint_uses_recorded_warmstart_as_previous_best(tmp_path: Path) -> None:
    warmstart = str(tmp_path / "warmstart")
    calls: list[tuple[Path, Path | None]] = []

    def evaluator(candidate: Path, opponent: Path | None) -> DevResult:
        calls.append((candidate, opponent))
        return _fake_dev(candidate, opponent)

    config = _run_config(tmp_path, "warmstart-run", warmstart_checkpoint=warmstart)
    runner = SelfPlayRunner(
        config,
        engine=_FakeEngine(),
        learner=_FakeLearner(),
        evaluator=evaluator,
        model=_tiny_model(seed=5),
        vocab=_VOCAB,
    )
    runner.run(max_cycles=1)
    assert len(calls) == 2
    assert calls[0][1] is None
    assert calls[1][1] == Path(warmstart)
    written = json.loads((config.run_dir / "config.json").read_text(encoding="utf-8"))
    assert written["warmstart_checkpoint"] == warmstart


def test_entropy_collapse_uses_fixed_network_probe(tmp_path: Path) -> None:
    config = _run_config(
        tmp_path,
        "entropy-run",
        entropy_collapse_fraction=0.10,
        entropy_collapse_window_s=7200.0,
    )
    model = _tiny_model(seed=7)
    runner = SelfPlayRunner(
        config,
        engine=_FakeEngine(),
        learner=_SharpeningLearner(model=model),
        evaluator=None,
        model=model,
        vocab=_VOCAB,
    )
    runner.run(max_cycles=3)
    cycles = runner.logger.cycles
    assert len(cycles) == 3
    assert cycles[0].policy_entropy is not None
    assert runner.stopping.entropy_baseline == pytest.approx(cycles[0].policy_entropy)
    assert cycles[-1].policy_entropy is not None
    assert cycles[-1].policy_entropy < cycles[0].policy_entropy
    assert cycles[-1].target_policy_entropy == pytest.approx(cycles[0].target_policy_entropy)
    # The probe is a network quantity, not the self-play target entropy.
    assert not pytest.approx(cycles[0].target_policy_entropy) == cycles[0].policy_entropy


def test_hard_failure_writes_regression_case_and_stops(tmp_path: Path) -> None:
    class _FailingEngine(_FakeEngine):
        def play_batch(self, num_games: int, *, stop_when=None) -> SelfPlayBatch:
            recorder = GameRecorder("train-0005", "tharsis", capture_seat=0)
            recorder.actions = [(0, 0), (1, 1)]
            return SelfPlayBatch(
                games=[
                    GameSummary(
                        seed_key="train-0005",
                        map_name="tharsis",
                        ok=False,
                        generations=None,
                        decisions=2,
                        error_type="AdapterBugError",
                        error="adapter rejected candidate",
                    )
                ],
                samples=[],
                recorders=[recorder],
                training_seats=[0],
                opponent_kinds=["latest"],
                decisions=2,
            )

    def _must_not_evaluate(_checkpoint: Path, _opponent: Path | None) -> DevResult:
        raise AssertionError("dev evaluation must not run after a hard stop")

    config = _run_config(tmp_path, "failure-run")
    runner = SelfPlayRunner(
        config,
        engine=_FailingEngine(),
        learner=_FakeLearner(),
        evaluator=_must_not_evaluate,
        model=_tiny_model(seed=4),
        vocab=_VOCAB,
    )
    summary = runner.run(max_cycles=5)
    assert summary.stop.kind is StopKind.ADAPTER_REJECTION
    cases = list((config.run_dir / "regression").glob("*.json"))
    assert len(cases) == 1
    payload = json.loads(cases[0].read_text(encoding="utf-8"))
    assert payload["seed_key"] == "train-0005"
    assert payload["error_type"] == "AdapterBugError"
    assert payload["actions"] == [[0, 0], [1, 1]]


def test_nonfinite_learner_rolls_back_and_stops(tmp_path: Path) -> None:
    config = _run_config(tmp_path, "nan-run")
    model = _tiny_model(seed=8)
    learner = _FakeLearner(model=model)
    runner = SelfPlayRunner(
        config,
        engine=_FakeEngine(),
        learner=learner,
        evaluator=None,
        model=model,
        vocab=_VOCAB,
    )
    runner.run(max_cycles=1)
    checkpoint = runner.last_checkpoint
    assert checkpoint is not None
    payload = torch.load(checkpoint / "checkpoint.pt", map_location="cpu", weights_only=False)
    good = payload["model_state_dict"]

    learner.nonfinite = True
    summary = runner.run(max_cycles=2, resume=True)
    assert summary.stop.kind is StopKind.NONFINITE_LOSS
    assert summary.stop.details["rolled_back"] is True
    for key, value in model.state_dict().items():
        assert torch.equal(value, good[key]), key
    final_payload = torch.load(
        runner.last_checkpoint / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    for key, value in final_payload["model_state_dict"].items():
        assert torch.isfinite(value).all(), key


def test_disk_cap_crash_becomes_resource_stop(tmp_path: Path) -> None:
    config = _run_config(
        tmp_path,
        "disk-run",
        limits=ResourceLimits(disk_gb=1e-9),
    )
    runner = SelfPlayRunner(
        config,
        engine=_FakeEngine(),
        learner=_FakeLearner(),
        evaluator=_fake_dev,
        model=_tiny_model(seed=9),
        vocab=_VOCAB,
    )
    summary = runner.run(max_cycles=1)
    assert summary.stop.stopped is True
    assert summary.stop.kind is StopKind.RESOURCE_LIMIT
    assert runner.manager.state_path.exists()


def test_monitoring_separates_rejections_and_records_game_mix(tmp_path: Path, monkeypatch) -> None:
    class _MixedEngine(_FakeEngine):
        def play_batch(self, num_games: int, *, stop_when=None) -> SelfPlayBatch:
            recorder = GameRecorder("train-0005", "tharsis", capture_seat=0)
            recorder.actions = [(0, 0)]
            games = [
                GameSummary("train-0005", "tharsis", True, 30, 100, (60, 50), 0, False),
                GameSummary("train-0006", "hellas", True, 12, 100, (40, 50), 1, False),
                GameSummary("train-0007", "elysium", True, 12, 100, (50, 50), None, True),
                GameSummary(
                    "train-0008",
                    "tharsis",
                    False,
                    None,
                    7,
                    error_type="AdapterBugError",
                    error="rejected",
                ),
            ]
            return SelfPlayBatch(
                games=games,
                samples=[_encoded_sample(value=1.0) for _ in range(3)],
                recorders=[recorder],
                training_seats=[0, 0, 0, 0],
                opponent_kinds=["latest", "pool", "b1", "latest"],
                decisions=400,
                searched_decisions=100,
            )

    from tm_bot.runs.monitoring import DeterminismResult

    monkeypatch.setattr(
        "tm_bot.runs.orchestrator.determinism_check",
        lambda original, actions: DeterminismResult(
            original.seed_key, original.map_name, True, "identical"
        ),
    )
    config = _run_config(
        tmp_path,
        "monitor-run",
        games_per_cycle=4,
        determinism_check_every_games=1,
    )
    runner = SelfPlayRunner(
        config,
        engine=_MixedEngine(),
        learner=_FakeLearner(),
        evaluator=None,
        model=_tiny_model(seed=10),
        vocab=_VOCAB,
    )
    summary = runner.run(max_cycles=5)
    assert summary.stop.kind is StopKind.ADAPTER_REJECTION
    record = runner.logger.cycles[-1]
    assert record.rejections == 1
    assert record.failures == 1
    assert (record.wins, record.losses, record.ties) == (1, 1, 1)
    assert record.opponent_kinds == {"latest": 2, "pool": 1, "b1": 1}
    assert record.searched_share == pytest.approx(0.25)
    assert record.shape_flag_count == 1
    assert record.determinism_checked is False  # hard stops do not replay a game
    assert record.vp_histogram


def test_determinism_spot_check_is_logged(tmp_path: Path, monkeypatch) -> None:
    class _RecorderEngine(_FakeEngine):
        def play_batch(self, num_games: int, *, stop_when=None) -> SelfPlayBatch:
            games = []
            recorders = []
            for i in range(num_games):
                seed = f"train-{self.seed_cursor:04d}"
                self.seed_cursor += 1
                games.append(
                    GameSummary(seed, "tharsis", True, 12, 120, (60, 50), 0, False)
                )
                recorder = GameRecorder(seed, "tharsis", capture_seat=0)
                recorder.actions = [(0, 0)]
                recorders.append(recorder)
            return SelfPlayBatch(
                games=games,
                samples=[_encoded_sample(value=1.0) for _ in games],
                recorders=recorders,
                training_seats=[0 for _ in games],
                opponent_kinds=["latest" for _ in games],
                decisions=120 * num_games,
                searched_decisions=20 * num_games,
            )

    from tm_bot.runs.monitoring import DeterminismResult

    monkeypatch.setattr(
        "tm_bot.runs.orchestrator.determinism_check",
        lambda original, actions: DeterminismResult(
            original.seed_key, original.map_name, True, "identical"
        ),
    )
    config = _run_config(
        tmp_path, "determinism-run", games_per_cycle=2, determinism_check_every_games=1
    )
    runner = SelfPlayRunner(
        config,
        engine=_RecorderEngine(),
        learner=_FakeLearner(),
        evaluator=None,
        model=_tiny_model(seed=12),
        vocab=_VOCAB,
    )
    runner.run(max_cycles=2)
    cycles = runner.logger.cycles
    assert cycles[0].determinism_checked is True
    assert cycles[0].determinism_ok is True
    assert runner.logger.summary()["determinism_checks"] == 2


def test_utility_for_seat_rejects_unknown_winner() -> None:
    with pytest.raises(ValueError, match="no declared winner"):
        utility_for_seat(
            GameSummary("train-0", "tharsis", True, 12, 10, (50, 50), None, False), 0
        )
    assert utility_for_seat(
        GameSummary("train-0", "tharsis", True, 12, 10, (50, 50), None, True), 1
    ) == 0.0


def test_dev_agent_cache_is_bounded(tmp_path: Path) -> None:
    clear_dev_agent_cache()
    paths = []
    for index in range(3):
        target = tmp_path / f"ckpt-{index}"
        save_checkpoint(target, _tiny_model(seed=index), _VOCAB)
        paths.append(target)
    for target in paths:
        cached_policy_value_agent(target, "cpu")
    info = _load_policy_value_agent.cache_info()
    assert info.maxsize == 2
    assert info.currsize <= 2
    clear_dev_agent_cache()


def test_pool_agent_cache_is_trimmed(tmp_path: Path) -> None:
    from tm_bot.runs.selfplay import SelfPlayEngine

    config = RunConfig(run_name="pool", output_root=tmp_path, workers=1)
    engine = SelfPlayEngine(config, model=_tiny_model(), vocab=_VOCAB)
    keep = tmp_path / "a"
    engine._pool_agents = {str(keep): object(), str(tmp_path / "b"): object()}  # type: ignore[assignment]
    engine.set_opponent_pool([keep])
    assert set(engine._pool_agents) == {str(keep)}
    assert engine.opponents.pool == (keep,)


def test_params_are_finite_detects_nan() -> None:
    from tm_bot.runs.learner import params_are_finite

    model = _tiny_model(seed=13)
    assert params_are_finite(model) is True
    with torch.no_grad():
        model.obs_norm.weight[0] = float("nan")
    assert params_are_finite(model) is False


def test_orchestrator_deadline_stop(tmp_path: Path) -> None:
    class _DeadlineEngine(_FakeEngine):
        def play_batch(self, num_games: int, *, stop_when=None) -> SelfPlayBatch:
            self.seed_cursor += num_games
            return SelfPlayBatch(
                games=[
                    GameSummary(
                        "train-0100",
                        "tharsis",
                        False,
                        None,
                        10,
                        error_type="DeadlineReached",
                        error="deadline",
                    )
                ],
                samples=[],
                recorders=[GameRecorder("train-0100", "tharsis")],
                training_seats=[0],
                opponent_kinds=["latest"],
                decisions=10,
            )

    config = _run_config(tmp_path, "deadline-run", games_per_cycle=1)
    runner = SelfPlayRunner(
        config,
        engine=_DeadlineEngine(),
        learner=_FakeLearner(),
        evaluator=None,
        model=_tiny_model(seed=11),
        vocab=_VOCAB,
    )
    summary = runner.run(max_cycles=5)
    assert summary.stop.kind is StopKind.BUDGET
    assert summary.selfplay_games == 0
    assert runner.logger.cycles[-1].failures == 0


def test_run_without_dev_eval_does_not_duplicate_the_final_checkpoint(tmp_path: Path) -> None:
    config = _run_config(tmp_path, "no-dup-no-eval", dev_eval_enabled=False)
    runner = SelfPlayRunner(
        config,
        engine=_FakeEngine(),
        learner=_FakeLearner(),
        evaluator=None,
        model=_tiny_model(seed=8),
        vocab=_VOCAB,
    )
    runner.run(max_cycles=2)
    steps = [
        torch.load(p / "checkpoint.pt", map_location="cpu", weights_only=False)["metadata"]["step"]
        for p in runner.manager.list_checkpoints()
    ]
    assert steps == sorted(set(steps))
    assert steps and steps[-1] == runner.steps


def test_first_checkpoint_without_previous_best_logs_no_previous_best_rate(
    tmp_path: Path,
) -> None:
    calls: list[Path | None] = []

    def evaluator(candidate: Path, opponent: Path | None) -> DevResult:
        calls.append(opponent)
        return _fake_dev(candidate, opponent)

    config = _run_config(tmp_path, "no-prev-best")
    runner = SelfPlayRunner(
        config,
        engine=_FakeEngine(),
        learner=_FakeLearner(),
        evaluator=evaluator,
        model=_tiny_model(seed=9),
        vocab=_VOCAB,
    )
    runner.run(max_cycles=1)
    first = runner.logger.checkpoints[0]
    assert calls[0] is None
    assert first.dev_win_rate_vs_previous_best is None
    assert first.dev_win_rate_vs_previous_best_ci_95 is None


class _NoUpdateLearner(_FakeLearner):
    def train(self, buffer, *, updates=None) -> LearnerStats:
        return LearnerStats(updates=0)


def test_periodic_checkpoint_skips_a_model_that_did_not_train(tmp_path: Path) -> None:
    # Each clock read advances 100 s, so every cycle is past the checkpoint interval.
    ticks = iter(float(i) * 100.0 for i in range(10_000))
    config = _run_config(tmp_path, "no-dup-periodic")
    runner = SelfPlayRunner(
        config,
        engine=_FakeEngine(),
        learner=_NoUpdateLearner(),
        evaluator=_fake_dev,
        model=_tiny_model(seed=10),
        vocab=_VOCAB,
        clock=lambda: next(ticks),
    )
    runner.run(max_cycles=3)
    assert runner.steps == 0
    assert len(runner.manager.list_checkpoints()) == 1
    assert [r.step for r in runner.logger.checkpoints] == [0]


class _ClockedEngine(_FakeEngine):
    """Each self-play batch advances the shared fake clock by `batch_s`."""

    def __init__(self, now: list[float], batch_s: float) -> None:
        super().__init__()
        self.now = now
        self.batch_s = batch_s

    def play_batch(self, num_games: int, *, stop_when=None) -> SelfPlayBatch:
        self.now[0] += self.batch_s
        return super().play_batch(num_games, stop_when=stop_when)


def _clocked_dev(now: list[float], eval_s: float):
    def evaluator(candidate: Path, opponent: Path | None) -> DevResult:
        now[0] += eval_s
        return _fake_dev(candidate, opponent)

    return evaluator


def test_budget_and_checkpoint_interval_run_on_the_training_clock(tmp_path: Path) -> None:
    # Self-play cycles take 100 s; every dev pairing takes 1,000 s of wall clock.
    now = [0.0]
    config = _run_config(
        tmp_path,
        "training-clock",
        wall_clock_budget_s=550.0,
        checkpoint_interval_s=200.0,
        keep_last_checkpoints=10,
    )
    runner = SelfPlayRunner(
        config,
        engine=_ClockedEngine(now, 100.0),
        learner=_FakeLearner(),
        evaluator=_clocked_dev(now, 1_000.0),
        model=_tiny_model(seed=12),
        vocab=_VOCAB,
        clock=lambda: now[0],
    )
    summary = runner.run()
    assert summary.stop.kind is StopKind.BUDGET
    # Dev eval is off the clock: six 100 s cycles fit the 550 s training budget.
    assert summary.cycles == 6
    assert summary.training_s == pytest.approx(600.0)
    checkpoints = runner.logger.checkpoints
    # Checkpoints every 200 s of training time: after cycles 2, 4 and 6.
    assert [c.cycle for c in checkpoints] == [2, 4, 6]
    assert [c.training_s for c in checkpoints] == pytest.approx([200.0, 400.0, 600.0])
    # The first checkpoint has no previous best, so it plays only the vs-B1 pairing.
    assert summary.dev_eval_s == pytest.approx(1_000.0 + 2_000.0 + 2_000.0)
    assert summary.elapsed_s == pytest.approx(summary.training_s + summary.dev_eval_s)
    cycles = runner.logger.cycles
    assert cycles[-1].training_s == pytest.approx(600.0)
    assert cycles[-1].dev_eval_s == pytest.approx(3_000.0)
    written = json.loads((config.run_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert written["training_s"] == pytest.approx(600.0)
    assert written["dev_eval_s"] == pytest.approx(5_000.0)
    monitor_summary = runner.logger.summary()
    assert monitor_summary["training_s"] == pytest.approx(600.0)
    assert monitor_summary["dev_eval_s"] == pytest.approx(5_000.0)
    # Both summaries report the same total wall clock, including the last dev eval.
    assert monitor_summary["elapsed_s"] == pytest.approx(summary.elapsed_s)


def test_entropy_window_runs_on_the_training_clock(tmp_path: Path) -> None:
    now = [0.0]
    config = _run_config(
        tmp_path,
        "entropy-clock",
        checkpoint_interval_s=100.0,
        entropy_collapse_window_s=250.0,
    )
    runner = SelfPlayRunner(
        config,
        engine=_ClockedEngine(now, 100.0),
        learner=_FakeLearner(),
        evaluator=_clocked_dev(now, 1_000.0),
        model=_tiny_model(seed=13),
        vocab=_VOCAB,
        clock=lambda: now[0],
    )
    runner.run(max_cycles=2)
    # Two 100 s cycles: entropy was last checked at 200 s of training time, inside the
    # 250 s window, although the wall clock is past 1,000 s after the first dev eval.
    assert runner.stopping.entropy_checked_at_s == pytest.approx(200.0)


def test_resume_keeps_the_training_and_dev_eval_clocks(tmp_path: Path) -> None:
    now = [0.0]
    config = _run_config(tmp_path, "clock-resume", checkpoint_interval_s=100.0)

    def make_runner() -> SelfPlayRunner:
        return SelfPlayRunner(
            config,
            engine=_ClockedEngine(now, 100.0),
            learner=_FakeLearner(),
            evaluator=_clocked_dev(now, 1_000.0),
            model=_tiny_model(seed=14),
            vocab=_VOCAB,
            clock=lambda: now[0],
        )

    first = make_runner()
    first.run(max_cycles=1)
    assert first.training_s() == pytest.approx(100.0)
    assert first.dev_eval_s == pytest.approx(1_000.0)
    now[0] += 50_000.0  # downtime between sessions is on neither clock
    resumed = make_runner()
    resumed.run(max_cycles=2, resume=True)
    assert resumed.training_s() == pytest.approx(200.0)
    assert resumed.dev_eval_s == pytest.approx(3_000.0)
    state = json.loads((config.run_dir / "state.json").read_text(encoding="utf-8"))
    assert state["dev_eval_s"] == pytest.approx(3_000.0)
    assert state["training_s"] == pytest.approx(200.0)


class _ClockedLearner(_FakeLearner):
    def __init__(self, now: list[float], step_s: float) -> None:
        super().__init__()
        self.now = now
        self.step_s = step_s

    def train(self, buffer, *, updates=None) -> LearnerStats:
        self.now[0] += self.step_s
        return super().train(buffer, updates=updates)


def test_cycle_records_split_wall_clock_into_training_and_dev_eval(tmp_path: Path) -> None:
    now = [0.0]
    config = _run_config(tmp_path, "clock-identity", checkpoint_interval_s=100.0)
    runner = SelfPlayRunner(
        config,
        engine=_ClockedEngine(now, 100.0),
        learner=_ClockedLearner(now, 40.0),
        evaluator=_clocked_dev(now, 1_000.0),
        model=_tiny_model(seed=15),
        vocab=_VOCAB,
        clock=lambda: now[0],
    )
    runner.run(max_cycles=3)
    for record in [*runner.logger.cycles, *runner.logger.checkpoints]:
        assert record.elapsed_s == pytest.approx(record.training_s + record.dev_eval_s)


def test_raising_dev_eval_still_counts_as_dev_eval_time(tmp_path: Path) -> None:
    now = [0.0]

    def evaluator(candidate: Path, opponent: Path | None) -> DevResult:
        now[0] += 1_000.0
        raise RuntimeError("dev bridge crashed")

    config = _run_config(tmp_path, "clock-raise", checkpoint_interval_s=100.0)
    runner = SelfPlayRunner(
        config,
        engine=_ClockedEngine(now, 100.0),
        learner=_FakeLearner(),
        evaluator=evaluator,
        model=_tiny_model(seed=16),
        vocab=_VOCAB,
        clock=lambda: now[0],
    )
    with pytest.raises(RuntimeError, match="dev bridge crashed"):
        runner.run(max_cycles=1)
    assert runner.dev_eval_s == pytest.approx(1_000.0)
    assert runner.training_s() == pytest.approx(100.0)
