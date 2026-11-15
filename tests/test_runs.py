"""Tests for self-play run orchestration (#24)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from tm_bot.agents import B0Agent, play_game
from tm_bot.agents.card_data import load_card_data
from tm_bot.env.types import Candidate, Choice
from tm_bot.eval.seeds import SeedKey
from tm_bot.models.encode import encode_choice, encode_observation
from tm_bot.models.network import TMModelConfig, TMNetwork
from tm_bot.models.vocab import CardVocab
from tm_bot.runs.buffer import DecisionSample, ReplayBuffer, collate_samples
from tm_bot.runs.checkpoints import CheckpointManager
from tm_bot.runs.config import ResourceLimits, RunConfig
from tm_bot.runs.deveval import DevResult, build_dev_games
from tm_bot.runs.errors import RunResumeError
from tm_bot.runs.learner import LearnerStats
from tm_bot.runs.monitoring import (
    CheckpointMetrics,
    CycleMetrics,
    GameSummary,
    MonitorLogger,
    compare_replay,
    game_shape_flags,
)
from tm_bot.runs.orchestrator import SelfPlayRunner
from tm_bot.runs.recorder import DecisionRecord, GameRecorder, RecordingAgent, capture_agent
from tm_bot.runs.selfplay import (
    OpponentSampler,
    SelfPlayBatch,
    assert_training_seed_keys,
    build_selfplay_specs,
    make_train_seed_keys,
    replay_game,
)
from tm_bot.runs.stopping import StopKind, StoppingMonitor

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
            # Decoy fields: only the opponent's hand differs between the twins.
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
        kind="branch",
        prompt="Take your first action",
        candidates=(Candidate("a", "A"), Candidate("b", "B")),
    )


def _encoded_sample() -> DecisionSample:
    observation = _obs(actor=0)
    choice = _choice()
    return DecisionSample(
        observation=encode_observation(observation, _VOCAB),  # type: ignore[arg-type]
        choice=encode_choice(choice, observation, _VOCAB),  # type: ignore[arg-type]
        policy_target=np.array([0.75, 0.25], dtype=np.float32),
        value_target=1.0,
        seat=0,
        was_searched=True,
        seed_key="train-0000",
        map_name="tharsis",
        generation=2,
    )


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
        checkpoint=f"step-{cycle:09d}",
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
    # Within 15 points of the 0.80 best: no streak.
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
    # After the two-hour window the same entropy does not trigger collapse.
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
        )


def test_selfplay_specs_use_train_seeds_and_one_capture_seat() -> None:
    keys = make_train_seed_keys(0, 3)
    sampler = OpponentSampler(b1_prob=1.0, latest_prob=0.0, pool_prob=0.0)
    specs, recorders, seats = build_selfplay_specs(
        seed_keys=keys,
        maps=["tharsis", "hellas"],
        learner_factory=lambda seed: B0Agent(seed=seed),
        sampler=sampler,
        opponent_factory=lambda choice, seed: B0Agent(seed=seed),
    )
    assert all(spec.seed_key.startswith("train-") for spec in specs)
    assert seats == [0, 1, 0]
    assert [r.capture_seat for r in recorders] == [0, 1, 0]
    assert [r.seed_key for r in recorders] == [k.key for k in keys]


def test_dev_games_are_exactly_n_and_dev_namespaced() -> None:
    games = build_dev_games(100, maps=["tharsis", "elysium", "hellas", "vastitas borealis nova"])
    assert len(games) == 100
    assert all(seed.startswith("dev-") for seed, _map, _seat in games)
    assert sum(1 for _s, _m, seat in games if seat == 0) == 50
    assert sum(1 for _s, _m, seat in games if seat == 1) == 50


# --- leakage -----------------------------------------------------------
def test_opponent_private_cards_never_enter_the_acting_seat_encoding() -> None:
    # From seat 0, both worlds look identical: the opponent's private cards are not in
    # the actor observation at all. (Only the public hand *count* is visible.)
    obs_a = _obs(actor=0, opponent_hand=("Algae",))
    obs_b = _obs(actor=0, opponent_hand=("Birds", "Debris"))
    enc_a = encode_observation(obs_a, _VOCAB)  # type: ignore[arg-type]
    enc_b = encode_observation(obs_b, _VOCAB)  # type: ignore[arg-type]
    assert np.array_equal(enc_a.obs_scalars, enc_b.obs_scalars)
    assert np.array_equal(enc_a.card_ids, enc_b.card_ids)
    assert np.array_equal(enc_a.card_dyn, enc_b.card_dyn)
    assert np.array_equal(enc_a.space_feats, enc_b.space_feats)

    # From seat 1 those same cards are its own private data, so the encoding differs:
    # the encoder is seat-scoped, and the recorder only ever stores the acting seat view.
    own_a = encode_observation(_obs(actor=1, opponent_hand=("Algae",)), _VOCAB)  # type: ignore[arg-type]
    own_b = encode_observation(_obs(actor=1, opponent_hand=("Birds", "Debris")), _VOCAB)  # type: ignore[arg-type]
    assert not np.array_equal(own_a.obs_scalars, own_b.obs_scalars)


def test_recorder_captures_only_the_training_seat() -> None:
    recorder = GameRecorder("train-0000", "tharsis", capture_seat=0)
    learner = RecordingAgent(B0Agent(seed=1), recorder, capture_seat=0)
    opponent = RecordingAgent(B0Agent(seed=2), recorder, capture_seat=None)
    learner.act(_obs(actor=0), _choice(seat=0))  # type: ignore[arg-type]
    opponent.act(_obs(actor=1), _choice(seat=1))  # type: ignore[arg-type]
    assert len(recorder.actions) == 2
    assert len(recorder.decisions) == 1
    assert int(recorder.decisions[0].observation["actor"]) == 0

    # An observation that belongs to another seat can never become a training row.
    from types import SimpleNamespace

    from tm_bot.runs.selfplay import SelfPlayEngine

    config = RunConfig(run_name="leak", output_root=Path("unused"), workers=1)
    engine = SelfPlayEngine(config, model=_tiny_model(), vocab=_VOCAB)
    bad = GameRecorder("train-0001", "tharsis", capture_seat=0)
    bad.decisions.append(
        DecisionRecord(
            observation=_obs(actor=1),  # type: ignore[arg-type]
            choice=_choice(seat=1),
            candidate_index=0,
            policy=np.array([1.0, 0.0]),
            was_searched=False,
            seat=1,
            generation=2,
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


# --- 3. resume ---------------------------------------------------------
class _FakeEngine:
    def __init__(self) -> None:
        from types import SimpleNamespace

        self.seed_cursor = 0
        self.opponents = SimpleNamespace(pool=())

    def set_opponent_pool(self, checkpoints) -> None:
        self.opponents.pool = tuple(Path(p) for p in checkpoints)

    def play_batch(self, num_games: int) -> SelfPlayBatch:
        games = []
        for i in range(num_games):
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
            samples=[_encoded_sample() for _ in games],
            training_seats=[0 for _ in games],
            decisions=120 * num_games,
            searched_decisions=20 * num_games,
        )


class _FakeLearner:
    def __init__(self) -> None:
        self.step = 0

    def train(self, buffer, *, updates=None) -> LearnerStats:
        self.step += 1
        return LearnerStats(
            updates=1,
            policy_loss=1.2,
            value_loss=0.7,
            total_loss=1.9,
            policy_entropy=1.5,
            value_brier=0.24,
            value_accuracy=0.55,
        )


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


def _run_config(tmp_path: Path, name: str) -> RunConfig:
    return RunConfig(
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


def test_resume_continues_from_last_good_checkpoint(tmp_path: Path) -> None:
    config = _run_config(tmp_path, "resume-run")
    model = _tiny_model(seed=1)
    learner = _FakeLearner()
    runner = SelfPlayRunner(
        config,
        engine=_FakeEngine(),
        learner=learner,
        evaluator=_fake_dev,
        model=model,
        vocab=_VOCAB,
    )
    first = runner.run(max_cycles=2)
    assert first.cycles == 2
    assert first.selfplay_games == 2
    assert first.last_checkpoint is not None
    assert first.last_checkpoint.name == "step-000000002"
    state = runner.manager.load_state()
    assert state is not None and state["cycle"] == 2

    # A fresh model must be replaced by the last good checkpoint on resume.
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

    def weights(path: Path) -> dict[str, torch.Tensor]:
        payload = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=False)
        return payload["model_state_dict"]

    step2 = weights(first.last_checkpoint)
    step3 = weights(second.last_checkpoint)
    for key in step2:
        assert torch.equal(step2[key], step3[key]), key

    # Starting over in the same directory without --resume is refused.
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
    for step in range(1, 15):
        target = manager.checkpoint_dir(step)
        target.mkdir(parents=True)
        (target / "checkpoint.pt").write_bytes(b"x" * 16)
    manager.best_dir.mkdir(parents=True, exist_ok=True)
    (manager.best_dir / "checkpoint.pt").write_bytes(b"best")
    manager.prune()
    remaining = [p.name for p in manager.list_checkpoints()]
    assert remaining == [f"step-{step:09d}" for step in range(5, 15)]
    assert (manager.best_dir / "checkpoint.pt").read_bytes() == b"best"


def test_disk_cap_prunes_oldest_then_raises(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path / "run", keep_last=10, disk_limit_gb=1e-9)
    for step in range(1, 4):
        target = manager.checkpoint_dir(step)
        target.mkdir(parents=True)
        (target / "checkpoint.pt").write_bytes(b"x" * 4096)
    removed = manager.enforce_disk_cap()
    assert removed
    assert manager.list_checkpoints() == []
    # A file that cannot be pruned keeps the run over its cap.
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
            rejections=0,
            failures=0,
            mean_generations=12.0,
            policy_entropy=1.5,
            policy_loss=1.2,
            value_loss=0.7,
            throughput_games_per_hour=5760.0,
        )
    )
    logger.log_checkpoint(
        CheckpointMetrics(
            cycle=1,
            elapsed_s=10.0,
            step=1,
            games=16,
            checkpoint="step-000000001",
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
    assert summary["policy_entropy"]["mean"] == pytest.approx(1.5)
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


def test_hard_failure_writes_regression_case_and_stops(tmp_path: Path) -> None:
    class _FailingEngine(_FakeEngine):
        def play_batch(self, num_games: int) -> SelfPlayBatch:
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
    assert summary.stop.stopped is True
    assert summary.stop.kind is StopKind.ADAPTER_REJECTION
    cases = list((config.run_dir / "regression").glob("*.json"))
    assert len(cases) == 1
    import json

    payload = json.loads(cases[0].read_text(encoding="utf-8"))
    assert payload["seed_key"] == "train-0005"
    assert payload["map_name"] == "tharsis"
    assert payload["error_type"] == "AdapterBugError"
    assert payload["actions"] == [[0, 0], [1, 1]]


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
