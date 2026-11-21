"""Tests for the Jev-inspired comparator agent (#25)."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tm_bot.agents.card_data import load_card_data
from tm_bot.agents.protocol import GameRunResult
from tm_bot.comparator.agent import (
    EVAL_ACTION_TEMPERATURE,
    ComparatorAgent,
    comparator_eval_factory,
)
from tm_bot.comparator.calibration import (
    calibration_metrics,
    fit_temperature,
    write_calibration_report,
)
from tm_bot.comparator.model import (
    ComparatorConfig,
    ComparatorNetwork,
    load_comparator_checkpoint,
    save_comparator_checkpoint,
)
from tm_bot.comparator.ppo import (
    PPOBuffer,
    PPOLearner,
    PPOSample,
    collate_ppo_samples,
    compute_gae,
    compute_surrogate_loss,
)
from tm_bot.comparator.ppo import _policy_terms as ppo_policy_terms
from tm_bot.comparator.selfplay import ComparatorSelfPlayEngine
from tm_bot.env.types import GameResult
from tm_bot.eval.harness import resolve_agent_factory
from tm_bot.models.encode import collate_encoded_batch, encode_choice, encode_observation
from tm_bot.models.network import TMModelConfig, TMNetwork
from tm_bot.models.vocab import CardVocab
from tm_bot.runs.buffer import normalize_policy_target
from tm_bot.runs.config import ResourceLimits, RunConfig
from tm_bot.runs.deveval import DevResult
from tm_bot.runs.learner import LearnerStats
from tm_bot.runs.orchestrator import SelfPlayRunner
from tm_bot.runs.recorder import GameRecorder
from tm_bot.runs.selfplay import GameSummary, SelfPlayBatch
from tm_bot.training.warmstart_builder import build_corp_choice

_CARDS = load_card_data()
_VOCAB = CardVocab.from_card_data(_CARDS)


def _tiny_comparator(*, seed: int = 0, dropout: float = 0.0) -> ComparatorNetwork:
    cfg = ComparatorConfig(
        vocab_size=_VOCAB.size,
        d_model=64,
        card_hidden=64,
        card_emb_dim=16,
        card_static_proj_dim=16,
        slot_emb_dim=8,
        card_dyn_proj_dim=8,
        space_hidden=16,
        choice_hidden=64,
        cand_hidden=64,
        value_hidden=64,
        dropout=dropout,
    )
    torch.manual_seed(seed)
    return ComparatorNetwork(cfg, torch.from_numpy(_VOCAB.static_features.copy()))


def _tiny_main(*, seed: int = 0) -> TMNetwork:
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


def _obs_choice():
    return build_corp_choice(
        "tharsis",
        0,
        offered_corps=("CrediCor", "Ecoline", "Helion"),
        offered_preludes=(),
        offered_projects=(),
    )


def _assert_encoded_observation_equal(actual, expected) -> None:
    assert np.array_equal(actual.obs_scalars, expected.obs_scalars)
    assert np.array_equal(actual.card_ids, expected.card_ids)
    assert np.array_equal(actual.card_slots, expected.card_slots)
    assert np.array_equal(actual.card_dyn, expected.card_dyn)
    assert np.array_equal(actual.space_feats, expected.space_feats)


def _assert_encoded_choice_equal(actual, expected) -> None:
    assert actual.prompt_type_id == expected.prompt_type_id
    assert np.array_equal(actual.choice_ctx, expected.choice_ctx)
    assert np.array_equal(actual.cand_card_ids, expected.cand_card_ids)
    assert np.array_equal(actual.cand_feats, expected.cand_feats)
    assert np.array_equal(actual.picked_card_ids, expected.picked_card_ids)


def _batch():
    observation, choice = _obs_choice()
    return observation, choice, collate_encoded_batch(
        [encode_observation(observation, _VOCAB)],  # type: ignore[arg-type]
        [encode_choice(choice, observation, _VOCAB)],  # type: ignore[arg-type]
    )


def _samples(model: ComparatorNetwork, n: int = 8) -> list[PPOSample]:
    observation, choice = _obs_choice()
    batch = collate_encoded_batch(
        [encode_observation(observation, _VOCAB)],  # type: ignore[arg-type]
        [encode_choice(choice, observation, _VOCAB)],  # type: ignore[arg-type]
    )
    with torch.inference_mode():
        probs = model(batch, compute_policy=True, compute_value=False)["policy_probs"][0]
    actions = [i % len(choice.candidates) for i in range(n)]
    advantages = [0.9, -0.7, 0.4, -0.2, 0.8, -0.5, 0.3, -0.9][:n]
    samples: list[PPOSample] = []
    for i, action in enumerate(actions):
        samples.append(
            PPOSample(
                observation=encode_observation(observation, _VOCAB),  # type: ignore[arg-type]
                choice=encode_choice(choice, observation, _VOCAB),  # type: ignore[arg-type]
                policy_target=normalize_policy_target(
                    probs.numpy(), len(choice.candidates)
                ),
                value_target=1.0 if advantages[i] > 0 else -1.0,
                seat=0,
                was_searched=False,
                seed_key="train-0000",
                map_name="tharsis",
                generation=1,
                action=action,
                old_log_prob=float(torch.log(probs[action].clamp_min(1e-12)).item()),
                behavior_value=float(probs[action].item()),
                advantage=float(advantages[i]),
                return_=1.0 if advantages[i] > 0 else 0.0,
            )
        )
    return samples


def test_comparator_shares_no_parameters_with_main_agent(tmp_path: Path) -> None:
    main = _tiny_main(seed=1)
    comparator = _tiny_comparator(seed=2)
    main_ids = {id(p) for p in main.parameters()}
    comparator_ids = {id(p) for p in comparator.parameters()}
    assert main_ids.isdisjoint(comparator_ids)

    _observation, _choice, batch = _batch()
    with torch.inference_mode():
        before = comparator(batch)["win_prob"].clone()
        for param in main.parameters():
            param.add_(1.0)
        after = comparator(batch)["win_prob"].clone()
    assert torch.allclose(before, after)

    with pytest.raises(RuntimeError):
        main.load_state_dict(comparator.state_dict())

    path = tmp_path / "comparator"
    save_comparator_checkpoint(
        path, comparator, _VOCAB, temperature=1.7, metadata={"version": "test"}
    )
    loaded, loaded_vocab, meta = load_comparator_checkpoint(path)
    assert loaded_vocab.size == _VOCAB.size
    assert meta["temperature"] == pytest.approx(1.7)
    assert loaded.config.architecture == "comparator_deepset_v1"
    loaded_ids = {id(p) for p in loaded.parameters()}
    assert loaded_ids.isdisjoint(main_ids)


def test_illegal_candidates_get_exact_zero_probability() -> None:
    model = _tiny_comparator()
    _observation, _choice, batch = _batch()
    batch = {k: v.clone() for k, v in batch.items()}
    batch["cand_mask"][0, 2] = False
    with torch.inference_mode():
        out = model(batch, compute_policy=True, compute_value=True)
    probs = out["policy_probs"][0]
    logits = out["policy_logits"][0]
    assert float(probs[2].item()) == 0.0
    assert torch.isinf(logits[2]) and logits[2] < 0
    assert float(probs[:2].sum().item()) == pytest.approx(1.0, abs=1e-6)

    # A forced single-candidate decision is deterministic and legal.
    agent = ComparatorAgent(model, _VOCAB, temperature=0.0, seed=0)
    observation, choice = _obs_choice()
    observation["public"]["generation"] = 1  # type: ignore[index]
    forced = choice.__class__(
        seat=choice.seat,
        kind=choice.kind,
        prompt=choice.prompt,
        candidates=(choice.candidates[0],),
    )
    assert agent.act(observation, forced) == 0  # type: ignore[arg-type]


def test_ppo_update_reduces_surrogate_loss_on_fixed_batch() -> None:
    torch.manual_seed(0)
    model = _tiny_comparator(seed=3)
    samples = _samples(model)
    initial = compute_surrogate_loss(model, samples, clip=0.2, device="cpu")
    config = RunConfig(
        run_name="ppo-test",
        output_root=Path("unused"),
        agent_kind="comparator",
        updates_per_cycle=5,
        learner_batch_size=len(samples),
        learning_rate=1e-3,
        grad_clip=0.0,
        use_amp=False,
    )
    learner = PPOLearner(config, model, device="cpu")
    stats = learner.update_on_batch(samples, steps=5)
    assert stats.updates == 5
    final = compute_surrogate_loss(model, samples, clip=0.2, device="cpu")
    assert final < initial - 1e-4


def test_ppo_update_is_on_policy_before_the_first_step() -> None:
    # Behaviour log-probs come from the deterministic (eval-mode) policy the agent acts with,
    # so with unchanged weights the PPO ratio must be exactly 1 even with dropout configured.
    model = _tiny_comparator(seed=4, dropout=0.3)
    model.eval()
    samples = _samples(model)
    config = RunConfig(
        run_name="ppo-ratio",
        output_root=Path("unused"),
        agent_kind="comparator",
        learner_batch_size=len(samples),
        learning_rate=0.0,
        weight_decay=0.0,
        grad_clip=0.0,
        use_amp=False,
    )
    learner = PPOLearner(config, model, device="cpu")
    stats = learner.train(PPOBuffer(capacity=16), samples=samples, updates=1)
    assert stats.updates == 1
    assert stats.mean_ratio == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(("win_target", "lo", "hi"), [(0.0, 0.0, 0.8), (1.0, 0.8, 1.0)])
def test_gae_value_targets_stay_in_win_probability_space(
    win_target: float, lo: float, hi: float
) -> None:
    # Win-probability head predicts 0.8 everywhere. A lost game must pull every value target
    # down (never above the prediction), a won game must pull every target up, and the
    # terminal target equals the outcome in [0, 1] (issue #25 review: targets were biased up).
    values = [0.8] * 50
    advantages, returns = compute_gae(values, win_target)
    assert returns[-1] == pytest.approx(win_target)
    assert np.all(returns >= lo - 1e-6) and np.all(returns <= hi + 1e-6)
    if win_target == 0.0:
        assert np.all(advantages < 0.0)
    else:
        assert np.all(advantages > 0.0)


def test_gae_rejects_utility_space_outcomes() -> None:
    # The terminal outcome must be a win probability (loss 0, tie 0.5, win 1), the same space as
    # the win-probability head's bootstrap values; a -1/+1 utility would bias every target.
    with pytest.raises(ValueError):
        compute_gae([0.8, 0.8], -1.0)


def test_ppo_value_loss_targets_returns_directly() -> None:
    model = _tiny_comparator()
    samples = _samples(model)
    batch, ppo = collate_ppo_samples(samples)
    with torch.inference_mode():
        _, parts = ppo_policy_terms(model, batch, ppo, clip=0.2, value_coef=1.0, entropy_coef=0.0)
        out = model(batch, compute_policy=False, compute_value=True)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        out["value_logit"].float(), ppo["returns"].float()
    )
    assert float(parts["value_loss"]) == pytest.approx(float(expected), rel=1e-5)


def test_ppo_train_with_empty_rollout_does_not_reuse_stale_buffer() -> None:
    config = RunConfig(
        run_name="t", agent_kind="comparator", updates_per_cycle=3, learner_batch_size=4
    )
    model = _tiny_comparator()
    buffer = PPOBuffer(capacity=16)
    buffer.extend(_samples(model))
    before = [p.detach().clone() for p in model.parameters()]
    stats = PPOLearner(config, model, device="cpu").train(buffer, samples=[])
    assert stats.updates == 0
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_gae_and_ppo_buffer_roundtrip() -> None:
    advantages, returns = compute_gae([0.5, 0.6, 0.7], 1.0)
    assert advantages.shape == returns.shape == (3,)
    assert advantages[-1] == pytest.approx(0.3)
    assert returns[-1] == pytest.approx(1.0)

    model = _tiny_comparator()
    samples = _samples(model)
    buffer = PPOBuffer(capacity=4)
    buffer.extend(samples)
    assert len(buffer) == 4
    payload = buffer.state_dict()
    restored = PPOBuffer(capacity=4)
    restored.load_state_dict(payload)
    batch, ppo = collate_ppo_samples(restored.samples)
    assert batch["cand_mask"].shape[0] == 4
    assert ppo["actions"].shape == (4,)


def test_temperature_scaling_and_calibration_report(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(0.0, 2.0, size=400)
    true_probs = 1.0 / (1.0 + np.exp(-(logits * 0.35)))
    targets = rng.random(400) < true_probs
    fit = fit_temperature(logits, targets)
    assert fit["n"] == 400
    assert fit["nll_after"] <= fit["nll_before"] + 1e-6
    assert 0.05 <= fit["temperature"] <= 20.0

    metrics = calibration_metrics(1.0 / (1.0 + np.exp(-logits)), targets)
    assert metrics["n"] == 400
    assert len(metrics["bins"]) == 10
    payload = write_calibration_report(
        tmp_path, logits, targets, temperature=fit["temperature"], name="dev"
    )
    assert payload["calibrated"]["n"] == 400
    for name in ("dev.json", "dev.md", "dev_reliability.svg"):
        assert (tmp_path / name).exists()


def test_comparator_selfplay_seats_use_the_play_temperature_schedule() -> None:
    # v2 (#48) replaces the flat action_temperature: every comparator seat samples at τ(g)
    # in the action phase and τ 1 elsewhere, with the run config's schedule.
    config = RunConfig(run_name="cmp", agent_kind="comparator")
    engine = ComparatorSelfPlayEngine(config, model=_tiny_comparator(), vocab=_VOCAB)
    agent = engine._learner_factory(1)
    assert agent.play_temperature == config.play_temperature
    assert agent.temperature == 1.0


def test_agent_kind_is_validated() -> None:
    with pytest.raises(ValueError, match="agent_kind"):
        RunConfig(run_name="bad", agent_kind="other")
    assert RunConfig(run_name="main").agent_kind == "main"
    assert RunConfig(run_name="cmp", agent_kind="comparator").agent_kind == "comparator"


def test_evaluation_comparator_plays_its_sampled_policy_reproducibly() -> None:
    """Greedy argmax play can loop forever (two greedy comparators kept buying
    Aquifer with oceans maxed and never raised temperature, so dev games hit the
    step cap). Evaluation therefore plays the trained stochastic policy, seeded
    per seat so the same seeds reproduce the same games."""
    model = _tiny_comparator()
    observation, choice = _obs_choice()
    factory = comparator_eval_factory(model, _VOCAB, device="cpu", version="eval-test")

    def picks(seed: int) -> list[int]:
        agent = factory()
        agent.reset(seed)
        return [agent.act(observation, choice) for _ in range(40)]  # type: ignore[arg-type]

    first = picks(7)
    assert picks(7) == first
    assert len(set(first)) > 1, "evaluation agent must sample, not take the argmax"
    assert factory().version == "eval-test"


class _RolloutEngine:
    """Comparator self-play stand-in: every cycle yields a fresh PPO rollout."""

    def __init__(self, model: ComparatorNetwork) -> None:
        self.model = model
        self.seed_cursor = 0
        self.opponents = SimpleNamespace(pool=())

    def set_opponent_pool(self, checkpoints) -> None:
        self.opponents.pool = tuple(Path(p) for p in checkpoints)

    def play_batch(self, num_games: int, *, stop_when=None) -> SelfPlayBatch:
        games = []
        for _ in range(num_games):
            games.append(
                GameSummary(
                    seed_key=f"train-{self.seed_cursor:04d}",
                    map_name="tharsis",
                    ok=True,
                    generations=12,
                    decisions=120,
                    scores=(60, 50),
                    winner=0,
                    tie=False,
                )
            )
            self.seed_cursor += 1
        return SelfPlayBatch(
            games=games,
            samples=[],
            training_seats=[0] * num_games,
            opponent_kinds=["latest"] * num_games,
            decisions=120 * num_games,
            ppo_samples=_samples(self.model),
        )


class _RecordingPPOLearner:
    def __init__(self) -> None:
        self.step = 0
        self.rollout_sizes: list[int] = []

    def train(self, buffer, *, updates=None, samples=None) -> LearnerStats:
        assert samples is not None, "PPO must train on the fresh rollout"
        self.step += 1
        self.rollout_sizes.append(len(samples))
        return LearnerStats(updates=1, policy_loss=1.0, value_loss=0.5, total_loss=1.5)


def _dev_result(_checkpoint: Path, _opponent: Path | None) -> DevResult:
    return DevResult(
        games=8,
        completed=8,
        failures=0,
        win_rate=0.5,
        win_rate_ci_95=(0.3, 0.7),
        mean_vp_margin=0.0,
        report={},
    )


def test_comparator_run_trains_on_fresh_rollouts_and_keeps_none(tmp_path: Path) -> None:
    """PPO is on-policy: old rollouts are never trained on again, so a run must not
    hold them in memory or rewrite them into state.pt every cycle (it reached 1.1 GB)."""
    config = RunConfig(
        run_name="cmp-run",
        output_root=tmp_path,
        agent_kind="comparator",
        workers=1,
        games_per_cycle=2,
        updates_per_cycle=1,
        learner_batch_size=4,
        checkpoint_interval_s=1e-6,
        dev_games_per_opponent=8,
        keep_last_checkpoints=2,
        limits=ResourceLimits(disk_gb=50.0),
    )
    model = _tiny_comparator(seed=1)
    model.temperature = 0.8  # loaded comparator checkpoints carry their calibration
    learner = _RecordingPPOLearner()
    runner = SelfPlayRunner(
        config,
        engine=_RolloutEngine(model),
        learner=learner,
        evaluator=_dev_result,
        model=model,  # type: ignore[arg-type]
        vocab=_VOCAB,
    )
    summary = runner.run(max_cycles=3)
    assert summary.cycles == 3
    assert learner.rollout_sizes == [8, 8, 8]
    assert len(runner.buffer) == 0
    state = torch.load(runner.manager.state_tensors_path, map_location="cpu", weights_only=False)
    assert "buffer" not in state

    # Resume restores the comparator weights from its own checkpoint format.
    resumed_model = _tiny_comparator(seed=99)
    resumed = SelfPlayRunner(
        config,
        engine=_RolloutEngine(resumed_model),
        learner=_RecordingPPOLearner(),
        evaluator=_dev_result,
        model=resumed_model,  # type: ignore[arg-type]
        vocab=_VOCAB,
    )
    resumed.run(max_cycles=3, resume=True)
    for key, value in model.state_dict().items():
        assert torch.equal(value, resumed_model.state_dict()[key]), key
    assert resumed_model.temperature == pytest.approx(0.8)


def test_assemble_batch_targets_follow_each_training_seats_outcome() -> None:
    """Seat 0 wins both games; the learner sat in seat 0 (win) then seat 1 (loss).
    Value targets are the seat's utility; GAE returns stay in win-probability space
    and end exactly on the outcome (1 for the win, 0 for the loss)."""
    config = RunConfig(run_name="cmp-assemble", agent_kind="comparator")
    engine = ComparatorSelfPlayEngine(config, model=_tiny_comparator(), vocab=_VOCAB)
    base_obs, choice = _obs_choice()
    recorders = []
    outcomes = []
    for game, seat in enumerate((0, 1)):
        recorder = GameRecorder(
            f"train-{game:04d}", "tharsis", capture_seat=seat, vocab=_VOCAB
        )
        observation = copy.deepcopy(base_obs)
        observation["actor"] = seat  # type: ignore[index]
        for step in range(3):
            recorder.record(
                observation, choice, step % len(choice.candidates), [0.5, 0.3, 0.2],
                was_searched=False,
            )
        recorders.append(recorder)
        outcomes.append(
            GameRunResult(
                seed_key=f"train-{game:04d}",
                map="tharsis",
                result=GameResult(scores=(70, 50), winner=0, tie=False, generations=12),
                generations=12,
                decisions=6,
                agent_versions=(None, None),
                ok=True,
            )
        )
    batch = engine.assemble_batch(outcomes, recorders, [0, 1], ["latest", "latest"])
    samples = batch.ppo_samples
    assert samples is not None and len(samples) == 6
    win, loss = samples[:3], samples[3:]
    assert {s.seat for s in win} == {0} and {s.seat for s in loss} == {1}
    assert all(s.value_target == 1.0 for s in win)
    assert all(s.value_target == -1.0 for s in loss)
    assert all(0.0 <= s.return_ <= 1.0 for s in samples)
    assert win[-1].return_ == pytest.approx(1.0)
    assert loss[-1].return_ == pytest.approx(0.0)
    assert [s.action for s in win] == [0, 1, 2]


def test_ppo_rows_match_encode_at_assembly_rows() -> None:
    """PPO rows assembled from encoded records equal the old encode-at-assembly rows."""
    config = RunConfig(run_name="cmp-encoded", agent_kind="comparator")
    model = _tiny_comparator()
    engine = ComparatorSelfPlayEngine(config, model=model, vocab=_VOCAB)
    base_obs, choice = _obs_choice()
    recorder = GameRecorder("train-0000", "tharsis", capture_seat=0, vocab=_VOCAB)
    cases = []
    for step in range(3):
        observation = copy.deepcopy(base_obs)
        observation["actor"] = 0  # type: ignore[index]
        index = step % len(choice.candidates)
        recorder.record(
            observation, choice, index, [0.5, 0.3, 0.2], was_searched=False
        )
        cases.append((observation, choice, index))
    outcome = GameRunResult(
        seed_key="train-0000",
        map="tharsis",
        result=GameResult(scores=(70, 50), winner=0, tie=False, generations=12),
        generations=12,
        decisions=6,
        agent_versions=(None, None),
        ok=True,
    )
    batch = engine.assemble_batch([outcome], [recorder], [0], ["latest"])
    samples = batch.ppo_samples
    assert samples is not None and len(samples) == 3

    # Old path: encode the raw observation/choice pairs at assembly time.
    encoded_obs = [encode_observation(obs, _VOCAB) for obs, _, _ in cases]  # type: ignore[arg-type]
    encoded_choice = [
        encode_choice(ch, obs, _VOCAB)  # type: ignore[arg-type]
        for obs, ch, _ in cases
    ]
    tensors = collate_encoded_batch(encoded_obs, encoded_choice)
    model.eval()
    with torch.inference_mode():
        out = model(tensors, compute_policy=True, compute_value=True)
        log_probs = torch.log_softmax(out["policy_logits"].float(), dim=-1)
        actions = torch.tensor([index for _, _, index in cases], dtype=torch.long)
        expected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        expected_values = out["win_prob"].float()
    expected_advantages, expected_returns = compute_gae(
        expected_values.numpy(),
        1.0,
        gamma=config.ppo_gamma,
        gae_lambda=config.ppo_gae_lambda,
    )
    for i, (sample, (obs, _, index)) in enumerate(zip(samples, cases)):
        _assert_encoded_observation_equal(
            sample.observation,
            encode_observation(obs, _VOCAB),  # type: ignore[arg-type]
        )
        _assert_encoded_choice_equal(
            sample.choice,
            encode_choice(choice, obs, _VOCAB),  # type: ignore[arg-type]
        )
        assert sample.action == index
        assert sample.old_log_prob == pytest.approx(float(expected_log_probs[i]))
        assert sample.behavior_value == pytest.approx(float(expected_values[i]))
        assert sample.advantage == pytest.approx(float(expected_advantages[i]))
        assert sample.return_ == pytest.approx(float(expected_returns[i]))
        assert np.array_equal(
            sample.policy_target, normalize_policy_target([0.5, 0.3, 0.2], 3)
        )
        assert sample.value_target == 1.0
        assert sample.seat == 0


def test_eval_cli_spec_loads_a_cached_sampling_comparator(tmp_path: Path) -> None:
    path = tmp_path / "comparator"
    save_comparator_checkpoint(path, _tiny_comparator(), _VOCAB, temperature=1.3)
    factory = resolve_agent_factory(
        f"tm_bot.comparator:make_comparator_agent?checkpoint={path}&device=cpu"
    )
    first, second = factory(), factory()
    assert first is not second
    assert first.model is second.model, "the checkpoint is loaded once per process"
    assert first.temperature == EVAL_ACTION_TEMPERATURE
    assert first.calibration_temperature == pytest.approx(1.3)
