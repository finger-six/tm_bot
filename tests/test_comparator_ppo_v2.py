"""v2 comparator PPO (#50): τ-scaled ratios and entropy, utility-U reward with the
(2p − 1) + 0.15·m̂ GAE baseline, outcome-only win head, margin head, and the warm-start
anchors shared with the main learner.

Decisions are synthetic (`tests.test_runs` observations, tiny comparator); human rows come
from the golden table pair's tiny encoded cache, never the real cache.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tests.test_comparator import _RolloutEngine, _dev_result, _tiny_comparator
from tests.test_learner_v2 import TINY_WARMSTART, _perturb, _RecordingHumanAux
from tests.test_runs import _VOCAB, _choice, _obs
from tests.test_selfplay_rules import _action_obs, _outcome
from tests.test_warmstart_v2 import golden_data, golden_dir, vocab  # noqa: F401  (fixtures)
from tm_bot.agents.play_rules import V2_PLAY_TEMPERATURE
from tm_bot.comparator.model import ComparatorNetwork
from tm_bot.comparator.ppo import (
    PPOBuffer,
    PPOLearner,
    PPOSample,
    collate_ppo_samples,
    compute_gae,
    scaled_log_probs,
)
from tm_bot.comparator.ppo import _policy_terms as ppo_policy_terms
from tm_bot.comparator.selfplay import ComparatorSelfPlayEngine
from tm_bot.env.types import Candidate, Choice
from tm_bot.models.network import VP_MARGIN_SMOOTH_L1_BETA
from tm_bot.runs.config import ResourceLimits, RunConfig
from tm_bot.runs.learner import compute_value_calibration
from tm_bot.runs.orchestrator import SelfPlayRunner
from tm_bot.runs.recorder import GameRecorder
from tm_bot.training.human_aux import HumanAuxLoss, build_human_aux_sample
from tm_bot.training.multitask import WarmstartData

# Generation 8 of τ(g): linear from 1.0 at g 4 to 0.3 at g 10.
ACTION_GENERATION = 8
ACTION_TAU = 1.0 + (0.3 - 1.0) * (ACTION_GENERATION - 4) / 6


def _config(**overrides: Any) -> RunConfig:
    data: dict[str, Any] = dict(
        run_name="ppo-v2", agent_kind="comparator", use_amp=False, grad_clip=0.0
    )
    data.update(overrides)
    return RunConfig(**data)


def _three_way_choice(seat: int = 0) -> Choice:
    return Choice(
        seat=seat,  # type: ignore[arg-type]
        kind="card",
        prompt="Select a card to keep",
        candidates=(Candidate("Algae", "Algae"), Candidate("Birds", "Birds"), Candidate("Ants", "Ants")),
    )


def _research_obs(*, actor: int = 0) -> dict[str, Any]:
    obs = _obs(actor=actor)
    obs["public"]["phase"] = "research"
    obs["public"]["generation"] = ACTION_GENERATION
    return obs


def _play_and_record(
    engine: ComparatorSelfPlayEngine, *, seat: int = 0, rounds: int = 3
) -> GameRecorder:
    """The learner agent acts on action-phase (τ(8) < 1) and research (τ = 1) decisions;
    each decision is recorded the way the game runner records it."""
    agent = engine._learner_factory(seed=11)
    recorder = GameRecorder("train-0000", "tharsis", capture_seat=seat, vocab=_VOCAB)
    for _ in range(rounds):
        for obs, choice in (
            (_action_obs(ACTION_GENERATION, actor=seat), _three_way_choice(seat)),
            (_research_obs(actor=seat), _choice(seat=seat)),
        ):
            index = agent.act(obs, choice)  # type: ignore[arg-type]
            recorder.record(
                obs, choice, index, agent.last_decision_policy,  # type: ignore[arg-type]
                was_searched=False, play_temperature=agent.last_decision_temperature,
            )
    return recorder


def _rollout(
    model: ComparatorNetwork,
    *,
    anchor: ComparatorNetwork | None = None,
    scores: tuple[int, int] = (70, 45),
    config: RunConfig | None = None,
) -> list[PPOSample]:
    engine = ComparatorSelfPlayEngine(
        config or _config(), model=model, vocab=_VOCAB, anchor_model=anchor
    )
    recorder = _play_and_record(engine)
    winner = 0 if scores[0] > scores[1] else 1
    samples = engine.assemble_batch([_outcome(scores, winner)], [recorder], [0]).ppo_samples
    assert samples is not None and len(samples) == len(recorder.decisions)
    return samples


def _model_tensors(model: ComparatorNetwork, samples: list[PPOSample]) -> dict[str, torch.Tensor]:
    batch, ppo = collate_ppo_samples(samples)
    model.eval()
    with torch.no_grad():
        out = model(batch, compute_policy=True, compute_value=True)
    return {"batch": batch, "ppo": ppo, **out}  # type: ignore[dict-item]


# --- AC: the PPO ratio is 1 on the sampled τ-scaled policy before an update -----------------


def test_ratio_is_one_on_the_sampled_tau_scaled_policy_before_an_update() -> None:
    # Dropout configured: behavior log-probs and the update both run the eval-mode policy.
    model = _tiny_comparator(seed=4, dropout=0.3)
    _perturb(model, scale=0.3)  # a peaked policy, so τ visibly matters
    samples = _rollout(model)
    taus = sorted({s.play_temperature for s in samples})
    assert taus == pytest.approx([ACTION_TAU, 1.0]) and ACTION_TAU < 1.0
    assert ACTION_TAU == pytest.approx(V2_PLAY_TEMPERATURE(ACTION_GENERATION))

    t = _model_tensors(model, samples)
    mask = t["batch"]["cand_mask"].bool()
    for i, s in enumerate(samples):
        # `old_log_prob` is the log-probability under the distribution the agent sampled
        # from: the τ = 1 policy raised to 1/τ and renormalised over the legal candidates.
        probs = t["policy_probs"][i, : s.num_candidates].double()
        sampled = probs ** (1.0 / s.play_temperature)
        sampled = sampled / sampled.sum()
        assert s.old_log_prob == pytest.approx(float(sampled[s.action].log()), abs=1e-5)
        raw = float(probs[s.action].log())
        if s.play_temperature < 1.0:
            assert abs(s.old_log_prob - raw) > 1e-3, "τ < 1 must change the behavior log-prob"
        else:
            assert s.old_log_prob == pytest.approx(raw, abs=1e-5)

    new = scaled_log_probs(t["policy_logits"], t["ppo"]["temperatures"], mask)
    new = new.gather(1, t["ppo"]["actions"].unsqueeze(1)).squeeze(1)
    ratios = torch.exp(new - t["ppo"]["old_log_probs"])
    np.testing.assert_allclose(ratios.numpy(), 1.0, atol=1e-5)

    config = _config(learner_batch_size=len(samples), learning_rate=0.0, weight_decay=0.0)
    stats = PPOLearner(config, model).train(PPOBuffer(16), samples=samples, updates=1)
    assert stats.updates == 1
    assert stats.mean_ratio == pytest.approx(1.0, abs=1e-6)


def test_entropy_uses_the_same_tau_scaled_logits() -> None:
    model = _tiny_comparator(seed=5)
    _perturb(model, scale=0.3)
    samples = _rollout(model)
    t = _model_tensors(model, samples)
    with torch.no_grad():
        _, parts = ppo_policy_terms(
            model, t["batch"], t["ppo"], clip=0.2, value_coef=0.5, entropy_coef=0.01
        )
    entropies, raw_entropies = [], []
    for i, s in enumerate(samples):
        logits = t["policy_logits"][i, : s.num_candidates].double()
        scaled = F.softmax(logits / s.play_temperature, dim=-1)
        raw = F.softmax(logits, dim=-1)
        entropies.append(float(-(scaled * scaled.log()).sum()))
        raw_entropies.append(float(-(raw * raw.log()).sum()))
    assert float(parts["entropy"]) == pytest.approx(float(np.mean(entropies)), rel=1e-5)
    assert float(parts["entropy"]) < float(np.mean(raw_entropies)) - 1e-4


def test_an_argmax_decision_cannot_enter_ppo() -> None:
    sample = replace(_rollout(_tiny_comparator(seed=6))[0], play_temperature=0.0)
    with pytest.raises(ValueError, match="sampled decision"):
        collate_ppo_samples([sample])


# --- AC: the KL term is 0 when the weights equal the anchor --------------------------------


def test_kl_term_is_zero_when_the_weights_equal_the_anchor() -> None:
    model = _tiny_comparator(seed=7)
    samples = _rollout(model, anchor=copy.deepcopy(model))
    assert all(s.anchor_logits is not None for s in samples)
    assert any(s.play_temperature < 1.0 for s in samples)
    batch, ppo = collate_ppo_samples(samples)
    model.eval()
    with torch.no_grad():
        _, parts = ppo_policy_terms(
            model, batch, ppo, clip=0.2, value_coef=0.5, entropy_coef=0.01, kl_beta=0.05
        )
    # Raw τ = 1 logits against the raw recorded anchor logits: 0 even on τ < 1 rows.
    assert float(parts["kl_anchor"]) == pytest.approx(0.0, abs=1e-6)

    _perturb(model, scale=0.2)
    with torch.no_grad():
        _, parts = ppo_policy_terms(
            model, batch, ppo, clip=0.2, value_coef=0.5, entropy_coef=0.01, kl_beta=0.05
        )
    assert float(parts["kl_anchor"]) > 1e-4


def test_ppo_learner_uses_the_run_kl_beta() -> None:
    assert PPOLearner(_config(), _tiny_comparator()).kl_beta == pytest.approx(0.05)
    assert PPOLearner(_config(kl_anchor_beta=0.2), _tiny_comparator()).kl_beta == pytest.approx(0.2)


# --- AC: returns/advantages use U; calibration never sees U ---------------------------------


def test_gae_hand_computed_example_in_utility_units() -> None:
    # A win by 20 VP: m = 0.4, U = 1 + 0.15·0.4 = 1.06; γ 1, λ 0.95.
    baselines = [0.2, -0.1, 0.4]
    advantages, returns = compute_gae(baselines, 1.06, gamma=1.0, gae_lambda=0.95)
    # t2: δ = 1.06 - 0.4 = 0.66
    # t1: δ = 0.4 - (-0.1) = 0.5,  A = 0.5 + 0.95·0.66 = 1.127
    # t0: δ = -0.1 - 0.2 = -0.3,  A = -0.3 + 0.95·1.127 = 0.77065
    np.testing.assert_allclose(advantages, [0.77065, 1.127, 0.66], rtol=1e-5)
    np.testing.assert_allclose(returns, [0.97065, 1.027, 1.06], rtol=1e-5)


def test_rollout_returns_use_u_against_the_margin_aware_baseline() -> None:
    model = _tiny_comparator(seed=8)
    samples = _rollout(model, scores=(70, 45))  # seat 0 wins by 25: m = 0.5
    utility = 1.0 + 0.15 * 25 / 50
    t = _model_tensors(model, samples)
    baselines = (2.0 * t["win_prob"] - 1.0 + 0.15 * t["vp_margin_pred"]).double().numpy()
    expected_adv, expected_ret = compute_gae(baselines, utility)
    np.testing.assert_allclose([s.advantage for s in samples], expected_adv, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose([s.return_ for s in samples], expected_ret, rtol=1e-5, atol=1e-6)
    assert samples[-1].return_ == pytest.approx(utility)
    assert samples[-1].advantage == pytest.approx(utility - baselines[-1], abs=1e-6)
    # The rows keep the raw win probability and the outcome-only target.
    np.testing.assert_allclose([s.behavior_value for s in samples], t["win_prob"].numpy(), rtol=1e-6)
    assert all(s.value_target == 1.0 for s in samples)
    assert all(s.margin_target == pytest.approx(0.5) for s in samples)


def test_win_head_and_calibration_never_see_u() -> None:
    model = _tiny_comparator(seed=9)
    samples = _rollout(model, scores=(40, 75))  # a loss by 35
    shifted = [
        replace(s, return_=-s.return_, advantage=s.advantage + 0.3, margin_target=0.9) for s in samples
    ]

    def value_head_grads(rows: list[PPOSample]) -> tuple[float, list[torch.Tensor]]:
        batch, ppo = collate_ppo_samples(rows)
        model.zero_grad(set_to_none=True)
        loss, parts = ppo_policy_terms(
            model, batch, ppo, clip=0.2, value_coef=0.5, entropy_coef=0.01, margin_coef=0.3
        )
        loss.backward()
        return float(parts["value_loss"]), [p.grad.clone() for p in model.value_head.parameters()]

    model.eval()
    loss_a, grads_a = value_head_grads(samples)
    loss_b, grads_b = value_head_grads(shifted)
    assert loss_a == loss_b
    assert all(torch.equal(a, b) for a, b in zip(grads_a, grads_b))
    logit = _model_tensors(model, samples)["value_logit"]
    expected = F.binary_cross_entropy_with_logits(logit, torch.zeros_like(logit))  # loss: o = -1
    assert loss_a == pytest.approx(float(expected), rel=1e-6)

    # In-run calibration reads the outcome o, not U or the returns.
    report = compute_value_calibration(model, shifted)  # type: ignore[arg-type]
    assert (report["wins"], report["losses"], report["ties"]) == (0, len(samples), 0)
    p = torch.sigmoid(logit).double().numpy()
    assert report["brier"] == pytest.approx(float(np.mean(p**2)), rel=1e-6)


# --- AC: the margin head trains ------------------------------------------------------------


def test_margin_head_trains_on_terminal_margins() -> None:
    model = _tiny_comparator(seed=10)
    samples = _rollout(model, scores=(80, 42))  # m = 38/50 = 0.76
    batch, ppo = collate_ppo_samples(samples)

    def margin_loss() -> float:
        model.eval()
        with torch.no_grad():
            _, parts = ppo_policy_terms(
                model, batch, ppo, clip=0.2, value_coef=0.5, entropy_coef=0.01, margin_coef=0.3
            )
        return float(parts["margin_loss"])

    pred = _model_tensors(model, samples)["vp_margin_pred"]
    expected = F.smooth_l1_loss(pred, torch.full_like(pred, 0.76), beta=VP_MARGIN_SMOOTH_L1_BETA)
    before = margin_loss()
    assert before == pytest.approx(float(expected), rel=1e-6)

    config = _config(learning_rate=3e-3, weight_decay=0.0)
    learner = PPOLearner(config, model)
    stats = learner.update_on_batch(samples, steps=1)
    assert model.vp_margin_head is not None
    assert any(float(p.grad.abs().sum()) > 0.0 for p in model.vp_margin_head.parameters())
    stats = learner.update_on_batch(samples, steps=30)
    assert stats.updates == 30 and stats.margin_loss is not None
    assert margin_loss() < 0.5 * before


# --- AC: mirror-game samples get per-seat GAE ----------------------------------------------


def test_mirror_game_rows_get_gae_per_seat_trajectory() -> None:
    model = _tiny_comparator(seed=12)
    engine = ComparatorSelfPlayEngine(_config(), model=model, vocab=_VOCAB)
    agent = engine._learner_factory(seed=3)
    recorder = GameRecorder(
        "train-0000", "tharsis", capture_seat=0, vocab=_VOCAB, record_both_seats=True
    )
    # Interleaved seats, as in a real mirror game.
    for generation in (2, 6, 9):
        for seat in (0, 1):
            obs, choice = _action_obs(generation, actor=seat), _three_way_choice(seat)
            index = agent.act(obs, choice)  # type: ignore[arg-type]
            recorder.record(
                obs, choice, index, agent.last_decision_policy,  # type: ignore[arg-type]
                was_searched=False, play_temperature=agent.last_decision_temperature,
            )
    samples = engine.assemble_batch([_outcome((52, 66), 1)], [recorder], [0]).ppo_samples
    assert samples is not None and len(samples) == 6
    for seat, sign in ((0, -1.0), (1, 1.0)):
        rows = [s for s in samples if s.seat == seat]
        assert [s.generation for s in rows] == [2, 6, 9]
        utility = sign * (1.0 + 0.15 * 14 / 50)
        t = _model_tensors(model, rows)
        baselines = (2.0 * t["win_prob"] - 1.0 + 0.15 * t["vp_margin_pred"]).double().numpy()
        advantages, returns = compute_gae(baselines, utility)
        np.testing.assert_allclose([s.advantage for s in rows], advantages, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose([s.return_ for s in rows], returns, rtol=1e-5, atol=1e-6)
        assert rows[-1].return_ == pytest.approx(utility)


# --- anchors: λ_h·L_human and the full loss ------------------------------------------------


@pytest.mark.parametrize(("hours", "expected"), [(0, 0.5), (5, 0.3), (12, 0.1)])
def test_ppo_learner_reads_lambda_from_the_training_clock(hours: float, expected: float) -> None:
    human = _RecordingHumanAux()
    model = _tiny_comparator(seed=13)
    samples = _rollout(model)
    learner = PPOLearner(
        _config(learner_batch_size=4), model, human_aux=human,
        training_clock=lambda: hours * 3600.0,
    )
    stats = learner.train(PPOBuffer(16), samples=samples, updates=2)
    assert human.weights == pytest.approx([expected, expected])
    assert stats.human_aux_weight == pytest.approx(expected)
    assert stats.human_aux_loss == pytest.approx(2.0)


def test_ppo_step_runs_every_term_and_logs_each(golden_data: WarmstartData) -> None:  # noqa: F811
    model = _tiny_comparator(seed=14)
    anchor = copy.deepcopy(model)
    _perturb(anchor, scale=0.2)
    samples = _rollout(model, anchor=anchor)
    sample = build_human_aux_sample(
        golden_data, max_rows=60, rng=np.random.default_rng(3), weights=TINY_WARMSTART
    )
    human = HumanAuxLoss(sample, rng=np.random.default_rng(0), weights=TINY_WARMSTART)
    learner = PPOLearner(_config(learner_batch_size=4), model, human_aux=human)
    before = [p.detach().clone() for p in model.parameters()]

    stats = learner.train(PPOBuffer(16), samples=samples, updates=3, training_s=0.0)

    assert stats.updates == 3 and learner.step == 3
    assert stats.nonfinite_loss is None and not stats.nonfinite_params
    assert any(not torch.equal(b, p) for b, p in zip(before, model.parameters()))
    for step in stats.per_step:
        assert step["kl_anchor"] > 0.0
        expected = (
            step["policy_loss"] + 0.5 * (step["value_loss"] + 0.3 * step["margin_loss"])
            - 0.01 * step["entropy"] + 0.05 * step["kl_anchor"] + 0.5 * step["human_aux_loss"]
        )
        assert step["total_loss"] == pytest.approx(expected, rel=1e-5)
    logged = stats.to_dict()
    for key in ("policy_loss", "value_loss", "margin_loss", "kl_anchor", "human_aux_loss", "total_loss"):
        assert logged[key] is not None and np.isfinite(logged[key]), key
    assert logged["human_aux_weight"] == pytest.approx(0.5)


def test_comparator_cycle_record_logs_each_ppo_term(tmp_path: Path) -> None:
    config = RunConfig(
        run_name="cmp-v2", output_root=tmp_path, agent_kind="comparator", workers=1,
        games_per_cycle=1, updates_per_cycle=2, learner_batch_size=4, use_amp=False,
        checkpoint_interval_s=1e-6, dev_games_per_opponent=8, keep_last_checkpoints=2,
        limits=ResourceLimits(disk_gb=50.0),
    )
    model = _tiny_comparator(seed=15)
    learner = PPOLearner(config, model, human_aux=_RecordingHumanAux())
    now = [0.0]
    runner = SelfPlayRunner(
        config, engine=_RolloutEngine(model), learner=learner, evaluator=_dev_result,
        model=model, vocab=_VOCAB, clock=lambda: now[0],  # type: ignore[arg-type]
    )
    learner.training_clock = runner.training_s  # as `SelfPlayRunner.create` wires it
    now[0] = 5 * 3600.0
    runner.run(max_cycles=1)
    record = runner.logger.cycles[-1]
    assert record.human_aux_weight == pytest.approx(0.3)
    assert record.human_aux_loss == pytest.approx(2.0)
    for value in (record.policy_loss, record.value_loss, record.margin_loss, record.total_loss):
        assert value is not None and np.isfinite(value)


def test_comparator_run_wires_the_human_sample_and_training_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded: dict[str, Any] = {}

    def fake_checkpoint(path, map_location=None, encoder_version=None):
        return _tiny_comparator(seed=16), _VOCAB, {}

    def fake_sample(data_dir, cache_dir, *, max_rows, seed):
        loaded.update(data_dir=data_dir, max_rows=max_rows, seed=seed)
        return "sample"

    class FakeLoss:
        def __init__(self, sample, *, rng):
            loaded["sample"] = sample

    monkeypatch.setattr("tm_bot.comparator.model.load_comparator_checkpoint", fake_checkpoint)
    monkeypatch.setattr("tm_bot.training.human_aux.load_human_aux_sample", fake_sample)
    monkeypatch.setattr("tm_bot.training.human_aux.HumanAuxLoss", FakeLoss)
    config = RunConfig(
        run_name="cmp-create", output_root=tmp_path, agent_kind="comparator", device="cpu",
        warmstart_checkpoint="unused.pt", human_aux_data_dir="human", dev_eval_enabled=False,
    )
    runner = SelfPlayRunner.create(config)
    assert loaded == {"data_dir": "human", "max_rows": 1_000_000, "seed": config.seed, "sample": "sample"}
    learner = runner.learner
    assert isinstance(learner, PPOLearner) and isinstance(learner.human_aux, FakeLoss)
    assert learner.training_clock == runner.training_s
    assert learner.kl_beta == pytest.approx(0.05)
