"""v2 main-agent learner (#49): margin-head loss, decaying human-data auxiliary loss, KL anchor.

Self-play rows are synthetic (`tests.test_runs` helpers); human rows come from the golden
table pair's tiny encoded cache (`tests.test_warmstart_v2` fixtures), never the real cache.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tests.test_runs import (
    _VOCAB,
    _FakeEngine,
    _choice,
    _fake_dev,
    _obs,
    _run_config,
    _tiny_model,
)
from tests.test_warmstart_v2 import golden_data, golden_dir, vocab  # noqa: F401  (fixtures)
from tm_bot.env.types import Candidate, Choice
from tm_bot.models.anchor_loss import (
    HumanAuxSchedule,
    kl_to_anchor,
    pad_anchor_logits,
)
from tm_bot.models.encode import encode_choice, encode_observation
from tm_bot.models.network import VP_MARGIN_SMOOTH_L1_BETA
from tm_bot.runs.anchor import anchor_policy_logits
from tm_bot.runs.buffer import DecisionSample, ReplayBuffer
from tm_bot.runs.config import RunConfig
from tm_bot.runs.learner import (
    AlphaZeroLearner,
    LossWeights,
    selfplay_losses,
    selfplay_targets,
)
from tm_bot.runs.orchestrator import SelfPlayRunner
from tm_bot.training.human_aux import (
    HumanAuxLoss,
    allocate_rows,
    build_human_aux_sample,
    pool_draws,
)
from tm_bot.training.multitask import WarmstartConfig, WarmstartData

TINY_WARMSTART = replace(
    WarmstartConfig(), use_amp=False, setup_batch_size=4, draft_batch_size=4, research_buy_batch_size=4,
    branch_batch_size=4, project_play_batch_size=4, standard_project_batch_size=4, card_action_batch_size=2,
    value_batch_size=6,
)


def _three_way_choice() -> Choice:
    return Choice(
        seat=0,  # type: ignore[arg-type]
        kind="card",
        prompt="Select a card to keep",
        candidates=(Candidate("Algae", "Algae"), Candidate("Birds", "Birds"), Candidate("Ants", "Ants")),
    )


def _samples(anchor_model: torch.nn.Module | None = None) -> list[DecisionSample]:
    """Win / loss / tie rows with 2 and 3 candidates; anchor logits from `anchor_model`."""
    rows = []
    for value, margin, choice in ((1.0, 0.4, _choice()), (-1.0, -0.6, _three_way_choice()), (0.0, 0.0, _choice())):
        obs = _obs(actor=0)
        enc_obs = encode_observation(obs, _VOCAB)  # type: ignore[arg-type]
        enc_choice = encode_choice(choice, obs, _VOCAB)  # type: ignore[arg-type]
        n = len(choice.candidates)
        logits = None
        if anchor_model is not None:
            logits = anchor_policy_logits(anchor_model, [enc_obs], [enc_choice], [n])[0]
        policy = np.zeros(n, dtype=np.float32)
        policy[0], policy[1:] = 0.7, 0.3 / (n - 1)
        rows.append(
            DecisionSample(
                observation=enc_obs, choice=enc_choice, policy_target=policy, value_target=value,
                seat=0, was_searched=True, seed_key="train-0000", map_name="tharsis", generation=2,
                play_temperature=1.0, margin_target=margin, anchor_logits=logits,
            )
        )
    return rows


def _perturb(model: torch.nn.Module, scale: float = 0.5, seed: int = 3) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(scale * torch.randn(p.shape, generator=gen))


# --- KL anchor ---------------------------------------------------------------------------------


def test_kl_matches_the_closed_form_over_legal_candidates() -> None:
    logits = torch.tensor([[1.0, -0.5, 3.0], [0.2, 0.1, 99.0]])
    anchor = torch.tensor([[0.0, 0.5, -1.0], [1.0, -1.0, 0.0]])
    mask = torch.tensor([[True, True, True], [True, True, False]])
    has = torch.tensor([True, True])
    expected = []
    for b, n in ((0, 3), (1, 2)):
        p, q = F.softmax(logits[b, :n], -1), F.softmax(anchor[b, :n], -1)
        expected.append(float((p * (p.log() - q.log())).sum()))
    kl = kl_to_anchor(logits, anchor, mask, has)
    assert kl is not None and float(kl) == pytest.approx(float(np.mean(expected)), rel=1e-5)
    # Rows without anchor logits are left out; no anchored row at all gives None.
    one = kl_to_anchor(logits, anchor, mask, torch.tensor([True, False]))
    assert float(one) == pytest.approx(expected[0], rel=1e-5)
    assert kl_to_anchor(logits, anchor, mask, torch.tensor([False, False])) is None


def test_pad_anchor_logits_keeps_candidate_order_and_flags_missing_rows() -> None:
    logits, has = pad_anchor_logits([np.array([1.0, 2.0], np.float32), None], 3)
    assert logits.tolist() == [[1.0, 2.0, 0.0], [0.0, 0.0, 0.0]]
    assert has.tolist() == [True, False]


def test_kl_term_is_zero_when_the_learner_equals_the_anchor_and_positive_after_a_perturbation() -> None:
    model = _tiny_model(seed=1).eval()
    anchor = copy.deepcopy(model)
    targets = selfplay_targets(_samples(anchor))
    assert targets["has_anchor"].all()
    _, parts = selfplay_losses(model, targets)
    assert parts["kl_anchor"] == pytest.approx(0.0, abs=1e-6)

    _perturb(model)
    _, parts = selfplay_losses(model, targets)
    assert parts["kl_anchor"] > 1e-4


def test_random_init_run_drops_the_kl_anchor_term() -> None:
    assert LossWeights.from_config(RunConfig("x")).kl_anchor == pytest.approx(0.05)
    assert LossWeights.from_config(RunConfig("x", random_init=True)).kl_anchor == 0.0


# --- λ_h schedule ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hours", "expected"), [(0, 0.5), (1, 0.5), (2, 0.5), (5, 0.3), (8, 0.1), (12, 0.1)]
)
def test_human_aux_weight_follows_the_schedule(hours: float, expected: float) -> None:
    schedule = RunConfig("x").human_aux_schedule
    assert schedule == HumanAuxSchedule(start=0.5, end=0.1, decay_from_h=2.0, decay_to_h=8.0)
    assert schedule(hours) == pytest.approx(expected)
    assert schedule.at_seconds(hours * 3600.0) == pytest.approx(expected)


class _RecordingHumanAux:
    """Stand-in L_human that records each λ_h and adds a constant loss with no gradient."""

    def __init__(self) -> None:
        self.weights: list[float] = []

    def backward(self, model, *, weight, device, use_amp):
        self.weights.append(weight)
        return {"loss_total": 2.0}


@pytest.mark.parametrize(("hours", "expected"), [(0, 0.5), (5, 0.3), (12, 0.1)])
def test_learner_reads_lambda_from_the_training_clock(hours: float, expected: float) -> None:
    human = _RecordingHumanAux()
    learner = AlphaZeroLearner(
        RunConfig("x", learner_batch_size=2, use_amp=False), _tiny_model(seed=2), human_aux=human,
        training_clock=lambda: hours * 3600.0,
    )
    buffer = ReplayBuffer(16)
    buffer.extend(_samples())
    stats = learner.train(buffer, updates=2)
    assert human.weights == pytest.approx([expected, expected])
    assert stats.human_aux_weight == pytest.approx(expected)
    assert stats.human_aux_loss == pytest.approx(2.0)


# --- resident human sample ---------------------------------------------------------------------


def test_run_config_records_the_v2_learner_defaults() -> None:
    cfg = RunConfig("x").to_dict()
    assert cfg["human_aux_max_rows"] == 1_000_000
    assert cfg["kl_anchor_beta"] == pytest.approx(0.05)
    assert cfg["w_margin"] == pytest.approx(0.3)
    assert cfg["w_value"] == pytest.approx(1.0) and cfg["w_policy"] == pytest.approx(1.0)
    assert (cfg["learner_batch_size"], cfg["updates_per_cycle"], cfg["buffer_capacity"]) == (256, 16, 200_000)
    assert cfg["learning_rate"] == pytest.approx(5e-4)


def test_allocate_rows_is_proportional_to_draws_capped_by_pool_size() -> None:
    out = allocate_rows({"a": 1_000, "b": 30, "c": 1_000, "empty": 0}, {"a": 2, "b": 1, "c": 1, "empty": 5}, 400)
    assert out["b"] == 30  # its 100-row share exceeds the pool: taken whole
    assert out["a"] == 247 and out["c"] == 123  # the remaining 370 split 2:1, flooring remainder to a
    assert out["empty"] == 0
    assert sum(out.values()) == 400
    # A cap below one row per pool still fills the cap instead of flooring every pool to 0.
    tiny = allocate_rows({"a": 1_000, "b": 1_000, "c": 1_000}, {"a": 512, "b": 64, "c": 64}, 2)
    assert sum(tiny.values()) == 2 and tiny["a"] >= 1
    assert allocate_rows({"a": 5, "b": 5}, {"a": 1, "b": 1}, 100) == {"a": 5, "b": 5}


@pytest.mark.parametrize("cap", [10, 25])
def test_resident_human_sample_honours_the_row_cap(golden_data: WarmstartData, cap: int) -> None:  # noqa: F811
    sample = build_human_aux_sample(golden_data, max_rows=cap, rng=np.random.default_rng(0), weights=TINY_WARMSTART)
    assert sample.num_rows == cap
    # Cached rows are resident copies holding exactly the sampled rows, all train split.
    by_cache: dict[str, int] = {}
    for pool, rows in sample.pools.items():
        if pool not in ("setup", "value_setup"):
            cache = "midgame_value" if pool.startswith("value_") else pool
            by_cache[cache] = by_cache.get(cache, 0) + len(rows)
    for name, table in sample.data.caches.items():
        assert table.num_rows == by_cache.get(name, 0)
        assert table.num_rows <= len(golden_data.caches[name].split_indices["train"])
        assert len(table.split_indices["val"]) == len(table.split_indices["test"]) == 0
        assert not any(isinstance(a, np.memmap) for a in table.arrays.values())
    for pool, table in (("setup", golden_data.corpus.setup_choices), ("value_setup", golden_data.corpus.setup_outcomes)):
        assert set(sample.pools[pool].tolist()) <= set(table.split_indices["train"].tolist())


def test_resident_rows_batch_identically_to_the_full_cache(golden_data: WarmstartData) -> None:  # noqa: F811
    sample = build_human_aux_sample(golden_data, max_rows=40, rng=np.random.default_rng(1), weights=TINY_WARMSTART)
    for name, sub in sample.data.caches.items():
        if sub.num_rows == 0:
            continue
        full = golden_data.caches[name]
        src = {int(r): i for i, r in enumerate(full["src_row"])}
        rows = np.array([src[int(r)] for r in sub["src_row"]], dtype=np.int64)
        idx = np.arange(sub.num_rows)[::-1].copy()
        if full.task.kind == "value":
            got, want = sub.value_batch(idx), full.value_batch(rows[idx])
        elif full.task.kind == "research":
            picked = [[] for _ in idx]
            got, want = sub.research_batch(idx, picked=picked), full.research_batch(rows[idx], picked=picked)
        else:
            got, want = sub.policy_batch(idx), full.policy_batch(rows[idx])
        for g, w in zip(got, want):
            if isinstance(g, dict):
                assert g.keys() == w.keys()
                for key in g:
                    assert torch.equal(g[key], w[key]), (name, key)
            else:
                assert torch.equal(torch.nan_to_num(g, nan=-7.0), torch.nan_to_num(w, nan=-7.0)), name


def test_human_aux_loss_backpropagates_the_weighted_multitask_loss(golden_data: WarmstartData) -> None:  # noqa: F811
    sample = build_human_aux_sample(golden_data, max_rows=60, rng=np.random.default_rng(2), weights=TINY_WARMSTART)
    model = _tiny_model(seed=4)
    human = HumanAuxLoss(sample, rng=np.random.default_rng(0), weights=TINY_WARMSTART)
    stats = human.backward(model, weight=0.5, device="cpu", use_amp=False)
    assert {"loss_value", "loss_branch", "loss_corp"} <= set(stats)
    assert np.isfinite(stats["loss_total"])
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0.0 for g in grads)


def test_pool_draws_cover_every_sampler_pool() -> None:
    draws = pool_draws(WarmstartConfig())
    assert sum(draws.values()) == 3_328
    assert draws["value_setup"] + draws["value_research"] + draws["value_action"] == 512


# --- outcome-only value head, margin head --------------------------------------------------------


def test_win_logit_loss_uses_outcome_only_and_margin_has_its_own_loss() -> None:
    model = _tiny_model(seed=5).eval()
    targets = selfplay_targets(_samples())
    _, base = selfplay_losses(model, targets)
    shifted = dict(targets, margin=targets["margin"] + 0.5)
    _, moved = selfplay_losses(model, shifted)
    assert moved["value_loss"] == base["value_loss"]
    assert moved["margin_loss"] != base["margin_loss"]

    with torch.no_grad():
        out = model(targets["batch"], compute_policy=False, compute_value=True)
    bce = F.binary_cross_entropy_with_logits(out["value_logit"], torch.tensor([1.0, 0.0, 0.5]))
    assert base["value_loss"] == pytest.approx(float(bce), rel=1e-6)
    smooth = F.smooth_l1_loss(out["vp_margin_pred"], torch.tensor([0.4, -0.6, 0.0]), beta=VP_MARGIN_SMOOTH_L1_BETA)
    assert base["margin_loss"] == pytest.approx(float(smooth), rel=1e-6)


def test_margin_head_trains_on_selfplay_margins() -> None:
    model = _tiny_model(seed=6)
    targets = selfplay_targets(_samples())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    _, before = selfplay_losses(model.eval(), targets)
    model.train()
    for step in range(30):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = selfplay_losses(model, targets)
        loss.backward()
        if step == 0:
            head = model.vp_margin_head
            assert head is not None
            assert any(float(p.grad.abs().sum()) > 0.0 for p in head.parameters())
        optimizer.step()
    _, after = selfplay_losses(model.eval(), targets)
    assert after["margin_loss"] < before["margin_loss"]


# --- end to end --------------------------------------------------------------------------------


def test_learner_step_runs_all_five_terms_and_logs_each(golden_data: WarmstartData) -> None:  # noqa: F811
    model = _tiny_model(seed=7)
    anchor = copy.deepcopy(model)
    _perturb(anchor, scale=0.2)
    sample = build_human_aux_sample(golden_data, max_rows=60, rng=np.random.default_rng(3), weights=TINY_WARMSTART)
    human = HumanAuxLoss(sample, rng=np.random.default_rng(0), weights=TINY_WARMSTART)
    config = RunConfig("x", learner_batch_size=4, use_amp=False)
    learner = AlphaZeroLearner(config, model, human_aux=human)
    buffer = ReplayBuffer(16)
    buffer.extend(_samples(anchor))
    before = [p.detach().clone() for p in model.parameters()]

    stats = learner.train(buffer, updates=3, training_s=0.0)

    assert stats.updates == 3 and learner.step == 3
    assert stats.nonfinite_loss is None and not stats.nonfinite_params
    assert any(not torch.equal(b, p) for b, p in zip(before, model.parameters()))
    for step in stats.per_step:
        assert set(step) == {"policy_loss", "value_loss", "margin_loss", "kl_anchor", "human_aux_loss", "total_loss"}
        assert step["kl_anchor"] > 0.0
        expected = (step["policy_loss"] + step["value_loss"] + 0.3 * step["margin_loss"]
                    + 0.05 * step["kl_anchor"] + 0.5 * step["human_aux_loss"])
        assert step["total_loss"] == pytest.approx(expected, rel=1e-5)
    logged = stats.to_dict()
    for key in ("policy_loss", "value_loss", "margin_loss", "kl_anchor", "human_aux_loss", "total_loss"):
        assert logged[key] is not None and np.isfinite(logged[key]), key
    assert logged["human_aux_weight"] == pytest.approx(0.5)


class _AnchoredEngine(_FakeEngine):
    def __init__(self, anchor: torch.nn.Module) -> None:
        super().__init__()
        self.rows = _samples(anchor)

    def play_batch(self, num_games: int, *, stop_when=None):
        batch = super().play_batch(num_games, stop_when=stop_when)
        batch.samples = [dataclasses.replace(self.rows[i % len(self.rows)]) for i in range(len(batch.samples))]
        return batch


def test_cycle_record_logs_each_learner_term_on_the_training_clock(tmp_path) -> None:
    model = _tiny_model(seed=8)
    anchor = copy.deepcopy(model)
    _perturb(anchor, scale=0.2)
    config = _run_config(tmp_path, "v2-learner", use_amp=False)
    human = _RecordingHumanAux()
    learner = AlphaZeroLearner(config, model, human_aux=human)
    now = [0.0]
    runner = SelfPlayRunner(
        config, engine=_AnchoredEngine(anchor), learner=learner, evaluator=_fake_dev,
        model=model, vocab=_VOCAB, clock=lambda: now[0],
    )
    learner.training_clock = runner.training_s  # as `SelfPlayRunner.create` wires it
    now[0] = 5 * 3600.0
    runner.run(max_cycles=1)
    record = runner.logger.cycles[-1]
    assert record.human_aux_weight == pytest.approx(0.3)
    assert record.human_aux_loss == pytest.approx(2.0)
    for value in (record.policy_loss, record.value_loss, record.margin_loss, record.kl_anchor, record.total_loss):
        assert value is not None and np.isfinite(value)
