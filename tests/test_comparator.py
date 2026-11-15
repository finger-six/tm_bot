"""Tests for the Jev-inspired comparator agent (#25)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tm_bot.agents.card_data import load_card_data
from tm_bot.comparator.agent import ComparatorAgent
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
from tm_bot.models.encode import collate_encoded_batch, encode_choice, encode_observation
from tm_bot.models.network import TMModelConfig, TMNetwork
from tm_bot.models.vocab import CardVocab
from tm_bot.runs.buffer import normalize_policy_target
from tm_bot.runs.config import RunConfig
from tm_bot.training.warmstart_builder import build_corp_choice

_CARDS = load_card_data()
_VOCAB = CardVocab.from_card_data(_CARDS)


def _tiny_comparator(*, seed: int = 0) -> ComparatorNetwork:
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
        dropout=0.0,
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
                return_=float(advantages[i]),
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


def test_agent_kind_is_validated() -> None:
    with pytest.raises(ValueError, match="agent_kind"):
        RunConfig(run_name="bad", agent_kind="other")
    assert RunConfig(run_name="main").agent_kind == "main"
    assert RunConfig(run_name="cmp", agent_kind="comparator").agent_kind == "comparator"
