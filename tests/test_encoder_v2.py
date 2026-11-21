"""Encoder v2 and VP-margin heads (#42)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tm_bot.comparator.model import (
    ComparatorConfig,
    ComparatorNetwork,
    load_comparator_checkpoint,
    save_comparator_checkpoint,
)
from tm_bot.env.types import Candidate, Choice
from tm_bot.models import CardVocab, PolicyValueAgent, TMModelConfig, TMNetwork
from tm_bot.models.encode import (
    CHOICE_CTX_DIM,
    ENCODER_VERSION,
    choice_ctx_dim,
    classify_prompt_type,
    collate_encoded_batch,
    encode_choice,
    encode_observation,
)
from tm_bot.models.network import (
    VP_MARGIN_BOUND,
    EncoderVersionMismatchError,
    load_checkpoint,
    save_checkpoint,
    vp_margin_loss,
    vp_margin_target,
)
from tm_bot.models.vocab import PROMPT_TYPE_TO_ID, PROMPT_TYPES
from tm_bot.training.warmstart_builder import build_draft_pick_choice

# Root action menu labels as the engine emits them (Player.ts).
ROOT_BRANCH_LABELS = {
    "end_turn": "End Turn",
    "pass_generation": "Pass for this generation",
    "fund_award": "Fund an award (8 M€)",
    "standard_projects": "Standard projects",
    "first_action": "Take first action of Valley Trust corporation",
    "milestone": "Claim a milestone",
    "greenery": "Convert 8 plants into greenery",
    "heat": "Convert 8 heat into temperature",
    "play_project_card": "Play project card",
    "card_action": "Perform an action from a played card",
}


@pytest.fixture(scope="module")
def vocab() -> CardVocab:
    return CardVocab.from_card_data()


def _obs():
    obs, _ = build_draft_pick_choice(
        "Tharsis",
        0,
        draft_round_id=2,
        packet_cards=["Earth Catapult", "Research", "Soletta", "Virus"],
        chosen_corp="Teractor",
        chosen_preludes=["Allied Bank", "Biolab"],
        chosen_projects=["AI Central"],
        offered_corps=["Ecoline", "Teractor"],
        offered_preludes=["Allied Bank", "Biolab", "Dome Farming", "Loans"],
        offered_projects=["AI Central", "Asteroid", "Comet"],
        drafted_this_round=[],
    )
    obs["public"]["phase"] = "action"
    obs["public"]["generation"] = 3
    return obs


def _root_branch_choice() -> Choice:
    return Choice(
        seat=0,
        kind="branch",
        prompt="Take your next action",
        candidates=tuple(
            Candidate(id=f"branch:{i}", label=label)
            for i, label in enumerate(ROOT_BRANCH_LABELS.values())
        ),
    )


def test_v2_dims():
    assert len(PROMPT_TYPES) == 16
    assert CHOICE_CTX_DIM == 22 == choice_ctx_dim(2)
    assert choice_ctx_dim(1) == 20
    # Appended, so every v1 prompt id is unchanged.
    assert PROMPT_TYPES[:14][-1] == "other_card"


def test_each_root_branch_kind_has_a_distinct_nonzero_feature_vector(vocab):
    enc = encode_choice(_root_branch_choice(), _obs(), vocab)
    rows = [tuple(row.tolist()) for row in enc.cand_feats]
    for kind, row in zip(ROOT_BRANCH_LABELS, rows):
        assert any(v != 0.0 for v in row), f"{kind} encodes as all-zero"
    assert len(set(rows)) == len(rows), "two root branch kinds alias"


def test_v1_encoding_keeps_the_old_aliasing(vocab):
    enc = encode_choice(_root_branch_choice(), _obs(), vocab, encoder_version=1)
    feats = dict(zip(ROOT_BRANCH_LABELS, enc.cand_feats))
    assert np.array_equal(feats["end_turn"], feats["pass_generation"])
    assert not feats["standard_projects"].any()
    assert enc.choice_ctx.shape == (20,)


@pytest.mark.parametrize(
    ("kind", "prompt", "expected_v2", "expected_v1"),
    [
        # Adapter kinds as the real engine emits them (adapter/test/adapter.test.ts).
        ("project_card", "Standard projects", "standard_project", "project_card"),
        ("project_card", "Play project card", "project_card", "project_card"),
        ("card", "Perform an action from a played card", "card_action", "other_card"),
        ("card", "Select card to add 2 microbes", "other_card", "other_card"),
    ],
)
def test_action_phase_card_prompts_route_to_their_own_types(
    vocab, kind, prompt, expected_v2, expected_v1
):
    choice = Choice(
        seat=0,
        kind=kind,
        prompt=prompt,
        candidates=(Candidate("Asteroid", "Asteroid"), Candidate("Comet", "Comet")),
    )
    obs = _obs()
    assert classify_prompt_type(choice, obs) == expected_v2
    assert classify_prompt_type(choice, obs, encoder_version=1) == expected_v1
    enc = encode_choice(choice, obs, vocab)
    assert enc.prompt_type_id == PROMPT_TYPE_TO_ID[expected_v2]
    assert enc.choice_ctx[PROMPT_TYPE_TO_ID[expected_v2]] == 1.0


def _tiny_main(vocab: CardVocab, **overrides) -> TMNetwork:
    cfg = TMModelConfig(
        vocab_size=vocab.size,
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
        **overrides,
    )
    return TMNetwork.from_vocab(vocab, cfg)


def _tiny_comparator(vocab: CardVocab, **overrides) -> ComparatorNetwork:
    cfg = ComparatorConfig(
        vocab_size=vocab.size,
        d_model=32,
        card_hidden=32,
        card_emb_dim=16,
        card_static_proj_dim=16,
        slot_emb_dim=8,
        card_dyn_proj_dim=8,
        space_hidden=16,
        choice_hidden=32,
        cand_hidden=32,
        value_hidden=32,
        dropout=0.0,
        **overrides,
    )
    return ComparatorNetwork.from_vocab(vocab, cfg)


def _batch(vocab: CardVocab, n: int = 6) -> dict[str, torch.Tensor]:
    obs = _obs()
    choice = _root_branch_choice()
    enc_obs, enc_ch = [], []
    for i in range(n):
        o = dict(obs)
        o["public"] = dict(obs["public"])
        o["public"]["generation"] = 2 + i
        enc_obs.append(encode_observation(o, vocab))
        enc_ch.append(encode_choice(choice, o, vocab))
    return collate_encoded_batch(enc_obs, enc_ch)


@pytest.mark.parametrize("make", [_tiny_main, _tiny_comparator])
def test_vp_margin_head_is_bounded_and_trainable(vocab, make):
    torch.manual_seed(0)
    model = make(vocab)
    batch = _batch(vocab)
    out = model(batch)
    pred = out["vp_margin_pred"]
    assert pred.shape == (6,)
    assert bool((pred.abs() <= VP_MARGIN_BOUND).all())
    # P(win) still comes only from the win logit.
    assert torch.allclose(out["win_prob"], torch.sigmoid(out["value_logit"]))

    my_vp = torch.tensor([80.0, 40.0, 120.0, 60.0, 75.0, 90.0])
    opp_vp = torch.tensor([60.0, 110.0, 30.0, 60.0, 80.0, 20.0])
    target = vp_margin_target(my_vp, opp_vp)
    assert float(target.max()) == pytest.approx(1.2) and float(target.min()) == pytest.approx(-1.2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    with torch.no_grad():
        before = vp_margin_loss(model(batch)["vp_margin_pred"], target).item()
    for _ in range(20):
        opt.zero_grad()
        loss = vp_margin_loss(model(batch)["vp_margin_pred"], target)
        loss.backward()
        opt.step()
    with torch.no_grad():
        after = vp_margin_loss(model(batch)["vp_margin_pred"], target).item()
    assert after < before


def test_vp_margin_target_clips_scalars():
    assert vp_margin_target(100, 10) == pytest.approx(1.2)
    assert vp_margin_target(10, 35) == pytest.approx(-0.5)


def _as_v1_checkpoint(path: Path) -> None:
    """Rewrite a saved checkpoint the way pre-#42 code stored it (no version keys)."""
    payload = torch.load(path, weights_only=False)
    cfg = payload["config"]
    del cfg["encoder_version"], cfg["vp_margin_head"]
    cfg["num_prompt_types"] = 14
    cfg["choice_ctx_dim"] = 20
    payload["model_state_dict"] = {
        k: v for k, v in payload["model_state_dict"].items() if not k.startswith("vp_margin_head.")
    }
    torch.save(payload, path)


def test_main_checkpoint_version_mismatch_raises(vocab, tmp_path):
    v1 = _tiny_main(vocab, encoder_version=1, vp_margin_head=False, num_prompt_types=14, choice_ctx_dim=20)
    ckpt = save_checkpoint(tmp_path / "v1", v1, vocab)
    _as_v1_checkpoint(ckpt)
    with pytest.raises(EncoderVersionMismatchError):
        load_checkpoint(tmp_path / "v1")
    with pytest.raises(EncoderVersionMismatchError):
        PolicyValueAgent.from_checkpoint(tmp_path / "v1", encoder_version=ENCODER_VERSION)
    model, _, _ = load_checkpoint(tmp_path / "v1", encoder_version=1)
    assert model.config.encoder_version == 1 and model.vp_margin_head is None
    # An agent accepts any version by default and encodes with v1 widths end to end.
    agent = PolicyValueAgent.from_checkpoint(tmp_path / "v1")
    probs, _ = agent.predict(_obs(), _root_branch_choice())
    assert probs.shape[0] >= len(ROOT_BRANCH_LABELS)

    v2 = _tiny_main(vocab)
    save_checkpoint(tmp_path / "v2", v2, vocab)
    with pytest.raises(EncoderVersionMismatchError):
        load_checkpoint(tmp_path / "v2", encoder_version=1)
    model, _, _ = load_checkpoint(tmp_path / "v2")
    assert model.config.encoder_version == ENCODER_VERSION


def test_comparator_checkpoint_version_mismatch_raises(vocab, tmp_path):
    v1 = _tiny_comparator(vocab, encoder_version=1, vp_margin_head=False, num_prompt_types=14, choice_ctx_dim=20)
    ckpt = save_comparator_checkpoint(tmp_path / "v1", v1, vocab)
    _as_v1_checkpoint(ckpt)
    with pytest.raises(EncoderVersionMismatchError):
        load_comparator_checkpoint(tmp_path / "v1")
    model, _, _ = load_comparator_checkpoint(tmp_path / "v1", encoder_version=1)
    assert model.config.encoder_version == 1

    save_comparator_checkpoint(tmp_path / "v2", _tiny_comparator(vocab), vocab)
    with pytest.raises(EncoderVersionMismatchError):
        load_comparator_checkpoint(tmp_path / "v2", encoder_version=1)


def test_config_rejects_widths_that_do_not_match_its_encoder_version(vocab):
    with pytest.raises(ValueError):
        TMModelConfig(vocab_size=vocab.size, encoder_version=1)
