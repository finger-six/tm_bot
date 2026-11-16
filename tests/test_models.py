"""Acceptance tests for the PyTorch observation encoder, heads, and checkpointing."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from tm_bot.agents.protocol import Agent
from tm_bot.env.types import Candidate, Choice
from tm_bot.models import (
    CardVocab,
    PolicyValueAgent,
    TMModelConfig,
    TMNetwork,
    build_random_init_model,
    collate_encoded_batch,
    encode_choice,
    encode_observation,
    load_checkpoint,
    save_checkpoint,
)
from tm_bot.training.warmstart_builder import build_corp_choice, build_draft_pick_choice


def _make_rich_obs_and_choice():
    obs, _ = build_draft_pick_choice(
        "Tharsis",
        0,
        draft_round_id=2,
        packet_cards=["Earth Catapult", "Research", "Soletta", "Virus"],
        chosen_corp="Teractor",
        chosen_preludes=["Allied Bank", "Biolab"],
        chosen_projects=["AI Central", "Anti-Gravity Technology", "Imported Nitrogen"],
        offered_corps=["Ecoline", "Teractor"],
        offered_preludes=["Allied Bank", "Biolab", "Dome Farming", "Loans"],
        offered_projects=[
            "AI Central",
            "Anti-Gravity Technology",
            "Asteroid",
            "Comet",
            "Deep Well Heating",
            "Imported Nitrogen",
            "Moss",
            "Search For Life",
            "Tardigrades",
            "Worms",
        ],
        drafted_this_round=["Arctic Algae", "Zeppelins"],
    )
    # Add opp tableau + usedActions + multiple hand cards to exercise all card sets
    obs["public"]["players"][1]["tableau"] = [
        {"name": "Ecoline", "resources": 0, "isDisabled": False},
        {"name": "Kelp Farming", "resources": 2, "isDisabled": False},
    ]
    obs["public"]["players"][1]["usedActions"] = ["Kelp Farming"]
    obs["public"]["players"][0]["usedActions"] = ["Teractor"]
    choice = Choice(
        seat=0,
        kind="card",
        prompt="Select a card to keep and pass the rest to blue",
        candidates=tuple(
            Candidate(id=c, label=c)
            for c in ["Earth Catapult", "Research", "Soletta", "Virus"]
        ),
    )
    return obs, choice


def test_parameter_count_in_target_range():
    vocab = CardVocab.from_card_data()
    model = TMNetwork.from_vocab(vocab)
    n_params = model.count_parameters()
    assert 2_000_000 <= n_params <= 5_000_000, f"Expected 2-5M params, got {n_params:,}"


def test_deterministic_and_order_invariant_encoding():
    vocab = CardVocab.from_card_data()
    obs1, ch1 = _make_rich_obs_and_choice()
    obs2 = copy.deepcopy(obs1)

    # Permute every set-like field in obs2
    obs2["private"]["cardsInHand"].reverse()
    obs2["private"]["dealtCorporationCards"].reverse()
    obs2["private"]["dealtPreludeCards"].reverse()
    obs2["private"]["dealtProjectCards"].reverse()
    obs2["private"]["draftedCards"].reverse()
    obs2["private"]["currentDraftPacket"].reverse()
    obs2["public"]["players"][0]["tableau"].reverse()
    obs2["public"]["players"][1]["tableau"].reverse()
    obs2["public"]["spaces"].reverse()
    obs2["public"]["milestones"].reverse()
    obs2["public"]["awards"].reverse()

    enc1 = encode_observation(obs1, vocab)
    enc2 = encode_observation(obs2, vocab)
    assert (enc1.obs_scalars == enc2.obs_scalars).all()
    assert (enc1.card_ids == enc2.card_ids).all()
    assert (enc1.card_slots == enc2.card_slots).all()
    assert (enc1.card_dyn == enc2.card_dyn).all()
    assert (enc1.space_feats == enc2.space_feats).all()

    # Check full model outputs are identical across permutations
    torch.manual_seed(0)
    model = TMNetwork.from_vocab(vocab)
    model.eval()
    ch1_enc = encode_choice(ch1, obs1, vocab)
    ch2_enc = encode_choice(ch1, obs2, vocab)
    b1 = collate_encoded_batch([enc1], [ch1_enc])
    b2 = collate_encoded_batch([enc2], [ch2_enc])

    with torch.no_grad():
        o1 = model(b1)
        o2 = model(b2)
    assert torch.allclose(o1["policy_probs"], o2["policy_probs"], atol=1e-6)
    assert torch.allclose(o1["value"], o2["value"], atol=1e-6)

    # Also verify candidate order permutation equivariance in policy logits
    rev_cands = tuple(reversed(ch1.candidates))
    ch_rev = Choice(seat=ch1.seat, kind=ch1.kind, prompt=ch1.prompt, candidates=rev_cands)
    b_rev = collate_encoded_batch([enc1], [encode_choice(ch_rev, obs1, vocab)])
    with torch.no_grad():
        o_rev = model(b_rev)
    assert torch.allclose(
        o1["policy_probs"][0],
        torch.flip(o_rev["policy_probs"][0], dims=[0]),
        atol=1e-5,
    )


def test_illegal_and_padded_candidates_get_exact_zero_probability():
    vocab = CardVocab.from_card_data()
    model = TMNetwork.from_vocab(vocab)
    model.eval()

    # Pair an observation with 2 candidates and one with 4 candidates so batch pads to 4
    obs_short, ch_short = build_corp_choice(
        "Tharsis",
        0,
        offered_corps=["CrediCor", "Ecoline"],
        offered_preludes=["Allied Bank", "Biolab", "Dome Farming", "Loans"],
        offered_projects=["Moss", "Worms"],
    )
    obs_long, ch_long = _make_rich_obs_and_choice()

    batch = collate_encoded_batch(
        [encode_observation(obs_short, vocab), encode_observation(obs_long, vocab)],
        [encode_choice(ch_short, obs_short, vocab), encode_choice(ch_long, obs_long, vocab)],
    )
    with torch.no_grad():
        out = model(batch)
    probs = out["policy_probs"]
    logits = out["policy_logits"]

    # Row 0 has 2 real candidates, indices 2 and 3 are padded -> exact 0.0
    assert probs[0, 2].item() == 0.0
    assert probs[0, 3].item() == 0.0
    assert torch.isneginf(logits[0, 2])
    assert torch.isneginf(logits[0, 3])
    assert abs(float(probs[0, :2].sum().item()) - 1.0) < 1e-6

    # Explicit illegal candidate mask on row 1
    batch_masked = dict(batch)
    mask = batch["cand_mask"].clone()
    mask[1, 1] = False
    mask[1, 3] = False
    batch_masked["cand_mask"] = mask
    with torch.no_grad():
        out_m = model(batch_masked)
    assert out_m["policy_probs"][1, 1].item() == 0.0
    assert out_m["policy_probs"][1, 3].item() == 0.0
    assert abs(float(out_m["policy_probs"][1].sum().item()) - 1.0) < 1e-6


def test_checkpoint_save_load_roundtrip(tmp_path: Path):
    vocab = CardVocab.from_card_data()
    cfg = TMModelConfig(vocab_size=vocab.size, d_model=128, n_heads=4, n_obs_layers=2, dim_feedforward=256, policy_ff_dim=256)
    model = TMNetwork.from_vocab(vocab, cfg)
    model.eval()

    obs, ch = _make_rich_obs_and_choice()
    batch = collate_encoded_batch([encode_observation(obs, vocab)], [encode_choice(ch, obs, vocab)])
    with torch.no_grad():
        before = model(batch)

    ckpt_dir = tmp_path / "ckpt"
    ckpt_path = save_checkpoint(ckpt_dir, model, vocab, metadata={"tag": "unit-test"})
    assert ckpt_path.exists()
    cfg_json_path = ckpt_dir / "config.json"
    assert cfg_json_path.exists()
    parsed_cfg = json.loads(cfg_json_path.read_text(encoding="utf-8"))
    assert parsed_cfg["parameter_count"] == model.count_parameters()
    assert parsed_cfg["config"]["d_model"] == 128
    assert parsed_cfg["metadata"]["tag"] == "unit-test"

    loaded_model, loaded_vocab, meta = load_checkpoint(ckpt_dir)
    assert meta["tag"] == "unit-test"
    assert loaded_vocab.names == vocab.names
    assert loaded_model.config == model.config
    with torch.no_grad():
        after = loaded_model(batch)
    assert torch.allclose(before["policy_probs"], after["policy_probs"], atol=0.0)
    assert torch.allclose(before["value"], after["value"], atol=0.0)

    # PolicyValueAgent implements Agent protocol cleanly
    agent = PolicyValueAgent.from_checkpoint(ckpt_dir)
    assert isinstance(agent, Agent)
    agent.reset(seed=7)
    idx = agent.act(obs, ch)
    assert 0 <= idx < len(ch.candidates)


def _small_config(vocab: CardVocab) -> TMModelConfig:
    return TMModelConfig(
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
    )


def _trained_tiny_model(vocab: CardVocab, *, seed: int) -> TMNetwork:
    """A small checkpoint whose learned weights differ from any fresh init."""
    torch.manual_seed(seed)
    model = TMNetwork.from_vocab(vocab, _small_config(vocab))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.1)
    return model


def test_build_random_init_model_keeps_architecture_not_weights(tmp_path: Path):
    vocab = CardVocab.from_card_data()
    warm = _trained_tiny_model(vocab, seed=3)
    checkpoint = tmp_path / "warmstart"
    save_checkpoint(checkpoint, warm, vocab, metadata={"tag": "warmstart"})

    model, loaded_vocab, metadata = build_random_init_model(checkpoint, seed=123)

    assert metadata["tag"] == "warmstart"
    assert loaded_vocab.names == vocab.names
    assert model.config == warm.config
    warm_state = warm.state_dict()
    fresh_state = model.state_dict()
    assert set(fresh_state) == set(warm_state)
    for name, tensor in fresh_state.items():
        assert tensor.shape == warm_state[name].shape
        if name == "card_static_matrix":
            # Non-learned card features come from the checkpoint unchanged.
            assert torch.equal(tensor, warm_state[name])
        else:
            assert not torch.equal(tensor, warm_state[name]), name


def test_build_random_init_model_is_deterministic_and_seeded(tmp_path: Path):
    vocab = CardVocab.from_card_data()
    checkpoint = tmp_path / "warmstart"
    save_checkpoint(checkpoint, _trained_tiny_model(vocab, seed=5), vocab)

    first, _, _ = build_random_init_model(checkpoint, seed=7)
    second, _, _ = build_random_init_model(checkpoint, seed=7)
    other, _, _ = build_random_init_model(checkpoint, seed=8)
    assert all(
        torch.equal(tensor, second.state_dict()[name])
        for name, tensor in first.state_dict().items()
    )
    assert any(
        not torch.equal(tensor, other.state_dict()[name])
        for name, tensor in first.state_dict().items()
        if name != "card_static_matrix"
    )
    # The seed fully determines the init: the caller's global RNG never leaks in.
    torch.manual_seed(1)
    caller_state = torch.get_rng_state()
    third, _, _ = build_random_init_model(checkpoint, seed=7)
    # ...and the call leaves the caller's RNG stream untouched.
    assert torch.equal(torch.get_rng_state(), caller_state)
    torch.manual_seed(999)
    fourth, _, _ = build_random_init_model(checkpoint, seed=7)
    assert all(
        torch.equal(tensor, fourth.state_dict()[name])
        for name, tensor in third.state_dict().items()
    )


def test_empty_card_and_space_sets_no_nan():
    vocab = CardVocab.from_card_data()
    model = TMNetwork.from_vocab(vocab)
    model.eval()
    obs, ch = build_corp_choice(
        "Tharsis",
        0,
        offered_corps=["CrediCor", "Ecoline"],
        offered_preludes=[],
        offered_projects=[],
    )
    obs["public"]["spaces"] = []
    obs["private"]["dealtCorporationCards"] = []
    batch = collate_encoded_batch([encode_observation(obs, vocab)], [encode_choice(ch, obs, vocab)])
    with torch.no_grad():
        out = model(batch)
    assert torch.isfinite(out["policy_probs"]).all()
    assert torch.isfinite(out["value"]).all()


def test_all_choice_kinds_encode_and_forward():
    vocab = CardVocab.from_card_data()
    cfg = TMModelConfig(vocab_size=vocab.size, d_model=64, n_heads=4, n_obs_layers=1, dim_feedforward=128, policy_ff_dim=128)
    model = TMNetwork.from_vocab(vocab, cfg)
    model.eval()
    obs, _ = _make_rich_obs_and_choice()
    sp0 = obs["public"]["spaces"][2]["id"]
    sp1 = obs["public"]["spaces"][3]["id"]
    choices = [
        Choice(0, "branch", "Take action", (Candidate("0", "Play project card"), Candidate("1", "Pass for this generation"))),
        Choice(0, "space", "Select space", (Candidate(sp0, sp0), Candidate(sp1, sp1))),
        Choice(0, "payment", "Select payment", (Candidate("megacredits=10", "10 MC"), Candidate("megacredits=8,steel=1", "8 MC + 1 steel"))),
        Choice(0, "resource", "Select resource", (Candidate("titanium", "Titanium"), Candidate("plants", "Plants"))),
        Choice(0, "player", "Select player", (Candidate("red", "red"), Candidate("blue", "blue"), Candidate("neutral", "neutral"))),
        Choice(0, "amount", "Select amount", (Candidate("1", "1"), Candidate("5", "5"))),
        Choice(0, "project_card", "Play project", (Candidate("Earth Catapult", "Earth Catapult"), Candidate("Research", "Research"))),
    ]
    enc_obs = [encode_observation(obs, vocab) for _ in choices]
    enc_chs = [encode_choice(ch, obs, vocab) for ch in choices]
    batch = collate_encoded_batch(enc_obs, enc_chs)
    with torch.no_grad():
        out = model(batch)
    assert out["policy_probs"].shape[0] == len(choices)
    assert torch.isfinite(out["policy_probs"]).all()
    assert torch.isfinite(out["value"]).all()

