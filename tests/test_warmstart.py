"""Acceptance tests for warm-start reconstruction parity, fast batching, and training step."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from tm_bot.agents.card_data import load_card_data
from tm_bot.env import WorkerClient
from tm_bot.models import (
    CardVocab,
    TMModelConfig,
    TMNetwork,
    collate_encoded_batch,
    encode_choice,
    encode_observation,
)
from tm_bot.training import (
    DraftPicksTable,
    FastBatchBuilder,
    SetupChoicesTable,
    SetupOutcomesTable,
    TrainConfig,
    build_corp_choice,
    build_draft_pick_choice,
    build_prelude_choice,
    build_project_buy_choice,
    build_setup_outcome_observation,
    train_one_step,
)


def _assert_structure_parity(synth: Any, live: Any, path: str = "obs") -> None:
    """Assert two nested observation dicts have identical keys and value types."""
    if live is None:
        assert synth is None or isinstance(synth, (str, int, float, bool)), (
            f"{path}: unexpected {type(synth)} when live is None"
        )
        return
    if isinstance(live, dict):
        assert isinstance(synth, dict), f"{path}: expected dict, got {type(synth)}"
        assert set(synth.keys()) == set(live.keys()), (
            f"{path}: key mismatch synth={sorted(synth.keys())} live={sorted(live.keys())}"
        )
        for k in live:
            if k in ("spaces", "milestones", "awards", "publicLog"):
                assert isinstance(synth[k], list), f"{path}.{k}: expected list"
                if live[k] and synth[k]:
                    _assert_structure_parity(synth[k][0], live[k][0], f"{path}.{k}[0]")
            elif k == "players":
                assert len(synth[k]) == len(live[k]) == 2
                for idx in range(2):
                    _assert_structure_parity(synth[k][idx], live[k][idx], f"{path}.players[{idx}]")
            elif k in ("tableau", "cardsInHand"):
                assert isinstance(synth[k], list), f"{path}.{k}: expected list"
                if live[k] and synth[k]:
                    _assert_structure_parity(synth[k][0], live[k][0], f"{path}.{k}[0]")
            else:
                _assert_structure_parity(synth[k], live[k], f"{path}.{k}")
    elif isinstance(live, list):
        assert isinstance(synth, list), f"{path}: expected list, got {type(synth)}"
        if live and synth:
            assert type(synth[0]) is type(live[0]), (
                f"{path}[0]: element type mismatch {type(synth[0])} vs {type(live[0])}"
            )
    elif isinstance(live, bool):
        assert isinstance(synth, bool), f"{path}: expected bool, got {type(synth)}"
    elif isinstance(live, int):
        assert isinstance(synth, int) and not isinstance(synth, bool), (
            f"{path}: expected int, got {type(synth)}"
        )
    elif isinstance(live, str):
        assert isinstance(synth, str), f"{path}: expected str, got {type(synth)}"


def test_builder_adapter_parity_initial_buy_and_draft():
    """Verify warmstart_builder outputs match live Node adapter schema, types, and candidates."""
    cards = load_card_data()
    with WorkerClient() as client:
        gid = client.create("warmstart-parity-seed-1", "tharsis")
        try:
            # 1. Corporation choice
            live_obs_corp = client.observe(gid, 0)
            live_ch_corp = client.choice(gid, 0)
            priv0 = live_obs_corp["private"]
            corps = list(priv0["dealtCorporationCards"])
            prels = list(priv0["dealtPreludeCards"])
            projs = list(priv0["dealtProjectCards"])

            syn_obs_corp, syn_ch_corp = build_corp_choice("Tharsis", 0, corps, prels, projs)
            _assert_structure_parity(syn_obs_corp, live_obs_corp, "corp_obs")
            assert syn_ch_corp.kind == live_ch_corp.kind == "card"
            assert syn_ch_corp.prompt == live_ch_corp.prompt
            assert [c.id for c in syn_ch_corp.candidates] == [c.id for c in live_ch_corp.candidates]
            assert syn_obs_corp["private"]["pickedCorporationCard"] is None

            # Pick corp 0 in live engine -> prelude step 0
            chosen_corp = live_ch_corp.candidates[0].id
            client.choose(gid, 0, 0)
            live_obs_p0 = client.observe(gid, 0)
            live_ch_p0 = client.choice(gid, 0)
            syn_obs_p0, syn_ch_p0 = build_prelude_choice("Tharsis", 0, corps, prels, projs)
            _assert_structure_parity(syn_obs_p0, live_obs_p0, "prelude0_obs")
            assert syn_ch_p0.kind == live_ch_p0.kind == "multi_card"
            assert syn_ch_p0.prompt == live_ch_p0.prompt
            assert [c.id for c in syn_ch_p0.candidates] == [c.id for c in live_ch_p0.candidates]

            # Pick prelude 0 -> prelude step 1
            p_first = live_ch_p0.candidates[0].id
            client.choose(gid, 0, 0)
            live_ch_p1 = client.choice(gid, 0)
            syn_obs_p1, syn_ch_p1 = build_prelude_choice(
                "Tharsis", 0, corps, prels, projs, picked_preludes=[p_first]
            )
            assert syn_ch_p1.kind == live_ch_p1.kind == "multi_card"
            assert syn_ch_p1.prompt == live_ch_p1.prompt
            assert [c.id for c in syn_ch_p1.candidates] == [c.id for c in live_ch_p1.candidates]

            # Pick prelude 1 -> done -> project buy step 0
            p_second = live_ch_p1.candidates[0].id
            client.choose(gid, 0, 0)
            ch_pdone = client.choice(gid, 0)
            client.choose(gid, 0, next(i for i, c in enumerate(ch_pdone.candidates) if c.id == "done"))

            live_obs_pj0 = client.observe(gid, 0)
            live_ch_pj0 = client.choice(gid, 0)
            syn_obs_pj0, syn_ch_pj0 = build_project_buy_choice("Tharsis", 0, corps, prels, projs)
            _assert_structure_parity(syn_obs_pj0, live_obs_pj0, "proj0_obs")
            assert syn_ch_pj0.kind == live_ch_pj0.kind == "multi_card"
            assert syn_ch_pj0.prompt == live_ch_pj0.prompt
            assert [c.id for c in syn_ch_pj0.candidates] == [c.id for c in live_ch_pj0.candidates]

            # Pick 1 project card -> project buy step 1
            pj_first = live_ch_pj0.candidates[0].id
            client.choose(gid, 0, 0)
            live_ch_pj1 = client.choice(gid, 0)
            _, syn_ch_pj1 = build_project_buy_choice(
                "Tharsis", 0, corps, prels, projs, picked_projects=[pj_first]
            )
            assert [c.id for c in syn_ch_pj1.candidates] == [c.id for c in live_ch_pj1.candidates]
            assert [c.label for c in syn_ch_pj1.candidates] == [c.label for c in live_ch_pj1.candidates]

            # Finish seat 0 initial setup -> seat 0 is deferred while seat 1 decides
            done_idx = next(i for i, c in enumerate(live_ch_pj1.candidates) if c.id == "done")
            client.choose(gid, 0, done_idx)
            assert client.pending(gid) == [1]

            # Seat 1 setup parity (must match warmstart_builder with no leak from seat 0)
            live_obs1_corp = client.observe(gid, 1)
            live_ch1_corp = client.choice(gid, 1)
            priv1 = live_obs1_corp["private"]
            corps1 = list(priv1["dealtCorporationCards"])
            prels1 = list(priv1["dealtPreludeCards"])
            projs1 = list(priv1["dealtProjectCards"])

            syn_obs1_corp, syn_ch1_corp = build_corp_choice("Tharsis", 1, corps1, prels1, projs1)
            _assert_structure_parity(syn_obs1_corp, live_obs1_corp, "seat1_corp_obs")
            assert [c.id for c in syn_ch1_corp.candidates] == [c.id for c in live_ch1_corp.candidates]
            assert live_obs1_corp["public"]["players"][0]["cardsInHandCount"] == syn_obs1_corp["public"]["players"][0]["cardsInHandCount"] == 0
            assert live_obs1_corp["public"]["players"][0]["megacredits"] == syn_obs1_corp["public"]["players"][0]["megacredits"] == 0
            assert live_obs1_corp["public"]["players"][0]["tableau"] == syn_obs1_corp["public"]["players"][0]["tableau"] == []
            assert live_obs1_corp["public"]["deckSize"] == syn_obs1_corp["public"]["deckSize"]
            assert live_obs1_corp["public"]["discardPileSize"] == syn_obs1_corp["public"]["discardPileSize"] == 0
            assert live_obs1_corp["public"]["otherDeckSizes"] == syn_obs1_corp["public"]["otherDeckSizes"]

            client.choose(gid, 1, 0)
            live_obs1_p0 = client.observe(gid, 1)
            live_ch1_p0 = client.choice(gid, 1)
            syn_obs1_p0, syn_ch1_p0 = build_prelude_choice("Tharsis", 1, corps1, prels1, projs1)
            _assert_structure_parity(syn_obs1_p0, live_obs1_p0, "seat1_prelude0_obs")
            assert [c.id for c in syn_ch1_p0.candidates] == [c.id for c in live_ch1_p0.candidates]

            p1_first = live_ch1_p0.candidates[0].id
            client.choose(gid, 1, 0)
            live_ch1_p1 = client.choice(gid, 1)
            _, syn_ch1_p1 = build_prelude_choice(
                "Tharsis", 1, corps1, prels1, projs1, picked_preludes=[p1_first]
            )
            assert [c.id for c in syn_ch1_p1.candidates] == [c.id for c in live_ch1_p1.candidates]
            client.choose(gid, 1, 0)
            ch1_pdone = client.choice(gid, 1)
            client.choose(gid, 1, next(i for i, c in enumerate(ch1_pdone.candidates) if c.id == "done"))

            live_obs1_pj0 = client.observe(gid, 1)
            live_ch1_pj0 = client.choice(gid, 1)
            syn_obs1_pj0, syn_ch1_pj0 = build_project_buy_choice("Tharsis", 1, corps1, prels1, projs1)
            _assert_structure_parity(syn_obs1_pj0, live_obs1_pj0, "seat1_proj0_obs")
            assert [c.id for c in syn_ch1_pj0.candidates] == [c.id for c in live_ch1_pj0.candidates]
            assert live_obs1_pj0["public"]["players"][0]["cardsInHandCount"] == syn_obs1_pj0["public"]["players"][0]["cardsInHandCount"] == 0
            assert live_obs1_pj0["public"]["discardPileSize"] == syn_obs1_pj0["public"]["discardPileSize"] == 0
            assert live_obs1_pj0["public"]["otherDeckSizes"] == syn_obs1_pj0["public"]["otherDeckSizes"]

            while 1 in client.pending(gid):
                ch1 = client.choice(gid, 1)
                idx1 = next((i for i, c in enumerate(ch1.candidates) if c.id == "done"), 0)
                client.choose(gid, 1, idx1)

            live_obs_draft = None
            live_ch_draft = None
            draft_seat = 0
            for _ in range(200):
                seats = client.pending(gid)
                if not seats:
                    break
                s = seats[0]
                obs_s = client.observe(gid, s)
                if obs_s["public"]["phase"] == "drafting":
                    live_obs_draft = obs_s
                    live_ch_draft = client.choice(gid, s)
                    draft_seat = s
                    break
                ch_s = client.choice(gid, s)
                if ch_s.kind == "branch":
                    idx = next((i for i, c in enumerate(ch_s.candidates) if "pass" in c.label.lower()), 0)
                else:
                    idx = next((i for i, c in enumerate(ch_s.candidates) if c.id == "done"), 0)
                client.choose(gid, s, idx)

            assert live_obs_draft is not None and live_ch_draft is not None
            pkt = [c.id for c in live_ch_draft.candidates]
            syn_obs_d, syn_ch_d = build_draft_pick_choice(
                "Tharsis",
                draft_seat,
                draft_round_id=1,
                packet_cards=pkt,
                chosen_corp=chosen_corp,
                chosen_preludes=[p_first, p_second],
                chosen_projects=[pj_first],
                offered_corps=corps,
                offered_preludes=prels,
                offered_projects=projs,
                cards_by_name=cards,
            )
            _assert_structure_parity(syn_obs_d, live_obs_draft, "draft_obs")
            assert syn_ch_d.kind == live_ch_draft.kind == "card"
            assert syn_ch_d.prompt == live_ch_draft.prompt
            assert [c.id for c in syn_ch_d.candidates] == [c.id for c in live_ch_draft.candidates]
        finally:
            client.close_game(gid)


def _synthetic_tables(vocab: CardVocab):
    cid = vocab.card_id
    sc = SetupChoicesTable(
        map_id=np.array([0, 2], dtype=np.uint8),
        seat=np.array([0, 1], dtype=np.uint8),
        split=np.array([0, 0], dtype=np.uint8),
        offered_corps=np.array(
            [[cid("CrediCor"), cid("Ecoline")], [cid("Helion"), cid("Teractor")]],
            dtype=np.uint16,
        ),
        chosen_corp=np.array([cid("Ecoline"), cid("Teractor")], dtype=np.uint16),
        offered_preludes=np.array(
            [
                [cid("Allied Bank"), cid("Biolab"), cid("Dome Farming"), cid("Loans")],
                [cid("Acquired Space Agency"), cid("Biofuels"), cid("Ecology Experts"), cid("Power Generation")],
            ],
            dtype=np.uint16,
        ),
        chosen_preludes=np.array(
            [[cid("Allied Bank"), cid("Biolab")], [cid("Biofuels"), cid("Ecology Experts")]],
            dtype=np.uint16,
        ),
        offered_projects=np.array(
            [
                [cid(x) for x in ["AI Central", "Asteroid", "Comet", "Earth Catapult", "Kelp Farming", "Moss", "Research", "Soletta", "Tardigrades", "Worms"]],
                [cid(x) for x in ["AI Central", "Asteroid", "Comet", "Earth Catapult", "Kelp Farming", "Moss", "Research", "Soletta", "Tardigrades", "Worms"]],
            ],
            dtype=np.uint16,
        ),
        chosen_projects=np.array(
            [
                [cid("AI Central"), cid("Earth Catapult"), cid("Research"), 0, 0, 0, 0, 0, 0, 0],
                [cid("Kelp Farming"), cid("Soletta"), 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.uint16,
        ),
        chosen_proj_len=np.array([3, 2], dtype=np.uint8),
        split_indices={"train": np.array([0, 1]), "val": np.array([]), "test": np.array([])},
    )
    dp = DraftPicksTable(
        setup_idx=np.array([0, 1], dtype=np.int32),
        map_id=np.array([0, 2], dtype=np.uint8),
        seat=np.array([0, 1], dtype=np.uint8),
        split=np.array([0, 0], dtype=np.uint8),
        generation=np.array([2, 3], dtype=np.uint8),
        draft_round_id=np.array([1, 2], dtype=np.uint16),
        pick_index=np.array([2, 1], dtype=np.uint8),
        packet_cards=np.array(
            [
                [cid("Earth Catapult"), cid("Moss"), cid("Virus"), 0, 0],
                [cid("AI Central"), cid("Comet"), cid("Research"), cid("Worms"), 0],
            ],
            dtype=np.uint16,
        ),
        packet_len=np.array([3, 4], dtype=np.uint8),
        chosen_card=np.array([cid("Earth Catapult"), cid("AI Central")], dtype=np.uint16),
        drafted_this_round=np.array([[cid("Tardigrades"), 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint16),
        drafted_len=np.array([1, 0], dtype=np.uint8),
        split_indices={"train": np.array([0, 1]), "val": np.array([]), "test": np.array([])},
    )
    so = SetupOutcomesTable(
        map_id=np.array([0, 2], dtype=np.uint8),
        seat=np.array([0, 1], dtype=np.uint8),
        split=np.array([0, 0], dtype=np.uint8),
        chosen_corp=sc.chosen_corp,
        chosen_preludes=sc.chosen_preludes,
        chosen_projects=sc.chosen_projects,
        chosen_proj_len=sc.chosen_proj_len,
        won=np.array([1.0, 0.0], dtype=np.float32),
        split_indices={"train": np.array([0, 1]), "val": np.array([]), "test": np.array([])},
    )
    return sc, dp, so


def test_fast_batch_builder_matches_warmstart_builder_tensors():
    """Verify FastBatchBuilder produces exact tensor equivalence with warmstart_builder + encode_*."""
    cards = load_card_data()
    vocab = CardVocab.from_card_data(cards)
    builder = FastBatchBuilder(vocab, cards)
    sc, dp, so = _synthetic_tables(vocab)
    idx = np.array([0, 1], dtype=np.int64)
    maps = ["tharsis", "hellas"]

    # Corp parity
    fast_c, _ = builder.build_corp_batch(sc, idx)
    slow_obs_c, slow_ch_c = [], []
    for r in range(2):
        o, c = build_corp_choice(
            maps[r],
            int(sc.seat[r]),
            [vocab.names[i] for i in sc.offered_corps[r]],
            [vocab.names[i] for i in sc.offered_preludes[r]],
            [vocab.names[i] for i in sc.offered_projects[r]],
        )
        slow_obs_c.append(encode_observation(o, vocab))
        slow_ch_c.append(encode_choice(c, o, vocab))
    slow_c = collate_encoded_batch(slow_obs_c, slow_ch_c)
    for k in slow_c:
        assert torch.allclose(fast_c[k].float(), slow_c[k].float()), f"corp mismatch on {k}"

    # Prelude parity (step 0 and step 1)
    fast_p, _ = builder.build_prelude_batch(sc, idx, step_choices=np.array([0, 1]))
    o0, c0 = build_prelude_choice(
        maps[0],
        0,
        [vocab.names[i] for i in sc.offered_corps[0]],
        [vocab.names[i] for i in sc.offered_preludes[0]],
        [vocab.names[i] for i in sc.offered_projects[0]],
    )
    o1, c1 = build_prelude_choice(
        maps[1],
        1,
        [vocab.names[i] for i in sc.offered_corps[1]],
        [vocab.names[i] for i in sc.offered_preludes[1]],
        [vocab.names[i] for i in sc.offered_projects[1]],
        picked_preludes=[vocab.names[int(sc.chosen_preludes[1, 0])]],
    )
    slow_p = collate_encoded_batch(
        [encode_observation(o0, vocab), encode_observation(o1, vocab)],
        [encode_choice(c0, o0, vocab), encode_choice(c1, o1, vocab)],
    )
    for k in slow_p:
        assert torch.allclose(fast_p[k].float(), slow_p[k].float()), f"prelude mismatch on {k}"

    # Draft parity
    fast_d, _ = builder.build_draft_batch(dp, sc, idx)
    slow_obs_d, slow_ch_d = [], []
    for r in range(2):
        o, c = build_draft_pick_choice(
            maps[r],
            int(dp.seat[r]),
            int(dp.draft_round_id[r]),
            [vocab.names[int(dp.packet_cards[r, j])] for j in range(int(dp.packet_len[r]))],
            vocab.names[int(sc.chosen_corp[r])],
            [vocab.names[int(x)] for x in sc.chosen_preludes[r]],
            [vocab.names[int(sc.chosen_projects[r, j])] for j in range(int(sc.chosen_proj_len[r]))],
            offered_corps=[vocab.names[int(x)] for x in sc.offered_corps[r]],
            offered_preludes=[vocab.names[int(x)] for x in sc.offered_preludes[r]],
            offered_projects=[vocab.names[int(x)] for x in sc.offered_projects[r]],
            drafted_this_round=[
                vocab.names[int(dp.drafted_this_round[r, j])]
                for j in range(int(dp.drafted_len[r]))
            ],
            cards_by_name=cards,
        )
        slow_obs_d.append(encode_observation(o, vocab))
        slow_ch_d.append(encode_choice(c, o, vocab))
    slow_d = collate_encoded_batch(slow_obs_d, slow_ch_d)
    for k in slow_d:
        assert torch.allclose(fast_d[k].float(), slow_d[k].float()), f"draft mismatch on {k}"

    # Outcome parity
    fast_v, _ = builder.build_outcome_batch(so, idx)
    slow_obs_v = []
    for r in range(2):
        o = build_setup_outcome_observation(
            maps[r],
            int(so.seat[r]),
            vocab.names[int(so.chosen_corp[r])],
            [vocab.names[int(x)] for x in so.chosen_preludes[r]],
            [vocab.names[int(so.chosen_projects[r, j])] for j in range(int(so.chosen_proj_len[r]))],
            cards_by_name=cards,
        )
        slow_obs_v.append(encode_observation(o, vocab))
    slow_v = collate_encoded_batch(slow_obs_v)
    for k in slow_v:
        assert torch.allclose(fast_v[k].float(), slow_v[k].float()), f"outcome mismatch on {k}"


def test_one_training_step_reduces_loss_on_fixed_batch():
    """Verify one gradient step strictly reduces multi-task loss on a fixed synthetic batch."""
    cards = load_card_data()
    vocab = CardVocab.from_card_data(cards)
    builder = FastBatchBuilder(vocab, cards)
    sc, dp, so = _synthetic_tables(vocab)
    idx = np.array([0, 1], dtype=np.int64)

    draft_b, draft_t = builder.build_draft_batch(dp, sc, idx)
    corp_b, corp_t = builder.build_corp_batch(sc, idx)
    prel_b, prel_t = builder.build_prelude_batch(sc, idx, step_choices=np.array([0, 1]))
    proj_b, proj_t = builder.build_project_buy_batch(sc, idx, picked_counts=np.array([1, 2]))
    out_b, out_t = builder.build_outcome_batch(so, idx)

    torch.manual_seed(123)
    cfg = TMModelConfig(
        vocab_size=vocab.size,
        d_model=64,
        n_heads=4,
        n_obs_layers=2,
        dim_feedforward=128,
        policy_ff_dim=128,
        dropout=0.0,
    )
    model = TMNetwork.from_vocab(vocab, cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tcfg = TrainConfig(use_amp=False)

    stats_before = train_one_step(
        model,
        optimizer,
        device=torch.device("cpu"),
        use_amp=False,
        weights=tcfg,
        draft_batch=draft_b,
        draft_targets=draft_t,
        corp_batch=corp_b,
        corp_targets=corp_t,
        prelude_batch=prel_b,
        prelude_targets=prel_t,
        project_batch=proj_b,
        project_targets=proj_t,
        outcome_batch=out_b,
        outcome_targets=out_t,
    )
    stats_after = train_one_step(
        model,
        optimizer,
        device=torch.device("cpu"),
        use_amp=False,
        weights=tcfg,
        draft_batch=draft_b,
        draft_targets=draft_t,
        corp_batch=corp_b,
        corp_targets=corp_t,
        prelude_batch=prel_b,
        prelude_targets=prel_t,
        project_batch=proj_b,
        project_targets=proj_t,
        outcome_batch=out_b,
        outcome_targets=out_t,
    )
    assert stats_after["loss_total"] < stats_before["loss_total"]


def test_opponent_private_isolation():
    """Reconstructed observations must keep opponent state at neutral defaults with zero private info."""
    obs, _ = build_draft_pick_choice(
        "Elysium",
        0,
        draft_round_id=3,
        packet_cards=["Earth Catapult", "Virus"],
        chosen_corp="Teractor",
        chosen_preludes=["Allied Bank", "Biolab"],
        chosen_projects=["Research"],
    )
    opp = obs["public"]["players"][1]
    assert opp["tableau"] == []
    assert opp["cardsInHandCount"] == 0
    assert opp["megacredits"] == 0
    assert opp["lastCardPlayed"] is None
    assert all(v == 0 for v in opp["tags"].values())
