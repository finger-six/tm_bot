"""Label parity: the v2 action-phase builders produce the `Choice` the live adapter produces (#44).

A seeded game is driven through the real adapter (`WorkerClient`) to one root `branch` prompt, one
`Standard projects` prompt and one blue-card action `SelectCard` prompt. For each, the builder is fed
an equivalent warm-start row (the live observation in the snapshot column layout, with the live
candidates as its menu) and must produce the same prompt type, the same candidate ids and labels,
and the same per-candidate encoder features (the v2 branch bits for `branch`).
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pytest

from tm_bot.agents.card_data import load_card_data
from tm_bot.data import actionmenu
from tm_bot.data.logstate import SNAPSHOT_TAGS
from tm_bot.env import WorkerClient
from tm_bot.env.types import ActorObservation, Choice
from tm_bot.models.encode import classify_prompt_type, encode_choice
from tm_bot.models.vocab import CardVocab
from tm_bot.training.warmstart_builder import (
    branch_label,
    build_branch_choice,
    build_card_action_choice,
    build_standard_project_choice,
)

SEED_KEY, MAP, DRIVER_SEED, MAX_STEPS = "parity-probe-1", "tharsis", 7, 4000
_STOCKS = (("mc", "megacredits"), ("steel", "steel"), ("titanium", "titanium"), ("plants", "plants"),
           ("energy", "energy"), ("heat", "heat"))
_PRODS = (("mc_prod", "megacreditProduction"), ("steel_prod", "steelProduction"),
          ("titanium_prod", "titaniumProduction"), ("plants_prod", "plantProduction"),
          ("energy_prod", "energyProduction"), ("heat_prod", "heatProduction"))


def _prompt_key(ch: Choice) -> str | None:
    if ch.kind == "branch" and ch.prompt.startswith("Take your"):
        return "root"
    if ch.kind == "project_card" and ch.prompt == "Standard projects":
        return "standard_project"
    if ch.kind == "card" and ch.prompt.startswith("Perform an action from a played card"):
        return "card_action"
    return None


@pytest.fixture(scope="module")
def live_prompts() -> dict[str, tuple[Choice, ActorObservation]]:
    """First root and `Standard projects` prompts, and the widest card-action prompt, of a seeded game."""
    rng = random.Random(DRIVER_SEED)
    seen: dict[str, tuple[Choice, ActorObservation]] = {}
    with WorkerClient() as client:
        gid = client.create(seed_key=SEED_KEY, map=MAP)
        for _ in range(MAX_STEPS):
            seats = client.pending(gid)
            if not seats:
                break
            ch = client.choice(gid, seats[0])
            labels = [c.label for c in ch.candidates]
            key = _prompt_key(ch)
            if key is not None and (key not in seen or (key == "card_action"
                                                        and len(ch.candidates) > len(seen[key][0].candidates))):
                seen[key] = (ch, client.observe(gid, seats[0]))
            idx = rng.randrange(len(ch.candidates))
            if key == "root":
                for want in ("Perform an action from a played card", "Standard projects", "Play project card"):
                    if want in labels and rng.random() < 0.7:
                        idx = labels.index(want)
                        break
                if labels[idx] == "Pass for this generation" and len(labels) > 1:
                    idx = (idx + 1) % len(labels)
            client.choose(gid, seats[0], idx)
        client.close_game(gid)
    assert set(seen) == {"root", "standard_project", "card_action"}
    assert len(seen["card_action"][0].candidates) >= 2
    return seen


def _row_from_obs(obs: ActorObservation, *, menu: list[str], turn_action: int = 0) -> dict[str, Any]:
    """The live observation written in the warm-start snapshot column layout (`tm_bot.data.logstate`)."""
    pub, actor = obs["public"], obs["actor"]
    gp = pub["globalParameters"]
    row: dict[str, Any] = {
        "map": pub["map"], "seat": actor, "generation": pub["generation"], "phase": "action",
        "temperature": gp["temperature"], "oxygen": gp["oxygenLevel"], "oceans": gp["oceans"],
        "menu": menu, "turn_action": turn_action,
    }
    for prefix, p in (("me", pub["players"][actor]), ("opp", pub["players"][1 - actor])):
        row[f"{prefix}_tr"] = p["terraformRating"]
        for col, key in _STOCKS + _PRODS:
            row[f"{prefix}_{col}"] = p[key]
        row[f"{prefix}_tags"] = [p["tags"].get(t, 0) for t in SNAPSHOT_TAGS]
        row[f"{prefix}_cities"] = p["citiesCount"]
        row[f"{prefix}_greeneries"] = 0
        row[f"{prefix}_hand_count"] = p["cardsInHandCount"]
        row[f"{prefix}_tableau"] = sorted(c["name"] for c in p["tableau"])
    ms = [(m["name"], m["claimedBy"]) for m in pub["milestones"] if m["claimedBy"] is not None]
    aw = [(a["name"], a["fundedBy"]) for a in pub["awards"] if a["fundedBy"] is not None]
    priv = obs["private"]
    row.update(
        milestone_names=[n for n, _ in ms], milestone_seats=[s for _, s in ms],
        award_names=[n for n, _ in aw], award_seats=[s for _, s in aw],
        me_hand_exact=True, me_hand=sorted(c["name"] for c in priv["cardsInHand"]),
        me_preludes_in_hand=sorted(priv["preludeCardsInHand"]), me_drafted_this_round=[],
        me_corporation=priv["pickedCorporationCard"], me_setup_preludes=[], me_setup_projects_bought=[],
        me_setup_corps_discarded=[], me_setup_preludes_discarded=[], me_setup_projects_discarded=[],
        me_research_generations=[], me_research_offered=[], me_research_kept=[], me_research_discarded=[],
        me_discarded_seen=[],
    )
    return row


def _assert_parity(live: Choice, live_obs: ActorObservation, built: Choice, built_obs: ActorObservation,
                   vocab: CardVocab) -> None:
    assert (built.kind, built.prompt) == (live.kind, live.prompt)
    assert classify_prompt_type(built, built_obs) == classify_prompt_type(live, live_obs)
    assert [(c.id, c.label) for c in built.candidates] == [(c.id, c.label) for c in live.candidates]
    enc_live, enc_built = encode_choice(live, live_obs, vocab), encode_choice(built, built_obs, vocab)
    assert enc_built.prompt_type_id == enc_live.prompt_type_id
    np.testing.assert_array_equal(enc_built.cand_card_ids, enc_live.cand_card_ids)
    # Every candidate feature except the hand-cost pair (2, 3), which reads the observation's hand.
    cols = [i for i in range(enc_live.cand_feats.shape[1]) if i not in (2, 3)]
    np.testing.assert_array_equal(enc_built.cand_feats[:, cols], enc_live.cand_feats[:, cols])


def test_root_branch_choice_matches_live_adapter(live_prompts, vocab: CardVocab) -> None:
    live, obs = live_prompts["root"]
    label_to_key = {}
    for key in actionmenu.BRANCH_ORDER:
        probe = _row_from_obs(obs, menu=[key])
        label_to_key[branch_label(key, probe)] = key
    menu = [label_to_key[c.label] for c in live.candidates]
    turn = 0 if live.prompt == "Take your first action" else 1
    row = _row_from_obs(obs, menu=menu, turn_action=turn)
    built_obs, built = build_branch_choice(row)
    _assert_parity(live, obs, built, built_obs, vocab)
    # The reconstructed menu for the same state is the live legal set.
    specs = load_card_data()
    assert actionmenu.branch_menu(row, map_name=MAP, specs=specs, used_actions=(), turn_action=turn,
                                  opp_passed=False, last_played=None) == menu


def test_standard_project_choice_matches_live_adapter(live_prompts, vocab: CardVocab) -> None:
    live, obs = live_prompts["standard_project"]
    row = _row_from_obs(obs, menu=actionmenu.standard_project_menu(_row_from_obs(obs, menu=[])))
    assert sorted(row["menu"]) == sorted(c.id for c in live.candidates)
    built_obs, built = build_standard_project_choice(row)
    _assert_parity(live, obs, built, built_obs, vocab)


def test_card_action_choice_matches_live_adapter(live_prompts, vocab: CardVocab) -> None:
    live, obs = live_prompts["card_action"]
    row = _row_from_obs(obs, menu=[c.id for c in live.candidates])
    built_obs, built = build_card_action_choice(row)
    _assert_parity(live, obs, built, built_obs, vocab)
    assert classify_prompt_type(built, built_obs) == "card_action"


@pytest.fixture(scope="module")
def vocab() -> CardVocab:
    return CardVocab.from_card_data()
