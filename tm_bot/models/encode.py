"""Pure, order-invariant tensor encoding for ActorObservation and Choice."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from tm_bot.env.types import SUPPORTED_MAPS, ActorObservation, Choice
from tm_bot.models.vocab import (
    DONE_ID,
    PAD_ID,
    PROMPT_TYPE_TO_ID,
    PROMPT_TYPES,
    SLOT_TO_ID,
    TAG_NAMES,
    CardVocab,
)

_MAP_TO_IDX: dict[str, int] = {m: i for i, m in enumerate(SUPPORTED_MAPS)}
_NUM_MAP_SLOTS = len(SUPPORTED_MAPS) + 1  # +1 for unknown/other

_PHASES: tuple[str, ...] = (
    "research",
    "initial_drafting",
    "drafting",
    "preludes",
    "action",
    "production",
    "end",
    "other",
)
_PHASE_TO_IDX: dict[str, int] = {p: i for i, p in enumerate(_PHASES)}

_SPACE_TYPES: tuple[str, ...] = ("land", "ocean", "colony", "other")
_SPACE_TYPE_TO_IDX: dict[str, int] = {s: i for i, s in enumerate(_SPACE_TYPES)}

_PAYMENT_KV_RE = re.compile(r"([a-zA-Z]+)=([0-9]+)")
_DONE_COUNT_RE = re.compile(r"\((\d+)\s+selected\)")

# 5 map + 6 actor/turn + 11 gen/phase + 5 global_params + 6 deck_sizes
# + 80 players (2 * 40) + 35 milestones (5 * 7) + 25 awards (5 * 5) + 12 priv/hist = 185
OBS_SCALAR_DIM = 185
SPACE_FEAT_DIM = 24
CARD_DYN_DIM = 4
CHOICE_CTX_DIM = len(PROMPT_TYPES) + 6  # 20
CAND_FEAT_DIM = 32


@dataclass(frozen=True, slots=True)
class EncodedObservation:
    """Pure numpy tensor bundle for one ActorObservation."""

    obs_scalars: np.ndarray  # float32[OBS_SCALAR_DIM]
    card_ids: np.ndarray  # int64[N_cards]
    card_slots: np.ndarray  # int64[N_cards]
    card_dyn: np.ndarray  # float32[N_cards, CARD_DYN_DIM]
    space_feats: np.ndarray  # float32[N_spaces, SPACE_FEAT_DIM]


@dataclass(frozen=True, slots=True)
class EncodedChoice:
    """Pure numpy tensor bundle for one Choice conditioned on its ActorObservation."""

    choice_ctx: np.ndarray  # float32[CHOICE_CTX_DIM]
    prompt_type_id: int
    cand_card_ids: np.ndarray  # int64[C]
    cand_feats: np.ndarray  # float32[C, CAND_FEAT_DIM]
    picked_card_ids: np.ndarray  # int64[N_picked]


def classify_prompt_type(choice: Choice, obs: ActorObservation) -> str:
    """Map `(choice, obs)` to a canonical prompt type string."""
    kind = choice.kind
    prompt_l = choice.prompt.lower()
    phase = str(obs["public"]["phase"]).lower()

    if kind == "card":
        if "corporation" in prompt_l:
            return "initial_corp"
        if phase in ("drafting", "initial_drafting") or "pass the rest" in prompt_l:
            return "draft_pick"
        if phase == "preludes" or "prelude" in prompt_l:
            return "prelude_play"
        return "other_card"
    if kind == "multi_card":
        if "prelude" in prompt_l:
            return "initial_prelude"
        if "initial" in prompt_l and "buy" in prompt_l:
            return "initial_project_buy"
        if "buy" in prompt_l:
            return "research_buy"
        return "other_card"
    if kind in PROMPT_TYPE_TO_ID:
        return kind
    return "other_card"


def _encode_space_bonus(bonuses: Sequence[int], out: np.ndarray, offset: int) -> None:
    # Engine SpaceBonus: 0=Ti, 1=Steel, 2=Plant, 3=Draw, 4=Heat, 5=Ocean, 6=MC, 13/18=Temp
    for b in bonuses:
        bi = int(b)
        if 0 <= bi <= 6:
            out[offset + bi] += 1.0
        elif bi in (13, 18):
            out[offset + 7] += 1.0
        else:
            out[offset + 8] += 1.0


def _encode_player_scalars(p: Mapping[str, Any], out: np.ndarray, offset: int) -> int:
    out[offset + 0] = float(p["terraformRating"]) / 50.0
    out[offset + 1] = float(p["megacredits"]) / 100.0
    out[offset + 2] = float(p["steel"]) / 20.0
    out[offset + 3] = float(p["titanium"]) / 20.0
    out[offset + 4] = float(p["plants"]) / 20.0
    out[offset + 5] = float(p["energy"]) / 20.0
    out[offset + 6] = float(p["heat"]) / 20.0

    out[offset + 7] = float(p["megacreditProduction"]) / 30.0
    out[offset + 8] = float(p["steelProduction"]) / 10.0
    out[offset + 9] = float(p["titaniumProduction"]) / 10.0
    out[offset + 10] = float(p["plantProduction"]) / 10.0
    out[offset + 11] = float(p["energyProduction"]) / 10.0
    out[offset + 12] = float(p["heatProduction"]) / 10.0

    out[offset + 13] = float(p["steelValue"]) / 5.0
    out[offset + 14] = float(p["titaniumValue"]) / 5.0
    out[offset + 15] = float(p["cardCost"]) / 5.0
    out[offset + 16] = float(p["cardDiscount"]) / 10.0

    out[offset + 17] = float(p["cardsInHandCount"]) / 20.0
    out[offset + 18] = float(p["citiesCount"]) / 10.0
    out[offset + 19] = float(p["noTagsCount"]) / 10.0

    out[offset + 20] = float(p["actionsTakenThisRound"]) / 10.0
    out[offset + 21] = float(p["actionsTakenThisGame"]) / 50.0
    out[offset + 22] = float(p["availableBlueCardActionCount"]) / 10.0

    tags = p["tags"]
    for i, tag in enumerate(TAG_NAMES):
        out[offset + 23 + i] = float(tags.get(tag, 0)) / 10.0

    return offset + 23 + len(TAG_NAMES)  # +40


def encode_observation(obs: ActorObservation, vocab: CardVocab) -> EncodedObservation:
    """Pure, permutation-invariant encoding of one ActorObservation."""
    actor = int(obs["actor"])
    if actor not in (0, 1):
        raise ValueError(f"Invalid actor seat: {actor!r}")

    pub = obs["public"]
    priv = obs["private"]
    hist = obs["history"]
    players = pub["players"]
    me = players[actor]
    opp = players[1 - actor]

    scalars = np.zeros(OBS_SCALAR_DIM, dtype=np.float32)
    off = 0

    # Map one-hot (5)
    map_idx = _MAP_TO_IDX.get(str(pub["map"]).lower(), len(SUPPORTED_MAPS))
    scalars[off + map_idx] = 1.0
    off += _NUM_MAP_SLOTS

    # Actor & turn order (6)
    scalars[off + 0] = 1.0 if actor == 0 else 0.0
    scalars[off + 1] = 1.0 if actor == 1 else 0.0
    scalars[off + 2] = 1.0 if int(pub["firstPlayer"]) == actor else 0.0
    scalars[off + 3] = 1.0 if int(pub["activePlayer"]) == actor else 0.0
    scalars[off + 4] = 1.0 if bool(me["passed"]) else 0.0
    scalars[off + 5] = 1.0 if bool(opp["passed"]) else 0.0
    off += 6

    # Generation & phase (11)
    gen = int(pub["generation"])
    phase_str = str(pub["phase"]).lower()
    scalars[off + 0] = float(gen) / 20.0
    scalars[off + 1] = 1.0 if gen == 1 else 0.0
    ph_idx = _PHASE_TO_IDX.get(phase_str, _PHASE_TO_IDX["other"])
    scalars[off + 2 + ph_idx] = 1.0
    scalars[off + 2 + len(_PHASES)] = 1.0 if (gen == 1 and phase_str == "research") else 0.0
    off += 2 + len(_PHASES) + 1

    # Global parameters (5)
    gp = pub["globalParameters"]
    temp = int(gp["temperature"])
    oxy = int(gp["oxygenLevel"])
    oceans = int(gp["oceans"])
    rem_steps = max(0, (8 - temp) // 2) + max(0, 14 - oxy) + max(0, 9 - oceans)
    scalars[off + 0] = float(temp + 30) / 38.0
    scalars[off + 1] = float(oxy) / 14.0
    scalars[off + 2] = float(oceans) / 9.0
    scalars[off + 3] = 1.0 if bool(gp["isTerraformed"]) else 0.0
    scalars[off + 4] = float(rem_steps) / 42.0
    off += 5

    # Deck sizes (6)
    ods = pub["otherDeckSizes"]
    prel_sizes = ods["preludes"]
    scalars[off + 0] = float(pub["deckSize"]) / 208.0
    scalars[off + 1] = float(pub["discardPileSize"]) / 208.0
    scalars[off + 2] = float(ods["corporations"]["drawPile"]) / 20.0
    scalars[off + 3] = float(ods["corporations"]["discardPile"]) / 20.0
    scalars[off + 4] = float(prel_sizes["drawPile"]) / 35.0 if prel_sizes is not None else 0.0
    scalars[off + 5] = float(prel_sizes["discardPile"]) / 35.0 if prel_sizes is not None else 0.0
    off += 6

    # Actor-relative public player features: [me, opp] (80)
    off = _encode_player_scalars(me, scalars, off)
    off = _encode_player_scalars(opp, scalars, off)

    # Milestones (5 * 7 = 35), canonically sorted by milestone name for order invariance
    milestones = sorted(pub["milestones"], key=lambda m: str(m["name"]))[:5]
    for i, ms in enumerate(milestones):
        base = off + i * 7
        cb = ms["claimedBy"]
        if cb is None:
            scalars[base + 2] = 1.0
        elif int(cb) == actor:
            scalars[base + 0] = 1.0
        else:
            scalars[base + 1] = 1.0
        for sc in ms["scores"]:
            if int(sc["seat"]) == actor:
                scalars[base + 3] = float(sc["score"]) / 10.0
                scalars[base + 5] = 1.0 if bool(sc["claimable"]) else 0.0
            else:
                scalars[base + 4] = float(sc["score"]) / 10.0
                scalars[base + 6] = 1.0 if bool(sc["claimable"]) else 0.0
    off += 35

    # Awards (5 * 5 = 25), canonically sorted by award name
    awards = sorted(pub["awards"], key=lambda a: str(a["name"]))[:5]
    for i, aw in enumerate(awards):
        base = off + i * 5
        fb = aw["fundedBy"]
        if fb is None:
            scalars[base + 2] = 1.0
        elif int(fb) == actor:
            scalars[base + 0] = 1.0
        else:
            scalars[base + 1] = 1.0
        for sc in aw["scores"]:
            if int(sc["seat"]) == actor:
                scalars[base + 3] = float(sc["score"]) / 10.0
            else:
                scalars[base + 4] = float(sc["score"]) / 10.0
    off += 25

    # Private & history summary counts (12)
    scalars[off + 0] = float(len(priv["cardsInHand"])) / 20.0
    scalars[off + 1] = float(len(priv["dealtCorporationCards"])) / 5.0
    scalars[off + 2] = float(len(priv["dealtPreludeCards"])) / 5.0
    scalars[off + 3] = float(len(priv["dealtProjectCards"])) / 10.0
    scalars[off + 4] = 1.0 if priv["pickedCorporationCard"] is not None else 0.0
    scalars[off + 5] = float(len(priv["preludeCardsInHand"])) / 4.0
    scalars[off + 6] = float(len(priv["draftedCards"])) / 4.0
    scalars[off + 7] = float(len(priv["currentDraftPacket"])) / 4.0
    scalars[off + 8] = float(len(priv["pendingCardSelection"])) / 10.0
    scalars[off + 9] = float(len(hist["drafts"])) / 40.0
    scalars[off + 10] = float(len(hist["researchBuys"])) / 20.0
    scalars[off + 11] = 1.0 if hist["initialBuy"] is not None else 0.0
    off += 12

    assert off == OBS_SCALAR_DIM

    # Collect card tokens canonically as (slot_id, card_id, d0, d1, d2, d3)
    tokens: list[tuple[int, int, float, float, float, float]] = []

    def add_simple(slot_name: str, names: Sequence[str]) -> None:
        sid = SLOT_TO_ID[slot_name]
        for nm in names:
            tokens.append((sid, vocab.card_id(nm), 0.0, 0.0, 0.0, 0.0))

    s_hand = SLOT_TO_ID["my_hand"]
    for item in priv["cardsInHand"]:
        tokens.append((s_hand, vocab.card_id(item["name"]), float(item["calculatedCost"]) / 50.0, 0.0, 0.0, 0.0))

    my_used = set(me["usedActions"])
    s_mytab = SLOT_TO_ID["my_tableau"]
    for item in me["tableau"]:
        nm = item["name"]
        tokens.append(
            (
                s_mytab,
                vocab.card_id(nm),
                0.0,
                float(item["resources"]) / 20.0,
                1.0 if bool(item["isDisabled"]) else 0.0,
                1.0 if nm in my_used else 0.0,
            )
        )

    opp_used = set(opp["usedActions"])
    s_opptab = SLOT_TO_ID["opp_tableau"]
    for item in opp["tableau"]:
        nm = item["name"]
        tokens.append(
            (
                s_opptab,
                vocab.card_id(nm),
                0.0,
                float(item["resources"]) / 20.0,
                1.0 if bool(item["isDisabled"]) else 0.0,
                1.0 if nm in opp_used else 0.0,
            )
        )

    add_simple("my_preludes_in_hand", priv["preludeCardsInHand"])
    add_simple("dealt_corps", priv["dealtCorporationCards"])
    add_simple("dealt_preludes", priv["dealtPreludeCards"])
    add_simple("dealt_projects", priv["dealtProjectCards"])
    if priv["pickedCorporationCard"] is not None:
        add_simple("picked_corp", (priv["pickedCorporationCard"],))
    add_simple("drafted_cards", priv["draftedCards"])
    add_simple("current_draft_packet", priv["currentDraftPacket"])
    add_simple("pending_card_selection", priv["pendingCardSelection"])

    passed_counts: Counter[int] = Counter()
    for d_entry in hist["drafts"]:
        for nm in d_entry["passed"]:
            passed_counts[vocab.card_id(nm)] += 1
    s_passed = SLOT_TO_ID["history_draft_passed"]
    for cid, cnt in passed_counts.items():
        tokens.append((s_passed, cid, float(cnt) / 3.0, 0.0, 0.0, 0.0))

    discarded_counts: Counter[int] = Counter()
    ib = hist["initialBuy"]
    if ib is not None:
        for nm in ib["corporationsDiscarded"]:
            discarded_counts[vocab.card_id(nm)] += 1
        for nm in ib["preludesDiscarded"]:
            discarded_counts[vocab.card_id(nm)] += 1
        for nm in ib["projectsDiscarded"]:
            discarded_counts[vocab.card_id(nm)] += 1
    for rb in hist["researchBuys"]:
        for nm in rb["discarded"]:
            discarded_counts[vocab.card_id(nm)] += 1
    for cs in hist["cardSelections"]:
        for nm in cs["unselected"]:
            discarded_counts[vocab.card_id(nm)] += 1
    s_disc = SLOT_TO_ID["history_discarded_seen"]
    for cid, cnt in discarded_counts.items():
        tokens.append((s_disc, cid, float(cnt) / 3.0, 0.0, 0.0, 0.0))

    tokens.sort()

    n_cards = len(tokens)
    if n_cards == 0:
        card_ids = np.zeros((0,), dtype=np.int64)
        card_slots = np.zeros((0,), dtype=np.int64)
        card_dyn = np.zeros((0, CARD_DYN_DIM), dtype=np.float32)
    else:
        card_ids = np.asarray([t[1] for t in tokens], dtype=np.int64)
        card_slots = np.asarray([t[0] for t in tokens], dtype=np.int64)
        card_dyn = np.asarray([[t[2], t[3], t[4], t[5]] for t in tokens], dtype=np.float32)

    # Spaces sorted canonically by id for order invariance
    spaces = sorted(pub["spaces"], key=lambda sp: str(sp["id"]))
    n_spaces = len(spaces)
    space_feats = np.zeros((n_spaces, SPACE_FEAT_DIM), dtype=np.float32)
    for i, sp in enumerate(spaces):
        space_feats[i, 0] = float(sp["x"]) / 10.0
        space_feats[i, 1] = float(sp["y"]) / 10.0
        st_idx = _SPACE_TYPE_TO_IDX.get(str(sp["spaceType"]).lower(), _SPACE_TYPE_TO_IDX["other"])
        space_feats[i, 2 + st_idx] = 1.0
        tt = sp["tileType"]
        if tt is None:
            space_feats[i, 6 + 0] = 1.0
        elif int(tt) in (0, 1, 2, 3):
            space_feats[i, 6 + 1 + int(tt)] = 1.0
        else:
            space_feats[i, 6 + 5] = 1.0
        own = sp["owner"]
        if own is None:
            space_feats[i, 12 + 0] = 1.0
        elif int(own) == actor:
            space_feats[i, 12 + 1] = 1.0
        else:
            space_feats[i, 12 + 2] = 1.0
        _encode_space_bonus(sp["bonus"], space_feats[i], 15)

    return EncodedObservation(
        obs_scalars=scalars,
        card_ids=card_ids,
        card_slots=card_slots,
        card_dyn=card_dyn,
        space_feats=space_feats,
    )


def encode_choice(choice: Choice, obs: ActorObservation, vocab: CardVocab) -> EncodedChoice:
    """Encode a Choice and its candidates conditioned on the ActorObservation."""
    prompt_type = classify_prompt_type(choice, obs)
    prompt_type_id = PROMPT_TYPE_TO_ID[prompt_type]
    cands = choice.candidates
    n_cands = len(cands)

    has_done = any(c.id == "done" for c in cands)
    already_selected = 0
    for c in cands:
        if c.id == "done":
            m = _DONE_COUNT_RE.search(c.label)
            if m:
                already_selected = int(m.group(1))
            break

    priv = obs["private"]
    cand_id_set = {c.id for c in cands if c.id != "done"}
    picked_names: list[str] = []
    if prompt_type == "initial_prelude" and priv["dealtPreludeCards"]:
        picked_names = sorted(c for c in priv["dealtPreludeCards"] if c not in cand_id_set)
        already_selected = len(picked_names)
    elif prompt_type == "initial_project_buy" and priv["dealtProjectCards"]:
        picked_names = sorted(c for c in priv["dealtProjectCards"] if c not in cand_id_set)
        if already_selected == 0:
            already_selected = len(picked_names)

    picked_card_ids = np.asarray([vocab.card_id(nm) for nm in picked_names], dtype=np.int64)

    ctx = np.zeros(CHOICE_CTX_DIM, dtype=np.float32)
    ctx[prompt_type_id] = 1.0
    o = len(PROMPT_TYPES)
    ctx[o + 0] = float(n_cands) / 30.0
    ctx[o + 1] = 1.0 if has_done else 0.0
    ctx[o + 2] = float(already_selected) / 10.0
    ctx[o + 3] = 1.0 if choice.kind == "multi_card" else 0.0
    ctx[o + 4] = float(max(0, 4 - n_cands)) / 4.0 if prompt_type == "draft_pick" else 0.0
    ctx[o + 5] = 1.0 if (int(obs["public"]["generation"]) == 1 and obs["public"]["phase"] == "research") else 0.0

    hand_costs = {item["name"]: int(item["calculatedCost"]) for item in priv["cardsInHand"]}
    space_by_id = {str(sp["id"]): sp for sp in obs["public"]["spaces"]} if choice.kind == "space" else {}
    actor = int(obs["actor"])
    my_color = str(obs["public"]["players"][actor]["color"])

    cand_card_ids = np.zeros((n_cands,), dtype=np.int64)
    cand_feats = np.zeros((n_cands, CAND_FEAT_DIM), dtype=np.float32)
    is_card_kind = choice.kind in ("card", "multi_card", "project_card")

    for j, cand in enumerate(cands):
        cid_str = cand.id
        if is_card_kind:
            if cid_str == "done":
                cand_card_ids[j] = DONE_ID
                cand_feats[j, 1] = 1.0
            else:
                cand_card_ids[j] = vocab.card_id(cid_str)
                cand_feats[j, 0] = 1.0
                if cid_str in hand_costs:
                    cand_feats[j, 2] = float(hand_costs[cid_str]) / 50.0
                    cand_feats[j, 3] = 1.0
        elif choice.kind == "space":
            sp = space_by_id.get(cid_str)
            if sp is not None:
                cand_feats[j, 4] = float(sp["x"]) / 10.0
                cand_feats[j, 5] = float(sp["y"]) / 10.0
                st_idx = _SPACE_TYPE_TO_IDX.get(str(sp["spaceType"]).lower(), _SPACE_TYPE_TO_IDX["other"])
                cand_feats[j, 6 + st_idx] = 1.0
                _encode_space_bonus(sp["bonus"], cand_feats[j], 10)  # 10..18
        elif choice.kind == "payment":
            for k, val_s in _PAYMENT_KV_RE.findall(cid_str):
                v = float(val_s)
                kl = k.lower()
                if kl == "megacredits":
                    cand_feats[j, 19] = v / 50.0
                elif kl == "steel":
                    cand_feats[j, 20] = v / 10.0
                elif kl == "titanium":
                    cand_feats[j, 21] = v / 10.0
                elif kl == "heat":
                    cand_feats[j, 22] = v / 20.0
        elif choice.kind == "branch":
            lbl = cand.label.lower()
            if "milestone" in lbl:
                cand_feats[j, 23] = 1.0
            elif "greenery" in lbl:
                cand_feats[j, 24] = 1.0
            elif "heat" in lbl:
                cand_feats[j, 25] = 1.0
            elif "play project card" in lbl:
                cand_feats[j, 26] = 1.0
            elif "action from a played card" in lbl:
                cand_feats[j, 27] = 1.0
            elif "pass" in lbl or "end turn" in lbl:
                cand_feats[j, 28] = 1.0
        elif choice.kind == "resource":
            res_map = {"megacredits": 23, "steel": 24, "titanium": 25, "plants": 26, "energy": 27, "heat": 28}
            r_slot = res_map.get(cid_str)
            if r_slot is not None:
                cand_feats[j, r_slot] = 1.0
        elif choice.kind == "player":
            if cid_str == my_color:
                cand_feats[j, 29] = 1.0
            elif cid_str == "neutral":
                cand_feats[j, 31] = 1.0
            else:
                cand_feats[j, 30] = 1.0
        elif choice.kind == "amount":
            try:
                cand_feats[j, 29] = float(cid_str) / 20.0
            except ValueError:
                pass
            cand_feats[j, 30] = float(j) / float(max(1, n_cands - 1))
            cand_feats[j, 31] = 1.0

    return EncodedChoice(
        choice_ctx=ctx,
        prompt_type_id=prompt_type_id,
        cand_card_ids=cand_card_ids,
        cand_feats=cand_feats,
        picked_card_ids=picked_card_ids,
    )


def collate_encoded_batch(
    observations: Sequence[EncodedObservation],
    choices: Sequence[EncodedChoice] | None = None,
) -> dict[str, torch.Tensor]:
    """Pad and stack encoded observations (and optional choices) into a batch dict of tensors."""
    bsz = len(observations)
    if bsz == 0:
        raise ValueError("Cannot collate an empty batch")

    obs_scalars = torch.from_numpy(np.stack([o.obs_scalars for o in observations], axis=0))

    max_cards = max((o.card_ids.shape[0] for o in observations), default=0)
    max_cards = max(1, max_cards)
    card_ids = np.zeros((bsz, max_cards), dtype=np.int64)
    card_slots = np.zeros((bsz, max_cards), dtype=np.int64)
    card_dyn = np.zeros((bsz, max_cards, CARD_DYN_DIM), dtype=np.float32)
    card_mask = np.zeros((bsz, max_cards), dtype=bool)
    for b, o in enumerate(observations):
        n = o.card_ids.shape[0]
        if n > 0:
            card_ids[b, :n] = o.card_ids
            card_slots[b, :n] = o.card_slots
            card_dyn[b, :n] = o.card_dyn
            card_mask[b, :n] = True

    max_spaces = max((o.space_feats.shape[0] for o in observations), default=0)
    max_spaces = max(1, max_spaces)
    space_feats = np.zeros((bsz, max_spaces, SPACE_FEAT_DIM), dtype=np.float32)
    space_mask = np.zeros((bsz, max_spaces), dtype=bool)
    for b, o in enumerate(observations):
        n = o.space_feats.shape[0]
        if n > 0:
            space_feats[b, :n] = o.space_feats
            space_mask[b, :n] = True

    batch: dict[str, torch.Tensor] = {
        "obs_scalars": obs_scalars,
        "card_ids": torch.from_numpy(card_ids),
        "card_slots": torch.from_numpy(card_slots),
        "card_dyn": torch.from_numpy(card_dyn),
        "card_mask": torch.from_numpy(card_mask),
        "space_feats": torch.from_numpy(space_feats),
        "space_mask": torch.from_numpy(space_mask),
    }

    if choices is not None:
        if len(choices) != bsz:
            raise ValueError(f"choices length ({len(choices)}) != observations length ({bsz})")
        choice_ctx = np.stack([c.choice_ctx for c in choices], axis=0)
        prompt_type_ids = np.asarray([c.prompt_type_id for c in choices], dtype=np.int64)

        max_cands = max((c.cand_card_ids.shape[0] for c in choices), default=1)
        max_cands = max(1, max_cands)
        cand_card_ids = np.zeros((bsz, max_cands), dtype=np.int64)
        cand_feats = np.zeros((bsz, max_cands, CAND_FEAT_DIM), dtype=np.float32)
        cand_mask = np.zeros((bsz, max_cands), dtype=bool)
        for b, c in enumerate(choices):
            nc = c.cand_card_ids.shape[0]
            if nc > 0:
                cand_card_ids[b, :nc] = c.cand_card_ids
                cand_feats[b, :nc] = c.cand_feats
                cand_mask[b, :nc] = True

        max_picked = max((c.picked_card_ids.shape[0] for c in choices), default=0)
        max_picked = max(1, max_picked)
        picked_card_ids = np.zeros((bsz, max_picked), dtype=np.int64)
        picked_mask = np.zeros((bsz, max_picked), dtype=bool)
        for b, c in enumerate(choices):
            npk = c.picked_card_ids.shape[0]
            if npk > 0:
                picked_card_ids[b, :npk] = c.picked_card_ids
                picked_mask[b, :npk] = True

        batch.update(
            {
                "choice_ctx": torch.from_numpy(choice_ctx),
                "prompt_type_ids": torch.from_numpy(prompt_type_ids),
                "cand_card_ids": torch.from_numpy(cand_card_ids),
                "cand_feats": torch.from_numpy(cand_feats),
                "cand_mask": torch.from_numpy(cand_mask),
                "picked_card_ids": torch.from_numpy(picked_card_ids),
                "picked_mask": torch.from_numpy(picked_mask),
            }
        )

    return batch
