"""Reconstruct adapter-compatible ActorObservation and Choice objects from warm-start records."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tm_bot.agents.card_data import CardSpec, load_card_data
from tm_bot.data import actionmenu
from tm_bot.data.logstate import SNAPSHOT_TAGS
from tm_bot.env.types import (
    SUPPORTED_MAPS,
    ActorHistory,
    ActorObservation,
    Candidate,
    CardSelectionHistoryEntry,
    Choice,
    DraftHistoryEntry,
    HandCardObservation,
    InitialBuyHistory,
    PrivateObservation,
    PublicObservation,
    PublicPlayerObservation,
    ResearchBuyHistoryEntry,
    Seat,
    TableauCardObservation,
)
from tm_bot.models.vocab import TAG_NAMES

_TEMPLATES_PATH = Path(__file__).resolve().with_name("map_templates.json")
_TEMPLATE_CACHE: dict[str, dict[str, Any]] | None = None

_MAP_ALIASES: dict[str, str] = {
    "tharsis": "tharsis",
    "elysium": "elysium",
    "hellas": "hellas",
    "vastitas borealis": "vastitas borealis nova",
    "vastitas borealis nova": "vastitas borealis nova",
    "vastitas_borealis_nova": "vastitas borealis nova",
}

_SEAT_COLORS: tuple[str, str] = ("red", "blue")


def normalize_map_name(map_name: str) -> str:
    """Normalize warm-start / display map names to adapter SUPPORTED_MAPS keys."""
    key = " ".join(str(map_name).strip().lower().replace("-", " ").split())
    if key in _MAP_ALIASES:
        return _MAP_ALIASES[key]
    if key in SUPPORTED_MAPS:
        return key
    raise ValueError(f"Unsupported map name for warm-start reconstruction: {map_name!r}")


def load_map_templates(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load static board/milestone/award/deck templates extracted from the adapter."""
    global _TEMPLATE_CACHE
    if path is None and _TEMPLATE_CACHE is not None:
        return _TEMPLATE_CACHE
    target = path or _TEMPLATES_PATH
    data = json.loads(target.read_text(encoding="utf-8"))
    if path is None:
        _TEMPLATE_CACHE = data
    return data


def _empty_tags() -> dict[str, int]:
    return {tag: 0 for tag in TAG_NAMES}


def _zero_toggle_map() -> dict[str, str]:
    return {
        "megacredits": "off",
        "steel": "off",
        "titanium": "off",
        "plants": "off",
        "energy": "off",
        "heat": "off",
    }


def _default_public_player(seat: Seat, *, is_active: bool = False) -> PublicPlayerObservation:
    return PublicPlayerObservation(
        seat=seat,
        color=_SEAT_COLORS[seat],
        terraformRating=20,
        megacredits=0,
        steel=0,
        titanium=0,
        plants=0,
        energy=0,
        heat=0,
        megacreditProduction=0,
        steelProduction=0,
        titaniumProduction=0,
        plantProduction=0,
        energyProduction=0,
        heatProduction=0,
        steelValue=2,
        titaniumValue=3,
        cardCost=3,
        cardDiscount=0,
        cardsInHandCount=0,
        citiesCount=0,
        noTagsCount=0,
        tags=_empty_tags(),
        tableau=[],
        usedActions=[],
        actionsTakenThisRound=0,
        actionsTakenThisGame=0,
        availableBlueCardActionCount=0,
        lastCardPlayed=None,
        isActive=is_active,
        passed=False,
        protectedResources=_zero_toggle_map(),
        protectedProduction=_zero_toggle_map(),
    )


def _base_public_observation(
    map_name: str,
    *,
    generation: int,
    phase: str,
    templates: Mapping[str, dict[str, Any]] | None = None,
) -> PublicObservation:
    norm_map = normalize_map_name(map_name)
    tpl_map = (templates if templates is not None else load_map_templates())[norm_map]
    # Seat 0 is the gen-1 starting player (warm-start seats are resolved that way); the engine
    # rotates the first player every generation and reports it as the active player during the
    # simultaneous setup and drafting decisions this module reconstructs.
    first: Seat = 0 if int(generation) % 2 == 1 else 1
    players = [
        _default_public_player(0, is_active=(first == 0)),
        _default_public_player(1, is_active=(first == 1)),
    ]
    return PublicObservation(
        map=norm_map,
        generation=int(generation),
        phase=phase,
        firstPlayer=first,
        activePlayer=first,
        passedPlayers=[],
        globalParameters={
            "temperature": -30,
            "oxygenLevel": 0,
            "oceans": 0,
            "isTerraformed": False,
        },
        deckSize=int(tpl_map["deckSize"]),
        discardPileSize=0,
        otherDeckSizes=copy.deepcopy(tpl_map["otherDeckSizes"]),
        spaces=copy.deepcopy(tpl_map["spaces"]),
        milestones=copy.deepcopy(tpl_map["milestones"]),
        awards=copy.deepcopy(tpl_map["awards"]),
        players=players,
        publicLog=[],
    )


def _empty_history() -> ActorHistory:
    return ActorHistory(
        initialBuy=None,
        drafts=[],
        researchBuys=[],
        cardSelections=[],
    )


def _validate_seat(seat: int) -> Seat:
    s = int(seat)
    if s not in (0, 1):
        raise ValueError(f"Expected seat 0 or 1, got {seat!r}")
    return 0 if s == 0 else 1


def _accumulate_tags(card_names: Sequence[str], cards_by_name: Mapping[str, CardSpec]) -> dict[str, int]:
    tags = _empty_tags()
    for name in card_names:
        spec = cards_by_name.get(name)
        if spec is None:
            continue
        for t in spec.tags:
            if t in tags:
                tags[t] += 1
    return tags


def _make_hand_cards(
    project_names: Sequence[str], cards_by_name: Mapping[str, CardSpec]
) -> list[HandCardObservation]:
    out: list[HandCardObservation] = []
    for name in sorted(project_names):
        spec = cards_by_name.get(name)
        cost = int(spec.cost) if spec is not None else 0
        out.append(HandCardObservation(name=name, calculatedCost=cost))
    return out


def _apply_committed_setup_to_actor(
    pub: PublicObservation,
    seat: Seat,
    chosen_corp: str,
    chosen_preludes: Sequence[str],
    chosen_projects: Sequence[str],
    cards_by_name: Mapping[str, CardSpec],
    *,
    include_preludes_in_tableau: bool,
) -> None:
    me = pub["players"][seat]
    corp_spec = cards_by_name.get(chosen_corp)
    card_cost = (
        int(corp_spec.card_cost)
        if (corp_spec is not None and corp_spec.card_cost is not None)
        else 3
    )
    starting_mc = int(corp_spec.starting_megacredits) if corp_spec is not None else 40
    me["cardCost"] = card_cost
    me["megacredits"] = max(0, starting_mc - len(chosen_projects) * card_cost)
    me["cardsInHandCount"] = len(chosen_projects)

    tab_names = [chosen_corp]
    if include_preludes_in_tableau:
        tab_names.extend(chosen_preludes)
    me["tableau"] = [
        TableauCardObservation(name=nm, resources=0, isDisabled=False) for nm in tab_names
    ]
    me["tags"] = _accumulate_tags(tab_names, cards_by_name)


def build_corp_choice(
    map_name: str,
    seat: int,
    offered_corps: Sequence[str],
    offered_preludes: Sequence[str],
    offered_projects: Sequence[str],
    *,
    templates: Mapping[str, dict[str, Any]] | None = None,
) -> tuple[ActorObservation, Choice]:
    """Reconstruct `(ActorObservation, Choice)` at the initial corporation selection step."""
    actor = _validate_seat(seat)
    corps = sorted(str(c) for c in offered_corps)
    preludes = sorted(str(p) for p in offered_preludes)
    projects = sorted(str(p) for p in offered_projects)

    pub = _base_public_observation(
        map_name, generation=1, phase="research", templates=templates
    )
    priv = PrivateObservation(
        cardsInHand=[],
        dealtCorporationCards=corps,
        dealtPreludeCards=preludes,
        dealtProjectCards=projects,
        pickedCorporationCard=None,
        preludeCardsInHand=[],
        draftedCards=[],
        currentDraftPacket=[],
        pendingCardSelection=[],
    )
    obs = ActorObservation(
        actor=actor,
        public=pub,
        private=priv,
        history=_empty_history(),
    )
    choice = Choice(
        seat=actor,
        kind="card",
        prompt="Select corporation",
        candidates=tuple(Candidate(id=c, label=c) for c in corps),
    )
    return obs, choice


def build_prelude_choice(
    map_name: str,
    seat: int,
    offered_corps: Sequence[str],
    offered_preludes: Sequence[str],
    offered_projects: Sequence[str],
    *,
    picked_preludes: Sequence[str] = (),
    templates: Mapping[str, dict[str, Any]] | None = None,
) -> tuple[ActorObservation, Choice]:
    """Reconstruct `(ActorObservation, Choice)` during initial 2-prelude selection.

    Matches the live adapter's DecisionStepper behavior: during `Select 2 Prelude cards`,
    `pickedCorporationCard` is still `None` and already-picked preludes are removed from
    `choice.candidates` while `dealtPreludeCards` retains all 4 dealt preludes.
    """
    actor = _validate_seat(seat)
    corps = sorted(str(c) for c in offered_corps)
    preludes = sorted(str(p) for p in offered_preludes)
    projects = sorted(str(p) for p in offered_projects)
    picked_set = set(str(p) for p in picked_preludes)

    pub = _base_public_observation(
        map_name, generation=1, phase="research", templates=templates
    )
    priv = PrivateObservation(
        cardsInHand=[],
        dealtCorporationCards=corps,
        dealtPreludeCards=preludes,
        dealtProjectCards=projects,
        pickedCorporationCard=None,
        preludeCardsInHand=[],
        draftedCards=[],
        currentDraftPacket=[],
        pendingCardSelection=[],
    )
    obs = ActorObservation(
        actor=actor,
        public=pub,
        private=priv,
        history=_empty_history(),
    )

    if len(picked_set) >= 2:
        cands = (Candidate(id="done", label=f"Done ({len(picked_set)} selected)"),)
    else:
        remaining = [p for p in preludes if p not in picked_set]
        cands = tuple(Candidate(id=p, label=p) for p in remaining)

    choice = Choice(
        seat=actor,
        kind="multi_card",
        prompt="Select 2 Prelude cards",
        candidates=cands,
    )
    return obs, choice


def build_project_buy_choice(
    map_name: str,
    seat: int,
    offered_corps: Sequence[str],
    offered_preludes: Sequence[str],
    offered_projects: Sequence[str],
    *,
    picked_projects: Sequence[str] = (),
    templates: Mapping[str, dict[str, Any]] | None = None,
) -> tuple[ActorObservation, Choice]:
    """Reconstruct `(ActorObservation, Choice)` during initial project buy selection."""
    actor = _validate_seat(seat)
    corps = sorted(str(c) for c in offered_corps)
    preludes = sorted(str(p) for p in offered_preludes)
    projects = sorted(str(p) for p in offered_projects)
    picked_set = set(str(p) for p in picked_projects)

    pub = _base_public_observation(
        map_name, generation=1, phase="research", templates=templates
    )
    priv = PrivateObservation(
        cardsInHand=[],
        dealtCorporationCards=corps,
        dealtPreludeCards=preludes,
        dealtProjectCards=projects,
        pickedCorporationCard=None,
        preludeCardsInHand=[],
        draftedCards=[],
        currentDraftPacket=[],
        pendingCardSelection=[],
    )
    obs = ActorObservation(
        actor=actor,
        public=pub,
        private=priv,
        history=_empty_history(),
    )

    remaining = [p for p in projects if p not in picked_set]
    cands = tuple(
        [Candidate(id=p, label=p) for p in remaining]
        + [Candidate(id="done", label=f"Done ({len(picked_set)} selected)")]
    )
    choice = Choice(
        seat=actor,
        kind="multi_card",
        prompt="Select initial cards to buy",
        candidates=cands,
    )
    return obs, choice


def build_draft_pick_choice(
    map_name: str,
    seat: int,
    draft_round_id: int,
    packet_cards: Sequence[str],
    chosen_corp: str,
    chosen_preludes: Sequence[str],
    chosen_projects: Sequence[str],
    *,
    offered_corps: Sequence[str] = (),
    offered_preludes: Sequence[str] = (),
    offered_projects: Sequence[str] = (),
    drafted_this_round: Sequence[str] = (),
    prior_drafts: Sequence[DraftHistoryEntry] = (),
    opponent_corp: str | None = None,
    opponent_preludes: Sequence[str] = (),
    cards_by_name: Mapping[str, CardSpec] | None = None,
    templates: Mapping[str, dict[str, Any]] | None = None,
) -> tuple[ActorObservation, Choice]:
    """Reconstruct `(ActorObservation, Choice)` for a drafting pick.

    Never reads or populates opponent private information. The opponent's corporation and
    preludes (`opponent_corp`, `opponent_preludes`) are public once played in generation 1 and
    are placed in their tableau when known; every other opponent field stays at its neutral
    starting default. `prior_drafts` are the actor's own earlier draft picks.
    """
    actor = _validate_seat(seat)
    specs = cards_by_name if cards_by_name is not None else load_card_data()
    # Round r is the draft of generation r + 1; the v1 table's generation column is all 1 (#30).
    gen = max(2, int(draft_round_id) + 1)

    corps_off = sorted(str(c) for c in (offered_corps or (chosen_corp,)))
    prel_off = sorted(str(p) for p in (offered_preludes or chosen_preludes))
    proj_off = sorted(str(p) for p in (offered_projects or chosen_projects))
    prel_ch = sorted(str(p) for p in chosen_preludes)
    proj_ch = sorted(str(p) for p in chosen_projects)
    packet_sorted = sorted(str(c) for c in packet_cards)
    drafted_sorted = sorted(str(c) for c in drafted_this_round)

    pub = _base_public_observation(
        map_name, generation=gen, phase="drafting", templates=templates
    )
    _apply_committed_setup_to_actor(
        pub,
        actor,
        chosen_corp,
        prel_ch,
        proj_ch,
        specs,
        include_preludes_in_tableau=True,
    )
    if opponent_corp is not None:
        opp_tab = [str(opponent_corp), *sorted(str(p) for p in opponent_preludes)]
        opp = pub["players"][1 - actor]
        opp["tableau"] = [
            TableauCardObservation(name=nm, resources=0, isDisabled=False) for nm in opp_tab
        ]
        opp["tags"] = _accumulate_tags(opp_tab, specs)

    priv = PrivateObservation(
        cardsInHand=_make_hand_cards(proj_ch, specs),
        dealtCorporationCards=corps_off,
        dealtPreludeCards=prel_off,
        dealtProjectCards=proj_off,
        pickedCorporationCard=chosen_corp,
        preludeCardsInHand=[],
        draftedCards=drafted_sorted,
        currentDraftPacket=packet_sorted,
        pendingCardSelection=[],
    )
    init_buy = InitialBuyHistory(
        corporationsOffered=corps_off,
        corporationChosen=chosen_corp,
        corporationsDiscarded=[c for c in corps_off if c != chosen_corp],
        preludesOffered=prel_off,
        preludesChosen=prel_ch,
        preludesDiscarded=[p for p in prel_off if p not in set(prel_ch)],
        projectsOffered=proj_off,
        projectsBought=proj_ch,
        projectsDiscarded=[p for p in proj_off if p not in set(proj_ch)],
    )
    history = ActorHistory(
        initialBuy=init_buy,
        drafts=list(prior_drafts),
        researchBuys=[],
        cardSelections=[],
    )
    obs = ActorObservation(
        actor=actor,
        public=pub,
        private=priv,
        history=history,
    )
    opp_color = _SEAT_COLORS[1 - actor]
    choice = Choice(
        seat=actor,
        kind="card",
        prompt=f"Select a card to keep and pass the rest to {opp_color}",
        candidates=tuple(Candidate(id=c, label=c) for c in packet_sorted),
    )
    return obs, choice


def build_setup_outcome_observation(
    map_name: str,
    seat: int,
    chosen_corp: str,
    chosen_preludes: Sequence[str],
    chosen_projects: Sequence[str],
    *,
    cards_by_name: Mapping[str, CardSpec] | None = None,
    templates: Mapping[str, dict[str, Any]] | None = None,
) -> ActorObservation:
    """Reconstruct post-setup `ActorObservation` (`phase='preludes'`) for value prediction."""
    actor = _validate_seat(seat)
    specs = cards_by_name if cards_by_name is not None else load_card_data()
    prel_ch = sorted(str(p) for p in chosen_preludes)
    proj_ch = sorted(str(p) for p in chosen_projects)

    pub = _base_public_observation(
        map_name, generation=1, phase="preludes", templates=templates
    )
    _apply_committed_setup_to_actor(
        pub,
        actor,
        chosen_corp,
        prel_ch,
        proj_ch,
        specs,
        include_preludes_in_tableau=False,
    )
    priv = PrivateObservation(
        cardsInHand=_make_hand_cards(proj_ch, specs),
        dealtCorporationCards=[],
        dealtPreludeCards=[],
        dealtProjectCards=[],
        pickedCorporationCard=chosen_corp,
        preludeCardsInHand=prel_ch,
        draftedCards=[],
        currentDraftPacket=[],
        pendingCardSelection=[],
    )
    return ActorObservation(
        actor=actor,
        public=pub,
        private=priv,
        history=_empty_history(),
    )


# ---------------------------------------------------------------------------------------------
# v2: generation 2+ rows carrying the reconstructed snapshot column group (`tm_bot.data.logstate`)
# ---------------------------------------------------------------------------------------------

_STOCK_FIELDS: tuple[tuple[str, str], ...] = (
    ("mc", "megacredits"), ("steel", "steel"), ("titanium", "titanium"), ("plants", "plants"),
    ("energy", "energy"), ("heat", "heat"),
)
_PROD_FIELDS: tuple[tuple[str, str], ...] = (
    ("mc_prod", "megacreditProduction"), ("steel_prod", "steelProduction"),
    ("titanium_prod", "titaniumProduction"), ("plants_prod", "plantProduction"),
    ("energy_prod", "energyProduction"), ("heat_prod", "heatProduction"),
)


def _apply_snapshot_player(p: PublicPlayerObservation, row: Mapping[str, Any], prefix: str) -> None:
    p["terraformRating"] = int(row[f"{prefix}_tr"])
    for col, key in _STOCK_FIELDS + _PROD_FIELDS:
        p[key] = int(row[f"{prefix}_{col}"])  # type: ignore[literal-required]
    p["cardsInHandCount"] = int(row[f"{prefix}_hand_count"])
    p["citiesCount"] = int(row[f"{prefix}_cities"])
    tags = _empty_tags()
    for tag, count in zip(SNAPSHOT_TAGS, row[f"{prefix}_tags"]):
        tags[tag] = int(count)
    p["tags"] = tags
    p["tableau"] = [
        TableauCardObservation(name=str(nm), resources=0, isDisabled=False) for nm in row[f"{prefix}_tableau"]
    ]


def build_state_observation(
    row: Mapping[str, Any],
    *,
    phase: str | None = None,
    cards_by_name: Mapping[str, CardSpec] | None = None,
    templates: Mapping[str, dict[str, Any]] | None = None,
) -> ActorObservation:
    """Reconstruct a gen 2+ `ActorObservation` from one v2 snapshot-group row.

    Populates globals, both seats' TR/stocks/productions/tags/cities/hand counts and public tableaus,
    milestone `claimedBy` and award `fundedBy`, the actor's tracked hand, and its setup and
    research-buy history. Unrecoverable fields (deck sizes, milestone/award scores, per-card
    resources, board spaces, used actions) stay at their map-template / neutral defaults (#38 2D).
    """
    actor = _validate_seat(row["seat"])
    specs = cards_by_name if cards_by_name is not None else load_card_data()
    gen = int(row["generation"])
    pub = _base_public_observation(
        row["map"], generation=gen, phase=phase or str(row["phase"]), templates=templates
    )
    temp, oxy, oceans = int(row["temperature"]), int(row["oxygen"]), int(row["oceans"])
    pub["globalParameters"] = {
        "temperature": temp,
        "oxygenLevel": oxy,
        "oceans": oceans,
        "isTerraformed": temp >= 8 and oxy >= 14 and oceans >= 9,
    }
    _apply_snapshot_player(pub["players"][actor], row, "me")
    _apply_snapshot_player(pub["players"][1 - actor], row, "opp")
    claimed = {str(n): int(s) for n, s in zip(row["milestone_names"], row["milestone_seats"]) if int(s) in (0, 1)}
    funded = {str(n): int(s) for n, s in zip(row["award_names"], row["award_seats"]) if int(s) in (0, 1)}
    for ms in pub["milestones"]:
        if ms["name"] in claimed:
            ms["claimedBy"] = _validate_seat(claimed[ms["name"]])
    for aw in pub["awards"]:
        if aw["name"] in funded:
            aw["fundedBy"] = _validate_seat(funded[aw["name"]])

    corp = str(row["me_corporation"])
    preludes = sorted(str(p) for p in row["me_setup_preludes"])
    bought = sorted(str(p) for p in row["me_setup_projects_bought"])
    corps_disc = sorted(str(c) for c in row["me_setup_corps_discarded"])
    prel_disc = sorted(str(p) for p in row["me_setup_preludes_discarded"])
    proj_disc = sorted(str(p) for p in row["me_setup_projects_discarded"])
    hand = [str(c) for c in row["me_hand"]] if bool(row["me_hand_exact"]) else []
    priv = PrivateObservation(
        cardsInHand=_make_hand_cards(hand, specs),
        dealtCorporationCards=sorted([corp, *corps_disc]),
        dealtPreludeCards=sorted([*preludes, *prel_disc]),
        dealtProjectCards=sorted([*bought, *proj_disc]),
        pickedCorporationCard=corp,
        preludeCardsInHand=sorted(str(p) for p in row["me_preludes_in_hand"]),
        draftedCards=sorted(str(c) for c in row["me_drafted_this_round"]),
        currentDraftPacket=[],
        pendingCardSelection=[],
    )
    research = [
        ResearchBuyHistoryEntry(
            generation=int(g),
            offered=sorted(str(c) for c in off),
            bought=sorted(str(c) for c in kept),
            discarded=sorted(str(c) for c in disc),
        )
        for g, off, kept, disc in zip(
            row["me_research_generations"],
            row["me_research_offered"],
            row["me_research_kept"],
            row["me_research_discarded"],
        )
    ]
    seen = sorted(str(c) for c in row["me_discarded_seen"])
    selections = (
        [CardSelectionHistoryEntry(generation=gen, phase="action", offered=seen, selected=[], unselected=seen)]
        if seen
        else []
    )
    history = ActorHistory(
        initialBuy=InitialBuyHistory(
            corporationsOffered=sorted([corp, *corps_disc]),
            corporationChosen=corp,
            corporationsDiscarded=corps_disc,
            preludesOffered=sorted([*preludes, *prel_disc]),
            preludesChosen=preludes,
            preludesDiscarded=prel_disc,
            projectsOffered=sorted([*bought, *proj_disc]),
            projectsBought=bought,
            projectsDiscarded=proj_disc,
        ),
        drafts=[],
        researchBuys=research,
        cardSelections=selections,
    )
    return ActorObservation(actor=actor, public=pub, private=priv, history=history)


def build_draft_pick_state_choice(
    row: Mapping[str, Any],
    *,
    cards_by_name: Mapping[str, CardSpec] | None = None,
    templates: Mapping[str, dict[str, Any]] | None = None,
) -> tuple[ActorObservation, Choice]:
    """Gen 2+ draft pick with the reconstructed pre-pick state and the tracked `cardsInHand`."""
    obs = build_state_observation(row, phase="drafting", cards_by_name=cards_by_name, templates=templates)
    packet = sorted(str(c) for c in row["packet_cards"])
    obs["private"]["currentDraftPacket"] = packet
    opp_color = _SEAT_COLORS[1 - obs["actor"]]
    choice = Choice(
        seat=obs["actor"],
        kind="card",
        prompt=f"Select a card to keep and pass the rest to {opp_color}",
        candidates=tuple(Candidate(id=c, label=c) for c in packet),
    )
    return obs, choice


def build_research_buy_choice(
    row: Mapping[str, Any],
    *,
    picked: Sequence[str] = (),
    cards_by_name: Mapping[str, CardSpec] | None = None,
    templates: Mapping[str, dict[str, Any]] | None = None,
) -> tuple[ActorObservation, Choice]:
    """Gen 2+ research buy as the adapter's sequential `multi_card` frame (`adapter/src/factor.ts`).

    Candidates are the unpicked cards of the 4-card drafted packet plus `done` (min 0 at 3 M€ per
    card, so `done` is always legal); `picked` are the cards already selected in this decision.
    """
    obs = build_state_observation(row, phase="research", cards_by_name=cards_by_name, templates=templates)
    packet = sorted(str(c) for c in row["packet_cards"])
    obs["private"]["draftedCards"] = packet
    picked_set = {str(p) for p in picked}
    cands = [Candidate(id=c, label=c) for c in packet if c not in picked_set]
    cands.append(Candidate(id="done", label=f"Done ({len(picked_set)} selected)"))
    choice = Choice(seat=obs["actor"], kind="multi_card", prompt="Select card(s) to buy", candidates=tuple(cands))
    return obs, choice


# Root branch labels as the adapter prints them (`formatEngineTitle` of the engine option titles in
# `Player.getActions`); convert-plants and fund-award titles carry a number.
_BRANCH_LABELS: dict[str, str] = {
    "claim_milestone": "Claim a milestone",
    "convert_heat": "Convert 8 heat into temperature",
    "card_action": "Perform an action from a played card",
    "play_project_card": "Play project card",
    "end_turn": "End Turn",
    "standard_projects": "Standard projects",
    "pass": "Pass for this generation",
    "sell_patents": "Sell patents",
}


def branch_label(key: str, row: Mapping[str, Any]) -> str:
    """Adapter label of one root branch key for the row's pre-move state."""
    if key == "convert_plants":
        return f"Convert {actionmenu.plants_for_greenery(row)} plants into greenery"
    if key == "fund_award":
        return f"Fund an award ({actionmenu.award_cost(row)} M€)"
    return _BRANCH_LABELS[key]


def _action_observation(
    row: Mapping[str, Any],
    cards_by_name: Mapping[str, CardSpec] | None,
    templates: Mapping[str, dict[str, Any]] | None,
) -> ActorObservation:
    """Snapshot observation plus the row's public action-phase state (`turn_action`, `opp_passed`,
    both seats' used blue-card actions). `availableBlueCardActionCount` is the reconstructed unused
    blue-action card count (the engine's per-card `canAct` checks are not reconstructed)."""
    specs = cards_by_name if cards_by_name is not None else load_card_data()
    obs = build_state_observation(row, phase="action", cards_by_name=specs, templates=templates)
    actor = obs["actor"]
    obs["public"]["activePlayer"] = actor
    me_used = sorted(str(c) for c in row["me_used_actions"])
    for p in obs["public"]["players"]:
        p["isActive"] = p["seat"] == actor
        if p["seat"] == actor:
            p["actionsTakenThisRound"] = int(row["turn_action"])
            p["usedActions"] = me_used
            p["availableBlueCardActionCount"] = len(actionmenu.unused_action_cards(row, specs, me_used))
        else:
            p["passed"] = bool(row["opp_passed"])
            p["usedActions"] = sorted(str(c) for c in row["opp_used_actions"])
    return obs


def build_branch_choice(
    row: Mapping[str, Any],
    *,
    cards_by_name: Mapping[str, CardSpec] | None = None,
    templates: Mapping[str, dict[str, Any]] | None = None,
) -> tuple[ActorObservation, Choice]:
    """Root action `branch` prompt (the engine's `getActions` OrOptions) for an `action_branches` row.

    Candidates follow `row["menu"]` (engine push order), so the target index is
    `row["menu"].index(row["chosen"])`. Ids are the adapter's `branch:<i>`, `i` being the option's
    index in the engine's pushed list (the menu plus the always-pushed `Standard projects` and `Pass`).
    The ids are exact only when the reconstructed menu equals the engine's: a claimable milestone the
    snapshot can't prove (e.g. Polar Explorer) or a blue card whose `canAct` fails shifts later indices.
    """
    obs = _action_observation(row, cards_by_name, templates)
    menu = [str(k) for k in row["menu"]]
    pushed = [k for k in actionmenu.BRANCH_ORDER if k in menu or k in actionmenu.ALWAYS_PUSHED]
    cands = tuple(Candidate(id=f"branch:{pushed.index(k)}", label=branch_label(k, row)) for k in menu)
    prompt = "Take your first action" if int(row["turn_action"]) == 0 else "Take your next action"
    return obs, Choice(seat=obs["actor"], kind="branch", prompt=prompt, candidates=cands)


def build_project_card_choice(
    row: Mapping[str, Any],
    *,
    cards_by_name: Mapping[str, CardSpec] | None = None,
    templates: Mapping[str, dict[str, Any]] | None = None,
) -> tuple[ActorObservation, Choice]:
    """`Play project card` prompt (adapter kind `project_card`) over the row's playable hand cards."""
    obs = _action_observation(row, cards_by_name, templates)
    cands = tuple(Candidate(id=c, label=c) for c in sorted(str(c) for c in row["menu"]))
    return obs, Choice(seat=obs["actor"], kind="project_card", prompt="Play project card", candidates=cands)


def build_card_action_choice(
    row: Mapping[str, Any],
    *,
    cards_by_name: Mapping[str, CardSpec] | None = None,
    templates: Mapping[str, dict[str, Any]] | None = None,
) -> tuple[ActorObservation, Choice]:
    """Blue-card action pick (the engine's `SelectCard` after the root branch; adapter kind `card`)."""
    obs = _action_observation(row, cards_by_name, templates)
    cands = tuple(Candidate(id=c, label=c) for c in sorted(str(c) for c in row["menu"]))
    return obs, Choice(seat=obs["actor"], kind="card", prompt="Perform an action from a played card",
                       candidates=cands)


def build_standard_project_choice(
    row: Mapping[str, Any],
    *,
    cards_by_name: Mapping[str, CardSpec] | None = None,
    templates: Mapping[str, dict[str, Any]] | None = None,
) -> tuple[ActorObservation, Choice]:
    """`Standard projects` prompt (adapter kind `project_card`); the adapter lists the candidates by name."""
    obs = _action_observation(row, cards_by_name, templates)
    cands = tuple(Candidate(id=c, label=c) for c in sorted(str(c) for c in row["menu"]))
    return obs, Choice(seat=obs["actor"], kind="project_card", prompt="Standard projects", candidates=cands)


def build_midgame_value_observation(
    row: Mapping[str, Any],
    *,
    cards_by_name: Mapping[str, CardSpec] | None = None,
    templates: Mapping[str, dict[str, Any]] | None = None,
) -> ActorObservation:
    """Gen 2+ value point (`phase` research at a deal boundary, or action) for the value heads."""
    return build_state_observation(row, cards_by_name=cards_by_name, templates=templates)
