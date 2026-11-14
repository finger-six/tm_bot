"""Reconstruct adapter-compatible ActorObservation and Choice objects from warm-start records."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tm_bot.agents.card_data import CardSpec, load_card_data
from tm_bot.env.types import (
    SUPPORTED_MAPS,
    ActorHistory,
    ActorObservation,
    Candidate,
    Choice,
    DraftHistoryEntry,
    HandCardObservation,
    InitialBuyHistory,
    PrivateObservation,
    PublicObservation,
    PublicPlayerObservation,
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
    # BGA parser recorded generation=1 on all draft_picks rows; round 1 corresponds to Gen 2+.
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
