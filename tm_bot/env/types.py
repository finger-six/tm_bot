"""Typed response models and observation schemas for the tm_bot bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

SUPPORTED_MAPS: tuple[str, ...] = (
    "tharsis",
    "elysium",
    "hellas",
    "vastitas borealis nova",
)

SupportedMap = Literal[
    "tharsis",
    "elysium",
    "hellas",
    "vastitas borealis nova",
]

Seat = Literal[0, 1]

ChoiceKind = Literal[
    "branch",
    "card",
    "multi_card",
    "project_card",
    "payment",
    "space",
    "player",
    "amount",
    "resource",
]


@dataclass(frozen=True, slots=True)
class Candidate:
    id: str
    label: str


@dataclass(frozen=True, slots=True)
class Choice:
    seat: Seat
    kind: str
    prompt: str
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True, slots=True)
class GameResult:
    scores: tuple[int, int]
    winner: Seat | None
    tie: bool
    generations: int


@dataclass(frozen=True, slots=True)
class BatchOp:
    method: str
    params: dict[str, Any]


class Op:
    """Factory helpers for batched bridge requests."""

    @staticmethod
    def create(seed_key: str, map: str, *, game_id: str | None = None) -> BatchOp:
        params: dict[str, Any] = {"seedKey": seed_key, "map": map}
        if game_id is not None:
            params["gameId"] = game_id
        return BatchOp("create", params)

    @staticmethod
    def pending(game_id: str) -> BatchOp:
        return BatchOp("pending", {"gameId": game_id})

    @staticmethod
    def observe(game_id: str, seat: Seat) -> BatchOp:
        return BatchOp("observe", {"gameId": game_id, "seat": seat})

    @staticmethod
    def choice(game_id: str, seat: Seat) -> BatchOp:
        return BatchOp("choice", {"gameId": game_id, "seat": seat})

    @staticmethod
    def choose(game_id: str, seat: Seat, candidate_index: int) -> BatchOp:
        return BatchOp("choose", {"gameId": game_id, "seat": seat, "candidateIndex": candidate_index})

    @staticmethod
    def result(game_id: str) -> BatchOp:
        return BatchOp("result", {"gameId": game_id})

    @staticmethod
    def clone(game_id: str, *, new_game_id: str | None = None) -> BatchOp:
        params: dict[str, Any] = {"gameId": game_id}
        if new_game_id is not None:
            params["newGameId"] = new_game_id
        return BatchOp("clone", params)

    @staticmethod
    def close(game_id: str) -> BatchOp:
        return BatchOp("close", {"gameId": game_id})


@dataclass(frozen=True, slots=True)
class BatchResult:
    ok: bool
    value: Any = None
    error: Exception | None = None


class SpaceObservation(TypedDict):
    id: str
    x: int
    y: int
    spaceType: str
    bonus: list[int]
    tileType: int | None
    owner: Seat | None
    highlight: str | None
    rotated: bool


class MilestoneScoreObservation(TypedDict):
    seat: Seat
    score: int
    claimable: bool


class MilestoneObservation(TypedDict):
    name: str
    claimedBy: Seat | None
    scores: list[MilestoneScoreObservation]


class AwardScoreObservation(TypedDict):
    seat: Seat
    score: int


class AwardObservation(TypedDict):
    name: str
    fundedBy: Seat | None
    scores: list[AwardScoreObservation]


class TableauCardObservation(TypedDict):
    name: str
    resources: int
    isDisabled: bool


class PublicPlayerObservation(TypedDict):
    seat: Seat
    color: str
    terraformRating: int
    megacredits: int
    steel: int
    titanium: int
    plants: int
    energy: int
    heat: int
    megacreditProduction: int
    steelProduction: int
    titaniumProduction: int
    plantProduction: int
    energyProduction: int
    heatProduction: int
    steelValue: int
    titaniumValue: int
    cardCost: int
    cardDiscount: int
    cardsInHandCount: int
    citiesCount: int
    noTagsCount: int
    tags: dict[str, int]
    tableau: list[TableauCardObservation]
    usedActions: list[str]
    actionsTakenThisRound: int
    actionsTakenThisGame: int
    availableBlueCardActionCount: int
    lastCardPlayed: str | None
    isActive: bool
    passed: bool
    protectedResources: dict[str, str]
    protectedProduction: dict[str, str]


class PublicLogDatumObservation(TypedDict):
    type: int | None
    value: str | list[str]


class PublicLogEntryObservation(TypedDict):
    type: int
    message: str
    data: list[PublicLogDatumObservation]


class GlobalParametersObservation(TypedDict):
    temperature: int
    oxygenLevel: int
    oceans: int
    isTerraformed: bool


class DeckPileSizesObservation(TypedDict):
    drawPile: int
    discardPile: int


class OtherDeckSizesObservation(TypedDict):
    corporations: DeckPileSizesObservation
    preludes: DeckPileSizesObservation | None


class PublicObservation(TypedDict):
    map: str
    generation: int
    phase: str
    firstPlayer: Seat
    activePlayer: Seat
    passedPlayers: list[Seat]
    globalParameters: GlobalParametersObservation
    deckSize: int
    discardPileSize: int
    otherDeckSizes: OtherDeckSizesObservation
    spaces: list[SpaceObservation]
    milestones: list[MilestoneObservation]
    awards: list[AwardObservation]
    players: list[PublicPlayerObservation]
    publicLog: list[PublicLogEntryObservation]


class HandCardObservation(TypedDict):
    name: str
    calculatedCost: int


class PrivateObservation(TypedDict):
    cardsInHand: list[HandCardObservation]
    dealtCorporationCards: list[str]
    dealtPreludeCards: list[str]
    dealtProjectCards: list[str]
    pickedCorporationCard: str | None
    preludeCardsInHand: list[str]
    draftedCards: list[str]
    currentDraftPacket: list[str]


class InitialBuyHistory(TypedDict):
    corporationsOffered: list[str]
    corporationChosen: str
    corporationsDiscarded: list[str]
    preludesOffered: list[str]
    preludesChosen: list[str]
    preludesDiscarded: list[str]
    projectsOffered: list[str]
    projectsBought: list[str]
    projectsDiscarded: list[str]


class DraftHistoryEntry(TypedDict):
    generation: int
    draftRound: int
    packet: list[str]
    taken: list[str]
    passed: list[str]


class ResearchBuyHistoryEntry(TypedDict):
    generation: int
    offered: list[str]
    bought: list[str]
    discarded: list[str]


class CardSelectionHistoryEntry(TypedDict):
    generation: int
    phase: str
    offered: list[str]
    selected: list[str]
    unselected: list[str]


class ActorHistory(TypedDict):
    initialBuy: InitialBuyHistory | None
    drafts: list[DraftHistoryEntry]
    researchBuys: list[ResearchBuyHistoryEntry]
    cardSelections: list[CardSelectionHistoryEntry]


class ActorObservation(TypedDict):
    actor: Seat
    public: PublicObservation
    private: PrivateObservation
    history: ActorHistory


class ActionRecord(TypedDict, total=False):
    decisionIndex: int
    seat: Seat
    phase: str
    generation: int
    inputType: str
    candidateIndices: list[int]
    candidateIds: list[str]
    response: Any
