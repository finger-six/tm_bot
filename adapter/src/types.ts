import type {InputResponse} from './engine';

export const SUPPORTED_MAPS = [
  'tharsis',
  'elysium',
  'hellas',
  'vastitas borealis nova',
] as const;

export type SupportedMap = typeof SUPPORTED_MAPS[number];

export type Seat = 0 | 1;

export type ChoiceKind =
  | 'branch'
  | 'card'
  | 'multi_card'
  | 'project_card'
  | 'payment'
  | 'space'
  | 'player'
  | 'amount'
  | 'resource';

export interface Candidate {
  readonly id: string;
  readonly label: string;
}

export interface Choice {
  readonly seat: Seat;
  readonly kind: ChoiceKind;
  readonly prompt: string;
  readonly candidates: ReadonlyArray<Candidate>;
}

export interface GameResult {
  readonly scores: readonly [number, number];
  readonly winner: Seat | null;
  readonly tie: boolean;
  readonly generations: number;
}

export interface ActionRecord {
  readonly decisionIndex: number;
  readonly seat: Seat;
  readonly phase: string;
  readonly generation: number;
  readonly inputType: string;
  readonly candidateIndices: ReadonlyArray<number>;
  readonly candidateIds: ReadonlyArray<string>;
  readonly response: InputResponse | undefined;
}

export interface SpaceObservation {
  readonly id: string;
  readonly x: number;
  readonly y: number;
  readonly spaceType: string;
  readonly bonus: ReadonlyArray<number>;
  readonly tileType: number | null;
  readonly owner: Seat | null;
  readonly highlight: string | null;
  readonly rotated: boolean;
}

export interface MilestoneObservation {
  readonly name: string;
  readonly claimedBy: Seat | null;
  readonly scores: ReadonlyArray<{
    readonly seat: Seat;
    readonly score: number;
    readonly claimable: boolean;
  }>;
}

export interface AwardObservation {
  readonly name: string;
  readonly fundedBy: Seat | null;
  readonly scores: ReadonlyArray<{
    readonly seat: Seat;
    readonly score: number;
  }>;
}

export interface TableauCardObservation {
  readonly name: string;
  readonly resources: number;
  readonly isDisabled: boolean;
}

export interface PublicPlayerObservation {
  readonly seat: Seat;
  readonly color: string;
  readonly terraformRating: number;
  readonly megacredits: number;
  readonly steel: number;
  readonly titanium: number;
  readonly plants: number;
  readonly energy: number;
  readonly heat: number;
  readonly megacreditProduction: number;
  readonly steelProduction: number;
  readonly titaniumProduction: number;
  readonly plantProduction: number;
  readonly energyProduction: number;
  readonly heatProduction: number;
  readonly steelValue: number;
  readonly titaniumValue: number;
  readonly cardCost: number;
  readonly cardDiscount: number;
  readonly cardsInHandCount: number;
  readonly citiesCount: number;
  readonly noTagsCount: number;
  readonly tags: Readonly<Record<string, number>>;
  readonly tableau: ReadonlyArray<TableauCardObservation>;
  readonly usedActions: ReadonlyArray<string>;
  readonly actionsTakenThisRound: number;
  readonly actionsTakenThisGame: number;
  readonly availableBlueCardActionCount: number;
  readonly lastCardPlayed: string | null;
  readonly isActive: boolean;
  readonly passed: boolean;
  readonly protectedResources: Readonly<Record<string, string>>;
  readonly protectedProduction: Readonly<Record<string, string>>;
}

export interface PublicLogEntryObservation {
  readonly type: number;
  readonly message: string;
  readonly data: ReadonlyArray<{
    readonly type: number | null;
    readonly value: string | ReadonlyArray<string>;
  }>;
}

export interface HandCardObservation {
  readonly name: string;
  readonly calculatedCost: number;
}

export interface DraftHistoryEntry {
  readonly generation: number;
  readonly draftRound: number;
  readonly packet: ReadonlyArray<string>;
  readonly taken: ReadonlyArray<string>;
  readonly passed: ReadonlyArray<string>;
}

export interface InitialBuyHistory {
  readonly corporationsOffered: ReadonlyArray<string>;
  readonly corporationChosen: string;
  readonly corporationsDiscarded: ReadonlyArray<string>;
  readonly preludesOffered: ReadonlyArray<string>;
  readonly preludesChosen: ReadonlyArray<string>;
  readonly preludesDiscarded: ReadonlyArray<string>;
  readonly projectsOffered: ReadonlyArray<string>;
  readonly projectsBought: ReadonlyArray<string>;
  readonly projectsDiscarded: ReadonlyArray<string>;
}

export interface ResearchBuyHistoryEntry {
  readonly generation: number;
  readonly offered: ReadonlyArray<string>;
  readonly bought: ReadonlyArray<string>;
  readonly discarded: ReadonlyArray<string>;
}

/** A card prompt outside setup, drafting and research that showed the actor cards not already in its hand or any tableau (e.g. Inventors' Guild, Business Network). */
export interface CardSelectionHistoryEntry {
  readonly generation: number;
  readonly phase: string;
  readonly offered: ReadonlyArray<string>;
  readonly selected: ReadonlyArray<string>;
  readonly unselected: ReadonlyArray<string>;
}

export interface ActorHistory {
  readonly initialBuy: InitialBuyHistory | null;
  readonly drafts: ReadonlyArray<DraftHistoryEntry>;
  readonly researchBuys: ReadonlyArray<ResearchBuyHistoryEntry>;
  readonly cardSelections: ReadonlyArray<CardSelectionHistoryEntry>;
}

export interface ActorObservation {
  readonly actor: Seat;
  readonly public: {
    readonly map: SupportedMap;
    readonly generation: number;
    readonly phase: string;
    readonly firstPlayer: Seat;
    readonly activePlayer: Seat;
    readonly passedPlayers: ReadonlyArray<Seat>;
    readonly globalParameters: {
      readonly temperature: number;
      readonly oxygenLevel: number;
      readonly oceans: number;
      readonly isTerraformed: boolean;
    };
    readonly deckSize: number;
    readonly discardPileSize: number;
    readonly otherDeckSizes: {
      readonly corporations: {readonly drawPile: number; readonly discardPile: number};
      readonly preludes: {readonly drawPile: number; readonly discardPile: number} | null;
    };
    readonly spaces: ReadonlyArray<SpaceObservation>;
    readonly milestones: ReadonlyArray<MilestoneObservation>;
    readonly awards: ReadonlyArray<AwardObservation>;
    readonly players: readonly [PublicPlayerObservation, PublicPlayerObservation];
    readonly publicLog: ReadonlyArray<PublicLogEntryObservation>;
  };
  readonly private: {
    readonly cardsInHand: ReadonlyArray<HandCardObservation>;
    readonly dealtCorporationCards: ReadonlyArray<string>;
    readonly dealtPreludeCards: ReadonlyArray<string>;
    readonly dealtProjectCards: ReadonlyArray<string>;
    readonly pickedCorporationCard: string | null;
    readonly preludeCardsInHand: ReadonlyArray<string>;
    readonly draftedCards: ReadonlyArray<string>;
    readonly currentDraftPacket: ReadonlyArray<string>;
  };
  readonly history: ActorHistory;
}

export interface CreateGameOptions {
  readonly seedKey: string;
  readonly map: SupportedMap;
}

export interface AdapterGame {
  readonly seedKey: string;
  readonly map: SupportedMap;
  pending(): Array<Seat>;
  choice(seat: Seat): Choice;
  choose(seat: Seat, candidateIndex: number): void;
  observe(seat: Seat): ActorObservation;
  result(): GameResult | null;
}

export class AdapterBugError extends Error {
  public readonly seedKey: string;
  public readonly map: SupportedMap;
  public readonly seat: Seat;
  public readonly history: ReadonlyArray<ActionRecord>;
  public readonly engineError: unknown;

  constructor(params: {
    message: string;
    seedKey: string;
    map: SupportedMap;
    seat: Seat;
    history: ReadonlyArray<ActionRecord>;
    engineError?: unknown;
  }) {
    super(params.message);
    this.name = 'AdapterBugError';
    this.seedKey = params.seedKey;
    this.map = params.map;
    this.seat = params.seat;
    this.history = params.history;
    this.engineError = params.engineError;
  }
}
