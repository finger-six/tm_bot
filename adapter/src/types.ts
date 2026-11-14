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
