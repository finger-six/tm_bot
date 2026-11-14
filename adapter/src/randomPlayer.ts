import {createGame} from './adapter';
import {hashSeedKey} from './config';
import {GameResult, Seat, SupportedMap} from './types';

/**
 * Mulberry32 deterministic PRNG for the B0-style uniform-random test chooser.
 */
export class SeededPrng {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  public nextInt(max: number): number {
    if (max <= 0) {
      throw new Error(`nextInt max must be > 0, got ${max}`);
    }
    let t = (this.state += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    const u = ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    return Math.floor(u * max);
  }
}

export interface RecordedAtomicChoice {
  readonly seat: Seat;
  readonly candidateIndex: number;
}

export interface RandomGameOptions {
  readonly seedKey: string;
  readonly map: SupportedMap;
  readonly driverSeed?: number;
  readonly recordChoices?: boolean;
}

export interface RandomGameOutcome {
  readonly seedKey: string;
  readonly map: SupportedMap;
  readonly result: GameResult;
  readonly atomicChoices: number;
  readonly wallTimeMs: number;
  readonly choices?: ReadonlyArray<RecordedAtomicChoice>;
}

/**
 * Plays a complete training-format game through the factored adapter interface
 * using a uniform-random chooser (B0 baseline logic) in canonical seat order.
 */
export function playRandomGame(options: RandomGameOptions): RandomGameOutcome {
  const startNs = process.hrtime.bigint();
  const driverSeed =
    options.driverSeed ?? Math.floor(hashSeedKey(`${options.seedKey}:driver`).seed * 4294967296);
  const prng = new SeededPrng(driverSeed);

  const game = createGame({seedKey: options.seedKey, map: options.map});
  const choices: Array<RecordedAtomicChoice> | undefined = options.recordChoices ? [] : undefined;
  let atomicChoices = 0;

  while (true) {
    const seats = game.pending();
    if (seats.length === 0) {
      break;
    }
    const seat = seats[0];
    const current = game.choice(seat);
    const candidateIndex = prng.nextInt(current.candidates.length);
    game.choose(seat, candidateIndex);
    atomicChoices += 1;
    if (choices !== undefined) {
      choices.push({seat, candidateIndex});
    }
  }

  const result = game.result();
  if (result === null) {
    throw new Error(`Game ended without a terminal result (seedKey="${options.seedKey}")`);
  }

  const endNs = process.hrtime.bigint();
  const wallTimeMs = Number(endNs - startNs) / 1e6;

  return {
    seedKey: options.seedKey,
    map: options.map,
    result,
    atomicChoices,
    wallTimeMs,
    ...(choices !== undefined ? {choices} : {}),
  };
}
