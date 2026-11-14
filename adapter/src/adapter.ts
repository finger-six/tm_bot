import {
  Game,
  GameId,
  IGame,
  IPlayer,
  InputResponse,
  Phase,
  Player,
  PlayerId,
  PlayerInput,
  SpectatorId,
  initializeEngine,
} from './engine';
import {buildTrainingGameOptions, hashSeedKey, validateCreateGameOptions} from './config';
import {DecisionStepper} from './factor';
import {
  ActionRecord,
  AdapterBugError,
  AdapterGame,
  Choice,
  CreateGameOptions,
  GameResult,
  Seat,
  SupportedMap,
} from './types';

export const DEFAULT_MAX_STEPS = 20_000;

const ENGINE_TO_ADAPTER_SYMBOL = Symbol('engineGame');

export function _unsafeGetEngineGame(game: AdapterGame): IGame {
  const internal = (game as unknown as {[ENGINE_TO_ADAPTER_SYMBOL]?: IGame})[ENGINE_TO_ADAPTER_SYMBOL];
  if (!internal) {
    throw new Error('Not an AdapterGame instance');
  }
  return internal;
}

interface DeferredSubmission {
  readonly record: ActionRecord;
  readonly response: InputResponse;
}

export class GameAdapter implements AdapterGame {
  public readonly seedKey: string;
  public readonly map: SupportedMap;
  private readonly [ENGINE_TO_ADAPTER_SYMBOL]: IGame;
  private readonly history: Array<ActionRecord> = [];
  private readonly activeSteppers: [DecisionStepper | undefined, DecisionStepper | undefined] = [undefined, undefined];
  private readonly maxSteps: number;
  // Completed responses held back so simultaneous prompts reach the engine in canonical seat order.
  private readonly deferred: [DeferredSubmission | undefined, DeferredSubmission | undefined] = [undefined, undefined];

  constructor(engineGame: IGame, seedKey: string, map: SupportedMap, maxSteps = DEFAULT_MAX_STEPS) {
    this[ENGINE_TO_ADAPTER_SYMBOL] = engineGame;
    this.seedKey = seedKey;
    this.map = map;
    this.maxSteps = maxSteps;
  }

  public pending(): Array<Seat> {
    const game = this[ENGINE_TO_ADAPTER_SYMBOL];
    if (game.phase === Phase.END) {
      return [];
    }
    const seats: Array<Seat> = [];
    for (let i = 0; i < game.players.length; i++) {
      const waiting = game.players[i].getWaitingFor();
      if (waiting !== undefined && !waiting.optional && this.deferred[i] === undefined) {
        seats.push(i as Seat);
      }
    }
    if (seats.length === 0) {
      throw new AdapterBugError({
        message: `Deadlock: game phase=${game.phase}, generation=${game.generation}, seedKey="${this.seedKey}" with no waiting players`,
        seedKey: this.seedKey,
        map: this.map,
        seat: 0,
        history: this.history.slice(),
      });
    }
    return seats;
  }

  public choice(seat: Seat): Choice {
    const stepper = this.ensureStepper(seat);
    return stepper.currentChoice();
  }

  public choose(seat: Seat, candidateIndex: number): void {
    const game = this[ENGINE_TO_ADAPTER_SYMBOL];
    const stepper = this.ensureStepper(seat);

    let outcome: {done: false} | {done: true; response: InputResponse};
    try {
      outcome = stepper.choose(candidateIndex);
    } catch (err) {
      if (err instanceof RangeError) {
        throw err;
      }
      this.activeSteppers[seat] = undefined;
      throw new AdapterBugError({
        message: `Adapter bug building response for seedKey="${this.seedKey}", map="${this.map}", seat=${seat}: ${err instanceof Error ? err.message : String(err)}`,
        seedKey: this.seedKey,
        map: this.map,
        seat,
        history: this.history.slice(),
        engineError: err,
      });
    }

    if (!outcome.done) {
      return;
    }

    const record: ActionRecord = {
      decisionIndex: this.history.length,
      seat,
      phase: stepper.phase,
      generation: stepper.generation,
      inputType: stepper.rootInput.type,
      candidateIndices: stepper.candidateIndices.slice(),
      candidateIds: stepper.candidateIds.slice(),
      response: outcome.response,
    };

    this.activeSteppers[seat] = undefined;

    // A lower seat still owes an answer to the same simultaneous phase: hold this response.
    if (this.lowerSeatPending(seat)) {
      this.deferred[seat] = {record, response: outcome.response};
      return;
    }
    this.submit(seat, record, outcome.response);
    for (let other = seat + 1; other < this.deferred.length; other++) {
      const held = this.deferred[other];
      if (held !== undefined && !this.lowerSeatPending(other as Seat)) {
        this.deferred[other] = undefined;
        this.submit(other as Seat, held.record, held.response);
      }
    }
  }

  private lowerSeatPending(seat: Seat): boolean {
    const game = this[ENGINE_TO_ADAPTER_SYMBOL];
    for (let i = 0; i < seat; i++) {
      const waiting = game.players[i].getWaitingFor();
      if (waiting !== undefined && !waiting.optional && this.deferred[i] === undefined) {
        return true;
      }
    }
    return false;
  }

  private submit(seat: Seat, record: ActionRecord, response: InputResponse): void {
    const game = this[ENGINE_TO_ADAPTER_SYMBOL];
    const player = game.players[seat];
    record = {...record, decisionIndex: this.history.length};

    try {
      player.process(response);
    } catch (err) {
      const failureHistory = [...this.history, record];
      throw new AdapterBugError({
        message: `Engine rejected adapter response at decision #${record.decisionIndex} (seedKey="${this.seedKey}", map="${this.map}", seat=${seat}, phase=${record.phase}, gen=${record.generation}, input=${record.inputType}): ${err instanceof Error ? err.message : String(err)}`,
        seedKey: this.seedKey,
        map: this.map,
        seat,
        history: failureHistory,
        engineError: err,
      });
    }

    this.history.push(record);

    if (game.phase !== Phase.END && this.history.length >= this.maxSteps) {
      throw new AdapterBugError({
        message: `Step cap (${this.maxSteps}) exceeded for seedKey="${this.seedKey}", map="${this.map}" at phase=${game.phase}, generation=${game.generation}`,
        seedKey: this.seedKey,
        map: this.map,
        seat,
        history: this.history.slice(),
      });
    }
  }

  public result(): GameResult | null {
    const game = this[ENGINE_TO_ADAPTER_SYMBOL];
    if (game.phase !== Phase.END) {
      return null;
    }
    const s0 = game.players[0].getVictoryPoints().total;
    const s1 = game.players[1].getVictoryPoints().total;
    // Official tie-break: most M€ wins; an exact tie needs equal VP and equal M€.
    const k0 = s0 !== s1 ? s0 : game.players[0].megaCredits;
    const k1 = s0 !== s1 ? s1 : game.players[1].megaCredits;
    const tie = k0 === k1;
    const winner: Seat | null = tie ? null : k0 > k1 ? 0 : 1;
    return {
      scores: [s0, s1],
      winner,
      tie,
      generations: game.generation,
    };
  }

  private ensureStepper(seat: Seat): DecisionStepper {
    if (seat !== 0 && seat !== 1) {
      throw new Error(`Invalid seat ${String(seat)}: expected 0 or 1`);
    }
    const game = this[ENGINE_TO_ADAPTER_SYMBOL];
    if (game.phase === Phase.END) {
      throw new Error(`Game already ended (seedKey="${this.seedKey}")`);
    }

    const player = game.players[seat];
    const waiting = player.getWaitingFor();
    if (waiting === undefined || waiting.optional || this.deferred[seat] !== undefined) {
      throw new Error(`Seat ${seat} is not pending a required decision`);
    }

    const existing = this.activeSteppers[seat];
    if (existing !== undefined && existing.rootInput === waiting) {
      return existing;
    }

    let stepper: DecisionStepper;
    try {
      stepper = new DecisionStepper(seat, waiting, player, game.phase, game.generation);
    } catch (err) {
      const pendingRecord: ActionRecord = {
        decisionIndex: this.history.length,
        seat,
        phase: game.phase,
        generation: game.generation,
        inputType: waiting.type,
        candidateIndices: [],
        candidateIds: [],
        response: undefined,
      };
      throw new AdapterBugError({
        message: `Adapter failed to factor prompt (seedKey="${this.seedKey}", map="${this.map}", seat=${seat}, phase=${game.phase}, gen=${game.generation}, input=${waiting.type}): ${err instanceof Error ? err.message : String(err)}`,
        seedKey: this.seedKey,
        map: this.map,
        seat,
        history: [...this.history, pendingRecord],
        engineError: err,
      });
    }

    this.activeSteppers[seat] = stepper;
    return stepper;
  }
}

export function createGame(options: CreateGameOptions): AdapterGame {
  const {seedKey, map} = validateCreateGameOptions(options);
  initializeEngine();

  const {seed, seedHex} = hashSeedKey(seedKey);
  const gameId: GameId = `g-ad-${seedHex}` as GameId;
  const spectatorId: SpectatorId = `s-ad-${seedHex}` as SpectatorId;
  const p1: IPlayer = new Player('Seat 1', 'red', false, 0, 'p-1' as PlayerId);
  const p2: IPlayer = new Player('Seat 2', 'blue', false, 0, 'p-2' as PlayerId);
  const players: [IPlayer, IPlayer] = [p1, p2];
  const gameOptions = buildTrainingGameOptions(map);
  const engineGame = Game.newInstance(gameId, players, players[0], spectatorId, gameOptions, seed);

  return new GameAdapter(engineGame, seedKey, map);
}
