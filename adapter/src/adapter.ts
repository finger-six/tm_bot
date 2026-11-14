import {
  CardName,
  Game,
  GameId,
  IGame,
  IPlayer,
  InputResponse,
  Phase,
  Player,
  PlayerId,
  SelectCard,
  SelectInitialCards,
  SpectatorId,
  initializeEngine,
} from './engine';
import {buildTrainingGameOptions, hashSeedKey, validateCreateGameOptions} from './config';
import {DecisionStepper} from './factor';
import {buildActorObservation} from './observe';
import {
  ActionRecord,
  ActorObservation,
  AdapterBugError,
  AdapterGame,
  CardSelectionHistoryEntry,
  Choice,
  CreateGameOptions,
  DraftHistoryEntry,
  GameResult,
  InitialBuyHistory,
  ResearchBuyHistoryEntry,
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

interface MutableActorHistory {
  initialBuy: InitialBuyHistory | null;
  readonly drafts: Array<DraftHistoryEntry>;
  readonly researchBuys: Array<ResearchBuyHistoryEntry>;
  readonly cardSelections: Array<CardSelectionHistoryEntry>;
}

function createEmptyActorHistory(): MutableActorHistory {
  return {
    initialBuy: null,
    drafts: [],
    researchBuys: [],
    cardSelections: [],
  };
}

function compareStrings(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

function subtractCards(offeredSorted: ReadonlyArray<string>, chosenSorted: ReadonlyArray<string>): Array<string> {
  const counts = new Map<string, number>();
  for (const name of chosenSorted) {
    counts.set(name, (counts.get(name) ?? 0) + 1);
  }
  const remaining: Array<string> = [];
  for (const name of offeredSorted) {
    const c = counts.get(name) ?? 0;
    if (c > 0) {
      counts.set(name, c - 1);
    } else {
      remaining.push(name);
    }
  }
  return remaining;
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
  private readonly actorHistories: [MutableActorHistory, MutableActorHistory] = [
    createEmptyActorHistory(),
    createEmptyActorHistory(),
  ];

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

  public observe(seat: Seat): ActorObservation {
    if (seat !== 0 && seat !== 1) {
      throw new Error(`Invalid seat ${String(seat)}: expected 0 or 1`);
    }
    return buildActorObservation(
      this[ENGINE_TO_ADAPTER_SYMBOL],
      this.map,
      seat,
      this.actorHistories[seat],
    );
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

    const historySnapshot = this.snapshotHistoryContext(player, game);

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
    this.commitHistoryContext(seat, historySnapshot, response);

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

  private snapshotHistoryContext(
    player: IPlayer,
    game: IGame,
  ):
    | {
        kind: 'initialCards';
        slots: Array<{role: 'corp' | 'prelude' | 'project' | 'other'; offered: Array<string>}>;
      }
    | {kind: 'draft'; generation: number; draftRound: number; packet: Array<string>}
    | {kind: 'research'; generation: number; offered: Array<string>}
    | {kind: 'cardSelection'; generation: number; phase: string; offered: Array<string>}
    | null {
    const waitingBefore = player.getWaitingFor();
    if (waitingBefore === undefined) {
      return null;
    }

    if (waitingBefore.type === 'initialCards') {
      const initInput = waitingBefore as SelectInitialCards;
      const slots = initInput.options.map((opt) => {
        const offered =
          opt.type === 'card'
            ? (opt as SelectCard).cards.map((c) => c.name).sort(compareStrings)
            : [];
        const role: 'corp' | 'prelude' | 'project' | 'other' =
          opt === initInput.inputs.corp
            ? 'corp'
            : opt === initInput.inputs.prelude
              ? 'prelude'
              : opt === initInput.inputs.project
                ? 'project'
                : 'other';
        return {role, offered};
      });
      return {kind: 'initialCards', slots};
    }

    if (
      (game.phase === Phase.DRAFTING || game.phase === Phase.INITIALDRAFTING) &&
      player.needsToDraft === true &&
      waitingBefore.type === 'card'
    ) {
      return {
        kind: 'draft',
        generation: game.generation,
        draftRound: game.draftRound,
        packet: (waitingBefore as SelectCard).cards.map((c) => c.name).sort(compareStrings),
      };
    }

    if (game.phase === Phase.RESEARCH && waitingBefore.type === 'card') {
      return {
        kind: 'research',
        generation: game.generation,
        offered: (waitingBefore as SelectCard).cards.map((c) => c.name).sort(compareStrings),
      };
    }

    if (waitingBefore.type === 'card') {
      // Only prompts showing the actor otherwise-unseen cards (look-and-buy, draw-and-keep) add history;
      // choices among its own hand or public tableau cards add no private knowledge.
      const offered = (waitingBefore as SelectCard).cards.map((c) => c.name).sort(compareStrings);
      const inHand = new Set<string>([...player.cardsInHand, ...player.preludeCardsInHand].map((c) => c.name));
      const isKnown = (name: string): boolean =>
        inHand.has(name) || game.players.some((p) => p.tableau.has(name as CardName));
      if (offered.some((name) => !isKnown(name))) {
        return {kind: 'cardSelection', generation: game.generation, phase: game.phase, offered};
      }
    }

    return null;
  }

  private commitHistoryContext(
    seat: Seat,
    snapshot: ReturnType<GameAdapter['snapshotHistoryContext']>,
    response: InputResponse,
  ): void {
    if (snapshot === null) {
      return;
    }
    const actorHistory = this.actorHistories[seat];

    if (snapshot.kind === 'initialCards' && response.type === 'initialCards') {
      let corporationsOffered: Array<string> = [];
      let corporationChosen = '';
      let preludesOffered: Array<string> = [];
      let preludesChosen: Array<string> = [];
      let projectsOffered: Array<string> = [];
      let projectsBought: Array<string> = [];

      for (let i = 0; i < snapshot.slots.length; i++) {
        const slot = snapshot.slots[i];
        const resp = response.responses[i];
        if (slot === undefined || resp === undefined || resp.type !== 'card') {
          continue;
        }
        const chosen = resp.cards.slice().sort(compareStrings);
        if (slot.role === 'corp') {
          corporationsOffered = slot.offered;
          corporationChosen = chosen[0] ?? '';
        } else if (slot.role === 'prelude') {
          preludesOffered = slot.offered;
          preludesChosen = chosen;
        } else if (slot.role === 'project') {
          projectsOffered = slot.offered;
          projectsBought = chosen;
        }
      }

      actorHistory.initialBuy = {
        corporationsOffered,
        corporationChosen,
        corporationsDiscarded: corporationChosen
          ? subtractCards(corporationsOffered, [corporationChosen])
          : corporationsOffered.slice(),
        preludesOffered,
        preludesChosen,
        preludesDiscarded: subtractCards(preludesOffered, preludesChosen),
        projectsOffered,
        projectsBought,
        projectsDiscarded: subtractCards(projectsOffered, projectsBought),
      };
      return;
    }

    if (snapshot.kind === 'draft' && response.type === 'card') {
      const taken = response.cards.slice().sort(compareStrings);
      const passed = subtractCards(snapshot.packet, taken);
      actorHistory.drafts.push({
        generation: snapshot.generation,
        draftRound: snapshot.draftRound,
        packet: snapshot.packet,
        taken,
        passed,
      });
      return;
    }

    if (snapshot.kind === 'research' && response.type === 'card') {
      const bought = response.cards.slice().sort(compareStrings);
      const discarded = subtractCards(snapshot.offered, bought);
      actorHistory.researchBuys.push({
        generation: snapshot.generation,
        offered: snapshot.offered,
        bought,
        discarded,
      });
      return;
    }

    if (snapshot.kind === 'cardSelection' && response.type === 'card') {
      const selected = response.cards.slice().sort(compareStrings);
      actorHistory.cardSelections.push({
        generation: snapshot.generation,
        phase: snapshot.phase,
        offered: snapshot.offered,
        selected,
        unselected: subtractCards(snapshot.offered, selected),
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
