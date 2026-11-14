import readline from 'node:readline';
import {createGame} from './adapter';
import {sampleWorldFromObservation} from './determinize';
import {serializeObservation} from './observe';
import {AdapterBugError, AdapterGame, Seat, SupportedMap} from './types';

export interface WorkerRequest {
  readonly id?: string | number | null;
  readonly method?: string;
  readonly params?: Record<string, unknown>;
  readonly batch?: ReadonlyArray<WorkerRequest>;
}

export interface StructuredWorkerError {
  readonly type: string;
  readonly message: string;
  readonly seedKey?: string;
  readonly map?: SupportedMap;
  readonly seat?: Seat;
  readonly history?: ReadonlyArray<unknown>;
}

export interface WorkerItemResponse {
  readonly id?: string | number | null;
  readonly ok: boolean;
  readonly result?: unknown;
  readonly error?: StructuredWorkerError;
}

export interface WorkerBatchResponse {
  readonly id?: string | number | null;
  readonly ok: true;
  readonly results: ReadonlyArray<WorkerItemResponse>;
}

export type WorkerResponse = WorkerItemResponse | WorkerBatchResponse;

class UnsupportedMethodError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'UnsupportedMethodError';
  }
}

class ProtocolError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ProtocolError';
  }
}

export interface WorkerActionStep {
  readonly seat: Seat;
  readonly candidateIndex: number;
}

export class GameWorker {
  private readonly games = new Map<string, AdapterGame>();
  private readonly actionLogs = new Map<string, Array<WorkerActionStep>>();
  private nextGameSeq = 1;
  private shouldShutdown = false;

  public get shutdownRequested(): boolean {
    return this.shouldShutdown;
  }

  public get gameCount(): number {
    return this.games.size;
  }

  public handleMessage(raw: unknown): WorkerResponse {
    if (raw === null || typeof raw !== 'object' || Array.isArray(raw)) {
      return {
        id: null,
        ok: false,
        error: {
          type: 'ProtocolError',
          message: 'Request must be a JSON object',
        },
      };
    }

    const req = raw as WorkerRequest;
    const reqId = req.id ?? null;

    if (Array.isArray(req.batch)) {
      const results: Array<WorkerItemResponse> = req.batch.map((subReq) => this.executeSingle(subReq));
      return {
        id: reqId,
        ok: true,
        results,
      };
    }

    return this.executeSingle(req);
  }

  private executeSingle(rawReq: unknown): WorkerItemResponse {
    if (rawReq === null || typeof rawReq !== 'object' || Array.isArray(rawReq)) {
      return {
        id: null,
        ok: false,
        error: {
          type: 'ProtocolError',
          message: 'Sub-request must be a JSON object',
        },
      };
    }

    const req = rawReq as WorkerRequest;
    const reqId = req.id ?? null;
    const method = req.method;
    const params = req.params ?? {};

    if (typeof method !== 'string' || method.length === 0) {
      return {
        id: reqId,
        ok: false,
        error: {
          type: 'ProtocolError',
          message: 'Request is missing a non-empty "method" string',
        },
      };
    }

    if (params === null || typeof params !== 'object' || Array.isArray(params)) {
      return {
        id: reqId,
        ok: false,
        error: {
          type: 'ProtocolError',
          message: 'Request "params" must be an object',
        },
      };
    }

    let targetGame: AdapterGame | undefined;
    let targetGameId: string | undefined;

    try {
      switch (method) {
        case 'ping': {
          return {id: reqId, ok: true, result: {ok: true, games: this.games.size}};
        }

        case 'shutdown': {
          this.games.clear();
          this.actionLogs.clear();
          this.shouldShutdown = true;
          return {id: reqId, ok: true, result: {ok: true}};
        }

        case 'create': {
          const seedKey = params.seedKey;
          const map = params.map;
          const requestedGameId = params.gameId;

          let gameId: string;
          if (requestedGameId !== undefined && requestedGameId !== null) {
            if (typeof requestedGameId !== 'string' || requestedGameId.length === 0) {
              throw new ProtocolError('create "gameId" must be a non-empty string when provided');
            }
            if (this.games.has(requestedGameId)) {
              throw new ProtocolError(`Game "${requestedGameId}" already exists`);
            }
            gameId = requestedGameId;
          } else {
            do {
              gameId = `g-${this.nextGameSeq++}`;
            } while (this.games.has(gameId));
          }

          const game = createGame({
            seedKey: seedKey as string,
            map: map as SupportedMap,
          });
          this.games.set(gameId, game);
          this.actionLogs.set(gameId, []);
          return {
            id: reqId,
            ok: true,
            result: {
              gameId,
              seedKey: game.seedKey,
              map: game.map,
            },
          };
        }

        case 'pending': {
          const {gameId, game} = this.requireGame(params);
          targetGameId = gameId;
          targetGame = game;
          return {id: reqId, ok: true, result: {pending: game.pending()}};
        }

        case 'observe': {
          const {gameId, game} = this.requireGame(params);
          targetGameId = gameId;
          targetGame = game;
          const seat = this.requireSeat(params);
          const observation = game.observe(seat);
          if (params.serialized === true) {
            return {id: reqId, ok: true, result: {observation: serializeObservation(observation)}};
          }
          return {id: reqId, ok: true, result: {observation}};
        }

        case 'choice': {
          const {gameId, game} = this.requireGame(params);
          targetGameId = gameId;
          targetGame = game;
          const seat = this.requireSeat(params);
          return {id: reqId, ok: true, result: {choice: game.choice(seat)}};
        }

        case 'choose': {
          const {gameId, game} = this.requireGame(params);
          targetGameId = gameId;
          targetGame = game;
          const seat = this.requireSeat(params);
          const candidateIndex = params.candidateIndex;
          if (typeof candidateIndex !== 'number' || !Number.isInteger(candidateIndex)) {
            throw new ProtocolError(`Invalid candidateIndex "${String(candidateIndex)}": expected integer`);
          }
          game.choose(seat, candidateIndex);
          this.actionLogs.get(gameId)?.push({seat, candidateIndex});
          return {id: reqId, ok: true, result: {ok: true}};
        }

        case 'result': {
          const {gameId, game} = this.requireGame(params);
          targetGameId = gameId;
          targetGame = game;
          return {id: reqId, ok: true, result: {result: game.result()}};
        }

        case 'clone': {
          const {gameId, game} = this.requireGame(params);
          targetGameId = gameId;
          targetGame = game;
          const maybeClone = (game as unknown as {clone?: () => AdapterGame}).clone;
          if (typeof maybeClone !== 'function') {
            throw new UnsupportedMethodError('AdapterGame.clone() is not available on this adapter build');
          }

          const requestedNewId = params.newGameId;
          let newGameId: string;
          if (requestedNewId !== undefined && requestedNewId !== null) {
            if (typeof requestedNewId !== 'string' || requestedNewId.length === 0) {
              throw new ProtocolError('clone "newGameId" must be a non-empty string when provided');
            }
            if (this.games.has(requestedNewId)) {
              throw new ProtocolError(`Game "${requestedNewId}" already exists`);
            }
            newGameId = requestedNewId;
          } else {
            do {
              newGameId = `g-${this.nextGameSeq++}`;
            } while (this.games.has(newGameId));
          }

          const cloned = maybeClone.call(game);
          this.games.set(newGameId, cloned);
          const sourceLog = this.actionLogs.get(gameId);
          this.actionLogs.set(newGameId, sourceLog ? sourceLog.slice() : []);
          return {
            id: reqId,
            ok: true,
            result: {
              gameId: newGameId,
              seedKey: cloned.seedKey,
              map: cloned.map,
            },
          };
        }

        case 'determinize_clone': {
          const {gameId, game} = this.requireGame(params);
          targetGameId = gameId;
          targetGame = game;
          const seat = this.requireSeat(params);
          const seed = params.seed;
          if (typeof seed !== 'number' || !Number.isFinite(seed)) {
            throw new ProtocolError(`Invalid seed "${String(seed)}": expected finite number`);
          }
          if (!game.pending().includes(seat)) {
            throw new ProtocolError(`determinize_clone(${seat}) requires seat ${seat} to be pending`);
          }

          const requestedNewId = params.newGameId;
          let newGameId: string;
          if (requestedNewId !== undefined && requestedNewId !== null) {
            if (typeof requestedNewId !== 'string' || requestedNewId.length === 0) {
              throw new ProtocolError('determinize_clone "newGameId" must be a non-empty string when provided');
            }
            if (this.games.has(requestedNewId)) {
              throw new ProtocolError(`Game "${requestedNewId}" already exists`);
            }
            newGameId = requestedNewId;
          } else {
            do {
              newGameId = `g-${this.nextGameSeq++}`;
            } while (this.games.has(newGameId));
          }

          // Sample strictly from the acting seat's own observation before cloning/applying.
          const obs = game.observe(seat);
          const world = sampleWorldFromObservation(obs, seed);
          const cloned = game.clone();
          cloned.determinize(seat, world);
          this.games.set(newGameId, cloned);
          const sourceLog = this.actionLogs.get(gameId);
          this.actionLogs.set(newGameId, sourceLog ? sourceLog.slice() : []);
          return {
            id: reqId,
            ok: true,
            result: {
              gameId: newGameId,
              seedKey: cloned.seedKey,
              map: cloned.map,
            },
          };
        }

        case 'close':
        case 'dispose': {
          const gameId = params.gameId;
          if (typeof gameId !== 'string' || gameId.length === 0) {
            throw new ProtocolError('Missing non-empty "gameId"');
          }
          const closed = this.games.delete(gameId);
          this.actionLogs.delete(gameId);
          return {id: reqId, ok: true, result: {closed}};
        }

        default: {
          throw new UnsupportedMethodError(`Unknown method "${method}"`);
        }
      }
    } catch (err) {
      return {
        id: reqId,
        ok: false,
        error: this.formatError(err, targetGame, targetGameId),
      };
    }
  }

  private requireGame(params: Record<string, unknown>): {gameId: string; game: AdapterGame} {
    const gameId = params.gameId;
    if (typeof gameId !== 'string' || gameId.length === 0) {
      throw new ProtocolError('Missing non-empty "gameId"');
    }
    const game = this.games.get(gameId);
    if (game === undefined) {
      throw new ProtocolError(`Unknown gameId "${gameId}"`);
    }
    return {gameId, game};
  }

  private requireSeat(params: Record<string, unknown>): Seat {
    const seat = params.seat;
    if (seat !== 0 && seat !== 1) {
      throw new ProtocolError(`Invalid seat "${String(seat)}": expected 0 or 1`);
    }
    return seat;
  }

  private formatError(
    err: unknown,
    game: AdapterGame | undefined,
    gameId: string | undefined,
  ): StructuredWorkerError {
    if (err instanceof AdapterBugError) {
      return {
        type: 'AdapterBugError',
        message: err.message,
        seedKey: err.seedKey,
        map: err.map,
        seat: err.seat,
        history: err.history,
      };
    }
    const type = err instanceof Error ? err.name : 'Error';
    const message = err instanceof Error ? err.message : String(err);
    if (game !== undefined && gameId !== undefined) {
      return {
        type,
        message,
        seedKey: game.seedKey,
        map: game.map,
        history: (this.actionLogs.get(gameId) ?? []).slice(),
      };
    }
    return {type, message};
  }
}

export function runStdioWorker(): void {
  // Reserve process.stdout strictly for line-delimited JSON protocol messages.
  const writeStdout = process.stdout.write.bind(process.stdout);
  const logStderr = (...args: Array<unknown>): void => {
    const line = args.map((a) => (typeof a === 'string' ? a : JSON.stringify(a))).join(' ');
    process.stderr.write(`${line}\n`);
  };
  console.log = logStderr;
  console.info = logStderr;
  console.debug = logStderr;
  console.warn = logStderr;

  const worker = new GameWorker();
  const rl = readline.createInterface({
    input: process.stdin,
    crlfDelay: Infinity,
    terminal: false,
  });

  rl.on('line', (line: string) => {
    if (line.trim().length === 0) {
      return;
    }
    let response: WorkerResponse;
    try {
      const parsed: unknown = JSON.parse(line);
      response = worker.handleMessage(parsed);
    } catch (err) {
      response = {
        id: null,
        ok: false,
        error: {
          type: 'ProtocolError',
          message: `Invalid JSON request: ${err instanceof Error ? err.message : String(err)}`,
        },
      };
      process.stderr.write(`[worker] Invalid JSON line: ${line.slice(0, 200)}\n`);
    }

    writeStdout(`${JSON.stringify(response)}\n`);
    if (worker.shutdownRequested) {
      rl.close();
    }
  });

  rl.on('close', () => {
    process.exit(0);
  });
}

if (require.main === module) {
  runStdioWorker();
}
