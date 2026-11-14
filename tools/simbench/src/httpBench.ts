import {ChildProcess, spawn} from 'node:child_process';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import {TARGET_MAPS} from './config';
import {BoardNameValue, InputResponse, Payment, PaymentUtil, RandomMAOptionType, SpendableResource, Units, UnitsUtil, constants} from './engine';
import {SeededPrng} from './prng';

export interface HttpBenchmarkResult {
  measured: boolean;
  reason?: string;
  dbBackend: string;
  warmupGames: number;
  gamesRun: number;
  gamesCompleted: number;
  decisions: number;
  waitingForMs: Array<number>;
  postInputMs: Array<number>;
  decisionTotalMs: Array<number>;
  gameWallMs: Array<number>;
}

interface HttpCardModel {
  name: string;
  calculatedCost?: number;
  isDisabled?: boolean;
  reserveUnits?: Readonly<Units>;
  standardProjectCanPayWith?: {
    steel?: boolean;
    titanium?: boolean;
    seeds?: boolean;
    kuiperAsteroids?: boolean;
  };
}

interface HttpInputModel {
  type: string;
  title?: unknown;
  buttonLabel?: string;
  optional?: boolean;
  options?: Array<HttpInputModel>;
  cards?: Array<HttpCardModel>;
  min?: number;
  max?: number;
  amount?: number;
  paymentOptions?: Record<string, boolean | undefined>;
  reserveUnits?: Readonly<Units>;
  spaces?: Array<string>;
  players?: Array<string>;
  include?: Array<keyof Units>;
  count?: number;
  payProduction?: {
    cost: number;
    units: Units;
  };
}

interface HttpPlayerViewModel {
  id: string;
  color: string;
  game: {
    phase: string;
    generation: number;
    gameAge: number;
    undoCount: number;
  };
  thisPlayer: {
    cardCost: number;
    megacredits: number;
    steel: number;
    titanium: number;
    plants: number;
    energy: number;
    heat: number;
    steelValue: number;
    titaniumValue: number;
    megacreditProduction: number;
    steelProduction: number;
    titaniumProduction: number;
    plantProduction: number;
    energyProduction: number;
    heatProduction: number;
  };
  waitingFor?: HttpInputModel;
}

function getFreePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.unref();
    srv.on('error', reject);
    srv.listen(0, '127.0.0.1', () => {
      const addr = srv.address();
      if (!addr || typeof addr === 'string') {
        srv.close();
        reject(new Error('Failed to get ephemeral port'));
        return;
      }
      const port = addr.port;
      srv.close((err) => {
        if (err) {
          reject(err);
        } else {
          resolve(port);
        }
      });
    });
  });
}

function getTitleText(title: unknown): string {
  if (typeof title === 'string') {
    return title;
  }
  if (title && typeof title === 'object' && 'message' in title) {
    return String((title as {message: unknown}).message);
  }
  return String(title ?? '');
}

function buildSimplePaymentFromView(
  requiredCost: number,
  paymentOptions: Record<string, boolean | undefined> | undefined,
  reserveUnits: Readonly<Units> | undefined,
  view: HttpPlayerViewModel,
): Payment | undefined {
  const payment: Record<SpendableResource, number> = {...PaymentUtil.EMPTY};
  if (requiredCost <= 0) {
    return payment;
  }
  const tp = view.thisPlayer;
  let remaining = requiredCost;

  if (paymentOptions?.steel && tp.steel > (reserveUnits?.steel ?? 0)) {
    const avail = tp.steel - (reserveUnits?.steel ?? 0);
    const val = tp.steelValue || 2;
    const use = Math.min(avail, Math.floor(remaining / val));
    payment.steel = use;
    remaining -= use * val;
  }
  if (remaining > 0 && paymentOptions?.titanium && tp.titanium > (reserveUnits?.titanium ?? 0)) {
    const avail = tp.titanium - (reserveUnits?.titanium ?? 0);
    const val = tp.titaniumValue || 3;
    const use = Math.min(avail, Math.floor(remaining / val));
    payment.titanium = use;
    remaining -= use * val;
  }
  const mcAvail = Math.max(0, tp.megacredits - (reserveUnits?.megacredits ?? 0));
  const useMc = Math.min(mcAvail, remaining);
  payment.megacredits = useMc;
  remaining -= useMc;

  if (remaining > 0 && paymentOptions?.heat) {
    const heatAvail = Math.max(0, tp.heat - (reserveUnits?.heat ?? 0));
    const useHeat = Math.min(heatAvail, remaining);
    payment.heat = useHeat;
    remaining -= useHeat;
  }
  if (remaining > 0) {
    return undefined;
  }
  return payment;
}

function hasChoiceInModel(input: HttpInputModel, view: HttpPlayerViewModel): boolean {
  switch (input.type) {
  case 'option':
  case 'amount':
  case 'payment':
  case 'productionToLose':
  case 'resource':
  case 'resources':
  case 'initialCards':
    return true;
  case 'space':
    return (input.spaces?.length ?? 0) > 0;
  case 'player':
    return (input.players?.length ?? 0) > 0;
  case 'card': {
    const enabled = (input.cards ?? []).filter((c) => !c.isDisabled);
    return enabled.length >= (input.min ?? 0);
  }
  case 'projectCard': {
    const cards = input.cards ?? [];
    for (const c of cards) {
      if (c.isDisabled) {
        continue;
      }
      const cost = c.calculatedCost ?? 0;
      const opts = c.standardProjectCanPayWith ?
        {...input.paymentOptions, ...c.standardProjectCanPayWith} :
        input.paymentOptions;
      if (buildSimplePaymentFromView(cost, opts, c.reserveUnits, view) !== undefined) {
        return true;
      }
    }
    return false;
  }
  case 'or':
    return (input.options ?? []).some((o) => hasChoiceInModel(o, view));
  case 'and':
    return (input.options ?? []).every((o) => hasChoiceInModel(o, view));
  default:
    return true;
  }
}

function buildResponseFromModel(
  input: HttpInputModel,
  view: HttpPlayerViewModel,
  prng: SeededPrng,
  failedOrIndices?: Set<number>,
): InputResponse {
  switch (input.type) {
  case 'option':
    return {type: 'option'};

  case 'or': {
    const options = input.options ?? [];
    const legal: Array<number> = [];
    for (let i = 0; i < options.length; i++) {
      if (failedOrIndices?.has(i)) {
        continue;
      }
      if (hasChoiceInModel(options[i], view)) {
        legal.push(i);
      }
    }
    if (legal.length === 0) {
      for (let i = 0; i < options.length; i++) {
        if (!failedOrIndices?.has(i)) {
          legal.push(i);
        }
      }
    }
    const idx = prng.pick(legal);
    return {
      type: 'or',
      index: idx,
      response: buildResponseFromModel(options[idx], view, prng),
    };
  }

  case 'and': {
    const options = input.options ?? [];
    return {
      type: 'and',
      responses: options.map((opt) => buildResponseFromModel(opt, view, prng)),
    };
  }

  case 'initialCards': {
    const options = input.options ?? [];
    let chosenCorp: HttpCardModel | undefined;
    const responses: Array<InputResponse> = [];
    for (const opt of options) {
      const title = getTitleText(opt.title).toLowerCase();
      if (title.includes('corporation')) {
        const enabled = (opt.cards ?? []).filter((c) => !c.isDisabled);
        chosenCorp = prng.pick(enabled);
        responses.push({type: 'card', cards: [chosenCorp.name]});
      } else if (title.includes('initial cards') || title.includes('project')) {
        const enabled = (opt.cards ?? []).filter((c) => !c.isDisabled);
        const cardCost = view.thisPlayer.cardCost || 3;
        // Conservative cap to ensure affordability across any starting corp MC
        const maxAff = Math.min(opt.max ?? 10, enabled.length, Math.floor(20 / cardCost));
        const minC = Math.min(opt.min ?? 0, maxAff);
        const count = prng.nextIntRange(minC, maxAff);
        const picked = prng.sample(enabled, count);
        responses.push({type: 'card', cards: picked.map((c) => c.name)});
      } else {
        responses.push(buildResponseFromModel(opt, view, prng));
      }
    }
    return {type: 'initialCards', responses};
  }

  case 'card': {
    const enabled = (input.cards ?? []).filter((c) => !c.isDisabled);
    const maxC = Math.min(input.max ?? 1, enabled.length);
    const minC = Math.min(input.min ?? 0, maxC);
    const count = prng.nextIntRange(minC, maxC);
    const picked = prng.sample(enabled, count);
    return {type: 'card', cards: picked.map((c) => c.name)};
  }

  case 'projectCard': {
    const candidates: Array<{name: string; payment: Payment}> = [];
    for (const c of input.cards ?? []) {
      if (c.isDisabled) {
        continue;
      }
      const opts = c.standardProjectCanPayWith ?
        {...input.paymentOptions, ...c.standardProjectCanPayWith} :
        input.paymentOptions;
      const payment = buildSimplePaymentFromView(c.calculatedCost ?? 0, opts, c.reserveUnits, view);
      if (payment !== undefined) {
        candidates.push({name: c.name, payment});
      }
    }
    if (candidates.length === 0) {
      throw new Error('No playable card in HTTP SelectProjectCardToPlayModel');
    }
    const picked = prng.pick(candidates);
    return {type: 'projectCard', card: picked.name, payment: picked.payment};
  }

  case 'payment': {
    const payment = buildSimplePaymentFromView(
      input.amount ?? 0,
      input.paymentOptions,
      input.reserveUnits,
      view,
    );
    if (payment === undefined) {
      throw new Error(`Cannot afford HTTP payment ${input.amount}`);
    }
    return {type: 'payment', payment};
  }

  case 'space': {
    const spaceId = prng.pick(input.spaces ?? []);
    return {type: 'space', spaceId};
  }

  case 'amount': {
    const min = input.min ?? 0;
    const max = input.max ?? 0;
    return {type: 'amount', amount: prng.nextIntRange(min, max)};
  }

  case 'player': {
    const player = prng.pick(input.players ?? []);
    return {type: 'player', player: player as never};
  }

  case 'productionToLose': {
    const minimums: Record<keyof Units, number> = {
      megacredits: -5,
      steel: 0,
      titanium: 0,
      plants: 0,
      energy: 0,
      heat: 0,
    };
    const tp = view.thisPlayer;
    const current: Units = {
      megacredits: tp.megacreditProduction,
      steel: tp.steelProduction,
      titanium: tp.titaniumProduction,
      plants: tp.plantProduction,
      energy: tp.energyProduction,
      heat: tp.heatProduction,
    };
    const lost: Units = {...UnitsUtil.EMPTY};
    let remaining = input.payProduction?.cost ?? 1;
    const keys: Array<keyof Units> = ['megacredits', 'steel', 'titanium', 'plants', 'energy', 'heat'];
    while (remaining > 0) {
      const reducible = keys.filter((k) => current[k] - lost[k] > minimums[k]);
      const k = prng.pick(reducible);
      lost[k] += 1;
      remaining -= 1;
    }
    return {type: 'productionToLose', units: lost};
  }

  case 'resource':
    return {type: 'resource', resource: prng.pick(input.include ?? ['megacredits'])};

  case 'resources': {
    const units: Units = {...UnitsUtil.EMPTY};
    const count = input.count ?? 0;
    for (let i = 0; i < count; i++) {
      const k = prng.pick(UnitsUtil.keys);
      units[k] += 1;
    }
    return {type: 'resources', units};
  }

  default:
    throw new Error(`Unsupported HTTP input model type: ${input.type}`);
  }
}

async function httpJson<T>(url: string, init?: RequestInit): Promise<{status: number; data: T; elapsedMs: number}> {
  const startNs = process.hrtime.bigint();
  const res = await fetch(url, init);
  const text = await res.text();
  const endNs = process.hrtime.bigint();
  const elapsedMs = Number(endNs - startNs) / 1e6;
  const data = (text ? JSON.parse(text) : {}) as T;
  return {status: res.status, data, elapsedMs};
}

function buildNewGamePayload(boardName: BoardNameValue): Record<string, unknown> {
  return {
    players: [
      {name: 'Seat 1', color: 'red', beginner: false, handicap: 0, first: true},
      {name: 'Seat 2', color: 'blue', beginner: false, handicap: 0, first: false},
    ],
    expansions: {
      corpera: true,
      promo: false,
      venus: false,
      colonies: false,
      prelude: true,
      prelude2: false,
      turmoil: false,
      community: false,
      ares: false,
      moon: false,
      pathfinders: false,
      ceo: false,
      starwars: false,
      underworld: false,
      deltaProject: false,
    },
    board: boardName,
    seed: 0,
    randomFirstPlayer: false,
    clonedGamedId: undefined,
    undoOption: false,
    showTimers: false,
    fastModeOption: false,
    showOtherPlayersVP: false,
    aresExtremeVariant: false,
    politicalAgendasExtension: 'Standard',
    solarPhaseOption: false,
    removeNegativeGlobalEventsOption: false,
    modularMA: false,
    draftVariant: true,
    initialDraft: false,
    preludeDraftVariant: false,
    ceosDraftVariant: false,
    startingCorporations: constants.CORPORATION_CARDS_DEALT_PER_PLAYER,
    shuffleMapOption: false,
    randomMA: RandomMAOptionType.NONE,
    includeFanMA: false,
    soloTR: false,
    customCorporationsList: [],
    bannedCards: [],
    includedCards: [],
    customColoniesList: [],
    customPreludes: [],
    customCeos: [],
    requiresMoonTrackCompletion: false,
    requiresVenusTrackCompletion: false,
    moonStandardProjectVariant: false,
    moonStandardProjectVariant1: false,
    altVenusBoard: false,
    escapeVelocity: undefined,
    twoCorpsVariant: false,
    startingCeos: constants.CEO_CARDS_DEALT_PER_PLAYER,
    startingPreludes: constants.PRELUDE_CARDS_DEALT_PER_PLAYER,
  };
}

async function waitForServerReady(baseUrl: string, child: ChildProcess, stderrBuf: Array<string>): Promise<void> {
  const deadline = Date.now() + 15_000;
  while (Date.now() < deadline) {
    if (child.exitCode !== null) {
      throw new Error(`Engine server exited early (code=${child.exitCode}): ${stderrBuf.join('')}`);
    }
    try {
      const res = await fetch(`${baseUrl}/api/game?id=g0`);
      if (res.status > 0) {
        return;
      }
    } catch {
      // retry until listening
    }
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error(`Timed out waiting for engine HTTP server readiness: ${stderrBuf.join('')}`);
}

async function playOneHttpGame(
  baseUrl: string,
  boardName: BoardNameValue,
  driverSeed: number,
  recordSamples: boolean,
  waitingForSamples: Array<number>,
  postInputSamples: Array<number>,
  decisionTotalSamples: Array<number>,
): Promise<{completed: boolean; decisions: number; wallMs: number; error?: string}> {
  const prng = new SeededPrng(driverSeed);
  const gameStartNs = process.hrtime.bigint();

  const createRes = await httpJson<{players: Array<{id: string; color: string}>}>(`${baseUrl}/api/creategame`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(buildNewGamePayload(boardName)),
  });
  if (createRes.status !== 200) {
    return {completed: false, decisions: 0, wallMs: 0, error: `creategame HTTP ${createRes.status}`};
  }

  const playerIds = createRes.data.players.map((p) => p.id);
  const views = new Map<string, HttpPlayerViewModel>();
  for (const pid of playerIds) {
    const pv = await httpJson<HttpPlayerViewModel>(`${baseUrl}/api/player?id=${pid}`);
    views.set(pid, pv.data);
  }

  let decisions = 0;
  const maxSteps = 20_000;

  while (decisions < maxSteps) {
    const anyEnd = Array.from(views.values()).some((v) => v.game.phase === 'end');
    if (anyEnd) {
      break;
    }

    let acted = false;
    // Canonical seat order: iterate playerIds[0], then playerIds[1]
    for (const pid of playerIds) {
      const curView = views.get(pid)!;
      const wfRes = await httpJson<{result: string; waitingFor: Array<string>}>(
        `${baseUrl}/api/waitingfor?id=${pid}&gameAge=${curView.game.gameAge}&undoCount=${curView.game.undoCount}`,
      );
      if (wfRes.status !== 200) {
        return {completed: false, decisions, wallMs: 0, error: `waitingfor HTTP ${wfRes.status}`};
      }

      if (wfRes.data.result === 'REFRESH') {
        const pv = await httpJson<HttpPlayerViewModel>(`${baseUrl}/api/player?id=${pid}`);
        views.set(pid, pv.data);
      }
      if (wfRes.data.result !== 'GO') {
        continue;
      }

      let activeView = views.get(pid)!;
      if (!activeView.waitingFor || activeView.waitingFor.optional) {
        const pv = await httpJson<HttpPlayerViewModel>(`${baseUrl}/api/player?id=${pid}`);
        activeView = pv.data;
        views.set(pid, activeView);
      }
      if (activeView.game.phase === 'end') {
        acted = true;
        break;
      }
      if (!activeView.waitingFor || activeView.waitingFor.optional) {
        continue;
      }

      const failedOrIndices = new Set<number>();
      let stepSucceeded = false;
      let lastErrMsg = '';
      for (let attempt = 0; attempt < 16; attempt++) {
        let resp: InputResponse;
        try {
          resp = buildResponseFromModel(activeView.waitingFor, activeView, prng, failedOrIndices);
        } catch (err) {
          lastErrMsg = err instanceof Error ? err.message : String(err);
          break;
        }
        const postRes = await httpJson<HttpPlayerViewModel & {message?: string}>(
          `${baseUrl}/player/input?id=${pid}`,
          {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(resp),
          },
        );
        if (postRes.status === 200) {
          views.set(pid, postRes.data);
          decisions += 1;
          acted = true;
          stepSucceeded = true;
          if (recordSamples) {
            waitingForSamples.push(wfRes.elapsedMs);
            postInputSamples.push(postRes.elapsedMs);
            decisionTotalSamples.push(wfRes.elapsedMs + postRes.elapsedMs);
          }
          break;
        } else {
          lastErrMsg = postRes.data?.message ?? `HTTP ${postRes.status}`;
          if (resp.type === 'or') {
            failedOrIndices.add(resp.index);
          } else {
            break;
          }
        }
      }

      if (!stepSucceeded) {
        return {
          completed: false,
          decisions,
          wallMs: Number(process.hrtime.bigint() - gameStartNs) / 1e6,
          error: `Failed HTTP step at gen=${activeView.game.generation}, phase=${activeView.game.phase}, input=${activeView.waitingFor?.type}: ${lastErrMsg}`,
        };
      }
      // Canonical order: break inner loop to re-check from seat 0
      break;
    }

    if (!acted) {
      // Refresh both views before declaring deadlock
      let refreshedGo = false;
      for (const pid of playerIds) {
        const pv = await httpJson<HttpPlayerViewModel>(`${baseUrl}/api/player?id=${pid}`);
        views.set(pid, pv.data);
        if (pv.data.game.phase === 'end' || (pv.data.waitingFor && !pv.data.waitingFor.optional)) {
          refreshedGo = true;
        }
      }
      if (!refreshedGo) {
        return {
          completed: false,
          decisions,
          wallMs: Number(process.hrtime.bigint() - gameStartNs) / 1e6,
          error: 'HTTP deadlock: no required waiting player',
        };
      }
    }
  }

  const gameEndNs = process.hrtime.bigint();
  const completed = Array.from(views.values()).some((v) => v.game.phase === 'end');
  return {
    completed,
    decisions,
    wallMs: Number(gameEndNs - gameStartNs) / 1e6,
  };
}

export async function runHttpComparisonBenchmark(gamesPerMap = 2): Promise<HttpBenchmarkResult> {
  const serverEntry = path.resolve(__dirname, '../engine/build/src/server/server.js');
  if (!fs.existsSync(serverEntry)) {
    return {
      measured: false,
      reason: `Engine build server not found at ${serverEntry}`,
      dbBackend: 'none',
      warmupGames: 0,
      gamesRun: 0,
      gamesCompleted: 0,
      decisions: 0,
      waitingForMs: [],
      postInputMs: [],
      decisionTotalMs: [],
      gameWallMs: [],
    };
  }

  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'tm-simbench-http-'));
  // ServeAsset's static initializer reads build/styles.css{,.gz,.br} from process.cwd()
  const tempBuildDir = path.join(tempDir, 'build');
  fs.mkdirSync(tempBuildDir, {recursive: true});
  fs.writeFileSync(path.join(tempBuildDir, 'styles.css'), '');
  fs.writeFileSync(path.join(tempBuildDir, 'styles.css.gz'), '');
  fs.writeFileSync(path.join(tempBuildDir, 'styles.css.br'), '');
  const port = await getFreePort();
  const baseUrl = `http://127.0.0.1:${port}`;
  const stderrBuf: Array<string> = [];

  const child = spawn(process.execPath, [serverEntry], {
    cwd: tempDir,
    env: {
      ...process.env,
      PORT: String(port),
      HOST: '127.0.0.1',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  const stdoutBuf: Array<string> = [];
  child.stdout?.on('data', (chunk: Buffer) => {
    if (stdoutBuf.length < 200) {
      stdoutBuf.push(chunk.toString('utf8'));
    }
  });
  child.stderr?.on('data', (chunk: Buffer) => {
    stderrBuf.push(chunk.toString('utf8'));
  });

  const waitingForMs: Array<number> = [];
  const postInputMs: Array<number> = [];
  const decisionTotalMs: Array<number> = [];
  const gameWallMs: Array<number> = [];

  try {
    await waitForServerReady(baseUrl, child, stderrBuf);

    // 1 warm-up game
    const warmRes = await playOneHttpGame(baseUrl, TARGET_MAPS[0], 99001, false, waitingForMs, postInputMs, decisionTotalMs);
    console.log(`[HTTP] Warm-up (${TARGET_MAPS[0]}): completed=${warmRes.completed}, decisions=${warmRes.decisions}, wall=${warmRes.wallMs.toFixed(0)}ms${warmRes.error ? `, err=${warmRes.error}` : ''}`);

    let gamesRun = 0;
    let gamesCompleted = 0;
    let decisions = 0;

    for (let g = 0; g < gamesPerMap; g++) {
      for (let m = 0; m < TARGET_MAPS.length; m++) {
        const boardName = TARGET_MAPS[m];
        const driverSeed = 9001 + g * 10 + m;
        const res = await playOneHttpGame(
          baseUrl,
          boardName,
          driverSeed,
          true,
          waitingForMs,
          postInputMs,
          decisionTotalMs,
        );
        gamesRun += 1;
        if (res.completed) {
          gamesCompleted += 1;
          decisions += res.decisions;
          gameWallMs.push(res.wallMs);
          console.log(`[HTTP] Game ${gamesRun}/${gamesPerMap * TARGET_MAPS.length} (${boardName}): ${res.decisions} dec in ${res.wallMs.toFixed(0)}ms`);
        } else {
          console.warn(`[HTTP] Game failed on ${boardName} (seed=${driverSeed}): ${res.error}`);
        }
      }
    }

    return {
      measured: gamesCompleted > 0,
      ...(gamesCompleted === 0 ? {reason: 'No HTTP games completed'} : {}),
      dbBackend: 'SQLite (engine default in temp dir)',
      warmupGames: 1,
      gamesRun,
      gamesCompleted,
      decisions,
      waitingForMs,
      postInputMs,
      decisionTotalMs,
      gameWallMs,
    };
  } catch (err) {
    return {
      measured: false,
      reason: err instanceof Error ? err.message : String(err),
      dbBackend: 'SQLite (engine default in temp dir)',
      warmupGames: 0,
      gamesRun: 0,
      gamesCompleted: 0,
      decisions: 0,
      waitingForMs: [],
      postInputMs: [],
      decisionTotalMs: [],
      gameWallMs: [],
    };
  } finally {
    if (child.exitCode === null) {
      const exitPromise = new Promise<void>((resolve) => {
        child.once('exit', () => resolve());
        setTimeout(resolve, 1500);
      });
      child.kill('SIGTERM');
      await exitPromise;
      if (child.exitCode === null) {
        child.kill('SIGKILL');
        await new Promise((r) => setTimeout(r, 200));
      }
    }
    try {
      fs.rmSync(tempDir, {recursive: true, force: true});
    } catch {
      // best-effort cleanup
    }
  }
}
