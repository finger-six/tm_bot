import fs from 'node:fs';
import path from 'node:path';
import {TARGET_MAPS, createExplicitTwoPlayerGame} from './config';
import {
  DEFAULT_MAX_STEPS,
  MAX_RETRIES_PER_DECISION,
  buildRandomResponse,
  getRequiredWaitingPlayers,
  pickNextWaitingPlayer,
  stepDecision,
} from './driver';
import {BoardNameValue, Game, InputResponse, Phase, Server} from './engine';
import {runHttpComparisonBenchmark} from './httpBench';
import {SeededPrng} from './prng';

interface DistStats {
  count: number;
  mean: number;
  median: number;
  p95: number;
  max: number;
  min: number;
}

interface TertileBreakdown {
  overall: DistStats;
  early: DistStats;
  mid: DistStats;
  late: DistStats;
}

interface TimedSample {
  generation: number;
  value: number;
}

function summarize(values: ReadonlyArray<number>, decimals = 4): DistStats {
  if (values.length === 0) {
    return {count: 0, mean: 0, median: 0, p95: 0, max: 0, min: 0};
  }
  const sorted = values.slice().sort((a, b) => a - b);
  const sum = sorted.reduce((acc, v) => acc + v, 0);
  const p50Idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil(0.5 * sorted.length) - 1));
  const p95Idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil(0.95 * sorted.length) - 1));
  const round = (x: number) => Number(x.toFixed(decimals));
  return {
    count: sorted.length,
    mean: round(sum / sorted.length),
    median: round(sorted[p50Idx]),
    p95: round(sorted[p95Idx]),
    max: round(sorted[sorted.length - 1]),
    min: round(sorted[0]),
  };
}

function summarizeTertiles(
  perGameSamples: ReadonlyArray<{totalGenerations: number; samples: ReadonlyArray<TimedSample>}>,
  decimals = 4,
): TertileBreakdown {
  const all: Array<number> = [];
  const early: Array<number> = [];
  const mid: Array<number> = [];
  const late: Array<number> = [];

  for (const g of perGameSamples) {
    const t1 = Math.ceil(g.totalGenerations / 3);
    const t2 = Math.ceil((2 * g.totalGenerations) / 3);
    for (const s of g.samples) {
      all.push(s.value);
      if (s.generation <= t1) {
        early.push(s.value);
      } else if (s.generation <= t2) {
        mid.push(s.value);
      } else {
        late.push(s.value);
      }
    }
  }

  return {
    overall: summarize(all, decimals),
    early: summarize(early, decimals),
    mid: summarize(mid, decimals),
    late: summarize(late, decimals),
  };
}

function runWarmup(): void {
  for (let m = 0; m < TARGET_MAPS.length; m++) {
    const boardName = TARGET_MAPS[m];
    const seed = 3901 + m;
    const prng = new SeededPrng((seed ^ 0x9e3779b9) >>> 0);
    const {game} = createExplicitTwoPlayerGame({boardName, seed});
    const unexpected = new Set<string>();
    let steps = 0;
    while (game.phase !== Phase.END && steps < DEFAULT_MAX_STEPS) {
      const waiting = getRequiredWaitingPlayers(game);
      if (waiting.length === 0) {
        break;
      }
      const player = pickNextWaitingPlayer(waiting, prng, 'canonical');
      const wf = player.getWaitingFor();
      if (!wf) {
        break;
      }
      Server.getPlayerModel(player);
      if (steps % 20 === 0) {
        const json = JSON.stringify(game.serialize());
        Game.deserialize(JSON.parse(json) as Record<string, unknown>);
      }
      const res = stepDecision(player, wf, prng, unexpected);
      if (!res.succeeded) {
        break;
      }
      steps += 1;
    }
  }
}

function runPassNoObs(boardName: BoardNameValue, seed: number): {
  completed: boolean;
  generations: number;
  decisions: number;
  resetMs: number;
  gameWallMs: number;
  rssMB: number;
  heapUsedMB: number;
} {
  const driverSeed = (seed ^ 0x9e3779b9) >>> 0;
  const prng = new SeededPrng(driverSeed);
  const unexpected = new Set<string>();

  const totalStart = process.hrtime.bigint();
  const resetStart = totalStart;
  const {game} = createExplicitTwoPlayerGame({boardName, seed});
  const resetEnd = process.hrtime.bigint();

  let decisions = 0;
  while (game.phase !== Phase.END && decisions < DEFAULT_MAX_STEPS) {
    const waiting = getRequiredWaitingPlayers(game);
    if (waiting.length === 0) {
      break;
    }
    const player = pickNextWaitingPlayer(waiting, prng, 'canonical');
    const wf = player.getWaitingFor();
    if (!wf) {
      break;
    }
    const step = stepDecision(player, wf, prng, unexpected);
    if (!step.succeeded) {
      break;
    }
    decisions += 1;
  }
  const totalEnd = process.hrtime.bigint();
  const mem = process.memoryUsage();

  return {
    completed: game.phase === Phase.END,
    generations: game.generation,
    decisions,
    resetMs: Number(resetEnd - resetStart) / 1e6,
    gameWallMs: Number(totalEnd - totalStart) / 1e6,
    rssMB: mem.rss / (1024 * 1024),
    heapUsedMB: mem.heapUsed / (1024 * 1024),
  };
}

function runPassWithObs(boardName: BoardNameValue, seed: number): {
  completed: boolean;
  generations: number;
  decisions: number;
  gameWallMs: number;
  obsSamples: Array<TimedSample>;
  stepSamples: Array<TimedSample>;
} {
  const driverSeed = (seed ^ 0x9e3779b9) >>> 0;
  const prng = new SeededPrng(driverSeed);
  const unexpected = new Set<string>();

  const obsSamples: Array<TimedSample> = [];
  const stepSamples: Array<TimedSample> = [];

  const totalStart = process.hrtime.bigint();
  const {game} = createExplicitTwoPlayerGame({boardName, seed});

  let decisions = 0;
  while (game.phase !== Phase.END && decisions < DEFAULT_MAX_STEPS) {
    const waiting = getRequiredWaitingPlayers(game);
    if (waiting.length === 0) {
      break;
    }
    const player = pickNextWaitingPlayer(waiting, prng, 'canonical');
    const wf = player.getWaitingFor();
    if (!wf) {
      break;
    }

    const gen = game.generation;
    const obsStart = process.hrtime.bigint();
    Server.getPlayerModel(player);
    const obsEnd = process.hrtime.bigint();
    obsSamples.push({generation: gen, value: Number(obsEnd - obsStart) / 1e6});

    const failedOrIndices = new Set<number>();
    let succeeded = false;
    for (let attempt = 0; attempt < MAX_RETRIES_PER_DECISION; attempt++) {
      let response: InputResponse;
      try {
        response = buildRandomResponse(wf, player, prng, unexpected, failedOrIndices);
      } catch {
        break;
      }
      const stepStart = process.hrtime.bigint();
      try {
        player.process(response);
        const stepEnd = process.hrtime.bigint();
        stepSamples.push({generation: gen, value: Number(stepEnd - stepStart) / 1e6});
        succeeded = true;
        decisions += 1;
        break;
      } catch {
        if (response.type === 'or') {
          failedOrIndices.add(response.index);
        }
      }
    }
    if (!succeeded) {
      break;
    }
  }
  const totalEnd = process.hrtime.bigint();

  return {
    completed: game.phase === Phase.END,
    generations: game.generation,
    decisions,
    gameWallMs: Number(totalEnd - totalStart) / 1e6,
    obsSamples,
    stepSamples,
  };
}

function runPassSerialize(boardName: BoardNameValue, seed: number): {
  completed: boolean;
  generations: number;
  serializeSamples: Array<TimedSample>;
  deserializeSamples: Array<TimedSample>;
  bytesSamples: Array<TimedSample>;
  deserializeErrors: number;
  rssMB: number;
  heapUsedMB: number;
} {
  const driverSeed = (seed ^ 0x9e3779b9) >>> 0;
  const prng = new SeededPrng(driverSeed);
  const unexpected = new Set<string>();

  const serializeSamples: Array<TimedSample> = [];
  const deserializeSamples: Array<TimedSample> = [];
  const bytesSamples: Array<TimedSample> = [];
  let deserializeErrors = 0;

  const {game} = createExplicitTwoPlayerGame({boardName, seed});
  let decisions = 0;

  while (game.phase !== Phase.END && decisions < DEFAULT_MAX_STEPS) {
    const waiting = getRequiredWaitingPlayers(game);
    if (waiting.length === 0) {
      break;
    }
    const player = pickNextWaitingPlayer(waiting, prng, 'canonical');
    const wf = player.getWaitingFor();
    if (!wf) {
      break;
    }

    const gen = game.generation;
    const serStart = process.hrtime.bigint();
    const jsonStr = JSON.stringify(game.serialize());
    const serEnd = process.hrtime.bigint();
    serializeSamples.push({generation: gen, value: Number(serEnd - serStart) / 1e6});
    bytesSamples.push({generation: gen, value: Buffer.byteLength(jsonStr, 'utf8')});

    const deserStart = process.hrtime.bigint();
    try {
      Game.deserialize(JSON.parse(jsonStr) as Record<string, unknown>);
      const deserEnd = process.hrtime.bigint();
      deserializeSamples.push({generation: gen, value: Number(deserEnd - deserStart) / 1e6});
    } catch {
      deserializeErrors += 1;
    }

    const step = stepDecision(player, wf, prng, unexpected);
    if (!step.succeeded) {
      break;
    }
    decisions += 1;
  }

  const mem = process.memoryUsage();
  return {
    completed: game.phase === Phase.END,
    generations: game.generation,
    serializeSamples,
    deserializeSamples,
    bytesSamples,
    deserializeErrors,
    rssMB: mem.rss / (1024 * 1024),
    heapUsedMB: mem.heapUsed / (1024 * 1024),
  };
}

function updateSummaryFile(resultsDir: string, patch: Record<string, unknown>): void {
  const summaryPath = path.join(resultsDir, 'summary.json');
  let existing: Record<string, unknown> = {};
  if (fs.existsSync(summaryPath)) {
    try {
      existing = JSON.parse(fs.readFileSync(summaryPath, 'utf8')) as Record<string, unknown>;
    } catch {
      existing = {};
    }
  }
  fs.writeFileSync(summaryPath, JSON.stringify({...existing, ...patch}, null, 2) + '\n', 'utf8');
}

async function main(): Promise<void> {
  const resultsDir = path.resolve(__dirname, '../results');
  fs.mkdirSync(resultsDir, {recursive: true});

  const gamesPerMap = 20;
  const baseSeed = 4001;

  console.log(`=== Stage 4: Latency (${gamesPerMap} games/map = ${gamesPerMap * TARGET_MAPS.length} games, canonical order) ===`);
  console.log('Running warm-up (4 games, 1 per map)...');
  runWarmup();

  const resetMsAll: Array<number> = [];
  const gameWallNoObsAll: Array<number> = [];
  const gameWallWithObsAll: Array<number> = [];
  const rssSamples: Array<number> = [];
  const heapUsedSamples: Array<number> = [];

  const noObsByMap: Record<string, Array<number>> = {};
  const withObsByMap: Record<string, Array<number>> = {};

  const obsPerGame: Array<{totalGenerations: number; samples: Array<TimedSample>}> = [];
  const stepPerGame: Array<{totalGenerations: number; samples: Array<TimedSample>}> = [];
  const serPerGame: Array<{totalGenerations: number; samples: Array<TimedSample>}> = [];
  const deserPerGame: Array<{totalGenerations: number; samples: Array<TimedSample>}> = [];
  const bytesPerGame: Array<{totalGenerations: number; samples: Array<TimedSample>}> = [];

  let totalDecisionsNoObs = 0;
  let totalDecisionsWithObs = 0;
  let totalDeserializeErrors = 0;
  let completedNoObs = 0;
  let completedWithObs = 0;

  for (const boardName of TARGET_MAPS) {
    noObsByMap[boardName] = [];
    withObsByMap[boardName] = [];

    for (let i = 0; i < gamesPerMap; i++) {
      const seed = baseSeed + i;

      const rNoObs = runPassNoObs(boardName, seed);
      if (rNoObs.completed) {
        completedNoObs += 1;
      }
      totalDecisionsNoObs += rNoObs.decisions;
      resetMsAll.push(rNoObs.resetMs);
      gameWallNoObsAll.push(rNoObs.gameWallMs);
      noObsByMap[boardName].push(rNoObs.gameWallMs);
      rssSamples.push(rNoObs.rssMB);
      heapUsedSamples.push(rNoObs.heapUsedMB);

      const rWithObs = runPassWithObs(boardName, seed);
      if (rWithObs.completed) {
        completedWithObs += 1;
      }
      totalDecisionsWithObs += rWithObs.decisions;
      gameWallWithObsAll.push(rWithObs.gameWallMs);
      withObsByMap[boardName].push(rWithObs.gameWallMs);
      obsPerGame.push({totalGenerations: rWithObs.generations, samples: rWithObs.obsSamples});
      stepPerGame.push({totalGenerations: rWithObs.generations, samples: rWithObs.stepSamples});

      const rSer = runPassSerialize(boardName, seed);
      totalDeserializeErrors += rSer.deserializeErrors;
      serPerGame.push({totalGenerations: rSer.generations, samples: rSer.serializeSamples});
      deserPerGame.push({totalGenerations: rSer.generations, samples: rSer.deserializeSamples});
      bytesPerGame.push({totalGenerations: rSer.generations, samples: rSer.bytesSamples});
      rssSamples.push(rSer.rssMB);
      heapUsedSamples.push(rSer.heapUsedMB);
    }

    const noObsStats = summarize(noObsByMap[boardName], 2);
    const withObsStats = summarize(withObsByMap[boardName], 2);
    console.log(
      `[${boardName}] gameWall: noObs median=${noObsStats.median}ms p95=${noObsStats.p95}ms max=${noObsStats.max}ms | ` +
        `withObs median=${withObsStats.median}ms p95=${withObsStats.p95}ms max=${withObsStats.max}ms`,
    );
  }

  const resetStats = summarize(resetMsAll, 4);
  const gameWallNoObsStats = summarize(gameWallNoObsAll, 2);
  const gameWallWithObsStats = summarize(gameWallWithObsAll, 2);
  const rssStats = summarize(rssSamples, 2);
  const heapUsedStats = summarize(heapUsedSamples, 2);

  const obsTertiles = summarizeTertiles(obsPerGame, 4);
  const stepTertiles = summarizeTertiles(stepPerGame, 4);
  const serTertiles = summarizeTertiles(serPerGame, 4);
  const deserTertiles = summarizeTertiles(deserPerGame, 4);
  const bytesTertiles = summarizeTertiles(bytesPerGame, 0);

  console.log('\n--- In-Process Aggregate Distributions ---');
  console.log(`reset (Game.newInstance): median=${resetStats.median}ms, p95=${resetStats.p95}ms, max=${resetStats.max}ms (n=${resetStats.count})`);
  console.log(`observation (Server.getPlayerModel): median=${obsTertiles.overall.median}ms, p95=${obsTertiles.overall.p95}ms, max=${obsTertiles.overall.max}ms (n=${obsTertiles.overall.count})`);
  console.log(`  early/mid/late median: ${obsTertiles.early.median}ms / ${obsTertiles.mid.median}ms / ${obsTertiles.late.median}ms`);
  console.log(`step (player.process): median=${stepTertiles.overall.median}ms, p95=${stepTertiles.overall.p95}ms, max=${stepTertiles.overall.max}ms (n=${stepTertiles.overall.count})`);
  console.log(`  early/mid/late median: ${stepTertiles.early.median}ms / ${stepTertiles.mid.median}ms / ${stepTertiles.late.median}ms`);
  console.log(`serialize()+JSON.stringify: median=${serTertiles.overall.median}ms, p95=${serTertiles.overall.p95}ms, max=${serTertiles.overall.max}ms (n=${serTertiles.overall.count})`);
  console.log(`  early/mid/late median: ${serTertiles.early.median}ms / ${serTertiles.mid.median}ms / ${serTertiles.late.median}ms`);
  console.log(`JSON.parse+Game.deserialize: median=${deserTertiles.overall.median}ms, p95=${deserTertiles.overall.p95}ms, max=${deserTertiles.overall.max}ms (n=${deserTertiles.overall.count}, errors=${totalDeserializeErrors})`);
  console.log(`  early/mid/late median: ${deserTertiles.early.median}ms / ${deserTertiles.mid.median}ms / ${deserTertiles.late.median}ms`);
  console.log(`snapshot bytes: median=${bytesTertiles.overall.median} B, p95=${bytesTertiles.overall.p95} B, max=${bytesTertiles.overall.max} B`);
  console.log(`  early/mid/late median: ${bytesTertiles.early.median} B / ${bytesTertiles.mid.median} B / ${bytesTertiles.late.median} B`);
  console.log(`completed-game wall (no obs): median=${gameWallNoObsStats.median}ms, p95=${gameWallNoObsStats.p95}ms, max=${gameWallNoObsStats.max}ms (bar<=30000ms: ${gameWallNoObsStats.max <= 30_000 ? 'PASS' : 'FAIL'})`);
  console.log(`completed-game wall (with obs): median=${gameWallWithObsStats.median}ms, p95=${gameWallWithObsStats.p95}ms, max=${gameWallWithObsStats.max}ms`);
  console.log(`memory: RSS median=${rssStats.median}MB max=${rssStats.max}MB | heapUsed median=${heapUsedStats.median}MB max=${heapUsedStats.max}MB`);

  console.log('\n--- Running Localhost HTTP Server Comparison (1 game/map = 4 games + 1 warmup) ---');
  const httpRaw = await runHttpComparisonBenchmark(1);
  let httpSummary: Record<string, unknown>;
  if (httpRaw.measured) {
    const wfStats = summarize(httpRaw.waitingForMs, 3);
    const postStats = summarize(httpRaw.postInputMs, 3);
    const totalStepStats = summarize(httpRaw.decisionTotalMs, 3);
    const httpGameStats = summarize(httpRaw.gameWallMs, 2);
    const inProcStepPlusObsMedian = obsTertiles.overall.median + stepTertiles.overall.median;
    const perDecisionRatio = inProcStepPlusObsMedian > 0 ?
      Number((totalStepStats.median / inProcStepPlusObsMedian).toFixed(1)) :
      0;
    const gameWallRatio = gameWallWithObsStats.median > 0 ?
      Number((httpGameStats.median / gameWallWithObsStats.median).toFixed(1)) :
      0;

    httpSummary = {
      measured: true,
      dbBackend: httpRaw.dbBackend,
      gamesRun: httpRaw.gamesRun,
      gamesCompleted: httpRaw.gamesCompleted,
      decisions: httpRaw.decisions,
      waitingForMs: wfStats,
      postInputMs: postStats,
      decisionTotalMs: totalStepStats,
      gameWallMs: httpGameStats,
      speedupInProcessVsHttpDecisionMedian: perDecisionRatio,
      speedupInProcessVsHttpGameWallMedian: gameWallRatio,
    };

    console.log(`HTTP (${httpRaw.dbBackend}): completed=${httpRaw.gamesCompleted}/${httpRaw.gamesRun}, decisions=${httpRaw.decisions}`);
    console.log(`  GET /api/waitingfor: median=${wfStats.median}ms, p95=${wfStats.p95}ms, max=${wfStats.max}ms`);
    console.log(`  POST /player/input: median=${postStats.median}ms, p95=${postStats.p95}ms, max=${postStats.max}ms`);
    console.log(`  Combined per-decision (waitingfor + input): median=${totalStepStats.median}ms, p95=${totalStepStats.p95}ms, max=${totalStepStats.max}ms`);
    console.log(`  HTTP completed-game wall: median=${httpGameStats.median}ms, p95=${httpGameStats.p95}ms, max=${httpGameStats.max}ms`);
    console.log(`  In-process vs HTTP speedup: ~${perDecisionRatio}x per-decision, ~${gameWallRatio}x full-game wall`);
  } else {
    httpSummary = {
      measured: false,
      reason: httpRaw.reason ?? 'Failed to run HTTP server benchmark',
      dbBackend: httpRaw.dbBackend,
    };
    console.log(`HTTP comparison NOT MEASURED: ${String(httpSummary.reason)}`);
  }

  const perMapSummary = TARGET_MAPS.map((boardName) => ({
    boardName,
    gameWallNoObsMs: summarize(noObsByMap[boardName], 2),
    gameWallWithObsMs: summarize(withObsByMap[boardName], 2),
  }));

  const payload = {
    generatedAt: new Date().toISOString(),
    nodeVersion: process.version,
    simultaneousOrder: 'canonical',
    gamesPerMap,
    totalGames: gamesPerMap * TARGET_MAPS.length,
    completedNoObs,
    completedWithObs,
    totalDecisionsNoObs,
    totalDecisionsWithObs,
    totalDeserializeErrors,
    passes30sBareEngineBar: gameWallNoObsStats.max <= 30_000,
    resetMs: resetStats,
    gameWallNoObsMs: gameWallNoObsStats,
    gameWallWithObsMs: gameWallWithObsStats,
    perMapSummary,
    observationMs: obsTertiles,
    stepMs: stepTertiles,
    serializeJsonMs: serTertiles,
    parseDeserializeMs: deserTertiles,
    snapshotBytes: bytesTertiles,
    rssMB: rssStats,
    heapUsedMB: heapUsedStats,
    httpComparison: httpSummary,
  };

  fs.writeFileSync(
    path.join(resultsDir, 'latency.json'),
    JSON.stringify(payload, null, 2) + '\n',
    'utf8',
  );

  updateSummaryFile(resultsDir, {
    stage4_latency: payload,
  });
}

void main();
