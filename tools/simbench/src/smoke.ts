import fs from 'node:fs';
import path from 'node:path';
import {BoardNameValue} from './engine';
import {TARGET_MAPS} from './config';
import {GameRunResult, runRandomGame} from './driver';

interface MapSmokeSummary {
  boardName: BoardNameValue;
  firstGame: GameRunResult;
  corpusCount: number;
  completedCount: number;
  completionRate: number;
  meanWallTimeMs: number;
  p95WallTimeMs: number;
  minWallTimeMs: number;
  maxWallTimeMs: number;
  meanGenerations: number;
  meanDecisions: number;
  totalInvalidActions: number;
  totalDeadlocks: number;
  totalStepCaps: number;
  unexpectedInputTypes: Array<string>;
}

function percentile(sorted: ReadonlyArray<number>, p: number): number {
  if (sorted.length === 0) {
    return 0;
  }
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil(p * sorted.length) - 1));
  return sorted[idx];
}

function parseSeedsArg(): number {
  const idx = process.argv.indexOf('--seeds');
  if (idx !== -1 && process.argv[idx + 1]) {
    const val = Number(process.argv[idx + 1]);
    if (Number.isInteger(val) && val >= 1) {
      return val;
    }
  }
  return 20;
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
  const updated = {
    ...existing,
    ...patch,
  };
  fs.writeFileSync(summaryPath, JSON.stringify(updated, null, 2) + '\n', 'utf8');
}

function main(): void {
  const seedsPerMap = parseSeedsArg();
  const baseSeed = 1001;
  const resultsDir = path.resolve(__dirname, '../results');
  fs.mkdirSync(resultsDir, {recursive: true});

  const allRuns: Array<GameRunResult> = [];
  const mapSummaries: Array<MapSmokeSummary> = [];

  console.log(`=== Stage 1: Smoke (${seedsPerMap} seeds/map, 2P Corporate Era + Prelude + Drafting, canonical order) ===`);

  for (const boardName of TARGET_MAPS) {
    const mapRuns: Array<GameRunResult> = [];
    for (let i = 0; i < seedsPerMap; i++) {
      const seed = baseSeed + i;
      const run = runRandomGame({boardName, seed, simultaneousOrder: 'canonical'});
      mapRuns.push(run);
      allRuns.push(run);
    }

    const firstGame = mapRuns[0];
    const completedRuns = mapRuns.filter((r) => r.completed);
    const wallTimes = mapRuns.map((r) => r.wallTimeMs).slice().sort((a, b) => a - b);
    const meanWallTimeMs = wallTimes.reduce((a, b) => a + b, 0) / wallTimes.length;
    const p95WallTimeMs = percentile(wallTimes, 0.95);
    const meanGenerations =
      mapRuns.reduce((a, r) => a + r.generations, 0) / mapRuns.length;
    const meanDecisions =
      mapRuns.reduce((a, r) => a + r.decisions, 0) / mapRuns.length;
    const unexpected = Array.from(
      new Set(mapRuns.flatMap((r) => r.unexpectedInputTypes)),
    );

    const summary: MapSmokeSummary = {
      boardName,
      firstGame,
      corpusCount: mapRuns.length,
      completedCount: completedRuns.length,
      completionRate: completedRuns.length / mapRuns.length,
      meanWallTimeMs: Number(meanWallTimeMs.toFixed(2)),
      p95WallTimeMs: Number(p95WallTimeMs.toFixed(2)),
      minWallTimeMs: Number(wallTimes[0].toFixed(2)),
      maxWallTimeMs: Number(wallTimes[wallTimes.length - 1].toFixed(2)),
      meanGenerations: Number(meanGenerations.toFixed(2)),
      meanDecisions: Number(meanDecisions.toFixed(1)),
      totalInvalidActions: mapRuns.reduce((a, r) => a + r.invalidActions, 0),
      totalDeadlocks: mapRuns.reduce((a, r) => a + r.deadlocks, 0),
      totalStepCaps: mapRuns.reduce((a, r) => a + (r.hitStepCap ? 1 : 0), 0),
      unexpectedInputTypes: unexpected,
    };
    mapSummaries.push(summary);

    console.log(
      `\n[${boardName}] Single-seed smoke (seed=${firstGame.seed}): ` +
        `completed=${firstGame.completed ? 'yes' : 'no'}, ` +
        `gen=${firstGame.generations}, decisions=${firstGame.decisions}, ` +
        `invalid=${firstGame.invalidActions}, deadlocks=${firstGame.deadlocks}, ` +
        `unexpected=[${firstGame.unexpectedInputTypes.join(',')}], ` +
        `scores=${JSON.stringify(firstGame.scores)}, tie=${firstGame.tie}, ` +
        `wall=${firstGame.wallTimeMs.toFixed(2)}ms` +
        (firstGame.error ? `, error=${firstGame.error}` : ''),
    );
    console.log(
      `[${boardName}] Corpus (${summary.corpusCount} seeds): ` +
        `completed=${summary.completedCount}/${summary.corpusCount} (${(summary.completionRate * 100).toFixed(1)}%), ` +
        `mean=${summary.meanWallTimeMs}ms, p95=${summary.p95WallTimeMs}ms, ` +
        `meanGen=${summary.meanGenerations}, meanDecisions=${summary.meanDecisions}, ` +
        `invalid=${summary.totalInvalidActions}, deadlocks=${summary.totalDeadlocks}, stepCaps=${summary.totalStepCaps}`,
    );
  }

  const smokePayload = {
    generatedAt: new Date().toISOString(),
    nodeVersion: process.version,
    simultaneousOrder: 'canonical',
    seedsPerMap,
    baseSeed,
    maps: mapSummaries,
    runs: allRuns,
  };

  fs.writeFileSync(
    path.join(resultsDir, 'smoke.json'),
    JSON.stringify(smokePayload, null, 2) + '\n',
    'utf8',
  );

  updateSummaryFile(resultsDir, {
    stage1_smoke: {
      generatedAt: smokePayload.generatedAt,
      nodeVersion: process.version,
      simultaneousOrder: 'canonical',
      seedsPerMap,
      baseSeed,
      maps: mapSummaries.map((m) => ({
        boardName: m.boardName,
        firstGame: {
          seed: m.firstGame.seed,
          completed: m.firstGame.completed,
          generations: m.firstGame.generations,
          decisions: m.firstGame.decisions,
          invalidActions: m.firstGame.invalidActions,
          deadlocks: m.firstGame.deadlocks,
          unexpectedInputTypes: m.firstGame.unexpectedInputTypes,
          scores: m.firstGame.scores,
          tie: m.firstGame.tie,
          wallTimeMs: Number(m.firstGame.wallTimeMs.toFixed(2)),
        },
        corpusCount: m.corpusCount,
        completedCount: m.completedCount,
        completionRate: m.completionRate,
        meanWallTimeMs: m.meanWallTimeMs,
        p95WallTimeMs: m.p95WallTimeMs,
        totalInvalidActions: m.totalInvalidActions,
        totalDeadlocks: m.totalDeadlocks,
        totalStepCaps: m.totalStepCaps,
        unexpectedInputTypes: m.unexpectedInputTypes,
      })),
    },
  });

  const anyFailure = mapSummaries.some((m) => m.completedCount < m.corpusCount);
  if (anyFailure) {
    process.exitCode = 1;
  }
}

main();
