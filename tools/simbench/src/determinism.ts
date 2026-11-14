import fs from 'node:fs';
import path from 'node:path';
import {TARGET_MAPS, createExplicitTwoPlayerGame} from './config';
import {RecordedAction, collectPlayerScores, getRequiredWaitingPlayers, runRandomGame} from './driver';
import {BoardNameValue, GameId, Phase, SpectatorId} from './engine';
import {findFirstDiff, normalizeSerializedGame} from './normalize';

interface AlternatedOrderResult {
  passed: boolean;
  passedIgnoringGameLog: boolean;
  simultaneousPairsFound: number;
  simultaneousPairsSwapped: number;
  barriersChecked: number;
  strictMismatches: number;
  firstStrictDiff: string | null;
  nonLogMismatches: number;
  firstNonLogDiff: string | null;
  scoresMatch: boolean;
  scores: Record<string, number>;
  error?: string;
}

interface MapDeterminismResult {
  boardName: BoardNameValue;
  seed: number;
  driverSeed: number;
  recordedCompleted: boolean;
  totalActions: number;
  strictReplay: {
    passed: boolean;
    stepsChecked: number;
    mismatches: number;
    firstDiff: string | null;
    scoresMatch: boolean;
    scores: Record<string, number>;
  };
  alternatedDraftAndResearchOrder: AlternatedOrderResult;
  alternatedDraftOnlyOrder: AlternatedOrderResult;
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

function replayExactWithSnapshots(
  boardName: BoardNameValue,
  seed: number,
  trace: ReadonlyArray<RecordedAction>,
  gameId: GameId,
  spectatorId: SpectatorId,
): {
  snapshots: Array<unknown>;
  scores: Record<string, number>;
  phase: string;
} {
  const {game, players} = createExplicitTwoPlayerGame({
    boardName,
    seed,
    gameId,
    spectatorId,
  });
  const byColor = new Map(players.map((p) => [p.color, p]));
  const snapshots: Array<unknown> = [];

  for (const action of trace) {
    const player = byColor.get(action.playerColor);
    if (!player) {
      throw new Error(`Missing player ${action.playerColor}`);
    }
    player.process(action.response);
    snapshots.push(normalizeSerializedGame(game.serialize()));
  }

  return {
    snapshots,
    scores: collectPlayerScores(game).scores,
    phase: game.phase,
  };
}

function replayWithSwappedPhases(
  boardName: BoardNameValue,
  seed: number,
  trace: ReadonlyArray<RecordedAction>,
  referenceSnapshots: ReadonlyArray<unknown>,
  referenceScores: Record<string, number>,
  phasesToSwap: ReadonlySet<string>,
  tag: string,
): AlternatedOrderResult {
  const {game, players} = createExplicitTwoPlayerGame({
    boardName,
    seed,
    gameId: `g-${tag}-${seed}` as GameId,
    spectatorId: `s-${tag}-${seed}` as SpectatorId,
  });
  const byColor = new Map(players.map((p) => [p.color, p]));

  let simultaneousPairsFound = 0;
  let simultaneousPairsSwapped = 0;
  let barriersChecked = 0;
  let strictMismatches = 0;
  let firstStrictDiff: string | null = null;
  let nonLogMismatches = 0;
  let firstNonLogDiff: string | null = null;
  let error: string | undefined;

  const checkBarrier = (refIndex: number) => {
    barriersChecked += 1;
    const actualNorm = normalizeSerializedGame(game.serialize());
    const refNorm = referenceSnapshots[refIndex];
    const strictDiff = findFirstDiff(refNorm, actualNorm);
    if (strictDiff !== null) {
      strictMismatches += 1;
      if (firstStrictDiff === null) {
        firstStrictDiff = `step=${refIndex + 1}: ${strictDiff}`;
      }
      const refNoLog = JSON.parse(JSON.stringify(refNorm)) as Record<string, unknown>;
      const actNoLog = JSON.parse(JSON.stringify(actualNorm)) as Record<string, unknown>;
      delete refNoLog.gameLog;
      delete actNoLog.gameLog;
      const nonLogDiff = findFirstDiff(refNoLog, actNoLog);
      if (nonLogDiff !== null) {
        nonLogMismatches += 1;
        if (firstNonLogDiff === null) {
          firstNonLogDiff = `step=${refIndex + 1}: ${nonLogDiff}`;
        }
      }
    }
  };

  let i = 0;
  try {
    while (i < trace.length) {
      const waitingNow = getRequiredWaitingPlayers(game);
      const a1 = trace[i];
      const a2 = i + 1 < trace.length ? trace[i + 1] : undefined;

      const isSimultaneousPair =
        waitingNow.length === 2 &&
        a2 !== undefined &&
        a1.playerColor !== a2.playerColor &&
        a1.waitingRequiredColors.length === 2 &&
        phasesToSwap.has(a1.phase);

      if (isSimultaneousPair && a2) {
        simultaneousPairsFound += 1;
        const pSecond = byColor.get(a2.playerColor)!;
        const pFirst = byColor.get(a1.playerColor)!;
        pSecond.process(a2.response);
        pFirst.process(a1.response);
        simultaneousPairsSwapped += 1;
        checkBarrier(i + 1);
        i += 2;
        continue;
      }

      const player = byColor.get(a1.playerColor)!;
      player.process(a1.response);
      checkBarrier(i);
      i += 1;
    }
  } catch (err) {
    error = `Replay aborted at step=${i + 1} (gen=${game.generation}, phase=${game.phase}): ${err instanceof Error ? err.message : String(err)}`;
  }

  const completed = error === undefined && game.phase === Phase.END;
  const actualScores = completed ? collectPlayerScores(game).scores : {};
  const scoresMatch = completed && JSON.stringify(referenceScores) === JSON.stringify(actualScores);

  return {
    passed: completed && strictMismatches === 0 && scoresMatch,
    passedIgnoringGameLog: completed && nonLogMismatches === 0 && scoresMatch,
    simultaneousPairsFound,
    simultaneousPairsSwapped,
    barriersChecked,
    strictMismatches,
    firstStrictDiff,
    nonLogMismatches,
    firstNonLogDiff,
    scoresMatch,
    scores: actualScores,
    ...(error ? {error} : {}),
  };
}

function main(): void {
  const resultsDir = path.resolve(__dirname, '../results');
  fs.mkdirSync(resultsDir, {recursive: true});

  const baseSeed = 2001;
  const mapResults: Array<MapDeterminismResult> = [];

  console.log('=== Stage 2: Determinism (Strict Replay + Alternated Simultaneous Submission Order) ===');

  for (let idx = 0; idx < TARGET_MAPS.length; idx++) {
    const boardName = TARGET_MAPS[idx];
    const seed = baseSeed + idx;

    const recorded = runRandomGame({boardName, seed, recordTrace: true});
    const trace = recorded.trace ?? [];
    if (!recorded.completed) {
      throw new Error(`Recording game did not complete on ${boardName} (seed=${seed}): ${recorded.error}`);
    }

    // Run A: baseline snapshot capture
    const runA = replayExactWithSnapshots(
      boardName,
      seed,
      trace,
      `g-det-a-${seed}` as GameId,
      `s-det-a-${seed}` as SpectatorId,
    );

    // Run B: exact replay with distinct gameId/spectatorId
    const runB = replayExactWithSnapshots(
      boardName,
      seed,
      trace,
      `g-det-b-${seed}` as GameId,
      `s-det-b-${seed}` as SpectatorId,
    );

    let strictMismatches = 0;
    let firstStrictDiff: string | null = null;
    for (let i = 0; i < runA.snapshots.length; i++) {
      const diff = findFirstDiff(runA.snapshots[i], runB.snapshots[i]);
      if (diff !== null) {
        strictMismatches += 1;
        if (firstStrictDiff === null) {
          firstStrictDiff = `step=${i + 1}: ${diff}`;
        }
      }
    }
    const strictScoresMatch = JSON.stringify(runA.scores) === JSON.stringify(runB.scores);
    const strictPassed = strictMismatches === 0 && strictScoresMatch && runB.phase === Phase.END;

    // Run C: swap both DRAFTING and RESEARCH simultaneous pairs
    const altDraftAndResearch = replayWithSwappedPhases(
      boardName,
      seed,
      trace,
      runA.snapshots,
      runA.scores,
      new Set([Phase.DRAFTING, Phase.RESEARCH, Phase.INITIALDRAFTING]),
      'alt-both',
    );

    // Run D: swap ONLY DRAFTING simultaneous pairs (keeping RESEARCH canonical)
    const altDraftOnly = replayWithSwappedPhases(
      boardName,
      seed,
      trace,
      runA.snapshots,
      runA.scores,
      new Set([Phase.DRAFTING, Phase.INITIALDRAFTING]),
      'alt-draft',
    );

    const res: MapDeterminismResult = {
      boardName,
      seed,
      driverSeed: recorded.driverSeed,
      recordedCompleted: recorded.completed,
      totalActions: trace.length,
      strictReplay: {
        passed: strictPassed,
        stepsChecked: runA.snapshots.length,
        mismatches: strictMismatches,
        firstDiff: firstStrictDiff,
        scoresMatch: strictScoresMatch,
        scores: runB.scores,
      },
      alternatedDraftAndResearchOrder: altDraftAndResearch,
      alternatedDraftOnlyOrder: altDraftOnly,
    };
    mapResults.push(res);

    console.log(
      `\n[${boardName}] (seed=${seed}, actions=${trace.length}):\n` +
        `  Strict replay: ${res.strictReplay.passed ? 'PASS' : 'FAIL'} ` +
        `(steps=${res.strictReplay.stepsChecked}, mismatches=${res.strictReplay.mismatches}, ` +
        `scoresMatch=${res.strictReplay.scoresMatch}, scores=${JSON.stringify(res.strictReplay.scores)})` +
        (res.strictReplay.firstDiff ? `\n    firstDiff: ${res.strictReplay.firstDiff}` : '') +
        `\n  Alternated Draft-only order: ${altDraftOnly.passed ? 'PASS' : 'FAIL'} ` +
        `(swapped=${altDraftOnly.simultaneousPairsSwapped}, barriers=${altDraftOnly.barriersChecked}, ` +
        `mismatches=${altDraftOnly.strictMismatches}, scoresMatch=${altDraftOnly.scoresMatch})` +
        (altDraftOnly.firstStrictDiff ? `\n    firstStrictDiff: ${altDraftOnly.firstStrictDiff}` : '') +
        `\n  Alternated Draft+Research order: ${altDraftAndResearch.passed ? 'PASS' : 'FAIL'} ` +
        `(swapped=${altDraftAndResearch.simultaneousPairsSwapped}, barriers=${altDraftAndResearch.barriersChecked}, ` +
        `strictMismatches=${altDraftAndResearch.strictMismatches}, nonLogMismatches=${altDraftAndResearch.nonLogMismatches}, ` +
        `scoresMatch=${altDraftAndResearch.scoresMatch})` +
        (altDraftAndResearch.firstNonLogDiff ? `\n    firstNonLogDiff: ${altDraftAndResearch.firstNonLogDiff}` : '') +
        (altDraftAndResearch.error ? `\n    error: ${altDraftAndResearch.error}` : ''),
    );
  }

  const payload = {
    generatedAt: new Date().toISOString(),
    nodeVersion: process.version,
    maps: mapResults,
  };

  fs.writeFileSync(
    path.join(resultsDir, 'determinism.json'),
    JSON.stringify(payload, null, 2) + '\n',
    'utf8',
  );

  updateSummaryFile(resultsDir, {
    stage2_determinism: payload,
  });

  if (mapResults.some((m) => !m.strictReplay.passed)) {
    process.exitCode = 1;
  }
}

main();
