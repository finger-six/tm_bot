import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {PerformanceObserver, constants as perfConstants, monitorEventLoopDelay} from 'node:perf_hooks';
import {Worker, isMainThread, parentPort, workerData} from 'node:worker_threads';
import {TARGET_MAPS} from './config';
import {runRandomGame} from './driver';
import {initializeEngine} from './engine';

interface WorkerInitData {
  workerIndex: number;
  durationMs: number;
  baseSeed: number;
}

interface WorkerReport {
  workerIndex: number;
  gamesCompleted: number;
  gamesFailed: number;
  decisions: number;
  elapsedMs: number;
  heapUsedPeakMB: number;
  heapUsedEndMB: number;
  gcCount: number;
  gcTotalMs: number;
  gcMaxMs: number;
  gcMajorCount: number;
  gcMajorTotalMs: number;
  gcMajorMaxMs: number;
  eldMeanMs: number;
  eldP50Ms: number;
  eldP95Ms: number;
  eldMaxMs: number;
}

interface ScalingTierResult {
  workers: number;
  durationSec: number;
  actualWallSec: number;
  totalGamesCompleted: number;
  totalGamesFailed: number;
  totalDecisions: number;
  gamesPerSec: number;
  decisionsPerSec: number;
  speedupVs1Worker: number;
  scalingEfficiencyPct: number;
  processRssMeanMB: number;
  processRssMaxMB: number;
  rssPerWorkerMaxMB: number;
  workerHeapPeakAvgMB: number;
  workerHeapPeakMaxMB: number;
  gcTotalEvents: number;
  gcTotalTimeMs: number;
  gcMaxPauseMs: number;
  gcMajorEvents: number;
  gcMajorMaxPauseMs: number;
  eldP50AvgMs: number;
  eldP95MaxMs: number;
  eldMaxPauseMs: number;
}

function yieldEventLoop(): Promise<void> {
  return new Promise((resolve) => setImmediate(resolve));
}

async function runWorkerMain(data: WorkerInitData): Promise<void> {
  initializeEngine();

  // Warm-up 2 games before timing window
  runRandomGame({boardName: TARGET_MAPS[0], seed: 800001 + data.workerIndex, simultaneousOrder: 'canonical'});
  runRandomGame({boardName: TARGET_MAPS[1], seed: 800101 + data.workerIndex, simultaneousOrder: 'canonical'});

  let gcCount = 0;
  let gcTotalMs = 0;
  let gcMaxMs = 0;
  let gcMajorCount = 0;
  let gcMajorTotalMs = 0;
  let gcMajorMaxMs = 0;
  let trackingActive = false;

  const gcObs = new PerformanceObserver((list) => {
    if (!trackingActive) {
      return;
    }
    for (const entry of list.getEntries()) {
      const dur = entry.duration;
      gcCount += 1;
      gcTotalMs += dur;
      if (dur > gcMaxMs) {
        gcMaxMs = dur;
      }
      // Node PerformanceEntry GC detail: kind 2 = Mark-Sweep-Compact (major GC)
      const detail = (entry as unknown as {detail?: {kind?: number}}).detail;
      const kind = detail?.kind ?? (entry as unknown as {kind?: number}).kind;
      if (kind === perfConstants.NODE_PERFORMANCE_GC_MAJOR) {
        gcMajorCount += 1;
        gcMajorTotalMs += dur;
        if (dur > gcMajorMaxMs) {
          gcMajorMaxMs = dur;
        }
      }
    }
  });
  gcObs.observe({entryTypes: ['gc']});

  const eld = monitorEventLoopDelay({resolution: 10});

  await new Promise<void>((resolve) => {
    parentPort!.once('message', (msg: unknown) => {
      if (msg === 'start') {
        resolve();
      }
    });
    parentPort!.postMessage('ready');
  });

  eld.enable();
  trackingActive = true;

  let gamesCompleted = 0;
  let gamesFailed = 0;
  let decisions = 0;
  let gameIdx = 0;
  let heapUsedPeakBytes = process.memoryUsage().heapUsed;

  const startNs = process.hrtime.bigint();
  const targetNs = BigInt(Math.round(data.durationMs * 1e6));

  while (process.hrtime.bigint() - startNs < targetNs) {
    const boardName = TARGET_MAPS[(data.workerIndex + gameIdx) % TARGET_MAPS.length];
    const seed = data.baseSeed + data.workerIndex * 100_000 + gameIdx;
    const res = runRandomGame({
      boardName,
      seed,
      simultaneousOrder: 'canonical',
    });
    if (res.completed) {
      gamesCompleted += 1;
    } else {
      gamesFailed += 1;
    }
    decisions += res.decisions;
    gameIdx += 1;

    if (gameIdx % 10 === 0) {
      const hu = process.memoryUsage().heapUsed;
      if (hu > heapUsedPeakBytes) {
        heapUsedPeakBytes = hu;
      }
    }
    // Yield once per game so monitorEventLoopDelay & GC observers remain responsive
    await yieldEventLoop();
  }

  const endNs = process.hrtime.bigint();
  trackingActive = false;
  eld.disable();
  gcObs.disconnect();

  const memEnd = process.memoryUsage();
  if (memEnd.heapUsed > heapUsedPeakBytes) {
    heapUsedPeakBytes = memEnd.heapUsed;
  }

  const report: WorkerReport = {
    workerIndex: data.workerIndex,
    gamesCompleted,
    gamesFailed,
    decisions,
    elapsedMs: Number(endNs - startNs) / 1e6,
    heapUsedPeakMB: Number((heapUsedPeakBytes / (1024 * 1024)).toFixed(2)),
    heapUsedEndMB: Number((memEnd.heapUsed / (1024 * 1024)).toFixed(2)),
    gcCount,
    gcTotalMs: Number(gcTotalMs.toFixed(2)),
    gcMaxMs: Number(gcMaxMs.toFixed(2)),
    gcMajorCount,
    gcMajorTotalMs: Number(gcMajorTotalMs.toFixed(2)),
    gcMajorMaxMs: Number(gcMajorMaxMs.toFixed(2)),
    eldMeanMs: Number((eld.mean / 1e6).toFixed(2)),
    eldP50Ms: Number((eld.percentile(50) / 1e6).toFixed(2)),
    eldP95Ms: Number((eld.percentile(95) / 1e6).toFixed(2)),
    eldMaxMs: Number((eld.max / 1e6).toFixed(2)),
  };

  parentPort!.postMessage(report);
}

async function runWorkerTier(workerCount: number, durationSec: number): Promise<{
  tier: Omit<ScalingTierResult, 'speedupVs1Worker' | 'scalingEfficiencyPct'>;
  workerReports: Array<WorkerReport>;
}> {
  const durationMs = durationSec * 1000;
  const workers: Array<Worker> = [];
  const readyPromises: Array<Promise<void>> = [];
  const reportPromises: Array<Promise<WorkerReport>> = [];

  for (let i = 0; i < workerCount; i++) {
    const worker = new Worker(__filename, {
      workerData: {
        workerIndex: i,
        durationMs,
        baseSeed: 50001,
      } satisfies WorkerInitData,
    });
    workers.push(worker);

    let resolveReady!: () => void;
    let resolveReport!: (r: WorkerReport) => void;
    let rejectWorker!: (err: Error) => void;

    readyPromises.push(new Promise<void>((res, rej) => {
      resolveReady = res;
      rejectWorker = rej;
    }));
    reportPromises.push(new Promise<WorkerReport>((res, rej) => {
      resolveReport = res;
      const prevReject = rejectWorker;
      rejectWorker = (e) => {
        prevReject(e);
        rej(e);
      };
    }));

    worker.on('message', (msg: unknown) => {
      if (msg === 'ready') {
        resolveReady();
      } else if (msg && typeof msg === 'object' && 'gamesCompleted' in msg) {
        resolveReport(msg as WorkerReport);
      }
    });
    worker.on('error', (err) => rejectWorker(err));
    worker.on('exit', (code) => {
      if (code !== 0) {
        rejectWorker(new Error(`Worker ${i} exited with code ${code}`));
      }
    });
  }

  await Promise.all(readyPromises);

  const rssSamplesMB: Array<number> = [process.memoryUsage().rss / (1024 * 1024)];
  const sampler = setInterval(() => {
    rssSamplesMB.push(process.memoryUsage().rss / (1024 * 1024));
  }, 1000);

  const wallStartNs = process.hrtime.bigint();
  for (const w of workers) {
    w.postMessage('start');
  }

  const workerReports = await Promise.all(reportPromises);
  const wallEndNs = process.hrtime.bigint();
  clearInterval(sampler);
  rssSamplesMB.push(process.memoryUsage().rss / (1024 * 1024));

  await Promise.all(workers.map((w) => w.terminate()));

  const actualWallSec = Number(wallEndNs - wallStartNs) / 1e9;
  const totalGamesCompleted = workerReports.reduce((a, r) => a + r.gamesCompleted, 0);
  const totalGamesFailed = workerReports.reduce((a, r) => a + r.gamesFailed, 0);
  const totalDecisions = workerReports.reduce((a, r) => a + r.decisions, 0);
  const gamesPerSec = totalGamesCompleted / actualWallSec;
  const decisionsPerSec = totalDecisions / actualWallSec;

  const rssMeanMB = rssSamplesMB.reduce((a, v) => a + v, 0) / rssSamplesMB.length;
  const rssMaxMB = Math.max(...rssSamplesMB);

  const tier: Omit<ScalingTierResult, 'speedupVs1Worker' | 'scalingEfficiencyPct'> = {
    workers: workerCount,
    durationSec,
    actualWallSec: Number(actualWallSec.toFixed(2)),
    totalGamesCompleted,
    totalGamesFailed,
    totalDecisions,
    gamesPerSec: Number(gamesPerSec.toFixed(2)),
    decisionsPerSec: Number(decisionsPerSec.toFixed(1)),
    processRssMeanMB: Number(rssMeanMB.toFixed(2)),
    processRssMaxMB: Number(rssMaxMB.toFixed(2)),
    rssPerWorkerMaxMB: Number((rssMaxMB / workerCount).toFixed(2)),
    workerHeapPeakAvgMB: Number(
      (workerReports.reduce((a, r) => a + r.heapUsedPeakMB, 0) / workerReports.length).toFixed(2),
    ),
    workerHeapPeakMaxMB: Number(Math.max(...workerReports.map((r) => r.heapUsedPeakMB)).toFixed(2)),
    gcTotalEvents: workerReports.reduce((a, r) => a + r.gcCount, 0),
    gcTotalTimeMs: Number(workerReports.reduce((a, r) => a + r.gcTotalMs, 0).toFixed(2)),
    gcMaxPauseMs: Number(Math.max(...workerReports.map((r) => r.gcMaxMs)).toFixed(2)),
    gcMajorEvents: workerReports.reduce((a, r) => a + r.gcMajorCount, 0),
    gcMajorMaxPauseMs: Number(Math.max(...workerReports.map((r) => r.gcMajorMaxMs)).toFixed(2)),
    eldP50AvgMs: Number(
      (workerReports.reduce((a, r) => a + r.eldP50Ms, 0) / workerReports.length).toFixed(2),
    ),
    eldP95MaxMs: Number(Math.max(...workerReports.map((r) => r.eldP95Ms)).toFixed(2)),
    eldMaxPauseMs: Number(Math.max(...workerReports.map((r) => r.eldMaxMs)).toFixed(2)),
  };

  return {tier, workerReports};
}

function parseSecondsArg(): number {
  const idx = process.argv.indexOf('--seconds');
  if (idx !== -1 && process.argv[idx + 1]) {
    const val = Number(process.argv[idx + 1]);
    if (Number.isFinite(val) && val >= 1) {
      return val;
    }
  }
  return 60;
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

async function runCoordinator(): Promise<void> {
  const durationSec = parseSecondsArg();
  const cpus = os.cpus();
  const cpuModel = cpus[0]?.model ?? 'unknown';
  const logicalCores = cpus.length;
  const totalRamGB = Number((os.totalmem() / 1024 ** 3).toFixed(2));

  const workerCounts = Array.from(new Set([1, 2, 4, 8, logicalCores]))
    .filter((n) => n >= 1)
    .sort((a, b) => a - b);

  const resultsDir = path.resolve(__dirname, '../results');
  fs.mkdirSync(resultsDir, {recursive: true});

  console.log(
    `=== Stage 5: Scaling (worker_threads, ${durationSec}s/tier, canonical order, 4 maps rotating) ===`,
  );
  console.log(`Host: ${cpuModel} | logicalCores=${logicalCores} | RAM=${totalRamGB} GB | Node=${process.version}`);
  console.log(`Tiers: [${workerCounts.join(', ')}] workers\n`);

  const rawTiers: Array<{
    tier: Omit<ScalingTierResult, 'speedupVs1Worker' | 'scalingEfficiencyPct'>;
    workerReports: Array<WorkerReport>;
  }> = [];

  for (const wc of workerCounts) {
    console.log(`Running ${wc} worker(s) for ${durationSec}s...`);
    const res = await runWorkerTier(wc, durationSec);
    rawTiers.push(res);
    const t = res.tier;
    console.log(
      `  workers=${String(wc).padStart(2)}: games=${t.totalGamesCompleted} (failed=${t.totalGamesFailed}), ` +
        `rate=${t.gamesPerSec} games/s (${t.decisionsPerSec} dec/s), ` +
        `RSS mean=${t.processRssMeanMB}MB max=${t.processRssMaxMB}MB (${t.rssPerWorkerMaxMB}MB/worker), ` +
        `heapPeakMax=${t.workerHeapPeakMaxMB}MB, ` +
        `GC(total=${t.gcTotalEvents}, major=${t.gcMajorEvents}, maxPause=${t.gcMaxPauseMs}ms), ` +
        `ELD(p50Avg=${t.eldP50AvgMs}ms, p95Max=${t.eldP95MaxMs}ms, max=${t.eldMaxPauseMs}ms)`,
    );
  }

  const baselineRate = rawTiers.find((r) => r.tier.workers === 1)?.tier.gamesPerSec ?? rawTiers[0].tier.gamesPerSec;
  const tiers: Array<ScalingTierResult> = rawTiers.map(({tier}) => {
    const speedup = baselineRate > 0 ? tier.gamesPerSec / baselineRate : 0;
    const eff = (speedup / tier.workers) * 100;
    return {
      ...tier,
      speedupVs1Worker: Number(speedup.toFixed(2)),
      scalingEfficiencyPct: Number(eff.toFixed(1)),
    };
  });

  console.log('\n=== Stage 5 Scaling Summary ===');
  for (const t of tiers) {
    console.log(
      `  ${String(t.workers).padStart(2)}w: ${String(t.gamesPerSec).padStart(7)} games/s | ` +
        `speedup=${t.speedupVs1Worker}x (${t.scalingEfficiencyPct}% eff) | ` +
        `RSS max=${t.processRssMaxMB}MB | GC max=${t.gcMaxPauseMs}ms | ELD max=${t.eldMaxPauseMs}ms`,
    );
  }

  const payload = {
    generatedAt: new Date().toISOString(),
    nodeVersion: process.version,
    simultaneousOrder: 'canonical',
    concurrencyModel: 'worker_threads (isolated V8 isolates; module globals isolated per worker)',
    host: {
      cpuModel,
      logicalCores,
      totalRamGB,
      platform: os.platform(),
      release: os.release(),
    },
    durationSecPerTier: durationSec,
    tiers,
    perTierWorkerReports: rawTiers.map((r) => ({
      workers: r.tier.workers,
      reports: r.workerReports,
    })),
  };

  fs.writeFileSync(
    path.join(resultsDir, 'scaling.json'),
    JSON.stringify(payload, null, 2) + '\n',
    'utf8',
  );

  updateSummaryFile(resultsDir, {
    stage5_scaling: {
      generatedAt: payload.generatedAt,
      nodeVersion: payload.nodeVersion,
      simultaneousOrder: payload.simultaneousOrder,
      concurrencyModel: payload.concurrencyModel,
      host: payload.host,
      durationSecPerTier: durationSec,
      tiers,
    },
  });
}

if (isMainThread) {
  void runCoordinator();
} else {
  void runWorkerMain(workerData as WorkerInitData);
}
