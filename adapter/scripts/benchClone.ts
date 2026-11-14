import {
  AdapterGame,
  SUPPORTED_MAPS,
  SupportedMap,
  _unsafeGetEngineGame,
  createGame,
} from '../src/index';

function parseCount(argv: ReadonlyArray<string>): number {
  const raw = argv.slice(2).find((arg) => !arg.startsWith('--'));
  if (raw === undefined) {
    return 20;
  }
  const parsed = Number(raw);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`Invalid game count "${raw}": expected a positive integer`);
  }
  return parsed;
}

function makePrng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return s / 0x1_0000_0000;
  };
}

function percentile(sorted: ReadonlyArray<number>, p: number): number {
  if (sorted.length === 0) {
    return 0;
  }
  const idx = Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * p));
  return sorted[idx];
}

function summarize(samples: ReadonlyArray<number>): {
  count: number;
  mean: number;
  p50: number;
  p95: number;
  p99: number;
  max: number;
} {
  if (samples.length === 0) {
    return {count: 0, mean: 0, p50: 0, p95: 0, p99: 0, max: 0};
  }
  const sorted = samples.slice().sort((a, b) => a - b);
  let sum = 0;
  for (const v of sorted) {
    sum += v;
  }
  return {
    count: sorted.length,
    mean: sum / sorted.length,
    p50: percentile(sorted, 0.5),
    p95: percentile(sorted, 0.95),
    p99: percentile(sorted, 0.99),
    max: sorted[sorted.length - 1],
  };
}

function warmup(): void {
  const g = createGame({seedKey: 'bench-warmup', map: 'tharsis'});
  const rng = makePrng(1);
  while (true) {
    const seats = g.pending();
    if (seats.length === 0) {
      break;
    }
    const c = g.clone();
    void c.pending();
    const s = seats[0];
    const ch = g.choice(s);
    g.choose(s, Math.floor(rng() * ch.candidates.length));
  }
}

function runBenchGame(
  seedKey: string,
  map: SupportedMap,
  seed: number,
  allSamples: Array<number>,
  byPhase: Map<string, Array<number>>,
): number {
  const game: AdapterGame = createGame({seedKey, map});
  const rng = makePrng(seed);
  let decisions = 0;

  while (true) {
    const seats = game.pending();
    if (seats.length === 0) {
      break;
    }
    const phase = _unsafeGetEngineGame(game).phase;
    const t0 = process.hrtime.bigint();
    const cloned = game.clone();
    const t1 = process.hrtime.bigint();
    const ms = Number(t1 - t0) / 1e6;

    // Prevent dead-code elimination while verifying minimal clone sanity.
    if (cloned.pending().length !== seats.length) {
      throw new Error(`Clone pending mismatch in ${seedKey}`);
    }

    allSamples.push(ms);
    let phaseBucket = byPhase.get(phase);
    if (phaseBucket === undefined) {
      phaseBucket = [];
      byPhase.set(phase, phaseBucket);
    }
    phaseBucket.push(ms);
    decisions += 1;

    const s = seats[0];
    const ch = game.choice(s);
    game.choose(s, Math.floor(rng() * ch.candidates.length));
  }

  return decisions;
}

function main(): void {
  const totalGames = parseCount(process.argv);
  warmup();

  const allSamples: Array<number> = [];
  const byPhase = new Map<string, Array<number>>();
  const wallStart = process.hrtime.bigint();
  let totalDecisions = 0;

  for (let i = 0; i < totalGames; i++) {
    const map = SUPPORTED_MAPS[i % SUPPORTED_MAPS.length];
    totalDecisions += runBenchGame(
      `bench-clone-${map}-${i}`,
      map,
      7000 + i * 31,
      allSamples,
      byPhase,
    );
  }

  const wallEnd = process.hrtime.bigint();
  const wallSec = Number(wallEnd - wallStart) / 1e9;
  const overall = summarize(allSamples);

  console.log(`=== Adapter Clone Latency Benchmark (${totalGames} games, ${totalDecisions} decisions) ===`);
  console.log(
    `Overall: mean=${overall.mean.toFixed(3)} ms, median(p50)=${overall.p50.toFixed(3)} ms, p95=${overall.p95.toFixed(3)} ms, p99=${overall.p99.toFixed(3)} ms, max=${overall.max.toFixed(3)} ms`,
  );
  for (const [phase, samples] of byPhase.entries()) {
    const stats = summarize(samples);
    console.log(
      `  phase=${phase.padEnd(11)} n=${String(stats.count).padStart(5)} mean=${stats.mean.toFixed(3)} ms  p50=${stats.p50.toFixed(3)} ms  p95=${stats.p95.toFixed(3)} ms  max=${stats.max.toFixed(3)} ms`,
    );
  }
  console.log(`Total wall (including rollout + clone): ${wallSec.toFixed(2)}s`);
}

main();
