import {
  AdapterBugError,
  SUPPORTED_MAPS,
  SupportedMap,
  playRandomGame,
} from '../src/index';

function parseCount(argv: ReadonlyArray<string>): number {
  const raw = argv.slice(2).find((arg) => !arg.startsWith('--'));
  if (raw === undefined) {
    return 1000;
  }
  const parsed = Number(raw);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`Invalid fuzz game count "${raw}": expected a positive integer`);
  }
  return parsed;
}

function main(): void {
  const totalGames = parseCount(process.argv);
  const observe = process.argv.includes('--observe');
  const startNs = process.hrtime.bigint();

  let completed = 0;
  let rejections = 0;
  let stepCaps = 0;
  let deadlocks = 0;
  let otherErrors = 0;
  let totalGenerations = 0;
  let totalAtomicChoices = 0;
  let totalObserveMs = 0;

  const perMapCompleted: Record<SupportedMap, number> = {
    'tharsis': 0,
    'elysium': 0,
    'hellas': 0,
    'vastitas borealis nova': 0,
  };
  const failures: Array<{seedKey: string; map: SupportedMap; message: string}> = [];

  for (let i = 0; i < totalGames; i++) {
    const map = SUPPORTED_MAPS[i % SUPPORTED_MAPS.length];
    const seedKey = `fuzz-${map}-${i}`;
    try {
      const outcome = playRandomGame({seedKey, map, observe});
      completed += 1;
      perMapCompleted[map] += 1;
      totalGenerations += outcome.result.generations;
      totalAtomicChoices += outcome.atomicChoices;
      totalObserveMs += outcome.observeMs ?? 0;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (err instanceof AdapterBugError) {
        if (/Step cap/i.test(msg)) {
          stepCaps += 1;
        } else if (/Deadlock/i.test(msg)) {
          deadlocks += 1;
        } else {
          rejections += 1;
        }
      } else {
        otherErrors += 1;
      }
      if (failures.length < 20) {
        failures.push({seedKey, map, message: msg});
      }
    }
  }

  const endNs = process.hrtime.bigint();
  const totalMs = Number(endNs - startNs) / 1e6;
  const meanMs = totalMs / totalGames;
  const meanGen = completed > 0 ? totalGenerations / completed : 0;
  const meanChoices = completed > 0 ? totalAtomicChoices / completed : 0;

  console.log(`=== Adapter Fuzz Report (${totalGames} games) ===`);
  console.log(`Completed:    ${completed}/${totalGames}`);
  console.log(`Rejections:   ${rejections}`);
  console.log(`Step caps:    ${stepCaps}`);
  console.log(`Deadlocks:    ${deadlocks}`);
  console.log(`Other errors: ${otherErrors}`);
  console.log(
    `Per map:      ${SUPPORTED_MAPS.map((m) => `${m}=${perMapCompleted[m]}`).join(', ')}`,
  );
  console.log(`Mean gens:    ${meanGen.toFixed(2)}`);
  console.log(`Mean choices: ${meanChoices.toFixed(1)}`);
  if (observe) {
    const meanObserve = totalAtomicChoices > 0 ? totalObserveMs / totalAtomicChoices : 0;
    console.log(`Observe:      ${meanObserve.toFixed(4)} ms/choice (observe + serialize)`);
  }
  console.log(`Total wall:   ${(totalMs / 1000).toFixed(2)}s (${meanMs.toFixed(2)} ms/game)`);

  if (failures.length > 0) {
    console.error('Sample failures:');
    for (const f of failures) {
      console.error(`  [${f.map}] ${f.seedKey}: ${f.message}`);
    }
    process.exitCode = 1;
  }
}

main();
