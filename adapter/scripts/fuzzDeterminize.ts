import assert from 'node:assert/strict';
import {
  AdapterGame,
  SUPPORTED_MAPS,
  SampledWorld,
  Seat,
  UnseenPool,
  _unsafeGetEngineGame,
  createGame,
  serializeObservation,
} from '../src/index';

function parseCount(argv: ReadonlyArray<string>): number {
  const raw = argv.slice(2).find((arg) => !arg.startsWith('--'));
  if (raw === undefined) {
    return 24;
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

function shuffleWithRng<T>(items: ReadonlyArray<T>, rng: () => number): Array<T> {
  const arr = items.slice();
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function sampleWorldFromObservation(
  pool: UnseenPool,
  rng: () => number,
): SampledWorld {
  const availableProjects = shuffleWithRng(pool.projects, rng);
  const availablePreludes = shuffleWithRng(pool.preludes, rng);
  const availableCorps = shuffleWithRng(pool.corporations, rng);
  const availablePossOpp = shuffleWithRng(pool.possibleOpponentHandProjects, rng);

  const takeProjects = (count: number): Array<string> => {
    assert.ok(availableProjects.length >= count, `Not enough unseen projects: need ${count}, have ${availableProjects.length}`);
    return availableProjects.splice(0, count);
  };

  let opponentDealtCorporations: Array<string> | undefined;
  let opponentDealtPreludes: Array<string> | undefined;
  let opponentDealtProjects: Array<string> | undefined;
  let opponentPickedCorporation: string | undefined;
  let opponentPreludes: Array<string> | undefined;
  let opponentDraftHand: Array<string> | undefined;
  let opponentDrafted: Array<string> | undefined;
  let opponentResearchOffer: Array<string> | undefined;

  if (!pool.opponentInitialBuyComplete) {
    opponentDealtCorporations = availableCorps.splice(0, 2);
    opponentDealtPreludes = availablePreludes.splice(0, 4);
    opponentDealtProjects = takeProjects(10);
  } else if (pool.opponentHasHiddenPickedCorp) {
    opponentPickedCorporation = shuffleWithRng(pool.opponentPickedCorporationCandidates, rng)[0];
  }

  if (pool.opponentPreludeHandSize > 0) {
    opponentPreludes = availablePreludes.splice(0, pool.opponentPreludeHandSize);
  }

  if (pool.opponentDraftHandSize > 0 || pool.opponentDraftedSize > 0) {
    const needDraftHandUnseen = Math.max(0, pool.opponentDraftHandSize - pool.knownOpponentDraftHandProjects.length);
    const needDraftedUnseen = Math.max(0, pool.opponentDraftedSize - pool.knownOpponentDraftedProjects.length);
    opponentDraftHand = [...pool.knownOpponentDraftHandProjects, ...takeProjects(needDraftHandUnseen)];
    opponentDrafted = [...pool.knownOpponentDraftedProjects, ...takeProjects(needDraftedUnseen)];
  }

  if (pool.opponentResearchOfferSize > 0) {
    const needResUnseen = Math.max(0, pool.opponentResearchOfferSize - pool.knownOpponentResearchProjects.length);
    opponentResearchOffer = [...pool.knownOpponentResearchProjects, ...takeProjects(needResUnseen)];
  }

  const opponentHand: Array<string> = [...pool.knownOpponentHandProjects];
  const minPossNeeded = Math.max(
    0,
    pool.knownProjectDiscardPile.length + availablePossOpp.length - pool.discardPileSize,
  );
  const maxPossAllowed = Math.min(availablePossOpp.length, pool.opponentHandSize - opponentHand.length);
  const takePoss = Math.min(maxPossAllowed, Math.max(minPossNeeded, Math.floor(rng() * (maxPossAllowed + 1))));
  opponentHand.push(...availablePossOpp.splice(0, takePoss));

  const remainingOppHand = pool.opponentHandSize - opponentHand.length;
  assert.ok(remainingOppHand >= 0, 'opponentHand exceeded opponentHandSize');
  opponentHand.push(...takeProjects(remainingOppHand));

  const projectDeck = takeProjects(pool.deckSize);

  return {
    opponentHand,
    projectDeck,
    ...(opponentDealtCorporations !== undefined ? {opponentDealtCorporations} : {}),
    ...(opponentDealtPreludes !== undefined ? {opponentDealtPreludes} : {}),
    ...(opponentDealtProjects !== undefined ? {opponentDealtProjects} : {}),
    ...(opponentPickedCorporation !== undefined ? {opponentPickedCorporation} : {}),
    ...(opponentPreludes !== undefined ? {opponentPreludes} : {}),
    ...(opponentDraftHand !== undefined ? {opponentDraftHand} : {}),
    ...(opponentDrafted !== undefined ? {opponentDrafted} : {}),
    ...(opponentResearchOffer !== undefined ? {opponentResearchOffer} : {}),
  };
}

function isResearchPaymentWindow(game: AdapterGame, seat: Seat): boolean {
  const e = _unsafeGetEngineGame(game);
  return e.phase === 'research' && e.generation >= 2 && e.players[seat].getWaitingFor()?.type === 'payment';
}

function censusDuplicates(game: AdapterGame): Array<string> {
  const e = _unsafeGetEngineGame(game) as unknown as {
    phase: string;
    generation: number;
    projectDeck: {drawPile: Array<{name: string}>; discardPile: Array<{name: string}>};
    corporationDeck: {drawPile: Array<{name: string}>; discardPile: Array<{name: string}>};
    preludeDeck: {drawPile: Array<{name: string}>; discardPile: Array<{name: string}>};
    players: Array<{
      cardsInHand: Array<{name: string}>;
      preludeCardsInHand: Array<{name: string}>;
      playedCards: Iterable<{name: string; type?: string}>;
      pickedCorporationCard?: {name: string};
      draftedCards: Array<{name: string}>;
      draftHand: Array<{name: string}>;
      dealtCorporationCards: Array<{name: string}>;
      dealtPreludeCards: Array<{name: string}>;
      dealtProjectCards: Array<{name: string}>;
      getWaitingFor(): {type?: string; cards?: ReadonlyArray<{name: string; type?: string}>} | undefined;
    }>;
  };
  const projects = new Map<string, number>();
  const corps = new Map<string, number>();
  const preludes = new Map<string, number>();
  const bump = (map: Map<string, number>, cards: ReadonlyArray<{name: string}> | undefined) => {
    for (const c of cards ?? []) {
      map.set(c.name, (map.get(c.name) ?? 0) + 1);
    }
  };
  bump(projects, e.projectDeck.drawPile);
  bump(projects, e.projectDeck.discardPile);
  bump(corps, e.corporationDeck.drawPile);
  bump(corps, e.corporationDeck.discardPile);
  bump(preludes, e.preludeDeck.drawPile);
  bump(preludes, e.preludeDeck.discardPile);

  for (const p of e.players) {
    bump(projects, p.cardsInHand);
    bump(projects, p.draftedCards);
    bump(projects, p.draftHand);
    bump(preludes, p.preludeCardsInHand);

    const played = Array.from(p.playedCards ?? []);
    const playedCorps = played.filter((c) => c.type === 'corporation');
    const playedPreludes = played.filter((c) => c.type === 'prelude');
    const playedProjects = played.filter((c) => c.type !== 'corporation' && c.type !== 'prelude');
    bump(projects, playedProjects);
    bump(corps, playedCorps);
    bump(preludes, playedPreludes);

    const initialBuyComplete = p.pickedCorporationCard !== undefined || playedCorps.length > 0;
    if (!initialBuyComplete) {
      bump(projects, p.dealtProjectCards);
      bump(corps, p.dealtCorporationCards);
      bump(preludes, p.dealtPreludeCards);
    }
    if (p.pickedCorporationCard !== undefined && !playedCorps.some((c) => c.name === p.pickedCorporationCard?.name)) {
      bump(corps, [p.pickedCorporationCard]);
    }
    const wf = p.getWaitingFor();
    if (wf !== undefined && wf.type === 'card' && wf.cards !== undefined) {
      if (e.phase === 'research' && e.generation >= 2) {
        bump(projects, wf.cards);
      } else if (e.phase === 'action' || e.phase === 'preludes') {
        const knownHere = new Set<string>(p.cardsInHand.map((c) => c.name));
        for (const pl of e.players) {
          for (const pc of Array.from(pl.playedCards ?? [])) knownHere.add(pc.name);
        }
        const wfProj = wf.cards.filter((c) => c.type !== 'corporation' && c.type !== 'prelude');
        if (wfProj.some((c) => !knownHere.has(c.name))) {
          bump(projects, wfProj.filter((c) => !knownHere.has(c.name)));
        }
      }
    }
  }

  const dups: Array<string> = [];
  for (const [label, map] of [['project', projects], ['corp', corps], ['prelude', preludes]] as const) {
    for (const [name, count] of map) {
      if (count > 1) dups.push(`${label}:${name}x${count}`);
    }
  }
  return dups;
}

function extractTrueWorld(game: AdapterGame, actor: Seat): SampledWorld {
  const e = _unsafeGetEngineGame(game);
  const obs = game.observe(actor);
  const pool = game.unseenPool(actor);
  const opp: Seat = actor === 0 ? 1 : 0;
  const oppPlayer = e.players[opp];

  let opponentDealtCorporations: Array<string> | undefined;
  let opponentDealtPreludes: Array<string> | undefined;
  let opponentDealtProjects: Array<string> | undefined;
  let opponentPickedCorporation: string | undefined;
  let opponentPreludes: Array<string> | undefined;
  let opponentDraftHand: Array<string> | undefined;
  let opponentDrafted: Array<string> | undefined;
  let opponentResearchOffer: Array<string> | undefined;

  if (!pool.opponentInitialBuyComplete) {
    opponentDealtCorporations = oppPlayer.dealtCorporationCards.map((c) => c.name);
    opponentDealtPreludes = oppPlayer.dealtPreludeCards.map((c) => c.name);
    opponentDealtProjects = oppPlayer.dealtProjectCards.map((c) => c.name);
  } else if (pool.opponentHasHiddenPickedCorp) {
    opponentPickedCorporation = oppPlayer.pickedCorporationCard?.name;
  }

  if (pool.opponentPreludeHandSize > 0) {
    opponentPreludes = oppPlayer.preludeCardsInHand.map((c) => c.name);
  }

  if (e.phase === 'drafting' && (pool.opponentDraftHandSize > 0 || pool.opponentDraftedSize > 0)) {
    if (oppPlayer.needsToDraft === false) {
      const oppHasMarsMaths = obs.public.players[opp].tableau.some((tc) => tc.name === 'Mars Maths');
      const keepCount = e.draftRound === 1 && oppHasMarsMaths ? 2 : 1;
      opponentDraftHand = [
        ...oppPlayer.draftHand.map((c) => c.name),
        ...oppPlayer.draftedCards.slice(-keepCount).map((c) => c.name),
      ];
      opponentDrafted = oppPlayer.draftedCards.slice(0, oppPlayer.draftedCards.length - keepCount).map((c) => c.name);
    } else {
      opponentDraftHand = oppPlayer.draftHand.map((c) => c.name);
      opponentDrafted = oppPlayer.draftedCards.map((c) => c.name);
    }
  }

  if (e.phase === 'research' && e.generation >= 2 && pool.opponentResearchOfferSize > 0) {
    const wf = oppPlayer.getWaitingFor();
    if (wf !== undefined && wf.type === 'card') {
      opponentResearchOffer = (wf as unknown as {cards: ReadonlyArray<{name: string}>}).cards.map((c) => c.name);
    }
  }

  return {
    opponentHand: oppPlayer.cardsInHand.map((c) => c.name),
    projectDeck: e.projectDeck.drawPile.map((c) => c.name),
    projectDiscardPile: e.projectDeck.discardPile.map((c) => c.name),
    ...(opponentDealtCorporations !== undefined ? {opponentDealtCorporations} : {}),
    ...(opponentDealtPreludes !== undefined ? {opponentDealtPreludes} : {}),
    ...(opponentDealtProjects !== undefined ? {opponentDealtProjects} : {}),
    ...(opponentPickedCorporation !== undefined ? {opponentPickedCorporation} : {}),
    ...(opponentPreludes !== undefined ? {opponentPreludes} : {}),
    ...(opponentDraftHand !== undefined ? {opponentDraftHand} : {}),
    ...(opponentDrafted !== undefined ? {opponentDrafted} : {}),
    ...(opponentResearchOffer !== undefined ? {opponentResearchOffer} : {}),
  };
}

function playToCompletion(game: AdapterGame, rng: () => number): void {
  while (true) {
    const seats = game.pending();
    if (seats.length === 0) break;
    const seat = seats[0];
    const choice = game.choice(seat);
    game.choose(seat, Math.floor(rng() * choice.candidates.length));
  }
}

function main(): void {
  const totalGames = parseCount(process.argv);
  const startNs = process.hrtime.bigint();
  let truthChecks = 0;
  let sampledRollouts = 0;

  for (let gi = 0; gi < totalGames; gi++) {
    const map = SUPPORTED_MAPS[gi % SUPPORTED_MAPS.length];
    const g = createGame({seedKey: `t28-truth-${gi}`, map});
    const mainRng = makePrng(9000 + gi * 17);
    const sampleRng = makePrng(19000 + gi * 29);
    let decision = 0;

    while (true) {
      const pending = g.pending();
      if (pending.length === 0) break;

      if (decision % 3 === 0) {
        for (const actor of pending) {
          const opp: Seat = actor === 0 ? 1 : 0;
          if (!isResearchPaymentWindow(g, opp)) {
            const cTruth = g.clone();
            const obsBefore = serializeObservation(cTruth.observe(actor));
            const trueWorld = extractTrueWorld(cTruth, actor);
            cTruth.determinize(actor, trueWorld);
            assert.equal(
              serializeObservation(cTruth.observe(actor)),
              obsBefore,
              `truth determinize altered observe(${actor}) in game ${gi} decision ${decision}`,
            );
            assert.deepEqual(
              censusDuplicates(cTruth),
              [],
              `truth determinize duplicates in game ${gi} decision ${decision}`,
            );
            truthChecks += 1;
          }

          const cSample = g.clone();
          const pool = cSample.unseenPool(actor);
          const sampledWorld = sampleWorldFromObservation(pool, sampleRng);
          cSample.determinize(actor, sampledWorld);
          assert.deepEqual(
            censusDuplicates(cSample),
            [],
            `sampled determinize duplicates in game ${gi} decision ${decision}`,
          );
          playToCompletion(cSample, makePrng(29000 + gi * 1000 + decision + actor));
          assert.ok(cSample.result() !== null, `sampled rollout did not finish in game ${gi} decision ${decision}`);
          assert.deepEqual(
            censusDuplicates(cSample),
            [],
            `sampled rollout duplicates in game ${gi} decision ${decision}`,
          );
          sampledRollouts += 1;
        }
      }

      const seat = pending[0];
      const choice = g.choice(seat);
      g.choose(seat, Math.floor(mainRng() * choice.candidates.length));
      decision += 1;
    }
  }

  const endNs = process.hrtime.bigint();
  const totalSec = Number(endNs - startNs) / 1e9;
  console.log(`=== Determinize Truth & Rollout Sweep (${totalGames} games) ===`);
  console.log(`Truth checks:     ${truthChecks} (0 rejections, 0 duplicates)`);
  console.log(`Sampled rollouts: ${sampledRollouts} (all completed, 0 duplicates)`);
  console.log(`Total wall:       ${totalSec.toFixed(2)}s`);
}

main();
