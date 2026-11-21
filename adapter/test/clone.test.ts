import {describe, it} from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import {
  AdapterGame,
  ActorObservation,
  SUPPORTED_MAPS,
  SampledWorld,
  Seat,
  UnseenPool,
  computeUnseenPool,
  createGame,
  sampleWorldFromObservation,
  serializeObservation,
  _unsafeGetEngineGame,
} from '../src/index';

function makePrng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return s / 0x1_0000_0000;
  };
}

function chooseSeeded(rng: () => number, count: number): number {
  return Math.floor(rng() * count);
}

function playToCompletion(game: AdapterGame, rng: () => number): Array<string> {
  const trace: Array<string> = [];
  while (true) {
    const seats = game.pending();
    if (seats.length === 0) {
      break;
    }
    const seat = seats[0];
    const choice = game.choice(seat);
    const idx = chooseSeeded(rng, choice.candidates.length);
    trace.push(`${seat}:${choice.kind}:${choice.candidates[idx].id}`);
    game.choose(seat, idx);
  }
  return trace;
}

function assertCloneMatchesAndContinuesIdentically(orig: AdapterGame, label: string, seed = 42): void {
  const clone = orig.clone();
  assert.notEqual(clone, orig, `${label}: clone must be a distinct instance`);
  assert.notEqual(
    _unsafeGetEngineGame(clone),
    _unsafeGetEngineGame(orig),
    `${label}: clone must not share engine instance`,
  );

  assert.deepEqual(clone.pending(), orig.pending(), `${label}: pending() mismatch`);
  assert.deepEqual(clone.result(), orig.result(), `${label}: result() mismatch`);
  assert.equal(
    serializeObservation(clone.observe(0)),
    serializeObservation(orig.observe(0)),
    `${label}: observe(0) mismatch`,
  );
  assert.equal(
    serializeObservation(clone.observe(1)),
    serializeObservation(orig.observe(1)),
    `${label}: observe(1) mismatch`,
  );

  for (const seat of orig.pending()) {
    assert.deepEqual(clone.choice(seat), orig.choice(seat), `${label}: choice(${seat}) mismatch`);
  }

  const origTrace = playToCompletion(orig, makePrng(seed));
  const cloneTrace = playToCompletion(clone, makePrng(seed));
  assert.deepEqual(cloneTrace, origTrace, `${label}: continuation trace diverged`);
  assert.deepEqual(clone.result(), orig.result(), `${label}: final result mismatch`);
  assert.equal(
    serializeObservation(clone.observe(0)),
    serializeObservation(orig.observe(0)),
    `${label}: final observe(0) mismatch`,
  );
  assert.equal(
    serializeObservation(clone.observe(1)),
    serializeObservation(orig.observe(1)),
    `${label}: final observe(1) mismatch`,
  );
}

function pickDoneOrZero(game: AdapterGame, seat: Seat): number {
  const cur = game.choice(seat);
  if (cur.kind === 'multi_card') {
    const doneIdx = cur.candidates.findIndex((c) => c.id === 'done');
    return doneIdx !== -1 ? doneIdx : 0;
  }
  return 0;
}

function finishInitialResearch(game: AdapterGame): void {
  for (const seat of [0, 1] as const) {
    game.choose(seat, 0); // corp
    game.choose(seat, 0); // prelude 1
    game.choose(seat, 0); // prelude 2
    const donePrelIdx = game.choice(seat).candidates.findIndex((c) => c.id === 'done');
    game.choose(seat, donePrelIdx);
    game.choose(seat, 0); // 1 project card
    const doneProjIdx = game.choice(seat).candidates.findIndex((c) => c.id === 'done');
    game.choose(seat, doneProjIdx);
  }
}

function advanceUntil(
  game: AdapterGame,
  predicate: (g: AdapterGame) => boolean,
  rng: () => number,
  maxDecisions = 3000,
): boolean {
  let steps = 0;
  while (steps < maxDecisions) {
    if (predicate(game)) {
      return true;
    }
    const seats = game.pending();
    if (seats.length === 0) {
      return false;
    }
    const seat = seats[0];
    const choice = game.choice(seat);
    const idx = chooseSeeded(rng, choice.candidates.length);
    game.choose(seat, idx);
    steps += 1;
  }
  return false;
}

function findGameAt(
  predicate: (g: AdapterGame) => boolean,
  seedPrefix: string,
  map: (typeof SUPPORTED_MAPS)[number] = 'tharsis',
  maxGames = 40,
): AdapterGame {
  for (let i = 0; i < maxGames; i++) {
    const game = createGame({seedKey: `${seedPrefix}-${i}`, map});
    const rng = makePrng(1000 + i * 7);
    if (advanceUntil(game, predicate, rng)) {
      return game;
    }
  }
  throw new Error(`Could not find state matching predicate for ${seedPrefix}`);
}

function shuffleWithRng<T>(items: ReadonlyArray<T>, rng: () => number): Array<T> {
  const arr = items.slice();
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

/**
 * Independent test oracle (the pre-#23 sampler, deliberately kept separate from production).
 * Pure sampler constructing a legal SampledWorld from ActorObservation + UnseenPool ONLY.
 */
function sampleWorldFromObservationLegacy(
  obs: ActorObservation,
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

  // Opponent hand must include knownOpponentHandProjects, plus enough possibleOpponentHandProjects
  // so remaining possibleOpponentHandProjects fit in discardPileSize along with knownProjectDiscardPile.
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

// #31: a stale belief makes the sampler throw only for some seeds (the opponent-hand split
// is seed-dependent), so sweep seeds instead of trying one.
function assertSamplesEverySeed(obs: ActorObservation, seeds = 64): void {
  for (let seed = 0; seed < seeds; seed++) {
    const world = sampleWorldFromObservation(obs, seed);
    assert.equal(world.projectDeck.length, obs.public.deckSize, `deck size, seed ${seed}`);
    assert.equal(
      world.opponentHand.length,
      obs.public.players[obs.actor === 0 ? 1 : 0].cardsInHandCount,
      `opponent hand size, seed ${seed}`,
    );
  }
}

describe('Ticket 16: clone(), unseenPool(), determinize()', () => {
  describe('1. clone() prompt/continuation equivalence across all decision categories', () => {
    it('clones identically at initial setup root and mid multi_card selection', () => {
      // Root corp pick
      const g1 = createGame({seedKey: 'clone-init-root', map: 'tharsis'});
      assertCloneMatchesAndContinuesIdentically(g1, 'initial root');

      // Mid-selection of prelude multi_card (1 of 2 selected) and project multi_card
      const g2 = createGame({seedKey: 'clone-init-mid', map: 'elysium'});
      g2.choose(0, 0); // corp picked -> now prelude multi_card
      g2.choose(0, 0); // 1st prelude picked -> mid multi_card
      assert.equal(g2.choice(0).kind, 'multi_card');
      assertCloneMatchesAndContinuesIdentically(g2, 'initial prelude mid-multi_card');

      const g3 = createGame({seedKey: 'clone-init-proj-mid', map: 'hellas'});
      g3.choose(0, 0); // corp
      g3.choose(0, 0); // prel 1
      g3.choose(0, 0); // prel 2
      g3.choose(0, g3.choice(0).candidates.findIndex((c) => c.id === 'done'));
      assert.equal(g3.choice(0).kind, 'multi_card');
      g3.choose(0, 0); // 1 project selected, still mid multi_card
      assertCloneMatchesAndContinuesIdentically(g3, 'initial project mid-multi_card');
    });

    it('clones identically with a deferred seat-1 initial answer and a deferred seat-1 drafting answer', () => {
      // Deferred seat-1 in initial setup
      const gInit = createGame({seedKey: 'clone-deferred-init', map: 'tharsis'});
      assert.deepEqual(gInit.pending(), [0, 1]);
      // Seat 1 finishes initial selection first (deferred by adapter)
      while (gInit.pending().includes(1)) {
        gInit.choose(1, pickDoneOrZero(gInit, 1));
      }
      assert.deepEqual(gInit.pending(), [0]);
      assertCloneMatchesAndContinuesIdentically(gInit, 'deferred seat-1 initial');

      // Deferred seat-1 in Gen 2 drafting
      const gDraft = findGameAt(
        (g) => _unsafeGetEngineGame(g).phase === 'drafting',
        'clone-draft-defer',
        'elysium',
      );
      assert.deepEqual(gDraft.pending(), [0, 1]);
      gDraft.choose(1, 0); // deferred while seat 0 still pending
      assert.deepEqual(gDraft.pending(), [0]);
      assertCloneMatchesAndContinuesIdentically(gDraft, 'deferred seat-1 drafting');

      // Non-deferred in Gen 2 drafting round 2 (after seat 0 picked R2 first)
      const gDraftR2 = findGameAt(
        (g) => {
          const eng = _unsafeGetEngineGame(g);
          return eng.phase === 'drafting' && eng.draftRound === 2 && g.pending()[0] === 0;
        },
        'clone-draft-r2',
        'hellas',
      );
      gDraftR2.choose(0, 0); // seat 0 submits R2 first, leaving seat 1 pending
      assert.deepEqual(gDraftR2.pending(), [1]);
      assertCloneMatchesAndContinuesIdentically(gDraftR2, 'drafting R2 after seat 0');
    });

    it('clones identically across preludes, root action, research, and final greenery', () => {
      // Preludes root
      const gPrel = findGameAt(
        (g) => _unsafeGetEngineGame(g).phase === 'preludes' && g.choice(g.pending()[0]).kind === 'card',
        'clone-prelude-root',
        'tharsis',
      );
      assertCloneMatchesAndContinuesIdentically(gPrel, 'prelude root');

      // Gen >= 2 Research (root + mid multi_card)
      const gRes = findGameAt(
        (g) => {
          const eng = _unsafeGetEngineGame(g);
          return eng.phase === 'research' && eng.generation >= 2 && g.choice(g.pending()[0]).kind === 'multi_card';
        },
        'clone-research',
        'vastitas borealis nova',
      );
      const gResMid = gRes.clone();
      assertCloneMatchesAndContinuesIdentically(gRes, 'gen>=2 research root');

      const resSeat = gResMid.pending()[0];
      if (gResMid.choice(resSeat).candidates.length > 1) {
        gResMid.choose(resSeat, 0); // pick 1 card, still in research multi_card or payment
        assertCloneMatchesAndContinuesIdentically(gResMid, 'gen>=2 research mid-selection');
      }

      // Root action
      const gAct = findGameAt(
        (g) => _unsafeGetEngineGame(g).phase === 'action' && g.choice(g.pending()[0]).kind === 'branch',
        'clone-action-root',
        'tharsis',
      );
      assertCloneMatchesAndContinuesIdentically(gAct, 'action root');

      // Final greenery production:branch and production:space
      const gProd = findGameAt(
        (g) => _unsafeGetEngineGame(g).phase === 'production' && _unsafeGetEngineGame(g).deferredActions.length === 0,
        'clone-final-greenery',
        'elysium',
      );
      const gEnd = gProd.clone();
      assertCloneMatchesAndContinuesIdentically(gProd, 'final greenery branch');

      // Phase.END
      playToCompletion(gEnd, makePrng(99));
      assert.deepEqual(gEnd.pending(), []);
      const endClone = gEnd.clone();
      assert.deepEqual(endClone.pending(), []);
      assert.deepEqual(endClone.result(), gEnd.result());
      assert.equal(serializeObservation(endClone.observe(0)), serializeObservation(gEnd.observe(0)));
      assert.equal(serializeObservation(endClone.observe(1)), serializeObservation(gEnd.observe(1)));
    });

    it('clones identically at nested card/payment/space/player/amount/branch prompts and action multi_card', () => {
      const targetKinds = new Set(['card', 'project_card', 'payment', 'space', 'player', 'amount']);
      const captured = new Map<string, AdapterGame>();

      for (let i = 0; i < 15 && captured.size < targetKinds.size; i++) {
        const g = createGame({seedKey: `clone-nested-${i}`, map: SUPPORTED_MAPS[i % SUPPORTED_MAPS.length]});
        const rng = makePrng(500 + i);
        while (true) {
          const seats = g.pending();
          if (seats.length === 0) break;
          const seat = seats[0];
          const ch = g.choice(seat);
          const eng = _unsafeGetEngineGame(g);
          if (eng.phase === 'action' && targetKinds.has(ch.kind) && !captured.has(ch.kind)) {
            captured.set(ch.kind, g.clone());
          }
          g.choose(seat, chooseSeeded(rng, ch.candidates.length));
        }
      }

      for (const kind of targetKinds) {
        const snap = captured.get(kind);
        assert.ok(snap !== undefined, `Expected to encounter action:${kind} in seed sweep`);
        assertCloneMatchesAndContinuesIdentically(snap, `action:${kind}`);
      }

      // Explicitly verify Sell Patents multi_card mid-selection and Mars University nested branch
      const engineRoot = path.resolve(__dirname, '../../../tools/simbench/engine/build/src');
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const {MarsUniversity} = require(path.join(engineRoot, 'server/cards/base/MarsUniversity.js'));
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const {Research} = require(path.join(engineRoot, 'server/cards/base/Research.js'));

      // Buy 2 project cards during initial research so Sell Patents is available in Phase.ACTION
      const gSetup = createGame({seedKey: 'clone-sell-patents', map: 'tharsis'});
      for (const seat of [0, 1] as const) {
        gSetup.choose(seat, 1); // non-first-action corp
        gSetup.choose(seat, 0); // prelude 1
        gSetup.choose(seat, 0); // prelude 2
        gSetup.choose(seat, gSetup.choice(seat).candidates.findIndex((c) => c.id === 'done'));
        gSetup.choose(seat, 0); // project 1
        gSetup.choose(seat, 0); // project 2
        gSetup.choose(seat, gSetup.choice(seat).candidates.findIndex((c) => c.id === 'done'));
      }
      // Advance preludes up to the last prelude step before Phase.ACTION starts, and inject Research + MarsUniversity
      // on seat 0 right before Phase.ACTION builds OrOptions.
      let injected = false;
      const prelRng = makePrng(12);
      while (_unsafeGetEngineGame(gSetup).phase !== 'action') {
        const seats = gSetup.pending();
        if (seats.length === 0) break;
        const s = seats[0];
        if (!injected && _unsafeGetEngineGame(gSetup).phase === 'preludes') {
          const engSetup = _unsafeGetEngineGame(gSetup);
          // Inject onto active player's hand/playedCards before takeAction constructs OrOptions
          (engSetup.players[0] as unknown as {playedCards: {push(c: unknown): void}; megaCredits: number}).playedCards.push(
            new MarsUniversity(),
          );
          (engSetup.players[0] as unknown as {megaCredits: number}).megaCredits = 50;
          engSetup.players[0].cardsInHand.push(new Research());
          injected = true;
        }
        gSetup.choose(s, chooseSeeded(prelRng, gSetup.choice(s).candidates.length));
      }

      const sellSeat = gSetup.pending()[0];
      assert.equal(sellSeat, 0);
      const gBranch = gSetup.clone();

      // Sell Patents mid multi_card on gSetup
      const sellIdx = gSetup.choice(sellSeat).candidates.findIndex((c) => /sell patents/i.test(c.label));
      assert.ok(sellIdx >= 0, 'Expected Sell patents action in OrOptions');
      gSetup.choose(sellSeat, sellIdx);
      assert.equal(gSetup.choice(sellSeat).kind, 'multi_card');
      gSetup.choose(sellSeat, 0); // select 1 card, stay in multi_card
      assert.equal(gSetup.choice(sellSeat).kind, 'multi_card');
      assertCloneMatchesAndContinuesIdentically(gSetup, 'action:multi_card mid-selection');

      // Nested branch via Mars University trigger on gBranch
      const playIdx = gBranch.choice(sellSeat).candidates.findIndex((c) => /play project card/i.test(c.label));
      assert.ok(playIdx >= 0, 'Expected Play project card action in OrOptions');
      gBranch.choose(sellSeat, playIdx);
      const cardPickIdx = gBranch.choice(sellSeat).candidates.findIndex((c) => c.id === 'Research');
      assert.ok(cardPickIdx >= 0, 'Expected Research in playable cards');
      gBranch.choose(sellSeat, cardPickIdx);
      if (gBranch.choice(sellSeat).kind === 'payment') {
        gBranch.choose(sellSeat, 0);
      }
      assert.equal(gBranch.choice(sellSeat).kind, 'branch');
      assertCloneMatchesAndContinuesIdentically(gBranch, 'nested action:branch');
    });
  });

  describe('2. Clone isolation across 20 decisions', () => {
    it('advancing a clone 20 decisions leaves original byte-identical (observations + engine.serialize)', () => {
      const canonicalEngineJson = (g: AdapterGame): string => {
        const serialized = _unsafeGetEngineGame(g).serialize() as unknown as {
          players: Array<{timer?: unknown}>;
        };
        for (const p of serialized.players) {
          delete p.timer;
        }
        return JSON.stringify(serialized);
      };

      const game = findGameAt(
        (g) => _unsafeGetEngineGame(g).phase === 'action',
        'clone-isolation',
        'tharsis',
      );
      // Advance partially into a card play if possible so suffix is non-empty too
      const seat = game.pending()[0];
      game.choose(seat, 0);

      const obs0Before = serializeObservation(game.observe(0));
      const obs1Before = serializeObservation(game.observe(1));
      const engineBefore = canonicalEngineJson(game);

      const clone = game.clone();
      const rng = makePrng(777);
      let advanced = 0;
      for (let i = 0; i < 20; i++) {
        const seats = clone.pending();
        if (seats.length === 0) break;
        const s = seats[0];
        clone.choose(s, chooseSeeded(rng, clone.choice(s).candidates.length));
        advanced += 1;
      }
      assert.equal(advanced, 20);

      assert.equal(serializeObservation(game.observe(0)), obs0Before);
      assert.equal(serializeObservation(game.observe(1)), obs1Before);
      assert.equal(canonicalEngineJson(game), engineBefore);
      assert.notEqual(serializeObservation(clone.observe(0)), obs0Before);
    });
  });

  describe('3. Unseen pool invariants', () => {
    it('is a pure function of observe(actor) and invariant when opponent hand or deck order changes', () => {
      const game = createGame({seedKey: 'pool-invariance-1', map: 'tharsis'});
      finishInitialResearch(game);
      advanceUntil(game, (g) => _unsafeGetEngineGame(g).phase === 'action', makePrng(31));

      const obs0 = game.observe(0);
      const pool0 = game.unseenPool(0);
      assert.deepEqual(pool0, computeUnseenPool(obs0));

      const eng = _unsafeGetEngineGame(game);
      assert.ok(eng.players[1].cardsInHand.length >= 1);
      assert.ok(eng.projectDeck.drawPile.length >= 5);

      const poolJsonBefore = JSON.stringify(pool0);

      // Reverse project deck order
      eng.projectDeck.drawPile.reverse();
      assert.equal(JSON.stringify(game.unseenPool(0)), poolJsonBefore);

      // Swap opponent hand card with unseen card from drawPile
      const origCard = eng.players[1].cardsInHand[0];
      const swapIdx = eng.projectDeck.drawPile.findIndex((c) => c.name !== origCard.name);
      assert.ok(swapIdx >= 0);
      const replacement = eng.projectDeck.drawPile[swapIdx];
      eng.players[1].cardsInHand[0] = replacement;
      eng.projectDeck.drawPile[swapIdx] = origCard;

      assert.equal(JSON.stringify(game.unseenPool(0)), poolJsonBefore);
    });

    it('tracks passed draft cards, forced opponent holdings, publicLog discards/draws, and reshuffles', () => {
      const g = findGameAt(
        (game) => _unsafeGetEngineGame(game).phase === 'drafting',
        'pool-draft-track',
        'hellas',
      );
      const r1Packet0 = g.choice(0).candidates.map((c) => c.id);
      const r1Pick0 = g.choice(0).candidates[0].id;
      const r1Passed0 = r1Packet0.filter((c) => c !== r1Pick0).sort();

      // Defer seat 1 -> pool(0) must stay identical
      const pool0BeforeDefer = JSON.stringify(g.unseenPool(0));
      g.choose(1, 0);
      assert.equal(JSON.stringify(g.unseenPool(0)), pool0BeforeDefer);
      g.choose(0, 0); // round 1 completes -> round 2 begins

      const pool0R2 = g.unseenPool(0);
      assert.deepEqual(pool0R2.knownOpponentDraftHandProjects, r1Passed0);
      for (const c of r1Packet0) {
        assert.equal(pool0R2.projects.includes(c), false, `Seen R1 card ${c} must not be in unseenPool.projects`);
      }

      // Pick in R2 -> advance to R3
      const r2Packet0 = g.choice(0).candidates.map((c) => c.id);
      const r2Pick0 = g.choice(0).candidates[0].id;
      const r2Passed0 = r2Packet0.filter((c) => c !== r2Pick0).sort();
      g.choose(0, 0);
      g.choose(1, 0);

      const r3Packet0 = g.choice(0).candidates.map((c) => c.id).sort();
      const expectedOppPick2 = r1Passed0.filter((c) => !r3Packet0.includes(c));
      assert.equal(expectedOppPick2.length, 1);

      const pool0R3 = g.unseenPool(0);
      assert.deepEqual(pool0R3.knownOpponentDraftHandProjects, r2Passed0);
      assert.deepEqual(pool0R3.knownOpponentDraftedProjects, expectedOppPick2);

      // Reshuffle check: move almost entire projectDeck into discardPile and trigger draw
      const eng = _unsafeGetEngineGame(g);
      const keep = eng.projectDeck.drawPile.splice(0, 1);
      eng.projectDeck.discardPile.push(...eng.projectDeck.drawPile.splice(0));
      eng.projectDeck.drawPile.push(...keep);
      const sawReshuffle = advanceUntil(
        g,
        (gm) => gm.observe(0).public.publicLog.some((e) => e.message.includes('discard pile has been shuffled')),
        makePrng(88),
      );
      assert.equal(sawReshuffle, true, 'Expected reshuffle log entry');
      const poolAfterReshuffle = g.unseenPool(0);
      assert.ok(poolAfterReshuffle.projects.length >= poolAfterReshuffle.deckSize);
      assert.equal(poolAfterReshuffle.knownProjectDiscardPile.length, 0);

      // Synthetic publicLog checks on computeUnseenPool for opponent public draw and public reveal+discard
      const baseObs = g.observe(0);
      const oppColor = baseObs.public.players[1].color;
      const basePool = computeUnseenPool(baseObs);
      assert.ok(basePool.projects.length >= 3);
      const [pubDrawCard, pubRevealCard, pubDiscardCard] = basePool.projects;
      const synthObs: ActorObservation = {
        ...baseObs,
        public: {
          ...baseObs.public,
          publicLog: [
            ...baseObs.public.publicLog,
            {
              type: 0,
              message: '${0} drew ${1}',
              data: [
                {type: 3, value: oppColor},
                {type: 1, value: pubDrawCard},
              ],
            },
            {
              type: 0,
              message: '${0} revealed and discarded ${1}',
              data: [
                {type: 3, value: oppColor},
                {type: 1, value: pubRevealCard},
              ],
            },
            {
              type: 0,
              message: 'Discarded ${0} cards ${1}',
              data: [
                {type: 0, value: '1'},
                {type: 1, value: pubDiscardCard},
              ],
            },
          ],
        },
      };
      const synthPool = computeUnseenPool(synthObs);
      assert.ok(
        synthPool.knownOpponentHandProjects.includes(pubDrawCard) ||
          synthPool.possibleOpponentHandProjects.includes(pubDrawCard),
        'Publicly drawn card should be tracked for opponent',
      );
      assert.equal(synthPool.projects.includes(pubDrawCard), false);
      assert.equal(synthPool.projects.includes(pubRevealCard), false);
      assert.equal(synthPool.projects.includes(pubDiscardCard), false);
      assert.equal(synthPool.knownProjectDiscardPile.includes(pubRevealCard), true);
      assert.equal(synthPool.knownProjectDiscardPile.includes(pubDiscardCard), true);
    });

    it('excludes pre-reshuffle unbought drafted cards when actor sells patents post-reshuffle', () => {
      // The actor must not have discarded from hand yet: the synthetic sale below is then
      // its only one, and the observation stays self-consistent.
      const actorDiscardedFromHand = (game: AdapterGame): boolean => {
        const obs = game.observe(0);
        return obs.public.publicLog.some(
          (e) =>
            e.data[0]?.value === obs.public.players[0].color &&
            (e.message === '${0} sold ${1} patents' ||
              e.message === '${0} is using their ${1} effect to draw a card by discarding a card.'),
        );
      };
      const g = findGameAt(
        (game) =>
          _unsafeGetEngineGame(game).phase === 'action' &&
          _unsafeGetEngineGame(game).generation >= 2 &&
          !actorDiscardedFromHand(game),
        'pool-draft-sell-reshuffle',
        'tharsis',
      );
      const baseObs = g.observe(0);
      const actorColor = baseObs.public.players[0].color;
      const basePool = computeUnseenPool(baseObs);
      assert.ok(basePool.projects.length >= 3);
      const [unbought1, unbought2, passedCard] = basePool.projects;
      const synthObs: ActorObservation = {
        ...baseObs,
        public: {
          ...baseObs.public,
          generation: 5,
          // The reshuffle moved the whole discard pile into the deck, so the belief is exactly
          // tight: every unseen card not in the opponent's hand must fit in the deck (#31).
          deckSize: baseObs.public.deckSize + baseObs.public.discardPileSize,
          discardPileSize: 0,
          publicLog: [
            ...baseObs.public.publicLog,
            {type: 0, message: 'Generation ${0}', data: [{type: 0, value: '4'}]},
            {type: 0, message: 'The project discard pile has been shuffled to form a new deck.', data: []},
            {type: 0, message: 'Generation ${0}', data: [{type: 0, value: '5'}]},
            {
              type: 0,
              message: '${0} sold ${1} patents',
              data: [
                {type: 3, value: actorColor},
                {type: 0, value: '1'},
              ],
            },
          ],
        },
        history: {
          ...baseObs.history,
          drafts: [
            ...baseObs.history.drafts,
            {
              generation: 2,
              draftRound: 1,
              packet: [unbought1, unbought2, passedCard],
              taken: [unbought1, unbought2],
              passed: [passedCard],
            },
          ],
          researchBuys: [
            ...baseObs.history.researchBuys,
            {
              generation: 2,
              offered: [unbought1, unbought2],
              bought: [],
              discarded: [unbought1, unbought2],
            },
          ],
        },
      };
      assertSamplesEverySeed(synthObs);
      const pool = computeUnseenPool(synthObs);
      assert.equal(
        pool.knownProjectDiscardPile.includes(unbought1),
        false,
        'Pre-reshuffle unbought drafted card must not be marked in post-reshuffle discard pile',
      );
      assert.equal(pool.knownProjectDiscardPile.includes(unbought2), false);
    });

    it('leaves Discarded N cards in unseen pool when a reshuffle occurs inside drawByConditionOrThrow', () => {
      const g = findGameAt(
        (game) => _unsafeGetEngineGame(game).phase === 'action' && _unsafeGetEngineGame(game).generation >= 2,
        'pool-discard-spans-reshuffle',
        'tharsis',
      );
      const baseObs = g.observe(0);
      const actorColor = baseObs.public.players[0].color;
      const basePool = computeUnseenPool(baseObs);
      assert.ok(basePool.projects.length >= 10);
      const discardedBatch = basePool.projects.slice(0, 10);
      const synthObs: ActorObservation = {
        ...baseObs,
        public: {
          ...baseObs.public,
          deckSize: basePool.deckSize + basePool.discardPileSize - 3,
          discardPileSize: 3,
          publicLog: [
            ...baseObs.public.publicLog,
            {type: 0, message: '${0} played ${1}', data: [{type: 2, value: actorColor}, {type: 3, value: 'Experimental Forest'}]},
            {type: 0, message: 'The project discard pile has been shuffled to form a new deck.', data: []},
            {
              type: 0,
              message: 'Discarded ${0} cards ${1}',
              data: [
                {type: 1, value: String(discardedBatch.length)},
                {type: 14, value: discardedBatch},
              ],
            },
          ],
        },
      };
      assertSamplesEverySeed(synthObs);
      const pool = computeUnseenPool(synthObs);
      for (const c of discardedBatch) {
        assert.equal(
          pool.knownProjectDiscardPile.includes(c),
          false,
          `Batch card ${c} spanning reshuffle must not be forced into knownProjectDiscardPile`,
        );
      }
    });
  });

  describe('4. determinize(actor, world)', () => {
    it('applies valid worlds preserving actor observation byte-for-byte and plays to completion', () => {
      // Test across: setup (both seat 0 and seat 1 after seat 0 picks), preludes, drafting R1/R2/R3,
      // research Gen>=2, action, and final greenery!
      const checkpoints: Array<{label: string; game: AdapterGame; actor: Seat}> = [];

      const gInit0 = createGame({seedKey: 'det-init-0', map: 'tharsis'});
      checkpoints.push({label: 'init seat 0', game: gInit0, actor: 0});

      const gInit0Defer = createGame({seedKey: 'det-init-defer', map: 'tharsis'});
      while (gInit0Defer.pending().includes(1)) {
        gInit0Defer.choose(1, pickDoneOrZero(gInit0Defer, 1));
      }
      checkpoints.push({label: 'init seat 0 with deferred seat 1', game: gInit0Defer, actor: 0});

      const gInit1After0 = createGame({seedKey: 'det-init-1-after-0', map: 'elysium'});
      while (gInit1After0.pending().includes(0)) {
        gInit1After0.choose(0, pickDoneOrZero(gInit1After0, 0));
      }
      checkpoints.push({label: 'init seat 1 after seat 0', game: gInit1After0, actor: 1});

      const gPrel = findGameAt((g) => _unsafeGetEngineGame(g).phase === 'preludes', 'det-prel', 'hellas');
      checkpoints.push({label: 'preludes', game: gPrel, actor: gPrel.pending()[0]});

      const gDraftR1 = findGameAt((g) => _unsafeGetEngineGame(g).phase === 'drafting' && _unsafeGetEngineGame(g).draftRound === 1, 'det-d1', 'tharsis');
      checkpoints.push({label: 'draft R1 seat 0', game: gDraftR1.clone(), actor: 0});
      gDraftR1.choose(0, 0); // seat 0 already picked in R1
      checkpoints.push({label: 'draft R1 seat 1 after seat 0 picked', game: gDraftR1, actor: 1});

      const gDraftR2 = findGameAt((g) => _unsafeGetEngineGame(g).phase === 'drafting' && _unsafeGetEngineGame(g).draftRound === 2, 'det-d2', 'elysium');
      checkpoints.push({label: 'draft R2', game: gDraftR2, actor: 0});

      const gDraftR3 = findGameAt((g) => _unsafeGetEngineGame(g).phase === 'drafting' && _unsafeGetEngineGame(g).draftRound === 3, 'det-d3', 'hellas');
      checkpoints.push({label: 'draft R3', game: gDraftR3, actor: 1});

      const gRes = findGameAt(
        (g) => _unsafeGetEngineGame(g).phase === 'research' && _unsafeGetEngineGame(g).generation >= 2,
        'det-res',
        'vastitas borealis nova',
      );
      checkpoints.push({label: 'gen>=2 research before opp buys', game: gRes.clone(), actor: 0});
      const rSeat = gRes.pending()[0];
      if (rSeat === 0) {
        gRes.choose(0, pickDoneOrZero(gRes, 0));
        if (_unsafeGetEngineGame(gRes).phase === 'research' && gRes.pending()[0] === 1) {
          checkpoints.push({label: 'gen>=2 research after seat 0 buys', game: gRes, actor: 1});
        }
      }

      const gAct = findGameAt(
        (g) => _unsafeGetEngineGame(g).phase === 'action' && _unsafeGetEngineGame(g).generation >= 2,
        'det-act',
        'tharsis',
      );
      checkpoints.push({label: 'action gen>=2', game: gAct, actor: gAct.pending()[0]});

      const gProd = findGameAt(
        (g) => _unsafeGetEngineGame(g).phase === 'production' && g.pending().length > 0,
        'det-prod',
        'elysium',
      );
      checkpoints.push({label: 'final greenery production', game: gProd, actor: gProd.pending()[0]});

      for (let i = 0; i < checkpoints.length; i++) {
        const {label, game, actor} = checkpoints[i];
        const clone = game.clone();
        const obsBefore = clone.observe(actor);
        const obsJsonBefore = serializeObservation(obsBefore);
        const pool = clone.unseenPool(actor);
        const world = sampleWorldFromObservationLegacy(obsBefore, pool, makePrng(900 + i));

        clone.determinize(actor, world);

        const obsJsonAfter = serializeObservation(clone.observe(actor));
        assert.equal(obsJsonAfter, obsJsonBefore, `${label}: observe(${actor}) changed after determinize`);

        // Verify world actually written into clone's engine state
        const opp = actor === 0 ? 1 : 0;
        const eng = _unsafeGetEngineGame(clone);
        assert.deepEqual(
          eng.players[opp].cardsInHand.map((c) => c.name).sort(),
          world.opponentHand.slice().sort(),
          `${label}: opponent hand not written`,
        );
        assert.deepEqual(
          eng.projectDeck.drawPile.map((c) => c.name),
          world.projectDeck,
          `${label}: projectDeck order not written`,
        );

        // Verify cloning a determinized game preserves the determinized world
        const cloneOfDet = clone.clone();
        assert.deepEqual(
          _unsafeGetEngineGame(cloneOfDet).projectDeck.drawPile.map((c) => c.name),
          world.projectDeck,
          `${label}: clone() of determinized game must keep determinized deck`,
        );

        // Play determinized clone all the way to legal completion
        playToCompletion(clone, makePrng(1200 + i));
        assert.ok(clone.result() !== null, `${label}: determinized clone failed to finish cleanly`);
      }
    });

    it('rejects inconsistent sampled worlds with clear errors', () => {
      const gAct = findGameAt(
        (g) => _unsafeGetEngineGame(g).phase === 'action' && _unsafeGetEngineGame(g).generation >= 2,
        'det-reject-act',
        'tharsis',
      );
      const actor = gAct.pending()[0];
      const obs = gAct.observe(actor);
      const pool = gAct.unseenPool(actor);
      const baseWorld = sampleWorldFromObservationLegacy(obs, pool, makePrng(42));

      // 1. Card outside unseen pool (card from actor's hand or tableau)
      const knownActorCard = obs.public.players[actor].tableau[1]?.name ?? obs.private.cardsInHand[0]?.name;
      assert.ok(knownActorCard);
      const clone1 = gAct.clone();
      const badOppHand = baseWorld.opponentHand.length > 0
        ? [knownActorCard, ...baseWorld.opponentHand.slice(1)]
        : baseWorld.opponentHand;
      if (badOppHand.length > 0) {
        assert.throws(
          () => clone1.determinize(actor, {...baseWorld, opponentHand: badOppHand}),
          (err: unknown) => err instanceof Error && /pool|known|unseen|invalid/i.test(err.message),
        );
      }
      const clone1b = gAct.clone();
      assert.throws(
        () => clone1b.determinize(actor, {...baseWorld, projectDeck: [knownActorCard, ...baseWorld.projectDeck.slice(1)]}),
        (err: unknown) => err instanceof Error && /pool|known|unseen|invalid/i.test(err.message),
      );

      // 2. Wrong opponent hand size
      const clone2 = gAct.clone();
      assert.throws(
        () => clone2.determinize(actor, {...baseWorld, opponentHand: [...baseWorld.opponentHand, baseWorld.projectDeck[0]]}),
        (err: unknown) => err instanceof Error && /size|length|count/i.test(err.message),
      );

      // 3. Wrong deck size
      const clone3 = gAct.clone();
      assert.throws(
        () => clone3.determinize(actor, {...baseWorld, projectDeck: baseWorld.projectDeck.slice(1)}),
        (err: unknown) => err instanceof Error && /deck|size|length/i.test(err.message),
      );

      // 4. Duplicate card
      const clone4 = gAct.clone();
      assert.throws(
        () => clone4.determinize(actor, {...baseWorld, projectDeck: [baseWorld.projectDeck[0], ...baseWorld.projectDeck.slice(0, -1)]}),
        (err: unknown) => err instanceof Error && /duplicate/i.test(err.message),
      );

      // 5. Violating forced draft passed cards in Drafting R2
      const gDraftR2 = findGameAt(
        (g) => _unsafeGetEngineGame(g).phase === 'drafting' && _unsafeGetEngineGame(g).draftRound === 2,
        'det-reject-draft',
        'elysium',
      );
      const dObs = gDraftR2.observe(0);
      const dPool = gDraftR2.unseenPool(0);
      const dWorld = sampleWorldFromObservationLegacy(dObs, dPool, makePrng(77));
      const badDraftHand = [dPool.projects[0], ...(dWorld.opponentDraftHand?.slice(1) ?? [])];
      assert.throws(
        () => gDraftR2.clone().determinize(0, {...dWorld, opponentDraftHand: badDraftHand}),
        (err: unknown) => err instanceof Error && /draft|forced|known|passed/i.test(err.message),
      );
    });
  });
});

// ---- Regression: research payment window (review findings H1/H2) ----

type EngineCard = {name: string};
const cardNames = (a: ReadonlyArray<EngineCard> | undefined): Array<string> => (a ?? []).map((c) => c.name);

/** Project-card census over every engine zone visible outside closures; duplicates mean a card was conjured. */
function projectCensusDuplicates(game: AdapterGame): Array<string> {
  const e = _unsafeGetEngineGame(game) as any;
  const seen = new Map<string, number>();
  const add = (cards: ReadonlyArray<EngineCard> | undefined) => {
    for (const n of cardNames(cards)) seen.set(n, (seen.get(n) ?? 0) + 1);
  };
  add(e.projectDeck.drawPile);
  add(e.projectDeck.discardPile);
  for (const p of e.players) {
    add(p.cardsInHand);
    add(Array.from<any>(p.playedCards ?? []).filter((c: any) => c.type !== 'corporation' && c.type !== 'prelude'));
    add(p.draftedCards);
    add(p.draftHand);
    const wf = p.getWaitingFor();
    if (wf !== undefined && wf.type === 'card') {
      if (e.phase === 'research' && e.generation >= 2) {
        add(wf.cards);
      } else if (e.phase === 'action' || e.phase === 'preludes') {
        const knownHere = new Set<string>(cardNames(p.cardsInHand));
        for (const pl of e.players) {
          for (const pc of Array.from<any>(pl.playedCards ?? [])) knownHere.add(pc.name);
        }
        const wfProj = ((wf as any).cards as Array<any> | undefined)?.filter(
          (c) => c.type !== 'corporation' && c.type !== 'prelude',
        );
        if (wfProj !== undefined && wfProj.some((c) => !knownHere.has(c.name))) {
          add(wfProj.filter((c) => !knownHere.has(c.name)));
        }
      }
    }
  }
  return [...seen].filter(([, n]) => n > 1).map(([name]) => name);
}

function isResearchPaymentWindow(game: AdapterGame, seat: Seat): boolean {
  const e = _unsafeGetEngineGame(game);
  return e.phase === 'research' && e.generation >= 2 && e.players[seat].getWaitingFor()?.type === 'payment';
}

describe('Ticket 16 regression: research payment window', () => {
  it('opponent mid-payment: determinized world never keeps the true research buy (no conjured cards)', () => {
    // Under Ticket 29, simultaneous research SelectCard prompts are submitted together, so
    // an opponent can only be in SelectPayment while actor is pending if both seats enter
    // SelectPayment simultaneously (e.g. both able to pay with heat). Exercise that path directly.
    const g = createGame({seedKey: 'T-opp-mid-pay', map: 'tharsis'});
    finishInitialResearch(g);
    advanceUntil(g, (gm) => _unsafeGetEngineGame(gm).phase === 'action', makePrng(501));
    const eng = _unsafeGetEngineGame(g) as unknown as {
      players: Array<{megaCredits: number; heat: number; canUseHeatAsMegaCredits: boolean}>;
    };
    for (const p of eng.players) {
      p.megaCredits = 30;
      p.heat = 6;
      p.canUseHeatAsMegaCredits = true;
    }
    (g as unknown as {refreshCheckpoint(e: unknown): void}).refreshCheckpoint(eng);
    while (!(_unsafeGetEngineGame(g).phase === 'research' && _unsafeGetEngineGame(g).generation >= 2)) {
      const s = g.pending()[0];
      const ch = g.choice(s);
      const passIdx = ch.candidates.findIndex((c) => /pass/i.test(c.label));
      const endIdx = ch.candidates.findIndex((c) => /end turn/i.test(c.label));
      const doneIdx = ch.candidates.findIndex((c) => c.id === 'done');
      const idx = passIdx >= 0 ? passIdx : endIdx >= 0 ? endIdx : doneIdx >= 0 ? doneIdx : 0;
      g.choose(s, idx);
    }
    for (const seat of [0, 1] as const) {
      g.choose(seat, 0); // buy 1 card
      const doneIdx = g.choice(seat).candidates.findIndex((c) => c.id === 'done');
      g.choose(seat, doneIdx);
    }
    assert.deepEqual(g.pending(), [0, 1]);

    let hits = 0;
    for (const actor of g.pending()) {
      const opp: Seat = actor === 0 ? 1 : 0;
      if (!isResearchPaymentWindow(g, opp)) continue;
      hits += 1;
      const c = g.clone();
      const pool = c.unseenPool(actor);
      c.determinize(actor, sampleWorldFromObservationLegacy(c.observe(actor), pool, makePrng(7 + actor)));
      assert.notEqual(
        _unsafeGetEngineGame(c).players[opp].getWaitingFor()?.type,
        'payment',
        'opponent payment closure (holding true cards) must be rewound',
      );
      assert.deepEqual(projectCensusDuplicates(c), [], `duplicates right after determinize (seat ${actor})`);
      playToCompletion(c, makePrng(99 + actor));
      assert.deepEqual(projectCensusDuplicates(c), [], `duplicates after rollout (seat ${actor})`);
    }
    assert.equal(hits, 2, 'expected both seats to witness opponent research payment window');
  });

  it('actor mid-payment: own pending buy is not unseen, and samples conserve cards', () => {
    let hits = 0;
    for (let gi = 0; gi < 12 && hits < 3; gi++) {
      const g = createGame({seedKey: `T-${gi}`, map: SUPPORTED_MAPS[gi % SUPPORTED_MAPS.length]});
      const rng = makePrng(500 + gi);
      let steps = 0;
      while (g.result() === null && steps < 4000 && hits < 3) {
        for (const actor of g.pending()) {
          if (!isResearchPaymentWindow(g, actor)) continue;
          hits += 1;
          const obs = g.observe(actor);
          const rb = obs.history.researchBuys.find((r) => r.generation === obs.public.generation);
          assert.ok(rb !== undefined);
          const pool = g.unseenPool(actor);
          for (const card of [...rb.bought, ...rb.discarded]) {
            assert.equal(pool.projects.includes(card), false, `pending ${card} must not be unseen`);
            assert.equal(pool.possibleOpponentHandProjects.includes(card), false);
            assert.equal(pool.knownProjectDiscardPile.includes(card), false, `${card} not yet discarded`);
          }
          const c = g.clone();
          c.determinize(actor, sampleWorldFromObservationLegacy(c.observe(actor), c.unseenPool(actor), makePrng(11 + steps)));
          playToCompletion(c, makePrng(98));
          assert.deepEqual(projectCensusDuplicates(c), [], `duplicates after rollout (T-${gi} step ${steps})`);
        }
        const s = g.pending()[0];
        g.choose(s, chooseSeeded(rng, g.choice(s).candidates.length));
        steps += 1;
      }
    }
    assert.ok(hits > 0, 'expected to hit an actor research payment window');
  });
});

function canonicalEngineStateForDeterminizeTest(
  game: AdapterGame,
  ignoreRng = false,
): Record<string, unknown> {
  const s = _unsafeGetEngineGame(game).serialize() as Record<string, unknown>;
  delete s.createdTimeMs;
  if (ignoreRng) {
    delete s.seed;
    delete s.currentSeed;
  }
  const players = s.players as Array<Record<string, unknown>>;
  for (const p of players) {
    delete p.timer;
  }
  const gameLog = s.gameLog as Array<Record<string, unknown>>;
  for (const entry of gameLog) {
    delete entry.timestamp;
  }
  return s;
}

function perturbHiddenEngineState(game: AdapterGame, actor: Seat): void {
  const e = _unsafeGetEngineGame(game);
  const opp: Seat = actor === 0 ? 1 : 0;
  const oppPlayer = e.players[opp];
  const pool = game.unseenPool(actor);
  const unseenProjSet = new Set(pool.projects);

  e.projectDeck.drawPile.reverse();
  e.projectDeck.discardPile.reverse();
  e.corporationDeck.drawPile.reverse();
  e.corporationDeck.discardPile.reverse();
  e.preludeDeck.drawPile.reverse();
  e.preludeDeck.discardPile.reverse();

  oppPlayer.dealtCorporationCards.reverse();
  oppPlayer.dealtPreludeCards.reverse();
  oppPlayer.dealtProjectCards.reverse();
  oppPlayer.cardsInHand.reverse();
  oppPlayer.preludeCardsInHand.reverse();
  oppPlayer.draftHand.reverse();
  oppPlayer.draftedCards.reverse();

  // Swap an unseen card between opp hand and projectDeck.drawPile when possible
  const handSwapIdx = oppPlayer.cardsInHand.findIndex((c) => unseenProjSet.has(c.name));
  const deckSwapIdx = e.projectDeck.drawPile.findIndex((c) => unseenProjSet.has(c.name));
  if (handSwapIdx !== -1 && deckSwapIdx !== -1) {
    const tmp = oppPlayer.cardsInHand[handSwapIdx];
    oppPlayer.cardsInHand[handSwapIdx] = e.projectDeck.drawPile[deckSwapIdx];
    e.projectDeck.drawPile[deckSwapIdx] = tmp;
  }

  // Swap an unseen prelude between opp preludeCardsInHand and preludeDeck.drawPile
  if (oppPlayer.preludeCardsInHand.length > 0 && e.preludeDeck.drawPile.length > 0) {
    const tmp = oppPlayer.preludeCardsInHand[0];
    oppPlayer.preludeCardsInHand[0] = e.preludeDeck.drawPile[0];
    e.preludeDeck.drawPile[0] = tmp;
  }

  // Swap hidden picked corporation with corporationDeck.discardPile
  if (pool.opponentHasHiddenPickedCorp && oppPlayer.pickedCorporationCard !== undefined && e.corporationDeck.discardPile.length > 0) {
    const tmp = oppPlayer.pickedCorporationCard;
    oppPlayer.pickedCorporationCard = e.corporationDeck.discardPile[0];
    e.corporationDeck.discardPile[0] = tmp;
  }

  if (e.phase === 'research' && e.generation >= 2) {
    const wf = oppPlayer.getWaitingFor();
    if (wf !== undefined && wf.type === 'card') {
      (wf as unknown as {cards: Array<unknown>}).cards.reverse();
    }
  }

  const rngObj = (e as unknown as {rng: {seed: number; currentSeed: number}}).rng;
  rngObj.seed = 0.987654321;
  rngObj.currentSeed = 0xdeadbeef;
}

describe('Ticket 28: determinize hardening (L1-L4)', () => {
  it('L1: identical observe(actor) with perturbed hidden state serializes identically after determinize across checkpoints and both seats', () => {
    const cases: Array<{label: string; game: AdapterGame; actor: Seat}> = [];

    // Setup (both seats)
    const gSetup0 = createGame({seedKey: 't28-l1-setup-0', map: 'tharsis'});
    cases.push({label: 'setup seat 0', game: gSetup0, actor: 0});

    const gSetup1 = createGame({seedKey: 't28-l1-setup-1', map: 'elysium'});
    cases.push({label: 'setup seat 1', game: gSetup1, actor: 1});

    const gSetup1After0 = createGame({seedKey: 't28-l1-setup-1-after-0', map: 'hellas'});
    while (gSetup1After0.pending().includes(0)) {
      gSetup1After0.choose(0, pickDoneOrZero(gSetup1After0, 0));
    }
    cases.push({label: 'setup seat 1 after seat 0 initial buy', game: gSetup1After0, actor: 1});

    // Preludes (both seats)
    for (const seat of [0, 1] as const) {
      const gPrel = findGameAt(
        (g) => _unsafeGetEngineGame(g).phase === 'preludes' && g.pending().includes(seat),
        `t28-l1-prel-${seat}`,
        'tharsis',
      );
      cases.push({label: `preludes seat ${seat}`, game: gPrel, actor: seat});
    }

    // Drafting (both seats)
    for (const seat of [0, 1] as const) {
      const gDraft = findGameAt(
        (g) => _unsafeGetEngineGame(g).phase === 'drafting' && _unsafeGetEngineGame(g).draftRound === 2 && g.pending().includes(seat),
        `t28-l1-draft-${seat}`,
        'elysium',
      );
      cases.push({label: `drafting R2 seat ${seat}`, game: gDraft, actor: seat});
    }

    // Research (Gen >= 2, both seats)
    for (const seat of [0, 1] as const) {
      const gRes = findGameAt(
        (g) => _unsafeGetEngineGame(g).phase === 'research' && _unsafeGetEngineGame(g).generation >= 2 && g.pending().includes(seat),
        `t28-l1-res-${seat}`,
        'hellas',
      );
      cases.push({label: `research gen>=2 seat ${seat}`, game: gRes, actor: seat});
    }

    // Action (Gen >= 2, both seats)
    for (const seat of [0, 1] as const) {
      const gAct = findGameAt(
        (g) => _unsafeGetEngineGame(g).phase === 'action' && _unsafeGetEngineGame(g).generation >= 2 && g.pending().includes(seat),
        `t28-l1-act-${seat}`,
        'vastitas borealis nova',
      );
      cases.push({label: `action gen>=2 seat ${seat}`, game: gAct, actor: seat});
    }

    for (let i = 0; i < cases.length; i++) {
      const {label, game, actor} = cases[i];
      const c1 = game.clone();
      const c2 = game.clone();
      perturbHiddenEngineState(c2, actor);

      const obs1 = c1.observe(actor);
      assert.equal(
        serializeObservation(c2.observe(actor)),
        serializeObservation(obs1),
        `${label}: perturbation leaked into observe(${actor})`,
      );

      const pool = c1.unseenPool(actor);
      const world = sampleWorldFromObservationLegacy(obs1, pool, makePrng(3000 + i));

      c1.determinize(actor, world);
      c2.determinize(actor, world);

      assert.deepEqual(
        canonicalEngineStateForDeterminizeTest(c1, false),
        canonicalEngineStateForDeterminizeTest(c2, false),
        `${label}: determinized engine states must serialize identically`,
      );
    }
  });

  it('L2: conserves every project, corporation, and prelude card across all engine zones (including fizzled preludes)', () => {
    const fullCensus = (game: AdapterGame): {
      projects: Map<string, number>;
      corps: Map<string, number>;
      preludes: Map<string, number>;
    } => {
      const e = _unsafeGetEngineGame(game) as any;
      const projects = new Map<string, number>();
      const corps = new Map<string, number>();
      const preludes = new Map<string, number>();
      const bump = (map: Map<string, number>, cards: ReadonlyArray<EngineCard> | undefined) => {
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

        const played: Array<any> = Array.from(p.playedCards ?? []);
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

        // Before corps are revealed, pickedCorporationCard is set without being in playedCards yet
        if (p.pickedCorporationCard !== undefined && !playedCorps.some((c) => c.name === p.pickedCorporationCard.name)) {
          bump(corps, [p.pickedCorporationCard]);
        }

        const wf = p.getWaitingFor();
        if (wf !== undefined && wf.type === 'card' && e.phase === 'research' && e.generation >= 2) {
          bump(projects, wf.cards);
        }
      }

      return {projects, corps, preludes};
    };

    const assertExactPool = (counts: Map<string, number>, expectedTotal: number, label: string) => {
      const duplicates = [...counts].filter(([, n]) => n > 1).map(([name]) => name);
      assert.deepEqual(duplicates, [], `${label}: duplicate cards found`);
      const total = [...counts.values()].reduce((a, b) => a + b, 0);
      assert.equal(total, expectedTotal, `${label}: expected ${expectedTotal} unique cards, got ${total}`);
    };

    // Check across setup, preludes, drafting, research, and action
    const gSetup = createGame({seedKey: 't28-l2-setup', map: 'tharsis'});
    const gPrel = findGameAt((g) => _unsafeGetEngineGame(g).phase === 'preludes', 't28-l2-prel', 'elysium');
    const gDraft = findGameAt((g) => _unsafeGetEngineGame(g).phase === 'drafting' && _unsafeGetEngineGame(g).draftRound === 2, 't28-l2-draft', 'hellas');
    const gRes = findGameAt((g) => _unsafeGetEngineGame(g).phase === 'research' && _unsafeGetEngineGame(g).generation >= 2, 't28-l2-res', 'tharsis');
    const gAct = findGameAt((g) => _unsafeGetEngineGame(g).phase === 'action' && _unsafeGetEngineGame(g).generation >= 2, 't28-l2-act', 'vastitas borealis nova');

    for (const [label, g] of [
      ['setup', gSetup],
      ['preludes', gPrel],
      ['drafting', gDraft],
      ['research', gRes],
      ['action', gAct],
    ] as const) {
      const actor = g.pending()[0];
      const c = g.clone();
      const obs = c.observe(actor);
      const pool = c.unseenPool(actor);
      c.determinize(actor, sampleWorldFromObservationLegacy(obs, pool, makePrng(4001)));
      const census = fullCensus(c);
      assertExactPool(census.projects, 215, `${label} projects`);
      assertExactPool(census.corps, 17, `${label} corps`);
      assertExactPool(census.preludes, 35, `${label} preludes`);
    }

    // Also verify a game where a prelude fizzles conserves all 35 preludes in action phase, for
    // each seat while it is the pending (deciding) seat.
    for (const actor of [0, 1] as const) {
      const gFizzle = findGameAt(
        (g) =>
          _unsafeGetEngineGame(g).phase === 'action' &&
          g.pending().includes(actor) &&
          g.observe(0).public.publicLog.some((e) => e.message.includes('fizzled.')),
        't28-l2-fizzle',
        'tharsis',
        150,
      );
      const c = gFizzle.clone();
      c.determinize(actor, sampleWorldFromObservationLegacy(c.observe(actor), c.unseenPool(actor), makePrng(4002 + actor)));
      const census = fullCensus(c);
      assertExactPool(census.projects, 215, `fizzle seat ${actor} projects`);
      assertExactPool(census.corps, 17, `fizzle seat ${actor} corps`);
      assertExactPool(census.preludes, 35, `fizzle seat ${actor} preludes`);
    }
  });

  it('L2: fills omitted projectDiscardPile, corporationDeck, and preludeDeck in sampled order, not alphabetically', () => {
    const gAct = findGameAt(
      (g) => _unsafeGetEngineGame(g).phase === 'action' && g.observe(0).public.discardPileSize >= 10,
      't28-l2-order',
      'tharsis',
    );
    const actor = gAct.pending()[0];
    const obs = gAct.observe(actor);
    const pool = gAct.unseenPool(actor);

    const w1 = sampleWorldFromObservationLegacy(obs, pool, makePrng(5001));
    const w2 = sampleWorldFromObservationLegacy(obs, pool, makePrng(5002));
    assert.equal(w1.projectDiscardPile, undefined);
    assert.equal(w2.projectDiscardPile, undefined);

    const c1 = gAct.clone();
    const c2 = gAct.clone();
    c1.determinize(actor, w1);
    c2.determinize(actor, w2);

    const e1 = _unsafeGetEngineGame(c1);
    const e2 = _unsafeGetEngineGame(c2);

    const isAlphabetical = (names: ReadonlyArray<string>): boolean => {
      const sorted = names.slice().sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
      return names.every((n, idx) => n === sorted[idx]);
    };

    const corpDraw1 = e1.corporationDeck.drawPile.map((c) => c.name);
    const prelDraw1 = e1.preludeDeck.drawPile.map((c) => c.name);
    const projDisc1 = e1.projectDeck.discardPile.map((c) => c.name);

    const corpDraw2 = e2.corporationDeck.drawPile.map((c) => c.name);
    const prelDraw2 = e2.preludeDeck.drawPile.map((c) => c.name);
    const projDisc2 = e2.projectDeck.discardPile.map((c) => c.name);

    assert.equal(isAlphabetical(corpDraw1), false, 'corporationDeck.drawPile must not be alphabetical');
    assert.equal(isAlphabetical(prelDraw1), false, 'preludeDeck.drawPile must not be alphabetical');
    assert.equal(isAlphabetical(projDisc1), false, 'omitted projectDeck.discardPile must not be alphabetical');

    assert.notDeepEqual(corpDraw1, corpDraw2, 'different sampled worlds should shuffle corporationDeck differently');
    assert.notDeepEqual(prelDraw1, prelDraw2, 'different sampled worlds should shuffle preludeDeck differently');
    assert.notDeepEqual(projDisc1, projDisc2, 'different sampled worlds should shuffle projectDiscardPile differently');
  });

  it('L3: opponentResearchOfferSize and draft sizes account for public Mars Maths (and deck exhaustion) and accept true world', () => {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const {MarsMaths} = require(path.resolve(__dirname, '../../../tools/simbench/engine/build/src/server/cards/pathfinders/MarsMaths.js')) as {
      MarsMaths: new () => unknown;
    };

    const extractTrueWorld = (game: AdapterGame, actor: Seat): SampledWorld => {
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
    };

    // 1. Opponent has Mars Maths: Gen 2 Drafting R1/R2/R3 and Gen 2 Research (offer of 5) + Gen 2 Action after buying 4
    const gOppMM = createGame({seedKey: 't28-l3-opp-mm', map: 'tharsis'});
    finishInitialResearch(gOppMM);
    advanceUntil(gOppMM, (g) => _unsafeGetEngineGame(g).phase === 'action', makePrng(6100));
    (_unsafeGetEngineGame(gOppMM).players[0] as unknown as {playedCards: Array<unknown>}).playedCards.push(new MarsMaths());
    advanceUntil(
      gOppMM,
      (g) => _unsafeGetEngineGame(g).phase === 'drafting' && _unsafeGetEngineGame(g).generation === 2,
      makePrng(6101),
    );

    const actor: Seat = 1;
    for (const expectedRound of [1, 2, 3]) {
      const e = _unsafeGetEngineGame(gOppMM);
      assert.equal(e.phase, 'drafting');
      assert.equal(e.draftRound, expectedRound);
      const pool = gOppMM.unseenPool(actor);
      assert.equal(pool.opponentDraftHandSize, expectedRound === 1 ? 5 : 5 - expectedRound);
      assert.equal(pool.opponentDraftedSize, expectedRound === 1 ? 0 : expectedRound);
      const clone = gOppMM.clone();
      clone.determinize(actor, extractTrueWorld(gOppMM, actor));
      // Advance this draft round
      while (_unsafeGetEngineGame(gOppMM).phase === 'drafting' && _unsafeGetEngineGame(gOppMM).draftRound === expectedRound) {
        const s = gOppMM.pending()[0];
        gOppMM.choose(s, 0);
      }
    }

    assert.equal(_unsafeGetEngineGame(gOppMM).phase, 'research');
    const resPool = gOppMM.unseenPool(actor);
    assert.equal(resPool.opponentResearchOfferSize, 5);
    const resClone = gOppMM.clone();
    resClone.determinize(actor, extractTrueWorld(gOppMM, actor));

    // Also verify opponent buying 4 of 5 cards in Mars Maths research does not falsely force all 3 known draft cards into knownOppHand
    for (let i = 0; i < 4; i++) {
      gOppMM.choose(0, 0);
    }
    const buyDoneIdx = gOppMM.choice(0).candidates.findIndex((c) => c.id === 'done');
    gOppMM.choose(0, buyDoneIdx);
    // Pay for opponent's 4 bought cards if prompted, then let actor buy 0
    while (_unsafeGetEngineGame(gOppMM).phase === 'research') {
      const pending = gOppMM.pending();
      if (pending[0] === 0) {
        gOppMM.choose(0, pickDoneOrZero(gOppMM, 0));
      } else {
        gOppMM.choose(1, pickDoneOrZero(gOppMM, 1));
      }
    }
    const actPool = gOppMM.unseenPool(actor);
    const actClone = gOppMM.clone();
    actClone.determinize(actor, extractTrueWorld(gOppMM, actor));
    assert.ok(actPool.opponentHandSize >= 0);

    // 2. Actor has Mars Maths (offer of 5), opponent does NOT (offer of 4)
    const gActorMM = createGame({seedKey: 't28-l3-actor-mm', map: 'elysium'});
    finishInitialResearch(gActorMM);
    advanceUntil(gActorMM, (g) => _unsafeGetEngineGame(g).phase === 'action', makePrng(6200));
    (_unsafeGetEngineGame(gActorMM).players[0] as unknown as {playedCards: Array<unknown>}).playedCards.push(new MarsMaths());
    advanceUntil(
      gActorMM,
      (g) => _unsafeGetEngineGame(g).phase === 'drafting' && _unsafeGetEngineGame(g).generation === 2,
      makePrng(6201),
    );
    // Seat 0 (actor) picks first in R1: actor sees 5 cards -> curRound must still be 1!
    const actorDraftPool = gActorMM.unseenPool(0);
    assert.equal(actorDraftPool.opponentDraftHandSize, 4);
    assert.equal(actorDraftPool.opponentDraftedSize, 0);
    gActorMM.clone().determinize(0, extractTrueWorld(gActorMM, 0));

    advanceUntil(gActorMM, (g) => _unsafeGetEngineGame(g).phase === 'research' && _unsafeGetEngineGame(g).generation === 2, makePrng(6202));
    const actorResPool = gActorMM.unseenPool(0);
    assert.equal(actorResPool.opponentResearchOfferSize, 4);
    gActorMM.clone().determinize(0, extractTrueWorld(gActorMM, 0));

    // 3. Deck exhaustion: only 6 project cards left entering Gen 2 Drafting (Seat 0 draws 4, Seat 1 draws 2)
    const gExhaust = createGame({seedKey: 't28-l3-exhaust', map: 'hellas'});
    finishInitialResearch(gExhaust);
    advanceUntil(gExhaust, (g) => _unsafeGetEngineGame(g).phase === 'action', makePrng(6300));
    while (_unsafeGetEngineGame(gExhaust).phase === 'action' && _unsafeGetEngineGame(gExhaust).generation === 1) {
      const s = gExhaust.pending()[0];
      const ch = gExhaust.choice(s);
      const passIdx = ch.candidates.findIndex((c) => /pass/i.test(c.label));
      if (passIdx !== -1 && _unsafeGetEngineGame(gExhaust).players.filter((p) => (p as unknown as {passed?: boolean}).passed).length === 1 ||
          passIdx !== -1 && _unsafeGetEngineGame(gExhaust).activePlayer === _unsafeGetEngineGame(gExhaust).players[s]) {
        // Check if only one player hasn't passed yet: drain deck right before the final pass
        const eCheck = _unsafeGetEngineGame(gExhaust);
        const passedCount = gExhaust.observe(0).public.passedPlayers.length;
        if (passedCount === 1) {
          const toDrain = eCheck.projectDeck.drawPile.length + eCheck.projectDeck.discardPile.length - 6;
          const moved = (eCheck.projectDeck as unknown as {drawN(logger: unknown, count: number): Array<unknown>}).drawN(eCheck, toDrain);
          eCheck.players[1].cardsInHand.push(...(moved as typeof eCheck.players[1]['cardsInHand']));
        }
        gExhaust.choose(s, passIdx);
      } else {
        gExhaust.choose(s, 0);
      }
    }
    assert.equal(_unsafeGetEngineGame(gExhaust).phase, 'drafting');
    assert.equal(_unsafeGetEngineGame(gExhaust).generation, 2);
    // Check true world accepted for both seats in exhausted draft and exhausted research
    const exDraftPool0 = gExhaust.unseenPool(0);
    assert.equal(exDraftPool0.opponentDraftHandSize, 2);
    gExhaust.clone().determinize(0, extractTrueWorld(gExhaust, 0));
    const exDraftPool1 = gExhaust.unseenPool(1);
    assert.equal(exDraftPool1.opponentDraftHandSize, 4);
    gExhaust.clone().determinize(1, extractTrueWorld(gExhaust, 1));

    advanceUntil(gExhaust, (g) => _unsafeGetEngineGame(g).phase === 'research' && _unsafeGetEngineGame(g).generation === 2, makePrng(6302));
    gExhaust.clone().determinize(0, extractTrueWorld(gExhaust, 0));
    gExhaust.clone().determinize(1, extractTrueWorld(gExhaust, 1));
  });

  it('L4: restoreDraft emits no Overwriting waitingFor warnings and determinize reseeds engine RNG', () => {
    const gDraft = findGameAt(
      (g) => _unsafeGetEngineGame(g).phase === 'drafting' && _unsafeGetEngineGame(g).draftRound === 2 && g.pending().includes(0),
      't28-l4-draft',
      'tharsis',
    );
    const pkt0 = gDraft.choice(0).candidates[0].id;
    // Under Ticket 29, seat 0 picking first holds seat 0 in deferred[] (engine needsToDraft stays true).
    const gDeferred = gDraft.clone();
    gDeferred.choose(0, 0);
    assert.equal(_unsafeGetEngineGame(gDeferred).players[0].needsToDraft, true);
    assert.deepEqual(gDeferred.pending(), [1]);

    const warnings: Array<string> = [];
    const origWarn = console.warn;
    console.warn = (...args: Array<unknown>) => {
      warnings.push(args.map((a) => String(a)).join(' '));
    };
    try {
      const obs = gDeferred.observe(1);
      const pool = gDeferred.unseenPool(1);
      const baseWorld = sampleWorldFromObservationLegacy(obs, pool, makePrng(7001));
      const origCurrentSeed = (_unsafeGetEngineGame(gDraft) as unknown as {rng: {current: number}}).rng.current;

      const c1 = gDeferred.clone();
      c1.determinize(1, {...baseWorld, seed: 101});
      const c2 = gDeferred.clone();
      c2.determinize(1, {...baseWorld, seed: 202});

      const s1 = (_unsafeGetEngineGame(c1) as unknown as {rng: {current: number}}).rng.current;
      const s2 = (_unsafeGetEngineGame(c2) as unknown as {rng: {current: number}}).rng.current;

      assert.notEqual(s1, origCurrentSeed, 'determinize must reseed engine RNG away from true RNG');
      assert.notEqual(s2, s1, 'different world.seed values must yield different engine RNG states');

      // Also exercise the defensive restoreDraft() branch when opp.needsToDraft === false directly on engine
      const cDirect = gDraft.clone();
      _unsafeGetEngineGame(cDirect).players[0].process({type: 'card', cards: [pkt0 as never]});
      assert.equal(_unsafeGetEngineGame(cDirect).players[0].needsToDraft, false);
      cDirect.determinize(1, {...baseWorld, seed: 303});
      assert.equal(_unsafeGetEngineGame(cDirect).players[0].needsToDraft, true);

      assert.equal(
        warnings.filter((w) => w.includes('Overwriting waitingFor')).length,
        0,
        `expected 0 Overwriting waitingFor warnings, got: ${JSON.stringify(warnings)}`,
      );
    } finally {
      console.warn = origWarn;
    }
  });

  it('conserves cards during pending unseen card-selection prompts and Buy payment closures', () => {
    const engineCardsRoot = path.resolve(__dirname, '../../../tools/simbench/engine/build/src/server/cards');
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const {InventorsGuild} = require(path.join(engineCardsRoot, 'base/InventorsGuild.js')) as {
      InventorsGuild: new () => unknown;
    };
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const {Helion} = require(path.join(engineCardsRoot, 'corporation/Helion.js')) as {
      Helion: new () => unknown;
    };

    const gIG = findGameAt(
      (g) => _unsafeGetEngineGame(g).phase === 'preludes',
      't28-pending-sel-ig',
      'tharsis',
    );
    const eIG = _unsafeGetEngineGame(gIG) as unknown as {
      phase: string;
      players: Array<{
        playedCards: Array<unknown>;
        cardsInHand: Array<{name: string}>;
        heat: number;
        megaCredits: number;
        canUseHeatAsMegaCredits: boolean;
        getWaitingFor(): {type?: string; cards?: ReadonlyArray<{name: string}>} | undefined;
      }>;
      projectDeck: {drawPile: Array<{name: string}>; discardPile: Array<{name: string}>};
    };
    // Move Inventors' Guild from projectDeck and Helion from corpDeck (or replace in-place) so card conservation holds!
    const eRaw = eIG as unknown as {
      corporationDeck: {drawPile: Array<{name: string}>; discardPile: Array<{name: string}>};
    };
    const moveCardByName = (piles: Array<Array<{name: string}>>, name: string): boolean => {
      for (const pile of piles) {
        const idx = pile.findIndex((c) => c.name === name);
        if (idx !== -1) {
          pile.splice(idx, 1);
          return true;
        }
      }
      return false;
    };
    assert.ok(
      moveCardByName([eIG.projectDeck.drawPile, eIG.projectDeck.discardPile], "Inventors' Guild"),
      "expected Inventors' Guild in projectDeck for test setup",
    );
    eIG.players[0].playedCards.push(new InventorsGuild());
    if (moveCardByName([eRaw.corporationDeck.drawPile, eRaw.corporationDeck.discardPile], 'Helion')) {
      eIG.players[0].playedCards.push(new Helion());
    }
    eIG.players[0].canUseHeatAsMegaCredits = true;
    eIG.players[0].heat = 5;
    eIG.players[0].megaCredits = 10;
    (gIG as unknown as {refreshCheckpoint(e: unknown): void}).refreshCheckpoint(eIG);
    advanceUntil(gIG, (g) => _unsafeGetEngineGame(g).phase === 'action' && g.pending()[0] === 0, makePrng(8100));

    // Trigger Inventors' Guild action
    const root = gIG.choice(0);
    const actIdx = root.candidates.findIndex((c) => /action from a played card/i.test(c.label));
    assert.ok(actIdx >= 0);
    gIG.choose(0, actIdx);
    const cardAct = gIG.choice(0);
    const igIdx = cardAct.candidates.findIndex((c) => c.id === "Inventors' Guild");
    assert.ok(igIdx >= 0);
    gIG.choose(0, igIdx);

    // 1) Open SelectCard prompt ("Select card(s) to buy")
    const wfOpen = eIG.players[0].getWaitingFor();
    assert.equal(wfOpen?.type, 'card');
    const seenCard = wfOpen?.cards?.[0]?.name;
    assert.ok(seenCard !== undefined);

    const poolOpen = gIG.unseenPool(0);
    assert.equal(poolOpen.projects.includes(seenCard), false, 'open SelectCard offer must not be unseen');
    const cOpen = gIG.clone();
    cOpen.determinize(0, {
      opponentHand: eIG.players[1].cardsInHand.map((c) => c.name),
      projectDeck: eIG.projectDeck.drawPile.map((c) => c.name),
      projectDiscardPile: eIG.projectDeck.discardPile.map((c) => c.name),
    });
    assert.deepEqual(projectCensusDuplicates(cOpen), []);
    playToCompletion(cOpen, makePrng(8101));
    assert.deepEqual(projectCensusDuplicates(cOpen), []);

    // 2) Select Buy -> enters SelectPaymentDeferred ("Select how to spend 3 M€ for 1 cards")
    const buyIdx = gIG.choice(0).candidates.findIndex((c) => c.id === seenCard);
    assert.ok(buyIdx >= 0);
    gIG.choose(0, buyIdx);
    const doneBuyIdx = gIG.choice(0).candidates.findIndex((c) => c.id === 'done');
    if (doneBuyIdx !== -1) {
      gIG.choose(0, doneBuyIdx);
    }
    const wfPay = eIG.players[0].getWaitingFor();
    assert.equal(wfPay?.type, 'payment');

    const poolPay = gIG.unseenPool(0);
    assert.equal(poolPay.projects.includes(seenCard), false, 'Buy payment closure card must not be unseen');
    const cPay = gIG.clone();
    cPay.determinize(0, {
      opponentHand: (eIG.players[1] as unknown as {cardsInHand: Array<{name: string}>}).cardsInHand.map((c) => c.name),
      projectDeck: eIG.projectDeck.drawPile.map((c) => c.name),
      projectDiscardPile: eIG.projectDeck.discardPile.map((c) => c.name),
    });
    assert.deepEqual(projectCensusDuplicates(cPay), []);
    playToCompletion(cPay, makePrng(8102));
    assert.deepEqual(projectCensusDuplicates(cPay), []);

    // 3) Valley Trust initialAction opening 3 preludes in SelectCard
    const gVT = findGameAt(
      (g) => {
        const e = _unsafeGetEngineGame(g);
        if (e.phase !== 'action' || e.generation !== 1 || g.pending().length !== 1) return false;
        const s = g.pending()[0];
        return g.observe(s).public.players[s].tableau.some((tc) => tc.name === 'Valley Trust');
      },
      't28-pending-sel-vt',
      'tharsis',
      200,
    );
    const vtSeat = gVT.pending()[0];
    const vtRoot = gVT.choice(vtSeat);
    const vtInitIdx = vtRoot.candidates.findIndex((c) => /Valley Trust/i.test(c.label));
    assert.ok(vtInitIdx >= 0, 'Expected Valley Trust initialAction option');
    gVT.choose(vtSeat, vtInitIdx);

    const eVT = _unsafeGetEngineGame(gVT);
    const vtWf = eVT.players[vtSeat].getWaitingFor();
    assert.equal(vtWf?.type, 'card');
    const vtOffered = (vtWf as unknown as {cards: ReadonlyArray<{name: string}>}).cards.map((c) => c.name);
    assert.equal(vtOffered.length, 3);
    const vtPool = gVT.unseenPool(vtSeat);
    for (const p of vtOffered) {
      assert.equal(vtPool.preludes.includes(p), false, `Valley Trust offered prelude ${p} must not be unseen`);
    }
    const cVT = gVT.clone();
    const vtOpp: Seat = vtSeat === 0 ? 1 : 0;
    cVT.determinize(vtSeat, {
      opponentHand: eVT.players[vtOpp].cardsInHand.map((c) => c.name),
      projectDeck: eVT.projectDeck.drawPile.map((c) => c.name),
      projectDiscardPile: eVT.projectDeck.discardPile.map((c) => c.name),
    });
    playToCompletion(cVT, makePrng(8103));
  });

  it('clone() preserves pendingInitialActions tableau object identity so funded awards skip Vitor cleanly', () => {
    const g = createGame({seedKey: 't28-truth-2', map: 'hellas'});
    const mainRng = makePrng(9000 + 2 * 17);
    for (let d = 0; d < 48; d++) {
      const s = g.pending()[0];
      g.choose(s, chooseSeeded(mainRng, g.choice(s).candidates.length));
    }
    const c = g.clone();
    const eng = _unsafeGetEngineGame(c) as unknown as {
      players: Array<{
        tableau: {get(name: string): unknown};
        pendingInitialActions: Array<{name: string}>;
      }>;
    };
    for (const p of eng.players) {
      for (const initCorp of p.pendingInitialActions) {
        assert.equal(initCorp, p.tableau.get(initCorp.name));
      }
    }
    playToCompletion(c, makePrng(29000 + 2 * 1000 + 48 + 1));
    assert.ok(c.result() !== null);
  });
});



describe('Ticket 28 regression: hidden picked corporation affordability', () => {
  it('hides seat 0 initial buy behind deferral and still enforces affordability if engine already committed seat 0', () => {
    // t28-truth-2 / hellas at decision 24: seat 0 has finished answering its initial setup
    // (including a 9-card project buy), while seat 1 is still deciding. Under Ticket #29, seat 0's
    // initial buy is held in deferred[0] and NOT submitted to the engine until seat 1 also finishes.
    const g = createGame({seedKey: 't28-truth-2', map: 'hellas'});
    const mainRng = makePrng(9000 + 2 * 17);
    for (let d = 0; d < 24; d++) {
      const s = g.pending()[0];
      g.choose(s, chooseSeeded(mainRng, g.choice(s).candidates.length));
    }
    assert.deepEqual(g.pending(), [1]);
    const deferredPool = g.unseenPool(1);
    assert.equal(deferredPool.opponentInitialBuyComplete, false);
    assert.equal(deferredPool.opponentHasHiddenPickedCorp, false);
    assert.equal(deferredPool.opponentHandSize, 0);

    // Also keep exercising determinize's defensive affordability filter on a direct-engine clone
    // where seat 0's 9-card initial buy has already been submitted to the engine.
    const cDirect = createGame({seedKey: 't28-truth-2', map: 'hellas'});
    const engDirect = _unsafeGetEngineGame(cDirect) as unknown as {
      players: Array<{
        dealtCorporationCards: ReadonlyArray<{name: string; startingMegaCredits: number}>;
        dealtPreludeCards: ReadonlyArray<{name: string}>;
        dealtProjectCards: ReadonlyArray<{name: string}>;
        process(input: unknown): void;
      }>;
    };
    const p0 = engDirect.players[0];
    const richCorp = p0.dealtCorporationCards.find((c) => c.startingMegaCredits >= 27);
    assert.ok(richCorp !== undefined);
    p0.process({
      type: 'initialCards',
      responses: [
        {type: 'card', cards: [richCorp.name]},
        {type: 'card', cards: p0.dealtPreludeCards.slice(0, 2).map((c) => c.name)},
        {type: 'card', cards: p0.dealtProjectCards.slice(0, 9).map((c) => c.name)},
      ],
    });
    (cDirect as unknown as {refreshCheckpoint(e: unknown): void}).refreshCheckpoint(engDirect);

    assert.ok(cDirect.pending().includes(1));
    const pool = cDirect.unseenPool(1);
    assert.equal(pool.opponentHasHiddenPickedCorp, true);
    assert.equal(pool.opponentHandSize, 9);
    const candidates = new Set(pool.opponentPickedCorporationCandidates);
    assert.ok(candidates.size > 0);
    for (const c of candidates) assert.ok(pool.corporations.includes(c));
    assert.ok(!candidates.has('PhoboLog'));
    assert.ok(pool.corporations.includes('PhoboLog'));

    const base = sampleWorldFromObservationLegacy(cDirect.observe(1), pool, makePrng(77));
    const bad = cDirect.clone();
    assert.throws(
      () => bad.determinize(1, {...base, opponentPickedCorporation: 'PhoboLog'}),
      /opponentPickedCorporation/,
    );

    const warnings: Array<string> = [];
    const origWarn = console.warn;
    console.warn = (m: unknown) => { warnings.push(String(m)); };
    try {
      for (const corp of candidates) {
        const c = cDirect.clone();
        c.determinize(1, {...base, opponentPickedCorporation: corp});
        playToCompletion(c, makePrng(4242));
      }
    } finally {
      console.warn = origWarn;
    }
    assert.deepEqual(warnings.filter((w) => w.startsWith('Illegal state')), []);
  });
});

describe('Ticket 28 regression: determinize contract', () => {
  it('rejects determinize for a seat that is not pending', () => {
    const g = createGame({seedKey: 't28-contract', map: 'tharsis'});
    const rng = makePrng(31);
    while (g.pending().length !== 1) {
      const s = g.pending()[0];
      g.choose(s, chooseSeeded(rng, g.choice(s).candidates.length));
    }
    const idle: Seat = g.pending()[0] === 0 ? 1 : 0;
    const c = g.clone();
    const world = sampleWorldFromObservationLegacy(c.observe(idle), c.unseenPool(idle), makePrng(5));
    assert.throws(() => c.determinize(idle, world), /requires seat \d to be pending/);
  });
});

describe('Ticket 23: sampleWorldFromObservation(obs, seed)', () => {
  it('takes only (obs, seed), is deterministic, and is unchanged when opponent hand changes', () => {
    const g = findGameAt(
      (gm) => _unsafeGetEngineGame(gm).phase === 'action' && _unsafeGetEngineGame(gm).generation >= 2,
      't23-sampler-obs-only',
      'tharsis',
    );
    const actor = g.pending()[0];
    const opp: Seat = actor === 0 ? 1 : 0;

    const obsBefore = g.observe(actor);
    const w1 = sampleWorldFromObservation(obsBefore, 123456);
    const w2 = sampleWorldFromObservation(obsBefore, 123456);
    const wDiff = sampleWorldFromObservation(obsBefore, 654321);
    assert.deepEqual(w1, w2);
    assert.notDeepEqual(w1, wDiff);

    // Mutate opponent's hand in the live engine by swapping an unseen hand card with drawPile
    const eng = _unsafeGetEngineGame(g);
    const pool = g.unseenPool(actor);
    const unseenSet = new Set(pool.projects);
    const twin = g.clone();
    const handIdx = eng.players[opp].cardsInHand.findIndex((c) => unseenSet.has(c.name));
    const deckIdx = eng.projectDeck.drawPile.findIndex((c) => unseenSet.has(c.name));
    assert.notEqual(handIdx, -1, 'fixture must give the opponent an unseen hand card');
    assert.notEqual(deckIdx, -1, 'fixture must have an unseen draw-pile card');
    const handBefore = eng.players[opp].cardsInHand.map((c) => c.name);
    const tmp = eng.players[opp].cardsInHand[handIdx];
    eng.players[opp].cardsInHand[handIdx] = eng.projectDeck.drawPile[deckIdx];
    eng.projectDeck.drawPile[deckIdx] = tmp;
    assert.notDeepEqual(
      eng.players[opp].cardsInHand.map((c) => c.name),
      handBefore,
      'opponent true hand must actually change',
    );

    const obsAfter = g.observe(actor);
    assert.equal(serializeObservation(obsAfter), serializeObservation(obsBefore));
    const wAfter = sampleWorldFromObservation(obsAfter, 123456);
    assert.deepEqual(wAfter, w1);

    // End to end: determinizing the mutated game and its unmutated twin with the same seed
    // yields identical hidden worlds, so nothing about the true opponent hand reaches the clone.
    const hiddenState = (gm: ReturnType<typeof createGame>) => {
      const e = _unsafeGetEngineGame(gm);
      return {
        oppHand: e.players[opp].cardsInHand.map((c) => c.name),
        drawPile: e.projectDeck.drawPile.map((c) => c.name),
      };
    };
    const cMut = g.clone();
    cMut.determinize(actor, sampleWorldFromObservation(cMut.observe(actor), 99));
    const cTwin = twin.clone();
    cTwin.determinize(actor, sampleWorldFromObservation(cTwin.observe(actor), 99));
    assert.deepEqual(hiddenState(cMut), hiddenState(cTwin));
    assert.notDeepEqual(hiddenState(g).oppHand, hiddenState(twin).oppHand);
  });
});

