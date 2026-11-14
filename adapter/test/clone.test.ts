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
 * Pure sampler constructing a legal SampledWorld from ActorObservation + UnseenPool ONLY.
 */
function sampleWorldFromObservation(
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
    opponentPickedCorporation = availableCorps.splice(0, 1)[0];
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
        const world = sampleWorldFromObservation(obsBefore, pool, makePrng(900 + i));

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
      const baseWorld = sampleWorldFromObservation(obs, pool, makePrng(42));

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
      const dWorld = sampleWorldFromObservation(dObs, dPool, makePrng(77));
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
    if (wf !== undefined && wf.type === 'card' && e.phase === 'research') add(wf.cards);
  }
  return [...seen].filter(([, n]) => n > 1).map(([name]) => name);
}

function isResearchPaymentWindow(game: AdapterGame, seat: Seat): boolean {
  const e = _unsafeGetEngineGame(game);
  return e.phase === 'research' && e.generation >= 2 && e.players[seat].getWaitingFor()?.type === 'payment';
}

describe('Ticket 16 regression: research payment window', () => {
  it('opponent mid-payment: determinized world never keeps the true research buy (no conjured cards)', () => {
    let hits = 0;
    for (let gi = 0; gi < 12 && hits < 3; gi++) {
      const g = createGame({seedKey: `T-${gi}`, map: SUPPORTED_MAPS[gi % SUPPORTED_MAPS.length]});
      const rng = makePrng(500 + gi);
      let steps = 0;
      while (g.result() === null && steps < 4000 && hits < 3) {
        for (const actor of g.pending()) {
          const opp: Seat = actor === 0 ? 1 : 0;
          if (!isResearchPaymentWindow(g, opp)) continue;
          hits += 1;
          const c = g.clone();
          const pool = c.unseenPool(actor);
          c.determinize(actor, sampleWorldFromObservation(c.observe(actor), pool, makePrng(7 + steps)));
          assert.notEqual(_unsafeGetEngineGame(c).players[opp].getWaitingFor()?.type, 'payment',
            'opponent payment closure (holding true cards) must be rewound');
          assert.deepEqual(projectCensusDuplicates(c), [], `duplicates right after determinize (T-${gi} step ${steps})`);
          playToCompletion(c, makePrng(99));
          assert.deepEqual(projectCensusDuplicates(c), [], `duplicates after rollout (T-${gi} step ${steps})`);
        }
        const s = g.pending()[0];
        g.choose(s, chooseSeeded(rng, g.choice(s).candidates.length));
        steps += 1;
      }
    }
    assert.ok(hits > 0, 'expected to hit an opponent research payment window');
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
          c.determinize(actor, sampleWorldFromObservation(c.observe(actor), c.unseenPool(actor), makePrng(11 + steps)));
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
