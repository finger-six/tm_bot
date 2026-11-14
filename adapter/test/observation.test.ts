import {describe, it} from 'node:test';
import assert from 'node:assert/strict';
import {
  AdapterGame,
  SUPPORTED_MAPS,
  Seat,
  createGame,
  serializeObservation,
  _unsafeGetEngineGame,
} from '../src/index';

function chooseDoneOrZero(game: AdapterGame, seat: Seat): void {
  const cur = game.choice(seat);
  if (cur.kind === 'multi_card') {
    const doneIdx = cur.candidates.findIndex((c) => c.id === 'done');
    game.choose(seat, doneIdx !== -1 ? doneIdx : 0);
  } else {
    game.choose(seat, 0);
  }
}

function advanceUntilPhase(game: AdapterGame, targetPhase: string, maxSteps = 5000): void {
  const engine = _unsafeGetEngineGame(game);
  let steps = 0;
  while (engine.phase !== targetPhase && engine.phase !== 'end' && steps < maxSteps) {
    const seats = game.pending();
    if (seats.length === 0) {
      break;
    }
    const seat = seats[0];
    const cur = game.choice(seat);
    if (cur.kind === 'branch') {
      const passIdx = cur.candidates.findIndex((c) => /pass/i.test(c.label));
      game.choose(seat, passIdx !== -1 ? passIdx : 0);
    } else {
      chooseDoneOrZero(game, seat);
    }
    steps += 1;
  }
}

function finishInitialResearchWithProjectBuys(game: AdapterGame): void {
  // For both seats: pick corp 0, preludes 0 & 1, buy 1 project card, then done.
  for (const seat of [0, 1] as const) {
    // corp
    game.choose(seat, 0);
    // prelude 1 & 2
    game.choose(seat, 0);
    game.choose(seat, 0);
    const donePreludeIdx = game.choice(seat).candidates.findIndex((c) => c.id === 'done');
    assert.ok(donePreludeIdx >= 0);
    game.choose(seat, donePreludeIdx);
    // buy 1 project card
    game.choose(seat, 0);
    const doneProjIdx = game.choice(seat).candidates.findIndex((c) => c.id === 'done');
    assert.ok(doneProjIdx >= 0);
    game.choose(seat, doneProjIdx);
  }
}

function advanceToNextDrafting(game: AdapterGame): void {
  const engine = _unsafeGetEngineGame(game);
  // Complete any initial research / preludes first, then pass through gen 1 action phase to gen 2 drafting
  while (engine.phase !== 'drafting' && engine.phase !== 'end') {
    const seats = game.pending();
    if (seats.length === 0) {
      break;
    }
    const seat = seats[0];
    const cur = game.choice(seat);
    if (cur.kind === 'branch') {
      const passIdx = cur.candidates.findIndex((c) => /pass/i.test(c.label));
      game.choose(seat, passIdx !== -1 ? passIdx : 0);
    } else {
      chooseDoneOrZero(game, seat);
    }
  }
  assert.equal(engine.phase, 'drafting');
}

describe('Ticket 15: Actor observation, history, and leakage tests', () => {
  it('1. yields byte-identical observation when opponent hand or deck order changes', () => {
    const game = createGame({seedKey: 'leak-hand-deck-1', map: 'tharsis'});
    finishInitialResearchWithProjectBuys(game);
    advanceUntilPhase(game, 'action');

    const engine = _unsafeGetEngineGame(game) as unknown as {
      players: Array<{cardsInHand: Array<{name: string}>}>;
      projectDeck: {drawPile: Array<{name: string}>};
    };
    assert.ok(engine.players[1].cardsInHand.length >= 1, 'Expected seat 1 to hold at least 1 card');
    assert.ok(engine.projectDeck.drawPile.length >= 5, 'Expected cards in project drawPile');

    const obsBefore = serializeObservation(game.observe(0));

    // Mutate only deck order
    engine.projectDeck.drawPile.reverse();
    const obsAfterDeckReverse = serializeObservation(game.observe(0));
    assert.equal(obsAfterDeckReverse, obsBefore);

    // Swap opponent's hand card with an unseen card from the drawPile
    const origOppCard = engine.players[1].cardsInHand[0];
    const swapIdx = engine.projectDeck.drawPile.findIndex((c) => c.name !== origOppCard.name);
    assert.ok(swapIdx >= 0);
    const replacement = engine.projectDeck.drawPile[swapIdx];
    engine.players[1].cardsInHand[0] = replacement;
    engine.projectDeck.drawPile[swapIdx] = origOppCard;
    assert.notEqual(engine.players[1].cardsInHand[0].name, origOppCard.name);

    const obsAfterHandSwap = serializeObservation(game.observe(0));
    assert.equal(obsAfterHandSwap, obsBefore);
  });

  it('2. hides opponent simultaneous draft pick before actor picks (deferred and non-deferred)', () => {
    // Part A: deferred by adapter (seat 1 picks first)
    const gameA = createGame({seedKey: 'leak-draft-simultaneous-1', map: 'elysium'});
    advanceToNextDrafting(gameA);
    assert.deepEqual(gameA.pending(), [0, 1]);

    const obs0BeforeA = serializeObservation(gameA.observe(0));
    const p1Choice = gameA.choice(1);
    assert.equal(p1Choice.kind, 'card');
    const p1PickedName = p1Choice.candidates[0].id;
    assert.equal(obs0BeforeA.includes(p1PickedName), false, 'Precondition: seat 1 card not already in seat 0 obs');

    gameA.choose(1, 0); // Deferred by adapter while seat 0 still pending
    assert.deepEqual(gameA.pending(), [0]);
    const obs0AfterDeferredA = serializeObservation(gameA.observe(0));
    assert.equal(obs0AfterDeferredA, obs0BeforeA);
    assert.equal(obs0AfterDeferredA.includes(p1PickedName), false);

    // Part B: even when NOT deferred (opponent processes directly on engine before seat 0 picks)
    const gameB = createGame({seedKey: 'leak-draft-simultaneous-1', map: 'elysium'});
    advanceToNextDrafting(gameB);
    const engineB = _unsafeGetEngineGame(gameB);
    const obs0BeforeB = serializeObservation(gameB.observe(0));
    assert.equal(obs0BeforeB.includes(p1PickedName), false);

    engineB.players[1].process({type: 'card', cards: [p1PickedName]});
    const obs0AfterDirectB = serializeObservation(gameB.observe(0));
    assert.equal(obs0AfterDirectB, obs0BeforeB);
    assert.equal(obs0AfterDirectB.includes(p1PickedName), false);
  });

  it('3. reveals an opponent card in actor observation only from the event that publicly reveals it', () => {
    const path = require('node:path') as typeof import('node:path');
    const engineRoot = path.resolve(__dirname, '../../../tools/simbench/engine/build/src');
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const {PowerPlant} = require(path.join(engineRoot, 'server/cards/base/PowerPlant.js'));

    const game = createGame({seedKey: 'leak-reveal-event-1', map: 'tharsis'});
    finishInitialResearchWithProjectBuys(game);

    const engine = _unsafeGetEngineGame(game) as unknown as {
      phase: string;
      activePlayer: {color: string};
      players: Array<{
        color: string;
        megaCredits: number;
        cardsInHand: Array<{name: string}>;
        playedCards: {asArray(): Array<{name: string}>};
      }>;
    };

    // Put a card not already seen by seat 0 into seat 1's hand while still in preludes,
    // before seat 1's ACTION prompt is generated.
    const secretCard = new PowerPlant();
    const obs0Initially = serializeObservation(game.observe(0));
    assert.equal(obs0Initially.includes(secretCard.name), false, 'Card must not be in seat 0 obs initially');

    engine.players[1].cardsInHand = [secretCard];
    engine.players[1].megaCredits = 20;

    // Advance through preludes and pass seat 0 until it is seat 1's turn in ACTION phase
    while (engine.phase !== 'action' || game.pending()[0] !== 1) {
      const seat = game.pending()[0];
      const cur = game.choice(seat);
      if (engine.phase === 'action' && seat === 0 && cur.kind === 'branch') {
        const passIdx = cur.candidates.findIndex((c) => /pass/i.test(c.label));
        game.choose(seat, passIdx !== -1 ? passIdx : 0);
      } else {
        chooseDoneOrZero(game, seat);
      }
    }

    // Still hidden while only in seat 1's hand!
    const obs0BeforePlay = serializeObservation(game.observe(0));
    assert.equal(obs0BeforePlay.includes(secretCard.name), false);

    // Seat 1 plays the card to tableau via adapter
    const rootChoice = game.choice(1);
    const playIdx = rootChoice.candidates.findIndex((c) => /play project card/i.test(c.label));
    assert.ok(playIdx >= 0, 'Expected Play project card option');
    game.choose(1, playIdx);

    const cardChoice = game.choice(1);
    assert.equal(cardChoice.kind, 'project_card');
    const targetIdx = cardChoice.candidates.findIndex((c) => c.id === secretCard.name);
    assert.ok(targetIdx >= 0);
    game.choose(1, targetIdx);
    // Complete payment if prompted
    if (game.pending()[0] === 1 && game.choice(1).kind === 'payment') {
      game.choose(1, 0);
    }

    // Now the card is publicly revealed in tableau and public log
    const obs0AfterPlay = game.observe(0);
    const obs0AfterPlayJson = serializeObservation(obs0AfterPlay);
    assert.equal(obs0AfterPlayJson.includes(secretCard.name), true);
    assert.ok(obs0AfterPlay.public.players[1].tableau.some((c) => c.name === secretCard.name));
    assert.ok(
      obs0AfterPlay.public.publicLog.some(
        (entry) => entry.data.some((d) => d.value === secretCard.name),
      ),
    );
  });

  it('4. records actor passed cards, initial buy, and research buys in per-actor history respecting deferral', () => {
    const game = createGame({seedKey: 'history-draft-passed-1', map: 'hellas'});

    // Seat 0 initial buy: pick corp 0, 2 preludes, buy 1 project card
    const corpChoice = game.choice(0);
    const chosenCorp = corpChoice.candidates[0].id;
    const allCorps = corpChoice.candidates.map((c) => c.id).sort();
    game.choose(0, 0);

    const pChoice1 = game.choice(0);
    const allPreludes = pChoice1.candidates.map((c) => c.id).sort();
    const p1 = pChoice1.candidates[0].id;
    game.choose(0, 0);
    const p2 = game.choice(0).candidates[0].id;
    game.choose(0, 0);
    game.choose(0, game.choice(0).candidates.findIndex((c) => c.id === 'done'));

    const projChoice = game.choice(0);
    const allProjects = projChoice.candidates.filter((c) => c.id !== 'done').map((c) => c.id).sort();
    const boughtProj = projChoice.candidates[0].id;
    game.choose(0, 0);
    game.choose(0, game.choice(0).candidates.findIndex((c) => c.id === 'done'));

    // Finish seat 1 initial buy
    finishSeatInitial(game, 1);

    const obsAfterInit = game.observe(0);
    assert.ok(obsAfterInit.history.initialBuy !== null);
    assert.deepEqual(obsAfterInit.history.initialBuy.corporationsOffered, allCorps);
    assert.equal(obsAfterInit.history.initialBuy.corporationChosen, chosenCorp);
    assert.deepEqual(
      obsAfterInit.history.initialBuy.corporationsDiscarded,
      allCorps.filter((c) => c !== chosenCorp),
    );
    assert.deepEqual(obsAfterInit.history.initialBuy.preludesOffered, allPreludes);
    assert.deepEqual(obsAfterInit.history.initialBuy.preludesChosen, [p1, p2].sort());
    assert.deepEqual(obsAfterInit.history.initialBuy.projectsOffered, allProjects);
    assert.deepEqual(obsAfterInit.history.initialBuy.projectsBought, [boughtProj]);
    assert.deepEqual(
      obsAfterInit.history.initialBuy.projectsDiscarded,
      allProjects.filter((c) => c !== boughtProj),
    );

    // Advance to Gen 2 drafting
    advanceToNextDrafting(game);
    assert.deepEqual(game.pending(), [0, 1]);

    // Test deferral: seat 1 picks draft round 1 BEFORE seat 0.
    // Seat 1's history must NOT record round 1 while held in deferred[].
    const s1Round1Packet = game.choice(1).candidates.map((c) => c.id).sort();
    const s1Round1Taken = game.choice(1).candidates[0].id;
    game.choose(1, 0);
    assert.equal(game.observe(1).history.drafts.length, 0, 'Deferred draft pick must not commit to history yet');

    // Now seat 0 picks draft round 1
    const s0Round1Packet = game.choice(0).candidates.map((c) => c.id).sort();
    const s0Round1Taken = game.choice(0).candidates[0].id;
    const s0Round1Passed = s0Round1Packet.filter((c) => c !== s0Round1Taken);
    game.choose(0, 0);

    // Both submissions have now committed in canonical order
    const obs0R1 = game.observe(0);
    assert.equal(obs0R1.history.drafts.length, 1);
    assert.deepEqual(obs0R1.history.drafts[0], {
      generation: 2,
      draftRound: 1,
      packet: s0Round1Packet,
      taken: [s0Round1Taken],
      passed: s0Round1Passed,
    });

    const obs1R1 = game.observe(1);
    assert.equal(obs1R1.history.drafts.length, 1);
    assert.deepEqual(obs1R1.history.drafts[0], {
      generation: 2,
      draftRound: 1,
      packet: s1Round1Packet,
      taken: [s1Round1Taken],
      passed: s1Round1Packet.filter((c) => c !== s1Round1Taken),
    });

    // Complete remaining draft rounds (2 and 3) and reach Gen 2 research buy
    const expectedDrafts0 = [obs0R1.history.drafts[0]];
    const engine = _unsafeGetEngineGame(game);
    while (engine.phase === 'drafting') {
      const seat = game.pending()[0];
      if (seat === 0) {
        const packet = game.choice(0).candidates.map((c) => c.id).sort();
        const taken = game.choice(0).candidates[0].id;
        expectedDrafts0.push({
          generation: engine.generation,
          draftRound: engine.draftRound,
          packet,
          taken: [taken],
          passed: packet.filter((c) => c !== taken),
        });
      }
      game.choose(seat, 0);
    }
    assert.equal(engine.phase, 'research');
    assert.deepEqual(game.observe(0).history.drafts, expectedDrafts0);
    assert.equal(expectedDrafts0.length, 3);

    // Seat 0 buys 0 or 1 cards in research buy
    const resChoice = game.choice(0);
    const offered = resChoice.candidates.filter((c) => c.id !== 'done').map((c) => c.id).sort();
    const doneIdx = resChoice.candidates.findIndex((c) => c.id === 'done');
    game.choose(0, doneIdx);

    const obs0AfterRes = game.observe(0);
    assert.equal(obs0AfterRes.history.researchBuys.length, 1);
    assert.deepEqual(obs0AfterRes.history.researchBuys[0], {
      generation: 2,
      offered,
      bought: [],
      discarded: offered,
    });
  });

  it('5. excludes opponent VP total and seed/RNG fields across all observation keys (allow-list check)', () => {
    const ALLOWED_TOP_KEYS = new Set(['actor', 'public', 'private', 'history']);
    const ALLOWED_PUBLIC_KEYS = new Set([
      'map',
      'generation',
      'phase',
      'firstPlayer',
      'activePlayer',
      'passedPlayers',
      'globalParameters',
      'deckSize',
      'discardPileSize',
      'otherDeckSizes',
      'spaces',
      'milestones',
      'awards',
      'players',
      'publicLog',
    ]);
    const ALLOWED_PLAYER_KEYS = new Set([
      'seat',
      'color',
      'terraformRating',
      'megacredits',
      'steel',
      'titanium',
      'plants',
      'energy',
      'heat',
      'megacreditProduction',
      'steelProduction',
      'titaniumProduction',
      'plantProduction',
      'energyProduction',
      'heatProduction',
      'steelValue',
      'titaniumValue',
      'cardCost',
      'cardDiscount',
      'cardsInHandCount',
      'citiesCount',
      'noTagsCount',
      'tags',
      'tableau',
      'usedActions',
      'actionsTakenThisRound',
      'actionsTakenThisGame',
      'availableBlueCardActionCount',
      'lastCardPlayed',
      'isActive',
      'passed',
      'protectedResources',
      'protectedProduction',
    ]);
    const FORBIDDEN_KEY_PATTERN = /victoryPoints|seed|rng|timer|timestamp|runId|expectedPurgeTimeMs|playerId|needsToDraft|needsToResearch|globalParameterSteps/i;

    const assertNoForbiddenKeys = (node: unknown, path = 'root'): void => {
      if (node === null || typeof node !== 'object') {
        return;
      }
      if (Array.isArray(node)) {
        node.forEach((item, idx) => assertNoForbiddenKeys(item, `${path}[${idx}]`));
        return;
      }
      for (const [k, v] of Object.entries(node as Record<string, unknown>)) {
        assert.equal(FORBIDDEN_KEY_PATTERN.test(k), false, `Forbidden key "${k}" at ${path}.${k}`);
        assertNoForbiddenKeys(v, `${path}.${k}`);
      }
    };

    const assertAllowList = (obs: ReturnType<AdapterGame['observe']>): void => {
      assert.deepEqual(new Set(Object.keys(obs)), ALLOWED_TOP_KEYS);
      assert.deepEqual(new Set(Object.keys(obs.public)), ALLOWED_PUBLIC_KEYS);
      assert.equal(obs.public.players.length, 2);
      for (const p of obs.public.players) {
        assert.deepEqual(new Set(Object.keys(p)), ALLOWED_PLAYER_KEYS);
      }
      assertNoForbiddenKeys(obs);
    };

    const game = createGame({seedKey: 'allowlist-check-1', map: 'vastitas borealis nova'});
    assertAllowList(game.observe(0));
    assertAllowList(game.observe(1));

    // Play through to game end and verify allow-list STILL holds at Phase.END (when showOtherPlayersVP would otherwise leak)
    while (true) {
      const seats = game.pending();
      if (seats.length === 0) {
        break;
      }
      const seat = seats[0];
      assertAllowList(game.observe(seat));
      game.choose(seat, 0);
    }

    assert.ok(game.result() !== null);
    assertAllowList(game.observe(0));
    assertAllowList(game.observe(1));
  });

  it('7. records cards seen through action-phase look-and-buy prompts (Inventors\' Guild) in the actor history only', () => {
    const path = require('node:path') as typeof import('node:path');
    const engineRoot = path.resolve(__dirname, '../../../tools/simbench/engine/build/src');
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const {InventorsGuild} = require(path.join(engineRoot, 'server/cards/base/InventorsGuild.js'));

    const game = createGame({seedKey: 'history-look-and-buy-1', map: 'tharsis'});
    finishInitialResearchWithProjectBuys(game);
    const engine = _unsafeGetEngineGame(game) as unknown as {
      phase: string;
      generation: number;
      players: Array<{megaCredits: number; playedCards: {push(card: unknown): void}}>;
    };
    engine.players[0].playedCards.push(new InventorsGuild());
    engine.players[0].megaCredits = 30;

    // Reach seat 0's action prompt (seat 1 passes whenever it acts).
    while (engine.phase !== 'action' || game.pending()[0] !== 0) {
      const seat = game.pending()[0];
      const cur = game.choice(seat);
      if (engine.phase === 'action' && cur.kind === 'branch') {
        const passIdx = cur.candidates.findIndex((c) => /pass/i.test(c.label));
        game.choose(seat, passIdx !== -1 ? passIdx : 0);
      } else {
        chooseDoneOrZero(game, seat);
      }
    }

    const root = game.choice(0);
    const actionIdx = root.candidates.findIndex((c) => /action from a played card/i.test(c.label));
    assert.ok(actionIdx >= 0, 'Expected a card-action branch');
    game.choose(0, actionIdx);
    const cardAction = game.choice(0);
    const igIdx = cardAction.candidates.findIndex((c) => c.id === "Inventors' Guild");
    assert.ok(igIdx >= 0, "Expected Inventors' Guild among card actions");
    game.choose(0, igIdx);

    const look = game.choice(0);
    const seen = look.candidates.filter((c) => c.id !== 'done').map((c) => c.id);
    assert.equal(seen.length, 1);
    assert.equal(game.observe(0).history.cardSelections.length, 0);
    assert.equal(serializeObservation(game.observe(1)).includes(seen[0]), false);
    game.choose(0, look.candidates.findIndex((c) => c.id === 'done'));

    assert.deepEqual(game.observe(0).history.cardSelections, [
      {generation: engine.generation, phase: 'action', offered: seen, selected: [], unselected: seen},
    ]);
    assert.deepEqual(game.observe(1).history.cardSelections, []);
    assert.equal(serializeObservation(game.observe(1)).includes(seen[0]), false, 'Discarded look card must stay hidden from the opponent');
  });


  it('6. serializes observations without error at every decision across 8 random games (2 per map; full sweep: npm run fuzz -- N --observe)', () => {
    const gamesPerMap = 2;
    let totalCalls = 0;
    let totalObserveNs = 0n;

    for (const map of SUPPORTED_MAPS) {
      for (let i = 0; i < gamesPerMap; i++) {
        const game = createGame({seedKey: `obs-400-${map}-${i}`, map});
        let stepSeed = (i + 1) * 2654435761;
        while (true) {
          const seats = game.pending();
          if (seats.length === 0) {
            break;
          }
          const seat = seats[0];
          const t0 = process.hrtime.bigint();
          const obs = game.observe(seat);
          const t1 = process.hrtime.bigint();
          totalObserveNs += t1 - t0;
          totalCalls += 1;

          const serialized = serializeObservation(obs);
          assert.ok(serialized.length > 0);
          assert.equal(obs.actor, seat);
          assert.equal(obs.public.map, map);

          // Deterministic PRNG choice
          stepSeed = (Math.imul(stepSeed, 1664525) + 1013904223) >>> 0;
          const choice = game.choice(seat);
          game.choose(seat, stepSeed % choice.candidates.length);
        }
        // Also observe both seats at game end
        const endObs0 = serializeObservation(game.observe(0));
        const endObs1 = serializeObservation(game.observe(1));
        assert.ok(endObs0.length > 0 && endObs1.length > 0);
      }
    }

    const meanMs = Number(totalObserveNs) / totalCalls / 1e6;
    console.log(`8-game observation sweep: ${totalCalls} decisions, mean observe() = ${meanMs.toFixed(4)} ms`);
    assert.ok(totalCalls > 0);
  });
});

function finishSeatInitial(game: AdapterGame, seat: Seat): void {
  game.choose(seat, 0); // corp
  game.choose(seat, 0); // prelude 1
  game.choose(seat, 0); // prelude 2
  game.choose(seat, game.choice(seat).candidates.findIndex((c) => c.id === 'done'));
  game.choose(seat, game.choice(seat).candidates.findIndex((c) => c.id === 'done')); // 0 project cards
}
