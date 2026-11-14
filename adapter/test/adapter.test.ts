import {describe, it} from 'node:test';
import assert from 'node:assert/strict';
import {
  AdapterBugError,
  SUPPORTED_MAPS,
  SupportedMap,
  createGame,
  playRandomGame,
  _unsafeGetEngineGame,
} from '../src/index';

describe('Slice 1: createGame map & training format validation', () => {
  it('accepts all four supported maps in training format', () => {
    for (const map of SUPPORTED_MAPS) {
      const game = createGame({seedKey: `train-init-${map}`, map});
      assert.deepEqual(game.pending(), [0, 1]);
      assert.equal(game.result(), null);

      const engine = _unsafeGetEngineGame(game);
      assert.equal(engine.players.length, 2);
      assert.equal(engine.gameOptions.boardName, map);
      assert.equal(engine.gameOptions.draftVariant, true);
      assert.equal(engine.gameOptions.preludeExtension, true);
      assert.equal(engine.gameOptions.corporateEra, true);
      assert.equal(engine.gameOptions.showOtherPlayersVP, false);
      assert.equal(engine.gameOptions.initialDraftVariant, false);
      assert.equal(engine.gameOptions.preludeDraftVariant, false);
      assert.equal(engine.gameOptions.venusNextExtension, false);
      assert.equal(engine.gameOptions.coloniesExtension, false);
      assert.equal(engine.gameOptions.turmoilExtension, false);
      assert.equal(engine.gameOptions.promoCardsOption, false);
      assert.equal(engine.gameOptions.communityCardsOption, false);
      assert.equal(engine.gameOptions.aresExtension, false);
      assert.equal(engine.gameOptions.moonExpansion, false);
      assert.equal(engine.gameOptions.pathfindersExpansion, false);
      assert.equal(engine.gameOptions.ceoExtension, false);
      assert.equal(engine.gameOptions.starWarsExpansion, false);
      assert.equal(engine.gameOptions.underworldExpansion, false);
      assert.equal(engine.gameOptions.deltaProjectExpansion, false);
      assert.equal(engine.gameOptions.prelude2Expansion, false);
      assert.equal(engine.gameOptions.expansions.corpera, true);
      assert.equal(engine.gameOptions.expansions.prelude, true);
      for (const [k, v] of Object.entries(engine.gameOptions.expansions)) {
        if (k !== 'corpera' && k !== 'prelude') {
          assert.equal(v, false, `expansion ${k} must be false`);
        }
      }
    }
  });

  it('rejects amazonis p. and unsupported maps with a clear error', () => {
    const unsupported = ['amazonis p.', 'vastitas borealis', 'utopia planitia', 'mars', ''];
    for (const badMap of unsupported) {
      assert.throws(
        () => createGame({seedKey: 'train-1', map: badMap as SupportedMap}),
        (err: unknown) => {
          assert.ok(err instanceof Error);
          assert.match(err.message, /Unsupported map/i);
          return true;
        },
      );
    }
  });

  it('rejects empty seedKey and any non-training-format options', () => {
    assert.throws(
      () => createGame({seedKey: '', map: 'tharsis'}),
      (err: unknown) => err instanceof Error && /seedKey/i.test(err.message),
    );

    const badOptionCases: Array<Record<string, unknown>> = [
      {seedKey: 'train-1', map: 'tharsis', draftVariant: false},
      {seedKey: 'train-1', map: 'tharsis', preludeExtension: false},
      {seedKey: 'train-1', map: 'tharsis', corporateEra: false},
      {seedKey: 'train-1', map: 'tharsis', showOtherPlayersVP: true},
      {seedKey: 'train-1', map: 'tharsis', players: 3},
      {seedKey: 'train-1', map: 'tharsis', expansions: {venus: true}},
      {seedKey: 'train-1', map: 'tharsis', venusNextExtension: true},
    ];
    for (const bad of badOptionCases) {
      assert.throws(
        () => createGame(bad as unknown as {seedKey: string; map: SupportedMap}),
        (err: unknown) => err instanceof Error && /option|training format/i.test(err.message),
      );
    }
  });
});

describe('Slice 2: Legality examples (payment canonicalization, research buy, ocean spaces)', () => {
  const path = require('node:path') as typeof import('node:path');
  const engineRoot = path.resolve(__dirname, '../../../tools/simbench/engine/build/src');
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const {FusionPower} = require(path.join(engineRoot, 'server/cards/base/FusionPower.js'));
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const {GeothermalPower} = require(path.join(engineRoot, 'server/cards/base/GeothermalPower.js'));
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const {SolarPower} = require(path.join(engineRoot, 'server/cards/base/SolarPower.js'));
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const {SubterraneanReservoir} = require(path.join(engineRoot, 'server/cards/base/SubterraneanReservoir.js'));
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const {SelectProjectCardToPlay} = require(path.join(engineRoot, 'server/inputs/SelectProjectCardToPlay.js'));
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const {ChooseCards} = require(path.join(engineRoot, 'server/deferredActions/ChooseCards.js'));

  it('offers exactly {2 steel + 10 MC, 0 steel + 14 MC} for a 14-cost building card with 2 steel and 20 MC', () => {
    const game = createGame({seedKey: 'legality-steel-14', map: 'tharsis'});
    const engine = _unsafeGetEngineGame(game);
    const p0 = engine.players[0] as unknown as {
      megaCredits: number;
      steel: number;
      titanium: number;
      heat: number;
      canUseHeatAsMegaCredits: boolean;
      cardsInHand: Array<unknown>;
      playedCards: {push(...cards: Array<unknown>): void};
      setWaitingFor(input: unknown, cb?: () => void): void;
      clearWaitingFor(): void;
      getWaitingFor(): unknown;
    };

    p0.clearWaitingFor();
    p0.playedCards.push(new GeothermalPower(), new SolarPower());
    p0.megaCredits = 20;
    p0.steel = 2;
    p0.titanium = 0;
    p0.heat = 0;
    p0.canUseHeatAsMegaCredits = false;
    p0.cardsInHand = [new FusionPower()];

    let processedCard = false;
    const selectPlay = new SelectProjectCardToPlay(p0).andThen(() => {
      processedCard = true;
      return undefined;
    });
    p0.setWaitingFor(selectPlay, () => {});

    const cardChoice = game.choice(0);
    assert.equal(cardChoice.kind, 'project_card');
    assert.equal(cardChoice.candidates.length, 1);
    assert.equal(cardChoice.candidates[0].id, 'Fusion Power');

    game.choose(0, 0);

    const payChoice = game.choice(0);
    assert.equal(payChoice.kind, 'payment');
    assert.equal(payChoice.candidates.length, 2);

    const ids = payChoice.candidates.map((c) => c.id).sort();
    assert.deepEqual(ids, [
      'megacredits=10,steel=2',
      'megacredits=14',
    ]);

    game.choose(0, 0);
    assert.equal(processedCard, true);
  });

  it('excludes buy candidates in research buy once remaining money < card cost (3)', () => {
    const game = createGame({seedKey: 'legality-research-buy', map: 'tharsis'});
    const engine = _unsafeGetEngineGame(game);
    const p0 = engine.players[0] as unknown as {
      megaCredits: number;
      heat: number;
      canUseHeatAsMegaCredits: boolean;
      cardCost: number;
      setWaitingFor(input: unknown, cb?: () => void): void;
      clearWaitingFor(): void;
      getWaitingFor(): unknown;
      defer(action: unknown): void;
    };
    const cards = [
      new FusionPower(),
      new GeothermalPower(),
      new SolarPower(),
      new SubterraneanReservoir(),
    ];

    p0.clearWaitingFor();
    p0.megaCredits = 8;
    p0.heat = 0;
    p0.canUseHeatAsMegaCredits = false;

    const chooseInput = new ChooseCards(p0, cards, {paying: true, keepMax: 4}).execute();
    p0.setWaitingFor(chooseInput, () => {});

    // 8 MC -> can buy up to 2 cards; 4 cards + done = 5 candidates
    const c0 = game.choice(0);
    assert.equal(c0.kind, 'multi_card');
    assert.equal(c0.candidates.length, 5);
    assert.equal(c0.candidates[c0.candidates.length - 1].id, 'done');

    // Buy 1st card (5 MC left >= 3) -> 3 cards + done = 4 candidates
    game.choose(0, 0);
    const c1 = game.choice(0);
    assert.equal(c1.kind, 'multi_card');
    assert.equal(c1.candidates.length, 4);
    assert.equal(c1.candidates[c1.candidates.length - 1].id, 'done');

    // Buy 2nd card (2 MC left < 3) -> buy excluded, only 'done' remains
    game.choose(0, 0);
    const c2 = game.choice(0);
    assert.equal(c2.kind, 'multi_card');
    assert.deepEqual(c2.candidates.map((c) => c.id), ['done']);

    game.choose(0, 0); // done
  });

  it('enumerates ocean candidates equal to the engine SelectSpace spaces', () => {
    const game = createGame({seedKey: 'legality-ocean-spaces', map: 'tharsis'});
    const engine = _unsafeGetEngineGame(game);
    const p0 = engine.players[0] as unknown as {
      megaCredits: number;
      steel: number;
      titanium: number;
      heat: number;
      cardsInHand: Array<unknown>;
      setWaitingFor(input: unknown, cb?: () => void): void;
      clearWaitingFor(): void;
      getWaitingFor(): {type: string; spaces?: ReadonlyArray<{id: string}>} | undefined;
    };

    p0.clearWaitingFor();
    p0.megaCredits = 20;
    p0.steel = 0;
    p0.titanium = 0;
    p0.heat = 0;
    p0.cardsInHand = [new SubterraneanReservoir()];

    const selectPlay = new SelectProjectCardToPlay(p0);
    p0.setWaitingFor(selectPlay, () => {
      (engine as unknown as {deferredActions: {runAll(cb: () => void): void}}).deferredActions.runAll(() => {});
    });

    const cardChoice = game.choice(0);
    assert.equal(cardChoice.kind, 'project_card');
    game.choose(0, 0); // unique payment -> submits and triggers PlaceOceanTile

    const waiting = p0.getWaitingFor();
    assert.ok(waiting && waiting.type === 'space' && Array.isArray(waiting.spaces));
    const expectedIds = waiting.spaces.map((s) => s.id);

    const spaceChoice = game.choice(0);
    assert.equal(spaceChoice.kind, 'space');
    assert.deepEqual(
      spaceChoice.candidates.map((c) => c.id),
      expectedIds,
    );
  });

  it('caps initial project card buys by chosen corporation starting megacredits', () => {
    const game = createGame({seedKey: 'legality-initial-buy', map: 'tharsis'});
    const engine = _unsafeGetEngineGame(game);
    const p0 = engine.players[0] as unknown as {
      dealtCorporationCards: Array<{name: string; startingMegaCredits: number; cardCost?: number}>;
      cardCost: number;
    };

    // Pick first corporation (index 0) and see its starting MC
    const corpChoice = game.choice(0);
    assert.equal(corpChoice.kind, 'card');
    const chosenCorpName = corpChoice.candidates[0].id;
    const chosenCorp = p0.dealtCorporationCards.find((c) => c.name === chosenCorpName)!;
    const cardCost = chosenCorp.cardCost ?? p0.cardCost;
    // Override startingMegaCredits getter on this instance so maxAffordable = 2 (< 10 dealt cards)
    Object.defineProperty(chosenCorp, 'startingMegaCredits', {value: 8, configurable: true});
    const expectedMax = Math.floor(8 / cardCost);
    assert.equal(expectedMax, 2);

    game.choose(0, 0); // pick corp
    game.choose(0, 0); // prelude 1
    game.choose(0, 0); // prelude 2
    const donePreludeIdx = game.choice(0).candidates.findIndex((c) => c.id === 'done');
    assert.ok(donePreludeIdx >= 0);
    game.choose(0, donePreludeIdx);

    // Initial project buy: 10 cards + done = 11
    assert.equal(game.choice(0).kind, 'multi_card');
    assert.equal(game.choice(0).candidates.length, 11);
    game.choose(0, 0); // buy 1 (5 MC left)
    assert.equal(game.choice(0).candidates.length, 10);
    game.choose(0, 0); // buy 2 (2 MC left < 3)
    assert.deepEqual(game.choice(0).candidates.map((c) => c.id), ['done']);
    game.choose(0, 0); // done -> submits SelectInitialCards cleanly
    assert.deepEqual(game.pending(), [1]);
  });
});

describe('Slice 3: Determinism & canonical seat order', () => {
  it('reproduces identical final scores for the same seedKey and candidate index sequence', () => {
    const run1 = playRandomGame({
      seedKey: 'det-replay-42',
      map: 'tharsis',
      driverSeed: 123456,
      recordChoices: true,
    });
    assert.ok(run1.result !== null);
    assert.ok(run1.choices && run1.choices.length > 0);
    assert.ok(run1.result.generations >= 1);

    // Replay the exact recorded (seat, candidateIndex) sequence on a fresh game with same seedKey
    const game2 = createGame({seedKey: 'det-replay-42', map: 'tharsis'});
    for (const step of run1.choices) {
      const seats = game2.pending();
      assert.ok(seats.length >= 1);
      // Verify canonical seat order whenever multiple seats are pending
      if (seats.length === 2) {
        assert.deepEqual(seats, [0, 1]);
      }
      assert.equal(seats[0], step.seat);
      game2.choose(step.seat, step.candidateIndex);
    }

    assert.deepEqual(game2.pending(), []);
    assert.deepEqual(game2.result(), run1.result);
  });
});

describe('Slice 4: Engine rejection surfaces as AdapterBugError with seed and history', () => {
  it('throws AdapterBugError carrying seedKey and full action history on forced bad response', () => {
    const game = createGame({seedKey: 'forced-bad-response-seed', map: 'elysium'});
    const engine = _unsafeGetEngineGame(game);

    // Complete initial card selections and preludes via the adapter to accumulate non-empty history
    while (engine.phase === 'initial_research' || engine.phase === 'research' || engine.phase === 'preliminary' || engine.phase === 'preludes') {
      const seats = game.pending();
      if (seats.length === 0) {
        break;
      }
      const seat = seats[0];
      const cur = game.choice(seat);
      if (cur.kind === 'multi_card') {
        const doneIdx = cur.candidates.findIndex((c) => c.id === 'done');
        game.choose(seat, doneIdx !== -1 ? doneIdx : 0);
      } else {
        game.choose(seat, 0);
      }
    }

    // Now in ACTION phase on the active seat: select Standard Projects -> Power Plant (costs 11 MC),
    // then zero out the player's megacredits AFTER choice() enumerated candidates so player.process() rejects.
    const seat = game.pending()[0];
    const player = engine.players[seat] as unknown as {
      megaCredits: number;
      steel: number;
      titanium: number;
      heat: number;
    };
    player.megaCredits = 25;
    player.steel = 0;
    player.titanium = 0;
    player.heat = 0;

    const rootChoice = game.choice(seat);
    assert.equal(rootChoice.kind, 'branch');
    const stdProjIdx = rootChoice.candidates.findIndex((c) => /standard projects/i.test(c.label));
    assert.ok(stdProjIdx >= 0, 'Expected Standard projects branch in ACTION phase');
    game.choose(seat, stdProjIdx);

    const spChoice = game.choice(seat);
    assert.equal(spChoice.kind, 'project_card');
    const powerPlantIdx = spChoice.candidates.findIndex((c) => c.id === 'Power Plant:SP');
    assert.ok(powerPlantIdx >= 0, 'Expected Power Plant:SP candidate when player has 25 MC');

    // Corrupt player resources after candidate enumeration to force an engine rejection on submission
    player.megaCredits = 0;

    let caught: unknown;
    try {
      game.choose(seat, powerPlantIdx);
    } catch (err) {
      caught = err;
    }

    assert.ok(caught instanceof AdapterBugError, 'Expected AdapterBugError');
    assert.equal(caught.seedKey, 'forced-bad-response-seed');
    assert.equal(caught.map, 'elysium');
    assert.equal(caught.seat, seat);
    assert.ok(caught.history.length >= 3, `Expected >= 3 history entries, got ${caught.history.length}`);
    const last = caught.history[caught.history.length - 1];
    assert.equal(last.seat, seat);
    assert.ok(last.response !== undefined, 'Expected rejected response recorded in history');
    assert.match(caught.message, /forced-bad-response-seed/);
  });
});

describe('Slice 5: 400 random-vs-random games (100 per supported map)', () => {
  it('completes 100 games per supported map with 0 adapter errors, 0 step caps, and 0 deadlocks', () => {
    const gamesPerMap = 100;
    for (const map of SUPPORTED_MAPS) {
      for (let i = 0; i < gamesPerMap; i++) {
        const outcome = playRandomGame({
          seedKey: `ci-400-${map}-${i}`,
          map,
        });
        assert.ok(outcome.result !== null, `Expected terminal result for ${map}#${i}`);
        assert.equal(outcome.result.scores.length, 2);
        assert.ok(Number.isInteger(outcome.result.scores[0]));
        assert.ok(Number.isInteger(outcome.result.scores[1]));
        assert.ok(outcome.result.generations >= 1);
      }
    }
  });
});





describe('Slice 3b: Simultaneous submissions are applied in canonical seat order', () => {
  it('yields identical engine state when seat 1 completes simultaneous decisions before seat 0', () => {
    const seedKey = 'canonical-order-7';
    const map: SupportedMap = 'hellas';
    const reference = playRandomGame({seedKey, map, driverSeed: 99, recordChoices: true});
    const perSeat: [number[], number[]] = [[], []];
    for (const step of reference.choices ?? []) {
      perSeat[step.seat].push(step.candidateIndex);
    }

    // Replay the same per-seat choices, but always serve the highest pending seat first.
    const reversed = createGame({seedKey, map});
    const cursor = [0, 0];
    while (true) {
      const seats = reversed.pending();
      if (seats.length === 0) {
        break;
      }
      const seat = seats[seats.length - 1];
      reversed.choose(seat, perSeat[seat][cursor[seat]++]);
    }

    const forward = createGame({seedKey, map});
    const cursorF = [0, 0];
    for (const step of reference.choices ?? []) {
      forward.choose(step.seat, perSeat[step.seat][cursorF[step.seat]++]);
    }

    assert.deepEqual(reversed.result(), forward.result());
    const normalized = (g: typeof reversed) =>
      JSON.stringify(_unsafeGetEngineGame(g).serialize(), (k, v) => (k === 'timestamp' || k === 'createdTimeMs' || k === 'name' || k === 'timer' || k === 'lastSaveId' ? undefined : v));
    assert.equal(normalized(reversed), normalized(forward));
  });
});

describe('Slice 3c: Result applies the M€ tie-break', () => {
  it('awards a VP tie to the player with more megacredits and only declares a tie when both are equal', () => {
    const outcome = playRandomGame({seedKey: 'tiebreak-1', map: 'tharsis', driverSeed: 5});
    assert.ok(outcome.result !== null);
    // Construct a finished game and force equal VP by comparing against the engine directly.
    const game = createGame({seedKey: 'tiebreak-1', map: 'tharsis'});
    const engine = _unsafeGetEngineGame(game);
    for (const step of playRandomGame({seedKey: 'tiebreak-1', map: 'tharsis', driverSeed: 5, recordChoices: true}).choices ?? []) {
      game.choose(step.seat, step.candidateIndex);
    }
    const [a, b] = engine.players;
    const vp = (p: typeof a) => p.getVictoryPoints().total;
    // Monkeypatch VP to force a tie in VP, then vary megacredits.
    const tiedVp = vp(a);
    (b as unknown as {getVictoryPoints: () => {total: number}}).getVictoryPoints = () => ({total: tiedVp});
    (a as unknown as {getVictoryPoints: () => {total: number}}).getVictoryPoints = () => ({total: tiedVp});
    a.megaCredits = 10;
    b.megaCredits = 20;
    assert.deepEqual([game.result()?.winner, game.result()?.tie], [1, false]);
    a.megaCredits = 20;
    assert.deepEqual([game.result()?.winner, game.result()?.tie], [null, true]);
  });
});
