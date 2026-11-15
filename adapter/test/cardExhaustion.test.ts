import {describe, it} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {
  AdapterGame,
  Seat,
  SupportedMap,
  _unsafeGetEngineGame,
  computeUnseenPool,
  createGame,
  sampleWorldFromObservation,
} from '../src/index';
import {SeededPrng} from '../src/randomPlayer';

interface RegressionCase {
  readonly seed_key: string;
  readonly map_name: string;
  readonly actions: ReadonlyArray<readonly [number, number]>;
}

const FIXTURE = path.resolve(
  __dirname,
  '../../test/fixtures/train-0027__vastitas_borealis_nova__AdapterBugError.json',
);
const regression = JSON.parse(fs.readFileSync(FIXTURE, 'utf8')) as RegressionCase;

/** Replays the saved issue #32 case to the prompt that used to fail factoring. */
function replayToEmptyDraft(): AdapterGame {
  const game = createGame({
    seedKey: regression.seed_key,
    map: regression.map_name as SupportedMap,
  });
  for (let i = 0; i < regression.actions.length; i++) {
    const [seat, candidateIndex] = regression.actions[i];
    assert.ok(
      game.pending().includes(seat as Seat),
      `replay action #${i}: seat ${seat} must be pending`,
    );
    game.choice(seat as Seat);
    game.choose(seat as Seat, candidateIndex);
  }
  return game;
}

/** Continues a game with the uniform-random policy until it ends. */
function playRandom(game: AdapterGame, seed: number, maxDecisions = 20_000): void {
  const prng = new SeededPrng(seed);
  let decisions = 0;
  while (true) {
    const pending = game.pending();
    if (pending.length === 0) {
      return;
    }
    const seat = pending[0];
    const choice = game.choice(seat);
    game.choose(seat, prng.nextInt(choice.candidates.length));
    decisions += 1;
    assert.ok(decisions <= maxDecisions, `random continuation exceeded ${maxDecisions} decisions`);
  }
}

describe('Issue #32: 0-card SelectCard when the project deck is exhausted', () => {
  it('factors the empty draft prompt as a forced done selection and finishes the game', () => {
    const game = replayToEmptyDraft();
    const engine = _unsafeGetEngineGame(game);
    assert.equal(engine.phase, 'drafting');
    assert.equal(engine.generation, 73);
    assert.equal(engine.projectDeck.drawPile.length + engine.projectDeck.discardPile.length, 0);
    assert.deepEqual(game.pending(), [0, 1]);

    // Seat 1 was dealt no cards; its only legal decision is the forced empty selection.
    const empty = game.choice(1);
    assert.equal(empty.kind, 'multi_card');
    assert.equal(empty.prompt, 'Select a card to keep and pass the rest to red');
    assert.deepEqual(empty.candidates.map((c) => c.id), ['done']);

    // Seat 0's real packet is unaffected.
    const real = game.choice(0);
    assert.equal(real.kind, 'card');
    assert.equal(real.candidates.length, 4);

    game.choose(0, 0);
    game.choose(1, 0);
    assert.equal(engine.phase, 'research');
    assert.deepEqual(game.pending(), [0, 1]);

    // The empty response must survive a checkpoint + suffix replay.
    const cloned = game.clone();
    for (const seat of [0, 1] as const) {
      assert.deepEqual(cloned.observe(seat), game.observe(seat));
      assert.deepEqual(cloned.choice(seat), game.choice(seat));
    }

    playRandom(game, 0x32_c0ffee);
    const result = game.result();
    assert.ok(result !== null);
    assert.ok(result.generations >= 73);
    assert.deepEqual(game.pending(), []);
  });

  it('samples and determinizes both seats with an empty project deck', () => {
    const game = replayToEmptyDraft();
    for (const seat of [0, 1] as const) {
      const clone = game.clone();
      const obs = clone.observe(seat);
      const pool = computeUnseenPool(obs);
      assert.equal(pool.deckSize, 0);
      assert.equal(pool.discardPileSize, 0);

      const world = sampleWorldFromObservation(obs, 0x51_0000 + seat);
      assert.equal(world.projectDeck.length, 0);
      clone.determinize(seat, world);

      // The empty prompt still factors in the determinized world, and play can continue.
      assert.deepEqual(clone.choice(1).candidates.map((c) => c.id), ['done']);
      playRandom(clone, 0xbeef + seat);
      const result = clone.result();
      assert.ok(result !== null);
      assert.ok(result.generations >= 73);
    }
  });

  it('clamps a partial selection when the dealt cards are below the prompt minimum', () => {
    const game = createGame({seedKey: 'zero-option-select-card', map: 'tharsis'});
    const engine = _unsafeGetEngineGame(game);
    const engineRoot = path.resolve(__dirname, '../../../tools/simbench/engine/build/src');
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const {SelectCard} = require(path.join(engineRoot, 'server/inputs/SelectCard.js')) as {
      SelectCard: new (
        title: string,
        buttonLabel: string,
        cards: ReadonlyArray<unknown>,
        config: {min: number; max: number},
      ) => {
        andThen(cb: (cards: ReadonlyArray<unknown>) => undefined): unknown;
        config: {min: number; max: number};
      };
    };
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const {FusionPower} = require(path.join(engineRoot, 'server/cards/base/FusionPower.js')) as {
      FusionPower: new () => unknown;
    };
    const p0 = engine.players[0] as unknown as {
      clearWaitingFor(): void;
      setWaitingFor(input: unknown, cb?: () => void): void;
    };
    engine.players[1].clearWaitingFor();
    p0.clearWaitingFor();

    let kept = -1;
    const prompt = new SelectCard('Keep the cards you drew', 'Keep', [new FusionPower()], {min: 2, max: 2});
    p0.setWaitingFor(
      prompt.andThen((cards) => {
        kept = cards.length;
        return undefined;
      }),
    );

    // One card dealt against min 2: the single available card is forced.
    const choice = game.choice(0);
    assert.equal(choice.kind, 'card');
    assert.equal(choice.candidates.length, 1);
    game.choose(0, 0);
    assert.equal(kept, 1);

    let keptEmpty = -1;
    const emptyPrompt = new SelectCard('Keep the cards you drew', 'Keep', [], {min: 2, max: 2});
    p0.setWaitingFor(
      emptyPrompt.andThen((cards) => {
        keptEmpty = cards.length;
        return undefined;
      }),
    );
    assert.deepEqual(game.choice(0).candidates.map((c) => c.id), ['done']);
    game.choose(0, 0);
    assert.equal(keptEmpty, 0);
  });
});
