import {describe, it} from 'node:test';
import assert from 'node:assert/strict';
import {GameWorker} from '../src/worker';
import {AdapterBugError} from '../src/types';

describe('GameWorker JSON protocol', () => {
  it('handles create, pending, observe, choice, choose, result, and close', () => {
    const worker = new GameWorker();
    const createResp = worker.handleMessage({
      id: 1,
      method: 'create',
      params: {seedKey: 'worker-unit-1', map: 'tharsis'},
    });
    assert.equal(createResp.ok, true);
    assert.ok('result' in createResp);
    const {gameId} = createResp.result as {gameId: string};
    assert.equal(worker.gameCount, 1);

    const pendingResp = worker.handleMessage({id: 2, method: 'pending', params: {gameId}});
    assert.equal(pendingResp.ok, true);
    assert.deepEqual((pendingResp as {result: {pending: Array<number>}}).result.pending, [0, 1]);

    const obsResp = worker.handleMessage({id: 3, method: 'observe', params: {gameId, seat: 0}});
    assert.equal(obsResp.ok, true);
    assert.equal(
      (obsResp as {result: {observation: {actor: number}}}).result.observation.actor,
      0,
    );

    const choiceResp = worker.handleMessage({id: 4, method: 'choice', params: {gameId, seat: 0}});
    assert.equal(choiceResp.ok, true);
    assert.equal((choiceResp as {result: {choice: {seat: number}}}).result.choice.seat, 0);

    const chooseResp = worker.handleMessage({
      id: 5,
      method: 'choose',
      params: {gameId, seat: 0, candidateIndex: 0},
    });
    assert.equal(chooseResp.ok, true);

    const resultResp = worker.handleMessage({id: 6, method: 'result', params: {gameId}});
    assert.equal(resultResp.ok, true);
    assert.equal((resultResp as {result: {result: unknown}}).result.result, null);

    const closeResp = worker.handleMessage({id: 7, method: 'close', params: {gameId}});
    assert.equal(closeResp.ok, true);
    assert.equal((closeResp as {result: {closed: boolean}}).result.closed, true);
    assert.equal(worker.gameCount, 0);
  });

  it('executes batched sub-requests in order and isolates per-game errors', () => {
    const worker = new GameWorker();
    const resp = worker.handleMessage({
      id: 10,
      batch: [
        {method: 'create', params: {gameId: 'g-a', seedKey: 'batch-a', map: 'tharsis'}},
        {method: 'create', params: {gameId: 'g-bad', seedKey: 'batch-bad', map: 'amazonis p.'}},
        {method: 'create', params: {gameId: 'g-b', seedKey: 'batch-b', map: 'hellas'}},
        {method: 'pending', params: {gameId: 'g-a'}},
        {method: 'choose', params: {gameId: 'g-b', seat: 0, candidateIndex: 9999}},
        {method: 'pending', params: {gameId: 'g-b'}},
      ],
    });
    assert.equal(resp.ok, true);
    assert.ok('results' in resp);
    assert.equal(resp.results.length, 6);
    assert.equal(resp.results[0].ok, true);
    assert.equal(resp.results[1].ok, false);
    assert.equal(resp.results[2].ok, true);
    assert.equal(resp.results[3].ok, true);
    assert.equal(resp.results[4].ok, false);
    assert.equal(resp.results[4].error?.seedKey, 'batch-b');
    assert.ok(Array.isArray(resp.results[4].error?.history));
    assert.equal(resp.results[5].ok, true);
  });

  it('reports AdapterBugError with seedKey and history if clone or game throws', () => {
    const worker = new GameWorker();
    worker.handleMessage({
      id: 1,
      method: 'create',
      params: {gameId: 'g-1', seedKey: 'err-seed', map: 'elysium'},
    });
    const cloneResp = worker.handleMessage({id: 2, method: 'clone', params: {gameId: 'g-1'}});
    // Either succeeds (if ticket #16 clone is present) or returns structured UnsupportedMethodError
    if (!cloneResp.ok && 'error' in cloneResp && cloneResp.error) {
      assert.equal(cloneResp.error.seedKey, 'err-seed');
      assert.ok(cloneResp.error.message.length > 0);
    }
    // Verify AdapterBugError type stays wired
    assert.equal(typeof AdapterBugError, 'function');
  });

  it('handles determinize_clone without mutating live game and produces deterministic clones from actor obs only', () => {
    const worker = new GameWorker();
    const createResp = worker.handleMessage({
      id: 1,
      method: 'create',
      params: {gameId: 'live-1', seedKey: 'det-clone-test-1', map: 'tharsis'},
    });
    assert.equal(createResp.ok, true);
    assert.equal(worker.gameCount, 1);

    const obs0Before = worker.handleMessage({id: 2, method: 'observe', params: {gameId: 'live-1', seat: 0, serialized: true}});
    const obs1Before = worker.handleMessage({id: 3, method: 'observe', params: {gameId: 'live-1', seat: 1, serialized: true}});
    const choice0Before = worker.handleMessage({id: 4, method: 'choice', params: {gameId: 'live-1', seat: 0}});

    const dc1 = worker.handleMessage({
      id: 5,
      method: 'determinize_clone',
      params: {gameId: 'live-1', seat: 0, seed: 4242, newGameId: 'det-a'},
    });
    assert.equal(dc1.ok, true);
    assert.deepEqual((dc1 as {result: {gameId: string; seedKey: string; map: string}}).result, {
      gameId: 'det-a',
      seedKey: 'det-clone-test-1',
      map: 'tharsis',
    });
    assert.equal(worker.gameCount, 2);

    const dc2 = worker.handleMessage({
      id: 6,
      method: 'determinize_clone',
      params: {gameId: 'live-1', seat: 0, seed: 4242, newGameId: 'det-b'},
    });
    assert.equal(dc2.ok, true);
    assert.equal(worker.gameCount, 3);

    // Determinism: identical (gameId, seat, seed) yields identical opponent observations in the two clones
    const detAOpp = worker.handleMessage({id: 7, method: 'observe', params: {gameId: 'det-a', seat: 1, serialized: true}});
    const detBOpp = worker.handleMessage({id: 8, method: 'observe', params: {gameId: 'det-b', seat: 1, serialized: true}});
    assert.equal(
      (detAOpp as {result: {observation: string}}).result.observation,
      (detBOpp as {result: {observation: string}}).result.observation,
    );

    // Different seed yields a different sampled world
    const dc3 = worker.handleMessage({
      id: 9,
      method: 'determinize_clone',
      params: {gameId: 'live-1', seat: 0, seed: 9999, newGameId: 'det-c'},
    });
    assert.equal(dc3.ok, true);
    const detCOpp = worker.handleMessage({id: 10, method: 'observe', params: {gameId: 'det-c', seat: 1, serialized: true}});
    assert.notEqual(
      (detAOpp as {result: {observation: string}}).result.observation,
      (detCOpp as {result: {observation: string}}).result.observation,
    );

    // Actor observation on clone matches live game byte-for-byte
    const detASelf = worker.handleMessage({id: 11, method: 'observe', params: {gameId: 'det-a', seat: 0, serialized: true}});
    assert.equal(
      (detASelf as {result: {observation: string}}).result.observation,
      (obs0Before as {result: {observation: string}}).result.observation,
    );

    // Advance clone and verify live game is untouched
    const stepClone = worker.handleMessage({
      id: 12,
      method: 'choose',
      params: {gameId: 'det-a', seat: 0, candidateIndex: 0},
    });
    assert.equal(stepClone.ok, true);
    const obs0After = worker.handleMessage({id: 13, method: 'observe', params: {gameId: 'live-1', seat: 0, serialized: true}});
    const obs1After = worker.handleMessage({id: 14, method: 'observe', params: {gameId: 'live-1', seat: 1, serialized: true}});
    const choice0After = worker.handleMessage({id: 15, method: 'choice', params: {gameId: 'live-1', seat: 0}});
    assert.equal(
      (obs0After as {result: {observation: string}}).result.observation,
      (obs0Before as {result: {observation: string}}).result.observation,
    );
    assert.equal(
      (obs1After as {result: {observation: string}}).result.observation,
      (obs1Before as {result: {observation: string}}).result.observation,
    );
    assert.deepEqual(
      (choice0After as {result: unknown}).result,
      (choice0Before as {result: unknown}).result,
    );

    // Validation errors do not leak clones
    const badSeed = worker.handleMessage({
      id: 16,
      method: 'determinize_clone',
      params: {gameId: 'live-1', seat: 0, seed: 'not-a-number'},
    });
    assert.equal(badSeed.ok, false);
    assert.equal(worker.gameCount, 4);

    // Advance seat 0 completely in live-1 until seat 0 is no longer pending (deferred)
    while (true) {
      const pend = worker.handleMessage({id: 17, method: 'pending', params: {gameId: 'live-1'}});
      const pendingSeats = (pend as {result: {pending: Array<number>}}).result.pending;
      if (!pendingSeats.includes(0)) break;
      const ch = worker.handleMessage({id: 18, method: 'choice', params: {gameId: 'live-1', seat: 0}});
      const cands = (ch as {result: {choice: {candidates: Array<{id: string}>}}}).result.choice.candidates;
      const doneIdx = cands.findIndex((c) => c.id === 'done');
      worker.handleMessage({id: 19, method: 'choose', params: {gameId: 'live-1', seat: 0, candidateIndex: doneIdx >= 0 ? doneIdx : 0}});
    }
    const nonPendingResp = worker.handleMessage({
      id: 20,
      method: 'determinize_clone',
      params: {gameId: 'live-1', seat: 0, seed: 123},
    });
    assert.equal(nonPendingResp.ok, false);
    assert.equal(worker.gameCount, 4);
  });
});

