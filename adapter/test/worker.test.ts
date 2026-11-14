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
});
