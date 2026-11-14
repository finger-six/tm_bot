from __future__ import annotations

import json
import os
import random
import signal
import subprocess
from pathlib import Path

import pytest

from tm_bot.env import (
    SUPPORTED_MAPS,
    AdapterBugError,
    BridgeError,
    BridgeProtocolError,
    GameResult,
    Op,
    WorkerClient,
    WorkerCrashError,
    WorkerPool,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ADAPTER_DIST_INDEX = REPO_ROOT / "adapter" / "dist" / "src" / "index.js"


def _run_node_random_game(seed_key: str, map_name: str, driver_seed: int = 42) -> dict:
    script = (
        f"const {{ playRandomGame }} = require({json.dumps(str(ADAPTER_DIST_INDEX))});"
        f"const out = playRandomGame({{ seedKey: {json.dumps(seed_key)}, map: {json.dumps(map_name)}, driverSeed: {driver_seed}, recordChoices: true }});"
        "process.stdout.write(JSON.stringify(out));"
    )
    proc = subprocess.run(
        ["node", "-e", script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def _replay_in_node(games: list[dict]) -> list[dict]:
    script = (
        f"const {{ createGame }} = require({json.dumps(str(ADAPTER_DIST_INDEX))});"
        "const fs = require('node:fs');"
        "const specs = JSON.parse(fs.readFileSync(0, 'utf8'));"
        "const out = specs.map((spec) => {"
        "  const g = createGame({ seedKey: spec.seedKey, map: spec.map });"
        "  for (const step of spec.choices) {"
        "    g.choose(step.seat, step.candidateIndex);"
        "  }"
        "  return g.result();"
        "});"
        "process.stdout.write(JSON.stringify(out));"
    )
    proc = subprocess.run(
        ["node", "-e", script],
        cwd=str(REPO_ROOT),
        input=json.dumps(games),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def test_full_random_game_matches_node_only_run() -> None:
    seed_key = "bridge-parity-tharsis-1"
    map_name = "tharsis"
    node_outcome = _run_node_random_game(seed_key, map_name, driver_seed=12345)
    recorded_choices = node_outcome["choices"]
    assert len(recorded_choices) > 0

    with WorkerClient() as client:
        game_id = client.create(seed_key=seed_key, map=map_name)
        seen_observations = 0

        for step_idx, step in enumerate(recorded_choices):
            seats = client.pending(game_id)
            assert len(seats) >= 1
            seat = step["seat"]
            assert seats[0] == seat

            choice = client.choice(game_id, seat)
            assert choice.seat == seat
            assert 0 <= step["candidateIndex"] < len(choice.candidates)

            # Spot-check observe() at the start and periodically throughout the game
            if step_idx == 0 or step_idx % 50 == 0:
                obs = client.observe(game_id, seat)
                assert obs["actor"] == seat
                assert obs["public"]["map"] == map_name
                seen_observations += 1

            client.choose(game_id, seat, step["candidateIndex"])

        assert seen_observations >= 2
        assert client.pending(game_id) == []
        res = client.result(game_id)
        assert res is not None
        assert res == GameResult(
            scores=(node_outcome["result"]["scores"][0], node_outcome["result"]["scores"][1]),
            winner=node_outcome["result"]["winner"],
            tie=node_outcome["result"]["tie"],
            generations=node_outcome["result"]["generations"],
        )
        client.close_game(game_id)


def test_batched_requests_play_several_games_concurrently_on_one_worker() -> None:
    specs = [
        {"seedKey": f"batch-concurrent-{i}", "map": SUPPORTED_MAPS[i]}
        for i in range(4)
    ]
    rng = random.Random(2026)

    with WorkerClient() as client:
        game_ids = client.batch([Op.create(s["seedKey"], s["map"]) for s in specs])
        assert len(game_ids) == 4
        assert len(set(game_ids)) == 4

        recorded: dict[str, list[dict[str, int]]] = {gid: [] for gid in game_ids}
        active_ids = list(game_ids)

        # Initial observation check across all games in a single batch
        init_obs = client.batch([Op.observe(gid, 0) for gid in game_ids])
        for obs, spec in zip(init_obs, specs):
            assert obs["actor"] == 0
            assert obs["public"]["map"] == spec["map"]

        while active_ids:
            pending_lists = client.batch([Op.pending(gid) for gid in active_ids])
            next_active: list[str] = []
            choice_ops = []
            pending_seats: list[tuple[str, int]] = []

            for gid, seats in zip(active_ids, pending_lists):
                if seats:
                    seat = seats[0]
                    next_active.append(gid)
                    pending_seats.append((gid, seat))
                    choice_ops.append(Op.choice(gid, seat))

            if not choice_ops:
                break

            choices = client.batch(choice_ops)
            choose_ops = []
            for (gid, seat), choice in zip(pending_seats, choices):
                assert choice.seat == seat
                idx = rng.randrange(len(choice.candidates))
                recorded[gid].append({"seat": seat, "candidateIndex": idx})
                choose_ops.append(Op.choose(gid, seat, idx))

            client.batch(choose_ops)
            active_ids = next_active

        results = client.batch([Op.result(gid) for gid in game_ids])
        closed = client.batch([Op.close(gid) for gid in game_ids])
        assert closed == [True, True, True, True]

    node_expected = _replay_in_node(
        [
            {"seedKey": spec["seedKey"], "map": spec["map"], "choices": recorded[gid]}
            for spec, gid in zip(specs, game_ids)
        ]
    )
    for res, exp in zip(results, node_expected):
        assert res == GameResult(
            scores=(exp["scores"][0], exp["scores"][1]),
            winner=exp["winner"],
            tie=exp["tie"],
            generations=exp["generations"],
        )


def test_game_session_interface() -> None:
    with WorkerClient() as client:
        with client.open("session-test-1", "elysium") as game:
            seats = game.pending()
            assert seats == [0, 1]
            obs = game.observe(0)
            assert obs["public"]["map"] == "elysium"
            ch = game.choice(0)
            assert len(ch.candidates) > 0
            game.choose(0, 0)
            assert game.result() is None


def test_adapter_bug_surfaces_with_seed_and_history_while_other_games_survive(tmp_path: Path) -> None:
    adapter_js = str(REPO_ROOT / "adapter" / "dist" / "src" / "adapter.js")
    preload = tmp_path / "force_bug_preload.cjs"
    preload.write_text(
        f"""
const adapter = require({json.dumps(adapter_js)});
const origCreateGame = adapter.createGame;
adapter.createGame = function(opts) {{
  const game = origCreateGame(opts);
  if (opts && opts.seedKey === 'forced-bug-seed') {{
    const engine = adapter._unsafeGetEngineGame(game);
    const origChoose = game.choose.bind(game);
    const origChoice = game.choice.bind(game);
    game.choice = function(seat) {{
      const ch = origChoice(seat);
      if (engine.phase === 'action' && ch.kind === 'branch') {{
        engine.players[seat].megaCredits = 25;
        engine.players[seat].steel = 0;
        engine.players[seat].titanium = 0;
        engine.players[seat].heat = 0;
      }}
      return ch;
    }};
    game.choose = function(seat, candidateIndex) {{
      const ch = origChoice(seat);
      if (ch.kind === 'project_card' && ch.candidates[candidateIndex]?.id === 'Power Plant:SP') {{
        engine.players[seat].megaCredits = 0;
      }}
      return origChoose(seat, candidateIndex);
    }};
  }}
  return game;
}};
""".lstrip(),
        encoding="utf-8",
    )

    with WorkerClient(node_args=["--require", str(preload)]) as client:
        bug_gid = client.create("forced-bug-seed", "elysium")
        ok_gid = client.create("healthy-surviving-seed", "tharsis")

        # Advance bug_gid through initial setup into action phase so history accumulates.
        while True:
            seats = client.pending(bug_gid)
            assert seats
            seat = seats[0]
            obs = client.observe(bug_gid, seat)
            if obs["public"]["phase"] == "action":
                break
            ch = client.choice(bug_gid, seat)
            if ch.kind == "multi_card":
                done_idx = next((i for i, c in enumerate(ch.candidates) if c.id == "done"), 0)
                client.choose(bug_gid, seat, done_idx)
            else:
                client.choose(bug_gid, seat, 0)

        active_seat = client.pending(bug_gid)[0]
        root_choice = client.choice(bug_gid, active_seat)
        std_idx = next(i for i, c in enumerate(root_choice.candidates) if "standard projects" in c.label.lower())
        client.choose(bug_gid, active_seat, std_idx)

        sp_choice = client.choice(bug_gid, active_seat)
        power_plant_idx = next(i for i, c in enumerate(sp_choice.candidates) if c.id == "Power Plant:SP")

        # Batched call: bug_gid triggers AdapterBugError, while ok_gid succeeds in the same batch.
        batch_res = client.batch(
            [
                Op.choose(bug_gid, active_seat, power_plant_idx),
                Op.pending(ok_gid),
            ],
            raise_on_error=False,
        )
        assert not batch_res[0].ok
        assert isinstance(batch_res[0].error, AdapterBugError)
        bug_err = batch_res[0].error
        assert bug_err.seed_key == "forced-bug-seed"
        assert bug_err.map == "elysium"
        assert bug_err.seat == active_seat
        assert len(bug_err.history) >= 3
        assert bug_err.history[-1]["seat"] == active_seat
        assert "response" in bug_err.history[-1]

        assert batch_res[1].ok
        assert batch_res[1].value == [0, 1]

        # Single-call AdapterBugError also raises directly, and ok_gid still works afterward.
        with pytest.raises(AdapterBugError) as exc_info:
            client.choose(bug_gid, active_seat, power_plant_idx)
        assert exc_info.value.seed_key == "forced-bug-seed"
        assert len(exc_info.value.history) >= 3

        assert client.pending(ok_gid) == [0, 1]
        client.choose(ok_gid, 0, 0)
        assert client.pending(ok_gid) == [0, 1]


def test_bad_json_response_raises_protocol_error(tmp_path: Path) -> None:
    bad_worker = tmp_path / "bad_worker.cjs"
    bad_worker.write_text(
        """
const readline = require('node:readline');
const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
let count = 0;
rl.on('line', (line) => {
  count += 1;
  const req = JSON.parse(line);
  if (count === 1) {
    process.stdout.write(JSON.stringify({ id: req.id, ok: true, result: { gameId: 'g-1', seedKey: 's1', map: 'tharsis' } }) + '\\n');
    return;
  }
  process.stderr.write('simulated bad response on stderr\\n');
  process.stdout.write('NOT_JSON_PAYLOAD\\n');
});
""".lstrip(),
        encoding="utf-8",
    )

    with WorkerClient(worker_script=bad_worker) as client:
        gid = client.create("s1", "tharsis")
        with pytest.raises(BridgeProtocolError) as exc_info:
            client.pending(gid)
        assert "NOT_JSON_PAYLOAD" in (exc_info.value.raw_response or "")


def test_killed_worker_raises_worker_crash_error_with_stderr_tail(tmp_path: Path) -> None:
    # Case 1: killing a real worker subprocess via OS signal
    with WorkerClient() as client:
        gid = client.create("crash-kill-seed", "tharsis")
        client.choose(gid, 0, 0)
        os.kill(client.pid, signal.SIGTERM)
        with pytest.raises(WorkerCrashError) as exc_info:
            client.pending(gid)
        assert exc_info.value.seed_key == "crash-kill-seed"
        assert exc_info.value.game_id == gid

    # Case 2: worker crashing mid-request with diagnostic stderr output
    crashing_worker = tmp_path / "crashing_worker.cjs"
    crashing_worker.write_text(
        """
const readline = require('node:readline');
const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
let count = 0;
rl.on('line', (line) => {
  count += 1;
  const req = JSON.parse(line);
  if (count === 1) {
    process.stdout.write(JSON.stringify({ id: req.id, ok: true, result: { gameId: 'g-1', seedKey: 'crash-stderr-seed', map: 'tharsis' } }) + '\\n');
    return;
  }
  process.stderr.write('FATAL: simulated engine segfault\\n');
  process.exit(42);
});
""".lstrip(),
        encoding="utf-8",
    )

    with WorkerClient(worker_script=crashing_worker) as client:
        gid = client.create("crash-stderr-seed", "tharsis")
        with pytest.raises(WorkerCrashError) as exc_info:
            client.pending(gid)
        assert exc_info.value.returncode == 42
        assert "simulated engine segfault" in exc_info.value.stderr_tail
        assert exc_info.value.seed_key == "crash-stderr-seed"


def test_worker_pool_runs_games_across_multiple_workers_and_shuts_down_cleanly() -> None:
    specs = [
        {"seedKey": f"pool-game-{i}", "map": SUPPORTED_MAPS[i]}
        for i in range(4)
    ]
    rng = random.Random(9001)

    with WorkerPool(num_workers=2) as pool:
        assert pool.num_workers == 2
        pids = pool.worker_pids
        assert len(pids) == 2
        assert len(set(pids)) == 2
        assert pool.alive_worker_count() == 2

        game_ids = pool.batch([Op.create(s["seedKey"], s["map"]) for s in specs])
        assert len(game_ids) == 4
        assignment = pool.game_worker_map
        assert set(assignment.values()) == {0, 1}
        assert sum(1 for w in assignment.values() if w == 0) == 2
        assert sum(1 for w in assignment.values() if w == 1) == 2

        recorded: dict[str, list[dict[str, int]]] = {gid: [] for gid in game_ids}
        active_ids = list(game_ids)

        while active_ids:
            pending_lists = pool.batch([Op.pending(gid) for gid in active_ids])
            next_active: list[str] = []
            choice_ops = []
            pending_seats: list[tuple[str, int]] = []

            for gid, seats in zip(active_ids, pending_lists):
                if seats:
                    seat = seats[0]
                    next_active.append(gid)
                    pending_seats.append((gid, seat))
                    choice_ops.append(Op.choice(gid, seat))

            if not choice_ops:
                break

            choices = pool.batch(choice_ops)
            choose_ops = []
            for (gid, seat), choice in zip(pending_seats, choices):
                idx = rng.randrange(len(choice.candidates))
                recorded[gid].append({"seat": seat, "candidateIndex": idx})
                choose_ops.append(Op.choose(gid, seat, idx))

            pool.batch(choose_ops)
            active_ids = next_active

        results = pool.batch([Op.result(gid) for gid in game_ids])
        assert pool.batch([Op.close(gid) for gid in game_ids]) == [True, True, True, True]
        assert pool.active_games == 0

    assert pool.alive_worker_count() == 0

    node_expected = _replay_in_node(
        [
            {"seedKey": spec["seedKey"], "map": spec["map"], "choices": recorded[gid]}
            for spec, gid in zip(specs, game_ids)
        ]
    )
    for res, exp in zip(results, node_expected):
        assert res == GameResult(
            scores=(exp["scores"][0], exp["scores"][1]),
            winner=exp["winner"],
            tie=exp["tie"],
            generations=exp["generations"],
        )


def test_worker_pool_other_worker_survives_single_worker_crash() -> None:
    with WorkerPool(num_workers=2) as pool:
        gid0 = pool.create("pool-crash-0", "tharsis", worker_index=0)
        gid1 = pool.create("pool-survive-1", "hellas", worker_index=1)

        os.kill(pool.worker_pids[0], signal.SIGTERM)

        with pytest.raises(WorkerCrashError) as exc_info:
            pool.pending(gid0)
        assert exc_info.value.seed_key == "pool-crash-0"

        # Worker 1 remains healthy and continues serving gid1.
        assert pool.pending(gid1) == [0, 1]
        pool.choose(gid1, 0, 0)
        assert pool.pending(gid1) == [0, 1]


def test_worker_clone_generic_routing() -> None:
    with WorkerPool(num_workers=2) as pool:
        gid = pool.create("clone-route-seed", "tharsis", worker_index=0)
        pool.choose(gid, 0, 0)
        cloned_id = pool.clone(gid)
        assert cloned_id != gid
        assert pool.game_worker_map[cloned_id] == 0
        assert pool.pending(cloned_id) == pool.pending(gid)
        assert pool.observe(cloned_id, 0) == pool.observe(gid, 0)
        assert pool.choice(cloned_id, 0) == pool.choice(gid, 0)
        pool.choose(cloned_id, 0, 0)
        assert pool.choice(cloned_id, 0) != pool.choice(gid, 0)
        # Complete seat 0's initial selection on the clone only; original stays untouched.
        while 0 in pool.pending(cloned_id):
            ch = pool.choice(cloned_id, 0)
            idx = next((i for i, c in enumerate(ch.candidates) if c.id == "done"), 0)
            pool.choose(cloned_id, 0, idx)
        assert pool.pending(cloned_id) == [1]
        assert pool.pending(gid) == [0, 1]
        assert pool.observe(cloned_id, 0) != pool.observe(gid, 0)
        pool.close_game(cloned_id)
        assert pool.pending(gid) == [0, 1]


def test_latency_benchmark_reports_single_and_batched_latencies() -> None:
    from tm_bot.env.bench import LatencyReport, measure_bridge_latency

    report = measure_bridge_latency(batch_size=4, steps=20, workers=2)
    assert isinstance(report, LatencyReport)
    assert report.single_step_mean_ms > 0
    assert report.single_step_p50_ms > 0
    assert report.single_observe_mean_ms > 0
    assert report.batched_amortized_step_ms > 0
    assert report.pool_amortized_step_ms > 0
    assert report.batched_amortized_step_ms < report.single_step_mean_ms




