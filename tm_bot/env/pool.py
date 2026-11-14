"""Multi-worker pool managing Node adapter worker subprocesses."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

from tm_bot.env.client import GameSession, WorkerClient
from tm_bot.env.errors import BridgeError
from tm_bot.env.types import ActorObservation, BatchOp, BatchResult, Choice, GameResult, Seat


class WorkerPool:
    """Pool of N Node worker processes with per-game routing and concurrent batching."""

    def __init__(
        self,
        num_workers: int = 1,
        *,
        worker_script: Path | str | None = None,
        node_executable: str = "node",
        node_args: Sequence[str] = (),
        cwd: Path | str | None = None,
        stderr_tail_lines: int = 100,
    ) -> None:
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")

        self._lock = threading.Lock()
        self._workers: list[WorkerClient] = []
        try:
            for _ in range(num_workers):
                self._workers.append(
                    WorkerClient(
                        worker_script=worker_script,
                        node_executable=node_executable,
                        node_args=node_args,
                        cwd=cwd,
                        stderr_tail_lines=stderr_tail_lines,
                    )
                )
        except Exception:
            for w in self._workers:
                w.close()
            raise

        self._executor = ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="tm-pool")
        self._game_to_worker: dict[str, int] = {}
        self._worker_game_counts: list[int] = [0] * num_workers
        self._next_game_seq = 1
        self._rr_cursor = 0
        self._closed = False

    @property
    def num_workers(self) -> int:
        return len(self._workers)

    @property
    def worker_pids(self) -> list[int]:
        return [w.pid for w in self._workers]

    @property
    def game_worker_map(self) -> dict[str, int]:
        with self._lock:
            return dict(self._game_to_worker)

    @property
    def active_games(self) -> int:
        with self._lock:
            return len(self._game_to_worker)

    def alive_worker_count(self) -> int:
        return sum(1 for w in self._workers if w.is_alive())

    def _allocate_game_id_locked(self, requested: str | None) -> str:
        if requested is not None:
            if not isinstance(requested, str) or not requested:
                raise ValueError("game_id must be a non-empty string")
            if requested in self._game_to_worker:
                raise BridgeError(f"Game {requested!r} already exists in pool")
            return requested
        while True:
            gid = f"pool-g-{self._next_game_seq}"
            self._next_game_seq += 1
            if gid not in self._game_to_worker:
                return gid

    def _pick_worker_locked(self, worker_index: int | None = None) -> int:
        if worker_index is not None:
            if not (0 <= worker_index < len(self._workers)):
                raise IndexError(f"worker_index {worker_index} out of range [0, {len(self._workers)})")
            return worker_index

        alive_indices = [i for i, w in enumerate(self._workers) if w.is_alive()]
        candidates = alive_indices if alive_indices else list(range(len(self._workers)))

        best_idx = candidates[0]
        best_count = self._worker_game_counts[best_idx]
        best_offset = (best_idx - self._rr_cursor) % len(self._workers)

        for idx in candidates[1:]:
            count = self._worker_game_counts[idx]
            offset = (idx - self._rr_cursor) % len(self._workers)
            if count < best_count or (count == best_count and offset < best_offset):
                best_idx = idx
                best_count = count
                best_offset = offset

        self._rr_cursor = (best_idx + 1) % len(self._workers)
        return best_idx

    def _worker_for_game(self, game_id: str) -> tuple[int, WorkerClient]:
        with self._lock:
            if game_id not in self._game_to_worker:
                raise BridgeError(f"Unknown game_id {game_id!r}")
            w_idx = self._game_to_worker[game_id]
            return w_idx, self._workers[w_idx]

    def create(
        self,
        seed_key: str,
        map: str,
        *,
        game_id: str | None = None,
        worker_index: int | None = None,
    ) -> str:
        with self._lock:
            if self._closed:
                raise BridgeError("WorkerPool is closed")
            w_idx = self._pick_worker_locked(worker_index)
            assigned_id = self._allocate_game_id_locked(game_id)
            self._game_to_worker[assigned_id] = w_idx
            self._worker_game_counts[w_idx] += 1
            worker = self._workers[w_idx]

        try:
            return worker.create(seed_key, map, game_id=assigned_id)
        except Exception:
            with self._lock:
                if self._game_to_worker.get(assigned_id) == w_idx:
                    self._game_to_worker.pop(assigned_id, None)
                    self._worker_game_counts[w_idx] -= 1
            raise

    def open(
        self,
        seed_key: str,
        map: str,
        *,
        game_id: str | None = None,
        worker_index: int | None = None,
    ) -> GameSession:
        assigned_id = self.create(seed_key, map, game_id=game_id, worker_index=worker_index)
        return GameSession(self, assigned_id, seed_key, map)

    def pending(self, game_id: str) -> list[Seat]:
        _, worker = self._worker_for_game(game_id)
        return worker.pending(game_id)

    def observe(self, game_id: str, seat: Seat) -> ActorObservation:
        _, worker = self._worker_for_game(game_id)
        return worker.observe(game_id, seat)

    def choice(self, game_id: str, seat: Seat) -> Choice:
        _, worker = self._worker_for_game(game_id)
        return worker.choice(game_id, seat)

    def choose(self, game_id: str, seat: Seat, candidate_index: int) -> None:
        _, worker = self._worker_for_game(game_id)
        worker.choose(game_id, seat, candidate_index)

    def result(self, game_id: str) -> GameResult | None:
        _, worker = self._worker_for_game(game_id)
        return worker.result(game_id)

    def clone(self, game_id: str, *, new_game_id: str | None = None) -> str:
        with self._lock:
            if game_id not in self._game_to_worker:
                raise BridgeError(f"Unknown game_id {game_id!r}")
            w_idx = self._game_to_worker[game_id]
            assigned_new_id = self._allocate_game_id_locked(new_game_id)
            self._game_to_worker[assigned_new_id] = w_idx
            self._worker_game_counts[w_idx] += 1
            worker = self._workers[w_idx]

        try:
            return worker.clone(game_id, new_game_id=assigned_new_id)
        except Exception:
            with self._lock:
                if self._game_to_worker.get(assigned_new_id) == w_idx:
                    self._game_to_worker.pop(assigned_new_id, None)
                    self._worker_game_counts[w_idx] -= 1
            raise

    def determinize_clone(
        self, game_id: str, seat: Seat, seed: int, *, new_game_id: str | None = None
    ) -> str:
        with self._lock:
            if game_id not in self._game_to_worker:
                raise BridgeError(f"Unknown game_id {game_id!r}")
            w_idx = self._game_to_worker[game_id]
            assigned_new_id = self._allocate_game_id_locked(new_game_id)
            self._game_to_worker[assigned_new_id] = w_idx
            self._worker_game_counts[w_idx] += 1
            worker = self._workers[w_idx]

        try:
            return worker.determinize_clone(game_id, seat, seed, new_game_id=assigned_new_id)
        except Exception:
            with self._lock:
                if self._game_to_worker.get(assigned_new_id) == w_idx:
                    self._game_to_worker.pop(assigned_new_id, None)
                    self._worker_game_counts[w_idx] -= 1
            raise

    def close_game(self, game_id: str) -> bool:
        _, worker = self._worker_for_game(game_id)
        closed = worker.close_game(game_id)
        with self._lock:
            w_idx = self._game_to_worker.pop(game_id, None)
            if w_idx is not None:
                self._worker_game_counts[w_idx] -= 1
        return closed

    def batch(
        self,
        requests: Sequence[BatchOp | Mapping[str, Any]],
        *,
        raise_on_error: bool = True,
    ) -> list[Any]:
        if not requests:
            return []

        per_worker_ops: dict[int, list[tuple[int, BatchOp]]] = {}
        new_registrations: list[tuple[int, str, int]] = []
        close_registrations: list[tuple[int, str, int]] = []

        with self._lock:
            if self._closed:
                raise BridgeError("WorkerPool is closed")

            for idx, item in enumerate(requests):
                if isinstance(item, BatchOp):
                    method = item.method
                    params = dict(item.params)
                elif isinstance(item, Mapping):
                    method = str(item["method"])
                    params = dict(item.get("params") or {})
                else:
                    raise TypeError(f"Unsupported batch operation type: {type(item).__name__}")

                if method == "create":
                    w_idx = self._pick_worker_locked()
                    assigned_id = self._allocate_game_id_locked(params.get("gameId"))
                    params["gameId"] = assigned_id
                    self._game_to_worker[assigned_id] = w_idx
                    self._worker_game_counts[w_idx] += 1
                    new_registrations.append((idx, assigned_id, w_idx))
                elif method in ("clone", "determinize_clone"):
                    src_id = params.get("gameId")
                    if not isinstance(src_id, str) or src_id not in self._game_to_worker:
                        raise BridgeError(f"Unknown game_id {src_id!r} in batched {method}")
                    w_idx = self._game_to_worker[src_id]
                    assigned_new_id = self._allocate_game_id_locked(params.get("newGameId"))
                    params["newGameId"] = assigned_new_id
                    self._game_to_worker[assigned_new_id] = w_idx
                    self._worker_game_counts[w_idx] += 1
                    new_registrations.append((idx, assigned_new_id, w_idx))
                else:
                    gid = params.get("gameId")
                    if not isinstance(gid, str) or gid not in self._game_to_worker:
                        raise BridgeError(f"Unknown game_id {gid!r} in batched {method}")
                    w_idx = self._game_to_worker[gid]
                    if method in ("close", "dispose"):
                        close_registrations.append((idx, gid, w_idx))

                per_worker_ops.setdefault(w_idx, []).append((idx, BatchOp(method, params)))

        ordered_results: list[BatchResult | None] = [None] * len(requests)
        worker_fatal_errors: dict[int, Exception] = {}

        if len(per_worker_ops) == 1:
            w_idx, entries = next(iter(per_worker_ops.items()))
            ops = [op for _, op in entries]
            try:
                sub_results = self._workers[w_idx].batch(ops, raise_on_error=False)
                for (orig_idx, _), res in zip(entries, sub_results):
                    ordered_results[orig_idx] = res
            except Exception as exc:
                worker_fatal_errors[w_idx] = exc
                for orig_idx, _ in entries:
                    ordered_results[orig_idx] = BatchResult(ok=False, value=None, error=exc)
        else:
            futures = {
                w_idx: self._executor.submit(
                    self._workers[w_idx].batch,
                    [op for _, op in entries],
                    raise_on_error=False,
                )
                for w_idx, entries in per_worker_ops.items()
            }
            for w_idx, fut in futures.items():
                entries = per_worker_ops[w_idx]
                try:
                    sub_results = fut.result()
                    for (orig_idx, _), res in zip(entries, sub_results):
                        ordered_results[orig_idx] = res
                except Exception as exc:
                    worker_fatal_errors[w_idx] = exc
                    for orig_idx, _ in entries:
                        ordered_results[orig_idx] = BatchResult(ok=False, value=None, error=exc)

        with self._lock:
            for orig_idx, gid, w_idx in new_registrations:
                res = ordered_results[orig_idx]
                if res is None or not res.ok:
                    if self._game_to_worker.get(gid) == w_idx:
                        self._game_to_worker.pop(gid, None)
                        self._worker_game_counts[w_idx] -= 1
            for orig_idx, gid, w_idx in close_registrations:
                res = ordered_results[orig_idx]
                if res is not None and res.ok:
                    if self._game_to_worker.pop(gid, None) is not None:
                        self._worker_game_counts[w_idx] -= 1

        final_results = [r for r in ordered_results if r is not None]
        if raise_on_error:
            for res in final_results:
                if not res.ok and res.error is not None:
                    raise res.error
            return [res.value for res in final_results]
        return final_results

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._executor.shutdown(wait=True)
        finally:
            for worker in self._workers:
                worker.close()

    def shutdown(self) -> None:
        self.close()

    def __enter__(self) -> WorkerPool:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
