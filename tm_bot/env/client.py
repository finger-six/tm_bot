"""Subprocess bridge client speaking line-delimited JSON to the Node adapter worker."""

from __future__ import annotations

import json
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

from tm_bot.env.errors import AdapterBugError, BridgeError, BridgeProtocolError, WorkerCrashError
from tm_bot.env.types import ActorObservation, BatchOp, BatchResult, Candidate, Choice, GameResult, Seat

_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_WORKER_SCRIPT = _DEFAULT_REPO_ROOT / "adapter" / "dist" / "src" / "worker.js"


def _parse_choice(payload: dict[str, Any]) -> Choice:
    candidates = tuple(
        Candidate(id=str(c["id"]), label=str(c["label"]))
        for c in payload["candidates"]
    )
    return Choice(
        seat=payload["seat"],
        kind=str(payload["kind"]),
        prompt=str(payload["prompt"]),
        candidates=candidates,
    )


def _parse_result(payload: dict[str, Any] | None) -> GameResult | None:
    if payload is None:
        return None
    scores_raw = payload["scores"]
    return GameResult(
        scores=(int(scores_raw[0]), int(scores_raw[1])),
        winner=payload["winner"],
        tie=bool(payload["tie"]),
        generations=int(payload["generations"]),
    )


class GameSession:
    """Single-game handle mirroring the AdapterGame interface over a client or pool."""

    def __init__(self, client: Any, game_id: str, seed_key: str, map_name: str) -> None:
        self._client = client
        self.game_id = game_id
        self.seed_key = seed_key
        self.map = map_name
        self._closed = False

    def pending(self) -> list[Seat]:
        return self._client.pending(self.game_id)

    def observe(self, seat: Seat) -> ActorObservation:
        return self._client.observe(self.game_id, seat)

    def choice(self, seat: Seat) -> Choice:
        return self._client.choice(self.game_id, seat)

    def choose(self, seat: Seat, candidate_index: int) -> None:
        self._client.choose(self.game_id, seat, candidate_index)

    def result(self) -> GameResult | None:
        return self._client.result(self.game_id)

    def clone(self, *, new_game_id: str | None = None) -> GameSession:
        cloned_id = self._client.clone(self.game_id, new_game_id=new_game_id)
        return GameSession(self._client, cloned_id, self.seed_key, self.map)

    def determinize_clone(
        self, seat: Seat, seed: int, *, new_game_id: str | None = None
    ) -> GameSession:
        cloned_id = self._client.determinize_clone(
            self.game_id, seat, seed, new_game_id=new_game_id
        )
        return GameSession(self._client, cloned_id, self.seed_key, self.map)

    def close(self) -> bool:
        if self._closed:
            return False
        self._closed = True
        return self._client.close_game(self.game_id)

    def __enter__(self) -> GameSession:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            self.close()
        except BridgeError:
            pass


class WorkerClient:
    """Line-delimited JSON client owning a single Node adapter worker subprocess."""

    def __init__(
        self,
        *,
        worker_script: Path | str | None = None,
        node_executable: str = "node",
        node_args: Sequence[str] = (),
        cwd: Path | str | None = None,
        stderr_tail_lines: int = 100,
    ) -> None:
        script_path = Path(worker_script) if worker_script is not None else _DEFAULT_WORKER_SCRIPT
        if not script_path.exists():
            raise FileNotFoundError(
                f"Adapter worker script not found at {script_path}. "
                "Build the adapter first (`cd adapter && npm run build`)."
            )
        work_dir = Path(cwd) if cwd is not None else _DEFAULT_REPO_ROOT
        self._command: tuple[str, ...] = (node_executable, *node_args, str(script_path))
        self._stderr_lines: deque[str] = deque(maxlen=stderr_tail_lines)
        self._stderr_lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._next_req_id = 1
        self._closed = False
        self._game_seeds: dict[str, str] = {}

        self._proc = subprocess.Popen(
            list(self._command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(work_dir),
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name="tm-worker-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        stderr = self._proc.stderr
        if stderr is None:
            return
        try:
            for line in stderr:
                with self._stderr_lock:
                    self._stderr_lines.append(line.rstrip("\r\n"))
        except Exception:
            pass

    @property
    def stderr_tail(self) -> str:
        with self._stderr_lock:
            return "\n".join(self._stderr_lines)

    @property
    def pid(self) -> int:
        return self._proc.pid

    def is_alive(self) -> bool:
        return self._proc.poll() is None

    def _send_raw(self, payload: dict[str, Any], *, game_id: str | None = None) -> dict[str, Any]:
        with self._io_lock:
            if self._closed:
                raise WorkerCrashError(
                    "WorkerClient is already closed",
                    returncode=self._proc.poll(),
                    stderr_tail=self.stderr_tail,
                    command=self._command,
                    game_id=game_id,
                    seed_key=self._game_seeds.get(game_id) if game_id else None,
                )

            rc = self._proc.poll()
            if rc is not None:
                raise WorkerCrashError(
                    "Worker subprocess exited before request",
                    returncode=rc,
                    stderr_tail=self.stderr_tail,
                    command=self._command,
                    game_id=game_id,
                    seed_key=self._game_seeds.get(game_id) if game_id else None,
                )

            req_id = self._next_req_id
            self._next_req_id += 1
            payload_with_id = {"id": req_id, **payload}
            line_out = json.dumps(payload_with_id, separators=(",", ":")) + "\n"

            assert self._proc.stdin is not None
            assert self._proc.stdout is not None
            try:
                self._proc.stdin.write(line_out)
                self._proc.stdin.flush()
            except OSError as exc:
                try:
                    rc = self._proc.wait(timeout=1.0) if self._proc.poll() is None else self._proc.returncode
                except subprocess.TimeoutExpired:
                    rc = self._proc.poll()
                self._stderr_thread.join(timeout=0.5)
                raise WorkerCrashError(
                    f"Failed writing to worker stdin: {exc}",
                    returncode=rc,
                    stderr_tail=self.stderr_tail,
                    command=self._command,
                    game_id=game_id,
                    seed_key=self._game_seeds.get(game_id) if game_id else None,
                ) from exc

            raw_line = self._proc.stdout.readline()
            if raw_line == "":
                try:
                    rc = self._proc.wait(timeout=2.0) if self._proc.poll() is None else self._proc.returncode
                except subprocess.TimeoutExpired:
                    rc = self._proc.poll()
                self._stderr_thread.join(timeout=0.5)
                raise WorkerCrashError(
                    "Worker subprocess closed stdout unexpectedly (EOF)",
                    returncode=rc,
                    stderr_tail=self.stderr_tail,
                    command=self._command,
                    game_id=game_id,
                    seed_key=self._game_seeds.get(game_id) if game_id else None,
                )

            try:
                resp = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise BridgeProtocolError(
                    f"Worker emitted invalid JSON: {exc}",
                    raw_response=raw_line,
                    stderr_tail=self.stderr_tail,
                ) from exc

            if not isinstance(resp, dict):
                raise BridgeProtocolError(
                    "Worker response must be a JSON object",
                    raw_response=raw_line,
                    stderr_tail=self.stderr_tail,
                )
            if resp.get("id") != req_id:
                raise BridgeProtocolError(
                    f"Mismatched response id: expected {req_id}, got {resp.get('id')!r}",
                    raw_response=raw_line,
                    stderr_tail=self.stderr_tail,
                )
            return resp

    def _error_from_payload(self, err: dict[str, Any], *, game_id: str | None = None) -> BridgeError:
        err_type = str(err.get("type", "Error"))
        message = str(err.get("message", "Unknown worker error"))
        seed_key = str(err.get("seedKey") or (self._game_seeds.get(game_id, "") if game_id else ""))
        map_name = err.get("map")
        seat = err.get("seat")
        history = err.get("history")
        if err_type == "AdapterBugError" or seed_key or history is not None:
            return AdapterBugError(
                message,
                seed_key=seed_key,
                map_name=map_name,
                seat=seat,
                history=history if isinstance(history, list) else [],
                error_type=err_type,
            )
        if err_type == "ProtocolError":
            return BridgeProtocolError(message, stderr_tail=self.stderr_tail)
        return BridgeError(f"{err_type}: {message}")

    def _call(self, method: str, params: dict[str, Any], *, game_id: str | None = None) -> dict[str, Any]:
        resp = self._send_raw({"method": method, "params": params}, game_id=game_id)
        if not resp.get("ok"):
            err = resp.get("error") or {}
            raise self._error_from_payload(err, game_id=game_id)
        result = resp.get("result")
        if not isinstance(result, dict):
            raise BridgeProtocolError(f"Expected dict result for {method}, got {type(result).__name__}")
        return result

    def create(self, seed_key: str, map: str, *, game_id: str | None = None) -> str:
        params: dict[str, Any] = {"seedKey": seed_key, "map": map}
        if game_id is not None:
            params["gameId"] = game_id
        res = self._call("create", params, game_id=game_id)
        assigned_id = str(res["gameId"])
        self._game_seeds[assigned_id] = seed_key
        return assigned_id

    def open(self, seed_key: str, map: str, *, game_id: str | None = None) -> GameSession:
        assigned_id = self.create(seed_key, map, game_id=game_id)
        return GameSession(self, assigned_id, seed_key, map)

    def pending(self, game_id: str) -> list[Seat]:
        res = self._call("pending", {"gameId": game_id}, game_id=game_id)
        return list(res["pending"])

    def observe(self, game_id: str, seat: Seat) -> ActorObservation:
        res = self._call("observe", {"gameId": game_id, "seat": seat}, game_id=game_id)
        return res["observation"]

    def choice(self, game_id: str, seat: Seat) -> Choice:
        res = self._call("choice", {"gameId": game_id, "seat": seat}, game_id=game_id)
        return _parse_choice(res["choice"])

    def choose(self, game_id: str, seat: Seat, candidate_index: int) -> None:
        self._call(
            "choose",
            {"gameId": game_id, "seat": seat, "candidateIndex": candidate_index},
            game_id=game_id,
        )

    def result(self, game_id: str) -> GameResult | None:
        res = self._call("result", {"gameId": game_id}, game_id=game_id)
        return _parse_result(res["result"])

    def clone(self, game_id: str, *, new_game_id: str | None = None) -> str:
        params: dict[str, Any] = {"gameId": game_id}
        if new_game_id is not None:
            params["newGameId"] = new_game_id
        res = self._call("clone", params, game_id=game_id)
        assigned_id = str(res["gameId"])
        self._game_seeds[assigned_id] = str(res.get("seedKey") or self._game_seeds.get(game_id, ""))
        return assigned_id

    def determinize_clone(
        self,
        game_id: str,
        seat: Seat,
        seed: int,
        *,
        new_game_id: str | None = None,
    ) -> str:
        params: dict[str, Any] = {"gameId": game_id, "seat": seat, "seed": int(seed)}
        if new_game_id is not None:
            params["newGameId"] = new_game_id
        res = self._call("determinize_clone", params, game_id=game_id)
        assigned_id = str(res["gameId"])
        self._game_seeds[assigned_id] = str(res.get("seedKey") or self._game_seeds.get(game_id, ""))
        return assigned_id

    def close_game(self, game_id: str) -> bool:
        res = self._call("close", {"gameId": game_id}, game_id=game_id)
        self._game_seeds.pop(game_id, None)
        return bool(res.get("closed", False))

    def ping(self) -> dict[str, Any]:
        return self._call("ping", {})

    def _interpret_result(self, method: str, params: Mapping[str, Any], res: dict[str, Any]) -> Any:
        if method == "create":
            assigned_id = str(res["gameId"])
            self._game_seeds[assigned_id] = str(params.get("seedKey", ""))
            return assigned_id
        if method == "pending":
            return list(res["pending"])
        if method == "observe":
            return res["observation"]
        if method == "choice":
            return _parse_choice(res["choice"])
        if method == "choose":
            return None
        if method == "result":
            return _parse_result(res["result"])
        if method in ("clone", "determinize_clone"):
            assigned_id = str(res["gameId"])
            src_id = params.get("gameId")
            fallback_seed = self._game_seeds.get(src_id, "") if isinstance(src_id, str) else ""
            self._game_seeds[assigned_id] = str(res.get("seedKey") or fallback_seed)
            return assigned_id
        if method in ("close", "dispose"):
            gid = params.get("gameId")
            if isinstance(gid, str):
                self._game_seeds.pop(gid, None)
            return bool(res.get("closed", False))
        return res

    def batch(
        self,
        requests: Sequence[BatchOp | Mapping[str, Any]],
        *,
        raise_on_error: bool = True,
    ) -> list[Any]:
        if not requests:
            return []

        normalized: list[tuple[str, dict[str, Any]]] = []
        wire_batch: list[dict[str, Any]] = []
        for item in requests:
            if isinstance(item, BatchOp):
                method = item.method
                params = dict(item.params)
            elif isinstance(item, Mapping):
                method = str(item["method"])
                raw_params = item.get("params") or {}
                params = dict(raw_params)
            else:
                raise TypeError(f"Unsupported batch operation type: {type(item).__name__}")
            normalized.append((method, params))
            wire_batch.append({"method": method, "params": params})

        resp = self._send_raw({"batch": wire_batch})
        if not resp.get("ok"):
            err = resp.get("error") or {}
            raise self._error_from_payload(err)

        raw_results = resp.get("results")
        if not isinstance(raw_results, list) or len(raw_results) != len(normalized):
            raise BridgeProtocolError(
                f"Expected {len(normalized)} batch results, got "
                f"{len(raw_results) if isinstance(raw_results, list) else type(raw_results).__name__}",
                stderr_tail=self.stderr_tail,
            )

        parsed_results: list[BatchResult] = []
        first_error: BridgeError | None = None

        for (method, params), sub_resp in zip(normalized, raw_results):
            if not isinstance(sub_resp, dict):
                raise BridgeProtocolError(
                    f"Batch sub-response must be a dict, got {type(sub_resp).__name__}",
                    stderr_tail=self.stderr_tail,
                )
            gid = params.get("gameId") if isinstance(params.get("gameId"), str) else None
            if sub_resp.get("ok"):
                raw_res = sub_resp.get("result")
                if not isinstance(raw_res, dict):
                    raise BridgeProtocolError(
                        f"Expected dict result for batched {method}, got {type(raw_res).__name__}",
                        stderr_tail=self.stderr_tail,
                    )
                val = self._interpret_result(method, params, raw_res)
                parsed_results.append(BatchResult(ok=True, value=val, error=None))
            else:
                err_payload = sub_resp.get("error") or {}
                err = self._error_from_payload(err_payload, game_id=gid)
                if first_error is None:
                    first_error = err
                parsed_results.append(BatchResult(ok=False, value=None, error=err))

        if raise_on_error:
            if first_error is not None:
                raise first_error
            return [r.value for r in parsed_results]
        return parsed_results

    def close(self) -> None:
        with self._io_lock:
            if self._closed:
                return
            if self._proc.poll() is None:
                try:
                    req_id = self._next_req_id
                    self._next_req_id += 1
                    assert self._proc.stdin is not None
                    self._proc.stdin.write(json.dumps({"id": req_id, "method": "shutdown", "params": {}}) + "\n")
                    self._proc.stdin.flush()
                    if self._proc.stdout is not None:
                        self._proc.stdout.readline()
                except Exception:
                    pass
                try:
                    if self._proc.stdin is not None:
                        self._proc.stdin.close()
                except Exception:
                    pass
                try:
                    self._proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    self._proc.terminate()
                    try:
                        self._proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        self._proc.kill()
                        self._proc.wait()
            else:
                try:
                    if self._proc.stdin is not None:
                        self._proc.stdin.close()
                except Exception:
                    pass
            self._closed = True
        self._stderr_thread.join(timeout=1.0)

    def __enter__(self) -> WorkerClient:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
