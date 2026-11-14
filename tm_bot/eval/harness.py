"""High-level evaluation runner reusing `tm_bot.agents.play_games`."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Sequence

from tm_bot.agents import B0Agent, B1Agent, load_card_data
from tm_bot.agents.protocol import Agent, GameRunResult
from tm_bot.agents.runner import DEFAULT_MAX_DECISIONS, play_games
from tm_bot.env.types import SUPPORTED_MAPS, GameResult, Seat
from tm_bot.eval.pairing import DEFAULT_EVAL_SEEDS, AgentFactory, build_paired_specs
from tm_bot.eval.report import summarize_evaluation
from tm_bot.eval.seeds import SeedKey, SeedNamespace
from tm_bot.eval.stats import DEFAULT_BOOTSTRAP_RESAMPLES, DEFAULT_BOOTSTRAP_SEED


def _coerce_spec_value(raw: str) -> Any:
    low = raw.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low == "none":
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def resolve_agent_factory(spec: str) -> Callable[[], Agent]:
    """Resolve a built-in agent name (`b0`, `b1`) or `module:callable[?k=v&...]` spec."""
    norm = spec.strip().lower()
    if norm == "b0":
        return lambda: B0Agent(seed=0)
    if norm == "b1":
        card_data = load_card_data()
        return lambda: B1Agent(seed=0, card_data=card_data)
    if ":" in spec:
        base_spec, _, query = spec.partition("?")
        mod_name, attr_name = base_spec.split(":", 1)
        mod = importlib.import_module(mod_name)
        factory = getattr(mod, attr_name)
        if not callable(factory):
            raise TypeError(f"Resolved object {spec!r} is not callable")
        if not query:
            return lambda: factory()
        kwargs: dict[str, Any] = {}
        for part in query.split("&"):
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"Invalid query parameter {part!r} in agent spec {spec!r}")
            k, v = part.split("=", 1)
            kwargs[k] = _coerce_spec_value(v)
        return lambda: factory(**kwargs)
    raise ValueError(
        f"Unknown agent spec {spec!r}; expected 'b0', 'b1', or 'module:factory'"
    )


def _serialize_outcome(out: GameRunResult) -> dict[str, Any]:
    res_dict: dict[str, Any] | None = None
    if out.result is not None:
        res_dict = {
            "scores": [int(out.result.scores[0]), int(out.result.scores[1])],
            "winner": out.result.winner,
            "tie": bool(out.result.tie),
            "generations": int(out.result.generations),
        }
    return {
        "seed_key": out.seed_key,
        "map": out.map,
        "result": res_dict,
        "generations": out.generations,
        "decisions": int(out.decisions),
        "agent_versions": [out.agent_versions[0], out.agent_versions[1]],
        "ok": bool(out.ok),
        "error_type": out.error_type,
        "error": out.error,
        "decision_latencies_ms": [
            list(out.decision_latencies_ms[0]),
            list(out.decision_latencies_ms[1]),
        ],
        "searched_decision_latencies_ms": [
            list(out.searched_decision_latencies_ms[0]),
            list(out.searched_decision_latencies_ms[1]),
        ],
    }


def _deserialize_outcome(payload: dict[str, Any]) -> GameRunResult:
    res_raw = payload.get("result")
    res_obj: GameResult | None = None
    if res_raw is not None:
        raw_winner = res_raw["winner"]
        winner: Seat | None = int(raw_winner) if raw_winner is not None else None  # type: ignore[assignment]
        res_obj = GameResult(
            scores=(int(res_raw["scores"][0]), int(res_raw["scores"][1])),
            winner=winner,
            tie=bool(res_raw["tie"]),
            generations=int(res_raw["generations"]),
        )
    vers = payload["agent_versions"]
    lats = payload.get("decision_latencies_ms", [[], []])
    slats = payload.get("searched_decision_latencies_ms", [[], []])
    return GameRunResult(
        seed_key=str(payload["seed_key"]),
        map=str(payload["map"]),
        result=res_obj,
        generations=payload.get("generations"),
        decisions=int(payload["decisions"]),
        agent_versions=(vers[0], vers[1]),
        ok=bool(payload["ok"]),
        error_type=payload.get("error_type"),
        error=payload.get("error"),
        decision_latencies_ms=(
            tuple(float(x) for x in lats[0]),
            tuple(float(x) for x in lats[1]),
        ),
        searched_decision_latencies_ms=(
            tuple(float(x) for x in slats[0]),
            tuple(float(x) for x in slats[1]),
        ),
    )


def evaluate_agents(
    *,
    agent_a: AgentFactory,
    agent_b: AgentFactory,
    agent_a_name: str = "agent_a",
    agent_b_name: str = "agent_b",
    agent_a_version: str | None = None,
    agent_b_version: str | None = None,
    a_training_seeds: int = 0,
    b_training_seeds: int = 0,
    a_compute: dict[str, Any] | None = None,
    b_compute: dict[str, Any] | None = None,
    seed_keys: Sequence[SeedKey | str] | None = None,
    namespace: SeedNamespace = "eval",
    num_seeds: int = DEFAULT_EVAL_SEEDS,
    seed_start: int = 0,
    maps: Sequence[str] = SUPPORTED_MAPS,
    num_workers: int = 8,
    batch_size: int = 32,
    max_decisions: int = DEFAULT_MAX_DECISIONS,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Run seat-swapped evaluation of `agent_a` vs `agent_b` via `play_games` and summarize."""
    specs, metas = build_paired_specs(
        agent_a=agent_a,
        agent_b=agent_b,
        seed_keys=seed_keys,
        namespace=namespace,
        num_seeds=num_seeds,
        seed_start=seed_start,
        maps=maps,
    )

    t0 = time.perf_counter()
    outcomes = play_games(
        specs,
        num_workers=num_workers,
        batch_size=batch_size,
        max_decisions=max_decisions,
    )
    wall_time_s = time.perf_counter() - t0

    return summarize_evaluation(
        outcomes=outcomes,
        metas=metas,
        namespace=namespace,
        agent_a_name=agent_a_name,
        agent_b_name=agent_b_name,
        agent_a_version=agent_a_version,
        agent_b_version=agent_b_version,
        a_training_seeds=a_training_seeds,
        b_training_seeds=b_training_seeds,
        a_compute=a_compute,
        b_compute=b_compute,
        workers=num_workers,
        batch_size=batch_size,
        wall_time_s=wall_time_s,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )


def evaluate_agents_chunked(
    *,
    checkpoint_path: Path,
    resume: bool = False,
    max_games_per_run: int | None = None,
    run_signature: dict[str, Any] | None = None,
    agent_a: AgentFactory,
    agent_b: AgentFactory,
    agent_a_name: str = "agent_a",
    agent_b_name: str = "agent_b",
    agent_a_version: str | None = None,
    agent_b_version: str | None = None,
    a_training_seeds: int = 0,
    b_training_seeds: int = 0,
    a_compute: dict[str, Any] | None = None,
    b_compute: dict[str, Any] | None = None,
    seed_keys: Sequence[SeedKey | str] | None = None,
    namespace: SeedNamespace = "eval",
    num_seeds: int = DEFAULT_EVAL_SEEDS,
    seed_start: int = 0,
    maps: Sequence[str] = SUPPORTED_MAPS,
    num_workers: int = 8,
    batch_size: int = 32,
    max_decisions: int = DEFAULT_MAX_DECISIONS,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[dict[str, Any] | None, int, int, float]:
    """Run paired evaluation incrementally with JSONL checkpointing.

    Returns `(report_or_none, completed_count, total_count, accumulated_wall_time_s)`.
    When all games are complete, `report_or_none` is the full report dict.
    """
    if max_games_per_run is not None and max_games_per_run < 1:
        raise ValueError(f"max_games_per_run must be >= 1, got {max_games_per_run}")

    specs, metas = build_paired_specs(
        agent_a=agent_a,
        agent_b=agent_b,
        seed_keys=seed_keys,
        namespace=namespace,
        num_seeds=num_seeds,
        seed_start=seed_start,
        maps=maps,
    )
    total_games = len(specs)
    sig = dict(run_signature or {})
    sig.setdefault("namespace", namespace)
    sig.setdefault("num_seeds", num_seeds)
    sig.setdefault("seed_start", seed_start)
    sig.setdefault("maps", list(maps))
    sig.setdefault("total_games", total_games)

    completed: dict[int, GameRunResult] = {}
    accumulated_wall_s = 0.0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if resume and checkpoint_path.exists():
        lines = checkpoint_path.read_text(encoding="utf-8").splitlines()
        if lines:
            header = json.loads(lines[0])
            if header.get("type") != "header" or header.get("signature") != sig:
                raise ValueError(
                    f"Checkpoint signature mismatch in {checkpoint_path}: "
                    f"found {header.get('signature')!r}, expected {sig!r}"
                )
            for line in lines[1:]:
                if not line.strip():
                    continue
                rec = json.loads(line)
                rtype = rec.get("type")
                if rtype == "outcome":
                    completed[int(rec["index"])] = _deserialize_outcome(rec["outcome"])
                elif rtype == "chunk_time":
                    accumulated_wall_s += float(rec["wall_time_s"])
        else:
            checkpoint_path.write_text(
                json.dumps({"type": "header", "signature": sig}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    else:
        checkpoint_path.write_text(
            json.dumps({"type": "header", "signature": sig}, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    pending_indices = [i for i in range(total_games) if i not in completed]
    if pending_indices:
        run_indices = (
            pending_indices[:max_games_per_run]
            if max_games_per_run is not None
            else pending_indices
        )
        t0 = time.perf_counter()
        chunk_outcomes = play_games(
            [specs[i] for i in run_indices],
            num_workers=num_workers,
            batch_size=batch_size,
            max_decisions=max_decisions,
        )
        chunk_wall_s = time.perf_counter() - t0
        accumulated_wall_s += chunk_wall_s

        with checkpoint_path.open("a", encoding="utf-8") as fh:
            for idx, out in zip(run_indices, chunk_outcomes):
                completed[idx] = out
                fh.write(
                    json.dumps(
                        {
                            "type": "outcome",
                            "index": idx,
                            "outcome": _serialize_outcome(out),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            fh.write(
                json.dumps({"type": "chunk_time", "wall_time_s": chunk_wall_s}, sort_keys=True)
                + "\n"
            )

    if len(completed) < total_games:
        return None, len(completed), total_games, accumulated_wall_s

    ordered_outcomes = [completed[i] for i in range(total_games)]
    report = summarize_evaluation(
        outcomes=ordered_outcomes,
        metas=metas,
        namespace=namespace,
        agent_a_name=agent_a_name,
        agent_b_name=agent_b_name,
        agent_a_version=agent_a_version,
        agent_b_version=agent_b_version,
        a_training_seeds=a_training_seeds,
        b_training_seeds=b_training_seeds,
        a_compute=a_compute,
        b_compute=b_compute,
        workers=num_workers,
        batch_size=batch_size,
        wall_time_s=accumulated_wall_s,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    return report, total_games, total_games, accumulated_wall_s
