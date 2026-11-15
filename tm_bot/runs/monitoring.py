"""Run monitoring: JSONL metrics stream, summary, shape sanity, determinism checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tm_bot.eval.stats import summarize_latency


def policy_entropy(probs: Sequence[float]) -> float:
    """Return the natural-log entropy of a candidate distribution."""
    arr = np.asarray(probs, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if arr.size == 0:
        return 0.0
    total = float(arr.sum())
    if total <= 0.0:
        return 0.0
    p = arr / total
    return float(-(p * np.log(p)).sum())


@dataclass(frozen=True, slots=True)
class GameSummary:
    """Compact per-game record used for monitoring and regression cases."""

    seed_key: str
    map_name: str
    ok: bool
    generations: int | None
    decisions: int
    scores: tuple[int, int] | None = None
    winner: int | None = None
    tie: bool = False
    error_type: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CycleMetrics:
    """One self-play + learner cycle written to the monitoring JSONL stream."""

    cycle: int
    elapsed_s: float
    wall_clock_s: float
    selfplay_games: int
    total_selfplay_games: int
    decisions: int
    searched_decisions: int
    rejections: int
    failures: int
    mean_generations: float | None = None
    min_generations: int | None = None
    max_generations: int | None = None
    mean_vp_margin: float | None = None
    mean_decisions_per_game: float | None = None
    policy_loss: float | None = None
    value_loss: float | None = None
    total_loss: float | None = None
    policy_entropy: float | None = None
    value_brier: float | None = None
    value_accuracy: float | None = None
    decision_latency_ms: dict[str, Any] = field(default_factory=dict)
    shape_flags: list[dict[str, Any]] = field(default_factory=list)
    determinism_checked: bool = False
    determinism_ok: bool | None = None
    throughput_games_per_hour: float | None = None
    stop_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["type"] = "cycle"
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CycleMetrics:
        data = {k: v for k, v in payload.items() if k != "type"}
        return cls(**data)


@dataclass(frozen=True, slots=True)
class CheckpointMetrics:
    """One periodic checkpoint evaluation written to the monitoring JSONL stream."""

    cycle: int
    elapsed_s: float
    step: int
    games: int
    checkpoint: str
    dev_games: int
    dev_win_rate_vs_b1: float | None
    dev_win_rate_vs_b1_ci_95: tuple[float, float] | None
    dev_win_rate_vs_previous_best: float | None
    dev_win_rate_vs_previous_best_ci_95: tuple[float, float] | None
    dev_mean_vp_margin_vs_b1: float | None
    new_best: bool
    best_dev_win_rate_vs_b1: float | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["type"] = "checkpoint"
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CheckpointMetrics:
        data = {k: v for k, v in payload.items() if k != "type"}
        for key in ("dev_win_rate_vs_b1_ci_95", "dev_win_rate_vs_previous_best_ci_95"):
            if data.get(key) is not None:
                data[key] = tuple(data[key])
        return cls(**data)


def game_shape_flags(
    games: Sequence[GameSummary],
    *,
    min_generations: int,
    max_generations: int,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return compact records for completed games outside the expected shape range."""
    flagged: list[dict[str, Any]] = []
    for game in games:
        if not game.ok or game.generations is None:
            continue
        if min_generations <= int(game.generations) <= max_generations:
            continue
        if len(flagged) >= limit:
            break
        flagged.append(
            {
                "seed_key": game.seed_key,
                "map": game.map_name,
                "generations": int(game.generations),
                "decisions": int(game.decisions),
            }
        )
    return flagged


@dataclass(frozen=True, slots=True)
class DeterminismResult:
    """Result of replaying a finished self-play game from seed + actions."""

    seed_key: str
    map_name: str
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_replay(original: GameSummary, replay: GameSummary) -> DeterminismResult:
    """Compare an original game with its seed + action-history replay."""
    differences: list[str] = []
    if original.ok != replay.ok:
        differences.append(f"ok {original.ok} != {replay.ok}")
    if original.generations != replay.generations:
        differences.append(f"generations {original.generations} != {replay.generations}")
    if original.scores != replay.scores:
        differences.append(f"scores {original.scores} != {replay.scores}")
    if original.winner != replay.winner:
        differences.append(f"winner {original.winner} != {replay.winner}")
    if original.tie != replay.tie:
        differences.append(f"tie {original.tie} != {replay.tie}")
    return DeterminismResult(
        seed_key=original.seed_key,
        map_name=original.map_name,
        ok=not differences,
        detail="identical" if not differences else "; ".join(differences),
    )


def _aggregate_numeric(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, float | int | None]:
    values = [
        float(r[key])
        for r in records
        if r.get(key) is not None and np.isfinite(float(r[key]))
    ]
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


class MonitorLogger:
    """Append-only JSONL metrics stream plus an aggregated summary."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cycles: list[CycleMetrics] = []
        self._checkpoints: list[CheckpointMetrics] = []
        self._rejections: list[dict[str, Any]] = []
        self._determinism: list[DeterminismResult] = []

    @property
    def cycles(self) -> tuple[CycleMetrics, ...]:
        return tuple(self._cycles)

    @property
    def checkpoints(self) -> tuple[CheckpointMetrics, ...]:
        return tuple(self._checkpoints)

    @property
    def rejections(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._rejections)

    def _append(self, payload: Mapping[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True) + "\n")

    def log_cycle(self, record: CycleMetrics) -> None:
        self._cycles.append(record)
        self._append(record.to_dict())

    def log_checkpoint(self, record: CheckpointMetrics) -> None:
        self._checkpoints.append(record)
        self._append(record.to_dict())

    def log_rejection(self, case: Mapping[str, Any]) -> None:
        payload = {"type": "rejection", **dict(case)}
        self._rejections.append(payload)
        self._append(payload)

    def log_determinism(self, result: DeterminismResult) -> None:
        self._determinism.append(result)
        self._append({"type": "determinism", **result.to_dict()})

    def load_existing(self) -> None:
        """Re-read an existing JSONL stream (used when resuming a run)."""
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            rtype = payload.get("type")
            if rtype == "cycle":
                self._cycles.append(CycleMetrics.from_dict(payload))
            elif rtype == "checkpoint":
                self._checkpoints.append(CheckpointMetrics.from_dict(payload))
            elif rtype == "rejection":
                self._rejections.append(dict(payload))
            elif rtype == "determinism":
                self._determinism.append(
                    DeterminismResult(
                        seed_key=str(payload["seed_key"]),
                        map_name=str(payload["map_name"]),
                        ok=bool(payload["ok"]),
                        detail=str(payload["detail"]),
                    )
                )

    def summary(self) -> dict[str, Any]:
        """Aggregate the stream into a compact run summary."""
        cycles = [c.to_dict() for c in self._cycles]
        total_selfplay_games = max(
            (int(c.total_selfplay_games) for c in self._cycles), default=0
        )
        elapsed = max((float(c.elapsed_s) for c in self._cycles), default=0.0)
        throughput = (
            total_selfplay_games / (elapsed / 3600.0) if elapsed > 0.0 else None
        )
        shape_flag_count = sum(len(c.shape_flags) for c in self._cycles)
        shape_examples = [flag for c in self._cycles for flag in c.shape_flags][:20]
        determ = [r.to_dict() for r in self._determinism]
        return {
            "schema_version": "selfplay-summary-v1",
            "cycles": len(self._cycles),
            "checkpoints": len(self._checkpoints),
            "selfplay_games": total_selfplay_games,
            "elapsed_s": elapsed,
            "throughput_games_per_hour": throughput,
            "decisions": sum(int(c.decisions) for c in self._cycles),
            "searched_decisions": sum(int(c.searched_decisions) for c in self._cycles),
            "rejections": len(self._rejections),
            "failures": sum(int(c.failures) for c in self._cycles),
            "policy_entropy": _aggregate_numeric(cycles, "policy_entropy"),
            "policy_loss": _aggregate_numeric(cycles, "policy_loss"),
            "value_loss": _aggregate_numeric(cycles, "value_loss"),
            "value_brier": _aggregate_numeric(cycles, "value_brier"),
            "mean_generations": _aggregate_numeric(cycles, "mean_generations"),
            "mean_vp_margin": _aggregate_numeric(cycles, "mean_vp_margin"),
            "mean_decisions_per_game": _aggregate_numeric(
                cycles, "mean_decisions_per_game"
            ),
            "shape_flag_count": shape_flag_count,
            "shape_examples": shape_examples,
            "determinism_checks": len(determ),
            "determinism_failures": [d for d in determ if not d["ok"]],
            "last_cycle": cycles[-1] if cycles else None,
            "last_checkpoint": (
                self._checkpoints[-1].to_dict() if self._checkpoints else None
            ),
        }

    def write_summary(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self.path.with_name("summary.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.summary(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target


def latency_summary(samples: Sequence[float]) -> dict[str, Any]:
    return summarize_latency(samples)
