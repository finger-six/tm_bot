"""Run the pre-self-play gate (#47, #38 §5) on one v2 warm-start checkpoint and write the report.

    python -m tm_bot.training.run_warmstart_gate --agent main --run-dir runs/warmstart-v3 \
        --out docs/gate-warmstart-v3.md
    python -m tm_bot.training.run_warmstart_gate --agent comparator --run-dir runs/comparator-warmstart-v2 \
        --out docs/gate-comparator-warmstart-v2.md

This module gathers the gate's inputs: the #46 run's val metrics (`<run-dir>/metrics.json`),
train-split frequency baselines scored on the val split (setup tasks from the parquet tables,
gen 2+ tasks from the encoded cache the #46 trainers built), and the live `dev-*` pairings, played
with the eval harness's seat-swapped schedule and runner. The checks and the report are pure
functions in `tm_bot.training.warmstart_gate`. Each finished pairing's games are kept under
`--work-dir` (default `<run-dir>/gate/`), so `--resume` skips pairings that already ran.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from tm_bot.agents.protocol import Agent, GameRunResult
from tm_bot.agents.runner import DEFAULT_MAX_DECISIONS, play_games
from tm_bot.env.types import SUPPORTED_MAPS
from tm_bot.eval.harness import _deserialize_outcome, _serialize_outcome, resolve_agent_factory
from tm_bot.eval.pairing import PairedGameMeta, build_paired_specs
from tm_bot.models.value_temperature import read_value_temperature_file, value_temperature_path
from tm_bot.search.agent import ISMCTSAgent
from tm_bot.training.warmstart_gate import (
    GATE_SCHEMA_VERSION,
    PairingSpec,
    evaluate_gate,
    format_gate_report,
    gate_seed_keys,
    pairings_for,
    summarize_pairing,
)

REPO = Path(__file__).resolve().parents[2]
V1_CHECKPOINTS = {
    "main": REPO / "runs" / "warmstart" / "best",
    "comparator": REPO / "runs" / "comparator-warmstart-full" / "best",
}
# The main agent is gated with light search: #39's fallback budget (8 simulations, K = 4), the
# lowest v2 will use, with the milestone's other evaluation settings. The raw-policy diagnostic
# is the same agent with 0 simulations (policy head, same temperatures).
LIGHT_SEARCH: dict[str, Any] = {
    "num_simulations": 8,
    "num_worlds": 4,
    "c_puct": 1.5,
    "temperature": 0.0,
    "action_temperature": 1.0,
    "wave_size": 4,
    # Pinned to the v1 play rules the gate was specified with (#48 made v2 the default):
    # flat action temperature, outcome-only search utility.
    "play_temperature": "flat",
    "margin_weight": 0.0,
    # The one change of the #58 re-gate: the completed-Q root policy π′ (#60).
    "root_rule": "completed_q",
}
RAW_POLICY: dict[str, Any] = {**LIGHT_SEARCH, "num_simulations": 0}
# The archived v1 main warm start plays as it was evaluated: the same light search with its
# visit-count root rule, so the vs-v1 opponent stays a fixed reference.
V1_LIGHT_SEARCH: dict[str, Any] = {**LIGHT_SEARCH, "root_rule": "visits"}
# Opponents whose games record the candidate's live P(win) (#60: per-opponent diagnostic).
VALUE_RECORDING_OPPONENTS = ("b1", "self", "v1")
BASELINE_TASKS: tuple[tuple[str, str], ...] = (
    # (gate task, encoded-cache task)
    ("draft", "draft"),
    ("research_buy", "research_buy"),
    ("branch", "branch"),
    ("project_card", "project_play"),
    ("standard_project", "standard_project"),
    ("card_action", "card_action"),
)


# --- validation inputs --------------------------------------------------------------------------


def load_validation(
    run_dir: Path, agent: str, checkpoint: Path | None = None
) -> tuple[dict[str, Any], dict[str, Any] | None, Path]:
    """`(metrics.json, value report the gate judges, path)` of a finished #46 run.

    Both agents are judged post-temperature (#58 amendment of check 14). A main warm start
    trained before #60 has no `value_calibrated` in its metrics; its report then comes from the
    checkpoint's value-temperature sidecar (`python -m tm_bot.training.value_temperature`).
    """
    path = run_dir / "metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    if metrics.get("status") != "completed":
        raise ValueError(f"{path}: the #46 run has not completed (status={metrics.get('status')!r})")
    value = metrics.get("value_calibrated")
    if value is None and agent == "main":
        ckpt = checkpoint or run_dir / "best"
        fit = read_value_temperature_file(ckpt)
        if fit is None:
            raise ValueError(
                f"{path} has no post-temperature value report and {value_temperature_path(ckpt)} is missing; "
                f"fit it with `python -m tm_bot.training.value_temperature --checkpoint {ckpt}`"
            )
        value = fit["value_calibrated"]
    return metrics, value, path


def _grouped_hits(hits: np.ndarray, map_ids: np.ndarray) -> dict[str, Any]:
    def entry(mask: np.ndarray) -> dict[str, Any]:
        n = int(mask.sum())
        ok = int(hits[mask].sum())
        return {"correct": ok, "total": n, "acc": (ok / n) if n else 0.0}

    return {
        "overall": entry(np.ones(hits.shape[0], dtype=bool)),
        "per_map": {m: entry(map_ids == i) for i, m in enumerate(SUPPORTED_MAPS)},
    }


def _candidate_keys(table: Any, lo: int, hi: int) -> np.ndarray:
    """Identity of flat candidates `lo:hi`: the card id, plus the branch bits for root branches."""
    keys = np.asarray(table["cand_card"][lo:hi], dtype=np.int64) * 1024
    if table.task.name == "branch":
        bits = np.asarray(table["cand_feats"][lo:hi]) != 0.0
        keys += (bits.astype(np.int64) << np.arange(bits.shape[1], dtype=np.int64)).sum(axis=1)
    return keys


def _positive_mask(table: Any, r0: int, r1: int) -> np.ndarray:
    """Flat candidates of rows `r0:r1` that are a correct top-1 answer.

    Policy rows: the chosen candidate. Research buys (step 0): any bought packet card, or `done`
    (the last candidate) when nothing was bought, the target `evaluate_state_task` scores.
    """
    off = table.cand_off
    lo = int(off[r0])
    pos = np.zeros(int(off[r1]) - lo, dtype=bool)
    starts = off[r0:r1] - lo
    if table.task.kind == "policy":
        pos[starts + np.asarray(table["target"][r0:r1], dtype=np.int64)] = True
        return pos
    counts = np.asarray(table["cand_count"][r0:r1], dtype=np.int64)
    kept = np.asarray(table["kept_mask"][r0:r1], dtype=np.int64)
    for j in range(8):
        sel = ((kept >> j) & 1).astype(bool) & (j < counts - 1)
        pos[starts[sel] + j] = True
    none = kept == 0
    pos[starts[none] + counts[none] - 1] = True
    return pos


def frequency_baseline(table: Any, *, fit_split: str = "train", eval_split: str = "val",
                       chunk_rows: int = 200_000) -> dict[str, Any]:
    """Top-1 of ranking each row's candidates by their `fit_split` selection counts.

    Ties go to the earlier candidate. Reads the cache in row chunks, so memory stays bounded.
    """
    split_ids = {"train": 0, "val": 1, "test": 2}
    if eval_split == "test":
        raise ValueError("the gate never evaluates the test split")
    split = np.asarray(table["split"])
    counts: dict[int, int] = {}
    n_rows = int(split.shape[0])
    for r0 in range(0, n_rows, chunk_rows):
        r1 = min(n_rows, r0 + chunk_rows)
        lo, hi = int(table.cand_off[r0]), int(table.cand_off[r1])
        row_of = np.repeat(np.arange(r0, r1), np.asarray(table.cand_off[r0 + 1 : r1 + 1] - table.cand_off[r0:r1]))
        take = _positive_mask(table, r0, r1) & (split[row_of] == split_ids[fit_split])
        uniq, cnt = np.unique(_candidate_keys(table, lo, hi)[take], return_counts=True)
        for k, c in zip(uniq.tolist(), cnt.tolist()):
            counts[k] = counts.get(k, 0) + c
    known = np.array(sorted(counts), dtype=np.int64)
    known_counts = np.array([counts[k] for k in known.tolist()], dtype=np.int64)

    def train_counts(keys: np.ndarray) -> np.ndarray:
        if known.size == 0:
            return np.zeros(keys.shape, dtype=np.int64)
        at = np.minimum(np.searchsorted(known, keys), known.size - 1)
        return np.where(known[at] == keys, known_counts[at], 0)

    hits: list[np.ndarray] = []
    maps: list[np.ndarray] = []
    for r0 in range(0, n_rows, chunk_rows):
        r1 = min(n_rows, r0 + chunk_rows)
        rows = np.arange(r0, r1)[split[r0:r1] == split_ids[eval_split]]
        rows = rows[np.asarray(table["cand_count"][rows]) > 0]
        if rows.size == 0:
            continue
        lo = int(table.cand_off[r0])
        scores = train_counts(_candidate_keys(table, lo, int(table.cand_off[r1])))
        pos = _positive_mask(table, r0, r1)
        starts = (table.cand_off[rows] - lo).astype(np.int64)
        n = np.asarray(table["cand_count"][rows], dtype=np.int64)
        width = int(n.max())
        grid = starts[:, None] + np.arange(width)[None, :]
        valid = np.arange(width)[None, :] < n[:, None]
        grid = np.where(valid, grid, starts[:, None])
        row_scores = np.where(valid, scores[grid], -1)
        best = row_scores.argmax(axis=1)
        hits.append(pos[starts + best])
        maps.append(np.asarray(table["map_id"][rows], dtype=np.int64))
    cat = (lambda xs, dt: np.concatenate(xs) if xs else np.zeros((0,), dtype=dt))
    return _grouped_hits(cat(hits, bool), cat(maps, np.int64))


def compute_frequency_baselines(
    data_dir: Path | None, cache_dir: Path | None, *, workers: int | None = None
) -> dict[str, Any]:
    """Train-split frequency baselines on the val split for every #38 §5.1 policy task."""
    from tm_bot.agents.card_data import load_card_data
    from tm_bot.models.vocab import CardVocab
    from tm_bot.training.baselines import FrequencyBaseline
    from tm_bot.training.dataset import load_warmstart_corpus, resolve_warmstart_dir
    from tm_bot.training.encoded_cache import ensure_encoded_cache
    from tm_bot.training.multitask import default_cache_dir

    data_path = resolve_warmstart_dir(data_dir)
    vocab = CardVocab.from_card_data(load_card_data())
    corpus = load_warmstart_corpus(data_path, vocab=vocab, load_drafts=False, load_outcomes=False)
    setup = FrequencyBaseline.fit_train(corpus).evaluate_split(corpus, "val")
    out: dict[str, Any] = {"corp": setup["corp"], "prelude": setup["prelude_top1"]}
    caches = ensure_encoded_cache(
        data_path, cache_dir if cache_dir is not None else default_cache_dir(data_path),
        tasks=[name for _, name in BASELINE_TASKS], vocab=vocab, workers=workers,
    )
    for task, name in BASELINE_TASKS:
        out[task] = frequency_baseline(caches[name])
    return out


# --- live agents --------------------------------------------------------------------------------


class ValueRecordingSearchAgent(ISMCTSAgent):
    """Search agent that also records its reported (post-temperature) P(win) at each non-forced
    decision.

    Used only for the live value-calibration diagnostics; actions are unchanged.
    """

    def __init__(self, inner: ISMCTSAgent, entries: list[tuple[int, float]]) -> None:
        super().__init__(inner.policy_agent, config=inner.config, seed=inner._seed, version=inner.version)
        self.value_entries = entries

    def __deepcopy__(self, memo: dict[int, Any]) -> ValueRecordingSearchAgent:
        clone = super().__deepcopy__(memo)
        clone.value_entries = self.value_entries
        return clone

    @classmethod
    def act_batch_with_samplers(cls, agents, observations, choices, samplers):  # type: ignore[no-untyped-def]
        out = super().act_batch_with_samplers(agents, observations, choices, samplers)
        idx = [i for i, c in enumerate(choices) if len(c.candidates) > 1]
        if idx:
            pv = agents[0].policy_agent
            _, values = pv.predict_batch(
                [observations[i] for i in idx], [choices[i] for i in idx], compute_value=True
            )
            for i, p in zip(idx, pv.win_probability(values)):
                gen = int(observations[i].get("public", {}).get("generation", 0))
                agents[i].value_entries.append((gen, float(p)))
        return out


class ValueRecordingComparatorAgent:
    """Comparator wrapper recording its calibrated win probability at each non-forced decision."""

    def __init__(self, inner: Any, entries: list[tuple[int, float]]) -> None:
        self.inner = inner
        self.value_entries = entries
        self.version = inner.version

    def __deepcopy__(self, memo: dict[int, Any]) -> ValueRecordingComparatorAgent:
        clone = self.__class__(copy.deepcopy(self.inner, memo), self.value_entries)
        memo[id(self)] = clone
        return clone

    def reset(self, seed: int | None = None) -> None:
        self.inner.reset(seed)

    def act(self, observation: Any, choice: Any) -> int:
        index = self.inner.act(observation, choice)
        if len(choice.candidates) > 1 and self.inner.last_decision_value is not None:
            gen = int(observation.get("public", {}).get("generation", 0))
            self.value_entries.append((gen, float(self.inner.last_decision_value)))
        return index


def agent_factory(agent: str, checkpoint: Path, device: str, settings: Mapping[str, Any] | None) -> Callable[[], Agent]:
    """Fresh agent per game sharing one loaded network: light search (main) or standalone (comparator)."""
    if agent == "main":
        from tm_bot.search.agent import make_warmstart_search_agent

        return lambda: make_warmstart_search_agent(checkpoint=str(checkpoint), device=device, **dict(settings or {}))
    from tm_bot.comparator.agent import make_comparator_agent

    return lambda: make_comparator_agent(checkpoint=str(checkpoint), device=device, play_temperature="flat")


def _recording(agent: str, factory: Callable[[], Agent], recorders: list[list[tuple[int, float]]]) -> Callable[[], Agent]:
    def make() -> Agent:
        entries: list[tuple[int, float]] = []
        recorders.append(entries)
        inner = factory()
        if agent == "main":
            return ValueRecordingSearchAgent(inner, entries)  # type: ignore[arg-type]
        return ValueRecordingComparatorAgent(inner, entries)  # type: ignore[return-value]

    return make


def value_entries_with_outcomes(
    outcomes: Sequence[GameRunResult],
    metas: Sequence[PairedGameMeta],
    recorders: Sequence[Sequence[tuple[int, float]]],
) -> list[tuple[int, float, float]]:
    """`(generation, win prob, candidate utility)` of every recorded decision in a completed game."""
    if len(recorders) != len(outcomes):
        raise RuntimeError(f"value recorder count ({len(recorders)}) != games ({len(outcomes)})")
    rows: list[tuple[int, float, float]] = []
    for out, meta, entries in zip(outcomes, metas, recorders):
        if not out.ok or out.result is None:
            continue
        res = out.result
        utility = 0.5 if res.tie else (1.0 if res.winner == meta.a_seat else 0.0)
        rows += [(int(g), float(p), utility) for g, p in entries]
    return rows


def run_pairing(
    spec: PairingSpec,
    *,
    agent: str,
    candidate: Callable[[], Agent],
    opponent: Callable[[], Agent],
    record_values: bool,
    num_seeds: int,
    seed_start: int,
    maps: Sequence[str],
    workers: int,
    batch_size: int,
    max_decisions: int,
) -> tuple[list[GameRunResult], list[PairedGameMeta], list[tuple[int, float, float]] | None, float]:
    """Play one seat-swapped `dev-*` pairing through the eval harness's schedule and runner."""
    recorders: list[list[tuple[int, float]]] = []
    cand = _recording(agent, candidate, recorders) if record_values else candidate
    specs, metas = build_paired_specs(
        agent_a=cand, agent_b=opponent, seed_keys=gate_seed_keys(num_seeds, start=seed_start), maps=maps
    )
    t0 = time.perf_counter()
    outcomes = play_games(specs, num_workers=workers, batch_size=batch_size, max_decisions=max_decisions)
    wall = time.perf_counter() - t0
    values = value_entries_with_outcomes(outcomes, metas, recorders) if record_values else None
    return outcomes, metas, values, wall


def _save_pairing(path: Path, signature: Mapping[str, Any], outcomes: Sequence[GameRunResult],
                  metas: Sequence[PairedGameMeta], values: Any, wall: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "signature": dict(signature),
        "wall_time_s": wall,
        "outcomes": [_serialize_outcome(o) for o in outcomes],
        "metas": [asdict(m) for m in metas],
        "value_entries": values,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _load_pairing(path: Path, signature: Mapping[str, Any]) -> tuple[Any, Any, Any, float] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("signature") != dict(signature):
        raise ValueError(f"{path} was played with different settings; delete it or drop --resume")
    outcomes = [_deserialize_outcome(o) for o in payload["outcomes"]]
    metas = [PairedGameMeta(**m) for m in payload["metas"]]
    values = payload.get("value_entries")
    values = None if values is None else [tuple(v) for v in values]
    return outcomes, metas, values, float(payload["wall_time_s"])


# --- command line -------------------------------------------------------------------------------


def _display(path: Path | None) -> str | None:
    """Repo-relative POSIX path when inside the repo (reports live in `docs/`)."""
    if path is None:
        return None
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip() or None
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pre-self-play gate (#47, #38 §5) for one v2 warm start.")
    parser.add_argument("--agent", required=True, choices=("main", "comparator"))
    parser.add_argument("--run-dir", type=Path, required=True, help="#46 run directory (metrics.json, best/).")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Default <run-dir>/best.")
    parser.add_argument("--v1-checkpoint", type=Path, default=None,
                        help="Archived v1 warm start (default runs/warmstart/best or runs/comparator-warmstart-full/best).")
    parser.add_argument("--out", type=Path, required=True, help="Markdown report path; JSON is written beside it.")
    parser.add_argument("--work-dir", type=Path, default=None, help="Per-pairing game records (default <run-dir>/gate).")
    parser.add_argument("--resume", action="store_true", help="Reuse pairings already recorded in --work-dir.")
    parser.add_argument("--data-dir", type=Path, default=None, help="v2 warm-start parquet directory.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Encoded cache (default <data-dir>_cache).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-decisions", type=int, default=DEFAULT_MAX_DECISIONS)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--skip-validation", action="store_true", help="Smoke runs: validation checks fail as missing.")
    parser.add_argument("--skip-live", action="store_true", help="Validation only: live checks fail as not run.")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Smoke runs only: seeds per pairing (the schedule checks then fail).")
    parser.add_argument("--maps", nargs="+", default=None, help="Smoke runs only: subset of supported maps.")
    args = parser.parse_args(argv)
    if args.seeds is not None and args.seeds < 1:
        parser.error("--seeds must be at least 1")

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = (args.checkpoint or args.run_dir / "best").resolve()
    v1_checkpoint = (args.v1_checkpoint or V1_CHECKPOINTS[args.agent]).resolve()
    work_dir = args.work_dir or args.run_dir / "gate"
    maps = tuple(args.maps) if args.maps else SUPPORTED_MAPS
    unknown = [m for m in maps if m not in SUPPORTED_MAPS]
    if unknown:
        parser.error(f"unsupported maps: {unknown}")

    metrics: dict[str, Any] = {}
    value = None
    baselines = None
    metrics_path = None
    if not args.skip_validation:
        metrics, value, metrics_path = load_validation(args.run_dir, args.agent, checkpoint)
        print("[gate] computing frequency baselines ...", flush=True)
        baselines = compute_frequency_baselines(args.data_dir, args.cache_dir)

    summaries: dict[str, Any] = {}
    value_temperature = None
    if not args.skip_live:
        main_agent = args.agent == "main"
        settings = LIGHT_SEARCH if main_agent else None
        v1_settings = V1_LIGHT_SEARCH if main_agent else None
        cand = agent_factory(args.agent, checkpoint, device, settings)
        if main_agent:
            # Recorded live P(win) is post-temperature; a refit must not reuse old records.
            value_temperature = cand().policy_agent.value_temperature  # type: ignore[attr-defined]
        opponents: dict[str, Callable[[], Agent]] = {
            "b0": resolve_agent_factory("b0"),
            "b1": resolve_agent_factory("b1"),
            "self": cand,
            "v1": agent_factory(args.agent, v1_checkpoint, device, v1_settings),
        }
        for spec in pairings_for(args.agent):
            raw = spec.candidate == "raw"
            num_seeds = spec.num_seeds if args.seeds is None else args.seeds
            record_values = spec.opponent in VALUE_RECORDING_OPPONENTS
            signature = {
                "pairing": spec.key, "agent": args.agent, "checkpoint": str(checkpoint),
                "opponent": str(v1_checkpoint) if spec.opponent == "v1" else spec.opponent,
                "settings": RAW_POLICY if raw else settings, "num_seeds": num_seeds,
                "seed_start": args.seed_start, "maps": list(maps), "max_decisions": args.max_decisions,
            }
            if spec.opponent == "v1" and v1_settings is not None:
                signature["opponent_settings"] = v1_settings
            if value_temperature is not None:
                signature["value_temperature"] = value_temperature
            path = work_dir / f"{spec.key}.json"
            loaded = _load_pairing(path, signature) if args.resume else None
            if loaded is None:
                print(f"[gate] {spec.label}: {num_seeds * len(maps) * 2} games ...", flush=True)
                candidate = agent_factory(args.agent, checkpoint, device, RAW_POLICY) if raw else cand
                loaded = run_pairing(
                    spec, agent=args.agent, candidate=candidate, opponent=opponents[spec.opponent],
                    record_values=record_values, num_seeds=num_seeds, seed_start=args.seed_start,
                    maps=maps, workers=args.workers, batch_size=args.batch_size, max_decisions=args.max_decisions,
                )
                _save_pairing(path, signature, *loaded)
            outcomes, metas, values, wall = loaded
            summaries[spec.key] = summarize_pairing(
                spec, outcomes, metas, candidate_name="candidate" + (" (raw policy)" if raw else ""),
                opponent_name=spec.opponent, value_entries=values, wall_time_s=wall,
            )
            print(f"[gate] {spec.label}: done in {wall:.0f} s", flush=True)

    val = metrics.get("val")
    gate = evaluate_gate(agent=args.agent, val=val, baselines=baselines, value=value, summaries=summaries)
    version = metrics.get("version")
    if version is None and summaries:
        version = next(iter(summaries.values()))["report"]["agent_a"]["version"]
    result = {
        "schema_version": GATE_SCHEMA_VERSION,
        "agent": args.agent,
        "version": version,
        "gate": gate,
        "config": {
            "checkpoint": _display(checkpoint),
            "metrics_path": _display(metrics_path),
            "v1_checkpoint": _display(v1_checkpoint),
            "candidate_setting": (
                "light search " + ", ".join(f"{k}={v}" for k, v in LIGHT_SEARCH.items())
                if args.agent == "main" else "comparator standalone (sampled eval policy)"
            ),
            "raw_setting": ("search agent with num_simulations=0 (policy head)" if args.agent == "main" else None),
            "v1_setting": (
                "light search " + ", ".join(f"{k}={v}" for k, v in V1_LIGHT_SEARCH.items())
                if args.agent == "main" else None
            ),
            "value_temperature": value_temperature if value is None else value.get("temperature"),
            "max_decisions": args.max_decisions,
            "workers": f"{args.workers} / {args.batch_size}",
            "seed_start": args.seed_start,
            "maps": list(maps),
            "git_commit": _git_commit(),
        },
        "pairings": {s.key: summaries.get(s.key) for s in pairings_for(args.agent)},
        "val": val,
        "value": value,
        "baselines": baselines,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = format_gate_report(result)
    args.out.write_text(report, encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report, flush=True)
    print(f"[gate] verdict {gate['verdict']}; wrote {args.out} and {args.out.with_suffix('.json')}", flush=True)
    return 0 if gate["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
