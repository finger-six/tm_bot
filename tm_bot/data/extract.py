"""Streaming warm-start dataset extraction, table grouping, splits, and yield reporting."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection

try:
    import orjson as _orjson
except ImportError:  # pragma: no cover
    _orjson = None

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from tm_bot.data.cards import CardRegistry, DEFAULT_CARD_REGISTRY_PATH
from tm_bot.data.gates import (
    DEFAULT_EXCLUDED_SCRAPER_VERSIONS,
    SUPPORTED_MAPS,
    GateResult,
    PlayerSetup,
    RecordEvaluation,
    check_index_row,
    evaluate_record,
)
from tm_bot.data.inventory import DEFAULT_INDEX_PATH, _get_peak_rss_bytes, normalize_map
from tm_bot.data.logstate import LogDecision, LogStateResult, reconstruct_log_state
from tm_bot.data.splits import SplitName, assign_split

DEFAULT_WARMSTART_DIR = Path("data") / "warmstart"

_SETUP_CHOICES_SCHEMA = pa.schema(
    [
        ("path", pa.string()),
        ("content_hash", pa.string()),
        ("table_id", pa.string()),
        ("perspective_player", pa.string()),
        ("scraper_version", pa.string()),
        ("map", pa.string()),
        ("seat", pa.int32()),
        ("player_id", pa.string()),
        ("offered_corporations", pa.list_(pa.string())),
        ("offered_preludes", pa.list_(pa.string())),
        ("offered_projects", pa.list_(pa.string())),
        ("chosen_corporation", pa.string()),
        ("chosen_preludes", pa.list_(pa.string())),
        ("chosen_projects", pa.list_(pa.string())),
        ("split", pa.string()),
        ("is_player_holdout", pa.bool_()),
    ]
)

_STR_LIST = pa.list_(pa.string())
_INT_LIST = pa.list_(pa.int32())


def _snapshot_fields() -> list[tuple[str, pa.DataType]]:
    """The shared snapshot column group of the gen 2+ v2 rows (see `tm_bot.data.logstate`)."""
    fields: list[tuple[str, pa.DataType]] = [
        ("phase", pa.string()),
        ("temperature", pa.int32()),
        ("oxygen", pa.int32()),
        ("oceans", pa.int32()),
    ]
    stocks = ("mc", "steel", "titanium", "plants", "energy", "heat")
    for prefix in ("me", "opp"):
        fields.append((f"{prefix}_tr", pa.int32()))
        fields += [(f"{prefix}_{stock}", pa.int32()) for stock in stocks]
        fields += [(f"{prefix}_{stock}_prod", pa.int32()) for stock in stocks]
        fields += [
            (f"{prefix}_tags", _INT_LIST),
            (f"{prefix}_cities", pa.int32()),
            (f"{prefix}_greeneries", pa.int32()),
            (f"{prefix}_hand_count", pa.int32()),
            (f"{prefix}_tableau", _STR_LIST),
        ]
    fields += [
        ("me_hand_exact", pa.bool_()),
        ("me_hand", _STR_LIST),
        ("me_preludes_in_hand", _STR_LIST),
        ("me_drafted_this_round", _STR_LIST),
        ("milestone_names", _STR_LIST),
        ("milestone_seats", _INT_LIST),
        ("award_names", _STR_LIST),
        ("award_seats", _INT_LIST),
        ("me_corporation", pa.string()),
        ("me_setup_preludes", _STR_LIST),
        ("me_setup_projects_bought", _STR_LIST),
        ("me_setup_corps_discarded", _STR_LIST),
        ("me_setup_preludes_discarded", _STR_LIST),
        ("me_setup_projects_discarded", _STR_LIST),
        ("me_research_generations", _INT_LIST),
        ("me_research_offered", pa.list_(_STR_LIST)),
        ("me_research_kept", pa.list_(_STR_LIST)),
        ("me_research_discarded", pa.list_(_STR_LIST)),
        ("me_discarded_seen", _STR_LIST),
    ]
    return fields


_SNAPSHOT_FIELDS = _snapshot_fields()
_TABLE_KEY_FIELDS: list[tuple[str, pa.DataType]] = [
    ("path", pa.string()),
    ("content_hash", pa.string()),
    ("table_id", pa.string()),
    ("perspective_player", pa.string()),
    ("scraper_version", pa.string()),
    ("map", pa.string()),
    ("seat", pa.int32()),
    ("player_id", pa.string()),
]
_SPLIT_FIELDS: list[tuple[str, pa.DataType]] = [("split", pa.string()), ("is_player_holdout", pa.bool_())]

_DRAFT_PICKS_SCHEMA = pa.schema(
    [
        ("path", pa.string()),
        ("content_hash", pa.string()),
        ("table_id", pa.string()),
        ("perspective_player", pa.string()),
        ("scraper_version", pa.string()),
        ("map", pa.string()),
        ("seat", pa.int32()),
        ("player_id", pa.string()),
        ("move_number", pa.int32()),
        ("draft_round_id", pa.int32()),
        ("pick_index", pa.int32()),
        ("generation", pa.int32()),
        ("packet_cards", pa.list_(pa.string())),
        ("chosen_card", pa.string()),
        ("split", pa.string()),
        ("is_player_holdout", pa.bool_()),
        ("has_snapshot", pa.bool_()),
        *_SNAPSHOT_FIELDS,
    ]
)

_RESEARCH_BUYS_SCHEMA = pa.schema(
    [
        *_TABLE_KEY_FIELDS,
        ("generation", pa.int32()),
        ("move_number", pa.int32()),
        ("packet_cards", _STR_LIST),
        ("kept_cards", _STR_LIST),
        ("paid_mc", pa.int32()),
        *_SPLIT_FIELDS,
        *_SNAPSHOT_FIELDS,
    ]
)

_MIDGAME_VALUES_SCHEMA = pa.schema(
    [
        *_TABLE_KEY_FIELDS,
        ("generation", pa.int32()),
        ("move_number", pa.int32()),
        ("won", pa.bool_()),
        ("vp_margin", pa.int32()),
        ("my_final_vp", pa.int32()),
        ("opp_final_vp", pa.int32()),
        *_SPLIT_FIELDS,
        *_SNAPSHOT_FIELDS,
    ]
)

_SETUP_OUTCOMES_SCHEMA = pa.schema(
    [
        ("path", pa.string()),
        ("content_hash", pa.string()),
        ("table_id", pa.string()),
        ("perspective_player", pa.string()),
        ("scraper_version", pa.string()),
        ("map", pa.string()),
        ("seat", pa.int32()),
        ("player_id", pa.string()),
        ("chosen_corporation", pa.string()),
        ("chosen_preludes", pa.list_(pa.string())),
        ("chosen_projects", pa.list_(pa.string())),
        ("won", pa.bool_()),
        ("final_vp", pa.int32()),
        ("vp_margin", pa.int32()),
        ("generations", pa.int32()),
        ("split", pa.string()),
        ("is_player_holdout", pa.bool_()),
    ]
)


@dataclass
class _EvaluatedPerspective:
    path: str
    content_hash: str | None
    evaluation: RecordEvaluation
    logstate: LogStateResult | None = None


_NEW_TABLES: tuple[tuple[str, pa.Schema], ...] = (
    ("research_buys", _RESEARCH_BUYS_SCHEMA),
    ("midgame_values", _MIDGAME_VALUES_SCHEMA),
)
_ALL_KINDS: tuple[str, ...] = ("setup_choices", "draft_picks", "setup_outcomes", "research_buys", "midgame_values")


def _v2_clean(persp: _EvaluatedPerspective) -> bool:
    return persp.logstate is not None and persp.logstate.table_reason is None


def _decision_row(
    persp: _EvaluatedPerspective, table_id: str, map_name: str, d: LogDecision, split: SplitName
) -> dict[str, Any]:
    ev = persp.evaluation
    return {
        "path": persp.path,
        "content_hash": persp.content_hash,
        "table_id": table_id,
        "perspective_player": ev.perspective_player,
        "scraper_version": ev.scraper_version,
        "map": map_name,
        "seat": d.seat,
        "player_id": d.player_id,
        "generation": d.generation,
        "move_number": d.move_number,
        "split": split,
        "is_player_holdout": False,
        **d.snapshot,
    }


def _loads_json_bytes(raw: bytes) -> Any:
    view = raw[3:] if raw.startswith(b"\xef\xbb\xbf") else raw
    if _orjson is not None:
        return _orjson.loads(view)
    return json.loads(view.decode("utf-8"))


def _bucket_map(raw_map: str | None) -> str:
    norm = normalize_map(raw_map)
    if norm in SUPPORTED_MAPS or norm == "Amazonis Planitia":
        return norm
    return "Other/Missing"


def _pick_best_perspective(
    perspectives: list[_EvaluatedPerspective],
    *,
    prefer_player_id: str | None = None,
) -> _EvaluatedPerspective:
    return max(
        perspectives,
        key=lambda item: (
            1 if item.evaluation.outcome.passed else 0,
            1 if (prefer_player_id is not None and item.evaluation.perspective_player == prefer_player_id) else 0,
            len(item.evaluation.drafts.picks),
            item.path,
        ),
    )


def _flush_buffer(writer: pq.ParquetWriter, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    if not rows:
        return
    table = pa.Table.from_pylist(rows, schema=schema)
    writer.write_table(table)
    rows.clear()


def _apply_holdout_postpass(
    part_paths: list[Path],
    final_path: Path,
    schema: pa.Schema,
    holdout_tables: set[str],
) -> tuple[dict[str, int], int]:
    """Rewrite parquet part files in streaming batches with finalized is_player_holdout flags."""
    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    holdout_count = 0
    holdout_set_arr = pa.array(sorted(holdout_tables), type=pa.string()) if holdout_tables else None
    table_id_idx = schema.get_field_index("table_id")
    split_idx = schema.get_field_index("split")
    holdout_idx = schema.get_field_index("is_player_holdout")

    writer = pq.ParquetWriter(final_path, schema, compression="snappy")
    try:
        for part_path in part_paths:
            if not part_path.is_file():
                continue
            pf = pq.ParquetFile(part_path)
            try:
                if pf.metadata.num_rows > 0:
                    for batch in pf.iter_batches(batch_size=65536):
                        splits = batch.column(split_idx)
                        for s in ("train", "val", "test"):
                            split_counts[s] += int(pc.sum(pc.equal(splits, s)).as_py() or 0)

                        if holdout_set_arr is not None:
                            holdout_col = pc.is_in(batch.column(table_id_idx), value_set=holdout_set_arr)
                        else:
                            holdout_col = pa.array([False] * batch.num_rows, type=pa.bool_())

                        holdout_count += int(pc.sum(holdout_col).as_py() or 0)
                        new_cols = list(batch.columns)
                        new_cols[holdout_idx] = holdout_col
                        new_batch = pa.RecordBatch.from_arrays(new_cols, schema=schema)
                        writer.write_batch(new_batch)
            finally:
                del pf
    finally:
        writer.close()

    return split_counts, holdout_count


def format_yield_report_markdown(report: dict[str, Any]) -> str:
    """Format the warm-start yield report dictionary as readable Markdown."""
    files = report["files"]
    tables = report["tables"]
    examples = report["examples"]
    splits = report["splits"]
    draft_diag = report["draft_diagnostics"]

    lines = [
        "# Warm-Start Dataset Yield Report",
        "",
        f"- Elapsed: {report['elapsed_seconds']:.2f}s",
        f"- Peak RSS: {report['peak_rss_mib']:.1f} MiB" if report.get("peak_rss_mib") is not None else "- Peak RSS: n/a",
        "",
        "## Gate Funnel",
        "",
        "| Stage | Files | Distinct `table_id`s |",
        "| --- | ---: | ---: |",
        f"| Total in index | {files['total']:,} | {tables['total_distinct']:,} |",
        f"| Passed index pre-gate | {files['passed_index']:,} | {tables['passed_index']:,} |",
        f"| Passed full eligibility | {files['passed_eligibility']:,} | {tables['passed_eligibility']:,} |",
        f"| Passed setup gate | {files['passed_setup']:,} | {tables['passed_setup']:,} |",
        f"| Passed outcome gate | {files['passed_outcome']:,} | {tables['passed_outcome']:,} |",
        "",
        "## Examples Extracted",
        "",
        f"- `setup_choices`: {examples['setup_choices']:,}",
        f"- `draft_picks`: {examples['draft_picks']:,} (dropped unverified/undone draft rows: {draft_diag['dropped_count']:,})",
        f"- `setup_outcomes`: {examples['setup_outcomes']:,}",
        f"- `research_buys`: {examples.get('research_buys', 0):,}",
        f"- `midgame_values`: {examples.get('midgame_values', 0):,}",
        f"- `draft_picks` without a gen 2+ snapshot (a generation 1 pick, or the table failed a table-level v2 gate): "
        f"{examples.get('draft_picks_without_snapshot', 0):,}",
        "",
        "## Split Sizes",
        "",
        "| Kind | Train | Val | Test | Test (Player Holdout) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for kind in [k for k in _ALL_KINDS if k in splits]:
        sp = splits[kind]
        lines.append(
            f"| `{kind}` | {sp['train']:,} | {sp['val']:,} | {sp['test']:,} | {sp['test_player_holdout']:,} |"
        )

    lines.extend(["", "## Per-Map Yield", ""])
    lines.append("| Map | Index Files | Distinct Tables | Setup Passed (Tables) | Outcome Passed (Tables) | Setup Examples | Draft Examples | Outcome Examples |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for mname, mstats in report["by_map"].items():
        lines.append(
            f"| {mname} | {mstats['index_files']:,} | {mstats['distinct_tables']:,} | "
            f"{mstats['passed_setup_tables']:,} | {mstats['passed_outcome_tables']:,} | "
            f"{mstats['examples']['setup_choices']:,} | {mstats['examples']['draft_picks']:,} | "
            f"{mstats['examples']['setup_outcomes']:,} |"
        )

    lines.extend(["", "## Quarantined Files by Reason (Eligibility + Setup)", ""])
    if files["quarantined_by_reason"]:
        for reason, cnt in sorted(files["quarantined_by_reason"].items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- `{reason}`: {cnt:,}")
    else:
        lines.append("- (none)")

    lines.extend(["", "## Quarantined Distinct Tables by Reason (Eligibility + Setup)", ""])
    if tables["quarantined_by_reason"]:
        for reason, cnt in sorted(tables["quarantined_by_reason"].items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- `{reason}`: {cnt:,}")
    else:
        lines.append("- (none)")

    lines.extend(["", "## Outcome Quarantines Among Setup-Valid Tables", ""])
    if tables["outcome_quarantined_by_reason"]:
        for reason, cnt in sorted(tables["outcome_quarantined_by_reason"].items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- `{reason}`: {cnt:,}")
    else:
        lines.append("- (none)")

    v2 = report.get("v2_quarantine") or {}
    lines.extend(["", "## v2 Gates and Quarantines (per player, from that player's chosen perspective)", ""])
    if v2:
        lines.append("Keys are `<task or gate>:<reason>`. `table:<reason>` counts players whose chosen perspective")
        lines.append("failed a table-level v2 gate; their draft picks keep only the v1 columns.")
        lines.append("")
        for reason, cnt in sorted(v2.items()):
            lines.append(f"- `{reason}`: {cnt:,}")
    else:
        lines.append("- (none)")

    if draft_diag["dropped_by_reason"]:
        lines.extend(["", "## Dropped Draft Rows by Reason", ""])
        for reason, cnt in sorted(draft_diag["dropped_by_reason"].items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- `{reason}`: {cnt:,}")

    return "\n".join(lines) + "\n"


def run_extraction(
    archive_dir: Path | str,
    index_path: Path | str = DEFAULT_INDEX_PATH,
    out_dir: Path | str = DEFAULT_WARMSTART_DIR,
    *,
    card_registry_path: Path | str = DEFAULT_CARD_REGISTRY_PATH,
    excluded_versions: Collection[str | None] = DEFAULT_EXCLUDED_SCRAPER_VERSIONS,
    progress_interval: int = 2000,
    limit_tables: int | None = None,
    limit_files: int | None = None,
    batch_tables: int = 1000,
    max_seconds: float | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Stream over indexed archive files, apply gates, group by table_id, and write Parquet + yield report."""
    archive_root = Path(archive_dir)
    if not archive_root.is_dir():
        raise FileNotFoundError(f"Archive directory does not exist: {archive_root}")
    db_path = Path(index_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"Index file does not exist: {db_path}")

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_root / "_checkpoint"
    state_path = ckpt_dir / "state.json"

    registry = CardRegistry.from_json(card_registry_path)
    chunk_start = time.perf_counter()

    map_buckets = list(SUPPORTED_MAPS) + ["Amazonis Planitia", "Other/Missing"]

    if resume and state_path.is_file():
        saved = json.loads(state_path.read_text(encoding="utf-8"))
        part_index = int(saved["part_index"])
        nulls_done = bool(saved["nulls_done"])
        last_table_id: str | None = saved.get("last_table_id")
        tables_processed = int(saved["tables_processed"])
        prior_elapsed = float(saved["elapsed_seconds"])
        prior_peak_rss: int | None = saved.get("peak_rss_bytes")

        file_counts = Counter[str](saved["file_counts"])
        file_quarantine = Counter[str](saved["file_quarantine"])
        table_counts = Counter[str](saved["table_counts"])
        table_quarantine = Counter[str](saved["table_quarantine"])
        outcome_quarantine = Counter[str](saved["outcome_quarantine"])
        draft_dropped_reasons = Counter[str](saved["draft_dropped_reasons"])
        example_counts = Counter[str](saved["example_counts"])
        v2_counts = Counter[str](saved.get("v2_counts", {}))
        by_map: dict[str, dict[str, Any]] = {}
        for m in map_buckets:
            ms = saved["by_map"][m]
            by_map[m] = {
                "index_files": int(ms["index_files"]),
                "distinct_tables": int(ms["distinct_tables"]),
                "passed_setup_tables": int(ms["passed_setup_tables"]),
                "passed_outcome_tables": int(ms["passed_outcome_tables"]),
                "quarantined_tables_by_reason": Counter(ms["quarantined_tables_by_reason"]),
                "examples": dict(ms["examples"]),
            }
        train_players: set[str] = set(saved["train_players"])
        test_table_players: dict[str, tuple[str, ...]] = {
            k: tuple(v) for k, v in saved["test_table_players"].items()
        }
    else:
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        part_index = 0
        nulls_done = False
        last_table_id = None
        tables_processed = 0
        prior_elapsed = 0.0
        prior_peak_rss = None

        file_counts = Counter[str]()
        file_quarantine = Counter[str]()
        table_counts = Counter[str]()
        table_quarantine = Counter[str]()
        outcome_quarantine = Counter[str]()
        draft_dropped_reasons = Counter[str]()
        example_counts = Counter[str]()
        v2_counts = Counter[str]()
        by_map = {
            m: {
                "index_files": 0,
                "distinct_tables": 0,
                "passed_setup_tables": 0,
                "passed_outcome_tables": 0,
                "quarantined_tables_by_reason": Counter(),
                "examples": {k: 0 for k in _ALL_KINDS},
            }
            for m in map_buckets
        }
        train_players = set()
        test_table_players = {}

    setup_part = ckpt_dir / f"setup_choices.part_{part_index:04d}.parquet"
    draft_part = ckpt_dir / f"draft_picks.part_{part_index:04d}.parquet"
    outcome_part = ckpt_dir / f"setup_outcomes.part_{part_index:04d}.parquet"

    setup_writer = pq.ParquetWriter(setup_part, _SETUP_CHOICES_SCHEMA, compression="snappy")
    draft_writer = pq.ParquetWriter(draft_part, _DRAFT_PICKS_SCHEMA, compression="snappy")
    outcome_writer = pq.ParquetWriter(outcome_part, _SETUP_OUTCOMES_SCHEMA, compression="snappy")

    new_writers = {
        name: pq.ParquetWriter(ckpt_dir / f"{name}.part_{part_index:04d}.parquet", schema, compression="snappy")
        for name, schema in _NEW_TABLES
    }
    new_bufs: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in _NEW_TABLES}

    setup_buf: list[dict[str, Any]] = []
    draft_buf: list[dict[str, Any]] = []
    outcome_buf: list[dict[str, Any]] = []

    current_tid: str | None = None
    group_map_bucket: str = "Other/Missing"
    group_index_failures: list[GateResult] = []
    group_perspectives: list[_EvaluatedPerspective] = []
    paused = False

    def _finalize_table_group() -> None:
        nonlocal tables_processed, last_table_id
        if current_tid is None:
            return
        tables_processed += 1
        last_table_id = current_tid
        table_counts["total_distinct"] += 1
        by_map[group_map_bucket]["distinct_tables"] += 1

        if not group_perspectives:
            reason = group_index_failures[0].reason if group_index_failures else "unknown_index_failure"
            assert reason is not None
            table_quarantine[reason] += 1
            by_map[group_map_bucket]["quarantined_tables_by_reason"][reason] += 1
            return

        table_counts["passed_index"] += 1

        elig_passing = [p for p in group_perspectives if p.evaluation.eligibility.passed]
        if not elig_passing:
            reason = group_perspectives[0].evaluation.eligibility.reason or "eligibility_failed"
            table_quarantine[reason] += 1
            by_map[group_map_bucket]["quarantined_tables_by_reason"][reason] += 1
            return

        table_counts["passed_eligibility"] += 1

        setup_passing = [p for p in elig_passing if p.evaluation.setup.passed]
        if not setup_passing:
            reason = elig_passing[0].evaluation.setup.reason or "setup_failed"
            table_quarantine[reason] += 1
            by_map[group_map_bucket]["quarantined_tables_by_reason"][reason] += 1
            return

        table_counts["passed_setup"] += 1
        by_map[group_map_bucket]["passed_setup_tables"] += 1

        split: SplitName = assign_split(current_tid)
        canon_persp = _pick_best_perspective(setup_passing)
        canon_players = canon_persp.evaluation.setup.players
        player_ids = tuple(ps.player_id for ps in canon_players)
        if split == "train":
            train_players.update(player_ids)
        elif split == "test":
            test_table_players[current_tid] = player_ids

        # Emit setup_choices (one row per player, strictly from that player's own offers/choices)
        for p_setup in canon_players:
            pid = p_setup.player_id
            chosen_persp = _pick_best_perspective(setup_passing, prefer_player_id=pid)
            chosen_p_setup: PlayerSetup = next(
                ps for ps in chosen_persp.evaluation.setup.players if ps.player_id == pid
            )
            ev = chosen_persp.evaluation
            setup_buf.append(
                {
                    "path": chosen_persp.path,
                    "content_hash": chosen_persp.content_hash,
                    "table_id": current_tid,
                    "perspective_player": ev.perspective_player,
                    "scraper_version": ev.scraper_version,
                    "map": ev.map_name or group_map_bucket,
                    "seat": chosen_p_setup.seat,
                    "player_id": pid,
                    "offered_corporations": list(chosen_p_setup.offered_corporations),
                    "offered_preludes": list(chosen_p_setup.offered_preludes),
                    "offered_projects": list(chosen_p_setup.offered_projects),
                    "chosen_corporation": chosen_p_setup.chosen_corporation,
                    "chosen_preludes": list(chosen_p_setup.chosen_preludes),
                    "chosen_projects": list(chosen_p_setup.chosen_projects),
                    "split": split,
                    "is_player_holdout": False,
                }
            )
            example_counts["setup_choices"] += 1
            by_map[group_map_bucket]["examples"]["setup_choices"] += 1

        # Emit verified draft_picks (and gen 2+ research_buys) per player from that player's preferred
        # perspective, preferring perspectives that pass the table-level v2 gates.
        v2_passing = [p for p in setup_passing if _v2_clean(p)] or setup_passing
        counted_draft_persps: set[str] = set()
        for p_setup in canon_players:
            pid = p_setup.player_id
            chosen_persp = _pick_best_perspective(v2_passing, prefer_player_id=pid)
            ev = chosen_persp.evaluation
            ls = chosen_persp.logstate
            decisions: dict[tuple[str, int], LogDecision] = {}
            if ls is not None and ls.table_reason is None:
                v2_counts.update(ls.counts_by_player.get(pid) or {})
                decisions = {(d.kind, d.move_number): d for d in ls.decisions if d.player_id == pid}
            else:
                v2_counts[f"table:{ls.table_reason if ls is not None else 'not_reconstructed'}"] += 1
            if chosen_persp.path not in counted_draft_persps:
                counted_draft_persps.add(chosen_persp.path)
                for reason in ev.drafts.dropped_reasons:
                    draft_dropped_reasons[reason] += 1
            for pick in ev.drafts.picks:
                if pick.player_id != pid:
                    continue
                state = decisions.get(("draft_pick", pick.move_number))
                if state is not None and state.drop_reason is not None:
                    continue  # hand-lost or income-quarantined seat (counted in v2_counts)
                if state is None:
                    example_counts["draft_picks_without_snapshot"] += 1
                draft_buf.append(
                    {
                        **(state.snapshot if state is not None else {}),
                        "has_snapshot": state is not None,
                        "path": chosen_persp.path,
                        "content_hash": chosen_persp.content_hash,
                        "table_id": current_tid,
                        "perspective_player": ev.perspective_player,
                        "scraper_version": ev.scraper_version,
                        "map": ev.map_name or group_map_bucket,
                        "seat": pick.seat,
                        "player_id": pid,
                        "move_number": pick.move_number,
                        "draft_round_id": pick.draft_round_id,
                        "pick_index": pick.pick_index,
                        "generation": pick.generation,
                        "packet_cards": list(pick.packet_cards),
                        "chosen_card": pick.chosen_card,
                        "split": split,
                        "is_player_holdout": False,
                    }
                )
                example_counts["draft_picks"] += 1
                by_map[group_map_bucket]["examples"]["draft_picks"] += 1

            for d in decisions.values():
                if d.kind != "research_buy" or d.drop_reason is not None:
                    continue
                row = _decision_row(chosen_persp, current_tid, ev.map_name or group_map_bucket, d, split)
                row.update(d.payload)
                new_bufs["research_buys"].append(row)
                example_counts["research_buys"] += 1
                by_map[group_map_bucket]["examples"]["research_buys"] += 1

        # Emit setup_outcomes only if an outcome-passing perspective exists
        outcome_passing = [p for p in setup_passing if p.evaluation.outcome.passed]
        if outcome_passing:
            table_counts["passed_outcome"] += 1
            by_map[group_map_bucket]["passed_outcome_tables"] += 1
            out_persp = _pick_best_perspective(outcome_passing)
            out_ev = out_persp.evaluation
            out_res = out_ev.outcome
            assert out_res.scores_by_player is not None
            scores = out_res.scores_by_player
            for p_setup in out_ev.setup.players:
                pid = p_setup.player_id
                opp_vp = next(v for k, v in scores.items() if k != pid)
                outcome_buf.append(
                    {
                        "path": out_persp.path,
                        "content_hash": out_persp.content_hash,
                        "table_id": current_tid,
                        "perspective_player": out_ev.perspective_player,
                        "scraper_version": out_ev.scraper_version,
                        "map": out_ev.map_name or group_map_bucket,
                        "seat": p_setup.seat,
                        "player_id": pid,
                        "chosen_corporation": p_setup.chosen_corporation,
                        "chosen_preludes": list(p_setup.chosen_preludes),
                        "chosen_projects": list(p_setup.chosen_projects),
                        "won": pid == out_res.winner_player_id,
                        "final_vp": out_res.scores_by_player[pid],
                        "vp_margin": scores[pid] - opp_vp,
                        "generations": out_res.generations,
                        "split": split,
                        "is_player_holdout": False,
                    }
                )
                example_counts["setup_outcomes"] += 1
                by_map[group_map_bucket]["examples"]["setup_outcomes"] += 1
                # Mid-game value points from this player's own preferred v2-clean perspective.
                val_persp = _pick_best_perspective(v2_passing, prefer_player_id=pid)
                if val_persp.logstate is None or val_persp.logstate.table_reason is not None:
                    continue
                for d in val_persp.logstate.decisions:
                    if d.player_id != pid or d.kind not in ("deal_boundary", "action") or d.drop_reason is not None:
                        continue
                    row = _decision_row(
                        val_persp, current_tid, val_persp.evaluation.map_name or group_map_bucket, d, split
                    )
                    row.update(
                        won=pid == out_res.winner_player_id,
                        vp_margin=scores[pid] - opp_vp,
                        my_final_vp=scores[pid],
                        opp_final_vp=opp_vp,
                    )
                    new_bufs["midgame_values"].append(row)
                    example_counts["midgame_values"] += 1
                    by_map[group_map_bucket]["examples"]["midgame_values"] += 1
        else:
            out_reason = setup_passing[0].evaluation.outcome.reason or "outcome_failed"
            outcome_quarantine[out_reason] += 1

        if tables_processed % batch_tables == 0:
            _flush_buffer(setup_writer, setup_buf, _SETUP_CHOICES_SCHEMA)
            _flush_buffer(draft_writer, draft_buf, _DRAFT_PICKS_SCHEMA)
            _flush_buffer(outcome_writer, outcome_buf, _SETUP_OUTCOMES_SCHEMA)
            for name, schema in _NEW_TABLES:
                _flush_buffer(new_writers[name], new_bufs[name], schema)

    _select_cols = (
        "path, size, content_hash, status, error_type, "
        "table_id, perspective_player, scraper_version, "
        "map_raw, map, player_count, "
        "draft_on, prelude_on, corporate_era_on, colonies_on, "
        "beginners_corporations_on, other_modules"
    )

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        if not nulls_done:
            null_cursor = conn.execute(
                f"SELECT {_select_cols} FROM files WHERE table_id IS NULL OR TRIM(table_id) = '' ORDER BY path"
            )
            for row in null_cursor:
                row_dict = dict(row)
                file_counts["total"] += 1
                map_bucket = _bucket_map(row_dict.get("map") or row_dict.get("map_raw"))
                by_map[map_bucket]["index_files"] += 1
                idx_gate = check_index_row(row_dict, excluded_versions=excluded_versions)
                if not idx_gate.passed:
                    reason = idx_gate.reason or "index_gate_failed"
                    file_quarantine[reason] += 1
                else:  # pragma: no cover - table_id IS NULL always fails check_index_row
                    file_quarantine["missing_or_inconsistent_table_id"] += 1
            nulls_done = True

        if last_table_id is not None:
            cursor = conn.execute(
                f"SELECT {_select_cols} FROM files "
                "WHERE table_id IS NOT NULL AND TRIM(table_id) != '' AND table_id > ? "
                "ORDER BY table_id, path",
                (last_table_id,),
            )
        else:
            cursor = conn.execute(
                f"SELECT {_select_cols} FROM files "
                "WHERE table_id IS NOT NULL AND TRIM(table_id) != '' "
                "ORDER BY table_id, path"
            )

        for row in cursor:
            row_dict = dict(row)
            tid = str(row_dict["table_id"]).strip()
            map_bucket = _bucket_map(row_dict.get("map") or row_dict.get("map_raw"))

            if tid != current_tid:
                if current_tid is not None:
                    _finalize_table_group()
                    group_index_failures.clear()
                    group_perspectives.clear()
                    if (limit_tables is not None and tables_processed >= limit_tables) or (
                        limit_files is not None and file_counts["passed_index"] >= limit_files
                    ):
                        current_tid = None
                        break
                    if max_seconds is not None and (time.perf_counter() - chunk_start) >= max_seconds:
                        paused = True
                        current_tid = None
                        break
                current_tid = tid
                group_map_bucket = map_bucket

            file_counts["total"] += 1
            by_map[map_bucket]["index_files"] += 1

            idx_gate = check_index_row(row_dict, excluded_versions=excluded_versions)
            if not idx_gate.passed:
                reason = idx_gate.reason or "index_gate_failed"
                file_quarantine[reason] += 1
                group_index_failures.append(idx_gate)
                continue

            file_counts["passed_index"] += 1
            rel_path = row_dict["path"]
            abs_path = archive_root / rel_path

            try:
                with open(abs_path, "rb") as fh:
                    raw = fh.read()
                data = _loads_json_bytes(raw)
                del raw
            except (OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                file_quarantine["corrupt_or_unreadable"] += 1
                group_index_failures.append(GateResult(False, "corrupt_or_unreadable", type(exc).__name__))
                continue

            if not isinstance(data, dict):
                file_quarantine["corrupt_or_unreadable"] += 1
                group_index_failures.append(GateResult(False, "corrupt_or_unreadable", type(data).__name__))
                del data
                continue

            ev = evaluate_record(
                data,
                path=rel_path,
                card_registry=registry,
                excluded_versions=excluded_versions,
            )
            logstate = (
                reconstruct_log_state(data, setup=ev.setup, drafts=ev.drafts, card_registry=registry)
                if ev.eligibility.passed and ev.setup.passed
                else None
            )
            del data
            if (row_dict.get("size") or 0) >= 10_000_000:
                gc.collect()

            if ev.eligibility.passed:
                file_counts["passed_eligibility"] += 1
            else:
                file_quarantine[ev.eligibility.reason or "eligibility_failed"] += 1

            if ev.eligibility.passed and ev.setup.passed:
                file_counts["passed_setup"] += 1
            elif ev.eligibility.passed:
                file_quarantine[ev.setup.reason or "setup_failed"] += 1

            if ev.eligibility.passed and ev.setup.passed and ev.outcome.passed:
                file_counts["passed_outcome"] += 1

            group_perspectives.append(
                _EvaluatedPerspective(
                    path=rel_path,
                    content_hash=row_dict.get("content_hash"),
                    evaluation=ev,
                    logstate=logstate,
                )
            )

            if progress_interval and file_counts["passed_index"] % progress_interval == 0:
                print(
                    f"[extract] opened={file_counts['passed_index']} "
                    f"tables={tables_processed} setup_ex={example_counts['setup_choices']} "
                    f"draft_ex={example_counts['draft_picks']} out_ex={example_counts['setup_outcomes']}",
                    file=sys.stderr,
                    flush=True,
                )

        if not paused and current_tid is not None and (limit_tables is None or tables_processed < limit_tables):
            _finalize_table_group()

        _flush_buffer(setup_writer, setup_buf, _SETUP_CHOICES_SCHEMA)
        _flush_buffer(draft_writer, draft_buf, _DRAFT_PICKS_SCHEMA)
        _flush_buffer(outcome_writer, outcome_buf, _SETUP_OUTCOMES_SCHEMA)
        for name, schema in _NEW_TABLES:
            _flush_buffer(new_writers[name], new_bufs[name], schema)
    finally:
        conn.close()
        setup_writer.close()
        draft_writer.close()
        outcome_writer.close()
        for writer in new_writers.values():
            writer.close()

    total_elapsed = prior_elapsed + (time.perf_counter() - chunk_start)
    chunk_rss = _get_peak_rss_bytes()
    if prior_peak_rss is not None and chunk_rss is not None:
        peak_rss: int | None = max(prior_peak_rss, chunk_rss)
    else:
        peak_rss = chunk_rss if chunk_rss is not None else prior_peak_rss

    if paused:
        state_payload = {
            "part_index": part_index + 1,
            "nulls_done": nulls_done,
            "last_table_id": last_table_id,
            "tables_processed": tables_processed,
            "elapsed_seconds": total_elapsed,
            "peak_rss_bytes": peak_rss,
            "file_counts": dict(file_counts),
            "file_quarantine": dict(file_quarantine),
            "table_counts": dict(table_counts),
            "table_quarantine": dict(table_quarantine),
            "outcome_quarantine": dict(outcome_quarantine),
            "draft_dropped_reasons": dict(draft_dropped_reasons),
            "example_counts": dict(example_counts),
            "v2_counts": dict(v2_counts),
            "by_map": {
                m: {
                    "index_files": s["index_files"],
                    "distinct_tables": s["distinct_tables"],
                    "passed_setup_tables": s["passed_setup_tables"],
                    "passed_outcome_tables": s["passed_outcome_tables"],
                    "quarantined_tables_by_reason": dict(s["quarantined_tables_by_reason"]),
                    "examples": s["examples"],
                }
                for m, s in by_map.items()
            },
            "train_players": sorted(train_players),
            "test_table_players": {k: list(v) for k, v in test_table_players.items()},
        }
        tmp_state = ckpt_dir / "state.json.tmp"
        tmp_state.write_text(json.dumps(state_payload), encoding="utf-8")
        os.replace(tmp_state, state_path)
        return {
            "status": "paused",
            "part_index": part_index + 1,
            "last_table_id": last_table_id,
            "tables_processed": tables_processed,
            "elapsed_seconds": round(total_elapsed, 3),
            "peak_rss_bytes": peak_rss,
            "peak_rss_mib": round(peak_rss / (1024**2), 2) if peak_rss is not None else None,
        }

    holdout_tables = {
        tid for tid, pids in test_table_players.items() if pids and set(pids).isdisjoint(train_players)
    }

    part_indices = range(part_index + 1)
    setup_splits, setup_holdout = _apply_holdout_postpass(
        [ckpt_dir / f"setup_choices.part_{i:04d}.parquet" for i in part_indices],
        out_root / "setup_choices.parquet",
        _SETUP_CHOICES_SCHEMA,
        holdout_tables,
    )
    draft_splits, draft_holdout = _apply_holdout_postpass(
        [ckpt_dir / f"draft_picks.part_{i:04d}.parquet" for i in part_indices],
        out_root / "draft_picks.parquet",
        _DRAFT_PICKS_SCHEMA,
        holdout_tables,
    )
    outcome_splits, outcome_holdout = _apply_holdout_postpass(
        [ckpt_dir / f"setup_outcomes.part_{i:04d}.parquet" for i in part_indices],
        out_root / "setup_outcomes.parquet",
        _SETUP_OUTCOMES_SCHEMA,
        holdout_tables,
    )
    new_splits: dict[str, dict[str, int]] = {}
    for name, schema in _NEW_TABLES:
        sp, holdout = _apply_holdout_postpass(
            [ckpt_dir / f"{name}.part_{i:04d}.parquet" for i in part_indices],
            out_root / f"{name}.parquet",
            schema,
            holdout_tables,
        )
        new_splits[name] = {**sp, "test_player_holdout": holdout}
    shutil.rmtree(ckpt_dir, ignore_errors=True)

    report: dict[str, Any] = {
        "status": "complete",
        "elapsed_seconds": round(total_elapsed, 3),
        "peak_rss_bytes": peak_rss,
        "peak_rss_mib": round(peak_rss / (1024**2), 2) if peak_rss is not None else None,
        "files": {
            "total": file_counts["total"],
            "passed_index": file_counts["passed_index"],
            "passed_eligibility": file_counts["passed_eligibility"],
            "passed_setup": file_counts["passed_setup"],
            "passed_outcome": file_counts["passed_outcome"],
            "quarantined_by_reason": dict(sorted(file_quarantine.items())),
        },
        "tables": {
            "total_distinct": table_counts["total_distinct"],
            "passed_index": table_counts["passed_index"],
            "passed_eligibility": table_counts["passed_eligibility"],
            "passed_setup": table_counts["passed_setup"],
            "passed_outcome": table_counts["passed_outcome"],
            "quarantined_by_reason": dict(sorted(table_quarantine.items())),
            "outcome_quarantined_by_reason": dict(sorted(outcome_quarantine.items())),
            "test_tables": len(test_table_players),
            "test_player_holdout_tables": len(holdout_tables),
        },
        "examples": {
            "setup_choices": example_counts["setup_choices"],
            "draft_picks": example_counts["draft_picks"],
            "setup_outcomes": example_counts["setup_outcomes"],
            "research_buys": example_counts["research_buys"],
            "midgame_values": example_counts["midgame_values"],
            "draft_picks_without_snapshot": example_counts["draft_picks_without_snapshot"],
        },
        "v2_quarantine": dict(sorted(v2_counts.items())),
        "splits": {
            "setup_choices": {**setup_splits, "test_player_holdout": setup_holdout},
            "draft_picks": {**draft_splits, "test_player_holdout": draft_holdout},
            "setup_outcomes": {**outcome_splits, "test_player_holdout": outcome_holdout},
            **new_splits,
        },
        "draft_diagnostics": {
            "dropped_count": sum(draft_dropped_reasons.values()),
            "dropped_by_reason": dict(sorted(draft_dropped_reasons.items())),
        },
        "by_map": {
            m: {
                "index_files": s["index_files"],
                "distinct_tables": s["distinct_tables"],
                "passed_setup_tables": s["passed_setup_tables"],
                "passed_outcome_tables": s["passed_outcome_tables"],
                "quarantined_tables_by_reason": dict(sorted(s["quarantined_tables_by_reason"].items())),
                "examples": s["examples"],
            }
            for m, s in by_map.items()
        },
    }

    (out_root / "yield_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_root / "yield_report.md").write_text(format_yield_report_markdown(report), encoding="utf-8")
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tm_bot.data.extract",
        description="Gate and extract warm-start dataset from the indexed Terraforming Mars log archive.",
    )
    parser.add_argument("--archive", type=Path, required=True, help="Path to extracted game log archive directory.")
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help=f"SQLite inventory index path (default: {DEFAULT_INDEX_PATH}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_WARMSTART_DIR,
        help=f"Output warmstart directory (default: {DEFAULT_WARMSTART_DIR}).",
    )
    parser.add_argument(
        "--card-registry",
        type=Path,
        default=DEFAULT_CARD_REGISTRY_PATH,
        help=f"Engine-derived card registry JSON path (default: {DEFAULT_CARD_REGISTRY_PATH}).",
    )
    parser.add_argument(
        "--exclude-version",
        action="append",
        default=None,
        help="Additional scraper_version to exclude (can be passed multiple times).",
    )
    parser.add_argument(
        "--no-default-excluded-versions",
        action="store_true",
        help="Disable built-in DEFAULT_EXCLUDED_SCRAPER_VERSIONS list.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=2000,
        help="Emit progress to stderr every N opened candidate files; 0 disables (default: 2000).",
    )
    parser.add_argument(
        "--limit-tables",
        type=int,
        default=None,
        help="Optional cap on number of distinct table_ids to process (for debugging).",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Optional cap on opened (index-passing) files; stops at the next table_id boundary (small samples).",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Optional wall-clock budget per invocation; checkpoints at table_id boundary and exits with code 2.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from <out>/_checkpoint/state.json if present.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    excluded: set[str | None] = set() if args.no_default_excluded_versions else set(DEFAULT_EXCLUDED_SCRAPER_VERSIONS)
    if args.exclude_version:
        excluded.update(args.exclude_version)

    report = run_extraction(
        archive_dir=args.archive,
        index_path=args.index,
        out_dir=args.out,
        card_registry_path=args.card_registry,
        excluded_versions=excluded,
        progress_interval=args.progress_interval,
        limit_tables=args.limit_tables,
        limit_files=args.limit_files,
        max_seconds=args.max_seconds,
        resume=args.resume,
    )
    if report.get("status") == "paused":
        print(
            f"[extract] paused at table_id={report['last_table_id']} "
            f"tables={report['tables_processed']:,} elapsed={report['elapsed_seconds']}s "
            f"peak_rss_mib={report['peak_rss_mib']}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(format_yield_report_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
