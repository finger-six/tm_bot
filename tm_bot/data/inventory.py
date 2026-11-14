"""Streaming inventory of the human Terraforming Mars log archive."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import gc
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

DEFAULT_INDEX_PATH = Path("data") / "archive_index.sqlite"

_CORE_MODULE_NAMES = {"corporate era", "prelude", "colonies"}
_CORE_FLAG_KEYS = {"draft_on", "prelude_on", "corporate_era_on", "colonies_on"}
_MODULE_RE = re.compile(r"Module:\s*([^|\r\n]+)", re.IGNORECASE)

_MAP_ALIASES: dict[str, str] = {
    # Tharsis
    "tharsis": "Tharsis",
    "tharsis (base)": "Tharsis",
    "tharsis base": "Tharsis",
    "tharsis planitia": "Tharsis",
    "base": "Tharsis",
    "original": "Tharsis",
    "mars": "Tharsis",
    "тарсис": "Tharsis",
    "тарсида": "Tharsis",
    "тарсис планиция": "Tharsis",
    "тарсис (базовая)": "Tharsis",
    "базовая": "Tharsis",
    "основная": "Tharsis",
    # Elysium
    "elysium": "Elysium",
    "élisium": "Elysium",
    "elisium": "Elysium",
    "élysium": "Elysium",
    "elysium planitia": "Elysium",
    "элизий": "Elysium",
    "элизиум": "Elysium",
    # Hellas
    "hellas": "Hellas",
    "hellas (south pole)": "Hellas",
    "hellas south pole": "Hellas",
    "hellas planitia": "Hellas",
    "эллада": "Hellas",
    "эллас": "Hellas",
    # Vastitas Borealis
    "vastitas borealis": "Vastitas Borealis",
    "vastitas": "Vastitas Borealis",
    "vastitas borealis (nova)": "Vastitas Borealis",
    "vastitas borealis nova": "Vastitas Borealis",
    "ваститас бореалис": "Vastitas Borealis",
    "ваститас": "Vastitas Borealis",
    "северная равнина": "Vastitas Borealis",
    "ボレアリス荒野": "Vastitas Borealis",
    # Amazonis Planitia
    "amazonis planitia": "Amazonis Planitia",
    "amazonis": "Amazonis Planitia",
    "амазония": "Amazonis Planitia",
    "амазонис": "Amazonis Planitia",
    "планиция амазонис": "Amazonis Planitia",
    "амазонис планиция": "Amazonis Planitia",
    # Other known BGA maps
    "terra cimmeria": "Terra Cimmeria",
    "utopia planitia": "Utopia Planitia",
    # Random
    "random": "Random",
    "aléatoire": "Random",
    "aleatoire": "Random",
    "zufällig": "Random",
    "zufallig": "Random",
    "aleatorio": "Random",
    "aleatoria": "Random",
    "casuale": "Random",
    "случайная": "Random",
    "случайная карта": "Random",
    "случайно": "Random",
    "willekeurig": "Random",
    "losowa": "Random",
    "slumpmässig": "Random",
    "slumpmassig": "Random",
    "tilfeldig": "Random",
    "tilfældig": "Random",
    "satunnainen": "Random",
    "véletlenszerű": "Random",
    "náhodná": "Random",
    "nahodna": "Random",
    "ランダム": "Random",
    "随机": "Random",
    "무작위": "Random",
}

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    content_hash TEXT,
    status TEXT NOT NULL,
    error_type TEXT,
    error_message TEXT,
    table_id TEXT,
    perspective_player TEXT,
    scraper_version TEXT,
    map_raw TEXT,
    map TEXT,
    player_count INTEGER,
    draft_on INTEGER,
    prelude_on INTEGER,
    corporate_era_on INTEGER,
    colonies_on INTEGER,
    beginners_corporations_on INTEGER,
    other_modules TEXT,
    conceded INTEGER,
    move_count INTEGER,
    metadata_total_moves INTEGER,
    indexed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_files_table_id ON files(table_id);
CREATE INDEX IF NOT EXISTS idx_files_status ON files(status);
CREATE INDEX IF NOT EXISTS idx_files_content_hash ON files(content_hash);
CREATE INDEX IF NOT EXISTS idx_files_map ON files(map);
"""

_UPSERT_SQL = """
INSERT INTO files (
    path, size, mtime_ns, content_hash, status, error_type, error_message,
    table_id, perspective_player, scraper_version, map_raw, map,
    player_count, draft_on, prelude_on, corporate_era_on, colonies_on,
    beginners_corporations_on, other_modules, conceded, move_count,
    metadata_total_moves, indexed_at
) VALUES (
    ?, ?, ?, ?, ?, ?, ?,
    ?, ?, ?, ?, ?,
    ?, ?, ?, ?, ?,
    ?, ?, ?, ?,
    ?, ?
)
ON CONFLICT(path) DO UPDATE SET
    size = excluded.size,
    mtime_ns = excluded.mtime_ns,
    content_hash = excluded.content_hash,
    status = excluded.status,
    error_type = excluded.error_type,
    error_message = excluded.error_message,
    table_id = excluded.table_id,
    perspective_player = excluded.perspective_player,
    scraper_version = excluded.scraper_version,
    map_raw = excluded.map_raw,
    map = excluded.map,
    player_count = excluded.player_count,
    draft_on = excluded.draft_on,
    prelude_on = excluded.prelude_on,
    corporate_era_on = excluded.corporate_era_on,
    colonies_on = excluded.colonies_on,
    beginners_corporations_on = excluded.beginners_corporations_on,
    other_modules = excluded.other_modules,
    conceded = excluded.conceded,
    move_count = excluded.move_count,
    metadata_total_moves = excluded.metadata_total_moves,
    indexed_at = excluded.indexed_at;
"""


@dataclass(frozen=True)
class InventoryRecord:
    path: str
    size: int
    mtime_ns: int
    content_hash: str | None
    status: str
    error_type: str | None = None
    error_message: str | None = None
    table_id: str | None = None
    perspective_player: str | None = None
    scraper_version: str | None = None
    map_raw: str | None = None
    map: str | None = None
    player_count: int | None = None
    draft_on: bool | None = None
    prelude_on: bool | None = None
    corporate_era_on: bool | None = None
    colonies_on: bool | None = None
    beginners_corporations_on: bool | None = None
    other_modules: tuple[str, ...] = ()
    conceded: bool | None = None
    move_count: int | None = None
    metadata_total_moves: int | None = None


@dataclass(frozen=True)
class InventoryRunStats:
    total_seen: int
    processed: int
    skipped: int
    errors: int
    elapsed_seconds: float
    peak_rss_bytes: int | None = None
    stopped_early: bool = False


@dataclass(frozen=True)
class InventorySummary:
    total_files: int
    ok_files: int
    error_files: int
    total_bytes: int
    distinct_table_ids: int
    by_map: dict[str | None, int]
    by_version: dict[str | None, int]
    by_player_count: dict[int | None, int]
    by_conceded: dict[bool | None, int]
    by_flags: dict[tuple[bool | None, bool | None, bool | None, bool | None, str], int]
    flag_value_counts: dict[str, dict[Any, int]]
    by_error: dict[str | None, int]
    candidate_2p_prelude_draft_files: int
    candidate_2p_prelude_draft_tables: int


def normalize_map(raw_map: str | None) -> str | None:
    """Normalize a Terraforming Mars map name across aliases and localizations."""
    if raw_map is None:
        return None
    cleaned = unicodedata.normalize("NFKC", str(raw_map)).strip()
    if not cleaned:
        return None

    key = re.sub(r"[_\-\s]+", " ", cleaned).strip().casefold()
    if key in _MAP_ALIASES:
        return _MAP_ALIASES[key]

    # Try stripping parenthetical suffixes like "Vastitas Borealis (Nova)"
    without_parens = re.sub(r"\([^)]*\)", "", key)
    without_parens = re.sub(r"\s+", " ", without_parens).strip()
    if without_parens in _MAP_ALIASES:
        return _MAP_ALIASES[without_parens]

    return cleaned


def _opt_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
    return None


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _extract_other_modules(data: dict[str, Any], moves: Any) -> tuple[str, ...]:
    modules: set[str] = set()

    for key, val in data.items():
        if (
            isinstance(key, str)
            and key.endswith("_on")
            and key not in _CORE_FLAG_KEYS
            and _opt_bool(val) is True
        ):
            modules.add(key[:-3])

    if isinstance(moves, list):
        for move in moves[:5]:
            if not isinstance(move, dict):
                continue
            desc = move.get("description")
            if not isinstance(desc, str):
                continue
            for match in _MODULE_RE.findall(desc):
                mod_name = re.sub(r"\s+", " ", match).strip().lower()
                if mod_name and mod_name not in _CORE_MODULE_NAMES:
                    modules.add(mod_name)

    return tuple(sorted(modules))


def _extract_conceded(data: dict[str, Any], moves: Any) -> bool | None:
    explicit = _opt_bool(data.get("conceded"))
    if explicit is not None:
        return explicit

    if isinstance(moves, list) and moves:
        for move in reversed(moves[-3:]):
            if not isinstance(move, dict):
                continue
            desc = move.get("description")
            if not isinstance(desc, str):
                continue
            low = desc.lower()
            if "concedes the game" in low:
                return True
            if "the end of the game:" in low or "mars is terraformed" in low:
                return False

    return None


def extract_record(
    data: dict[str, Any],
    *,
    path: str,
    size: int,
    mtime_ns: int,
    content_hash: str,
) -> InventoryRecord:
    """Extract normalized inventory metadata from one parsed game log dict."""
    metadata = data.get("metadata")
    metadata_dict = metadata if isinstance(metadata, dict) else {}

    scraper_version = (
        _opt_str(metadata_dict.get("scraper_version"))
        or _opt_str(metadata_dict.get("parser_version"))
        or _opt_str(data.get("scraper_version"))
        or _opt_str(data.get("parser_version"))
    )

    players = data.get("players")
    if isinstance(players, (dict, list)):
        player_count: int | None = len(players)
    else:
        player_count = None

    moves = data.get("moves")
    move_count = len(moves) if isinstance(moves, list) else None

    raw_total_moves = metadata_dict.get("total_moves")
    if isinstance(raw_total_moves, int) and not isinstance(raw_total_moves, bool):
        metadata_total_moves: int | None = raw_total_moves
    else:
        metadata_total_moves = None

    map_raw = _opt_str(data.get("map"))

    return InventoryRecord(
        path=path,
        size=size,
        mtime_ns=mtime_ns,
        content_hash=content_hash,
        status="ok",
        table_id=_opt_str(data.get("table_id")),
        perspective_player=_opt_str(data.get("player_perspective")),
        scraper_version=scraper_version,
        map_raw=map_raw,
        map=normalize_map(map_raw),
        player_count=player_count,
        draft_on=_opt_bool(data.get("draft_on")),
        prelude_on=_opt_bool(data.get("prelude_on")),
        corporate_era_on=_opt_bool(data.get("corporate_era_on")),
        colonies_on=_opt_bool(data.get("colonies_on")),
        beginners_corporations_on=_opt_bool(data.get("beginners_corporations_on")),
        other_modules=_extract_other_modules(data, moves),
        conceded=_extract_conceded(data, moves),
        move_count=move_count,
        metadata_total_moves=metadata_total_moves,
    )


def _iter_json_entries(archive_dir: Path) -> Iterator[tuple[str, Path, os.stat_result]]:
    """Stream `.json` files under `archive_dir` one directory at a time."""
    stack: list[tuple[Path, str]] = [(archive_dir, "")]
    while stack:
        current_dir, rel_prefix = stack.pop()
        subdirs: list[tuple[Path, str]] = []
        files: list[tuple[str, Path, os.stat_result]] = []

        with os.scandir(current_dir) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False):
                    sub_rel = f"{rel_prefix}/{entry.name}" if rel_prefix else entry.name
                    subdirs.append((Path(entry.path), sub_rel))
                elif entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(".json"):
                    file_rel = f"{rel_prefix}/{entry.name}" if rel_prefix else entry.name
                    files.append((file_rel, Path(entry.path), entry.stat(follow_symlinks=False)))

        files.sort(key=lambda item: item[0])
        for item in files:
            yield item

        subdirs.sort(key=lambda item: item[1], reverse=True)
        stack.extend(subdirs)


def _bool_to_db(val: bool | None) -> int | None:
    if val is None:
        return None
    return 1 if val else 0


def _db_to_bool(val: int | None) -> bool | None:
    if val is None:
        return None
    return bool(val)


def _open_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    return conn


def _get_peak_rss_bytes() -> int | None:
    """Return OS-reported peak resident set size in bytes, if available."""
    if os.name == "nt":
        class _ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.wintypes.DWORD),
                ("PageFaultCount", ctypes.wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
        handle = kernel32.GetCurrentProcess()
        get_mem = getattr(kernel32, "K32GetProcessMemoryInfo", None) or ctypes.windll.psapi.GetProcessMemoryInfo
        get_mem.argtypes = [ctypes.wintypes.HANDLE, ctypes.POINTER(_ProcessMemoryCounters), ctypes.wintypes.DWORD]
        get_mem.restype = ctypes.wintypes.BOOL
        if get_mem(handle, ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize)
        return None
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(rss * 1024) if sys.platform != "darwin" else int(rss)
    except ImportError:
        return None


def _hash_stream(fh: Any, chunk_size: int = 1024 * 1024) -> str:
    if hasattr(fh, "readinto"):
        return hashlib.file_digest(fh, "sha256").hexdigest()
    hasher = hashlib.sha256()
    while True:
        chunk = fh.read(chunk_size)
        if not chunk:
            break
        hasher.update(chunk)
    return hasher.hexdigest()


def run_inventory(
    archive_dir: Path | str,
    out_path: Path | str = DEFAULT_INDEX_PATH,
    *,
    verify_hash: bool = False,
    retry_errors: bool = False,
    batch_size: int = 500,
    progress_interval: int = 0,
    max_seconds: float | None = None,
) -> InventoryRunStats:
    """Stream over `archive_dir` one file at a time and update the SQLite index at `out_path`."""
    archive_root = Path(archive_dir)
    if not archive_root.is_dir():
        raise FileNotFoundError(f"Archive directory does not exist: {archive_root}")

    db_path = Path(out_path)
    conn = _open_db(db_path)

    start_time = time.perf_counter()
    total_seen = 0
    processed = 0
    skipped = 0
    errors = 0
    pending_writes = 0
    stopped_early = False

    try:
        for rel_path, abs_path, st in _iter_json_entries(archive_root):
            if max_seconds is not None and (time.perf_counter() - start_time) >= max_seconds:
                stopped_early = True
                break

            total_seen += 1
            size = st.st_size
            mtime_ns = st.st_mtime_ns

            existing = conn.execute(
                "SELECT size, mtime_ns, content_hash, status FROM files WHERE path = ?",
                (rel_path,),
            ).fetchone()

            if existing is not None:
                ex_size, ex_mtime_ns, ex_hash, ex_status = existing
                if (
                    ex_size == size
                    and ex_mtime_ns == mtime_ns
                    and ex_hash is not None
                    and not verify_hash
                    and (not retry_errors or ex_status == "ok")
                ):
                    skipped += 1
                    if progress_interval and total_seen % progress_interval == 0:
                        print(
                            f"[inventory] seen={total_seen} processed={processed} skipped={skipped} errors={errors}",
                            file=sys.stderr,
                            flush=True,
                        )
                    continue

            content_hash: str | None = None
            record: InventoryRecord
            now_iso = datetime.now(timezone.utc).isoformat()

            try:
                with open(abs_path, "rb") as fh:
                    if existing is None and size < 64 * 1024 * 1024:
                        raw = fh.read()
                        content_hash = hashlib.sha256(raw).hexdigest()
                        try:
                            data = json.loads(raw)
                        finally:
                            del raw
                    else:
                        content_hash = _hash_stream(fh)

                        if (
                            existing is not None
                            and existing[2] == content_hash
                            and (not retry_errors or existing[3] == "ok")
                        ):
                            conn.execute(
                                "UPDATE files SET size = ?, mtime_ns = ? WHERE path = ?",
                                (size, mtime_ns, rel_path),
                            )
                            pending_writes += 1
                            skipped += 1
                            if pending_writes >= batch_size:
                                conn.commit()
                                pending_writes = 0
                            if progress_interval and total_seen % progress_interval == 0:
                                print(
                                    f"[inventory] seen={total_seen} processed={processed} skipped={skipped} errors={errors}",
                                    file=sys.stderr,
                                    flush=True,
                                )
                            continue

                        fh.seek(0)
                        data = json.load(fh)

                if not isinstance(data, dict):
                    raise ValueError(f"Expected JSON object, got {type(data).__name__}")

                record = extract_record(
                    data,
                    path=rel_path,
                    size=size,
                    mtime_ns=mtime_ns,
                    content_hash=content_hash,
                )
                del data
                if size >= 10_000_000:
                    gc.collect()
            except (OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError, TypeError) as exc:
                errors += 1
                record = InventoryRecord(
                    path=rel_path,
                    size=size,
                    mtime_ns=mtime_ns,
                    content_hash=content_hash,
                    status="error",
                    error_type=type(exc).__name__,
                    error_message=str(exc)[:500],
                )

            conn.execute(
                _UPSERT_SQL,
                (
                    record.path,
                    record.size,
                    record.mtime_ns,
                    record.content_hash,
                    record.status,
                    record.error_type,
                    record.error_message,
                    record.table_id,
                    record.perspective_player,
                    record.scraper_version,
                    record.map_raw,
                    record.map,
                    record.player_count,
                    _bool_to_db(record.draft_on),
                    _bool_to_db(record.prelude_on),
                    _bool_to_db(record.corporate_era_on),
                    _bool_to_db(record.colonies_on),
                    _bool_to_db(record.beginners_corporations_on),
                    json.dumps(list(record.other_modules)),
                    _bool_to_db(record.conceded),
                    record.move_count,
                    record.metadata_total_moves,
                    now_iso,
                ),
            )
            processed += 1
            pending_writes += 1

            if pending_writes >= batch_size:
                conn.commit()
                pending_writes = 0

            if progress_interval and total_seen % progress_interval == 0:
                print(
                    f"[inventory] seen={total_seen} processed={processed} skipped={skipped} errors={errors}",
                    file=sys.stderr,
                    flush=True,
                )

        if pending_writes > 0:
            conn.commit()
    finally:
        conn.close()

    elapsed = time.perf_counter() - start_time
    return InventoryRunStats(
        total_seen=total_seen,
        processed=processed,
        skipped=skipped,
        errors=errors,
        elapsed_seconds=elapsed,
        peak_rss_bytes=_get_peak_rss_bytes(),
        stopped_early=stopped_early,
    )


def summarize_index(index_path: Path | str = DEFAULT_INDEX_PATH) -> InventorySummary:
    """Compute aggregate counts from an existing inventory SQLite index."""
    db_path = Path(index_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"Index file does not exist: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        total_files, ok_files, error_files, total_bytes = conn.execute(
            """
            SELECT
                COUNT(*),
                COALESCE(SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END), 0),
                COALESCE(SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END), 0),
                COALESCE(SUM(size), 0)
            FROM files
            """
        ).fetchone()

        (distinct_table_ids,) = conn.execute(
            "SELECT COUNT(DISTINCT table_id) FROM files WHERE status = 'ok' AND table_id IS NOT NULL"
        ).fetchone()

        by_map = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT map, COUNT(*) AS c FROM files WHERE status = 'ok' GROUP BY map ORDER BY c DESC, map ASC"
            )
        }
        by_version = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT scraper_version, COUNT(*) AS c FROM files WHERE status = 'ok' "
                "GROUP BY scraper_version ORDER BY c DESC, scraper_version ASC"
            )
        }
        by_player_count = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT player_count, COUNT(*) AS c FROM files WHERE status = 'ok' "
                "GROUP BY player_count ORDER BY c DESC, player_count ASC"
            )
        }
        by_conceded = {
            _db_to_bool(row[0]): row[1]
            for row in conn.execute(
                "SELECT conceded, COUNT(*) AS c FROM files WHERE status = 'ok' "
                "GROUP BY conceded ORDER BY c DESC, conceded ASC"
            )
        }
        by_flags = {
            (
                _db_to_bool(row[0]),
                _db_to_bool(row[1]),
                _db_to_bool(row[2]),
                _db_to_bool(row[3]),
                row[4] if row[4] is not None else "[]",
            ): row[5]
            for row in conn.execute(
                """
                SELECT draft_on, prelude_on, corporate_era_on, colonies_on, other_modules, COUNT(*) AS c
                FROM files
                WHERE status = 'ok'
                GROUP BY draft_on, prelude_on, corporate_era_on, colonies_on, other_modules
                ORDER BY c DESC
                """
            )
        }

        flag_value_counts: dict[str, dict[Any, int]] = {}
        for flag_col in ("draft_on", "prelude_on", "corporate_era_on", "colonies_on"):
            flag_value_counts[flag_col] = {
                _db_to_bool(row[0]): row[1]
                for row in conn.execute(
                    f"SELECT {flag_col}, COUNT(*) AS c FROM files WHERE status = 'ok' "
                    f"GROUP BY {flag_col} ORDER BY c DESC"
                )
            }
        flag_value_counts["other_modules"] = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT other_modules, COUNT(*) AS c FROM files WHERE status = 'ok' "
                "GROUP BY other_modules ORDER BY c DESC"
            )
        }

        by_error = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT error_type, COUNT(*) AS c FROM files WHERE status = 'error' "
                "GROUP BY error_type ORDER BY c DESC, error_type ASC"
            )
        }

        cand_files, cand_tables = conn.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT table_id)
            FROM files
            WHERE status = 'ok'
              AND player_count = 2
              AND map IN ('Tharsis', 'Elysium', 'Hellas', 'Vastitas Borealis')
              AND draft_on = 1
              AND prelude_on = 1
              AND corporate_era_on = 1
              AND colonies_on = 0
              AND other_modules = '[]'
              AND table_id IS NOT NULL
            """
        ).fetchone()
    finally:
        conn.close()

    return InventorySummary(
        total_files=total_files,
        ok_files=ok_files,
        error_files=error_files,
        total_bytes=total_bytes,
        distinct_table_ids=distinct_table_ids,
        by_map=by_map,
        by_version=by_version,
        by_player_count=by_player_count,
        by_conceded=by_conceded,
        by_flags=by_flags,
        flag_value_counts=flag_value_counts,
        by_error=by_error,
        candidate_2p_prelude_draft_files=cand_files,
        candidate_2p_prelude_draft_tables=cand_tables,
    )


def _fmt_label(val: Any) -> str:
    return "<missing>" if val is None else str(val)


def format_summary(summary: InventorySummary) -> str:
    """Format an `InventorySummary` for human-readable terminal reporting."""
    gib = summary.total_bytes / (1024**3)
    lines = [
        "=== Archive Inventory Summary ===",
        f"Total files: {summary.total_files}",
        f"OK files: {summary.ok_files}",
        f"Errors: {summary.error_files}",
        f"Total size: {summary.total_bytes:,} bytes ({gib:.2f} GiB)",
        f"Distinct table_ids: {summary.distinct_table_ids}",
        (
            "Candidate 2p supported-map Prelude+Draft+CorpEra (no Colonies/other): "
            f"{summary.candidate_2p_prelude_draft_files} files, "
            f"{summary.candidate_2p_prelude_draft_tables} distinct table_ids"
        ),
        "",
        "By map:",
    ]
    for k, v in summary.by_map.items():
        lines.append(f"  {_fmt_label(k)}: {v}")

    lines.append("")
    lines.append("By scraper_version:")
    for k, v in summary.by_version.items():
        lines.append(f"  {_fmt_label(k)}: {v}")

    lines.append("")
    lines.append("By player_count:")
    for k, v in summary.by_player_count.items():
        lines.append(f"  {_fmt_label(k)}: {v}")

    lines.append("")
    lines.append("By conceded:")
    for k, v in summary.by_conceded.items():
        lines.append(f"  {_fmt_label(k)}: {v}")

    lines.append("")
    lines.append("By individual option flags:")
    for flag_name, counts in summary.flag_value_counts.items():
        formatted = ", ".join(f"{_fmt_label(k)}={v}" for k, v in counts.items())
        lines.append(f"  {flag_name}: {formatted}")

    lines.append("")
    lines.append("By flag combination (draft, prelude, corporate_era, colonies, other_modules):")
    for (d, p, ce, col, other), v in list(summary.by_flags.items())[:25]:
        lines.append(f"  draft={_fmt_label(d)}, prelude={_fmt_label(p)}, corp={_fmt_label(ce)}, colonies={_fmt_label(col)}, other={other}: {v}")
    if len(summary.by_flags) > 25:
        lines.append(f"  ... ({len(summary.by_flags) - 25} more combinations)")

    lines.append("")
    lines.append("By error:")
    if summary.by_error:
        for k, v in summary.by_error.items():
            lines.append(f"  {_fmt_label(k)}: {v}")
    else:
        lines.append("  (none)")

    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tm_bot.data.inventory",
        description="Streaming inventory and summary of Terraforming Mars game log archives.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Path to the extracted game log archive directory.",
    )
    parser.add_argument(
        "--out",
        "--index",
        dest="out",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help=f"Output SQLite index path (default: {DEFAULT_INDEX_PATH}).",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print the summary for an existing index without scanning an archive.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip printing the index summary after running inventory.",
    )
    parser.add_argument(
        "--verify-hash",
        action="store_true",
        help="Recompute content hashes for already-indexed files even when size and mtime match.",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry files previously recorded with status='error'.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="SQLite commit batch size (default: 500).",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=5000,
        help="Emit progress to stderr every N seen files; 0 disables (default: 5000).",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Gracefully stop and commit after this many seconds (resumable on next run).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    summary_subcommand = False
    if raw_argv and raw_argv[0] == "summary":
        summary_subcommand = True
        raw_argv = raw_argv[1:]

    parser = _build_parser()
    args = parser.parse_args(raw_argv)

    if summary_subcommand or args.summary_only:
        summary = summarize_index(args.out)
        print(format_summary(summary))
        return 0

    if args.archive is None:
        parser.error("--archive is required unless running 'summary' or '--summary-only'")

    stats = run_inventory(
        args.archive,
        args.out,
        verify_hash=args.verify_hash,
        retry_errors=args.retry_errors,
        batch_size=args.batch_size,
        progress_interval=args.progress_interval,
        max_seconds=args.max_seconds,
    )
    peak_str = (
        f", peak_rss={stats.peak_rss_bytes / (1024 ** 2):.1f} MiB"
        if stats.peak_rss_bytes is not None
        else ""
    )
    state = "paused (time budget)" if stats.stopped_early else "complete"
    print(
        f"Inventory {state} in {stats.elapsed_seconds:.2f}s: "
        f"seen={stats.total_seen}, processed={stats.processed}, "
        f"skipped={stats.skipped}, errors={stats.errors}{peak_str}"
    )

    if not args.no_summary:
        summary = summarize_index(args.out)
        print()
        print(format_summary(summary))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
