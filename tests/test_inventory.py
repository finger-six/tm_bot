from __future__ import annotations

import builtins
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tracemalloc
from pathlib import Path

import pytest

from tm_bot.data.inventory import (
    format_summary,
    normalize_map,
    run_inventory,
    summarize_index,
)

FIXTURES_ARCHIVE = Path(__file__).parent / "fixtures" / "archive"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fetch_rows(db_path: Path) -> dict[str, sqlite3.Row]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM files ORDER BY path").fetchall()
        return {row["path"]: row for row in rows}
    finally:
        conn.close()


def test_normalize_map_aliases_and_localizations() -> None:
    assert normalize_map("Tharsis") == "Tharsis"
    assert normalize_map("tharsis (base)") == "Tharsis"
    assert normalize_map("Тарсис") == "Tharsis"
    assert normalize_map("Hellas") == "Hellas"
    assert normalize_map("Эллада") == "Hellas"
    assert normalize_map("Elysium") == "Elysium"
    assert normalize_map("Элизий") == "Elysium"
    assert normalize_map("Vastitas Borealis") == "Vastitas Borealis"
    assert normalize_map("vastitas_borealis") == "Vastitas Borealis"
    assert normalize_map("Vastitas Borealis (Nova)") == "Vastitas Borealis"
    assert normalize_map("ボレアリス荒野") == "Vastitas Borealis"
    assert normalize_map("Amazonis Planitia") == "Amazonis Planitia"
    assert normalize_map("Amazonis") == "Amazonis Planitia"
    assert normalize_map("Random") == "Random"
    assert normalize_map("Aléatoire") == "Random"
    assert normalize_map("  Custom Map  ") == "Custom Map"
    assert normalize_map("") is None
    assert normalize_map(None) is None


def test_inventory_field_extraction_and_malformed_handling(tmp_path: Path) -> None:
    db_path = tmp_path / "index.sqlite"
    stats = run_inventory(FIXTURES_ARCHIVE, db_path)

    assert stats.total_seen == 6
    assert stats.processed == 6
    assert stats.skipped == 0
    assert stats.errors == 2

    rows = _fetch_rows(db_path)
    assert set(rows.keys()) == {
        "1001/game_500001_1001.json",
        "1002/game_500001_1002.json",
        "1003/game_500002_1003.json",
        "1004/game_500003_1004.json",
        "1005/game_500004_1005.json",
        "1006/game_500005_1006.json",
    }

    r1 = rows["1001/game_500001_1001.json"]
    f1 = FIXTURES_ARCHIVE / "1001" / "game_500001_1001.json"
    assert r1["status"] == "ok"
    assert r1["size"] == f1.stat().st_size
    assert r1["content_hash"] == _sha256(f1)
    assert r1["table_id"] == "500001"
    assert r1["perspective_player"] == "1001"
    assert r1["scraper_version"] == "v2025.10.07.1844"
    assert r1["map"] == "Hellas"
    assert r1["map_raw"] == "Hellas"
    assert r1["player_count"] == 2
    assert r1["draft_on"] == 1
    assert r1["prelude_on"] == 1
    assert r1["corporate_era_on"] == 1
    assert r1["colonies_on"] == 0
    assert json.loads(r1["other_modules"]) == []
    assert r1["conceded"] == 0
    assert r1["move_count"] == 3
    assert r1["metadata_total_moves"] == 3

    # Older perspective: missing root conceded, move_count != metadata.total_moves
    r2 = rows["1002/game_500001_1002.json"]
    assert r2["status"] == "ok"
    assert r2["table_id"] == "500001"
    assert r2["perspective_player"] == "1002"
    assert r2["scraper_version"] == "v2025.08.06.1719"
    assert r2["conceded"] == 0  # inferred from terminal completion text
    assert r2["move_count"] == 2
    assert r2["metadata_total_moves"] == 10

    # Localized map, 3 players, colonies, beginners_corporations + Venus Next
    r3 = rows["1003/game_500002_1003.json"]
    assert r3["status"] == "ok"
    assert r3["map"] == "Elysium"
    assert r3["map_raw"] == "Элизий"
    assert r3["player_count"] == 3
    assert r3["draft_on"] == 0
    assert r3["prelude_on"] == 1
    assert r3["corporate_era_on"] == 0
    assert r3["colonies_on"] == 1
    assert json.loads(r3["other_modules"]) == ["beginners_corporations", "venus next"]
    assert r3["conceded"] == 1

    # Missing metadata & missing conceded without terminal text
    r4 = rows["1004/game_500003_1004.json"]
    assert r4["status"] == "ok"
    assert r4["map"] == "Tharsis"
    assert r4["scraper_version"] is None
    assert r4["conceded"] is None
    assert r4["move_count"] == 1

    # Malformed JSON recorded as error, not fatal
    r5 = rows["1005/game_500004_1005.json"]
    assert r5["status"] == "error"
    assert r5["error_type"] == "JSONDecodeError"
    assert r5["content_hash"] == _sha256(FIXTURES_ARCHIVE / "1005" / "game_500004_1005.json")

    # Non-dict JSON root recorded as error
    r6 = rows["1006/game_500005_1006.json"]
    assert r6["status"] == "error"
    assert r6["error_type"] == "ValueError"


def test_inventory_resume_skips_unchanged_and_rechecks_hash_on_mtime_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_copy = tmp_path / "archive"
    shutil.copytree(FIXTURES_ARCHIVE, archive_copy)
    db_path = tmp_path / "index.sqlite"

    first = run_inventory(archive_copy, db_path)
    assert first.processed == 6
    assert first.skipped == 0

    # Second run: no files should be opened for reading
    opened_archive_files: list[str] = []
    real_open = builtins.open

    def tracking_open(file, *args, **kwargs):
        file_str = str(file)
        if str(archive_copy) in file_str:
            opened_archive_files.append(file_str)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracking_open)
    second = run_inventory(archive_copy, db_path)
    assert second.processed == 0
    assert second.skipped == 6
    assert opened_archive_files == []

    # Touch mtime without changing bytes on 1001 -> opens to verify hash, skips JSON parse
    f1 = archive_copy / "1001" / "game_500001_1001.json"
    st = f1.stat()
    os.utime(f1, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))

    # Modify 1004 content and add a new file 1008
    f4 = archive_copy / "1004" / "game_500003_1004.json"
    f4_data = json.loads(f4.read_text(encoding="utf-8"))
    f4_data["prelude_on"] = True
    f4.write_text(json.dumps(f4_data), encoding="utf-8")

    new_dir = archive_copy / "1008"
    new_dir.mkdir()
    (new_dir / "game_500009_1008.json").write_text(
        json.dumps(
            {
                "table_id": "500009",
                "player_perspective": "1008",
                "map": "Vastitas",
                "conceded": False,
                "draft_on": True,
                "prelude_on": True,
                "corporate_era_on": True,
                "colonies_on": False,
                "players": {"1008": {}, "1009": {}},
                "metadata": {"scraper_version": "v2025.12.27.1015"},
                "moves": [],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(builtins, "open", real_open)
    third = run_inventory(archive_copy, db_path)
    assert third.total_seen == 7
    assert third.skipped == 5  # 4 untouched + 1 mtime-only hash match
    assert third.processed == 2  # modified 1004 + new 1008

    rows = _fetch_rows(db_path)
    assert rows["1004/game_500003_1004.json"]["prelude_on"] == 1
    assert rows["1008/game_500009_1008.json"]["map"] == "Vastitas Borealis"


def test_inventory_streams_one_file_at_a_time_with_bounded_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def make_archive(name: str, count: int) -> Path:
        root = tmp_path / name
        root.mkdir()
        payload = {
            "table_id": "900000",
            "player_perspective": "1",
            "map": "Tharsis",
            "conceded": False,
            "draft_on": True,
            "prelude_on": True,
            "corporate_era_on": True,
            "colonies_on": False,
            "players": {"1": {}, "2": {}},
            "metadata": {"scraper_version": "v2025.10.07.1844"},
            "moves": [{"move_number": i, "description": "x" * 200} for i in range(40)],
        }
        raw = json.dumps(payload)
        for i in range(count):
            sub = root / f"{i // 10:03d}"
            sub.mkdir(exist_ok=True)
            (sub / f"game_{900000 + i}_1.json").write_text(raw, encoding="utf-8")
        return root

    small_dir = make_archive("small", 10)
    large_dir = make_archive("large", 100)

    active_opens = 0
    max_concurrent_opens = 0
    total_archive_opens = 0
    real_open = builtins.open

    class TrackedFile:
        def __init__(self, inner):
            self._inner = inner

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            nonlocal active_opens
            try:
                return self._inner.__exit__(exc_type, exc, tb)
            finally:
                active_opens -= 1

        def close(self):
            nonlocal active_opens
            try:
                return self._inner.close()
            finally:
                active_opens -= 1

        def __getattr__(self, item):
            return getattr(self._inner, item)

    def counting_open(file, *args, **kwargs):
        nonlocal active_opens, max_concurrent_opens, total_archive_opens
        handle = real_open(file, *args, **kwargs)
        if str(large_dir) in str(file):
            active_opens += 1
            total_archive_opens += 1
            if active_opens > max_concurrent_opens:
                max_concurrent_opens = active_opens
            return TrackedFile(handle)
        return handle

    monkeypatch.setattr(builtins, "open", counting_open)
    tracemalloc.start()
    run_inventory(small_dir, tmp_path / "small.sqlite", batch_size=10)
    _, small_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    stats = run_inventory(large_dir, tmp_path / "large.sqlite", batch_size=10)
    _, large_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert stats.processed == 100
    assert total_archive_opens == 100
    assert max_concurrent_opens == 1
    assert active_opens == 0
    # 10x more files should not cause ~10x peak memory growth
    assert large_peak < max(small_peak * 2.5, small_peak + 1_000_000)


def test_summary_counts_and_cli(tmp_path: Path) -> None:
    db_path = tmp_path / "index.sqlite"
    run_inventory(FIXTURES_ARCHIVE, db_path)

    summary = summarize_index(db_path)
    assert summary.total_files == 6
    assert summary.ok_files == 4
    assert summary.error_files == 2
    assert summary.distinct_table_ids == 3  # 500001 (x2 perspectives), 500002, 500003
    assert summary.by_map == {"Hellas": 2, "Elysium": 1, "Tharsis": 1}
    assert summary.by_player_count == {2: 3, 3: 1}
    assert summary.by_version == {
        "v2025.10.07.1844": 1,
        "v2025.08.06.1719": 1,
        "v2025.12.27.1015": 1,
        None: 1,
    }
    assert summary.by_error == {"JSONDecodeError": 1, "ValueError": 1}
    assert summary.by_conceded == {False: 2, True: 1, None: 1}
    assert summary.flag_value_counts["draft_on"] == {True: 3, False: 1}
    assert summary.flag_value_counts["prelude_on"] == {True: 3, False: 1}
    assert summary.by_flags == {
        (True, True, True, False, "[]"): 2,
        (False, True, False, True, '["beginners_corporations", "venus next"]'): 1,
        (True, False, True, False, "[]"): 1,
    }
    assert summary.candidate_2p_prelude_draft_files == 2
    assert summary.candidate_2p_prelude_draft_tables == 1

    text = format_summary(summary)
    assert "Distinct table_ids: 3" in text
    assert "Hellas: 2" in text
    assert "JSONDecodeError: 1" in text

    # Test CLI invocation for indexing + summary
    cli_db = tmp_path / "cli.sqlite"
    res_index = subprocess.run(
        [
            sys.executable,
            "-m",
            "tm_bot.data.inventory",
            "--archive",
            str(FIXTURES_ARCHIVE),
            "--out",
            str(cli_db),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Distinct table_ids: 3" in res_index.stdout
    assert "peak_rss=" in res_index.stdout

    res_summary = subprocess.run(
        [
            sys.executable,
            "-m",
            "tm_bot.data.inventory",
            "summary",
            "--out",
            str(cli_db),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Distinct table_ids: 3" in res_summary.stdout
    assert "Errors: 2" in res_summary.stdout


def test_invalid_utf8_bom_and_null_fields(tmp_path: Path) -> None:
    archive = tmp_path / "edge"
    archive.mkdir()

    # Invalid UTF-8 bytes
    (archive / "bad_utf8.json").write_bytes(b'{"table_id": "\xff\xfe"}')

    # UTF-8 BOM + null metadata/players/moves + integer table_id
    bom_payload = json.dumps(
        {
            "table_id": 42,
            "player_perspective": 7,
            "map": "Amazonis",
            "players": None,
            "metadata": None,
            "moves": None,
        }
    ).encode("utf-8-sig")
    (archive / "bom_nulls.json").write_bytes(bom_payload)

    db_path = tmp_path / "edge.sqlite"
    stats = run_inventory(archive, db_path)
    assert stats.total_seen == 2
    assert stats.processed == 2
    assert stats.errors == 1
    assert stats.peak_rss_bytes is None or stats.peak_rss_bytes > 0

    rows = _fetch_rows(db_path)
    assert rows["bad_utf8.json"]["status"] == "error"
    assert rows["bad_utf8.json"]["error_type"] == "UnicodeDecodeError"

    bom_row = rows["bom_nulls.json"]
    assert bom_row["status"] == "ok"
    assert bom_row["table_id"] == "42"
    assert bom_row["perspective_player"] == "7"
    assert bom_row["map"] == "Amazonis Planitia"
    assert bom_row["player_count"] is None
    assert bom_row["move_count"] is None


def test_max_seconds_graceful_pause_and_resume(tmp_path: Path) -> None:
    db_path = tmp_path / "paused.sqlite"
    paused = run_inventory(FIXTURES_ARCHIVE, db_path, max_seconds=0.0)
    assert paused.stopped_early is True
    assert paused.processed == 0

    resumed = run_inventory(FIXTURES_ARCHIVE, db_path)
    assert resumed.stopped_early is False
    assert resumed.processed == 6


