from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

from tm_bot.data.extract import run_extraction
from tm_bot.data.inventory import run_inventory
from tm_bot.data.splits import assign_split, compute_player_holdout_tables

_GOLDEN = Path(__file__).resolve().parent / "fixtures" / "golden_v2"
_REPO = Path(__file__).resolve().parents[1]


def test_assign_split_is_deterministic_and_independent_of_pythonhashseed() -> None:
    expected = assign_split("500001")
    assert expected in {"train", "val", "test"}
    assert assign_split("500001") == expected

    # Same table_id produces identical split across fresh subprocesses with different PYTHONHASHSEED
    out1 = subprocess.check_output(
        [sys.executable, "-c", "from tm_bot.data.splits import assign_split; print(assign_split('767184202'))"],
        env={"PYTHONHASHSEED": "1"},
        text=True,
    ).strip()
    out2 = subprocess.check_output(
        [sys.executable, "-c", "from tm_bot.data.splits import assign_split; print(assign_split('767184202'))"],
        env={"PYTHONHASHSEED": "99999"},
        text=True,
    ).strip()
    assert out1 == out2
    assert out1 in {"train", "val", "test"}


def test_assign_split_proportions_approx_90_5_5() -> None:
    counts = {"train": 0, "val": 0, "test": 0}
    n = 4000
    for i in range(n):
        counts[assign_split(str(100_000 + i))] += 1
    assert 0.87 <= counts["train"] / n <= 0.93
    assert 0.03 <= counts["val"] / n <= 0.07
    assert 0.03 <= counts["test"] / n <= 0.07


def test_compute_player_holdout_tables() -> None:
    table_splits = {
        "t_train": "train",
        "t_val": "val",
        "t_test_overlap_one": "test",
        "t_test_overlap_both": "test",
        "t_test_holdout": "test",
    }
    table_players = {
        "t_train": ("p1", "p2"),
        "t_val": ("p3", "p4"),  # val players do NOT taint test holdout; only train does
        "t_test_overlap_one": ("p1", "p5"),  # p1 appeared in train
        "t_test_overlap_both": ("p1", "p2"),
        "t_test_holdout": ("p3", "p6"),  # neither p3 nor p6 appeared in train
    }

    holdout = compute_player_holdout_tables(table_players, table_splits)
    assert holdout == {"t_test_holdout"}


def _table_ids_covering_all_splits() -> list[str]:
    found: dict[str, str] = {}
    tid = 920000000
    while len(found) < 3:
        split = assign_split(str(tid))
        found.setdefault(split, str(tid))
        tid += 1
    return sorted(found.values())


def test_split_determinism_over_v2_tables(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    table_ids = _table_ids_covering_all_splits()
    for tid in table_ids:
        for pid in ("10000001", "10000002"):
            data = json.loads((_GOLDEN / pid / f"game_900000001_{pid}.json").read_text(encoding="utf-8"))
            data["table_id"] = data["replay_id"] = tid
            dest = archive / pid
            dest.mkdir(parents=True, exist_ok=True)
            (dest / f"game_{tid}_{pid}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    index = tmp_path / "index.sqlite"
    run_inventory(archive, index, progress_interval=0)

    runs = []
    for name in ("run1", "run2"):
        run_extraction(archive, index, tmp_path / name,
                       card_registry_path=_REPO / "data" / "card_registry.json", progress_interval=0)
        runs.append(tmp_path / name)

    for table in ("draft_picks", "research_buys", "midgame_values"):
        rows1 = pq.read_table(runs[0] / f"{table}.parquet").to_pylist()
        rows2 = pq.read_table(runs[1] / f"{table}.parquet").to_pylist()
        assert rows1 and rows1 == rows2, table
        assert {r["table_id"] for r in rows1} == set(table_ids), table
        for r in rows1:
            assert r["split"] == assign_split(r["table_id"]), (table, r["table_id"])
        flags: dict[str, set[bool]] = {}
        for r in rows1:
            flags.setdefault(r["table_id"], set()).add(r["is_player_holdout"])
        assert all(len(v) == 1 for v in flags.values()), table
