from __future__ import annotations

import subprocess
import sys

from tm_bot.data.splits import assign_split, compute_player_holdout_tables


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
