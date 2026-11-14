"""Deterministic game-grouped splits and player-holdout flags."""

from __future__ import annotations

import hashlib
from typing import Iterable, Literal, Mapping

SplitName = Literal["train", "val", "test"]


def assign_split(table_id: str) -> SplitName:
    """Assign a deterministic 90/5/5 train/val/test split from SHA-256 of table_id."""
    if not isinstance(table_id, str) or not table_id.strip():
        raise ValueError("table_id must be a non-empty string")
    digest = hashlib.sha256(table_id.strip().encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "val"
    return "test"


def compute_player_holdout_tables(
    table_players: Mapping[str, Iterable[str]],
    table_splits: Mapping[str, SplitName],
) -> set[str]:
    """Return test-split table_ids whose players never appear in any train-split table."""
    train_players: set[str] = set()
    for tid, split in table_splits.items():
        if split == "train":
            for pid in table_players.get(tid, ()):
                if pid:
                    train_players.add(pid)

    holdout: set[str] = set()
    for tid, split in table_splits.items():
        if split != "test":
            continue
        players = {pid for pid in table_players.get(tid, ()) if pid}
        if players and players.isdisjoint(train_players):
            holdout.add(tid)
    return holdout
