from __future__ import annotations

import builtins
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from tm_bot.data.extract import run_extraction
from tm_bot.data.inventory import run_inventory
from tm_bot.data.splits import assign_split

FIXTURE_CARDS = [
    "CrediCor",
    "Ecoline",
    "Helion",
    "Vitor",
    "Tharsis Republic",
    "Allied Bank",
    "Biolab",
    "Business Empire",
    "Dome Farming",
    "Donation",
    "Galilean Mining",
    "Loan",
    "Power Generation",
    "Acquired Company",
    "Algae",
    "Arctic Algae",
    "Birds",
    "Business Contacts",
    "CEO's Favorite Project",
    "Comet",
    "Decomposers",
    "Dust Seals",
    "Farming",
    "Fish",
    "Food Factory",
    "Fuel Factory",
    "Giant Space Mirror",
    "Grass",
    "Heather",
    "Herbivores",
    "Ice Cap Melting",
    "Imported Nitrogen",
    "Lichen",
    "Livestock",
    "Mangrove",
    "Mass Converter",
    "Mine",
    "Mining Area",
    "Noctis City",
    "Permafrost Extraction",
    "Research",
    "Shuttles",
    "Space Elevator",
    "Towing A Comet",
    "Trees",
    "Wave Power",
    "Worms",
    # Opponent-isolation canary cards
    "Saturn Systems",
    "Mining Operations",
    "Capital",
]


def _base_game(
    *,
    table_id: str,
    perspective: str,
    p0: str,
    p0_name: str,
    p1: str,
    p1_name: str,
    map_name: str = "Hellas",
    conceded: bool = False,
    scraper_version: str = "v2025.10.07.1844",
    p0_vp: int = 104,
    p1_vp: int = 90,
    term_p0_vp: int = 104,
    term_p1_vp: int = 90,
) -> dict[str, Any]:
    last_desc = (
        f"{p1_name} concedes the game. | The end of the game: {p0_name} wins!"
        if conceded
        else (
            f"It is the end of the generation 10 and Mars is terraformed! | "
            f"{p0_name} scores {term_p0_vp} TOTAL VP | "
            f"{p1_name} scores {term_p1_vp} TOTAL VP | "
            f"The end of the game: {p0_name} wins!"
        )
    )
    return {
        "table_id": table_id,
        "replay_id": table_id,
        "player_perspective": perspective,
        "map": map_name,
        "conceded": conceded,
        "draft_on": True,
        "prelude_on": True,
        "corporate_era_on": True,
        "colonies_on": False,
        "beginners_corporations_on": False,
        "winner": p0_name,
        "generations": 10,
        "metadata": {"total_moves": 8, "scraper_version": scraper_version},
        "players": {
            p0: {
                "name": p0_name,
                "corporation": "CrediCor",
                "final_vp": p0_vp,
                "starting_hand": {
                    "corporations": ["CrediCor", "Ecoline"],
                    "preludes": ["Allied Bank", "Biolabs", "Donation", "Loan"],
                    "project_cards": [
                        "Acquired Company",
                        "Algae",
                        "Arctic Algae",
                        "Birds",
                        "Business Contacts",
                        "CEO's Favourite Project",
                        "Comet",
                        "Decomposers",
                        "Dust Seals",
                        "Farming",
                    ],
                },
            },
            p1: {
                "name": p1_name,
                "corporation": "Vitor",
                "final_vp": p1_vp,
                "starting_hand": {
                    "corporations": ["Helion", "Vitor"],
                    "preludes": ["Business Empire", "Dome Farming", "Galilean Mining", "Power Generation"],
                    "project_cards": [
                        "Fish",
                        "Food Factory",
                        "Fuel Factory",
                        "Giant Space Mirror",
                        "Grass",
                        "Heather",
                        "Herbivores",
                        "Ice cap Melting",
                        "Imported Nitrogen",
                        "Lichen",
                    ],
                },
            },
        },
        "moves": [
            {
                "move_number": 1,
                "player_id": p0,
                "player_name": p0_name,
                "action_type": "other",
                "description": f"Module: Corporate Era | Map: {map_name} | {p0_name} is starting player for this generation",
                "cards_kept": None,
            },
            {
                "move_number": 2,
                "player_id": p1,
                "player_name": p1_name,
                "action_type": "other",
                "description": f"{p1_name} keeps 5 cards",
                "cards_kept": {p1: ["Vitor", "Business Empire", "Dome Farming", "Fish", "Grass"]},
            },
            {
                "move_number": 3,
                "player_id": p0,
                "player_name": p0_name,
                "action_type": "other",
                "description": f"{p0_name} keeps 6 cards",
                "cards_kept": {
                    p0: ["CrediCor", "Allied Bank", "Biolabs", "Algae", "Birds", "Comet"],
                    p1: ["Vitor", "Business Empire", "Dome Farming", "Fish", "Grass"],
                },
            },
            {
                "move_number": 21,
                "player_id": p0,
                "player_name": p0_name,
                "action_type": "pass",
                "description": f"{p0_name} drafts 4 cards: Mass Converter, Towing a Comet, Wave Power, Worms",
                "card_options": {
                    p0: ["Mass Converter", "Towing a Comet", "Wave Power", "Worms"],
                    p1: ["Mangrove", "Mine", "Research", "Shuttles"],
                },
            },
            {
                "move_number": 22,
                "player_id": p0,
                "player_name": p0_name,
                "action_type": "draft",
                "description": "You draft Towing a Comet",
                "card_drafted": "Towing a Comet",
                "reason": "draft",
            },
            {
                "move_number": 23,
                "player_id": p1,
                "player_name": p1_name,
                "action_type": "draft",
                "description": f"{p1_name} drafts Mangrove",
                "card_drafted": "Mangrove",
                "reason": "draft",
                "card_options": {
                    p0: ["Mine", "Research", "Shuttles"],
                    p1: ["Mass Converter", "Wave Power", "Worms"],
                },
            },
            {
                "move_number": 24,
                "player_id": p0,
                "player_name": p0_name,
                "action_type": "other",
                "description": f"{p0_name} keeps Towing a Comet",
                "cards_kept": {p0: ["Towing a Comet"]},
            },
            {
                "move_number": 30,
                "player_id": p0,
                "player_name": p0_name,
                "action_type": "other",
                "description": last_desc,
            },
        ],
    }


def _find_table_id_for_split(target_split: str, start: int = 500000) -> str:
    tid = start
    while True:
        if assign_split(str(tid)) == target_split:
            return str(tid)
        tid += 1


def test_end_to_end_extraction_grouping_isolation_splits_and_yield_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_dir = tmp_path / "archive"
    reg_path = tmp_path / "card_registry.json"
    reg_path.write_text(json.dumps(FIXTURE_CARDS), encoding="utf-8")

    tid_train = _find_table_id_for_split("train", 100000)
    tid_test_overlap = _find_table_id_for_split("test", 200000)
    tid_test_holdout = _find_table_id_for_split("test", 300000)
    tid_conceded = _find_table_id_for_split("val", 400000)
    tid_score_bad = _find_table_id_for_split("train", 500000)
    tid_amazonis = _find_table_id_for_split("train", 600000)

    def write_game(rel: str, payload: dict[str, Any]) -> Path:
        p = archive_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload), encoding="utf-8")
        return p

    # 1. Duplicate perspectives for tid_train:
    #    - 1001 perspective fails outcome (score mismatch)
    #    - 1002 perspective passes outcome -> group prefers 1002 and produces NO duplicate rows!
    g_train_1001 = _base_game(
        table_id=tid_train,
        perspective="1001",
        p0="1001",
        p0_name="Alice",
        p1="1002",
        p1_name="Bob",
        map_name="Hellas",
        p0_vp=20,
        p1_vp=20,
        term_p0_vp=104,
        term_p1_vp=90,
    )
    g_train_1002 = _base_game(
        table_id=tid_train,
        perspective="1002",
        p0="1001",
        p0_name="Alice",
        p1="1002",
        p1_name="Bob",
        map_name="Hellas",
    )
    write_game(f"1001/game_{tid_train}_1001.json", g_train_1001)
    write_game(f"1002/game_{tid_train}_1002.json", g_train_1002)

    # 2. Opponent isolation canary test (in test_overlap with player 1001 from train)
    g_canary = _base_game(
        table_id=tid_test_overlap,
        perspective="1001",
        p0="1001",
        p0_name="Alice",
        p1="2002",
        p1_name="Carol",
        map_name="Tharsis",
    )
    g_canary["players"]["2002"]["starting_hand"]["corporations"] = ["Saturn Systems", "Vitor"]
    g_canary["players"]["2002"]["starting_hand"]["preludes"][2] = "Mining Operations"
    g_canary["players"]["2002"]["starting_hand"]["project_cards"][1] = "Capital"
    g_canary["moves"][3]["card_options"]["2002"] = ["Mangrove", "Capital", "Research", "Shuttles"]
    write_game(f"1001/game_{tid_test_overlap}_1001.json", g_canary)

    # 3. True player-holdout test game (neither 3001 nor 3002 appears in train)
    g_holdout = _base_game(
        table_id=tid_test_holdout,
        perspective="3001",
        p0="3001",
        p0_name="Dan",
        p1="3002",
        p1_name="Eve",
        map_name="Vastitas Borealis (Nova)",
    )
    write_game(f"3001/game_{tid_test_holdout}_3001.json", g_holdout)

    # 4. Conceded game: keeps setup_choices and draft_picks, NO setup_outcomes
    g_conc = _base_game(
        table_id=tid_conceded,
        perspective="4001",
        p0="4001",
        p0_name="Fay",
        p1="4002",
        p1_name="Gus",
        map_name="Элизий",
        conceded=True,
    )
    write_game(f"4001/game_{tid_conceded}_4001.json", g_conc)

    # 5. Score/text disagreement game: keeps setup_choices and draft_picks, NO setup_outcomes
    g_bad_score = _base_game(
        table_id=tid_score_bad,
        perspective="5001",
        p0="5001",
        p0_name="Hal",
        p1="5002",
        p1_name="Ivy",
        map_name="Tharsis",
        p0_vp=20,
        p1_vp=20,
        term_p0_vp=110,
        term_p1_vp=95,
    )
    write_game(f"5001/game_{tid_score_bad}_5001.json", g_bad_score)

    # 6. Non-candidate Amazonis game (must be quarantined at index level without opening)
    g_amaz = _base_game(
        table_id=tid_amazonis,
        perspective="6001",
        p0="6001",
        p0_name="Jon",
        p1="6002",
        p1_name="Kim",
        map_name="Amazonis Planitia",
    )
    amaz_path = write_game(f"6001/game_{tid_amazonis}_6001.json", g_amaz)

    index_path = tmp_path / "index.sqlite"
    run_inventory(archive_dir, index_path, progress_interval=0)

    # Track file opens during extraction: max 1 open at a time, Amazonis never opened
    active_opens = 0
    max_concurrent_opens = 0
    opened_paths: list[str] = []
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
        nonlocal active_opens, max_concurrent_opens
        handle = real_open(file, *args, **kwargs)
        if str(archive_dir) in str(file):
            active_opens += 1
            opened_paths.append(str(file))
            if active_opens > max_concurrent_opens:
                max_concurrent_opens = active_opens
            return TrackedFile(handle)
        return handle

    monkeypatch.setattr(builtins, "open", counting_open)

    out_dir = tmp_path / "warmstart"
    report = run_extraction(
        archive_dir=archive_dir,
        index_path=index_path,
        out_dir=out_dir,
        card_registry_path=reg_path,
        progress_interval=0,
    )

    assert max_concurrent_opens == 1
    assert active_opens == 0
    assert str(amaz_path) not in opened_paths

    # Verify Parquet outputs exist and have expected row counts (5 eligible tables * 2 players = 10 setup rows)
    setup_tbl = pq.read_table(out_dir / "setup_choices.parquet")
    draft_tbl = pq.read_table(out_dir / "draft_picks.parquet")
    outcome_tbl = pq.read_table(out_dir / "setup_outcomes.parquet")

    setup_rows = setup_tbl.to_pylist()
    draft_rows = draft_tbl.to_pylist()
    outcome_rows = outcome_tbl.to_pylist()

    assert len(setup_rows) == 10  # no duplicate from 2 perspectives of tid_train
    assert len(draft_rows) == 10  # 2 picks per table * 5 tables
    # Outcomes only for the 3 non-conceded, score-consistent tables (tid_train, tid_test_overlap, tid_test_holdout)
    assert len(outcome_rows) == 6
    outcome_tids = {r["table_id"] for r in outcome_rows}
    assert outcome_tids == {tid_train, tid_test_overlap, tid_test_holdout}
    assert tid_conceded not in outcome_tids
    assert tid_score_bad not in outcome_tids

    # Preferred outcome-passing perspective (1002) used for tid_train
    tid_train_setup = [r for r in setup_rows if r["table_id"] == tid_train]
    assert {r["perspective_player"] for r in tid_train_setup} == {"1002"}

    # Opponent isolation: canary cards from Carol (2002) never appear in Alice (1001) rows
    alice_setup = next(r for r in setup_rows if r["table_id"] == tid_test_overlap and r["player_id"] == "1001")
    alice_blobs = json.dumps(alice_setup)
    for canary in ("Saturn Systems", "Mining Operations", "Capital"):
        assert canary not in alice_blobs

    alice_draft = next(r for r in draft_rows if r["table_id"] == tid_test_overlap and r["player_id"] == "1001")
    assert "Capital" not in alice_draft["packet_cards"]

    # Splits and player-holdout flags
    for r in setup_rows:
        assert r["split"] == assign_split(r["table_id"])
    holdout_tids = {r["table_id"] for r in setup_rows if r["is_player_holdout"]}
    assert holdout_tids == {tid_test_holdout}

    # Yield report JSON and Markdown
    report_json = json.loads((out_dir / "yield_report.json").read_text(encoding="utf-8"))
    report_md = (out_dir / "yield_report.md").read_text(encoding="utf-8")
    assert report["examples"]["setup_choices"] == 10
    assert report_json["examples"]["setup_outcomes"] == 6
    assert "unsupported_map" in report_json["tables"]["quarantined_by_reason"]
    assert "Tharsis" in report_md
    assert "Vastitas Borealis" in report_md

    # CLI invocation
    cli_out = tmp_path / "warmstart_cli"
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "tm_bot.data.extract",
            "--archive",
            str(archive_dir),
            "--index",
            str(index_path),
            "--out",
            str(cli_out),
            "--card-registry",
            str(reg_path),
            "--progress-interval",
            "0",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "`setup_choices`: 10" in res.stdout
    assert (cli_out / "setup_choices.parquet").is_file()


def test_checkpoint_pause_and_resume_matches_uninterrupted_run(tmp_path: Path) -> None:
    archive_dir = tmp_path / "archive"
    reg_path = tmp_path / "card_registry.json"
    reg_path.write_text(json.dumps(FIXTURE_CARDS), encoding="utf-8")

    tid_train = _find_table_id_for_split("train", 100000)
    tid_test = _find_table_id_for_split("test", 300000)
    tid_conc = _find_table_id_for_split("val", 400000)

    def write_game(rel: str, payload: dict[str, Any]) -> None:
        p = archive_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload), encoding="utf-8")

    write_game(
        f"1001/game_{tid_train}_1001.json",
        _base_game(table_id=tid_train, perspective="1001", p0="1001", p0_name="Alice", p1="1002", p1_name="Bob"),
    )
    write_game(
        f"3001/game_{tid_test}_3001.json",
        _base_game(table_id=tid_test, perspective="3001", p0="3001", p0_name="Dan", p1="3002", p1_name="Eve"),
    )
    write_game(
        f"4001/game_{tid_conc}_4001.json",
        _base_game(
            table_id=tid_conc,
            perspective="4001",
            p0="4001",
            p0_name="Fay",
            p1="4002",
            p1_name="Gus",
            conceded=True,
        ),
    )

    index_path = tmp_path / "index.sqlite"
    run_inventory(archive_dir, index_path, progress_interval=0)

    full_dir = tmp_path / "full"
    full_report = run_extraction(
        archive_dir=archive_dir,
        index_path=index_path,
        out_dir=full_dir,
        card_registry_path=reg_path,
        progress_interval=0,
    )
    assert full_report["status"] == "complete"

    resumed_dir = tmp_path / "resumed"
    # Force pause after every single table group (max_seconds=0.0) and resume until complete.
    chunks = 0
    while True:
        chunks += 1
        res = run_extraction(
            archive_dir=archive_dir,
            index_path=index_path,
            out_dir=resumed_dir,
            card_registry_path=reg_path,
            progress_interval=0,
            max_seconds=0.0,
            resume=chunks > 1,
        )
        if res["status"] == "complete":
            resumed_report = res
            break
        assert res["status"] == "paused"
        assert chunks < 10

    assert chunks >= 2
    assert not (resumed_dir / "_checkpoint").exists()
    assert resumed_report["examples"] == full_report["examples"]
    assert resumed_report["splits"] == full_report["splits"]
    assert resumed_report["tables"] == full_report["tables"]

    for fname in ("setup_choices.parquet", "draft_picks.parquet", "setup_outcomes.parquet"):
        assert pq.read_table(resumed_dir / fname).to_pylist() == pq.read_table(full_dir / fname).to_pylist()

