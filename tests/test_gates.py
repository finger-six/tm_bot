from __future__ import annotations

import copy
from typing import Any

from tm_bot.data.cards import CardRegistry
from tm_bot.data.gates import (
    check_eligibility,
    check_outcome,
    evaluate_record,
    extract_verified_drafts,
    extract_verified_setup,
)

FIXTURE_CARDS = [
    "CrediCor",
    "Ecoline",
    "Helion",
    "Vitor",
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
]


def _make_valid_record() -> dict[str, Any]:
    return {
        "table_id": "500001",
        "replay_id": "500001",
        "player_perspective": "1001",
        "map": "Hellas",
        "conceded": False,
        "draft_on": True,
        "prelude_on": True,
        "corporate_era_on": True,
        "colonies_on": False,
        "beginners_corporations_on": False,
        "winner": "Alice",
        "generations": 10,
        "metadata": {"total_moves": 5, "scraper_version": "v2025.10.07.1844"},
        "players": {
            "1001": {
                "name": "Alice",
                "corporation": "CrediCor",
                "final_vp": 104,
                "final_tr": 50,
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
            "1002": {
                "name": "Bob",
                "corporation": "Vitor",
                "final_vp": 90,
                "final_tr": 42,
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
                "player_id": "1001",
                "player_name": "Alice",
                "action_type": "other",
                "description": "Module: Corporate Era | Map: Hellas | Alice is starting player for this generation",
                "cards_kept": None,
            },
            {
                "move_number": 2,
                "player_id": "1002",
                "player_name": "Bob",
                "action_type": "other",
                "description": "Bob keeps 5 cards",
                "cards_kept": {
                    "1002": ["Vitor", "Business Empire", "Dome Farming", "Fish", "Grass"],
                },
            },
            {
                "move_number": 3,
                "player_id": "1001",
                "player_name": "Alice",
                "action_type": "other",
                "description": "Alice keeps 6 cards",
                "cards_kept": {
                    "1001": ["CrediCor", "Allied Bank", "Biolabs", "Algae", "Birds", "Comet"],
                    "1002": ["Vitor", "Business Empire", "Dome Farming", "Fish", "Grass"],
                },
            },
            {
                "move_number": 4,
                "player_id": "1001",
                "player_name": "Alice",
                "action_type": "play_card",
                "description": "Alice plays card Algae",
                "card_played": "Algae",
            },
            {
                "move_number": 5,
                "player_id": "1001",
                "player_name": "Alice",
                "action_type": "other",
                "description": (
                    "It is the end of the generation 10 and Mars is terraformed! | "
                    "Alice has 40 M€ left (for tiebreaker purposes) | "
                    "Alice scores 104 TOTAL VP | "
                    "Bob has 30 M€ left (for tiebreaker purposes) | "
                    "Bob scores 90 TOTAL VP | "
                    "The end of the game: Alice wins!"
                ),
            },
        ],
    }


def test_eligibility_passes_for_supported_maps_and_localizations() -> None:
    reg = CardRegistry.from_names(FIXTURE_CARDS)
    for raw_map in ("Tharsis", "Элизий", "Hellas", "Vastitas Borealis (Nova)"):
        rec = _make_valid_record()
        rec["map"] = raw_map
        res = check_eligibility(rec, path="1001/game_500001_1001.json", card_registry=reg)
        assert res.passed is True, f"Expected pass for map {raw_map}, got {res}"
        assert res.reason is None


def test_eligibility_rejects_player_count_and_unsupported_maps() -> None:
    reg = CardRegistry.from_names(FIXTURE_CARDS)

    one_p = _make_valid_record()
    del one_p["players"]["1002"]
    res_1p = check_eligibility(one_p, path="1001/game_500001_1001.json", card_registry=reg)
    assert res_1p.passed is False
    assert res_1p.reason == "not_two_players"

    three_p = _make_valid_record()
    three_p["players"]["1003"] = {"name": "Carol"}
    res_3p = check_eligibility(three_p, path="1001/game_500001_1001.json", card_registry=reg)
    assert res_3p.passed is False
    assert res_3p.reason == "not_two_players"

    for bad_map in ("Amazonis Planitia", "Amazonis", "Random", "Terra Cimmeria", None):
        rec = _make_valid_record()
        rec["map"] = bad_map
        res = check_eligibility(rec, path="1001/game_500001_1001.json", card_registry=reg)
        assert res.passed is False
        assert res.reason == "unsupported_map"


def test_eligibility_rejects_bad_or_missing_setup_flags_and_other_modules() -> None:
    reg = CardRegistry.from_names(FIXTURE_CARDS)

    cases = [
        ("draft_on", False, "draft_not_on"),
        ("draft_on", None, "draft_not_on"),
        ("prelude_on", False, "prelude_not_on"),
        ("prelude_on", None, "prelude_not_on"),
        ("corporate_era_on", False, "corporate_era_not_on"),
        ("corporate_era_on", None, "corporate_era_not_on"),
        ("colonies_on", True, "colonies_not_off"),
        ("colonies_on", None, "colonies_not_off"),
    ]
    for field, val, expected_reason in cases:
        rec = _make_valid_record()
        rec[field] = val
        res = check_eligibility(rec, path="1001/game_500001_1001.json", card_registry=reg)
        assert res.passed is False
        assert res.reason == expected_reason

    rec_beg = _make_valid_record()
    rec_beg["beginners_corporations_on"] = True
    res_beg = check_eligibility(rec_beg, path="1001/game_500001_1001.json", card_registry=reg)
    assert res_beg.passed is False
    assert res_beg.reason == "other_module_present"

    rec_txt = _make_valid_record()
    rec_txt["moves"][0]["description"] = "Module: Venus Next | Map: Hellas | Alice is starting player"
    res_txt = check_eligibility(rec_txt, path="1001/game_500001_1001.json", card_registry=reg)
    assert res_txt.passed is False
    assert res_txt.reason == "other_module_present"


def test_eligibility_rejects_table_id_mismatch_excluded_version_and_unknown_card() -> None:
    reg = CardRegistry.from_names(FIXTURE_CARDS)

    rec_no_tid = _make_valid_record()
    rec_no_tid["table_id"] = None
    res_no_tid = check_eligibility(rec_no_tid, path="1001/game_500001_1001.json", card_registry=reg)
    assert res_no_tid.passed is False
    assert res_no_tid.reason == "missing_or_inconsistent_table_id"

    rec_bad_tid = _make_valid_record()
    rec_bad_tid["table_id"] = "999999"
    res_bad_tid = check_eligibility(rec_bad_tid, path="1001/game_500001_1001.json", card_registry=reg)
    assert res_bad_tid.passed is False
    assert res_bad_tid.reason == "missing_or_inconsistent_table_id"

    rec_ver = _make_valid_record()
    rec_ver["metadata"]["scraper_version"] = "v2025.08.06.1719"
    res_ver = check_eligibility(
        rec_ver,
        path="1001/game_500001_1001.json",
        card_registry=reg,
        excluded_versions={"v2025.08.06.1719"},
    )
    assert res_ver.passed is False
    assert res_ver.reason == "excluded_scraper_version"

    rec_card = copy.deepcopy(_make_valid_record())
    rec_card["players"]["1001"]["starting_hand"]["project_cards"][0] = "Mystery Card From Venus"
    res_card = check_eligibility(rec_card, path="1001/game_500001_1001.json", card_registry=reg)
    assert res_card.passed is False
    assert res_card.reason == "unresolvable_card_name"


def test_setup_gate_passes_and_canonicalizes_cards_and_zero_projects() -> None:
    reg = CardRegistry.from_names(FIXTURE_CARDS)
    rec = _make_valid_record()
    # Bob buys 0 project cards (1 corp + 2 preludes = 3 total)
    rec["moves"][1]["cards_kept"]["1002"] = ["Vitor", "Business Empire", "Dome Farming"]
    rec["moves"][2]["cards_kept"]["1002"] = ["Vitor", "Business Empire", "Dome Farming"]

    res = extract_verified_setup(rec, card_registry=reg)
    assert res.passed is True
    assert res.reason is None
    assert len(res.players) == 2

    by_pid = {p.player_id: p for p in res.players}
    alice = by_pid["1001"]
    assert alice.seat == 0
    assert alice.chosen_corporation == "CrediCor"
    assert alice.chosen_preludes == ("Allied Bank", "Biolab")
    assert alice.chosen_projects == ("Algae", "Birds", "Comet")
    assert "CEO's Favorite Project" in alice.offered_projects

    bob = by_pid["1002"]
    assert bob.seat == 1
    assert bob.chosen_corporation == "Vitor"
    assert bob.chosen_preludes == ("Business Empire", "Dome Farming")
    assert bob.chosen_projects == ()


def test_setup_gate_rejects_unoffered_cards_and_bad_counts() -> None:
    reg = CardRegistry.from_names(FIXTURE_CARDS)

    # Chosen card not in offered
    rec_unoffered = copy.deepcopy(_make_valid_record())
    rec_unoffered["moves"][2]["cards_kept"]["1001"].append("Research")
    res_unoffered = extract_verified_setup(rec_unoffered, card_registry=reg)
    assert res_unoffered.passed is False
    assert res_unoffered.reason == "setup_chosen_not_subset_of_offered"

    # Wrong prelude count (1 instead of 2)
    rec_bad_preludes = copy.deepcopy(_make_valid_record())
    rec_bad_preludes["moves"][2]["cards_kept"]["1001"] = ["CrediCor", "Allied Bank", "Algae"]
    res_bad_preludes = extract_verified_setup(rec_bad_preludes, card_registry=reg)
    assert res_bad_preludes.passed is False
    assert res_bad_preludes.reason == "invalid_setup_counts"

    # Wrong corp count (2 instead of 1)
    rec_two_corps = copy.deepcopy(_make_valid_record())
    rec_two_corps["moves"][2]["cards_kept"]["1001"] = ["CrediCor", "Ecoline", "Allied Bank", "Biolabs"]
    res_two_corps = extract_verified_setup(rec_two_corps, card_registry=reg)
    assert res_two_corps.passed is False
    assert res_two_corps.reason == "invalid_setup_counts"

    # Wrong offered counts (only 3 preludes offered)
    rec_bad_offer = copy.deepcopy(_make_valid_record())
    rec_bad_offer["players"]["1001"]["starting_hand"]["preludes"] = ["Allied Bank", "Biolabs", "Donation"]
    res_bad_offer = extract_verified_setup(rec_bad_offer, card_registry=reg)
    assert res_bad_offer.passed is False
    assert res_bad_offer.reason == "invalid_starting_hand"

    # Missing setup cards_kept
    rec_no_kept = copy.deepcopy(_make_valid_record())
    rec_no_kept["moves"] = [rec_no_kept["moves"][0], rec_no_kept["moves"][3], rec_no_kept["moves"][4]]
    res_no_kept = extract_verified_setup(rec_no_kept, card_registry=reg)
    assert res_no_kept.passed is False
    assert res_no_kept.reason == "missing_setup_choices"


def test_draft_gate_extracts_verified_picks_handles_undo_and_drops_unverified() -> None:
    reg = CardRegistry.from_names(FIXTURE_CARDS)
    rec = copy.deepcopy(_make_valid_record())
    rec["moves"] = [
        rec["moves"][0],
        rec["moves"][1],
        rec["moves"][2],
        # Gen 2 draft deal
        {
            "move_number": 21,
            "player_id": "1001",
            "player_name": "Alice",
            "action_type": "pass",
            "description": "Alice drafts 4 cards: Mass Converter, Towing a Comet, Wave Power, Worms | Bob drafts 4 cards: CEO's Favourite Project, Mangrove, Mine, Research",
            "card_drafted": None,
            "card_options": {
                "1001": ["Mass Converter", "Towing a Comet", "Wave Power", "Worms"],
                "1002": ["CEO's Favourite Project", "Mangrove", "Mine", "Research"],
            },
            "cards_kept": None,
            "reason": None,
        },
        # Alice picks Mass Converter, then undoes when re-dealt 4 cards in move 23
        {
            "move_number": 22,
            "player_id": "1001",
            "player_name": "Alice",
            "action_type": "draft",
            "description": "You draft Mass Converter",
            "card_drafted": "Mass Converter",
            "card_options": None,
            "cards_kept": None,
            "reason": "draft",
        },
        {
            "move_number": 23,
            "player_id": "1001",
            "player_name": "Alice",
            "action_type": "draft",
            "description": "Alice drafts 4 cards: Mass Converter, Towing a Comet, Wave Power, Worms",
            "card_drafted": None,
            "card_options": {
                "1001": ["Mass Converter", "Towing a Comet", "Wave Power", "Worms"],
            },
            "cards_kept": None,
            "reason": None,
        },
        # Alice's actual pick 1 + Bob's pick 1 (carrying swapped 3-card packets)
        {
            "move_number": 24,
            "player_id": "1001",
            "player_name": "Alice",
            "action_type": "draft",
            "description": "You draft Towing a Comet",
            "card_drafted": "Towing a Comet",
            "card_options": None,
            "cards_kept": None,
            "reason": "draft",
        },
        {
            "move_number": 25,
            "player_id": "1002",
            "player_name": "Bob",
            "action_type": "draft",
            "description": "Bob drafts CEO's Favourite Project",
            "card_drafted": "CEO's Favourite Project",
            "card_options": {
                "1002": ["Mass Converter", "Wave Power", "Worms"],
                "1001": ["Mangrove", "Mine", "Research"],
            },
            "cards_kept": None,
            "reason": "draft",
        },
        # Pick 2 (3-card packets)
        {
            "move_number": 26,
            "player_id": "1001",
            "player_name": "Alice",
            "action_type": "draft",
            "description": "You draft Mangrove",
            "card_drafted": "Mangrove",
            "card_options": None,
            "cards_kept": None,
            "reason": "draft",
        },
        # Unverified pick: Research is NOT in Bob's 3-card packet (it's in Alice's!) -> must be dropped
        {
            "move_number": 27,
            "player_id": "1002",
            "player_name": "Bob",
            "action_type": "draft",
            "description": "Bob drafts Research",
            "card_drafted": "Research",
            "card_options": {
                "1001": ["Wave Power", "Worms"],
                "1002": ["Mine", "Research"],
            },
            "cards_kept": None,
            "reason": "draft",
        },
        # Keep phase ends draft round
        {
            "move_number": 28,
            "player_id": "1001",
            "player_name": "Alice",
            "action_type": "other",
            "description": "Alice keeps Towing a Comet",
            "card_drafted": "Towing a Comet",
            "card_options": None,
            "cards_kept": {"1001": ["Towing a Comet"]},
            "reason": None,
        },
        # Non-draft draw move must not emit draft picks
        {
            "move_number": 35,
            "player_id": "1001",
            "player_name": "Alice",
            "action_type": "draw",
            "description": "Alice draws 4 cards: Trees, Shuttles, Worms, Mine",
            "card_drafted": None,
            "card_options": {"1001": ["Trees", "Shuttles", "Worms", "Mine"]},
            "cards_kept": None,
            "reason": None,
        },
        {
            "move_number": 36,
            "player_id": "1001",
            "player_name": "Alice",
            "action_type": "draft",
            "description": "Alice keeps Trees",
            "card_drafted": "Trees",
            "card_options": None,
            "cards_kept": {"1001": ["Trees"]},
            "reason": None,
        },
    ]

    res = extract_verified_drafts(rec, card_registry=reg)
    assert res.dropped_count == 2  # undone move 22 + unverified move 27
    assert [(p.move_number, p.player_id, p.chosen_card, p.packet_cards) for p in res.picks] == [
        (24, "1001", "Towing A Comet", ("Mass Converter", "Towing A Comet", "Wave Power", "Worms")),
        (25, "1002", "CEO's Favorite Project", ("CEO's Favorite Project", "Mangrove", "Mine", "Research")),
        (26, "1001", "Mangrove", ("Mangrove", "Mine", "Research")),
    ]


def test_outcome_gate_passes_for_valid_and_tiebreaker_games() -> None:
    rec = _make_valid_record()
    res = check_outcome(rec)
    assert res.passed is True
    assert res.winner_player_id == "1001"
    assert res.scores_by_player == {"1001": 104, "1002": 90}

    # Tied VP resolved by M€ left
    rec_tie = copy.deepcopy(_make_valid_record())
    rec_tie["players"]["1001"]["final_vp"] = 100
    rec_tie["players"]["1002"]["final_vp"] = 100
    rec_tie["winner"] = "Bob"
    rec_tie["moves"][-1]["description"] = (
        "Alice has 12 M€ left (for tiebreaker purposes) | Alice scores 100 TOTAL VP | "
        "Bob has 25 M€ left (for tiebreaker purposes) | Bob scores 100 TOTAL VP | "
        "The end of the game: Bob wins!"
    )
    res_tie = check_outcome(rec_tie)
    assert res_tie.passed is True
    assert res_tie.winner_player_id == "1002"


def test_conceded_keeps_choice_examples_but_has_no_outcome() -> None:
    reg = CardRegistry.from_names(FIXTURE_CARDS)
    rec = copy.deepcopy(_make_valid_record())
    rec["conceded"] = True
    rec["moves"][-1]["description"] = "Bob concedes the game. | The end of the game: Alice wins!"

    ev = evaluate_record(rec, path="1001/game_500001_1001.json", card_registry=reg)
    assert ev.eligibility.passed is True
    assert ev.setup.passed is True
    assert ev.outcome.passed is False
    assert ev.outcome.reason == "conceded"


def test_score_text_disagreement_and_winner_mismatch_drop_only_outcome_label() -> None:
    reg = CardRegistry.from_names(FIXTURE_CARDS)

    # Classic 20/20 vs 102/120 disagreement from audit #3
    rec_bad_score = copy.deepcopy(_make_valid_record())
    rec_bad_score["players"]["1001"]["final_vp"] = 20
    rec_bad_score["players"]["1002"]["final_vp"] = 20
    rec_bad_score["winner"] = "Bob"
    rec_bad_score["moves"][-1]["description"] = (
        "Alice scores 102 TOTAL VP | Bob scores 120 TOTAL VP | The end of the game: Bob wins!"
    )
    ev_bad = evaluate_record(rec_bad_score, path="1001/game_500001_1001.json", card_registry=reg)
    assert ev_bad.eligibility.passed is True
    assert ev_bad.setup.passed is True
    assert ev_bad.outcome.passed is False
    assert ev_bad.outcome.reason == "score_text_mismatch"

    # Winner disagrees with scores
    rec_bad_win = copy.deepcopy(_make_valid_record())
    rec_bad_win["winner"] = "Bob"
    res_win = check_outcome(rec_bad_win)
    assert res_win.passed is False
    assert res_win.reason == "winner_mismatch"


def test_setup_gate_ignores_post_setup_draw_move() -> None:
    reg = CardRegistry.from_names(FIXTURE_CARDS)
    rec = copy.deepcopy(_make_valid_record())
    # Insert a Point Luna / start-of-game draw move right after move 3 setup keep
    rec["moves"].insert(
        3,
        {
            "move_number": 4,
            "player_id": "1001",
            "player_name": "Alice",
            "action_type": "draw",
            "description": "Alice draws Research",
            "cards_kept": {"1001": ["Research"]},
        },
    )
    res = extract_verified_setup(rec, card_registry=reg)
    assert res.passed is True
    by_pid = {p.player_id: p for p in res.players}
    assert by_pid["1001"].chosen_corporation == "CrediCor"
    assert by_pid["1001"].chosen_projects == ("Algae", "Birds", "Comet")


def test_setup_gate_handles_play_card_setup_keep_and_does_not_overwrite() -> None:
    reg = CardRegistry.from_names(FIXTURE_CARDS)
    rec = copy.deepcopy(_make_valid_record())
    # Real BGA scraper pattern: P2 keeps with action_type="draft", P1 keeps with action_type="play_card",
    # and a subsequent prelude draw move has cards_kept with 2 cards for P1 plus cumulative 5 for P2.
    rec["moves"][1]["action_type"] = "draft"
    rec["moves"][2]["action_type"] = "play_card"
    rec["moves"][2]["cards_kept"] = {
        "1001": ["CrediCor", "Allied Bank", "Biolabs", "Algae", "Birds", "Comet"],
    }
    rec["moves"].insert(
        3,
        {
            "move_number": 4,
            "player_id": "1001",
            "player_name": "Alice",
            "action_type": "play_card",
            "description": "Alice keeps 2 cards: Algae, Farming | Bob keeps 5 cards: Vitor, Business Empire, Dome Farming, Fish, Grass",
            "cards_kept": {
                "1001": ["Algae", "Farming"],
                "1002": ["Vitor", "Business Empire", "Dome Farming", "Fish", "Grass"],
            },
        },
    )
    res = extract_verified_setup(rec, card_registry=reg)
    assert res.passed is True
    by_pid = {p.player_id: p for p in res.players}
    assert by_pid["1001"].chosen_corporation == "CrediCor"
    assert by_pid["1001"].chosen_projects == ("Algae", "Birds", "Comet")
    assert by_pid["1002"].chosen_corporation == "Vitor"

    # Also verify v2025.12.27 pattern where Move 3 setup keep is tagged action_type="draw"
    rec_draw = copy.deepcopy(_make_valid_record())
    rec_draw["moves"][2]["action_type"] = "draw"
    rec_draw["moves"][2]["description"] = "Alice draws 6 cards: CrediCor, Allied Bank, Biolabs, Algae, Birds, Comet"
    res_draw = extract_verified_setup(rec_draw, card_registry=reg)
    assert res_draw.passed is True
    assert {p.player_id: p.chosen_corporation for p in res_draw.players} == {"1001": "CrediCor", "1002": "Vitor"}


def test_setup_gate_handles_setup_undo_and_corp_switch() -> None:
    reg = CardRegistry.from_names(FIXTURE_CARDS)
    rec = copy.deepcopy(_make_valid_record())
    # Bob initially picks Helion (m2), undoes with "Bob drafts Helion" (m3), Alice keeps (m4),
    # then Bob confirms final keep with Vitor (m5) before gameplay starts (m6).
    rec["moves"] = [
        rec["moves"][0],
        {
            "move_number": 2,
            "player_id": "1002",
            "player_name": "Bob",
            "action_type": "draft",
            "description": "Bob keeps 5 cards: Helion, Business Empire, Dome Farming, Fish, Grass",
            "card_drafted": "Helion",
            "cards_kept": {"1002": ["Helion", "Business Empire", "Dome Farming", "Fish", "Grass"]},
        },
        {
            "move_number": 3,
            "player_id": "1002",
            "player_name": "Bob",
            "action_type": "draft",
            "description": "Bob drafts Helion",
            "card_drafted": "Helion",
            "cards_kept": None,
        },
        {
            "move_number": 4,
            "player_id": "1001",
            "player_name": "Alice",
            "action_type": "other",
            "description": "Alice keeps 6 cards: CrediCor, Allied Bank, Biolabs, Algae, Birds, Comet",
            "cards_kept": {"1001": ["CrediCor", "Allied Bank", "Biolabs", "Algae", "Birds", "Comet"]},
        },
        {
            "move_number": 5,
            "player_id": "1002",
            "player_name": "Bob",
            "action_type": "other",
            "description": "Bob keeps 6 cards: Vitor, Business Empire, Dome Farming, Fish, Grass, Heather",
            "cards_kept": {
                "1002": ["Vitor", "Business Empire", "Dome Farming", "Fish", "Grass", "Heather"],
                "1001": ["CrediCor", "Allied Bank", "Biolabs", "Algae", "Birds", "Comet"],
            },
        },
        {
            "move_number": 6,
            "player_id": "1001",
            "player_name": "Alice",
            "action_type": "play_card",
            "description": "Alice plays card Allied Bank",
            "card_played": "Allied Bank",
            "cards_kept": None,
        },
        *rec["moves"][3:],
    ]
    res = extract_verified_setup(rec, card_registry=reg)
    assert res.passed is True
    by_pid = {p.player_id: p for p in res.players}
    assert by_pid["1002"].chosen_corporation == "Vitor"
    assert by_pid["1002"].chosen_projects == ("Fish", "Grass", "Heather")


def test_setup_gate_ignores_3card_prelude_draw_after_setup_keep() -> None:
    reg = CardRegistry.from_names(FIXTURE_CARDS)
    rec = copy.deepcopy(_make_valid_record())
    # Real BGA Biolabs/Research Network pattern: both players keep in moves 2-3,
    # then Alice's prelude draws 3 project cards (action_type="play_card", no corporation) in move 4.
    rec["moves"].insert(
        3,
        {
            "move_number": 4,
            "player_id": "1001",
            "player_name": "Alice",
            "action_type": "play_card",
            "description": "Alice keeps 3 cards: Herbivores, Research, Space Elevator | Alice increases Plant Production by 1",
            "cards_kept": {"1001": ["Herbivores", "Research", "Space Elevator"]},
        },
    )
    res = extract_verified_setup(rec, card_registry=reg)
    assert res.passed is True
    by_pid = {p.player_id: p for p in res.players}
    assert by_pid["1001"].chosen_corporation == "CrediCor"
    assert by_pid["1001"].chosen_projects == ("Algae", "Birds", "Comet")


def _deal(move_number: int, alice: list[str], bob: list[str]) -> dict:
    return {
        "move_number": move_number,
        "player_id": "1001",
        "player_name": "Alice",
        "action_type": "pass",
        "description": f"Alice drafts 4 cards: {', '.join(alice)} | Bob drafts 4 cards: {', '.join(bob)}",
        "card_drafted": None,
        "card_options": {"1001": alice, "1002": bob},
        "cards_kept": None,
        "reason": None,
    }


def _pick(move_number: int, pid: str, card: str, swap: dict | None = None) -> dict:
    name = "Alice" if pid == "1001" else "Bob"
    return {
        "move_number": move_number,
        "player_id": pid,
        "player_name": name,
        "action_type": "draft",
        "description": f"{name} drafts {card}",
        "card_drafted": card,
        "card_options": swap,
        "cards_kept": None,
        "reason": "draft",
    }


def _keep(move_number: int) -> dict:
    return {
        "move_number": move_number,
        "player_id": "1001",
        "player_name": "Alice",
        "action_type": "other",
        "description": "Alice keeps Worms",
        "card_drafted": None,
        "card_options": None,
        "cards_kept": {"1001": ["Worms"]},
        "reason": None,
    }


def test_draft_generation_follows_round_and_inconsistent_rounds_are_quarantined() -> None:
    """#30: generation = draft_round_id + 1; duplicate keys and gaps never reach the dataset."""
    reg = CardRegistry.from_names(FIXTURE_CARDS)
    rec = copy.deepcopy(_make_valid_record())
    rec["moves"] = rec["moves"][:3] + [
        # Round 1 (generation 2): clean two-pick sequence for Alice, one pick for Bob.
        _deal(21, ["Mass Converter", "Towing A Comet", "Wave Power", "Worms"], ["Mangrove", "Mine", "Research", "Trees"]),
        _pick(22, "1001", "Mass Converter"),
        _pick(23, "1002", "Mangrove", {"1001": ["Mine", "Research", "Trees"], "1002": ["Towing A Comet", "Wave Power", "Worms"]}),
        _pick(24, "1001", "Mine"),
        _keep(25),
        # Round 2 (generation 3): Alice's pick event is logged twice from the same packet.
        _deal(31, ["Birds", "Comet", "Fish", "Grass"], ["Algae", "Heather", "Lichen", "Shuttles"]),
        _pick(32, "1001", "Birds"),
        _pick(33, "1001", "Birds"),
        _pick(34, "1002", "Algae", {"1001": ["Heather", "Lichen", "Shuttles"], "1002": ["Comet", "Fish", "Grass"]}),
        _keep(35),
        # Round 3 (generation 4): Alice's first pick fails verification, so her second has a gap.
        _deal(41, ["Dust Seals", "Farming", "Livestock", "Trees"], ["Decomposers", "Herbivores", "Mining Area", "Noctis City"]),
        _pick(42, "1001", "Research"),
        _pick(43, "1002", "Decomposers", {"1001": ["Herbivores", "Mining Area", "Noctis City"], "1002": ["Farming", "Livestock", "Trees"]}),
        _pick(44, "1001", "Herbivores"),
        _keep(45),
    ]

    res = extract_verified_drafts(rec, card_registry=reg)
    assert [(p.move_number, p.player_id, p.draft_round_id, p.pick_index, p.generation) for p in res.picks] == [
        (22, "1001", 1, 0, 2),
        (23, "1002", 1, 0, 2),
        (24, "1001", 1, 1, 2),
        (34, "1002", 2, 0, 3),
        (43, "1002", 3, 0, 4),
    ]
    assert sorted(res.dropped_reasons) == [
        "chosen_not_in_packet",
        "duplicate_pick_key",
        "duplicate_pick_key",
        "missing_earlier_pick",
    ]
    keys = [(p.player_id, p.draft_round_id, p.pick_index) for p in res.picks]
    assert len(keys) == len(set(keys))
