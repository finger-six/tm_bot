"""Each v2 gate quarantines its failure case and is counted in the yield report.

Every case is a small mutation of the synthetic golden table (`tests/fixtures/golden_v2/`), seen from
Alice's perspective (seat 0). Move numbers refer to the fixture's move list.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from tm_bot.data.cards import CardRegistry
from tm_bot.data.extract import run_extraction
from tm_bot.data.gates import DEFAULT_EXCLUDED_SCRAPER_VERSIONS, check_index_row, evaluate_record
from tm_bot.data.inventory import run_inventory
from tm_bot.data.logstate import LogStateResult, check_v2_file, reconstruct_log_state

REPO = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).resolve().parent / "fixtures" / "golden_v2" / "10000001" / "game_900000001_10000001.json"
A = "10000001"
MC = "M€"


@pytest.fixture(scope="module")
def registry() -> CardRegistry:
    return CardRegistry.from_json(REPO / "data" / "card_registry.json")


def _golden() -> dict[str, Any]:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def _row(data: dict[str, Any], move: int) -> dict[str, Any]:
    row = data["moves"][move - 1]
    assert row["move_number"] == move
    return row


def _set_shown_mc(data: dict[str, Any], moves: range, delta: int) -> None:
    for mv in moves:
        _row(data, mv)["game_state"]["player_trackers"][A][MC] += delta


# ---- mutations: one failure case per gate --------------------------------------------------------


def _timing(d):
    _row(d, 10)["game_state"]["move_number"] = 999


def _empty_trackers(d):
    for mv in range(5, len(d["moves"]) + 1):
        _row(d, mv)["game_state"]["player_trackers"] = {}


def _unstructured_draft(d):
    _row(d, 18)["card_drafted"] = None  # "You draft Trees" with no structured pick


def _payment_mismatch(d):
    _row(d, 22)["cards_kept"][A] = ["Media Group", "Trees", "Bushes"]  # 3 cards kept, 6 M€ paid


def _kept_not_in_packet(d):
    _row(d, 22)["cards_kept"][A] = ["Media Group", "Capital"]


def _incomplete_packet(d):
    _row(d, 21)["card_options"] = None  # the final 1-card pass packet is never shown


def _missing_payment_text(d):
    _row(d, 22)["description"] = "Alice Testa keeps 2 cards: Media Group, Trees"


def _unaffordable(d):
    _set_shown_mc(d, range(14, 25), -61)  # gen-2 start M€ = -21 + 4 + 21 = 4 < 6


def _income_fail(d):
    # A wrong deal-row M€ production: both posted gen-2 M€ values then disagree with the identity.
    _row(d, 15)["game_state"]["player_trackers"][A][f"{MC} Production"] = 9


def _count_only_discard(d):
    _row(d, 36)["description"] = "Alice Testa discards 1 card/s (Sell patents)"


def _unnamed_draw(d):
    _row(d, 36)["description"] = "Alice Testa draws 2 cards"


def _play_not_in_hand(d):
    row = _row(d, 35)
    row["description"] = row["description"].replace("plays card Wave Power", "plays card Windmills")
    row["card_played"] = "Windmills"


# ---- action-phase gates (#44) --------------------------------------------------------------------


def _resignation(d):
    # A resignation row never becomes a Pass (here typed `pass`, the worst case).
    _row(d, 52).update(action_type="pass", description="Alice Testa concedes the game.")


def _attribution_mismatch(d):
    _row(d, 56)["description"] = "Alice Testa pays 3 M€ | Bram Fixture plays card Search For Life"


def _unsupported_standard_project(d):
    _row(d, 50)["description"] = "Alice Testa plays standard project Build Colony | Alice Testa pays 11 M€"


def _card_action_not_in_tableau(d):
    _row(d, 57)["description"] = "Alice Testa activates Birds | Alice Testa pays 1 M€"


def _card_action_used(d):
    _row(d, 59)["description"] = "Alice Testa activates Search For Life"  # second use in generation 3


def _standard_project_unaffordable(d):
    _row(d, 63)["description"] = "Alice Testa plays standard project City | Alice Testa pays 25 M€"  # 13 M€


def _end_turn_on_first_action(d):
    _row(d, 56).update(action_type="other", description="Alice Testa skips second action", card_played=None)


CASES: dict[str, tuple[Callable[[dict[str, Any]], None], dict[str, int]]] = {
    "timing": (_timing, {"table:timing_mismatch": 1}),
    "empty_trackers": (_empty_trackers, {"table:degenerate_empty_trackers": 1}),
    "unstructured_draft": (_unstructured_draft, {"table:degenerate_unstructured_draft": 1}),
    "payment_mismatch": (_payment_mismatch, {"research_buy:payment_mismatch": 1}),
    "kept_not_in_packet": (_kept_not_in_packet, {"research_buy:kept_not_in_packet": 1}),
    "incomplete_packet": (_incomplete_packet, {"research_buy:incomplete_packet": 1}),
    "missing_payment_text": (_missing_payment_text, {"research_buy:missing_payment_text": 1}),
    "unaffordable": (_unaffordable, {"research_buy:unaffordable": 1}),
    "income_fail": (_income_fail, {"income_identity:fail": 1, "draft_pick:income_identity_failed": 3,
                                   "deal_boundary:income_identity_failed": 1, "action:income_identity_failed": 2,
                                   "research_buy:income_identity_failed": 1}),
    "count_only_discard": (_count_only_discard, {"hand_lost:count_only_discard": 1, "draft_pick:hand_lost": 3,
                                                 "research_buy:hand_lost": 1, "action:hand_lost": 2}),
    "unnamed_draw": (_unnamed_draw, {"hand_lost:unnamed_draw": 1, "draft_pick:hand_lost": 3,
                                     "research_buy:hand_lost": 1, "action:hand_lost": 2}),
    "play_not_in_hand": (_play_not_in_hand, {"hand_lost:play_not_in_hand": 1, "research_buy:hand_lost": 1}),
    "resignation": (_resignation, {"gate5:resignation": 1}),
    "attribution_mismatch": (_attribution_mismatch, {"gate5:attribution_mismatch": 1}),
    "unsupported_standard_project": (_unsupported_standard_project, {"gate5:unsupported_standard_project": 1}),
    "card_action_not_in_tableau": (_card_action_not_in_tableau, {"card_action:menu_miss:not_in_tableau": 1}),
    "card_action_used": (_card_action_used, {"card_action:menu_miss:used": 1}),
    "standard_project_unaffordable": (_standard_project_unaffordable,
                                      {"standard_project:menu_miss:unaffordable": 1}),
    "end_turn_on_first_action": (_end_turn_on_first_action, {"branch:menu_miss:end_turn_first_action": 1}),
}


def _reconstruct(data: dict[str, Any], registry: CardRegistry) -> LogStateResult:
    ev = evaluate_record(data, path=f"{A}/game_{data['table_id']}_{A}.json", card_registry=registry)
    assert ev.setup.passed
    return reconstruct_log_state(data, setup=ev.setup, drafts=ev.drafts, card_registry=registry)


def test_golden_table_passes_every_v2_gate(registry: CardRegistry) -> None:
    res = _reconstruct(_golden(), registry)
    assert res.table_reason is None
    # #43 gates all pass. The #44 counts are the golden's hand-computed action-phase outcomes (see
    # tests/test_golden_v2.py): stripped rows, nested prompts never emitted, 1-candidate drops, the board
    # kept valid to the end (#61; Alice's space row at move 47 is kept, so it adds no count), and the
    # three synthetic plays whose chosen candidate fails the static checks (quarantined).
    assert dict(res.counts_by_player[A]) == {
        "income_identity:pass": 2,
        "gate5:cancel": 1, "gate5:other_row": 1, "gate5:bundled_pass_split": 1,
        "excluded:payment": 10, "excluded:prelude_play": 2, "board:valid_at_end": 1,
        "branch:menu_miss:claim_milestone": 1, "branch:single_candidate": 1,
        "project_card:menu_miss:requirement": 2, "project_card:single_candidate": 1,
        "card_action:single_candidate": 2, "standard_project:single_candidate": 1,
    }


@pytest.mark.parametrize("case", sorted(CASES))
def test_gate_quarantines_its_failure_case(case: str, registry: CardRegistry) -> None:
    mutate, expected = CASES[case]
    data = _golden()
    mutate(data)
    res = _reconstruct(data, registry)
    table_keys = {k: v for k, v in expected.items() if k.startswith("table:")}
    if table_keys:
        assert f"table:{res.table_reason}" in table_keys
        assert res.decisions == []
        return
    mine = res.counts_by_player[A]
    for key, cnt in expected.items():
        assert mine[key] == cnt, (case, key, dict(mine))
    dropped = {(d.kind, d.drop_reason) for d in res.decisions if d.player_id == A and d.drop_reason is not None}
    for key in expected:
        if not key.startswith(("hand_lost", "income_identity", "gate5")):
            assert tuple(key.split(":", 1)) in dropped, (case, key, dropped)


def _decision(res: LogStateResult, kind: str, move: int):
    return next(d for d in res.decisions if d.player_id == A and d.kind == kind and d.move_number == move)


def test_named_discard_keeps_the_hand_exact(registry: CardRegistry) -> None:
    golden_hand = _decision(_reconstruct(_golden(), registry), "research_buy", 44).snapshot["me_hand"]
    card = golden_hand[0]
    data = _golden()
    _row(data, 36)["description"] = f"Alice Testa discards {card}"
    res = _reconstruct(data, registry)
    # The discard itself keeps the identity; the hand is lost only when the discarded card is played later.
    assert res.hand_lost_at[A] > 44
    assert dict(res.counts_by_player[A]).get("hand_lost:play_not_in_hand") == 1
    assert not any(k.startswith("hand_lost") and k != "hand_lost:play_not_in_hand" for k in res.counts_by_player[A])
    snap = _decision(res, "research_buy", 44).snapshot
    assert snap["me_hand_exact"] is True
    assert snap["me_hand"] == [c for c in golden_hand if c != card]
    assert card in snap["me_discarded_seen"]


def test_look_and_buy_records_the_unkept_cards_as_seen(registry: CardRegistry) -> None:
    data = _golden()
    offer = _row(data, 34)  # Alice is shown two cards and keeps one of them two rows later
    offer.update(action_type="draw", description="Alice Testa draws 2 card/s", card_options={A: ["Ants", "Birds"]})
    keep = _row(data, 36)
    keep.update(description="Alice Testa keeps 1 card/s", cards_kept={A: ["Ants"]})
    res = _reconstruct(data, registry)
    # Ants stays in the hand, so the gen-3 Sell patents of 1 card (count-only, move 62) loses it.
    assert [k for k in res.counts_by_player[A] if k.startswith("hand_lost")] == ["hand_lost:count_only_discard"]
    assert res.hand_lost_at[A] == 62
    snap = _decision(res, "research_buy", 44).snapshot
    assert "Ants" in snap["me_hand"] and "Ants" not in snap["me_discarded_seen"]
    assert "Birds" in snap["me_discarded_seen"] and "Birds" not in snap["me_hand"]


def test_forward_quarantine_keeps_earlier_rows(registry: CardRegistry) -> None:
    data = _golden()
    _count_only_discard(data)  # hand lost at move 36 (end of generation 2)
    res = _reconstruct(data, registry)
    assert res.hand_lost_at == {A: 36, "10000002": 45}
    v1_kinds = ("draft_pick", "research_buy", "deal_boundary", "action")
    kept = sorted(d.move_number for d in res.decisions
                  if d.player_id == A and d.drop_reason is None and d.kind in v1_kinds)
    # Every gen-2 row survives; gen 3 keeps only the deal-boundary value point, with no hand identity.
    assert kept == [15, 16, 18, 20, 22, 30, 34, 37]
    boundary = next(d for d in res.decisions if d.player_id == A and d.move_number == 37 and d.kind == "deal_boundary")
    assert boundary.snapshot["me_hand_exact"] is False and boundary.snapshot["me_hand"] == []


def test_action_forward_quarantine_drops_later_rows_and_keeps_earlier_ones(registry: CardRegistry) -> None:
    golden = {(d.kind, d.move_number): d.drop_reason
              for d in _reconstruct(_golden(), registry).decisions if d.player_id == A}
    data = _golden()
    _count_only_discard(data)  # Alice's hand is lost at move 36
    res = _reconstruct(data, registry)
    for d in res.decisions:
        if d.player_id != A or d.kind not in ("branch", "project_card", "card_action", "standard_project"):
            continue
        if d.move_number <= 36:
            assert d.drop_reason == golden[(d.kind, d.move_number)], (d.kind, d.move_number)  # unchanged
        elif d.kind == "standard_project":
            # Its menu reads only M€, so it does not need the exact hand.
            assert d.drop_reason == golden[(d.kind, d.move_number)], d.move_number
        else:
            assert d.drop_reason == "hand_lost", (d.kind, d.move_number)
    counts = res.counts_by_player[A]
    # Gen 3 roots of Alice after the loss: 37 (pass), 46, 50, 52, 56, 57, 59, 60, 61, 63, 65.
    assert counts["branch:hand_lost"] == 11
    assert counts["project_card:hand_lost"] == 3 and counts["card_action:hand_lost"] == 2
    kept_before = sorted(d.move_number for d in res.decisions
                         if d.player_id == A and d.kind == "branch" and d.drop_reason is None)
    assert kept_before == [25, 32, 34, 35]


def test_resignation_is_stripped_and_never_a_pass(registry: CardRegistry) -> None:
    data = _golden()
    _resignation(data)
    res = _reconstruct(data, registry)
    assert not [d for d in res.decisions if d.player_id == A and d.move_number == 52]
    assert res.counts_by_player[A]["gate5:resignation"] == 1
    passes = sorted(d.move_number for d in res.decisions if d.kind == "branch" and d.payload["chosen"] == "pass")
    assert passes == [33, 37, 58, 65]


def test_version_gate_excludes_stripped_payment_scraper() -> None:
    assert "v2025.08.15.1706" in DEFAULT_EXCLUDED_SCRAPER_VERSIONS
    row = {"status": "ok", "player_count": 2, "map": "Tharsis", "draft_on": 1, "prelude_on": 1,
           "corporate_era_on": 1, "colonies_on": 0, "table_id": "900000001",
           "path": f"{A}/game_900000001_{A}.json", "scraper_version": "v2025.08.15.1706"}
    gate = check_index_row(row)
    assert (gate.passed, gate.reason) == (False, "excluded_scraper_version")


def test_check_v2_file_flags_missing_snapshots() -> None:
    data = _golden()
    for row in data["moves"]:
        del row["game_state"]
    assert check_v2_file(data["moves"]) == "missing_game_state"


def test_every_gate_is_counted_in_the_yield_report(tmp_path: Path) -> None:
    archive = tmp_path / "archive" / A
    archive.mkdir(parents=True)
    expected_total: dict[str, int] = {}
    for i, case in enumerate(sorted(CASES)):
        data = _golden()
        CASES[case][0](data)
        tid = f"9100000{i:02d}"
        data["table_id"] = data["replay_id"] = tid
        (archive / f"game_{tid}_{A}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        for key, cnt in CASES[case][1].items():
            expected_total[key] = expected_total.get(key, 0) + cnt
    version = _golden()
    version["table_id"] = version["replay_id"] = "910000099"
    version["metadata"]["scraper_version"] = "v2025.08.15.1706"
    (archive / f"game_910000099_{A}.json").write_text(json.dumps(version, ensure_ascii=False), encoding="utf-8")

    index = tmp_path / "index.sqlite"
    run_inventory(tmp_path / "archive", index, progress_interval=0)
    out = tmp_path / "out"
    report = run_extraction(tmp_path / "archive", index, out,
                            card_registry_path=REPO / "data" / "card_registry.json", progress_interval=0)
    v2 = report["v2_quarantine"]
    for key, cnt in expected_total.items():
        if key.startswith("table:"):
            assert v2[key] == 2 * cnt, key  # one count per player of the table
        else:
            assert v2[key] >= cnt, key
    assert report["tables"]["quarantined_by_reason"]["excluded_scraper_version"] == 1
    md = (out / "yield_report.md").read_text(encoding="utf-8")
    for key in expected_total:
        assert f"`{key}`" in md


def test_plant_conversion_needs_the_greenery_placement(registry: CardRegistry) -> None:
    # An attacked seat's plant loss is logged as a bare `NAME pays N Plant` too: not a root.
    data = _golden()
    _row(data, 36)["description"] = "Alice Testa pays 3 Plant"
    res = _reconstruct(data, registry)
    assert not [d for d in res.decisions if d.player_id == A and d.move_number == 36]

    # A conversion: 8 plants posted on move 49, paid on move 50, greenery placed on move 51.
    data = _golden()
    _row(data, 49)["game_state"]["player_trackers"][A]["Plant"] = 8
    _row(data, 50).update(action_type="other", description="Alice Testa pays 8 Plant", reason=None)
    _row(data, 51)["description"] = ("Alice Testa places Greenery on Tharsis Hex 5,5 (5,5) | "
                                     "Alice Testa increases Oxygen Level by 1 step/s to a value of 2 | Alice Testa gains 1 TR")
    res = _reconstruct(data, registry)
    branch = _decision(res, "branch", 50)
    assert branch.drop_reason is None and branch.payload["chosen"] == "convert_plants"
    assert branch.payload["menu"][:2] == ["convert_plants", "card_action"]
