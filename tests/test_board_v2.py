"""Unit tests for the v2 board reconstruction and legal space menus (#61).

Coordinates, adjacency and menus below are worked by hand from `tm_bot/training/map_templates.json`
and the engine rules cited in `tm_bot/data/board.py` / `tm_bot/data/spacemenu.py`. The replay tests
mutate copies of the golden fixture (`tests/fixtures/golden_v2/`, see `tests/test_golden_v2.py`).
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tm_bot.agents.card_data import load_card_data
from tm_bot.data.board import (
    CITY,
    GREENERY,
    MINING_AREA,
    NO_OWNER,
    NO_TILE,
    OCEAN,
    SPECIAL_TILES,
    Board,
    apply_compact_board,
    geometry,
    locate,
    log_coords_to_xy,
    parse_placement,
    structured_placement,
)
from tm_bot.data.cards import CardRegistry
from tm_bot.data.gates import evaluate_record
from tm_bot.data.logstate import reconstruct_log_state
from tm_bot.data.spacemenu import space_menu
from tm_bot.models.encode import encode_observation
from tm_bot.models.vocab import CardVocab
from tm_bot.training.encoded_cache import BOARD_SPACES, expand_board
from tm_bot.training.warmstart_builder import build_state_observation

REPO = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden_v2"
A, B = "10000001", "10000002"
TABLE = "900000001"
MAPS = ("tharsis", "hellas", "elysium", "vastitas borealis nova")


# ---------------------------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------------------------


def test_structured_location_resolves_by_coordinates() -> None:
    # Row 5 is engine y = 4 (width 9, first column 1): column 6 -> x = 5 -> Tharsis (5, 4) = '34'.
    loc = "Tharsis Hex 6,5 (6,5)"
    assert log_coords_to_xy(6, 5) == (5, 4)
    assert locate(geometry("tharsis"), structured_placement("Ocean", loc), loc) == "34"


def test_named_volcano_resolves_by_name_and_by_coordinates() -> None:
    p = parse_placement("places Mining on Elysium Mons (2,3)")
    assert p is not None and (p.tile, p.where, p.coords, p.kind) == ("Mining", "Elysium Mons", (2, 3), "Mining")
    geo = geometry("elysium")
    assert locate(geo, p) == "14"
    # By coordinates alone: row 3 is y = 2 (width 7, first column 2), column 2 -> x = 2 -> (2, 2) = '14'.
    assert log_coords_to_xy(2, 3) == (2, 2)
    assert locate(geo, parse_placement("places Mining on Hex (2,3)")) == "14"


def test_tile_forest_into_hex() -> None:
    p = parse_placement("places tile Forest into Hex at (5,2)")
    assert p is not None and p.tile == "Forest" and p.kind == "greenery" and p.coords == (5, 2)
    # Row 2 is y = 1 (width 6, first column 3), column 5 -> x = 5 -> Tharsis (5, 1) = '10'.
    assert locate(geometry("tharsis"), p) == "10"


def test_hellas_double_space_location() -> None:
    p = parse_placement("places Ocean on  Hex 4,8 (4,8)")
    assert p is not None and p.kind == "ocean" and p.where == "Hex 4,8" and p.coords == (4, 8)
    # Row 8 is y = 7 (width 6, first column 3), column 4 -> x = 4 -> Hellas (4, 7) = '54'.
    assert locate(geometry("hellas"), p) == "54"


@pytest.mark.parametrize("map_name", MAPS)
def test_off_mars_cities_by_name(map_name: str) -> None:
    geo = geometry(map_name)
    assert locate(geo, parse_placement("places City on Ganymede Colony (0,3)")) == "01"
    assert locate(geo, parse_placement("places City on Phobos Space Haven")) == "02"


def test_noctis_city_by_name_without_coordinates() -> None:
    p = parse_placement("places tile City into Noctis City")
    assert p is not None and p.coords is None and p.kind == "city"
    assert locate(geometry("tharsis"), p) == "31"


def test_unparseable_and_unlocatable() -> None:
    assert parse_placement("pays 23 M€") is None
    assert locate(geometry("tharsis"), parse_placement("places Ocean on Hex (9,9)")) is None
    assert locate(geometry("tharsis"), parse_placement("places Ocean on Somewhere")) is None


@pytest.mark.parametrize("map_name", MAPS)
def test_log_coordinates_are_a_bijection_onto_mars(map_name: str) -> None:
    geo = geometry(map_name)
    seen = []
    for row in range(1, 10):
        width = 9 - abs(row - 5)
        first = math.ceil((9 - width) / 2) + 1
        for col in range(first, first + width):
            seen.append(geo.xy_to_id[log_coords_to_xy(col, row)])
    mars = [sid for sid in geo.space_ids if geo.space_type(sid) != "colony"]
    assert len(seen) == len(set(seen)) == len(mars) == 61
    assert sorted(seen) == mars


def test_engine_adjacency_by_hand() -> None:
    geo = geometry("tharsis")
    # Clockwise from the top left. (2, 3) above the middle row; (1, 4) on it; (4, 5) below it.
    assert geo.adjacent["22"] == ("14", "15", "23", "31", "30", "21")
    assert geo.adjacent["30"] == ("21", "22", "31", "39", "38", "29")
    assert geo.adjacent["41"] == ("32", "33", "42", "49", "48", "40")
    assert geo.adjacent["01"] == ()
    for sid, adj in geo.adjacent.items():
        assert all(sid in geo.adjacent[a] for a in adj)


# ---------------------------------------------------------------------------------------------
# Menus on synthetic boards
# ---------------------------------------------------------------------------------------------


def _land(geo, *exclude: str) -> list[str]:
    return [s for s in geo.space_ids if geo.space_type(s) == "land" and s != geo.noctis and s not in exclude]


def test_land_excludes_noctis_and_occupied() -> None:
    geo = geometry("tharsis")
    board = Board(geo)
    land = space_menu("land", board, 0, 100)
    assert "31" not in land and len(land) == len(_land(geo)) == 48
    board.place("41", GREENERY, 1)
    assert space_menu("land", board, 0, 100) == _land(geo, "41")


def test_city_adjacency() -> None:
    geo = geometry("tharsis")
    board = Board(geo)
    board.place("22", CITY, 1)
    # Neither seat may put a city on 22 or next to it (14, 15, 21, 23, 30; 31 is Noctis).
    expected = _land(geo, "22", "14", "15", "21", "23", "30")
    assert space_menu("city", board, 0, 100) == expected
    assert space_menu("city", board, 1, 100) == expected


def test_noctis_rule() -> None:
    # Tharsis reserves Noctis City's space: no prompt. Elsewhere Noctis City places like any city.
    assert space_menu("noctis", Board(geometry("tharsis")), 0, 100) == []
    hellas = Board(geometry("hellas"))
    assert space_menu("noctis", hellas, 0, 100) == space_menu("city", hellas, 0, 100) == _land(hellas.geo)


def test_greenery_own_adjacency_with_fallback() -> None:
    geo = geometry("tharsis")
    board = Board(geo)
    assert space_menu("greenery", board, 0, 100) == _land(geo)  # no own tile: any land space
    board.place("22", CITY, 0)
    assert space_menu("greenery", board, 0, 100) == ["14", "15", "21", "23", "30"]
    assert space_menu("greenery", board, 1, 100) == _land(geo, "22")  # the opponent's tile does not count
    board.place("32", OCEAN, 0)  # oceans are unowned: no adjacency for seat 0
    assert space_menu("greenery", board, 0, 100) == ["14", "15", "21", "23", "30"]


def test_ocean_spaces() -> None:
    geo = geometry("tharsis")
    board = Board(geo)
    oceans = ["04", "06", "07", "13", "28", "32", "33", "34", "43", "44", "45", "63"]
    assert space_menu("ocean", board, 0, 0) == oceans
    board.place("04", OCEAN, 0)
    assert board.owner("04") == NO_OWNER
    assert space_menu("ocean", board, 1, 0) == oceans[1:]


@pytest.mark.parametrize("map_name,sid,cost", [("hellas", "61", 6), ("vastitas borealis nova", "33", 4)])
def test_bonus_space_costs(map_name: str, sid: str, cost: int) -> None:
    board = Board(geometry(map_name))
    for rule in ("land", "city", "greenery"):
        assert sid not in space_menu(rule, board, 0, cost - 1)
        assert sid in space_menu(rule, board, 0, cost)
    assert len(space_menu("land", board, 0, cost - 1)) == len(space_menu("land", board, 0, cost)) - 1


def test_isolated() -> None:
    geo = geometry("tharsis")
    board = Board(geo)
    board.place("22", CITY, 1)
    board.place("41", OCEAN, 0)  # (illegal for an ocean, but any tile blocks its neighbours)
    near = {"22", "14", "15", "21", "23", "30", "41", "32", "33", "42", "49", "48", "40"}
    assert space_menu("isolated", board, 0, 100) == _land(geo, *near)


def test_volcanic() -> None:
    board = Board(geometry("tharsis"))
    assert space_menu("volcanic", board, 0, 100) == ["09", "14", "21", "29"]
    board.place("14", CITY, 1)
    assert space_menu("volcanic", board, 0, 100) == ["09", "21", "29"]
    assert space_menu("volcanic", Board(geometry("elysium")), 0, 100) == ["08", "14", "20", "37"]
    hellas = Board(geometry("hellas"))  # no volcanic spaces: any land space
    assert space_menu("volcanic", hellas, 0, 100) == _land(hellas.geo)
    assert space_menu("lava_tube_settlement", hellas, 0, 100) == space_menu("city", hellas, 0, 100)
    # Lava Tube Settlement on a volcanic map: a volcanic space (no city-adjacency filter in the engine).
    assert space_menu("lava_tube_settlement", board, 0, 100) == ["09", "21", "29"]


def test_mining_rights_and_area() -> None:
    board = Board(geometry("tharsis"))
    # Land spaces with a steel (1) or titanium (0) bonus.
    steel_ti = ["03", "09", "20", "21", "53", "58", "59", "60"]
    assert space_menu("mining_rights", board, 0, 100) == steel_ti
    assert space_menu("mining_area", board, 0, 100) == []
    board.place("10", CITY, 0)  # neighbours 04, 05, 11, 17, 16, 09
    assert space_menu("mining_area", board, 0, 100) == ["09"]
    assert space_menu("mining_area", board, 1, 100) == []
    board.place("09", MINING_AREA, 0)
    assert space_menu("mining_rights", board, 1, 100) == [s for s in steel_ti if s != "09"]


def test_urbanized_industrial_ecological() -> None:
    geo = geometry("tharsis")
    board = Board(geo)
    board.place("22", CITY, 0)
    board.place("24", CITY, 1)  # 23 is the only space next to both
    assert space_menu("urbanized_area", board, 0, 100) == ["23"]
    assert space_menu("industrial_center", board, 1, 100) == sorted(
        {"14", "15", "21", "23", "30", "16", "17", "25"})
    assert space_menu("ecological_zone", board, 0, 100) == []
    board.place("48", GREENERY, 1)  # neighbours of (4, 6): 40, 41, 49, 55, 54, 47
    assert space_menu("ecological_zone", board, 0, 100) == ["40", "41", "47", "49", "54", "55"]


def test_land_claim_marker_closes_the_space_to_the_opponent_only() -> None:
    board = Board(geometry("tharsis"))
    board.tiles["10"] = (NO_TILE, 1)  # seat 1's Land Claim marker
    assert "10" not in space_menu("land", board, 0, 100)
    assert "10" in space_menu("land", board, 1, 100)
    assert "10" not in space_menu("city", board, 0, 100) and "10" in space_menu("city", board, 1, 100)


# ---------------------------------------------------------------------------------------------
# Replay and forward quarantine on the golden fixture
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def registry() -> CardRegistry:
    return CardRegistry.from_json(REPO / "data" / "card_registry.json")


def _load(pid: str) -> dict[str, Any]:
    return json.loads((FIXTURE / pid / f"game_{TABLE}_{pid}.json").read_text(encoding="utf-8"))


def _reconstruct(data: dict[str, Any], registry: CardRegistry):
    ev = evaluate_record(data, path=f"{data['player_perspective']}/game_{TABLE}_{data['player_perspective']}.json",
                         card_registry=registry)
    assert ev.setup.passed
    return reconstruct_log_state(data, setup=ev.setup, drafts=ev.drafts, card_registry=registry)


def _move(data: dict[str, Any], n: int) -> dict[str, Any]:
    (m,) = [m for m in data["moves"] if m["move_number"] == n]
    return m


def _boarded(res) -> list:
    return [d for d in res.decisions if d.snapshot.get("board_tiles")]


def test_untouched_fixture_keeps_its_board(registry: CardRegistry) -> None:
    res = _reconstruct(_load(A), registry)
    assert res.board_lost is None
    assert any(d.move_number > 47 for d in _boarded(res))
    assert {(d.player_id, d.move_number) for d in res.decisions if d.kind == "space"} == {(A, 47), (B, 27)}


def test_missing_placement_quarantines_at_the_ocean_check(registry: CardRegistry) -> None:
    data = copy.deepcopy(_load(A))
    m47 = _move(data, 47)
    m47["description"] = " | ".join(s for s in m47["description"].split(" | ") if " places " not in s)
    m47["tile_placed"] = m47["tile_location"] = None
    res = _reconstruct(data, registry)
    # The snapshot posts the ocean at 47 but the board has none.
    assert res.board_lost == (47, "check:oceans")
    assert res.counts_by_player[A]["board_quarantine:check:oceans"] == 1
    assert res.counts_by_player[B]["board_quarantine:check:oceans"] == 1
    assert res.counts_by_player[A]["board:quarantined"] == 1
    # Rows up to and including 47 keep the board (the check runs after the row's decisions); later
    # rows have none, and no space rows follow.
    assert any(d.move_number < 47 for d in _boarded(res))
    assert all(d.move_number <= 47 for d in _boarded(res))
    later = [d for d in res.decisions if d.move_number > 47 and d.generation >= 2 and d.kind != "deal_boundary"]
    assert later and all(d.snapshot.get("board_tiles") == [] for d in later)
    assert [d.move_number for d in res.decisions if d.kind == "space"] == [27]


def test_snapshot_greenery_above_the_board_quarantines(registry: CardRegistry) -> None:
    data = copy.deepcopy(_load(A))
    _move(data, 28)["game_state"]["player_vp"][A]["total_details"]["greeneries"] = 1  # Alice has none
    res = _reconstruct(data, registry)
    assert res.board_lost == (28, "check:greeneries")
    assert [d.move_number for d in res.decisions if d.kind == "space"] == [27]


def test_snapshot_greenery_lagging_the_board_does_not_quarantine(registry: CardRegistry) -> None:
    data = copy.deepcopy(_load(A))
    for m in data["moves"]:
        if 27 <= m["move_number"] <= 40:
            m["game_state"]["player_vp"][B]["total_details"]["greeneries"] = 0  # Bram's VP details lag
    res = _reconstruct(data, registry)
    assert res.board_lost is None
    assert res.counts_by_player[A]["board:valid_at_end"] == 1


def test_unlocatable_placement_quarantines(registry: CardRegistry) -> None:
    data = copy.deepcopy(_load(A))
    m47 = _move(data, 47)
    m47["description"] = m47["description"].replace("Hex 4,1 (4,1)", "Hex 9,9 (9,9)")
    m47["tile_location"] = "Tharsis Hex 9,9 (9,9)"
    res = _reconstruct(data, registry)
    assert res.board_lost == (47, "unlocatable_placement")
    assert res.counts_by_player[A]["board_quarantine:unlocatable_placement"] == 1
    assert [d.move_number for d in res.decisions if d.kind == "space"] == [27]


# ---------------------------------------------------------------------------------------------
# Compact codes round trip
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def space_row(registry: CardRegistry) -> dict[str, Any]:
    res = _reconstruct(_load(A), registry)
    (d,) = [d for d in res.decisions if d.kind == "space" and d.player_id == A]
    return {"seat": d.seat, "generation": d.generation, **d.payload, **d.snapshot}


@pytest.mark.parametrize("map_name,display", [("tharsis", "Tharsis"), ("hellas", "Hellas"), ("elysium", "Elysium"),
                                              ("vastitas borealis nova", "Vastitas Borealis")])
@pytest.mark.parametrize("actor", [0, 1])
def test_compact_board_round_trip(space_row: dict[str, Any], map_name: str, display: str, actor: int) -> None:
    geo = geometry(map_name)
    board = Board(geo)
    land = _land(geo)
    oceans = [s for s in geo.space_ids if geo.space_type(s) == "ocean"]
    board.place("01", CITY, 1 - actor)  # off-Mars city
    board.place(land[0], CITY, actor)
    board.place(land[5], GREENERY, 1 - actor)
    board.place(land[9], SPECIAL_TILES["Capital"], actor)
    board.place(land[12], SPECIAL_TILES["Nuclear Zone"], 1 - actor)
    board.place(land[20], MINING_AREA, actor)
    board.place(oceans[0], OCEAN, actor)
    tiles, owners = board.compact(actor)
    assert len(tiles) == len(owners) == BOARD_SPACES

    vocab = CardVocab.from_card_data(load_card_data())
    row = {**space_row, "map": display, "seat": actor}
    empty = encode_observation(build_state_observation({**row, "board_tiles": [], "board_owners": []}), vocab)
    obs = build_state_observation({**row, "board_tiles": tiles, "board_owners": owners})
    by_id = {str(sp["id"]): sp for sp in obs["public"]["spaces"]}
    assert (by_id[land[0]]["tileType"], by_id[land[0]]["owner"]) == (CITY, actor)
    assert (by_id[land[5]]["tileType"], by_id[land[5]]["owner"]) == (GREENERY, 1 - actor)
    assert (by_id[oceans[0]]["tileType"], by_id[oceans[0]]["owner"]) == (OCEAN, None)
    assert (by_id[land[1]]["tileType"], by_id[land[1]]["owner"]) == (None, None)
    full = encode_observation(obs, vocab)
    expanded = expand_board(empty.space_feats[None], np.asarray([tiles], dtype=np.int8),
                            np.asarray([owners], dtype=np.int8), np.ones(1, dtype=bool))[0]
    assert np.array_equal(full.space_feats, expanded)
    assert not np.array_equal(full.space_feats, empty.space_feats)


def test_apply_compact_board_rejects_a_wrong_map_size() -> None:
    spaces = [{"id": f"{i:02d}", "tileType": None, "owner": None} for i in range(1, 10)]
    with pytest.raises(ValueError):
        apply_compact_board(spaces, [NO_TILE] * 63, [NO_OWNER] * 63, 0)
