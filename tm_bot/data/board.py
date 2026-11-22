"""Mars board geometry and per-table board state rebuilt from human-log placements (v2 warm start, #61).

The supported base + Prelude maps have fixed layouts (`tm_bot/training/map_templates.json`, the
adapter's empty board). Space ids, `x`/`y`, space types and bonuses come from the template; the
adjacency rule, the volcanic spaces and the Noctis reservation follow the pinned engine
(`tools/simbench/engine/src/server/boards/Board.ts` `computeAdjacentSpaces`, the `*Board.ts`
builders, `TharsisBoard` `NOCTIS_CITY`).

Log coordinates. A placement names its space as `(<col>,<row>)`: `row` is the engine `y + 1` and
`col` counts the hex row's spaces from a left edge that stays put as the rows widen, so a row of
width `w` starts at column `ceil((9 - w) / 2) + 1`. Rows above the middle start at engine
`x = 4 - y`, the middle row and rows below at `x = y - 4`. The off-Mars cities and the volcanoes are
also logged by name, sometimes without coordinates (`places tile City into Noctis City`).

Board codes. A space's tile is the engine `TileType` number (`NO_TILE` when empty) and its owner a
seat (`NO_OWNER` for oceans and empty spaces); the live observation carries the same numbers.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

_TEMPLATES_PATH = Path(__file__).resolve().parents[1] / "training" / "map_templates.json"

NO_TILE = -1
NO_OWNER = -1
# Engine `TileType` numbers of the base + Prelude tiles.
GREENERY, OCEAN, CITY, CAPITAL = 0, 1, 2, 3
MINING_AREA, MINING_RIGHTS = 8, 9
SPECIAL_TILES: dict[str, int] = {
    "Capital": CAPITAL, "Commercial District": 4, "Ecological Zone": 5, "Industrial Center": 6,
    "Lava Flows": 7, "Mohole Area": 10, "Natural Preserve": 11, "Nuclear Zone": 12, "Restricted Area": 13,
}
CITY_TILES = frozenset({CITY, CAPITAL})  # engine `CITY_TILES` restricted to base + Prelude
# Log tile name -> placed tile kind: "city", "greenery", "ocean", or the special tile's own name.
# `Mining` is either Mining Area or Mining Rights; the source card tells which.
TILE_KINDS: dict[str, str] = {
    "City": "city", "Greenery": "greenery", "Forest": "greenery", "Ocean": "ocean", "Mining": "Mining",
    **{name: name for name in SPECIAL_TILES},
}

# Engine `BoardBuilder.volcanic(...)` spaces (Hellas has none).
VOLCANIC_SPACES: dict[str, frozenset[str]] = {
    "tharsis": frozenset({"09", "14", "21", "29"}),
    "elysium": frozenset({"08", "14", "20", "37"}),
    "hellas": frozenset(),
    "vastitas borealis nova": frozenset({"05", "13", "21", "46"}),
}
# Log map name (`normalize_map`, lower-cased) -> template key; the archive's Vastitas Borealis is the
# engine's Vastitas Borealis Nova (as in `warmstart_builder.normalize_map_name`).
BOARD_MAPS: dict[str, str] = {
    "tharsis": "tharsis", "elysium": "elysium", "hellas": "hellas",
    "vastitas borealis": "vastitas borealis nova", "vastitas borealis nova": "vastitas borealis nova",
}
NOCTIS_SPACE: dict[str, str] = {"tharsis": "31"}  # engine `SpaceName.NOCTIS_CITY` reservation
# Spaces the log can name without coordinates.
NAMED_SPACES: dict[str, dict[str, str]] = {
    "tharsis": {"Tharsis Tholus": "09", "Ascraeus Mons": "14", "Pavonis Mons": "21", "Arsia Mons": "29",
                "Noctis City": "31"},
    "elysium": {"Hecatus Tholus": "08", "Elysium Mons": "14", "Olympus Mons": "20", "Arsia Mons": "37"},
}
OFF_MARS_SPACES: dict[str, str] = {"Ganymede Colony": "01", "Phobos Space Haven": "02"}
# Engine `SpaceBonus` numbers that charge M€ to place on (`spaceCosts`): Hellas' ocean bonus space
# (`HELLAS_BONUS_OCEAN_COST`) and Vastitas Borealis Nova's temperature space.
OCEAN_BONUS, TEMPERATURE_4MC = 5, 18
STEEL_BONUS, TITANIUM_BONUS = 1, 0
BONUS_COSTS: dict[str, dict[int, int]] = {"hellas": {OCEAN_BONUS: 6}, "vastitas borealis nova": {TEMPERATURE_4MC: 4}}

_PLACES_RE = re.compile(r"^places (?:tile )?(?P<tile>.+?) (?:on|into) (?P<where>.*?)\s*(?:\((?P<c>\d+),(?P<r>\d+)\))?\s*$")
_COORDS_RE = re.compile(r"\((\d+),(\d+)\)")


@dataclass(frozen=True)
class MapGeometry:
    name: str
    space_ids: tuple[str, ...]  # template ids in sorted order (the encoder's space order)
    spaces: Mapping[str, Mapping[str, Any]]
    xy_to_id: Mapping[tuple[int, int], str]
    adjacent: Mapping[str, tuple[str, ...]]
    volcanic: frozenset[str]
    noctis: str | None
    named: Mapping[str, str]
    bonus_cost: Mapping[str, int]

    def space_type(self, sid: str) -> str:
        return str(self.spaces[sid]["spaceType"])


def _engine_neighbours(x: int, y: int, middle: int = 4) -> list[tuple[int, int]]:
    """Engine `computeAdjacentSpaces` coordinates (clockwise from the top left)."""
    top_left, top_right, right = [x, y - 1], [x, y - 1], [x + 1, y]
    bottom_right, bottom_left, left = [x, y + 1], [x, y + 1], [x - 1, y]
    if y < middle:
        bottom_left[0] -= 1
        top_right[0] += 1
    elif y == middle:
        bottom_right[0] += 1
        top_right[0] += 1
    else:
        bottom_right[0] += 1
        top_left[0] -= 1
    return [tuple(c) for c in (top_left, top_right, right, bottom_right, bottom_left, left)]  # type: ignore[misc]


@lru_cache(maxsize=None)
def geometry(map_name: str) -> MapGeometry:
    """Geometry of one supported map (lower-case name, as in the templates)."""
    templates = json.loads(_TEMPLATES_PATH.read_text(encoding="utf-8"))
    key = map_name.lower()
    spaces = {str(s["id"]): s for s in templates[key]["spaces"]}
    xy_to_id = {(int(s["x"]), int(s["y"])): sid for sid, s in spaces.items() if s["spaceType"] != "colony"}
    adjacent = {
        sid: tuple(xy_to_id[c] for c in _engine_neighbours(int(s["x"]), int(s["y"])) if c in xy_to_id)
        if s["spaceType"] != "colony" else ()
        for sid, s in spaces.items()
    }
    costs = BONUS_COSTS.get(key, {})
    bonus_cost = {sid: max(costs.get(int(b), 0) for b in s["bonus"]) for sid, s in spaces.items()
                  if any(int(b) in costs for b in s["bonus"])}
    return MapGeometry(
        name=key, space_ids=tuple(sorted(spaces)), spaces=spaces, xy_to_id=xy_to_id, adjacent=adjacent,
        volcanic=VOLCANIC_SPACES[key], noctis=NOCTIS_SPACE.get(key),
        named={**NAMED_SPACES.get(key, {}), **OFF_MARS_SPACES}, bonus_cost=bonus_cost,
    )


def log_coords_to_xy(col: int, row: int) -> tuple[int, int]:
    """Log `(col,row)` -> engine `(x, y)` (see the module docstring)."""
    y = row - 1
    width = 9 - abs(y - 4)
    return col - (math.ceil((9 - width) / 2) + 1) + abs(y - 4), y


@dataclass(frozen=True)
class Placement:
    tile: str  # the log's tile name ("City", "Forest", "Mining", ...)
    where: str  # the log's space text, coordinates stripped
    coords: tuple[int, int] | None

    @property
    def kind(self) -> str | None:
        return TILE_KINDS.get(self.tile)


def parse_placement(rest: str) -> Placement | None:
    """`places <tile> on|into <space> [(<col>,<row>)]` (actor already stripped), else None."""
    m = _PLACES_RE.match(rest)
    if m is None:
        return None
    coords = (int(m.group("c")), int(m.group("r"))) if m.group("c") is not None else None
    return Placement(tile=m.group("tile").strip(), where=m.group("where").strip(), coords=coords)


def structured_placement(tile_placed: Any, tile_location: str) -> Placement:
    """A placement from the structured `tile_placed` / `tile_location` fields alone."""
    return Placement(tile=str(tile_placed).strip(), where=_COORDS_RE.sub("", tile_location).strip(), coords=None)


def locate(geo: MapGeometry, placement: Placement, tile_location: Any = None) -> str | None:
    """Engine space id of a placement: a named space first (off-Mars cities, volcanoes, Noctis), then
    the structured `tile_location` coordinates, then the description's; None when unresolvable."""
    for name, sid in geo.named.items():
        if placement.where == name or placement.where.endswith(" " + name):
            return sid
    coords = None
    if isinstance(tile_location, str) and (m := _COORDS_RE.search(tile_location)):
        coords = (int(m.group(1)), int(m.group(2)))
    coords = coords or placement.coords
    if coords is None:
        return None
    return geo.xy_to_id.get(log_coords_to_xy(*coords))


@dataclass
class Board:
    """One table's board: space id -> (engine tile code, owner seat or `NO_OWNER`)."""

    geo: MapGeometry
    tiles: dict[str, tuple[int, int]] = field(default_factory=dict)

    def tile(self, sid: str) -> int:
        return self.tiles.get(sid, (NO_TILE, NO_OWNER))[0]

    def owner(self, sid: str) -> int:
        return self.tiles.get(sid, (NO_TILE, NO_OWNER))[1]

    def place(self, sid: str, code: int, seat: int) -> None:
        self.tiles[sid] = (code, NO_OWNER if code == OCEAN else seat)

    def count(self, codes: frozenset[int] | set[int], seat: int | None = None) -> int:
        return sum(1 for code, own in self.tiles.values() if code in codes and (seat is None or own == seat))

    def compact(self, actor: int) -> tuple[list[int], list[int]]:
        """Per template space (sorted ids): tile code, and owner relative to `actor` (0 me, 1 opponent)."""
        tiles, owners = [], []
        for sid in self.geo.space_ids:
            code, own = self.tiles.get(sid, (NO_TILE, NO_OWNER))
            tiles.append(code)
            owners.append(NO_OWNER if own == NO_OWNER else (0 if own == actor else 1))
        return tiles, owners


def apply_compact_board(spaces: Sequence[dict[str, Any]], tiles: Sequence[int], owners: Sequence[int],
                        actor: int) -> None:
    """Write compact codes (`Board.compact` order: sorted space ids) into observation `spaces` in place."""
    order = sorted(range(len(spaces)), key=lambda i: str(spaces[i]["id"]))
    if len(order) != len(tiles) or len(tiles) != len(owners):
        raise ValueError(f"board codes cover {len(tiles)} spaces, the map has {len(spaces)}")
    for i, code, own in zip(order, tiles, owners):
        spaces[i]["tileType"] = None if int(code) == NO_TILE else int(code)
        spaces[i]["owner"] = None if int(own) == NO_OWNER else (actor if int(own) == 0 else 1 - actor)
