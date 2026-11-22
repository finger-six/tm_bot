"""Legal tile-placement space menus from a rebuilt board (v2 warm start, #61).

Pure functions over a `tm_bot.data.board.Board`, the placing seat and its M€ funds. They port the
pinned engine's placement rules (`tools/simbench/engine/src/server/boards/Board.ts`
`getAvailableSpacesOnLand`, `MarsBoard.ts` `getAvailableSpacesForType` / `ForCity` / `ForGreenery` /
`ForOcean` / `IsolatedSpaces` / `VolcanicSpaces`) and the base + Prelude cards' own placement
filters (`MiningCard`, `MiningArea`, `EcologicalZone`, `IndustrialCenter`, `UrbanizedArea`,
`LavaTubeSettlement`). Board features outside base + Prelude (Ares hazards, cubes, Land Claim
markers, Gordon, Red City, Kingdom of Tauraro) never occur in the supported tables. Like the action
menus these are supervision approximations: the live adapter stays the legality authority.

A placement's `rule` comes from its source (`PLACEMENT_SOURCES`): the card played or the card
action taken, a standard project, or the plant conversion. Two sources place without a prompt on
the supported maps (`NO_PROMPT`): Noctis City on Tharsis (the reserved space) and the off-Mars
cities.
"""

from __future__ import annotations

from typing import Mapping

from tm_bot.agents.card_data import CardSpec
from tm_bot.data.board import (
    CITY,
    CITY_TILES,
    GREENERY,
    MINING_AREA,
    MINING_RIGHTS,
    NO_OWNER,
    NO_TILE,
    OCEAN,
    SPECIAL_TILES,
    STEEL_BONUS,
    TITANIUM_BONUS,
    Board,
)

RULES: tuple[str, ...] = (
    "land", "city", "greenery", "ocean", "isolated", "volcanic", "noctis", "mining_area", "mining_rights",
    "ecological_zone", "industrial_center", "urbanized_area", "lava_tube_settlement",
)
NO_PROMPT = "no_prompt"
CONVERT_PLANTS = "convert_plants"
HELLAS_OCEAN_BONUS = "hellas_ocean_bonus"
TEMPERATURE_OCEAN_BONUS = "temperature_ocean_bonus"

# Source -> (placed tile kind, rule). Kinds are `tm_bot.data.board.TILE_KINDS` values.
_FIXED_SOURCES: dict[str, tuple[str, str]] = {
    # Standard projects (engine names), the plant conversion and the Hellas ocean bonus.
    "City": ("city", "city"), "Greenery": ("greenery", "greenery"), "Aquifer": ("ocean", "ocean"),
    CONVERT_PLANTS: ("greenery", "greenery"), HELLAS_OCEAN_BONUS: ("ocean", "ocean"),
    TEMPERATURE_OCEAN_BONUS: ("ocean", "ocean"),
    # Cities.
    "Corporate Stronghold": ("city", "city"), "Cupola City": ("city", "city"), "Domed Crater": ("city", "city"),
    "Early Settlement": ("city", "city"), "Immigrant City": ("city", "city"), "Open City": ("city", "city"),
    "Self-Sufficient Settlement": ("city", "city"), "Underground City": ("city", "city"),
    "Tharsis Republic": ("city", "city"), "Noctis City": ("city", "noctis"),
    "Research Outpost": ("city", "isolated"), "Urbanized Area": ("city", "urbanized_area"),
    "Lava Tube Settlement": ("city", "lava_tube_settlement"),
    "Ganymede Colony": ("city", NO_PROMPT), "Phobos Space Haven": ("city", NO_PROMPT),
    # Greeneries; Mangrove and Protected Valley go on ocean spaces.
    "Plantation": ("greenery", "greenery"), "Experimental Forest": ("greenery", "greenery"),
    "Mangrove": ("greenery", "ocean"), "Protected Valley": ("greenery", "ocean"),
    # Oceans not covered by `CardSpec.oceans`: Flooding and the two ocean card actions; Artificial Lake
    # goes on land.
    "Flooding": ("ocean", "ocean"), "Aquifer Pumping": ("ocean", "ocean"),
    "Water Import From Europa": ("ocean", "ocean"), "Artificial Lake": ("ocean", "land"),
    # Special tiles.
    "Capital": ("Capital", "city"), "Commercial District": ("Commercial District", "land"),
    "Ecological Zone": ("Ecological Zone", "ecological_zone"),
    "Industrial Center": ("Industrial Center", "industrial_center"), "Lava Flows": ("Lava Flows", "volcanic"),
    "Mining Area": ("Mining", "mining_area"), "Mining Rights": ("Mining", "mining_rights"),
    "Mohole Area": ("Mohole Area", "ocean"), "Natural Preserve": ("Natural Preserve", "isolated"),
    "Nuclear Zone": ("Nuclear Zone", "land"), "Restricted Area": ("Restricted Area", "land"),
}


def placement_sources(specs: Mapping[str, CardSpec]) -> dict[str, tuple[str, str]]:
    """Every known placement source: the fixed table plus each card whose `oceans` places oceans."""
    out = {name: ("ocean", "ocean") for name, spec in specs.items()
           if spec.oceans > 0 and spec.card_type != "standard_project"}
    out.update(_FIXED_SOURCES)
    return out


def tile_code(kind: str, source: str | None) -> int:
    """Engine tile code of a placed tile kind (`Mining` resolves through its source card)."""
    if kind == "city":
        return CITY
    if kind == "greenery":
        return GREENERY
    if kind == "ocean":
        return OCEAN
    if kind == "Mining":
        return MINING_RIGHTS if source == "Mining Rights" else MINING_AREA
    return SPECIAL_TILES[kind]


def _is_city(board: Board, sid: str) -> bool:
    return board.tile(sid) in CITY_TILES


def land_spaces(board: Board, seat: int, funds: int) -> list[str]:
    """Engine `getAvailableSpacesOnLand`: empty land spaces, not the Noctis reservation, whose
    placement M€ cost (Hellas ocean bonus 6, Vastitas Borealis Nova temperature 4) is affordable."""
    geo = board.geo
    return [sid for sid in geo.space_ids
            if geo.space_type(sid) == "land" and sid != geo.noctis and board.tile(sid) == NO_TILE
            and board.owner(sid) in (NO_OWNER, seat) and funds >= geo.bonus_cost.get(sid, 0)]


def ocean_spaces(board: Board, seat: int) -> list[str]:
    """Engine `getAvailableSpacesForOcean`: empty ocean-reserved spaces."""
    geo = board.geo
    return [sid for sid in geo.space_ids
            if geo.space_type(sid) == "ocean" and board.tile(sid) == NO_TILE and board.owner(sid) in (NO_OWNER, seat)]


def city_spaces(board: Board, seat: int, funds: int) -> list[str]:
    """Engine `getAvailableSpacesForCity`: land spaces with no adjacent city."""
    return [sid for sid in land_spaces(board, seat, funds)
            if not any(_is_city(board, a) for a in board.geo.adjacent[sid])]


def greenery_spaces(board: Board, seat: int, funds: int) -> list[str]:
    """Engine `getAvailableSpacesForGreenery`: land spaces next to an own tile, or any land space when none is."""
    land = land_spaces(board, seat, funds)
    own = [sid for sid in land
           if any(board.tile(a) != NO_TILE and board.owner(a) == seat for a in board.geo.adjacent[sid])]
    return own or land


def space_menu(rule: str, board: Board, seat: int, funds: int) -> list[str]:
    """Legal space ids (sorted) for one placement rule; `NO_PROMPT` and unknown rules have none."""
    geo, adj = board.geo, board.geo.adjacent
    if rule == "land":
        return land_spaces(board, seat, funds)
    if rule == "city" or (rule == "noctis" and geo.noctis is None):
        return city_spaces(board, seat, funds)
    if rule == "greenery":
        return greenery_spaces(board, seat, funds)
    if rule == "ocean":
        return ocean_spaces(board, seat)
    if rule == "isolated":
        return [s for s in land_spaces(board, seat, funds) if all(board.tile(a) == NO_TILE for a in adj[s])]
    if rule == "volcanic":
        land = land_spaces(board, seat, funds)
        return [s for s in land if s in geo.volcanic] if geo.volcanic else land
    if rule == "lava_tube_settlement":
        return space_menu("volcanic" if geo.volcanic else "city", board, seat, funds)
    if rule in ("mining_rights", "mining_area"):
        out = [s for s in land_spaces(board, seat, funds)
               if any(int(b) in (STEEL_BONUS, TITANIUM_BONUS) for b in geo.spaces[s]["bonus"])]
        if rule == "mining_area":
            out = [s for s in out if any(board.tile(a) not in (NO_TILE, OCEAN) and board.owner(a) == seat
                                         for a in adj[s])]
        return out
    if rule == "ecological_zone":
        return [s for s in land_spaces(board, seat, funds) if any(board.tile(a) == GREENERY for a in adj[s])]
    if rule == "industrial_center":
        return [s for s in land_spaces(board, seat, funds) if any(_is_city(board, a) for a in adj[s])]
    if rule == "urbanized_area":
        return [s for s in land_spaces(board, seat, funds) if sum(_is_city(board, a) for a in adj[s]) >= 2]
    return []
