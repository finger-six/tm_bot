"""Space-menu parity: the reconstructed placement menus and `space` rows match the live adapter (#61).

Seeded games on every supported map are driven through the real adapter (`WorkerClient`). At each
`SelectSpace` prompt whose placement source is known (the card played or activated, the standard
project, the plant conversion), the live board is read into a `Board`, and
`spacemenu.space_menu` for that source's rule must equal the live candidate ids. The `space`
builder, fed the live state with its compact board codes, must produce the same prompt, candidate
ids and features, and the same encoded board (`space_feats`) as the live observation.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

import numpy as np
import pytest

from tests.test_action_parity_v2 import _row_from_obs
from tm_bot.data import actionmenu, spacemenu
from tm_bot.data.board import NO_OWNER, NO_TILE, Board, geometry
from tm_bot.env import WorkerClient
from tm_bot.env.types import ActorObservation, Choice
from tm_bot.models.encode import encode_choice, encode_observation
from tm_bot.models.vocab import CardVocab
from tm_bot.training.warmstart_builder import build_space_choice

MAPS = ("tharsis", "hellas", "elysium", "vastitas borealis nova")
GAMES_PER_MAP, MAX_STEPS = 3, 4000
_PREFERRED_ROOTS = ("Play project card", "Standard projects", "Convert")
_PREFERRED_PROJECTS = ("City", "Greenery", "Aquifer")
# Ocean prompts a global-parameter or placement bonus inserts into another source's placements.
_BONUS_PROMPTS = {
    "Select space for ocean from temperature increase": spacemenu.TEMPERATURE_OCEAN_BONUS,
    "Select space for ocean from placement bonus": spacemenu.HELLAS_OCEAN_BONUS,
}


def _live_board(obs: ActorObservation, map_name: str) -> Board:
    board = Board(geometry(map_name))
    for sp in obs["public"]["spaces"]:  # a Land Claim marker is an owner without a tile
        if sp["tileType"] is not None or sp["owner"] is not None:
            board.tiles[str(sp["id"])] = (NO_TILE if sp["tileType"] is None else int(sp["tileType"]),
                                          NO_OWNER if sp["owner"] is None else int(sp["owner"]))
    return board


def _drive(map_name: str, game: int, sources: dict[str, tuple[str, str]]) -> list[dict[str, Any]]:
    """Space prompts of one seeded game with a known placement source."""
    rng = random.Random(1000 * game + MAPS.index(map_name))
    out: list[dict[str, Any]] = []
    source: dict[int, str | None] = {0: None, 1: None}
    with WorkerClient() as client:
        gid = client.create(seed_key=f"space-parity-{map_name}-{game}", map=map_name)
        for _ in range(MAX_STEPS):
            seats = client.pending(gid)
            if not seats:
                break
            seat = seats[0]
            ch = client.choice(gid, seat)
            labels = [c.label for c in ch.candidates]
            idx = rng.randrange(len(labels))
            if ch.kind == "branch" and ch.prompt.startswith("Take your"):
                for want in _PREFERRED_ROOTS:
                    hits = [i for i, lb in enumerate(labels) if lb.startswith(want)]
                    if hits and rng.random() < 0.6:
                        idx = hits[0]
                        break
                source[seat] = spacemenu.CONVERT_PLANTS if labels[idx].startswith("Convert") and "plants" in labels[idx] \
                    else None
            elif ch.kind == "project_card" and ch.prompt == "Standard projects":
                hits = [i for i, c in enumerate(ch.candidates) if c.id in _PREFERRED_PROJECTS]
                idx = rng.choice(hits) if hits and rng.random() < 0.8 else idx
                source[seat] = ch.candidates[idx].id
            elif ch.kind == "project_card" or (ch.kind == "card" and ch.prompt.startswith("Perform an action")):
                source[seat] = ch.candidates[idx].id
            elif ch.kind == "space":
                src = _BONUS_PROMPTS.get(ch.prompt, source[seat])
                if src in sources:
                    out.append({"choice": ch, "obs": client.observe(gid, seat), "source": src, "seat": seat,
                                "placement_index": int(ch.prompt == "Select space for second ocean")})
            client.choose(gid, seat, idx)
        client.close_game(gid)
    return out


@pytest.fixture(scope="module")
def live_space_prompts() -> list[dict[str, Any]]:
    from tm_bot.agents.card_data import load_card_data

    sources = spacemenu.placement_sources(load_card_data())
    prompts = []
    for map_name in MAPS:
        for game in range(GAMES_PER_MAP):
            for p in _drive(map_name, game, sources):
                p["map"], p["rule"] = map_name, sources[p["source"]][1]
                prompts.append(p)
    return prompts


def _space_row(p: dict[str, Any], board: Board, menu: list[str]) -> dict[str, Any]:
    row = _row_from_obs(p["obs"], menu=menu)
    row["board_tiles"], row["board_owners"] = board.compact(p["seat"])
    row.update(source=p["source"], rule=p["rule"], chosen=menu[0] if menu else None,
               placement_index=p["placement_index"])
    return row


def test_space_menus_match_live_adapter(live_space_prompts) -> None:
    rules = Counter(p["rule"] for p in live_space_prompts)
    assert len(live_space_prompts) >= 60 and {"city", "greenery", "ocean"} <= set(rules), rules
    misses = []
    for p in live_space_prompts:
        board = _live_board(p["obs"], p["map"])
        funds = actionmenu.funds_mc(_row_from_obs(p["obs"], menu=[]))
        menu = spacemenu.space_menu(p["rule"], board, p["seat"], funds)
        live = [c.id for c in p["choice"].candidates]
        if menu != live:
            misses.append((p["map"], p["source"], p["rule"], sorted(set(menu) ^ set(live))))
    assert not misses, misses


def test_space_choice_matches_live_adapter(live_space_prompts, vocab: CardVocab) -> None:
    for p in live_space_prompts:
        live: Choice = p["choice"]
        board = _live_board(p["obs"], p["map"])
        built_obs, built = build_space_choice(_space_row(p, board, [c.id for c in live.candidates]))
        assert (built.kind, built.prompt) == (live.kind, live.prompt)
        assert [(c.id, c.label) for c in built.candidates] == [(c.id, c.label) for c in live.candidates]
        np.testing.assert_array_equal(encode_observation(built_obs, vocab).space_feats,
                                      encode_observation(p["obs"], vocab).space_feats)
        enc_live, enc_built = encode_choice(live, p["obs"], vocab), encode_choice(built, built_obs, vocab)
        np.testing.assert_array_equal(enc_built.cand_feats, enc_live.cand_feats)
        assert enc_built.prompt_type_id == enc_live.prompt_type_id


@pytest.fixture(scope="module")
def vocab() -> CardVocab:
    return CardVocab.from_card_data()
