from __future__ import annotations

from typing import Any

import pytest

from tm_bot.agents import (
    B0_VERSION,
    B0Agent,
    GameRunResult,
    play_game,
)
from tm_bot.env import SUPPORTED_MAPS, Choice, WorkerClient


class _SpyAgent:
    version = "spy-v1"

    def __init__(self) -> None:
        self.calls: list[tuple[int, str, int]] = []
        self.reset_calls: list[int | None] = []

    def reset(self, seed: int | None = None) -> None:
        self.reset_calls.append(seed)

    def act(self, observation: Any, choice: Any) -> int:
        assert isinstance(observation, dict)
        assert set(observation.keys()) == {"actor", "public", "private", "history"}
        assert "players" in observation["public"]
        assert len(observation["public"]["players"]) == 2
        for player_obs in observation["public"]["players"]:
            assert "victoryPoints" not in player_obs
            assert "cardsInHand" not in player_obs
        assert isinstance(choice, Choice)
        assert choice.seat == observation["actor"]
        assert len(choice.candidates) > 0
        self.calls.append((choice.seat, choice.kind, len(choice.candidates)))
        return 0


def test_b0_plays_complete_game_deterministically_through_bridge() -> None:
    with WorkerClient() as client:
        res1 = play_game(
            (B0Agent(seed=101), B0Agent(seed=202)),
            seed_key="test-b0-det-1",
            map_name="tharsis",
            client=client,
        )
        res2 = play_game(
            (B0Agent(seed=101), B0Agent(seed=202)),
            seed_key="test-b0-det-1",
            map_name="tharsis",
            client=client,
        )

    assert isinstance(res1, GameRunResult)
    assert res1.ok is True
    assert res1.error is None
    assert res1.result is not None
    assert res1.generations is not None and res1.generations >= 1
    assert res1.decisions > 0
    assert res1.agent_versions == (B0_VERSION, B0_VERSION)
    assert res1 == res2


def test_b0_reset_reproduces_identical_choice_sequence() -> None:
    a = B0Agent(seed=77)
    b = B0Agent(seed=77)
    obs: Any = {"actor": 0}
    choices = [
        Choice(seat=0, kind="branch", prompt="p", candidates=tuple()),  # replaced below
    ]
    from tm_bot.env import Candidate

    choices = [
        Choice(
            seat=0,
            kind="branch",
            prompt="p",
            candidates=tuple(Candidate(id=f"c{i}", label=f"C{i}") for i in range(7)),
        )
        for _ in range(25)
    ]

    seq1 = [a.act(obs, c) for c in choices]
    seq2 = [b.act(obs, c) for c in choices]
    assert seq1 == seq2

    # Advance a further, then reset() back to original seed
    _ = [a.act(obs, c) for c in choices]
    a.reset()
    seq3 = [a.act(obs, c) for c in choices]
    assert seq3 == seq1


def test_play_game_passes_only_actor_observation_and_choice() -> None:
    spy0 = _SpyAgent()
    spy1 = _SpyAgent()
    with WorkerClient() as client:
        outcome = play_game(
            (spy0, spy1),
            seed_key="test-spy-isolation",
            map_name=SUPPORTED_MAPS[0],
            client=client,
            seat_seeds=(11, 22),
        )
    assert outcome.ok is True
    assert outcome.result is not None
    assert spy0.reset_calls == [11]
    assert spy1.reset_calls == [22]
    assert len(spy0.calls) > 0
    assert len(spy1.calls) > 0
    assert len(spy0.calls) + len(spy1.calls) == outcome.decisions


def test_play_game_reports_step_cap_and_bridge_errors_as_failures() -> None:
    with WorkerClient() as client:
        step_cap_out = play_game(
            (B0Agent(seed=1), B0Agent(seed=2)),
            seed_key="test-step-cap",
            map_name="tharsis",
            client=client,
            max_decisions=5,
        )
        assert step_cap_out.ok is False
        assert step_cap_out.result is None
        assert step_cap_out.error_type == "StepCapError"
        assert step_cap_out.decisions == 5

        bad_map_out = play_game(
            (B0Agent(seed=1), B0Agent(seed=2)),
            seed_key="test-bad-map",
            map_name="amazonis p.",
            client=client,
        )
        assert bad_map_out.ok is False
        assert bad_map_out.result is None
        assert bad_map_out.error is not None


def test_card_data_exported_from_pinned_engine(tmp_path) -> None:
    from tm_bot.agents import export_card_data, load_card_data

    out_path = tmp_path / "cards.json"
    exported = export_card_data(out_path=out_path)
    assert out_path.exists()
    assert len(exported) >= 200

    loaded = load_card_data(path=out_path)
    assert "Adaptation Technology" in loaded
    adapt = loaded["Adaptation Technology"]
    assert adapt.cost == 12
    assert adapt.card_type == "active"
    assert "science" in adapt.tags
    assert adapt.static_vp == 1

    lichen = loaded["Adapted Lichen"]
    assert lichen.cost == 9
    assert lichen.production["plants"] == 1
    assert "plant" in lichen.tags

    # Standard projects and corporations are also present
    assert "Power Plant:SP" in loaded
    assert loaded["Power Plant:SP"].cost == 11
    assert "CrediCor" in loaded
    assert loaded["CrediCor"].starting_megacredits == 57


def test_b1_plays_complete_games_on_all_supported_maps_with_version_tag() -> None:
    from tm_bot.agents import B1_VERSION, B1Agent

    assert B1_VERSION == "b1-v2"
    with WorkerClient() as client:
        for idx, map_name in enumerate(SUPPORTED_MAPS):
            b1 = B1Agent(seed=100 + idx)
            b0 = B0Agent(seed=200 + idx)
            assert b1.version == B1_VERSION
            outcome = play_game(
                (b1, b0),
                seed_key=f"test-b1-map-{map_name}",
                map_name=map_name,
                client=client,
            )
            assert outcome.ok is True, f"Failed on {map_name}: {outcome.error}"
            assert outcome.result is not None
            assert outcome.generations is not None and outcome.generations >= 1
            assert outcome.decisions > 0
            assert outcome.agent_versions == (B1_VERSION, B0_VERSION)


def test_b1_is_deterministic_given_seed() -> None:
    from tm_bot.agents import B1Agent

    with WorkerClient() as client:
        r1 = play_game(
            (B1Agent(seed=42), B0Agent(seed=99)),
            seed_key="test-b1-det-1",
            map_name="elysium",
            client=client,
        )
        r2 = play_game(
            (B1Agent(seed=42), B0Agent(seed=99)),
            seed_key="test-b1-det-1",
            map_name="elysium",
            client=client,
        )
    assert r1.ok is True
    assert r1 == r2


def test_b1_takes_corporation_first_action_instead_of_passing() -> None:
    from tm_bot.agents import B1Agent
    from tm_bot.env import Candidate

    # Until a corporation's initial action is taken, the root action menu offers
    # only that action and Pass (e.g. Valley Trust); passing repeats every generation.
    choice = Choice(
        seat=0,
        kind="branch",
        prompt="Select one option",
        candidates=(
            Candidate(id="branch:0", label="Take first action of Valley Trust corporation"),
            Candidate(id="branch:1", label="Pass for this generation"),
        ),
    )
    obs = {"actor": 0, "public": {"globalParameters": {"temperature": -30, "oxygenLevel": 0, "oceans": 0}}}
    assert B1Agent(seed=0).act(obs, choice) == 0


def test_b1_finishes_first_action_corporation_game_normally() -> None:
    """Regression (#34): B1 in seat 0 with Valley Trust passed every generation."""
    from tm_bot.agents import B1Agent, load_card_data

    with WorkerClient() as client:
        outcome = play_game(
            (B1Agent(seed=0, card_data=load_card_data()), B0Agent(seed=0)),
            seed_key="dev-0006",
            map_name="tharsis",
            client=client,
            seat_seeds=(1608211574, 1168229789),
        )
    assert outcome.ok is True, outcome.error
    assert outcome.result is not None
    assert outcome.generations is not None and outcome.generations <= 25
    assert outcome.result.winner == 0
    assert outcome.result.scores[0] > 60


def test_b1_heuristic_decisions_at_public_seam() -> None:
    from tm_bot.agents import B1Agent
    from tm_bot.env import Candidate

    agent = B1Agent(seed=0)

    def make_obs(
        *,
        mc: int = 20,
        temp: int = -20,
        oxy: int = 4,
        oceans: int = 2,
        generation: int = 2,
        awards: list[dict[str, Any]] | None = None,
        cards_in_hand: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return {
            "actor": 0,
            "public": {
                "map": "tharsis",
                "generation": generation,
                "phase": "action",
                "firstPlayer": 0,
                "activePlayer": 0,
                "passedPlayers": [],
                "globalParameters": {
                    "temperature": temp,
                    "oxygenLevel": oxy,
                    "oceans": oceans,
                    "isTerraformed": False,
                },
                "deckSize": 100,
                "discardPileSize": 0,
                "otherDeckSizes": {"corporations": {"drawPile": 10, "discardPile": 0}, "preludes": None},
                "spaces": [],
                "milestones": [],
                "awards": awards or [],
                "players": [
                    {
                        "seat": 0,
                        "color": "red",
                        "terraformRating": 22,
                        "megacredits": mc,
                        "steel": 2,
                        "titanium": 1,
                        "plants": 0,
                        "energy": 0,
                        "heat": 0,
                        "megacreditProduction": 3,
                        "steelProduction": 1,
                        "titaniumProduction": 0,
                        "plantProduction": 1,
                        "energyProduction": 0,
                        "heatProduction": 1,
                        "steelValue": 2,
                        "titaniumValue": 3,
                        "cardCost": 3,
                        "cardDiscount": 0,
                        "cardsInHandCount": len(cards_in_hand or []),
                        "citiesCount": 0,
                        "noTagsCount": 0,
                        "tags": {},
                        "tableau": [],
                        "usedActions": [],
                        "actionsTakenThisRound": 0,
                        "actionsTakenThisGame": 0,
                        "availableBlueCardActionCount": 0,
                        "lastCardPlayed": None,
                        "isActive": True,
                        "passed": False,
                        "protectedResources": {},
                        "protectedProduction": {},
                    },
                    {
                        "seat": 1,
                        "color": "blue",
                        "terraformRating": 20,
                        "megacredits": 15,
                        "steel": 0,
                        "titanium": 0,
                        "plants": 0,
                        "energy": 0,
                        "heat": 0,
                        "megacreditProduction": 0,
                        "steelProduction": 0,
                        "titaniumProduction": 0,
                        "plantProduction": 0,
                        "energyProduction": 0,
                        "heatProduction": 0,
                        "steelValue": 2,
                        "titaniumValue": 3,
                        "cardCost": 3,
                        "cardDiscount": 0,
                        "cardsInHandCount": 2,
                        "citiesCount": 0,
                        "noTagsCount": 0,
                        "tags": {},
                        "tableau": [],
                        "usedActions": [],
                        "actionsTakenThisRound": 0,
                        "actionsTakenThisGame": 0,
                        "availableBlueCardActionCount": 0,
                        "lastCardPlayed": None,
                        "isActive": False,
                        "passed": False,
                        "protectedResources": {},
                        "protectedProduction": {},
                    },
                ],
                "publicLog": [],
            },
            "private": {
                "cardsInHand": cards_in_hand or [],
                "dealtCorporationCards": [],
                "dealtPreludeCards": [],
                "dealtProjectCards": [],
                "pickedCorporationCard": "CrediCor",
                "preludeCardsInHand": [],
                "draftedCards": [],
                "currentDraftPacket": [],
            },
            "history": {
                "initialBuy": None,
                "drafts": [],
                "researchBuys": [],
                "cardSelections": [],
            },
        }

    # 1. Prefers Claim a milestone over Pass / Standard projects
    obs = make_obs(mc=10)
    branch_milestone = Choice(
        seat=0,
        kind="branch",
        prompt="Take your next action",
        candidates=(
            Candidate(id="branch:3", label="Standard projects"),
            Candidate(id="branch:4", label="Pass for this generation"),
            Candidate(id="branch:0", label="Claim a milestone"),
        ),
    )
    assert agent.act(obs, branch_milestone) == 2

    # 2. Prefers canonical (first) payment candidate
    pay_choice = Choice(
        seat=0,
        kind="payment",
        prompt="Select payment for Mine",
        candidates=(
            Candidate(id="megacredits=0,steel=2", label="2 steel"),
            Candidate(id="megacredits=4", label="4 M€"),
        ),
    )
    assert agent.act(obs, pay_choice) == 0

    # 3. Targets opponent on player selection
    player_choice = Choice(
        seat=0,
        kind="player",
        prompt="Select player to remove 2 plants",
        candidates=(
            Candidate(id="red", label="Seat 1 (red)"),
            Candidate(id="blue", label="Seat 2 (blue)"),
        ),
    )
    assert agent.act(obs, player_choice) == 1

    # 4. Initial buy uses chosen corp startingMegaCredits even while me["megacredits"] == 0
    init_obs = make_obs(mc=0)
    init_obs["private"]["pickedCorporationCard"] = None
    init_obs["private"]["dealtProjectCards"] = ["Acquired Company", "Underground Detonations"]
    corp_choice = Choice(
        seat=0,
        kind="card",
        prompt="Select corporation",
        candidates=(
            Candidate(id="CrediCor", label="CrediCor"),
            Candidate(id="PhoboLog", label="PhoboLog"),
        ),
    )
    assert agent.act(init_obs, corp_choice) == 0
    init_buy = Choice(
        seat=0,
        kind="multi_card",
        prompt="Select initial cards to buy",
        candidates=(
            Candidate(id="Underground Detonations", label="Underground Detonations"),
            Candidate(id="Acquired Company", label="Acquired Company"),
            Candidate(id="done", label="Done (0 selected)"),
        ),
    )
    assert agent.act(init_obs, init_buy) == 1

    # 5. Funds an award when strictly ahead
    award_obs = make_obs(
        mc=20,
        awards=[
            {
                "name": "Banker",
                "fundedBy": None,
                "scores": [{"seat": 0, "score": 6}, {"seat": 1, "score": 1}],
            }
        ],
    )
    branch_award = Choice(
        seat=0,
        kind="branch",
        prompt="Take your next action",
        candidates=(
            Candidate(id="branch:0", label="Pass for this generation"),
            Candidate(id="branch:1", label="Fund an award (8 M€)"),
        ),
    )
    assert agent.act(award_obs, branch_award) == 1

    # 6. Space placement prefers adjacency to own city over opponent city
    space_obs = make_obs()
    space_obs["public"]["spaces"] = [
        {"id": "01", "x": 4, "y": 4, "spaceType": "land", "bonus": [], "tileType": 2, "owner": 0, "highlight": None, "rotated": False},
        {"id": "02", "x": 5, "y": 4, "spaceType": "land", "bonus": [], "tileType": None, "owner": None, "highlight": None, "rotated": False},
        {"id": "03", "x": 0, "y": 0, "spaceType": "land", "bonus": [], "tileType": 2, "owner": 1, "highlight": None, "rotated": False},
        {"id": "04", "x": 1, "y": 0, "spaceType": "land", "bonus": [], "tileType": None, "owner": None, "highlight": None, "rotated": False},
    ]
    greenery_space = Choice(
        seat=0,
        kind="space",
        prompt="Select space for greenery tile",
        candidates=(
            Candidate(id="04", label="04"),
            Candidate(id="02", label="02"),
        ),
    )
    assert agent.act(space_obs, greenery_space) == 1


def test_play_games_batches_across_pool_and_isolates_failures() -> None:
    from tm_bot.agents import B1Agent, GameSpec, play_games

    specs = [
        GameSpec(seed_key="batch-g-0", map_name="tharsis", agents=(B1Agent(seed=1), B0Agent(seed=2))),
        GameSpec(seed_key="batch-g-1", map_name="elysium", agents=(B0Agent(seed=3), B1Agent(seed=4))),
        GameSpec(seed_key="batch-g-bad", map_name="amazonis p.", agents=(B1Agent(seed=5), B0Agent(seed=6))),
        GameSpec(seed_key="batch-g-2", map_name="hellas", agents=(B1Agent(seed=7), B0Agent(seed=8))),
    ]

    results = play_games(specs, num_workers=2, batch_size=2)
    assert len(results) == 4
    assert results[0].ok is True
    assert results[1].ok is True
    assert results[2].ok is False
    assert results[2].result is None
    assert results[3].ok is True

    # Verify parity with single-game play_game
    with WorkerClient() as client:
        single_0 = play_game(
            (B1Agent(seed=1), B0Agent(seed=2)),
            seed_key="batch-g-0",
            map_name="tharsis",
            client=client,
        )
    assert results[0] == single_0



