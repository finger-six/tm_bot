from __future__ import annotations

import json
from pathlib import Path

from tm_bot.data.cards import CardRegistry


def test_card_registry_resolves_exact_casefold_and_bga_aliases(tmp_path: Path) -> None:
    reg_path = tmp_path / "card_registry.json"
    reg_path.write_text(
        json.dumps(
            {
                "cards": [
                    "ArchaeBacteria",
                    "Beam From A Thorium Asteroid",
                    "Biolab",
                    "CEO's Favorite Project",
                    "Cheung Shing MARS",
                    "Eccentric Sponsor",
                    "Ice Cap Melting",
                    "Permafrost Extraction",
                    "ThorGate",
                    "Towing A Comet",
                ]
            }
        ),
        encoding="utf-8",
    )
    reg = CardRegistry.from_json(reg_path)

    # Exact matches
    assert reg.resolve("ThorGate") == "ThorGate"
    assert reg.resolve("CEO's Favorite Project") == "CEO's Favorite Project"

    # Casefold matches
    assert reg.resolve("Archaebacteria") == "ArchaeBacteria"
    assert reg.resolve("Beam From a Thorium Asteroid") == "Beam From A Thorium Asteroid"
    assert reg.resolve("Cheung Shing Mars") == "Cheung Shing MARS"
    assert reg.resolve("Ice cap Melting") == "Ice Cap Melting"
    assert reg.resolve("Permafrost extraction") == "Permafrost Extraction"
    assert reg.resolve("Towing a Comet") == "Towing A Comet"
    assert reg.resolve("  thorgate  ") == "ThorGate"

    # Known BGA spelling aliases
    assert reg.resolve("Biolabs") == "Biolab"
    assert reg.resolve("CEO's Favourite Project") == "CEO's Favorite Project"
    assert reg.resolve("Excentric Sponsor") == "Eccentric Sponsor"

    # Unknown cards must never resolve
    assert reg.resolve("Totally Fake Card") is None
    assert reg.resolve("") is None
    assert reg.resolve(None) is None


def test_card_registry_does_not_resolve_alias_if_target_missing_from_registry() -> None:
    small_reg = CardRegistry.from_names(["CrediCor", "Vitor"])
    assert small_reg.resolve("CrediCor") == "CrediCor"
    assert small_reg.resolve("Biolabs") is None
    assert small_reg.resolve("CEO's Favourite Project") is None
    assert small_reg.resolve("Excentric Sponsor") is None


def test_card_registry_from_plain_json_list(tmp_path: Path) -> None:
    reg_path = tmp_path / "cards.json"
    reg_path.write_text(json.dumps(["Acquired Company", "Vitor"]), encoding="utf-8")
    reg = CardRegistry.from_json(reg_path)
    assert reg.resolve("acquired company") == "Acquired Company"
    assert reg.resolve("Vitor") == "Vitor"
