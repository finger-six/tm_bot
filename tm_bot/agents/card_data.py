"""Static card metadata derived from the pinned Terraforming Mars engine."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CARD_DATA_PATH = _REPO_ROOT / "data" / "cards.json"
_EXPORT_SCRIPT = _REPO_ROOT / "adapter" / "dist" / "scripts" / "export_cards.js"

_CACHE: dict[Path, dict[str, CardSpec]] = {}


@dataclass(frozen=True, slots=True)
class CardSpec:
    """Public static rules metadata for a single Terraforming Mars card or standard project."""

    name: str
    module: str
    card_type: str
    cost: int
    tags: tuple[str, ...]
    starting_megacredits: int
    card_cost: int | None
    static_vp: float
    has_dynamic_vp: bool
    vp_per_resource: float
    production: dict[str, int]
    has_dynamic_production: bool
    stock: dict[str, int]
    tr: int
    temperature_steps: int
    oxygen_steps: int
    oceans: int
    greenery: int
    city: int
    draw_card: int
    has_action: bool
    resource_type: str | None
    discount_amount: int
    tiles_built: tuple[int, ...]
    requirements_count: int
    requirements: tuple[dict[str, Any], ...]


def _parse_card(raw: dict[str, Any]) -> CardSpec:
    return CardSpec(
        name=str(raw["name"]),
        module=str(raw["module"]),
        card_type=str(raw["cardType"]),
        cost=int(raw.get("cost", 0)),
        tags=tuple(str(t) for t in raw.get("tags", ())),
        starting_megacredits=int(raw.get("startingMegaCredits", 0)),
        card_cost=int(raw["cardCost"]) if raw.get("cardCost") is not None else None,
        static_vp=float(raw.get("staticVp", 0)),
        has_dynamic_vp=bool(raw.get("hasDynamicVp", False)),
        vp_per_resource=float(raw.get("vpPerResource", 0.0)),
        production={k: int(v) for k, v in dict(raw.get("production", {})).items()},
        has_dynamic_production=bool(raw.get("hasDynamicProduction", False)),
        stock={k: int(v) for k, v in dict(raw.get("stock", {})).items()},
        tr=int(raw.get("tr", 0)),
        temperature_steps=int(raw.get("temperatureSteps", 0)),
        oxygen_steps=int(raw.get("oxygenSteps", 0)),
        oceans=int(raw.get("oceans", 0)),
        greenery=int(raw.get("greenery", 0)),
        city=int(raw.get("city", 0)),
        draw_card=int(raw.get("drawCard", 0)),
        has_action=bool(raw.get("hasAction", False)),
        resource_type=str(raw["resourceType"]) if raw.get("resourceType") is not None else None,
        discount_amount=int(raw.get("discountAmount", 0)),
        tiles_built=tuple(int(t) for t in raw.get("tilesBuilt", ())),
        requirements_count=int(raw.get("requirementsCount", 0)),
        requirements=tuple(dict(r) for r in raw.get("requirements", ())),
    )


def export_card_data(
    out_path: Path | str | None = None,
    *,
    node_executable: str = "node",
) -> dict[str, CardSpec]:
    """Regenerate the static card-data JSON from the pinned engine and return parsed CardSpecs."""
    if not _EXPORT_SCRIPT.exists():
        raise FileNotFoundError(
            f"Card export script not found at {_EXPORT_SCRIPT}. "
            "Build the adapter first (`cd adapter && npm run build`)."
        )
    target = Path(out_path) if out_path is not None else DEFAULT_CARD_DATA_PATH
    subprocess.run(
        [node_executable, str(_EXPORT_SCRIPT), "--out", str(target)],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    _CACHE.pop(target.resolve(), None)
    return load_card_data(path=target, auto_export=False)


def load_card_data(
    path: Path | str | None = None,
    *,
    auto_export: bool = True,
) -> dict[str, CardSpec]:
    """Load static card metadata from JSON, exporting from the pinned engine if missing."""
    target = Path(path) if path is not None else DEFAULT_CARD_DATA_PATH
    resolved = target.resolve()
    cached = _CACHE.get(resolved)
    if cached is not None:
        return cached

    if not target.exists():
        if not auto_export:
            raise FileNotFoundError(f"Static card data file not found at {target}")
        return export_card_data(out_path=target)

    payload = json.loads(target.read_text(encoding="utf-8"))
    cards = {spec.name: spec for spec in (_parse_card(item) for item in payload["cards"])}
    _CACHE[resolved] = cards
    return cards


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export static Terraforming Mars card metadata from the pinned engine.")
    parser.add_argument("--export", action="store_true", help="Regenerate card metadata JSON.")
    parser.add_argument("--out", type=Path, default=DEFAULT_CARD_DATA_PATH, help="Output JSON path.")
    args = parser.parse_args(argv)
    if args.export or not args.out.exists():
        cards = export_card_data(out_path=args.out)
        print(f"Exported {len(cards)} cards to {args.out}")
    else:
        cards = load_card_data(path=args.out)
        print(f"Loaded {len(cards)} cards from {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
