"""Action-phase candidate menus reconstructed from pre-move state (v2 warm start, #44).

Pure functions over one actor-relative snapshot (the `tm_bot.data.logstate` column group) plus a few
action-phase extras (used blue-card actions, turn position, whether the opponent passed, the actor's
last played card). Menus are supervision approximations only: the live adapter stays the legality
authority at play time. The rules and strings follow the pinned engine
(`tools/simbench/engine/src/server/Player.ts` `getActions`) and the adapter
(`adapter/src/factor.ts`), not the human platform:

- Root order is the engine's `getActions` push order (`BRANCH_ORDER`). `Sell patents` is a root option
  of its own and never part of the `Standard projects` prompt, so it is a separate branch key.
- Heat and plant conversions stay legal at max temperature / max oxygen (the engine only warns).
- `Standard projects` is pushed every turn but the adapter keeps it only when one standard project is
  affordable; the asteroid and aquifer projects stay legal at max temperature / max oceans.
- Card playability: cost after tableau discounts, payable with M€ + steel (building tag) + titanium
  (space tag) at their values + heat (Helion); global, tag, production, greenery and city requirements
  from `data/cards.json`, with Inventrix / Adaptation Technology / Special Design requirement bonuses;
  own production decreases must stay above the engine minimum. A requirement kind this module cannot
  check is treated as met (the card stays in the menu) and counted by the caller.
- Card actions: the actor's tableau cards with a blue action (`hasAction`), corporations included, not
  yet used this generation. The per-card `canAct` resource checks are not reconstructed.
- Milestones: only milestones whose score is derivable from the snapshot are proven claimable;
  `Polar Explorer`, `Diversifier` and the Vastitas Borealis Nova set never are.
"""

from __future__ import annotations

from typing import Any, Collection, Mapping

from tm_bot.agents.card_data import CardSpec

# Root branch keys in the engine's `getActions` push order.
BRANCH_ORDER: tuple[str, ...] = (
    "claim_milestone", "convert_plants", "convert_heat", "card_action", "play_project_card", "end_turn",
    "fund_award", "standard_projects", "pass", "sell_patents",
)
# Pushed by the engine every turn whatever the state (they still shift later option indices).
ALWAYS_PUSHED: frozenset[str] = frozenset({"standard_projects", "pass"})
# Classified log root (`tm_bot.data.logstate`) -> the root branch key it chose.
ROOT_TO_BRANCH: dict[str, str] = {
    "claim_milestone": "claim_milestone", "convert_plants": "convert_plants", "convert_heat": "convert_heat",
    "card_action": "card_action", "play_card": "play_project_card", "end_turn": "end_turn",
    "fund_award": "fund_award", "standard_project": "standard_projects", "pass": "pass",
    "sell_patents": "sell_patents",
}

MILESTONE_COST = 8
AWARD_COSTS: tuple[int, ...] = (8, 14, 20)
MAX_MILESTONES = 3
MAX_AWARDS = 3
HEAT_FOR_TEMPERATURE = 8

# Engine `Game.getStandardProjects()` (cost order) for the supported 2-player format; `Buffer Gas` is
# solo-only and `Sell Patents` is not displayed as a card.
STANDARD_PROJECTS: tuple[tuple[str, int], ...] = (
    ("Power Plant:SP", 11), ("Asteroid:SP", 14), ("Aquifer", 18), ("Greenery", 23), ("City", 25),
)
# Log spelling (`plays standard project <name>`, lowercased) -> engine card name.
LOG_STANDARD_PROJECTS: dict[str, str] = {
    "power plant": "Power Plant:SP", "asteroid": "Asteroid:SP", "aquifer": "Aquifer", "greenery": "Greenery",
    "city": "City",
}

# Engine `cardDiscount` (tag, amount, per-card) of the supported cards.
_CARD_DISCOUNTS: dict[str, tuple[str | None, int, bool]] = {
    "Anti-Gravity Technology": (None, 2, False), "Earth Catapult": (None, 2, False),
    "Research Outpost": (None, 1, False), "Earth Office": ("earth", 3, False), "Teractor": ("earth", 3, False),
    "Mass Converter": ("space", 2, True), "Quantum Extractor": ("space", 2, False),
    "Shuttles": ("space", 2, False), "Space Station": ("space", 2, False), "ThorGate": ("power", 3, False),
    "Cheung Shing MARS": ("building", 2, False), "Valley Trust": ("science", 2, False),
}
_NEXT_CARD_DISCOUNTS: dict[str, int] = {"Indentured Workers": 8}
_REQUIREMENT_BONUS_CARDS = ("Inventrix", "Adaptation Technology")
_NEXT_CARD_REQUIREMENT_BONUS = "Special Design"
_PRODUCTION_MINIMUM = {"megacredits": -5}
_PROD_COLS = {
    "megacredits": "mc_prod", "steel": "steel_prod", "titanium": "titanium_prod", "plants": "plants_prod",
    "energy": "energy_prod", "heat": "heat_prod",
}
_TAG_INDEX = {t: i for i, t in enumerate((
    "building", "space", "science", "power", "earth", "jovian", "plant", "microbe", "animal", "city", "wild",
    "event",
))}

MAP_MILESTONES: dict[str, tuple[str, ...]] = {
    "tharsis": ("Terraformer", "Mayor", "Gardener", "Builder", "Planner"),
    "hellas": ("Diversifier", "Tactician", "Polar Explorer", "Energizer", "Rim Settler"),
    "elysium": ("Generalist", "Specialist", "Ecologist", "Tycoon", "Legend"),
    "vastitas borealis nova": ("Agronomist", "V. Spacefarer", "Geologist", "Engineer", "Farmer"),
}


def _tag(snap: Mapping[str, Any], tag: str, *, wild: bool = True) -> int:
    tags = snap["me_tags"]
    return int(tags[_TAG_INDEX[tag]]) + (int(tags[_TAG_INDEX["wild"]]) if wild else 0)


def funds_mc(snap: Mapping[str, Any]) -> int:
    """M€ available for an M€ cost: M€ plus heat for Helion (`canUseHeatAsMegaCredits`)."""
    return int(snap["me_mc"]) + (int(snap["me_heat"]) if snap["me_corporation"] == "Helion" else 0)


def plants_for_greenery(snap: Mapping[str, Any]) -> int:
    return 7 if snap["me_corporation"] == "Ecoline" else 8


def award_cost(snap: Mapping[str, Any]) -> int | None:
    """Next award's funding cost, or None when all awards are funded."""
    funded = len([s for s in snap["award_seats"] if int(s) in (0, 1)])
    return AWARD_COSTS[funded] if funded < MAX_AWARDS else None


def _milestone_score_reached(name: str, snap: Mapping[str, Any], specs: Mapping[str, CardSpec]) -> bool | None:
    """True/False when the milestone's score is derivable from the snapshot, None when it is not."""
    prods = [int(snap[f"me_{c}"]) for c in _PROD_COLS.values()]
    tableau = [specs[c] for c in snap["me_tableau"] if c in specs]
    key = name.lower()
    if key == "terraformer":
        return int(snap["me_tr"]) >= 35
    if key == "mayor":
        return int(snap["me_cities"]) >= 3
    if key == "gardener":
        return int(snap["me_greeneries"]) >= 3
    if key == "builder":
        return _tag(snap, "building") >= 8
    if key == "planner":
        return bool(snap["me_hand_exact"]) and len(snap["me_hand"]) >= 16
    if key == "energizer":
        return int(snap["me_energy_prod"]) >= 6
    if key == "rim settler":
        return _tag(snap, "jovian") >= 3
    if key == "generalist":
        return all(p >= 1 for p in prods)
    if key == "specialist":
        return max(prods) >= 10
    if key == "ecologist":
        return _tag(snap, "plant", wild=False) + _tag(snap, "microbe", wild=False) + _tag(snap, "animal") >= 4
    if key == "tycoon":
        return sum(1 for s in tableau if s.card_type in ("active", "automated")) >= 15
    if key == "tactician":
        return sum(1 for s in tableau if s.card_type in ("active", "automated") and s.requirements) >= 5
    if key == "legend":
        return int(snap["me_tags"][_TAG_INDEX["event"]]) >= 5
    return None


def claimable_milestones(snap: Mapping[str, Any], map_name: str, specs: Mapping[str, CardSpec]) -> list[str]:
    """Unclaimed milestones of the map proven claimable from the snapshot (score reached, M€ >= 8)."""
    claimed = {str(n).lower() for n, s in zip(snap["milestone_names"], snap["milestone_seats"]) if int(s) in (0, 1)}
    if len(claimed) >= MAX_MILESTONES or funds_mc(snap) < MILESTONE_COST:
        return []
    return [m for m in MAP_MILESTONES.get(map_name.lower(), ()) if m.lower() not in claimed
            and _milestone_score_reached(m, snap, specs)]


def card_cost(spec: CardSpec, snap: Mapping[str, Any], last_played: str | None) -> int:
    """Engine `getCardCost`: printed cost minus tableau and next-card discounts, floored at 0."""
    cost = int(spec.cost)
    for name in snap["me_tableau"]:
        disc = _CARD_DISCOUNTS.get(name)
        if disc is None:
            continue
        tag, amount, per_card = disc
        if tag is None:
            cost -= amount
        else:
            n = sum(1 for t in spec.tags if t == tag)
            cost -= (amount if n else 0) if per_card else amount * n
    if last_played is not None:
        cost -= _NEXT_CARD_DISCOUNTS.get(last_played, 0)
    return max(cost, 0)


def _requirement_met(req: Mapping[str, Any], snap: Mapping[str, Any], bonus: int) -> bool | None:
    count = int(req.get("count", 1))
    is_max = bool(req.get("max"))
    for key, scale in (("temperature", 2), ("oxygen", 1), ("oceans", 1)):
        if key in req:
            level = int(snap[key])
            target = int(req[key])
            return level <= target + bonus * scale if is_max else level >= target - bonus * scale
    if "tag" in req:
        return _tag(snap, str(req["tag"])) >= count
    if "production" in req:
        col = _PROD_COLS.get(str(req["production"]))
        return None if col is None else int(snap[f"me_{col}"]) >= count
    if "greeneries" in req and not req.get("all"):
        return int(snap["me_greeneries"]) >= count
    if "cities" in req and req.get("all"):
        return int(snap["me_cities"]) + int(snap["opp_cities"]) >= count
    return None


def card_playability(
    spec: CardSpec, snap: Mapping[str, Any], last_played: str | None
) -> tuple[str | None, bool]:
    """`(failure, fully_checked)` for one hand card: failure is None (playable), `requirement`,
    `production` or `unaffordable`; `fully_checked` is False when a requirement kind was not checkable."""
    tableau = set(snap["me_tableau"])
    bonus = sum(2 for c in _REQUIREMENT_BONUS_CARDS if c in tableau)
    if last_played == _NEXT_CARD_REQUIREMENT_BONUS:
        bonus += 2
    checked = True
    for req in spec.requirements:
        met = _requirement_met(req, snap, bonus)
        if met is None:
            checked = False
        elif not met:
            return "requirement", True
    if not spec.has_dynamic_production:
        for res, delta in spec.production.items():
            col = _PROD_COLS.get(res)
            if col is not None and delta < 0 and int(snap[f"me_{col}"]) + delta < _PRODUCTION_MINIMUM.get(res, 0):
                return "production", True
    steel_value = 2 + (1 if "Advanced Alloys" in tableau else 0)
    titanium_value = 3 + (1 if "Advanced Alloys" in tableau else 0) + (1 if "PhoboLog" in tableau else 0)
    budget = funds_mc(snap)
    if "building" in spec.tags:
        budget += int(snap["me_steel"]) * steel_value
    if "space" in spec.tags:
        budget += int(snap["me_titanium"]) * titanium_value
    return (None if budget >= card_cost(spec, snap, last_played) else "unaffordable"), checked


def playable_project_cards(
    snap: Mapping[str, Any], specs: Mapping[str, CardSpec], last_played: str | None
) -> tuple[list[str], int]:
    """Sorted tracked hand cards passing static playability, and how many of them had an unchecked requirement."""
    out: list[str] = []
    unchecked = 0
    for name in sorted(snap["me_hand"]):
        spec = specs.get(name)
        if spec is None:
            continue
        failure, checked = card_playability(spec, snap, last_played)
        if failure is None:
            out.append(name)
            unchecked += 0 if checked else 1
    return out, unchecked


def unused_action_cards(
    snap: Mapping[str, Any], specs: Mapping[str, CardSpec], used: Collection[str]
) -> list[str]:
    """Sorted tableau cards with a blue action not used this generation (standard projects excluded)."""
    return sorted({c for c in snap["me_tableau"] if c not in used and c in specs and specs[c].has_action
                   and specs[c].card_type != "standard_project"})


def standard_project_menu(snap: Mapping[str, Any]) -> list[str]:
    """Affordable standard projects in the engine's cost order (Thorgate pays 3 M€ less for Power Plant)."""
    funds = funds_mc(snap)
    thorgate = "ThorGate" in snap["me_tableau"]
    return [name for name, cost in STANDARD_PROJECTS
            if funds >= cost - (3 if thorgate and name == "Power Plant:SP" else 0)]


def branch_menu(
    snap: Mapping[str, Any],
    *,
    map_name: str,
    specs: Mapping[str, CardSpec],
    used_actions: Collection[str],
    turn_action: int,
    opp_passed: bool,
    last_played: str | None,
) -> list[str]:
    """Legal root branch keys in `BRANCH_ORDER` (requires the exact tracked hand)."""
    legal = {
        "claim_milestone": bool(claimable_milestones(snap, map_name, specs)),
        "convert_plants": int(snap["me_plants"]) >= plants_for_greenery(snap),
        "convert_heat": int(snap["me_heat"]) >= HEAT_FOR_TEMPERATURE,
        "card_action": bool(unused_action_cards(snap, specs, used_actions)),
        "play_project_card": bool(playable_project_cards(snap, specs, last_played)[0]),
        "end_turn": turn_action > 0 and not opp_passed,
        "fund_award": (cost := award_cost(snap)) is not None and funds_mc(snap) >= cost,
        "standard_projects": bool(standard_project_menu(snap)),
        "pass": True,
        "sell_patents": len(snap["me_hand"]) > 0,
    }
    return [k for k in BRANCH_ORDER if legal[k]]
