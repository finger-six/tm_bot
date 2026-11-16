"""Baseline B1: frozen, documented deterministic greedy heuristic agent."""

from __future__ import annotations

import copy
import random
from typing import Any

from tm_bot.agents.card_data import CardSpec, load_card_data
from tm_bot.env.types import ActorObservation, Choice

B1_VERSION = "b1-v2"

# Engine SpaceBonus enum values (see tools/simbench/engine/src/common/boards/SpaceBonus.ts)
_BONUS_TITANIUM = 0
_BONUS_STEEL = 1
_BONUS_PLANT = 2
_BONUS_DRAW_CARD = 3
_BONUS_HEAT = 4
_BONUS_OCEAN = 5
_BONUS_MEGACREDITS = 6
_BONUS_TEMPERATURE = 13
_BONUS_TEMPERATURE_4MC = 18

# Engine TileType enum values (see tools/simbench/engine/src/common/TileType.ts)
_TILE_GREENERY = 0
_TILE_OCEAN = 1
_TILE_CITY = 2
_TILE_CAPITAL = 3

_CITY_TILES = {_TILE_CITY, _TILE_CAPITAL}
_GREENERY_TILES = {_TILE_GREENERY}
_OCEAN_TILES = {_TILE_OCEAN}

_BONUS_VALUES: dict[int, float] = {
    _BONUS_TITANIUM: 3.0,
    _BONUS_STEEL: 2.0,
    _BONUS_PLANT: 2.5,
    _BONUS_DRAW_CARD: 3.5,
    _BONUS_HEAT: 1.2,
    _BONUS_OCEAN: 6.0,
    _BONUS_MEGACREDITS: 1.0,
    _BONUS_TEMPERATURE: 5.0,
    _BONUS_TEMPERATURE_4MC: 5.0,
}

_RESOURCE_GAIN_ORDER: tuple[str, ...] = (
    "titanium",
    "plants",
    "steel",
    "megacredits",
    "heat",
    "energy",
)

_RESOURCE_LOSE_ORDER: tuple[str, ...] = (
    "energy",
    "heat",
    "megacredits",
    "plants",
    "steel",
    "titanium",
)


def _remaining_steps(obs: ActorObservation) -> int:
    gp = obs["public"]["globalParameters"]
    temp_steps = max(0, (8 - int(gp["temperature"])) // 2)
    oxy_steps = max(0, 14 - int(gp["oxygenLevel"]))
    ocean_steps = max(0, 9 - int(gp["oceans"]))
    return temp_steps + oxy_steps + ocean_steps


def _adjacent_coords(x: int, y: int, max_y: int) -> tuple[tuple[int, int], ...]:
    middle_row = max_y / 2.0
    if y < middle_row:
        return (
            (x, y - 1),
            (x + 1, y - 1),
            (x + 1, y),
            (x, y + 1),
            (x - 1, y + 1),
            (x - 1, y),
        )
    if y == middle_row:
        return (
            (x, y - 1),
            (x + 1, y - 1),
            (x + 1, y),
            (x + 1, y + 1),
            (x, y + 1),
            (x - 1, y),
        )
    return (
        (x - 1, y - 1),
        (x, y - 1),
        (x + 1, y),
        (x + 1, y + 1),
        (x, y + 1),
        (x - 1, y),
    )


class B1Agent:
    """Deterministic greedy heuristic baseline over `(ActorObservation, Choice)`."""

    version: str = B1_VERSION

    def __init__(self, seed: int = 0, card_data: dict[str, CardSpec] | None = None) -> None:
        self._seed = int(seed)
        self._rng = random.Random(self._seed)
        self._cards: dict[str, CardSpec] = card_data if card_data is not None else load_card_data()
        self._pending_corp_name: str | None = None

    def __deepcopy__(self, memo: dict[int, Any]) -> B1Agent:
        # CardSpec dictionary is immutable public rules knowledge; share across clones.
        cls = self.__class__
        clone = cls.__new__(cls)
        memo[id(self)] = clone
        clone._seed = self._seed
        clone._rng = copy.deepcopy(self._rng, memo)
        clone._cards = self._cards
        clone._pending_corp_name = self._pending_corp_name
        return clone

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = int(seed)
        self._rng = random.Random(self._seed)
        self._pending_corp_name = None

    def act(self, observation: ActorObservation, choice: Choice) -> int:
        n = len(choice.candidates)
        if n <= 0:
            raise ValueError("Choice has no candidates")
        if n == 1:
            return 0

        kind = choice.kind
        if kind == "payment":
            return 0
        if kind == "branch":
            return self._choose_branch(observation, choice)
        if kind in ("card", "multi_card"):
            return self._choose_card(observation, choice)
        if kind == "project_card":
            return self._choose_project_card(observation, choice)
        if kind == "space":
            return self._choose_space(observation, choice)
        if kind == "player":
            return self._choose_player(observation, choice)
        if kind == "amount":
            return self._choose_amount(observation, choice)
        if kind == "resource":
            return self._choose_resource(observation, choice)
        return 0

    # ------------------------------------------------------------------
    # Card & state valuation helpers
    # ------------------------------------------------------------------

    def _me_and_opp(self, obs: ActorObservation) -> tuple[dict[str, Any], dict[str, Any]]:
        actor = obs["actor"]
        players = obs["public"]["players"]
        return players[actor], players[1 - actor]

    def _card_static_value(self, spec: CardSpec, obs: ActorObservation) -> float:
        rem_steps = _remaining_steps(obs)
        rem_frac = rem_steps / 42.0
        prod_horizon = 0.8 + 6.2 * rem_frac
        vp_value = 5.0 + 3.0 * (1.0 - rem_frac)

        me, _ = self._me_and_opp(obs)
        steel_val = float(me["steelValue"])
        ti_val = float(me["titaniumValue"])

        gp = obs["public"]["globalParameters"]
        temp_left = max(0, (8 - int(gp["temperature"])) // 2)
        oxy_left = max(0, 14 - int(gp["oxygenLevel"]))
        ocean_left = max(0, 9 - int(gp["oceans"]))

        # Effective TR raises (capped by remaining track steps)
        effective_tr = (
            min(spec.temperature_steps, temp_left)
            + min(spec.oxygen_steps, oxy_left)
            + min(spec.oceans, ocean_left)
        )
        # Cards with direct TR (not from global params)
        direct_tr = max(0, spec.tr - (spec.temperature_steps + spec.oxygen_steps + spec.oceans + spec.greenery))
        effective_tr += direct_tr
        if spec.greenery > 0 and spec.oxygen_steps == 0:
            effective_tr += min(spec.greenery, oxy_left)

        tr_unit_val = vp_value + 1.0 * prod_horizon
        score = effective_tr * tr_unit_val

        # Greenery tile itself scores 1 VP + adjacency potential
        if spec.greenery > 0:
            score += spec.greenery * (vp_value + 2.0)

        # City tile scores ~2 VP + placement potential
        if spec.city > 0:
            score += spec.city * (2.0 * vp_value + 2.0)

        # Production box
        prod = spec.production
        score += prod.get("megacredits", 0) * 1.0 * prod_horizon
        score += prod.get("steel", 0) * steel_val * 0.9 * prod_horizon
        score += prod.get("titanium", 0) * ti_val * 0.9 * prod_horizon
        score += prod.get("plants", 0) * 1.8 * prod_horizon
        score += prod.get("energy", 0) * 1.3 * prod_horizon
        score += prod.get("heat", 0) * 1.0 * prod_horizon

        if spec.has_dynamic_production:
            score += 4.5 * rem_frac + 1.5

        # Immediate stock
        stock = spec.stock
        score += stock.get("megacredits", 0) * 1.0
        score += stock.get("steel", 0) * steel_val * 0.9
        score += stock.get("titanium", 0) * ti_val * 0.9
        score += stock.get("plants", 0) * 1.8
        score += stock.get("energy", 0) * 0.8
        score += stock.get("heat", 0) * 1.0

        # Victory points
        score += spec.static_vp * vp_value
        if spec.vp_per_resource > 0:
            score += spec.vp_per_resource * (3.5 if spec.has_action else 1.5) * vp_value
        elif spec.has_dynamic_vp:
            score += 1.5 * vp_value

        # Card draw, ongoing actions, discounts, and tags
        score += spec.draw_card * 3.5
        if spec.has_action:
            score += 5.5 * (0.3 + 0.7 * rem_frac)
        if spec.discount_amount > 0:
            score += spec.discount_amount * 4.5 * (0.3 + 0.7 * rem_frac)
        score += len(spec.tags) * 0.6
        return score

    def _max_budget_for_card(self, spec: CardSpec | None, obs: ActorObservation) -> int:
        me, _ = self._me_and_opp(obs)
        budget = int(me["megacredits"])
        if obs["private"]["pickedCorporationCard"] == "Helion":
            budget += int(me["heat"])
        if spec is not None:
            if "building" in spec.tags:
                budget += int(me["steel"]) * int(me["steelValue"])
            if "space" in spec.tags:
                budget += int(me["titanium"]) * int(me["titaniumValue"])
        return budget

    def _satisfies_requirements_approx(self, spec: CardSpec, obs: ActorObservation) -> bool:
        if not spec.requirements:
            return True
        gp = obs["public"]["globalParameters"]
        temp = int(gp["temperature"])
        oxy = int(gp["oxygenLevel"])
        oceans = int(gp["oceans"])
        me, opp = self._me_and_opp(obs)
        tags = me["tags"]

        # Account for potential +/-2 requirement relaxation (e.g. Adaptation Technology / Inventrix)
        for req in spec.requirements:
            is_max = bool(req.get("max", False))
            if "temperature" in req:
                target = int(req["temperature"])
                if is_max and temp > target + 4:
                    return False
                if not is_max and temp < target - 4:
                    return False
            elif "oxygen" in req:
                target = int(req["oxygen"])
                if is_max and oxy > target + 2:
                    return False
                if not is_max and oxy < target - 2:
                    return False
            elif "oceans" in req:
                target = int(req["oceans"])
                if is_max and oceans > target + 2:
                    return False
                if not is_max and oceans < target - 2:
                    return False
            elif "tag" in req:
                tag_name = str(req["tag"])
                needed = int(req.get("count", 1) or 1)
                if int(tags.get(tag_name, 0)) < needed:
                    return False
            elif "tr" in req:
                needed = int(req.get("count", req["tr"]))
                if int(me["terraformRating"]) < needed:
                    return False
            elif "greeneries" in req:
                needed = int(req.get("count", req["greeneries"]))
                all_players = bool(req.get("all", False))
                count = sum(
                    1
                    for sp in obs["public"]["spaces"]
                    if sp["tileType"] in _GREENERY_TILES and (all_players or sp["owner"] == obs["actor"])
                )
                if count < needed:
                    return False
            elif "cities" in req:
                needed = int(req.get("count", req["cities"]))
                all_players = bool(req.get("all", False))
                count = int(me["citiesCount"]) + (int(opp["citiesCount"]) if all_players else 0)
                if count < needed:
                    return False
        return True

    def _hand_card_net_value(self, card_name: str, calculated_cost: int, obs: ActorObservation) -> float:
        spec = self._cards.get(card_name)
        if spec is None:
            return -float(calculated_cost)
        return self._card_static_value(spec, obs) - float(calculated_cost)

    def _best_hand_candidate(self, obs: ActorObservation) -> float | None:
        best: float | None = None
        for item in obs["private"]["cardsInHand"]:
            name = item["name"]
            cost = int(item["calculatedCost"])
            spec = self._cards.get(name)
            if cost > self._max_budget_for_card(spec, obs):
                continue
            if spec is not None and not self._satisfies_requirements_approx(spec, obs):
                continue
            net = self._hand_card_net_value(name, cost, obs)
            if best is None or net > best:
                best = net
        return best

    # ------------------------------------------------------------------
    # Branch choices
    # ------------------------------------------------------------------

    def _award_funding_choice(self, obs: ActorObservation) -> tuple[int, float] | None:
        me, opp = self._me_and_opp(obs)
        funded_count = sum(1 for a in obs["public"]["awards"] if a["fundedBy"] is not None)
        cost = 8 if funded_count == 0 else (14 if funded_count == 1 else 20)
        if int(me["megacredits"]) < cost:
            return None

        rem_steps = _remaining_steps(obs)
        actor = obs["actor"]
        best_idx: int | None = None
        best_lead = -10**9
        best_score = 0

        unfunded_idx = 0
        for award in obs["public"]["awards"]:
            if award["fundedBy"] is not None:
                continue
            scores = {entry["seat"]: int(entry["score"]) for entry in award["scores"]}
            my_s = scores.get(actor, 0)
            opp_s = scores.get(1 - actor, 0)
            lead = my_s - opp_s
            if lead > best_lead or (lead == best_lead and my_s > best_score):
                best_lead = lead
                best_score = my_s
                best_idx = unfunded_idx
            unfunded_idx += 1

        if best_idx is None or best_score <= 0:
            return None

        if funded_count == 0:
            worth = best_lead >= 2 or (best_lead >= 1 and best_score >= 3 and rem_steps <= 28)
        elif funded_count == 1:
            worth = best_lead >= 2 and best_score >= 3 and rem_steps <= 22
        else:
            worth = best_lead >= 3 and best_score >= 4 and rem_steps <= 16

        if not worth:
            return None
        return best_idx, float(best_lead)

    def _standard_project_worthy(self, obs: ActorObservation) -> bool:
        me, _ = self._me_and_opp(obs)
        mc = int(me["megacredits"])
        gp = obs["public"]["globalParameters"]
        temp = int(gp["temperature"])
        oxy = int(gp["oxygenLevel"])
        oceans = int(gp["oceans"])
        rem_steps = _remaining_steps(obs)
        generation = int(obs["public"]["generation"])

        if mc >= 23 and (oxy < 14 or rem_steps <= 14 or int(me["citiesCount"]) >= 1):
            return True
        if mc >= 18 and oceans < 9 and (rem_steps <= 26 or mc >= 24):
            return True
        if mc >= 14 and temp < 8 and (rem_steps <= 22 or mc >= 26 or generation >= 4):
            return True
        if mc >= 25 and (rem_steps <= 14 or mc >= 34):
            return True
        return False

    def _choose_branch(self, obs: ActorObservation, choice: Choice) -> int:
        prompt_l = choice.prompt.lower()
        candidates = choice.candidates

        # Sub-branch: Fund an award
        if "fund an award" in prompt_l:
            award_pick = self._award_funding_choice(obs)
            if award_pick is not None and 0 <= award_pick[0] < len(candidates):
                return award_pick[0]
            return 0

        # Sub-branch: Claim a milestone -> pick first/highest
        if "claim a milestone" in prompt_l:
            return 0

        # Root action menu
        by_type: dict[str, int] = {}
        for idx, cand in enumerate(candidates):
            lbl = cand.label.lower()
            if lbl.startswith("take first action of"):
                by_type.setdefault("first_action", idx)
            elif "claim a milestone" in lbl:
                by_type.setdefault("milestone", idx)
            elif "convert" in lbl and "greenery" in lbl:
                by_type.setdefault("greenery", idx)
            elif "convert" in lbl and "heat" in lbl:
                by_type.setdefault("heat", idx)
            elif "play project card" in lbl:
                by_type.setdefault("play_card", idx)
            elif "perform an action" in lbl:
                by_type.setdefault("card_action", idx)
            elif "fund an award" in lbl:
                by_type.setdefault("award", idx)
            elif "standard projects" in lbl:
                by_type.setdefault("std_proj", idx)
            elif "pass for this generation" in lbl:
                by_type.setdefault("pass", idx)
            elif "end turn" in lbl:
                by_type.setdefault("end_turn", idx)

        # A corporation's mandatory first action blocks every other root action
        # until taken; passing would repeat each generation.
        if "first_action" in by_type:
            return by_type["first_action"]

        if "milestone" in by_type:
            return by_type["milestone"]

        if "greenery" in by_type:
            return by_type["greenery"]

        gp = obs["public"]["globalParameters"]
        if "heat" in by_type and int(gp["temperature"]) < 8:
            return by_type["heat"]

        rem_steps = _remaining_steps(obs)
        play_threshold = -2.0 if rem_steps <= 10 else -0.5

        if "play_card" in by_type:
            best_hand = self._best_hand_candidate(obs)
            if best_hand is None or best_hand >= play_threshold:
                return by_type["play_card"]

        if "card_action" in by_type:
            return by_type["card_action"]

        if "award" in by_type and self._award_funding_choice(obs) is not None:
            return by_type["award"]

        if "std_proj" in by_type and self._standard_project_worthy(obs):
            return by_type["std_proj"]

        if "pass" in by_type:
            return by_type["pass"]

        # Non-root OrOptions: avoid End Turn / Sell patents if something else exists
        for idx, cand in enumerate(candidates):
            lbl = cand.label.lower()
            if "pass" not in lbl and "end turn" not in lbl and "sell patents" not in lbl:
                return idx
        return 0

    # ------------------------------------------------------------------
    # Card & multi_card choices
    # ------------------------------------------------------------------

    def _choose_card(self, obs: ActorObservation, choice: Choice) -> int:
        prompt_l = choice.prompt.lower()
        phase = obs["public"]["phase"]
        candidates = choice.candidates
        me, _ = self._me_and_opp(obs)

        done_idx = next((i for i, c in enumerate(candidates) if c.id == "done"), None)
        card_indices = [i for i, c in enumerate(candidates) if c.id != "done"]
        if not card_indices:
            return done_idx if done_idx is not None else 0

        # Sell patents: avoid selling unless forced
        if "sell patents" in prompt_l:
            if done_idx is not None:
                return done_idx
            return self._min_value_card_index(obs, candidates, card_indices)

        # Discard prompt: dump lowest-value card
        if "discard" in prompt_l:
            return self._min_value_card_index(obs, candidates, card_indices)

        # Corporation selection
        if "corporation" in prompt_l:
            dealt_tags: set[str] = set()
            for name in obs["private"]["dealtProjectCards"]:
                sp = self._cards.get(name)
                if sp is not None:
                    dealt_tags.update(sp.tags)

            best_idx = card_indices[0]
            best_score = -10**9
            for idx in card_indices:
                name = candidates[idx].id
                spec = self._cards.get(name)
                if spec is None:
                    score = 0.0
                else:
                    card_cost = spec.card_cost if spec.card_cost is not None else 3
                    tag_syn = sum(1.0 for t in spec.tags if t in dealt_tags)
                    score = (
                        float(spec.starting_megacredits)
                        + self._card_static_value(spec, obs)
                        - (card_cost - 3) * 4.0
                        + 1.5 * tag_syn
                    )
                if score > best_score:
                    best_score = score
                    best_idx = idx
            self._pending_corp_name = candidates[best_idx].id
            return best_idx

        # Prelude selection (both research pick-2 and preludes play phase)
        if "prelude" in prompt_l or phase == "preludes":
            best_idx = card_indices[0]
            best_score = -10**9
            for idx in card_indices:
                spec = self._cards.get(candidates[idx].id)
                score = (
                    self._card_static_value(spec, obs) + float(spec.starting_megacredits)
                    if spec is not None
                    else 0.0
                )
                if score > best_score:
                    best_score = score
                    best_idx = idx
            return best_idx

        # Buying cards (initial buy or research buy or look-and-buy)
        if "buy" in prompt_l:
            mc = int(me["megacredits"])
            card_cost = int(me["cardCost"])
            is_initial = "initial" in prompt_l
            if is_initial:
                corp_name = obs["private"]["pickedCorporationCard"] or self._pending_corp_name
                corp_spec = self._cards.get(corp_name) if corp_name else None
                if corp_spec is not None:
                    mc = max(mc, corp_spec.starting_megacredits)
                    if corp_spec.card_cost is not None:
                        card_cost = corp_spec.card_cost
                elif mc == 0:
                    mc = 35

            already_picked = 0
            if done_idx is not None:
                lbl = candidates[done_idx].label
                if "(" in lbl and "selected" in lbl:
                    try:
                        already_picked = int(lbl.split("(", 1)[1].split()[0])
                    except (ValueError, IndexError):
                        already_picked = 0

            remaining_mc = mc - already_picked * card_cost
            rem_steps = _remaining_steps(obs)
            # In late game, don't buy cards unless very high immediate value
            buy_threshold = float(card_cost) + (2.0 if rem_steps > 14 else 4.0)
            max_buy_budget = remaining_mc - (12 if is_initial else 0)

            if done_idx is not None:
                if is_initial and (already_picked >= 5 or max_buy_budget < card_cost):
                    return done_idx
                if not is_initial and remaining_mc < card_cost:
                    return done_idx

            best_idx = card_indices[0]
            best_score = -10**9
            for idx in card_indices:
                spec = self._cards.get(candidates[idx].id)
                if spec is None:
                    score = -100.0
                else:
                    # Prefer affordable or reasonably playable cards
                    max_playable = (
                        remaining_mc
                        + int(me["steel"]) * int(me["steelValue"])
                        + int(me["titanium"]) * int(me["titaniumValue"])
                        + max(8, 3 * int(me["megacreditProduction"]) + 10)
                    )
                    cost_penalty = 0.0 if spec.cost <= max_playable else 12.0
                    score = self._card_static_value(spec, obs) - float(spec.cost) - cost_penalty
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if done_idx is not None and best_score < buy_threshold:
                return done_idx
            return best_idx

        # Drafting
        if phase in ("drafting", "initial_drafting") or "pass the rest" in prompt_l:
            best_idx = card_indices[0]
            best_score = -10**9
            for idx in card_indices:
                spec = self._cards.get(candidates[idx].id)
                score = (
                    self._card_static_value(spec, obs) - 0.85 * float(spec.cost)
                    if spec is not None
                    else 0.0
                )
                if score > best_score:
                    best_score = score
                    best_idx = idx
            return best_idx

        # Blue card action selection ("Perform an action from a played card")
        if "action from a played card" in prompt_l:
            best_idx = card_indices[0]
            best_score = -10**9
            for idx in card_indices:
                spec = self._cards.get(candidates[idx].id)
                if spec is None:
                    score = 0.0
                else:
                    score = 1.0
                    if spec.tr > 0 or spec.oxygen_steps > 0 or spec.temperature_steps > 0 or spec.oceans > 0:
                        score += 5.0
                    if spec.vp_per_resource > 0:
                        score += 4.0 * spec.vp_per_resource
                    if spec.draw_card > 0:
                        score += 3.0
                if score > best_score:
                    best_score = score
                    best_idx = idx
            return best_idx

        # Generic card selection (e.g. "Select N card(s) to keep") -> highest net value
        best_idx = card_indices[0]
        best_score = -10**9
        for idx in card_indices:
            spec = self._cards.get(candidates[idx].id)
            score = (
                self._card_static_value(spec, obs) - 0.8 * float(spec.cost)
                if spec is not None
                else 0.0
            )
            if score > best_score:
                best_score = score
                best_idx = idx
        if done_idx is not None and best_score < 0.0:
            return done_idx
        return best_idx

    def _min_value_card_index(
        self,
        obs: ActorObservation,
        candidates: tuple[Any, ...],
        card_indices: list[int],
    ) -> int:
        worst_idx = card_indices[0]
        worst_score = 10**9
        for idx in card_indices:
            spec = self._cards.get(candidates[idx].id)
            score = (
                self._card_static_value(spec, obs) - float(spec.cost)
                if spec is not None
                else 0.0
            )
            if score < worst_score:
                worst_score = score
                worst_idx = idx
        return worst_idx

    # ------------------------------------------------------------------
    # Project card / Standard project choice
    # ------------------------------------------------------------------

    def _choose_project_card(self, obs: ActorObservation, choice: Choice) -> int:
        prompt_l = choice.prompt.lower()
        candidates = choice.candidates
        hand_costs = {c["name"]: int(c["calculatedCost"]) for c in obs["private"]["cardsInHand"]}
        gp = obs["public"]["globalParameters"]
        rem_steps = _remaining_steps(obs)

        if "standard projects" in prompt_l:
            best_idx = 0
            best_score = -10**9
            for idx, cand in enumerate(candidates):
                cid = cand.id
                if cid == "Greenery":
                    oxy_bonus = 10.0 if int(gp["oxygenLevel"]) < 14 else 0.0
                    score = 10.0 + oxy_bonus - 23.0
                elif cid == "Aquifer":
                    score = (13.5 if int(gp["oceans"]) < 9 else -20.0) - 18.0
                elif cid == "Asteroid:SP":
                    score = (11.5 if int(gp["temperature"]) < 8 else -20.0) - 14.0
                elif cid == "City":
                    late_bonus = 4.0 if rem_steps <= 14 else 0.0
                    score = 12.0 + late_bonus - 25.0
                elif cid == "Power Plant:SP":
                    score = -12.0
                else:
                    score = -20.0
                if score > best_score:
                    best_score = score
                    best_idx = idx
            return best_idx

        best_idx = 0
        best_score = -10**9
        for idx, cand in enumerate(candidates):
            name = cand.id
            spec = self._cards.get(name)
            cost = hand_costs.get(name, spec.cost if spec is not None else 0)
            score = (
                self._card_static_value(spec, obs) - float(cost)
                if spec is not None
                else -float(cost)
            )
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    # ------------------------------------------------------------------
    # Space placement scoring
    # ------------------------------------------------------------------

    def _choose_space(self, obs: ActorObservation, choice: Choice) -> int:
        spaces = obs["public"]["spaces"]
        if not spaces:
            return 0

        space_by_id = {sp["id"]: sp for sp in spaces}
        max_y = max(int(sp["y"]) for sp in spaces)
        by_coord = {
            (int(sp["x"]), int(sp["y"])): sp
            for sp in spaces
            if sp["spaceType"] != "colony"
        }

        prompt_l = choice.prompt.lower()
        actor = obs["actor"]
        is_city = "city" in prompt_l or "capital" in prompt_l
        is_greenery = "greenery" in prompt_l
        is_ocean = "ocean" in prompt_l

        best_idx = 0
        best_score = -10**9

        for idx, cand in enumerate(choice.candidates):
            sp = space_by_id.get(cand.id)
            if sp is None:
                continue
            score = sum(_BONUS_VALUES.get(int(b), 1.0) for b in sp["bonus"])
            neighbors = [
                by_coord[coord]
                for coord in _adjacent_coords(int(sp["x"]), int(sp["y"]), max_y)
                if coord in by_coord
            ]

            for nb in neighbors:
                tile = nb["tileType"]
                owner = nb["owner"]
                if is_city:
                    if tile in _GREENERY_TILES:
                        score += 4.5
                    elif tile in _OCEAN_TILES:
                        score += 2.0
                    elif tile is None and nb["spaceType"] == "land":
                        score += 0.8
                elif is_greenery:
                    if tile in _CITY_TILES:
                        score += 4.5 if owner == actor else -3.5
                    elif tile in _OCEAN_TILES:
                        score += 2.0
                    elif tile is None and nb["spaceType"] == "land":
                        score += 0.3
                elif is_ocean:
                    if tile in _CITY_TILES:
                        score += 2.0 if owner == actor else -1.5
                    elif owner == actor:
                        score += 1.0
                else:
                    if tile in _OCEAN_TILES:
                        score += 2.0
                    elif tile in _CITY_TILES and owner == actor:
                        score += 0.5

            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    # ------------------------------------------------------------------
    # Player, amount, resource prompts
    # ------------------------------------------------------------------

    def _choose_player(self, obs: ActorObservation, choice: Choice) -> int:
        me, _ = self._me_and_opp(obs)
        my_color = me["color"]
        for idx, cand in enumerate(choice.candidates):
            if cand.id != my_color and cand.id != "neutral":
                return idx
        for idx, cand in enumerate(choice.candidates):
            if cand.id != my_color:
                return idx
        return 0

    def _choose_amount(self, obs: ActorObservation, choice: Choice) -> int:
        prompt_l = choice.prompt.lower()
        if "heat production to decrease" in prompt_l:
            gp = obs["public"]["globalParameters"]
            if int(gp["temperature"]) >= 0 or _remaining_steps(obs) <= 14:
                return len(choice.candidates) - 1
            return 0
        if "lose" in prompt_l or "discard" in prompt_l:
            return 0
        return len(choice.candidates) - 1

    def _choose_resource(self, obs: ActorObservation, choice: Choice) -> int:
        prompt_l = choice.prompt.lower()
        order = _RESOURCE_LOSE_ORDER if "lose" in prompt_l else _RESOURCE_GAIN_ORDER
        rank = {name: i for i, name in enumerate(order)}
        best_idx = 0
        best_rank = 10**9
        for idx, cand in enumerate(choice.candidates):
            r = rank.get(cand.id, 999)
            if r < best_rank:
                best_rank = r
                best_idx = idx
        return best_idx
