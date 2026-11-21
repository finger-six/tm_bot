"""Card vocabulary and static card feature matrix derived from engine card data."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from tm_bot.agents.card_data import CardSpec, load_card_data

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
DONE_TOKEN = "done"

PAD_ID = 0
UNK_ID = 1
DONE_ID = 2
SPECIAL_TOKENS: tuple[str, ...] = (PAD_TOKEN, UNK_TOKEN, DONE_TOKEN)

TAG_NAMES: tuple[str, ...] = (
    "building",
    "space",
    "science",
    "power",
    "earth",
    "jovian",
    "venus",
    "moon",
    "mars",
    "plant",
    "microbe",
    "animal",
    "crime",
    "city",
    "wild",
    "clone",
    "event",
)
_TAG_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(TAG_NAMES)}

CARD_TYPES: tuple[str, ...] = (
    "corporation",
    "prelude",
    "active",
    "automated",
    "event",
    "standard_project",
    "other",
)
_CARD_TYPE_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(CARD_TYPES)}

RESOURCE_TYPES: tuple[str, ...] = (
    "none",
    "Microbe",
    "Animal",
    "Floater",
    "Science",
    "Fighter",
    "Asteroid",
    "other",
)
_RES_TYPE_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(RESOURCE_TYPES)}

UNIT_KEYS: tuple[str, ...] = (
    "megacredits",
    "steel",
    "titanium",
    "plants",
    "energy",
    "heat",
)

SLOT_NAMES: tuple[str, ...] = (
    "<PAD>",
    "my_hand",
    "my_tableau",
    "opp_tableau",
    "my_preludes_in_hand",
    "dealt_corps",
    "dealt_preludes",
    "dealt_projects",
    "picked_corp",
    "drafted_cards",
    "current_draft_packet",
    "pending_card_selection",
    "history_draft_passed",
    "history_discarded_seen",
)
SLOT_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(SLOT_NAMES)}

PROMPT_TYPES: tuple[str, ...] = (
    "initial_corp",
    "initial_prelude",
    "initial_project_buy",
    "draft_pick",
    "research_buy",
    "prelude_play",
    "project_card",
    "branch",
    "payment",
    "space",
    "player",
    "amount",
    "resource",
    "other_card",
    # Encoder v2 (#42): split out of `project_card` and `other_card` respectively.
    # Appended so v1 ids stay unchanged.
    "standard_project",
    "card_action",
)
PROMPT_TYPE_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(PROMPT_TYPES)}

# 7 card_type + 1 cost + 1 starting_mc + 2 card_cost + 3 vp + 6 prod + 1 dyn_prod
# + 6 stock + 10 effects + 17 tags + 8 resource_type + 8 requirements = 69
CARD_STATIC_DIM = 7 + 1 + 1 + 2 + 3 + 6 + 1 + 6 + 10 + len(TAG_NAMES) + len(RESOURCE_TYPES) + 8


def card_spec_to_features(spec: CardSpec | None) -> np.ndarray:
    """Convert a CardSpec (or None for special tokens) into a fixed float32 vector."""
    vec = np.zeros(CARD_STATIC_DIM, dtype=np.float32)
    if spec is None:
        return vec

    offset = 0
    # Card type (7)
    ct_idx = _CARD_TYPE_TO_IDX.get(spec.card_type, _CARD_TYPE_TO_IDX["other"])
    vec[offset + ct_idx] = 1.0
    offset += len(CARD_TYPES)

    # Cost, starting MC, custom card cost (4)
    vec[offset] = float(spec.cost) / 50.0
    vec[offset + 1] = float(spec.starting_megacredits) / 60.0
    if spec.card_cost is not None:
        vec[offset + 2] = (float(spec.card_cost) - 3.0) / 3.0
        vec[offset + 3] = 1.0
    offset += 4

    # VP (3)
    vec[offset] = float(spec.static_vp) / 10.0
    vec[offset + 1] = 1.0 if spec.has_dynamic_vp else 0.0
    vec[offset + 2] = float(spec.vp_per_resource)
    offset += 3

    # Production (6) + dynamic_prod (1)
    prod_scales = (10.0, 5.0, 5.0, 5.0, 5.0, 5.0)
    for i, (uk, sc) in enumerate(zip(UNIT_KEYS, prod_scales)):
        vec[offset + i] = float(spec.production.get(uk, 0)) / sc
    vec[offset + 6] = 1.0 if spec.has_dynamic_production else 0.0
    offset += 7

    # Stock (6)
    stock_scales = (20.0, 10.0, 10.0, 10.0, 10.0, 10.0)
    for i, (uk, sc) in enumerate(zip(UNIT_KEYS, stock_scales)):
        vec[offset + i] = float(spec.stock.get(uk, 0)) / sc
    offset += 6

    # Global / action / discount effects (10)
    vec[offset + 0] = float(spec.tr) / 5.0
    vec[offset + 1] = float(spec.temperature_steps) / 3.0
    vec[offset + 2] = float(spec.oxygen_steps) / 3.0
    vec[offset + 3] = float(spec.oceans) / 3.0
    vec[offset + 4] = float(spec.greenery) / 2.0
    vec[offset + 5] = float(spec.city) / 2.0
    vec[offset + 6] = float(spec.draw_card) / 3.0
    vec[offset + 7] = 1.0 if spec.has_action else 0.0
    vec[offset + 8] = float(spec.discount_amount) / 5.0
    vec[offset + 9] = float(spec.requirements_count) / 3.0
    offset += 10

    # Tags (17)
    for tag in spec.tags:
        t_idx = _TAG_TO_IDX.get(tag)
        if t_idx is not None:
            vec[offset + t_idx] = 1.0
    offset += len(TAG_NAMES)

    # Resource type (8)
    if spec.resource_type is None:
        vec[offset + _RES_TYPE_TO_IDX["none"]] = 1.0
    else:
        r_idx = _RES_TYPE_TO_IDX.get(spec.resource_type, _RES_TYPE_TO_IDX["other"])
        vec[offset + r_idx] = 1.0
    offset += len(RESOURCE_TYPES)

    # Requirements summary (8):
    # [has_min_temp, has_max_temp, has_min_oxy, has_max_oxy, has_min_oceans, has_max_oceans, has_tag_req, has_other_req]
    for req in spec.requirements:
        is_max = bool(req.get("max", False))
        if "temperature" in req:
            vec[offset + (1 if is_max else 0)] = 1.0
        elif "oxygen" in req:
            vec[offset + (3 if is_max else 2)] = 1.0
        elif "oceans" in req:
            vec[offset + (5 if is_max else 4)] = 1.0
        elif "tag" in req:
            vec[offset + 6] = 1.0
        else:
            vec[offset + 7] = 1.0
    offset += 8

    assert offset == CARD_STATIC_DIM
    return vec


class CardVocab:
    """Deterministic card-name vocabulary and static feature matrix."""

    def __init__(
        self,
        names: tuple[str, ...],
        static_features: np.ndarray,
    ) -> None:
        if len(names) != static_features.shape[0]:
            raise ValueError(
                f"names length ({len(names)}) != static_features rows ({static_features.shape[0]})"
            )
        if static_features.shape[1] != CARD_STATIC_DIM:
            raise ValueError(
                f"Expected CARD_STATIC_DIM={CARD_STATIC_DIM}, got {static_features.shape[1]}"
            )
        self.names: tuple[str, ...] = tuple(names)
        self.name_to_id: dict[str, int] = {name: i for i, name in enumerate(self.names)}
        self.static_features: np.ndarray = np.ascontiguousarray(static_features, dtype=np.float32)

    @property
    def size(self) -> int:
        return len(self.names)

    def __len__(self) -> int:
        return len(self.names)

    def card_id(self, name: str | None) -> int:
        if not name:
            return PAD_ID
        return self.name_to_id.get(name, UNK_ID)

    @classmethod
    def from_card_data(cls, cards: Mapping[str, CardSpec] | None = None) -> CardVocab:
        specs = cards if cards is not None else load_card_data()
        sorted_names = sorted(name for name in specs.keys() if name not in SPECIAL_TOKENS)
        names = SPECIAL_TOKENS + tuple(sorted_names)
        feats = np.zeros((len(names), CARD_STATIC_DIM), dtype=np.float32)
        for idx, name in enumerate(names):
            if idx < len(SPECIAL_TOKENS):
                continue
            feats[idx] = card_spec_to_features(specs.get(name))
        return cls(names=names, static_features=feats)

    def to_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "static_features": self.static_features.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CardVocab:
        names = tuple(str(x) for x in payload["names"])
        feats = np.asarray(payload["static_features"], dtype=np.float32)
        return cls(names=names, static_features=feats)
