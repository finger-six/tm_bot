"""Shared v2 self-play rules (#39, #48): the play temperature τ(g) and the utility U.

Both agents use these, so the main agent's search, the comparator, and the self-play
engines import them from here:

- `PlayTemperatureSchedule` is τ(g): 1.0 through generation 4, linear to 0.3 at
  generation 10, 0.3 after, never 0. It applies to action-phase decisions only; every
  learned seat (learner, `latest`, pool, warm-start anchor, dev and eval agents) samples
  at τ(g) there.
- `game_utility` is U = o + 0.15·m with o ∈ {+1, 0, -1} and
  m = clip(my_vp - opp_vp, -60, 60) / 50, so any win outranks any loss.
- `search_utility` is the search leaf (2p - 1) + 0.15·m̂ from the value and margin heads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from tm_bot.models.network import VP_MARGIN_BOUND, vp_margin_target

SELFPLAY_RULES_VERSION = "v2"
MARGIN_UTILITY_WEIGHT = 0.15


@dataclass(frozen=True, slots=True)
class PlayTemperatureSchedule:
    """τ(g): `start` through `decay_from`, linear to `floor` at `floor_at`, `floor` after."""

    start: float = 1.0
    floor: float = 0.3
    decay_from: int = 4
    floor_at: int = 10

    def __post_init__(self) -> None:
        if self.floor <= 0.0:
            raise ValueError(f"play temperature floor must be > 0, got {self.floor}")
        if self.start < self.floor:
            raise ValueError(
                f"play temperature start {self.start} must be >= floor {self.floor}"
            )
        if self.decay_from >= self.floor_at:
            raise ValueError(
                f"decay_from ({self.decay_from}) must be < floor_at ({self.floor_at})"
            )

    def __call__(self, generation: int) -> float:
        g = int(generation)
        if g <= self.decay_from:
            return float(self.start)
        if g >= self.floor_at:
            return float(self.floor)
        frac = (g - self.decay_from) / (self.floor_at - self.decay_from)
        return float(self.start + (self.floor - self.start) * frac)

    def for_observation(self, observation: Mapping[str, Any]) -> float | None:
        """τ(g) for an action-phase observation; None outside the action phase."""
        public = observation.get("public", {}) or {}
        if str(public.get("phase", "")) != "action":
            return None
        return self(int(public.get("generation", 1) or 1))

    @property
    def spec(self) -> str:
        """Compact id recorded in agent versions, e.g. `tau1g4-0.3g10`."""
        return f"tau{self.start:g}g{self.decay_from}-{self.floor:g}g{self.floor_at}"


V2_PLAY_TEMPERATURE = PlayTemperatureSchedule()


def _check_weight(weight: float) -> None:
    # A win's U must stay above any loss's: 1 - w·1.2 > -1 + w·1.2.
    if not (0.0 <= weight and weight * VP_MARGIN_BOUND < 1.0):
        raise ValueError(
            f"margin weight must be in [0, {1.0 / VP_MARGIN_BOUND:g}), got {weight}"
        )


def game_utility(
    outcome: float, my_vp: float, opp_vp: float, *, weight: float = MARGIN_UTILITY_WEIGHT
) -> float:
    """U = o + weight·m for a finished game, from the seat whose VP is `my_vp`."""
    _check_weight(weight)
    return float(outcome) + weight * float(vp_margin_target(my_vp, opp_vp))


def search_utility(
    value: float, margin_pred: float | None, *, weight: float = MARGIN_UTILITY_WEIGHT
) -> float:
    """Search leaf utility (2p - 1) + weight·m̂; m̂ counts as 0 without a margin head."""
    _check_weight(weight)
    return float(value) + weight * (0.0 if margin_pred is None else float(margin_pred))
