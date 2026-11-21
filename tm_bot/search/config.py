"""Configuration and decision classification for determinized IS-MCTS search."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from tm_bot.agents.play_rules import (
    MARGIN_UTILITY_WEIGHT,
    V2_PLAY_TEMPERATURE,
    PlayTemperatureSchedule,
)
from tm_bot.env.types import ActorObservation, Choice

# #39 search budget for self-play, `dev-*` and `eval-*`: 16 simulations over K = 4 worlds,
# with a fallback of 8 simulations (never below 8 / 4).
V2_NUM_SIMULATIONS = 16
FALLBACK_NUM_SIMULATIONS = 8
V2_NUM_WORLDS = 4

_ACTION_ROOT_PROMPTS = frozenset(
    {"take your first action", "take your next action"}
)


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """Hyperparameters controlling determinized IS-MCTS search.

    Defaults are the v2 self-play rules (#48): 16 simulations x K = 4, τ(g) at
    action-phase decisions (`play_temperature`), and leaf/terminal utility
    U = o + `margin_weight`·m. `temperature` is the non-action root temperature
    (0 = most-visited). `action_temperature` is the v1 flat action-phase temperature,
    used only when `play_temperature` is None.
    """

    num_simulations: int = V2_NUM_SIMULATIONS
    num_worlds: int = V2_NUM_WORLDS
    c_puct: float = 1.5
    # First-play urgency: Q assigned to unvisited children, already in the perspective of the
    # player selecting at that node (visited Q values are sign-adjusted the same way).
    fpu_value: float = 0.0
    temperature: float = 0.0
    action_temperature: float = 0.0
    wave_size: int | None = None
    max_nested_steps: int = 32
    play_temperature: PlayTemperatureSchedule | None = V2_PLAY_TEMPERATURE
    margin_weight: float = MARGIN_UTILITY_WEIGHT

    def __post_init__(self) -> None:
        if self.num_simulations < 0:
            raise ValueError(f"num_simulations must be >= 0, got {self.num_simulations}")
        if self.num_worlds < 1:
            raise ValueError(f"num_worlds must be >= 1, got {self.num_worlds}")
        if self.c_puct < 0.0:
            raise ValueError(f"c_puct must be >= 0, got {self.c_puct}")
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.action_temperature < 0.0:
            raise ValueError(
                f"action_temperature must be >= 0, got {self.action_temperature}"
            )
        if self.wave_size is not None and self.wave_size < 1:
            raise ValueError(f"wave_size must be >= 1, got {self.wave_size}")
        if self.max_nested_steps < 1:
            raise ValueError(f"max_nested_steps must be >= 1, got {self.max_nested_steps}")
        if not (0.0 <= self.margin_weight < 1.0 / 1.2):
            raise ValueError(f"margin_weight must be in [0, 1/1.2), got {self.margin_weight}")

    @classmethod
    def v2(cls, *, fallback: bool = False, **overrides: Any) -> SearchConfig:
        """The #39 self-play/dev/eval search: 16 x 4, or 8 x 4 with `fallback`."""
        sims = FALLBACK_NUM_SIMULATIONS if fallback else V2_NUM_SIMULATIONS
        return replace(cls(num_simulations=sims, num_worlds=V2_NUM_WORLDS), **overrides)

    def decision_temperature(self, observation: ActorObservation, default: float) -> float:
        """Sampling temperature for one decision: τ(g) in the action phase, else `default`.

        Without a schedule the v1 flat `action_temperature` applies in the action phase
        when it is > 0.
        """
        if str(observation.get("public", {}).get("phase", "")) == "action":
            if self.play_temperature is not None:
                tau = self.play_temperature.for_observation(observation)
                assert tau is not None
                return tau
            if self.action_temperature > 0.0:
                return self.action_temperature
        return default


def is_search_decision(observation: ActorObservation, choice: Choice) -> bool:
    """Return True for multi-candidate action-root, draft, and buy decisions.

    Nested sub-choices (`payment`, `space`, `player`, `amount`, nested `card`,
    nested `branch`, `initial_corp`, `initial_prelude`, and single-candidate
    forced choices) return False so they use the neural policy head directly.
    """
    if len(choice.candidates) <= 1:
        return False

    phase = str(observation.get("public", {}).get("phase", ""))
    prompt_l = choice.prompt.strip().lower()

    if phase == "action":
        return choice.kind == "branch" and prompt_l in _ACTION_ROOT_PROMPTS
    if phase in ("drafting", "initial_drafting"):
        return choice.kind == "card"
    if phase == "research":
        return choice.kind == "multi_card" and "buy" in prompt_l
    return False
