"""Configuration and decision classification for determinized IS-MCTS search."""

from __future__ import annotations

from dataclasses import dataclass

from tm_bot.env.types import ActorObservation, Choice

_ACTION_ROOT_PROMPTS = frozenset(
    {"take your first action", "take your next action"}
)


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """Hyperparameters controlling determinized IS-MCTS search."""

    num_simulations: int = 32
    num_worlds: int = 8
    c_puct: float = 1.5
    # First-play urgency: Q assigned to unvisited children, already in the perspective of the
    # player selecting at that node (visited Q values are sign-adjusted the same way).
    fpu_value: float = 0.0
    temperature: float = 0.0
    action_temperature: float = 0.0
    wave_size: int | None = None
    max_nested_steps: int = 32

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
