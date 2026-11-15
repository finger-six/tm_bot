"""Jev-inspired comparator agent: Deep Sets encoder + typed-candidate PPO scorer."""

from __future__ import annotations

from tm_bot.comparator.agent import ComparatorAgent, make_comparator_agent
from tm_bot.comparator.model import (
    ComparatorConfig,
    ComparatorNetwork,
    load_comparator_checkpoint,
    save_comparator_checkpoint,
)

__all__ = [
    "ComparatorAgent",
    "ComparatorConfig",
    "ComparatorNetwork",
    "load_comparator_checkpoint",
    "make_comparator_agent",
    "save_comparator_checkpoint",
]
