"""Baseline B0: uniform random agent over legal candidates."""

from __future__ import annotations

import random

from tm_bot.env.types import ActorObservation, Choice

B0_VERSION = "b0-v1"


class B0Agent:
    """Uniform random agent using an isolated seeded `random.Random` instance."""

    version: str = B0_VERSION

    def __init__(self, seed: int = 0) -> None:
        self._seed = int(seed)
        self._rng = random.Random(self._seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = int(seed)
        self._rng = random.Random(self._seed)

    def act(self, observation: ActorObservation, choice: Choice) -> int:
        del observation
        n = len(choice.candidates)
        if n <= 0:
            raise ValueError("Choice has no candidates")
        return self._rng.randrange(n)
