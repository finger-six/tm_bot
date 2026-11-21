"""Replay buffer of recent self-play decisions (AlphaZero-style targets)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from tm_bot.models.encode import (
    EncodedChoice,
    EncodedObservation,
    collate_encoded_batch,
)


@dataclass(frozen=True, slots=True)
class DecisionSample:
    """One training row: acting-seat observation, choice, visit policy, utility target.

    v2 self-play fields (#48), consumed by the learners (#49 main, #50 comparator):

    - `play_temperature`: the τ this seat sampled the decision with (τ(g) in the
      action phase, 0.0 for an argmax decision, None when the agent did not report
      one, e.g. forced single-candidate decisions).
    - `margin_target`: m = clip(my_vp - opp_vp, -60, 60) / 50 for this seat at game
      end (`vp_margin_target`), the margin head's target; U = o + 0.15·m.
    - `anchor_logits`: float32[num_candidates] raw (τ = 1) policy logits of the frozen
      warm-start anchor on this same observation and choice; index i is candidate i,
      the same order as `policy_target` (and `action`). None when the run has no
      anchor.
    """

    observation: EncodedObservation
    choice: EncodedChoice
    policy_target: np.ndarray  # float32[C], sums to 1 over legal candidates
    value_target: float  # +1 win, -1 loss, 0 tie (acting seat's perspective)
    seat: int
    was_searched: bool
    seed_key: str
    map_name: str
    generation: int
    play_temperature: float | None = None
    margin_target: float = 0.0
    anchor_logits: np.ndarray | None = None

    @property
    def num_candidates(self) -> int:
        return int(self.policy_target.shape[0])


def normalize_policy_target(raw: Sequence[float], num_candidates: int) -> np.ndarray:
    """Validate and renormalize a candidate visit distribution."""
    arr = np.asarray(raw[:num_candidates], dtype=np.float32)
    if arr.shape[0] != num_candidates:
        raise ValueError(
            f"policy target has {arr.shape[0]} entries for {num_candidates} candidates"
        )
    arr = np.where(np.isfinite(arr) & (arr > 0.0), arr, 0.0)
    total = float(arr.sum())
    if total <= 0.0:
        return np.full((num_candidates,), 1.0 / max(1, num_candidates), dtype=np.float32)
    return (arr / total).astype(np.float32)


class ReplayBuffer:
    """Fixed-capacity ring buffer keeping the most recent decisions."""

    def __init__(self, capacity: int = 200_000) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.capacity = int(capacity)
        self._items: deque[DecisionSample] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._items)

    @property
    def samples(self) -> tuple[DecisionSample, ...]:
        return tuple(self._items)

    def add(self, sample: DecisionSample) -> None:
        if sample.num_candidates < 1:
            raise ValueError("decision sample has no candidates")
        self._items.append(sample)

    def extend(self, samples: Sequence[DecisionSample]) -> None:
        for sample in samples:
            self.add(sample)

    def sample_batch(
        self,
        batch_size: int,
        *,
        rng: np.random.Generator | None = None,
    ) -> list[DecisionSample]:
        """Sample decisions uniformly with replacement (deterministic given `rng`)."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if not self._items:
            raise ValueError("cannot sample from an empty replay buffer")
        generator = rng or np.random.default_rng(0)
        indices = generator.integers(0, len(self._items), size=batch_size)
        items = list(self._items)
        return [items[int(i)] for i in indices]

    def state_dict(self) -> dict[str, list]:
        """Serialize recent samples for run resume (numpy arrays stay arrays)."""
        return {
            "observation": [s.observation for s in self._items],
            "choice": [s.choice for s in self._items],
            "policy_target": [s.policy_target for s in self._items],
            "value_target": [s.value_target for s in self._items],
            "seat": [s.seat for s in self._items],
            "was_searched": [s.was_searched for s in self._items],
            "seed_key": [s.seed_key for s in self._items],
            "map_name": [s.map_name for s in self._items],
            "generation": [s.generation for s in self._items],
            "play_temperature": [s.play_temperature for s in self._items],
            "margin_target": [s.margin_target for s in self._items],
            "anchor_logits": [s.anchor_logits for s in self._items],
        }

    def load_state_dict(self, payload: dict[str, list]) -> None:
        self._items.clear()
        n = len(payload["value_target"])
        play_temperature = payload.get("play_temperature", [None] * n)
        margin_target = payload.get("margin_target", [0.0] * n)
        anchor_logits = payload.get("anchor_logits", [None] * n)
        for i in range(n):
            self._items.append(
                DecisionSample(
                    observation=payload["observation"][i],
                    choice=payload["choice"][i],
                    policy_target=np.asarray(payload["policy_target"][i], dtype=np.float32),
                    value_target=float(payload["value_target"][i]),
                    seat=int(payload["seat"][i]),
                    was_searched=bool(payload["was_searched"][i]),
                    seed_key=str(payload["seed_key"][i]),
                    map_name=str(payload["map_name"][i]),
                    generation=int(payload["generation"][i]),
                    play_temperature=_optional_float(play_temperature[i]),
                    margin_target=float(margin_target[i]),
                    anchor_logits=_optional_logits(anchor_logits[i]),
                )
            )


def _optional_float(value: float | None) -> float | None:
    return None if value is None else float(value)


def _optional_logits(value: np.ndarray | None) -> np.ndarray | None:
    return None if value is None else np.asarray(value, dtype=np.float32)


def collate_samples(
    samples: Sequence[DecisionSample],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Collate decision samples into `(model_batch, policy_targets, value_targets)`."""
    if not samples:
        raise ValueError("cannot collate an empty sample list")
    batch = collate_encoded_batch(
        [s.observation for s in samples],
        [s.choice for s in samples],
    )
    max_c = max(s.num_candidates for s in samples)
    targets = np.zeros((len(samples), max_c), dtype=np.float32)
    values = np.zeros((len(samples),), dtype=np.float32)
    mask = np.zeros((len(samples), max_c), dtype=bool)
    for i, sample in enumerate(samples):
        n = sample.num_candidates
        targets[i, :n] = sample.policy_target
        values[i] = float(sample.value_target)
        mask[i, :n] = True
    cand_mask = batch["cand_mask"]
    # Any candidate beyond a sample's own list is padding and must be masked.
    batch_mask = cand_mask.numpy() & mask
    batch["cand_mask"] = torch.from_numpy(batch_mask)
    return batch, torch.from_numpy(targets), torch.from_numpy(values)
