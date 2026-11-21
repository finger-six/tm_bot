"""Warm-start retention terms shared by both v2 learners (#39 §1d; #49 main, #50 comparator).

Each learner update adds two terms that tie the policy to the frozen v2 warm start:

- `λ_h · L_human`: the full #38 multi-task warm-start loss on a human-data minibatch.
  `HumanAuxSchedule` is λ_h over the run's training clock (wall clock minus in-run dev
  evaluation, `SelfPlayRunner.training_s`): 0.5 for hours 0-2, linear to 0.1 at hour 8,
  0.1 after. The resident human sample and the loss itself live in `human_aux.py`.
- `β · KL(π_θ ‖ π_anchor)` on the learner's own recorded self-play decisions, from raw
  (τ = 1) logits, β = 0.05. The anchor logits are recorded with each decision
  (`DecisionSample.anchor_logits`, `PPOSample.anchor_logits`); `pad_anchor_logits` lays
  them out like the collated policy logits and `kl_to_anchor` computes the KL.

This module is import-light (torch and numpy only, outside the heavy `tm_bot.training`
package) so `RunConfig` can read its defaults without an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

HUMAN_AUX_WEIGHT_START = 0.5
HUMAN_AUX_WEIGHT_END = 0.1
HUMAN_AUX_DECAY_FROM_H = 2.0
HUMAN_AUX_DECAY_TO_H = 8.0
HUMAN_AUX_MAX_ROWS = 2_000_000
KL_ANCHOR_BETA = 0.05

_MASKED_LOGIT = -1e9


@dataclass(frozen=True, slots=True)
class HumanAuxSchedule:
    """λ_h(t): `start` through hour `decay_from_h`, linear to `end` at `decay_to_h`, `end` after."""

    start: float = HUMAN_AUX_WEIGHT_START
    end: float = HUMAN_AUX_WEIGHT_END
    decay_from_h: float = HUMAN_AUX_DECAY_FROM_H
    decay_to_h: float = HUMAN_AUX_DECAY_TO_H

    def __post_init__(self) -> None:
        if self.start < 0.0 or self.end < 0.0:
            raise ValueError(f"λ_h must be >= 0, got start={self.start}, end={self.end}")
        if not (0.0 <= self.decay_from_h < self.decay_to_h):
            raise ValueError(
                f"need 0 <= decay_from_h ({self.decay_from_h}) < decay_to_h ({self.decay_to_h})"
            )

    def __call__(self, training_hours: float) -> float:
        h = float(training_hours)
        if h <= self.decay_from_h:
            return float(self.start)
        if h >= self.decay_to_h:
            return float(self.end)
        frac = (h - self.decay_from_h) / (self.decay_to_h - self.decay_from_h)
        return float(self.start + (self.end - self.start) * frac)

    def at_seconds(self, training_s: float) -> float:
        """λ_h at `training_s` seconds of the training clock."""
        return self(float(training_s) / 3600.0)


def pad_anchor_logits(
    anchor_logits: Sequence[np.ndarray | None], width: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(logits[B, width], has_anchor[B])` from per-decision anchor logits.

    Row b holds decision b's float32[n_b] logits in its first n_b slots (candidate order,
    as recorded) and zeros after; rows recorded without an anchor are all zero with
    `has_anchor[b] = False`.
    """
    out = np.zeros((len(anchor_logits), int(width)), dtype=np.float32)
    has = np.zeros((len(anchor_logits),), dtype=bool)
    for b, row in enumerate(anchor_logits):
        if row is None:
            continue
        arr = np.asarray(row, dtype=np.float32).reshape(-1)
        if arr.shape[0] > width:
            raise ValueError(f"anchor logits have {arr.shape[0]} entries for width {width}")
        out[b, : arr.shape[0]] = arr
        has[b] = True
    return torch.from_numpy(out), torch.from_numpy(has)


def kl_to_anchor(
    logits: torch.Tensor,
    anchor_logits: torch.Tensor,
    cand_mask: torch.Tensor,
    has_anchor: torch.Tensor,
) -> torch.Tensor | None:
    """Mean over anchored rows of KL(π_θ ‖ π_anchor) over each row's legal candidates.

    Both distributions are the softmax of raw (τ = 1) logits restricted to `cand_mask`.
    Gradient flows through `logits` only. Returns None when no row has anchor logits.
    """
    rows = has_anchor.bool()
    if not bool(rows.any()):
        return None
    mask = cand_mask.bool()[rows]
    log_p = F.log_softmax(logits.float()[rows].masked_fill(~mask, _MASKED_LOGIT), dim=-1)
    log_q = F.log_softmax(
        anchor_logits.float().to(logits.device)[rows].masked_fill(~mask, _MASKED_LOGIT), dim=-1
    )
    per_cand = torch.where(mask, log_p.exp() * (log_p - log_q), torch.zeros_like(log_p))
    return per_cand.sum(dim=-1).mean()
