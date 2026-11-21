"""AlphaZero-style learner plus pre-update calibration and policy-entropy probes.

The v2 learner loss (#39 §1c-1d, §3; #49) on replay-buffer decisions is

    CE(policy, visits) + w_value·BCE(win_logit, o) + w_margin·SmoothL1(vp_margin_pred, m)
    + λ_h·L_human + β·KL(π_θ ‖ π_anchor)

`win_logit` trains on the terminal outcome o alone (target (o + 1) / 2 ∈ {0, 0.5, 1}),
`vp_margin_pred` on the terminal margin m. L_human is the #38 multi-task warm-start loss
on a resident human sample (`tm_bot/training/human_aux.py`), weighted by λ_h on the
training clock; the KL uses the anchor logits recorded with each decision
(`tm_bot/models/anchor_loss.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tm_bot.models.anchor_loss import kl_to_anchor, pad_anchor_logits
from tm_bot.models.network import TMNetwork, vp_margin_loss
from tm_bot.runs.buffer import DecisionSample, ReplayBuffer, collate_samples
from tm_bot.runs.config import RunConfig


@dataclass(slots=True)
class LearnerStats:
    """Aggregated learner metrics for one cycle (training-batch statistics only)."""

    updates: int = 0
    policy_loss: float | None = None
    value_loss: float | None = None
    total_loss: float | None = None
    nonfinite_loss: float | None = None
    nonfinite_params: bool = False
    per_step_policy: list[float] = field(default_factory=list)
    per_step_value: list[float] = field(default_factory=list)
    # v2 terms (#49): unweighted cycle means; None when the term never applied.
    margin_loss: float | None = None
    kl_anchor: float | None = None
    human_aux_loss: float | None = None
    human_aux_weight: float | None = None
    per_step: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "updates": self.updates,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "margin_loss": self.margin_loss,
            "kl_anchor": self.kl_anchor,
            "human_aux_loss": self.human_aux_loss,
            "human_aux_weight": self.human_aux_weight,
            "total_loss": self.total_loss,
            "nonfinite_loss": self.nonfinite_loss,
            "nonfinite_params": self.nonfinite_params,
        }


class HumanAuxSource(Protocol):
    """L_human provider (`tm_bot.training.human_aux.HumanAuxLoss`)."""

    def backward(
        self, model: torch.nn.Module, *, weight: float, device: torch.device | str, use_amp: bool
    ) -> dict[str, float]: ...


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@dataclass(frozen=True, slots=True)
class LossWeights:
    """Weights of the self-play terms of the v2 learner loss (#39 §3)."""

    policy: float = 1.0
    value: float = 1.0
    margin: float = 0.3
    kl_anchor: float = 0.05

    @classmethod
    def from_config(cls, config: RunConfig) -> LossWeights:
        # A random-init run is pure self-play: no warm-start anchor terms (#39 §4).
        return cls(
            policy=config.w_policy,
            value=config.w_value,
            margin=config.w_margin,
            kl_anchor=0.0 if config.random_init else config.kl_anchor_beta,
        )


def selfplay_targets(samples: Sequence[DecisionSample]) -> dict[str, Any]:
    """Collated model batch plus every self-play target of `samples`.

    Keys: `batch` (model inputs), `policy` (visit distribution), `outcome` (o ∈ {+1, 0, -1}),
    `margin` (m), `anchor_logits` and `has_anchor` (see `pad_anchor_logits`).
    """
    batch, policy, outcome = collate_samples(samples)
    anchor, has_anchor = pad_anchor_logits([s.anchor_logits for s in samples], policy.shape[1])
    return {
        "batch": batch,
        "policy": policy,
        "outcome": outcome,
        "margin": torch.tensor([float(s.margin_target) for s in samples], dtype=torch.float32),
        "anchor_logits": anchor,
        "has_anchor": has_anchor,
    }


def selfplay_losses(
    model: TMNetwork,
    targets: Mapping[str, Any],
    weights: LossWeights = LossWeights(),
) -> tuple[torch.Tensor, dict[str, float | None]]:
    """Weighted self-play part of the v2 loss, plus each unweighted term.

    Policy: masked cross-entropy against the visit distribution. Value: BCE of
    `win_logit` against (o + 1) / 2, outcome only. Margin: SmoothL1 (β 0.2) of
    `vp_margin_pred` against m (skipped for a model without the head). KL: mean
    KL(π_θ ‖ π_anchor) over rows with recorded anchor logits (skipped when its weight is
    0 or no row has them). Skipped terms report None.
    """
    batch = targets["batch"]
    out = model(batch, compute_policy=True, compute_value=True)
    logits = out["policy_logits"].float()
    mask = batch["cand_mask"].bool()
    log_probs = F.log_softmax(logits, dim=-1)
    safe_log_probs = torch.where(mask, log_probs, torch.zeros_like(log_probs))
    policy_loss = -(targets["policy"].float() * safe_log_probs).sum(dim=-1).mean()

    win_targets = (targets["outcome"].float() + 1.0) * 0.5
    value_loss = F.binary_cross_entropy_with_logits(out["value_logit"].float(), win_targets)
    total = weights.policy * policy_loss + weights.value * value_loss
    parts: dict[str, float | None] = {
        "policy_loss": float(policy_loss.detach().item()),
        "value_loss": float(value_loss.detach().item()),
        "margin_loss": None,
        "kl_anchor": None,
    }
    if "vp_margin_pred" in out:
        margin_loss = vp_margin_loss(out["vp_margin_pred"], targets["margin"])
        total = total + weights.margin * margin_loss
        parts["margin_loss"] = float(margin_loss.detach().item())
    if weights.kl_anchor > 0.0:
        kl = kl_to_anchor(logits, targets["anchor_logits"], mask, targets["has_anchor"])
        if kl is not None:
            total = total + weights.kl_anchor * kl
            parts["kl_anchor"] = float(kl.detach().item())
    return total, parts


@torch.inference_mode()
def compute_value_calibration(
    model: TMNetwork,
    samples: Sequence[DecisionSample],
    *,
    device: str | torch.device = "cpu",
    num_bins: int = 10,
    chunk_size: int = 256,
) -> dict[str, Any]:
    """Measure value-head calibration on held-out decisions with frozen weights.

    Call this before the cycle's samples enter the replay buffer and before any
    optimizer step, so the numbers are out-of-sample for the updated network.
    Returns sample counts, outcome counts, Brier score, ECE, and reliability bins.
    """
    empty: dict[str, Any] = {
        "n": 0,
        "wins": 0,
        "losses": 0,
        "ties": 0,
        "brier": None,
        "ece": None,
        "bins": [],
    }
    if not samples:
        return empty
    dev = torch.device(device)
    model.eval()
    probs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for start in range(0, len(samples), chunk_size):
        chunk = samples[start : start + chunk_size]
        batch, _target_probs, values = collate_samples(chunk)
        out = model(_move(batch, dev), compute_policy=False, compute_value=True)
        probs.append(torch.sigmoid(out["value_logit"].float()).detach().cpu().numpy())
        targets.append((values.numpy().astype(np.float64) + 1.0) * 0.5)
    win_prob = np.concatenate(probs)
    win_targets = np.concatenate(targets)
    value_arr = win_targets * 2.0 - 1.0

    brier = float(np.mean((win_prob - win_targets) ** 2))
    bins: list[dict[str, Any]] = []
    ece = 0.0
    n = int(win_prob.size)
    for i in range(num_bins):
        lo = i / num_bins
        hi = (i + 1) / num_bins
        if i == num_bins - 1:
            mask = (win_prob >= lo) & (win_prob <= hi)
        else:
            mask = (win_prob >= lo) & (win_prob < hi)
        count = int(mask.sum())
        if count == 0:
            bins.append({"lo": lo, "hi": hi, "count": 0, "mean_pred": None, "mean_target": None})
            continue
        mean_pred = float(win_prob[mask].mean())
        mean_target = float(win_targets[mask].mean())
        ece += (count / n) * abs(mean_pred - mean_target)
        bins.append(
            {
                "lo": lo,
                "hi": hi,
                "count": count,
                "mean_pred": mean_pred,
                "mean_target": mean_target,
            }
        )
    return {
        "n": n,
        "wins": int(np.sum(value_arr > 0.0)),
        "losses": int(np.sum(value_arr < 0.0)),
        "ties": int(np.sum(value_arr == 0.0)),
        "brier": brier,
        "ece": float(ece),
        "bins": bins,
    }


@torch.inference_mode()
def probe_policy_entropy(
    model: TMNetwork,
    samples: Sequence[DecisionSample],
    *,
    device: str | torch.device = "cpu",
) -> float | None:
    """Mean policy entropy of `model` on a fixed observation set."""
    if not samples:
        return None
    batch, _targets, _values = collate_samples(samples)
    dev = torch.device(device)
    model.eval()
    out = model(_move(batch, dev), compute_policy=True, compute_value=False)
    probs = out["policy_probs"].float()
    mask = batch["cand_mask"].bool().to(dev)
    ent = -torch.where(mask, probs * torch.log(probs.clamp_min(1e-12)), torch.zeros_like(probs))
    return float(ent.sum(dim=-1).mean().item())


def params_are_finite(model: TMNetwork) -> bool:
    """True when every parameter tensor is finite."""
    return all(bool(torch.isfinite(p).all().item()) for p in model.parameters())


class AlphaZeroLearner:
    """Samples decision rows from the replay buffer and applies gradient updates."""

    def __init__(
        self,
        config: RunConfig,
        model: TMNetwork,
        *,
        device: str | torch.device = "cpu",
        optimizer: torch.optim.Optimizer | None = None,
        human_aux: HumanAuxSource | None = None,
        training_clock: Callable[[], float] | None = None,
    ) -> None:
        """`human_aux` supplies L_human (None: no human-data term); `training_clock`
        returns the run's training seconds for λ_h (the orchestrator wires
        `SelfPlayRunner.training_s`; None: hour 0)."""
        self.config = config
        self.model = model
        self.device = torch.device(device)
        self.weights = LossWeights.from_config(config)
        self.human_aux = human_aux
        self.training_clock = training_clock
        self.human_aux_schedule = config.human_aux_schedule
        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.rng = np.random.default_rng(config.seed)
        self.step = 0

    def train(
        self,
        buffer: ReplayBuffer,
        *,
        updates: int | None = None,
        training_s: float | None = None,
    ) -> LearnerStats:
        """Run `updates` gradient steps; returns aggregated training-batch losses.

        λ_h is read once per call, at `training_s` (default: `training_clock()`).
        A non-finite loss or any non-finite parameter stops updates immediately
        and is reported so the orchestrator can roll back to the last checkpoint.
        """
        n_updates = self.config.updates_per_cycle if updates is None else updates
        stats = LearnerStats()
        if n_updates <= 0 or len(buffer) == 0:
            return stats
        if training_s is None:
            training_s = self.training_clock() if self.training_clock is not None else 0.0
        human_weight = self.human_aux_schedule.at_seconds(training_s)
        amp_enabled = bool(self.config.use_amp and self.device.type == "cuda")
        amp_dtype = torch.bfloat16 if amp_enabled else torch.float32
        self.model.train()
        for _ in range(n_updates):
            samples = buffer.sample_batch(self.config.learner_batch_size, rng=self.rng)
            targets = selfplay_targets(samples)
            targets = {
                k: _move(v, self.device) if k == "batch" else v.to(self.device)
                for k, v in targets.items()
            }
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=self.device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                loss, parts = selfplay_losses(self.model, targets, self.weights)
            loss_value = float(loss.detach().item())
            if not math.isfinite(loss_value):
                stats.nonfinite_loss = loss_value
                break
            loss.backward()
            step: dict[str, float] = {k: v for k, v in parts.items() if v is not None}
            if self.human_aux is not None and human_weight > 0.0:
                human = self.human_aux.backward(
                    self.model, weight=human_weight, device=self.device, use_amp=amp_enabled
                )
                human_loss = float(human["loss_total"])
                if not math.isfinite(human_loss):
                    stats.nonfinite_loss = human_loss
                    break
                step["human_aux_loss"] = human_loss
                loss_value += human_weight * human_loss
            step["total_loss"] = loss_value
            if self.config.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            if not params_are_finite(self.model):
                stats.nonfinite_params = True
                break
            self.step += 1
            stats.updates += 1
            stats.per_step.append(step)
            stats.per_step_policy.append(step["policy_loss"])
            stats.per_step_value.append(step["value_loss"])

        if stats.updates:
            stats.policy_loss = float(np.mean(stats.per_step_policy))
            stats.value_loss = float(np.mean(stats.per_step_value))
            stats.margin_loss = _mean_of(stats.per_step, "margin_loss")
            stats.kl_anchor = _mean_of(stats.per_step, "kl_anchor")
            stats.human_aux_loss = _mean_of(stats.per_step, "human_aux_loss")
            if stats.human_aux_loss is not None:
                stats.human_aux_weight = human_weight
            stats.total_loss = _mean_of(stats.per_step, "total_loss")
        self.model.eval()
        return stats


def _mean_of(rows: Sequence[Mapping[str, float]], key: str) -> float | None:
    values = [row[key] for row in rows if key in row]
    return float(np.mean(values)) if values else None


def select_entropy_probe(
    samples: Iterable[DecisionSample],
    *,
    limit: int = 128,
) -> list[DecisionSample]:
    """Deterministically pick a fixed spread of samples for the entropy probe."""
    items = list(samples)
    if not items:
        return []
    if len(items) <= limit:
        return items
    stride = max(1, len(items) // limit)
    picked = items[::stride][:limit]
    return picked
