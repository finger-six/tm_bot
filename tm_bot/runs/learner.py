"""AlphaZero-style learner plus pre-update calibration and policy-entropy probes."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tm_bot.models.network import TMNetwork
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "updates": self.updates,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "total_loss": self.total_loss,
            "nonfinite_loss": self.nonfinite_loss,
            "nonfinite_params": self.nonfinite_params,
        }


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def policy_value_losses(
    model: TMNetwork,
    batch: dict[str, torch.Tensor],
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
    *,
    w_policy: float = 1.0,
    w_value: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute masked policy cross-entropy plus value BCE from acting-seat rows."""
    out = model(batch, compute_policy=True, compute_value=True)
    log_probs = F.log_softmax(out["policy_logits"].float(), dim=-1)
    mask = batch["cand_mask"].bool()
    safe_log_probs = torch.where(mask, log_probs, torch.zeros_like(log_probs))
    policy_loss = -(policy_targets.float() * safe_log_probs).sum(dim=-1)
    policy_loss = policy_loss.mean()

    value_logit = out["value_logit"].float()
    win_targets = (value_targets.float() + 1.0) * 0.5
    value_loss = F.binary_cross_entropy_with_logits(value_logit, win_targets)
    total = w_policy * policy_loss + w_value * value_loss
    return total, {
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
    }


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
    ) -> None:
        self.config = config
        self.model = model
        self.device = torch.device(device)
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
    ) -> LearnerStats:
        """Run `updates` gradient steps; returns aggregated training-batch losses.

        A non-finite loss or any non-finite parameter stops updates immediately
        and is reported so the orchestrator can roll back to the last checkpoint.
        """
        n_updates = self.config.updates_per_cycle if updates is None else updates
        stats = LearnerStats()
        if n_updates <= 0 or len(buffer) == 0:
            return stats
        amp_enabled = bool(self.config.use_amp and self.device.type == "cuda")
        amp_dtype = torch.bfloat16 if amp_enabled else torch.float32
        self.model.train()
        for _ in range(n_updates):
            samples = buffer.sample_batch(self.config.learner_batch_size, rng=self.rng)
            batch, targets, values = collate_samples(samples)
            batch = _move(batch, self.device)
            targets = targets.to(self.device)
            values = values.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=self.device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                loss, parts = policy_value_losses(
                    self.model,
                    batch,
                    targets,
                    values,
                    w_policy=self.config.w_policy,
                    w_value=self.config.w_value,
                )
            loss_value = float(loss.detach().item())
            if not math.isfinite(loss_value):
                stats.nonfinite_loss = loss_value
                break
            loss.backward()
            if self.config.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            if not params_are_finite(self.model):
                stats.nonfinite_params = True
                break
            self.step += 1
            stats.updates += 1
            stats.per_step_policy.append(float(parts["policy_loss"].item()))
            stats.per_step_value.append(float(parts["value_loss"].item()))

        if stats.updates:
            stats.policy_loss = float(np.mean(stats.per_step_policy))
            stats.value_loss = float(np.mean(stats.per_step_value))
            stats.total_loss = float(
                self.config.w_policy * stats.policy_loss + self.config.w_value * stats.value_loss
            )
        self.model.eval()
        return stats


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
