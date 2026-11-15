"""AlphaZero-style learner updating the shared policy/value network from the replay buffer."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tm_bot.models.network import TMNetwork
from tm_bot.runs.buffer import ReplayBuffer, collate_samples
from tm_bot.runs.config import RunConfig


@dataclass(slots=True)
class LearnerStats:
    """Aggregated learner metrics for one cycle."""

    updates: int = 0
    policy_loss: float | None = None
    value_loss: float | None = None
    total_loss: float | None = None
    policy_entropy: float | None = None
    value_brier: float | None = None
    value_accuracy: float | None = None
    nonfinite_loss: float | None = None
    per_step_policy: list[float] = field(default_factory=list)
    per_step_value: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "updates": self.updates,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "total_loss": self.total_loss,
            "policy_entropy": self.policy_entropy,
            "value_brier": self.value_brier,
            "value_accuracy": self.value_accuracy,
            "nonfinite_loss": self.nonfinite_loss,
        }


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

    probs = out["policy_probs"].float()
    entropy = -(torch.where(mask, probs * torch.log(probs.clamp_min(1e-12)), torch.zeros_like(probs)))
    entropy = entropy.sum(dim=-1).mean()
    win_prob = torch.sigmoid(value_logit)
    brier = ((win_prob - win_targets) ** 2).mean()
    non_tie = (value_targets != 0.0).float()
    pred = (win_prob >= 0.5).float()
    correct = ((pred == (value_targets > 0.0)).float() * non_tie).sum()
    accuracy = correct / non_tie.sum().clamp_min(1.0)
    return total, {
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "entropy": entropy.detach(),
        "brier": brier.detach(),
        "accuracy": accuracy.detach(),
    }


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

    def _move(self, batch: dict[str, torch.Tensor], targets: Any, values: Any):
        moved = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        return moved, targets.to(self.device), values.to(self.device)

    def train(
        self,
        buffer: ReplayBuffer,
        *,
        updates: int | None = None,
    ) -> LearnerStats:
        """Run `updates` gradient steps; returns aggregated losses.

        A non-finite loss stops updating immediately and is reported in
        `LearnerStats.nonfinite_loss` so the orchestrator can roll back.
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
            batch, targets, values = self._move(batch, targets, values)
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
            # Entropy/calibration are measured on the last batch.
            stats.policy_entropy = float(parts["entropy"].item())
            stats.value_brier = float(parts["brier"].item())
            stats.value_accuracy = float(parts["accuracy"].item())
        self.model.eval()
        return stats
