"""PPO learner for the comparator: clipped surrogate, value critic, and GAE."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tm_bot.comparator.model import ComparatorNetwork
from tm_bot.models.encode import EncodedChoice, EncodedObservation
from tm_bot.runs.buffer import collate_samples


@dataclass(frozen=True, slots=True)
class PPOSample:
    """One recorded comparator decision with behavior log-probability and GAE targets."""

    observation: EncodedObservation
    choice: EncodedChoice
    policy_target: np.ndarray  # behavior distribution over legal candidates
    value_target: float  # terminal utility from the acting seat's perspective
    seat: int
    was_searched: bool
    seed_key: str
    map_name: str
    generation: int
    action: int
    old_log_prob: float
    behavior_value: float  # raw win probability at collection time
    advantage: float = 0.0
    return_: float = 0.0

    @property
    def num_candidates(self) -> int:
        return int(self.policy_target.shape[0])


class PPOBuffer:
    """Fixed-capacity ring buffer of comparator PPO decisions."""

    def __init__(self, capacity: int = 200_000) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.capacity = int(capacity)
        self._items: deque[PPOSample] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._items)

    @property
    def samples(self) -> tuple[PPOSample, ...]:
        return tuple(self._items)

    def add(self, sample: PPOSample) -> None:
        if sample.num_candidates < 1:
            raise ValueError("PPO sample has no candidates")
        self._items.append(sample)

    def extend(self, samples: Sequence[PPOSample]) -> None:
        for sample in samples:
            self.add(sample)

    def state_dict(self) -> dict[str, list]:
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
            "action": [s.action for s in self._items],
            "old_log_prob": [s.old_log_prob for s in self._items],
            "behavior_value": [s.behavior_value for s in self._items],
            "advantage": [s.advantage for s in self._items],
            "return_": [s.return_ for s in self._items],
        }

    def load_state_dict(self, payload: dict[str, list]) -> None:
        self._items.clear()
        for i in range(len(payload["action"])):
            self._items.append(
                PPOSample(
                    observation=payload["observation"][i],
                    choice=payload["choice"][i],
                    policy_target=np.asarray(payload["policy_target"][i], dtype=np.float32),
                    value_target=float(payload["value_target"][i]),
                    seat=int(payload["seat"][i]),
                    was_searched=bool(payload["was_searched"][i]),
                    seed_key=str(payload["seed_key"][i]),
                    map_name=str(payload["map_name"][i]),
                    generation=int(payload["generation"][i]),
                    action=int(payload["action"][i]),
                    old_log_prob=float(payload["old_log_prob"][i]),
                    behavior_value=float(payload["behavior_value"][i]),
                    advantage=float(payload["advantage"][i]),
                    return_=float(payload["return_"][i]),
                )
            )


def collate_ppo_samples(
    samples: Sequence[PPOSample],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Collate PPO samples into `(model_batch, ppo_tensors)`."""
    if not samples:
        raise ValueError("cannot collate an empty PPO sample list")
    batch, _targets, _values = collate_samples(samples)  # type: ignore[arg-type]
    ppo = {
        "actions": torch.tensor([s.action for s in samples], dtype=torch.long),
        "old_log_probs": torch.tensor(
            [s.old_log_prob for s in samples], dtype=torch.float32
        ),
        "advantages": torch.tensor(
            [s.advantage for s in samples], dtype=torch.float32
        ),
        "returns": torch.tensor([s.return_ for s in samples], dtype=torch.float32),
    }
    return batch, ppo


def compute_gae(
    values: Sequence[float],
    win_target: float,
    *,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE for one episode with zero intermediate rewards and `win_target` at the end.

    `values` are the win-probability head's predictions, so the terminal outcome must be in
    the same space: loss 0, tie 0.5, win 1. The returns are then win-probability targets.
    """
    if not 0.0 <= float(win_target) <= 1.0:
        raise ValueError(f"win_target must be in [0, 1] (got {win_target}); pass (utility+1)/2")
    n = len(values)
    if n == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    advantages = np.zeros((n,), dtype=np.float32)
    returns = np.zeros((n,), dtype=np.float32)
    next_value = 0.0
    next_advantage = 0.0
    for t in reversed(range(n)):
        terminal = t == n - 1
        reward = float(win_target) if terminal else 0.0
        nonterminal = 0.0 if terminal else 1.0
        delta = reward + gamma * next_value * nonterminal - float(values[t])
        next_advantage = delta + gamma * gae_lambda * nonterminal * next_advantage
        advantages[t] = next_advantage
        returns[t] = next_advantage + float(values[t])
        next_value = float(values[t])
    return advantages, returns


def _normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    """Standardize advantages per batch so the step size does not depend on the win/loss mix."""
    if advantages.numel() < 2:
        return advantages
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)


def _policy_terms(
    model: ComparatorNetwork,
    batch: dict[str, torch.Tensor],
    ppo: dict[str, torch.Tensor],
    *,
    clip: float,
    value_coef: float,
    entropy_coef: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    out = model(batch, compute_policy=True, compute_value=True)
    mask = batch["cand_mask"].bool()
    logits = out["policy_logits"].float()
    log_probs = F.log_softmax(logits, dim=-1)
    safe_log_probs = torch.where(mask, log_probs, torch.zeros_like(log_probs))
    actions = ppo["actions"].to(logits.device).long()
    new_log_probs = safe_log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    old_log_probs = ppo["old_log_probs"].to(logits.device).float()
    advantages = _normalize_advantages(ppo["advantages"].to(logits.device).float())
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()

    value_logit = out["value_logit"].float()
    returns = ppo["returns"].to(value_logit.device).float()
    value_targets = returns.clamp(0.0, 1.0)
    value_loss = F.binary_cross_entropy_with_logits(value_logit, value_targets)

    probs = out["policy_probs"].float()
    entropy = -torch.where(
        mask, probs * torch.log(probs.clamp_min(1e-12)), torch.zeros_like(probs)
    ).sum(dim=-1).mean()
    total = policy_loss + value_coef * value_loss - entropy_coef * entropy
    return total, {
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "entropy": entropy.detach(),
        "total_loss": total.detach(),
        "mean_ratio": ratio.detach().mean(),
    }


def compute_surrogate_loss(
    model: ComparatorNetwork,
    samples: Sequence[PPOSample],
    *,
    clip: float = 0.2,
    device: str | torch.device = "cpu",
) -> float:
    """Return the clipped PPO surrogate loss on `samples` with the current weights."""
    batch, ppo = collate_ppo_samples(samples)
    dev = torch.device(device)
    moved = {k: v.to(dev) for k, v in batch.items()}
    ppo = {k: v.to(dev) for k, v in ppo.items()}
    model.eval()
    with torch.inference_mode():
        out = model(moved, compute_policy=True, compute_value=False)
        log_probs = F.log_softmax(out["policy_logits"].float(), dim=-1)
        safe = torch.where(moved["cand_mask"].bool(), log_probs, torch.zeros_like(log_probs))
        actions = ppo["actions"].long()
        new_log_probs = safe.gather(1, actions.unsqueeze(1)).squeeze(1)
        ratio = torch.exp(new_log_probs - ppo["old_log_probs"].float())
        advantages = _normalize_advantages(ppo["advantages"].float())
        loss = -torch.minimum(
            ratio * advantages,
            torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages,
        ).mean()
    return float(loss.item())


@dataclass(slots=True)
class PPOStats:
    """Aggregated learner metrics for one cycle."""

    updates: int = 0
    policy_loss: float | None = None
    value_loss: float | None = None
    entropy: float | None = None
    total_loss: float | None = None
    mean_ratio: float | None = None
    nonfinite_loss: float | None = None
    nonfinite_params: bool = False
    per_step_policy: list[float] = field(default_factory=list)
    per_step_value: list[float] = field(default_factory=list)
    per_step_total: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "updates": self.updates,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "entropy": self.entropy,
            "total_loss": self.total_loss,
            "mean_ratio": self.mean_ratio,
            "nonfinite_loss": self.nonfinite_loss,
            "nonfinite_params": self.nonfinite_params,
        }


def _params_are_finite(model: torch.nn.Module) -> bool:
    return all(bool(torch.isfinite(p).all().item()) for p in model.parameters())


class PPOLearner:
    """Clipped-surrogate PPO over fresh self-play rollouts (with buffer fallback)."""

    def __init__(
        self,
        config: Any,
        model: ComparatorNetwork,
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
        buffer: PPOBuffer,
        *,
        updates: int | None = None,
        samples: Sequence[PPOSample] | None = None,
    ) -> PPOStats:
        """Run `updates` clipped-surrogate steps on a fresh rollout (or buffer sample)."""
        n_updates = self.config.updates_per_cycle if updates is None else updates
        stats = PPOStats()
        # An explicit empty rollout (e.g. every game hit the deadline) must not fall back to the
        # buffer: its log-probs and advantages come from older weights.
        source = list(samples) if samples is not None else list(buffer.samples)
        if n_updates <= 0 or not source:
            return stats

        amp_enabled = bool(self.config.use_amp and self.device.type == "cuda")
        amp_dtype = torch.bfloat16 if amp_enabled else torch.float32
        # Dropout off: PPO's ratio compares against the deterministic policy the agent acted
        # with (`_behavior_stats` runs in eval mode), so the update must evaluate it too.
        self.model.eval()
        batch_size = min(self.config.learner_batch_size, len(source))
        for _ in range(n_updates):
            indices = self.rng.integers(0, len(source), size=batch_size)
            chunk = [source[int(i)] for i in indices]
            batch, ppo = collate_ppo_samples(chunk)
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            ppo = {k: v.to(self.device, non_blocking=True) for k, v in ppo.items()}
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=self.device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                loss, parts = _policy_terms(
                    self.model,
                    batch,
                    ppo,
                    clip=self.config.ppo_clip,
                    value_coef=self.config.ppo_value_coef,
                    entropy_coef=self.config.ppo_entropy_coef,
                )
            loss_value = float(loss.detach().item())
            if not math.isfinite(loss_value):
                stats.nonfinite_loss = loss_value
                break
            loss.backward()
            if self.config.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            if not _params_are_finite(self.model):
                stats.nonfinite_params = True
                break
            self.step += 1
            stats.updates += 1
            stats.per_step_policy.append(float(parts["policy_loss"].item()))
            stats.per_step_value.append(float(parts["value_loss"].item()))
            stats.per_step_total.append(loss_value)

        if stats.updates:
            stats.policy_loss = float(np.mean(stats.per_step_policy))
            stats.value_loss = float(np.mean(stats.per_step_value))
            stats.total_loss = float(np.mean(stats.per_step_total))
            stats.entropy = float(parts["entropy"].item())
            stats.mean_ratio = float(parts["mean_ratio"].item())
        self.model.eval()
        return stats

    def update_on_batch(
        self,
        samples: Sequence[PPOSample],
        *,
        steps: int = 1,
    ) -> PPOStats:
        """Apply `steps` full-batch PPO updates (used by the surrogate-loss test)."""
        stats = PPOStats()
        if not samples or steps <= 0:
            return stats
        batch, ppo = collate_ppo_samples(samples)
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        ppo = {k: v.to(self.device, non_blocking=True) for k, v in ppo.items()}
        # Dropout off: PPO's ratio compares against the deterministic policy the agent acted
        # with (`_behavior_stats` runs in eval mode), so the update must evaluate it too.
        self.model.eval()
        for _ in range(steps):
            self.optimizer.zero_grad(set_to_none=True)
            loss, parts = _policy_terms(
                self.model,
                batch,
                ppo,
                clip=self.config.ppo_clip,
                value_coef=self.config.ppo_value_coef,
                entropy_coef=self.config.ppo_entropy_coef,
            )
            loss_value = float(loss.detach().item())
            if not math.isfinite(loss_value):
                stats.nonfinite_loss = loss_value
                break
            loss.backward()
            if self.config.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            if not _params_are_finite(self.model):
                stats.nonfinite_params = True
                break
            self.step += 1
            stats.updates += 1
            stats.per_step_policy.append(float(parts["policy_loss"].item()))
            stats.per_step_value.append(float(parts["value_loss"].item()))
            stats.per_step_total.append(loss_value)
        if stats.updates:
            stats.policy_loss = float(np.mean(stats.per_step_policy))
            stats.value_loss = float(np.mean(stats.per_step_value))
            stats.total_loss = float(np.mean(stats.per_step_total))
        self.model.eval()
        return stats
