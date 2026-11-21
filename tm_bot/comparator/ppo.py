"""PPO learner for the comparator: clipped surrogate, value critic, and GAE.

v2 PPO (#39 §1c, §1d, §2; #50):

- The ratio and the entropy bonus use the τ-scaled policy each decision was sampled
  from: softmax(logits / τ) over its legal candidates, where τ is the decision's recorded
  `play_temperature` (τ(g) in the action phase, 1 elsewhere). `scaled_log_probs` computes
  that policy for both the self-play engine's behavior log-probs and the learner, so the
  ratio is exactly 1 before an update.
- The reward is the terminal utility U = o + 0.15·m. GAE's baseline is the critic
  (2p − 1) + 0.15·m̂ from the win and margin heads (`critic_baseline`), so advantages and
  returns are in U units (γ 1.0, λ 0.95).
- The two heads behind that critic train on their own terminal targets, never on U:
  `win_logit` on the outcome alone (BCE vs (o + 1) / 2 ∈ {0, 0.5, 1}), so `P(win)` and its
  calibration stay unshaped, and `vp_margin_pred` on m (SmoothL1, β 0.2).
- The warm-start anchors are shared with the main learner (`tm_bot/models/anchor_loss.py`,
  `tm_bot/training/human_aux.py`): β·KL(π_θ ‖ π_anchor) on raw τ = 1 logits against the
  recorded anchor logits, and λ_h·L_human on a resident human sample.

Loss per update:

    −clipped surrogate + value_coef·(BCE(win_logit, (o+1)/2) + w_margin·SmoothL1(m̂, m))
    − entropy_coef·H(π_τ) + β·KL(π_θ ‖ π_anchor) + λ_h·L_human
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import itertools
import math
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tm_bot.agents.play_rules import MARGIN_UTILITY_WEIGHT
from tm_bot.comparator.model import ComparatorNetwork
from tm_bot.models.anchor_loss import kl_to_anchor, pad_anchor_logits
from tm_bot.models.encode import EncodedChoice, EncodedObservation
from tm_bot.models.network import VP_MARGIN_BOUND, vp_margin_loss
from tm_bot.runs.buffer import _optional_float, _optional_logits, collate_samples
from tm_bot.runs.learner import HumanAuxSource

# |U| = |o + 0.15·m| <= 1 + 0.15·1.2 = 1.18, the range of the reward and of the GAE baseline.
UTILITY_BOUND = 1.0 + MARGIN_UTILITY_WEIGHT * VP_MARGIN_BOUND


@dataclass(frozen=True, slots=True)
class PPOSample:
    """One recorded comparator decision with behavior log-probability and GAE targets.

    `old_log_prob` is the log-probability of `action` under the τ-scaled policy it was
    sampled from (`scaled_log_probs` at `play_temperature`). `value_target` is the outcome
    o, the win head's only target. `advantage` and `return_` are GAE in U units.

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
    policy_target: np.ndarray  # τ = 1 policy over legal candidates (sampling used play_temperature)
    value_target: float  # terminal outcome o ∈ {+1, 0, -1} from the acting seat's perspective
    seat: int
    was_searched: bool
    seed_key: str
    map_name: str
    generation: int
    action: int
    old_log_prob: float
    behavior_value: float  # raw win probability at collection time (not the U baseline)
    advantage: float = 0.0
    return_: float = 0.0
    play_temperature: float | None = None
    margin_target: float = 0.0
    anchor_logits: np.ndarray | None = None

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
            "play_temperature": [s.play_temperature for s in self._items],
            "margin_target": [s.margin_target for s in self._items],
            "anchor_logits": [s.anchor_logits for s in self._items],
        }

    def load_state_dict(self, payload: dict[str, list]) -> None:
        self._items.clear()
        n = len(payload["action"])
        play_temperature = payload.get("play_temperature", [None] * n)
        margin_target = payload.get("margin_target", [0.0] * n)
        anchor_logits = payload.get("anchor_logits", [None] * n)
        for i in range(n):
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
                    play_temperature=_optional_float(play_temperature[i]),
                    margin_target=float(margin_target[i]),
                    anchor_logits=_optional_logits(anchor_logits[i]),
                )
            )


def collate_ppo_samples(
    samples: Sequence[PPOSample],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Collate PPO samples into `(model_batch, ppo_tensors)`.

    Besides actions, log-probs, advantages and returns, `ppo_tensors` holds `temperatures`
    (the τ each decision was sampled at), `outcomes` (o), `margins` (m), and
    `anchor_logits` / `has_anchor` (`pad_anchor_logits`).
    """
    if not samples:
        raise ValueError("cannot collate an empty PPO sample list")
    batch, _targets, outcomes = collate_samples(samples)  # type: ignore[arg-type]
    anchor, has_anchor = pad_anchor_logits(
        [s.anchor_logits for s in samples], int(batch["cand_mask"].shape[1])
    )
    ppo = {
        "actions": torch.tensor([s.action for s in samples], dtype=torch.long),
        "old_log_probs": torch.tensor(
            [s.old_log_prob for s in samples], dtype=torch.float32
        ),
        "advantages": torch.tensor(
            [s.advantage for s in samples], dtype=torch.float32
        ),
        "returns": torch.tensor([s.return_ for s in samples], dtype=torch.float32),
        "temperatures": sampling_temperatures([s.play_temperature for s in samples]),
        "outcomes": outcomes.float(),
        "margins": torch.tensor([s.margin_target for s in samples], dtype=torch.float32),
        "anchor_logits": anchor,
        "has_anchor": has_anchor,
    }
    return batch, ppo


def sampling_temperatures(play_temperatures: Sequence[float | None]) -> torch.Tensor:
    """float32[B]: the τ each decision was sampled at, 1 where none was reported.

    The comparator samples at τ(g) in the action phase and at τ = 1 elsewhere. A forced
    single-candidate decision reports no τ; its log-probability is 0 at any τ. An argmax
    decision (τ = 0) has no sampling policy to form PPO's ratio, so it is rejected.
    """
    out: list[float] = []
    for value in play_temperatures:
        tau = 1.0 if value is None else float(value)
        if not (math.isfinite(tau) and tau > 0.0):
            raise ValueError(f"PPO needs a sampled decision; got play temperature {value}")
        out.append(tau)
    return torch.tensor(out, dtype=torch.float32)


def scaled_log_probs(
    logits: torch.Tensor, temperatures: torch.Tensor, cand_mask: torch.Tensor
) -> torch.Tensor:
    """log softmax(logits / τ) per row over its legal candidates, 0 at illegal slots.

    This is the policy `ComparatorAgent.act` samples from. The behavior log-probs and
    the learner's new log-probs both come from here.
    """
    mask = cand_mask.bool()
    scaled = logits.float() / temperatures.to(logits.device).float().unsqueeze(1)
    log_probs = F.log_softmax(scaled.masked_fill(~mask, float("-inf")), dim=-1)
    return torch.where(mask, log_probs, torch.zeros_like(log_probs))


def critic_baseline(win_prob: np.ndarray, margin_pred: np.ndarray | None) -> np.ndarray:
    """The GAE baseline (2p − 1) + 0.15·m̂ in U units; m̂ counts as 0 without a margin head."""
    value = 2.0 * np.asarray(win_prob, dtype=np.float64) - 1.0
    if margin_pred is None:
        return value
    return value + MARGIN_UTILITY_WEIGHT * np.asarray(margin_pred, dtype=np.float64)


def compute_gae(
    values: Sequence[float],
    utility: float,
    *,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE over one seat's trajectory, with zero intermediate rewards and `utility` at the end.

    `utility` is the terminal U = o + 0.15·m and `values` is the critic baseline
    (2p − 1) + 0.15·m̂ (`critic_baseline`). Both are in U units, so the advantages and
    returns are too. Both must lie in [-1.18, 1.18].
    """
    if not (math.isfinite(float(utility)) and abs(float(utility)) <= UTILITY_BOUND + 1e-6):
        raise ValueError(f"utility must be U = o + 0.15·m in [-1.18, 1.18], got {utility}")
    values_arr = np.asarray(values, dtype=np.float64)
    if values_arr.size and not (
        np.all(np.isfinite(values_arr)) and np.all(np.abs(values_arr) <= UTILITY_BOUND + 1e-6)
    ):
        raise ValueError("GAE values must be (2p - 1) + 0.15·m̂ baselines in [-1.18, 1.18]")
    n = len(values)
    if n == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    advantages = np.zeros((n,), dtype=np.float32)
    returns = np.zeros((n,), dtype=np.float32)
    next_value = 0.0
    next_advantage = 0.0
    for t in reversed(range(n)):
        terminal = t == n - 1
        reward = float(utility) if terminal else 0.0
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
    margin_coef: float = 0.0,
    kl_beta: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
    """The weighted PPO loss on one batch, plus each unweighted term (None when skipped).

    The ratio and the entropy use the τ-scaled policy (`scaled_log_probs`), and the KL
    anchor uses raw τ = 1 logits. The win head's BCE target is (o + 1) / 2, never a return.
    The margin term is skipped for a model without the head, and the KL term when
    `kl_beta` is 0 or no row has anchor logits.
    """
    out = model(batch, compute_policy=True, compute_value=True)
    mask = batch["cand_mask"].bool()
    logits = out["policy_logits"].float()
    safe_log_probs = scaled_log_probs(logits, ppo["temperatures"], mask)
    actions = ppo["actions"].to(logits.device).long()
    new_log_probs = safe_log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    old_log_probs = ppo["old_log_probs"].to(logits.device).float()
    advantages = _normalize_advantages(ppo["advantages"].to(logits.device).float())
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()

    value_logit = out["value_logit"].float()
    win_targets = (ppo["outcomes"].to(value_logit.device).float() + 1.0) * 0.5
    value_loss = F.binary_cross_entropy_with_logits(value_logit, win_targets)
    critic_loss = value_loss
    margin_loss: torch.Tensor | None = None
    if "vp_margin_pred" in out:
        margin_loss = vp_margin_loss(out["vp_margin_pred"], ppo["margins"].to(value_logit.device))
        critic_loss = critic_loss + margin_coef * margin_loss

    probs = safe_log_probs.exp() * mask
    entropy = -(probs * safe_log_probs).sum(dim=-1).mean()
    total = policy_loss + value_coef * critic_loss - entropy_coef * entropy
    kl: torch.Tensor | None = None
    if kl_beta > 0.0:
        kl = kl_to_anchor(logits, ppo["anchor_logits"], mask, ppo["has_anchor"].to(logits.device))
        if kl is not None:
            total = total + kl_beta * kl
    return total, {
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "margin_loss": None if margin_loss is None else margin_loss.detach(),
        "kl_anchor": None if kl is None else kl.detach(),
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
        safe = scaled_log_probs(out["policy_logits"], ppo["temperatures"], moved["cand_mask"])
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
    """Aggregated learner metrics for one cycle.

    Loss terms are unweighted cycle means (`total_loss` is the weighted sum including
    λ_h·L_human). Terms that never applied are None.
    """

    updates: int = 0
    policy_loss: float | None = None
    value_loss: float | None = None
    margin_loss: float | None = None
    kl_anchor: float | None = None
    human_aux_loss: float | None = None
    human_aux_weight: float | None = None
    entropy: float | None = None
    total_loss: float | None = None
    mean_ratio: float | None = None
    nonfinite_loss: float | None = None
    nonfinite_params: bool = False
    per_step_policy: list[float] = field(default_factory=list)
    per_step_value: list[float] = field(default_factory=list)
    per_step_total: list[float] = field(default_factory=list)
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
            "entropy": self.entropy,
            "total_loss": self.total_loss,
            "mean_ratio": self.mean_ratio,
            "nonfinite_loss": self.nonfinite_loss,
            "nonfinite_params": self.nonfinite_params,
        }


def _params_are_finite(model: torch.nn.Module) -> bool:
    return all(bool(torch.isfinite(p).all().item()) for p in model.parameters())


def _mean_of(rows: Sequence[dict[str, float]], key: str) -> float | None:
    values = [row[key] for row in rows if key in row]
    return float(np.mean(values)) if values else None


class PPOLearner:
    """Clipped-surrogate PPO over fresh self-play rollouts (with buffer fallback)."""

    def __init__(
        self,
        config: Any,
        model: ComparatorNetwork,
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
        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.human_aux = human_aux
        self.training_clock = training_clock
        self.human_aux_schedule = config.human_aux_schedule
        # A random-init run is pure self-play: no warm-start anchor terms (#39 §4).
        self.kl_beta = 0.0 if config.random_init else float(config.kl_anchor_beta)
        self.rng = np.random.default_rng(config.seed)
        self.step = 0

    def _human_weight(self, training_s: float | None) -> float:
        if training_s is None:
            training_s = self.training_clock() if self.training_clock is not None else 0.0
        return self.human_aux_schedule.at_seconds(training_s)

    def train(
        self,
        buffer: PPOBuffer,
        *,
        updates: int | None = None,
        samples: Sequence[PPOSample] | None = None,
        training_s: float | None = None,
    ) -> PPOStats:
        """Run `updates` clipped-surrogate steps on a fresh rollout (or buffer sample).

        λ_h is read once per call, at `training_s` (default: `training_clock()`).
        """
        n_updates = self.config.updates_per_cycle if updates is None else updates
        # An explicit empty rollout (e.g. every game hit the deadline) must not fall back to the
        # buffer: its log-probs and advantages come from older weights.
        source = list(samples) if samples is not None else list(buffer.samples)
        if n_updates <= 0 or not source:
            return PPOStats()
        batch_size = min(self.config.learner_batch_size, len(source))

        def minibatches() -> Iterable[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]]:
            for _ in range(n_updates):
                indices = self.rng.integers(0, len(source), size=batch_size)
                yield collate_ppo_samples([source[int(i)] for i in indices])

        return self._run_updates(minibatches(), human_weight=self._human_weight(training_s))

    def update_on_batch(
        self,
        samples: Sequence[PPOSample],
        *,
        steps: int = 1,
        training_s: float | None = None,
    ) -> PPOStats:
        """Apply `steps` full-batch PPO updates (used by the surrogate-loss test)."""
        if not samples or steps <= 0:
            return PPOStats()
        collated = collate_ppo_samples(samples)
        return self._run_updates(
            itertools.repeat(collated, steps), human_weight=self._human_weight(training_s)
        )

    def _run_updates(
        self,
        minibatches: Iterable[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]],
        *,
        human_weight: float,
    ) -> PPOStats:
        """One optimizer step per minibatch. A non-finite loss or parameter stops the
        updates and is reported so the orchestrator can roll back."""
        stats = PPOStats()
        amp_enabled = bool(self.config.use_amp and self.device.type == "cuda")
        amp_dtype = torch.bfloat16 if amp_enabled else torch.float32
        # Dropout off: PPO's ratio compares against the deterministic policy the agent acted
        # with (`_behavior_stats` runs in eval mode), so the update must evaluate it too.
        # L_human runs in the same mode (the main learner runs it in train mode).
        self.model.eval()
        for batch, ppo in minibatches:
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
                    margin_coef=self.config.w_margin,
                    kl_beta=self.kl_beta,
                )
            loss_value = float(loss.detach().item())
            if not math.isfinite(loss_value):
                stats.nonfinite_loss = loss_value
                break
            loss.backward()
            step = {k: float(v.item()) for k, v in parts.items() if v is not None}
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
            if not _params_are_finite(self.model):
                stats.nonfinite_params = True
                break
            self.step += 1
            stats.updates += 1
            stats.per_step.append(step)
            stats.per_step_policy.append(step["policy_loss"])
            stats.per_step_value.append(step["value_loss"])
            stats.per_step_total.append(loss_value)

        if stats.updates:
            stats.policy_loss = float(np.mean(stats.per_step_policy))
            stats.value_loss = float(np.mean(stats.per_step_value))
            stats.total_loss = float(np.mean(stats.per_step_total))
            stats.margin_loss = _mean_of(stats.per_step, "margin_loss")
            stats.kl_anchor = _mean_of(stats.per_step, "kl_anchor")
            stats.human_aux_loss = _mean_of(stats.per_step, "human_aux_loss")
            if self.human_aux is not None:
                stats.human_aux_weight = human_weight
            last = stats.per_step[-1]
            stats.entropy = last["entropy"]
            stats.mean_ratio = last["mean_ratio"]
        self.model.eval()
        return stats
