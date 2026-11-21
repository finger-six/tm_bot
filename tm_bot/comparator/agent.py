"""Inference-time comparator agent: direct typed-candidate scoring, no search."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any, Callable

import numpy as np
import torch

from tm_bot.agents.play_rules import V2_PLAY_TEMPERATURE, PlayTemperatureSchedule
from tm_bot.comparator.model import ComparatorNetwork, load_comparator_checkpoint
from tm_bot.env.types import ActorObservation, Choice
from tm_bot.models.encode import collate_encoded_batch, encode_choice, encode_observation
from tm_bot.models.network import check_encoder_version
from tm_bot.models.vocab import CardVocab


class ComparatorAgent:
    """Agent backed by the independently designed comparator network.

    `predict` returns raw candidate probabilities and the win probability
    temperature-scaled by the checkpoint's calibration temperature. The value
    logit behind the last decision is exposed for calibration reporting.

    With a `play_temperature` schedule (v2 rules, #48) the agent samples at τ(g) in the
    action phase and at τ = 1 elsewhere (#39); without one it uses the flat `temperature`
    everywhere (v1). `last_decision_temperature` is the τ it sampled with.
    """

    def __init__(
        self,
        model: ComparatorNetwork,
        vocab: CardVocab,
        *,
        device: str | torch.device = "cpu",
        temperature: float = 0.0,
        seed: int | None = None,
        version: str = "comparator-v1",
        play_temperature: PlayTemperatureSchedule | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.vocab = vocab
        self.temperature = float(temperature)
        self.play_temperature = play_temperature
        self.version = version
        self._seed = seed
        self._rng = random.Random(seed)
        self.last_decision_temperature: float | None = None
        self.last_decision_policy: np.ndarray | None = None
        self.last_decision_value: float | None = None
        self.last_decision_logit: float | None = None

    @property
    def calibration_temperature(self) -> float:
        return float(getattr(self.model, "temperature", 1.0))

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = seed
        self._rng = random.Random(self._seed)
        self.last_decision_temperature = None
        self.last_decision_policy = None
        self.last_decision_value = None
        self.last_decision_logit = None

    def __deepcopy__(self, memo: dict[int, Any]) -> ComparatorAgent:
        clone = self.__class__(
            self.model,
            self.vocab,
            device=self.device,
            temperature=self.temperature,
            seed=self._seed,
            version=self.version,
            play_temperature=self.play_temperature,
        )
        memo[id(self)] = clone
        return clone

    @torch.inference_mode()
    def predict(
        self, observation: ActorObservation, choice: Choice
    ) -> tuple[np.ndarray, float]:
        """Return `(candidate_probs, calibrated_win_prob)` for one decision."""
        enc_obs = encode_observation(observation, self.vocab)
        enc_choice = encode_choice(
            choice, observation, self.vocab, encoder_version=self.model.config.encoder_version
        )
        batch = collate_encoded_batch([enc_obs], [enc_choice])
        tensors = {k: v.to(self.device) for k, v in batch.items()}
        out = self.model(tensors, compute_policy=True, compute_value=True)
        probs = out["policy_probs"][0].detach().cpu().numpy()
        logit = float(out["value_logit"][0].item())
        self._last_predicted_logit = logit
        win_prob = float(torch.sigmoid(torch.tensor(logit / self.calibration_temperature)).item())
        return probs, win_prob

    def decision_temperature(self, observation: ActorObservation) -> float:
        """τ(g) in the action phase and 1 elsewhere with a schedule; else the flat temperature."""
        if self.play_temperature is None:
            return self.temperature
        tau = self.play_temperature.for_observation(observation)
        return 1.0 if tau is None else tau

    def act(self, observation: ActorObservation, choice: Choice) -> int:
        n = len(choice.candidates)
        if n == 0:
            raise ValueError("Choice has no candidates")
        if n == 1:
            self.last_decision_temperature = None
            self.last_decision_policy = np.ones((1,), dtype=np.float64)
            self.last_decision_value = None
            self.last_decision_logit = None
            return 0
        probs, win_prob = self.predict(observation, choice)
        probs = np.asarray(probs[:n], dtype=np.float64)
        total = float(probs.sum())
        if total <= 0.0 or not np.isfinite(total):
            probs = np.full((n,), 1.0 / n, dtype=np.float64)
        else:
            probs = probs / total
        temperature = self.decision_temperature(observation)
        self.last_decision_temperature = float(temperature)
        if temperature <= 0.0:
            index = int(np.argmax(probs))
        elif temperature != 1.0:
            logits = np.log(np.clip(probs, 1e-12, None)) / temperature
            logits -= logits.max()
            weights = np.exp(logits)
            # Sample at τ but record the τ = 1 policy, matching `old_log_prob`.
            sampled = weights / weights.sum()
            index = int(self._rng.choices(range(n), weights=sampled.tolist(), k=1)[0])
        else:
            index = int(self._rng.choices(range(n), weights=probs.tolist(), k=1)[0])
        self.last_decision_policy = probs
        self.last_decision_value = win_prob
        self.last_decision_logit = float(getattr(self, "_last_predicted_logit", 0.0))
        return index

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
        temperature: float = 0.0,
        seed: int | None = None,
        version: str | None = None,
        encoder_version: int | None = None,
    ) -> ComparatorAgent:
        """Load an agent; it encodes with the checkpoint's own encoder version."""
        model, vocab, meta = load_comparator_checkpoint(
            path, map_location=device, encoder_version=encoder_version
        )
        resolved_version = version or str(meta.get("version", "comparator-warmstart-v1"))
        return cls(
            model=model,
            vocab=vocab,
            device=device,
            temperature=temperature,
            seed=seed,
            version=resolved_version,
        )


# Evaluation plays the trained stochastic policy. Greedy argmax play can loop
# forever: two greedy comparators kept taking the same standard project with its
# parameter maxed, never raised temperature, and every dev game hit the step cap.
# Each seat's agent is reset with its seat seed, so evaluation stays reproducible.
# Under the v2 rules (#39/#48) this is the non-action temperature and the action
# phase samples at τ(g); with `play_temperature="flat"` it applies everywhere (v1).
EVAL_ACTION_TEMPERATURE = 1.0


def comparator_eval_factory(
    model: ComparatorNetwork,
    vocab: CardVocab,
    *,
    device: str | torch.device = "cpu",
    version: str = "comparator-v1",
    play_temperature: PlayTemperatureSchedule | None = V2_PLAY_TEMPERATURE,
) -> Callable[[], ComparatorAgent]:
    """Fresh evaluation agent per game sharing `model`, sampling its policy at τ(g)."""
    return lambda: ComparatorAgent(
        model,
        vocab,
        device=device,
        temperature=EVAL_ACTION_TEMPERATURE,
        version=version,
        play_temperature=play_temperature,
    )


_EVAL_MODEL_CACHE: dict[tuple[str, str], tuple[ComparatorNetwork, CardVocab, str]] = {}


def make_comparator_agent(
    *,
    checkpoint: str | Path,
    device: str | None = None,
    encoder_version: int | None = None,
    play_temperature: str = "v2",
) -> ComparatorAgent:
    """Eval-CLI factory (`tm_bot.comparator:make_comparator_agent?checkpoint=...`).

    Loads each checkpoint once per process and returns a fresh agent playing the
    evaluation policy. The agent encodes with the checkpoint's own encoder version;
    `encoder_version` additionally requires that version. `play_temperature` is
    `"v2"` (τ(g) in the action phase, τ = 1 elsewhere) or `"flat"` (τ = 1 everywhere, v1).
    """
    if play_temperature not in ("v2", "flat"):
        raise ValueError(f"play_temperature must be 'v2' or 'flat', got {play_temperature!r}")
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = (str(Path(checkpoint).resolve()), resolved_device)
    cached = _EVAL_MODEL_CACHE.get(key)
    if cached is None:
        model, vocab, meta = load_comparator_checkpoint(
            checkpoint, map_location=resolved_device, encoder_version=encoder_version
        )
        model.eval()
        version = str(meta.get("version") or f"comparator-{Path(checkpoint).name}")
        cached = (model, vocab, version)
        _EVAL_MODEL_CACHE[key] = cached
    model, vocab, version = cached
    if encoder_version is not None:
        check_encoder_version(model.config.encoder_version, encoder_version, checkpoint)
    schedule = V2_PLAY_TEMPERATURE if play_temperature == "v2" else None
    return comparator_eval_factory(
        model, vocab, device=resolved_device, version=version, play_temperature=schedule
    )()
