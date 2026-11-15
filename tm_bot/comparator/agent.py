"""Inference-time comparator agent: direct typed-candidate scoring, no search."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np
import torch

from tm_bot.comparator.model import ComparatorNetwork, load_comparator_checkpoint
from tm_bot.env.types import ActorObservation, Choice
from tm_bot.models.encode import collate_encoded_batch, encode_choice, encode_observation
from tm_bot.models.vocab import CardVocab


class ComparatorAgent:
    """Agent backed by the independently designed comparator network.

    `predict` returns raw candidate probabilities and the win probability
    temperature-scaled by the checkpoint's calibration temperature. The value
    logit behind the last decision is exposed for calibration reporting.
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
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.vocab = vocab
        self.temperature = float(temperature)
        self.version = version
        self._seed = seed
        self._rng = random.Random(seed)
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
        )
        memo[id(self)] = clone
        return clone

    @torch.inference_mode()
    def predict(
        self, observation: ActorObservation, choice: Choice
    ) -> tuple[np.ndarray, float]:
        """Return `(candidate_probs, calibrated_win_prob)` for one decision."""
        enc_obs = encode_observation(observation, self.vocab)
        enc_choice = encode_choice(choice, observation, self.vocab)
        batch = collate_encoded_batch([enc_obs], [enc_choice])
        tensors = {k: v.to(self.device) for k, v in batch.items()}
        out = self.model(tensors, compute_policy=True, compute_value=True)
        probs = out["policy_probs"][0].detach().cpu().numpy()
        logit = float(out["value_logit"][0].item())
        self._last_predicted_logit = logit
        win_prob = float(torch.sigmoid(torch.tensor(logit / self.calibration_temperature)).item())
        return probs, win_prob

    @torch.inference_mode()
    def predict_batch(
        self,
        observations: Sequence[ActorObservation],
        choices: Sequence[Choice],
        *,
        compute_value: bool = True,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Batched inference returning `(list_of_prob_arrays, calibrated_win_probs)`."""
        enc_obs = [encode_observation(o, self.vocab) for o in observations]
        enc_ch = [encode_choice(c, o, self.vocab) for o, c in zip(observations, choices)]
        batch = collate_encoded_batch(enc_obs, enc_ch)
        tensors = {k: v.to(self.device) for k, v in batch.items()}
        out = self.model(tensors, compute_policy=True, compute_value=compute_value)
        probs_tensor = out["policy_probs"].detach().cpu().numpy()
        if compute_value:
            logits = out["value_logit"].detach().cpu().numpy()
            values = 1.0 / (1.0 + np.exp(-logits / self.calibration_temperature))
        else:
            values = np.zeros((len(observations),), dtype=np.float32)
        probs_list = [
            probs_tensor[i, : len(c.candidates)].copy() for i, c in enumerate(choices)
        ]
        return probs_list, values

    def act(self, observation: ActorObservation, choice: Choice) -> int:
        n = len(choice.candidates)
        if n == 0:
            raise ValueError("Choice has no candidates")
        if n == 1:
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
        if self.temperature <= 0.0:
            index = int(np.argmax(probs))
        elif self.temperature != 1.0:
            logits = np.log(np.clip(probs, 1e-12, None)) / self.temperature
            logits -= logits.max()
            weights = np.exp(logits)
            probs = weights / weights.sum()
            index = int(self._rng.choices(range(n), weights=probs.tolist(), k=1)[0])
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
    ) -> ComparatorAgent:
        model, vocab, meta = load_comparator_checkpoint(path, map_location=device)
        resolved_version = version or str(meta.get("version", "comparator-warmstart-v1"))
        return cls(
            model=model,
            vocab=vocab,
            device=device,
            temperature=temperature,
            seed=seed,
            version=resolved_version,
        )
