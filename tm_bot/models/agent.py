"""Inference agent wrapper around TMNetwork implementing the Agent protocol."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np
import torch

from tm_bot.env.types import ActorObservation, Choice
from tm_bot.models.encode import collate_encoded_batch, encode_choice, encode_observation
from tm_bot.models.network import TMNetwork, load_checkpoint
from tm_bot.models.value_temperature import calibrated_win_prob, load_value_temperature
from tm_bot.models.vocab import CardVocab


class PolicyValueAgent:
    """Agent backed by a shared TMNetwork policy/value checkpoint."""

    def __init__(
        self,
        model: TMNetwork,
        vocab: CardVocab,
        *,
        device: str | torch.device = "cpu",
        temperature: float = 0.0,
        seed: int | None = None,
        version: str = "policy-value-v1",
        value_temperature: float = 1.0,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.vocab = vocab
        self.temperature = float(temperature)
        # Temperature of the reported P(win) only (#60); search leaves use the raw value head.
        self.value_temperature = float(value_temperature)
        self.version = version
        self._seed = seed
        self._rng = random.Random(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = seed
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random(self._seed)

    def __deepcopy__(self, memo: dict[int, Any]) -> PolicyValueAgent:
        cls = self.__class__
        clone = cls.__new__(cls)
        memo[id(self)] = clone
        clone.device = self.device
        clone.model = self.model
        clone.vocab = self.vocab
        clone.temperature = self.temperature
        clone.value_temperature = self.value_temperature
        clone.version = self.version
        clone._seed = self._seed
        clone._rng = random.Random(self._seed)
        return clone

    @torch.inference_mode()
    def predict(
        self, observation: ActorObservation, choice: Choice
    ) -> tuple[np.ndarray, float]:
        """Return `(candidate_probs, value)` for a single observation and choice."""
        enc_obs = encode_observation(observation, self.vocab)
        enc_choice = encode_choice(
            choice, observation, self.vocab, encoder_version=self.model.config.encoder_version
        )
        batch = collate_encoded_batch([enc_obs], [enc_choice])
        tensors = {k: v.to(self.device) for k, v in batch.items()}
        out = self.model(tensors, compute_policy=True, compute_value=True)
        probs = out["policy_probs"][0].detach().cpu().numpy()
        value = float(out["value"][0].item())
        return probs, value

    @torch.inference_mode()
    def predict_batch(
        self,
        observations: Sequence[ActorObservation],
        choices: Sequence[Choice],
        *,
        compute_value: bool = True,
        return_margin: bool = False,
    ) -> Any:
        """Batched inference returning `(list_of_prob_arrays, values_array)`.

        With `return_margin` a third array holds the VP-margin head's predictions
        (zeros when the network has no margin head or `compute_value` is False).
        """
        enc_obs = [encode_observation(o, self.vocab) for o in observations]
        ev = self.model.config.encoder_version
        enc_ch = [
            encode_choice(c, o, self.vocab, encoder_version=ev)
            for o, c in zip(observations, choices)
        ]
        batch = collate_encoded_batch(enc_obs, enc_ch)
        tensors = {k: v.to(self.device) for k, v in batch.items()}
        out = self.model(tensors, compute_policy=True, compute_value=compute_value)
        probs_tensor = out["policy_probs"].detach().cpu().numpy()
        values = (
            out["value"].detach().cpu().numpy()
            if compute_value
            else np.zeros((len(observations),), dtype=np.float32)
        )
        probs_list = [probs_tensor[i, : len(c.candidates)].copy() for i, c in enumerate(choices)]
        if not return_margin:
            return probs_list, values
        margin = out.get("vp_margin_pred") if compute_value else None
        margins = (
            margin.detach().float().cpu().numpy()
            if margin is not None
            else np.zeros((len(observations),), dtype=np.float32)
        )
        return probs_list, values, margins

    def win_probability(self, values: Any) -> np.ndarray:
        """Reported P(win) of value-head outputs, scaled by the checkpoint's value temperature."""
        return calibrated_win_prob(values, self.value_temperature)

    def act(self, observation: ActorObservation, choice: Choice) -> int:
        n = len(choice.candidates)
        if n == 0:
            raise ValueError("Choice has no candidates")
        if n == 1:
            return 0
        probs, _ = self.predict(observation, choice)
        probs = np.asarray(probs[:n], dtype=np.float64)
        total = float(probs.sum())
        if total <= 0.0 or not np.isfinite(total):
            return 0
        if self.temperature <= 0.0:
            return int(np.argmax(probs))
        if self.temperature != 1.0:
            logits = np.log(np.clip(probs, 1e-12, None)) / self.temperature
            logits -= logits.max()
            w = np.exp(logits)
            probs = w / w.sum()
        else:
            probs = probs / total
        return self._rng.choices(range(n), weights=probs.tolist(), k=1)[0]

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
    ) -> PolicyValueAgent:
        """Load an agent; it encodes with the checkpoint's own encoder version.

        `encoder_version` additionally requires that version (raising
        `EncoderVersionMismatchError` otherwise); `None` accepts any. The reported P(win)
        uses the checkpoint's value temperature (`load_value_temperature`).
        """
        model, vocab, meta = load_checkpoint(
            path, map_location=device, encoder_version=encoder_version
        )
        resolved_version = version or str(meta.get("version", "warmstart-v1"))
        return cls(
            model=model,
            vocab=vocab,
            device=device,
            temperature=temperature,
            seed=seed,
            version=resolved_version,
            value_temperature=load_value_temperature(path, meta),
        )
