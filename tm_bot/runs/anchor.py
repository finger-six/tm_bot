"""Frozen warm-start anchor (#39, #48): compatibility check and recorded anchor logits.

Both self-play engines play a share of their games against the frozen warm-start
checkpoint and store its policy logits with every recorded decision, so the learners
(#49 main, #50 comparator) can add a KL term toward it without re-running the anchor.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from tm_bot.models.encode import EncodedChoice, EncodedObservation, collate_encoded_batch
from tm_bot.models.vocab import CardVocab


def check_anchor_compatible(
    anchor_model: torch.nn.Module,
    anchor_vocab: CardVocab,
    model: torch.nn.Module,
    vocab: CardVocab,
) -> None:
    """Raise unless the anchor reads the learner's encodings (same encoder and vocab)."""
    anchor_version = int(anchor_model.config.encoder_version)  # type: ignore[union-attr]
    version = int(model.config.encoder_version)  # type: ignore[union-attr]
    if anchor_version != version:
        raise ValueError(
            f"warm-start anchor uses encoder v{anchor_version}, the learner v{version}"
        )
    if tuple(anchor_vocab.names) != tuple(vocab.names):
        raise ValueError("warm-start anchor vocab differs from the learner's vocab")


def freeze(model: torch.nn.Module) -> torch.nn.Module:
    """Put `model` in eval mode with gradients off (the anchor never trains)."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def anchor_policy_logits(
    model: torch.nn.Module,
    observations: Sequence[EncodedObservation],
    choices: Sequence[EncodedChoice],
    num_candidates: Sequence[int],
    *,
    device: str | torch.device = "cpu",
    chunk_size: int = 256,
) -> list[np.ndarray]:
    """Raw (τ = 1) anchor policy logits, one float32[n] array per decision.

    Index i of each array is candidate i of that decision's choice: the same order as
    the recorded policy target and action index.
    """
    if not (len(observations) == len(choices) == len(num_candidates)):
        raise ValueError("observations, choices and num_candidates must align")
    model.eval()
    out: list[np.ndarray] = []
    for start in range(0, len(observations), chunk_size):
        stop = start + chunk_size
        batch = collate_encoded_batch(observations[start:stop], choices[start:stop])
        tensors = {k: v.to(device) for k, v in batch.items()}
        logits = model(tensors, compute_policy=True, compute_value=False)["policy_logits"]
        logits = logits.float().cpu().numpy()
        for row, n in zip(logits, num_candidates[start:stop]):
            out.append(np.asarray(row[: int(n)], dtype=np.float32).copy())
    return out
