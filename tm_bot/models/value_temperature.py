"""Main value-head temperature (#58, #60): reported P(win) = σ(logit / T); search leaves stay raw.

T is fitted on the val split's gen >= 2 value logits (`fit_midgame_temperature`, as the comparator
does). A warm start trained since #60 stores it in its checkpoint metadata (`"temperature"`); an
older checkpoint gets a sidecar `<checkpoint stem>.value_temperature.json` beside it, written by
`python -m tm_bot.training.value_temperature`. The sidecar records the checkpoint's SHA-256, so a
fit is never applied to a different checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

VALUE_TEMPERATURE_SCHEMA = "value-temperature-v1"


def checkpoint_file(checkpoint: str | Path) -> Path:
    """The `.pt` file of a checkpoint given as a file or a `save_checkpoint` directory."""
    target = Path(checkpoint)
    return target / "checkpoint.pt" if target.is_dir() else target


def value_temperature_path(checkpoint: str | Path) -> Path:
    """`<dir>/<stem>.value_temperature.json` beside the checkpoint file."""
    ckpt = checkpoint_file(checkpoint)
    return ckpt.with_name(f"{ckpt.stem}.value_temperature.json")


def checkpoint_sha256(checkpoint: str | Path) -> str:
    digest = hashlib.sha256()
    with checkpoint_file(checkpoint).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_value_temperature_file(checkpoint: str | Path) -> dict[str, Any] | None:
    """The checkpoint's sidecar fit, or None; raises if it was fitted on another checkpoint."""
    path = value_temperature_path(checkpoint)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("checkpoint_sha256") != checkpoint_sha256(checkpoint):
        raise ValueError(f"{path} was fitted on a different checkpoint; refit or delete it")
    return payload


def load_value_temperature(checkpoint: str | Path, metadata: Mapping[str, Any] | None = None) -> float:
    """T of a main checkpoint: its sidecar, else its metadata's `"temperature"`, else 1 (raw)."""
    payload = read_value_temperature_file(checkpoint)
    if payload is not None:
        return float(payload["temperature"])
    temperature = (metadata or {}).get("temperature")
    return 1.0 if temperature is None else float(temperature)


def calibrated_win_prob(values: Any, temperature: float) -> np.ndarray:
    """P(win) = σ(logit / T) from value-head outputs v = 2σ(logit) − 1."""
    p = (np.asarray(values, dtype=np.float64) + 1.0) / 2.0
    if temperature == 1.0:
        return p
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return 1.0 / (1.0 + np.exp(-np.log(p / (1.0 - p)) / float(temperature)))
