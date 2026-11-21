"""Fit and store a main warm start's value temperature on the val split (#58, #60).

    python -m tm_bot.training.value_temperature --checkpoint runs/warmstart-v3/best

T is fitted on gen >= 2 value logits (`fit_midgame_temperature`, as the comparator's warm start
does) and written to `<checkpoint stem>.value_temperature.json` beside the checkpoint, with the
raw and post-temperature val value reports the gate's check 14 reads. Warm starts trained since
#60 already carry T in their metadata and `metrics.json`; this is for checkpoints trained before.
The test split is never read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tm_bot.comparator.calibration import fit_midgame_temperature
from tm_bot.models.value_temperature import (
    VALUE_TEMPERATURE_SCHEMA,
    checkpoint_sha256,
    value_temperature_path,
)
from tm_bot.training.multitask import value_report


def value_temperature_payload(values: Mapping[str, np.ndarray], *, checkpoint_sha: str | None = None) -> dict[str, Any]:
    """The fit on `collect_value_outputs` rows and the raw / post-temperature value reports."""
    fit = fit_midgame_temperature(values["logit"], values["won"], values["generation"])
    return {
        "schema_version": VALUE_TEMPERATURE_SCHEMA,
        "checkpoint_sha256": checkpoint_sha,
        "split": "val",
        "temperature": float(fit["temperature"]),
        "val_calibration": fit,
        "value_raw": value_report(values),
        "value_calibrated": value_report(values, temperature=fit["temperature"]),
    }


def write_value_temperature(checkpoint: str | Path, payload: Mapping[str, Any]) -> Path:
    path = value_temperature_path(checkpoint)
    path.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")
    return path


def fit_checkpoint_value_temperature(
    checkpoint: str | Path,
    *,
    data_dir: Path | None = None,
    cache_dir: Path | None = None,
    device: str | None = None,
    workers: int | None = None,
    batch_size: int = 1024,
) -> dict[str, Any]:
    """Fit T for one main checkpoint on the val split's value rows."""
    import torch

    from tm_bot.models.network import load_checkpoint
    from tm_bot.training.multitask import collect_value_outputs, load_warmstart_data

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _, _ = load_checkpoint(checkpoint, map_location=dev, encoder_version=None)
    data = load_warmstart_data(data_dir, cache_dir, workers=workers)
    values = collect_value_outputs(model, data, "val", device=dev, batch_size=batch_size)
    return value_temperature_payload(values, checkpoint_sha=checkpoint_sha256(checkpoint))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True, help="Main checkpoint (file or best/ dir).")
    parser.add_argument("--data-dir", type=Path, default=None, help="v2 warm-start parquet directory.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Encoded cache (default <data-dir>_cache).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=None, help="Cache-building processes.")
    args = parser.parse_args(argv)
    payload = fit_checkpoint_value_temperature(
        args.checkpoint, data_dir=args.data_dir, cache_dir=args.cache_dir, device=args.device, workers=args.workers
    )
    path = write_value_temperature(args.checkpoint, payload)
    raw, cal = payload["value_raw"]["midgame"], payload["value_calibrated"]["midgame"]
    print(
        f"[value-temperature] T = {payload['temperature']:.4f} (n = {payload['val_calibration']['n']:,} gen >= 2 "
        f"val rows); ECE {raw['ece']:.4f} -> {cal['ece']:.4f}, Brier {raw['brier']:.4f} -> {cal['brier']:.4f}, "
        f"accuracy {cal['accuracy']:.4f}; wrote {path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
