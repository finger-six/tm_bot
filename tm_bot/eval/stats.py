"""Bootstrap confidence intervals and latency summary statistics."""

from __future__ import annotations

from typing import Sequence

import numpy as np

DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 0


def cluster_bootstrap_mean_ci(
    cluster_totals: Sequence[tuple[float, int]],
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence: float = 0.95,
    rng_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[float | None, tuple[float, float] | None]:
    """Per-game mean and cluster bootstrap percentile CI over `(value_sum, count)` clusters.

    Each element is one resampling cluster: `value_sum` is the sum of a per-game value over
    the cluster's `count` games. Clusters with `count <= 0` are ignored. The CI is not
    clipped, so it also serves signed quantities such as a paired score difference.
    """
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    valid = [(float(pts), int(cnt)) for pts, cnt in cluster_totals if cnt > 0]
    if not valid:
        return None, None

    points = np.array([pts for pts, _ in valid], dtype=np.float64)
    counts = np.array([cnt for _, cnt in valid], dtype=np.float64)
    total_count = float(counts.sum())
    mean = float(points.sum() / total_count)

    m = len(valid)
    rng = np.random.default_rng(rng_seed)

    # Chunk resamples to bound peak memory even for large pair counts.
    chunk_size = max(1, min(n_resamples, max(1, 2_000_000 // max(1, m))))
    boot_means = np.empty(n_resamples, dtype=np.float64)
    offset = 0
    while offset < n_resamples:
        cur = min(chunk_size, n_resamples - offset)
        idx = rng.integers(0, m, size=(cur, m))
        boot_means[offset : offset + cur] = points[idx].sum(axis=1) / counts[idx].sum(axis=1)
        offset += cur

    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(boot_means, [alpha, 1.0 - alpha])
    return mean, (float(low), float(high))


def paired_bootstrap_ci(
    pair_totals: Sequence[tuple[float, int]],
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence: float = 0.95,
    rng_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[float | None, tuple[float, float] | None]:
    """Compute mean win rate and paired bootstrap percentile CI over `(points, completed_count)` pairs.

    Each element of `pair_totals` is one resampling cluster of seat-swapped games (a whole seed
    for the overall rate, a `(seed_key, map)` pair for per-map rates): `points` is the sum of
    Agent A scores (`win=1.0, tie=0.5, loss=0.0`), and `completed_count` is the number of
    completed games in the cluster. Clusters with `completed_count <= 0` are ignored.
    """
    mean_wr, ci = cluster_bootstrap_mean_ci(
        pair_totals, n_resamples=n_resamples, confidence=confidence, rng_seed=rng_seed
    )
    if ci is None:
        return mean_wr, None
    return mean_wr, (float(np.clip(ci[0], 0.0, 1.0)), float(np.clip(ci[1], 0.0, 1.0)))


def percentile_ms(samples: Sequence[float], pct: float) -> float | None:
    """Return a quantile in milliseconds over `samples`, or `None` if empty."""
    if not samples:
        return None
    arr = np.asarray(samples, dtype=np.float64)
    return float(np.percentile(arr, pct))


def summarize_latency(samples: Sequence[float]) -> dict[str, float | int | None]:
    """Summarize per-decision latency samples in milliseconds."""
    if not samples:
        return {
            "decisions": 0,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
        }
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "decisions": int(arr.size),
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50.0)),
        "p95_ms": float(np.percentile(arr, 95.0)),
    }
