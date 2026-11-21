"""Resident human-data sample and the human-data auxiliary loss L_human (#39 §1d; #49, #50).

Each v2 learner update adds `λ_h · L_human`, where L_human is the full #38 multi-task
warm-start loss (every policy head plus both value heads, `WarmstartConfig` weights and
per-task minibatch sizes) on train-split human rows. Rows come only from the human
warm-start tables, never from `b1-v2` games (#36).

The learner must stay under its 20.0 GiB RAM guard, which counts the process working set,
so memory-mapped pages it touches count too. `build_human_aux_sample` therefore draws a
fixed random subset of at most `max_rows` (default 1M) train rows once, at run start, and
copies the encoded-cache rows it chose into RAM (`resident_table`, ~1.4 KB per row:
measured 2.72 GiB at 2M rows). The full cache is released afterwards and never read again. The
generation-1 setup tables stay loaded whole (they are small); only the chosen setup rows
are sampled. Rows are split across the sampler's pools in proportion to the rows each pool
contributes per step, so every pool cycles through its share at the same rate; a pool
smaller than its share is taken whole and the remainder goes to the others.

API, shared by the main learner and the comparator PPO:

- `load_human_aux_sample(data_dir, cache_dir, max_rows=..., seed=...)` -> `HumanAuxSample`.
- `HumanAuxLoss(sample, rng=...)`; `.backward(model, weight=λ_h, device=..., use_amp=...)`
  draws one minibatch, back-propagates `weight · L_human` group by group into the
  existing gradients, and returns the unweighted per-task losses plus `loss_total`.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
import torch.nn as nn

from tm_bot.models.anchor_loss import HUMAN_AUX_MAX_ROWS
from tm_bot.training.encoded_cache import (
    _SPLIT_TO_IDX,
    _TOKEN_FIELDS,
    EncodedTable,
    _cand_fields,
)
from tm_bot.training.multitask import (
    STATE_HEADS,
    MultiTaskSampler,
    WarmstartConfig,
    WarmstartData,
    _to_device,
    load_warmstart_data,
    multitask_loss_groups,
)

# Sampler pools that index the setup tables instead of an encoded cache.
SETUP_POOLS: tuple[str, ...] = ("setup", "value_setup")
# Encoded cache behind each cache-backed sampler pool.
POOL_CACHE: dict[str, str] = {
    "value_research": "midgame_value",
    "value_action": "midgame_value",
    **{name: name for name, _, _ in STATE_HEADS},
}


def pool_draws(cfg: WarmstartConfig) -> dict[str, int]:
    """Rows each `MultiTaskSampler` pool contributes to one step."""
    n_setup, n_research, n_action = cfg.value_counts()
    sizes = cfg.task_batch_sizes()
    return {
        "setup": cfg.setup_batch_size,
        "value_setup": n_setup,
        "value_research": n_research,
        "value_action": n_action,
        **{name: sizes[name] for name, _, _ in STATE_HEADS},
    }


def allocate_rows(
    available: Mapping[str, int], draws: Mapping[str, int], max_rows: int
) -> dict[str, int]:
    """Rows to keep per pool: proportional to `draws`, capped by `available`, sum <= `max_rows`.

    Water-filling: a pool whose proportional share exceeds its size is taken whole and
    the rows it frees are shared among the remaining pools by the same rule. Rows lost to
    flooring go one each to the pools with the largest remainders, so the total is
    `min(max_rows, sum(available))` and a tiny cap never leaves every pool empty.
    """
    if max_rows < 0:
        raise ValueError(f"max_rows must be >= 0, got {max_rows}")
    out = {name: 0 for name in available}
    active = {n for n in available if available[n] > 0 and draws.get(n, 0) > 0}
    remaining = int(max_rows)
    while active and remaining > 0:
        weight = sum(draws[n] for n in active)
        share = {n: remaining * draws[n] / weight for n in active}
        full = {n for n in active if available[n] <= share[n]}
        if not full:
            for n in active:
                out[n] = int(share[n])
            leftover = remaining - sum(out[n] for n in active)
            by_remainder = sorted(active, key=lambda n: (out[n] - share[n], n))
            for n in by_remainder[:leftover]:
                out[n] += 1
            break
        for n in full:
            out[n] = int(available[n])
            remaining -= int(available[n])
        active -= full
    return out


def _ragged_flat(offsets: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Flat element indices of `rows` in a ragged array with row `offsets`, in row order."""
    starts = offsets[rows]
    counts = offsets[rows + 1] - starts
    total = int(counts.sum())
    if total == 0:
        return np.zeros((0,), dtype=np.int64)
    out_starts = np.cumsum(counts) - counts
    return np.repeat(starts - out_starts, counts) + np.arange(total, dtype=np.int64)


def _gather_ragged(
    arr: np.ndarray, offsets: np.ndarray, rows: np.ndarray, out_offsets: np.ndarray, chunk_rows: int
) -> np.ndarray:
    out = np.empty((int(out_offsets[-1]),) + arr.shape[1:], dtype=arr.dtype)
    for lo in range(0, rows.shape[0], chunk_rows):
        chunk = rows[lo : lo + chunk_rows]
        out[out_offsets[lo] : out_offsets[lo + chunk.shape[0]]] = arr[_ragged_flat(offsets, chunk)]
    return out


def _offsets(counts: np.ndarray) -> np.ndarray:
    out = np.zeros(counts.shape[0] + 1, dtype=np.int64)
    np.cumsum(counts, out=out[1:])
    return out


def resident_table(table: EncodedTable, rows: np.ndarray, *, chunk_rows: int = 65_536) -> EncodedTable:
    """In-RAM copy of `rows` of `table` (sorted, deduplicated), renumbered 0..n-1.

    Batches from the copy equal the source table's batches of the same rows; nothing in
    it refers to the source's memory maps.
    """
    rows = np.unique(np.asarray(rows, dtype=np.int64))
    n = int(rows.shape[0])
    token_names = set(_TOKEN_FIELDS)
    cand_names = set(_cand_fields(table.task))
    sub = EncodedTable.__new__(EncodedTable)
    sub.dir = table.dir
    sub.meta = {**table.meta, "num_rows": n, "resident_of": table.num_rows}
    sub.task = table.task
    sub.num_rows = n
    sub.tok_off = _offsets(table.tok_off[rows + 1] - table.tok_off[rows])
    has_cands = hasattr(table, "cand_off")
    if has_cands:
        sub.cand_off = _offsets(table.cand_off[rows + 1] - table.cand_off[rows])
    arrays: dict[str, np.ndarray] = {}
    for name, arr in table.arrays.items():
        if name in token_names:
            arrays[name] = _gather_ragged(arr, table.tok_off, rows, sub.tok_off, chunk_rows)
        elif name in cand_names and has_cands:
            arrays[name] = _gather_ragged(arr, table.cand_off, rows, sub.cand_off, chunk_rows)
        else:
            arrays[name] = np.ascontiguousarray(arr[rows])
    sub.arrays = arrays
    sub.space_feats = table.space_feats
    sub.space_mask = table.space_mask
    split = arrays["split"]
    sub.split_indices = {
        name: np.nonzero(split == idx)[0].astype(np.int64) for name, idx in _SPLIT_TO_IDX.items()
    }
    return sub


def _table_nbytes(table: EncodedTable) -> int:
    total = sum(int(a.nbytes) for a in table.arrays.values()) + int(table.tok_off.nbytes)
    if hasattr(table, "cand_off"):
        total += int(table.cand_off.nbytes)
    return total


@dataclass
class HumanAuxSample:
    """A bounded, RAM-resident set of train-split human rows for L_human.

    `data.caches` are `resident_table` copies holding only the chosen rows; `pools` maps
    each `MultiTaskSampler` pool to its rows (indices into the resident caches, or into
    the setup tables for `SETUP_POOLS`).
    """

    data: WarmstartData
    pools: dict[str, np.ndarray]

    @property
    def num_rows(self) -> int:
        """Rows in the sample, summed over pools."""
        return int(sum(len(rows) for rows in self.pools.values()))

    @property
    def cache_nbytes(self) -> int:
        """Bytes held by the resident encoded-cache copies (setup tables excluded)."""
        return sum(_table_nbytes(t) for t in self.data.caches.values())

    def sampler(self, rng: np.random.Generator, weights: WarmstartConfig | None = None) -> MultiTaskSampler:
        """A `MultiTaskSampler` restricted to this sample's rows."""
        sampler = MultiTaskSampler(self.data, weights or WarmstartConfig(), rng)
        sampler.pools = {name: rows.copy() for name, rows in self.pools.items()}
        return sampler


def build_human_aux_sample(
    data: WarmstartData,
    *,
    max_rows: int = HUMAN_AUX_MAX_ROWS,
    rng: np.random.Generator,
    weights: WarmstartConfig | None = None,
) -> HumanAuxSample:
    """Draw at most `max_rows` train rows from `data` and copy the cached ones into RAM."""
    cfg = weights or WarmstartConfig()
    full_pools = MultiTaskSampler(data, cfg, rng).pools  # train split only
    counts = allocate_rows({k: len(v) for k, v in full_pools.items()}, pool_draws(cfg), max_rows)
    chosen = {
        name: np.sort(rng.choice(rows, size=counts[name], replace=False)) if counts[name] else rows[:0]
        for name, rows in full_pools.items()
    }
    by_cache: dict[str, list[np.ndarray]] = {}
    for pool, cache in POOL_CACHE.items():
        by_cache.setdefault(cache, []).append(chosen[pool])
    caches = {
        cache: resident_table(data.caches[cache], np.concatenate(parts))
        for cache, parts in by_cache.items()
    }
    resident = WarmstartData(vocab=data.vocab, corpus=data.corpus, builder=data.builder, caches=caches)
    pools = dict(MultiTaskSampler(resident, cfg, rng).pools)
    for name in SETUP_POOLS:
        pools[name] = chosen[name]
    return HumanAuxSample(data=resident, pools=pools)


def load_human_aux_sample(
    data_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
    *,
    max_rows: int = HUMAN_AUX_MAX_ROWS,
    seed: int = 0,
    workers: int | None = None,
    log: Callable[[str], None] = print,
) -> HumanAuxSample:
    """Load the v2 warm-start tables, keep a resident sample of `max_rows`, drop the rest."""
    data = load_warmstart_data(data_dir, cache_dir, workers=workers, log=log)
    sample = build_human_aux_sample(data, max_rows=max_rows, rng=np.random.default_rng(seed))
    del data
    gc.collect()
    log(
        f"human aux sample: {sample.num_rows:,} rows, "
        f"{sample.cache_nbytes / 1024**3:.2f} GiB of resident cache rows"
    )
    return sample


class HumanAuxLoss:
    """Per-update L_human from a `HumanAuxSample` (#38 multi-task loss, train rows only)."""

    def __init__(
        self,
        sample: HumanAuxSample,
        *,
        rng: np.random.Generator,
        weights: WarmstartConfig | None = None,
    ) -> None:
        self.sample = sample
        self.weights = weights or WarmstartConfig()
        self.sampler = sample.sampler(rng, self.weights)

    def backward(
        self,
        model: nn.Module,
        *,
        weight: float,
        device: torch.device | str = "cpu",
        use_amp: bool = False,
    ) -> dict[str, float]:
        """Back-propagate `weight · L_human` on one fresh minibatch, accumulating gradients.

        Does not zero gradients or step the optimizer. Returns the unweighted per-task
        losses and their weighted-by-task sum `loss_total` (L_human itself).
        """
        dev = torch.device(device)
        amp_enabled = bool(use_amp and dev.type == "cuda")
        amp_dtype = torch.bfloat16 if amp_enabled else torch.float32
        batches = {k: _to_device(v, dev) for k, v in self.sampler.next_step().items()}
        groups = multitask_loss_groups(model, self.weights, batches)
        stats: dict[str, float] = {}
        total = 0.0
        while True:
            with torch.amp.autocast(device_type=dev.type, dtype=amp_dtype, enabled=amp_enabled):
                item = next(groups, None)
            if item is None:
                break
            group_stats, loss = item
            (float(weight) * loss).backward()
            stats.update(group_stats)
            total += float(loss.detach().item())
        if not stats:
            raise ValueError("the human aux sample produced no task batches")
        stats["loss_total"] = total
        return stats
