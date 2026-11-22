"""Shared v2 multi-task warm start for both agents (#46; decisions in #38 §4).

`train_warmstart.py` (main agent, `TMNetwork`) and `comparator/warmstart.py` (comparator,
`ComparatorNetwork`) both run `run_multitask_warmstart` with the one `WarmstartConfig`: the same
tables, per-task minibatches, loss weights, 15,000-step budget, composite checkpoint selection and
validation report. Only the network, its checkpoint format and the comparator's gen 2+ temperature
fit differ, through `AgentSpec`.

Gen 2+ rows come from the encoded cache (`encoded_cache.py`); generation-1 setup rows from the
compact setup tables and `FastBatchBuilder`. Since #62 the gen 2+ tasks include `space` (tile
placement over the legal spaces of the rebuilt board, #61), so training needs `action_spaces.parquet`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields, replace
import json
from pathlib import Path
import queue
import threading
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tm_bot.agents.card_data import load_card_data
from tm_bot.comparator.calibration import calibration_metrics, fit_midgame_temperature
from tm_bot.env.types import SUPPORTED_MAPS
from tm_bot.models.encode import ENCODER_VERSION
from tm_bot.models.network import vp_margin_loss, vp_margin_target
from tm_bot.models.vocab import CardVocab
from tm_bot.training.baselines import evaluate_model
from tm_bot.training.dataset import (
    FastBatchBuilder,
    WarmstartCorpus,
    load_warmstart_corpus,
    resolve_warmstart_dir,
)
from tm_bot.training.encoded_cache import (
    CACHE_TASKS,
    PHASE_ACTION,
    PHASE_RESEARCH,
    EncodedTable,
    concat_obs_batches,
    ensure_encoded_cache,
)

WARMSTART_DATA_VERSION = 2
GEN_BUCKETS: tuple[str, ...] = ("gen 1", "gen 2-5", "gen 6-9", "gen 10+")

# Policy heads: (task, loss kind, config weight). Setup heads share one observation encoding.
SETUP_HEADS: tuple[tuple[str, str, str], ...] = (
    ("corp", "hard", "w_corp"),
    ("prelude", "soft", "w_prelude"),
    ("project", "soft", "w_init_project"),
)
STATE_HEADS: tuple[tuple[str, str, str], ...] = (
    ("draft", "hard", "w_draft"),
    ("research_buy", "soft", "w_research_buy"),
    ("branch", "hard", "w_branch"),
    ("project_play", "hard", "w_project_play"),
    ("standard_project", "hard", "w_standard_project"),
    ("card_action", "hard", "w_card_action"),
    ("space", "hard", "w_space"),
)
FIXED_MENU_TASKS: tuple[str, ...] = (
    "draft", "branch", "project_play", "standard_project", "card_action", "space",
)


@dataclass(frozen=True)
class WarmstartConfig:
    """The one v2 warm-start configuration both agents train with (#38 §4)."""

    # Per-step task minibatches.
    setup_batch_size: int = 256
    draft_batch_size: int = 512
    research_buy_batch_size: int = 512
    branch_batch_size: int = 512
    project_play_batch_size: int = 512
    standard_project_batch_size: int = 256
    card_action_batch_size: int = 256
    # #62: tile placement (~1.1M train rows) takes the band #38 gave the other ~0.5-1.5M-row
    # action sub-choices (standard project, card action): 256 rows, weight 0.40.
    space_batch_size: int = 256
    value_batch_size: int = 512
    # Value minibatch mix: gen-1 setup outcomes / gen 2+ pre-research / gen 2+ action points.
    value_mix_setup: float = 0.20
    value_mix_research: float = 0.40
    value_mix_action: float = 0.40
    # Loss weights.
    w_corp: float = 0.40
    w_prelude: float = 0.40
    w_init_project: float = 0.30
    w_draft: float = 0.80
    w_research_buy: float = 0.70
    w_branch: float = 1.00
    w_project_play: float = 1.00
    w_standard_project: float = 0.40
    w_card_action: float = 0.40
    w_space: float = 0.40
    w_value_win: float = 0.80
    w_value_margin: float = 0.30
    # Step budget and optimizer: 6 x 2,500 = 15,000 steps, cosine LR with warmup, bf16 AMP.
    epochs: int = 6
    steps_per_epoch: int = 2500
    lr: float = 5e-4
    min_lr: float = 3e-5
    warmup_steps: int = 500
    weight_decay: float = 2e-4
    grad_clip: float = 1.0
    use_amp: bool = True
    seed: int = 42
    eval_batch_size: int = 1024
    log_every: int = 250

    @property
    def total_steps(self) -> int:
        return self.epochs * self.steps_per_epoch

    def task_weights(self) -> dict[str, float]:
        return {f.name[2:]: getattr(self, f.name) for f in fields(self) if f.name.startswith("w_")}

    def task_batch_sizes(self) -> dict[str, int]:
        return {
            f.name[: -len("_batch_size")]: getattr(self, f.name)
            for f in fields(self)
            if f.name.endswith("_batch_size") and f.name != "eval_batch_size"
        }

    def value_counts(self) -> tuple[int, int, int]:
        """Rows of the value minibatch per source: (setup, pre-research, action)."""
        n_setup = int(round(self.value_batch_size * self.value_mix_setup))
        n_research = int(round(self.value_batch_size * self.value_mix_research))
        return n_setup, n_research, self.value_batch_size - n_setup - n_research

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["total_steps"] = self.total_steps
        return out


def gen_bucket_ids(generations: np.ndarray) -> np.ndarray:
    """Index into `GEN_BUCKETS` per generation."""
    g = np.asarray(generations, dtype=np.int64)
    return np.select([g <= 1, g <= 5, g <= 9], [0, 1, 2], default=3)


def composite_score(m: Mapping[str, float | None]) -> float:
    """#38 §4 checkpoint-selection score from the validation metrics it names (`space` since #62)."""
    setup = np.mean([m["corp_top1"], m["prelude0_top1"], m["draft_top1"], m["research_buy_acc"]])
    action = np.mean([
        m["branch_top1"], m["project_play_top1"], m["standard_project_top1"], m["card_action_top1"],
        m["space_top1"],
    ])
    brier = m["midgame_val_brier"]
    value = 0.0 if brier is None else max(0.0, 1.0 - float(brier) / 0.25)
    return float(0.25 * setup + 0.50 * action + 0.25 * value)


# --- loss ------------------------------------------------------------------------------------


def soft_cross_entropy(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    cand_mask: torch.Tensor,
) -> torch.Tensor:
    """Numerically safe cross-entropy with soft target distributions over masked candidates."""
    log_probs = F.log_softmax(logits.float(), dim=-1)
    safe_log_probs = torch.where(cand_mask, log_probs, torch.zeros_like(log_probs))
    return -(soft_targets * safe_log_probs).sum(dim=-1).mean()


def _encode(model: nn.Module, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """Observation encoding as a tuple ending in `obs_pooled` (`TMNetwork` also returns tokens)."""
    enc = model.encode_observations(batch)
    return enc if isinstance(enc, tuple) else (enc,)


def _policy_loss(
    model: nn.Module,
    enc: tuple[torch.Tensor, ...],
    batch: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    kind: str,
) -> torch.Tensor:
    logits, _ = model.forward_policy(*enc, batch)
    if kind == "hard":
        return F.cross_entropy(logits.float(), targets)
    return soft_cross_entropy(logits, targets, batch["cand_mask"].bool())


_BATCH_KEYS = frozenset(
    [f"{name}_{part}" for name, _, _ in (*SETUP_HEADS, *STATE_HEADS) for part in ("batch", "targets")]
    + ["outcome_batch", "outcome_targets", "outcome_margins"]
)


def _loss_groups(
    model: nn.Module, cfg: WarmstartConfig, batches: Mapping[str, Any]
) -> Iterator[tuple[dict[str, float], torch.Tensor]]:
    """Yield `(stats, weighted loss)` per forward group: the shared-encoding setup heads, each gen 2+
    task, then the value heads. Each group's graph is independent of the others'."""
    setup_present = [
        (name, kind, w) for name, kind, w in SETUP_HEADS
        if batches.get(f"{name}_batch") is not None and batches.get(f"{name}_targets") is not None
    ]
    if setup_present:
        enc = _encode(model, batches[f"{setup_present[0][0]}_batch"])
        stats: dict[str, float] = {}
        total: torch.Tensor | None = None
        for name, kind, w in setup_present:
            loss = _policy_loss(model, enc, batches[f"{name}_batch"], batches[f"{name}_targets"], kind)
            stats[f"loss_{name}"] = float(loss.detach().item())
            total = getattr(cfg, w) * loss if total is None else total + getattr(cfg, w) * loss
        yield stats, total  # type: ignore[misc]

    for name, kind, w in STATE_HEADS:
        batch, targets = batches.get(f"{name}_batch"), batches.get(f"{name}_targets")
        if batch is not None and targets is not None:
            loss = _policy_loss(model, _encode(model, batch), batch, targets, kind)
            yield {f"loss_{name}": float(loss.detach().item())}, getattr(cfg, w) * loss

    value_batch, won = batches.get("outcome_batch"), batches.get("outcome_targets")
    if value_batch is not None and won is not None:
        pooled = _encode(model, value_batch)[-1]
        logit = model.forward_value(pooled).float()
        lv = F.binary_cross_entropy_with_logits(logit, won.float())
        stats = {"loss_value": float(lv.detach().item())}
        total = cfg.w_value_win * lv
        margins = batches.get("outcome_margins")
        head = getattr(model, "vp_margin_head", None)
        if margins is not None and head is not None:
            valid = torch.isfinite(margins)
            if bool(valid.any()):
                pred = head(pooled).float()
                lm = vp_margin_loss(pred[valid], vp_margin_target(margins[valid].float(), 0.0))
                stats["loss_value_margin"] = float(lm.detach().item())
                total = total + cfg.w_value_margin * lm
        yield stats, total


def _check_batch_keys(batches: Mapping[str, Any]) -> None:
    unknown = set(batches) - _BATCH_KEYS
    if unknown:
        raise TypeError(f"unknown task batches: {sorted(unknown)}")


def compute_multitask_step_losses(
    model: nn.Module,
    *,
    weights: WarmstartConfig | None = None,
    **batches: Any,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted multi-task warm-start loss from any subset of task batches.

    Keys are `<task>_batch` / `<task>_targets` for the tasks in `SETUP_HEADS` and `STATE_HEADS`
    (`project` is the initial project buy), plus `outcome_batch`, `outcome_targets` (won) and
    `outcome_margins` (raw final VP margin, NaN where unknown; such rows get no margin loss) for the
    value heads. The setup batches come from the same rows and share one `encode_observations` call.
    """
    _check_batch_keys(batches)
    total: torch.Tensor | None = None
    stats: dict[str, float] = {}
    for group_stats, loss in _loss_groups(model, weights or WarmstartConfig(), batches):
        stats.update(group_stats)
        total = loss if total is None else total + loss
    if total is None:
        raise ValueError("At least one task batch must be provided")
    stats["loss_total"] = float(total.detach().item())
    return total, stats


def multitask_loss_groups(
    model: nn.Module, weights: WarmstartConfig | None, batches: Mapping[str, Any]
) -> Iterator[tuple[dict[str, float], torch.Tensor]]:
    """`(stats, weighted loss)` per independent forward group of `compute_multitask_step_losses`.

    Back-propagating each group as it is yielded gives the summed loss's gradient with one
    group's activations alive at a time (as `train_one_step` does).
    """
    _check_batch_keys(batches)
    return _loss_groups(model, weights or WarmstartConfig(), batches)


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Mapping):
        return {k: v.to(device, non_blocking=True) for k, v in value.items()}
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    return value


def train_one_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    use_amp: bool = True,
    grad_clip: float = 1.0,
    weights: WarmstartConfig | None = None,
    **batches: Any,
) -> dict[str, float]:
    """One optimizer step on the supplied task batches (see `compute_multitask_step_losses`)."""
    model.train()
    amp_enabled = use_amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32
    _check_batch_keys(batches)
    moved = {k: _to_device(v, device) for k, v in batches.items()}
    optimizer.zero_grad(set_to_none=True)
    # Back-propagate each forward group as soon as it is computed: the accumulated gradient is
    # that of the summed loss, while only one group's activations are alive at a time.
    groups = _loss_groups(model, weights or WarmstartConfig(), moved)
    stats: dict[str, float] = {}
    total = 0.0
    while True:
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            item = next(groups, None)
        if item is None:
            break
        group_stats, loss = item
        loss.backward()
        stats.update(group_stats)
        total += float(loss.detach().item())
    if not stats:
        raise ValueError("At least one task batch must be provided")
    stats["loss_total"] = total
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return stats


# --- data and sampling ---------------------------------------------------------------------


@dataclass
class WarmstartData:
    """Setup tables (generation 1) plus the encoded gen 2+ task caches."""

    vocab: CardVocab
    corpus: WarmstartCorpus
    builder: FastBatchBuilder
    caches: dict[str, EncodedTable]

    def table_sizes(self) -> dict[str, dict[str, int]]:
        sizes = {
            "setup_choices": self.corpus.setup_choices.split_indices,
            "setup_outcomes": self.corpus.setup_outcomes.split_indices,
            **{name: c.split_indices for name, c in self.caches.items()},
        }
        return {name: {s: int(len(idx)) for s, idx in split.items()} for name, split in sizes.items()}


def default_cache_dir(data_dir: Path) -> Path:
    """`<data_dir>_cache`, a sibling of the read-only parquet directory."""
    return data_dir.with_name(data_dir.name + "_cache")


def load_warmstart_data(
    data_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
    *,
    tasks: Sequence[str] | None = None,
    workers: int | None = None,
    log: Callable[[str], None] = print,
) -> WarmstartData:
    """Setup tables plus the encoded caches of `tasks` (default: every cached task, `space` included;
    a missing parquet table raises instead of being skipped)."""
    data_path = resolve_warmstart_dir(data_dir)
    cards = load_card_data()
    vocab = CardVocab.from_card_data(cards)
    caches = ensure_encoded_cache(
        data_path, Path(cache_dir) if cache_dir is not None else default_cache_dir(data_path),
        tasks=tuple(tasks) if tasks is not None else tuple(CACHE_TASKS), vocab=vocab, workers=workers, log=log,
    )
    corpus = load_warmstart_corpus(data_path, vocab=vocab, load_drafts=False)
    return WarmstartData(vocab=vocab, corpus=corpus, builder=FastBatchBuilder(vocab, cards), caches=caches)


def _midgame_rows(data: WarmstartData, split: str, phase: int) -> np.ndarray:
    mv = data.caches["midgame_value"]
    idx = mv.split_indices[split]
    return idx[mv["phase"][idx] == phase]


class MultiTaskSampler:
    """Draws every optimizer step's dedicated per-task minibatches from the train split."""

    def __init__(self, data: WarmstartData, cfg: WarmstartConfig, rng: np.random.Generator) -> None:
        self.data = data
        self.cfg = cfg
        self.rng = rng
        self.pools: dict[str, np.ndarray] = {
            "setup": data.corpus.setup_choices.split_indices["train"],
            "value_setup": data.corpus.setup_outcomes.split_indices["train"],
            "value_research": _midgame_rows(data, "train", PHASE_RESEARCH),
            "value_action": _midgame_rows(data, "train", PHASE_ACTION),
            **{name: data.caches[name].split_indices["train"] for name, _, _ in STATE_HEADS},
        }
        self.perms: dict[str, np.ndarray] = {}
        self.cursors: dict[str, int] = {}

    def take(self, pool: str, n: int) -> np.ndarray:
        """`n` train rows of `pool`, walking a fresh permutation per pass (with replacement if tiny)."""
        rows = self.pools[pool]
        if n <= 0 or len(rows) == 0:
            return np.zeros((0,), dtype=np.int64)
        if len(rows) < n:
            return self.rng.choice(rows, size=n, replace=True)
        perm = self.perms.get(pool)
        cur = self.cursors.get(pool, 0)
        if perm is None or cur + n > len(perm):
            perm = self.rng.permutation(rows)
            self.perms[pool] = perm
            cur = 0
        self.cursors[pool] = cur + n
        return perm[cur : cur + n]

    def next_step(self) -> dict[str, Any]:
        cfg, data = self.cfg, self.data
        out: dict[str, Any] = {}
        sc = data.corpus.setup_choices
        s_idx = self.take("setup", cfg.setup_batch_size)
        if len(s_idx):
            out["corp_batch"], out["corp_targets"] = data.builder.build_corp_batch(sc, s_idx)
            out["prelude_batch"], out["prelude_targets"] = data.builder.build_prelude_batch(sc, s_idx, rng=self.rng)
            out["project_batch"], out["project_targets"] = data.builder.build_project_buy_batch(
                sc, s_idx, rng=self.rng
            )
        sizes = cfg.task_batch_sizes()
        for name in FIXED_MENU_TASKS:
            idx = self.take(name, sizes[name])
            if len(idx):
                out[f"{name}_batch"], out[f"{name}_targets"] = data.caches[name].policy_batch(idx)
        idx = self.take("research_buy", sizes["research_buy"])
        if len(idx):
            out["research_buy_batch"], out["research_buy_targets"] = data.caches["research_buy"].research_batch(
                idx, rng=self.rng
            )

        n_setup, n_research, n_action = cfg.value_counts()
        parts: list[dict[str, torch.Tensor]] = []
        won: list[torch.Tensor] = []
        margins: list[torch.Tensor] = []
        so = data.corpus.setup_outcomes
        v_idx = self.take("value_setup", n_setup)
        if len(v_idx):
            b, w = data.builder.build_outcome_batch(so, v_idx)
            parts.append(b)
            won.append(w)
            margins.append(_setup_margins(so, v_idx))
        m_idx = np.concatenate([self.take("value_research", n_research), self.take("value_action", n_action)])
        if len(m_idx):
            b, w, m = data.caches["midgame_value"].value_batch(m_idx)
            parts.append(b)
            won.append(w)
            margins.append(m)
        if parts:
            out["outcome_batch"] = concat_obs_batches(parts)
            out["outcome_targets"] = torch.cat(won)
            out["outcome_margins"] = torch.cat(margins)
        return out

    def state_dict(self) -> dict[str, Any]:
        return {"rng": self.rng.bit_generator.state, "perms": self.perms, "cursors": self.cursors}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.rng.bit_generator.state = state["rng"]
        self.perms = dict(state["perms"])
        self.cursors = dict(state["cursors"])


def _setup_margins(so: Any, idx: np.ndarray) -> torch.Tensor:
    if so.vp_margin is None:
        return torch.full((len(idx),), float("nan"))
    return torch.from_numpy(so.vp_margin[idx].astype(np.float32))


def _prefetch(sampler: MultiTaskSampler, n: int, depth: int = 3) -> Iterator[dict[str, Any]]:
    """Yield `n` steps from `sampler`, assembled one step ahead on a background thread."""
    q: queue.Queue[Any] = queue.Queue(maxsize=depth)

    def work() -> None:
        try:
            for _ in range(n):
                q.put(sampler.next_step())
        except BaseException as exc:  # surfaced in the training thread
            q.put(exc)

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    for _ in range(n):
        item = q.get()
        if isinstance(item, BaseException):
            raise item
        yield item
    thread.join()


# --- evaluation ------------------------------------------------------------------------------


def _grouped(correct: np.ndarray, map_ids: np.ndarray, generations: np.ndarray) -> dict[str, Any]:
    """Accuracy overall, per supported map and per generation bucket."""

    def entry(mask: np.ndarray) -> dict[str, Any]:
        n = int(mask.sum())
        ok = int(correct[mask].sum())
        return {"correct": ok, "total": n, "acc": (ok / n) if n else 0.0}

    buckets = gen_bucket_ids(generations)
    every = np.ones(correct.shape[0], dtype=bool)
    return {
        "overall": entry(every),
        "per_map": {m: entry(map_ids == i) for i, m in enumerate(SUPPORTED_MAPS)},
        "per_gen": {b: entry(buckets == i) for i, b in enumerate(GEN_BUCKETS)},
    }


def _autocast(device: torch.device, use_amp: bool) -> Any:
    enabled = use_amp and device.type == "cuda"
    return torch.amp.autocast(
        device_type=device.type, dtype=torch.bfloat16 if enabled else torch.float32, enabled=enabled
    )


def _policy_logits(model: nn.Module, batch: Mapping[str, torch.Tensor], device: torch.device, use_amp: bool) -> np.ndarray:
    with _autocast(device, use_amp):
        out = model(_to_device(batch, device), compute_policy=True, compute_value=False)
    return out["policy_logits"].float().cpu().numpy()


@torch.inference_mode()
def evaluate_state_task(
    model: nn.Module,
    table: EncodedTable,
    rows: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 1024,
    use_amp: bool = True,
) -> dict[str, Any]:
    """Top-1 of a fixed-menu gen 2+ policy task (research buys: step-0 and done-stop accuracy)."""
    research = table.task.kind == "research"
    hits: list[np.ndarray] = []
    done_hits: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        b_idx = rows[start : start + batch_size]
        if research:
            batch, soft = table.research_batch(b_idx, picked=[[] for _ in b_idx])
            pred = _policy_logits(model, batch, device, use_amp).argmax(axis=-1)
            hits.append(soft.numpy()[np.arange(len(b_idx)), pred] > 0.0)
            kept = table["kept_mask"][b_idx]
            all_kept = [[j for j in range(8) if (int(k) >> j) & 1] for k in kept]
            batch_d, soft_d = table.research_batch(b_idx, picked=all_kept)
            pred_d = _policy_logits(model, batch_d, device, use_amp).argmax(axis=-1)
            done_hits.append(pred_d == soft_d.numpy().argmax(axis=-1))
        else:
            batch, target = table.policy_batch(b_idx)
            pred = _policy_logits(model, batch, device, use_amp).argmax(axis=-1)
            hits.append(pred == target.numpy())
    maps, gens = table["map_id"][rows], table["generation"][rows]
    cat = (lambda xs: np.concatenate(xs) if xs else np.zeros((0,), dtype=bool))
    result = _grouped(cat(hits), maps, gens)
    if research:
        result = {"step0": result, "done": _grouped(cat(done_hits), maps, gens)}
    return result


@torch.inference_mode()
def collect_value_outputs(
    model: nn.Module,
    data: WarmstartData,
    split: str,
    *,
    device: torch.device,
    batch_size: int = 1024,
    use_amp: bool = True,
) -> dict[str, np.ndarray]:
    """Value logits, margin predictions and row labels of a split's setup-outcome and mid-game rows.

    Setup outcomes are generation 1 (`phase` -1); mid-game rows carry their phase code.
    """
    cols: dict[str, list[np.ndarray]] = {k: [] for k in ("logit", "won", "margin", "margin_pred", "generation", "map_id", "phase")}

    def run(batch: Mapping[str, torch.Tensor]) -> None:
        with _autocast(device, use_amp):
            out = model(_to_device(batch, device), compute_policy=False, compute_value=True)
        cols["logit"].append(out["value_logit"].float().cpu().numpy())
        pred = out.get("vp_margin_pred")
        n = out["value_logit"].shape[0]
        cols["margin_pred"].append(pred.float().cpu().numpy() if pred is not None else np.full(n, np.nan))

    so = data.corpus.setup_outcomes
    idx = so.split_indices[split]
    for start in range(0, len(idx), batch_size):
        b_idx = idx[start : start + batch_size]
        batch, won = data.builder.build_outcome_batch(so, b_idx)
        run(batch)
        cols["won"].append(won.numpy())
        cols["margin"].append(_setup_margins(so, b_idx).numpy())
        cols["generation"].append(np.ones(len(b_idx), dtype=np.int64))
        cols["map_id"].append(so.map_id[b_idx].astype(np.int64))
        cols["phase"].append(np.full(len(b_idx), -1, dtype=np.int64))
    mv = data.caches["midgame_value"]
    idx = mv.split_indices[split]
    for start in range(0, len(idx), batch_size):
        b_idx = idx[start : start + batch_size]
        batch, won, margin = mv.value_batch(b_idx)
        run(batch)
        cols["won"].append(won.numpy())
        cols["margin"].append(margin.numpy())
        cols["generation"].append(mv["generation"][b_idx].astype(np.int64))
        cols["map_id"].append(mv["map_id"][b_idx].astype(np.int64))
        cols["phase"].append(mv["phase"][b_idx].astype(np.int64))
    return {k: (np.concatenate(v) if v else np.zeros((0,))) for k, v in cols.items()}


def value_report(values: Mapping[str, np.ndarray], *, temperature: float = 1.0) -> dict[str, Any]:
    """Value Brier / ECE / accuracy by generation bucket, map and phase, at `temperature`."""
    probs = 1.0 / (1.0 + np.exp(-np.asarray(values["logit"], dtype=np.float64) / temperature))
    won = np.asarray(values["won"], dtype=np.float64)
    gens = np.asarray(values["generation"])
    buckets = gen_bucket_ids(gens)

    def metrics(mask: np.ndarray) -> dict[str, Any]:
        m = calibration_metrics(probs[mask], won[mask])
        m.pop("bins", None)
        return m

    mid = gens >= 2
    margin = np.asarray(values["margin"], dtype=np.float64)
    pred = np.asarray(values["margin_pred"], dtype=np.float64)
    has_margin = mid & np.isfinite(margin) & np.isfinite(pred)
    target = np.clip(margin[has_margin], -60.0, 60.0) / 50.0
    phases = np.asarray(values["phase"])
    return {
        "temperature": float(temperature),
        "per_gen": {b: metrics(buckets == i) for i, b in enumerate(GEN_BUCKETS)},
        "midgame": metrics(mid),
        "midgame_by_phase": {
            "research": metrics(mid & (phases == PHASE_RESEARCH)),
            "action": metrics(mid & (phases == PHASE_ACTION)),
        },
        "midgame_per_map": {m: metrics(mid & (values["map_id"] == i)) for i, m in enumerate(SUPPORTED_MAPS)},
        "midgame_margin_mae": float(np.abs(pred[has_margin] - target).mean()) if has_margin.any() else None,
    }


def evaluate_warmstart(
    model: nn.Module,
    data: WarmstartData,
    split: str,
    *,
    device: torch.device,
    batch_size: int = 1024,
    use_amp: bool = True,
) -> dict[str, Any]:
    """Every task's validation metrics, the value report and the composite selection score."""
    was_training = model.training
    model.eval()
    setup = evaluate_model(
        model, data.corpus, data.builder, split, device=device, batch_size=batch_size, use_amp=use_amp
    )
    metrics: dict[str, Any] = {
        key: setup[key] for key in ("corp", "prelude_top1", "prelude_pair", "prelude_step1_cond", "project_buy")
    }
    for name, _, _ in STATE_HEADS:
        table = data.caches[name]
        metrics[name] = evaluate_state_task(
            model, table, table.split_indices[split], device=device, batch_size=batch_size, use_amp=use_amp
        )
    values = collect_value_outputs(model, data, split, device=device, batch_size=batch_size, use_amp=use_amp)
    metrics["value"] = value_report(values)
    inputs = {
        "corp_top1": metrics["corp"]["overall"]["acc"],
        "prelude0_top1": metrics["prelude_top1"]["overall"]["acc"],
        "draft_top1": metrics["draft"]["overall"]["acc"],
        "research_buy_acc": metrics["research_buy"]["step0"]["overall"]["acc"],
        "branch_top1": metrics["branch"]["overall"]["acc"],
        "project_play_top1": metrics["project_play"]["overall"]["acc"],
        "standard_project_top1": metrics["standard_project"]["overall"]["acc"],
        "card_action_top1": metrics["card_action"]["overall"]["acc"],
        "space_top1": metrics["space"]["overall"]["acc"],
        "midgame_val_brier": metrics["value"]["midgame"]["brier"],
    }
    metrics["composite"] = {"score": composite_score(inputs), "inputs": inputs}
    if was_training:
        model.train()
    return metrics


# --- reporting -------------------------------------------------------------------------------

_POLICY_ROWS: tuple[tuple[str, str], ...] = (
    ("Corporation", "corp"),
    ("Prelude (step-0 top-1)", "prelude_top1"),
    ("Draft pick", "draft"),
    ("Research buy (step-0)", "research_buy.step0"),
    ("Research buy (done-stop)", "research_buy.done"),
    ("Root branch", "branch"),
    ("Project card play", "project_play"),
    ("Standard project", "standard_project"),
    ("Card action", "card_action"),
    ("Tile placement (space)", "space"),
)


def _lookup(metrics: Mapping[str, Any], dotted: str) -> Mapping[str, Any]:
    out: Any = metrics
    for part in dotted.split("."):
        out = out[part]
    return out


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def format_report(results: Mapping[str, Any]) -> str:
    """Markdown validation report: composite score, per-task top-1 by map and generation, value."""
    val = results["val"]
    comp = val["composite"]
    lines = [
        f"## {results['version']} validation report",
        "",
        f"- Composite score: **{comp['score']:.4f}** (best epoch {results['best_epoch']}, "
        f"step {results['best_step']:,} of {results['train_config']['total_steps']:,})",
        "- Inputs: " + ", ".join(f"{k}=`{_fmt(v)}`" for k, v in comp["inputs"].items()),
        f"- Parameters: `{results['parameter_count']:,}`; train wall-clock `{results['train_wall_clock_sec'] / 60.0:.1f} min`",
        "",
        "### Policy top-1",
        "",
        "| Task | Overall | " + " | ".join(SUPPORTED_MAPS) + " | " + " | ".join(GEN_BUCKETS) + " |",
        "| --- | ---: | " + " | ".join("---:" for _ in (*SUPPORTED_MAPS, *GEN_BUCKETS)) + " |",
    ]
    for label, key in _POLICY_ROWS:
        m = _lookup(val, key)
        cells = [f"{m['overall']['acc']:.4f} (n={m['overall']['total']:,})"]
        cells += [f"{m['per_map'][mp]['acc']:.4f}" if m["per_map"][mp]["total"] else "-" for mp in SUPPORTED_MAPS]
        per_gen = m.get("per_gen")
        if per_gen is None:  # setup tasks are all generation 1
            cells += [cells[0].split(" ")[0]] + ["-"] * (len(GEN_BUCKETS) - 1)
        else:
            cells += [f"{per_gen[b]['acc']:.4f}" if per_gen[b]["total"] else "-" for b in GEN_BUCKETS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    pj = val["project_buy"]
    lines += [
        "",
        f"- Initial project buy: step-0 `{pj['step0_acc']:.4f}`, done-stop `{pj['done_acc']:.4f}` (n={pj['total']:,})",
    ]
    for title, rep in (("raw", val["value"]), ("calibrated", results.get("value_calibrated"))):
        if rep is None:
            continue
        lines += [
            "",
            f"### Value head ({title}, temperature {rep['temperature']:.4f})",
            "",
            "| Scope | n | Accuracy | Brier | ECE |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        scopes = [(b, rep["per_gen"][b]) for b in GEN_BUCKETS]
        scopes += [("gen 2+ (all)", rep["midgame"])]
        scopes += [(f"gen 2+ {k}", v) for k, v in rep["midgame_by_phase"].items()]
        scopes += [(f"gen 2+ {k}", v) for k, v in rep["midgame_per_map"].items()]
        for name, m in scopes:
            lines.append(f"| {name} | {m['n']:,} | {_fmt(m['accuracy'])} | {_fmt(m['brier'])} | {_fmt(m['ece'])} |")
        lines.append(f"\n- Gen 2+ VP-margin MAE (normalized target): `{_fmt(rep['midgame_margin_mae'])}`")
    return "\n".join(lines) + "\n"


# --- the shared run loop ---------------------------------------------------------------------


@dataclass(frozen=True)
class AgentSpec:
    """What differs between the two agents' warm starts: network, checkpoint format, calibration."""

    name: str
    version: str
    default_out_dir: Path
    build_model: Callable[[CardVocab], nn.Module]
    save_checkpoint: Callable[[Path, nn.Module, CardVocab, dict[str, Any]], Any]
    load_checkpoint: Callable[[Path, torch.device], nn.Module]
    calibrate_temperature: bool = False


def _lr_lambda(cfg: WarmstartConfig) -> Callable[[int], float]:
    def fn(step: int) -> float:
        if step < cfg.warmup_steps:
            return float(step + 1) / float(max(1, cfg.warmup_steps))
        progress = float(step - cfg.warmup_steps) / float(max(1, cfg.total_steps - cfg.warmup_steps))
        progress = min(1.0, max(0.0, progress))
        min_ratio = cfg.min_lr / cfg.lr
        return float(min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + np.cos(np.pi * progress)))

    return fn


def _checkpoint_metadata(
    spec: AgentSpec, cfg: WarmstartConfig, model: nn.Module, data: WarmstartData, record: Mapping[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    return {
        "version": spec.version,
        "agent": spec.name,
        "warmstart_data_version": WARMSTART_DATA_VERSION,
        "encoder_version": ENCODER_VERSION,
        "best_epoch": record["epoch"],
        "best_step": record["global_step"],
        "val_composite": record["val"]["composite"]["score"],
        "val_composite_inputs": record["val"]["composite"]["inputs"],
        "train_config": cfg.to_dict(),
        "model_config": model.config.to_dict(),
        "table_sizes": data.table_sizes(),
        **extra,
    }


def run_multitask_warmstart(
    spec: AgentSpec,
    *,
    data_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
    out_dir: Path | str | None = None,
    cfg: WarmstartConfig | None = None,
    device_str: str | None = None,
    resume: bool = False,
    epochs_per_run: int | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    """Train one agent's v2 warm start, keep the best composite-score checkpoint, report on val.

    The test split is never evaluated here; it stays reserved for the later gates.
    """
    cfg = cfg or WarmstartConfig()
    out_path = Path(out_dir) if out_dir is not None else spec.default_out_dir
    out_path.mkdir(parents=True, exist_ok=True)
    metrics_path = out_path / "metrics.json"
    state_path = out_path / "latest_state.pt"
    best_dir = out_path / "best"
    if metrics_path.exists():
        prior = json.loads(metrics_path.read_text(encoding="utf-8"))
        if "test_model" in prior:
            raise RuntimeError(
                f"The test split was already evaluated for the run in {out_path}; "
                "use a fresh --out-dir instead of re-running or resuming a finished run."
            )
        if prior.get("status") == "completed":
            raise RuntimeError(f"The run in {out_path} already completed; use a fresh --out-dir.")

    torch.manual_seed(cfg.seed)
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    t_start = time.perf_counter()
    data = load_warmstart_data(data_dir, cache_dir, workers=workers)
    model = spec.build_model(data.vocab).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[{spec.name}] {spec.version} on {device}: {param_count:,} parameters; tables {data.table_sizes()}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda(cfg))
    sampler = MultiTaskSampler(data, cfg, np.random.default_rng(cfg.seed))

    start_epoch, global_step, train_sec = 1, 0, 0.0
    best_score, best_epoch = -1.0, 0
    history: list[dict[str, Any]] = []
    if resume and state_path.exists():
        st = torch.load(state_path, map_location=device, weights_only=False)
        if st["train_config"] != cfg.to_dict():
            raise ValueError("resume config differs from the saved run's config")
        model.load_state_dict(st["model_state_dict"])
        optimizer.load_state_dict(st["optimizer_state_dict"])
        scheduler.load_state_dict(st["scheduler_state_dict"])
        sampler.load_state_dict(st["sampler_state"])
        torch.set_rng_state(st["torch_rng_state"])
        start_epoch = int(st["completed_epoch"]) + 1
        global_step = int(st["global_step"])
        train_sec = float(st["train_sec"])
        best_score, best_epoch = float(st["best_score"]), int(st["best_epoch"])
        history = list(st["history"])
        print(f"[{spec.name}] resumed after epoch {start_epoch - 1}/{cfg.epochs}", flush=True)

    end_epoch = cfg.epochs if epochs_per_run is None else min(cfg.epochs, start_epoch + epochs_per_run - 1)
    for epoch in range(start_epoch, end_epoch + 1):
        ep_t0 = time.perf_counter()
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        win_t0 = time.perf_counter()
        for step_batches in _prefetch(sampler, cfg.steps_per_epoch):
            stats = train_one_step(
                model, optimizer, device=device, use_amp=cfg.use_amp, grad_clip=cfg.grad_clip,
                weights=cfg, **step_batches,
            )
            scheduler.step()
            global_step += 1
            for k, v in stats.items():
                sums[k] = sums.get(k, 0.0) + v
                counts[k] = counts.get(k, 0) + 1
            if cfg.log_every and global_step % cfg.log_every == 0:
                rate = cfg.log_every / max(1e-9, time.perf_counter() - win_t0)
                win_t0 = time.perf_counter()
                print(
                    f"[{spec.name}] step {global_step}/{cfg.total_steps} {rate:.2f} it/s "
                    f"lr={scheduler.get_last_lr()[0]:.2e} loss={stats['loss_total']:.4f}",
                    flush=True,
                )
        ep_train = time.perf_counter() - ep_t0
        train_sec += ep_train
        val = evaluate_warmstart(
            model, data, "val", device=device, batch_size=cfg.eval_batch_size, use_amp=cfg.use_amp
        )
        score = val["composite"]["score"]
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "epoch_train_sec": ep_train,
            "train_loss": {k: sums[k] / counts[k] for k in sums},
            "val": val,
        }
        history.append(record)
        print(
            f"[{spec.name}] epoch {epoch}/{cfg.epochs} ({ep_train:.0f}s) composite={score:.4f} "
            + " ".join(f"{k}={_fmt(v)}" for k, v in val["composite"]["inputs"].items()),
            flush=True,
        )
        if score > best_score:
            best_score, best_epoch = score, epoch
            spec.save_checkpoint(best_dir, model, data.vocab, _checkpoint_metadata(spec, cfg, model, data, record))
        torch.save(
            {
                "completed_epoch": epoch,
                "global_step": global_step,
                "train_sec": train_sec,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "history": history,
                "train_config": cfg.to_dict(),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "sampler_state": sampler.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
            },
            state_path,
        )

    if end_epoch < cfg.epochs:
        print(f"[{spec.name}] paused after epoch {end_epoch}/{cfg.epochs}; re-run with --resume.", flush=True)
        return {"status": "paused", "completed_epoch": end_epoch, "best_epoch": best_epoch, "best_score": best_score}

    best = history[best_epoch - 1]
    best_model = spec.load_checkpoint(best_dir, device)
    extra: dict[str, Any] = {}
    value_calibrated = None
    if spec.calibrate_temperature:
        values = collect_value_outputs(
            best_model, data, "val", device=device, batch_size=cfg.eval_batch_size, use_amp=cfg.use_amp
        )
        fit = fit_midgame_temperature(values["logit"], values["won"], values["generation"])
        value_calibrated = value_report(values, temperature=fit["temperature"])
        extra = {"temperature": fit["temperature"], "val_calibration": fit}
    results: dict[str, Any] = {
        "status": "completed",
        "version": spec.version,
        "agent": spec.name,
        "best_epoch": best_epoch,
        "best_step": best["global_step"],
        "checkpoint_path": str((best_dir / "checkpoint.pt").resolve()),
        "parameter_count": param_count,
        "train_wall_clock_sec": train_sec,
        "total_wall_clock_sec": time.perf_counter() - t_start,
        "device": str(device),
        "train_config": cfg.to_dict(),
        "model_config": best_model.config.to_dict(),
        "table_sizes": data.table_sizes(),
        "val": best["val"],
        "value_calibrated": value_calibrated,
        "history": history,
        **extra,
    }
    report = format_report(results)
    meta = _checkpoint_metadata(spec, cfg, best_model, data, best, **extra)
    meta["report"] = report
    spec.save_checkpoint(best_dir, best_model, data.vocab, meta)
    (best_dir / "report.md").write_text(report, encoding="utf-8")
    (out_path / "report.md").write_text(report, encoding="utf-8")
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(report, flush=True)
    return results


# --- shared command line ---------------------------------------------------------------------


def add_cli_args(parser: argparse.ArgumentParser, *, default_out_dir: Path) -> None:
    parser.add_argument("--data-dir", type=Path, default=None, help="v2 warm-start parquet directory.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Encoded cache (default <data-dir>_cache).")
    parser.add_argument("--out-dir", type=Path, default=default_out_dir)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--workers", type=int, default=None, help="Cache-building processes.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs-per-run", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override (smoke runs only).")
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="Override (smoke runs only).")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")


def config_from_args(args: argparse.Namespace) -> WarmstartConfig:
    overrides: dict[str, Any] = {
        k: getattr(args, k) for k in ("epochs", "steps_per_epoch", "seed") if getattr(args, k) is not None
    }
    if args.no_amp:
        overrides["use_amp"] = False
    return replace(WarmstartConfig(), **overrides)


def run_from_args(spec: AgentSpec, args: argparse.Namespace) -> dict[str, Any]:
    return run_multitask_warmstart(
        spec,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        out_dir=args.out_dir,
        cfg=config_from_args(args),
        device_str=args.device,
        resume=args.resume,
        epochs_per_run=args.epochs_per_run,
        workers=args.workers,
    )
