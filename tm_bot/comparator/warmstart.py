"""Supervised warm start for the comparator on the same human data and split."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tm_bot.agents.card_data import load_card_data
from tm_bot.comparator.model import (
    ComparatorConfig,
    ComparatorNetwork,
    load_comparator_checkpoint,
    save_comparator_checkpoint,
)
from tm_bot.comparator.calibration import fit_temperature
from tm_bot.models.vocab import CardVocab
from tm_bot.training.baselines import FrequencyBaseline, evaluate_model
from tm_bot.training.dataset import (
    FastBatchBuilder,
    load_warmstart_corpus,
    resolve_warmstart_dir,
)

DEFAULT_OUT_DIR = Path("runs/comparator-warmstart")


@dataclass
class ComparatorTrainConfig:
    """Hyperparameters for comparator warm-start training."""

    epochs: int = 2
    draft_batch_size: int = 1024
    setup_batch_size: int = 256
    value_batch_size: int = 128
    value_every_n_steps: int = 2
    lr: float = 5e-4
    min_lr: float = 3e-5
    weight_decay: float = 2e-4
    grad_clip: float = 1.0
    warmup_steps: int = 100
    w_draft: float = 1.0
    w_corp: float = 0.6
    w_prelude: float = 0.6
    w_project: float = 0.35
    w_value: float = 0.25
    seed: int = 42
    eval_batch_size: int = 1024
    use_amp: bool = True
    max_steps_per_epoch: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def soft_cross_entropy(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    cand_mask: torch.Tensor,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    safe = torch.where(cand_mask, log_probs, torch.zeros_like(log_probs))
    return -(soft_targets * safe).sum(dim=-1).mean()


def comparator_multitask_losses(
    model: ComparatorNetwork,
    *,
    draft_batch: Mapping[str, torch.Tensor] | None = None,
    draft_targets: torch.Tensor | None = None,
    corp_batch: Mapping[str, torch.Tensor] | None = None,
    corp_targets: torch.Tensor | None = None,
    prelude_batch: Mapping[str, torch.Tensor] | None = None,
    prelude_targets: torch.Tensor | None = None,
    project_batch: Mapping[str, torch.Tensor] | None = None,
    project_targets: torch.Tensor | None = None,
    outcome_batch: Mapping[str, torch.Tensor] | None = None,
    outcome_targets: torch.Tensor | None = None,
    weights: ComparatorTrainConfig | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted multi-task warm-start loss using the comparator's own forward pass."""
    cfg = weights or ComparatorTrainConfig()
    losses: list[torch.Tensor] = []
    stats: dict[str, float] = {}

    if draft_batch is not None and draft_targets is not None:
        out = model(draft_batch, compute_policy=True, compute_value=False)
        loss = F.cross_entropy(out["policy_logits"].float(), draft_targets)
        losses.append(cfg.w_draft * loss)
        stats["loss_draft"] = float(loss.detach().item())

    if corp_batch is not None and corp_targets is not None:
        out = model(corp_batch, compute_policy=True, compute_value=False)
        loss = F.cross_entropy(out["policy_logits"].float(), corp_targets)
        losses.append(cfg.w_corp * loss)
        stats["loss_corp"] = float(loss.detach().item())

    if prelude_batch is not None and prelude_targets is not None:
        out = model(prelude_batch, compute_policy=True, compute_value=False)
        loss = soft_cross_entropy(
            out["policy_logits"], prelude_targets, prelude_batch["cand_mask"].bool()
        )
        losses.append(cfg.w_prelude * loss)
        stats["loss_prelude"] = float(loss.detach().item())

    if project_batch is not None and project_targets is not None:
        out = model(project_batch, compute_policy=True, compute_value=False)
        loss = soft_cross_entropy(
            out["policy_logits"], project_targets, project_batch["cand_mask"].bool()
        )
        losses.append(cfg.w_project * loss)
        stats["loss_project"] = float(loss.detach().item())

    if outcome_batch is not None and outcome_targets is not None:
        out = model(outcome_batch, compute_policy=False, compute_value=True)
        loss = F.binary_cross_entropy_with_logits(
            out["value_logit"].float(), outcome_targets
        )
        losses.append(cfg.w_value * loss)
        stats["loss_value"] = float(loss.detach().item())

    if not losses:
        raise ValueError("At least one task batch must be provided")
    total = losses[0]
    for extra in losses[1:]:
        total = total + extra
    stats["loss_total"] = float(total.detach().item())
    return total, stats


def _train_step(
    model: ComparatorNetwork,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    use_amp: bool,
    grad_clip: float,
    weights: ComparatorTrainConfig,
    **batches: Any,
) -> dict[str, float]:
    model.train()
    amp_enabled = use_amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32
    moved: dict[str, Any] = {}
    for key, value in batches.items():
        if isinstance(value, Mapping):
            moved[key] = {k: v.to(device, non_blocking=True) for k, v in value.items()}
        elif isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        total, stats = comparator_multitask_losses(model, weights=weights, **moved)
    total.backward()
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return stats


def _selection_score(val_metrics: Mapping[str, Any]) -> float:
    return (
        val_metrics["corp"]["overall"]["acc"]
        + val_metrics["prelude_top1"]["overall"]["acc"]
        + val_metrics["draft"]["overall"]["acc"]
    ) / 3.0


@torch.inference_mode()
def collect_value_logits(
    model: ComparatorNetwork,
    corpus: Any,
    builder: FastBatchBuilder,
    split: str,
    *,
    device: torch.device,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(logits, targets)` for the split's setup-outcome rows."""
    so = corpus.setup_outcomes
    idx = so.split_indices[split]
    logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(idx), batch_size):
        b_idx = idx[start : start + batch_size]
        batch, target = builder.build_outcome_batch(so, b_idx)
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch, compute_policy=False, compute_value=True)
        logits.append(out["value_logit"].float().cpu().numpy())
        targets.append(target.numpy())
    if not logits:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.concatenate(logits), np.concatenate(targets)


def run_comparator_warmstart(
    *,
    data_dir: Path | str | None = None,
    out_dir: Path | str = DEFAULT_OUT_DIR,
    train_cfg: ComparatorTrainConfig | None = None,
    model_cfg: ComparatorConfig | None = None,
    device_str: str | None = None,
) -> dict[str, Any]:
    """Train the comparator on human setup/draft/value tasks, then fit temperature on val."""
    cfg = train_cfg or ComparatorTrainConfig()
    out_path = Path(out_dir)
    best_dir = out_path / "best"
    out_path.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)
    device = torch.device(
        device_str or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    resolved_data = resolve_warmstart_dir(data_dir)
    cards = load_card_data()
    vocab = CardVocab.from_card_data(cards)
    corpus = load_warmstart_corpus(resolved_data, vocab=vocab)
    builder = FastBatchBuilder(vocab, cards)

    mcfg = model_cfg or ComparatorConfig(vocab_size=vocab.size)
    model = ComparatorNetwork.from_vocab(vocab, mcfg).to(device)
    param_count = model.count_parameters()
    print(
        f"Comparator warm start on {device}: {param_count:,} parameters, "
        f"data={resolved_data}"
    )

    tr_dp = corpus.draft_picks.split_indices["train"]
    tr_sc = corpus.setup_choices.split_indices["train"]
    tr_so = corpus.setup_outcomes.split_indices["train"]
    steps_per_epoch = max(1, (len(tr_dp) + cfg.draft_batch_size - 1) // cfg.draft_batch_size)
    if cfg.max_steps_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, cfg.max_steps_per_epoch)
    total_steps = steps_per_epoch * cfg.epochs

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return float(step + 1) / float(max(1, cfg.warmup_steps))
        progress = float(step - cfg.warmup_steps) / float(max(1, total_steps - cfg.warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        min_ratio = cfg.min_lr / cfg.lr
        return float(min_ratio + (1.0 - min_ratio) * cosine)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    sc_cursor = 0
    so_cursor = 0
    sc_perm = np_rng.permutation(tr_sc)
    so_perm = np_rng.permutation(tr_so)
    best_score = -1.0
    best_epoch = 0
    best_metrics: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    global_step = 0
    t0 = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        ep_t0 = time.perf_counter()
        dp_perm = np_rng.permutation(tr_dp)
        running: dict[str, float] = {}
        value_steps = 0
        for step_idx in range(steps_per_epoch):
            d_start = step_idx * cfg.draft_batch_size
            d_idx = dp_perm[d_start : d_start + cfg.draft_batch_size]
            draft_b, draft_t = builder.build_draft_batch(
                corpus.draft_picks, corpus.setup_choices, d_idx
            )
            if sc_cursor + cfg.setup_batch_size > len(sc_perm):
                sc_perm = np_rng.permutation(tr_sc)
                sc_cursor = 0
            s_idx = sc_perm[sc_cursor : sc_cursor + cfg.setup_batch_size]
            sc_cursor += cfg.setup_batch_size
            corp_b, corp_t = builder.build_corp_batch(corpus.setup_choices, s_idx)
            prel_b, prel_t = builder.build_prelude_batch(
                corpus.setup_choices, s_idx, rng=np_rng
            )
            proj_b, proj_t = builder.build_project_buy_batch(
                corpus.setup_choices, s_idx, rng=np_rng
            )
            outcome_b = outcome_t = None
            if (global_step % cfg.value_every_n_steps) == 0 and len(so_perm) > 0:
                if so_cursor + cfg.value_batch_size > len(so_perm):
                    so_perm = np_rng.permutation(tr_so)
                    so_cursor = 0
                v_idx = so_perm[so_cursor : so_cursor + cfg.value_batch_size]
                so_cursor += cfg.value_batch_size
                outcome_b, outcome_t = builder.build_outcome_batch(
                    corpus.setup_outcomes, v_idx
                )
                value_steps += 1
            stats = _train_step(
                model,
                optimizer,
                device=device,
                use_amp=cfg.use_amp,
                grad_clip=cfg.grad_clip,
                weights=cfg,
                draft_batch=draft_b,
                draft_targets=draft_t,
                corp_batch=corp_b,
                corp_targets=corp_t,
                prelude_batch=prel_b,
                prelude_targets=prel_t,
                project_batch=proj_b,
                project_targets=proj_t,
                outcome_batch=outcome_b,
                outcome_targets=outcome_t,
            )
            scheduler.step()
            global_step += 1
            for key, value in stats.items():
                running[key] = running.get(key, 0.0) + value

        ep_time = time.perf_counter() - ep_t0
        val_metrics = evaluate_model(
            model,
            corpus,
            builder,
            "val",
            device=device,
            batch_size=cfg.eval_batch_size,
            use_amp=cfg.use_amp,
        )
        score = _selection_score(val_metrics)
        avg = {
            k: v / max(1, value_steps if k == "loss_value" else steps_per_epoch)
            for k, v in running.items()
        }
        print(
            f"[comparator epoch {epoch}/{cfg.epochs}] {ep_time:.1f}s | "
            f"loss={avg.get('loss_total', float('nan')):.4f} | "
            f"val corp={val_metrics['corp']['overall']['acc']:.4f} | "
            f"val prel1={val_metrics['prelude_top1']['overall']['acc']:.4f} | "
            f"val draft={val_metrics['draft']['overall']['acc']:.4f} | "
            f"val value acc={val_metrics['value']['overall']['acc']:.4f}"
        )
        record = {
            "epoch": epoch,
            "epoch_time_sec": ep_time,
            "train_loss": avg,
            "val_score": score,
            "val": val_metrics,
        }
        history.append(record)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = val_metrics
            save_comparator_checkpoint(
                best_dir,
                model,
                vocab,
                temperature=1.0,
                metadata={
                    "version": "comparator-warmstart-v1",
                    "best_epoch": best_epoch,
                    "val_score": best_score,
                    "train_config": cfg.to_dict(),
                    "model_config": mcfg.to_dict(),
                },
            )

    if best_metrics is None:
        raise RuntimeError("comparator warm start produced no checkpoint")
    best_model, _, _meta = load_comparator_checkpoint(best_dir, map_location=device)
    val_logits, val_targets = collect_value_logits(
        best_model, corpus, builder, "val", device=device, batch_size=cfg.eval_batch_size
    )
    fit = fit_temperature(val_logits, val_targets)
    save_comparator_checkpoint(
        best_dir,
        best_model,
        vocab,
        temperature=fit["temperature"],
        metadata={
            "version": "comparator-warmstart-v1",
            "best_epoch": best_epoch,
            "val_score": best_score,
            "val_calibration": fit,
            "train_config": cfg.to_dict(),
            "model_config": mcfg.to_dict(),
        },
    )

    total_wall = time.perf_counter() - t0
    results = {
        "status": "completed",
        "parameter_count": param_count,
        "device": str(device),
        "data_dir": str(resolved_data),
        "wall_clock_sec": total_wall,
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "best_val": best_metrics,
        "temperature": fit["temperature"],
        "val_calibration": fit,
        "train_config": cfg.to_dict(),
        "model_config": mcfg.to_dict(),
        "history": history,
        "checkpoint_path": str((best_dir / "checkpoint.pt").resolve()),
    }
    (out_path / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(
        f"Comparator warm start done in {total_wall:.1f}s; temperature="
        f"{fit['temperature']:.4f}; checkpoint={best_dir}"
    )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Warm-start the comparator on human data.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--draft-batch-size", type=int, default=1024)
    parser.add_argument("--setup-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    args = parser.parse_args(argv)
    run_comparator_warmstart(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        train_cfg=ComparatorTrainConfig(
            epochs=args.epochs,
            draft_batch_size=args.draft_batch_size,
            setup_batch_size=args.setup_batch_size,
            lr=args.lr,
            seed=args.seed,
            use_amp=not args.no_amp,
            max_steps_per_epoch=args.max_steps_per_epoch,
        ),
        device_str=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
