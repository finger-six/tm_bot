"""Warm-start multi-task training pipeline on setup choices, draft picks, and setup outcomes."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from tm_bot.agents.card_data import load_card_data
from tm_bot.env.types import SUPPORTED_MAPS
from tm_bot.models.network import (
    TMModelConfig,
    TMNetwork,
    load_checkpoint,
    save_checkpoint,
)
from tm_bot.models.vocab import CardVocab
from tm_bot.training.baselines import FrequencyBaseline, evaluate_model
from tm_bot.training.dataset import (
    FastBatchBuilder,
    load_warmstart_corpus,
    resolve_warmstart_dir,
)


@dataclass
class TrainConfig:
    """Hyperparameters and run configuration for warm-start training."""

    epochs: int = 4
    draft_batch_size: int = 1536
    setup_batch_size: int = 256
    value_batch_size: int = 128
    value_every_n_steps: int = 2
    lr: float = 5e-4
    min_lr: float = 3e-5
    weight_decay: float = 2e-4
    grad_clip: float = 1.0
    warmup_steps: int = 200
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
    """Numerically safe cross-entropy with soft target distributions over masked candidates."""
    log_probs = F.log_softmax(logits.float(), dim=-1)
    safe_log_probs = torch.where(cand_mask, log_probs, torch.zeros_like(log_probs))
    return -(soft_targets * safe_log_probs).sum(dim=-1).mean()


def compute_multitask_step_losses(
    model: TMNetwork,
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
    weights: TrainConfig | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute weighted multi-task warm-start loss from any combination of task batches.

    When `corp_batch`, `prelude_batch`, and `project_batch` come from the same `setup_choices`
    rows, `encode_observations` is executed once and reused across all three choice heads.
    """
    cfg = weights or TrainConfig()
    losses: list[torch.Tensor] = []
    stats: dict[str, float] = {}

    if draft_batch is not None and draft_targets is not None:
        out_d = model(draft_batch, compute_policy=True, compute_value=False)
        ld = F.cross_entropy(out_d["policy_logits"].float(), draft_targets)
        losses.append(cfg.w_draft * ld)
        stats["loss_draft"] = float(ld.detach().item())

    setup_obs_source = corp_batch or prelude_batch or project_batch
    if setup_obs_source is not None:
        obs_tokens, obs_mask, obs_pooled = model.encode_observations(setup_obs_source)
        if corp_batch is not None and corp_targets is not None:
            logits_c, _ = model.forward_policy(obs_tokens, obs_mask, obs_pooled, corp_batch)
            lc = F.cross_entropy(logits_c.float(), corp_targets)
            losses.append(cfg.w_corp * lc)
            stats["loss_corp"] = float(lc.detach().item())
        if prelude_batch is not None and prelude_targets is not None:
            logits_p, _ = model.forward_policy(obs_tokens, obs_mask, obs_pooled, prelude_batch)
            lp = soft_cross_entropy(logits_p, prelude_targets, prelude_batch["cand_mask"].bool())
            losses.append(cfg.w_prelude * lp)
            stats["loss_prelude"] = float(lp.detach().item())
        if project_batch is not None and project_targets is not None:
            logits_pj, _ = model.forward_policy(obs_tokens, obs_mask, obs_pooled, project_batch)
            lpj = soft_cross_entropy(
                logits_pj, project_targets, project_batch["cand_mask"].bool()
            )
            losses.append(cfg.w_project * lpj)
            stats["loss_project"] = float(lpj.detach().item())

    if outcome_batch is not None and outcome_targets is not None:
        out_v = model(outcome_batch, compute_policy=False, compute_value=True)
        lv = F.binary_cross_entropy_with_logits(out_v["value_logit"].float(), outcome_targets)
        losses.append(cfg.w_value * lv)
        stats["loss_value"] = float(lv.detach().item())

    if not losses:
        raise ValueError("At least one task batch must be provided")

    total = losses[0]
    for extra in losses[1:]:
        total = total + extra
    stats["loss_total"] = float(total.detach().item())
    return total, stats


def train_one_step(
    model: TMNetwork,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    use_amp: bool = True,
    grad_clip: float = 1.0,
    weights: TrainConfig | None = None,
    **batches: Any,
) -> dict[str, float]:
    """Perform one optimization step on the supplied task batches and return scalar loss stats."""
    model.train()
    amp_enabled = use_amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32

    moved: dict[str, Any] = {}
    for k, v in batches.items():
        if isinstance(v, Mapping):
            moved[k] = {tk: tv.to(device, non_blocking=True) for tk, tv in v.items()}
        elif isinstance(v, torch.Tensor):
            moved[k] = v.to(device, non_blocking=True)
        else:
            moved[k] = v

    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        total_loss, stats = compute_multitask_step_losses(
            model, weights=weights, **moved
        )
    total_loss.backward()
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


def format_comparison_report(
    *,
    split_name: str,
    model_metrics: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
    param_count: int,
    wall_clock_sec: float,
) -> str:
    """Format an aligned Markdown report table comparing model vs frequency baseline."""
    lines: list[str] = [
        f"### {split_name.capitalize()} Split Results",
        "",
        "| Task | Scope | Model Top-1 | Freq Baseline Top-1 | Delta | Samples |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    tasks = [
        ("Corporation", "corp"),
        ("Prelude (step-0 top-1)", "prelude_top1"),
        ("Prelude (cond step-1)", "prelude_step1_cond"),
        ("Prelude (exact pair)", "prelude_pair"),
        ("Draft pick", "draft"),
    ]
    for label, key in tasks:
        m_ov = model_metrics[key]["overall"]
        b_ov = baseline_metrics[key]["overall"]
        delta = m_ov["acc"] - b_ov["acc"]
        lines.append(
            f"| {label} | Overall | {m_ov['acc']:.4f} | {b_ov['acc']:.4f} | {delta:+.4f} | {m_ov['total']:,} |"
        )
        for mname in SUPPORTED_MAPS:
            mm = model_metrics[key]["per_map"][mname]
            bm = baseline_metrics[key]["per_map"][mname]
            md = mm["acc"] - bm["acc"]
            lines.append(
                f"| {label} | {mname} | {mm['acc']:.4f} | {bm['acc']:.4f} | {md:+.4f} | {mm['total']:,} |"
            )

    pj = model_metrics["project_buy"]
    lines.append("")
    lines.append(
        f"- **Project buy**: step-0 accuracy = `{pj['step0_acc']:.4f}`, "
        f"done-stop accuracy = `{pj['done_acc']:.4f}` (`n={pj['total']:,}`)"
    )

    val = model_metrics["value"]
    vov = val["overall"]
    lines.append(
        f"- **Value head (setup outcome)**: BCE = `{vov['bce']:.4f}`, "
        f"Brier = `{vov['brier']:.4f}`, Accuracy = `{vov['acc']:.4f}` (`n={vov['total']:,}`)"
    )
    lines.append("- **Value accuracy per map**: " + ", ".join(
        f"{m}=`{val['per_map'][m]['acc']:.4f}`" for m in SUPPORTED_MAPS
    ))
    lines.append(f"- **Player-holdout**: `N/A` (corpus contributors appear in all splits)")
    lines.append(f"- **Model parameters**: `{param_count:,}`")
    lines.append(f"- **Training wall-clock**: `{wall_clock_sec:.1f}s` (`{wall_clock_sec / 60.0:.2f} min`)")
    return "\n".join(lines)


def run_warmstart_training(
    *,
    data_dir: Path | str | None = None,
    out_dir: Path | str = Path("runs/warmstart"),
    train_cfg: TrainConfig | None = None,
    model_cfg: TMModelConfig | None = None,
    device_str: str | None = None,
    resume: bool = False,
    epochs_per_run: int | None = None,
) -> dict[str, Any]:
    """Run warm-start training, select best checkpoint on val, evaluate test once at completion."""
    cfg = train_cfg or TrainConfig()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    state_path = out_path / "latest_state.pt"
    best_ckpt_dir = out_path / "best"

    torch.manual_seed(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)

    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_start = time.perf_counter()
    resolved_data = resolve_warmstart_dir(data_dir)
    cards = load_card_data()
    vocab = CardVocab.from_card_data(cards)

    print(f"Loading warm-start corpus from {resolved_data} ...")
    corpus = load_warmstart_corpus(resolved_data, vocab=vocab)
    builder = FastBatchBuilder(vocab, cards)

    mcfg = model_cfg or TMModelConfig(vocab_size=vocab.size)
    model = TMNetwork.from_vocab(vocab, mcfg).to(device)
    param_count = model.count_parameters()
    print(f"Model initialized on {device} with {param_count:,} parameters.")

    freq_baseline = FrequencyBaseline.fit_train(corpus)
    val_baseline = freq_baseline.evaluate_split(corpus, "val")

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

    start_epoch = 1
    best_score = -1.0
    best_epoch = 0
    history: list[dict[str, Any]] = []
    global_step = 0
    accum_train_sec = 0.0

    if resume and state_path.exists():
        st = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model_state_dict"])
        optimizer.load_state_dict(st["optimizer_state_dict"])
        scheduler.load_state_dict(st["scheduler_state_dict"])
        start_epoch = int(st["completed_epoch"]) + 1
        best_score = float(st["best_score"])
        best_epoch = int(st["best_epoch"])
        history = list(st["history"])
        global_step = int(st["global_step"])
        accum_train_sec = float(st["accum_train_sec"])
        np_rng.bit_generator.state = st["np_rng_state"]
        print(f"Resumed from epoch {start_epoch - 1}/{cfg.epochs} (best_epoch={best_epoch}, best_score={best_score:.4f}).")

    end_epoch = cfg.epochs
    if epochs_per_run is not None:
        end_epoch = min(cfg.epochs, start_epoch + epochs_per_run - 1)

    sc_cursor = 0
    so_cursor = 0
    sc_perm = np_rng.permutation(tr_sc)
    so_perm = np_rng.permutation(tr_so)

    run_t0 = time.perf_counter()

    for epoch in range(start_epoch, end_epoch + 1):
        ep_t0 = time.perf_counter()
        dp_perm = np_rng.permutation(tr_dp)
        running = {
            "loss_total": 0.0,
            "loss_draft": 0.0,
            "loss_corp": 0.0,
            "loss_prelude": 0.0,
            "loss_project": 0.0,
            "loss_value": 0.0,
        }
        val_steps = 0

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

            outcome_b = None
            outcome_t = None
            if (global_step % cfg.value_every_n_steps) == 0 and len(so_perm) > 0:
                if so_cursor + cfg.value_batch_size > len(so_perm):
                    so_perm = np_rng.permutation(tr_so)
                    so_cursor = 0
                v_idx = so_perm[so_cursor : so_cursor + cfg.value_batch_size]
                so_cursor += cfg.value_batch_size
                outcome_b, outcome_t = builder.build_outcome_batch(corpus.setup_outcomes, v_idx)
                val_steps += 1

            step_stats = train_one_step(
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

            for k in running:
                if k in step_stats:
                    running[k] += step_stats[k]

        ep_time = time.perf_counter() - ep_t0
        accum_train_sec += ep_time
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
        avg_loss = {
            k: (v / max(1, val_steps if k == "loss_value" else steps_per_epoch))
            for k, v in running.items()
        }

        print(
            f"[Epoch {epoch}/{cfg.epochs}] {ep_time:.1f}s | "
            f"loss={avg_loss['loss_total']:.4f} | "
            f"val corp={val_metrics['corp']['overall']['acc']:.4f} (base {val_baseline['corp']['overall']['acc']:.4f}) | "
            f"val prel1={val_metrics['prelude_top1']['overall']['acc']:.4f} (base {val_baseline['prelude_top1']['overall']['acc']:.4f}) | "
            f"val draft={val_metrics['draft']['overall']['acc']:.4f} (base {val_baseline['draft']['overall']['acc']:.4f}) | "
            f"val value_acc={val_metrics['value']['overall']['acc']:.4f} bce={val_metrics['value']['overall']['bce']:.4f}"
        )

        ep_record = {
            "epoch": epoch,
            "epoch_time_sec": ep_time,
            "train_loss": avg_loss,
            "val_score": score,
            "val": val_metrics,
        }
        history.append(ep_record)

        if score > best_score:
            best_score = score
            best_epoch = epoch
            save_checkpoint(
                best_ckpt_dir,
                model,
                vocab,
                metadata={
                    "version": "warmstart-v1",
                    "best_epoch": best_epoch,
                    "val_score": best_score,
                    "train_config": cfg.to_dict(),
                },
            )

        torch.save(
            {
                "completed_epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_score": best_score,
                "best_epoch": best_epoch,
                "history": history,
                "global_step": global_step,
                "accum_train_sec": accum_train_sec,
                "np_rng_state": np_rng.bit_generator.state,
            },
            state_path,
        )

    if end_epoch < cfg.epochs:
        print(f"Paused after epoch {end_epoch}/{cfg.epochs}; re-run with --resume to continue.")
        return {
            "status": "paused",
            "completed_epoch": end_epoch,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "accum_train_sec": accum_train_sec,
        }

    train_wall_sec = accum_train_sec
    total_wall_sec = accum_train_sec + (time.perf_counter() - t_start) - (time.perf_counter() - run_t0)

    # Load best validation checkpoint and evaluate on test EXACTLY ONCE
    best_model, _, _ = load_checkpoint(best_ckpt_dir, map_location=device)
    best_val_metrics = history[best_epoch - 1]["val"]
    test_baseline = freq_baseline.evaluate_split(corpus, "test")
    test_metrics = evaluate_model(
        best_model,
        corpus,
        builder,
        "test",
        device=device,
        batch_size=cfg.eval_batch_size,
        use_amp=cfg.use_amp,
    )

    val_report = format_comparison_report(
        split_name="validation",
        model_metrics=best_val_metrics,
        baseline_metrics=val_baseline,
        param_count=param_count,
        wall_clock_sec=train_wall_sec,
    )
    test_report = format_comparison_report(
        split_name="test",
        model_metrics=test_metrics,
        baseline_metrics=test_baseline,
        param_count=param_count,
        wall_clock_sec=train_wall_sec,
    )

    print("\n" + "=" * 72)
    print(f"Best validation checkpoint: epoch {best_epoch} -> {best_ckpt_dir / 'checkpoint.pt'}")
    print(val_report)
    print("\n" + test_report)
    print("=" * 72)

    results = {
        "status": "completed",
        "best_epoch": best_epoch,
        "checkpoint_path": str((best_ckpt_dir / "checkpoint.pt").resolve()),
        "config_path": str((best_ckpt_dir / "config.json").resolve()),
        "parameter_count": param_count,
        "train_wall_clock_sec": train_wall_sec,
        "total_wall_clock_sec": total_wall_sec,
        "device": str(device),
        "train_config": cfg.to_dict(),
        "model_config": mcfg.to_dict(),
        "val_baseline": val_baseline,
        "val_model": best_val_metrics,
        "test_baseline": test_baseline,
        "test_model": test_metrics,
        "history": history,
    }
    (out_path / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_path / "report.md").write_text(
        val_report + "\n\n" + test_report + "\n", encoding="utf-8"
    )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Warm-start policy/value training on TM datasets.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Warmstart parquet directory.")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/warmstart"), help="Output checkpoint directory.")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--draft-batch-size", type=int, default=1536)
    parser.add_argument("--setup-batch-size", type=int, default=256)
    parser.add_argument("--value-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs-per-run", type=int, default=None)
    args = parser.parse_args(argv)

    tcfg = TrainConfig(
        epochs=args.epochs,
        draft_batch_size=args.draft_batch_size,
        setup_batch_size=args.setup_batch_size,
        value_batch_size=args.value_batch_size,
        lr=args.lr,
        seed=args.seed,
        use_amp=not args.no_amp,
        max_steps_per_epoch=args.max_steps_per_epoch,
    )
    run_warmstart_training(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        train_cfg=tcfg,
        device_str=args.device,
        resume=args.resume,
        epochs_per_run=args.epochs_per_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
