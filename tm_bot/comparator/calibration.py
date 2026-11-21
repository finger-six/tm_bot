"""Win-probability calibration: temperature scaling, Brier/NLL, reliability reports."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import torch

from tm_bot.agents import B1Agent, load_card_data
from tm_bot.agents.protocol import Agent
from tm_bot.agents.runner import play_games
from tm_bot.comparator.agent import ComparatorAgent, comparator_eval_factory
from tm_bot.comparator.model import load_comparator_checkpoint
from tm_bot.env.types import SUPPORTED_MAPS, ActorObservation, Choice
from tm_bot.eval.pairing import build_paired_specs
from tm_bot.eval.report import summarize_evaluation
from tm_bot.eval.seeds import generate_seed_keys
from tm_bot.runs.deveval import dev_seed_count


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _nll(probs: np.ndarray, targets: np.ndarray) -> float:
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    y = np.asarray(targets, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _brier(probs: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((np.asarray(probs) - np.asarray(targets)) ** 2))


def fit_temperature(
    logits: Sequence[float],
    targets: Sequence[float],
    *,
    min_temp: float = 0.05,
    max_temp: float = 20.0,
    max_iter: int = 100,
) -> dict[str, Any]:
    """Fit a single temperature on validation logits by minimizing NLL."""
    logit_arr = np.asarray(logits, dtype=np.float64)
    target_arr = np.asarray(targets, dtype=np.float64)
    n = int(logit_arr.size)
    if n == 0:
        return {
            "temperature": 1.0,
            "n": 0,
            "nll_before": None,
            "nll_after": None,
            "brier_before": None,
            "brier_after": None,
        }
    if n != target_arr.size:
        raise ValueError("logits and targets must have the same length")

    before = _sigmoid(logit_arr)
    nll_before = _nll(before, target_arr)
    brier_before = _brier(before, target_arr)

    logits_t = torch.tensor(logit_arr, dtype=torch.float64)
    targets_t = torch.tensor(target_arr, dtype=torch.float64)
    log_temp = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=max_iter)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temp = torch.exp(log_temp).clamp(min_temp, max_temp)
        probs = torch.sigmoid(logits_t / temp)
        loss = torch.nn.functional.binary_cross_entropy(probs, targets_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(torch.exp(log_temp.detach()).clamp(min_temp, max_temp).item())
    after = _sigmoid(logit_arr / temperature)
    return {
        "temperature": temperature,
        "n": n,
        "nll_before": nll_before,
        "nll_after": _nll(after, target_arr),
        "brier_before": brier_before,
        "brier_after": _brier(after, target_arr),
    }


def fit_midgame_temperature(
    logits: Sequence[float],
    targets: Sequence[float],
    generations: Sequence[int],
    **kwargs: Any,
) -> dict[str, Any]:
    """Fit the temperature on generation 2+ value logits only (#38 §4, #46).

    Generation-1 setup-outcome rows are excluded: the calibrated win probability serves mid-game
    play, and post-setup outcomes are a different, much less predictable regime.
    """
    gen = np.asarray(generations, dtype=np.int64)
    logit_arr = np.asarray(logits, dtype=np.float64)
    target_arr = np.asarray(targets, dtype=np.float64)
    if not (gen.size == logit_arr.size == target_arr.size):
        raise ValueError("logits, targets and generations must have the same length")
    keep = gen >= 2
    fit = fit_temperature(logit_arr[keep], target_arr[keep], **kwargs)
    fit["excluded_gen1"] = int((~keep).sum())
    return fit


def calibration_metrics(
    probs: Sequence[float],
    targets: Sequence[float],
    *,
    num_bins: int = 10,
) -> dict[str, Any]:
    """Brier, NLL, ECE, accuracy, and reliability bins."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if p.size == 0:
        return {
            "n": 0,
            "brier": None,
            "nll": None,
            "ece": None,
            "accuracy": None,
            "bins": [],
        }
    if p.size != y.size:
        raise ValueError("probs and targets must have the same length")
    bins: list[dict[str, Any]] = []
    ece = 0.0
    n = int(p.size)
    for i in range(num_bins):
        lo = i / num_bins
        hi = (i + 1) / num_bins
        if i == num_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        count = int(mask.sum())
        if count == 0:
            bins.append(
                {
                    "lo": lo,
                    "hi": hi,
                    "count": 0,
                    "mean_pred": None,
                    "observed": None,
                }
            )
            continue
        mean_pred = float(p[mask].mean())
        observed = float(y[mask].mean())
        ece += (count / n) * abs(mean_pred - observed)
        bins.append(
            {
                "lo": lo,
                "hi": hi,
                "count": count,
                "mean_pred": mean_pred,
                "observed": observed,
            }
        )
    return {
        "n": n,
        "brier": _brier(p, y),
        "nll": _nll(p, y),
        "ece": float(ece),
        "accuracy": float(np.mean((p >= 0.5) == (y >= 0.5))),
        "bins": bins,
    }


def reliability_table_md(metrics: dict[str, Any], *, title: str = "Reliability") -> str:
    """Markdown reliability table for one calibration metric block."""
    lines = [
        f"### {title}",
        "",
        f"- n = {metrics['n']}, Brier = {metrics['brier']:.4f}, "
        f"NLL = {metrics['nll']:.4f}, ECE = {metrics['ece']:.4f}, "
        f"accuracy = {metrics['accuracy']:.4f}"
        if metrics["n"]
        else f"### {title}\n\nNo samples.",
        "",
        "| Bin | Count | Mean predicted | Observed | Gap |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for i, b in enumerate(metrics["bins"]):
        lo, hi = b["lo"], b["hi"]
        if b["count"] == 0:
            lines.append(f"| [{lo:.1f}, {hi:.1f}] | 0 | — | — | — |")
        else:
            gap = b["mean_pred"] - b["observed"]
            lines.append(
                f"| [{lo:.1f}, {hi:.1f}] | {b['count']} | "
                f"{b['mean_pred']:.4f} | {b['observed']:.4f} | {gap:+.4f} |"
            )
    return "\n".join(lines)


def reliability_svg(metrics: dict[str, Any], *, title: str = "Reliability") -> str:
    """Dependency-free reliability diagram SVG (predicted vs observed)."""
    width, height = 420, 420
    left, right, top, bottom = 56, 20, 36, 48
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(x: float) -> float:
        return left + x * plot_w

    def sy(y: float) -> float:
        return top + (1.0 - y) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="22" font-family="sans-serif" font-size="14">{title}</text>',
        f'<line x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(1):.1f}" y2="{sy(0):.1f}" stroke="#333"/>',
        f'<line x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(0):.1f}" y2="{sy(1):.1f}" stroke="#333"/>',
        f'<line x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(1):.1f}" y2="{sy(1):.1f}" '
        'stroke="#999" stroke-dasharray="4 4"/>',
        f'<text x="{sx(0.5):.1f}" y="{height - 12}" font-family="sans-serif" '
        'font-size="11" text-anchor="middle">mean predicted win probability</text>',
        f'<text x="14" y="{sy(0.5):.1f}" font-family="sans-serif" font-size="11" '
        'text-anchor="middle" transform="rotate(-90 14 '
        f'{sy(0.5):.1f})">observed win rate</text>',
    ]
    points = [
        (b["mean_pred"], b["observed"], b["count"])
        for b in metrics["bins"]
        if b["count"] > 0 and b["mean_pred"] is not None
    ]
    if points:
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y, _ in points)
        parts.append(
            f'<polyline points="{coords}" fill="none" stroke="#1f77b4" stroke-width="2"/>'
        )
        max_count = max(c for _, _, c in points)
        for x, y, count in points:
            radius = 3.0 + 4.0 * (count / max_count) ** 0.5
            parts.append(
                f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{radius:.1f}" '
                'fill="#1f77b4" opacity="0.75"/>'
            )
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        parts.append(
            f'<text x="{sx(tick):.1f}" y="{sy(0.0) + 16:.1f}" font-family="sans-serif" '
            f'font-size="10" text-anchor="middle">{tick:.2f}</text>'
        )
        parts.append(
            f'<text x="{sx(0.0) - 6:.1f}" y="{sy(tick) + 3:.1f}" font-family="sans-serif" '
            f'font-size="10" text-anchor="end">{tick:.2f}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def write_calibration_report(
    out_dir: str | Path,
    logits: Sequence[float],
    targets: Sequence[float],
    *,
    temperature: float,
    metadata: dict[str, Any] | None = None,
    name: str = "calibration",
) -> dict[str, Any]:
    """Write JSON, Markdown tables, and an SVG reliability diagram."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = _sigmoid(np.asarray(logits, dtype=np.float64))
    calibrated = _sigmoid(np.asarray(logits, dtype=np.float64) / float(temperature))
    raw_metrics = calibration_metrics(raw, targets)
    calibrated_metrics = calibration_metrics(calibrated, targets)
    payload = {
        "schema_version": "comparator-calibration-v1",
        "name": name,
        "temperature": float(temperature),
        "metadata": dict(metadata or {}),
        "raw": raw_metrics,
        "calibrated": calibrated_metrics,
    }
    (out / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = "\n\n".join(
        [
            f"# Comparator win-probability calibration: {name}",
            f"Temperature scaling fitted on the warm-start validation split only: "
            f"T = {temperature:.4f}.",
            reliability_table_md(raw_metrics, title="Raw (T = 1)"),
            reliability_table_md(
                calibrated_metrics, title=f"Temperature scaled (T = {temperature:.4f})"
            ),
        ]
    )
    (out / f"{name}.md").write_text(md + "\n", encoding="utf-8")
    (out / f"{name}_reliability.svg").write_text(
        reliability_svg(calibrated_metrics, title=f"{name} (T={temperature:.3f})"),
        encoding="utf-8",
    )
    return payload


class WinProbRecorder:
    """Shared sink for `(actor, raw value logit)` pairs across copied agents."""

    def __init__(self) -> None:
        self.entries: list[tuple[int, float]] = []


class WinProbRecordingAgent:
    """Agent wrapper recording the comparator's value logit at each non-forced decision."""

    def __init__(self, inner: ComparatorAgent, recorder: WinProbRecorder) -> None:
        self.inner = inner
        self.recorder = recorder
        self.version = f"{inner.version}-calibration"

    def __deepcopy__(self, memo: dict[int, Any]) -> WinProbRecordingAgent:
        clone = self.__class__(copy.deepcopy(self.inner, memo), self.recorder)
        memo[id(self)] = clone
        return clone

    def reset(self, seed: int | None = None) -> None:
        self.inner.reset(seed)

    def act(self, observation: ActorObservation, choice: Choice) -> int:
        index = self.inner.act(observation, choice)
        if len(choice.candidates) > 1 and self.inner.last_decision_logit is not None:
            self.recorder.entries.append(
                (int(observation["actor"]), float(self.inner.last_decision_logit))
            )
        return index


def collect_dev_calibration(
    checkpoint: str | Path,
    *,
    opponent_checkpoint: str | Path | None = None,
    num_games: int = 52,
    maps: Sequence[str] = SUPPORTED_MAPS,
    seed_start: int = 0,
    num_workers: int = 16,
    batch_size: int = 32,
    device: str = "cpu",
) -> dict[str, Any]:
    """Play seat-swapped `dev-*` games and collect comparator win-prob predictions."""
    model, vocab, meta = load_comparator_checkpoint(
        checkpoint, map_location=device, encoder_version=None
    )
    model.eval()
    temperature = float(getattr(model, "temperature", 1.0))
    recorders: list[WinProbRecorder] = []

    play = comparator_eval_factory(
        model, vocab, device=device, version=f"dev-{Path(checkpoint).name}"
    )

    def candidate_factory() -> Agent:
        recorder = WinProbRecorder()
        recorders.append(recorder)
        return WinProbRecordingAgent(play(), recorder)

    if opponent_checkpoint is None:
        card_data = load_card_data()
        opponent_factory = lambda: B1Agent(seed=0, card_data=card_data)
        opponent_name = "b1"
    else:
        opp_model, opp_vocab, _ = load_comparator_checkpoint(
            opponent_checkpoint, map_location=device, encoder_version=None
        )
        opp_model.eval()
        opponent_factory = comparator_eval_factory(
            opp_model,
            opp_vocab,
            device=device,
            version=f"dev-{Path(opponent_checkpoint).name}",
        )
        opponent_name = str(Path(opponent_checkpoint).name)

    seeds = generate_seed_keys(
        "dev", dev_seed_count(num_games, maps), start=seed_start
    )
    specs, metas = build_paired_specs(
        agent_a=candidate_factory,
        agent_b=opponent_factory,
        seed_keys=seeds,
        maps=maps,
    )
    t0 = time.perf_counter()
    outcomes = play_games(specs, num_workers=num_workers, batch_size=batch_size)
    wall_s = time.perf_counter() - t0
    if len(recorders) != len(outcomes):
        raise RuntimeError(
            f"calibration recorder count ({len(recorders)}) != games ({len(outcomes)})"
        )

    logits: list[float] = []
    targets: list[float] = []
    for outcome, recorder, meta_row in zip(outcomes, recorders, metas):
        if not outcome.ok or outcome.result is None:
            continue
        actor = int(meta_row.a_seat)
        if outcome.result.tie:
            utility = 0.5
        elif outcome.result.winner == actor:
            utility = 1.0
        else:
            utility = 0.0
        for seat, logit in recorder.entries:
            if seat != actor:
                continue
            logits.append(logit)
            targets.append(utility)

    report = summarize_evaluation(
        outcomes=outcomes,
        metas=metas,
        namespace="dev",
        agent_a_name="comparator",
        agent_b_name=opponent_name,
        agent_a_version=None,
        agent_b_version=None,
        workers=num_workers,
        batch_size=batch_size,
        wall_time_s=wall_s,
    )
    deterministic = report["deterministic"]
    failure_details = [
        {
            "seed_key": outcome.seed_key,
            "map": outcome.map,
            "error_type": outcome.error_type,
            "error": outcome.error,
        }
        for outcome in outcomes
        if not outcome.ok
    ]
    return {
        "checkpoint": str(checkpoint),
        "opponent": opponent_name,
        "temperature": temperature,
        "meta": meta,
        "logits": logits,
        "targets": targets,
        "games": deterministic["total_games"],
        "completed": deterministic["completed_games"],
        "failures": deterministic["failure_count"],
        "failure_details": failure_details,
        "win_rate": deterministic["win_rate"],
        "win_rate_ci_95": deterministic["win_rate_ci_95"],
        "mean_vp_margin": deterministic["mean_vp_margin"],
        "mean_generations": deterministic["mean_generations"],
        "total_decisions": deterministic["total_decisions"],
        "per_map": deterministic["per_map"],
        "wall_time_s": wall_s,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Dev-seed win-probability calibration report for the comparator."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--opponent", type=str, default="b1")
    parser.add_argument("--games", type=int, default=52)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", type=str, default="dev")
    args = parser.parse_args(argv)

    opponent = None if args.opponent.lower() == "b1" else args.opponent
    collected = collect_dev_calibration(
        args.checkpoint,
        opponent_checkpoint=opponent,
        num_games=args.games,
        seed_start=args.seed_start,
        num_workers=args.workers,
        batch_size=args.batch_size,
        device=args.device,
    )
    payload = write_calibration_report(
        args.out,
        collected["logits"],
        collected["targets"],
        temperature=collected["temperature"],
        name=args.name,
        metadata={
            "checkpoint": collected["checkpoint"],
            "opponent": collected["opponent"],
            "games": collected["games"],
            "completed": collected["completed"],
            "failures": collected["failures"],
            "win_rate": collected["win_rate"],
            "win_rate_ci_95": collected["win_rate_ci_95"],
            "mean_vp_margin": collected["mean_vp_margin"],
            "mean_generations": collected["mean_generations"],
            "total_decisions": collected["total_decisions"],
            "failure_details": collected["failure_details"],
            "wall_time_s": collected["wall_time_s"],
        },
    )
    print(
        json.dumps(
            {
                "n": payload["calibrated"]["n"],
                "temperature": payload["temperature"],
                "raw_brier": payload["raw"]["brier"],
                "calibrated_brier": payload["calibrated"]["brier"],
                "raw_nll": payload["raw"]["nll"],
                "calibrated_nll": payload["calibrated"]["nll"],
                "games": collected["games"],
                "failures": collected["failures"],
                "win_rate": collected["win_rate"],
                "win_rate_ci_95": collected["win_rate_ci_95"],
                "wall_time_s": collected["wall_time_s"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
