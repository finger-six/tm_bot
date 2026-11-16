"""CLI for `python -m tm_bot.runs`: launch or resume a self-play run."""

from __future__ import annotations

import argparse
from pathlib import Path

from tm_bot.runs.config import ResourceLimits, RunConfig
from tm_bot.runs.orchestrator import SelfPlayRunner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Self-play training run for the Terraforming Mars main agent."
    )
    parser.add_argument("--run-name", required=True, help="Output directory name under runs/.")
    parser.add_argument("--out-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--agent-kind",
        choices=("main", "comparator"),
        default="main",
        help="main = AlphaZero search agent; comparator = direct PPO scorer.",
    )
    parser.add_argument("--budget-hours", type=float, default=24.0, help="Training-time budget (wall clock minus in-run dev eval).")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sims", type=int, default=64, help="Simulations per searched decision.")
    parser.add_argument("--worlds", type=int, default=4, help="Determinizations (K).")
    parser.add_argument("--games-per-cycle", type=int, default=32)
    parser.add_argument("--updates-per-cycle", type=int, default=16)
    parser.add_argument("--learner-batch-size", type=int, default=256)
    parser.add_argument("--buffer-capacity", type=int, default=200_000)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--checkpoint-minutes", type=float, default=30.0)
    parser.add_argument("--dev-games", type=int, default=100)
    parser.add_argument("--skip-dev-eval", action="store_true")
    parser.add_argument("--dev-seed-start", type=int, default=0)
    parser.add_argument("--train-seed-start", type=int, default=0)
    parser.add_argument("--keep-last-checkpoints", type=int, default=10)
    parser.add_argument("--plateau-checkpoints", type=int, default=12)
    parser.add_argument("--step-cap", type=int, default=None)
    parser.add_argument("--determinism-every-games", type=int, default=1_000)
    parser.add_argument("--warmstart", type=str, default=None)
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Main agent only: train from random weights (architecture/vocab from --warmstart).",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--ppo-gamma", type=float, default=1.0)
    parser.add_argument("--ppo-gae-lambda", type=float, default=0.95)
    parser.add_argument("--ppo-value-coef", type=float, default=0.5)
    parser.add_argument("--ppo-entropy-coef", type=float, default=0.01)
    parser.add_argument("--disk-gb", type=float, default=50.0)
    parser.add_argument("--vram-gb", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cycles", type=int, default=None)
    args = parser.parse_args(argv)

    config = RunConfig(
        run_name=args.run_name,
        output_root=args.out_root,
        agent_kind=args.agent_kind,
        workers=args.workers,
        batch_size=args.batch_size,
        wall_clock_budget_s=args.budget_hours * 3600.0,
        num_simulations=args.sims,
        num_worlds=args.worlds,
        games_per_cycle=args.games_per_cycle,
        updates_per_cycle=args.updates_per_cycle,
        learner_batch_size=args.learner_batch_size,
        buffer_capacity=args.buffer_capacity,
        learning_rate=args.learning_rate,
        checkpoint_interval_s=args.checkpoint_minutes * 60.0,
        dev_games_per_opponent=args.dev_games,
        dev_eval_enabled=not args.skip_dev_eval,
        dev_seed_start=args.dev_seed_start,
        train_seed_start=args.train_seed_start,
        keep_last_checkpoints=args.keep_last_checkpoints,
        plateau_checkpoints=args.plateau_checkpoints,
        step_cap=args.step_cap,
        determinism_check_every_games=args.determinism_every_games,
        warmstart_checkpoint=args.warmstart,
        random_init=args.random_init,
        device=args.device,
        use_amp=not args.no_amp,
        seed=args.seed,
        ppo_clip=args.ppo_clip,
        ppo_gamma=args.ppo_gamma,
        ppo_gae_lambda=args.ppo_gae_lambda,
        ppo_value_coef=args.ppo_value_coef,
        ppo_entropy_coef=args.ppo_entropy_coef,
        limits=ResourceLimits(disk_gb=args.disk_gb, vram_gb=args.vram_gb),
    )
    runner = SelfPlayRunner.create(config)
    summary = runner.run(resume=args.resume, max_cycles=args.max_cycles)
    print(summary.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
