"""CLI entrypoint for `python -m tm_bot.eval`."""

from __future__ import annotations

import argparse
from pathlib import Path

from tm_bot.agents.runner import DEFAULT_MAX_DECISIONS
from tm_bot.env.types import SUPPORTED_MAPS
from tm_bot.eval.harness import evaluate_agents, resolve_agent_factory
from tm_bot.eval.report import format_markdown_report, write_report_files
from tm_bot.eval.seeds import SEED_NAMESPACES, validate_namespace
from tm_bot.eval.stats import DEFAULT_BOOTSTRAP_RESAMPLES, DEFAULT_BOOTSTRAP_SEED


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate two Terraforming Mars agents with seat-swapped pairs and bootstrap CIs."
    )
    parser.add_argument("--a", required=True, help="Agent A spec ('b0', 'b1', or 'module:factory').")
    parser.add_argument("--b", required=True, help="Agent B spec ('b0', 'b1', or 'module:factory').")
    parser.add_argument(
        "--games",
        type=int,
        default=400,
        help="Total games across all supported maps and both seats (default: 400 = 50 seeds x 4 maps x 2 seats).",
    )
    parser.add_argument(
        "--namespace",
        default="eval",
        choices=SEED_NAMESPACES,
        help="Seed namespace to use (default: eval).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for report.json, report_deterministic.json, and report.md.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Number of Node worker processes (default: 8).")
    parser.add_argument("--batch-size", type=int, default=32, help="Concurrent games per pool batch (default: 32).")
    parser.add_argument("--seed-start", type=int, default=0, help="Starting integer index for seed keys (default: 0).")
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
        help=f"Bootstrap resample count (default: {DEFAULT_BOOTSTRAP_RESAMPLES}).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help=f"Deterministic RNG seed for bootstrap CI (default: {DEFAULT_BOOTSTRAP_SEED}).",
    )
    parser.add_argument("--a-training-seeds", type=int, default=0, help="Training seed count for Agent A (0 for baselines).")
    parser.add_argument("--b-training-seeds", type=int, default=0, help="Training seed count for Agent B (0 for baselines).")
    parser.add_argument("--max-decisions", type=int, default=DEFAULT_MAX_DECISIONS, help="Per-game decision step cap.")
    parser.add_argument(
        "--omit-timing",
        action="store_true",
        help="Omit runtime timing, latency, and hardware sections from report.json and report.md.",
    )

    args = parser.parse_args(argv)
    namespace = validate_namespace(args.namespace)

    games_per_seed = 2 * len(SUPPORTED_MAPS)
    if args.games <= 0 or args.games % games_per_seed != 0:
        parser.error(
            f"--games must be a positive multiple of {games_per_seed} "
            f"({len(SUPPORTED_MAPS)} supported maps x 2 swapped seats), got {args.games}"
        )
    num_seeds = args.games // games_per_seed

    factory_a = resolve_agent_factory(args.a)
    factory_b = resolve_agent_factory(args.b)

    report = evaluate_agents(
        agent_a=factory_a,
        agent_b=factory_b,
        agent_a_name=args.a,
        agent_b_name=args.b,
        a_training_seeds=args.a_training_seeds,
        b_training_seeds=args.b_training_seeds,
        namespace=namespace,
        num_seeds=num_seeds,
        seed_start=args.seed_start,
        maps=SUPPORTED_MAPS,
        num_workers=args.workers,
        batch_size=args.batch_size,
        max_decisions=args.max_decisions,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )

    paths = write_report_files(report, args.out, omit_timing=args.omit_timing)
    print(format_markdown_report(report, omit_timing=args.omit_timing))
    print(f"Wrote reports: {paths['json']}, {paths['deterministic_json']}, {paths['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
