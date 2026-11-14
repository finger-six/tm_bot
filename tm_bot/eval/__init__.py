"""Fair paired evaluation harness with disjoint seed namespaces and reproducible reports."""

from __future__ import annotations

from tm_bot.eval.harness import evaluate_agents, resolve_agent_factory
from tm_bot.eval.pairing import DEFAULT_EVAL_SEEDS, AgentFactory, PairedGameMeta, build_paired_specs
from tm_bot.eval.report import (
    AMAZONIS_MAP_NAME,
    EVAL_REPORT_SCHEMA_VERSION,
    FIVE_MAP_SCOPE,
    collect_hardware_info,
    format_markdown_report,
    summarize_evaluation,
    write_report_files,
)
from tm_bot.eval.seeds import (
    SEED_NAMESPACES,
    SeedKey,
    SeedNamespace,
    derive_agent_seed,
    generate_seed_keys,
    make_seed_key,
    validate_namespace,
)
from tm_bot.eval.stats import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    paired_bootstrap_ci,
    percentile_ms,
    summarize_latency,
)

__all__ = [
    "AMAZONIS_MAP_NAME",
    "AgentFactory",
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_EVAL_SEEDS",
    "EVAL_REPORT_SCHEMA_VERSION",
    "FIVE_MAP_SCOPE",
    "PairedGameMeta",
    "SEED_NAMESPACES",
    "SeedKey",
    "SeedNamespace",
    "build_paired_specs",
    "collect_hardware_info",
    "derive_agent_seed",
    "evaluate_agents",
    "format_markdown_report",
    "generate_seed_keys",
    "make_seed_key",
    "paired_bootstrap_ci",
    "percentile_ms",
    "resolve_agent_factory",
    "summarize_evaluation",
    "summarize_latency",
    "validate_namespace",
    "write_report_files",
]
