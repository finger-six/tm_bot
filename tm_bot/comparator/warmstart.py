"""Comparator (`ComparatorNetwork`) v2 warm start: the shared multi-task trainer of
`tm_bot.training.multitask` (#46), then temperature scaling fit on generation 2+ validation value
logits only.

    python -m tm_bot.comparator.warmstart --data-dir data/warmstart_space --out-dir runs/comparator-warmstart-v3
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from tm_bot.comparator.model import (
    ComparatorConfig,
    ComparatorNetwork,
    load_comparator_checkpoint,
    save_comparator_checkpoint,
)
from tm_bot.models.vocab import CardVocab
from tm_bot.training.multitask import (
    AgentSpec,
    WarmstartConfig,
    add_cli_args,
    config_from_args,
    run_from_args,
    run_multitask_warmstart,
)

# Same config object as the main agent: identical tasks, loss weights and step budget (#38 §4).
ComparatorTrainConfig = WarmstartConfig

DEFAULT_OUT_DIR = Path("runs/comparator-warmstart-v3")


def _build_model(vocab: CardVocab) -> ComparatorNetwork:
    return ComparatorNetwork.from_vocab(vocab, ComparatorConfig(vocab_size=vocab.size))


def _save(path: Path, model: Any, vocab: CardVocab, metadata: dict[str, Any]) -> None:
    save_comparator_checkpoint(
        path, model, vocab, temperature=float(metadata.get("temperature", 1.0)), metadata=metadata
    )


def _load(path: Path, device: torch.device) -> ComparatorNetwork:
    return load_comparator_checkpoint(path, map_location=device)[0]


COMPARATOR_AGENT = AgentSpec(
    name="comparator",
    version="comparator-warmstart-v3",  # v2 plus the space task and rebuilt boards (#62)
    default_out_dir=DEFAULT_OUT_DIR,
    build_model=_build_model,
    save_checkpoint=_save,
    load_checkpoint=_load,
    calibrate_temperature=True,
)


def run_comparator_warmstart(
    *,
    data_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
    out_dir: Path | str = DEFAULT_OUT_DIR,
    train_cfg: WarmstartConfig | None = None,
    device_str: str | None = None,
    resume: bool = False,
    epochs_per_run: int | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    """Train the `comparator-warmstart-v3` checkpoint and fit its gen 2+ temperature on val."""
    return run_multitask_warmstart(
        COMPARATOR_AGENT,
        data_dir=data_dir,
        cache_dir=cache_dir,
        out_dir=out_dir,
        cfg=train_cfg,
        device_str=device_str,
        resume=resume,
        epochs_per_run=epochs_per_run,
        workers=workers,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Comparator v2 warm start on the human-data tables.")
    add_cli_args(parser, default_out_dir=DEFAULT_OUT_DIR)
    return parser


def build_train_config(argv: list[str] | None = None) -> WarmstartConfig:
    """The config `main(argv)` trains with."""
    return config_from_args(_parser().parse_args(argv))


def main(argv: list[str] | None = None) -> int:
    run_from_args(COMPARATOR_AGENT, _parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
