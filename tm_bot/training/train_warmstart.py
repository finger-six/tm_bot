"""Main-agent (`TMNetwork`) v2 warm start: the shared multi-task trainer of `multitask.py` (#46).

    python -m tm_bot.training.train_warmstart --data-dir data/warmstart --out-dir runs/warmstart-v3
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from tm_bot.models.network import TMModelConfig, TMNetwork, load_checkpoint, save_checkpoint
from tm_bot.models.vocab import CardVocab
from tm_bot.training.multitask import (
    AgentSpec,
    WarmstartConfig,
    add_cli_args,
    compute_multitask_step_losses,
    config_from_args,
    run_from_args,
    run_multitask_warmstart,
    soft_cross_entropy,
    train_one_step,
)

# The main agent trains with the shared config; this name is kept for existing callers.
TrainConfig = WarmstartConfig

DEFAULT_OUT_DIR = Path("runs/warmstart-v3")


def _build_model(vocab: CardVocab) -> TMNetwork:
    return TMNetwork.from_vocab(vocab, TMModelConfig(vocab_size=vocab.size))


def _save(path: Path, model: Any, vocab: CardVocab, metadata: dict[str, Any]) -> None:
    save_checkpoint(path, model, vocab, metadata=metadata)


def _load(path: Path, device: torch.device) -> TMNetwork:
    return load_checkpoint(path, map_location=device)[0]


MAIN_AGENT = AgentSpec(
    name="main",
    version="warmstart-v3",
    default_out_dir=DEFAULT_OUT_DIR,
    build_model=_build_model,
    save_checkpoint=_save,
    load_checkpoint=_load,
)


def run_warmstart_training(
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
    """Train the main agent's `warmstart-v3` checkpoint on the v2 tables."""
    return run_multitask_warmstart(
        MAIN_AGENT,
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
    parser = argparse.ArgumentParser(description="Main-agent v2 warm start on the human-data tables.")
    add_cli_args(parser, default_out_dir=DEFAULT_OUT_DIR)
    return parser


def build_train_config(argv: list[str] | None = None) -> WarmstartConfig:
    """The config `main(argv)` trains with."""
    return config_from_args(_parser().parse_args(argv))


def main(argv: list[str] | None = None) -> int:
    run_from_args(MAIN_AGENT, _parser().parse_args(argv))
    return 0


__all__ = [
    "MAIN_AGENT",
    "TrainConfig",
    "build_train_config",
    "compute_multitask_step_losses",
    "run_warmstart_training",
    "soft_cross_entropy",
    "train_one_step",
]


if __name__ == "__main__":
    raise SystemExit(main())
