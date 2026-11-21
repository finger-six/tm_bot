"""Warm-start observation reconstruction, dataset loading, baselines, and training."""

from tm_bot.training.baselines import FrequencyBaseline, evaluate_model
from tm_bot.training.dataset import (
    DEFAULT_WARMSTART_DIR,
    DraftPicksTable,
    FastBatchBuilder,
    SetupChoicesTable,
    SetupOutcomesTable,
    WarmstartCorpus,
    load_warmstart_corpus,
    resolve_warmstart_dir,
)
from tm_bot.training.train_warmstart import (
    TrainConfig,
    compute_multitask_step_losses,
    run_warmstart_training,
    soft_cross_entropy,
    train_one_step,
)
from tm_bot.training.warmstart_builder import (
    build_corp_choice,
    build_draft_pick_choice,
    build_prelude_choice,
    build_project_buy_choice,
    build_setup_outcome_observation,
    load_map_templates,
    normalize_map_name,
)

__all__ = [
    "DEFAULT_WARMSTART_DIR",
    "DraftPicksTable",
    "FastBatchBuilder",
    "FrequencyBaseline",
    "SetupChoicesTable",
    "SetupOutcomesTable",
    "TrainConfig",
    "WarmstartCorpus",
    "build_corp_choice",
    "build_draft_pick_choice",
    "build_prelude_choice",
    "build_project_buy_choice",
    "build_setup_outcome_observation",
    "compute_multitask_step_losses",
    "evaluate_model",
    "load_map_templates",
    "load_warmstart_corpus",
    "normalize_map_name",
    "resolve_warmstart_dir",
    "run_warmstart_training",
    "soft_cross_entropy",
    "train_one_step",
]
