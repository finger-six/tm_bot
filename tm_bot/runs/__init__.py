"""Self-play run orchestration for the main agent (#24)."""

from __future__ import annotations

from tm_bot.runs.buffer import DecisionSample, ReplayBuffer, collate_samples
from tm_bot.runs.checkpoints import CheckpointManager
from tm_bot.runs.config import ResourceLimits, RunConfig
from tm_bot.runs.errors import RunError, RunResourceError, RunResumeError
from tm_bot.runs.learner import AlphaZeroLearner, LearnerStats
from tm_bot.runs.monitoring import (
    CheckpointMetrics,
    CycleMetrics,
    GameSummary,
    MonitorLogger,
    game_shape_flags,
    policy_entropy,
)
from tm_bot.runs.orchestrator import RunSummary, SelfPlayRunner
from tm_bot.runs.recorder import (
    DecisionRecord,
    GameRecorder,
    RecordingAgent,
    RecordingSearchAgent,
)
from tm_bot.runs.selfplay import (
    OpponentSampler,
    SelfPlayBatch,
    SelfPlayEngine,
    build_selfplay_specs,
    make_train_seed_keys,
    replay_game,
)
from tm_bot.runs.stopping import StopDecision, StopKind, StoppingMonitor

__all__ = [
    "AlphaZeroLearner",
    "CheckpointManager",
    "CheckpointMetrics",
    "CycleMetrics",
    "DecisionRecord",
    "DecisionSample",
    "GameRecorder",
    "GameSummary",
    "LearnerStats",
    "MonitorLogger",
    "OpponentSampler",
    "RecordingAgent",
    "RecordingSearchAgent",
    "ReplayBuffer",
    "ResourceLimits",
    "RunConfig",
    "RunError",
    "RunResourceError",
    "RunResumeError",
    "RunSummary",
    "SelfPlayBatch",
    "SelfPlayEngine",
    "SelfPlayRunner",
    "StopDecision",
    "StopKind",
    "StoppingMonitor",
    "build_selfplay_specs",
    "collate_samples",
    "game_shape_flags",
    "make_train_seed_keys",
    "policy_entropy",
    "replay_game",
]
