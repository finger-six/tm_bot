"""Run configuration, resource limits, and search hyperparameters for self-play."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path
from typing import Any, Mapping

from tm_bot.agents.runner import DEFAULT_MAX_DECISIONS
from tm_bot.env.types import SUPPORTED_MAPS
from tm_bot.agents.play_rules import (
    MARGIN_UTILITY_WEIGHT,
    SELFPLAY_RULES_VERSION,
    PlayTemperatureSchedule,
)
from tm_bot.search.config import (
    FALLBACK_NUM_SIMULATIONS,
    V2_NUM_SIMULATIONS,
    V2_NUM_WORLDS,
    SearchConfig,
)

DEFAULT_RUNS_DIR = Path("runs")
WARMSTART_CHECKPOINT_ENV = "TM_SEARCH_CHECKPOINT"


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    """Per-run resource ceilings from the local experiment budget (#10)."""

    workers_ram_gb: float = 6.0
    learner_ram_gb: float = 20.0
    vram_gb: float = 12.0
    disk_gb: float = 50.0

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if value <= 0.0:
                raise ValueError(f"{f.name} must be > 0, got {value}")


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Configuration for one self-play run.

    Defaults match the #10 budget decision: 16 workers, 24 h of training time
    (`wall_clock_budget_s`: wall clock minus in-run dev evaluation, #26), a
    200k-decision replay buffer, 30-minute checkpoints evaluated on 100 `dev-*`
    games against B1 and 100 against the previous best.

    Self-play follows the v2 rules (#39, #48; `selfplay_rules == "v2"`): 16
    simulations x K = 4 worlds per searched decision (8 x 4 with
    `search_fallback`, which normalises `num_simulations` to 8), τ(g) at every
    learned seat's action-phase decisions (`play_temperature`), utility
    U = o + `utility_margin_weight`·m, and the opponent mix 50% `latest` / 25%
    pool / 10% warm-start anchor / 15% B1. A v1 run config (no
    `selfplay_rules`) cannot be loaded.
    """

    run_name: str
    output_root: Path = DEFAULT_RUNS_DIR
    agent_kind: str = "main"
    workers: int = 16
    batch_size: int = 32
    wall_clock_budget_s: float = 24.0 * 3600.0
    max_decisions_per_game: int = DEFAULT_MAX_DECISIONS

    # Search hyperparameters (shared by self-play and dev evaluation).
    num_simulations: int = V2_NUM_SIMULATIONS
    num_worlds: int = V2_NUM_WORLDS
    c_puct: float = 1.5
    temperature: float = 0.0
    wave_size: int = 4
    search_fallback: bool = False

    # v2 self-play rules (#39, #48), shared by both agent kinds.
    selfplay_rules: str = SELFPLAY_RULES_VERSION
    play_temperature_start: float = 1.0
    play_temperature_floor: float = 0.3
    play_temperature_decay_from: int = 4
    play_temperature_floor_at: int = 10
    utility_margin_weight: float = MARGIN_UTILITY_WEIGHT

    # Seed namespaces. Training never leaves `train-*`; checkpoint selection is
    # `dev-*`; `eval-*` is never touched by a run.
    train_seed_start: int = 0
    dev_seed_start: int = 0
    dev_games_per_opponent: int = 100

    # Self-play and learning.
    games_per_cycle: int = 32
    updates_per_cycle: int = 16
    learner_batch_size: int = 256
    buffer_capacity: int = 200_000
    learning_rate: float = 5e-4
    weight_decay: float = 2e-4
    grad_clip: float = 1.0
    w_policy: float = 1.0
    w_value: float = 1.0
    use_amp: bool = True
    seed: int = 42

    # PPO hyperparameters for the comparator agent kind (ignored by the main agent).
    ppo_clip: float = 0.2
    ppo_gamma: float = 1.0
    ppo_gae_lambda: float = 0.95
    ppo_value_coef: float = 0.5
    ppo_entropy_coef: float = 0.01

    # Opponent sampling (latest / frozen pool / warm-start anchor / B1).
    opponent_latest_prob: float = 0.50
    opponent_pool_prob: float = 0.25
    opponent_anchor_prob: float = 0.10
    opponent_b1_prob: float = 0.15
    opponent_pool_size: int = 10

    # Checkpointing and evaluation.
    checkpoint_interval_s: float = 30.0 * 60.0
    keep_last_checkpoints: int = 10
    dev_eval_enabled: bool = True
    warmstart_checkpoint: str | None = None
    # Train the main agent from random initialisation instead of the
    # warm-start weights (#27). The warm-start checkpoint still supplies the
    # architecture, vocabulary, and card static matrix. Main agent only.
    random_init: bool = False
    device: str | None = None

    # Stopping rules.
    collapse_win_rate_drop: float = 0.15
    collapse_checkpoints: int = 3
    entropy_collapse_fraction: float = 0.10
    entropy_collapse_window_s: float = 2.0 * 3600.0
    plateau_checkpoints: int = 12
    step_cap: int | None = None

    # Monitoring.
    determinism_check_every_games: int = 1_000
    shape_min_generations: int = 6
    shape_max_generations: int = 25

    limits: ResourceLimits = field(default_factory=ResourceLimits)
    maps: tuple[str, ...] = SUPPORTED_MAPS

    def __post_init__(self) -> None:
        if not self.run_name or any(c in self.run_name for c in '<>:"/\\|?*'):
            raise ValueError(f"Invalid run_name {self.run_name!r}")
        if self.agent_kind not in ("main", "comparator"):
            raise ValueError(
                f"agent_kind must be 'main' or 'comparator', got {self.agent_kind!r}"
            )
        if self.workers < 1:
            raise ValueError(f"workers must be >= 1, got {self.workers}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.wall_clock_budget_s <= 0.0:
            raise ValueError("wall_clock_budget_s must be > 0")
        if self.selfplay_rules != SELFPLAY_RULES_VERSION:
            raise ValueError(
                f"selfplay_rules must be {SELFPLAY_RULES_VERSION!r}, got {self.selfplay_rules!r}"
            )
        if self.search_fallback:
            # The fallback budget replaces the default one; `num_simulations` stays the
            # effective count so metrics and search_config() read one field.
            if self.num_simulations == V2_NUM_SIMULATIONS:
                object.__setattr__(self, "num_simulations", FALLBACK_NUM_SIMULATIONS)
            elif self.num_simulations != FALLBACK_NUM_SIMULATIONS:
                raise ValueError(
                    f"search_fallback uses {FALLBACK_NUM_SIMULATIONS} simulations, "
                    f"got {self.num_simulations}"
                )
        if self.num_simulations < FALLBACK_NUM_SIMULATIONS:
            raise ValueError(
                f"num_simulations must be >= {FALLBACK_NUM_SIMULATIONS}, "
                f"got {self.num_simulations}"
            )
        if self.num_worlds < V2_NUM_WORLDS:
            raise ValueError(f"num_worlds must be >= {V2_NUM_WORLDS}, got {self.num_worlds}")
        self.play_temperature  # validates the schedule
        if not (0.0 <= self.utility_margin_weight < 1.0 / 1.2):
            raise ValueError(
                "utility_margin_weight must be in [0, 1/1.2), "
                f"got {self.utility_margin_weight}"
            )
        if not self.maps:
            raise ValueError("maps must be non-empty")
        if self.train_seed_start < 0 or self.dev_seed_start < 0:
            raise ValueError("seed starts must be >= 0")
        if self.dev_games_per_opponent < 1:
            raise ValueError("dev_games_per_opponent must be >= 1")
        if self.games_per_cycle < 1:
            raise ValueError("games_per_cycle must be >= 1")
        if self.updates_per_cycle < 0:
            raise ValueError("updates_per_cycle must be >= 0")
        if self.learner_batch_size < 1:
            raise ValueError("learner_batch_size must be >= 1")
        if self.buffer_capacity < 1:
            raise ValueError("buffer_capacity must be >= 1")
        if self.keep_last_checkpoints < 1:
            raise ValueError("keep_last_checkpoints must be >= 1")
        if self.checkpoint_interval_s <= 0.0:
            raise ValueError("checkpoint_interval_s must be > 0")
        if self.determinism_check_every_games < 1:
            raise ValueError("determinism_check_every_games must be >= 1")
        if self.shape_min_generations >= self.shape_max_generations:
            raise ValueError("shape_min_generations must be < shape_max_generations")
        if not (0.0 <= self.ppo_clip < 1.0):
            raise ValueError(f"ppo_clip must be in [0, 1), got {self.ppo_clip}")
        if not (0.0 <= self.ppo_gae_lambda <= 1.0):
            raise ValueError(
                f"ppo_gae_lambda must be in [0, 1], got {self.ppo_gae_lambda}"
            )
        if self.ppo_gamma <= 0.0:
            raise ValueError(f"ppo_gamma must be > 0, got {self.ppo_gamma}")
        if self.random_init and self.agent_kind == "comparator":
            raise ValueError(
                "random_init is only supported for the main agent; "
                "the comparator must start from its warm-start checkpoint"
            )
        probs = (
            self.opponent_latest_prob,
            self.opponent_pool_prob,
            self.opponent_anchor_prob,
            self.opponent_b1_prob,
        )
        if any(p < 0.0 for p in probs):
            raise ValueError(f"opponent probabilities must be >= 0, got {probs}")
        if abs(sum(probs) - 1.0) > 1e-9:
            raise ValueError(f"opponent probabilities must sum to 1, got {sum(probs)}")

    @property
    def run_dir(self) -> Path:
        return Path(self.output_root) / self.run_name

    @property
    def play_temperature(self) -> PlayTemperatureSchedule:
        """τ(g) for every learned seat's action-phase decisions (#39)."""
        return PlayTemperatureSchedule(
            start=self.play_temperature_start,
            floor=self.play_temperature_floor,
            decay_from=self.play_temperature_decay_from,
            floor_at=self.play_temperature_floor_at,
        )

    def search_config(self) -> SearchConfig:
        """Search hyperparameters used by both self-play seats and dev evaluation."""
        return SearchConfig(
            num_simulations=self.num_simulations,
            num_worlds=self.num_worlds,
            c_puct=self.c_puct,
            temperature=self.temperature,
            wave_size=self.wave_size,
            play_temperature=self.play_temperature,
            margin_weight=self.utility_margin_weight,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_root"] = str(self.output_root)
        payload["maps"] = list(self.maps)
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunConfig:
        rules = payload.get("selfplay_rules")
        if rules != SELFPLAY_RULES_VERSION:
            raise ValueError(
                f"run config uses self-play rules {rules!r}; this code runs only "
                f"{SELFPLAY_RULES_VERSION!r} (#48), so a v1 run cannot be resumed"
            )
        valid = {f.name for f in fields(cls)}
        data = {k: v for k, v in payload.items() if k in valid}
        if "output_root" in data:
            data["output_root"] = Path(data["output_root"])
        if "maps" in data:
            data["maps"] = tuple(data["maps"])
        limits = data.get("limits")
        if isinstance(limits, Mapping):
            data["limits"] = ResourceLimits(**limits)
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path) -> RunConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
