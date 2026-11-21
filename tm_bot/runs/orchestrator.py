"""Self-play run orchestration: cycles, checkpoints, dev selection, stopping, resume."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Protocol

import numpy as np
import torch

from tm_bot.models.network import TMNetwork
from tm_bot.models.vocab import CardVocab
from tm_bot.runs.buffer import DecisionSample, ReplayBuffer
from tm_bot.runs.checkpoints import CheckpointManager
from tm_bot.runs.config import RunConfig
from tm_bot.eval.seeds import generate_seed_keys
from tm_bot.runs.deveval import (
    DevGame,
    DevResult,
    b1_agent_factory,
    build_checkpoint_metrics,
    dev_agent_factory,
    dev_eval_interval,
    dev_score,
    dev_seed_count,
    dev_win_rate,
    evaluate_dev_pairing,
    games_on_seeds,
    paired_delta_d,
)
from tm_bot.runs.errors import RunResourceError, RunResumeError
from tm_bot.runs.learner import (
    AlphaZeroLearner,
    LearnerStats,
    compute_value_calibration,
    probe_policy_entropy,
    select_entropy_probe,
)
from tm_bot.runs.monitoring import (
    CycleMetrics,
    DeterminismResult,
    MonitorLogger,
    count_shape_flags,
    game_shape_flags,
    latency_summary,
    vp_margin_bucket,
)
from tm_bot.runs.resources import check_learner_memory, current_rss_bytes
from tm_bot.runs.selfplay import (
    SelfPlayBatch,
    SelfPlayEngine,
    determinism_check,
)
from tm_bot.runs.stopping import (
    FailureEvent,
    StopDecision,
    StopKind,
    StoppingMonitor,
    new_best_route,
)


class SelfPlaySource(Protocol):
    """Anything that can produce a batch of self-play games + training rows."""

    def play_batch(
        self, num_games: int, *, stop_when: Callable[[], bool] | None = None
    ) -> SelfPlayBatch: ...


class Learner(Protocol):
    step: int

    def train(self, buffer: ReplayBuffer, *, updates: int | None = None) -> LearnerStats: ...


# `evaluator(candidate, opponent, num_games=n)`: opponent None is `b1-v2`.
DevEvaluator = Callable[..., DevResult]

_HARD_STOPS = frozenset(
    {
        StopKind.ENGINE_ERROR,
        StopKind.ADAPTER_REJECTION,
        StopKind.STEP_CAP,
        StopKind.NONFINITE_LOSS,
        StopKind.RESOURCE_LIMIT,
        # A stalled or rule-broken self-play run must not spend more on dev games.
        StopKind.GAME_SHAPE,
    }
)


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Final outcome of a self-play run."""

    run_name: str
    run_dir: Path
    cycles: int
    selfplay_games: int
    steps: int
    elapsed_s: float
    training_s: float
    dev_eval_s: float
    best_dev_win_rate: float | None
    best_checkpoint: Path | None
    last_checkpoint: Path | None
    stop: StopDecision
    summary_path: Path
    init: str = "warmstart"
    init_checkpoint: Path | None = None
    best_dev_score: float | None = None
    anchor_dev_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "selfplay-run-v1",
            "run_name": self.run_name,
            "run_dir": str(self.run_dir),
            "cycles": self.cycles,
            "selfplay_games": self.selfplay_games,
            "steps": self.steps,
            "elapsed_s": self.elapsed_s,
            "training_s": self.training_s,
            "dev_eval_s": self.dev_eval_s,
            "best_dev_win_rate": self.best_dev_win_rate,
            "best_dev_score": self.best_dev_score,
            "anchor_dev_score": self.anchor_dev_score,
            "best_checkpoint": str(self.best_checkpoint) if self.best_checkpoint else None,
            "last_checkpoint": str(self.last_checkpoint) if self.last_checkpoint else None,
            "init": self.init,
            "init_checkpoint": str(self.init_checkpoint) if self.init_checkpoint else None,
            "stop": self.stop.to_dict(),
        }


def _rel(path: Path, base: Path) -> str:
    try:
        return os.path.relpath(path, base)
    except ValueError:  # pragma: no cover - different drives on Windows
        return str(path)


def _rss_gb() -> float | None:
    """Learner process RSS in GiB, the quantity `check_learner_memory` guards."""
    rss = current_rss_bytes()
    return None if rss is None else rss / 1024**3


def _resolve(path: str, base: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (base / p)


class SelfPlayRunner:
    """Runs the self-play loop with checkpointing, dev evaluation, and stopping rules."""

    def __init__(
        self,
        config: RunConfig,
        *,
        engine: SelfPlaySource,
        learner: Learner | None = None,
        evaluator: DevEvaluator | None = None,
        manager: CheckpointManager | None = None,
        clock: Callable[[], float] = time.monotonic,
        model: TMNetwork | None = None,
        vocab: CardVocab | None = None,
        device: str = "cpu",
        determinism_every: int | None = None,
    ) -> None:
        self.config = config
        self.engine = engine
        self.learner = learner
        self.evaluator = evaluator
        self.manager = manager or CheckpointManager(
            config.run_dir,
            keep_last=config.keep_last_checkpoints,
            disk_limit_gb=config.limits.disk_gb,
        )
        self.clock = clock
        self.model = model
        self.vocab = vocab
        self.device = device
        self.determinism_every = (
            config.determinism_check_every_games
            if determinism_every is None
            else determinism_every
        )
        self.logger = MonitorLogger(self.manager.metrics_path)
        self.stopping = StoppingMonitor(
            entropy_collapse_fraction=config.entropy_collapse_fraction,
            entropy_collapse_window_s=config.entropy_collapse_window_s,
            plateau_s=config.plateau_s,
            regression_anchor_win_rate=config.regression_anchor_win_rate,
            margin_collapse_drop=config.margin_collapse_drop,
            calibration_ece=config.calibration_ece,
            consecutive_evaluations=config.stop_consecutive_evaluations,
            # Calibration stops the comparator; for main it is a diagnostic (#40).
            calibration_stops=config.agent_kind == "comparator",
            game_shape_median_generations=config.game_shape_median_generations,
            game_shape_window_s=config.game_shape_window_s,
            step_cap=config.step_cap,
        )
        if config.agent_kind == "comparator":
            from tm_bot.comparator.ppo import PPOBuffer

            self.buffer: Any = PPOBuffer(config.buffer_capacity)
        else:
            self.buffer = ReplayBuffer(config.buffer_capacity)
        self.optimizer: torch.optim.Optimizer | None = None
        self.entropy_probe: list[DecisionSample] | None = None

        self.cycle = 0
        self.selfplay_games = 0
        self.steps = 0
        self.best_dev_win_rate: float | None = None
        self.last_checkpoint: Path | None = None
        self.last_checkpoint_training_s = 0.0
        self.last_checkpoint_step: int | None = None
        # A random-init run saves its fresh weights in <run>/init and uses them
        # as the first previous-best opponent (#27); warm-started runs use the
        # recorded warm-start checkpoint instead. Either is the run's warm-start
        # anchor: the incumbent until a first new best, and the `a` opponent (#40).
        self.init_checkpoint: Path | None = None
        self.previous_best_fallback: Path | None = (
            Path(config.warmstart_checkpoint) if config.warmstart_checkpoint else None
        )
        # Floor-aware selection state (#40 section 1): the incumbent's per-game
        # b1-v2 results for the paired dD, its D, and the anchor's own D.
        self.incumbent_b1_games: tuple[DevGame, ...] | None = None
        self.best_dev_score: float | None = None
        self.anchor_dev_score: float | None = None
        # Dev-eval cadence on the training clock, doubled by the cost cap (#40 section 2).
        self.dev_eval_interval_s = float(config.dev_eval_interval_s)
        self.last_dev_eval_training_s = 0.0
        self.last_dev_eval_step: int | None = None
        self._elapsed_base = 0.0
        # Wall clock spent in in-run dev evaluation; it is off the training clock (#26).
        self.dev_eval_s = 0.0
        self._t_start = self.clock()
        self._determinism_checks_done = 0
        self._stop = StopDecision(stopped=False)

    # --- construction --------------------------------------------------
    @classmethod
    def create(cls, config: RunConfig) -> SelfPlayRunner:
        """Build the real engine, learner, model, and dev evaluator for `config`."""
        from tm_bot.models.network import build_random_init_model, load_checkpoint
        from tm_bot.search.agent import DEFAULT_WARMSTART_CHECKPOINT

        if config.agent_kind == "comparator" and not config.warmstart_checkpoint:
            raise ValueError(
                "comparator runs require --warmstart <comparator warm-start checkpoint>"
            )
        warmstart = config.warmstart_checkpoint or str(DEFAULT_WARMSTART_CHECKPOINT)
        config = replace(config, warmstart_checkpoint=str(warmstart))

        device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if device.startswith("cuda") and config.limits.vram_gb > 0:
            try:
                total = torch.cuda.get_device_properties(0).total_memory
                torch.cuda.set_per_process_memory_fraction(
                    min(1.0, (config.limits.vram_gb * 1024**3) / float(total))
                )
            except Exception:
                pass

        if config.agent_kind == "comparator":
            return cls._create_comparator(config, device=device)

        if config.random_init:
            # The warm start supplies architecture, vocab, and the card static
            # matrix only; every learned weight is re-drawn from config.seed.
            model, vocab, _meta = build_random_init_model(
                config.warmstart_checkpoint, seed=config.seed, map_location=device
            )
        else:
            model, vocab, _meta = load_checkpoint(
                config.warmstart_checkpoint, map_location=device
            )
        model.eval()
        human_aux = None
        if not config.random_init:
            # A random-init run is pure self-play, without the warm-start anchors (#39 §4).
            from tm_bot.training.human_aux import HumanAuxLoss, load_human_aux_sample

            human_aux = HumanAuxLoss(
                load_human_aux_sample(
                    config.human_aux_data_dir,
                    config.human_aux_cache_dir,
                    max_rows=config.human_aux_max_rows,
                    seed=config.seed,
                ),
                rng=np.random.default_rng(config.seed + 1),
            )
        learner = AlphaZeroLearner(config, model, device=device, human_aux=human_aux)
        engine = SelfPlayEngine(
            config,
            model=model,
            vocab=vocab,
            device=device,
            rng=np.random.default_rng(config.seed),
        )
        runner = cls(
            config,
            engine=engine,
            learner=learner,
            manager=CheckpointManager(
                config.run_dir,
                keep_last=config.keep_last_checkpoints,
                disk_limit_gb=config.limits.disk_gb,
            ),
            model=model,
            vocab=vocab,
            device=device,
            clock=time.monotonic,
        )
        runner.optimizer = learner.optimizer
        learner.training_clock = runner.training_s

        def evaluator(candidate: Path, opponent: Path | None, *, num_games: int) -> DevResult:
            candidate_factory = dev_agent_factory(
                candidate, device=device, search_config=config.search_config()
            )
            opponent_factory = (
                b1_agent_factory()
                if opponent is None
                else dev_agent_factory(
                    opponent, device=device, search_config=config.search_config()
                )
            )
            return evaluate_dev_pairing(
                candidate_factory=candidate_factory,
                opponent_factory=opponent_factory,
                num_games=num_games,
                maps=config.maps,
                seed_start=config.dev_seed_start,
                num_workers=config.workers,
                batch_size=config.batch_size,
                agent_a_name="candidate",
                agent_b_name="b1" if opponent is None else "checkpoint",
                record_values="main" if opponent is None else None,
            )

        runner.evaluator = evaluator
        if not config.dev_eval_enabled:
            runner.evaluator = None
        return runner

    @classmethod
    def _create_comparator(cls, config: RunConfig, *, device: str) -> SelfPlayRunner:
        """Build the independently designed comparator engine and PPO learner."""
        from tm_bot.comparator.agent import comparator_eval_factory
        from tm_bot.comparator.model import load_comparator_checkpoint
        from tm_bot.comparator.ppo import PPOLearner
        from tm_bot.comparator.selfplay import ComparatorSelfPlayEngine

        model, vocab, _meta = load_comparator_checkpoint(
            config.warmstart_checkpoint, map_location=device
        )
        model.eval()
        # Same warm-start retention as the main learner (#39 §1d, #50). Comparator runs
        # always warm start (`random_init` is main-only), so the human sample always loads.
        from tm_bot.training.human_aux import HumanAuxLoss, load_human_aux_sample

        human_aux = HumanAuxLoss(
            load_human_aux_sample(
                config.human_aux_data_dir,
                config.human_aux_cache_dir,
                max_rows=config.human_aux_max_rows,
                seed=config.seed,
            ),
            rng=np.random.default_rng(config.seed + 1),
        )
        learner = PPOLearner(config, model, device=device, human_aux=human_aux)
        engine = ComparatorSelfPlayEngine(
            config,
            model=model,
            vocab=vocab,
            device=device,
            rng=np.random.default_rng(config.seed),
        )
        runner = cls(
            config,
            engine=engine,
            learner=learner,
            manager=CheckpointManager(
                config.run_dir,
                keep_last=config.keep_last_checkpoints,
                disk_limit_gb=config.limits.disk_gb,
            ),
            model=model,  # type: ignore[arg-type]
            vocab=vocab,
            device=device,
            clock=time.monotonic,
        )
        runner.optimizer = learner.optimizer
        learner.training_clock = runner.training_s

        def comparator_factory(checkpoint: Path):
            cmp_model, cmp_vocab, _ = load_comparator_checkpoint(
                checkpoint, map_location=device, encoder_version=None
            )
            cmp_model.eval()
            return comparator_eval_factory(
                cmp_model,
                cmp_vocab,
                device=device,
                version=f"dev-{Path(checkpoint).name}",
            )

        def evaluator(candidate: Path, opponent: Path | None, *, num_games: int) -> DevResult:
            candidate_factory = comparator_factory(candidate)
            opponent_factory = (
                b1_agent_factory()
                if opponent is None
                else comparator_factory(opponent)
            )
            return evaluate_dev_pairing(
                candidate_factory=candidate_factory,
                opponent_factory=opponent_factory,
                num_games=num_games,
                maps=config.maps,
                seed_start=config.dev_seed_start,
                num_workers=config.workers,
                batch_size=config.batch_size,
                agent_a_name="comparator",
                agent_b_name="b1" if opponent is None else "checkpoint",
                record_values="comparator" if opponent is None else None,
            )

        runner.evaluator = evaluator
        if not config.dev_eval_enabled:
            runner.evaluator = None
        return runner

    # --- time ----------------------------------------------------------
    def elapsed_s(self) -> float:
        """Total wall clock of the run, including in-run dev evaluation."""
        return self._elapsed_base + (self.clock() - self._t_start)

    def training_s(self) -> float:
        """Training clock: wall clock minus in-run dev evaluation (#26).

        The budget, checkpoint interval, and entropy-collapse window run on it.
        """
        return self.elapsed_s() - self.dev_eval_s

    def _deadline_reached(self) -> bool:
        return self.training_s() >= self.config.wall_clock_budget_s

    # --- resume --------------------------------------------------------
    def _restore(self) -> bool:
        state = self.manager.load_state()
        if state is None:
            return False
        init = state.get("init_checkpoint")
        if init:
            # Resume must keep comparing against the run's own init checkpoint,
            # not the warm start it borrowed its architecture from (#27).
            self.init_checkpoint = _resolve(str(init), self.manager.run_dir)
            if not (self.init_checkpoint / "checkpoint.pt").exists():
                raise RunResumeError(
                    f"Run {self.config.run_name} records init checkpoint "
                    f"{self.init_checkpoint}, but it is missing"
                )
            self.previous_best_fallback = self.init_checkpoint
        checkpoint = self.manager.resolve_resume_checkpoint(state)
        if checkpoint is None and self.init_checkpoint is not None:
            # A random-init run records init from the start, so a crash before
            # the first periodic checkpoint still has something to resume from.
            checkpoint = self.init_checkpoint
        if checkpoint is None:
            raise RunResumeError(
                f"Run {self.config.run_name} has state but no loadable checkpoint"
            )
        if self.model is not None:
            self.manager.restore_model(self.model, checkpoint)
        payload = self.manager.load_training_state(
            optimizer=self.optimizer,
            buffer=self.buffer,
        )
        if payload is not None and payload.get("entropy_probe") is not None:
            self.entropy_probe = list(payload["entropy_probe"])
        self.cycle = int(state.get("cycle", 0))
        self.steps = int(state.get("step", 0))
        self.selfplay_games = int(state.get("selfplay_games", 0))
        self._determinism_checks_done = int(state.get("determinism_checks_done", 0))
        best = state.get("best_dev_win_rate")
        self.best_dev_win_rate = None if best is None else float(best)
        self._elapsed_base = float(state.get("elapsed_s", 0.0))
        self.dev_eval_s = float(state.get("dev_eval_s", 0.0))
        self.last_checkpoint_training_s = float(state.get("last_checkpoint_training_s", 0.0))
        step = state.get("last_checkpoint_step")
        self.last_checkpoint_step = None if step is None else int(step)
        self.last_checkpoint = checkpoint
        if self.learner is not None and hasattr(self.learner, "step"):
            self.learner.step = self.steps  # type: ignore[attr-defined]
        pool = [_resolve(str(p), self.manager.run_dir) for p in state.get("pool", [])]
        setter = getattr(self.engine, "set_opponent_pool", None)
        if callable(setter):
            setter(pool)
        seed_cursor = state.get("seed_cursor")
        if seed_cursor is not None and hasattr(self.engine, "seed_cursor"):
            self.engine.seed_cursor = int(seed_cursor)  # type: ignore[attr-defined]
        entropy_baseline = state.get("entropy_baseline")
        if entropy_baseline is not None:
            self.stopping.set_entropy_baseline(
                float(entropy_baseline), elapsed_s=self._elapsed_base - self.dev_eval_s
            )
        self.stopping.load_state_dict(state.get("stopping") or {})
        games = state.get("incumbent_b1_games")
        self.incumbent_b1_games = (
            None if games is None else tuple(DevGame.from_dict(g) for g in games)
        )
        for name in ("best_dev_score", "anchor_dev_score"):
            value = state.get(name)
            setattr(self, name, None if value is None else float(value))
        self.dev_eval_interval_s = float(
            state.get("dev_eval_interval_s", self.config.dev_eval_interval_s)
        )
        self.last_dev_eval_training_s = float(state.get("last_dev_eval_training_s", 0.0))
        step = state.get("last_dev_eval_step")
        self.last_dev_eval_step = None if step is None else int(step)
        self.logger.load_existing()
        return True

    def _state_payload(self) -> dict[str, Any]:
        pool = getattr(getattr(self.engine, "opponents", None), "pool", ())
        return {
            "cycle": self.cycle,
            "step": self.steps,
            "selfplay_games": self.selfplay_games,
            "best_dev_win_rate": self.best_dev_win_rate,
            "elapsed_s": self.elapsed_s(),
            "training_s": self.training_s(),
            "dev_eval_s": self.dev_eval_s,
            "last_checkpoint_training_s": self.last_checkpoint_training_s,
            "last_checkpoint_step": self.last_checkpoint_step,
            "last_checkpoint": (
                _rel(self.last_checkpoint, self.manager.run_dir)
                if self.last_checkpoint is not None
                else None
            ),
            "init": "random" if self.config.random_init else "warmstart",
            "init_checkpoint": (
                _rel(self.init_checkpoint, self.manager.run_dir)
                if self.init_checkpoint is not None
                else None
            ),
            "pool": [_rel(Path(p), self.manager.run_dir) for p in pool],
            "seed_cursor": getattr(self.engine, "seed_cursor", None),
            "determinism_checks_done": self._determinism_checks_done,
            "entropy_baseline": self.stopping.entropy_baseline,
            "stopping": self.stopping.state_dict(),
            "best_dev_score": self.best_dev_score,
            "anchor_dev_score": self.anchor_dev_score,
            "incumbent_b1_games": (
                None
                if self.incumbent_b1_games is None
                else [g.to_dict() for g in self.incumbent_b1_games]
            ),
            "dev_eval_interval_s": self.dev_eval_interval_s,
            "last_dev_eval_training_s": self.last_dev_eval_training_s,
            "last_dev_eval_step": self.last_dev_eval_step,
            "stop": self._stop.to_dict(),
        }

    def _save_state(self) -> None:
        self.manager.save_state(
            payload=self._state_payload(),
            model=self.model,
            optimizer=self.optimizer,
            buffer=self.buffer if len(self.buffer) else None,
            entropy_probe=self.entropy_probe,
        )

    def _init_metadata(self) -> dict[str, Any]:
        """Checkpoint provenance: warm start, or random weights seeded from config."""
        if self.config.random_init:
            return {
                "init": "random",
                "warmstart_checkpoint": None,
                "architecture_from": self.config.warmstart_checkpoint,
            }
        return {
            "init": "warmstart",
            "warmstart_checkpoint": self.config.warmstart_checkpoint,
        }

    def _prepare_random_init(self) -> None:
        """Save the fresh weights as the first previous-best opponent (#27)."""
        if not self.config.random_init or self.model is None or self.vocab is None:
            return
        self.init_checkpoint = self.manager.save_init(
            model=self.model,
            vocab=self.vocab,
            metadata={
                "version": f"selfplay-{self.config.run_name}",
                "agent_kind": self.config.agent_kind,
                "cycle": 0,
                "step": 0,
                "selfplay_games": 0,
                "seed": self.config.seed,
                "temperature": getattr(self.model, "temperature", None),
                **self._init_metadata(),
            },
        )
        self.previous_best_fallback = self.init_checkpoint
        # Record the reference immediately so an early crash still resumes with it.
        self._save_state()

    # --- pre-update measurements --------------------------------------
    def _measure_pre_update(
        self, batch: SelfPlayBatch
    ) -> tuple[dict[str, Any] | None, float | None]:
        """Calibration + fixed-probe entropy on frozen, pre-update weights."""
        calibration: dict[str, Any] | None = None
        probe_entropy: float | None = None
        training_samples = batch.training_samples
        if self.model is not None and training_samples:
            calibration = compute_value_calibration(
                self.model, training_samples, device=self.device  # type: ignore[arg-type]
            )
            if self.entropy_probe is None:
                self.entropy_probe = select_entropy_probe(training_samples)
        if self.model is not None and self.entropy_probe:
            probe_entropy = probe_policy_entropy(
                self.model, self.entropy_probe, device=self.device
            )
            if self.stopping.entropy_baseline is None and probe_entropy is not None:
                self.stopping.set_entropy_baseline(
                    probe_entropy, elapsed_s=self.training_s()
                )
        return calibration, probe_entropy

    # --- one cycle -----------------------------------------------------
    def _log_cycle(
        self,
        batch: SelfPlayBatch,
        *,
        learner_stats: LearnerStats | None,
        calibration: dict[str, Any] | None,
        probe_entropy: float | None,
        determinism: DeterminismResult | None,
        cycle_wall_s: float,
    ) -> CycleMetrics:
        games = batch.games
        completed = [g for g in games if g.ok]
        generations = [int(g.generations) for g in completed if g.generations is not None]
        generation_counts: dict[str, int] = {}
        for gen in generations:
            generation_counts[str(gen)] = generation_counts.get(str(gen), 0) + 1
        margins: list[float] = []
        wins = losses = ties = 0
        by_opponent: dict[str, dict[str, int]] = {}
        for i, g in enumerate(games):
            if not g.ok or g.scores is None:
                continue
            seat = batch.training_seats[i] if i < len(batch.training_seats) else 0
            margins.append(float(g.scores[seat] - g.scores[1 - seat]))
            if g.tie:
                ties += 1
                result = "ties"
            elif g.winner == seat:
                wins += 1
                result = "wins"
            else:
                losses += 1
                result = "losses"
            if i < len(batch.opponent_kinds):
                counts = by_opponent.setdefault(
                    batch.opponent_kinds[i], {"wins": 0, "losses": 0, "ties": 0}
                )
                counts[result] += 1
        histogram: dict[str, int] = {}
        for margin in margins:
            label = vp_margin_bucket(margin)
            histogram[label] = histogram.get(label, 0) + 1
        kinds: dict[str, int] = {}
        for kind in batch.opponent_kinds:
            kinds[kind] = kinds.get(kind, 0) + 1
        rejections = sum(
            1 for g in games if not g.ok and g.error_type == "AdapterBugError"
        )
        wall = self.elapsed_s()
        record = CycleMetrics(
            cycle=self.cycle,
            elapsed_s=float(wall),
            training_s=float(wall - self.dev_eval_s),
            dev_eval_s=float(self.dev_eval_s),
            wall_clock_s=float(cycle_wall_s),
            selfplay_games=len(games),
            total_selfplay_games=self.selfplay_games,
            decisions=int(batch.decisions),
            searched_decisions=int(batch.searched_decisions),
            searched_share=(
                float(batch.searched_decisions) / float(batch.decisions)
                if batch.decisions > 0
                else None
            ),
            rejections=rejections,
            failures=len(games) - len(completed),
            wins=wins,
            losses=losses,
            ties=ties,
            mean_generations=(float(np.mean(generations)) if generations else None),
            min_generations=(int(min(generations)) if generations else None),
            max_generations=(int(max(generations)) if generations else None),
            median_generations=(float(np.median(generations)) if generations else None),
            generation_counts=generation_counts,
            results_by_opponent=by_opponent,
            rss_gb=_rss_gb(),
            mean_vp_margin=(float(np.mean(margins)) if margins else None),
            vp_histogram=histogram,
            opponent_kinds=kinds,
            mean_decisions_per_game=(
                float(np.mean([g.decisions for g in completed])) if completed else None
            ),
            policy_loss=None if learner_stats is None else learner_stats.policy_loss,
            value_loss=None if learner_stats is None else learner_stats.value_loss,
            total_loss=None if learner_stats is None else learner_stats.total_loss,
            margin_loss=None if learner_stats is None else learner_stats.margin_loss,
            kl_anchor=None if learner_stats is None else learner_stats.kl_anchor,
            human_aux_loss=None if learner_stats is None else learner_stats.human_aux_loss,
            human_aux_weight=None if learner_stats is None else learner_stats.human_aux_weight,
            policy_entropy=probe_entropy,
            target_policy_entropy=batch.entropy,
            value_calibration=calibration or {},
            decision_latency_ms=latency_summary(batch.latencies_ms),
            shape_flag_count=count_shape_flags(
                games,
                min_generations=self.config.shape_min_generations,
                max_generations=self.config.shape_max_generations,
            ),
            shape_flags=game_shape_flags(
                games,
                min_generations=self.config.shape_min_generations,
                max_generations=self.config.shape_max_generations,
            ),
            determinism_checked=determinism is not None,
            determinism_ok=None if determinism is None else determinism.ok,
            throughput_games_per_hour=(
                self.selfplay_games / (wall / 3600.0) if wall > 0.0 else None
            ),
            stop_reason=(self._stop.kind.value if self._stop.stopped else None),
        )
        self.logger.log_cycle(record)
        return record

    def _handle_failures(self, batch: SelfPlayBatch, elapsed: float) -> None:
        failures: list[FailureEvent] = []
        for i, game in enumerate(batch.games):
            if game.ok:
                continue
            recorder = batch.recorders[i] if i < len(batch.recorders) else None
            case = {
                "seed_key": game.seed_key,
                "map_name": game.map_name,
                "error_type": game.error_type or "UnknownError",
                "error": game.error or "",
                "decisions": int(game.decisions),
                "actions": list(recorder.actions) if recorder is not None else [],
                "training_seat": (
                    batch.training_seats[i] if i < len(batch.training_seats) else None
                ),
                "config": {
                    "num_simulations": self.config.num_simulations,
                    "num_worlds": self.config.num_worlds,
                    "checkpoint_interval_s": self.config.checkpoint_interval_s,
                },
            }
            path = self.manager.write_regression_case(case)
            self.logger.log_rejection({**case, "regression_case": str(path)})
            failures.append(
                FailureEvent(
                    seed_key=game.seed_key,
                    map_name=game.map_name,
                    error_type=game.error_type or "UnknownError",
                    error=game.error or "",
                    decisions=int(game.decisions),
                )
            )
        if failures:
            decision = self.stopping.observe_failures(failures, elapsed_s=elapsed)
            if decision.stopped:
                self._stop = decision

    def _maybe_determinism_check(self, batch: SelfPlayBatch) -> DeterminismResult | None:
        if self.determinism_every <= 0:
            return None
        due = self.selfplay_games // self.determinism_every
        if due <= self._determinism_checks_done:
            return None
        self._determinism_checks_done = due
        index = None
        for i, game in enumerate(batch.games):
            if game.ok and i < len(batch.recorders):
                index = i
        if index is None:
            return None
        recorder = batch.recorders[index]
        return determinism_check(batch.games[index], recorder.actions)

    def _save_checkpoint(self, elapsed: float) -> bool:
        """Save a periodic checkpoint for the opponent pool; False on a disk-cap stop."""
        if self.model is None or self.vocab is None:
            return False
        try:
            self.last_checkpoint = self.manager.save(
                step=self.steps,
                model=self.model,
                vocab=self.vocab,
                metadata={
                    "version": f"selfplay-{self.config.run_name}",
                    "agent_kind": self.config.agent_kind,
                    "cycle": self.cycle,
                    "step": self.steps,
                    "selfplay_games": self.selfplay_games,
                    "num_simulations": self.config.num_simulations,
                    "num_worlds": self.config.num_worlds,
                    "temperature": getattr(self.model, "temperature", None),
                    **self._init_metadata(),
                },
            )
        except RunResourceError as exc:
            self._stop = StopDecision(
                stopped=True,
                kind=StopKind.RESOURCE_LIMIT,
                reason=str(exc),
                details={"elapsed_s": elapsed},
            )
            return False
        self.last_checkpoint_training_s = elapsed
        self.last_checkpoint_step = self.steps
        setter = getattr(self.engine, "set_opponent_pool", None)
        if callable(setter):
            pool = self.manager.list_checkpoints()[-self.config.opponent_pool_size :]
            setter(pool)
        # Make the new checkpoint resumable before spending time on dev games.
        self._save_state()
        return True

    def _dev_eval_pending(self) -> bool:
        """The newest checkpoint exists and has not been dev-evaluated yet."""
        return (
            self.evaluator is not None
            and self.model is not None
            and self.vocab is not None
            and self.last_checkpoint is not None
            and self.last_checkpoint_step is not None
            and self.last_dev_eval_step != self.last_checkpoint_step
            and not (self._stop.stopped and self._stop.kind in _HARD_STOPS)
        )

    def _anchor_seed_keys(self) -> list[str]:
        count = dev_seed_count(self.config.dev_anchor_games, self.config.maps)
        return [
            k.key for k in generate_seed_keys("dev", count, start=self.config.dev_seed_start)
        ]

    def _dev_evaluate(self, elapsed: float) -> None:
        """Dev-evaluate the newest checkpoint and apply floor-aware selection (#40 sections 1-2).

        Pairings: `b1-v2` (dev score D, ECE), the incumbent best (head-to-head h),
        and the warm-start anchor (a). Until a first new best the anchor is the
        incumbent, so `a` comes from the incumbent pairing's anchor seeds, and the
        anchor's own `b1-v2` games are measured once for the paired dD.
        """
        assert self.evaluator is not None and self.last_checkpoint is not None
        config = self.config
        candidate = self.last_checkpoint
        anchor = self.previous_best_fallback
        best = self.manager.best_checkpoint()
        incumbent = best if best is not None else anchor
        anchor_is_incumbent = best is None and anchor is not None

        t_eval = self.clock()
        baseline_s = 0.0
        baseline_failures = 0
        vs_anchor: DevResult | None = None
        try:
            if incumbent is not None and self.incumbent_b1_games is None:
                t_base = self.clock()
                baseline = self.evaluator(
                    incumbent, None, num_games=config.dev_games_per_opponent
                )
                baseline_s = self.clock() - t_base
                baseline_failures = baseline.failures
                self.incumbent_b1_games = baseline.records
                self.best_dev_score = dev_score(baseline.records)
                if anchor_is_incumbent:
                    self.anchor_dev_score = self.best_dev_score
            vs_b1 = self.evaluator(candidate, None, num_games=config.dev_games_per_opponent)
            vs_best = (
                self.evaluator(candidate, incumbent, num_games=config.dev_games_per_opponent)
                if incumbent is not None
                else None
            )
            if anchor is not None and not anchor_is_incumbent:
                vs_anchor = self.evaluator(candidate, anchor, num_games=config.dev_anchor_games)
        finally:
            eval_wall_s = self.clock() - t_eval
            self.dev_eval_s += eval_wall_s

        anchor_games: tuple[DevGame, ...] = ()
        anchor_win_rate: float | None = None
        if vs_anchor is not None:
            anchor_games = vs_anchor.records
            anchor_win_rate = vs_anchor.win_rate
        elif anchor_is_incumbent and vs_best is not None:
            anchor_games = games_on_seeds(vs_best.records, self._anchor_seed_keys())
            anchor_win_rate = dev_win_rate(anchor_games)
        # A rule bug in the one-time anchor baseline games is still a rule bug.
        failures = (
            baseline_failures
            + vs_b1.failures
            + (0 if vs_best is None else vs_best.failures)
            + (0 if vs_anchor is None else vs_anchor.failures)
        )
        score = dev_score(vs_b1.records)
        delta: float | None = None
        delta_ci: tuple[float, float] | None = None
        if incumbent is not None and self.incumbent_b1_games is not None:
            delta, delta_ci = paired_delta_d(vs_b1.records, self.incumbent_b1_games)
        route = new_best_route(
            failures=failures,
            anchor_win_rate=anchor_win_rate,
            head_to_head=None if vs_best is None else vs_best.win_rate,
            head_to_head_ci_95=None if vs_best is None else vs_best.win_rate_ci_95,
            delta_d_ci_95=delta_ci,
        )
        if route is not None:
            try:
                self.manager.save_best(
                    model=self.model,  # type: ignore[arg-type]
                    vocab=self.vocab,  # type: ignore[arg-type]
                    metadata={
                        "version": f"selfplay-{config.run_name}",
                        "cycle": self.cycle,
                        "step": self.last_checkpoint_step,
                        "selfplay_games": self.selfplay_games,
                        "dev_win_rate_vs_b1": vs_b1.win_rate,
                        "dev_score_vs_b1": score,
                        "new_best_route": route,
                        "temperature": getattr(self.model, "temperature", None),
                        **self._init_metadata(),
                    },
                )
            except RunResourceError as exc:
                self._stop = StopDecision(
                    stopped=True,
                    kind=StopKind.RESOURCE_LIMIT,
                    reason=str(exc),
                    details={"elapsed_s": elapsed},
                )
                return
            self.best_dev_win_rate = vs_b1.win_rate
            self.best_dev_score = score
            self.incumbent_b1_games = vs_b1.records

        # The one-time anchor baseline is not part of the recurring cost.
        self.dev_eval_interval_s = dev_eval_interval(
            eval_wall_s - baseline_s,
            base_interval_s=config.dev_eval_interval_s,
            cost_fraction=config.dev_eval_cost_fraction,
        )
        self.last_dev_eval_training_s = elapsed
        self.last_dev_eval_step = self.last_checkpoint_step

        wall = self.elapsed_s()
        record = build_checkpoint_metrics(
            cycle=self.cycle,
            elapsed_s=wall,
            step=int(self.last_checkpoint_step or 0),
            games=self.selfplay_games,
            checkpoint=candidate,
            vs_b1=vs_b1,
            vs_previous_best=vs_best,
            new_best=route is not None,
            best_dev_win_rate=self.best_dev_win_rate,
            training_s=wall - self.dev_eval_s,
            dev_eval_s=self.dev_eval_s,
            dev_score_vs_b1=score,
            best_dev_score_vs_b1=self.best_dev_score,
            dev_score_delta=delta,
            dev_score_delta_ci_95=delta_ci,
            dev_win_rate_vs_anchor=anchor_win_rate,
            dev_anchor_games=(
                vs_anchor.games if vs_anchor is not None else len(anchor_games)
            ),
            dev_ece_vs_b1=vs_b1.ece,
            dev_failures=failures,
            new_best_route=route,
            dev_eval_wall_s=eval_wall_s,
            dev_eval_interval_s=self.dev_eval_interval_s,
            dev_eval_baseline_s=baseline_s,
            rss_gb=_rss_gb(),
        )
        self.logger.log_checkpoint(record)
        decision = self.stopping.observe_checkpoint(record)
        if decision.stopped and not self._stop.stopped:
            self._stop = decision
        self._save_state()

    def _rollback(self) -> tuple[bool, str]:
        """Restore model and optimizer from the last good checkpoint (no silent failure)."""
        checkpoint = self.last_checkpoint or self.manager.resolve_resume_checkpoint(None)
        if checkpoint is None or self.model is None:
            return False, "no checkpoint available"
        try:
            self.manager.restore_model(self.model, checkpoint)
            if self.optimizer is not None:
                self.manager.load_training_state(optimizer=self.optimizer)
            self._save_state()
            return True, str(checkpoint)
        except Exception as exc:  # reported in the stop decision
            return False, f"{type(exc).__name__}: {exc}"

    def run(self, *, max_cycles: int | None = None, resume: bool = False) -> RunSummary:
        """Run until a stopping rule fires, the budget is exhausted, or `max_cycles`."""
        self.manager.initialize(self.config)
        if resume:
            self._restore()
        elif self.manager.load_state() is not None:
            raise RunResumeError(
                f"Run {self.config.run_name} already exists in {self.manager.run_dir}; "
                "pass resume=True or choose a new run name"
            )
        else:
            self._prepare_random_init()

        if self.model is not None and self.vocab is not None:
            setter = getattr(self.engine, "set_opponent_pool", None)
            if callable(setter):
                setter(self.manager.list_checkpoints()[-self.config.opponent_pool_size :])

        while True:
            elapsed = self.training_s()
            budget = self.stopping.check_budget(elapsed, self.config.wall_clock_budget_s)
            if budget.stopped:
                self._stop = budget
                break
            if max_cycles is not None and self.cycle >= max_cycles:
                break

            t_cycle = self.clock()
            batch = self.engine.play_batch(
                self.config.games_per_cycle, stop_when=self._deadline_reached
            )
            deadline_hit = any(g.error_type == "DeadlineReached" for g in batch.games)
            if deadline_hit:
                keep = [i for i, g in enumerate(batch.games) if g.error_type != "DeadlineReached"]
                batch.games = [batch.games[i] for i in keep]
                batch.recorders = [batch.recorders[i] for i in keep]
                batch.training_seats = [batch.training_seats[i] for i in keep]
                batch.opponent_kinds = [batch.opponent_kinds[i] for i in keep]
            self.cycle += 1
            self.selfplay_games += len(batch.games)
            elapsed = self.training_s()

            calibration, probe_entropy = self._measure_pre_update(batch)
            self._handle_failures(batch, elapsed)
            if self._stop.stopped:
                self._log_cycle(
                    batch,
                    learner_stats=None,
                    calibration=calibration,
                    probe_entropy=probe_entropy,
                    determinism=None,
                    cycle_wall_s=self.clock() - t_cycle,
                )
                break
            if deadline_hit:
                self._stop = StopDecision(
                    stopped=True,
                    kind=StopKind.BUDGET,
                    reason="training-time deadline reached inside the self-play batch",
                    details={"elapsed_s": elapsed, "budget_s": self.config.wall_clock_budget_s},
                )

            # PPO is on-policy and trains only on this cycle's rollout, so comparator
            # rollouts are never kept (or persisted) after their update.
            is_comparator = self.config.agent_kind == "comparator"
            if batch.samples and not is_comparator:
                self.buffer.extend(batch.samples)
            has_training_data = bool(batch.ppo_samples) if is_comparator else len(self.buffer) > 0
            learner_stats: LearnerStats | None = None
            if self.learner is not None and has_training_data and not self._stop.stopped:
                if is_comparator:
                    learner_stats = self.learner.train(  # type: ignore[call-arg]
                        self.buffer, samples=batch.ppo_samples
                    )
                else:
                    learner_stats = self.learner.train(self.buffer)
                self.steps = int(getattr(self.learner, "step", self.steps))
                if (
                    learner_stats.nonfinite_loss is not None
                    or learner_stats.nonfinite_params
                ):
                    nonfinite_value = (
                        learner_stats.nonfinite_loss
                        if learner_stats.nonfinite_loss is not None
                        else float("nan")
                    )
                    loss_decision = self.stopping.observe_loss(
                        nonfinite_value, elapsed_s=elapsed
                    )
                    rolled_back, rollback_detail = self._rollback()
                    self._stop = StopDecision(
                        stopped=True,
                        kind=loss_decision.kind or StopKind.NONFINITE_LOSS,
                        reason=(
                            f"non-finite loss/params ({nonfinite_value!r}, "
                            f"params_finite={not learner_stats.nonfinite_params}); "
                            f"rollback={'ok' if rolled_back else 'FAILED'}"
                        ),
                        details={
                            "loss": learner_stats.nonfinite_loss,
                            "nonfinite_params": learner_stats.nonfinite_params,
                            "rolled_back": rolled_back,
                            "rollback_detail": rollback_detail,
                            "elapsed_s": elapsed,
                        },
                    )

            determinism = self._maybe_determinism_check(batch)
            cycle_record = self._log_cycle(
                batch,
                learner_stats=learner_stats,
                calibration=calibration,
                probe_entropy=probe_entropy,
                determinism=determinism,
                cycle_wall_s=self.clock() - t_cycle,
            )
            if determinism is not None:
                self.logger.log_determinism(determinism)
            shape = self.stopping.observe_game_shape(
                cycle_record.median_generations, elapsed_s=elapsed
            )
            if shape.stopped and not self._stop.stopped:
                self._stop = shape

            if not self._stop.stopped:
                try:
                    check_learner_memory(self.config.limits)
                except RunResourceError as exc:
                    self._stop = StopDecision(
                        stopped=True,
                        kind=StopKind.RESOURCE_LIMIT,
                        reason=str(exc),
                        details={"elapsed_s": elapsed},
                    )

            if probe_entropy is not None and not self._stop.stopped:
                decision = self.stopping.observe_entropy(probe_entropy, elapsed_s=elapsed)
                if decision.stopped:
                    self._stop = decision

            step_decision = self.stopping.check_steps(self.steps, elapsed_s=elapsed)
            if step_decision.stopped and not self._stop.stopped:
                self._stop = step_decision

            # Checkpoints are saved every `checkpoint_interval_s` of training clock
            # for the opponent pool; the newest is dev-evaluated when the (cost-
            # capped) dev cadence is due, and before a soft stop or the last cycle.
            # A model that has not trained since the last checkpoint is already saved.
            closing = (self._stop.stopped and self._stop.kind not in _HARD_STOPS) or (
                max_cycles is not None and self.cycle >= max_cycles
            )
            due = (
                self.model is not None
                and self.vocab is not None
                and (self.last_checkpoint is None or self.last_checkpoint_step != self.steps)
                and (
                    closing
                    or elapsed - self.last_checkpoint_training_s
                    >= self.config.checkpoint_interval_s
                )
            )
            saved = False
            if due and not (self._stop.stopped and self._stop.kind in _HARD_STOPS):
                saved = self._save_checkpoint(elapsed)
            eval_due = elapsed - self.last_dev_eval_training_s >= self.dev_eval_interval_s
            if self._dev_eval_pending() and (closing or (saved and eval_due)):
                self._dev_evaluate(elapsed)

            if self._stop.stopped:
                break

        return self._finalize()

    def _finalize(self) -> RunSummary:
        elapsed = self.training_s()
        hard_stop = self._stop.stopped and self._stop.kind in _HARD_STOPS
        # The last checkpoint already holds the model when no update ran since it.
        unchanged = self.last_checkpoint is not None and self.last_checkpoint_step == self.steps
        if self.model is not None and self.vocab is not None and hard_stop:
            # A broken run must not spend more compute on dev games; save the
            # rolled-back state so the run can be resumed after the bug is fixed.
            if self._stop.kind != StopKind.RESOURCE_LIMIT and not unchanged:
                try:
                    self.last_checkpoint = self.manager.save(
                        step=self.steps,
                        model=self.model,
                        vocab=self.vocab,
                        metadata={
                            "cycle": self.cycle,
                            "step": self.steps,
                            "stop": self._stop.to_dict(),
                            "temperature": getattr(self.model, "temperature", None),
                        },
                    )
                    self.last_checkpoint_step = self.steps
                except RunResourceError:
                    self._stop = StopDecision(
                        stopped=True,
                        kind=StopKind.RESOURCE_LIMIT,
                        reason=(
                            f"{self._stop.reason}; final checkpoint skipped "
                            "(disk cap)"
                        ),
                        details=self._stop.details,
                    )
        elif self.model is not None and self.vocab is not None and self.evaluator is not None:
            if not unchanged:
                self._save_checkpoint(elapsed)
            if self._dev_eval_pending():
                self._dev_evaluate(elapsed)
        elif self.model is not None and self.vocab is not None and not unchanged:
            try:
                self.last_checkpoint = self.manager.save(
                    step=self.steps,
                    model=self.model,
                    vocab=self.vocab,
                    metadata={
                        "cycle": self.cycle,
                        "step": self.steps,
                        "temperature": getattr(self.model, "temperature", None),
                    },
                )
                self.last_checkpoint_step = self.steps
            except RunResourceError as exc:
                self._stop = StopDecision(
                    stopped=True,
                    kind=StopKind.RESOURCE_LIMIT,
                    reason=str(exc),
                    details={"elapsed_s": elapsed},
                )
        self._save_state()
        summary_path = self.logger.write_summary(self.manager.summary_path)
        run_summary = RunSummary(
            run_name=self.config.run_name,
            run_dir=self.manager.run_dir,
            cycles=self.cycle,
            selfplay_games=self.selfplay_games,
            steps=self.steps,
            elapsed_s=self.elapsed_s(),
            training_s=self.training_s(),
            dev_eval_s=self.dev_eval_s,
            best_dev_win_rate=self.best_dev_win_rate,
            best_checkpoint=self.manager.best_checkpoint(),
            last_checkpoint=self.last_checkpoint,
            stop=self._stop,
            summary_path=summary_path,
            init="random" if self.config.random_init else "warmstart",
            init_checkpoint=self.init_checkpoint,
            best_dev_score=self.best_dev_score,
            anchor_dev_score=self.anchor_dev_score,
        )
        (self.manager.run_dir / "run_summary.json").write_text(
            json.dumps(run_summary.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return run_summary
