"""Self-play run orchestration: cycles, checkpoints, dev selection, stopping, resume."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Protocol

import numpy as np
import torch

from tm_bot.models.network import TMNetwork
from tm_bot.models.vocab import CardVocab
from tm_bot.runs.buffer import ReplayBuffer
from tm_bot.runs.checkpoints import CheckpointManager
from tm_bot.runs.config import RunConfig
from tm_bot.runs.deveval import (
    DevResult,
    b1_agent_factory,
    build_checkpoint_metrics,
    dev_agent_factory,
    evaluate_dev_pairing,
)
from tm_bot.runs.errors import RunResumeError
from tm_bot.runs.learner import AlphaZeroLearner, LearnerStats
from tm_bot.runs.monitoring import (
    CycleMetrics,
    MonitorLogger,
    game_shape_flags,
    latency_summary,
)
from tm_bot.runs.resources import check_learner_memory
from tm_bot.runs.selfplay import (
    SelfPlayBatch,
    SelfPlayEngine,
    determinism_check,
)
from tm_bot.runs.stopping import FailureEvent, StopDecision, StopKind, StoppingMonitor


class SelfPlaySource(Protocol):
    """Anything that can produce a batch of self-play games + training rows."""

    def play_batch(self, num_games: int) -> SelfPlayBatch: ...


class Learner(Protocol):
    step: int

    def train(self, buffer: ReplayBuffer, *, updates: int | None = None) -> LearnerStats: ...


DevEvaluator = Callable[[Path, Path | None], DevResult]

_HARD_STOPS = frozenset(
    {
        StopKind.ENGINE_ERROR,
        StopKind.ADAPTER_REJECTION,
        StopKind.STEP_CAP,
        StopKind.NONFINITE_LOSS,
        StopKind.RESOURCE_LIMIT,
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
    best_dev_win_rate: float | None
    best_checkpoint: Path | None
    last_checkpoint: Path | None
    stop: StopDecision
    summary_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "selfplay-run-v1",
            "run_name": self.run_name,
            "run_dir": str(self.run_dir),
            "cycles": self.cycles,
            "selfplay_games": self.selfplay_games,
            "steps": self.steps,
            "elapsed_s": self.elapsed_s,
            "best_dev_win_rate": self.best_dev_win_rate,
            "best_checkpoint": str(self.best_checkpoint) if self.best_checkpoint else None,
            "last_checkpoint": str(self.last_checkpoint) if self.last_checkpoint else None,
            "stop": self.stop.to_dict(),
        }


def _rel(path: Path, base: Path) -> str:
    try:
        return os.path.relpath(path, base)
    except ValueError:  # pragma: no cover - different drives on Windows
        return str(path)


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
            collapse_win_rate_drop=config.collapse_win_rate_drop,
            collapse_checkpoints=config.collapse_checkpoints,
            entropy_collapse_fraction=config.entropy_collapse_fraction,
            entropy_collapse_window_s=config.entropy_collapse_window_s,
            plateau_checkpoints=config.plateau_checkpoints,
            step_cap=config.step_cap,
        )
        self.buffer = ReplayBuffer(config.buffer_capacity)
        self.optimizer: torch.optim.Optimizer | None = None

        self.cycle = 0
        self.selfplay_games = 0
        self.steps = 0
        self.best_dev_win_rate: float | None = None
        self.last_checkpoint: Path | None = None
        self.last_checkpoint_elapsed_s = 0.0
        self._elapsed_base = 0.0
        self._t_start = self.clock()
        self._determinism_checks_done = 0
        self._stop = StopDecision(stopped=False)

    # --- construction --------------------------------------------------
    @classmethod
    def create(cls, config: RunConfig) -> SelfPlayRunner:
        """Build the real engine, learner, model, and dev evaluator for `config`."""
        device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if device.startswith("cuda") and config.limits.vram_gb > 0:
            try:
                total = torch.cuda.get_device_properties(0).total_memory
                torch.cuda.set_per_process_memory_fraction(
                    min(1.0, (config.limits.vram_gb * 1024**3) / float(total))
                )
            except Exception:
                pass

        checkpoint = config.warmstart_checkpoint
        if checkpoint is not None:
            from tm_bot.models.agent import PolicyValueAgent
            from tm_bot.models.network import load_checkpoint

            model, vocab, _meta = load_checkpoint(checkpoint, map_location=device)
        else:
            from tm_bot.search.agent import DEFAULT_WARMSTART_CHECKPOINT
            from tm_bot.models.network import load_checkpoint

            model, vocab, _meta = load_checkpoint(
                DEFAULT_WARMSTART_CHECKPOINT, map_location=device
            )
        model.eval()
        learner = AlphaZeroLearner(config, model, device=device)
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

        def evaluator(candidate: Path, opponent: Path | None) -> DevResult:
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
                num_games=config.dev_games_per_opponent,
                maps=config.maps,
                seed_start=config.dev_seed_start,
                num_workers=config.workers,
                batch_size=config.batch_size,
                agent_a_name="candidate",
                agent_b_name="b1" if opponent is None else "previous_best",
            )

        runner.evaluator = evaluator
        return runner

    # --- time ----------------------------------------------------------
    def elapsed_s(self) -> float:
        return self._elapsed_base + (self.clock() - self._t_start)

    # --- resume --------------------------------------------------------
    def _restore(self) -> bool:
        state = self.manager.load_state()
        if state is None:
            return False
        checkpoint = self.manager.resolve_resume_checkpoint(state)
        if checkpoint is None:
            raise RunResumeError(
                f"Run {self.config.run_name} has state but no loadable checkpoint"
            )
        if self.model is not None:
            self.manager.restore_model(self.model, checkpoint)
            if self.optimizer is not None:
                self.manager.load_training_state(
                    model=self.model, optimizer=self.optimizer, buffer=self.buffer
                )
        self.cycle = int(state.get("cycle", 0))
        self.steps = int(state.get("step", 0))
        self.selfplay_games = int(state.get("selfplay_games", 0))
        best = state.get("best_dev_win_rate")
        self.best_dev_win_rate = None if best is None else float(best)
        self._elapsed_base = float(state.get("elapsed_s", 0.0))
        self.last_checkpoint_elapsed_s = float(state.get("last_checkpoint_elapsed_s", 0.0))
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
                float(entropy_baseline), elapsed_s=self._elapsed_base
            )
        stopping = state.get("stopping") or {}
        if stopping.get("best_dev_win_rate") is not None:
            self.stopping.best_dev_win_rate = float(stopping["best_dev_win_rate"])
        self.stopping.below_best_streak = int(stopping.get("below_best_streak", 0))
        self.stopping.checkpoints_without_improvement = int(
            stopping.get("checkpoints_without_improvement", 0)
        )
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
            "last_checkpoint_elapsed_s": self.last_checkpoint_elapsed_s,
            "last_checkpoint": (
                _rel(self.last_checkpoint, self.manager.run_dir)
                if self.last_checkpoint is not None
                else None
            ),
            "pool": [_rel(Path(p), self.manager.run_dir) for p in pool],
            "seed_cursor": getattr(self.engine, "seed_cursor", None),
            "entropy_baseline": self.stopping.entropy_baseline,
            "stopping": {
                "best_dev_win_rate": self.stopping.best_dev_win_rate,
                "below_best_streak": self.stopping.below_best_streak,
                "checkpoints_without_improvement": self.stopping.checkpoints_without_improvement,
            },
            "stop": self._stop.to_dict(),
        }

    def _save_state(self) -> None:
        self.manager.save_state(
            payload=self._state_payload(),
            model=self.model,
            optimizer=self.optimizer,
            buffer=self.buffer if len(self.buffer) else None,
        )

    # --- one cycle -----------------------------------------------------
    def _log_cycle(
        self,
        batch: SelfPlayBatch,
        *,
        learner_stats: LearnerStats | None,
        elapsed: float,
        cycle_wall_s: float,
    ) -> CycleMetrics:
        games = batch.games
        completed = [g for g in games if g.ok]
        generations = [int(g.generations) for g in completed if g.generations is not None]
        margins: list[float] = []
        for g, seat in zip(games, batch.training_seats or [0] * len(games)):
            if not g.ok or g.scores is None:
                continue
            margins.append(float(g.scores[seat] - g.scores[1 - seat]))
        record = CycleMetrics(
            cycle=self.cycle,
            elapsed_s=float(elapsed),
            wall_clock_s=float(cycle_wall_s),
            selfplay_games=len(games),
            total_selfplay_games=self.selfplay_games,
            decisions=int(batch.decisions),
            searched_decisions=int(batch.searched_decisions),
            rejections=len(batch.failures),
            failures=len(batch.failures),
            mean_generations=(float(np.mean(generations)) if generations else None),
            min_generations=(int(min(generations)) if generations else None),
            max_generations=(int(max(generations)) if generations else None),
            mean_vp_margin=(float(np.mean(margins)) if margins else None),
            mean_decisions_per_game=(
                float(np.mean([g.decisions for g in completed])) if completed else None
            ),
            policy_loss=None if learner_stats is None else learner_stats.policy_loss,
            value_loss=None if learner_stats is None else learner_stats.value_loss,
            total_loss=None if learner_stats is None else learner_stats.total_loss,
            policy_entropy=(
                learner_stats.policy_entropy
                if learner_stats is not None and learner_stats.policy_entropy is not None
                else batch.entropy
            ),
            value_brier=None if learner_stats is None else learner_stats.value_brier,
            value_accuracy=None if learner_stats is None else learner_stats.value_accuracy,
            decision_latency_ms=latency_summary(batch.latencies_ms),
            shape_flags=game_shape_flags(
                games,
                min_generations=self.config.shape_min_generations,
                max_generations=self.config.shape_max_generations,
            ),
            throughput_games_per_hour=(
                self.selfplay_games / (elapsed / 3600.0) if elapsed > 0.0 else None
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

    def _maybe_determinism_check(self, batch: SelfPlayBatch) -> None:
        if self.determinism_every <= 0:
            return
        due = self.selfplay_games // self.determinism_every
        if due <= self._determinism_checks_done:
            return
        self._determinism_checks_done = due
        index = None
        for i, game in enumerate(batch.games):
            if game.ok and i < len(batch.recorders):
                index = i
        if index is None:
            return
        recorder = batch.recorders[index]
        result = determinism_check(batch.games[index], recorder.actions)
        self.logger.log_determinism(result)

    def _checkpoint_and_evaluate(self, elapsed: float) -> None:
        if self.model is None or self.vocab is None:
            return
        self.last_checkpoint = self.manager.save(
            step=self.steps,
            model=self.model,
            vocab=self.vocab,
            metadata={
                "version": f"selfplay-{self.config.run_name}",
                "cycle": self.cycle,
                "step": self.steps,
                "selfplay_games": self.selfplay_games,
                "num_simulations": self.config.num_simulations,
                "num_worlds": self.config.num_worlds,
            },
        )
        self.last_checkpoint_elapsed_s = elapsed
        previous_best = self.manager.best_checkpoint()
        if previous_best is None and self.config.warmstart_checkpoint:
            previous_best = Path(self.config.warmstart_checkpoint)

        if self.evaluator is not None:
            vs_b1 = self.evaluator(self.last_checkpoint, None)
            vs_best = (
                self.evaluator(self.last_checkpoint, previous_best)
                if previous_best is not None
                else vs_b1
            )
            win_rate = vs_b1.win_rate
            new_best = win_rate is not None and (
                self.best_dev_win_rate is None or win_rate > self.best_dev_win_rate
            )
            if new_best:
                self.manager.save_best(
                    model=self.model,
                    vocab=self.vocab,
                    metadata={
                        "version": f"selfplay-{self.config.run_name}",
                        "cycle": self.cycle,
                        "step": self.steps,
                        "selfplay_games": self.selfplay_games,
                        "dev_win_rate_vs_b1": win_rate,
                    },
                )
                self.best_dev_win_rate = win_rate
            record = build_checkpoint_metrics(
                cycle=self.cycle,
                elapsed_s=elapsed,
                step=self.steps,
                games=self.selfplay_games,
                checkpoint=self.last_checkpoint,
                vs_b1=vs_b1,
                vs_previous_best=vs_best,
                new_best=bool(new_best),
                best_dev_win_rate=self.best_dev_win_rate,
            )
            self.logger.log_checkpoint(record)
            decision = self.stopping.observe_checkpoint(record)
            if decision.stopped and not self._stop.stopped:
                self._stop = decision

        setter = getattr(self.engine, "set_opponent_pool", None)
        if callable(setter):
            pool = self.manager.list_checkpoints()[
                -self.config.opponent_pool_size :
            ]
            setter(pool)
        self._save_state()

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

        if self.model is not None and self.vocab is not None:
            setter = getattr(self.engine, "set_opponent_pool", None)
            if callable(setter):
                setter(self.manager.list_checkpoints()[-self.config.opponent_pool_size :])

        while True:
            elapsed = self.elapsed_s()
            budget = self.stopping.check_budget(elapsed, self.config.wall_clock_budget_s)
            if budget.stopped:
                self._stop = budget
                break
            if max_cycles is not None and self.cycle >= max_cycles:
                break

            t_cycle = self.clock()
            batch = self.engine.play_batch(self.config.games_per_cycle)
            self.cycle += 1
            self.selfplay_games += len(batch.games)
            elapsed = self.elapsed_s()

            self._handle_failures(batch, elapsed)
            if self._stop.stopped:
                self._log_cycle(
                    batch,
                    learner_stats=None,
                    elapsed=elapsed,
                    cycle_wall_s=self.clock() - t_cycle,
                )
                break

            # The warm-start policy entropy baseline is measured on the first
            # cycle's self-play targets, before any update from this run.
            if self.stopping.entropy_baseline is None and batch.entropy is not None:
                self.stopping.set_entropy_baseline(batch.entropy, elapsed_s=elapsed)

            if batch.samples:
                self.buffer.extend(batch.samples)
            learner_stats: LearnerStats | None = None
            if self.learner is not None and len(self.buffer) > 0:
                learner_stats = self.learner.train(self.buffer)
                self.steps = int(getattr(self.learner, "step", self.steps))
                if learner_stats.nonfinite_loss is not None:
                    self._rollback()
                    self._stop = StopDecision(
                        stopped=True,
                        kind=StopKind.NONFINITE_LOSS,
                        reason=f"non-finite loss {learner_stats.nonfinite_loss!r}; rolled back",
                        details={
                            "loss": learner_stats.nonfinite_loss,
                            "elapsed_s": elapsed,
                            "rolled_back_to": (
                                str(self.last_checkpoint) if self.last_checkpoint else None
                            ),
                        },
                    )

            self._log_cycle(
                batch,
                learner_stats=learner_stats,
                elapsed=elapsed,
                cycle_wall_s=self.clock() - t_cycle,
            )
            self._maybe_determinism_check(batch)

            if not self._stop.stopped:
                try:
                    check_learner_memory(self.config.limits)
                except Exception as exc:
                    self._stop = StopDecision(
                        stopped=True,
                        kind=StopKind.RESOURCE_LIMIT,
                        reason=str(exc),
                        details={"elapsed_s": elapsed},
                    )

            entropy = learner_stats.policy_entropy if learner_stats else batch.entropy
            if entropy is not None:
                decision = self.stopping.observe_entropy(entropy, elapsed_s=elapsed)
                if decision.stopped and not self._stop.stopped:
                    self._stop = decision

            step_decision = self.stopping.check_steps(self.steps, elapsed_s=elapsed)
            if step_decision.stopped and not self._stop.stopped:
                self._stop = step_decision

            due = (
                self.model is not None
                and self.evaluator is not None
                and (
                    (
                        self._stop.stopped
                        and self._stop.kind not in _HARD_STOPS
                    )
                    or elapsed - self.last_checkpoint_elapsed_s
                    >= self.config.checkpoint_interval_s
                    or (max_cycles is not None and self.cycle >= max_cycles)
                )
            )
            if due:
                self._checkpoint_and_evaluate(elapsed)

            if self._stop.stopped:
                break

        return self._finalize()

    def _rollback(self) -> None:
        """Restore the last periodic checkpoint after a non-finite loss."""
        if self.model is None:
            return
        checkpoint = self.last_checkpoint or self.manager.resolve_resume_checkpoint(None)
        if checkpoint is None:
            return
        self.manager.restore_model(self.model, checkpoint)
        if self.optimizer is not None and self.manager.state_tensors_path.exists():
            try:
                payload = torch.load(
                    self.manager.state_tensors_path,
                    map_location="cpu",
                    weights_only=False,
                )
                if "optimizer_state_dict" in payload:
                    self.optimizer.load_state_dict(payload["optimizer_state_dict"])
            except Exception:
                pass
        self._save_state()

    def _finalize(self) -> RunSummary:
        elapsed = self.elapsed_s()
        hard_stop = self._stop.stopped and self._stop.kind in _HARD_STOPS
        if self.model is not None and self.vocab is not None and hard_stop:
            # A broken run must not spend more compute on dev games; save the
            # rolled-back state so the run can be resumed after the bug is fixed.
            self.last_checkpoint = self.manager.save(
                step=self.steps,
                model=self.model,
                vocab=self.vocab,
                metadata={
                    "cycle": self.cycle,
                    "step": self.steps,
                    "stop": self._stop.to_dict(),
                },
            )
        elif self.model is not None and self.vocab is not None and self.evaluator is not None:
            if elapsed - self.last_checkpoint_elapsed_s > 60.0 or self.last_checkpoint is None:
                self._checkpoint_and_evaluate(elapsed)
        elif self.model is not None and self.vocab is not None:
            self.last_checkpoint = self.manager.save(
                step=self.steps,
                model=self.model,
                vocab=self.vocab,
                metadata={"cycle": self.cycle, "step": self.steps},
            )
        self._save_state()
        summary_path = self.logger.write_summary(self.manager.summary_path)
        run_summary = RunSummary(
            run_name=self.config.run_name,
            run_dir=self.manager.run_dir,
            cycles=self.cycle,
            selfplay_games=self.selfplay_games,
            steps=self.steps,
            elapsed_s=elapsed,
            best_dev_win_rate=self.best_dev_win_rate,
            best_checkpoint=self.manager.best_checkpoint(),
            last_checkpoint=self.last_checkpoint,
            stop=self._stop,
            summary_path=summary_path,
        )
        (self.manager.run_dir / "run_summary.json").write_text(
            json.dumps(run_summary.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return run_summary
