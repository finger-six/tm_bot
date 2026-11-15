"""Stopping rules for self-play runs (#10): hard errors, NaN loss, collapse, plateau, budget."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Sequence

from tm_bot.runs.monitoring import CheckpointMetrics


class StopKind(str, Enum):
    """Reason a run stopped."""

    BUDGET = "budget"
    ENGINE_ERROR = "engine_error"
    ADAPTER_REJECTION = "adapter_rejection"
    STEP_CAP = "step_cap"
    NONFINITE_LOSS = "nonfinite_loss"
    COLLAPSE_WIN_RATE = "collapse_win_rate"
    COLLAPSE_ENTROPY = "collapse_entropy"
    PLATEAU = "plateau"
    MAX_STEPS = "max_steps"
    RESOURCE_LIMIT = "resource_limit"


@dataclass(frozen=True, slots=True)
class StopDecision:
    """Outcome of evaluating every stopping rule."""

    stopped: bool
    kind: StopKind | None = None
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stopped": self.stopped,
            "kind": self.kind.value if self.kind else None,
            "reason": self.reason,
            "details": self.details,
        }


def _not_stopped() -> StopDecision:
    return StopDecision(stopped=False)


def _stop(kind: StopKind, reason: str, **details: Any) -> StopDecision:
    return StopDecision(stopped=True, kind=kind, reason=reason, details=details)


@dataclass(frozen=True, slots=True)
class FailureEvent:
    """A failed game or rejected decision that hard-stops the run."""

    seed_key: str
    map_name: str
    error_type: str
    error: str
    decisions: int = 0


class StoppingMonitor:
    """Incremental evaluation of the #10 stopping rules.

    The monitor is fed metric events as they arrive (cycle losses, failed games,
    periodic dev checkpoints) and returns a `StopDecision`. It is intentionally
    deterministic and free of clocks: callers supply elapsed seconds.
    """

    def __init__(
        self,
        *,
        collapse_win_rate_drop: float = 0.15,
        collapse_checkpoints: int = 3,
        entropy_collapse_fraction: float = 0.10,
        entropy_collapse_window_s: float = 2.0 * 3600.0,
        plateau_checkpoints: int = 12,
        step_cap: int | None = None,
    ) -> None:
        if collapse_win_rate_drop <= 0.0:
            raise ValueError("collapse_win_rate_drop must be > 0")
        if collapse_checkpoints < 1:
            raise ValueError("collapse_checkpoints must be >= 1")
        if not (0.0 < entropy_collapse_fraction < 1.0):
            raise ValueError("entropy_collapse_fraction must be in (0, 1)")
        if entropy_collapse_window_s <= 0.0:
            raise ValueError("entropy_collapse_window_s must be > 0")
        if plateau_checkpoints < 1:
            raise ValueError("plateau_checkpoints must be >= 1")

        self.collapse_win_rate_drop = float(collapse_win_rate_drop)
        self.collapse_checkpoints = int(collapse_checkpoints)
        self.entropy_collapse_fraction = float(entropy_collapse_fraction)
        self.entropy_collapse_window_s = float(entropy_collapse_window_s)
        self.plateau_checkpoints = int(plateau_checkpoints)
        self.step_cap = step_cap

        self.best_dev_win_rate: float | None = None
        self.below_best_streak = 0
        self.checkpoints_without_improvement = 0
        self.entropy_baseline: float | None = None
        self.entropy_checked_at_s: float | None = None

    def set_entropy_baseline(self, entropy: float, *, elapsed_s: float = 0.0) -> None:
        """Record the warm-start policy entropy used by the collapse rule."""
        if not math.isfinite(entropy) or entropy <= 0.0:
            return
        self.entropy_baseline = float(entropy)
        self.entropy_checked_at_s = float(elapsed_s)

    # --- hard failures -------------------------------------------------
    def observe_failures(
        self,
        failures: Sequence[FailureEvent],
        *,
        elapsed_s: float = 0.0,
    ) -> StopDecision:
        """Hard stop on adapter rejection, engine error, or step cap."""
        for failure in failures:
            if failure.error_type == "StepCapError":
                return _stop(
                    StopKind.STEP_CAP,
                    f"step cap reached for {failure.seed_key} on {failure.map_name}: {failure.error}",
                    seed_key=failure.seed_key,
                    map_name=failure.map_name,
                    error_type=failure.error_type,
                    error=failure.error,
                    decisions=failure.decisions,
                    elapsed_s=elapsed_s,
                )
            if failure.error_type == "AdapterBugError":
                return _stop(
                    StopKind.ADAPTER_REJECTION,
                    f"adapter rejection for {failure.seed_key} on {failure.map_name}: {failure.error}",
                    seed_key=failure.seed_key,
                    map_name=failure.map_name,
                    error_type=failure.error_type,
                    error=failure.error,
                    decisions=failure.decisions,
                    elapsed_s=elapsed_s,
                )
            # Any other failed game is treated as an engine/bridge failure: a
            # self-play game that cannot finish must not be silently skipped.
            return _stop(
                StopKind.ENGINE_ERROR,
                f"{failure.error_type} for {failure.seed_key} on {failure.map_name}: {failure.error}",
                seed_key=failure.seed_key,
                map_name=failure.map_name,
                error_type=failure.error_type,
                error=failure.error,
                decisions=failure.decisions,
                elapsed_s=elapsed_s,
            )
        return _not_stopped()

    # --- non-finite loss -----------------------------------------------
    def observe_loss(self, loss: float, *, elapsed_s: float = 0.0) -> StopDecision:
        """Hard stop on NaN/Inf training loss (the caller rolls back first)."""
        if not math.isfinite(float(loss)):
            return _stop(
                StopKind.NONFINITE_LOSS,
                f"non-finite loss {loss!r}",
                loss=float(loss),
                elapsed_s=elapsed_s,
            )
        return _not_stopped()

    # --- entropy collapse ----------------------------------------------
    def observe_entropy(self, entropy: float, *, elapsed_s: float) -> StopDecision:
        """Collapse when policy entropy falls below 10% of warm start within 2 h."""
        if self.entropy_baseline is None:
            self.set_entropy_baseline(entropy, elapsed_s=elapsed_s)
            return _not_stopped()
        self.entropy_checked_at_s = float(elapsed_s)
        if elapsed_s > self.entropy_collapse_window_s:
            return _not_stopped()
        threshold = self.entropy_collapse_fraction * self.entropy_baseline
        if math.isfinite(entropy) and entropy < threshold:
            return _stop(
                StopKind.COLLAPSE_ENTROPY,
                f"policy entropy {entropy:.6f} < {threshold:.6f} "
                f"(10% of warm-start {self.entropy_baseline:.6f}) at {elapsed_s:.1f}s",
                entropy=float(entropy),
                baseline=float(self.entropy_baseline),
                threshold=float(threshold),
                elapsed_s=float(elapsed_s),
            )
        return _not_stopped()

    # --- dev checkpoints: collapse + plateau ----------------------------
    def observe_checkpoint(
        self,
        metrics: CheckpointMetrics,
    ) -> StopDecision:
        """Apply the dev-win-rate collapse and plateau rules to a checkpoint."""
        win_rate = metrics.dev_win_rate_vs_b1
        if win_rate is None:
            return _not_stopped()

        new_best = self.best_dev_win_rate is None or win_rate > self.best_dev_win_rate
        if new_best:
            self.best_dev_win_rate = win_rate
            self.below_best_streak = 0
            self.checkpoints_without_improvement = 0
        else:
            self.checkpoints_without_improvement += 1
            drop = self.best_dev_win_rate - win_rate
            if drop > self.collapse_win_rate_drop:
                self.below_best_streak += 1
            else:
                self.below_best_streak = 0
            if self.below_best_streak >= self.collapse_checkpoints:
                return _stop(
                    StopKind.COLLAPSE_WIN_RATE,
                    f"dev win rate vs B1 {win_rate:.4f} more than "
                    f"{self.collapse_win_rate_drop:.2f} below best "
                    f"{self.best_dev_win_rate:.4f} for {self.below_best_streak} checkpoints",
                    win_rate=float(win_rate),
                    best=float(self.best_dev_win_rate),
                    streak=self.below_best_streak,
                    cycle=metrics.cycle,
                    elapsed_s=metrics.elapsed_s,
                )
            if self.checkpoints_without_improvement >= self.plateau_checkpoints:
                return _stop(
                    StopKind.PLATEAU,
                    f"no new best dev win rate for {self.checkpoints_without_improvement} checkpoints "
                    f"(best {self.best_dev_win_rate:.4f})",
                    best=float(self.best_dev_win_rate),
                    checkpoints_without_improvement=self.checkpoints_without_improvement,
                    cycle=metrics.cycle,
                    elapsed_s=metrics.elapsed_s,
                )
        return _not_stopped()

    # --- budget / step cap ---------------------------------------------
    def check_budget(self, elapsed_s: float, budget_s: float) -> StopDecision:
        if elapsed_s >= budget_s:
            return _stop(
                StopKind.BUDGET,
                f"wall-clock budget exhausted ({elapsed_s:.1f}s >= {budget_s:.1f}s)",
                elapsed_s=float(elapsed_s),
                budget_s=float(budget_s),
            )
        return _not_stopped()

    def check_steps(self, step: int, *, elapsed_s: float = 0.0) -> StopDecision:
        if self.step_cap is not None and step >= self.step_cap:
            return _stop(
                StopKind.MAX_STEPS,
                f"learner step cap reached ({step} >= {self.step_cap})",
                step=int(step),
                step_cap=int(self.step_cap),
                elapsed_s=float(elapsed_s),
            )
        return _not_stopped()
