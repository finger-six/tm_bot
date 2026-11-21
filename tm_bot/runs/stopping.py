"""Stopping rules and checkpoint selection for self-play runs (#10, v2 rules from #40 section 1)."""

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
    COLLAPSE_ENTROPY = "collapse_entropy"
    PLATEAU = "plateau"
    REGRESSION_ANCHOR = "regression_anchor"
    MARGIN_COLLAPSE = "margin_collapse"
    GAME_SHAPE = "game_shape"
    CALIBRATION = "calibration"
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


# Selection thresholds (#40 section 1).
ANCHOR_WIN_RATE_FLOOR = 0.45  # G2: no regression against the warm-start anchor
HEAD_TO_HEAD_FLOOR = 0.50  # R1 point estimate / R2 CI lower bound
DELTA_D_NON_INFERIORITY = -0.03  # R2: dD CI lower bound, about -10 VP at the floor

ROUTE_DELTA_D = "delta_d"
ROUTE_HEAD_TO_HEAD = "head_to_head"
ROUTE_NO_INCUMBENT = "no_incumbent"


def new_best_route(
    *,
    failures: int,
    anchor_win_rate: float | None,
    head_to_head: float | None,
    head_to_head_ci_95: Sequence[float] | None,
    delta_d_ci_95: Sequence[float] | None,
) -> str | None:
    """Floor-aware best-checkpoint rule on `dev-*` results; returns the accepting route.

    Gates: G1 zero failures across all pairings; G2 win rate `a` vs the
    warm-start anchor >= 0.45 (skipped only when no anchor is configured).
    Routes: R1 (`delta_d`) the paired dD 95% CI lower bound is above 0 and the
    head-to-head rate `h` vs the incumbent is >= 0.50; R2 (`head_to_head`) the
    `h` 95% CI lower bound is above 0.50 and the dD CI lower bound is >= -0.03.
    With no incumbent at all (no anchor, no earlier best), the gates alone decide.
    """
    if failures > 0:
        return None
    if anchor_win_rate is not None and anchor_win_rate < ANCHOR_WIN_RATE_FLOOR:
        return None
    if head_to_head is None and head_to_head_ci_95 is None and delta_d_ci_95 is None:
        return ROUTE_NO_INCUMBENT
    if delta_d_ci_95 is None:
        return None
    d_low = float(delta_d_ci_95[0])
    if d_low > 0.0 and head_to_head is not None and head_to_head >= HEAD_TO_HEAD_FLOOR:
        return ROUTE_DELTA_D
    if (
        head_to_head_ci_95 is not None
        and float(head_to_head_ci_95[0]) > HEAD_TO_HEAD_FLOOR
        and d_low >= DELTA_D_NON_INFERIORITY
    ):
        return ROUTE_HEAD_TO_HEAD
    return None


def is_new_best(**evidence: Any) -> bool:
    """True when `new_best_route` accepts the candidate by either route."""
    return new_best_route(**evidence) is not None


@dataclass(frozen=True, slots=True)
class FailureEvent:
    """A failed game or rejected decision that hard-stops the run."""

    seed_key: str
    map_name: str
    error_type: str
    error: str
    decisions: int = 0


class StoppingMonitor:
    """Incremental evaluation of the stopping rules (#10, v2 rules from #40 section 1).

    The monitor is fed metric events as they arrive (cycle losses, failed games,
    self-play game shape, dev evaluations) and returns a `StopDecision`. It is
    intentionally deterministic and free of clocks: callers supply training-clock
    seconds.
    """

    def __init__(
        self,
        *,
        entropy_collapse_fraction: float = 0.10,
        entropy_collapse_window_s: float = 2.0 * 3600.0,
        plateau_s: float = 6.0 * 3600.0,
        regression_anchor_win_rate: float = 0.35,
        margin_collapse_drop: float = 0.05,
        calibration_ece: float = 0.15,
        consecutive_evaluations: int = 3,
        calibration_stops: bool = True,
        game_shape_median_generations: float = 22.0,
        game_shape_window_s: float = 2.0 * 3600.0,
        step_cap: int | None = None,
    ) -> None:
        if not (0.0 < entropy_collapse_fraction < 1.0):
            raise ValueError("entropy_collapse_fraction must be in (0, 1)")
        if entropy_collapse_window_s <= 0.0:
            raise ValueError("entropy_collapse_window_s must be > 0")
        if plateau_s <= 0.0:
            raise ValueError("plateau_s must be > 0")
        if not (0.0 <= regression_anchor_win_rate <= 1.0):
            raise ValueError("regression_anchor_win_rate must be in [0, 1]")
        if margin_collapse_drop <= 0.0:
            raise ValueError("margin_collapse_drop must be > 0")
        if calibration_ece <= 0.0:
            raise ValueError("calibration_ece must be > 0")
        if consecutive_evaluations < 1:
            raise ValueError("consecutive_evaluations must be >= 1")
        if game_shape_median_generations <= 0.0 or game_shape_window_s <= 0.0:
            raise ValueError("game-shape threshold and window must be > 0")

        self.entropy_collapse_fraction = float(entropy_collapse_fraction)
        self.entropy_collapse_window_s = float(entropy_collapse_window_s)
        self.plateau_s = float(plateau_s)
        self.regression_anchor_win_rate = float(regression_anchor_win_rate)
        self.margin_collapse_drop = float(margin_collapse_drop)
        self.calibration_ece = float(calibration_ece)
        self.consecutive_evaluations = int(consecutive_evaluations)
        self.calibration_stops = bool(calibration_stops)
        self.game_shape_median_generations = float(game_shape_median_generations)
        self.game_shape_window_s = float(game_shape_window_s)
        self.step_cap = step_cap

        # Training clock of the last new best (run start until the first one).
        self.last_best_training_s = 0.0
        self.regression_streak = 0
        self.margin_streak = 0
        self.calibration_streak = 0
        # Training clock at which self-play game shape was last within bounds.
        self.shape_ok_training_s = 0.0
        self.entropy_baseline: float | None = None
        self.entropy_checked_at_s: float | None = None

    _PERSISTED = (
        "last_best_training_s",
        "regression_streak",
        "margin_streak",
        "calibration_streak",
        "shape_ok_training_s",
    )

    def state_dict(self) -> dict[str, Any]:
        """Rule state that must survive a resume."""
        return {name: getattr(self, name) for name in self._PERSISTED}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for name in self._PERSISTED:
            if state.get(name) is not None:
                kind = type(getattr(self, name))
                setattr(self, name, kind(state[name]))

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

    # --- self-play game shape ----------------------------------------------
    def observe_game_shape(
        self, median_generations: float | None, *, elapsed_s: float
    ) -> StopDecision:
        """Stop when the self-play median stays above 22 generations for 2 h of training clock."""
        if median_generations is None:
            return _not_stopped()
        if median_generations <= self.game_shape_median_generations:
            self.shape_ok_training_s = float(elapsed_s)
            return _not_stopped()
        over_s = float(elapsed_s) - self.shape_ok_training_s
        if over_s >= self.game_shape_window_s:
            return _stop(
                StopKind.GAME_SHAPE,
                f"self-play median {median_generations:.1f} generations > "
                f"{self.game_shape_median_generations:.0f} for {over_s / 3600.0:.2f} h "
                "of training clock (rule bug or stall: investigate)",
                median_generations=float(median_generations),
                over_s=over_s,
                elapsed_s=float(elapsed_s),
            )
        return _not_stopped()

    # --- dev evaluations: plateau, regression, margin, calibration --------
    def _streak(self, name: str, hit: bool) -> int:
        value = getattr(self, name) + 1 if hit else 0
        setattr(self, name, value)
        return value

    def observe_checkpoint(self, metrics: CheckpointMetrics) -> StopDecision:
        """Apply the v2 dev-evaluation stop rules to one full dev evaluation."""
        training_s = float(
            metrics.training_s if metrics.training_s is not None else metrics.elapsed_s
        )
        base = {"cycle": metrics.cycle, "elapsed_s": metrics.elapsed_s, "training_s": training_s}
        a = metrics.dev_win_rate_vs_anchor
        d = metrics.dev_score_vs_b1
        best_d = metrics.best_dev_score_vs_b1
        ece = metrics.dev_ece_vs_b1
        if metrics.new_best:
            self.last_best_training_s = training_s
        regression = self._streak(
            "regression_streak", a is not None and a < self.regression_anchor_win_rate
        )
        margin = self._streak(
            "margin_streak",
            not metrics.new_best
            and d is not None
            and best_d is not None
            and d < best_d - self.margin_collapse_drop,
        )
        calibration = self._streak(
            "calibration_streak", ece is not None and ece > self.calibration_ece
        )
        n = self.consecutive_evaluations
        if regression >= n:
            return _stop(
                StopKind.REGRESSION_ANCHOR,
                f"win rate vs the warm-start anchor {a:.4f} < "
                f"{self.regression_anchor_win_rate:.2f} for {regression} evaluations",
                anchor_win_rate=float(a),  # type: ignore[arg-type]
                streak=regression,
                **base,
            )
        if margin >= n:
            return _stop(
                StopKind.MARGIN_COLLAPSE,
                f"dev score D {d:.4f} more than {self.margin_collapse_drop:.2f} below the "
                f"best's {best_d:.4f} for {margin} evaluations",
                dev_score=float(d),  # type: ignore[arg-type]
                best_dev_score=float(best_d),  # type: ignore[arg-type]
                streak=margin,
                **base,
            )
        if calibration >= n and self.calibration_stops:
            return _stop(
                StopKind.CALIBRATION,
                f"dev ECE vs b1-v2 {ece:.4f} > {self.calibration_ece:.2f} for "
                f"{calibration} evaluations (investigate)",
                ece=float(ece),  # type: ignore[arg-type]
                streak=calibration,
                **base,
            )
        since_best = training_s - self.last_best_training_s
        if not metrics.new_best and since_best >= self.plateau_s:
            return _stop(
                StopKind.PLATEAU,
                f"no new best for {since_best / 3600.0:.2f} h of training clock",
                since_best_s=since_best,
                last_best_training_s=self.last_best_training_s,
                **base,
            )
        return _not_stopped()

    # --- budget / step cap ---------------------------------------------
    def check_budget(self, elapsed_s: float, budget_s: float) -> StopDecision:
        if elapsed_s >= budget_s:
            return _stop(
                StopKind.BUDGET,
                f"training-time budget exhausted ({elapsed_s:.1f}s >= {budget_s:.1f}s)",
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
