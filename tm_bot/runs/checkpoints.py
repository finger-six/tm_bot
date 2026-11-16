"""Run directories, checkpoint retention (best + last 10), disk cap, and resume state."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence

import torch

from tm_bot.models.network import TMNetwork, load_checkpoint, save_checkpoint
from tm_bot.models.vocab import CardVocab
from tm_bot.runs.buffer import ReplayBuffer
from tm_bot.runs.config import RunConfig
from tm_bot.runs.errors import RunResourceError, RunResumeError

_SAVE_DIR_RE = re.compile(r"^(?:step|ckpt)-(\d{9,})$")
CONFIG_NAME = "config.json"
STATE_NAME = "state.json"
STATE_TENSORS_NAME = "state.pt"


def _directory_size_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return total


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class CheckpointManager:
    """Owns `runs/<run>/`: checkpoints, best copy, run state, and regression cases."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        keep_last: int = 10,
        disk_limit_gb: float = 50.0,
    ) -> None:
        if keep_last < 1:
            raise ValueError("keep_last must be >= 1")
        self.run_dir = Path(run_dir)
        self.checkpoint_root = self.run_dir / "checkpoints"
        self.best_dir = self.run_dir / "best"
        self.monitor_dir = self.run_dir / "monitor"
        self.regression_dir = self.run_dir / "regression"
        self.keep_last = int(keep_last)
        self.disk_limit_bytes = float(disk_limit_gb) * (1024.0**3)
        self._save_counter = 0
        for path in (
            self.run_dir,
            self.checkpoint_root,
            self.monitor_dir,
            self.regression_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        existing = self._list_save_counters()
        self._save_counter = (max(existing) + 1) if existing else 1

    # --- paths ---------------------------------------------------------
    @property
    def config_path(self) -> Path:
        return self.run_dir / CONFIG_NAME

    @property
    def state_path(self) -> Path:
        return self.run_dir / STATE_NAME

    @property
    def state_tensors_path(self) -> Path:
        return self.run_dir / STATE_TENSORS_NAME

    @property
    def metrics_path(self) -> Path:
        return self.monitor_dir / "metrics.jsonl"

    @property
    def summary_path(self) -> Path:
        return self.monitor_dir / "summary.json"

    @property
    def init_dir(self) -> Path:
        return self.run_dir / "init"

    def initialize(self, config: RunConfig) -> None:
        """Write the run config if the directory is new; verify it on resume."""
        if self.config_path.exists():
            existing = RunConfig.load(self.config_path)
            if existing.to_dict() != config.to_dict():
                raise RunResumeError(
                    f"Run config in {self.config_path} does not match the requested config"
                )
        else:
            self.config_path.write_text(config.to_json(), encoding="utf-8")

    # --- checkpoints ---------------------------------------------------
    def checkpoint_dir(self, counter: int) -> Path:
        return self.checkpoint_root / f"ckpt-{int(counter):09d}"

    def _list_save_counters(self) -> list[int]:
        if not self.checkpoint_root.exists():
            return []
        found: list[int] = []
        for entry in self.checkpoint_root.iterdir():
            match = _SAVE_DIR_RE.match(entry.name)
            if match and entry.is_dir():
                found.append(int(match.group(1)))
        return found

    def list_checkpoints(self) -> list[Path]:
        """Numbered checkpoints in monotonic save order (legacy `step-` dirs included)."""
        if not self.checkpoint_root.exists():
            return []
        found: list[tuple[int, Path]] = []
        for entry in self.checkpoint_root.iterdir():
            match = _SAVE_DIR_RE.match(entry.name)
            if match and entry.is_dir():
                found.append((int(match.group(1)), entry))
        return [path for _counter, path in sorted(found)]

    def save(
        self,
        *,
        step: int,
        model: TMNetwork,
        vocab: CardVocab,
        metadata: Mapping[str, Any],
        is_best: bool = False,
    ) -> Path:
        """Save a checkpoint named by a monotonic save counter (never by step)."""
        target = self.checkpoint_dir(self._save_counter)
        self._save_counter += 1
        save_checkpoint(target, model, vocab, metadata=dict(metadata))
        if is_best:
            save_checkpoint(self.best_dir, model, vocab, metadata=dict(metadata))
        self.prune()
        return target

    def save_best(
        self,
        *,
        model: TMNetwork,
        vocab: CardVocab,
        metadata: Mapping[str, Any],
    ) -> Path:
        """Refresh the `best/` copy without writing another numbered checkpoint."""
        save_checkpoint(self.best_dir, model, vocab, metadata=dict(metadata))
        return self.best_dir

    def prune(self) -> list[Path]:
        """Keep the best copy and the last `keep_last` numbered checkpoints."""
        checkpoints = self.list_checkpoints()
        removed: list[Path] = []
        for path in checkpoints[: max(0, len(checkpoints) - self.keep_last)]:
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path)
        removed.extend(self.enforce_disk_cap())
        return removed

    def disk_usage_bytes(self) -> int:
        return _directory_size_bytes(self.run_dir)

    def enforce_disk_cap(self) -> list[Path]:
        """Delete the oldest non-best checkpoints until the run fits its disk cap."""
        removed: list[Path] = []
        checkpoints = self.list_checkpoints()
        while self.disk_usage_bytes() > self.disk_limit_bytes and checkpoints:
            oldest = checkpoints.pop(0)
            shutil.rmtree(oldest, ignore_errors=True)
            removed.append(oldest)
        if self.disk_usage_bytes() > self.disk_limit_bytes:
            raise RunResourceError(
                f"run directory {self.run_dir} exceeds disk cap "
                f"{self.disk_limit_bytes / 1024**3:.2f} GB even after pruning"
            )
        return removed

    def best_checkpoint(self) -> Path | None:
        path = self.best_dir / "checkpoint.pt"
        return self.best_dir if path.exists() else None

    def save_init(
        self,
        *,
        model: TMNetwork,
        vocab: CardVocab,
        metadata: Mapping[str, Any],
    ) -> Path:
        """Save the run's initial weights outside the pruned checkpoint pool.

        The random-init run (#27) uses this checkpoint as the first
        previous-best opponent. It lives in `<run>/init/`, which
        `prune`/`enforce_disk_cap` (which only scan `checkpoints/`) never remove.
        """
        save_checkpoint(self.init_dir, model, vocab, metadata=dict(metadata))
        return self.init_dir

    # --- run state / resume -------------------------------------------
    def save_state(
        self,
        *,
        payload: Mapping[str, Any],
        model: TMNetwork | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        buffer: ReplayBuffer | None = None,
        entropy_probe: Any | None = None,
    ) -> None:
        data = dict(payload)
        _write_json_atomic(self.state_path, data)
        tensors: dict[str, Any] = {}
        if model is not None:
            tensors["model_state_dict"] = model.state_dict()
        if optimizer is not None:
            tensors["optimizer_state_dict"] = optimizer.state_dict()
        if buffer is not None:
            tensors["buffer"] = buffer.state_dict()
        if entropy_probe is not None:
            tensors["entropy_probe"] = entropy_probe
        if tensors:
            torch.save(tensors, self.state_tensors_path)

    def load_state(self) -> dict[str, Any] | None:
        if not self.state_path.exists():
            return None
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def resolve_resume_checkpoint(self, state: Mapping[str, Any] | None) -> Path | None:
        """Return the last good checkpoint: recorded last, else newest numbered, else best."""
        if state is not None:
            recorded = state.get("last_checkpoint")
            if recorded:
                candidate = Path(recorded)
                if not candidate.is_absolute():
                    candidate = self.run_dir / candidate
                if (candidate / "checkpoint.pt").exists():
                    return candidate
        checkpoints = self.list_checkpoints()
        if checkpoints:
            return checkpoints[-1]
        return self.best_checkpoint()

    def restore_model(self, model: TMNetwork, checkpoint: Path) -> None:
        from tm_bot.comparator.model import (
            comparator_checkpoint_architecture,
            load_comparator_checkpoint,
        )

        if comparator_checkpoint_architecture(checkpoint).startswith("comparator_"):
            loaded, _vocab, _meta = load_comparator_checkpoint(
                checkpoint, map_location="cpu"
            )
            model.load_state_dict(loaded.state_dict())
            model.temperature = float(getattr(loaded, "temperature", 1.0))
        else:
            loaded, _vocab, _meta = load_checkpoint(checkpoint, map_location="cpu")
            model.load_state_dict(loaded.state_dict())
        model.eval()

    def load_training_state(
        self,
        *,
        model: TMNetwork | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        buffer: ReplayBuffer | None = None,
    ) -> dict[str, Any] | None:
        """Restore any of model/optimizer/buffer independently; return the raw payload."""
        if not self.state_tensors_path.exists():
            return None
        payload = torch.load(self.state_tensors_path, map_location="cpu", weights_only=False)
        if model is not None and "model_state_dict" in payload:
            model.load_state_dict(payload["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if buffer is not None and "buffer" in payload:
            buffer.load_state_dict(payload["buffer"])
        return payload

    # --- regression cases ---------------------------------------------
    def write_regression_case(self, payload: Mapping[str, Any]) -> Path:
        """Persist a failed game's seed, map, and action history for replay."""
        seed = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(payload.get("seed_key", "unknown")))
        map_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(payload.get("map_name", "unknown")))
        error = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(payload.get("error_type", "error")))
        path = self.regression_dir / f"{seed}__{map_name}__{error}.json"
        _write_json_atomic(path, payload)
        return path
