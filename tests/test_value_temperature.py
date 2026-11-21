"""Main value temperature (#60): fitted on val gen >= 2 logits, persisted beside (or inside) the
checkpoint, applied to the reported P(win) only, and scored by gate check 14."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.test_runs import _VOCAB, _tiny_model
from tm_bot.comparator.calibration import calibration_metrics
from tm_bot.models.agent import PolicyValueAgent
from tm_bot.models.network import save_checkpoint
from tm_bot.models.value_temperature import (
    calibrated_win_prob,
    load_value_temperature,
    read_value_temperature_file,
    value_temperature_path,
)
from tm_bot.training import run_warmstart_gate as runner
from tm_bot.training import warmstart_gate as wg
from tm_bot.training.value_temperature import value_temperature_payload, write_value_temperature

TRUE_T = 1.5


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _synthetic_values(n: int = 40_000, seed: int = 0) -> dict[str, np.ndarray]:
    """`collect_value_outputs`-shaped rows whose gen >= 2 logits are TRUE_T x over-confident.

    Gen-1 rows are pure noise with extreme logits: a fit that used them would not recover TRUE_T.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.5, n)
    won = (rng.random(n) < _sigmoid(z)).astype(np.float64)
    gen = rng.integers(1, 14, n)
    logit = np.where(gen >= 2, TRUE_T * z, rng.normal(0.0, 8.0, n))
    return {
        "logit": logit, "won": won, "generation": gen,
        "margin": np.full(n, np.nan), "margin_pred": np.full(n, np.nan),
        "map_id": rng.integers(0, 4, n), "phase": np.where(gen >= 2, 1, -1),
    }


def test_fit_on_synthetic_logits_recovers_the_temperature_from_gen_2_plus_rows() -> None:
    values = _synthetic_values()
    payload = value_temperature_payload(values, checkpoint_sha="abc")
    mid = values["generation"] >= 2
    assert payload["temperature"] == pytest.approx(TRUE_T, rel=0.05)
    assert payload["val_calibration"]["n"] == int(mid.sum())
    assert payload["val_calibration"]["excluded_gen1"] == int((~mid).sum())
    raw, cal = payload["value_raw"]["midgame"], payload["value_calibrated"]["midgame"]
    assert payload["value_calibrated"]["temperature"] == pytest.approx(payload["temperature"])
    assert cal["ece"] < raw["ece"] and cal["ece"] < 0.02
    assert cal["accuracy"] == pytest.approx(raw["accuracy"])  # a temperature never flips a side of 0.5
    json.dumps(payload)


def test_calibrated_win_prob_divides_the_logit_by_t() -> None:
    logits = np.array([-3.0, -0.5, 0.0, 1.2, 4.0])
    values = 2.0 * _sigmoid(logits) - 1.0
    np.testing.assert_allclose(calibrated_win_prob(values, 1.0), _sigmoid(logits))
    np.testing.assert_allclose(calibrated_win_prob(values, TRUE_T), _sigmoid(logits / TRUE_T), rtol=1e-6)


def test_temperature_is_persisted_beside_the_checkpoint_and_applied_to_reported_win_prob(tmp_path: Path) -> None:
    from tm_bot.models.value_temperature import checkpoint_sha256

    best = tmp_path / "best"
    save_checkpoint(best, _tiny_model(), _VOCAB, metadata={"version": "warmstart-v3"})
    assert PolicyValueAgent.from_checkpoint(best).value_temperature == 1.0  # no fit: raw

    payload = value_temperature_payload(_synthetic_values(), checkpoint_sha=checkpoint_sha256(best))
    path = write_value_temperature(best, payload)
    assert path == value_temperature_path(best) == best / "checkpoint.value_temperature.json"
    assert value_temperature_path(best / "checkpoint.pt") == path
    agent = PolicyValueAgent.from_checkpoint(best / "checkpoint.pt")
    assert agent.value_temperature == pytest.approx(payload["temperature"])
    values = np.array([-0.8, 0.0, 0.6], dtype=np.float32)
    np.testing.assert_allclose(agent.win_probability(values), calibrated_win_prob(values, payload["temperature"]))
    import copy

    assert copy.deepcopy(agent).value_temperature == agent.value_temperature

    # A fit belongs to one checkpoint: overwriting the weights invalidates it.
    save_checkpoint(best, _tiny_model(seed=9), _VOCAB, metadata={"version": "warmstart-v3"})
    with pytest.raises(ValueError, match="different checkpoint"):
        PolicyValueAgent.from_checkpoint(best)


def test_metadata_temperature_is_used_without_a_sidecar(tmp_path: Path) -> None:
    ckpt = save_checkpoint(tmp_path / "m.pt", _tiny_model(), _VOCAB, metadata={"temperature": 2.0})
    assert PolicyValueAgent.from_checkpoint(ckpt).value_temperature == 2.0
    assert load_value_temperature(tmp_path / "m.pt", {"temperature": None}) == 1.0  # self-play checkpoints
    assert read_value_temperature_file(tmp_path / "m.pt") is None


def test_gate_check_14_scores_the_main_post_temperature_ece(tmp_path: Path) -> None:
    from tm_bot.models.value_temperature import checkpoint_sha256

    run = tmp_path / "run"
    best = run / "best"
    save_checkpoint(best, _tiny_model(), _VOCAB, metadata={"version": "warmstart-v3"})
    values = _synthetic_values()
    payload = value_temperature_payload(values, checkpoint_sha=checkpoint_sha256(best))
    raw_report = payload["value_raw"]
    (run / "metrics.json").write_text(json.dumps(
        {"status": "completed", "version": "warmstart-v3", "val": {"value": raw_report}, "value_calibrated": None}
    ), encoding="utf-8")
    with pytest.raises(ValueError, match="value_temperature"):
        runner.load_validation(run, "main")
    write_value_temperature(best, payload)
    _, value, _ = runner.load_validation(run, "main")
    assert value == json.loads(json.dumps(payload["value_calibrated"]))

    gate = wg.evaluate_gate(agent="main", val=None, baselines=None, value=value, summaries={})
    check = next(c for c in gate["checks"] if c["name"].startswith("Gen >= 2 value ECE"))
    assert check["name"] == "Gen >= 2 value ECE (post-temperature)"
    mid = values["generation"] >= 2
    p = _sigmoid(values["logit"][mid] / payload["temperature"])
    assert check["measured"] == pytest.approx(calibration_metrics(p, values["won"][mid])["ece"])
    assert check["passed"] and raw_report["midgame"]["ece"] > wg.VALUE_MAX_ECE  # raw would fail
