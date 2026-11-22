"""v2 warm-start training (#46): encoded-cache equivalence, shared multi-task loss, composite
selection, gen 2+ calibration, and both agents' identical configuration; the `space` task (#62).

Rows come from extracting the synthetic golden table pair (`tests/fixtures/golden_v2/`), which
emits every v2 table.
"""

from __future__ import annotations

from dataclasses import replace
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from tm_bot.agents.card_data import load_card_data
from tm_bot.comparator import warmstart as comparator_warmstart
from tm_bot.comparator.calibration import fit_midgame_temperature, fit_temperature
from tm_bot.comparator.model import ComparatorConfig, ComparatorNetwork, load_comparator_checkpoint
from tm_bot.data.extract import run_extraction
from tm_bot.data.inventory import run_inventory
from tm_bot.models.agent import PolicyValueAgent
from tm_bot.models.encode import collate_encoded_batch, encode_choice, encode_observation
from tm_bot.models.network import TMModelConfig, TMNetwork, load_checkpoint
from tm_bot.models.vocab import PROMPT_TYPE_TO_ID, CardVocab
from tm_bot.training import train_warmstart
from tm_bot.training.dataset import FastBatchBuilder, load_warmstart_corpus
from tm_bot.training.encoded_cache import CACHE_TASKS, build_row, ensure_encoded_cache
from tm_bot.training.multitask import (
    STATE_HEADS,
    MultiTaskSampler,
    WarmstartConfig,
    WarmstartData,
    composite_score,
    compute_multitask_step_losses,
    evaluate_state_task,
    load_warmstart_data,
    train_one_step,
)
from tm_bot.training.warmstart_builder import build_research_buy_choice

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "fixtures" / "golden_v2"
TABLE = "900000001"
ALL_TABLES = (
    "setup_choices", "setup_outcomes", "draft_picks", "research_buys", "midgame_values",
    "action_branches", "action_project_cards", "action_card_actions", "action_standard_projects",
    "action_spaces",
)


@pytest.fixture(scope="module")
def vocab() -> CardVocab:
    return CardVocab.from_card_data(load_card_data())


@pytest.fixture(scope="module")
def golden_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The v2 tables extracted from the golden pair, splits rewritten to alternate train/val."""
    root = tmp_path_factory.mktemp("golden")
    archive = root / "archive"
    for pid in ("10000001", "10000002"):
        dest = archive / pid
        dest.mkdir(parents=True)
        name = f"game_{TABLE}_{pid}.json"
        (dest / name).write_bytes((GOLDEN / pid / name).read_bytes())
    index = root / "index.sqlite"
    run_inventory(archive, index, progress_interval=0)
    out = root / "warmstart"
    run_extraction(archive, index, out, card_registry_path=REPO / "data" / "card_registry.json", progress_interval=0)
    for name in ALL_TABLES:
        path = out / f"{name}.parquet"
        table = pq.read_table(path)
        split = pa.array(["train" if i % 2 == 0 else "val" for i in range(table.num_rows)], pa.string())
        pq.write_table(table.set_column(table.schema.get_field_index("split"), "split", split), path)
    return out


@pytest.fixture(scope="module")
def golden_data(golden_dir: Path, vocab: CardVocab) -> WarmstartData:
    caches = ensure_encoded_cache(golden_dir, golden_dir.with_name("cache"), vocab=vocab, workers=0, log=lambda _: None)
    corpus = load_warmstart_corpus(golden_dir, vocab=vocab, load_drafts=False)
    return WarmstartData(vocab=vocab, corpus=corpus, builder=FastBatchBuilder(vocab, load_card_data()), caches=caches)


def _rows(golden_dir: Path, task: str) -> list[dict[str, Any]]:
    return pq.read_table(golden_dir / f"{CACHE_TASKS[task].table}.parquet").to_pylist()


def _assert_row_equal(fast: dict[str, torch.Tensor], b: int, ref: dict[str, torch.Tensor]) -> None:
    """Row `b` of a padded fast batch equals a single-row `collate_encoded_batch` output."""
    assert torch.equal(fast["obs_scalars"][b], ref["obs_scalars"][0])
    for prefix, keys in (("card", ("card_ids", "card_slots", "card_dyn")), ("space", ("space_feats",)),
                         ("cand", ("cand_card_ids", "cand_feats"))):
        mask_key = "cand_mask" if prefix == "cand" else f"{prefix}_mask"
        if mask_key not in ref:
            continue
        m_fast, m_ref = fast[mask_key][b], ref[mask_key][0]
        assert int(m_fast.sum()) == int(m_ref.sum())
        assert bool(m_fast[: int(m_fast.sum())].all())
        for key in keys:
            assert torch.equal(fast[key][b][m_fast], ref[key][0][m_ref]), key
    if "choice_ctx" in ref:
        assert torch.equal(fast["choice_ctx"][b], ref["choice_ctx"][0])
        assert int(fast["prompt_type_ids"][b]) == int(ref["prompt_type_ids"][0])
        assert int(fast["picked_mask"][b].sum()) == int(ref["picked_mask"][0].sum()) == 0


@pytest.mark.parametrize(
    "task", ["draft", "branch", "project_play", "standard_project", "card_action", "space", "midgame_value"]
)
def test_cache_rows_equal_reference_builder_encoding(golden_dir: Path, golden_data: WarmstartData,
                                                     vocab: CardVocab, task: str) -> None:
    table = golden_data.caches[task]
    rows = _rows(golden_dir, task)
    assert table.num_rows == len(rows) > 0
    # Gen 2+ action-phase rows carry the rebuilt board (#61); the fast path expands the compact codes
    # into the same `space_feats` as `encode_observation` on the builder's board below.
    has_board = [bool(r["board_tiles"]) for r in rows]
    assert all(has_board) if task in ("branch", "space") else not any(has_board) if task == "draft" else True
    idx = np.arange(table.num_rows)[::-1].copy()  # reversed: exercises padding to the widest row
    if task == "midgame_value":
        fast, won, margin = table.value_batch(idx)
    else:
        fast, target = table.policy_batch(idx)
    for b, i in enumerate(idx):
        row = rows[int(table["src_row"][i])]
        obs, choice, ref_target = build_row(task, row)
        enc = encode_observation(obs, vocab)
        ref = collate_encoded_batch([enc], [encode_choice(choice, obs, vocab)] if choice is not None else None)
        _assert_row_equal(fast, b, ref)
        if task == "midgame_value":
            assert float(won[b]) == float(row["won"])
            assert float(margin[b]) == float(row["vp_margin"])
        else:
            assert int(target[b]) == ref_target
            chosen = row["chosen_card"] if task == "draft" else row["chosen"]
            if task != "branch":
                assert choice.candidates[ref_target].id == chosen


def test_research_cache_rows_equal_reference_for_every_picked_subset(golden_dir: Path, golden_data: WarmstartData,
                                                                     vocab: CardVocab) -> None:
    table = golden_data.caches["research_buy"]
    rows = _rows(golden_dir, "research_buy")
    assert table.num_rows == len(rows) > 0
    checked = 0
    for i in range(table.num_rows):
        row = rows[int(table["src_row"][i])]
        packet = sorted(str(c) for c in row["packet_cards"])
        kept = sorted(str(c) for c in row["kept_cards"])
        for k in range(len(kept) + 1):
            for picked in combinations(kept, k):
                positions = [packet.index(c) for c in picked]
                fast, soft = table.research_batch(np.array([i, i]), picked=[positions, []])
                obs, choice = build_research_buy_choice(row, picked=picked)
                ref = collate_encoded_batch([encode_observation(obs, vocab)], [encode_choice(choice, obs, vocab)])
                _assert_row_equal(fast, 0, ref)
                ids = [c.id for c in choice.candidates]
                remaining = [c for c in kept if c not in picked]
                expected = np.zeros(soft.shape[1], dtype=np.float32)
                for c in remaining:
                    expected[ids.index(c)] = 1.0 / len(remaining)
                if not remaining:
                    expected[ids.index("done")] = 1.0
                assert np.allclose(soft[0].numpy(), expected)
                checked += 1
    assert checked >= 8


def test_sampler_draws_every_task_and_mixes_value_sources(golden_data: WarmstartData) -> None:
    cfg = replace(WarmstartConfig(), value_batch_size=10)
    sampler = MultiTaskSampler(golden_data, cfg, np.random.default_rng(0))
    step = sampler.next_step()
    for name in ("corp", "prelude", "project", "draft", "research_buy", "branch", "project_play",
                 "standard_project", "card_action", "space", "outcome"):
        assert step[f"{name}_batch"] is not None and step[f"{name}_targets"] is not None, name
    assert step["corp_targets"].shape[0] == 256 and step["branch_targets"].shape[0] == 512
    assert step["standard_project_targets"].shape[0] == 256 and step["card_action_targets"].shape[0] == 256
    assert step["space_targets"].shape[0] == 256
    assert cfg.value_counts() == (2, 4, 4)
    assert step["outcome_targets"].shape[0] == 10
    assert bool(torch.isfinite(step["outcome_margins"]).all())


def _tiny_models(vocab: CardVocab) -> list[torch.nn.Module]:
    torch.manual_seed(0)
    main = TMNetwork.from_vocab(vocab, TMModelConfig(
        vocab_size=vocab.size, d_model=64, n_heads=4, n_obs_layers=2, dim_feedforward=128, policy_ff_dim=128,
        dropout=0.0))
    comp = ComparatorNetwork.from_vocab(vocab, ComparatorConfig(
        vocab_size=vocab.size, d_model=96, card_hidden=96, card_emb_dim=32, card_static_proj_dim=32,
        choice_hidden=64, cand_hidden=96, value_hidden=96, dropout=0.0))
    return [main, comp]


def test_tiny_multitask_steps_reduce_every_task_loss_including_margin(golden_data: WarmstartData,
                                                                     vocab: CardVocab) -> None:
    cfg = replace(
        WarmstartConfig(), use_amp=False, setup_batch_size=4, draft_batch_size=8, research_buy_batch_size=4,
        branch_batch_size=8, project_play_batch_size=4, standard_project_batch_size=4, card_action_batch_size=2,
        space_batch_size=4, value_batch_size=10,
    )
    batches = MultiTaskSampler(golden_data, cfg, np.random.default_rng(1)).next_step()
    for model in _tiny_models(vocab):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        def losses() -> dict[str, float]:
            model.eval()
            with torch.no_grad():
                return compute_multitask_step_losses(model, weights=cfg, **batches)[1]

        before = losses()
        assert {"loss_value", "loss_value_margin", "loss_branch", "loss_research_buy", "loss_space"} <= set(before)
        for _ in range(20):
            train_one_step(model, optimizer, device=torch.device("cpu"), use_amp=False, weights=cfg, **batches)
        after = losses()
        for key, value in before.items():
            assert after[key] < value, (type(model).__name__, key, value, after[key])


def test_margin_loss_skips_rows_without_a_final_margin(golden_data: WarmstartData, vocab: CardVocab) -> None:
    cfg = replace(WarmstartConfig(), value_batch_size=10)
    step = MultiTaskSampler(golden_data, cfg, np.random.default_rng(2)).next_step()
    model = _tiny_models(vocab)[0].eval()
    value = {k: step[k] for k in ("outcome_batch", "outcome_targets", "outcome_margins")}
    with torch.no_grad():
        full = compute_multitask_step_losses(model, **value)[1]
        no_margin = compute_multitask_step_losses(
            model, **{**value, "outcome_margins": torch.full_like(value["outcome_margins"], float("nan"))})[1]
        partial = dict(value)
        partial["outcome_margins"] = value["outcome_margins"].clone()
        partial["outcome_margins"][1:] = float("nan")
        one = compute_multitask_step_losses(model, **partial)[1]
        out = model(value["outcome_batch"], compute_policy=False, compute_value=True)
    assert "loss_value_margin" in full and "loss_value_margin" not in no_margin
    target = float(np.clip(float(value["outcome_margins"][0]), -60, 60) / 50.0)
    expected = torch.nn.functional.smooth_l1_loss(out["vp_margin_pred"][:1], torch.tensor([target]), beta=0.2)
    assert one["loss_value_margin"] == pytest.approx(float(expected), rel=1e-5)


# --- the space task (#62) -------------------------------------------------------------------


def test_space_batches_are_legal_space_menus_on_the_rebuilt_board(golden_dir: Path,
                                                                  golden_data: WarmstartData) -> None:
    table = golden_data.caches["space"]
    rows = _rows(golden_dir, "space")
    assert len(table.split_indices["train"]) > 0 and len(table.split_indices["val"]) > 0
    idx = np.arange(table.num_rows)
    batch, target = table.policy_batch(idx)
    mask = batch["cand_mask"]
    menus = [len(rows[int(table["src_row"][i])]["menu"]) for i in idx]
    assert mask.sum(dim=1).tolist() == menus and min(menus) >= 2
    assert bool(((target >= 0) & (target < torch.tensor(menus))).all())
    assert (batch["prompt_type_ids"] == PROMPT_TYPE_TO_ID["space"]).all()
    # Space candidates carry no card id, only position (4, 5), one space-type bit (6..9) and bonuses.
    feats = batch["cand_feats"][mask]
    assert (batch["cand_card_ids"][mask] == 0).all()
    assert (feats[:, :4] == 0).all() and (feats[:, 19:] == 0).all()
    assert (feats[:, 6:10].sum(dim=1) == 1).all()
    for b in range(len(idx)):  # every menu entry is a different space
        xy = batch["cand_feats"][b][mask[b]][:, 4:6]
        assert len({tuple(v) for v in xy.tolist()}) == menus[b]
    # Every space row carries its table's rebuilt board, not the empty map template.
    assert bool(table["has_board"].all())
    template = torch.from_numpy(table.space_feats[table["map_id"][idx]])
    assert not torch.equal(batch["space_feats"], template)


def test_space_loss_is_hard_cross_entropy_scaled_by_w_space(golden_data: WarmstartData, vocab: CardVocab) -> None:
    table = golden_data.caches["space"]
    batch, target = table.policy_batch(table.split_indices["train"])
    cfg = replace(WarmstartConfig(), w_space=0.25)
    for model in _tiny_models(vocab):
        model.eval()
        with torch.no_grad():
            total, stats = compute_multitask_step_losses(model, weights=cfg, space_batch=batch, space_targets=target)
            logits = model(batch, compute_policy=True, compute_value=False)["policy_logits"].float()
        expected = float(torch.nn.functional.cross_entropy(logits, target))
        assert set(stats) == {"loss_space", "loss_total"}
        assert stats["loss_space"] == pytest.approx(expected, rel=1e-5)
        assert float(total) == pytest.approx(0.25 * expected, rel=1e-5)


def test_space_val_top1_scores_argmax_hits_on_the_val_rows(golden_data: WarmstartData, vocab: CardVocab) -> None:
    table = golden_data.caches["space"]
    rows = table.split_indices["val"]
    model = _tiny_models(vocab)[0].eval()
    out = evaluate_state_task(model, table, rows, device=torch.device("cpu"), batch_size=3, use_amp=False)
    batch, target = table.policy_batch(rows)
    with torch.no_grad():
        pred = model(batch, compute_policy=True, compute_value=False)["policy_logits"].argmax(dim=-1)
    hits = int((pred == target).sum())
    assert out["overall"] == {"correct": hits, "total": len(rows), "acc": hits / len(rows)}
    assert sum(m["total"] for m in out["per_map"].values()) == len(rows)
    assert out["per_gen"]["gen 1"]["total"] == 0  # only gen 2+ placements emit rows
    assert sum(g["total"] for g in out["per_gen"].values()) == len(rows)


def test_training_data_requires_the_space_table(golden_dir: Path, tmp_path: Path) -> None:
    data = tmp_path / "warmstart"
    data.mkdir()
    for path in golden_dir.glob("*.parquet"):
        if path.name != "action_spaces.parquet":
            (data / path.name).write_bytes(path.read_bytes())
    with pytest.raises(FileNotFoundError, match="action_spaces"):
        load_warmstart_data(data, tmp_path / "cache", workers=0, log=lambda _: None)
    # A value-only consumer (`value_temperature`) still loads a pre-#61 extraction.
    out = load_warmstart_data(data, tmp_path / "cache", tasks=("midgame_value",), workers=0, log=lambda _: None)
    assert set(out.caches) == {"midgame_value"}


def test_temperature_fit_uses_generation_two_plus_rows_only() -> None:
    rng = np.random.default_rng(0)
    n_mid, n_setup = 400, 300
    mid_logits = rng.normal(0.0, 2.0, n_mid)
    mid_targets = (rng.random(n_mid) < 1.0 / (1.0 + np.exp(-mid_logits / 2.5))).astype(np.float64)
    setup_logits = rng.normal(0.0, 4.0, n_setup)
    setup_targets = (setup_logits < 0).astype(np.float64)  # confidently wrong: would shrink T if included
    logits = np.concatenate([setup_logits, mid_logits])
    targets = np.concatenate([setup_targets, mid_targets])
    gens = np.concatenate([np.ones(n_setup, dtype=np.int64), rng.integers(2, 14, n_mid)])

    fit = fit_midgame_temperature(logits, targets, gens)
    reference = fit_temperature(mid_logits, mid_targets)
    assert fit["n"] == n_mid and fit["excluded_gen1"] == n_setup
    assert fit["temperature"] == pytest.approx(reference["temperature"])
    assert abs(fit_temperature(logits, targets)["temperature"] - fit["temperature"]) > 0.5


def test_composite_score_matches_hand_computation() -> None:
    m = {
        "corp_top1": 0.7, "prelude0_top1": 0.8, "draft_top1": 0.6, "research_buy_acc": 0.5,
        "branch_top1": 0.4, "project_play_top1": 0.3, "standard_project_top1": 0.5, "card_action_top1": 0.6,
        "space_top1": 0.7, "midgame_val_brier": 0.2,
    }
    # 0.25 * 0.65 + 0.50 * mean(0.4, 0.3, 0.5, 0.6, 0.7) + 0.25 * (1 - 0.2 / 0.25) = 0.1625 + 0.25 + 0.05
    assert composite_score(m) == pytest.approx(0.4625)
    # A Brier worse than 0.25 contributes nothing (not a negative term).
    assert composite_score({**m, "midgame_val_brier": 0.3}) == pytest.approx(0.4125)
    # `space` counts as one of the five action-phase tasks, like the other four.
    assert composite_score({**m, "space_top1": 0.2}) == pytest.approx(0.4625 - 0.5 * 0.5 / 5)


def test_both_agents_share_tasks_weights_and_step_budget() -> None:
    main_cfg = train_warmstart.build_train_config([])
    comp_cfg = comparator_warmstart.build_train_config([])
    assert main_cfg == comp_cfg
    assert main_cfg.task_weights() == comp_cfg.task_weights() == {
        "corp": 0.40, "prelude": 0.40, "init_project": 0.30, "draft": 0.80, "research_buy": 0.70,
        "branch": 1.00, "project_play": 1.00, "standard_project": 0.40, "card_action": 0.40, "space": 0.40,
        "value_win": 0.80, "value_margin": 0.30,
    }
    assert main_cfg.task_batch_sizes() == comp_cfg.task_batch_sizes() == {
        "setup": 256, "draft": 512, "research_buy": 512, "branch": 512, "project_play": 512,
        "standard_project": 256, "card_action": 256, "space": 256, "value": 512,
    }
    # Every gen 2+ policy head has a weight and a minibatch in the shared config.
    assert {name for name, _, _ in STATE_HEADS} <= set(main_cfg.task_batch_sizes())
    assert main_cfg.value_counts() == (102, 205, 205)
    assert (main_cfg.epochs, main_cfg.steps_per_epoch, main_cfg.total_steps) == (6, 2500, 15000)
    assert main_cfg.use_amp and main_cfg.warmup_steps > 0 and main_cfg.min_lr < main_cfg.lr
    assert train_warmstart.MAIN_AGENT.version == "warmstart-v4"
    assert comparator_warmstart.COMPARATOR_AGENT.version == "comparator-warmstart-v3"


def test_both_agents_train_save_tagged_checkpoints_and_reports(golden_dir: Path, tmp_path: Path) -> None:
    cfg = replace(
        WarmstartConfig(), use_amp=False, epochs=1, steps_per_epoch=2, warmup_steps=1, log_every=0,
        setup_batch_size=4, draft_batch_size=4, research_buy_batch_size=4, branch_batch_size=4,
        project_play_batch_size=4, standard_project_batch_size=4, card_action_batch_size=2, space_batch_size=4,
        value_batch_size=10,
    )
    cache = golden_dir.with_name("cache")
    main = train_warmstart.run_warmstart_training(
        data_dir=golden_dir, cache_dir=cache, out_dir=tmp_path / "main", train_cfg=cfg, device_str="cpu", workers=0)
    comp = comparator_warmstart.run_comparator_warmstart(
        data_dir=golden_dir, cache_dir=cache, out_dir=tmp_path / "comp", train_cfg=cfg, device_str="cpu", workers=0)

    _, _, main_meta = load_checkpoint(tmp_path / "main" / "best", map_location="cpu")
    _, _, comp_meta = load_comparator_checkpoint(tmp_path / "comp" / "best", map_location="cpu")
    for meta, version, out in ((main_meta, "warmstart-v4", "main"), (comp_meta, "comparator-warmstart-v3", "comp")):
        assert meta["version"] == version
        assert meta["warmstart_data_version"] == 2 and meta["encoder_version"] == 2
        assert meta["model_config"]["encoder_version"] == 2
        assert meta["train_config"] == cfg.to_dict()
        assert "Composite score" in meta["report"]
        assert (tmp_path / out / "best" / "report.md").exists()
        metrics = json.loads((tmp_path / out / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["status"] == "completed" and "test_model" not in metrics
        assert set(metrics["val"]["value"]["per_gen"]) == {"gen 1", "gen 2-5", "gen 6-9", "gen 10+"}
        assert metrics["val"]["space"]["overall"]["total"] > 0
        assert metrics["val"]["composite"]["inputs"]["space_top1"] == metrics["val"]["space"]["overall"]["acc"]
        assert "| Tile placement (space) |" in meta["report"]
    assert main["val"]["composite"]["score"] == pytest.approx(main_meta["val_composite"])

    # The comparator's temperature is fit on the val split's gen 2+ value rows only.
    data_val = {name: pq.read_table(golden_dir / f"{name}.parquet").to_pylist() for name in ("setup_outcomes", "midgame_values")}
    fit = comp_meta["val_calibration"]
    assert fit["n"] == sum(r["split"] == "val" for r in data_val["midgame_values"])
    assert fit["excluded_gen1"] == sum(r["split"] == "val" for r in data_val["setup_outcomes"]) > 0
    assert comp_meta["temperature"] == pytest.approx(fit["temperature"]) == pytest.approx(comp["temperature"])
    assert "value_calibrated" in comp and comp["value_calibrated"]["temperature"] == pytest.approx(fit["temperature"])
    # Since #60 the main warm start fits and stores its value temperature the same way.
    main_fit = main_meta["val_calibration"]
    assert (main_fit["n"], main_fit["excluded_gen1"]) == (fit["n"], fit["excluded_gen1"])
    assert main_meta["temperature"] == pytest.approx(main_fit["temperature"]) == pytest.approx(main["temperature"])
    assert main["value_calibrated"]["temperature"] == pytest.approx(main_fit["temperature"])
    assert PolicyValueAgent.from_checkpoint(tmp_path / "main" / "best").value_temperature == pytest.approx(
        main_fit["temperature"]
    )

    with pytest.raises(RuntimeError, match="already completed"):
        train_warmstart.run_warmstart_training(
            data_dir=golden_dir, cache_dir=cache, out_dir=tmp_path / "main", train_cfg=cfg, device_str="cpu")
