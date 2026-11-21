"""On-disk encoded cache of the v2 warm-start state tables (#46).

Every gen 2+ row (`draft_picks`, `research_buys`, the four action tables and `midgame_values`) is
run once through its reference builder in `warmstart_builder` and `encode_observation` /
`encode_choice`; the encoder outputs are stored as flat memory-mapped arrays. Training then
gathers a minibatch with a few vectorized index operations instead of re-encoding rows, and both
agents share one cache. Equality with the reference path is by construction and is pinned per task
by `tests/test_warmstart_v2.py`.

Layout of one task directory: `meta.json` plus one raw `<field>.bin` per field. Row fields hold
one entry per cached row, `tok_*` fields one entry per card token and `cand_*` fields one entry per
candidate; `tok_count` / `cand_count` give each row's ragged length.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch

from tm_bot.agents.card_data import load_card_data
from tm_bot.env.types import SUPPORTED_MAPS
from tm_bot.models.encode import (
    CAND_FEAT_DIM,
    CARD_DYN_DIM,
    CHOICE_CTX_DIM,
    ENCODER_VERSION,
    OBS_SCALAR_DIM,
    SPACE_FEAT_DIM,
    encode_choice,
    encode_observation,
)
from tm_bot.models.vocab import DONE_ID, PROMPT_TYPES, CardVocab
from tm_bot.training import warmstart_builder as wb

CACHE_FORMAT = 1
DEFAULT_CACHE_DIR = Path("data") / "warmstart_cache"

_SPLIT_TO_IDX: dict[str, int] = {"train": 0, "val": 1, "test": 2}
PHASE_RESEARCH = 0
PHASE_ACTION = 1
PHASE_OTHER = 2
_PHASE_TO_IDX = {"research": PHASE_RESEARCH, "action": PHASE_ACTION}

# Candidate-feature columns the cached prompt kinds can set (`encode_choice`): card kinds use
# card/done/hand-cost/in-hand (0..3), v2 root branches one bit in 22..31. Building asserts every
# other column is zero, so the compact storage is exact.
_CARD_COLS: tuple[int, ...] = (0, 1, 2, 3)
_BRANCH_COLS: tuple[int, ...] = tuple(range(22, 32))


@dataclass(frozen=True)
class CacheTask:
    """One cached warm-start task: its parquet table, row kind and stored candidate columns."""

    name: str
    table: str
    kind: str  # "policy" | "research" | "value"
    feat_cols: tuple[int, ...] = ()


CACHE_TASKS: dict[str, CacheTask] = {
    t.name: t
    for t in (
        CacheTask("draft", "draft_picks", "policy", _CARD_COLS),
        CacheTask("research_buy", "research_buys", "research", _CARD_COLS),
        CacheTask("branch", "action_branches", "policy", _BRANCH_COLS),
        CacheTask("project_play", "action_project_cards", "policy", _CARD_COLS),
        CacheTask("standard_project", "action_standard_projects", "policy", _CARD_COLS),
        CacheTask("card_action", "action_card_actions", "policy", _CARD_COLS),
        CacheTask("midgame_value", "midgame_values", "value"),
    )
}

_POLICY_BUILDERS: dict[str, Callable[..., Any]] = {
    "draft": wb.build_draft_pick_state_choice,
    "branch": wb.build_branch_choice,
    "project_play": wb.build_project_card_choice,
    "standard_project": wb.build_standard_project_choice,
    "card_action": wb.build_card_action_choice,
}


def _row_fields(task: CacheTask) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Per-row fields: name -> (numpy dtype, trailing shape)."""
    fields: dict[str, tuple[str, tuple[int, ...]]] = {
        "src_row": ("<i8", ()),
        "map_id": ("u1", ()),
        "seat": ("u1", ()),
        "split": ("u1", ()),
        "generation": ("u1", ()),
        "obs_scalars": ("<f4", (OBS_SCALAR_DIM,)),
        "tok_count": ("<i4", ()),
    }
    if task.kind in ("policy", "research"):
        fields.update(
            {
                "choice_ctx": ("<f4", (CHOICE_CTX_DIM,)),
                "prompt_type": ("u1", ()),
                "cand_count": ("<i4", ()),
            }
        )
    if task.kind == "policy":
        fields["target"] = ("<i2", ())
    if task.kind == "research":
        fields["kept_mask"] = ("u1", ())  # bit j: packet candidate j was bought
    if task.kind == "value":
        fields.update({"won": ("u1", ()), "vp_margin": ("<f4", ()), "phase": ("u1", ())})
    return fields


_TOKEN_FIELDS: dict[str, tuple[str, tuple[int, ...]]] = {
    "tok_card": ("<u2", ()),
    "tok_slot": ("u1", ()),
    "tok_dyn0": ("<f4", ()),
    "tok_dyn3": ("u1", ()),
}


def _cand_fields(task: CacheTask) -> dict[str, tuple[str, tuple[int, ...]]]:
    if task.kind == "value":
        return {}
    return {"cand_card": ("<u2", ()), "cand_feats": ("<f4", (len(task.feat_cols),))}


# --- encoding one chunk of rows (runs in worker processes) --------------------------------

_WORKER: dict[str, Any] = {}


def _init_worker() -> None:
    cards = load_card_data()
    _WORKER["cards"] = cards
    _WORKER["vocab"] = CardVocab.from_card_data(cards)
    _WORKER["templates"] = wb.load_map_templates()


def _ensure_worker() -> None:
    if not _WORKER:
        _init_worker()


def build_row(task_name: str, row: Mapping[str, Any]) -> tuple[Any, Any, int | None]:
    """Reference `(ActorObservation, Choice | None, target index | None)` of one table row.

    Research buys are built with nothing picked yet; the sampler derives later sequential steps.
    """
    _ensure_worker()
    kwargs = {"cards_by_name": _WORKER["cards"], "templates": _WORKER["templates"]}
    if task_name == "midgame_value":
        return wb.build_midgame_value_observation(row, **kwargs), None, None
    if task_name == "research_buy":
        obs, choice = wb.build_research_buy_choice(row, **kwargs)
        return obs, choice, None
    obs, choice = _POLICY_BUILDERS[task_name](row, **kwargs)
    if task_name == "branch":
        target = [str(k) for k in row["menu"]].index(str(row["chosen"]))
    else:
        chosen = str(row["chosen_card"] if task_name == "draft" else row["chosen"])
        target = [c.id for c in choice.candidates].index(chosen)
    return obs, choice, target


def row_is_cacheable(task_name: str, row: Mapping[str, Any]) -> bool:
    """Draft rows without a reconstructed pre-pick snapshot have no gen 2+ state to encode."""
    return task_name != "draft" or bool(row.get("has_snapshot"))


def _encode_chunk(task_name: str, start: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    _ensure_worker()
    vocab: CardVocab = _WORKER["vocab"]
    task = CACHE_TASKS[task_name]
    row_f = _row_fields(task)
    out: dict[str, list[Any]] = {k: [] for k in (*row_f, *_TOKEN_FIELDS, *_cand_fields(task))}
    spaces: dict[int, np.ndarray] = {}
    skipped = 0
    for i, row in enumerate(rows):
        if not row_is_cacheable(task_name, row):
            skipped += 1
            continue
        obs, choice, target = build_row(task_name, row)
        enc = encode_observation(obs, vocab)
        map_id = SUPPORTED_MAPS.index(wb.normalize_map_name(row["map"]))
        if map_id in spaces:
            if not np.array_equal(spaces[map_id], enc.space_feats):
                raise ValueError(f"{task_name} row {start + i}: board spaces differ from the map template")
        else:
            spaces[map_id] = enc.space_feats
        if np.any(enc.card_dyn[:, 1:3] != 0.0):
            raise ValueError(f"{task_name} row {start + i}: card resources/disabled flags are not cached")
        out["src_row"].append(start + i)
        out["map_id"].append(map_id)
        out["seat"].append(int(row["seat"]))
        out["split"].append(_SPLIT_TO_IDX[str(row["split"])])
        out["generation"].append(min(255, int(row["generation"])))
        out["obs_scalars"].append(enc.obs_scalars)
        out["tok_count"].append(enc.card_ids.shape[0])
        out["tok_card"].append(enc.card_ids)
        out["tok_slot"].append(enc.card_slots)
        out["tok_dyn0"].append(enc.card_dyn[:, 0])
        out["tok_dyn3"].append(enc.card_dyn[:, 3])
        if choice is not None:
            ech = encode_choice(choice, obs, vocab)
            if ech.picked_card_ids.shape[0] != 0:
                raise ValueError(f"{task_name} row {start + i}: picked cards are not cached")
            other = np.delete(ech.cand_feats, task.feat_cols, axis=1)
            if np.any(other != 0.0):
                raise ValueError(f"{task_name} row {start + i}: candidate features outside {task.feat_cols}")
            out["choice_ctx"].append(ech.choice_ctx)
            out["prompt_type"].append(ech.prompt_type_id)
            out["cand_count"].append(ech.cand_card_ids.shape[0])
            out["cand_card"].append(ech.cand_card_ids)
            out["cand_feats"].append(ech.cand_feats[:, list(task.feat_cols)])
        if task.kind == "policy":
            out["target"].append(int(target))  # type: ignore[arg-type]
        elif task.kind == "research":
            kept = {str(c) for c in row["kept_cards"]}
            mask = 0
            for j, cand in enumerate(choice.candidates):
                if cand.id in kept:
                    mask |= 1 << j
            out["kept_mask"].append(mask)
        else:
            out["won"].append(1 if bool(row["won"]) else 0)
            margin = row.get("vp_margin")
            out["vp_margin"].append(np.nan if margin is None else float(margin))
            out["phase"].append(_PHASE_TO_IDX.get(str(row["phase"]), PHASE_OTHER))

    arrays: dict[str, np.ndarray] = {}
    all_fields = {**row_f, **_TOKEN_FIELDS, **_cand_fields(task)}
    for name, (dtype, trail) in all_fields.items():
        vals = out[name]
        if name in row_f:
            arr = np.asarray(vals, dtype=dtype).reshape((len(vals), *trail))
        elif vals:
            arr = np.concatenate([np.asarray(v).reshape((-1, *trail)) for v in vals]).astype(dtype)
        else:
            arr = np.zeros((0, *trail), dtype=dtype)
        if name in ("tok_card", "cand_card") and arr.size and int(arr.max()) > np.iinfo(np.uint16).max:
            raise ValueError("card ids exceed uint16")
        arrays[name] = arr
    return {"arrays": arrays, "spaces": spaces, "skipped": skipped, "n_in": len(rows)}


# --- building ------------------------------------------------------------------------------


def _vocab_fingerprint(vocab: CardVocab) -> str:
    return hashlib.sha1("\n".join(vocab.names).encode("utf-8")).hexdigest()


def _source_signature(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {"path": str(path.resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _expected_meta(task: CacheTask, source: Path, vocab: CardVocab) -> dict[str, Any]:
    return {
        "format": CACHE_FORMAT,
        "task": task.name,
        "encoder_version": ENCODER_VERSION,
        "obs_scalar_dim": OBS_SCALAR_DIM,
        "choice_ctx_dim": CHOICE_CTX_DIM,
        "num_prompt_types": len(PROMPT_TYPES),
        "feat_cols": list(task.feat_cols),
        "vocab_size": vocab.size,
        "vocab_sha1": _vocab_fingerprint(vocab),
        "source": _source_signature(source),
    }


def cache_is_current(task_dir: Path, task: CacheTask, source: Path, vocab: CardVocab) -> bool:
    """True when `task_dir` holds a complete cache of `source` for this encoder and vocab."""
    meta_path = task_dir / "meta.json"
    if not meta_path.exists() or not source.exists():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = _expected_meta(task, source, vocab)
    return all(meta.get(k) == v for k, v in expected.items())


def build_task_cache(
    task_name: str,
    source: Path,
    task_dir: Path,
    *,
    vocab: CardVocab,
    workers: int = 0,
    chunk_rows: int = 2048,
    log: Callable[[str], None] = print,
) -> None:
    """Encode every row of `source` into `task_dir` (written to a temp dir, then renamed)."""
    task = CACHE_TASKS[task_name]
    tmp_dir = task_dir.with_name(task_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    all_fields = {**_row_fields(task), **_TOKEN_FIELDS, **_cand_fields(task)}
    files = {name: open(tmp_dir / f"{name}.bin", "wb") for name in all_fields}
    counts = {name: 0 for name in all_fields}
    spaces: dict[int, np.ndarray] = {}
    skipped = 0
    n_in = 0
    t0 = time.perf_counter()

    def consume(res: dict[str, Any]) -> None:
        nonlocal skipped, n_in
        for name, arr in res["arrays"].items():
            files[name].write(np.ascontiguousarray(arr).tobytes())
            counts[name] += int(arr.shape[0])
        for mid, sp in res["spaces"].items():
            if mid in spaces and not np.array_equal(spaces[mid], sp):
                raise ValueError(f"{task_name}: board spaces differ between rows of map {mid}")
            spaces.setdefault(mid, sp)
        skipped += int(res["skipped"])
        n_in += int(res["n_in"])

    pf = pq.ParquetFile(source)
    chunks = (
        batch.to_pylist() for batch in pf.iter_batches(batch_size=chunk_rows)
    )
    try:
        if workers > 0:
            ctx = mp.get_context("spawn")
            with ctx.Pool(workers, initializer=_init_worker) as pool:
                pending: deque[Any] = deque()
                start = 0
                for rows in chunks:
                    pending.append(pool.apply_async(_encode_chunk, (task_name, start, rows)))
                    start += len(rows)
                    if len(pending) >= 2 * workers:
                        consume(pending.popleft().get())
                while pending:
                    consume(pending.popleft().get())
        else:
            start = 0
            for rows in chunks:
                consume(_encode_chunk(task_name, start, rows))
                start += len(rows)
    finally:
        for fh in files.values():
            fh.close()

    n_rows = counts["src_row"]
    if n_rows + skipped != n_in or n_in != pf.metadata.num_rows:
        raise RuntimeError(f"{task_name}: encoded {n_rows} + skipped {skipped} != {pf.metadata.num_rows} rows")
    max_spaces = max((sp.shape[0] for sp in spaces.values()), default=1)
    space_feats = np.zeros((len(SUPPORTED_MAPS), max_spaces, SPACE_FEAT_DIM), dtype=np.float32)
    space_mask = np.zeros((len(SUPPORTED_MAPS), max_spaces), dtype=bool)
    for mid, sp in spaces.items():
        space_feats[mid, : sp.shape[0]] = sp
        space_mask[mid, : sp.shape[0]] = True
    np.save(tmp_dir / "space_feats.npy", space_feats)
    np.save(tmp_dir / "space_mask.npy", space_mask)
    meta = _expected_meta(task, source, vocab)
    meta.update(
        {
            "num_rows": n_rows,
            "skipped_rows": skipped,
            "fields": {
                name: {"dtype": dtype, "shape": [counts[name], *trail]}
                for name, (dtype, trail) in all_fields.items()
            },
            "build_seconds": round(time.perf_counter() - t0, 1),
        }
    )
    (tmp_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if task_dir.exists():
        shutil.rmtree(task_dir)
    tmp_dir.rename(task_dir)
    log(f"[cache] {task_name}: {n_rows:,} rows ({skipped:,} skipped) in {meta['build_seconds']}s")


def ensure_encoded_cache(
    data_dir: Path | str,
    cache_dir: Path | str,
    *,
    tasks: Sequence[str] | None = None,
    vocab: CardVocab | None = None,
    workers: int | None = None,
    rebuild: bool = False,
    log: Callable[[str], None] = print,
) -> dict[str, EncodedTable]:
    """Build any missing or stale task caches of `data_dir` under `cache_dir` and open them all."""
    data_path = Path(data_dir)
    cache_path = Path(cache_dir)
    resolved_vocab = vocab if vocab is not None else CardVocab.from_card_data()
    if workers is None:
        workers = max(0, min(16, (os.cpu_count() or 1) - 2))
    opened: dict[str, EncodedTable] = {}
    for name in tasks or tuple(CACHE_TASKS):
        task = CACHE_TASKS[name]
        source = data_path / f"{task.table}.parquet"
        if not source.exists():
            raise FileNotFoundError(f"v2 warm-start table missing: {source}")
        task_dir = cache_path / name
        if rebuild or not cache_is_current(task_dir, task, source, resolved_vocab):
            log(f"[cache] encoding {source} -> {task_dir} ({workers} workers) ...")
            build_task_cache(name, source, task_dir, vocab=resolved_vocab, workers=workers, log=log)
        opened[name] = EncodedTable(task_dir)
    return opened


# --- reading and batch assembly ------------------------------------------------------------


class EncodedTable:
    """Read-only view of one task cache: small row fields in RAM, token/candidate arrays memory-mapped."""

    _IN_RAM = ("map_id", "seat", "split", "generation", "tok_count", "cand_count", "target",
               "kept_mask", "won", "vp_margin", "phase", "prompt_type", "src_row")

    def __init__(self, task_dir: Path | str) -> None:
        self.dir = Path(task_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        if self.meta.get("encoder_version") != ENCODER_VERSION or self.meta.get("format") != CACHE_FORMAT:
            raise ValueError(f"{self.dir} was cached for another encoder or cache format")
        self.task = CACHE_TASKS[self.meta["task"]]
        self.arrays: dict[str, np.ndarray] = {}
        for name, spec in self.meta["fields"].items():
            shape = tuple(spec["shape"])
            path = self.dir / f"{name}.bin"
            if shape[0] == 0:
                arr = np.zeros(shape, dtype=spec["dtype"])
            else:
                arr = np.memmap(path, dtype=spec["dtype"], mode="r", shape=shape)
            self.arrays[name] = np.array(arr) if name in self._IN_RAM else arr
        self.num_rows = int(self.meta["num_rows"])
        self.tok_off = np.zeros(self.num_rows + 1, dtype=np.int64)
        np.cumsum(self.arrays["tok_count"], out=self.tok_off[1:])
        if "cand_count" in self.arrays:
            self.cand_off = np.zeros(self.num_rows + 1, dtype=np.int64)
            np.cumsum(self.arrays["cand_count"], out=self.cand_off[1:])
        self.space_feats = np.load(self.dir / "space_feats.npy")
        self.space_mask = np.load(self.dir / "space_mask.npy")
        split = self.arrays["split"]
        self.split_indices = {
            name: np.nonzero(split == idx)[0].astype(np.int64) for name, idx in _SPLIT_TO_IDX.items()
        }

    def __getitem__(self, name: str) -> np.ndarray:
        return self.arrays[name]

    def obs_batch(self, rows: np.ndarray) -> dict[str, torch.Tensor]:
        """Padded observation tensors of `rows`, as `collate_encoded_batch` would produce."""
        rows = np.asarray(rows, dtype=np.int64)
        bsz = rows.shape[0]
        a = self.arrays
        valid, flat = _ragged_index(self.tok_off, rows)
        width = valid.shape[1]
        card_ids = np.zeros((bsz, width), dtype=np.int64)
        card_slots = np.zeros((bsz, width), dtype=np.int64)
        card_dyn = np.zeros((bsz, width, CARD_DYN_DIM), dtype=np.float32)
        card_ids[valid] = a["tok_card"][flat]
        card_slots[valid] = a["tok_slot"][flat]
        card_dyn[..., 0][valid] = a["tok_dyn0"][flat]
        card_dyn[..., 3][valid] = a["tok_dyn3"][flat]
        mids = a["map_id"][rows]
        return {
            "obs_scalars": torch.from_numpy(np.ascontiguousarray(a["obs_scalars"][rows], dtype=np.float32)),
            "card_ids": torch.from_numpy(card_ids),
            "card_slots": torch.from_numpy(card_slots),
            "card_dyn": torch.from_numpy(card_dyn),
            "card_mask": torch.from_numpy(valid),
            "space_feats": torch.from_numpy(self.space_feats[mids]),
            "space_mask": torch.from_numpy(self.space_mask[mids]),
        }

    def _cand_arrays(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        valid, flat = _ragged_index(self.cand_off, rows)
        bsz, width = valid.shape
        cand_ids = np.zeros((bsz, width), dtype=np.int64)
        cand_feats = np.zeros((bsz, width, CAND_FEAT_DIM), dtype=np.float32)
        cand_ids[valid] = self.arrays["cand_card"][flat]
        packed = np.asarray(self.arrays["cand_feats"][flat], dtype=np.float32)
        for k, col in enumerate(self.task.feat_cols):
            cand_feats[..., col][valid] = packed[:, k]
        return cand_ids, cand_feats, valid

    def policy_batch(self, rows: np.ndarray) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """`(batch, target index)` of a policy task with a fixed candidate menu per row."""
        rows = np.asarray(rows, dtype=np.int64)
        batch = self.obs_batch(rows)
        cand_ids, cand_feats, cand_mask = self._cand_arrays(rows)
        batch.update(self._choice_common(rows))
        batch["cand_card_ids"] = torch.from_numpy(cand_ids)
        batch["cand_feats"] = torch.from_numpy(cand_feats)
        batch["cand_mask"] = torch.from_numpy(cand_mask)
        return batch, torch.from_numpy(self.arrays["target"][rows].astype(np.int64))

    def _choice_common(self, rows: np.ndarray) -> dict[str, torch.Tensor]:
        bsz = rows.shape[0]
        return {
            "choice_ctx": torch.from_numpy(np.ascontiguousarray(self.arrays["choice_ctx"][rows], dtype=np.float32)),
            "prompt_type_ids": torch.from_numpy(self.arrays["prompt_type"][rows].astype(np.int64)),
            "picked_card_ids": torch.zeros((bsz, 1), dtype=torch.int64),
            "picked_mask": torch.zeros((bsz, 1), dtype=torch.bool),
        }

    def research_batch(
        self,
        rows: np.ndarray,
        *,
        picked: Sequence[Sequence[int]] | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """`(batch, soft targets)` of one sequential research-buy step per row.

        `picked[b]` lists the packet positions already selected (a subset of the bought cards);
        without it each row draws a uniform number of its bought cards, in random order, like the
        initial project buy. The frame matches `build_research_buy_choice(row, picked=...)`: the
        unpicked packet cards plus `done`; the target is uniform over the unpicked bought cards, or
        `done` once every bought card is picked.
        """
        rows = np.asarray(rows, dtype=np.int64)
        bsz = rows.shape[0]
        r = rng if rng is not None else np.random.default_rng(0)
        batch = self.obs_batch(rows)
        full_ids, full_feats, full_mask = self._cand_arrays(rows)
        common = self._choice_common(rows)
        ctx = common["choice_ctx"].numpy().copy()
        o = len(PROMPT_TYPES)
        width = full_ids.shape[1]
        cand_ids = np.zeros((bsz, width), dtype=np.int64)
        cand_feats = np.zeros((bsz, width, CAND_FEAT_DIM), dtype=np.float32)
        cand_mask = np.zeros((bsz, width), dtype=bool)
        targets = np.zeros((bsz, width), dtype=np.float32)
        kept_masks = self.arrays["kept_mask"][rows]
        for b in range(bsz):
            n_full = int(full_mask[b].sum())
            n_packet = n_full - 1  # the last candidate is `done`
            kept = [j for j in range(n_packet) if (int(kept_masks[b]) >> j) & 1]
            if picked is not None:
                chosen = [int(j) for j in picked[b]]
            else:
                k_cur = int(r.integers(0, len(kept) + 1))
                chosen = [kept[i] for i in r.permutation(len(kept))[:k_cur]]
            chosen_set = set(chosen)
            keep_pos = [j for j in range(n_packet) if j not in chosen_set] + [n_packet]
            n = len(keep_pos)
            cand_ids[b, :n] = full_ids[b, keep_pos]
            cand_feats[b, :n] = full_feats[b, keep_pos]
            cand_mask[b, :n] = True
            ctx[b, o + 0] = float(n) / 30.0
            ctx[b, o + 2] = float(len(chosen_set)) / 10.0
            remaining = [i for i, j in enumerate(keep_pos[:-1]) if j in kept and j not in chosen_set]
            if remaining:
                targets[b, remaining] = 1.0 / float(len(remaining))
            else:
                targets[b, n - 1] = 1.0
        if bsz and not np.all(cand_ids[np.arange(bsz), cand_mask.sum(axis=1) - 1] == DONE_ID):
            raise ValueError("research cache rows must end with the done candidate")
        batch.update(common)
        batch["choice_ctx"] = torch.from_numpy(ctx)
        batch["cand_card_ids"] = torch.from_numpy(cand_ids)
        batch["cand_feats"] = torch.from_numpy(cand_feats)
        batch["cand_mask"] = torch.from_numpy(cand_mask)
        return batch, torch.from_numpy(targets)

    def value_batch(self, rows: np.ndarray) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """`(obs batch, won, raw final VP margin or NaN)` of value rows."""
        rows = np.asarray(rows, dtype=np.int64)
        batch = self.obs_batch(rows)
        won = torch.from_numpy(self.arrays["won"][rows].astype(np.float32))
        margin = torch.from_numpy(self.arrays["vp_margin"][rows].astype(np.float32))
        return batch, won, margin


def _ragged_index(offsets: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(valid[B, W], flat indices of the valid cells)` of ragged rows padded to width `W`."""
    starts = offsets[rows]
    counts = offsets[rows + 1] - starts
    width = max(1, int(counts.max())) if rows.shape[0] else 1
    pos = np.arange(width, dtype=np.int64)
    valid = pos[None, :] < counts[:, None]
    flat = (starts[:, None] + pos[None, :])[valid]
    return valid, flat


_PAD_DIM1 = ("card_ids", "card_slots", "card_dyn", "card_mask", "space_feats", "space_mask")


def concat_obs_batches(batches: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Concatenate observation batches along the batch axis, zero-padding the ragged dimensions."""
    out: dict[str, torch.Tensor] = {}
    keys = [k for k in batches[0] if all(k in b for b in batches)]
    for key in keys:
        parts = [b[key] for b in batches]
        if key in _PAD_DIM1:
            width = max(p.shape[1] for p in parts)
            padded = []
            for p in parts:
                if p.shape[1] < width:
                    pad_shape = (p.shape[0], width - p.shape[1], *p.shape[2:])
                    p = torch.cat([p, torch.zeros(pad_shape, dtype=p.dtype)], dim=1)
                padded.append(p)
            parts = padded
        out[key] = torch.cat(parts, dim=0)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the encoded v2 warm-start cache (#46).")
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "warmstart")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)
    ensure_encoded_cache(args.data_dir, args.cache_dir, workers=args.workers, rebuild=args.rebuild)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
