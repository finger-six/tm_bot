"""Compact warm-start dataset loader and fast tensor batch builder."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch

from tm_bot.agents.card_data import CardSpec, load_card_data
from tm_bot.env.types import SUPPORTED_MAPS
from tm_bot.models.encode import (
    CAND_FEAT_DIM,
    CARD_DYN_DIM,
    CHOICE_CTX_DIM,
    OBS_SCALAR_DIM,
    SPACE_FEAT_DIM,
    encode_observation,
)
from tm_bot.models.vocab import (
    DONE_ID,
    PAD_ID,
    PROMPT_TYPE_TO_ID,
    PROMPT_TYPES,
    SLOT_TO_ID,
    TAG_NAMES,
    CardVocab,
)
from tm_bot.training.warmstart_builder import (
    build_corp_choice,
    load_map_templates,
    normalize_map_name,
)

DEFAULT_WARMSTART_DIR = Path(r"C:\2026\tm_bot\data\warmstart")

_TAG_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(TAG_NAMES)}
_SPLIT_TO_IDX: dict[str, int] = {"train": 0, "val": 1, "test": 2}
_IDX_TO_SPLIT: tuple[str, ...] = ("train", "val", "test")

_SLOT_MY_HAND = SLOT_TO_ID["my_hand"]
_SLOT_MY_TABLEAU = SLOT_TO_ID["my_tableau"]
_SLOT_MY_PRELUDES = SLOT_TO_ID["my_preludes_in_hand"]
_SLOT_DEALT_CORPS = SLOT_TO_ID["dealt_corps"]
_SLOT_DEALT_PRELUDES = SLOT_TO_ID["dealt_preludes"]
_SLOT_DEALT_PROJECTS = SLOT_TO_ID["dealt_projects"]
_SLOT_PICKED_CORP = SLOT_TO_ID["picked_corp"]
_SLOT_DRAFTED_CARDS = SLOT_TO_ID["drafted_cards"]
_SLOT_CURRENT_PACKET = SLOT_TO_ID["current_draft_packet"]
_SLOT_HIST_DISCARDED = SLOT_TO_ID["history_discarded_seen"]
_SLOT_HIST_PASSED = SLOT_TO_ID["history_draft_passed"]
_SLOT_OPP_TABLEAU = SLOT_TO_ID["opp_tableau"]

# obs_scalars indices written by the draft batch builder (see encode_observation layout).
_SC_FIRST_PLAYER = 7
_SC_ACTIVE_PLAYER = 8
_SC_OPP_TAGS = 96
_SC_HIST_DRAFTS = 182


def resolve_warmstart_dir(data_dir: Path | str | None = None) -> Path:
    """Resolve the read-only warm-start parquet directory."""
    if data_dir is not None:
        return Path(data_dir)
    repo_local = Path(__file__).resolve().parents[2] / "data" / "warmstart"
    if repo_local.exists():
        return repo_local
    return DEFAULT_WARMSTART_DIR


@dataclass
class SetupChoicesTable:
    """Compact array representation of `setup_choices.parquet`."""

    map_id: np.ndarray  # uint8[N]
    seat: np.ndarray  # uint8[N]
    split: np.ndarray  # uint8[N] (0=train, 1=val, 2=test)
    offered_corps: np.ndarray  # uint16[N, 2]
    chosen_corp: np.ndarray  # uint16[N]
    offered_preludes: np.ndarray  # uint16[N, 4]
    chosen_preludes: np.ndarray  # uint16[N, 2]
    offered_projects: np.ndarray  # uint16[N, 10]
    chosen_projects: np.ndarray  # uint16[N, 10]
    chosen_proj_len: np.ndarray  # uint8[N]
    split_indices: dict[str, np.ndarray]

    @property
    def num_rows(self) -> int:
        return int(self.map_id.shape[0])


@dataclass
class DraftPicksTable:
    """Compact array representation of `draft_picks.parquet`."""

    setup_idx: np.ndarray  # int32[M]
    map_id: np.ndarray  # uint8[M]
    seat: np.ndarray  # uint8[M]
    split: np.ndarray  # uint8[M]
    generation: np.ndarray  # uint8[M]
    draft_round_id: np.ndarray  # uint16[M]
    pick_index: np.ndarray  # uint8[M]
    packet_cards: np.ndarray  # uint16[M, 5]
    packet_len: np.ndarray  # uint8[M]
    chosen_card: np.ndarray  # uint16[M]
    drafted_this_round: np.ndarray  # uint16[M, 4]
    drafted_len: np.ndarray  # uint8[M]
    # The actor's own previous draft row (same table and seat; earlier round or earlier pick),
    # or -1. Following the chain reconstructs `history.drafts`.
    prev_row: np.ndarray  # int32[M]
    # Setup row of the opponent at the same table (its corporation and preludes are public
    # after generation 1), or -1 when that perspective did not pass the gates.
    opp_setup_idx: np.ndarray  # int32[M]
    split_indices: dict[str, np.ndarray]

    @property
    def num_rows(self) -> int:
        return int(self.map_id.shape[0])


@dataclass
class SetupOutcomesTable:
    """Compact array representation of `setup_outcomes.parquet`."""

    map_id: np.ndarray  # uint8[K]
    seat: np.ndarray  # uint8[K]
    split: np.ndarray  # uint8[K]
    chosen_corp: np.ndarray  # uint16[K]
    chosen_preludes: np.ndarray  # uint16[K, 2]
    chosen_projects: np.ndarray  # uint16[K, 10]
    chosen_proj_len: np.ndarray  # uint8[K]
    won: np.ndarray  # float32[K]
    split_indices: dict[str, np.ndarray]

    @property
    def num_rows(self) -> int:
        return int(self.map_id.shape[0])


@dataclass
class WarmstartCorpus:
    """All three warm-start tables loaded into compact RAM arrays."""

    vocab: CardVocab
    setup_choices: SetupChoicesTable
    draft_picks: DraftPicksTable
    setup_outcomes: SetupOutcomesTable


def _map_to_id(name: str) -> int:
    norm = normalize_map_name(name)
    return SUPPORTED_MAPS.index(norm)


def _split_dict(split_arr: np.ndarray) -> dict[str, np.ndarray]:
    return {
        name: np.nonzero(split_arr == idx)[0].astype(np.int64)
        for name, idx in _SPLIT_TO_IDX.items()
    }


def _load_setup_choices(
    path: Path, vocab: CardVocab
) -> tuple[SetupChoicesTable, dict[tuple[str, int], int]]:
    table = pq.read_table(
        path,
        columns=[
            "table_id",
            "map",
            "seat",
            "split",
            "offered_corporations",
            "chosen_corporation",
            "offered_preludes",
            "chosen_preludes",
            "offered_projects",
            "chosen_projects",
        ],
    )
    n = table.num_rows
    map_id = np.empty(n, dtype=np.uint8)
    seat = np.empty(n, dtype=np.uint8)
    split = np.empty(n, dtype=np.uint8)
    offered_corps = np.zeros((n, 2), dtype=np.uint16)
    chosen_corp = np.zeros(n, dtype=np.uint16)
    offered_preludes = np.zeros((n, 4), dtype=np.uint16)
    chosen_preludes = np.zeros((n, 2), dtype=np.uint16)
    offered_projects = np.zeros((n, 10), dtype=np.uint16)
    chosen_projects = np.zeros((n, 10), dtype=np.uint16)
    chosen_proj_len = np.zeros(n, dtype=np.uint8)

    cid = vocab.card_id
    tids = table["table_id"].to_pylist()
    maps = table["map"].to_pylist()
    seats = table["seat"].to_pylist()
    splits = table["split"].to_pylist()
    off_c = table["offered_corporations"].to_pylist()
    ch_c = table["chosen_corporation"].to_pylist()
    off_pr = table["offered_preludes"].to_pylist()
    ch_pr = table["chosen_preludes"].to_pylist()
    off_pj = table["offered_projects"].to_pylist()
    ch_pj = table["chosen_projects"].to_pylist()

    map_cache = {m: _map_to_id(m) for m in set(maps)}
    key_to_idx: dict[tuple[str, int], int] = {}

    for i in range(n):
        s_i = int(seats[i])
        key_to_idx[(str(tids[i]), s_i)] = i
        map_id[i] = map_cache[maps[i]]
        seat[i] = s_i
        split[i] = _SPLIT_TO_IDX[splits[i]]

        oc_ids = sorted(cid(x) for x in off_c[i])
        offered_corps[i, : len(oc_ids)] = oc_ids[:2]
        chosen_corp[i] = cid(ch_c[i])

        op_ids = sorted(cid(x) for x in off_pr[i])
        offered_preludes[i, : len(op_ids)] = op_ids[:4]
        cp_ids = sorted(cid(x) for x in ch_pr[i])
        chosen_preludes[i, : len(cp_ids)] = cp_ids[:2]

        oj_ids = sorted(cid(x) for x in off_pj[i])
        offered_projects[i, : len(oj_ids)] = oj_ids[:10]
        cj_ids = sorted(cid(x) for x in ch_pj[i])
        chosen_proj_len[i] = min(10, len(cj_ids))
        if cj_ids:
            chosen_projects[i, : len(cj_ids)] = cj_ids[:10]

    sc = SetupChoicesTable(
        map_id=map_id,
        seat=seat,
        split=split,
        offered_corps=offered_corps,
        chosen_corp=chosen_corp,
        offered_preludes=offered_preludes,
        chosen_preludes=chosen_preludes,
        offered_projects=offered_projects,
        chosen_projects=chosen_projects,
        chosen_proj_len=chosen_proj_len,
        split_indices=_split_dict(split),
    )
    return sc, key_to_idx


def _load_draft_picks(
    path: Path, vocab: CardVocab, key_to_setup_idx: Mapping[tuple[str, int], int]
) -> DraftPicksTable:
    table = pq.read_table(
        path,
        columns=[
            "table_id",
            "map",
            "seat",
            "split",
            "draft_round_id",
            "pick_index",
            "packet_cards",
            "chosen_card",
        ],
    )
    m = table.num_rows
    setup_idx = np.empty(m, dtype=np.int32)
    opp_setup_idx = np.empty(m, dtype=np.int32)
    map_id = np.empty(m, dtype=np.uint8)
    seat = np.empty(m, dtype=np.uint8)
    split = np.empty(m, dtype=np.uint8)
    generation = np.empty(m, dtype=np.uint8)
    draft_round_id = np.empty(m, dtype=np.uint16)
    pick_index = np.empty(m, dtype=np.uint8)
    packet_cards = np.zeros((m, 5), dtype=np.uint16)
    packet_len = np.zeros(m, dtype=np.uint8)
    chosen_card = np.zeros(m, dtype=np.uint16)

    cid = vocab.card_id
    tids = table["table_id"].to_pylist()
    maps = table["map"].to_pylist()
    seats = table["seat"].to_pylist()
    splits = table["split"].to_pylist()
    drids = table["draft_round_id"].to_pylist()
    picks = table["pick_index"].to_pylist()
    pkts = table["packet_cards"].to_pylist()
    chsn = table["chosen_card"].to_pylist()

    map_cache = {mp: _map_to_id(mp) for mp in set(maps)}

    for i in range(m):
        tid = str(tids[i])
        s_i = int(seats[i])
        setup_idx[i] = key_to_setup_idx[(tid, s_i)]
        opp_setup_idx[i] = key_to_setup_idx.get((tid, 1 - s_i), -1)
        map_id[i] = map_cache[maps[i]]
        seat[i] = s_i
        split[i] = _SPLIT_TO_IDX[splits[i]]
        dr = int(drids[i])
        draft_round_id[i] = dr
        generation[i] = max(2, min(255, dr + 1))
        pick_index[i] = int(picks[i])

        pkt_ids = sorted(cid(x) for x in pkts[i])
        plen = min(5, len(pkt_ids))
        packet_len[i] = plen
        if plen > 0:
            packet_cards[i, :plen] = pkt_ids[:plen]
        chosen_card[i] = cid(chsn[i])

    del tids, maps, seats, splits, drids, picks, pkts, chsn, table

    # Reconstruct prior picks within the same (table_id, seat, draft_round_id)
    drafted_this_round = np.zeros((m, 4), dtype=np.uint16)
    drafted_len = np.zeros(m, dtype=np.uint8)
    prev_row = np.full(m, -1, dtype=np.int32)
    if m > 0:
        sort_key = (
            (setup_idx.astype(np.uint64) << 24)
            | (draft_round_id.astype(np.uint64) << 8)
            | pick_index.astype(np.uint64)
        )
        order = np.argsort(sort_key, kind="mergesort")
        s_setup = setup_idx[order]
        s_round = draft_round_id[order]
        s_pick = pick_index[order]
        s_chosen = chosen_card[order]
        s_drafted = np.zeros((m, 4), dtype=np.uint16)
        s_dlen = np.zeros(m, dtype=np.uint8)

        for k in (1, 2, 3, 4):
            if m <= k:
                break
            same = (
                (s_setup[k:] == s_setup[:-k])
                & (s_round[k:] == s_round[:-k])
                & (s_pick[k:] > s_pick[:-k])
            )
            idx_cur = np.nonzero(same)[0] + k
            pos = s_dlen[idx_cur]
            valid_pos = pos < 4
            idx_cur = idx_cur[valid_pos]
            pos = pos[valid_pos]
            s_drafted[idx_cur, pos] = s_chosen[idx_cur - k]
            s_dlen[idx_cur] += 1

        # Sort drafted cards ascending per row (matching sorted observation convention)
        for row_idx in range(m):
            dl = int(s_dlen[row_idx])
            if dl > 1:
                s_drafted[row_idx, :dl].sort()

        drafted_this_round[order] = s_drafted
        drafted_len[order] = s_dlen

        # Sorted by (setup, round, pick): the previous sorted row of the same setup is the
        # actor's previous pick.
        same_setup = s_setup[1:] == s_setup[:-1]
        prev_row[order[1:][same_setup]] = order[:-1][same_setup]

    return DraftPicksTable(
        setup_idx=setup_idx,
        map_id=map_id,
        seat=seat,
        split=split,
        generation=generation,
        draft_round_id=draft_round_id,
        pick_index=pick_index,
        packet_cards=packet_cards,
        packet_len=packet_len,
        chosen_card=chosen_card,
        drafted_this_round=drafted_this_round,
        drafted_len=drafted_len,
        prev_row=prev_row,
        opp_setup_idx=opp_setup_idx,
        split_indices=_split_dict(split),
    )


def _load_setup_outcomes(path: Path, vocab: CardVocab) -> SetupOutcomesTable:
    table = pq.read_table(
        path,
        columns=[
            "map",
            "seat",
            "split",
            "chosen_corporation",
            "chosen_preludes",
            "chosen_projects",
            "won",
        ],
    )
    k = table.num_rows
    map_id = np.empty(k, dtype=np.uint8)
    seat = np.empty(k, dtype=np.uint8)
    split = np.empty(k, dtype=np.uint8)
    chosen_corp = np.zeros(k, dtype=np.uint16)
    chosen_preludes = np.zeros((k, 2), dtype=np.uint16)
    chosen_projects = np.zeros((k, 10), dtype=np.uint16)
    chosen_proj_len = np.zeros(k, dtype=np.uint8)
    won = np.empty(k, dtype=np.float32)

    cid = vocab.card_id
    maps = table["map"].to_pylist()
    seats = table["seat"].to_pylist()
    splits = table["split"].to_pylist()
    ch_c = table["chosen_corporation"].to_pylist()
    ch_pr = table["chosen_preludes"].to_pylist()
    ch_pj = table["chosen_projects"].to_pylist()
    wons = table["won"].to_pylist()
    map_cache = {mp: _map_to_id(mp) for mp in set(maps)}

    for i in range(k):
        map_id[i] = map_cache[maps[i]]
        seat[i] = int(seats[i])
        split[i] = _SPLIT_TO_IDX[splits[i]]
        chosen_corp[i] = cid(ch_c[i])
        cp_ids = sorted(cid(x) for x in ch_pr[i])
        chosen_preludes[i, : len(cp_ids)] = cp_ids[:2]
        cj_ids = sorted(cid(x) for x in ch_pj[i])
        chosen_proj_len[i] = min(10, len(cj_ids))
        if cj_ids:
            chosen_projects[i, : len(cj_ids)] = cj_ids[:10]
        won[i] = 1.0 if bool(wons[i]) else 0.0

    return SetupOutcomesTable(
        map_id=map_id,
        seat=seat,
        split=split,
        chosen_corp=chosen_corp,
        chosen_preludes=chosen_preludes,
        chosen_projects=chosen_projects,
        chosen_proj_len=chosen_proj_len,
        won=won,
        split_indices=_split_dict(split),
    )


def load_warmstart_corpus(
    data_dir: Path | str | None = None,
    *,
    vocab: CardVocab | None = None,
    load_drafts: bool = True,
    load_outcomes: bool = True,
) -> WarmstartCorpus:
    """Load warm-start parquet datasets into compact RAM structures."""
    root = resolve_warmstart_dir(data_dir)
    resolved_vocab = vocab if vocab is not None else CardVocab.from_card_data()

    sc, key_to_idx = _load_setup_choices(root / "setup_choices.parquet", resolved_vocab)
    if load_drafts:
        dp = _load_draft_picks(root / "draft_picks.parquet", resolved_vocab, key_to_idx)
    else:
        empty_split = {name: np.zeros((0,), dtype=np.int64) for name in _SPLIT_TO_IDX}
        dp = DraftPicksTable(
            setup_idx=np.zeros((0,), dtype=np.int32),
            map_id=np.zeros((0,), dtype=np.uint8),
            seat=np.zeros((0,), dtype=np.uint8),
            split=np.zeros((0,), dtype=np.uint8),
            generation=np.zeros((0,), dtype=np.uint8),
            draft_round_id=np.zeros((0,), dtype=np.uint16),
            pick_index=np.zeros((0,), dtype=np.uint8),
            packet_cards=np.zeros((0, 5), dtype=np.uint16),
            packet_len=np.zeros((0,), dtype=np.uint8),
            chosen_card=np.zeros((0,), dtype=np.uint16),
            drafted_this_round=np.zeros((0, 4), dtype=np.uint16),
            drafted_len=np.zeros((0,), dtype=np.uint8),
            prev_row=np.zeros((0,), dtype=np.int32),
            opp_setup_idx=np.zeros((0,), dtype=np.int32),
            split_indices=empty_split,
        )

    if load_outcomes:
        so = _load_setup_outcomes(root / "setup_outcomes.parquet", resolved_vocab)
    else:
        empty_split = {name: np.zeros((0,), dtype=np.int64) for name in _SPLIT_TO_IDX}
        so = SetupOutcomesTable(
            map_id=np.zeros((0,), dtype=np.uint8),
            seat=np.zeros((0,), dtype=np.uint8),
            split=np.zeros((0,), dtype=np.uint8),
            chosen_corp=np.zeros((0,), dtype=np.uint16),
            chosen_preludes=np.zeros((0, 2), dtype=np.uint16),
            chosen_projects=np.zeros((0, 10), dtype=np.uint16),
            chosen_proj_len=np.zeros((0,), dtype=np.uint8),
            won=np.zeros((0,), dtype=np.float32),
            split_indices=empty_split,
        )

    return WarmstartCorpus(
        vocab=resolved_vocab,
        setup_choices=sc,
        draft_picks=dp,
        setup_outcomes=so,
    )


class FastBatchBuilder:
    """Vectorized tensor batch builder matching `warmstart_builder` + `encode_*` exactly."""

    def __init__(
        self,
        vocab: CardVocab,
        cards_by_name: Mapping[str, CardSpec] | None = None,
    ) -> None:
        self.vocab = vocab
        self.specs = cards_by_name if cards_by_name is not None else load_card_data()
        templates = load_map_templates()

        # Pre-compute static space tensors and base obs_scalars for each (map_id, seat)
        self.map_space_feats: list[np.ndarray] = []
        self.base_scalars = np.zeros((len(SUPPORTED_MAPS), 2, OBS_SCALAR_DIM), dtype=np.float32)

        for mid, map_name in enumerate(SUPPORTED_MAPS):
            for s in (0, 1):
                obs, _ = build_corp_choice(
                    map_name,
                    s,
                    offered_corps=("CrediCor", "Ecoline"),
                    offered_preludes=(),
                    offered_projects=(),
                    templates=templates,
                )
                enc = encode_observation(obs, vocab)
                if s == 0:
                    self.map_space_feats.append(enc.space_feats)
                # Zero out dealtCorporationCards count (scalar[174]) in base template
                sc = enc.obs_scalars.copy()
                sc[174] = 0.0
                self.base_scalars[mid, s] = sc

        self.max_spaces = max(arr.shape[0] for arr in self.map_space_feats)

        # Pre-compute per-card cost, startingMC, cardCost, and tag vector indexed by vocab card_id
        vsize = vocab.size
        self.card_cost_arr = np.zeros(vsize, dtype=np.float32)
        self.corp_starting_mc = np.full(vsize, 40.0, dtype=np.float32)
        self.corp_buy_cost = np.full(vsize, 3.0, dtype=np.float32)
        self.card_tag_vec = np.zeros((vsize, len(TAG_NAMES)), dtype=np.float32)

        for idx, name in enumerate(vocab.names):
            spec = self.specs.get(name)
            if spec is None:
                continue
            self.card_cost_arr[idx] = float(spec.cost)
            self.corp_starting_mc[idx] = float(spec.starting_megacredits)
            self.corp_buy_cost[idx] = float(spec.card_cost) if spec.card_cost is not None else 3.0
            for t in spec.tags:
                ti = _TAG_TO_IDX.get(t)
                if ti is not None:
                    self.card_tag_vec[idx, ti] += 1.0

        # Also validate vocab card IDs are in strictly alphabetical order for all non-special cards
        self._non_special_sorted = all(
            vocab.names[i] < vocab.names[i + 1] for i in range(3, vocab.size - 1)
        )

    def _collate_spaces(self, map_ids: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = int(map_ids.shape[0])
        space_feats = np.zeros((bsz, self.max_spaces, SPACE_FEAT_DIM), dtype=np.float32)
        space_mask = np.zeros((bsz, self.max_spaces), dtype=bool)
        for b in range(bsz):
            sp = self.map_space_feats[int(map_ids[b])]
            n = sp.shape[0]
            space_feats[b, :n] = sp
            space_mask[b, :n] = True
        return torch.from_numpy(space_feats), torch.from_numpy(space_mask)

    def _pack_card_tokens(
        self, token_lists: list[list[tuple[int, int, float]]]
    ) -> dict[str, torch.Tensor]:
        """Canonically sort `(slot_id, card_id, dyn0)` lists and pad into batch tensors."""
        bsz = len(token_lists)
        max_cards = max((len(t) for t in token_lists), default=0)
        max_cards = max(1, max_cards)
        card_ids = np.zeros((bsz, max_cards), dtype=np.int64)
        card_slots = np.zeros((bsz, max_cards), dtype=np.int64)
        card_dyn = np.zeros((bsz, max_cards, CARD_DYN_DIM), dtype=np.float32)
        card_mask = np.zeros((bsz, max_cards), dtype=bool)
        for b, toks in enumerate(token_lists):
            if not toks:
                continue
            toks.sort()
            n = len(toks)
            for k, (sid, cid, d0) in enumerate(toks):
                card_slots[b, k] = sid
                card_ids[b, k] = cid
                card_dyn[b, k, 0] = d0
            card_mask[b, :n] = True
        return {
            "card_ids": torch.from_numpy(card_ids),
            "card_slots": torch.from_numpy(card_slots),
            "card_dyn": torch.from_numpy(card_dyn),
            "card_mask": torch.from_numpy(card_mask),
        }

    def _sort_card_ids_alpha(self, cids: Sequence[int]) -> list[int]:
        if self._non_special_sorted and all(c >= 3 for c in cids):
            return sorted(int(c) for c in cids)
        names = self.vocab.names
        return sorted((int(c) for c in cids), key=lambda c: names[c])

    def _build_setup_tokens_and_scalars(
        self, sc: SetupChoicesTable, indices: np.ndarray
    ) -> tuple[np.ndarray, list[list[tuple[int, int, float]]]]:
        bsz = int(indices.shape[0])
        mids = sc.map_id[indices]
        seats = sc.seat[indices]
        scalars = self.base_scalars[mids, seats].copy()
        scalars[:, 174] = (sc.offered_corps[indices] != PAD_ID).sum(axis=1).astype(np.float32) / 5.0
        scalars[:, 175] = (sc.offered_preludes[indices] != PAD_ID).sum(axis=1).astype(np.float32) / 5.0
        scalars[:, 176] = (sc.offered_projects[indices] != PAD_ID).sum(axis=1).astype(np.float32) / 10.0

        off_c = sc.offered_corps[indices]
        off_pr = sc.offered_preludes[indices]
        off_pj = sc.offered_projects[indices]

        token_lists: list[list[tuple[int, int, float]]] = []
        for b in range(bsz):
            toks: list[tuple[int, int, float]] = []
            for c in off_c[b]:
                ci = int(c)
                if ci != PAD_ID:
                    toks.append((_SLOT_DEALT_CORPS, ci, 0.0))
            for p in off_pr[b]:
                pi = int(p)
                if pi != PAD_ID:
                    toks.append((_SLOT_DEALT_PRELUDES, pi, 0.0))
            for pj in off_pj[b]:
                pji = int(pj)
                if pji != PAD_ID:
                    toks.append((_SLOT_DEALT_PROJECTS, pji, 0.0))
            token_lists.append(toks)
        return scalars, token_lists

    def build_corp_batch(
        self, sc: SetupChoicesTable, indices: np.ndarray
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Build `(batch_dict, target_index[B])` for `initial_corp`."""
        bsz = int(indices.shape[0])
        scalars, token_lists = self._build_setup_tokens_and_scalars(sc, indices)
        space_feats, space_mask = self._collate_spaces(sc.map_id[indices])

        pid = PROMPT_TYPE_TO_ID["initial_corp"]
        ctx = np.zeros((bsz, CHOICE_CTX_DIM), dtype=np.float32)
        ctx[:, pid] = 1.0
        o = len(PROMPT_TYPES)
        ctx[:, o + 0] = 2.0 / 30.0
        ctx[:, o + 5] = 1.0

        cand_ids = np.zeros((bsz, 2), dtype=np.int64)
        cand_feats = np.zeros((bsz, 2, CAND_FEAT_DIM), dtype=np.float32)
        cand_mask = np.ones((bsz, 2), dtype=bool)
        targets = np.zeros((bsz,), dtype=np.int64)

        off_c = sc.offered_corps[indices]
        ch_c = sc.chosen_corp[indices]
        for b in range(bsz):
            c_sorted = self._sort_card_ids_alpha(off_c[b])
            cand_ids[b, 0] = c_sorted[0]
            cand_ids[b, 1] = c_sorted[1]
            cand_feats[b, :, 0] = 1.0
            tgt_cid = int(ch_c[b])
            targets[b] = 0 if c_sorted[0] == tgt_cid else 1

        batch: dict[str, torch.Tensor] = {
            "obs_scalars": torch.from_numpy(scalars),
            "space_feats": space_feats,
            "space_mask": space_mask,
            "choice_ctx": torch.from_numpy(ctx),
            "prompt_type_ids": torch.full((bsz,), pid, dtype=torch.int64),
            "cand_card_ids": torch.from_numpy(cand_ids),
            "cand_feats": torch.from_numpy(cand_feats),
            "cand_mask": torch.from_numpy(cand_mask),
            "picked_card_ids": torch.zeros((bsz, 1), dtype=torch.int64),
            "picked_mask": torch.zeros((bsz, 1), dtype=bool),
        }
        batch.update(self._pack_card_tokens(token_lists))
        return batch, torch.from_numpy(targets)

    def build_prelude_batch(
        self,
        sc: SetupChoicesTable,
        indices: np.ndarray,
        *,
        step_choices: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Build `(batch_dict, soft_targets[B, 4])` for `initial_prelude`.

        `step_choices[b]`:
          - `0`: step 0 (0 picked, 4 candidates, 0.5/0.5 target on both chosen preludes)
          - `1`: step 1 with `chosen_preludes[0]` already picked (3 candidates, 1-hot on `[1]`)
          - `2`: step 1 with `chosen_preludes[1]` already picked (3 candidates, 1-hot on `[0]`)
        """
        bsz = int(indices.shape[0])
        if step_choices is None:
            r = rng if rng is not None else np.random.default_rng(0)
            step_choices = r.integers(0, 3, size=bsz)

        scalars, token_lists = self._build_setup_tokens_and_scalars(sc, indices)
        space_feats, space_mask = self._collate_spaces(sc.map_id[indices])

        pid = PROMPT_TYPE_TO_ID["initial_prelude"]
        ctx = np.zeros((bsz, CHOICE_CTX_DIM), dtype=np.float32)
        ctx[:, pid] = 1.0
        o = len(PROMPT_TYPES)
        ctx[:, o + 3] = 1.0  # multi_card
        ctx[:, o + 5] = 1.0  # gen1 research

        cand_ids = np.zeros((bsz, 4), dtype=np.int64)
        cand_feats = np.zeros((bsz, 4, CAND_FEAT_DIM), dtype=np.float32)
        cand_mask = np.zeros((bsz, 4), dtype=bool)
        picked_ids = np.zeros((bsz, 1), dtype=np.int64)
        picked_mask = np.zeros((bsz, 1), dtype=bool)
        soft_targets = np.zeros((bsz, 4), dtype=np.float32)

        off_pr = sc.offered_preludes[indices]
        ch_pr = sc.chosen_preludes[indices]

        for b in range(bsz):
            sc_step = int(step_choices[b])
            all_p = self._sort_card_ids_alpha(off_pr[b])
            c0, c1 = int(ch_pr[b, 0]), int(ch_pr[b, 1])
            if sc_step == 0:
                cands = all_p
                tgts = {c0, c1}
                n_picked = 0
            elif sc_step == 1:
                cands = [x for x in all_p if x != c0]
                tgts = {c1}
                n_picked = 1
                picked_ids[b, 0] = c0
                picked_mask[b, 0] = True
            else:
                cands = [x for x in all_p if x != c1]
                tgts = {c0}
                n_picked = 1
                picked_ids[b, 0] = c1
                picked_mask[b, 0] = True

            nc = len(cands)
            ctx[b, o + 0] = float(nc) / 30.0
            ctx[b, o + 2] = float(n_picked) / 10.0
            weight = 1.0 / float(max(1, len(tgts)))
            for j, cid_val in enumerate(cands):
                cand_ids[b, j] = cid_val
                cand_feats[b, j, 0] = 1.0
                cand_mask[b, j] = True
                if cid_val in tgts:
                    soft_targets[b, j] = weight

        batch: dict[str, torch.Tensor] = {
            "obs_scalars": torch.from_numpy(scalars),
            "space_feats": space_feats,
            "space_mask": space_mask,
            "choice_ctx": torch.from_numpy(ctx),
            "prompt_type_ids": torch.full((bsz,), pid, dtype=torch.int64),
            "cand_card_ids": torch.from_numpy(cand_ids),
            "cand_feats": torch.from_numpy(cand_feats),
            "cand_mask": torch.from_numpy(cand_mask),
            "picked_card_ids": torch.from_numpy(picked_ids),
            "picked_mask": torch.from_numpy(picked_mask),
        }
        batch.update(self._pack_card_tokens(token_lists))
        return batch, torch.from_numpy(soft_targets)

    def build_project_buy_batch(
        self,
        sc: SetupChoicesTable,
        indices: np.ndarray,
        *,
        picked_counts: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Build `(batch_dict, soft_targets[B, 11])` for `initial_project_buy`."""
        bsz = int(indices.shape[0])
        r = rng if rng is not None else np.random.default_rng(0)
        scalars, token_lists = self._build_setup_tokens_and_scalars(sc, indices)
        space_feats, space_mask = self._collate_spaces(sc.map_id[indices])

        pid = PROMPT_TYPE_TO_ID["initial_project_buy"]
        ctx = np.zeros((bsz, CHOICE_CTX_DIM), dtype=np.float32)
        ctx[:, pid] = 1.0
        o = len(PROMPT_TYPES)
        ctx[:, o + 1] = 1.0  # has_done
        ctx[:, o + 3] = 1.0  # multi_card
        ctx[:, o + 5] = 1.0  # gen1 research

        cand_ids = np.zeros((bsz, 11), dtype=np.int64)
        cand_feats = np.zeros((bsz, 11, CAND_FEAT_DIM), dtype=np.float32)
        cand_mask = np.zeros((bsz, 11), dtype=bool)
        soft_targets = np.zeros((bsz, 11), dtype=np.float32)

        off_pj = sc.offered_projects[indices]
        ch_pj = sc.chosen_projects[indices]
        ch_len = sc.chosen_proj_len[indices]

        picked_lists: list[list[int]] = []
        max_picked = 1
        for b in range(bsz):
            k_tot = int(ch_len[b])
            if picked_counts is not None:
                k_cur = min(k_tot, max(0, int(picked_counts[b])))
            else:
                k_cur = int(r.integers(0, k_tot + 1))

            all_pj = self._sort_card_ids_alpha(off_pj[b])
            chosen_list = [int(ch_pj[b, i]) for i in range(k_tot)]
            if k_cur > 0 and k_cur < k_tot:
                perm = r.permutation(k_tot)[:k_cur]
                picked = self._sort_card_ids_alpha([chosen_list[i] for i in perm])
            elif k_cur == k_tot:
                picked = self._sort_card_ids_alpha(chosen_list)
            else:
                picked = []

            picked_lists.append(picked)
            max_picked = max(max_picked, len(picked))

            picked_set = set(picked)
            remaining = [x for x in all_pj if x not in picked_set]
            nc = len(remaining) + 1  # +1 for done
            ctx[b, o + 0] = float(nc) / 30.0
            ctx[b, o + 2] = float(len(picked)) / 10.0

            for j, cid_val in enumerate(remaining):
                cand_ids[b, j] = cid_val
                cand_feats[b, j, 0] = 1.0
                cand_mask[b, j] = True

            done_idx = len(remaining)
            cand_ids[b, done_idx] = DONE_ID
            cand_feats[b, done_idx, 1] = 1.0
            cand_mask[b, done_idx] = True

            if k_cur == k_tot:
                soft_targets[b, done_idx] = 1.0
            else:
                unpicked_chosen = set(chosen_list) - picked_set
                w = 1.0 / float(max(1, len(unpicked_chosen)))
                for j, cid_val in enumerate(remaining):
                    if cid_val in unpicked_chosen:
                        soft_targets[b, j] = w

        picked_ids = np.zeros((bsz, max_picked), dtype=np.int64)
        picked_mask = np.zeros((bsz, max_picked), dtype=bool)
        for b, pk in enumerate(picked_lists):
            if pk:
                picked_ids[b, : len(pk)] = pk
                picked_mask[b, : len(pk)] = True

        batch: dict[str, torch.Tensor] = {
            "obs_scalars": torch.from_numpy(scalars),
            "space_feats": space_feats,
            "space_mask": space_mask,
            "choice_ctx": torch.from_numpy(ctx),
            "prompt_type_ids": torch.full((bsz,), pid, dtype=torch.int64),
            "cand_card_ids": torch.from_numpy(cand_ids),
            "cand_feats": torch.from_numpy(cand_feats),
            "cand_mask": torch.from_numpy(cand_mask),
            "picked_card_ids": torch.from_numpy(picked_ids),
            "picked_mask": torch.from_numpy(picked_mask),
        }
        batch.update(self._pack_card_tokens(token_lists))
        return batch, torch.from_numpy(soft_targets)

    def build_draft_batch(
        self,
        dp: DraftPicksTable,
        sc: SetupChoicesTable,
        indices: np.ndarray,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Build `(batch_dict, target_index[B])` for `draft_pick`."""
        bsz = int(indices.shape[0])
        mids = dp.map_id[indices]
        seats = dp.seat[indices]
        s_idx = dp.setup_idx[indices]
        gens = dp.generation[indices]

        scalars = self.base_scalars[mids, seats].copy()
        space_feats, space_mask = self._collate_spaces(mids)

        ch_corp = sc.chosen_corp[s_idx]
        ch_prel = sc.chosen_preludes[s_idx]
        ch_proj = sc.chosen_projects[s_idx]
        ch_proj_len = sc.chosen_proj_len[s_idx]
        off_c = sc.offered_corps[s_idx]
        off_pr = sc.offered_preludes[s_idx]
        off_pj = sc.offered_projects[s_idx]

        pkts = dp.packet_cards[indices]
        pkt_lens = dp.packet_len[indices]
        ch_card = dp.chosen_card[indices]
        drafted = dp.drafted_this_round[indices]
        drafted_lens = dp.drafted_len[indices]

        # Vectorized scalar updates for generation, phase=drafting, megacredits, cardCost, tags, counts
        scalars[:, 11] = gens.astype(np.float32) / 20.0
        scalars[:, 12] = 0.0  # gen != 1
        scalars[:, 13 + 0] = 0.0  # not research
        scalars[:, 13 + 2] = 1.0  # phase == drafting
        scalars[:, 21] = 0.0  # not (gen==1 and research)
        # First player rotates each generation (seat 0 starts gen 1) and is reported as active.
        is_first = ((gens.astype(np.int64) - 1) % 2 == seats.astype(np.int64)).astype(np.float32)
        scalars[:, _SC_FIRST_PLAYER] = is_first
        scalars[:, _SC_ACTIVE_PLAYER] = is_first

        card_cost = self.corp_buy_cost[ch_corp]
        start_mc = self.corp_starting_mc[ch_corp]
        mc = np.maximum(0.0, start_mc - ch_proj_len.astype(np.float32) * card_cost)
        scalars[:, 34] = mc / 100.0
        scalars[:, 48] = card_cost / 5.0
        scalars[:, 50] = ch_proj_len.astype(np.float32) / 20.0

        tags_sum = (
            self.card_tag_vec[ch_corp]
            + self.card_tag_vec[ch_prel[:, 0]]
            + self.card_tag_vec[ch_prel[:, 1]]
        )
        scalars[:, 56 : 56 + len(TAG_NAMES)] = tags_sum / 10.0

        scalars[:, 173] = ch_proj_len.astype(np.float32) / 20.0  # cardsInHand
        scalars[:, 174] = (off_c != PAD_ID).sum(axis=1).astype(np.float32) / 5.0
        scalars[:, 175] = (off_pr != PAD_ID).sum(axis=1).astype(np.float32) / 5.0
        scalars[:, 176] = (off_pj != PAD_ID).sum(axis=1).astype(np.float32) / 10.0
        scalars[:, 177] = 1.0  # pickedCorporationCard != None
        scalars[:, 178] = 0.0
        scalars[:, 179] = drafted_lens.astype(np.float32) / 4.0
        scalars[:, 180] = pkt_lens.astype(np.float32) / 4.0
        scalars[:, 184] = 1.0  # initialBuy is not None

        pid = PROMPT_TYPE_TO_ID["draft_pick"]
        ctx = np.zeros((bsz, CHOICE_CTX_DIM), dtype=np.float32)
        ctx[:, pid] = 1.0
        o = len(PROMPT_TYPES)
        ctx[:, o + 0] = pkt_lens.astype(np.float32) / 30.0
        ctx[:, o + 4] = np.maximum(0.0, 4.0 - pkt_lens.astype(np.float32)) / 4.0

        max_cands = int(max(1, pkt_lens.max())) if bsz > 0 else 1
        cand_ids = np.zeros((bsz, max_cands), dtype=np.int64)
        cand_feats = np.zeros((bsz, max_cands, CAND_FEAT_DIM), dtype=np.float32)
        cand_mask = np.zeros((bsz, max_cands), dtype=bool)
        targets = np.zeros((bsz,), dtype=np.int64)

        token_lists: list[list[tuple[int, int, float]]] = []
        for b in range(bsz):
            toks: list[tuple[int, int, float]] = []
            corp_id = int(ch_corp[b])
            p0_id = int(ch_prel[b, 0])
            p1_id = int(ch_prel[b, 1])

            # Tableau: corp + 2 preludes
            toks.append((_SLOT_MY_TABLEAU, corp_id, 0.0))
            if p0_id != PAD_ID:
                toks.append((_SLOT_MY_TABLEAU, p0_id, 0.0))
            if p1_id != PAD_ID:
                toks.append((_SLOT_MY_TABLEAU, p1_id, 0.0))
            toks.append((_SLOT_PICKED_CORP, corp_id, 0.0))

            # Hand projects + kept set
            n_pj = int(ch_proj_len[b])
            kept_set: set[int] = set()
            for k in range(n_pj):
                cid_val = int(ch_proj[b, k])
                kept_set.add(cid_val)
                toks.append((_SLOT_MY_HAND, cid_val, self.card_cost_arr[cid_val] / 50.0))

            # Dealt setup cards
            for cid_val in off_c[b]:
                if cid_val != PAD_ID:
                    toks.append((_SLOT_DEALT_CORPS, int(cid_val), 0.0))
            for cid_val in off_pr[b]:
                if cid_val != PAD_ID:
                    toks.append((_SLOT_DEALT_PRELUDES, int(cid_val), 0.0))
            for cid_val in off_pj[b]:
                if cid_val != PAD_ID:
                    toks.append((_SLOT_DEALT_PROJECTS, int(cid_val), 0.0))

            # Drafted this round + current packet
            dl = int(drafted_lens[b])
            for k in range(dl):
                cid_val = int(drafted[b, k])
                if cid_val != PAD_ID:
                    toks.append((_SLOT_DRAFTED_CARDS, cid_val, 0.0))

            pl = int(pkt_lens[b])
            pkt_sorted = self._sort_card_ids_alpha([int(pkts[b, k]) for k in range(pl)])
            tgt_cid = int(ch_card[b])
            for j, cid_val in enumerate(pkt_sorted):
                toks.append((_SLOT_CURRENT_PACKET, cid_val, 0.0))
                cand_ids[b, j] = cid_val
                cand_feats[b, j, 0] = 1.0
                if cid_val in kept_set:
                    cand_feats[b, j, 2] = self.card_cost_arr[cid_val] / 50.0
                    cand_feats[b, j, 3] = 1.0
                cand_mask[b, j] = True
                if cid_val == tgt_cid:
                    targets[b] = j

            # Own draft history: cards passed at every earlier pick
            passed_counts: Counter[int] = Counter()
            n_prior = 0
            k_row = int(dp.prev_row[indices[b]])
            while k_row >= 0:
                n_prior += 1
                taken = int(dp.chosen_card[k_row])
                for j in range(int(dp.packet_len[k_row])):
                    pc = int(dp.packet_cards[k_row, j])
                    if pc != taken:
                        passed_counts[pc] += 1
                k_row = int(dp.prev_row[k_row])
            for cid_val, cnt in passed_counts.items():
                toks.append((_SLOT_HIST_PASSED, cid_val, float(cnt) / 3.0))
            scalars[b, _SC_HIST_DRAFTS] = float(n_prior) / 40.0

            # Opponent's public setup (corporation + preludes)
            opp_i = int(dp.opp_setup_idx[indices[b]])
            if opp_i >= 0:
                opp_cards = [int(sc.chosen_corp[opp_i])] + [
                    int(x) for x in sc.chosen_preludes[opp_i] if int(x) != PAD_ID
                ]
                opp_tags = np.zeros(len(TAG_NAMES), dtype=np.float32)
                for oc in opp_cards:
                    toks.append((_SLOT_OPP_TABLEAU, oc, 0.0))
                    opp_tags += self.card_tag_vec[oc]
                scalars[b, _SC_OPP_TAGS : _SC_OPP_TAGS + len(TAG_NAMES)] = opp_tags / 10.0

            # Initial buy discarded
            disc_counts: Counter[int] = Counter()
            for cid_val in off_c[b]:
                ci = int(cid_val)
                if ci != PAD_ID and ci != corp_id:
                    disc_counts[ci] += 1
            prel_kept = {p0_id, p1_id}
            for cid_val in off_pr[b]:
                pi = int(cid_val)
                if pi != PAD_ID and pi not in prel_kept:
                    disc_counts[pi] += 1
            for cid_val in off_pj[b]:
                pji = int(cid_val)
                if pji != PAD_ID and pji not in kept_set:
                    disc_counts[pji] += 1
            for cid_val, cnt in disc_counts.items():
                toks.append((_SLOT_HIST_DISCARDED, cid_val, float(cnt) / 3.0))

            token_lists.append(toks)

        batch: dict[str, torch.Tensor] = {
            "obs_scalars": torch.from_numpy(scalars),
            "space_feats": space_feats,
            "space_mask": space_mask,
            "choice_ctx": torch.from_numpy(ctx),
            "prompt_type_ids": torch.full((bsz,), pid, dtype=torch.int64),
            "cand_card_ids": torch.from_numpy(cand_ids),
            "cand_feats": torch.from_numpy(cand_feats),
            "cand_mask": torch.from_numpy(cand_mask),
            "picked_card_ids": torch.zeros((bsz, 1), dtype=torch.int64),
            "picked_mask": torch.zeros((bsz, 1), dtype=bool),
        }
        batch.update(self._pack_card_tokens(token_lists))
        return batch, torch.from_numpy(targets)

    def build_outcome_batch(
        self, so: SetupOutcomesTable, indices: np.ndarray
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Build `(obs_batch_dict, won_float32[B])` for `setup_outcomes`."""
        bsz = int(indices.shape[0])
        mids = so.map_id[indices]
        seats = so.seat[indices]
        scalars = self.base_scalars[mids, seats].copy()
        space_feats, space_mask = self._collate_spaces(mids)

        ch_corp = so.chosen_corp[indices]
        ch_prel = so.chosen_preludes[indices]
        ch_proj = so.chosen_projects[indices]
        ch_proj_len = so.chosen_proj_len[indices]

        scalars[:, 13 + 0] = 0.0  # not research
        scalars[:, 13 + 3] = 1.0  # phase == preludes
        scalars[:, 21] = 0.0

        card_cost = self.corp_buy_cost[ch_corp]
        start_mc = self.corp_starting_mc[ch_corp]
        mc = np.maximum(0.0, start_mc - ch_proj_len.astype(np.float32) * card_cost)
        scalars[:, 34] = mc / 100.0
        scalars[:, 48] = card_cost / 5.0
        scalars[:, 50] = ch_proj_len.astype(np.float32) / 20.0
        scalars[:, 56 : 56 + len(TAG_NAMES)] = self.card_tag_vec[ch_corp] / 10.0

        scalars[:, 173] = ch_proj_len.astype(np.float32) / 20.0
        scalars[:, 174] = 0.0
        scalars[:, 175] = 0.0
        scalars[:, 176] = 0.0
        scalars[:, 177] = 1.0  # pickedCorporationCard != None
        scalars[:, 178] = (ch_prel != PAD_ID).sum(axis=1).astype(np.float32) / 4.0

        token_lists: list[list[tuple[int, int, float]]] = []
        for b in range(bsz):
            toks: list[tuple[int, int, float]] = []
            corp_id = int(ch_corp[b])
            toks.append((_SLOT_MY_TABLEAU, corp_id, 0.0))
            toks.append((_SLOT_PICKED_CORP, corp_id, 0.0))
            for k in range(2):
                pi = int(ch_prel[b, k])
                if pi != PAD_ID:
                    toks.append((_SLOT_MY_PRELUDES, pi, 0.0))
            for k in range(int(ch_proj_len[b])):
                pji = int(ch_proj[b, k])
                toks.append((_SLOT_MY_HAND, pji, self.card_cost_arr[pji] / 50.0))
            token_lists.append(toks)

        batch: dict[str, torch.Tensor] = {
            "obs_scalars": torch.from_numpy(scalars),
            "space_feats": space_feats,
            "space_mask": space_mask,
        }
        batch.update(self._pack_card_tokens(token_lists))
        won = torch.from_numpy(so.won[indices].copy())
        return batch, won
