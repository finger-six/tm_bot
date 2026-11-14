"""Frequency baselines and unified evaluation metrics for warm-start tasks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from tm_bot.env.types import SUPPORTED_MAPS
from tm_bot.models.network import TMNetwork
from tm_bot.models.vocab import CardVocab
from tm_bot.training.dataset import (
    FastBatchBuilder,
    WarmstartCorpus,
)


@dataclass
class FrequencyBaseline:
    """Train-split card choice frequency baseline."""

    vocab: CardVocab
    corp_counts: Counter[int]
    prelude_counts: Counter[int]
    draft_counts: Counter[int]
    project_keep_counts: Counter[int]
    project_offer_counts: Counter[int]

    @classmethod
    def fit_train(cls, corpus: WarmstartCorpus) -> FrequencyBaseline:
        sc = corpus.setup_choices
        tr_sc = sc.split_indices["train"]
        corp_counts: Counter[int] = Counter()
        prelude_counts: Counter[int] = Counter()
        project_keep_counts: Counter[int] = Counter()
        project_offer_counts: Counter[int] = Counter()

        for idx in tr_sc:
            i = int(idx)
            corp_counts[int(sc.chosen_corp[i])] += 1
            for p in sc.chosen_preludes[i]:
                pi = int(p)
                if pi > 0:
                    prelude_counts[pi] += 1
            for pj in sc.offered_projects[i]:
                pji = int(pj)
                if pji > 0:
                    project_offer_counts[pji] += 1
            for k in range(int(sc.chosen_proj_len[i])):
                project_keep_counts[int(sc.chosen_projects[i, k])] += 1

        dp = corpus.draft_picks
        tr_dp = dp.split_indices["train"]
        draft_counts: Counter[int] = Counter()
        if len(tr_dp) > 0:
            chosen_tr = dp.chosen_card[tr_dp]
            uniq, cnts = np.unique(chosen_tr, return_counts=True)
            for u, c in zip(uniq, cnts):
                draft_counts[int(u)] = int(c)

        return cls(
            vocab=corpus.vocab,
            corp_counts=corp_counts,
            prelude_counts=prelude_counts,
            draft_counts=draft_counts,
            project_keep_counts=project_keep_counts,
            project_offer_counts=project_offer_counts,
        )

    def evaluate_split(self, corpus: WarmstartCorpus, split: str) -> dict[str, Any]:
        """Compute frequency-baseline accuracies overall and per map for `split`."""
        sc = corpus.setup_choices
        sc_idx = sc.split_indices[split]
        names = self.vocab.names

        # Corp & prelude per-map accumulators
        corp_ok = {m: 0 for m in SUPPORTED_MAPS}
        corp_n = {m: 0 for m in SUPPORTED_MAPS}
        prel_top1_ok = {m: 0 for m in SUPPORTED_MAPS}
        prel_pair_ok = {m: 0 for m in SUPPORTED_MAPS}
        prel_step1_ok = {m: 0 for m in SUPPORTED_MAPS}

        for idx in sc_idx:
            i = int(idx)
            m = SUPPORTED_MAPS[int(sc.map_id[i])]
            corp_n[m] += 1

            # Corp
            oc = [int(x) for x in sc.offered_corps[i] if int(x) > 0]
            pred_c = max(oc, key=lambda c: (self.corp_counts[c], -c, names[c]))
            if pred_c == int(sc.chosen_corp[i]):
                corp_ok[m] += 1

            # Prelude
            op = [int(x) for x in sc.offered_preludes[i] if int(x) > 0]
            cp = {int(x) for x in sc.chosen_preludes[i] if int(x) > 0}
            ranked = sorted(op, key=lambda p: (-self.prelude_counts[p], p))
            if ranked[0] in cp:
                prel_top1_ok[m] += 1
            if set(ranked[:2]) == cp:
                prel_pair_ok[m] += 1
            # Conditional step 1 given ranked[0] (or given true pick)
            c0, c1 = int(sc.chosen_preludes[i, 0]), int(sc.chosen_preludes[i, 1])
            rem0 = [p for p in op if p != c0]
            pred1_from_0 = max(rem0, key=lambda p: (self.prelude_counts[p], -p))
            rem1 = [p for p in op if p != c1]
            pred1_from_1 = max(rem1, key=lambda p: (self.prelude_counts[p], -p))
            if pred1_from_0 == c1:
                prel_step1_ok[m] += 1
            if pred1_from_1 == c0:
                prel_step1_ok[m] += 1

        # Draft picks
        dp = corpus.draft_picks
        dp_idx = dp.split_indices[split]
        draft_ok = {m: 0 for m in SUPPORTED_MAPS}
        draft_n = {m: 0 for m in SUPPORTED_MAPS}

        if len(dp_idx) > 0:
            # Vectorized lookup using array of counts
            counts_arr = np.zeros(self.vocab.size, dtype=np.int64)
            for k, v in self.draft_counts.items():
                if 0 <= k < self.vocab.size:
                    counts_arr[k] = v

            pkts = dp.packet_cards[dp_idx]
            plens = dp.packet_len[dp_idx]
            chsn = dp.chosen_card[dp_idx]
            mids = dp.map_id[dp_idx]

            # Score valid packet slots; masked positions get -1
            scores = counts_arr[pkts]
            slot_range = np.arange(pkts.shape[1])[None, :]
            valid_mask = slot_range < plens[:, None]
            scores = np.where(valid_mask, scores, -1)
            best_pos = np.argmax(scores, axis=1)
            pred_cards = pkts[np.arange(len(dp_idx)), best_pos]
            correct = pred_cards == chsn

            for mid, mname in enumerate(SUPPORTED_MAPS):
                mask_m = mids == mid
                draft_n[mname] = int(mask_m.sum())
                draft_ok[mname] = int(correct[mask_m].sum())

        def _agg(ok_dict: dict[str, int], n_dict: dict[str, int], scale: int = 1) -> dict[str, Any]:
            tot_ok = sum(ok_dict.values())
            tot_n = sum(n_dict.values()) * scale
            per_map = {
                m: {
                    "correct": ok_dict[m],
                    "total": n_dict[m] * scale,
                    "acc": (ok_dict[m] / (n_dict[m] * scale)) if n_dict[m] > 0 else 0.0,
                }
                for m in SUPPORTED_MAPS
            }
            return {
                "overall": {
                    "correct": tot_ok,
                    "total": tot_n,
                    "acc": (tot_ok / tot_n) if tot_n > 0 else 0.0,
                },
                "per_map": per_map,
            }

        return {
            "corp": _agg(corp_ok, corp_n),
            "prelude_top1": _agg(prel_top1_ok, corp_n),
            "prelude_pair": _agg(prel_pair_ok, corp_n),
            "prelude_step1_cond": _agg(prel_step1_ok, corp_n, scale=2),
            "draft": _agg(draft_ok, draft_n),
        }


def _acc_entry(correct: int, total: int, loss_sum: float = 0.0) -> dict[str, Any]:
    return {
        "correct": int(correct),
        "total": int(total),
        "acc": (float(correct) / float(total)) if total > 0 else 0.0,
        "loss": (float(loss_sum) / float(total)) if total > 0 else 0.0,
    }


@torch.inference_mode()
def evaluate_model(
    model: TMNetwork,
    corpus: WarmstartCorpus,
    builder: FastBatchBuilder,
    split: str,
    *,
    device: torch.device,
    batch_size: int = 1024,
    use_amp: bool = True,
) -> dict[str, Any]:
    """Evaluate TMNetwork on `split` across corp, prelude, project_buy, draft, and value."""
    was_training = model.training
    model.eval()
    amp_enabled = use_amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32

    sc = corpus.setup_choices
    sc_idx = sc.split_indices[split]

    corp_ok = {m: 0 for m in SUPPORTED_MAPS}
    corp_n = {m: 0 for m in SUPPORTED_MAPS}
    corp_loss = 0.0

    prel_top1_ok = {m: 0 for m in SUPPORTED_MAPS}
    prel_pair_ok = {m: 0 for m in SUPPORTED_MAPS}
    prel_step1_ok = {m: 0 for m in SUPPORTED_MAPS}
    prel_loss = 0.0

    proj_step0_ok = 0
    proj_done_ok = 0
    proj_n = 0

    for start in range(0, len(sc_idx), batch_size):
        b_idx = sc_idx[start : start + batch_size]
        mids = sc.map_id[b_idx]
        bsz = len(b_idx)

        # 1. Corp
        batch_c, tgt_c = builder.build_corp_batch(sc, b_idx)
        batch_c = {k: v.to(device, non_blocking=True) for k, v in batch_c.items()}
        tgt_c = tgt_c.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            out_c = model(batch_c, compute_policy=True, compute_value=False)
        logits_c = out_c["policy_logits"].float()
        corp_loss += float(F.cross_entropy(logits_c, tgt_c, reduction="sum").item())
        pred_c = logits_c.argmax(dim=-1).cpu().numpy()
        tgt_c_np = tgt_c.cpu().numpy()
        for k in range(bsz):
            m = SUPPORTED_MAPS[int(mids[k])]
            corp_n[m] += 1
            if pred_c[k] == tgt_c_np[k]:
                corp_ok[m] += 1

        # 2. Prelude step 0
        zeros_step = np.zeros(bsz, dtype=np.int64)
        batch_p0, soft_p0 = builder.build_prelude_batch(sc, b_idx, step_choices=zeros_step)
        batch_p0_dev = {k: v.to(device, non_blocking=True) for k, v in batch_p0.items()}
        soft_p0_dev = soft_p0.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            out_p0 = model(batch_p0_dev, compute_policy=True, compute_value=False)
        logits_p0 = out_p0["policy_logits"].float()
        logp0 = torch.where(
            batch_p0_dev["cand_mask"],
            F.log_softmax(logits_p0, dim=-1),
            torch.zeros_like(logits_p0),
        )
        prel_loss += float(-(soft_p0_dev * logp0).sum().item())

        pred_p0_idx = logits_p0.argmax(dim=-1).cpu().numpy()
        cand_ids_p0 = batch_p0["cand_card_ids"].numpy()
        ch_prel_b = sc.chosen_preludes[b_idx]

        # Track greedy sequential step 1 for rows whose step-0 pick was in chosen_preludes
        step1_cond_choices = np.empty(bsz, dtype=np.int64)
        step0_in_chosen = np.zeros(bsz, dtype=bool)
        for k in range(bsz):
            picked_cid = int(cand_ids_p0[k, pred_p0_idx[k]])
            c0, c1 = int(ch_prel_b[k, 0]), int(ch_prel_b[k, 1])
            m = SUPPORTED_MAPS[int(mids[k])]
            if picked_cid == c0:
                step0_in_chosen[k] = True
                step1_cond_choices[k] = 1
                prel_top1_ok[m] += 1
            elif picked_cid == c1:
                step0_in_chosen[k] = True
                step1_cond_choices[k] = 2
                prel_top1_ok[m] += 1
            else:
                step1_cond_choices[k] = 1

        # Greedy step 1 conditioned on step 0 pick (when valid)
        batch_p1, soft_p1 = builder.build_prelude_batch(sc, b_idx, step_choices=step1_cond_choices)
        batch_p1_dev = {k: v.to(device, non_blocking=True) for k, v in batch_p1.items()}
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            out_p1 = model(batch_p1_dev, compute_policy=True, compute_value=False)
        logits_p1 = out_p1["policy_logits"].float()
        pred_p1 = logits_p1.argmax(dim=-1).cpu().numpy()
        tgt_p1 = soft_p1.argmax(dim=-1).numpy()
        for k in range(bsz):
            if step0_in_chosen[k] and pred_p1[k] == tgt_p1[k]:
                m = SUPPORTED_MAPS[int(mids[k])]
                prel_pair_ok[m] += 1

        # Also evaluate conditional step 1 over both ground-truth first picks
        for sc_step in (1, 2):
            step_arr = np.full(bsz, sc_step, dtype=np.int64)
            bp_s, sp_s = builder.build_prelude_batch(sc, b_idx, step_choices=step_arr)
            bp_s_dev = {k: v.to(device, non_blocking=True) for k, v in bp_s.items()}
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                op_s = model(bp_s_dev, compute_policy=True, compute_value=False)
            pred_s = op_s["policy_logits"].float().argmax(dim=-1).cpu().numpy()
            tgt_s = sp_s.argmax(dim=-1).numpy()
            for k in range(bsz):
                if pred_s[k] == tgt_s[k]:
                    m = SUPPORTED_MAPS[int(mids[k])]
                    prel_step1_ok[m] += 1

        # 3. Project buy (step 0 & final done)
        zeros_cnt = np.zeros(bsz, dtype=np.int64)
        batch_pj0, soft_pj0 = builder.build_project_buy_batch(
            sc, b_idx, picked_counts=zeros_cnt
        )
        batch_pj0_dev = {k: v.to(device, non_blocking=True) for k, v in batch_pj0.items()}
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            out_pj0 = model(batch_pj0_dev, compute_policy=True, compute_value=False)
        pred_pj0 = out_pj0["policy_logits"].float().argmax(dim=-1).cpu().numpy()
        soft_pj0_np = soft_pj0.numpy()
        for k in range(bsz):
            if soft_pj0_np[k, pred_pj0[k]] > 0.0:
                proj_step0_ok += 1

        all_kept_cnt = sc.chosen_proj_len[b_idx].astype(np.int64)
        batch_pjd, soft_pjd = builder.build_project_buy_batch(
            sc, b_idx, picked_counts=all_kept_cnt
        )
        batch_pjd_dev = {k: v.to(device, non_blocking=True) for k, v in batch_pjd.items()}
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            out_pjd = model(batch_pjd_dev, compute_policy=True, compute_value=False)
        pred_pjd = out_pjd["policy_logits"].float().argmax(dim=-1).cpu().numpy()
        tgt_pjd = soft_pjd.argmax(dim=-1).numpy()
        proj_done_ok += int((pred_pjd == tgt_pjd).sum())
        proj_n += bsz

    # 4. Draft picks
    dp = corpus.draft_picks
    dp_idx = dp.split_indices[split]
    draft_ok = {m: 0 for m in SUPPORTED_MAPS}
    draft_n = {m: 0 for m in SUPPORTED_MAPS}
    draft_loss = 0.0

    for start in range(0, len(dp_idx), batch_size):
        b_idx = dp_idx[start : start + batch_size]
        mids = dp.map_id[b_idx]
        batch_d, tgt_d = builder.build_draft_batch(dp, sc, b_idx)
        batch_d = {k: v.to(device, non_blocking=True) for k, v in batch_d.items()}
        tgt_d = tgt_d.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            out_d = model(batch_d, compute_policy=True, compute_value=False)
        logits_d = out_d["policy_logits"].float()
        draft_loss += float(F.cross_entropy(logits_d, tgt_d, reduction="sum").item())
        correct = (logits_d.argmax(dim=-1) == tgt_d).cpu().numpy()
        for mid, mname in enumerate(SUPPORTED_MAPS):
            mask_m = mids == mid
            cnt_m = int(mask_m.sum())
            if cnt_m > 0:
                draft_n[mname] += cnt_m
                draft_ok[mname] += int(correct[mask_m].sum())

    # 5. Setup outcomes (Value head)
    so = corpus.setup_outcomes
    so_idx = so.split_indices[split]
    val_ok = {m: 0 for m in SUPPORTED_MAPS}
    val_n = {m: 0 for m in SUPPORTED_MAPS}
    val_bce = 0.0
    val_brier = 0.0

    for start in range(0, len(so_idx), batch_size):
        b_idx = so_idx[start : start + batch_size]
        mids = so.map_id[b_idx]
        batch_v, tgt_v = builder.build_outcome_batch(so, b_idx)
        batch_v = {k: v.to(device, non_blocking=True) for k, v in batch_v.items()}
        tgt_v = tgt_v.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            out_v = model(batch_v, compute_policy=False, compute_value=True)
        v_logit = out_v["value_logit"].float()
        v_prob = out_v["win_prob"].float()
        val_bce += float(F.binary_cross_entropy_with_logits(v_logit, tgt_v, reduction="sum").item())
        val_brier += float(((v_prob - tgt_v) ** 2).sum().item())
        correct = ((v_prob >= 0.5) == (tgt_v >= 0.5)).cpu().numpy()
        for mid, mname in enumerate(SUPPORTED_MAPS):
            mask_m = mids == mid
            cnt_m = int(mask_m.sum())
            if cnt_m > 0:
                val_n[mname] += cnt_m
                val_ok[mname] += int(correct[mask_m].sum())

    if was_training:
        model.train()

    def _agg(
        ok_dict: dict[str, int],
        n_dict: dict[str, int],
        loss_sum: float = 0.0,
        scale: int = 1,
    ) -> dict[str, Any]:
        tot_ok = sum(ok_dict.values())
        tot_n = sum(n_dict.values()) * scale
        per_map = {
            m: _acc_entry(ok_dict[m], n_dict[m] * scale) for m in SUPPORTED_MAPS
        }
        return {
            "overall": _acc_entry(tot_ok, tot_n, loss_sum),
            "per_map": per_map,
        }

    tot_so = sum(val_n.values())
    return {
        "corp": _agg(corp_ok, corp_n, corp_loss),
        "prelude_top1": _agg(prel_top1_ok, corp_n, prel_loss),
        "prelude_pair": _agg(prel_pair_ok, corp_n),
        "prelude_step1_cond": _agg(prel_step1_ok, corp_n, scale=2),
        "project_buy": {
            "step0_acc": (proj_step0_ok / proj_n) if proj_n > 0 else 0.0,
            "done_acc": (proj_done_ok / proj_n) if proj_n > 0 else 0.0,
            "total": proj_n,
        },
        "draft": _agg(draft_ok, draft_n, draft_loss),
        "value": {
            "overall": {
                "correct": sum(val_ok.values()),
                "total": tot_so,
                "acc": (sum(val_ok.values()) / tot_so) if tot_so > 0 else 0.0,
                "bce": (val_bce / tot_so) if tot_so > 0 else 0.0,
                "brier": (val_brier / tot_so) if tot_so > 0 else 0.0,
            },
            "per_map": {m: _acc_entry(val_ok[m], val_n[m]) for m in SUPPORTED_MAPS},
        },
    }
