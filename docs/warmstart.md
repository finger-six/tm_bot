# Observation Encoder and Warm-Start Training

## Overview

Ticket #22 implements the shared neural observation encoder, candidate-scoring policy head, and value head in [`tm_bot/models/`](../tm_bot/models/), together with the human warm-start reconstruction and training pipeline in [`tm_bot/training/`](../tm_bot/training/).

## 1. Shared Observation Encoder & Policy/Value Network

- **Pure Observation Encoding** ([`tm_bot/models/encode.py`](../tm_bot/models/encode.py), [`tm_bot/models/vocab.py`](../tm_bot/models/vocab.py)):
  - `encode_observation(obs, vocab)` is a pure, deterministic function of the adapter's `ActorObservation` (`tm_bot/env/types.py`).
  - Public and private scalar state (`OBS_SCALAR_DIM = 185`) captures map one-hot, actor/turn order, generation/phase, global parameters (`temperature`, `oxygenLevel`, `oceans`, `isTerraformed`, remaining steps), deck/discard pile sizes, actor-relative `[me, opp]` public player scalars and tag counts, canonically sorted milestones and awards, and private/history summary counts.
  - Card tokens across 13 distinct semantic slots (`my_hand`, `my_tableau`, `opp_tableau`, `my_preludes_in_hand`, `dealt_corps`, `dealt_preludes`, `dealt_projects`, `picked_corp`, `drafted_cards`, `current_draft_packet`, `pending_card_selection`, `history_draft_passed`, `history_discarded_seen`) are paired with static card metadata from `data/cards.json` (`CARD_STATIC_DIM = 70`: card type, cost, starting MC, custom card cost, VP summary, production box, starting stock, global/action/discount effects, 17 tags, resource type, requirements) and 4 dynamic per-instance features (`calculatedCost`/count, resources, disabled flag, used-action flag).
  - Card tokens, board spaces (`SPACE_FEAT_DIM = 24`), milestones, and awards are canonically sorted before encoding, and the network uses no positional embeddings on sets, guaranteeing strict order invariance.
- **Model Architecture** ([`tm_bot/models/network.py`](../tm_bot/models/network.py)):
  - `TMNetwork` (`4,459,330` parameters, `d_model = 256`, `n_heads = 8`, `n_obs_layers = 3`, `dim_feedforward = 768`, `policy_ff_dim = 512`):
    - Permutation-invariant board space set encoder (mean + max + multi-head attention pooling).
    - 3-layer Transformer encoder over `[scalar_token, space_summary_token, *card_tokens]` followed by attention pooling to produce `obs_pooled`.
    - **Value head**: MLP predicting game win-probability logit (`win_prob = sigmoid(value_logit)`, `value = 2 * win_prob - 1`).
    - **Candidate-scoring policy head**: encodes `Choice` context (`prompt_type`, `choice_ctx`, pooled `picked_card_ids`), embeds each candidate (shared card embedding + static projection + `cand_feats` covering cards, `done`, spaces, payments, branches, resources, players, amounts), cross-attends to `obs_tokens`, applies 1-layer candidate self-attention (no positional encoding), and outputs masked logits/probabilities where illegal/padded candidates (`~cand_mask`) receive `-inf` logits and exact `0.0` probability.
  - Designed for direct reuse in self-play (#23) via [`PolicyValueAgent`](../tm_bot/models/agent.py) implementing the `Agent` protocol (`reset`, `act`, `predict`, `predict_batch`).

## 2. Warm-Start Decision Reconstruction & Fidelity Limits

- **Reconstructor** ([`tm_bot/training/warmstart_builder.py`](../tm_bot/training/warmstart_builder.py)) + static map templates ([`tm_bot/training/map_templates.json`](../tm_bot/training/map_templates.json)):
  - Warm-start rows from `C:\2026\tm_bot\data\warmstart\{setup_choices,draft_picks,setup_outcomes}.parquet` are reconstructed into exact `ActorObservation` + `Choice` structures matching the Node adapter's `DecisionStepper` and `buildActorObservation`:
    - `initial_corp`: `phase="research"`, `generation=1`, `pickedCorporationCard=None`, sorted `offered_corporations` candidates.
    - `initial_prelude`: `phase="research"`, `generation=1`, `kind="multi_card"`, `prompt="Select 2 Prelude cards"`. Matches the live adapter's `DecisionStepper`: `pickedCorporationCard` remains `None` during initial sub-steps while `choice.candidates` removes already-selected preludes (soft 0.5/0.5 target at step 0, conditional exact target at step 1).
    - `initial_project_buy`: `phase="research"`, `generation=1`, `kind="multi_card"`, `prompt="Select initial cards to buy"`, including `Candidate("done", "Done (k selected)")` so the policy learns both which cards to buy and when to stop.
    - `draft_pick`: `phase="drafting"`, `generation=max(2, draft_round_id + 1)`, `pickedCorporationCard=chosen_corporation`, actor tableau containing corporation + chosen preludes, actor starting MC adjusted for bought initial projects (`max(0, startingMC - len(chosen_projects) * cardCost)`), `private.currentDraftPacket`, `private.draftedCards` reconstructed from earlier picks in the same `(table_id, seat, draft_round_id)`, `history.drafts` (packet/taken/passed) from all of the actor's own earlier picks, the opponent's corporation + preludes in their tableau (public after generation 1; taken from the opponent's own setup row of the same table), `firstPlayer = activePlayer = (generation - 1) % 2` (seat 0 is the recorded starting player), and `history.initialBuy` discards.
    - `setup_outcome`: post-setup `phase="preludes"`, `generation=1`, `pickedCorporationCard`, `preludeCardsInHand`, and `cardsInHand` for value-head win/loss training.
  - **Privacy / no opponent leakage**: Opponent private data (offers, hand, draft packets/picks) is never read or placed into features. The only opponent data used is their chosen corporation and preludes, which are public from generation 1 onward; all other opponent public fields remain at neutral game-start defaults.
  - **Adapter parity** ([`tests/test_warmstart.py`](../tests/test_warmstart.py)):
    - `test_builder_values_match_live_adapter` plays a live game to generation 3: both seats' setup observations (corporation, both prelude steps, every project-buy step) are **tensor-identical** to the adapter's; generation 2-3 draft observations match exactly on every reconstructable card slot (`picked_corp`, `dealt_*`, `drafted_cards`, `current_draft_packet`, `history_draft_passed`) and the turn-order/generation scalars, and the opponent's public setup is a subset of the live opponent tableau.
    - [`FastBatchBuilder`](../tm_bot/training/dataset.py) is asserted tensor-identical to `warmstart_builder` + `encode_*` on synthetic rows (incl. history and opponent setup); a 600-row check on real draft rows with history also matched exactly.

### Fidelity Limits

1. **Draft generation column**: The v1 `draft_picks.parquet` has `generation=1` on every row (#21 parser bug). Since #30 the extractor writes `generation = draft_round_id + 1` (`draft_round_id` runs 1..G-1; there is no gen-1 draft) and quarantines inconsistent rounds before they reach the table: picks outside a draft round (`no_draft_round`), whole rounds with a repeated `pick_index` (`duplicate_pick_key`: two merged rounds or a doubly logged pick), and picks after a failed earlier pick (`missing_earlier_pick`). Each reason is counted under "Dropped Draft Rows by Reason" in `yield_report.md`. The v1 table predates the fix and is regenerated by the v2 extraction (#45); until then reconstruction keeps using `generation = max(2, draft_round_id + 1)`.
2. **Unrecorded mid-game public/private state during drafts**: `draft_picks.parquet` does not record played project cards after setup, board tile placements, global parameters, milestone claims, award fundings, current MC/production/resources, or which previously drafted cards were kept vs discarded during research buy phases. Consequently, draft observations reflect setup tableau/hand, the opponent's public setup, the actor's own draft history and the current round's draft state, with neutral board/global/resource defaults (about 30 of 185 scalars and the actor's played-card/research-discard tokens differ from a live gen-2+ observation); this gap will be closed when training on full engine trajectories during self-play (#23).
3. **Setup-outcome horizon**: `setup_outcomes.parquet` predicts game outcome (`won`) purely from post-setup cards (`chosen_corporation`, `chosen_preludes`, `chosen_projects`), `map`, and `seat`, which caps achievable accuracy (~62%) because post-setup drafts, play, and opponent cards strongly influence wins.
4. **Contributor overlap across splits**: As established in #21, the same BGA corpus contributors appear in `train`, `val`, and `test`. Therefore, **player-holdout evaluation is `N/A`**, and test accuracies reflect unseen games/tables rather than unseen human players.

## 3. Training Setup

- **Corpus**: `train` / `val` / `test` splits from `C:\2026\tm_bot\data\warmstart`:
  - `setup_choices`: `100,656` train / `5,394` val / `5,726` test
  - `draft_picks`: `2,839,121` train / `152,719` val / `161,550` test
  - `setup_outcomes`: `47,846` train / `2,662` val / `2,716` test (50.0% win rate)
- **Optimization**: `TMNetwork` (`4,459,330` params) trained on RTX 4080 Super (`bfloat16` AMP, peak VRAM `< 2.5 GB`, corpus RAM `< 200 MB`) for 4 epochs (`draft_batch_size=1536`, `setup_batch_size=256`, `value_batch_size=128` every 2 steps, AdamW `lr=5e-4 -> 3e-5` cosine schedule, `weight_decay=2e-4`, grad clip `1.0`).
- **Checkpoint selection**: Selected strictly on `val` mean top-1 accuracy across corporation, prelude step-0, and draft pick (**Epoch 2**, `val_score = 0.7468`); evaluated on `test` **once** at the end.
- **Saved artifacts** (gitignored):
  - Checkpoint: `runs/warmstart/best/checkpoint.pt`
  - Config & parameter count: `runs/warmstart/best/config.json`
  - Full metrics JSON: `runs/warmstart/metrics.json`

### Epoch Progression (Validation)

| Epoch | Time (s) | Train Loss | Val Corp | Val Prelude Step-0 | Val Draft | Val Value Acc / BCE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 752.4 | 2.2729 | 0.7236 | 0.8116 | 0.6969 | 0.6153 / 0.6497 |
| **2 (best)** | **826.2** | **2.1480** | **0.7247** | **0.8133** | **0.7024** | **0.6063 / 0.6492** |
| 3 | 847.2 | 2.0648 | 0.7010 | 0.8016 | 0.7052 | 0.6153 / 0.6491 |
| 4 | 869.6 | 1.9446 | 0.6921 | 0.8033 | 0.7059 | 0.6150 / 0.6484 |

## 4. Results vs. Frequency Baseline

Frequency baselines rank offered candidates by their `train`-split selection counts (with deterministic tie-breaking).

Numbers are from the `warmstart-v2` run (draft rows carry draft history, opponent public setup, and corrected first/active player; see Fidelity Limits). The superseded v1 run had val draft top-1 `0.6979` and test draft top-1 `0.6962`; corp and prelude accuracies are within ~0.01 of v1. On test, Vastitas Borealis Nova prelude step-1 and exact-pair accuracy fall slightly below baseline (`n=34` games, within noise).

### Validation Split Results

| Task | Scope | Model Top-1 | Freq Baseline Top-1 | Delta | Samples |
| --- | --- | ---: | ---: | ---: | ---: |
| Corporation | Overall | **0.7247** | 0.6850 | +0.0397 | 5,394 |
| Corporation | tharsis | 0.7156 | 0.6778 | +0.0379 | 3,988 |
| Corporation | elysium | 0.7593 | 0.6481 | +0.1111 | 54 |
| Corporation | hellas | 0.7508 | 0.7102 | +0.0405 | 1,308 |
| Corporation | vastitas borealis nova | 0.7273 | 0.6364 | +0.0909 | 44 |
| Prelude (step-0 top-1) | Overall | **0.8133** | 0.7840 | +0.0293 | 5,394 |
| Prelude (step-0 top-1) | tharsis | 0.8074 | 0.7778 | +0.0296 | 3,988 |
| Prelude (step-0 top-1) | elysium | 0.8704 | 0.7593 | +0.1111 | 54 |
| Prelude (step-0 top-1) | hellas | 0.8257 | 0.8028 | +0.0229 | 1,308 |
| Prelude (step-0 top-1) | vastitas borealis nova | 0.9091 | 0.8182 | +0.0909 | 44 |
| Prelude (cond step-1) | Overall | **0.6733** | 0.6217 | +0.0516 | 10,788 |
| Prelude (cond step-1) | tharsis | 0.6665 | 0.6180 | +0.0485 | 7,976 |
| Prelude (cond step-1) | elysium | 0.7315 | 0.6204 | +0.1111 | 108 |
| Prelude (cond step-1) | hellas | 0.6911 | 0.6361 | +0.0550 | 2,616 |
| Prelude (cond step-1) | vastitas borealis nova | 0.6932 | 0.5341 | +0.1591 | 88 |
| Prelude (exact pair) | Overall | **0.5159** | 0.4594 | +0.0565 | 5,394 |
| Prelude (exact pair) | tharsis | 0.5110 | 0.4581 | +0.0529 | 3,988 |
| Prelude (exact pair) | elysium | 0.5926 | 0.4815 | +0.1111 | 54 |
| Prelude (exact pair) | hellas | 0.5298 | 0.4694 | +0.0604 | 1,308 |
| Prelude (exact pair) | vastitas borealis nova | 0.4545 | 0.2500 | +0.2045 | 44 |
| Draft pick | Overall | **0.7024** | 0.6293 | +0.0731 | 152,719 |
| Draft pick | tharsis | 0.7000 | 0.6286 | +0.0714 | 112,656 |
| Draft pick | elysium | 0.6973 | 0.6306 | +0.0668 | 1,543 |
| Draft pick | hellas | 0.7098 | 0.6312 | +0.0787 | 37,350 |
| Draft pick | vastitas borealis nova | 0.7051 | 0.6385 | +0.0667 | 1,170 |

- **Project buy**: step-0 accuracy = `0.9499`, done-stop accuracy = `0.8059` (`n=5,394`)
- **Value head (setup outcome)**: BCE = `0.6492`, Brier = `0.2293`, Accuracy = `0.6063` (`n=2,662`)
- **Value accuracy per map**: `tharsis=0.6075`, `elysium=0.5385`, `hellas=0.6074`, `vastitas borealis nova=0.5455`
- **Player-holdout**: `N/A` (corpus contributors appear in all splits)

### Test Split Results (Evaluated Once)

| Task | Scope | Model Top-1 | Freq Baseline Top-1 | Delta | Samples |
| --- | --- | ---: | ---: | ---: | ---: |
| Corporation | Overall | **0.7232** | 0.6876 | +0.0356 | 5,726 |
| Corporation | tharsis | 0.7224 | 0.6818 | +0.0406 | 4,186 |
| Corporation | elysium | 0.7639 | 0.6667 | +0.0972 | 72 |
| Corporation | hellas | 0.7252 | 0.7071 | +0.0181 | 1,434 |
| Corporation | vastitas borealis nova | 0.6471 | 0.6176 | +0.0294 | 34 |
| Prelude (step-0 top-1) | Overall | **0.8137** | 0.7712 | +0.0424 | 5,726 |
| Prelude (step-0 top-1) | tharsis | 0.8082 | 0.7637 | +0.0444 | 4,186 |
| Prelude (step-0 top-1) | elysium | 0.8889 | 0.7778 | +0.1111 | 72 |
| Prelude (step-0 top-1) | hellas | 0.8243 | 0.7929 | +0.0314 | 1,434 |
| Prelude (step-0 top-1) | vastitas borealis nova | 0.8824 | 0.7647 | +0.1176 | 34 |
| Prelude (cond step-1) | Overall | **0.6694** | 0.6113 | +0.0581 | 11,452 |
| Prelude (cond step-1) | tharsis | 0.6621 | 0.6052 | +0.0569 | 8,372 |
| Prelude (cond step-1) | elysium | 0.7569 | 0.6181 | +0.1389 | 144 |
| Prelude (cond step-1) | hellas | 0.6879 | 0.6287 | +0.0593 | 2,868 |
| Prelude (cond step-1) | vastitas borealis nova | 0.6029 | 0.6176 | -0.0147 | 68 |
| Prelude (exact pair) | Overall | **0.5117** | 0.4514 | +0.0603 | 5,726 |
| Prelude (exact pair) | tharsis | 0.5026 | 0.4467 | +0.0559 | 4,186 |
| Prelude (exact pair) | elysium | 0.6389 | 0.4583 | +0.1806 | 72 |
| Prelude (exact pair) | hellas | 0.5335 | 0.4644 | +0.0690 | 1,434 |
| Prelude (exact pair) | vastitas borealis nova | 0.4412 | 0.4706 | -0.0294 | 34 |
| Draft pick | Overall | **0.7025** | 0.6296 | +0.0729 | 161,550 |
| Draft pick | tharsis | 0.6994 | 0.6298 | +0.0696 | 117,666 |
| Draft pick | elysium | 0.6791 | 0.6093 | +0.0698 | 2,035 |
| Draft pick | hellas | 0.7127 | 0.6302 | +0.0825 | 40,907 |
| Draft pick | vastitas borealis nova | 0.6932 | 0.6115 | +0.0817 | 942 |

- **Project buy**: step-0 accuracy = `0.9422`, done-stop accuracy = `0.7944` (`n=5,726`)
- **Value head (setup outcome)**: BCE = `0.6496`, Brier = `0.2293`, Accuracy = `0.6189` (`n=2,716`)
- **Value accuracy per map**: `tharsis=0.6209`, `elysium=0.6364`, `hellas=0.6102`, `vastitas borealis nova=0.6364`
- **Player-holdout**: `N/A` (corpus contributors appear in all splits)
- **Model parameters**: `4,459,330`
- **Training wall-clock**: `3,295.4s` (`54.92 min`, 4 epochs on RTX 4080 Super)

## 5. v2 Warm Start (#46)

Both agents' v2 warm starts (`warmstart-v3` for `TMNetwork`, `comparator-warmstart-v2` for `ComparatorNetwork`) run one shared trainer, [`tm_bot/training/multitask.py`](../tm_bot/training/multitask.py), with one `WarmstartConfig` (decisions in #38 §4). The v1 sections above stay as the record of the v1 runs.

```
python -m tm_bot.training.encoded_cache --data-dir data/warmstart          # optional: the trainers build it on demand
python -m tm_bot.training.train_warmstart --data-dir data/warmstart --out-dir runs/warmstart-v3
python -m tm_bot.comparator.warmstart --data-dir data/warmstart --out-dir runs/comparator-warmstart-v2
```

### Tasks, sampling and loss

| Task | Source | Minibatch / step | Loss | Weight |
| --- | --- | ---: | --- | ---: |
| Corporation | `setup_choices` (gen 1) | 256 (one shared encoding) | CE | 0.40 |
| Prelude | `setup_choices` | same rows | soft CE (sequential) | 0.40 |
| Initial project buy | `setup_choices` | same rows | soft CE (sequential, `done`) | 0.30 |
| Draft pick (gen 2+ state) | `draft_picks` | 512 | CE | 0.80 |
| Research buy | `research_buys` | 512 | soft CE (sequential, `done`) | 0.70 |
| Root branch | `action_branches` | 512 | CE | 1.00 |
| Project card play | `action_project_cards` | 512 | CE | 1.00 |
| Standard project | `action_standard_projects` | 256 | CE | 0.40 |
| Card action | `action_card_actions` | 256 | CE | 0.40 |
| Value: win | 512 = 102 `setup_outcomes` + 205 `midgame_values` research + 205 action | 512 | BCE on `win_logit` | 0.80 |
| Value: VP margin | same rows | | SmoothL1 (β 0.2) on `clip(margin, ±60) / 50` | 0.30 |

- The two gen 2+ value kinds are told apart by the `midgame_values.phase` column (`research` at a generation boundary, `action` for the strided action points). Every v2 value row carries `vp_margin`; a row without one (e.g. the v1 `setup_outcomes` backup) gets no margin loss.
- Research buys use the adapter's sequential `multi_card` frame like the initial project buy: each sample picks a uniform number of the bought cards, the target is uniform over the unpicked bought cards, or `done` once all are picked.
- Each task walks its own shuffled train permutation, so the 6M+ action rows can't crowd out setup, research buys or value. Each forward group is back-propagated as soon as it is computed (same gradient as the summed loss, one group's activations alive at a time).
- Budget, identical for both agents: 6 epochs × 2,500 = 15,000 AdamW steps, `lr 5e-4 → 3e-5` cosine after 500 warmup steps, weight decay `2e-4`, grad clip 1.0, bf16 autocast. `tests/test_warmstart_v2.py` asserts both agents' CLI configs are equal and pins every number above.

### Encoded cache (throughput)

The reference builders (`warmstart_builder.build_*`) plus `encode_observation`/`encode_choice` cost 0.42–0.46 ms per gen 2+ row on one core (measured on a 1,000-file sample, 70–97 card tokens per row). A step draws ~3,000 gen 2+ rows, so on-the-fly encoding would be ~1.35 CPU-s per step: ~5.6 CPU-hours per agent, 11 h for both.

Instead [`encoded_cache.py`](../tm_bot/training/encoded_cache.py) encodes every row once (16 worker processes, ~10k rows/s) into flat memory-mapped arrays under `<data-dir>_cache/` (default `data/warmstart_cache/`; ~1.5 KB per row on disk). Both agents share it; a cache is rebuilt automatically when its parquet file, the encoder version or the card vocabulary changes. Gathering a padded batch from it costs ~0.01 ms per row (the whole step's sampling, setup batches included, ~34 ms), so training is GPU-bound. Only small per-row fields sit in RAM; observation scalars, card tokens and candidates are read through `np.memmap`.

Exactness: the cache stores the encoder's own outputs, with the board taken from the per-map template (asserted identical on every row) and candidate features restricted to the columns those prompt kinds can set (asserted zero elsewhere). `tests/test_warmstart_v2.py` checks, for every row of every task of the golden table pair (research buys: every picked subset), that the cached batch equals `collate_encoded_batch(encode_*(build_*(row)))`; a one-off check on 1,500 random real rows per task (10,500 rows) also matched exactly.

### Selection, metrics and calibration

- Checkpoint selection after each 2,500-step epoch on the val split: `0.25 · mean(corp, prelude step-0, draft, research-buy step-0) + 0.50 · mean(branch, project play, standard project, card action) + 0.25 · max(0, 1 − gen 2+ value Brier / 0.25)`. Research-buy accuracy is step-0 accuracy (the top candidate is one of the bought cards); done-stop accuracy is reported alongside.
- Reports give per-task top-1 overall, per map and per generation bucket (`gen 1`, `2–5`, `6–9`, `10+`), and value accuracy / Brier / ECE per bucket, per phase and per map, for both agents.
- The comparator's temperature is fit on gen 2+ val value logits only (`calibration.fit_midgame_temperature`); its report adds the calibrated value table.
- The test split is not evaluated by these runs; it stays reserved for the pre-self-play gate (#47).
- Checkpoints carry `version`, `warmstart_data_version: 2`, `encoder_version`, the model config (whose `encoder_version` the loaders check), the full train config, table sizes, the composite inputs and the Markdown report; `best/report.md` and `metrics.json` sit beside them.
