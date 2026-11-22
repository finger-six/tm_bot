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

Both agents' v2 warm starts (`warmstart-v3` for `TMNetwork`, `comparator-warmstart-v2` for `ComparatorNetwork`) run one shared trainer, [`tm_bot/training/multitask.py`](../tm_bot/training/multitask.py), with one `WarmstartConfig` (decisions in #38 §4). The v1 sections above stay as the record of the v1 runs. Since #62 the trainer also has the `space` task and needs an extraction with `action_spaces.parquet` (#61); its checkpoints are `warmstart-v4` and `comparator-warmstart-v3`.

```
python -m tm_bot.training.encoded_cache --data-dir data/warmstart_space          # optional: the trainers build it on demand
python -m tm_bot.training.train_warmstart --data-dir data/warmstart_space --out-dir runs/warmstart-v4
python -m tm_bot.comparator.warmstart --data-dir data/warmstart_space --out-dir runs/comparator-warmstart-v3
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
| Tile placement (`space`, #62) | `action_spaces` | 256 | CE over the legal spaces | 0.40 |
| Value: win | 512 = 102 `setup_outcomes` + 205 `midgame_values` research + 205 action | 512 | BCE on `win_logit` | 0.80 |
| Value: VP margin | same rows | | SmoothL1 (β 0.2) on `clip(margin, ±60) / 50` | 0.30 |

- The two gen 2+ value kinds are told apart by the `midgame_values.phase` column (`research` at a generation boundary, `action` for the strided action points). Every v2 value row carries `vp_margin`; a row without one (e.g. the v1 `setup_outcomes` backup) gets no margin loss.
- Research buys use the adapter's sequential `multi_card` frame like the initial project buy: each sample picks a uniform number of the bought cards, the target is uniform over the unpicked bought cards, or `done` once all are picked.
- `space` (#62) takes the band #38 gave the other action sub-choices of about 0.5–1.5M train rows (standard project, card action): 256 rows at weight 0.40. It has 1,135,626 train rows, against several million for root branch and project play (512 at 1.00).
- Each task walks its own shuffled train permutation, so the 6M+ action rows can't crowd out setup, research buys or value. Each forward group is back-propagated as soon as it is computed (same gradient as the summed loss, one group's activations alive at a time).
- Budget, identical for both agents: 6 epochs × 2,500 = 15,000 AdamW steps, `lr 5e-4 → 3e-5` cosine after 500 warmup steps, weight decay `2e-4`, grad clip 1.0, bf16 autocast. `tests/test_warmstart_v2.py` asserts both agents' CLI configs are equal and pins every number above.

### Encoded cache (throughput)

The reference builders (`warmstart_builder.build_*`) plus `encode_observation`/`encode_choice` cost 0.42–0.46 ms per gen 2+ row on one core (measured on a 1,000-file sample, 70–97 card tokens per row). A step draws ~3,000 gen 2+ rows, so on-the-fly encoding would be ~1.35 CPU-s per step: ~5.6 CPU-hours per agent, 11 h for both.

Instead [`encoded_cache.py`](../tm_bot/training/encoded_cache.py) encodes every row once (16 worker processes, ~10k rows/s) into flat memory-mapped arrays under `<data-dir>_cache/` (default `data/warmstart_cache/`; ~1.5 KB per row on disk). Both agents share it; a cache is rebuilt automatically when its parquet file, the cache format, the encoder version, the card vocabulary or the code that shapes the encoding (`_code_fingerprint`) changes. Gathering a padded batch from it costs ~0.01 ms per row (the whole step's sampling, setup batches included, ~34 ms), so training is GPU-bound. Only small per-row fields sit in RAM; observation scalars, card tokens and candidates are read through `np.memmap`.

Exactness: the cache stores the encoder's own outputs, with the empty board taken from the per-map template (asserted identical on every row) and a rebuilt board (#61) stored as compact per-space tile/owner codes that `expand_board` writes over the template at batch time (asserted equal to the encoder's spaces on every row), and candidate features restricted to the columns those prompt kinds can set (asserted zero elsewhere). `tests/test_warmstart_v2.py` checks, for every row of every task of the golden table pair (research buys: every picked subset), that the cached batch equals `collate_encoded_batch(encode_*(build_*(row)))`; a one-off check on 1,500 random real rows per task (10,500 rows) also matched exactly.

### Space supervision data (#61)

The extractor rebuilds each supported table's board from the log placements ([`board.py`](../tm_bot/data/board.py), replay and checks in [`logstate.py`](../tm_bot/data/logstate.py)) and emits one `action_spaces` row per gen 2+ tile placement, with a legal-space menu from a port of the engine placement rules ([`spacemenu.py`](../tm_bot/data/spacemenu.py); columns `rule`, `source`, `tile`, `placement_index`). Gen 2+ action-phase rows of every table also carry the rebuilt board as `board_tiles` / `board_owners` while it is valid. `tests/test_space_parity_v2.py` checks the menus and the built prompts against the live adapter on all four maps; `tests/test_board_v2.py` covers coordinates, menus, replay quarantine and the compact board round trip. The space task is cached (`space`) and, since #62, trained (§5 task table).

Full extraction (2026-11-21, `data/warmstart_space`, 4,450 s wall, peak working set 5.4 GiB): 1,261,604 space rows (train 1,135,626 / val 61,262 / test 64,716; Tharsis 919,659, Hellas 323,003, Elysium 11,226, Vastitas Borealis 7,716), menu miss 0.13% (1,730 of 1,288,176; largest rules `greenery` 1,061 and `mining_area` 424). A per-map rerun over every eligible Elysium and Vastitas Borealis file and the first 1,200 Hellas files found menu miss 0.21% / 0.34% / 0.14%. 16,435 of 111,682 seat-boards were quarantined (mostly `check:greeneries`, 15,457: a greenery the snapshot counts but the log never places; Vastitas Borealis quarantines 26% of seat-boards against ~15% elsewhere, all on this check), and 82,525 rows were dropped as `land_claim_unknown` (Land Claim's space is never logged).

Known approximations: city and greenery counts are checked one-sided (VP details lag placements); the archive's Vastitas Borealis uses the Vastitas Borealis Nova layout and costs; only gen 2+ placements emit rows.

### Selection, metrics and calibration

- Checkpoint selection after each 2,500-step epoch on the val split: `0.25 · mean(corp, prelude step-0, draft, research-buy step-0) + 0.50 · mean(branch, project play, standard project, card action, space) + 0.25 · max(0, 1 − gen 2+ value Brier / 0.25)` (`space` since #62; the v3/v2 runs below used the four-task mean). Research-buy accuracy is step-0 accuracy (the top candidate is one of the bought cards); done-stop accuracy is reported alongside.
- Reports give per-task top-1 overall, per map and per generation bucket (`gen 1`, `2–5`, `6–9`, `10+`), and value accuracy / Brier / ECE per bucket, per phase and per map, for both agents.
- Both agents' value temperatures are fit on gen 2+ val value logits only (`calibration.fit_midgame_temperature`; the main agent's since #60), stored in the checkpoint metadata (`temperature`) and `metrics.json` (`value_calibrated`), and the reports add the calibrated value table. The temperature only scales the reported P(win) (`PolicyValueAgent.win_probability`, the gate); the main agent's search leaves keep the raw value head. A main checkpoint trained before #60 gets its fit from `python -m tm_bot.training.value_temperature --checkpoint <run>/best`, written beside it as `checkpoint.value_temperature.json` (with the checkpoint's SHA-256, so it never applies to other weights).
- The test split is not evaluated by these runs, and the pre-self-play gate (#47) uses the val split too (#38 §5).
- Checkpoints carry `version`, `warmstart_data_version: 2`, `encoder_version`, the model config (whose `encoder_version` the loaders check), the full train config, table sizes, the composite inputs and the Markdown report; `best/report.md` and `metrics.json` sit beside them.

### Results (2026-11-21)

Both runs used the same 15,000-step budget on the local GPU from the shared encoded cache (`runs/warmstart-v3`, `runs/comparator-warmstart-v2`; full tables in each run's `best/report.md`). Both selected **epoch 2 (step 5,000)**.

| | Main (`TMNetwork`) | Comparator (`ComparatorNetwork`) |
| --- | ---: | ---: |
| Parameters | 4,492,995 | 2,575,715 |
| Train wall clock | 143.0 min | 38.6 min |
| Composite (best) | 0.6232 | 0.6161 |
| Corporation top-1 | 0.7266 | 0.7296 |
| Prelude step-0 top-1 | 0.8121 | 0.8118 |
| Draft pick top-1 | 0.7243 | 0.7150 |
| Research buy step-0 / done-stop | 0.9505 / 0.8758 | 0.9446 / 0.8569 |
| Root branch top-1 | 0.6597 | 0.6407 |
| Project card play top-1 | 0.5783 | 0.5312 |
| Standard project top-1 | 0.7024 | 0.6796 |
| Card action top-1 | 0.7412 | 0.7271 |
| Initial project buy step-0 / done-stop | 0.9546 / 0.7843 | 0.9555 / 0.7183 |
| Gen 2+ value acc / Brier / ECE (raw) | 0.7579 / 0.1628 / 0.0608 | 0.7631 / 0.1563 / 0.0347 |
| Gen 2+ value Brier / ECE (calibrated) | — | 0.1549 / 0.0128 (T = 1.2603) |
| Gen 2+ VP-margin MAE (normalised) | 0.2360 | 0.2346 |

Value accuracy / Brier by generation bucket (raw):

| Bucket | Main acc | Main Brier | Main ECE | Comparator acc | Comparator Brier | Comparator ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gen 1 | 0.6093 | 0.2293 | 0.0397 | 0.6210 | 0.2271 | 0.0133 |
| gen 2-5 | 0.7112 | 0.1932 | 0.0749 | 0.7125 | 0.1864 | 0.0482 |
| gen 6-9 | 0.7879 | 0.1428 | 0.0551 | 0.7989 | 0.1349 | 0.0291 |
| gen 10+ | 0.8338 | 0.1150 | 0.0305 | 0.8368 | 0.1130 | 0.0246 |

Per map, policy top-1 is within a few points across Tharsis, Elysium, Hellas and Vastitas Borealis (Nova); the gen 2+ value tables on Elysium (n = 553) and Vastitas Borealis (n = 349) are small and noisier (ECE 0.13–0.17 raw), since the outcome-verified mid-game rows are mostly Tharsis and Hellas.

**Value overfitting after epoch 2.** Policy top-1 kept improving to step 15,000 (main root branch 0.6456 → 0.6597 → 0.6813, project play 0.5492 → 0.5783 → 0.6092 at epochs 1 / 2 / 6), but the gen 2+ value Brier rose after epoch 2, sharply for main (0.1604, 0.1628, 0.1981, 0.2264, 0.2368, 0.2414) and mildly for the comparator (0.1659, 0.1563, 0.1620, 0.1656, 0.1716, 0.1731). The composite's value term therefore picked epoch 2 for both. The ~1.09M mid-game value points come from far fewer games (many correlated points per game), so the larger main network memorises game outcomes. The selected checkpoints trade a few points of policy accuracy for an un-overfit value head; self-play's KL anchor and human-data auxiliary loss (#39) start from these epoch-2 weights. If the pre-self-play gate (§6) fails on value-dependent checks, the first lever is a shorter value schedule or lower value weight rather than more steps.

The cache rebuild before training took ~9 min (16 workers). A first attempt hung when its worker pool could no longer spawn processes after the launching session ended; rerun detached, it completed normally.

### Results with the space task (#62, 2026-11-21)

Both agents retrained from scratch on `data/warmstart_space` with the `space` task (same 15,000-step budget and settings; `runs/warmstart-v4`, `runs/comparator-warmstart-v3`; full tables in each run's `best/report.md`). The composite now averages five action tasks, so it is not comparable with the runs above. Space baselines are the #47 gate's frequency baselines (check 9).

| | Main (`warmstart-v4`) | Comparator (`comparator-warmstart-v3`) |
| --- | ---: | ---: |
| Selected epoch (step) | 2 (5,000) | **4 (10,000)** |
| Train wall clock | 157.1 min | 42.4 min |
| Composite (best) | 0.6046 | 0.5999 |
| Space top-1 (frequency baseline 0.2851) | 0.4764 | 0.4613 |
| Space top-1 gen 2–5 / 6–9 / 10+ | 0.5365 / 0.5032 / 0.3986 | 0.5239 / 0.4897 / 0.3797 |
| Root branch / project card play top-1 | 0.6589 / 0.5800 | 0.6556 / 0.5510 |
| Standard project / card action top-1 | 0.7058 / 0.7399 | 0.6982 / 0.7316 |
| Draft pick / research buy step-0 top-1 | 0.7254 / 0.9485 | 0.7202 / 0.9471 |
| Gen 2+ value acc / Brier / ECE (raw) | 0.7547 / 0.1625 / 0.0503 | 0.7597 / 0.1611 / 0.0524 |
| Gen 2+ value Brier / ECE (calibrated) | 0.1601 / 0.0376 (T = 1.3556) | 0.1578 / 0.0179 (T = 1.4890) |
| Gen 2+ VP-margin MAE (normalised) | 0.2412 | 0.2344 |

- Space accuracy is lowest on the small Elysium and Vastitas Borealis (Nova) val sets (0.36–0.38 for both agents, against 0.45–0.49 on Tharsis and Hellas) and in gen 10+.
- No other task got worse. Main is within 0.4 points of `warmstart-v3` on every task. The comparator is up to 2 points above `comparator-warmstart-v2` (root branch, project play and standard project), mostly because a later epoch was selected.
- Main repeats the value overfit above: gen 2+ val Brier 0.1530, 0.1625, 0.1985, 0.2262, 0.2373, 0.2405 over epochs 1–6, so epoch 2 is selected again (space top-1 keeps rising to 0.5257 at epoch 6).
- The comparator's value Brier rises only mildly (0.1562, 0.1556, 0.1581, 0.1611, 0.1665, 0.1706), so its space and action-task gains carry the composite to epoch 4.
- Main's post-temperature ECE (0.0376) clears the gate's 0.040 bar by a thin margin; a single temperature does less for `warmstart-v4` than #58 measured on `warmstart-v3` (0.012).

## 6. Pre-self-play gate (#47)

Both v2 warm starts must pass every #38 §5 check before self-play. The checks and the report are pure functions in [`tm_bot/training/warmstart_gate.py`](../tm_bot/training/warmstart_gate.py); [`run_warmstart_gate.py`](../tm_bot/training/run_warmstart_gate.py) gathers their inputs and writes `<out>.md` plus `<out>.json`.

```
python -m tm_bot.training.run_warmstart_gate --agent main --run-dir runs/warmstart-v4 --data-dir data/warmstart_space --out docs/gate-warmstart-v4.md
python -m tm_bot.training.run_warmstart_gate --agent comparator --run-dir runs/comparator-warmstart-v3 --data-dir data/warmstart_space --out docs/gate-comparator-warmstart-v3.md
```

- **Validation** reads `<run-dir>/metrics.json` (`val`). Both agents' value checks are judged post-temperature (`value_calibrated`; #58 amended check 14 for the main agent), a pre-#60 main warm start through its `checkpoint.value_temperature.json`. Frequency baselines rank each row's candidates by train-split selection counts and are scored on val: corporation and prelude from `setup_choices`, the gen 2+ tasks from the encoded cache (the same rows the model was scored on; ties go to the earlier candidate; research buys score step 0 like the model; a space is its map and grid position, since space candidates have no card id). A policy task passes only if the model beats its baseline overall; per-map numbers are reported, not gated. Since #62 (decision 4 of #59) `space` is one of these tasks, so each agent has 31 gating checks instead of 30: the space baseline is check 9 and every later check moves down one (#58's check 14, the value ECE, is now check 15). Value accuracy must rise strictly over `gen 2-5` → `6-9` → `10+`.
- **Live games** use `dev-*` seeds from `dev-0000` and the eval harness's seat-swapped schedule: 104 games vs `b0-v1`, 40 self-mirror games, 104 vs the archived v1 warm start (`runs/warmstart/best` or `runs/comparator-warmstart-full/best`, same settings as the candidate, except that the v1 main warm start keeps its visit-count root rule), and 104 vs `b1-v2`. The main agent plays light search, #39's 8/4 fallback budget (`c_puct` 1.5, temperature 0, action temperature 1, wave 4) with the completed-Q root policy π′ (#60); its raw-policy diagnostic is the same agent at 0 simulations, 104 more games vs `b1-v2`. The comparator plays its sampled evaluation policy. Failures are counted over every pairing played, the diagnostic one included (`IndexError` = illegal response, `StepCapError` = step cap, anything else = adapter error). Logged, not gated: win rate vs `b1-v2`, search minus raw policy (own VP and margin vs `b1-v2`), and live value Brier / ECE / accuracy of the reported (post-temperature) P(win) per opponent (`b1-v2`, self-mirror, v1 warm start).
- Each finished pairing's games are saved under `--work-dir` (default `<run-dir>/gate/`); `--resume` reuses them. `--skip-validation`, `--skip-live`, `--seeds` and `--maps` are for smoke runs: skipped or shortened parts fail their checks, so such a report never reads PASS.

### Results

| Run | Date | Verdict | Own VP / margin vs `b1-v2` | Win rate vs `b1-v2` | Report |
| --- | --- | --- | ---: | ---: | --- |
| `warmstart-v3` (main) | 2026-11-21 | FAIL 27/30 (value ECE; own VP and margin vs `b1-v2`) | 57.30 / −55.05 | 1.0% [0.0, 2.9] | [`gate-warmstart-v3.md`](gate-warmstart-v3.md) |
| `comparator-warmstart-v2` | 2026-11-21 | PASS 30/30 | 76.92 / −19.34 | 21.2% [11.5, 30.8] | [`gate-comparator-warmstart-v2.md`](gate-comparator-warmstart-v2.md) |
| `warmstart-v4` (main, #62) | 2026-11-22 | **PASS 31/31** | 82.77 / −10.65 | 29.8% [14.4, 47.1] | [`gate-warmstart-v4.md`](gate-warmstart-v4.md) |
| `comparator-warmstart-v3` (#62) | 2026-11-22 | **PASS 31/31** | 80.03 / −13.81 | 26.9% [17.3, 38.5] | [`gate-comparator-warmstart-v3.md`](gate-comparator-warmstart-v3.md) |

- The re-run combines #60 (completed-Q root policy, calibrated value output) and #62 (space supervision). The main agent's light search now beats its raw policy against `b1-v2` (+2.46 own VP, +3.23 margin; raw 80.31 / −13.88), where `warmstart-v3`'s search lost 23 VP to its raw policy (#57).
- Both agents win all 104 games vs `b0-v1` (margin +84.0 main, +82.1 comparator) and all 104 vs their v1 warm starts; self-mirror median length is 13 / 12 generations; there are 0 failures across 456 (main) and 352 (comparator) games.
- Closest calls: main's post-temperature val ECE is 0.0376 against the 0.040 bar. The live value is still miscalibrated against `b1-v2` (post-temperature ECE 0.34 main, 0.22 comparator), which is not gated; #58 leaves it to the `b1-v2` self-play slice.
- With both agents passing, self-play may start with the 1 h gate (#52).
