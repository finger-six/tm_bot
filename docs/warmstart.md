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
    - `draft_pick`: `phase="drafting"`, `generation=max(2, draft_round_id + 1)`, `pickedCorporationCard=chosen_corporation`, actor tableau containing corporation + chosen preludes, actor starting MC adjusted for bought initial projects (`max(0, startingMC - len(chosen_projects) * cardCost)`), `private.currentDraftPacket`, `private.draftedCards` reconstructed from earlier picks in the same `(table_id, seat, draft_round_id)`, and `history.initialBuy` discards.
    - `setup_outcome`: post-setup `phase="preludes"`, `generation=1`, `pickedCorporationCard`, `preludeCardsInHand`, and `cardsInHand` for value-head win/loss training.
  - **Privacy / no opponent leakage**: Opponent private data is never read or placed into features; opponent public player fields remain at neutral game-start defaults.
  - **Adapter parity**: Verified in [`tests/test_warmstart.py`](../tests/test_warmstart.py) against live `WorkerClient` observations and choices at corporation, prelude (step 0 & 1), project buy (step 0 & 1), and drafting decision points, and [`FastBatchBuilder`](../tm_bot/training/dataset.py) is asserted to produce bit-for-bit identical tensors to `warmstart_builder` + `encode_*`.

### Fidelity Limits

1. **Draft generation column**: All rows in `draft_picks.parquet` have `generation=1` due to an upstream BGA parser quirk; reconstruction infers `generation = max(2, draft_round_id + 1)`.
2. **Unrecorded mid-game public/private state during drafts**: `draft_picks.parquet` does not record played project cards after setup, board tile placements, global parameters, milestone claims, award fundings, current MC/production/resources, or which previously drafted cards were kept vs discarded during research buy phases. Consequently, draft observations reflect setup tableau/hand + current round draft state with neutral board/opponent/global defaults; this gap will be closed when training on full engine trajectories during self-play (#23).
3. **Setup-outcome horizon**: `setup_outcomes.parquet` predicts game outcome (`won`) purely from post-setup cards (`chosen_corporation`, `chosen_preludes`, `chosen_projects`), `map`, and `seat`, which caps achievable accuracy (~62%) because post-setup drafts, play, and opponent cards strongly influence wins.
4. **Contributor overlap across splits**: As established in #21, the same BGA corpus contributors appear in `train`, `val`, and `test`. Therefore, **player-holdout evaluation is `N/A`**, and test accuracies reflect unseen games/tables rather than unseen human players.

## 3. Training Setup

- **Corpus**: `train` / `val` / `test` splits from `C:\2026\tm_bot\data\warmstart`:
  - `setup_choices`: `100,656` train / `5,394` val / `5,726` test
  - `draft_picks`: `2,839,121` train / `152,719` val / `161,550` test
  - `setup_outcomes`: `47,846` train / `2,662` val / `2,716` test (50.0% win rate)
- **Optimization**: `TMNetwork` (`4,459,330` params) trained on RTX 4080 Super (`bfloat16` AMP, peak VRAM `< 2.5 GB`, corpus RAM `< 200 MB`) for 4 epochs (`draft_batch_size=1536`, `setup_batch_size=256`, `value_batch_size=128` every 2 steps, AdamW `lr=5e-4 -> 3e-5` cosine schedule, `weight_decay=2e-4`, grad clip `1.0`).
- **Checkpoint selection**: Selected strictly on `val` mean top-1 accuracy across corporation, prelude step-0, and draft pick (**Epoch 2**, `val_score = 0.7488`); evaluated on `test` **once** at the end.
- **Saved artifacts** (gitignored):
  - Checkpoint: `runs/warmstart/best/checkpoint.pt`
  - Config & parameter count: `runs/warmstart/best/config.json`
  - Full metrics JSON: `runs/warmstart/metrics.json`

### Epoch Progression (Validation)

| Epoch | Time (s) | Train Loss | Val Corp | Val Prelude Step-0 | Val Draft | Val Value Acc / BCE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 248.0 | 2.2727 | 0.7271 | 0.8144 | 0.6920 | 0.6161 / 0.6491 |
| **2 (best)** | **247.7** | **2.1552** | **0.7306** | **0.8179** | **0.6979** | **0.6202 / 0.6486** |
| 3 | 306.8 | 2.0713 | 0.7117 | 0.8187 | 0.6998 | 0.6247 / 0.6459 |
| 4 | 305.5 | 1.9498 | 0.6973 | 0.8116 | 0.7010 | 0.6236 / 0.6458 |

## 4. Results vs. Frequency Baseline

Frequency baselines rank offered candidates by their `train`-split selection counts (with deterministic tie-breaking).

### Validation Split Results

| Task | Scope | Model Top-1 | Freq Baseline Top-1 | Delta | Samples |
| --- | --- | ---: | ---: | ---: | ---: |
| Corporation | Overall | **0.7306** | 0.6850 | +0.0456 | 5,394 |
| Corporation | tharsis | 0.7197 | 0.6778 | +0.0419 | 3,988 |
| Corporation | elysium | 0.7593 | 0.6481 | +0.1111 | 54 |
| Corporation | hellas | 0.7622 | 0.7102 | +0.0520 | 1,308 |
| Corporation | vastitas borealis nova | 0.7500 | 0.6364 | +0.1136 | 44 |
| Prelude (step-0 top-1) | Overall | **0.8179** | 0.7840 | +0.0339 | 5,394 |
| Prelude (step-0 top-1) | tharsis | 0.8109 | 0.7778 | +0.0331 | 3,988 |
| Prelude (step-0 top-1) | elysium | 0.8333 | 0.7593 | +0.0741 | 54 |
| Prelude (step-0 top-1) | hellas | 0.8356 | 0.8028 | +0.0329 | 1,308 |
| Prelude (step-0 top-1) | vastitas borealis nova | 0.9091 | 0.8182 | +0.0909 | 44 |
| Prelude (cond step-1) | Overall | **0.6789** | 0.6217 | +0.0572 | 10,788 |
| Prelude (cond step-1) | tharsis | 0.6728 | 0.6180 | +0.0548 | 7,976 |
| Prelude (cond step-1) | elysium | 0.7315 | 0.6204 | +0.1111 | 108 |
| Prelude (cond step-1) | hellas | 0.6938 | 0.6361 | +0.0577 | 2,616 |
| Prelude (cond step-1) | vastitas borealis nova | 0.7273 | 0.5341 | +0.1932 | 88 |
| Prelude (exact pair) | Overall | **0.5267** | 0.4594 | +0.0673 | 5,394 |
| Prelude (exact pair) | tharsis | 0.5208 | 0.4581 | +0.0627 | 3,988 |
| Prelude (exact pair) | elysium | 0.5926 | 0.4815 | +0.1111 | 54 |
| Prelude (exact pair) | hellas | 0.5413 | 0.4694 | +0.0719 | 1,308 |
| Prelude (exact pair) | vastitas borealis nova | 0.5455 | 0.2500 | +0.2955 | 44 |
| Draft pick | Overall | **0.6979** | 0.6293 | +0.0686 | 152,719 |
| Draft pick | tharsis | 0.6954 | 0.6286 | +0.0668 | 112,656 |
| Draft pick | elysium | 0.6909 | 0.6306 | +0.0603 | 1,543 |
| Draft pick | hellas | 0.7059 | 0.6312 | +0.0747 | 37,350 |
| Draft pick | vastitas borealis nova | 0.7009 | 0.6385 | +0.0624 | 1,170 |

- **Project buy**: step-0 accuracy = `0.9509`, done-stop accuracy = `0.8090` (`n=5,394`)
- **Value head (setup outcome)**: BCE = `0.6486`, Brier = `0.2289`, Accuracy = `0.6202` (`n=2,662`)
- **Value accuracy per map**: `tharsis=0.6288`, `elysium=0.5385`, `hellas=0.5990`, `vastitas borealis nova=0.5000`
- **Player-holdout**: `N/A` (corpus contributors appear in all splits)

### Test Split Results (Evaluated Once)

| Task | Scope | Model Top-1 | Freq Baseline Top-1 | Delta | Samples |
| --- | --- | ---: | ---: | ---: | ---: |
| Corporation | Overall | **0.7311** | 0.6876 | +0.0435 | 5,726 |
| Corporation | tharsis | 0.7284 | 0.6818 | +0.0466 | 4,186 |
| Corporation | elysium | 0.7083 | 0.6667 | +0.0417 | 72 |
| Corporation | hellas | 0.7427 | 0.7071 | +0.0356 | 1,434 |
| Corporation | vastitas borealis nova | 0.6176 | 0.6176 | +0.0000 | 34 |
| Prelude (step-0 top-1) | Overall | **0.8144** | 0.7712 | +0.0431 | 5,726 |
| Prelude (step-0 top-1) | tharsis | 0.8048 | 0.7637 | +0.0411 | 4,186 |
| Prelude (step-0 top-1) | elysium | 0.9028 | 0.7778 | +0.1250 | 72 |
| Prelude (step-0 top-1) | hellas | 0.8368 | 0.7929 | +0.0439 | 1,434 |
| Prelude (step-0 top-1) | vastitas borealis nova | 0.8529 | 0.7647 | +0.0882 | 34 |
| Prelude (cond step-1) | Overall | **0.6712** | 0.6113 | +0.0599 | 11,452 |
| Prelude (cond step-1) | tharsis | 0.6634 | 0.6052 | +0.0582 | 8,372 |
| Prelude (cond step-1) | elysium | 0.7500 | 0.6181 | +0.1319 | 144 |
| Prelude (cond step-1) | hellas | 0.6907 | 0.6287 | +0.0621 | 2,868 |
| Prelude (cond step-1) | vastitas borealis nova | 0.6471 | 0.6176 | +0.0294 | 68 |
| Prelude (exact pair) | Overall | **0.5101** | 0.4514 | +0.0587 | 5,726 |
| Prelude (exact pair) | tharsis | 0.4988 | 0.4467 | +0.0521 | 4,186 |
| Prelude (exact pair) | elysium | 0.6111 | 0.4583 | +0.1528 | 72 |
| Prelude (exact pair) | hellas | 0.5391 | 0.4644 | +0.0746 | 1,434 |
| Prelude (exact pair) | vastitas borealis nova | 0.4706 | 0.4706 | +0.0000 | 34 |
| Draft pick | Overall | **0.6962** | 0.6296 | +0.0667 | 161,550 |
| Draft pick | tharsis | 0.6933 | 0.6298 | +0.0635 | 117,666 |
| Draft pick | elysium | 0.6781 | 0.6093 | +0.0688 | 2,035 |
| Draft pick | hellas | 0.7055 | 0.6302 | +0.0752 | 40,907 |
| Draft pick | vastitas borealis nova | 0.6932 | 0.6115 | +0.0817 | 942 |

- **Project buy**: step-0 accuracy = `0.9471`, done-stop accuracy = `0.7979` (`n=5,726`)
- **Value head (setup outcome)**: BCE = `0.6481`, Brier = `0.2286`, Accuracy = `0.6175` (`n=2,716`)
- **Value accuracy per map**: `tharsis=0.6204`, `elysium=0.5909`, `hellas=0.6102`, `vastitas borealis nova=0.5909`
- **Player-holdout**: `N/A` (corpus contributors appear in all splits)
- **Model parameters**: `4,459,330`
- **Training wall-clock**: `1,107.9s` (`18.47 min`, 4 epochs on RTX 4080 Super)
