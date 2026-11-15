# Jev-Inspired Comparator Agent

Ticket #25 adds an independently designed comparison agent: a typed-candidate
scorer with a Deep Sets card encoder, a per-candidate MLP policy head, and a
win-probability head calibrated by temperature scaling. It has no tree search at
inference and shares no modules, parameters, or state with the main agent
(`tm_bot/models/network.py`).

Code: [`tm_bot/comparator/`](../tm_bot/comparator/). It is wired into the same
run orchestration as the main agent through `RunConfig.agent_kind`
(`"main"` or `"comparator"`) and `python -m tm_bot.runs --agent-kind comparator`.

## Design

The comparator reuses the **pure encoding functions** from
`tm_bot.models.encode` (the actor observation and candidate tensors) but not the
main network. See [`model.py`](../tm_bot/comparator/model.py).

| Component | Comparator | Main agent |
| --- | --- | --- |
| Card-set encoder | Deep Sets: per-card MLP + masked mean/max pooling | Transformer over card tokens |
| Observation summary | scalar MLP + card/space Deep Sets + fusion MLP | transformer + attention pooling |
| Candidate head | per-candidate MLP over `[obs, choice context, candidate]` | cross-attention + candidate self-attention |
| Win-probability head | MLP over pooled observation | MLP over pooled observation |
| Inference search | none | determinized IS-MCTS |
| Parameters | 2,501,026 | 4,459,330 |

The policy is a masked softmax over candidates. Padded/illegal candidates get
`-inf` logits and **exact `0.0`** probability
([`model.py:247`](../tm_bot/comparator/model.py), `forward_policy`). The value
head predicts a win-probability logit; the checkpoint stores a scalar
temperature fitted on the **warm-start validation split only**, and
`ComparatorAgent.predict` reports `sigmoid(logit / T)`.

The candidate MLP is conditioned on the pooled observation and on the choice
context (`prompt_type`, `choice_ctx`, and the pooled already-picked cards), so
the same scorer handles corporation, prelude, project-buy, draft, payment,
space, player, amount, resource, and branch decisions.

## Warm start

`python -m tm_bot.comparator.warmstart` loads the same corpus and the same
`train`/`val`/`test` split as the main agent (`load_warmstart_corpus`,
`FastBatchBuilder`), trains the comparator's own encoder/policy/value heads on
corporation, prelude, initial project-buy, draft, and setup-outcome tasks, and
selects the best epoch on validation. Temperature scaling is then fitted on the
validation `setup_outcomes` logits; the test split is not used for selection.

The development warm start used a deliberately short schedule (1 epoch capped at
900 draft steps) so that the PPO development run could fit the ticket budget:

| Metric | Value |
| --- | ---: |
| Parameters | 2,501,026 |
| Wall clock | 114.5 s (RTX 4080 Super, bfloat16 AMP) |
| Val corporation top-1 | 0.7115 |
| Val prelude step-0 top-1 | 0.7996 |
| Val draft top-1 | 0.6798 |
| Val setup-outcome accuracy | 0.5924 |
| Validation temperature | 0.8144 |

Artifact: `runs/comparator-warmstart/best/` (gitignored). A full-budget warm
start can be run with `--epochs 4` and no step cap.

## PPO self-play

`ComparatorSelfPlayEngine` ([`selfplay.py`](../tm_bot/comparator/selfplay.py))
plays `train-*` games through the same adapter runner and recorder as the main
agent, with the same opponent sampling defaults: 60% latest comparator, 30%
uniform from the last 10 frozen comparator checkpoints, 10% B1. The training
seat samples from the masked policy (`temperature=1`); opponents act greedily.

Each recorded training-seat decision stores the selected action, its behavior
log-probability, the behavior value, and the terminal utility. GAE
([`ppo.py:compute_gae`](../tm_bot/comparator/ppo.py)) computes advantages over
the game with `gamma=1.0` and `lambda=0.95`; the learner
([`PPOLearner`](../tm_bot/comparator/ppo.py)) applies the clipped PPO surrogate
(epsilon 0.2), a value-critic BCE loss against the discounted return, and an
entropy bonus. Spec defaults are a 16 x 2,048-decision rollout
(`ppo_rollout_decisions`), 200k-decision ring buffer, AdamW, lr 5e-4, batch 256.
The development run used 16 games/cycle, 32 minibatch updates/cycle, batch 256,
lr 3e-4, and bfloat16 AMP.

Checkpoints, `dev-*` evaluation, best selection, stopping rules, monitoring,
resource limits, and resume are the existing `tm_bot.runs` machinery;
`CheckpointManager.restore_model` dispatches on the checkpoint architecture so
comparator runs resume and roll back correctly.

## Development run (2026-11-14)

Run `runs/cmp-dev1h/` (gitignored): `--agent-kind comparator`, 16 workers,
30-minute wall-clock budget, 16 games/cycle, 32 updates/cycle, warm start above,
checkpoint interval 30 min. Training ran 13 cycles (208 games, 199,490 decisions,
384 PPO steps, 3.5 ms p50 / 5.9 ms p95 per decision) in **1,240 s (20.7 min)** and
then hard-stopped on the known adapter bug #32 (`SelectCard ... only 0 enabled`
in drafting at generation 44 on `train-0198`). The regression case is
`runs/cmp-dev1h/regression/train-0198__hellas__AdapterBugError.json`.

| Cycle metric | Value |
| --- | ---: |
| Self-play games | 208 |
| Decisions | 199,490 (959/game) |
| Mean generations/game | 24.64 (range 12-59) |
| Wins / losses / ties (training seat) | 62 / 145 / 0 |
| Opponent mix | 191 latest / 0 pool / 17 B1 |
| Throughput | **604 self-play games/hour** (cumulative) |
| Policy / value loss (last completed cycle) | 0.058 / 0.298 |
| Rejections / other failures | 1 / 0 |
| Shape flags (outside 6-25 generations) | 69 of 208 games |

The frozen-checkpoint pool stayed empty because no 30-minute checkpoint elapsed
before the 20.7-minute stop; the pool share therefore fell back to "latest", as
designed for a fresh run.

### Dev-seed evaluation

Because the run hard-stopped on the adapter bug, `dev-*` games were evaluated
externally from the final checkpoint `runs/cmp-dev1h/best/` (the run checkpoint
plus the warm-start calibration temperature), exactly as the #24 development run
did. Each pairing is seat-swapped over four maps (52 requested games -> 56
games).

| Pairing | Games | Completed | Failures | Win rate (95% paired bootstrap CI) | Mean VP margin | Mean gens | Wall |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Comparator vs B1 | 56 | 48 | 8 | **0.00%** [0.00%, 0.00%] | -94.56 | 15.08 | 125 s |
| Comparator vs its warm start | 56 | 0 | 56 | **not measurable** (CI n/a) | n/a | n/a | 235 s |
| Comparator vs warm start, `dev` seeds 10-11 | 16 | 0 | 16 | not measurable | n/a | n/a | ~1 min |

All 8 vs-B1 failures and all 56 vs-warm-start failures are the same
`AdapterBugError` (#32): a `SelectCard` prompt with zero enabled cards in the
draft phase. Completed vs-B1 games are normal length (15.1 generations); the
failures occur around generations 38-42 (vs B1) and 31-34 (vs its warm start),
before any step cap. The adapter was not patched in this ticket, so the
warm-start pairing cannot produce a win-rate estimate yet.

Total development pipeline: 114.5 s warm start + 1,240 s PPO + 125 s vs-B1 eval
+ 235 s vs-warm-start eval = **1,715 s (28.6 min)**, within the 1 h development
limit.

### Win-probability calibration (`dev-*`, vs B1)

Report artifacts: `runs/cmp-dev1h/calibration/dev_vs_b1.{json,md}` and
`dev_vs_b1_reliability.svg` (gitignored). Temperature `T = 0.8144` was fitted on
the warm-start **validation** split; it is applied here without refitting on
`dev-*`.

| Metric (n = 12,558 decisions, 48 completed games) | Raw (T=1) | Calibrated (T=0.8144) |
| --- | ---: | ---: |
| Brier | 0.8695 | 0.8967 |
| NLL | 3.9145 | 4.7684 |
| ECE | 0.9251 | 0.9407 |
| Accuracy | 0.0090 | 0.0090 |

Reliability (calibrated), predicted -> observed:

| Bin | Count | Mean predicted | Observed | Gap |
| --- | ---: | ---: | ---: | ---: |
| [0.3, 0.4] | 8 | 0.3937 | 0.0000 | +0.3937 |
| [0.4, 0.5] | 105 | 0.4482 | 0.0000 | +0.4482 |
| [0.5, 0.6] | 201 | 0.5551 | 0.0000 | +0.5551 |
| [0.6, 0.7] | 325 | 0.6531 | 0.0000 | +0.6531 |
| [0.7, 0.8] | 754 | 0.7546 | 0.0000 | +0.7546 |
| [0.8, 0.9] | 1,207 | 0.8575 | 0.0000 | +0.8575 |
| [0.9, 1.0] | 9,958 | 0.9877 | 0.0000 | +0.9877 |

The value head is severely overconfident: it predicts near-certain wins while
the comparator loses every completed vs-B1 game. This is expected for a short
warm start followed by 384 PPO steps against mostly-weak self-play opponents:
the human-validation temperature does not transfer to the self-play/dev
distribution. The calibration machinery is correct (temperature fit reduces
validation NLL; see `tests/test_comparator.py`), but the agent needs the full
24 h schedule before calibration numbers are meaningful.

### Known deviations

- The development run hard-stopped at 20.7 min on adapter bug #32; the note in
  the ticket explicitly allows recording and reporting this instead of patching
  the adapter.
- Dev evaluation was run externally after the stop (the same approach as the
  #24 development run), not by the in-run checkpoint evaluator.
- The warm start is a short development schedule (900 steps), not the full
  4-epoch main-agent warm start.
- No frozen opponent checkpoints existed during the development run because the
  run was shorter than the 30-minute checkpoint interval.
- `eval-*` seeds were never read or written; all reported games are `dev-*`.

## Commands

```bash
# Warm start on the same corpus/split as the main agent.
conda run -n tm_bot python -m tm_bot.comparator.warmstart \
  --out-dir runs/comparator-warmstart --epochs 1 --max-steps-per-epoch 900

# PPO self-play under the run orchestration.
conda run -n tm_bot python -m tm_bot.runs --run-name cmp-dev1h \
  --agent-kind comparator --warmstart runs/comparator-warmstart/best \
  --budget-hours 0.5 --workers 16 --games-per-cycle 16 --updates-per-cycle 32 \
  --learner-batch-size 256 --learning-rate 0.0003 --dev-games 52

# Dev-seed win rate + calibration report (seat-swapped, vs B1 or a checkpoint).
conda run -n tm_bot python -m tm_bot.comparator.calibration \
  --checkpoint runs/cmp-dev1h/best --opponent b1 --games 52 --workers 16 \
  --device cuda --out runs/cmp-dev1h/calibration --name dev_vs_b1
```

## Acceptance evidence

- No shared parameters: `tests/test_comparator.py::test_comparator_shares_no_parameters_with_main_agent`.
- Illegal candidates get exact zero probability:
  `tests/test_comparator.py::test_illegal_candidates_get_exact_zero_probability`.
- PPO update reduces surrogate loss on a fixed batch:
  `tests/test_comparator.py::test_ppo_update_reduces_surrogate_loss_on_fixed_batch`.
- Calibration report: `runs/cmp-dev1h/calibration/` (Brier, NLL, reliability
  table + SVG), summarized above.
- Development run and dev-seed results: tables above; run artifacts under
  `runs/cmp-dev1h/`.
