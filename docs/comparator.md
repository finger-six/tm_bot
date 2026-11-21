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
agent, with the same v2 opponent mix (#48): 50% latest comparator, 25%
uniform from the last 10 frozen comparator checkpoints, 10% the frozen
comparator warm start (the anchor), 15% B1. Every comparator seat (learner,
latest, pool, anchor) samples the masked policy at τ(g) in the action phase and
at τ 1 elsewhere (see [selfplay.md](selfplay.md#v2-self-play-rules-48)). No seat
plays greedily. Mirror games against latest record both seats, and GAE runs on
each seat's trajectory separately.

v2 PPO (#39 §1c, §1d, §2; #50) in [`ppo.py`](../tm_bot/comparator/ppo.py):

- **τ-scaled ratio and entropy.** Each row records the τ it was sampled at
  (`play_temperature`). The behavior log-probability and the learner's new
  log-probability both come from `scaled_log_probs`: softmax(logits / τ) over
  the legal candidates, which is the policy the agent sampled from. Dropout is
  off in both, so the ratio is exactly 1 before an update. The entropy bonus
  uses the same scaled policy. An argmax row (τ 0) is rejected.
- **Reward U, margin-aware baseline.** The terminal reward is
  U = o + 0.15·m. GAE ([`compute_gae`](../tm_bot/comparator/ppo.py)) uses
  `gamma=1.0` and `lambda=0.95` against the critic baseline (2p − 1) + 0.15·m̂
  from the win and margin heads (`critic_baseline`). Advantages and returns are
  in U units, within ±1.18. Advantages are standardized per minibatch.
- **Outcome-only win head.** `win_logit` trains on BCE against (o + 1) / 2
  (0 / 0.5 / 1), never on a return or U, so `P(win)`, the in-run calibration
  (which reads the row's outcome) and the val-split temperature scaling stay
  unshaped. `vp_margin_pred` trains on the terminal m (SmoothL1, β 0.2).
- **Anchors**, shared with the main learner (see
  [selfplay.md](selfplay.md)): β·KL(π_θ ‖ π_anchor) with β 0.05 on the raw τ 1
  logits against the recorded anchor logits, and λ_h·L_human on the resident
  human sample (`human_aux_*` settings, λ_h on the training clock).

The learner ([`PPOLearner`](../tm_bot/comparator/ppo.py)) minimises, per
minibatch update:

```
−clipped surrogate (ε 0.2) + 0.5·(BCE(win_logit, (o+1)/2) + 0.3·SmoothL1(m̂, m))
  − 0.01·H(π_τ) + 0.05·KL(π_θ ‖ π_anchor) + λ_h·L_human
```

The value coefficient (0.5) scales the whole critic, and the margin term keeps the
main learner's 0.3 weight (`w_margin`) relative to the win BCE. Each cycle record
logs `policy_loss`, `value_loss`, `margin_loss`, `kl_anchor`, `human_aux_loss`,
`human_aux_weight` and `total_loss`.

Every cycle trains only on that cycle's fresh rollout. PPO is on-policy, so a
comparator run keeps no decision buffer: older rows carry log-probabilities and
advantages from older weights. An empty rollout (e.g. every game cut by the
deadline) performs no update. This also keeps the comparator's resumable
`state.pt` at ~32 MB instead of ~1.1 GB. A rollout is measured in games
(`games_per_cycle`), not decisions. Only the training seat's decisions become
PPO rows (~437 per game, half of the ~874 decisions both seats make), so the
spec's 16 x 2,048 = 32,768-row rollout corresponds to roughly 75 games per
cycle. Defaults: AdamW, lr 5e-4, batch 256. The development run used 16
games/cycle (~7,000 rows, a deviation from the spec rollout size), 32 minibatch
updates/cycle, batch 256, lr 3e-4 and bfloat16 AMP.

Checkpoints, `dev-*` evaluation, best selection, stopping rules, monitoring,
resource limits, and resume are the existing `tm_bot.runs` machinery.
`CheckpointManager.restore_model` dispatches on the checkpoint architecture, so
comparator runs resume and roll back correctly.

### Value-target bug fixed during review

The first development run reported a value head predicting ~0.99 win
probability while losing every game vs B1 (Brier 0.87, ECE 0.93). The cause was
a GAE bug, not overconfidence from a short warm start. GAE mixed a -1/+1
terminal utility with win-probability (0..1) bootstrap values. A lost game then
produced returns well above 0, which biased the BCE value targets upward.
Commit `b5d5fd8` computes GAE entirely in win-probability space
(`tests/test_comparator.py` covers the targets for wins and losses from both
seats). After the fix, in-run value calibration on finished self-play games
improved from Brier 0.094 / ECE 0.233 (cycle 1) to Brier 0.053 / ECE 0.098
(cycle 36).

## Evaluation policy

Dev evaluation, calibration and the eval-CLI factory play the comparator's
**sampled** policy, seeded per seat
(`comparator_eval_factory`, [`agent.py`](../tm_bot/comparator/agent.py)), not
greedy argmax. Since #48 they sample at τ(g) in the action phase and at τ 1
elsewhere. Before #48 they sampled at a flat temperature 1, and the #47 gate
still pins that (`make_comparator_agent(play_temperature="flat")`).

Greedy play stalled the first in-run dev evaluation. Greedy comparator vs
greedy warm start looped forever: on `tharsis` `dev-0000` the game reached
generation 244 with oceans (9) and oxygen (14) maxed and temperature stuck at
-10, while both seats kept buying the Aquifer standard project. Every dev game
hit the 20,000-decision step cap. With sampled play, games take 14-20
generations on average and a few seconds each. Each seat's agent is reset with
its seat seed, so evaluating the same checkpoints on the same seeds reproduces
exactly (`tests/test_comparator.py`). The main agent's IS-MCTS evaluation is
unaffected (the #24 candidate-vs-warm-start run completed 104/104 games).

## Development run (2026-11-15)

Run `runs/cmp-dev1h-final/` (gitignored): `--agent-kind comparator`, warm start
above, 16 workers, 1 h wall-clock budget, 16 games/cycle, 32 updates/cycle,
batch 256, lr 3e-4, 15-minute checkpoints with **in-run** dev evaluation (32
games vs B1 and 32 vs the previous best per checkpoint). It stopped on the
budget rule after 37 cycles, with 0 adapter rejections and 0 failures.

| Metric | Value |
| --- | ---: |
| Self-play games | 576 |
| Decisions | 503,500 (874/game) |
| PPO steps | 1,152 |
| Training-loop wall clock | 3,601 s (3,012 s self-play + learner, the rest in-run dev eval) |
| Total wall clock incl. final dev eval | 3,773 s |
| Throughput | **576 self-play games/hour** over the training loop incl. in-run dev eval (549.6/hour incl. the final dev eval) |
| Mean generations/game | 21.7 (range 11-65) |
| Shape flags (outside 6-25 generations) | 103 of 576 games |
| Wins / losses / ties (training seat) | 194 / 382 / 0 |
| Opponent mix | 399 latest / 126 pool / 51 B1 |
| Decision latency p50 / p95 (median over cycles) | 3.5 ms / 6.0 ms |
| Policy entropy, cycle 1 -> 36 | 1.17 -> 1.12 |
| Value loss, cycle 1 -> 36 | 0.58 -> 0.17 |

Mean generations per game dropped from 38.9 in cycle 1 to 17-19 from cycle 10 on.

### In-run dev evaluation

| Checkpoint | Cycle | Games | vs B1 (32 games) | VP margin vs B1 | vs previous best (32 games, 95% CI) | New best |
| --- | ---: | ---: | --- | ---: | --- | :---: |
| ckpt-000000001 | 11 | 176 | 0.0% [0.0, 0.0] | -81.7 | 68.8% [46.9, 84.4] vs warm start | yes |
| ckpt-000000002 | 21 | 336 | 0.0% [0.0, 0.0] | -77.5 | 71.9% [50.0, 87.5] vs ckpt-1 | no |
| ckpt-000000003 | 30 | 480 | 0.0% [0.0, 0.0] | -73.9 | 96.9% [90.6, 100] vs ckpt-1 | no |
| ckpt-000000004 | 37 | 576 | 0.0% [0.0, 0.0] | -77.1 | 96.9% [90.6, 100] vs ckpt-1 | no |
| ckpt-000000005 | 37 | 576 | 0.0% [0.0, 0.0] | -77.1 | 96.9% [90.6, 100] vs ckpt-1 | no |

The run exposed two orchestration issues. Both are fixed on this branch, with tests:

- **Best selection.** It used the vs-B1 win rate with a strict `>`. At 0% vs B1
  the best stayed `ckpt-000000001`, so "previous best" never advanced, and the
  plateau rule would have fired after 12 checkpoints while the agent kept
  improving. Now an exact vs-B1 tie becomes the new best only if it beats the
  previous best with the 95% CI lower bound above 50% (`is_new_best` in
  [`stopping.py`](../tm_bot/runs/stopping.py), shared by best selection and the
  stopping monitor). Under this rule `ckpt-000000003` would have become the
  best.
- **Duplicate final checkpoint.** `ckpt-000000005` holds the same weights as
  `ckpt-000000004`. The final soft-stop checkpoint fired whenever 60 s had
  passed since the previous checkpoint *started*, and that checkpoint's own dev
  eval took longer than that. It now fires only if the model trained since the
  last checkpoint.

### External dev evaluation and calibration (200 games per pairing)

From `ckpt-000000005` with `python -m tm_bot.comparator.calibration` (seat
swapped, four maps, `dev-*` seeds; no `eval-*` seed was touched):

| Pairing | Games | Completed | Failures | Win rate (95% paired bootstrap CI) | Mean VP margin | Mean gens | Wall |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Comparator vs B1 | 200 | 200 | 0 | **8.0%** [2.0, 16.0] | -60.7 | 14.0 | 332 s |
| Comparator vs its warm start | 200 | 200 | 0 | **89.5%** [85.0, 93.5] | +34.8 | 19.8 | 742 s |

After 1 h of PPO the comparator clearly beats its warm start. It is still far
below B1.

### Win-probability calibration

Artifacts: `runs/cmp-dev1h-final/calibration/dev_vs_{b1,warmstart}.{json,md}`
and `*_reliability.svg` (gitignored). Every non-forced decision (more than one
candidate) of the comparator's seat in the 200 games is scored against that
game's outcome. Temperature `T = 0.8144` was fitted on the warm-start
validation split with the **warm-start** weights and reused unchanged after
PPO; it is not refitted on `dev-*`. These are the same `dev-*` seeds used for
checkpoint selection, so the numbers are not a held-out estimate.

The base-rate column is the score of a constant prediction equal to the
decision-weighted win frequency (0.172 vs B1, 0.907 vs warm start; won games
are longer, so this exceeds the per-game win rate).

| Pairing | n decisions | Brier raw / T / base rate | NLL raw / T / base rate | ECE raw / T | Accuracy (base rate) |
| --- | ---: | --- | --- | --- | ---: |
| vs B1 | 52,566 | 0.089 / 0.094 / 0.143 | 0.280 / 0.300 / 0.460 | 0.084 / 0.102 | 0.877 (0.828) |
| vs warm start | 82,871 | 0.603 / 0.646 / 0.084 | 1.776 / 2.095 / 0.309 | 0.708 / 0.735 | 0.202 (0.907) |

Vs B1 the head beats the constant baseline; vs its warm start it is far worse.

Reliability vs B1 (raw): 85% of decisions fall in the [0, 0.1] bin, with mean
prediction 0.041 vs observed 0.052. The few won games are under-predicted
(predictions 0.1-0.7, observed 0.55-1.0). Vs its warm start the head is
strongly **under**confident: predictions 0.07-0.64, while the observed win rate
is 0.88-1.0 in every bin.

The head estimates the win probability from the state alone, against the
opponent mix it was trained on (mostly itself). It has no input for who the
opponent is, so it cannot know it faces a weaker warm start. Its calibration is
therefore opponent-dependent: close vs B1, far off vs a much weaker opponent.

Known caveat (review P2-5): the temperature is fitted on the human-data
validation `setup_outcomes`, not on a self-play-representative set. Here it
makes every metric slightly worse (T < 1 sharpens). A held-out slice of
`train-*` self-play games would be a representative fitting set. v1 does not
use one because the spec fixes fitting to validation data.

### Known deviations

- The warm start is a short development schedule (1 epoch, 900 draft steps),
  not the full 4-epoch main-agent warm start. Run `--epochs 4` before the
  milestone run (#26) for parity with the main agent.
- The in-run dev evaluation used 32 games per pairing to fit the 1 h limit; the
  milestone default is 100.
- Self-play comparator opponents act greedily, while evaluation samples.
- Rollouts were 16 games (~7,000 PPO rows) per cycle, not the spec's 32,768
  rows (~75 games); use `--games-per-cycle 75` for the milestone run.
- The collapse rule (vs-B1 drop of 15 points below the best) cannot fire while
  the best vs-B1 win rate is 0%.
- 103 of 576 games were flagged as shape anomalies (outside 6-25 generations),
  88 of them in cycles 1-6 and none after cycle 21; they were not investigated
  individually (deferred to the milestone run, #26).
- The reproducibility test covers the evaluation agent's seeded `act` calls,
  not a full `play_games` replay.
- The development run trained with dropout (0.05) active in the PPO update, so
  its ratios were slightly off-policy (log-prob gaps up to ~2% on a small probe
  model); fixed afterwards.
- `eval-*` seeds were never read or written; all reported games are `dev-*`.
- Adapter bug #32, which cut the first development run short, is fixed on `main`.

## Commands

```bash
# v2 warm start: the shared 15,000-step multi-task schedule (docs/warmstart.md section 5).
# The v1 flags --epochs/--max-steps-per-epoch below are gone; the v1 sections above are a record.
conda run -n tm_bot python -m tm_bot.comparator.warmstart \
  --data-dir data/warmstart --out-dir runs/comparator-warmstart-v2

# PPO self-play under the run orchestration (1 h development run).
conda run -n tm_bot python -m tm_bot.runs --run-name cmp-dev1h-final \
  --agent-kind comparator --warmstart runs/comparator-warmstart/best \
  --budget-hours 1.0 --workers 16 --games-per-cycle 16 --updates-per-cycle 32 \
  --learner-batch-size 256 --learning-rate 0.0003 --checkpoint-minutes 15 \
  --dev-games 32

# Dev-seed win rate + calibration report (seat-swapped, vs B1 or a checkpoint).
conda run -n tm_bot python -m tm_bot.comparator.calibration \
  --checkpoint runs/cmp-dev1h-final/checkpoints/ckpt-000000005 --opponent b1 \
  --games 200 --workers 16 --device cuda \
  --out runs/cmp-dev1h-final/calibration --name dev_vs_b1
conda run -n tm_bot python -m tm_bot.comparator.calibration \
  --checkpoint runs/cmp-dev1h-final/checkpoints/ckpt-000000005 \
  --opponent runs/comparator-warmstart/best \
  --games 200 --workers 16 --device cuda \
  --out runs/cmp-dev1h-final/calibration --name dev_vs_warmstart

# The general eval CLI loads the comparator through its factory:
#   tm_bot.comparator:make_comparator_agent?checkpoint=<dir>&device=cuda
```

## 1 h development gate (#52)

The comparator runs the same 1 h gate as the main agent before its 24 h run
(checks and thresholds in [selfplay.md](selfplay.md#1-h-development-gate-52)),
with a throughput floor of 800 self-play games per training hour. The 8/4
search fallback does not apply: the comparator scores candidates directly.

```bash
conda run -n tm_bot python -m tm_bot.runs --gate-1h --run-name gate-comparator-v2 --agent-kind comparator --warmstart runs/comparator-warmstart-v2/best
conda run -n tm_bot python -m tm_bot.runs.dev_gate runs/gate-comparator-v2 --out runs/gate-comparator-v2/dev_gate.md
```

## Acceptance evidence

- No shared parameters: `tests/test_comparator.py::test_comparator_shares_no_parameters_with_main_agent`.
- Illegal candidates get exact zero probability:
  `tests/test_comparator.py::test_illegal_candidates_get_exact_zero_probability`.
- PPO update reduces surrogate loss on a fixed batch:
  `tests/test_comparator.py::test_ppo_update_reduces_surrogate_loss_on_fixed_batch`.
- Calibration report (Brier, NLL, reliability diagram): the tables above;
  artifacts in `runs/cmp-dev1h-final/calibration/`.
- 1 h development run under the run orchestration with dev-seed results: the
  tables above; artifacts in `runs/cmp-dev1h-final/`.
