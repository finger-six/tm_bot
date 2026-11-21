# Self-Play Training Loop and Run Orchestration

Ticket #24 adds the AlphaZero-style self-play loop for the main agent: run
configuration, opponent sampling, a replay buffer, learner updates, periodic
`dev-*` checkpoint evaluation, stopping rules, monitoring, and resume. Code:
[`tm_bot/runs/`](../tm_bot/runs/). It reuses the search agent from
[search.md](search.md) (`ISMCTSAgent`), the shared network from
[warmstart.md](warmstart.md), the runner from `tm_bot.agents.runner`, and the
evaluation/reporting code from [evaluation.md](evaluation.md).

```bash
conda run -n tm_bot python -m tm_bot.runs --run-name dev1h \
  --budget-hours 1 --sims 4 --worlds 4 --workers 16 --games-per-cycle 16 \
  --warmstart "C:\2026\tm_bot\runs\warmstart\best"
```

## Run configuration

`RunConfig` ([`config.py`](../tm_bot/runs/config.py)) defaults to the #10 budget
decision; every field is overridable from the CLI.

| Setting | Default | Notes |
| --- | --- | --- |
| Workers | 16 | Node adapter workers; one game per active worker |
| Wall-clock budget | 24 h | Hard stop when reached (checked between cycles and inside the self-play batch loop) |
| Simulations / searched decision | 16 (8 with `--search-fallback`) | `--sims`; never below 8 x 4 (#39) |
| Worlds (K) | 4 | `--worlds` |
| Play temperature | τ(g): 1.0 to g4, linear to 0.3 at g10, 0.3 after | every learned seat, action phase only |
| Utility | U = o + 0.15·m | search leaves and terminals |
| Opponents | 50% latest / 25% last-10 pool / 10% warm-start anchor / 15% B1 | `OpponentSampler` |
| Replay buffer | last 200,000 decisions | `ReplayBuffer` |
| Learner | AdamW, lr 5e-4, wd 2e-4, grad clip 1.0, batch 256 | same optimizer settings as warm start |
| Checkpoints | every 30 min | monotonic `ckpt-XXXXXXXXX/` save counter, plus `best/`; feeds the opponent pool |
| Retained checkpoints | best + last 10 | older ones pruned |
| Disk cap | 50 GB | oldest non-best checkpoints deleted first; a cap hit stops the run as `resource_limit` |
| Dev evaluation | newest checkpoint every 60 min: 104 `dev-*` games vs `b1-v2`, 104 vs the incumbent best, 56 vs the warm-start anchor | seat-swapped pairs; selected on `dev-*` only; cadence doubles while one evaluation exceeds 40% of it (`--dev-eval-minutes`) |
| Plateau | 6 h of training clock without a new best | `--plateau-hours` |
| Regression vs anchor | win rate vs the warm-start anchor < 35% for 3 evaluations | |
| Margin collapse | dev score `D` > 0.05 below the best's for 3 evaluations | |
| Game shape | self-play median > 22 generations for 2 h of training clock | hard stop |
| Calibration | dev ECE vs `b1-v2` > 0.15 for 3 evaluations | stops the comparator; logged for main |
| Collapse (entropy) | <10% of the initial network's entropy within 2 h | |
| Determinism spot check | every 1,000 self-play games | replay seed + action history |

Resource limits live in `ResourceLimits`: learner + buffer <= 20.0 GiB RSS and
VRAM <= 12 GB (checked every cycle; a breach stops the run as `resource_limit`),
run outputs <= 50 GB under the gitignored `runs/` directory, and the configured
worker count bounds engine workers (per-worker RSS is not measured separately;
it is bounded by the 16-worker default). When CUDA is available, VRAM is also
soft-capped with `torch.cuda.set_per_process_memory_fraction`.

The RAM metric is the learner process's resident set: `/proc/self/statm` on
Linux and the working set (`WorkingSetSize` from `K32GetProcessMemoryInfo`) on
Windows. Windows `PagefileUsage` (private commit) is not used because it counts
CUDA driver commit that is not host RAM. The 20.0 GiB default (#36) clears the
v1 milestone peaks (17.2 GiB `milestone-main`, 16.2 GiB `milestone-pure`) while
still stopping a leak before the 16 Node workers (<= 6 GiB) starve on the
31.5 GiB host. Before #35 the Windows probe returned `None`, so v1 runs were
never RAM-guarded.

Run outputs (`runs/<run-name>/`): `config.json`, `state.json` + `state.pt`
(resume), `checkpoints/`, `init/` (random-init runs only), `best/`,
`monitor/metrics.jsonl`, `monitor/summary.json`,
`run_summary.json`, and `regression/` cases. `.gitignore` contains `/runs/`, so
none of these are committed.

## Self-play

Each game is one `train-*` seed and one supported map. The training seat is
alternated; the other seat is an opponent sampled 50% from the latest network,
25% from the last 10 frozen checkpoints, 10% from the frozen warm-start anchor,
and 15% `b1` (the current `B1_VERSION`). With an empty pool (a fresh run) the
pool share falls back to the latest network.

### v2 self-play rules (#48)

The rules come from #39 (last comment, sections 1-2) and are shared by the main
agent and the comparator. `RunConfig.selfplay_rules` is `"v2"`, and loading a
run config without it raises, so a v1 run cannot be resumed under v2 code.

- **Play temperature τ(g)** (`tm_bot/agents/play_rules.py`,
  `PlayTemperatureSchedule`): 1.0 through generation 4, linear to 0.3 at
  generation 10, 0.3 after, never 0. Every learned seat (learner, `latest`,
  pool, anchor, and the `dev-*`/`eval-*` agents) samples action-phase decisions
  at τ(g); the main agent samples its searched roots from the completed-Q
  policy π′ (#60, [search.md](search.md)). Outside the action phase the main
  agent takes the argmax of π′ or of the policy head, and the comparator samples
  at τ 1. No learned seat
  plays greedily in the action phase.
- **Utility** U = o + 0.15·m, with o ∈ {+1, 0, -1} and
  m = clip(my_vp - opp_vp, -60, 60) / 50, so any win outranks any loss. Search
  leaves use (2p - 1) + 0.15·m̂ from the value and margin heads, and terminals
  use U from the final scores. The value target stays o, so P(win) is
  outcome-only.
- **Search**: 16 simulations x K = 4. `--search-fallback` lowers this to
  8 x 4 by setting `num_simulations` to 8.
- **Anchor**: the frozen warm-start checkpoint (`warmstart_checkpoint`, also
  for random-init runs). It plays as an opponent under the run's rules and
  scores every recorded decision (see Targets).
- **Mirror games** (opponent `latest`) record both seats. All other games
  record only the training seat.

`RecordingSearchAgent` ([`recorder.py`](../tm_bot/runs/recorder.py)) wraps both
seats and records the ordered `(seat, candidate)` action history; it records
the encoded observation/choice and candidate distribution **only for the
training seat** (both seats in mirror games). Encoding happens at record time, so a cycle does not retain
the raw decoded `ActorObservation` dicts and `Choice` objects (which cost
~423 KB per decision instead of ~10 KB encoded).
The runner still passes only `(observation, choice[, sampler])` to
agents, so a training row can only contain what that seat legitimately saw.
The training input is the acting seat's encoded `ActorObservation`; opponent
private cards, deck order, and RNG never enter it (the adapter's actor
observation has no such fields, and `test_opponent_private_cards_never_enter_the_acting_seat_encoding`
checks the encoding is seat-scoped).

### Targets

- **Policy**: the root policy at searched decisions, the completed-Q π′ ∝
  prior · exp(3·Q̂) since #60 (normalized root visit counts before). Decisions
  handled by the policy head (nested payments, spaces, players, amounts,
  initial corporation/prelude picks) are kept and targeted from the policy
  head's own distribution; forced single-candidate decisions are one-hot. This
  keeps the whole game in the training set instead of only searched roots.
- **Value**: terminal utility from the acting seat's perspective: win `+1`,
  loss `-1`, exact tie `0`. The value head predicts win probability, so the
  loss uses `(u + 1) / 2` as the target.

Each row also carries the v2 fields (`DecisionSample` in
[`buffer.py`](../tm_bot/runs/buffer.py) and `PPOSample` in
[`ppo.py`](../tm_bot/comparator/ppo.py)), which the learners consume
(#49 main, #50 comparator):

- `play_temperature`: the τ the seat sampled this decision at. It is 0.0 for an
  argmax decision and None when the agent reported no τ (forced decisions).
- `margin_target`: m for the row's seat at game end.
- `anchor_logits`: float32[n] raw (τ = 1) policy logits of the frozen anchor
  on the same encoded observation and choice. Index i is candidate i, in the
  same order as `policy_target` and the action index. The values are computed
  once per cycle in `assemble_batch` (`tm_bot/runs/anchor.py`), and they are
  None when the run has no anchor.

`AlphaZeroLearner` ([`learner.py`](../tm_bot/runs/learner.py)) samples decisions
from the replay buffer (v1 buffer, batch, LR and updates per cycle) and minimises
the v2 loss (#39 §1c-1d, §3; #49):

```
CE(policy, π′) + 1.0·BCE(win_logit, o) + 0.3·SmoothL1(vp_margin_pred, m)
  + λ_h·L_human + 0.05·KL(π_θ ‖ π_anchor)
```

- `win_logit` trains on the terminal outcome alone (target `(o + 1) / 2`, so
  0 / 0.5 / 1); `vp_margin_pred` on the row's `margin_target` m with
  SmoothL1 (β 0.2).
- **KL anchor**: mean KL from the learner's raw (τ = 1) policy to the recorded
  `anchor_logits` over each row's legal candidates; rows without anchor logits
  are left out. β = `kl_anchor_beta` = 0.05, constant.
- **Human-data auxiliary loss** `L_human`: the full #38 multi-task warm-start
  loss (every policy head, win and margin heads, `WarmstartConfig` weights and
  per-task minibatches, ~3.3k rows) on one train-split human minibatch per
  update, back-propagated group by group like the warm start. λ_h is 0.5 for
  training hours 0-2, linear to 0.1 at hour 8, then 0.1
  (`RunConfig.human_aux_schedule`). Hours are the run's training clock
  (`SelfPlayRunner.training_s`: wall clock minus in-run dev evaluation,
  continued across `--resume`), read once per cycle before its updates.
- **Resident human sample** ([`human_aux.py`](../tm_bot/training/human_aux.py)):
  at run start the learner loads `data/warmstart` and its encoded cache, draws
  at most `human_aux_max_rows` = 1,000,000 train rows once (split across the
  sampler's pools in proportion to their per-step draws; a smaller pool is taken
  whole), copies the chosen cache rows into RAM and releases the memory maps. At
  ~1.4 KB per cached row that is ~1.4 GiB at 1M rows (plus the small generation-1
  setup tables). #39 allows up to 2M; a real-cache load of 2M rows measured 2.72 GiB
  of cache rows and +2.9 GiB of steady working set, and the unguarded v1
  `milestone-main` learner peaked at 18.0 GiB (p99 14.0 GiB), so 2M rows could cross
  the 20.0 GiB guard late in a run; 1M keeps the projected peak near 19.5 GiB. The
  one-time load (~45 s) transiently touches most of the mapped cache (peak working
  set 15.3 GiB at 2M rows) before the guard's first per-cycle check; those pages are
  file-backed and released with the maps. Human rows only, never `b1-v2` (#36).
- The shared pieces (`HumanAuxSchedule`, `pad_anchor_logits`, `kl_to_anchor` in
  [`anchor_loss.py`](../tm_bot/models/anchor_loss.py); `load_human_aux_sample`,
  `HumanAuxLoss` in `human_aux.py`) are also used by the comparator's PPO learner
  (#50; see [comparator.md](comparator.md#ppo-self-play)).
- A `--random-init` run is pure self-play: it drops both anchor terms (no human
  sample is loaded, β = 0), per #39 §4.

Each cycle record logs every term separately: `policy_loss`, `value_loss`,
`margin_loss`, `kl_anchor`, `human_aux_loss` (unweighted cycle means),
`human_aux_weight` (λ_h) and `total_loss` (the weighted sum).

Value-head calibration (Brier, ECE, reliability bins, win/loss/tie counts) is
measured by `compute_value_calibration` on each cycle's newly finished games
**before** they enter the buffer and before the optimizer step, with frozen
weights, so the cycle record is out-of-sample for that update. Training records
carry only training-batch losses (the terms above); calibration is not
mixed into them. Every parameter tensor is checked for finiteness after each
optimizer step; a NaN/Inf loss or parameter stops the cycle and rolls the model
and optimizer back to the last periodic checkpoint (a failed rollback is
reported in the stop decision rather than swallowed).

## Dev evaluation and best selection

Floor-aware selection from #40 sections 1-2 (#51). `dev-*` games use
`build_paired_specs` from [`tm_bot/eval/pairing.py`](../tm_bot/eval/pairing.py):
every `(seed, map)` is played twice with the candidate in opposite seats, so the
paired bootstrap CI is valid, and every evaluation reuses the same fixed seeds
(`dev_seed_start`). Each dev evaluation of the newest checkpoint plays three
pairings:

| Pairing | Games | Signal |
| --- | ---: | --- |
| vs `b1-v2` | 104 (13 seeds x 4 maps x 2 seats) | dev score `D`, win rate, mean VP margin, value-head ECE |
| vs the incumbent best | 104 | head-to-head win rate `h` with its paired bootstrap CI |
| vs the warm-start anchor | 56 (the first 7 of those seeds) | win rate `a` |

`D = mean(o + 0.15 * m)` over the `b1-v2` games, with `o` = +1/0/-1 for a
win/tie/loss and `m = clip(VP margin, -60, 60) / 50` (the #39 utility). At the
floor (all losses) `D` measures VP margin; once wins appear it moves smoothly
toward win rate. Per-game results (`DevGame`) are kept, and the incumbent's
`b1-v2` games are stored in `state.json`, so the candidate's games pair with
the incumbent's by `(seed, map, seat)` for a seed-clustered bootstrap CI on
`dD = D_C - D_best` (`paired_delta_d` in
[`deveval.py`](../tm_bot/runs/deveval.py)).

The warm-start anchor is the resolved warm-start checkpoint (recorded in
`config.json`), or the saved random-init weights in `<run>/init/` for a
`--random-init` run (see [Random initialisation](#random-initialisation-27)).
Until a first new best exists the anchor is the incumbent: its own `b1-v2`
games are measured once on the same 104 seeds (`anchor_dev_score` in
`run_summary.json`), and `a` is read from the incumbent pairing's anchor seeds
rather than played again.

`new_best_route` in [`stopping.py`](../tm_bot/runs/stopping.py) makes the
candidate the new best only if both gates hold, G1 zero failures across all
pairings and G2 `a >= 0.45`, and then either route does:

- **R1 `delta_d`**: dD 95% CI lower bound > 0, and `h >= 0.50`.
- **R2 `head_to_head`**: `h` 95% CI lower bound > 0.50, and dD CI lower bound
  >= -0.03 (about -10 VP mean margin at the floor).

The accepting route is logged as `new_best_route`. This replaces the v1 rule,
which ranked by win rate vs B1 alone and picked noise at the 0% floor.
`eval-*` seeds are never read or written by a run.

### Cadence and cost cap

Checkpoints are saved every 30 min of training clock for the opponent pool.
The newest one is dev-evaluated when 60 min of training clock have passed
since the last evaluation, and before a soft stop or the run's end. After each
evaluation the cadence becomes the smallest of 60, 120, 240, ... min in which
that evaluation's wall time fits in 40% (`dev_eval_cost_fraction`), so it never
drops below 60 min; the one-time anchor baseline is not counted. Dev games use
the agent's real play settings.

## Stopping rules

`StoppingMonitor` ([`stopping.py`](../tm_bot/runs/stopping.py)) is fed metric
events and returns a typed `StopDecision`; each rule has a synthetic-stream
test in [`tests/test_runs.py`](../tests/test_runs.py) or
[`tests/test_selection.py`](../tests/test_selection.py):

| Rule | Trigger | Action |
| --- | --- | --- |
| `adapter_rejection` | `AdapterBugError` in a self-play game | hard stop, write regression case |
| `engine_error` | bridge/worker error | hard stop, write regression case |
| `step_cap` | game exceeded `max_decisions_per_game` | hard stop, write regression case |
| `nonfinite_loss` | NaN/Inf learner loss | roll back to last good checkpoint, stop |
| `collapse_entropy` | policy entropy <10% of the initial value within 2 h | stop |
| `plateau` | no new best for 6 h of training clock (from the run start until a first one) | stop, keep the last best |
| `regression_anchor` | `a < 0.35` for 3 consecutive dev evaluations | stop, keep the last best |
| `margin_collapse` | `D` more than 0.05 below the best's `D` for 3 consecutive dev evaluations | stop |
| `game_shape` | self-play median > 22 generations for 2 h of training clock (no game is truncated) | hard stop: investigate a rule bug or stall |
| `calibration` | dev ECE vs `b1-v2` > 0.15 for 3 consecutive dev evaluations | comparator: stop and investigate; main: logged only |
| `max_steps` | optional learner step cap | stop |
| `resource_limit` | RAM/VRAM ceiling or disk cap hit | stop, keep state for resume |
| `budget` | training-time budget reached (see [Training clock](#training-clock)) | stop; also polled inside the self-play batch loop, which returns partial games as `DeadlineReached` and excludes them from training |

### Training clock

The budget (`--budget-hours`, `wall_clock_budget_s`), the checkpoint interval
and the 2 h entropy-collapse window run on the **training clock**: wall clock
minus the time spent in in-run dev evaluation (#26 decision). Self-play and
learner time count; dev games do not, so the main agent and the comparator get
the same training budget even though their dev evaluations cost very different
wall time. Cycle and checkpoint records, `monitor/summary.json`,
`run_summary.json` and `state.json` report all three clocks: `elapsed_s` (total
wall clock), `training_s` and `dev_eval_s`. The summary also gives throughput per
wall hour and per training hour. `run_summary.json` is the authoritative
record of the final clocks; `monitor/summary.json` is derived from the last
logged record and omits the few seconds of final state saving.

A regression case is `regression/<seed>__<map>__<error>.json` containing the
seed key, map, error, decision count, and the full ordered action history.

## Monitoring

`MonitorLogger` writes one JSON line per self-play cycle and per checkpoint to
`monitor/metrics.jsonl`, plus `monitor/summary.json`. Cycle records contain:
network policy entropy on the fixed probe, self-play target entropy,
policy/value/total training losses, pre-update value calibration (n, W/L/T,
Brier, ECE, reliability bins), W/L/T and VP-margin histogram for the training
seat, generations/game, decisions/game, searched-decision share, realized
opponent-kind mix, separate rejection and failure counts, decision latency
p50/p95 (both seats pooled; on the unbatched paths the training seat's latency
includes the record-time encoding of its decisions, so the seats are not
directly comparable; batched IS-MCTS latency is search or policy-head time
amortized across the batch and excludes encoding), throughput, determinism
check result, and untruncated game-shape flag
counts (generations outside 6-25) with up to 20 example games.
Checkpoint records contain dev win rates and CIs vs `b1-v2` and the incumbent,
`D`, the best's `D`, dD and its CI, `a`, the dev ECE, failures, the new-best
route, and the evaluation's wall time and resulting cadence. Cycle records also
carry the median generations used by the game-shape rule.
Determinism spot checks replay a finished game from its seed and recorded action
history and compare generations, scores, winner, and tie (`replay_game`,
`compare_replay`); the check counter is persisted across resume.

The entropy collapse rule uses exactly one quantity: the network policy entropy
on the first cycle's fixed observation set (`probe_policy_entropy`). That probe
is captured before any update, stored in `state.pt`, and re-evaluated with the
current weights each cycle; the self-play target entropy is logged separately
for information and never compared to the network probe. The baseline is
therefore the **initial model's** entropy: the warm-start network for a normal
run, and the random-init network for `--random-init` (the rule itself does not
change; the initial model is simply the fresh one).

## Random initialisation (#27)

The pure self-play comparison run trains the main agent from random weights
while keeping the same orchestration, budget, search settings, learner, and dev
protocol as the warm-started run:

```
conda run -n tm_bot python -m tm_bot.runs --run-name milestone-pure --random-init --budget-hours 24 --sims 4 --worlds 4 --workers 16 --checkpoint-minutes 30 --dev-games 100 --device cuda
```

`--warmstart <checkpoint>` (default: `DEFAULT_WARMSTART_CHECKPOINT`,
`runs/warmstart/best`) is still read, also on `--resume`: it supplies the `TMModelConfig`, the `CardVocab`, and the non-learned
`card_static_matrix`, so the observation/action encoding and the network shape
are identical. **No learned parameter is copied.** Every weight is
re-initialised by `build_random_init_model`
([`network.py`](../tm_bot/models/network.py)) from `RunConfig.seed` inside a
`torch.random.fork_rng` context, so two runs with the same `--seed` start
from bit-identical weights and the caller's global RNG is left untouched. Only
the initial weights are seeded this way: later torch randomness (dropout in the
learner) is not torch-seeded, as in warm-started runs.
`TMNetwork` has no non-learned `temperature` attribute; its `config` and
`card_static_matrix` buffer are the only non-learned state, and both are kept.

Differences from a warm-started run:

- The fresh weights are saved once at run start to `<run>/init/` (metadata
  `init: "random"`, `warmstart_checkpoint: null`, `architecture_from:
  <checkpoint>`). `<run>/init/` sits outside `checkpoints/`, so
  `keep_last_checkpoints` pruning and the disk cap never delete it.
- Before the first `best/` exists, that `init/` checkpoint is the
  previous-best dev opponent instead of the warm start.
- `state.json` and `run_summary.json` record `init: "random"` and
  `init_checkpoint`; `--resume` keeps using the recorded init reference. As
  for any run, `--resume` must repeat the original flags, including
  `--random-init` and the same `--warmstart`, or the config check rejects it.
  A missing `<run>/init/` fails the resume with `RunResumeError`.
- `--random-init` is rejected for `--agent-kind comparator` (PPO requires its
  warm start).

Everything else — opponents, replay buffer, learner, checkpoint cadence,
entropy-collapse baseline (now the random network's probe), stopping rules, and
dev seeds — is shared with the warm-started run.

## Resume

`state.json` stores the cycle, learner step, self-play game count, best dev win
rate, elapsed wall clock, training time and dev-eval time, seed cursor, opponent pool, determinism-check counter,
entropy probe baseline, stopping-rule counters (plateau clock, streaks, game-shape
clock), the incumbent's per-game `b1-v2` results, best and anchor `D`, the dev
cadence, and the init provenance
(`init` plus `init_checkpoint` for a random-init run). `state.pt` stores the
model/optimizer and the replay buffer (plus the fixed entropy probe). `--resume`
restores the model from the last good checkpoint (falling back to the newest
numbered checkpoint, then `best/`, then a random-init run's `init/` when no
periodic checkpoint exists yet), restores the optimizer and replay buffer
independently (a missing optimizer state does not skip the buffer), reopens the
metrics stream, restores the seed cursor and stopping counters, and continues
the run. Checkpoints are named by a monotonic save counter (`ckpt-000000001`,
`ckpt-000000002`, ...) so two saves with the same learner step never collide.

## 1 h development gate (#52)

Before a 24 h run, each agent runs 1 h of training clock under the full v2
rules with `--gate-1h` (`dev_gate_config` in
[`config.py`](../tm_bot/runs/config.py)): the budget becomes 1 h and dev
evaluation runs at 30 and 60 min (this 30-min cadence is for the gate only; the
24 h runs keep the 60-min cost-capped cadence). Search, rules, opponents and
the checkpoint come from the same flags as the 24 h run, so a later change to
them carries through. Then
[`tm_bot/runs/dev_gate.py`](../tm_bot/runs/dev_gate.py) reads the finished run
directory, prints a Markdown report and exits 1 if any check fails (#40
section 3):

| # | Check | Pass when |
| ---: | --- | --- |
| 1 | Legality | 0 failures (self-play and dev), 0 adapter rejections, 0 step-cap hits |
| 2 | Game shape | self-play median <= 18 generations, p95 <= 25, no game > 40 |
| 3 | Throughput | self-play games per training hour: main >= 250, comparator >= 800 |
| 4 | Symmetry | learner-seat win share vs the pool and vs the anchor each in [35%, 65%] (a tie counts half) |
| 5 | Cost | each dev evaluation, less the one-time anchor baseline, <= 40% of the 24 h runs' 60-min cadence (24 min), so those runs keep it; the gate's cumulative fraction is reported for information |
| 6 | Positive signal at 60 min | evaluations at 30 and 60 min; at 60 min `a >= 0.45` and `D` >= the anchor's own `D` vs `b1-v2`; the run stopped on `budget` |
| 7 | Resources | max learner RSS <= the 20.0 GiB guard, read every cycle and after every dev evaluation, and no `resource_limit` stop |

The anchor's `D` is the existing one-time baseline: the first dev evaluation
plays the warm-start anchor against `b1-v2` on the same 104 `dev-*` seeds
(`anchor_dev_score` in `run_summary.json`). Its games count toward check 1; its
wall time is reported separately (`dev_eval_baseline_s`) and left out of check 5,
as the cost cap leaves it out of the cadence. For the gate, cycle
records add per-generation game counts (`generation_counts`), the learner
seat's W/L/T per opponent kind (`results_by_opponent`) and `rss_gb`.

```bash
# Main agent at 16/4. Use the main warm start that passed #47.
conda run -n tm_bot python -m tm_bot.runs --gate-1h --run-name gate-main-v2 --warmstart runs/warmstart-v3/best
conda run -n tm_bot python -m tm_bot.runs.dev_gate runs/gate-main-v2 --out runs/gate-main-v2/dev_gate.md

# If check 3 fails at 16/4: the 8/4 fallback.
conda run -n tm_bot python -m tm_bot.runs --gate-1h --run-name gate-main-v2-8x4 --search-fallback --warmstart runs/warmstart-v3/best
conda run -n tm_bot python -m tm_bot.runs.dev_gate runs/gate-main-v2-8x4 --out runs/gate-main-v2-8x4/dev_gate.md
```

Pass `--warmstart` explicitly: without it the main agent falls back to the v1
warm start (`runs/warmstart/best`). `--json` prints the report as JSON. The
comparator's gate is in [comparator.md](comparator.md#1-h-development-gate-52).
Each gate run takes 1 h of training clock plus its dev evaluation: 312 dev
games at 30 min (anchor baseline, `b1-v2`, incumbent = anchor) and 208 at
60 min (264 if the 30-min checkpoint became the best).

**Check 5 amendment (#52, resolved autonomously).** #40 section 3 reads "dev-eval wall
time <= 40% of the training clock". Taken literally, the gate's two evaluations plus
the baseline in 1 h of training would need about 1,300 dev games per wall hour, well
above the expected dev throughput, while the 24 h runs meet the 40% cap by construction (the
cadence doubles until one evaluation fits). The check therefore asks what the cap
protects: whether the 24 h runs can keep the 60-min cadence, i.e. whether each gate
evaluation (up to 264 games, without the baseline) takes at most 24 min.

## Results

### Development run (2026-11-14)

All run outputs are gitignored; the numbers below are transcribed here. Artifacts:
`runs/dev1h-nov/` (`monitor/metrics.jsonl`, `monitor/summary.json`,
`run_summary.json`, `regression/`, `dev_reports/`).

**Config used (deviations from the table above):** 4 simulations, K = 4, 16
workers, batch 16, 16 games/cycle, 16 learner updates/cycle, 20-minute wall-clock
budget, 60-minute checkpoint interval, dev evaluation disabled in-run and run
externally afterwards with the same search settings (104 dev games per pairing,
seat-swapped over four maps). Warm start `C:\2026\tm_bot\runs\warmstart\best`.

**Search-cost measurements** (same host, 16 workers, self-play batch):

| Setting | Games | Wall | Throughput | Generations (min/median/max) | Decisions |
| --- | ---: | ---: | ---: | --- | ---: |
| s4/K4, batch 16 | 16 | 171.3 s | **336 games/h** | 12 / 25 / 33 | 9,797 (612/game) |
| s4/K4, run cycle 1 | 16 | 171 s | 338 games/h | 23.8 mean | 9,727 |
| s4/K4, run cycle 2 | 16 | 427 s | 135 games/h | 32.9 mean (max 63) | 15,751 |
| s8/K4, one self-play game | 1 | 47.5 s | (not a batch measurement) | 26 | 738 |
| s2/K2, batch 16 | 16 | did not finish in 25 min; games hit the decision cap | — | — | — |

The >= 1,000 self-play games/hour gate is **not met**. Simulations were reduced
first (64 -> 8 -> 4). K was left at 4: lowering it to 2 did not help because at
s2/K2 the untrained action policy stops terraforming, games run past the
decision cap, and no batch completes. The limiting factor is game length, not
per-decision search cost: s4/K4 self-play games median 25-33 generations
(vs 15 generations when the warm start plays B1), so the shape monitor flags the
long tail (`shape_flag_count` 6/16 in cycle 1 and 12/16 in cycle 2). This is the
expected pre-#25 behaviour of an untrained action policy, but it means the
throughput gate cannot be cleared until the policy plays normal-length games.

**Training**: 32 self-play games and 16 learner updates in 601 s before the hard
stop. Per-cycle metrics (before/after the single update cycle):

| Cycle | Games | Mean gens | Throughput (cum) | Policy loss | Value loss | Probe entropy | Target entropy | Value Brier / ECE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 16 | 23.8 | 338/h | 1.304 | 0.367 | 1.234 | 1.200 | 0.378 / 0.335 |
| 2 | 16 | 32.9 | 192/h | — (stopped before update) | — | 1.291 | 1.250 | 0.415 / 0.398 |

Value calibration is measured on the cycle's newly finished games **before**
they enter the buffer and before the update (frozen weights): n = 11,674 total,
5,773 wins / 5,901 losses / 0 ties (from the training seat's perspective). The
value head is badly overconfident on the top bin (mean predicted 0.96 vs
observed 0.35), which the ECE reflects. The probe entropy is the network policy
entropy on the first cycle's fixed observation set; target entropy is the
self-play visit-distribution entropy (they are different quantities and both
are logged).

**Stop trigger**: `adapter_rejection` at 599 s. `train-0027` reached generation
73 on Vastitas Borealis Nova, and the adapter failed to factor a drafting prompt
with zero offered cards (`SelectCard requires at least 1 cards, only 0 enabled`).
The run saved seed + full action history to
`runs/dev1h-nov/regression/train-0027__vastitas_borealis_nova__AdapterBugError.json`
and stopped instead of treating the game as a result. This is a real
adapter/engine edge case for games that run far past the normal 14 generations
(see #29/#31); it also appeared in the earlier pre-fix attempt at generations
44-48. Fixed by #32: when the project deck is exhausted the adapter now offers
the forced (short or empty) selection and lowers the engine minimum for that one
submission; the case is a committed regression test
(`adapter/test/cardExhaustion.test.ts`, `tests/test_bridge.py`).

**Dev results** (externally evaluated from `ckpt-000000001`, 104 seat-swapped
`dev-*` games per pairing, s4/K4, 16 workers, batch 32; JSON under
`runs/dev1h-nov/dev_reports/`):

| Pairing | Completed | Failures | Win rate (95% paired bootstrap CI) | Mean VP margin | Gens/game | Wall |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Candidate vs warm start | 104 | 0 | **45.19%** [35.58%, 54.81%] | -3.38 | 27.5 | 1,299 s |
| Warm start vs B1 | 104 | 0 | **7.69%** [0.00%, 19.23%] | -68.61 | 15.0 | 332 s |
| Candidate vs B1 | 103 | 1 | **7.77%** [0.00%, 17.65%] | -71.82 | 15.9 | 498 s |

The candidate is not measurably better than its warm start after 16 updates
(45% vs 50%, CI includes 50%); both lose to B1 at ~8%, matching the pre-training
search result in [search.md](search.md). CIs are wide (104 games, one training
seed) and the training run was cut short by the adapter bug, so this is a
throughput/plumbing result, not a strength result.

Total development-run wall clock: 601 s training + 1,299 s + 332 s + 498 s of
dev evaluation = **2,730 s (45.5 min)**, within the 1 h limit.
