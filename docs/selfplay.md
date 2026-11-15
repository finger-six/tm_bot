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
| Simulations / searched decision | 64 | `--sims` |
| Worlds (K) | 4 | `--worlds` |
| Opponents | 60% latest / 30% last-10 pool / 10% B1 | `OpponentSampler` |
| Replay buffer | last 200,000 decisions | `ReplayBuffer` |
| Learner | AdamW, lr 5e-4, wd 2e-4, grad clip 1.0, batch 256 | same optimizer settings as warm start |
| Checkpoints | every 30 min | monotonic `ckpt-XXXXXXXXX/` save counter, plus `best/` |
| Retained checkpoints | best + last 10 | older ones pruned |
| Disk cap | 50 GB | oldest non-best checkpoints deleted first; a cap hit stops the run as `resource_limit` |
| Dev evaluation | 100 `dev-*` games vs B1 and 100 vs previous best | seat-swapped pairs; selected on `dev-*` only |
| Plateau | 12 checkpoints without a new best | |
| Collapse (win rate) | >15 points below best for 3 checkpoints | |
| Collapse (entropy) | <10% of warm-start network entropy within 2 h | |
| Determinism spot check | every 1,000 self-play games | replay seed + action history |

Resource limits live in `ResourceLimits`: learner + buffer <= 10 GB RSS and
VRAM <= 12 GB (checked every cycle; a breach stops the run as `resource_limit`),
run outputs <= 50 GB under the gitignored `runs/` directory, and the configured
worker count bounds engine workers (per-worker RSS is not measured separately;
it is bounded by the 16-worker default). When CUDA is available, VRAM is also
soft-capped with `torch.cuda.set_per_process_memory_fraction`.

Run outputs (`runs/<run-name>/`): `config.json`, `state.json` + `state.pt`
(resume), `checkpoints/`, `best/`, `monitor/metrics.jsonl`, `monitor/summary.json`,
`run_summary.json`, and `regression/` cases. `.gitignore` contains `/runs/`, so
none of these are committed.

## Self-play

Each game is one `train-*` seed and one supported map. The training seat is
alternated; the other seat is an opponent sampled 60% from the latest network,
30% from the last 10 frozen checkpoints, and 10% `b1-v1`. With an empty pool (a
fresh run) the pool share falls back to the latest network.

`RecordingSearchAgent` ([`recorder.py`](../tm_bot/runs/recorder.py)) wraps both
seats and records the ordered `(seat, candidate)` action history; it records
`(observation, choice, candidate distribution)` rows **only for the training
seat**. The runner still passes only `(observation, choice[, sampler])` to
agents, so a training row can only contain what that seat legitimately saw.
The training input is the acting seat's encoded `ActorObservation`; opponent
private cards, deck order, and RNG never enter it (the adapter's actor
observation has no such fields, and `test_opponent_private_cards_never_enter_the_acting_seat_encoding`
checks the encoding is seat-scoped).

### Targets

- **Policy**: normalized root visit counts at searched decisions. Decisions
  handled by the policy head (nested payments, spaces, players, amounts,
  initial corporation/prelude picks) are kept and targeted from the policy
  head's own distribution; forced single-candidate decisions are one-hot. This
  keeps the whole game in the training set instead of only searched roots.
- **Value**: terminal utility from the acting seat's perspective: win `+1`,
  loss `-1`, exact tie `0`. The value head predicts win probability, so the
  loss uses `(u + 1) / 2` as the target.

`AlphaZeroLearner` ([`learner.py`](../tm_bot/runs/learner.py)) samples decisions
from the replay buffer and runs a masked policy cross-entropy plus value BCE.
Value-head calibration (Brier, ECE, reliability bins, win/loss/tie counts) is
measured by `compute_value_calibration` on each cycle's newly finished games
**before** they enter the buffer and before the optimizer step, with frozen
weights, so the cycle record is out-of-sample for that update. Training records
carry only training-batch losses (policy/value/total means); calibration is not
mixed into them. Every parameter tensor is checked for finiteness after each
optimizer step; a NaN/Inf loss or parameter stops the cycle and rolls the model
and optimizer back to the last periodic checkpoint (a failed rollback is
reported in the stop decision rather than swallowed).

## Dev evaluation and best selection

`dev-*` games use `build_paired_specs` from
[`tm_bot/eval/pairing.py`](../tm_bot/eval/pairing.py): every `(seed, map)` is
played twice with the candidate in opposite seats, so the paired bootstrap CI is
valid. The schedule is rounded up to whole seat-swapped pairs (104 games for a
requested 100). The candidate plays both B1 and the previous `best/` checkpoint;
before the first `best/` exists, the resolved warm-start checkpoint (recorded in
`config.json`) is the previous best, so the first checkpoint still gets a real
second pairing. Win rate, 95% paired bootstrap CI, and mean VP margin are
reported by the existing evaluation report code. Best is selected on `dev-*`
results only (`is_new_best` in [`stopping.py`](../tm_bot/runs/stopping.py),
shared with the plateau and collapse rules): a higher win rate vs B1 wins; on an
exact tie (e.g. 0% vs B1 early in training) the checkpoint becomes the new best
only if it beats the previous best with the 95% CI lower bound above 50%.
`eval-*` seeds are never read or written by a run.

## Stopping rules

`StoppingMonitor` ([`stopping.py`](../tm_bot/runs/stopping.py)) is fed metric
events and returns a typed `StopDecision`; each rule has a synthetic-stream
test in [`tests/test_runs.py`](../tests/test_runs.py):

| Rule | Trigger | Action |
| --- | --- | --- |
| `adapter_rejection` | `AdapterBugError` in a self-play game | hard stop, write regression case |
| `engine_error` | bridge/worker error | hard stop, write regression case |
| `step_cap` | game exceeded `max_decisions_per_game` | hard stop, write regression case |
| `nonfinite_loss` | NaN/Inf learner loss | roll back to last good checkpoint, stop |
| `collapse_win_rate` | dev win rate vs B1 >15 points below best for 3 checkpoints | stop |
| `collapse_entropy` | policy entropy <10% of warm-start value within 2 h | stop |
| `plateau` | no new best for 12 checkpoints | stop |
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
p50/p95, throughput, determinism check result, and untruncated game-shape flag
counts (generations outside 6-25) with up to 20 example games.
Checkpoint records contain dev win rates and CIs vs B1 and the previous best.
Determinism spot checks replay a finished game from its seed and recorded action
history and compare generations, scores, winner, and tie (`replay_game`,
`compare_replay`); the check counter is persisted across resume.

The entropy collapse rule uses exactly one quantity: the network policy entropy
on the first cycle's fixed observation set (`probe_policy_entropy`). That probe
is captured before any update, stored in `state.pt`, and re-evaluated with the
current weights each cycle; the self-play target entropy is logged separately
for information and never compared to the network probe.

## Resume

`state.json` stores the cycle, learner step, self-play game count, best dev win
rate, elapsed wall clock, training time and dev-eval time, seed cursor, opponent pool, determinism-check counter,
entropy probe baseline, and stopping-rule counters. `state.pt` stores the
model/optimizer and the replay buffer (plus the fixed entropy probe). `--resume`
restores the model from the last good checkpoint (falling back to the newest
numbered checkpoint, then `best/`), restores the optimizer and replay buffer
independently (a missing optimizer state does not skip the buffer), reopens the
metrics stream, restores the seed cursor and stopping counters, and continues
the run. Checkpoints are named by a monotonic save counter (`ckpt-000000001`,
`ckpt-000000002`, ...) so two saves with the same learner step never collide.

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
