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
| Wall-clock budget | 24 h | Hard stop when reached (checked per cycle) |
| Simulations / searched decision | 64 | `--sims` |
| Worlds (K) | 4 | `--worlds` |
| Opponents | 60% latest / 30% last-10 pool / 10% B1 | `OpponentSampler` |
| Replay buffer | last 200,000 decisions | `ReplayBuffer` |
| Learner | AdamW, lr 5e-4, wd 2e-4, grad clip 1.0, batch 256 | same optimizer settings as warm start |
| Checkpoints | every 30 min | numbered `step-XXXXXXXXX/`, plus `best/` |
| Retained checkpoints | best + last 10 | older ones pruned |
| Disk cap | 50 GB | oldest non-best checkpoints deleted first |
| Dev evaluation | 100 `dev-*` games vs B1 and 100 vs previous best | selected on `dev-*` only |
| Plateau | 12 checkpoints without a new best | |
| Collapse (win rate) | >15 points below best for 3 checkpoints | |
| Collapse (entropy) | <10% of warm-start entropy within 2 h | |
| Determinism spot check | every 1,000 self-play games | replay seed + action history |

Resource limits live in `ResourceLimits`: workers ≤ 6 GB RAM (one worker per
game, capped by the worker count), learner + buffer ≤ 10 GB, VRAM ≤ 12 GB (a
`torch.cuda.set_per_process_memory_fraction` guard when CUDA is available), and
run outputs ≤ 50 GB under the gitignored `runs/` directory.

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
from the replay buffer, runs a masked policy cross-entropy plus value BCE, and
reports policy/value loss, predicted-policy entropy, Brier score, and value
accuracy. A non-finite loss immediately stops the cycle and rolls the model back
to the last periodic checkpoint.

## Dev evaluation and best selection

`dev-*` games are built by `build_dev_games` exactly `N` at a time: each
`(seed, map)` is played once, alternating the candidate's seat so seats are
balanced, and the candidate plays both B1 and the previous `best/` checkpoint.
Win rate, 95% paired bootstrap CI, and mean VP margin are reported by the
existing evaluation report code. Best is selected on `dev-*` win rate vs B1
only; `eval-*` seeds are never read or written by a run.

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
| `budget` | wall-clock budget reached | stop |

A regression case is `regression/<seed>__<map>__<error>.json` containing the
seed key, map, error, decision count, and the full ordered action history.

## Monitoring

`MonitorLogger` writes one JSON line per self-play cycle and per checkpoint to
`monitor/metrics.jsonl`, plus `monitor/summary.json`. Cycle records contain:
policy entropy, policy/value/total losses, value Brier and accuracy,
generations/game, VP margin, decisions/game, rejections/failures, decision
latency p50/p95, throughput, and game-shape flags (generations outside 6-25).
Checkpoint records contain dev win rates and CIs vs B1 and the previous best.
Determinism spot checks replay a finished game from its seed and recorded action
history and compare generations, scores, winner, and tie
(`replay_game`, `compare_replay`).

## Resume

`state.json` stores the cycle, learner step, self-play game count, best dev win
rate, elapsed wall clock, seed cursor, opponent pool, and the last checkpoint.
`state.pt` stores the model/optimizer and the replay buffer. `--resume` restores
the model from the last good checkpoint (falling back to the newest numbered
checkpoint, then `best/`), reopens the metrics stream, restores the seed cursor
so training never reuses `dev-*`/`eval-*` seeds, and continues the run.

## Results

### Throughput measurement (1 h development run)

The 1 h development run used the config below; `runs/dev1h/` (gitignored) holds
the full metrics. Results are recorded here because run outputs stay out of git.

| Setting | Default (#10) | Development run | Why |
| --- | --- | --- | --- |
| simulations | 64 | **4** | throughput gate; see below |
| worlds (K) | 4 | 4 | unchanged |
| workers | 16 | 16 | unchanged |
| games/cycle | 32 | 16 | keep cycles short enough to checkpoint and stop promptly |
| checkpoint interval | 30 min | 20 min | fit three dev evaluations into the 45 min budget |
| dev games per opponent | 100 | 8 | time budget; CIs are correspondingly wide |

Search cost measurements on the same host (16 workers, batch of 16 self-play
games, `max_decisions_per_game=8000`):

| Setting | Games | Wall | Throughput | Generations (min/median/max) | Decisions |
| --- | ---: | ---: | ---: | --- | ---: |
| s8/K4 (eval reference) | 400 eval games | 2,051 s | ~700/h | 14.8 mean | ~490/game |
| s4/K4 | 16 | 171.3 s | **336/h** | 12 / 25 / 33 | 612/game |
| s2/K2 | 16 | > 25 min (measured attempt) | — | games did not finish (step cap) | — |

Reducing simulations from 64 to 4 was necessary to make games finish at all in
self-play; at s2/K2 the untrained action policy no longer terraforms and games
run to the decision cap. Note that in self-play **both** seats come from the
warm-start network, so games are much longer than the s8/K4 evaluation against
B1 (which was 14.8 generations/game); at s4/K4 self-play games median 25
generations, which is already at the top of the #10 6-25 sanity band and is
flagged by the shape monitor for the long tail. The 1,000 games/hour speed gate
is **not met** (336 games/hour measured); the limiting factor is game length,
not search cost (p50 searched-decision latency is 14 ms). Recorded per the
ticket: simulations were reduced first, then K was left at 4 because lowering K
did not help while games failed to terminate.

Run results (throughput, losses, dev results with CIs, stop triggers) are
recorded below from `runs/dev1h/monitor/summary.json` and the checkpoint rows.
