# First Milestone Report (#26)

This report covers the 24 h milestone runs of the main agent (IS-MCTS + policy/value
network, [selfplay.md](selfplay.md)) and of the Jev-inspired comparator (typed-candidate
scorer + PPO, [comparator.md](comparator.md)), then the final evaluation of both best
checkpoints on `eval-*` seeds.

**Verdict: the milestone is NOT met by either agent.** Neither agent won a single one of
400 `eval-*` games against B1. Both also lose to their own warm-start checkpoint, so
self-play made each of them weaker.

The milestone criterion comes from spec #13: 400 evaluation games, 0 failures, and a
lower 95% CI bound above 50% against B1 **and** against the agent's own warm start.

All run and report directories are gitignored (`runs/`, `data/reports/`), so every
number below is transcribed from them. Final evaluation reports:
`data/reports/milestone/<pairing>/report.{md,json}`. Calibration:
`data/reports/milestone/calibration/`.

## Verdict per agent

| Criterion (#13) | Main agent | Comparator |
| --- | --- | --- |
| 400 `eval-*` games, 0 failures | yes (all pairings 400/400, 0 failures) | yes (all pairings 400/400, 0 failures) |
| vs B1: lower 95% CI > 50% | **no**: 0.00% [0.00, 0.00] | **no**: 0.00% [0.00, 0.00] |
| vs own warm start: lower 95% CI > 50% | **no**: 30.00% [25.00, 35.25] | **no**: 11.50% [8.00, 15.25] |
| **Milestone** | **not met** | **not met** |

## Final evaluation (`eval-*`, 400 games per pairing)

Every pairing uses the default protocol from [evaluation.md](evaluation.md):
- 50 `eval-*` seeds × 4 supported maps × 2 seats.
- Paired bootstrap 95% CI (10,000 resamples, seed 0).
- 16 workers, 20,000-decision step cap.

Settings:
- Code: main @ 1dce4e9, so B1 is **`b1-v2`** (#34 fix).
- Main-agent agents (trained and warm start) play at the run's search settings: 4
  simulations, K = 4 worlds, c_puct 1.5, temperature 0, action temperature 1, wave
  size 4 (`tm_bot.search:make_warmstart_search_agent?...`).
- Comparator agents play the sampled evaluation policy
  (`tm_bot.comparator:make_comparator_agent?...`).

Win rate is for the first-named agent.

| Pairing | Games | Failures | Record | Win rate [95% CI] | Mean VP margin | Mean gens | Wall |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| Main best vs B1 | 400 | 0 | 0-400-0 | **0.00%** [0.00, 0.00] | -88.75 (40.2 vs 128.9) | 13.57 | 1,224 s |
| Main best vs main warm start | 400 | 0 | 120-280-0 | **30.00%** [25.00, 35.25] | -17.30 (85.2 vs 102.5) | 27.28 | 4,577 s |
| Comparator best vs B1 | 400 | 0 | 0-400-0 | **0.00%** [0.00, 0.00] | -84.92 (40.9 vs 125.8) | 13.46 | 464 s |
| Comparator best vs comparator warm start | 400 | 0 | 46-354-0 | **11.50%** [8.00, 15.25] | -47.04 (66.5 vs 113.6) | 39.11 | 2,224 s |
| **Head to head**: main best vs comparator best | 400 | 0 | 365-35-0 | **91.25%** [88.25, 94.00] | +56.80 (125.6 vs 68.8) | 41.13 | 4,904 s |
| *Context*: main warm start vs B1 | 400 | 0 | 0-400-0 | 0.00% [0.00, 0.00] | -87.64 | 13.69 | 1,156 s |
| *Context*: comparator warm start vs B1 | 400 | 0 | 0-400-0 | 0.00% [0.00, 0.00] | -93.42 | 13.83 | 555 s |

The two context rows are not milestone pairings. They show that neither warm start wins
against `b1-v2` either. Self-play has not moved either agent off the floor against B1.
Against B1 the trained main agent averages a VP margin of -88.8, against -87.6 for its
warm start, and the comparator -84.9 against -93.4. In head-to-head and
vs-warm-start games, both agents play learned policies against each other and the games
run 27-41 generations. Against B1 they run about 13.5 generations, because B1 ends the
game by terraforming.

### Per-map results

Win rate [95% CI] for the first-named agent. Every map has 100 games and 0 failures.

| Map | Status | Main vs B1 | Main vs warm start | Comparator vs B1 | Comparator vs warm start | Main vs comparator |
| --- | --- | --- | --- | --- | --- | --- |
| Tharsis | supported | 0.00% [0, 0] | 37.00% [28.00, 46.00] | 0.00% [0, 0] | 6.00% [1.00, 12.00] | 90.00% [83.00, 96.00] |
| Elysium | supported | 0.00% [0, 0] | 27.00% [19.00, 36.00] | 0.00% [0, 0] | 19.00% [12.00, 26.00] | 91.00% [85.00, 96.00] |
| Hellas | supported | 0.00% [0, 0] | 28.00% [19.00, 38.00] | 0.00% [0, 0] | 12.00% [5.00, 20.00] | 92.00% [87.00, 97.00] |
| Vastitas Borealis Nova | supported | 0.00% [0, 0] | 28.00% [19.98, 37.00] | 0.00% [0, 0] | 9.00% [4.00, 14.00] | 92.00% [86.00, 97.00] |
| Amazonis Planitia | **not supported** | not supported | not supported | not supported | not supported | not supported |

Mean VP margins vs B1 by map (Tharsis / Elysium / Hellas / Vastitas):
- Main agent: -84.8 / -89.0 / -92.8 / -88.4.
- Comparator: -80.6 / -84.6 / -87.9 / -86.5.

### Failures

There were 0 failures (step cap, adapter or bridge errors, illegal or raising agents) in
any final-evaluation game. There were also 0 failures and 0 adapter rejections across
both training runs (1,248 + 6,150 self-play games), and both `regression/` directories
are empty. The main run's determinism spot check (train-1023, Vastitas Borealis Nova)
replayed identically.

### Training seeds

**1 independent training seed per agent** (run seed 42). That is **fewer than 3**, so
none of these results says anything about seed-to-seed variance. The eval reports carry
the same warning.

## Comparator calibration

Report: `python -m tm_bot.comparator.calibration` on the best checkpoint, 400 games per
pairing, seat-swapped. These run on **`dev-*` seeds**, because the calibration CLI is
dev-only by design (#25). The `eval-*` win rates above are the milestone numbers.

Temperature T = 0.9878 is fitted on the warm-start validation split only (#13 story 51).
Reliability diagrams: `data/reports/milestone/calibration/dev_vs_{b1,warmstart}_reliability.svg`.

| Opponent | Games (dev) | Comparator win rate | Decisions scored | Brier raw / scaled | NLL raw / scaled | ECE raw / scaled | Base-rate Brier |
| --- | ---: | --- | ---: | --- | --- | --- | ---: |
| B1 (`b1-v2`) | 400 | 0.0% [0.0, 0.0] | 68,212 | 0.232 / 0.233 | 0.710 / 0.715 | 0.321 / 0.321 | 0.000 |
| Comparator warm start | 400 | 11.5% [9.0, 14.0] | 147,737 | 0.639 / 0.641 | 2.250 / 2.275 | 0.684 / 0.685 | 0.111 |

Base-rate Brier is the score of a constant prediction equal to the decision-weighted win
frequency (0 vs B1, 0.127 vs warm start). The head does far worse than that constant
against both opponents, and temperature scaling changes almost nothing because T ≈ 1.

The head is **badly overconfident**. Against the warm start, 95k of 148k decisions
predict ≥ 0.9 win probability, but the observed win rate in that bin is 13.8% (13.7%
after temperature scaling). Against B1, 9.8k decisions predict ≥ 0.9 and the comparator
never wins.

This matches the self-play anomaly below. In training, the learner won 89% of its games
(5,465-685), mostly against greedy copies of itself. The head therefore learned "I am
winning" from positions that do not carry over to games against B1 or the warm start.
The 1 h development run, before this drift, had Brier 0.089 vs B1
([comparator.md](comparator.md)).

## Runs

Both runs used the run orchestration from [selfplay.md](selfplay.md):
- 16 workers, 30-min checkpoints on the training clock.
- 100 (104 seat-swapped) `dev-*` games vs B1 and vs the previous best per checkpoint.
- 24 h training-time budget.
- Stopping rules from #10/#13.

Both ran on one machine, one after the other: Windows 11, 20-CPU Intel host, one RTX
4080 SUPER.

### Stopping

| | Main agent | Comparator |
| --- | --- | --- |
| Run | `runs/milestone-main` | `runs/milestone-comparator` (attempt 2) |
| Warm start | `runs/warmstart/best` | `runs/comparator-warmstart-full/best` (full 4 epochs) |
| Started | 2026-11-16 02:15 CST | 2026-11-15 08:21 CST |
| Stop | **plateau**: 12 checkpoints without a new best, cycle 39 | **plateau**: 12 checkpoints without a new best, cycle 82 |
| Best checkpoint | cycle 15 (`best/`, 480 self-play games), dev vs `b1-v1` 9/104 = 8.65% | cycle 42 (`best/`, 3,150 games), dev vs `b1-v1` 12/104 = 11.54% |
| 72 h extension | not applicable (stopped at 10.96 h of training) | not applicable (stopped at 12.81 h of training) |

Both runs stopped under a documented rule (#26 acceptance criterion 1). The plateau rule
fired well before 24 h, so the extension rule (still improving on `dev-*` at hour 24)
never applied.

### Compute

| | Main agent | Comparator |
| --- | ---: | ---: |
| Training clock (wall minus in-run dev eval) | **10.96 h** (39,444 s) | **12.81 h** (46,109 s) |
| Total wall clock | **28.54 h** (102,740 s) | **17.85 h** (64,277 s) |
| In-run dev evaluation | 17.58 h (63,296 s; 17 checkpoints, about 62 min each) | 5.05 h (18,168 s; 22 checkpoints, about 14 min each) |
| GPU-hours (1 × RTX 4080 SUPER held for the whole run) | 28.5 | 17.9 |
| Self-play games | 1,248 (114 per training hour) | 6,150 (480 per training hour) |
| Decisions generated | 1,282,932 | 6,511,229 |
| Learner steps | 624 (16 updates × 39 cycles) | 10,496 PPO minibatch updates |
| Simulations per searched decision | 4 sims × K = 4 worlds (about 33% of decisions searched) | 0 (no search) |
| Parameters | 4,459,330 | 2,501,026 |
| Per-decision latency, eval vs B1 (p50 / p95) | 16.9 / 28.6 ms (searched only: 19.8 / 30.4 ms) | 4.8 / 6.1 ms |
| Warm start cost | 3,295 s (4 epochs, [warmstart.md](warmstart.md)) | 1,310 s (4 epochs) |
| Final eval wall, both agents combined (5 pairings + 2 context + 2 calibration) | about 5.0 h (17,860 s) | (combined) |

The two clocks follow the #26 decision (issuecomment-6892016073). The 24 h budget, the
30-min checkpoint interval and the entropy window count training time only. The table
reports the wall clock alongside it.

The main agent's wall clock is 2.6× its training time because each IS-MCTS dev
evaluation (208 games) took about 62 min, against about 39 min of training between
checkpoints.

**Throughput gate:** this is the reported negative result (#26 decision,
issuecomment-6891028833). The #10 gate requires 1,000 self-play games per hour; the main
agent reached 114 per training hour. Sims had already been lowered to 4 (K stays 4). The
cause is game length (mean 23-54 generations per cycle), not search cost.

### Memory

These figures come from the memory logs `runs/milestone-*.mem.csv`, sampled every
120 s by an external sampler (not in the repo), for the run's Python process. All
figures are decimal GB (MB / 1000).
- Comparator attempt 2: private memory median 11.0 GB, p95 14.4 GB, peak 17.0 GB;
  working set (RSS) at most 4.7 GB. Free commit memory never fell below 59.4 GB.
- Main run: private memory median 17.9 GB, p95 22.9 GB, peak 33.3 GB; working set
  median 6.1 GB, p95 11.2 GB, peak 18.5 GB. Free commit memory never fell below 34.2 GB.
- Main-run memory grew while the replay buffer filled and then stayed in that band,
  and neither run failed for lack of memory. We did not test separately for a slow leak.
- **The 10 GB learner RAM limit was never enforced.** The main run's working set was above
  10 GiB in 171 of 855 samples. The run did not stop, because the Windows RSS probe
  `current_rss_bytes()` returns `None` on this host, so the RAM half of the guard never
  fires (filed as #35). The 12 GB VRAM half (`torch.cuda.memory_allocated`) works, and
  neither run stopped on it.

### Dev-checkpoint history

Dev win rate vs `b1-v1` (see the B1 bug below) per checkpoint, and vs the previous best:

| Main: cycle (training h) | vs B1 | vs prev best | Comparator: cycle (training h) | vs B1 | vs prev best |
| --- | --- | --- | --- | --- | --- |
| 4 (0.51) | 7.7% | 0.42 | 4 (0.51) | 7.7% | 0.62 |
| 15 (3.03) **best** | 8.7% | 0.38 | 23 (2.72) | 7.7% | 0.26 |
| 25 (6.30) | 8.7% (tie) | 0.48 | 40 (5.10) | 7.7% | 0.06 |
| 35 (9.71) | 8.7% (tie) | 0.39 | 42 (5.64) **best** | 11.5% | 0.07 |
| 39 (10.91) | 7.7% | 0.49 | 47 (6.72) | 10.6% | 0.68 |
| all others | 7.7% | 0.35-0.56 | all others | 7.7-8.7% | 0.21-0.70 |

## Incidents and anomalies

### #33: comparator attempt 1 ran out of memory

The first comparator run (started 2026-11-15 04:17 CST) stopped at cycle 2 (150 games,
about 71 min of training) with `engine_error`. Node workers died with "Zone Allocation
failed - process out of memory", because Python had used up the machine's commit memory.

Root cause: the self-play recorder kept every raw decoded observation (about 423 KB per
record), which reached 25-37 GB per 75-game cycle. #33 now encodes each observation when
it is recorded and keeps no raw copy. That brought peak memory per 75-game cycle down to
9.1 GiB, and the fix does not change play: attempt 2's cycle 1 had the same mean
generations as attempt 1's (53.68).

The comparator run was restarted from scratch with the same settings. Attempt 1 is kept
at `runs/milestone-comparator-oom-attempt1`. Both milestone runs used the #33 code
frozen at 2fb4921 (a worktree, so main could move on). The run artifacts do not record
a git SHA; this comes from the launch setup (cwd and `PYTHONPATH` pointed at that
worktree). The #33 review follow-ups merged later do not change training behavior.

### #34: B1 never took a corporation's first action (`b1-v1`)

**The bug:** `b1-v1` always chose Pass over "Take first action of <corp> corporation".
It then passed every generation and scored about 22 VP. This is the rule bug found during
the milestone. Its regression tests are
`tests/test_agents.py::test_b1_takes_corporation_first_action_instead_of_passing` and
`::test_b1_finishes_first_action_corporation_game_normally`. `B1_VERSION` is now
`b1-v2`, which beats B0 400/400.

**Effect on in-run dev evaluation:** both runs had started before the fix, so their
in-run dev evaluation used `b1-v1`.
- On `dev-0006` and `dev-0009` with B1 in seat 0, `b1-v1` holds the affected corporation
  and loses on every map. That is **8 free games out of 104** for any opponent; B0 also scores
  exactly 8/104.
- So the main agent's best (9/104) has **1** genuine dev win against B1.
- The comparator's best (12/104) has **4** (dev-0004, dev-0007).
- Every checkpoint at 7.7% won no genuine games.

**Decision:** the running main-agent run was not restarted. The offset is a constant 8
games, so it does not change the checkpoint ranking, and B1 is only 10% of self-play
opponents. All final evaluation and calibration use **`b1-v2`**, from main @ 1dce4e9.
The eval reports record `b1-v2` as the agent version. The calibration reports record only
`opponent: "b1"`, but they ran from the same commit.

### Checkpoint selection sat at the floor

**Neither run got clear of the floor:** both selected their best checkpoint on dev win
rate vs B1, and neither got meaningfully above the 8/104 floor. The main run's "best" is
one genuine dev game above the other checkpoints, and the plateau rule stopped it.

**The comparator's "best" (cycle 42) was a poor checkpoint:**
- It won only 6.7% of dev games against the previous best.
- Final evaluation confirms it: 11.5% against its own warm start.

So when the win rate vs B1 sits at the floor, best-checkpoint selection mostly picks
noise. The vs-previous-best signal was the more informative one, but the rules use it
only as a tie-break when win rates vs B1 are exactly equal (`is_new_best` in
`tm_bot/runs/stopping.py`). In the main run, that tie-break kept the cycle-15 checkpoint
against the two later ties (cycles 25 and 35: vs-previous-best CI lower bounds 0.375 and
0.288). We recorded this and changed no rule (#10).

### Comparator long-game anomaly (cycles 42-43)

**What happened:**
- Comparator self-play mean generations went from 50-73 (cycles 31-40) to 136 and 106
  (cycles 42-43).
- The longest game reached 275 generations (cycle 42). The largest decision count
  among the recorded shape flags (capped at 20 per cycle) was 3,326, in a 240-generation
  game in cycle 43.
- Generations then dropped to 91 (cycle 44) and stayed at 52-82 from cycle 46 on.

Across the run, 5,408 of 6,150 games fall outside the 6-25 generation shape window. No
game hit the 20,000-decision step cap.

**Likely cause** (inferred from the setup; we traced no game from cycles 42-43): the
training seat samples its policy (temperature 1), but its comparator opponents (latest
and pool) act greedily ([comparator.md](comparator.md)). Greedy comparators stall
terraforming, which is the looping failure that made greedy dev evaluation unusable in
#25. The sampling learner beats these stalled copies by large margins (89% of self-play
games; median per-cycle mean VP margin +83.7, and negative in only 1 of 82 cycles). The
PPO reward therefore mostly measures exploiting a stalling opponent.

This fits the drop against the warm start and the overconfident win-probability head.
Per #10, game-shape anomalies trigger an investigation, not a training change; it is
reported here and not fixed in this milestone.

### Early greedy-opponent anomaly

In attempt 1's first cycle, the learner beat the greedy `latest` opponent 71/71
(+113 VP), and games averaged 54 generations (71/75 flagged by the shape monitor).
Attempt 2 reproduced the same cycle exactly.

In the 1 h development run the same setup fell to about 17 generations within about 6
cycles. In the milestone run it did not: games stayed long all run (see above). It is
the same mechanism as the long-game anomaly.

### Main-agent game shape

Main-agent self-play is symmetric (both seats search with the same network). The learner
won 569 and lost 679 (46%). Mean generations per cycle rose from 23 to 35-54, with a
maximum of 155, and 950 of 1,248 games were shape-flagged. Policy entropy went from 1.19
to 1.27, so it did not collapse. The run never hit the step cap and never stopped on a
collapse rule.

## Reproduce

```bash
# Final evaluation (main @ 1dce4e9 or later, b1-v2). S = run search settings.
S="num_simulations=4&num_worlds=4&c_puct=1.5&temperature=0.0&action_temperature=1.0&wave_size=4&device=cuda"
MAIN="tm_bot.search:make_warmstart_search_agent?checkpoint=runs/milestone-main/best&$S"
MAINWS="tm_bot.search:make_warmstart_search_agent?checkpoint=runs/warmstart/best&$S"
CMP="tm_bot.comparator:make_comparator_agent?checkpoint=runs/milestone-comparator/best&device=cuda"
CMPWS="tm_bot.comparator:make_comparator_agent?checkpoint=runs/comparator-warmstart-full/best&device=cuda"
python -m tm_bot.eval --a "$MAIN" --b b1 --games 400 --namespace eval --workers 16 \
  --max-decisions 20000 --a-training-seeds 1 --out data/reports/milestone/main_vs_b1
# likewise: main_vs_ws ($MAIN vs $MAINWS), cmp_vs_b1, cmp_vs_ws ($CMP vs $CMPWS),
# main_vs_cmp ($MAIN vs $CMP), and the context rows mainws_vs_b1, cmpws_vs_b1;
# pass --a-training-seeds 1 for every learned A and --b-training-seeds 1 for every
# learned B, so each report carries the fewer-than-3-seeds warning.
python -m tm_bot.comparator.calibration --checkpoint runs/milestone-comparator/best \
  --opponent b1 --games 400 --workers 16 --device cuda \
  --out data/reports/milestone/calibration --name dev_vs_b1
python -m tm_bot.comparator.calibration --checkpoint runs/milestone-comparator/best \
  --opponent runs/comparator-warmstart-full/best --games 400 --workers 16 --device cuda \
  --out data/reports/milestone/calibration --name dev_vs_warmstart
```

Training commands: [selfplay.md](selfplay.md) and the #26 run-settings comment
(issuecomment-6892868152). Both runs used `--budget-hours 24 --workers 16
--checkpoint-minutes 30 --dev-games 100 --device cuda`. The main agent also used
`--sims 4 --worlds 4`. The comparator also used `--agent-kind comparator
--games-per-cycle 75 --updates-per-cycle 128 --learner-batch-size 256
--learning-rate 0.0003`.
