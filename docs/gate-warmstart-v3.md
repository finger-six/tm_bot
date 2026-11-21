# Pre-self-play gate: `warmstart-v3` (main agent)

**Verdict: FAIL** (27/30 gating checks pass). Thresholds: #38 §5; `val` split and `dev-*` seeds only.

- Checkpoint: `C:\2026\tm_bot\runs\warmstart-v3\best`
- Validation metrics: `runs\warmstart-v3\metrics.json`
- v1 opponent: `C:\2026\tm_bot\runs\warmstart\best`
- Candidate setting: `light search num_simulations=8, num_worlds=4, c_puct=1.5, temperature=0.0, action_temperature=1.0, wave_size=4`
- Raw-policy diagnostic: `search agent with num_simulations=0 (policy head)`
- Max decisions per game: `20000`
- Workers / batch size: `8 / 32`
- Code: `1014f5f`

## Checks

| # | Group | Check | Measured | Threshold | Result | Note |
| ---: | --- | --- | ---: | --- | --- | --- |
| 1 | Validation (val split) | Corporation: top-1 beats frequency baseline | 72.66% | > 68.49% | PASS | baseline 68.49%, delta +4.17 pp |
| 2 | Validation (val split) | Prelude (step-0): top-1 beats frequency baseline | 81.21% | > 78.39% | PASS | baseline 78.39%, delta +2.82 pp |
| 3 | Validation (val split) | Draft pick: top-1 beats frequency baseline | 72.43% | > 63.98% | PASS | baseline 63.98%, delta +8.45 pp |
| 4 | Validation (val split) | Research buy (step-0): top-1 beats frequency baseline | 95.05% | > 89.73% | PASS | baseline 89.73%, delta +5.32 pp |
| 5 | Validation (val split) | Root branch: top-1 beats frequency baseline | 65.97% | > 46.81% | PASS | baseline 46.81%, delta +19.15 pp |
| 6 | Validation (val split) | Project card play: top-1 beats frequency baseline | 57.83% | > 30.25% | PASS | baseline 30.25%, delta +27.58 pp |
| 7 | Validation (val split) | Standard project: top-1 beats frequency baseline | 70.24% | > 34.99% | PASS | baseline 34.99%, delta +35.25 pp |
| 8 | Validation (val split) | Card action: top-1 beats frequency baseline | 74.12% | > 59.36% | PASS | baseline 59.36%, delta +14.76 pp |
| 9 | Validation (val split) | Corporation: within -1.5 pp of v1 (72.50%) | 72.66% | >= 71.00% | PASS |  |
| 10 | Validation (val split) | Prelude (step-0): within -1.5 pp of v1 (81.30%) | 81.21% | >= 79.80% | PASS |  |
| 11 | Validation (val split) | Draft pick: within -1.5 pp of v1 (70.20%) | 72.43% | >= 68.70% | PASS |  |
| 12 | Validation (val split) | Gen >= 2 value accuracy (raw) | 75.79% | >= 72.00% | PASS |  |
| 13 | Validation (val split) | Gen >= 2 value Brier (raw) | 0.1628 | <= 0.1900 | PASS |  |
| 14 | Validation (val split) | Gen >= 2 value ECE (raw) | 0.0608 | <= 0.0400 | **FAIL** |  |
| 15 | Validation (val split) | Value accuracy increases over gen 2-5 -> gen 6-9 -> gen 10+ (raw) | 71.12% -> 78.79% -> 83.38% | strictly increasing | PASS |  |
| 16 | Zero failures | 0 illegal responses across all pairings | 0 | == 0 | PASS |  |
| 17 | Zero failures | 0 adapter errors across all pairings | 0 | == 0 | PASS |  |
| 18 | Zero failures | 0 step-cap truncations across all pairings | 0 | == 0 | PASS |  |
| 19 | vs b0-v1 | Win rate vs b0-v1 | 100.00% | >= 90.00% | PASS |  |
| 20 | vs b0-v1 | Win rate 95% CI lower bound vs b0-v1 | 100.00% | > 80.00% | PASS | paired bootstrap over seeds |
| 21 | vs b0-v1 | Mean VP margin vs b0-v1 | +54.55 | >= +35.00 | PASS |  |
| 22 | Self-mirror | Self-mirror median game length (generations) | 15.0 | <= 16.0 | PASS |  |
| 23 | Self-mirror | Self-mirror p95 game length (generations) | 20.0 | <= 22.0 | PASS |  |
| 24 | vs v1 warm start | Win rate vs archived v1 warm start | 100.00% | >= 85.00% | PASS |  |
| 25 | vs b1-v2 | Mean own VP vs b1-v2 | 57.30 | >= 75.00 | **FAIL** |  |
| 26 | vs b1-v2 | Mean VP margin vs b1-v2 | -55.05 | >= -35.00 | **FAIL** |  |
| 27 | Schedule | vs b0-v1: 104 seat-swapped dev-* games (13 seeds x 4 maps x 2 seats) | 104 | == 104 | PASS |  |
| 28 | Schedule | self-mirror: 40 seat-swapped dev-* games (5 seeds x 4 maps x 2 seats) | 40 | == 40 | PASS |  |
| 29 | Schedule | vs archived v1 warm start: 104 seat-swapped dev-* games (13 seeds x 4 maps x 2 seats) | 104 | == 104 | PASS |  |
| 30 | Schedule | vs b1-v2: 104 seat-swapped dev-* games (13 seeds x 4 maps x 2 seats) | 104 | == 104 | PASS |  |
| 31 | Schedule | raw policy vs b1-v2 (diagnostic): 104 seat-swapped dev-* games (13 seeds x 4 maps x 2 seats) | 104 | == 104 | PASS (diagnostic) |  |

## Diagnostics (not gating)

| Pairing | Measure | Value |
| --- | --- | --- |
| vs b1-v2 | Win rate | 0.96% [0.00%, 2.88%] |
| vs b1-v2 | Live value Brier / ECE / accuracy (gen >= 2) | 0.1934 / 0.2892 / 72.31% (n=19,417) |
| vs b1-v2 | Live value Brier / ECE / accuracy (all gens) | 0.1981 / 0.3023 / 71.08% (n=21,197) |
| raw policy vs b1-v2 (diagnostic) | Win rate | 21.15% [10.58%, 32.72%] |
| raw policy vs b1-v2 (diagnostic) | Mean own VP / VP margin | 80.38 / -17.48 |
| raw policy vs b1-v2 (diagnostic) | Live value Brier / ECE / accuracy (gen >= 2) | 0.3479 / 0.3730 / 53.79% (n=27,983) |
| raw policy vs b1-v2 (diagnostic) | Live value Brier / ECE / accuracy (all gens) | 0.3431 / 0.3636 / 53.54% (n=29,939) |

## Live pairings

### vs b0-v1

- Games: 104 (104 completed, 0 failures: illegal_response=0, adapter_error=0, step_cap=0)
- Record: 104W-0L-0T; win rate 100.00% [100.00%, 100.00%]
- Mean VP: 107.10 vs 52.55 (margin +54.55)
- Generations: median 16.0, p95 22.0, max 27
- Wall time: 879 s

| Map | Games | Failures | Win rate | 95% CI | Mean VP margin | Mean gens |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Tharsis | 26 | 0 | 100.00% | [100.00%, 100.00%] | +54.35 | 17.31 |
| Elysium | 26 | 0 | 100.00% | [100.00%, 100.00%] | +59.58 | 17.31 |
| Hellas | 26 | 0 | 100.00% | [100.00%, 100.00%] | +58.85 | 17.46 |
| Vastitas Borealis Nova | 26 | 0 | 100.00% | [100.00%, 100.00%] | +45.42 | 15.96 |

### self-mirror

- Games: 40 (40 completed, 0 failures: illegal_response=0, adapter_error=0, step_cap=0)
- Record: 14W-26L-0T; win rate 35.00% [20.00%, 50.00%]
- Mean VP: 79.92 vs 87.40 (margin -7.47)
- Generations: median 15.0, p95 20.0, max 23
- Wall time: 443 s

| Map | Games | Failures | Win rate | 95% CI | Mean VP margin | Mean gens |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Tharsis | 10 | 0 | 30.00% | [10.00%, 50.00%] | -8.70 | 15.50 |
| Elysium | 10 | 0 | 30.00% | [10.00%, 50.00%] | -7.60 | 15.10 |
| Hellas | 10 | 0 | 30.00% | [0.00%, 70.00%] | -11.90 | 14.70 |
| Vastitas Borealis Nova | 10 | 0 | 50.00% | [50.00%, 50.00%] | -1.70 | 15.20 |

### vs archived v1 warm start

- Games: 104 (104 completed, 0 failures: illegal_response=0, adapter_error=0, step_cap=0)
- Record: 104W-0L-0T; win rate 100.00% [100.00%, 100.00%]
- Mean VP: 115.02 vs 55.01 (margin +60.01)
- Generations: median 17.0, p95 26.0, max 34
- Wall time: 1502 s

| Map | Games | Failures | Win rate | 95% CI | Mean VP margin | Mean gens |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Tharsis | 26 | 0 | 100.00% | [100.00%, 100.00%] | +59.81 | 17.15 |
| Elysium | 26 | 0 | 100.00% | [100.00%, 100.00%] | +59.81 | 17.38 |
| Hellas | 26 | 0 | 100.00% | [100.00%, 100.00%] | +62.54 | 19.00 |
| Vastitas Borealis Nova | 26 | 0 | 100.00% | [100.00%, 100.00%] | +57.88 | 18.65 |

### vs b1-v2

- Games: 104 (104 completed, 0 failures: illegal_response=0, adapter_error=0, step_cap=0)
- Record: 1W-103L-0T; win rate 0.96% [0.00%, 2.88%]
- Mean VP: 57.30 vs 112.35 (margin -55.05)
- Generations: median 13.0, p95 15.0, max 15
- Wall time: 583 s

| Map | Games | Failures | Win rate | 95% CI | Mean VP margin | Mean gens |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Tharsis | 26 | 0 | 3.85% | [0.00%, 11.54%] | -53.92 | 12.77 |
| Elysium | 26 | 0 | 0.00% | [0.00%, 0.00%] | -57.08 | 12.58 |
| Hellas | 26 | 0 | 0.00% | [0.00%, 0.00%] | -57.15 | 12.73 |
| Vastitas Borealis Nova | 26 | 0 | 0.00% | [0.00%, 0.00%] | -52.04 | 12.58 |

### raw policy vs b1-v2 (diagnostic)

- Games: 104 (104 completed, 0 failures: illegal_response=0, adapter_error=0, step_cap=0)
- Record: 22W-82L-0T; win rate 21.15% [10.58%, 32.72%]
- Mean VP: 80.38 vs 97.87 (margin -17.48)
- Generations: median 12.0, p95 14.0, max 14
- Wall time: 176 s

| Map | Games | Failures | Win rate | 95% CI | Mean VP margin | Mean gens |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Tharsis | 26 | 0 | 15.38% | [0.00%, 34.62%] | -20.04 | 12.27 |
| Elysium | 26 | 0 | 15.38% | [3.85%, 26.92%] | -22.77 | 12.23 |
| Hellas | 26 | 0 | 34.62% | [15.38%, 53.85%] | -13.31 | 12.00 |
| Vastitas Borealis Nova | 26 | 0 | 19.23% | [3.85%, 38.46%] | -13.81 | 11.92 |

## Validation detail (val split)

| Task | Scope | Model top-1 | Frequency baseline | Delta | n |
| --- | --- | ---: | ---: | ---: | ---: |
| Corporation | overall | 72.66% | 68.49% | +4.17 pp | 5,392 |
| Corporation | tharsis | 72.10% | 67.76% | +4.34 pp | 3,986 |
| Corporation | elysium | 72.22% | 64.81% | +7.41 pp | 54 |
| Corporation | hellas | 74.62% | 71.02% | +3.59 pp | 1,308 |
| Corporation | vastitas borealis nova | 65.91% | 63.64% | +2.27 pp | 44 |
| Prelude (step-0) | overall | 81.21% | 78.39% | +2.82 pp | 5,392 |
| Prelude (step-0) | tharsis | 80.46% | 77.77% | +2.68 pp | 3,986 |
| Prelude (step-0) | elysium | 79.63% | 75.93% | +3.70 pp | 54 |
| Prelude (step-0) | hellas | 83.49% | 80.28% | +3.21 pp | 1,308 |
| Prelude (step-0) | vastitas borealis nova | 84.09% | 81.82% | +2.27 pp | 44 |
| Draft pick | overall | 72.43% | 63.98% | +8.45 pp | 98,383 |
| Draft pick | tharsis | 72.36% | 63.99% | +8.36 pp | 72,736 |
| Draft pick | elysium | 70.07% | 63.23% | +6.84 pp | 862 |
| Draft pick | hellas | 72.78% | 64.01% | +8.77 pp | 24,125 |
| Draft pick | vastitas borealis nova | 70.91% | 63.03% | +7.88 pp | 660 |
| Research buy (step-0) | overall | 95.05% | 89.73% | +5.32 pp | 32,232 |
| Research buy (step-0) | tharsis | 94.93% | 89.73% | +5.20 pp | 23,808 |
| Research buy (step-0) | elysium | 95.77% | 89.44% | +6.34 pp | 284 |
| Research buy (step-0) | hellas | 95.41% | 89.76% | +5.65 pp | 7,927 |
| Research buy (step-0) | vastitas borealis nova | 94.37% | 89.67% | +4.69 pp | 213 |
| Root branch | overall | 65.97% | 46.81% | +19.15 pp | 216,421 |
| Root branch | tharsis | 65.85% | 47.06% | +18.79 pp | 157,841 |
| Root branch | elysium | 68.54% | 49.37% | +19.17 pp | 1,977 |
| Root branch | hellas | 66.21% | 46.03% | +20.18 pp | 55,229 |
| Root branch | vastitas borealis nova | 65.50% | 46.29% | +19.21 pp | 1,374 |
| Project card play | overall | 57.83% | 30.25% | +27.58 pp | 89,803 |
| Project card play | tharsis | 58.03% | 30.50% | +27.52 pp | 65,737 |
| Project card play | elysium | 55.43% | 33.95% | +21.48 pp | 866 |
| Project card play | hellas | 57.45% | 29.44% | +28.01 pp | 22,625 |
| Project card play | vastitas borealis nova | 54.61% | 27.48% | +27.13 pp | 575 |
| Standard project | overall | 70.24% | 34.99% | +35.25 pp | 12,082 |
| Standard project | tharsis | 69.80% | 33.74% | +36.06 pp | 9,344 |
| Standard project | elysium | 68.87% | 33.02% | +35.85 pp | 106 |
| Standard project | hellas | 71.82% | 39.48% | +32.34 pp | 2,548 |
| Standard project | vastitas borealis nova | 72.62% | 39.29% | +33.33 pp | 84 |
| Card action | overall | 74.12% | 59.36% | +14.76 pp | 23,706 |
| Card action | tharsis | 74.01% | 59.22% | +14.78 pp | 16,951 |
| Card action | elysium | 83.59% | 68.21% | +15.38 pp | 195 |
| Card action | hellas | 74.46% | 59.77% | +14.68 pp | 6,381 |
| Card action | vastitas borealis nova | 62.01% | 48.04% | +13.97 pp | 179 |

### Value head (raw)

Temperature 1.0000.

| Scope | n | Accuracy | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| gen 1 | 2,662 | 60.93% | 0.2293 | 0.0397 |
| gen 2-5 | 26,267 | 71.12% | 0.1932 | 0.0749 |
| gen 6-9 | 21,611 | 78.79% | 0.1428 | 0.0551 |
| gen 10+ | 7,591 | 83.38% | 0.1150 | 0.0305 |
| gen 2+ (all) | 55,469 | 75.79% | 0.1628 | 0.0608 |
| gen 2+ tharsis | 41,976 | 76.05% | 0.1602 | 0.0637 |
| gen 2+ elysium | 553 | 67.63% | 0.2194 | 0.1379 |
| gen 2+ hellas | 12,591 | 75.47% | 0.1671 | 0.0533 |
| gen 2+ vastitas borealis nova | 349 | 68.19% | 0.2316 | 0.1742 |
