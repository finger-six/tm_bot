# Pre-self-play gate: `comparator-warmstart-v3` (comparator agent)

**Verdict: PASS** (31/31 gating checks pass). Thresholds: #38 §5; `val` split and `dev-*` seeds only.

- Checkpoint: `runs/comparator-warmstart-v3/best`
- Validation metrics: `runs/comparator-warmstart-v3/metrics.json`
- v1 opponent: `runs/comparator-warmstart-full/best`
- Candidate setting: `comparator standalone (sampled eval policy)`
- Value temperature (reported P(win), value checks): `1.489048571277834`
- Max decisions per game: `20000`
- Workers / batch size: `8 / 32`
- Code: `967517b`

## Checks

| # | Group | Check | Measured | Threshold | Result | Note |
| ---: | --- | --- | ---: | --- | --- | --- |
| 1 | Validation (val split) | Corporation: top-1 beats frequency baseline | 73.16% | > 68.49% | PASS | baseline 68.49%, delta +4.67 pp |
| 2 | Validation (val split) | Prelude (step-0): top-1 beats frequency baseline | 82.05% | > 78.39% | PASS | baseline 78.39%, delta +3.65 pp |
| 3 | Validation (val split) | Draft pick: top-1 beats frequency baseline | 72.02% | > 63.98% | PASS | baseline 63.98%, delta +8.03 pp |
| 4 | Validation (val split) | Research buy (step-0): top-1 beats frequency baseline | 94.71% | > 89.73% | PASS | baseline 89.73%, delta +4.98 pp |
| 5 | Validation (val split) | Root branch: top-1 beats frequency baseline | 65.56% | > 46.81% | PASS | baseline 46.81%, delta +18.75 pp |
| 6 | Validation (val split) | Project card play: top-1 beats frequency baseline | 55.10% | > 30.25% | PASS | baseline 30.25%, delta +24.84 pp |
| 7 | Validation (val split) | Standard project: top-1 beats frequency baseline | 69.82% | > 34.99% | PASS | baseline 34.99%, delta +34.84 pp |
| 8 | Validation (val split) | Card action: top-1 beats frequency baseline | 73.16% | > 59.36% | PASS | baseline 59.36%, delta +13.80 pp |
| 9 | Validation (val split) | Tile placement (space): top-1 beats frequency baseline | 46.13% | > 28.51% | PASS | baseline 28.51%, delta +17.62 pp |
| 10 | Validation (val split) | Corporation: within -1.5 pp of v1 (72.50%) | 73.16% | >= 71.00% | PASS |  |
| 11 | Validation (val split) | Prelude (step-0): within -1.5 pp of v1 (81.30%) | 82.05% | >= 79.80% | PASS |  |
| 12 | Validation (val split) | Draft pick: within -1.5 pp of v1 (70.20%) | 72.02% | >= 68.70% | PASS |  |
| 13 | Validation (val split) | Gen >= 2 value accuracy (post-temperature) | 75.97% | >= 72.00% | PASS |  |
| 14 | Validation (val split) | Gen >= 2 value Brier (post-temperature) | 0.1578 | <= 0.1900 | PASS |  |
| 15 | Validation (val split) | Gen >= 2 value ECE (post-temperature) | 0.0179 | <= 0.0400 | PASS |  |
| 16 | Validation (val split) | Value accuracy increases over gen 2-5 -> gen 6-9 -> gen 10+ (post-temperature) | 71.03% -> 79.27% -> 83.69% | strictly increasing | PASS |  |
| 17 | Zero failures | 0 illegal responses across all pairings | 0 | == 0 | PASS |  |
| 18 | Zero failures | 0 adapter errors across all pairings | 0 | == 0 | PASS |  |
| 19 | Zero failures | 0 step-cap truncations across all pairings | 0 | == 0 | PASS |  |
| 20 | vs b0-v1 | Win rate vs b0-v1 | 100.00% | >= 90.00% | PASS |  |
| 21 | vs b0-v1 | Win rate 95% CI lower bound vs b0-v1 | 100.00% | > 80.00% | PASS | paired bootstrap over seeds |
| 22 | vs b0-v1 | Mean VP margin vs b0-v1 | +82.14 | >= +35.00 | PASS |  |
| 23 | Self-mirror | Self-mirror median game length (generations) | 12.0 | <= 16.0 | PASS |  |
| 24 | Self-mirror | Self-mirror p95 game length (generations) | 16.0 | <= 22.0 | PASS |  |
| 25 | vs v1 warm start | Win rate vs archived v1 warm start | 100.00% | >= 85.00% | PASS |  |
| 26 | vs b1-v2 | Mean own VP vs b1-v2 | 80.03 | >= 75.00 | PASS |  |
| 27 | vs b1-v2 | Mean VP margin vs b1-v2 | -13.81 | >= -35.00 | PASS |  |
| 28 | Schedule | vs b0-v1: 104 seat-swapped dev-* games (13 seeds x 4 maps x 2 seats) | 104 | == 104 | PASS |  |
| 29 | Schedule | self-mirror: 40 seat-swapped dev-* games (5 seeds x 4 maps x 2 seats) | 40 | == 40 | PASS |  |
| 30 | Schedule | vs archived v1 warm start: 104 seat-swapped dev-* games (13 seeds x 4 maps x 2 seats) | 104 | == 104 | PASS |  |
| 31 | Schedule | vs b1-v2: 104 seat-swapped dev-* games (13 seeds x 4 maps x 2 seats) | 104 | == 104 | PASS |  |

## Diagnostics (not gating)

Live value rows score the candidate's reported (post-temperature) P(win) against the final result of its games.

| Pairing | Measure | Value |
| --- | --- | --- |
| vs b1-v2 | Win rate | 26.92% [17.31%, 38.46%] |
| vs b1-v2 | Live value Brier / ECE / accuracy (gen >= 2) | 0.2276 / 0.2157 / 64.35% (n=25,071) |
| vs b1-v2 | Live value Brier / ECE / accuracy (all gens) | 0.2353 / 0.2207 / 62.92% (n=27,275) |
| self-mirror | Live value Brier / ECE / accuracy (gen >= 2) | 0.2483 / 0.1389 / 59.57% (n=11,041) |
| self-mirror | Live value Brier / ECE / accuracy (all gens) | 0.2517 / 0.1456 / 58.81% (n=11,825) |
| vs archived v1 warm start | Live value Brier / ECE / accuracy (gen >= 2) | 0.1654 / 0.2710 / 75.38% (n=37,393) |
| vs archived v1 warm start | Live value Brier / ECE / accuracy (all gens) | 0.1728 / 0.2835 / 73.93% (n=39,634) |

## Live pairings

### vs b0-v1

- Games: 104 (104 completed, 0 failures: illegal_response=0, adapter_error=0, step_cap=0)
- Record: 104W-0L-0T; win rate 100.00% [100.00%, 100.00%]
- Mean VP: 126.12 vs 43.98 (margin +82.14)
- Generations: median 14.0, p95 17.0, max 19
- Wall time: 191 s

| Map | Games | Failures | Win rate | 95% CI | Mean VP margin | Mean gens |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Tharsis | 26 | 0 | 100.00% | [100.00%, 100.00%] | +81.38 | 14.23 |
| Elysium | 26 | 0 | 100.00% | [100.00%, 100.00%] | +77.50 | 14.35 |
| Hellas | 26 | 0 | 100.00% | [100.00%, 100.00%] | +86.62 | 14.00 |
| Vastitas Borealis Nova | 26 | 0 | 100.00% | [100.00%, 100.00%] | +83.08 | 13.77 |

### self-mirror

- Games: 40 (40 completed, 0 failures: illegal_response=0, adapter_error=0, step_cap=0)
- Record: 21W-19L-0T; win rate 52.50% [32.50%, 67.50%]
- Mean VP: 91.20 vs 92.33 (margin -1.12)
- Generations: median 12.0, p95 16.0, max 19
- Wall time: 107 s

| Map | Games | Failures | Win rate | 95% CI | Mean VP margin | Mean gens |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Tharsis | 10 | 0 | 40.00% | [20.00%, 50.00%] | -7.80 | 14.20 |
| Elysium | 10 | 0 | 60.00% | [30.00%, 90.00%] | +6.10 | 12.60 |
| Hellas | 10 | 0 | 40.00% | [20.00%, 50.00%] | -7.50 | 12.70 |
| Vastitas Borealis Nova | 10 | 0 | 70.00% | [30.00%, 100.00%] | +4.70 | 12.10 |

### vs archived v1 warm start

- Games: 104 (104 completed, 0 failures: illegal_response=0, adapter_error=0, step_cap=0)
- Record: 104W-0L-0T; win rate 100.00% [100.00%, 100.00%]
- Mean VP: 131.95 vs 42.08 (margin +89.88)
- Generations: median 15.0, p95 18.8, max 28
- Wall time: 269 s

| Map | Games | Failures | Win rate | 95% CI | Mean VP margin | Mean gens |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Tharsis | 26 | 0 | 100.00% | [100.00%, 100.00%] | +90.92 | 16.27 |
| Elysium | 26 | 0 | 100.00% | [100.00%, 100.00%] | +91.23 | 15.38 |
| Hellas | 26 | 0 | 100.00% | [100.00%, 100.00%] | +89.96 | 15.08 |
| Vastitas Borealis Nova | 26 | 0 | 100.00% | [100.00%, 100.00%] | +87.38 | 14.12 |

### vs b1-v2

- Games: 104 (104 completed, 0 failures: illegal_response=0, adapter_error=0, step_cap=0)
- Record: 28W-76L-0T; win rate 26.92% [17.31%, 38.46%]
- Mean VP: 80.03 vs 93.84 (margin -13.81)
- Generations: median 12.0, p95 13.0, max 14
- Wall time: 154 s

| Map | Games | Failures | Win rate | 95% CI | Mean VP margin | Mean gens |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Tharsis | 26 | 0 | 30.77% | [11.54%, 50.00%] | -14.54 | 11.88 |
| Elysium | 26 | 0 | 15.38% | [3.85%, 26.92%] | -16.96 | 11.81 |
| Hellas | 26 | 0 | 34.62% | [15.38%, 53.85%] | -11.35 | 12.00 |
| Vastitas Borealis Nova | 26 | 0 | 26.92% | [11.54%, 38.46%] | -12.38 | 11.42 |

## Validation detail (val split)

| Task | Scope | Model top-1 | Frequency baseline | Delta | n |
| --- | --- | ---: | ---: | ---: | ---: |
| Corporation | overall | 73.16% | 68.49% | +4.67 pp | 5,392 |
| Corporation | tharsis | 72.68% | 67.76% | +4.92 pp | 3,986 |
| Corporation | elysium | 72.22% | 64.81% | +7.41 pp | 54 |
| Corporation | hellas | 74.54% | 71.02% | +3.52 pp | 1,308 |
| Corporation | vastitas borealis nova | 77.27% | 63.64% | +13.64 pp | 44 |
| Prelude (step-0) | overall | 82.05% | 78.39% | +3.65 pp | 5,392 |
| Prelude (step-0) | tharsis | 81.71% | 77.77% | +3.94 pp | 3,986 |
| Prelude (step-0) | elysium | 77.78% | 75.93% | +1.85 pp | 54 |
| Prelude (step-0) | hellas | 83.33% | 80.28% | +3.06 pp | 1,308 |
| Prelude (step-0) | vastitas borealis nova | 79.55% | 81.82% | -2.27 pp | 44 |
| Draft pick | overall | 72.02% | 63.98% | +8.03 pp | 98,383 |
| Draft pick | tharsis | 71.95% | 63.99% | +7.95 pp | 72,736 |
| Draft pick | elysium | 69.61% | 63.23% | +6.38 pp | 862 |
| Draft pick | hellas | 72.29% | 64.01% | +8.28 pp | 24,125 |
| Draft pick | vastitas borealis nova | 72.88% | 63.03% | +9.85 pp | 660 |
| Research buy (step-0) | overall | 94.71% | 89.73% | +4.98 pp | 32,232 |
| Research buy (step-0) | tharsis | 94.66% | 89.73% | +4.93 pp | 23,808 |
| Research buy (step-0) | elysium | 94.72% | 89.44% | +5.28 pp | 284 |
| Research buy (step-0) | hellas | 94.88% | 89.76% | +5.12 pp | 7,927 |
| Research buy (step-0) | vastitas borealis nova | 94.84% | 89.67% | +5.16 pp | 213 |
| Root branch | overall | 65.56% | 46.81% | +18.75 pp | 216,421 |
| Root branch | tharsis | 65.51% | 47.06% | +18.44 pp | 157,841 |
| Root branch | elysium | 68.03% | 49.37% | +18.66 pp | 1,977 |
| Root branch | hellas | 65.65% | 46.03% | +19.63 pp | 55,229 |
| Root branch | vastitas borealis nova | 64.70% | 46.29% | +18.41 pp | 1,374 |
| Project card play | overall | 55.10% | 30.25% | +24.84 pp | 89,803 |
| Project card play | tharsis | 55.13% | 30.50% | +24.63 pp | 65,737 |
| Project card play | elysium | 52.42% | 33.95% | +18.48 pp | 866 |
| Project card play | hellas | 55.17% | 29.44% | +25.73 pp | 22,625 |
| Project card play | vastitas borealis nova | 51.48% | 27.48% | +24.00 pp | 575 |
| Standard project | overall | 69.82% | 34.99% | +34.84 pp | 12,082 |
| Standard project | tharsis | 69.72% | 33.74% | +35.98 pp | 9,344 |
| Standard project | elysium | 61.32% | 33.02% | +28.30 pp | 106 |
| Standard project | hellas | 70.45% | 39.48% | +30.97 pp | 2,548 |
| Standard project | vastitas borealis nova | 72.62% | 39.29% | +33.33 pp | 84 |
| Card action | overall | 73.16% | 59.36% | +13.80 pp | 23,706 |
| Card action | tharsis | 73.06% | 59.22% | +13.83 pp | 16,951 |
| Card action | elysium | 80.00% | 68.21% | +11.79 pp | 195 |
| Card action | hellas | 73.61% | 59.77% | +13.84 pp | 6,381 |
| Card action | vastitas borealis nova | 59.22% | 48.04% | +11.17 pp | 179 |
| Tile placement (space) | overall | 46.13% | 28.51% | +17.62 pp | 61,262 |
| Tile placement (space) | tharsis | 46.98% | 29.59% | +17.38 pp | 45,232 |
| Tile placement (space) | elysium | 36.26% | 30.47% | +5.79 pp | 535 |
| Tile placement (space) | hellas | 44.18% | 25.30% | +18.88 pp | 15,159 |
| Tile placement (space) | vastitas borealis nova | 36.01% | 25.00% | +11.01 pp | 336 |

### Value head (post-temperature)

Temperature 1.4890.

| Scope | n | Accuracy | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| gen 1 | 2,662 | 61.57% | 0.2318 | 0.0298 |
| gen 2-5 | 26,267 | 71.03% | 0.1873 | 0.0168 |
| gen 6-9 | 21,611 | 79.27% | 0.1378 | 0.0168 |
| gen 10+ | 7,591 | 83.69% | 0.1127 | 0.0340 |
| gen 2+ (all) | 55,469 | 75.97% | 0.1578 | 0.0179 |
| gen 2+ tharsis | 41,976 | 76.14% | 0.1559 | 0.0161 |
| gen 2+ elysium | 553 | 70.16% | 0.1990 | 0.0951 |
| gen 2+ hellas | 12,591 | 75.86% | 0.1610 | 0.0241 |
| gen 2+ vastitas borealis nova | 349 | 69.05% | 0.2117 | 0.1359 |
