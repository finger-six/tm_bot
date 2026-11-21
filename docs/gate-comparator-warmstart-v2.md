# Pre-self-play gate: `comparator-warmstart-v2` (comparator agent)

**Verdict: PASS** (30/30 gating checks pass). Thresholds: #38 §5; `val` split and `dev-*` seeds only.

- Checkpoint: `C:\2026\tm_bot\runs\comparator-warmstart-v2\best`
- Validation metrics: `runs\comparator-warmstart-v2\metrics.json`
- v1 opponent: `C:\2026\tm_bot\runs\comparator-warmstart-full\best`
- Candidate setting: `comparator standalone (sampled eval policy)`
- Max decisions per game: `20000`
- Workers / batch size: `8 / 32`
- Code: `1014f5f`

## Checks

| # | Group | Check | Measured | Threshold | Result | Note |
| ---: | --- | --- | ---: | --- | --- | --- |
| 1 | Validation (val split) | Corporation: top-1 beats frequency baseline | 72.96% | > 68.49% | PASS | baseline 68.49%, delta +4.47 pp |
| 2 | Validation (val split) | Prelude (step-0): top-1 beats frequency baseline | 81.18% | > 78.39% | PASS | baseline 78.39%, delta +2.78 pp |
| 3 | Validation (val split) | Draft pick: top-1 beats frequency baseline | 71.50% | > 63.98% | PASS | baseline 63.98%, delta +7.52 pp |
| 4 | Validation (val split) | Research buy (step-0): top-1 beats frequency baseline | 94.46% | > 89.73% | PASS | baseline 89.73%, delta +4.73 pp |
| 5 | Validation (val split) | Root branch: top-1 beats frequency baseline | 64.07% | > 46.81% | PASS | baseline 46.81%, delta +17.25 pp |
| 6 | Validation (val split) | Project card play: top-1 beats frequency baseline | 53.12% | > 30.25% | PASS | baseline 30.25%, delta +22.87 pp |
| 7 | Validation (val split) | Standard project: top-1 beats frequency baseline | 67.96% | > 34.99% | PASS | baseline 34.99%, delta +32.97 pp |
| 8 | Validation (val split) | Card action: top-1 beats frequency baseline | 72.71% | > 59.36% | PASS | baseline 59.36%, delta +13.35 pp |
| 9 | Validation (val split) | Corporation: within -1.5 pp of v1 (72.50%) | 72.96% | >= 71.00% | PASS |  |
| 10 | Validation (val split) | Prelude (step-0): within -1.5 pp of v1 (81.30%) | 81.18% | >= 79.80% | PASS |  |
| 11 | Validation (val split) | Draft pick: within -1.5 pp of v1 (70.20%) | 71.50% | >= 68.70% | PASS |  |
| 12 | Validation (val split) | Gen >= 2 value accuracy (post-temperature) | 76.31% | >= 72.00% | PASS |  |
| 13 | Validation (val split) | Gen >= 2 value Brier (post-temperature) | 0.1549 | <= 0.1900 | PASS |  |
| 14 | Validation (val split) | Gen >= 2 value ECE (post-temperature) | 0.0128 | <= 0.0400 | PASS |  |
| 15 | Validation (val split) | Value accuracy increases over gen 2-5 -> gen 6-9 -> gen 10+ (post-temperature) | 71.25% -> 79.89% -> 83.68% | strictly increasing | PASS |  |
| 16 | Zero failures | 0 illegal responses across all pairings | 0 | == 0 | PASS |  |
| 17 | Zero failures | 0 adapter errors across all pairings | 0 | == 0 | PASS |  |
| 18 | Zero failures | 0 step-cap truncations across all pairings | 0 | == 0 | PASS |  |
| 19 | vs b0-v1 | Win rate vs b0-v1 | 100.00% | >= 90.00% | PASS |  |
| 20 | vs b0-v1 | Win rate 95% CI lower bound vs b0-v1 | 100.00% | > 80.00% | PASS | paired bootstrap over seeds |
| 21 | vs b0-v1 | Mean VP margin vs b0-v1 | +84.28 | >= +35.00 | PASS |  |
| 22 | Self-mirror | Self-mirror median game length (generations) | 13.0 | <= 16.0 | PASS |  |
| 23 | Self-mirror | Self-mirror p95 game length (generations) | 17.0 | <= 22.0 | PASS |  |
| 24 | vs v1 warm start | Win rate vs archived v1 warm start | 100.00% | >= 85.00% | PASS |  |
| 25 | vs b1-v2 | Mean own VP vs b1-v2 | 76.92 | >= 75.00 | PASS |  |
| 26 | vs b1-v2 | Mean VP margin vs b1-v2 | -19.34 | >= -35.00 | PASS |  |
| 27 | Schedule | vs b0-v1: 104 seat-swapped dev-* games (13 seeds x 4 maps x 2 seats) | 104 | == 104 | PASS |  |
| 28 | Schedule | self-mirror: 40 seat-swapped dev-* games (5 seeds x 4 maps x 2 seats) | 40 | == 40 | PASS |  |
| 29 | Schedule | vs archived v1 warm start: 104 seat-swapped dev-* games (13 seeds x 4 maps x 2 seats) | 104 | == 104 | PASS |  |
| 30 | Schedule | vs b1-v2: 104 seat-swapped dev-* games (13 seeds x 4 maps x 2 seats) | 104 | == 104 | PASS |  |

## Diagnostics (not gating)

| Pairing | Measure | Value |
| --- | --- | --- |
| vs b1-v2 | Win rate | 21.15% [11.54%, 30.77%] |
| vs b1-v2 | Live value Brier / ECE / accuracy (gen >= 2) | 0.2876 / 0.3268 / 52.58% (n=25,780) |
| vs b1-v2 | Live value Brier / ECE / accuracy (all gens) | 0.2889 / 0.3272 / 51.79% (n=28,050) |

## Live pairings

### vs b0-v1

- Games: 104 (104 completed, 0 failures: illegal_response=0, adapter_error=0, step_cap=0)
- Record: 104W-0L-0T; win rate 100.00% [100.00%, 100.00%]
- Mean VP: 129.04 vs 44.76 (margin +84.28)
- Generations: median 15.0, p95 17.0, max 18
- Wall time: 236 s

| Map | Games | Failures | Win rate | 95% CI | Mean VP margin | Mean gens |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Tharsis | 26 | 0 | 100.00% | [100.00%, 100.00%] | +82.12 | 14.62 |
| Elysium | 26 | 0 | 100.00% | [100.00%, 100.00%] | +76.08 | 14.31 |
| Hellas | 26 | 0 | 100.00% | [100.00%, 100.00%] | +94.00 | 15.23 |
| Vastitas Borealis Nova | 26 | 0 | 100.00% | [100.00%, 100.00%] | +84.92 | 14.27 |

### self-mirror

- Games: 40 (40 completed, 0 failures: illegal_response=0, adapter_error=0, step_cap=0)
- Record: 18W-22L-0T; win rate 45.00% [37.50%, 55.00%]
- Mean VP: 91.85 vs 94.17 (margin -2.33)
- Generations: median 13.0, p95 17.0, max 18
- Wall time: 133 s

| Map | Games | Failures | Win rate | 95% CI | Mean VP margin | Mean gens |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Tharsis | 10 | 0 | 30.00% | [10.00%, 50.00%] | -7.50 | 14.00 |
| Elysium | 10 | 0 | 50.00% | [50.00%, 50.00%] | +3.80 | 13.00 |
| Hellas | 10 | 0 | 30.00% | [10.00%, 50.00%] | -9.60 | 13.90 |
| Vastitas Borealis Nova | 10 | 0 | 70.00% | [30.00%, 100.00%] | +4.00 | 12.30 |

### vs archived v1 warm start

- Games: 104 (104 completed, 0 failures: illegal_response=0, adapter_error=0, step_cap=0)
- Record: 104W-0L-0T; win rate 100.00% [100.00%, 100.00%]
- Mean VP: 133.40 vs 44.08 (margin +89.33)
- Generations: median 15.0, p95 20.0, max 22
- Wall time: 309 s

| Map | Games | Failures | Win rate | 95% CI | Mean VP margin | Mean gens |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Tharsis | 26 | 0 | 100.00% | [100.00%, 100.00%] | +88.62 | 15.35 |
| Elysium | 26 | 0 | 100.00% | [100.00%, 100.00%] | +88.12 | 15.38 |
| Hellas | 26 | 0 | 100.00% | [100.00%, 100.00%] | +93.15 | 15.92 |
| Vastitas Borealis Nova | 26 | 0 | 100.00% | [100.00%, 100.00%] | +87.42 | 15.35 |

### vs b1-v2

- Games: 104 (104 completed, 0 failures: illegal_response=0, adapter_error=0, step_cap=0)
- Record: 22W-82L-0T; win rate 21.15% [11.54%, 30.77%]
- Mean VP: 76.92 vs 96.26 (margin -19.34)
- Generations: median 12.0, p95 14.0, max 15
- Wall time: 166 s

| Map | Games | Failures | Win rate | 95% CI | Mean VP margin | Mean gens |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Tharsis | 26 | 0 | 19.23% | [7.69%, 30.77%] | -17.96 | 12.04 |
| Elysium | 26 | 0 | 15.38% | [3.85%, 26.92%] | -25.19 | 12.00 |
| Hellas | 26 | 0 | 23.08% | [11.54%, 38.46%] | -16.50 | 11.96 |
| Vastitas Borealis Nova | 26 | 0 | 26.92% | [11.54%, 46.15%] | -17.69 | 11.92 |

## Validation detail (val split)

| Task | Scope | Model top-1 | Frequency baseline | Delta | n |
| --- | --- | ---: | ---: | ---: | ---: |
| Corporation | overall | 72.96% | 68.49% | +4.47 pp | 5,392 |
| Corporation | tharsis | 72.23% | 67.76% | +4.47 pp | 3,986 |
| Corporation | elysium | 70.37% | 64.81% | +5.56 pp | 54 |
| Corporation | hellas | 75.31% | 71.02% | +4.28 pp | 1,308 |
| Corporation | vastitas borealis nova | 72.73% | 63.64% | +9.09 pp | 44 |
| Prelude (step-0) | overall | 81.18% | 78.39% | +2.78 pp | 5,392 |
| Prelude (step-0) | tharsis | 80.68% | 77.77% | +2.91 pp | 3,986 |
| Prelude (step-0) | elysium | 79.63% | 75.93% | +3.70 pp | 54 |
| Prelude (step-0) | hellas | 82.72% | 80.28% | +2.45 pp | 1,308 |
| Prelude (step-0) | vastitas borealis nova | 81.82% | 81.82% | +0.00 pp | 44 |
| Draft pick | overall | 71.50% | 63.98% | +7.52 pp | 98,383 |
| Draft pick | tharsis | 71.50% | 63.99% | +7.50 pp | 72,736 |
| Draft pick | elysium | 69.61% | 63.23% | +6.38 pp | 862 |
| Draft pick | hellas | 71.61% | 64.01% | +7.60 pp | 24,125 |
| Draft pick | vastitas borealis nova | 70.76% | 63.03% | +7.73 pp | 660 |
| Research buy (step-0) | overall | 94.46% | 89.73% | +4.73 pp | 32,232 |
| Research buy (step-0) | tharsis | 94.37% | 89.73% | +4.64 pp | 23,808 |
| Research buy (step-0) | elysium | 95.07% | 89.44% | +5.63 pp | 284 |
| Research buy (step-0) | hellas | 94.71% | 89.76% | +4.96 pp | 7,927 |
| Research buy (step-0) | vastitas borealis nova | 94.37% | 89.67% | +4.69 pp | 213 |
| Root branch | overall | 64.07% | 46.81% | +17.25 pp | 216,421 |
| Root branch | tharsis | 63.87% | 47.06% | +16.81 pp | 157,841 |
| Root branch | elysium | 67.22% | 49.37% | +17.86 pp | 1,977 |
| Root branch | hellas | 64.52% | 46.03% | +18.50 pp | 55,229 |
| Root branch | vastitas borealis nova | 63.83% | 46.29% | +17.54 pp | 1,374 |
| Project card play | overall | 53.12% | 30.25% | +22.87 pp | 89,803 |
| Project card play | tharsis | 53.15% | 30.50% | +22.65 pp | 65,737 |
| Project card play | elysium | 52.19% | 33.95% | +18.24 pp | 866 |
| Project card play | hellas | 53.11% | 29.44% | +23.67 pp | 22,625 |
| Project card play | vastitas borealis nova | 50.96% | 27.48% | +23.48 pp | 575 |
| Standard project | overall | 67.96% | 34.99% | +32.97 pp | 12,082 |
| Standard project | tharsis | 67.54% | 33.74% | +33.80 pp | 9,344 |
| Standard project | elysium | 64.15% | 33.02% | +31.13 pp | 106 |
| Standard project | hellas | 69.62% | 39.48% | +30.14 pp | 2,548 |
| Standard project | vastitas borealis nova | 69.05% | 39.29% | +29.76 pp | 84 |
| Card action | overall | 72.71% | 59.36% | +13.35 pp | 23,706 |
| Card action | tharsis | 72.46% | 59.22% | +13.23 pp | 16,951 |
| Card action | elysium | 81.54% | 68.21% | +13.33 pp | 195 |
| Card action | hellas | 73.39% | 59.77% | +13.62 pp | 6,381 |
| Card action | vastitas borealis nova | 62.57% | 48.04% | +14.53 pp | 179 |

### Value head (post-temperature)

Temperature 1.2603.

| Scope | n | Accuracy | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| gen 1 | 2,662 | 62.10% | 0.2271 | 0.0141 |
| gen 2-5 | 26,267 | 71.25% | 0.1840 | 0.0193 |
| gen 6-9 | 21,611 | 79.89% | 0.1342 | 0.0178 |
| gen 10+ | 7,591 | 83.68% | 0.1134 | 0.0252 |
| gen 2+ (all) | 55,469 | 76.31% | 0.1549 | 0.0128 |
| gen 2+ tharsis | 41,976 | 76.51% | 0.1533 | 0.0106 |
| gen 2+ elysium | 553 | 67.09% | 0.2091 | 0.1227 |
| gen 2+ hellas | 12,591 | 76.26% | 0.1566 | 0.0358 |
| gen 2+ vastitas borealis nova | 349 | 69.05% | 0.2067 | 0.0909 |
