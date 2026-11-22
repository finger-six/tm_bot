# Pre-self-play gate: `warmstart-v4` (main agent)

**Verdict: PASS** (31/31 gating checks pass). Thresholds: #38 §5; `val` split and `dev-*` seeds only.

- Checkpoint: `runs/warmstart-v4/best`
- Validation metrics: `runs/warmstart-v4/metrics.json`
- v1 opponent: `runs/warmstart/best`
- Candidate setting: `light search num_simulations=8, num_worlds=4, c_puct=1.5, temperature=0.0, action_temperature=1.0, wave_size=4, play_temperature=flat, margin_weight=0.0, root_rule=completed_q`
- v1 opponent setting: `light search num_simulations=8, num_worlds=4, c_puct=1.5, temperature=0.0, action_temperature=1.0, wave_size=4, play_temperature=flat, margin_weight=0.0, root_rule=visits`
- Raw-policy diagnostic: `search agent with num_simulations=0 (policy head)`
- Value temperature (reported P(win), value checks): `1.3556331664041352`
- Max decisions per game: `20000`
- Workers / batch size: `8 / 32`
- Code: `967517b`

## Checks

| # | Group | Check | Measured | Threshold | Result | Note |
| ---: | --- | --- | ---: | --- | --- | --- |
| 1 | Validation (val split) | Corporation: top-1 beats frequency baseline | 73.02% | > 68.49% | PASS | baseline 68.49%, delta +4.53 pp |
| 2 | Validation (val split) | Prelude (step-0): top-1 beats frequency baseline | 81.29% | > 78.39% | PASS | baseline 78.39%, delta +2.89 pp |
| 3 | Validation (val split) | Draft pick: top-1 beats frequency baseline | 72.54% | > 63.98% | PASS | baseline 63.98%, delta +8.55 pp |
| 4 | Validation (val split) | Research buy (step-0): top-1 beats frequency baseline | 94.85% | > 89.73% | PASS | baseline 89.73%, delta +5.12 pp |
| 5 | Validation (val split) | Root branch: top-1 beats frequency baseline | 65.89% | > 46.81% | PASS | baseline 46.81%, delta +19.08 pp |
| 6 | Validation (val split) | Project card play: top-1 beats frequency baseline | 58.00% | > 30.25% | PASS | baseline 30.25%, delta +27.75 pp |
| 7 | Validation (val split) | Standard project: top-1 beats frequency baseline | 70.58% | > 34.99% | PASS | baseline 34.99%, delta +35.59 pp |
| 8 | Validation (val split) | Card action: top-1 beats frequency baseline | 73.99% | > 59.36% | PASS | baseline 59.36%, delta +14.62 pp |
| 9 | Validation (val split) | Tile placement (space): top-1 beats frequency baseline | 47.64% | > 28.51% | PASS | baseline 28.51%, delta +19.12 pp |
| 10 | Validation (val split) | Corporation: within -1.5 pp of v1 (72.50%) | 73.02% | >= 71.00% | PASS |  |
| 11 | Validation (val split) | Prelude (step-0): within -1.5 pp of v1 (81.30%) | 81.29% | >= 79.80% | PASS |  |
| 12 | Validation (val split) | Draft pick: within -1.5 pp of v1 (70.20%) | 72.54% | >= 68.70% | PASS |  |
| 13 | Validation (val split) | Gen >= 2 value accuracy (post-temperature) | 75.47% | >= 72.00% | PASS |  |
| 14 | Validation (val split) | Gen >= 2 value Brier (post-temperature) | 0.1601 | <= 0.1900 | PASS |  |
| 15 | Validation (val split) | Gen >= 2 value ECE (post-temperature) | 0.0376 | <= 0.0400 | PASS |  |
| 16 | Validation (val split) | Value accuracy increases over gen 2-5 -> gen 6-9 -> gen 10+ (post-temperature) | 70.61% -> 78.92% -> 82.45% | strictly increasing | PASS |  |
| 17 | Zero failures | 0 illegal responses across all pairings | 0 | == 0 | PASS |  |
| 18 | Zero failures | 0 adapter errors across all pairings | 0 | == 0 | PASS |  |
| 19 | Zero failures | 0 step-cap truncations across all pairings | 0 | == 0 | PASS |  |
| 20 | vs b0-v1 | Win rate vs b0-v1 | 100.00% | >= 90.00% | PASS |  |
| 21 | vs b0-v1 | Win rate 95% CI lower bound vs b0-v1 | 100.00% | > 80.00% | PASS | paired bootstrap over seeds |
| 22 | vs b0-v1 | Mean VP margin vs b0-v1 | +84.01 | >= +35.00 | PASS |  |
| 23 | Self-mirror | Self-mirror median game length (generations) | 13.0 | <= 16.0 | PASS |  |
| 24 | Self-mirror | Self-mirror p95 game length (generations) | 17.2 | <= 22.0 | PASS |  |
| 25 | vs v1 warm start | Win rate vs archived v1 warm start | 100.00% | >= 85.00% | PASS |  |
| 26 | vs b1-v2 | Mean own VP vs b1-v2 | 82.77 | >= 75.00 | PASS |  |
| 27 | vs b1-v2 | Mean VP margin vs b1-v2 | -10.65 | >= -35.00 | PASS |  |
| 28 | Schedule | vs b0-v1: 104 seat-swapped dev-* games (13 seeds x 4 maps x 2 seats) | 104 | == 104 | PASS |  |
| 29 | Schedule | self-mirror: 40 seat-swapped dev-* games (5 seeds x 4 maps x 2 seats) | 40 | == 40 | PASS |  |
| 30 | Schedule | vs archived v1 warm start: 104 seat-swapped dev-* games (13 seeds x 4 maps x 2 seats) | 104 | == 104 | PASS |  |
| 31 | Schedule | vs b1-v2: 104 seat-swapped dev-* games (13 seeds x 4 maps x 2 seats) | 104 | == 104 | PASS |  |
| 32 | Schedule | raw policy vs b1-v2 (diagnostic): 104 seat-swapped dev-* games (13 seeds x 4 maps x 2 seats) | 104 | == 104 | PASS (diagnostic) |  |

## Diagnostics (not gating)

Live value rows score the candidate's reported (post-temperature) P(win) against the final result of its games.

| Pairing | Measure | Value |
| --- | --- | --- |
| vs b1-v2 | Win rate | 29.81% [14.42%, 47.12%] |
| vs b1-v2 | Live value Brier / ECE / accuracy (gen >= 2) | 0.3081 / 0.3373 / 54.55% (n=25,705) |
| vs b1-v2 | Live value Brier / ECE / accuracy (all gens) | 0.3036 / 0.3273 / 54.79% (n=27,653) |
| raw policy vs b1-v2 (diagnostic) | Win rate | 27.88% [17.31%, 42.31%] |
| raw policy vs b1-v2 (diagnostic) | Mean own VP / VP margin | 80.31 / -13.88 |
| search vs raw policy (vs b1-v2) | Own VP / VP margin, search minus raw | +2.46 / +3.23 (search 82.77 / -10.65, raw 80.31 / -13.88) |
| raw policy vs b1-v2 (diagnostic) | Live value Brier / ECE / accuracy (gen >= 2) | 0.3202 / 0.3779 / 51.86% (n=27,699) |
| raw policy vs b1-v2 (diagnostic) | Live value Brier / ECE / accuracy (all gens) | 0.3157 / 0.3665 / 52.24% (n=29,738) |
| self-mirror | Live value Brier / ECE / accuracy (gen >= 2) | 0.1815 / 0.1299 / 70.30% (n=11,510) |
| self-mirror | Live value Brier / ECE / accuracy (all gens) | 0.1869 / 0.1186 / 69.00% (n=12,261) |
| vs archived v1 warm start | Live value Brier / ECE / accuracy (gen >= 2) | 0.0484 / 0.1370 / 94.87% (n=37,435) |
| vs archived v1 warm start | Live value Brier / ECE / accuracy (all gens) | 0.0607 / 0.1560 / 91.97% (n=39,284) |

## Live pairings

### vs b0-v1

- Games: 104 (104 completed, 0 failures: illegal_response=0, adapter_error=0, step_cap=0)
- Record: 104W-0L-0T; win rate 100.00% [100.00%, 100.00%]
- Mean VP: 127.19 vs 43.18 (margin +84.01)
- Generations: median 14.0, p95 16.0, max 20
- Wall time: 1083 s

| Map | Games | Failures | Win rate | 95% CI | Mean VP margin | Mean gens |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Tharsis | 26 | 0 | 100.00% | [100.00%, 100.00%] | +84.65 | 14.00 |
| Elysium | 26 | 0 | 100.00% | [100.00%, 100.00%] | +81.50 | 13.92 |
| Hellas | 26 | 0 | 100.00% | [100.00%, 100.00%] | +85.27 | 13.73 |
| Vastitas Borealis Nova | 26 | 0 | 100.00% | [100.00%, 100.00%] | +84.62 | 14.08 |

### self-mirror

- Games: 40 (40 completed, 0 failures: illegal_response=0, adapter_error=0, step_cap=0)
- Record: 19W-21L-0T; win rate 47.50% [37.50%, 57.50%]
- Mean VP: 97.70 vs 94.25 (margin +3.45)
- Generations: median 13.0, p95 17.2, max 24
- Wall time: 862 s

| Map | Games | Failures | Win rate | 95% CI | Mean VP margin | Mean gens |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Tharsis | 10 | 0 | 60.00% | [30.00%, 90.00%] | -0.50 | 13.80 |
| Elysium | 10 | 0 | 40.00% | [20.00%, 50.00%] | +4.10 | 14.60 |
| Hellas | 10 | 0 | 50.00% | [50.00%, 50.00%] | +10.10 | 13.20 |
| Vastitas Borealis Nova | 10 | 0 | 40.00% | [20.00%, 50.00%] | +0.10 | 12.70 |

### vs archived v1 warm start

- Games: 104 (104 completed, 0 failures: illegal_response=0, adapter_error=0, step_cap=0)
- Record: 104W-0L-0T; win rate 100.00% [100.00%, 100.00%]
- Mean VP: 134.16 vs 46.03 (margin +88.13)
- Generations: median 14.0, p95 19.0, max 28
- Wall time: 1701 s

| Map | Games | Failures | Win rate | 95% CI | Mean VP margin | Mean gens |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Tharsis | 26 | 0 | 100.00% | [100.00%, 100.00%] | +84.92 | 15.38 |
| Elysium | 26 | 0 | 100.00% | [100.00%, 100.00%] | +85.15 | 14.69 |
| Hellas | 26 | 0 | 100.00% | [100.00%, 100.00%] | +90.35 | 14.38 |
| Vastitas Borealis Nova | 26 | 0 | 100.00% | [100.00%, 100.00%] | +92.12 | 14.81 |

### vs b1-v2

- Games: 104 (104 completed, 0 failures: illegal_response=0, adapter_error=0, step_cap=0)
- Record: 31W-73L-0T; win rate 29.81% [14.42%, 47.12%]
- Mean VP: 82.77 vs 93.42 (margin -10.65)
- Generations: median 12.0, p95 13.0, max 14
- Wall time: 845 s

| Map | Games | Failures | Win rate | 95% CI | Mean VP margin | Mean gens |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Tharsis | 26 | 0 | 42.31% | [19.23%, 65.38%] | -8.04 | 11.92 |
| Elysium | 26 | 0 | 19.23% | [3.85%, 38.46%] | -14.54 | 11.73 |
| Hellas | 26 | 0 | 30.77% | [11.54%, 53.85%] | -8.77 | 11.77 |
| Vastitas Borealis Nova | 26 | 0 | 26.92% | [11.54%, 46.15%] | -11.27 | 11.65 |

### raw policy vs b1-v2 (diagnostic)

- Games: 104 (104 completed, 0 failures: illegal_response=0, adapter_error=0, step_cap=0)
- Record: 29W-75L-0T; win rate 27.88% [17.31%, 42.31%]
- Mean VP: 80.31 vs 94.19 (margin -13.88)
- Generations: median 12.0, p95 13.8, max 15
- Wall time: 191 s

| Map | Games | Failures | Win rate | 95% CI | Mean VP margin | Mean gens |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Tharsis | 26 | 0 | 34.62% | [19.23%, 50.00%] | -10.35 | 11.77 |
| Elysium | 26 | 0 | 26.92% | [11.54%, 46.15%] | -16.42 | 11.88 |
| Hellas | 26 | 0 | 23.08% | [7.69%, 42.31%] | -15.69 | 11.92 |
| Vastitas Borealis Nova | 26 | 0 | 26.92% | [7.69%, 50.00%] | -13.08 | 11.73 |

## Validation detail (val split)

| Task | Scope | Model top-1 | Frequency baseline | Delta | n |
| --- | --- | ---: | ---: | ---: | ---: |
| Corporation | overall | 73.02% | 68.49% | +4.53 pp | 5,392 |
| Corporation | tharsis | 72.18% | 67.76% | +4.42 pp | 3,986 |
| Corporation | elysium | 79.63% | 64.81% | +14.81 pp | 54 |
| Corporation | hellas | 75.23% | 71.02% | +4.20 pp | 1,308 |
| Corporation | vastitas borealis nova | 75.00% | 63.64% | +11.36 pp | 44 |
| Prelude (step-0) | overall | 81.29% | 78.39% | +2.89 pp | 5,392 |
| Prelude (step-0) | tharsis | 80.56% | 77.77% | +2.78 pp | 3,986 |
| Prelude (step-0) | elysium | 77.78% | 75.93% | +1.85 pp | 54 |
| Prelude (step-0) | hellas | 83.26% | 80.28% | +2.98 pp | 1,308 |
| Prelude (step-0) | vastitas borealis nova | 93.18% | 81.82% | +11.36 pp | 44 |
| Draft pick | overall | 72.54% | 63.98% | +8.55 pp | 98,383 |
| Draft pick | tharsis | 72.60% | 63.99% | +8.61 pp | 72,736 |
| Draft pick | elysium | 69.84% | 63.23% | +6.61 pp | 862 |
| Draft pick | hellas | 72.50% | 64.01% | +8.49 pp | 24,125 |
| Draft pick | vastitas borealis nova | 70.61% | 63.03% | +7.58 pp | 660 |
| Research buy (step-0) | overall | 94.85% | 89.73% | +5.12 pp | 32,232 |
| Research buy (step-0) | tharsis | 94.90% | 89.73% | +5.17 pp | 23,808 |
| Research buy (step-0) | elysium | 96.13% | 89.44% | +6.69 pp | 284 |
| Research buy (step-0) | hellas | 94.65% | 89.76% | +4.89 pp | 7,927 |
| Research buy (step-0) | vastitas borealis nova | 95.31% | 89.67% | +5.63 pp | 213 |
| Root branch | overall | 65.89% | 46.81% | +19.08 pp | 216,421 |
| Root branch | tharsis | 65.75% | 47.06% | +18.69 pp | 157,841 |
| Root branch | elysium | 69.45% | 49.37% | +20.08 pp | 1,977 |
| Root branch | hellas | 66.15% | 46.03% | +20.12 pp | 55,229 |
| Root branch | vastitas borealis nova | 66.23% | 46.29% | +19.94 pp | 1,374 |
| Project card play | overall | 58.00% | 30.25% | +27.75 pp | 89,803 |
| Project card play | tharsis | 58.07% | 30.50% | +27.57 pp | 65,737 |
| Project card play | elysium | 55.20% | 33.95% | +21.25 pp | 866 |
| Project card play | hellas | 57.90% | 29.44% | +28.46 pp | 22,625 |
| Project card play | vastitas borealis nova | 57.74% | 27.48% | +30.26 pp | 575 |
| Standard project | overall | 70.58% | 34.99% | +35.59 pp | 12,082 |
| Standard project | tharsis | 70.42% | 33.74% | +36.68 pp | 9,344 |
| Standard project | elysium | 69.81% | 33.02% | +36.79 pp | 106 |
| Standard project | hellas | 71.15% | 39.48% | +31.67 pp | 2,548 |
| Standard project | vastitas borealis nova | 71.43% | 39.29% | +32.14 pp | 84 |
| Card action | overall | 73.99% | 59.36% | +14.62 pp | 23,706 |
| Card action | tharsis | 73.81% | 59.22% | +14.59 pp | 16,951 |
| Card action | elysium | 84.62% | 68.21% | +16.41 pp | 195 |
| Card action | hellas | 74.46% | 59.77% | +14.68 pp | 6,381 |
| Card action | vastitas borealis nova | 62.01% | 48.04% | +13.97 pp | 179 |
| Tile placement (space) | overall | 47.64% | 28.51% | +19.12 pp | 61,262 |
| Tile placement (space) | tharsis | 48.69% | 29.59% | +19.09 pp | 45,232 |
| Tile placement (space) | elysium | 37.76% | 30.47% | +7.29 pp | 535 |
| Tile placement (space) | hellas | 45.08% | 25.30% | +19.78 pp | 15,159 |
| Tile placement (space) | vastitas borealis nova | 37.80% | 25.00% | +12.80 pp | 336 |

### Value head (post-temperature)

Temperature 1.3556.

| Scope | n | Accuracy | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| gen 1 | 2,662 | 61.87% | 0.2293 | 0.0353 |
| gen 2-5 | 26,267 | 70.61% | 0.1869 | 0.0338 |
| gen 6-9 | 21,611 | 78.92% | 0.1412 | 0.0358 |
| gen 10+ | 7,591 | 82.45% | 0.1213 | 0.0559 |
| gen 2+ (all) | 55,469 | 75.47% | 0.1601 | 0.0376 |
| gen 2+ tharsis | 41,976 | 75.88% | 0.1575 | 0.0417 |
| gen 2+ elysium | 553 | 74.14% | 0.1825 | 0.1002 |
| gen 2+ hellas | 12,591 | 74.47% | 0.1662 | 0.0291 |
| gen 2+ vastitas borealis nova | 349 | 64.18% | 0.2191 | 0.1389 |
