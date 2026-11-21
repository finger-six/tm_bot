# v2 main warm start vs b1-v2: why light search collapses and where the VP goes

Measurement for [#57](https://github.com/finger-six/tm_bot/issues/57), feeding the fix tickets #58 (search/value) and #59 (space/payment).
Checkpoint `runs/warmstart-v3/best`, gate report `docs/gate-warmstart-v3.md` (checks 25/26 FAIL: own VP 57.3 < 75, margin −55.1 < −35).
Driver `tools/diag57.py` (this branch) plus the measurement hook `2eb84b6` (`ISMCTSAgent.last_root_stats`); no search default, gate threshold, encoder or training-data change.

## Method

- Driver: `tools/diag57.py` (one command per variant, see its docstring). It reuses the gate's pairing (`build_paired_specs`), runner (`play_games`), summary (`summarize_pairing`) and live-value recording, and swaps only the candidate agent.
- Schedule: the gate's seat-swapped `dev-*` games, 13 seeds × 4 maps × 2 seats = 104 games per variant vs `b1-v2` (5 seeds = 40 games for the self/v1 opponent reruns). No `eval-*` or `test` data.
- Base = the gate's `LIGHT_SEARCH`: 8 sims × K = 4 worlds, wave 4, c_puct 1.5, flat v1 play rules (action-phase root τ = 1.0, other phases τ = 0), `margin_weight` 0, FPU 0.
- Knobs changed one at a time: action-phase root rule (`--root-tau 0` = argmax visits; `--root prior_q` = π' ∝ prior · exp(3·Q) at τ = 1, unvisited children get the mean visited Q), FPU (`parent` = mean visited Q of the node, optionally − 0.2), wave size, leaf value (`tscale` = logit / T with T fitted on val gen ≥ 2 by `fit_midgame_temperature`, T = 1.524; `zero` = constant 0, i.e. P(win) 0.5), policy sub-choice τ, and `--hybrid` (`space`/`payment` sub-choices from `b1-v2`'s own heuristic, `--hybrid-kinds` to split them; **measurement only, never training data**, #36).
- Live value: the gate's diagnostic (value head on every multi-candidate candidate decision, outcome = final result), ECE/accuracy for gen ≥ 2.
- Wall times: one run at a time on the RTX 4080 Super, 8 workers, batch 32.

## Variant table

| variant | knobs | n | win % [95% CI] | own VP | margin | live ECE g≥2 (raw / T) | acc g≥2 | wall s |
|---|---|---|---|---|---|---|---|---|
| light | gate LIGHT_SEARCH | 104 | 1.0 [0.0, 2.9] | 57.3 | −55.0 | 0.289 / 0.305 | 0.723 | 597 |
| raw | `--sims 0` (root sampled from the prior, τ 1) | 104 | 21.2 [10.6, 32.7] | 80.4 | −17.5 | 0.373 / 0.356 | 0.538 | 181 |
| raw_sub0 | raw, sub-choices argmax | 104 | 24.0 [13.5, 34.6] | 76.3 | −27.4 | 0.331 / 0.303 | 0.569 | 182 |
| root_argmax | root τ 0 (argmax visits) | 104 | 10.6 [2.9, 20.2] | 74.7 | −25.7 | 0.339 / 0.343 | 0.582 | 696 |
| prior_q | root π' ∝ prior·exp(3Q), τ 1 | 104 | 11.5 [4.8, 20.2] | 76.8 | −22.0 | 0.346 / 0.344 | 0.585 | 686 |
| fpu_parent | FPU = parent Q | 104 | 1.9 [0.0, 4.8] | 63.5 | −44.5 | 0.295 / 0.310 | 0.697 | 627 |
| fpu_parent_r02 | FPU = parent Q − 0.2 | 104 | 1.9 [0.0, 4.8] | 64.2 | −42.8 | 0.282 / 0.299 | 0.722 | 633 |
| wave1 | wave size 1 | 104 | 25.0 [12.5, 38.5] | 72.9 | −30.2 | 0.292 / 0.270 | 0.629 | 1064 |
| wave1_argmax | wave 1 + root τ 0 | 104 | 22.1 [10.6, 34.6] | 75.2 | −27.6 | 0.307 / 0.288 | 0.603 | 1104 |
| wave1_priorq | wave 1 + prior_q | 104 | 22.1 [9.6, 38.5] | 74.4 | −26.8 | 0.263 / 0.249 | 0.617 | 1078 |
| leaf_tscale | leaf logit / 1.524 | 104 | 1.0 [0.0, 2.9] | 57.2 | −54.8 | 0.295 / 0.310 | 0.716 | 607 |
| leaf_zero | leaf value ≡ 0 | 104 | 3.8 [0.0, 8.7] | 64.7 | −41.9 | 0.343 / 0.356 | 0.645 | 775 |
| priorq_16x4 | prior_q at 16 × 4 (the #39 budget) | 104 | 10.6 [3.8, 18.3] | 72.7 | −29.0 | 0.308 / 0.306 | 0.645 | 1146 |
| hybrid_raw | raw + b1 space & payment | 104 | 50.0 [37.5, 62.5] | 90.7 | −1.8 | 0.183 / 0.124 | 0.640 | 172 |
| hybrid_space | raw + b1 space only | 104 | 48.1 [32.7, 63.5] | 89.1 | −3.1 | 0.209 / 0.156 | 0.629 | 167 |
| hybrid_pay | raw + b1 payment only | 104 | 26.0 [14.4, 38.5] | 81.8 | −14.2 | 0.338 / 0.309 | 0.534 | 176 |
| hybrid_best | prior_q (8 × 4) + b1 space & payment | 104 | 30.8 [19.2, 42.3] | 85.2 | −10.1 | 0.211 / 0.177 | 0.629 | 689 |
| raw_self | raw vs itself (raw) | 40 | 52.5 [45.0, 60.0] | 97.1 | +0.7 | 0.115 / 0.071 | 0.713 | 111 |
| raw_v1 | raw vs v1 warm start (raw) | 40 | 100.0 [100.0, 100.0] | 128.6 | +84.5 | 0.122 / 0.151 | 0.907 | 98 |

`light` and `raw` reproduce the gate exactly (gate: 1.0 % / 57.3 / −55.1 and 21.2 % / 80.4 / −17.5). Per-game margin SD is 18–25 VP (standard error ≈ 2 VP per 104-game mean; heavy-tailed at 28–41 VP for wave1*, raw_sub0), so the difference of two variants has SE ≈ 3 VP: gaps under ~6 VP are not resolved.
Win rate and margin disagree in places (wave1: best win rate, mid margin): the margin is the gated quantity, so the attribution uses it.

## Where the light-search deficit comes from

Gap to explain: light −55.0 → raw −17.5 = **37.5 VP of margin** (own VP −23.1, b1-v2 +14.4).

| axis | variant (best of axis) | margin | recovered of the 37.5 | reading |
|---|---|---|---|---|
| root move rule | prior_q / root_argmax | −22.0 / −25.7 | 33.0 (88 %) / 29.3 (78 %) | the main cause: τ = 1 sampling from 8-sim visit counts |
| virtual-loss spread | wave1 | −30.2 | 24.8 (66 %) | same mechanism from the other end; not additive with the root fix (wave1_argmax −27.6, wave1_priorq −26.8) |
| FPU | fpu_parent_r02 | −42.8 | 12.2 (33 %) | partial: fewer unvisited-child grabs, root still sampled ∝ N |
| value head (ranking) | leaf_zero | −41.9 | 13.1 (35 %) | the value head is worse than no value at all vs b1-v2 |
| value head (calibration) | leaf_tscale | −54.8 | 0.2 (0 %) | a monotone rescale changes nothing that matters |
| remainder | best search (prior_q) vs raw | −22.0 vs −17.5 | 4.5 not recovered | no search variant beats the raw policy |
| more search | priorq_16x4 vs prior_q | −29.0 vs −22.0 | −7.0 | doubling sims makes it worse: more weight on a misleading Q |

The axes overlap (the root rule and the wave both fix "a near-uniform spread of a few visits gets sampled"), so the shares do not sum to 100 %.

What the collapse looks like (`analyze` on the per-decision log, action-phase search roots only):

| variant | override of prior argmax | children visited / legal | pass chosen when prior top ≠ pass | std project chosen when prior top ≠ std project | action roots / game |
|---|---|---|---|---|---|
| raw | 31.3 % (τ 1 prior sampling) | – | 3.6 % | 2.5 % | 96 |
| light | 53.9 % | 4.05 / 4.82 | 11.5 % | 12.3 % | 64 |
| fpu_parent | 48.7 % | 3.55 / 4.83 | 8.7 % | 11.2 % | 70 |
| leaf_zero | 50.0 % | 3.48 / 4.94 | 6.4 % | 8.5 % | 83 |
| leaf_tscale | 53.7 % | 4.05 / 4.89 | 11.0 % | 12.1 % | 65 |
| root_argmax | 22.9 % | 3.96 / 5.03 | 3.9 % | 2.3 % | 79 |
| prior_q | 33.2 % | 3.95 / 5.18 | 6.0 % | 4.6 % | 80 |
| wave1 | 24.1 % | 2.11 / 5.18 | 6.5 % | 2.8 % | 79 |

Light search visits almost every top-level branch once or twice and then samples one ∝ N, so it picks a move the prior ranks second or lower in 54 % of roots (mean prior of the chosen move 0.41 vs 0.71 for the prior argmax). The cost is concrete: it **passes early** (11.5 % of non-pass roots, 3× raw) and **buys standard projects** the prior does not want (12.3 %, 5× raw), so it takes 64 action-phase decisions per game instead of 96 in the same ~12 generations. In VP terms (approx. split from the last observation, `GameResult` has totals only), light loses to raw on cards/other (6.5 vs 16.3), milestones (3.8 vs 8.4), greenery (3.9 vs 5.9), TR (33.7 vs 38.7) and awards (5.0 vs 7.3), and lets b1-v2 score 112 instead of 98. With leaf ≡ 0 the std-project excess shrinks (8.5 %), so the value head adds a preference for immediate-TR moves on top of the noise.

## The value head vs b1-v2

| opponent | live ECE g≥2 raw / T=1.524 | acc g≥2 | mean live P(win), gen 6–9 | actual win rate |
|---|---|---|---|---|
| val split (gate check 14) | 0.061 / – | 0.758 | – | – |
| self (raw_self, 40 g) | 0.115 / 0.071 | 0.713 | 0.590 | 0.525 |
| v1 warm start (raw_v1, 40 g) | 0.122 / 0.151 | 0.907 | 0.832 | 1.000 |
| b1-v2 (raw, 104 g) | 0.373 / 0.356 | 0.538 | 0.691 | 0.210 |
| b1-v2 (hybrid_raw, 104 g) | 0.183 / 0.124 | 0.640 | 0.716 | 0.490 |

The live miscalibration is **specific to b1-v2**: against itself and the v1 warm start, the value head is within ~0.05–0.12 of val ECE, and T fitted on val fixes the self-mirror case (0.115 → 0.071). Against b1-v2 it predicts ~0.69 P(win) through gen 2–9 for a 21 % win rate and does not separate wins from losses (accuracy 54 %, near chance). Temperature cannot repair a ranking that is wrong, which is why `leaf_tscale` equals `light` and `leaf_zero` beats both. The over-optimism falls with hybrid space/payment play, so part of it is the value head not seeing the board-placement deficit that b1-v2 exploits.

## space / payment share (E)

Raw policy's own deficit (−17.5) and its headroom above the −35 floor (17.5 VP) are almost entirely tile placement:

| variant | margin | Δ vs raw | own VP | b1-v2 VP | city-adjacency VP (own) |
|---|---|---|---|---|---|
| raw | −17.5 | – | 80.4 | 97.9 | 3.8 |
| hybrid_pay (b1 payment) | −14.2 | +3.3 (not resolved) | 81.8 | 96.0 | 3.9 |
| hybrid_space (b1 space) | −3.1 | **+14.4** | 89.1 | 92.2 | 9.5 |
| hybrid_raw (b1 space + payment) | −1.8 | +15.7 | 90.7 | 92.5 | 10.2 |
| raw_sub0 (policy sub-choices argmax) | −27.4 | −9.9 | 76.3 | 103.6 | 3.7 |
| hybrid_best (prior_q 8×4 + b1 space + payment) | −10.1 | +7.4 | 85.2 | 95.3 | 10.9 |

- `space` explains ~14 of raw's 17.5 VP margin deficit (82 %; win rate 21 % → 48 %); `payment` ~3 (within noise). The space gain is own VP +8.7 (+5.7 of it city adjacency: greenery next to its cities) and b1-v2 −5.7 (placement contest).
- Argmax over the policy's sub-choices is *worse* than sampling them (−27.4 vs −17.5, ~2σ): the policy head's top space/payment candidate is systematically poor, not just noisy.
- Search does not stack with better placement: hybrid_best (−10.1) is 8.3 VP below hybrid_raw (−1.8), the same order as prior_q's 4.5 VP below raw.
- Per-category VP (approx.; raw vs b1-v2 in the same games): TR 38.7 vs 48.6, greenery 5.9 vs 8.3, city adjacency 3.8 vs 5.9, milestones 8.4 vs 6.4, awards 7.3 vs 7.7, cards/other 16.3 vs 20.9. The TR gap (−9.9) is the largest single line and only partly closes with b1 placement (39.7 vs 47.2 in hybrid_raw), so the rest is action-level play, which this note does not split further.

## Confirmed code facts

1. **Virtual loss spreads each wave across the top priors.** `_select_puct` scores in-flight children with q = (W − VL)/(N + VL) (`tm_bot/search/agent.py:121-126`), so a child already taken in the current wave scores below FPU 0; VL is added per selection at `tm_bot/search/agent.py:438`; the gate's wave size is 4 (`tm_bot/training/run_warmstart_gate.py:59`, applied at `tm_bot/search/agent.py:378`). With 8 sims in 2 waves, sims 1–4 take the top-4 priors once each.
2. **FPU 0 pulls visits to unvisited children when behind** (`tm_bot/search/config.py:42`, used at `tm_bot/search/agent.py:125`): with visited Q < 0 (the usual state vs b1-v2) every unvisited child (Q = 0) outranks the visited ones in wave 2, so most legal branches end with N = 1.
3. **The action-phase root is sampled ∝ N at τ = 1** (`tm_bot/search/agent.py:699-704`); τ comes from `SearchConfig.decision_temperature` (`tm_bot/search/config.py:76-90`, flat `action_temperature` 1.0 returned at `:88`; gate setting `tm_bot/training/run_warmstart_gate.py:58`). The τ = 0 path (`tm_bot/search/agent.py:682-692`) takes argmax N, ties by single-sample Q then prior.
4. **Suspect 3 (value scale / perspective) is refuted.** Value = 2σ(logit) − 1 (`tm_bot/models/network.py:464`); leaf value is signed to the root actor (`tm_bot/search/agent.py:636-645`); selection applies the selector's sign (`:452`); backup stays in root perspective (`:664`); terminal utilities are ±1 from the root actor (`:78-96`). No sign or scale error; nothing to sketch a failing test for.
5. **Policy sub-choices are sampled at the action-phase τ** (`tm_bot/search/agent.py:340-345`, τ 1.0 under the gate's flat rules) for both raw and search play; nested sub-choices inside simulations use argmax (`:599`). `b1-v2` pays with candidate 0 (`tm_bot/agents/b1.py:140`) and places tiles with `_choose_space` (`tm_bot/agents/b1.py:149`, `740-809`).
6. **Raw policy (`--sims 0`) samples the root from the prior** at τ 1 (`tm_bot/search/agent.py:674-680`, `total_N = 0`).
7. `GameResult` carries totals only (`tm_bot/env/types.py:52-56`); the per-category VP split here is reconstructed from the candidate's last observation (TR, greenery tiles, city-adjacent greenery, milestones, funded-award winners; the rest = cards/other) and is approximate (award ties, last-observation lag).
8. The v2 default search config differs from the gate's pinned v1 rules: 16 sims × 4 worlds (`tm_bot/search/config.py:17-19`) and τ(g) = 1.0 through gen 4, linear to 0.3 at gen 10 (`tm_bot/agents/play_rules.py:26-32`, `V2_PLAY_TEMPERATURE` `:70`), wave 4 by default in `make_warmstart_search_agent` (`tm_bot/search/agent.py:837-842`).

## Recommendation

**#58 (search / value).**
1. Root rule first. Never sample the action-phase root ∝ N at τ = 1 with 8 sims and wave 4 (16 sims was only measured with the fixed root rule, where it was worse): the visit counts are a near-uniform spread over ~4 children, not a policy. For evaluation use either argmax visits (`--root-tau 0`) or π' ∝ prior · exp(3·Q) (`--root prior_q`). The only gate settings measured to pass checks 25/26 vs b1-v2 are:
   - `LIGHT_SEARCH` (8 × 4, wave 4, c_puct 1.5, FPU 0, raw leaf) + action-phase root π' ∝ prior · exp(3·Q), unvisited Q = mean visited Q, sampled at τ 1: own VP 76.8 (≥ 75), margin −22.0 (≥ −35). PASS 25 by 1.8 VP, inside one CI, so not a robust pass.
   - wave 1 + argmax visits: 75.2 / −27.6. PASS by 0.2 VP, at 1.6× wall time.
   - argmax visits at wave 4: 74.7 / −25.7. FAILS 25 by 0.3 VP.
   Checks 19–24 were not rerun under these settings.
2. Value: temperature scaling is not the fix for search. T = 1.524 (val NLL 0.494 → 0.472, Brier 0.163 → 0.158) changes play by nothing (leaf_tscale = light). It may still be worth adopting for gate check 14 (val ECE) on its own. The live failure is ranking: against b1-v2 the value head is 54 % accurate and over-optimistic (P(win) ≈ 0.69 at a 21 % win rate), while against itself and v1 it is fine. A constant leaf beats it (−41.9 vs −55.0), and doubling sims hurts (−29.0 vs −22.0). #58 should measure value-ranking quality per opponent and train on outcomes of games against b1-v2-style opponents, using outcome labels only; whether that is compatible with #36 is for the issue owner. Until then, add a "search ≥ raw policy vs b1-v2" diagnostic, because every search variant here plays worse than the raw policy.
3. Lower priority: wave size and FPU. Wave 1 helps win rate but costs 1.6–1.8× wall time and does not stack with the root fix. FPU = parent Q (− 0.2) recovers ~12 VP alone; it was not measured together with the root fix.
4. Self-play (#39) uses the v2 defaults: 16 × 4, wave 4, τ(g) = 1.0 through gen 4, and root sampling ∝ N. The same spread therefore lands in early-generation self-play moves and visit targets. Apply the root-rule fix there too (sample from π', or keep τ but sample from the prior re-weighted by Q).

**#59 (space / payment supervision).** Space placement is the lever; payment is minor. b1-v2's own space heuristic alone takes the raw policy from −17.5 to −3.1 (win rate 21 % → 48 %), about 82 % of raw's deficit and well clear of the −35 floor. Payment alone is worth ~3 VP and is not resolved. The policy head's argmax space choice is worse than sampling, so the space head needs better targets or features. City adjacency (own cities next to greenery) is the visible symptom. #59 should prioritise `space` and treat `payment` as a cheap follow-up. The hybrid is a ceiling measurement, never a data source (#36).

## Not measured

- The other search-dependent gate checks (19–24: vs b0-v1, self-mirror length, vs v1) under the recommended settings; the driver only pairs against b1-v2, self and v1, and the self/v1 reruns were raw policy.
- The value head's ranking on b1-v2 games offline (AUC by generation): there are no labelled b1-v2 positions in the training data, only these live rows.
- Per-category VP is reconstructed, not read from the engine's score breakdown (fact 7).
- Search variants with more than 16 sims, Dirichlet noise, or c_puct changes; and the v2 τ(g) schedule itself (the gate pins flat τ 1).
- Statistical significance between close variants (root_argmax vs prior_q vs wave1_*): margins within ~5 VP of each other are not separated at n = 104.

## Reproduce

```
# from the repo root, with data/ and runs/ present; W outside the repo
conda run -n tm_bot python tools/diag57.py fit-t --work-dir W
conda run -n tm_bot python tools/diag57.py run --work-dir W --name light
conda run -n tm_bot python tools/diag57.py run --work-dir W --name prior_q --root prior_q
conda run -n tm_bot python tools/diag57.py run --work-dir W --name hybrid_raw --sims 0 --hybrid
conda run -n tm_bot python tools/diag57.py run --work-dir W --name hybrid_space --sims 0 --hybrid --hybrid-kinds space
conda run -n tm_bot python tools/diag57.py table --work-dir W
conda run -n tm_bot python tools/diag57.py analyze --work-dir W light raw prior_q hybrid_raw
```
