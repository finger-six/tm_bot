# Human archive: action-phase, research-buy, and mid-game supervision audit

Research for [Audit human archive action-phase, research-buy, and mid-game supervision](https://github.com/finger-six/tm_bot/issues/37), 2026-11-18. This report measures what the eligible TFMStats tables can supply for action-phase `Decision` examples, generation-2+ research-buy examples, and mid-game value supervision. It proposes gates and yield bounds; it does not change the extraction pipeline or select a training-data policy.

## Finding

- **Snapshots are usable with per-field timing rules.** Every move row carries a `game_state` whose `move_number` equals the row's `move_number` (63,595/63,595 sampled rows). Oceans (765/765), milestone claims (162/171), and award fundings (234/253) are already post-move on their own row, so row *k−1*'s snapshot is a valid pre-move observation for those fields. M€ payments post on the payment row itself in 78.5% of checked rows (6,714/8,548) and never post within 5 rows in 21.2%; hand counts post with a 1-row netting lag around buys/draws. The generation-start M€ is stale by the production income and is **exactly** recoverable as `snapshot M€ + M€ Production + TR − payments since the draft deal` (verified example below); this closes the bulk of [docs/warmstart.md](../docs/warmstart.md) Fidelity Limit 2 for gen 2+ drafts.
- **Research buys (generation ≥ 2) are reconstructable and verifiable.** The offered packet is the player's four drafted cards (3 verified draft picks + the final 1-card pass packet); kept cards are structured (`cards_kept`), the payment is `3 M€ × |kept|`, and discarded drafted cards follow by set difference. On the sample, 1,813/1,872 round-matched keep rows (96.8%) pass all checks; every eligible scraper version passes 100% except `v2025.08.15.1706`, whose rows omit the payment text (59 rows, 0% verifiable). No corporation in the supported format pays a non-standard card price (details below).
- **Action-phase root choices are attributable and mostly state-checkable, but full legal-candidate enumeration is not possible without engine replay.** 76.3% of root action rows (79.0% among setup-passed, non-conceded tables) carry an exactly-tracked hand for the acting seat; only 1 of 7,452 tracked plays was not in the tracked hand. Chosen branches (card play, card action, standard project, greenery conversion, heat conversion, milestone claim, award funding, pass) are unambiguous and their legality can be re-checked against the pre-move snapshot; the *complete* root branch menu and nested payment/space candidate sets cannot be proven from the log.
- **Corpus-scale yields (concession-weighted, per supported map) are tabulated below** for state-enriched drafts, research buys, action-phase roots, exact-hand card plays, and mid-game value points on outcome-verified tables.

## Method and limits

Primary sources: `C:\2026\tm_bot\data\archive_index.sqlite` (queried read-only, `mode=ro`), `C:\2026\tm_bot\data\card_registry.json`, `C:\2026\tm_bot\data\cards.json`, `C:\2026\tm_bot\data\warmstart\yield_report.json`, the repo gates/encoder/builder/adapter sources, and bounded samples of `C:\2026\tm_bot\game-logs-archive-2026-01-10-023900\`. The archive directory was never enumerated directly; file paths came from SQLite queries, and files were parsed one at a time.

1. **Index funnel reproduction.** Streaming all 111,982 index rows through the repo's own `check_index_row` ([tm_bot/data/gates.py:91-137](../tm_bot/data/gates.py)) reproduces `yield_report.json` exactly: 63,001 index-passing files, 101,194 distinct `table_id`s, 59,323 distinct tables with ≥1 index-passing perspective. The remaining gap to 58,794 fully-eligible tables is 529 parse-level `unresolvable_card_name` tables (yield_report `tables.quarantined_by_reason`), which cannot be attributed per version without parsing.
2. **Stratified sample.** 92 index-eligible tables (91 unique; table `726846523` appears in two version strata and is deduplicated in aggregates), spread across 10 eligible scraper versions and all four supported maps: Tharsis 33, Hellas 24, Elysium 20, Vastitas Borealis 14; 63 non-conceded / 29 conceded. Per-table perspective selection mirrors `_pick_best_perspective` ([tm_bot/data/extract.py:127-140](../tm_bot/data/extract.py)): prefer `conceded=0`, then row count, then file size. Sample pass rates under the repo's full gates ([tm_bot/data/gates.py:371-483,657-784](../tm_bot/data/gates.py)): setup 85%/100%/95%/100% per map; outcome 55%/67%/60%/79%.
3. **Per-row measurement.** Each sampled file was parsed once (63,595 rows total) and every row was classified: draft deal/pick, research/effect keep, draw, play/action/placement/conversion/milestone/award/pass roots, resolutions, cancels, sell-patents discards. The same pass re-ran the repo gates (`evaluate_record`) to keep pipeline-comparable counts, tracked both seats' hands (projects + unplayed preludes), and measured snapshot-timing lags per version.

Limits: the sample is ≤1 file per table, so intra-table multi-perspective recovery is not measured here. Version strata with tiny eligible populations (`dev`, `v2025.08.15.1706`, `v2025.11.01.1808`) are represented but noisy. Elysium/Vastitas extrapolations carry wide confidence intervals from small samples. "Never posts within 5 rows" lags were not chased to end-of-file. No claim below depends on uninspected files.

## 1. Snapshot coverage across eligible scraper versions

### 1.1 Scraper-version distribution (index-eligible tables)

Reproduced by applying `check_index_row` (with `DEFAULT_EXCLUDED_SCRAPER_VERSIONS`, [tm_bot/data/gates.py:19-38](../tm_bot/data/gates.py)) to every index row, grouped by distinct `table_id` with ≥1 passing perspective. Multi-version eligible tables: 2,084.

| Scraper version | Tharsis | Hellas | Elysium | Vastitas | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v2025.10.07.1844` | 19,617 | 10,869 | 208 | 176 | 30,870 |
| `v2025.08.27.1615` | 13,210 | 217 | 96 | 122 | 13,645 |
| `v2025.08.16.0012` | 3,654 | 21 | 21 | 17 | 3,713 |
| `v2025.10.07.1845` | 2,018 | 689 | 35 | 20 | 2,762 |
| `v2025.12.27.1015` | 850 | 1,394 | 66 | 55 | 2,365 |
| `v2025.10.07.1547` | 1,269 | 541 | 0 | 5 | 1,815 |
| `v2025.08.15.2331` | 654 | 12 | 11 | 9 | 686 |
| `dev` | 83 | 2 | 7 | 2 | 94 |
| `v2025.08.15.1706` | 48 | 0 | 1 | 1 | 50 |
| `v2025.11.01.1808` | 5 | 0 | 0 | 0 | 5 |
| **All eligible** | **43,792** | **14,517** | **608** | **406** | **59,323** |

For context, the 12 excluded versions ([tm_bot/data/gates.py:23-38](../tm_bot/data/gates.py)) account for the bulk of the quarantine funnel (e.g. 6,281 Tharsis tables exist only under `v2025.08.06.1719`; corpus-wide `excluded_scraper_version` = 8,992 tables). Index-level `conceded=0` subsets: Tharsis 22,366, Hellas 6,490, Elysium 367, Vastitas 250 (outcome-verified per `yield_report.json` `by_map`: 19,878 / 6,169 / 333 / 232).

**Schema uniformity.** All 10 eligible versions in the sample expose the same per-row schema (`move_number, timestamp, player_id, player_name, action_type, description, card_played, card_drafted, tile_placed, tile_location, card_options, cards_kept, reason, game_state`) and the same `game_state` keys (`move_number, generation, temperature, oxygen, oceans, player_vp, milestones, awards, player_trackers`). Two degenerate file styles were found and must be gated on: `90190162/game_636520303_90190162.json` (table `636520303`, `v2025.10.07.1845`) has empty `player_trackers` on all 730 rows and unclassifiable `'123alex other'`/`'123alex draw'` descriptions; `84881377/game_583173353_84881377.json` (table `583173353`, `dev`) logs drafts as `action_type='draft_card'` with `card_drafted=null`, packets only in description text, pronoun attribution (`You draft X`) only for the perspective seat, and `pass Undo` markers. Structured draft fields are therefore version- and file-quality-dependent, not version-guaranteed.

### 1.2 Timing: which snapshot is pre-decision?

- `game_state.move_number == row.move_number` on 63,595/63,595 rows.
- **Post-move on the same row** (so row *k−1* is a valid pre-move source): oceans `765/765`; milestone claims `162/171` (the snapshot `milestones` dict gains the claimed milestone with `claimed_by/player_id/move_number`, e.g. table `643390962` move 122); award fundings `234/253` (e.g. table `643390962` move 196, `awards["Celebrity"].funded_by` present on the funding row).
- **Generation increments on the draft-deal row**: 1,034/1,034 generation increments occur on the row that deals the next draft (the pass-transition row that also carries the production text, e.g. table `643390962` move 29).
- **M€ payments** post on the payment row in 78.5% of isolated checks (per-version 74–81%), post with a 1–5 row lag rarely, and never post within 5 rows in 21.2% (1,814/8,548); `v2025.08.15.1706` writes payments without the `M€` unit (`'Niiyaaa pays 10 | Niiyaaa pays 1'`, table `684318561` move 11) and its M€ tracker is unreliable.
- **Hand counts**: play decrements post at lag 0 in 61.6% of isolated checks (5,294/8,592); buys/draws post netted into the actor's next row in several versions; some versions never post draw increments at all (table `713816826`, `v2025.10.07.1845`, moves 41–47: three named draws leave the `Hand Counter` unchanged while the tracked hand advances). The `Hand Counter` is therefore a cross-check, not a source of truth; on isolated comparison rows it agrees exactly with the reconstructed hand in 24.8% of cases overall (3,070/12,355) and ±1 in 13.3% more, best on `v2025.10.07.1844` (40% exact, 14% ±1).
- **Generation-start M€ correction (exact).** At the draft-deal row the M€ tracker is stale by the production income. The actor's true M€ at the research/draft decision is `snapshot M€ + M€ Production + TR − payments since the deal`. Worked example, table `643390962` (`96940540/game_643390962_96940540.json`): at move 29 the snapshot shows M€ 3, `M€ Production` 2, `player_vp.tr` 21 → 3+2+21 = 26; subtract the research buy of move 36 (`pays 6 M€`) and move 38's card payment (`pays 4 M€`) → 16, exactly the M€ shown at move 38. Same identity on the other seat (3+1+20−6−17 = 1, the M€ at move 42). Under payment-only netting the corrected value re-appears within 60 rows in 59.3% of 2,212 deal-row checks (per-version 48–68%, except `v2025.08.15.1706` at 3%); the misses involve un-netted M€ gains (triggered effects) and the versions with broken M€ trackers, not a failure of the identity.

### 1.3 What the 185-scalars observation can and cannot be populated with

The 185 `OBS_SCALAR_DIM` layout ([tm_bot/models/encode.py:44-51](../tm_bot/models/encode.py)) mapped to log sources (snapshot = row *k−1* `game_state` + timing rules above):

| Scalar group (count) | Populatable before a decision? |
| --- | --- |
| Map one-hot (5) | Yes — root `map`, exact. |
| Actor/turn (6) | Actor seat yes (row `player_id` + starting-player resolution, [tm_bot/data/gates.py:268-294](../tm_bot/data/gates.py)); `firstPlayer`/`activePlayer`/`passed` only via move-history reconstruction (`X is starting player for this generation`, pass rows), not in snapshots. |
| Generation/phase (11) | `generation` exact (snapshot); phase never in snapshots (inferable from row kind); gen==1 derivable. |
| Global parameters (5) | Yes — `temperature/oxygen/oceans` exact; `isTerraformed`/remaining-steps derivable. |
| Deck sizes (6) | **No — never recorded** in any eligible-version snapshot. |
| Public players (2×40) | TR exact (`player_vp.*.total_details.tr`); 6 stocks + 6 productions exact (`player_trackers`); `citiesCount`/`greeneries` exact (`player_vp.*.total_details`); 11 tag counts + played-events exact; `cardsInHandCount` exact-ish (`Hand Counter`, broken in some versions — use the tracked hand); `steelValue/titaniumValue/cardDiscount` 0 direct (reconstructable from corporation + played cards, e.g. `Advanced Alloys` rows say `gains 1 Steel Exchange Rate`); `cardCost` known from setup corporation; `noTagsCount`, `actionsTaken*`, `availableBlueCardActionCount` derivable by replaying rows (activate rows give used-actions per generation). ≈ 30/40 direct per seat. |
| Milestones (5×7) | `claimedBy` exact (snapshot `milestones` dict); per-milestone progress **scores not recorded** — needs engine recompute. |
| Awards (5×5) | `fundedBy` exact (snapshot `awards` dict); scores not recorded. |
| Private/history (12) | Hand count + dealt/packet/drafted/kept-discarded all reconstructable (setup + draft + keep rows); `pendingCardSelection` never in snapshots. |

Card tokens and board: hand *identity* is not in snapshots but is trackable (section 3); public tableaus reconstruct from cumulative `play_card` rows (size cross-check: `Player Area Counter` counts tableau cards including corporation and preludes); per-card stored resources are **not** recorded (only a `Microbe` aggregate; no floater/animal resource fields), so the 4 dynamic per-instance features are only partially populatable. Board spaces are entirely absent from snapshots and must be rebuilt from placement rows: 65.8% of placement rows have structured `tile_placed`/`tile_location` (e.g. `'Hex 6,5 (6,5)'`, table `643390962` move 16); 34.2% are text-only (e.g. `'Roguestorm places Mining on Elysium Mons (2,3)'`, table `753473373` move 28).

**Fidelity-limit-2 verdict.** The neutral defaults in `_base_public_observation` ([tm_bot/training/warmstart_builder.py:118-157](../tm_bot/training/warmstart_builder.py)) can now be replaced at gen 2+ draft rows with: true globals, TR, all stocks/productions, tag counts, hand counts, milestone claims, award fundings, and the kept-vs-discarded research-buy outcome (keep rows give both sets). Still not closable from logs alone: deck/discard sizes (6 scalars), milestone/award progress scores (60 scalars), per-card resources, and payment-mode detail.

**Value-target enablement.** On outcome-verified tables (26,612 corpus-wide; `check_outcome` validates final VP against terminal text, [tm_bot/data/gates.py:688-774](../tm_bot/data/gates.py)), the deal row of every generation ≥ 2 yields one pre-research observation per seat with the corrected M€ (24.4 points per table at the sample mean of 12.2 gen≥2 deal rows), and the outcome gate supplies `P(win)` plus the final VP margin (`scores_by_player`). Extrapolated per map in section 4.

## 2. Research-phase (generation ≥ 2) card-buy reconstruction

**Mechanics (verified on sample).** A generation's draft round is dealt on the transition row that also ends the previous generation (e.g. table `643390962` move 29: `card_options` carries both seats' 4-card packets). Each seat makes 3 explicit picks (`action_type='draft'`, `card_drafted`, pre-pick packet in the previous row's `card_options` — this is the existing verified-draft logic, [tm_bot/data/gates.py:508-644](../tm_bot/data/gates.py)); the 4th card arrives on the final 1-card pass packet carried by the last pick row. The offered research packet is therefore the seat's 3 picks + final pass card (4 cards). The buy row then records `cards_kept` (the bought subset) with `NAME keeps N cards: ...` plus `NAME pays 3N M€` (newer versions also set `action_type='buy_card'`, e.g. table `753473373` moves 43–44, `pays 9 M€` for 3 cards). Discarded drafted cards = packet minus kept. Re-running the repo's draft gate on the sample yields 5,736 verified picks (63.0 per table; drops: 36 missing-pre-pick-packet, 80 undone), mirroring the corpus-wide drop mix in `yield_report.json`.

**Card price rule (primary sources).** The pinned engine's default is `CARD_COST = 3` ([tools/simbench/engine/src/common/constants.ts:2](../tools/simbench/engine/src/common/constants.ts), applied at [adapter/src/factor.ts:631-636](../adapter/src/factor.ts) as the initial-buy cap `floor(startingMegaCredits / cardCost)`). The only corporations with a non-default buy price are `Polyphemos` at 5 M€ ([tools/simbench/engine/src/server/cards/colonies/Polyphemos.ts:12](../tools/simbench/engine/src/server/cards/colonies/Polyphemos.ts), Colonies — excluded by the eligibility gate) and `Terralabs Research` at 1 M€ ([tools/simbench/engine/src/server/cards/turmoil/TerralabsResearch.ts:13](../tools/simbench/engine/src/server/cards/turmoil/TerralabsResearch.ts), Turmoil — excluded). `Teractor` (the base-game corporation the issue calls "Terractor") has **no** `cardCost`; its effect is a 3 M€ discount on *playing* Earth-tag cards ([tools/simbench/engine/src/server/cards/corporation/Teractor.ts](../tools/simbench/engine/src/server/cards/corporation/Teractor.ts)). Neither of the 18 corporations in `C:\2026\tm_bot\data\cards.json` sets `cardCost`. **Conclusion: in the supported format every seat pays exactly 3 M€ per bought card, both in the initial buy and in gen ≥ 2 research.** No 2 M€ corporation exists in this format.

**Measured verification (91 tables, 1,872 round-matched keep rows).**

| Check | Result |
| --- | --- |
| kept ⊆ drafted-4 (3 picks + final packet) and \|kept\| ≤ 4 | 1,872/1,872 rows that matched a round |
| payment text == 3 M€ × \|kept\| | 1,813/1,872 (96.8%) |
| by version | 100% in every eligible version except `v2025.08.15.1706` (0/59: payment text stripped, e.g. table `684318561` moves 39–40 keep 3 cards with no `pays` segment) |
| drafted-4 reconstruction complete (both seats × 4 cards) | 920/1,135 rounds (81.1%); failures are the degenerate `dev`/`1706` deal formats, undone/re-dealt rounds, and setup-artifact rows |
| cards bought / drafted cards discarded | 4,709 bought / 2,543 discarded on matched rows |
| look-and-buy effects (Business Network / Inventors' Guild style) | 630 keep rows resolved against a named draw-offer; 485 (77.0%) carry the 3 M€ payment text |

Pre-buy affordability (kept × 3 ≤ true M€) follows from the payment text plus the exact income identity of section 1.2; no negative-M€ buy is possible in the platform's enforcement, and every observed payment matched `3 × |kept|`. Sample citations: clean — table `643390962` move 36 (keeps 2 of `[Protected Habitats, Adapted Lichen, Great Escarpment Consortium, Ice cap Melting]`, pays 6), move 64 (keeps 3, pays 9); table `753473373` move 43 (`buy_card`, keeps 3, pays 9). Unverifiable — table `684318561` move 40 (keeps 3, no payment text in `v2025.08.15.1706`).

The reconstructed `Choice` matches the live adapter's `multi_card` research-buy frame ([adapter/src/factor.ts:464-486](../adapter/src/factor.ts)): candidates = unbought drafted cards + `done`, with the adapter's `maxOverride` never binding because no eligible corporation changes `cardCost`.

## 3. Action-phase decision reconstruction without engine replay

### 3.1 Root-choice taxonomy and attribution

Root action rows are identifiable by `action_type` plus description verbs. Per-table means across both seats (91 tables, 18,973 root rows): 208.5 root rows per game — card plays 104.3, card actions 60.3, standard projects 13.8 (Sell patents 6.4, City 2.5, Greenery 2.4, Power plant/Aquifer/Asteroid ≈ 2.5), greenery conversions 6.3, heat conversions 6.0, milestone claims 3.0, award fundings 2.8, passes 12.1. A card-play decision with tile placement spans two rows (play + nested placement), e.g. table `643390962` moves 15–16; the nested space candidates are never recorded. Greenery conversion appears as a `places Greenery ... increases Oxygen` row with no card play (table `753473373` move 51, `713816826` move 57); heat conversion as `pays 8 Heat | increases Temperature` (table `753473373` move 589). Actor attribution is reliable via `player_id` — the description pronoun `You` is **not** (`You draft Cloud Seeding` on a row whose `player_id` is the opponent, table `643390962` move 35) — and misclassified `action_type` is rare but real (a card play logged with `action_type='pass'`, table `734242321` move 236).

### 3.2 Continuous hand tracking (the actor's own cards)

Tracking both seats from the verified setup (`chosen_projects` + unplayed preludes), with events for research buys, look-and-buy keeps, named draws (3,930 of 3,931 draw rows in the sample name the drawn cards; the one exception is an unparseable name — no `private:`-prefixed or count-only draws in any eligible version), plays, prelude plays, named discards, and Sell-patents discards:

- **76.3% of root action rows have an exact known hand** (14,478/18,973); 79.0% among setup-passed, non-conceded tables (9,605/12,165). Draft rows inherit the same tracker, so state-enriched draft picks get exact hand identity for free.
- **50/91 games keep both seats' hands exact end-to-end.** Hand loss is driven by: setup-gate failures (6 sample files, quarantined upstream anyway), partial Sell-patents ambiguity (`discards N card/s (Sell patents)` names a count, not identities — 61 events; e.g. table `753473373` move 315), and one residual mismatch (table `593543494` move 213).
- **Plays vs tracked hand: 7,451 found (7,112 project plays + 339 prelude plays), 1 not found** (0.013%). Cancellations (`cancels their move`, ~7.6 rows/game) revert *unlogged* in-progress moves: logged rows survive them (table `753473373` moves 101→106: a research buy survives a later cancel by the same seat).
- The snapshot `Hand Counter` includes unplayed preludes and posts with netting lags (section 1.2); use it only as a cross-check.

### 3.3 What can vs cannot be verified as legal `Candidate`s

| Tier | Verifiable from logs (with pre-move snapshot) | Not verifiable without engine replay |
| --- | --- | --- |
| Root branch choice | Chosen branch unambiguous for all 8 root kinds; legality re-checkable for: pass (always), heat conversion (Heat ≥ 8, temperature < 8), greenery conversion (Plant ≥ 8, oxygen < 14), milestone claim (unclaimed + M€ ≥ 8), award funding (funded count + 8/14/20 M€ cost, e.g. table `643390962` moves 196/439/484), standard project (name + fixed cost vs M€), card play (card ∈ tracked hand; cost vs M€/Steel/Titanium with corporation/tableau discounts recomputable from `cards.json`) | The **complete** branch menu (proving "no legal Play project card" requires full playability of every hand card; blue-card action availability requires per-card used-action state beyond activate rows) |
| Card-play selection | Which card was played (`card_played`), from which seat, from a tracked hand; payment components partially in text (`pays 3 Steel | pays 3 M€`) | Canonical payment enumeration ([adapter/src/payments.ts](../adapter/src/payments.ts) `enumerateCanonicalPayments` — steel/Titanium/heat splits unrecorded); requirement-heavy playability beyond global thresholds; per-card resource prerequisites (only a `Microbe` aggregate is recorded) |
| Nested choices | Chosen tile (`tile_placed`/`tile_location`, 65.8% structured; text-only otherwise), chosen amount (Sell-patents discard count), chosen target | Space candidate sets (never logged); effect branches (`add X microbes to card Y` rows give the chosen card but not the legal set); look-and-buy offers are reconstructable (draw row names them) but only when the draw row itself is logged |

Concrete quarantines: bundled rows join a root verb with effect/resolution segments rather than two root verbs (no sampled row carries two root verbs; 998 of ~2,097 pass rows — 47.6% — bundle the generation transition, the next draft deal, and production income, e.g. table `643390962` move 29; the terminal row bundles a greenery placement with end-of-game scoring, table `643390962` move 796; table `734242321` move 236 mislabels a play as `pass`); hidden per-card resources (no floater/animal fields in any snapshot); text-only tile coordinates (34.2% of placement rows); degenerate `'NAME other'` rows (11,653 rows = 18.3% of the sample — mostly effect resolutions, not decisions, but they must be classified before pairing observations); the two degenerate-file styles of section 1.1.

## 4. Estimated corpus yields and quarantine rates

Per-table means from the sample (95% CI), concession-weighted for the setup-passed populations (conceded share among setup-passed: Tharsis 0.50, Hellas 0.56, Elysium 0.42, Vastitas 0.39), then multiplied by the corpus populations from `yield_report.json` (`passed_setup` and `passed_outcome` per map). Value points use non-conceded means only and the outcome-verified population.

| Task (population) | Tharsis | Hellas | Elysium | Vastitas |
| --- | ---: | ---: | ---: | ---: |
| Setup-passed tables | 40,670 | 14,253 | 579 | 386 |
| Outcome-verified tables | 19,878 | 6,169 | 333 | 232 |
| Verified draft picks (setup-passed) | 2.40M [2.03, 2.78] | 0.98M [0.92, 1.03] | 35.8k [31.5, 40.0] | 25.8k [24.5, 27.2] |
| Verified research buys, gen ≥ 2 (setup-passed) | 771k [661, 882] | 313k [294, 331] | 10.6k [8.8, 12.3] | 8.1k [7.6, 8.6] |
| Research rounds started (setup-passed) | 483k [432, 534] | 183k [173, 194] | 6.8k [6.0, 7.6] | 4.9k [4.6, 5.2] |
| Mid-game value points, 2 per gen ≥ 2 (outcome-verified) | 237k [212, 262] | 80.2k [75.6, 84.8] | 3.7k [3.3, 4.1] | 2.8k [2.6, 3.0] |
| Root action rows (setup-passed) | 8.82M [7.90, 9.75] | 3.09M [2.72, 3.46] | 108k [91, 124] | 80.5k [72.5, 88.5] |
| Root rows with exact hand | 6.90M [5.56, 8.24] | 2.07M [1.76, 2.39] | 99.2k [83.9, 114] | 59.7k [46.6, 72.9] |
| Card plays from exact hand | 3.34M [2.69, 3.99] | 1.04M [0.88, 1.20] | 50.3k [43.2, 57.5] | 28.7k [22.7, 34.7] |
| Card actions (exact-hand subset implied) | 2.62M [2.16, 3.08] | 0.88M [0.69, 1.07] | 30.8k [23.0, 38.6] | 23.1k [18.6, 27.5] |
| Greenery conversions | 212k [136, 287] | 110k [80, 141] | 3.0k [1.6, 4.4] | 2.9k [1.6, 4.1] |
| Heat conversions | 249k [179, 319] | 98.8k [74, 123] | 2.0k [1.0, 2.9] | 2.8k [2.0, 3.5] |
| Milestone claims / award fundings | 122k / 116k | 42.8k / 40.6k | 1.7k / 1.5k | 1.2k / 1.1k |
| Standard projects | 619k [531, 706] | 177k [145, 209] | 6.2k [5.1, 7.3] | 5.7k [4.8, 6.6] |
| Pass roots | 494k [463, 525] | 169k [158, 179] | 6.6k [6.1, 7.0] | 4.6k [4.4, 4.8] |
| Quarantine: text-only tile rows | 492k [326, 659] | 205k [152, 257] | 6.3k [3.5, 9.1] | 5.9k [3.6, 8.2] |
| Quarantine: degenerate `other` rows | 4.74M [3.97, 5.51] | 1.87M [1.58, 2.16] | 80.3k [56.5, 104] | 55.8k [48.6, 63.0] |
| Quarantine: cancel rows | 254k [205, 303] | 121k [97, 147] | 4.6k [3.5, 5.7] | 3.7k [2.6, 4.9] |

Corpus-wide sanity: the draft-pick extrapolation (≈3.4M) brackets the measured pipeline count (3,153,390 picks in `yield_report.json`), whose stricter undo handling drops more rows; the research-buy total (≈1.10M) is a genuinely new task; the value-point total (≈323k `ActorObservation`→outcome pairs) exists only on outcome-verified tables. Elysium/Vastitas columns rest on 20/14 sampled tables and carry the widest bounds.

## Proposed gates and remaining research

1. **Timing gate:** accept row *k−1* `game_state` as pre-move only for globals/milestones/awards/TR/stocks/productions; for M€ at generation-start rows apply `M€ + M€ Production + TR − payments-since-deal` and cross-check against the actor's next snapshot (drop rows that fail twice). Quarantine versions whose M€ tracker fails the worked-identity spot check (`v2025.08.15.1706`).
2. **Degenerate-file gate:** quarantine tables whose `player_trackers` are empty past move 1 or whose draft rows carry no structured `card_drafted` (both styles cited in 1.1) — they pass the index gate today.
3. **Research-buy gate:** require kept ⊆ drafted-4 (3 verified picks + final pass packet), payment == 3 × |kept|, and discard-by-difference; quarantine keep rows without payment text unless the version is known to strip it.
4. **Hand-tracking gate:** require exact tracked hand at the row (setup + buys + named draws − plays − named discards; Sell-patents partial and the residual mismatch case quarantine the seat from that row on). The `Hand Counter` is a cross-check only.
5. **Action-root gate:** classify roots from `action_type` **plus** description verbs (mismatch citation in 3.1); verify the chosen branch's state legality (plants/heat ≥ 8, unclaimed/under-funded, cost vs M€); never claim full candidate-set legality without engine replay.
6. **Remaining research:** multi-perspective joins (two files of one table) for opponent private recovery without leakage; a text-coordinate parser for the 34.2% text-only placements; a version-to-scraper-commit mapping to explain the `1706`/`dev` regressions; measuring how much policy value the missing 60 milestone/award score scalars and deck-size scalars actually cost.

The most promising bounded next work is an event-level reconstruction check on a `v2025.10.07.1844` table pair (richest schema, 52% of eligible tables) that emits one full `ActorObservation`+`Choice` per research buy and one mid-game value point per generation, against the adapter contracts.
