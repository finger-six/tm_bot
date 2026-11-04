# TFMStats archive: suitability and replay-gap audit

Research for [Audit TFMStats logs for warm-start training and replay gaps](https://github.com/finger-six/tm_bot/issues/3), 2026-10-06. This report proposes validation checks; it does not select a training-data policy or implement a converter.

## Finding

The inspected archive supports game/setup analysis and supplies useful *candidate* examples for initial corporation/Prelude selection, draft choices, purchases, card plays, and placement choices. It is not yet demonstrated to supply complete, leakage-free observation/action/legal-alternative triples or trustworthy outcome labels without reconstruction and validation. Matching setup flags is a candidate eligibility test, not a replay-quality guarantee.

Two-player, drafting, Prelude-on/Colonies-off examples were verified for Tharsis, Hellas, Elysium, and Vastitas Borealis. The inspected Elysium example ends by concession. Amazonis Planitia was found, but its inspected example has Colonies enabled and drafting disabled. Availability of eligible Amazonis games remains **unknown**. Availability of fully validated trajectories on every map remains **unknown**.

## Bounded method and limits

Source: the already extracted local directory `game-logs-archive-2026-01-10-023900/`; the adjacent ZIP was not extracted or loaded. No corpus-wide file listing, conversion, data mutation, or total-game count was needed. Reads were sequential, with one JSON object in memory at a time.

1. Directory-spread probe: sort immediate archive directory names; select 24 evenly spaced indices; collect at most 300 JSON entries per selected directory; sort those names and select up to six evenly spaced file indices. Skip files above 10,000,000 bytes. Result: 140 parsed game objects, four oversized skips. Maps: Tharsis 106, Hellas 30, Elysium 1, Vastitas Borealis 3. Of these, 60 Tharsis and 27 Hellas records matched two players, Prelude on, drafting on, Colonies off; that is not a quality assessment.
2. Targeted version/perspective probe: fixed directories `95138349`, `84909925`, `84979765`; at most 600 JSON names each, sorted; up to 30 evenly spaced reads each, capped at 20,000,000 bytes. Result: 90 parsed game objects: Tharsis 69, Hellas 15, Elysium 4, Vastitas Borealis 1, Random 1.
3. Rare-map metadata probe: 40 evenly spaced immediate directories; at most the first 1,000 directory entries each; sort collected JSON filenames and inspect their first/last 25 names; read only the first 4,096 text characters per file. Result: 1,868 headers with a map field: Tharsis 1,179; Hellas 592; Elysium 66; Vastitas Borealis 12; Amazonis Planitia 1; Random 15; three localized map strings. Read errors: zero. These are prefix observations, not full-JSON validity or player-count checks. Candidate files were then read in full.
4. Purposeful detailed inspection of the files in the evidence table below, including the supplied duplicate pair and rich Hellas example.

The probes overlap; do not add their counts to estimate unique games. Filesystem enumeration order before each cap, directory boundaries, game-ID ordering, selected player histories, size exclusions, and any concurrent extraction all affect this non-random sample. No immutable corpus inventory or aggregate yield estimate was established. No sampled absence proves corpus absence. Exact map-name filters also miss aliases/localizations until normalized. Missing metadata is unknown, not a default expansion setting.

## Reproducible local evidence

Paths below are relative to the archive root. Move references mean the row whose `move_number` equals the stated value, **not** that array index. Other references use JSON field names.

| ID | File | Map / setup | Version / rows | Purpose |
|---|---|---|---|---|
| S1 | `84503166/game_491566079_84503166.json` | Tharsis; 2 players; draft; no Prelude; Colonies off | `v2025.08.27.1615`; 512 | Ineligible no-Prelude control |
| S2 | `84909925/game_486805304_84909925.json` | Tharsis; 2 players; draft; no Prelude; Colonies off | `v2025.10.07.1844`; 659 | Richer duplicate; payments and score inconsistency |
| S3 | `84979765/game_486805304_84979765.json` | Same table as S2 | `v2025.08.06.1719`; 471 | Older perspective; metadata claims 659 |
| S4 | `95138349/game_767184202_95138349.json` | Hellas; 2 players; Prelude; draft; Colonies off; no concession | `v2025.10.07.1844`; 513 | Detailed choice/private-card/placement traces |
| S5 | `84909925/game_510673396_84909925.json` | Tharsis; 2 players; Prelude; draft; Colonies off; no concession | `v2025.10.07.1844`; 642 | Bundled actions and missing tile fields |
| S6 | `20418489/game_632269608_20418489.json` | Tharsis; 2 players; Prelude; draft; Colonies off; concession | `v2025.08.15.2331`; 817 | Outcome and invalid-date control |
| S7 | `92470110/game_750714172_92470110.json` | Vastitas Borealis; 2 players; Prelude; draft; Colonies off; no concession | `v2025.10.07.1844`; 544 | Eligible setup on fourth map |
| S8 | `83875287/game_653414707_83875287.json` | Elysium; 2 players; Prelude; draft; Colonies off; concession | `v2025.12.27.1015`; 284 | Eligible setup; partial trajectory |
| S9 | `96336906/game_718503898_96336906.json` | Amazonis Planitia; 2 players; Prelude; Colonies on; draft off; Corporate Era off | `v2025.08.27.1615`; 689 | Map present, requested setup unproven |
| S10 | `83875287/game_617080553_83875287.json` | Vastitas Borealis; Prelude; draft; Colonies off; **4 players** | `v2025.12.27.1015`; 746 | Header flag match alone is insufficient |
| S11 | `83875287/game_628532195_83875287.json` | Elysium; Prelude; draft; Colonies off; **3 players** | `v2025.12.27.1015`; 634 | Player-count exclusion control |

These setup claims come from the root `map`, `players`, expansion flags, and `conceded`; version/row claims come from `metadata` and the actual `moves` length. S8's last row says `sidvictema concedes the game`; S9 move 1 explicitly includes `Module: Colonies`, `Basic mode`, and `Map: Amazonis`, independently supporting its mismatched setup.

## Schemas and version/perspective differences

The richer files have root IDs (`table_id`, `replay_id`, `player_perspective`), dates/duration, winner/generations, setup flags, `players`, `moves`, and parsing metadata. Player summaries include corporation, final VP/TR, VP breakdown, cards played, milestones/awards, starting offers, and ELO data. Move rows include actor ID/name, move number, timestamp, `action_type`, description, singular card/tile fields, `card_options`, `cards_kept`, `reason`, and `game_state`. Null and missing fields both occur. [S1–S9 root/player/move objects.]

S3 lacks root `conceded` and the richer card-choice fields in its move schema. Its 471 retained rows still end at move number 659, and `metadata.total_moves` is 659. S2 has 659 rows and the same table/player set. Thus `total_moves` and final move number are not interchangeable with row count or proof of completeness. Do not blindly zip perspectives by array index or label the 188-row difference as 188 missing *decisions*: it may include private/synthetic rows and parser filtering. The cause requires raw replay comparison. [S2/S3 `metadata`, `moves`, first/last rows.]

The first probe observed nine non-null version strings plus 21 records without a version; this is sample diversity, not a full version inventory. Versions ranged from `v2025.07.29.1618` to `v2025.12.27.1015`. Version-specific field/completeness tests are needed rather than assuming all records share a schema.

## Actor-time knowledge and choices

S4 `players.*.starting_hand` records both players' two corporation offers, four Prelude offers, and ten project offers. Move 1 exposes both project-card lists through `card_options`; moves 2–3 expose retained lists for both players. These are useful retrospective choice labels, but a record associated with one `player_perspective` clearly contains the opponent's private offers. Neither the perspective ID nor the presence/absence of the literal word `private` establishes actor-time visibility. S3 move 1 explicitly prefixes the perspective player's card draw with `private:`; S2 rewrites both players' initial draws as named card lists. [S2/S3 move 1; S4 move 1–3, `players.*.starting_hand`.]

A safe actor observation needs a provenance rule for every field: public event known by that time, that actor's private offer/hand, or retrospective disclosure. An opponent's later card play can establish public knowledge *from that play onward*; it cannot reveal their prior hand in the observation. Retrospective opponent offers, game-level `cards_played`, final outcome, and final summaries must not be copied into earlier policy features. For drafts, distinguish offers seen by the actor from packets acquired through later passes. This audit did not prove the timing/visibility of every reconstructed private field.

`card_options` is not uniformly the pre-decision candidate set. S4 move 23 chooses `Mangrove`, but the same row's actor options are `Symbiotic Fungus`, `Shuttles`, `Mine`: the chosen card is absent. This is consistent with a packet/state update after selection, but exact semantics need preceding notification alignment. S4 move 22 and 24 have named chosen cards with null options. S5 move 18 has two four-card packets but no singular chosen card. S4 has 44 rows with options, 81 with `card_drafted`, and 50 with kept-card maps; field counts are neither mutually exclusive nor decision counts. [S4 moves 22–24; S5 move 18; all S4 rows.]

Even `card_drafted` is not a clean draft label: S4 move 2 is `action_type=other`, keeps eight items including corporation/Preludes/projects, and sets `card_drafted=Vitor`. `You draft ...` appears on rows assigned to both actors (S4 moves 22–24), so pronouns are insufficient actor attribution.

## Replay reconstruction gaps

| Component | Evidence and practical implication |
|---|---|
| Hands | Starting offers and kept lists provide a reconstruction seed; snapshots in **all rows** of S4, S5, S7, S8, S9 have no hand-identity field. `Hand Counter` is only a count. Draw/buy/play/discard/steal and effect-specific choices must be aligned and checked before claiming a continuous actor hand. |
| Board | Coordinates occur in descriptions/`tile_location`, e.g. S4 move 55 Ocean at `Argyre Planitia (2,7)` and S7 move 49 Ocean at `(7,5)`. Snapshots lack a complete board. S4 move 54 places Nuclear Zone at South Pole `(5,9)` with both structured tile fields null; S5 move 8 likewise names Mohole Area `(8,2)` only in text. Board reconstruction needs map geometry, tile ownership/type, off-board placements, land claims, bonuses, and all text-only events. |
| State timing | S4 move 54 has oceans 0; move 55 records placement and oceans 1. Its snapshot is already after the event. Treating the same row's state as a pre-action observation leaks action results. Previous retained row may still be insufficient if rows are missing or aggregate several events. |
| Phase/turn | Generation, passes, starting-player descriptions, and effects offer clues; no explicit phase, active-decision specification, pending choice stack, or remaining-actions field exists in these snapshot schemas. Clock-only timestamps can decrease (S4 move 14 follows a later clock value); do not use them to reorder rows. Preserve source ordering and check event boundaries. |
| Payments | Some text gives exact components: S2 move 12 pays 2 Steel for Mining Area; S4 move 13 pays 9 M€ for Advanced Alloys. No structured payment-choice vector exists in the examined move schema. Inventory deltas also include gains/effects and cannot always identify the player's chosen split. |
| Multi-action rows | S5 move 4 describes Biosphere Support **and** Business Empire, but `card_played` names only Biosphere Support. A JSON row is not reliably one training decision. Split validated action boundaries before pairing observations and labels. |
| Resources/card state | `player_trackers` supplies resources, production, hand counts and tag counts; globals/milestones/awards/VP appear in snapshots. This is useful for cross-checks, not a complete engine state: per-card stored resources, used actions, all triggers and pending effects are not established by these fields. |
| Legal alternatives | No complete general legal-action list or mask appears in the examined schemas. Card offers cover some localized choices; they do not establish playable cards, all valid payments, all placement cells, targets, or effect branches. Recompute with a matching rules engine after reconstruction, then verify the observed action is legal. |

[Evidence: all snapshot-key unions in S4/S5/S7/S8/S9; indicated move rows in S2/S4/S5/S7. These claims concern inspected schemas, not every archive version.]

## Scores, concessions, and duplicates

S2 and S3 both report `final_vp=20`, `final_tr=20` for both players with default-looking VP breakdowns, while their final move 659 explicitly gives Lumin_S **102 TOTAL VP** and LeHunter **120 TOTAL VP**, with LeHunter winning. The replay globals nevertheless end at temperature 8, oxygen 14, oceans 9. Therefore a completed replay and non-concession flag do not validate root final scores. [S2/S3 `players`, last move, last `game_state`.]

In S4, root totals 104/90 agree with the final scoring text. S7 likewise agrees at 124/78. Agreement in those examples is positive evidence for those games only; verify breakdown sums, terminal scoring, winner and tiebreaker rather than accepting every score field. [S4/S7 `players.*.final_vp`, final row.]

S6 ends with `-Biscotto concedes the game` and declares rexkow winner, despite both player summaries showing 137 VP; its date is `0001-01-01`. S8 also ends by concession. A concession outcome can be an explicit win/loss label, but partial/current score is not a completed-game score target. Retaining pre-concession behavior examples is a separate policy choice, with the same reconstruction gates. Missing `conceded` (S3) is unknown until terminal text is checked.

Group by platform/table identity (`table_id`/`replay_id` checked against filename) and verify participant/setup agreement. Preserve perspective and parser version as provenance; never treat each perspective file as an independent game. Keep an entire duplicate group in one train/validation/test split. Prefer the most validated reconstruction, not simply newest version or longest row count; S2 is richer yet still has invalid root scores. Complementary perspectives may help recover private choices but must never expand actor observations with opponent-only information.

## First-party parser trace and scope

The [HStrand scraper README](https://github.com/HStrand/bga-tm-scraper/blob/8621fa764422161de0709368292c0c2bf2f2f196/README.md) describes scraping BGA table/replay pages and producing structured game data. The primary source was inspected at revision `8621fa764422161de0709368292c0c2bf2f2f196`:

- [`parser.py`, lines 45–97](https://github.com/HStrand/bga-tm-scraper/blob/8621fa764422161de0709368292c0c2bf2f2f196/bga_tm_scraper/parser.py#L45): the move state is documented as after the move. The current schema additionally declares hands, special tiles, starting player, and draw/discard counts, which are absent from the audited older snapshot schemas.
- [Lines 3988–4005](https://github.com/HStrand/bga-tm-scraper/blob/8621fa764422161de0709368292c0c2bf2f2f196/bga_tm_scraper/parser.py#L3988): rows can be synthesized from gamelogs when HTML lacks a move block; timestamps can be synthesized as UTC clock times.
- [Lines 4155–4244](https://github.com/HStrand/bga-tm-scraper/blob/8621fa764422161de0709368292c0c2bf2f2f196/bga_tm_scraper/parser.py#L4155): `newPrivateState` player operations contribute draft/setup targets, unified with draw IDs into `card_options`. This supports treating options as heterogeneous replay-derived data rather than an automatic general legal-action set.
- [Lines 3023 onward](https://github.com/HStrand/bga-tm-scraper/blob/8621fa764422161de0709368292c0c2bf2f2f196/bga_tm_scraper/parser.py#L3023): starting offers are extracted for both players from private-state notifications.
- [Lines 4587–4633](https://github.com/HStrand/bga-tm-scraper/blob/8621fa764422161de0709368292c0c2bf2f2f196/bga_tm_scraper/parser.py#L4587): current code extracts hands, supplements kept cards, removes played/discarded cards, then constructs snapshots. These improvements cannot be assumed present in the supplied JSON.

This is a relevant public scraper source, not a verified exact source revision for every archived version string. No archive-to-commit mapping, raw replay capture, or original BGA notification visibility contract was obtained. Current parser behavior is explanatory evidence and a follow-up lead, not proof that an older record was produced identically.

## Proposed gates and remaining research

1. **Setup gate:** require two actual participants, a normalized one-of-five map, explicit drafting/Prelude flags, and explicit exclusions for supported other expansions. Cross-check setup text/module/card identities; current flags only explicitly cover Prelude, Colonies, Corporate Era and beginner corporations, so Prelude-only must not be inferred from absent expansion fields. Quarantine unknown/contradictory setups. Do not exclude legitimate base cards merely because their names mention colonies.
2. **Identity/schema gate:** persist source path/content hash, game/perspective IDs, parser version and actual row count; detect duplicate groups, missing IDs/versions, inconsistent settings, malformed dates, row-number gaps/repeats, unknown cards/maps, takebacks, and unreconciled aggregate actions. A row gap is a diagnostic, not automatic proof of a missing decision.
3. **Visibility/decision gate:** produce pre-decision actor observations only after private/public provenance and event timing are validated. Recover initial offers separately from chosen items; align draft packet, choice, pass and purchase. Require the chosen item to be in the verified pre-choice candidate set.
4. **Replay gate:** replay complete action/effect sequences against the matching map/rules/card definitions, compare globals/resources/hand counts/placements, verify actor/order/pending effects, and recompute legal alternatives. Ambiguous actor, tile, payment or effect choices remain quarantined rather than guessed.
5. **Outcome gate:** independently check terminal scores, breakdown sums, winner/tiebreaker, and concession status. Record complete versus partial outcomes explicitly; do not train completed-score regression on concession/current/default scores.
6. **Split gate:** keep all perspectives/versions of the same game in one split; account for repeated player histories if evaluating generalization. Corpus weights or concession-retention choices are policy decisions beyond this audit.

The most promising bounded next work is an event-level reconstruction check on S4/S5/S7, a raw replay comparison for S2/S3, and a version-to-source mapping. An eligible Amazonis sample and a completed eligible Elysium sample are still useful follow-ups. Exhaustive metadata coverage, valid-example yield, map balance, complete actor hands/boards/phases, exact payment alternatives, raw-notification visibility, and training performance remain unproven. No full corpus scan is required to establish this initial suitability/gap finding.