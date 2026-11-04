# Existing Terraforming Mars engine: local simulator feasibility

Research date: 2026-10-06. Resolves the investigation in [Investigate existing simulator support for local training](https://github.com/finger-six/tm_bot/issues/2); it does not select an engine or implement an adapter.

## Evidence boundary

Inspected repository: `terraforming-mars/terraforming-mars`, pinned to [`f91e1e2574175c6affa2440a2c66a8fc00412e1e`](https://github.com/terraforming-mars/terraforming-mars/commit/f91e1e2574175c6affa2440a2c66a8fc00412e1e), commit timestamp 2026-10-06T16:54:09Z. Source files were retrieved through the GitHub API with that explicit `ref`; line citations below refer to that revision. Publisher references were read on the research date and are separate from the pinned engine.

Evidence here is **source inspection**, including test source, plus a limited comparison with publisher map rules. The engine was not installed, built, tested, run through a complete game, or benchmarked in this investigation. Test existence is not a passing test result. No games/hour, decisions/second, search budget, or local hardware feasibility is established.

Target: two players, drafting, Prelude as the only expansion, five named maps; local RTX 4080 Super with 16 GB VRAM and 32 GB RAM. These hardware specifications are supplied requirements, not measured resource budgets.

## Findings

The engine is a substantial candidate for reuse: server-side TypeScript implements card pools, inputs, game phases, production, final greenery and scoring; it exposes browser-free JSON routes and callable core methods. However, the requested publisher Amazonis rules are demonstrably different from the implemented `amazonis p.` map. Arbitrary mid-decision cloning, leak-free observations, complete-game replay determinism, and throughput remain unverified. Those gaps prevent treating this revision as a ready-made simulator for the full target format. [Core lifecycle][game] [Player processing][process] [Routes][routes]

### Map IDs and rule versions

Use explicit map IDs; the API's random “official” pool contains only Tharsis, Hellas and Elysium. The enum and board factory register these implementations: [IDs][board-ids] [Factories][board-factory] [Random pool][random-pool]

| Requested name | Core enum / API `board` value | Factory | Source-inspected status |
| --- | --- | --- | --- |
| Tharsis | `THARSIS` / `tharsis` | `TharsisBoard` | Registered; complete rules fidelity not audited. |
| Elysium | `ELYSIUM` / `elysium` | `ElysiumBoard` | Registered; complete rules fidelity not audited. |
| Hellas | `HELLAS` / `hellas` | `HellasBoard` | Registered; complete rules fidelity not audited. |
| Vastitas Borealis | `VASTITAS_BOREALIS` / `vastitas borealis` | `VastitasBorealisBoard` | Original variant: polar placement costs 3 M€; milestone/award sets differ from publisher map. |
| Vastitas Borealis, Nova variant | `VASTITAS_BOREALIS_NOVA` / `vastitas borealis nova` | `VastitasBorealisNovaBoard` | Candidate for publisher Vastitas: polar bonus costs 4 M€, with matching named milestone/award sets. Full map and card-interaction fidelity remain unverified. |
| Amazonis Planitia | `AMAZONIS` / `amazonis p.` | `AmazonisBoard` | Registered alternative map; incompatible with publisher Amazonis rules described below. |

This is a material version distinction. FryxGames' map sheet specifies a four-credit polar temperature bonus on Vastitas. The original engine board charges three; Nova charges four and has the publisher's named milestone/award sets. Matching those features is evidence for choosing Nova for a fuller audit, not proof of complete equivalence. [Publisher map sheet][publisher-maps] [Original board][vastitas] [Nova board][nova] [Costs][constants] [Milestone registry][milestones] [Award registry][awards]

For Amazonis, FryxGames specifies a larger surface with **11 oceans**. The engine's `AmazonisBoard` instead defines a nine-row, 61-hex Mars footprint, including a restricted central hex. Its `Game.canAddOcean()` uses the global nine-ocean constant with no Amazonis override. Its milestone and award sets also differ from the publisher sheet. Therefore the pinned implementation does **not** supply publisher Amazonis Planitia fidelity by selecting `amazonis p.`. The UI also groups these later map choices under “Fan-made”; that label alone cannot settle publisher compatibility. [Publisher ocean requirement][publisher-oceans] [Publisher map sheet][publisher-maps] [Amazonis layout][amazonis] [Ocean termination][ocean-cap] [Constants][constants] [Milestones][milestones] [Awards][awards] [UI grouping and list][map-ui]

Follow-up decision: explicitly define the required map editions. If publisher Amazonis is required, choose between adapting this engine's board/track/milestone support, finding another engine/revision, or changing scope with the user's agreement. Do not silently substitute the registered alternative.

### Two players, Prelude and drafting configuration

`Game.newInstance(id, players, firstPlayer, spectatorId, partialOptions, seed)` takes an array of players; the solo-specific modifications apply only to length one. Passing two non-beginner, zero-handicap players and selecting the first seat explicitly is the likely core setup. This is an available construction path, not a demonstrated two-player rollout. [Construction][construction] [Setup and solo handling][game]

For an ordinary Corporate Era game with original Prelude:

- Set `corporateEra: true`, `preludeExtension: true`, `prelude2Expansion: false`, and `draftVariant: true`.
- Keep all other expansion flags false, including promos and community cards; keep custom/included/banned card lists empty unless intentionally changing the format.
- Set `boardName` explicitly; retain unshuffled layouts and native milestones/awards (`shuffleMapOption: false`, `randomMA: NONE`, `modularMA: false`).
- Keep `initialDraftVariant` and `preludeDraftVariant` false unless the agreed format includes those separate variants. “Drafting” does not automatically specify either.
- For the HTTP configuration, these become `expansions.corpera: true`, `expansions.prelude: true`, all other expansion entries false, `board`, `draftVariant`, `initialDraft`, and `preludeDraftVariant`. Use the actual `NewGameConfig` shape, not core option names.

The code contains both legacy expansion booleans and an `expansions` record. Direct core construction derives the record when absent; callers supplying it should keep the two forms consistent. Card pools currently use the booleans, whereas compatibility checks can use the record. Also, `DEFAULT_GAME_OPTIONS.corporateEra` is true while its default record says `corpera: false`; avoid depending on mixed defaults. Corporate Era inclusion is a proposed conventional interpretation of the training format and should be recorded explicitly. [Defaults and options][options] [Core normalization][construction] [HTTP mapping][create-api] [Card-pool manifests][card-pools] [Compatibility][ma-compatible]

Standard drafting, initial project drafting and Prelude drafting have distinct implementations. Draft rounds can request inputs from both players; a player who already picked may receive an **optional repick** prompt until the other has chosen. A controller must schedule required inputs rather than always stepping only `game.activePlayer`, and must not loop indefinitely on optional repicks. [Draft lifecycle][draft] [Required-input filtering][waiting-api]

### Browser-free control and complete-game execution

Two seams exist:

| Seam | Available source path | Adaptation needed |
| --- | --- | --- |
| Local HTTP server | `POST /api/creategame`, `GET /api/player?id=...`, `GET /api/waitingfor?id=...&gameAge=...&undoCount=...`, `POST /player/input?id=...` | JSON controller, player IDs/access handling, prompt encoding, polling/scheduling; explicit seeded creation is absent from the inspected create route. |
| In-process TypeScript | `globalInitialize()`, `Game.newInstance(..., seed)`, `player.getWaitingFor()`, `Server.getPlayerModel(player)`, `player.process(InputResponse)` | Training wrapper; isolate storage/lifecycle/timers, normalize observations and actions, define safe branch boundaries. |

Paths are registered as server handlers and return JSON independently of Vue rendering, so a browser is not intrinsically required for these calls. The package specifies Node 22.x, a server build, and server test commands. No packaged reset/step/clone training interface was demonstrated by the inspected entry points. [Path strings][paths] [Handlers][routes] [Create route][create-api] [Player route][player-api] [Input route][input-api] [Core processing][process] [Initialization][initialize] [Package scripts][package]

The core contains the full phase progression: research/draft, actions, production, next generation, final greenery, final scoring and `Phase.END`. In multiplayer, `gameIsOver()` follows terraform completion; reaching that predicate is not itself equivalent to reaching the completed end phase. A driver must finish final production and greenery before reading final rewards. End-game work also calls database and game-loader methods asynchronously. Test setup replaces those singletons with fakes, showing a useful isolation example, not a finished storage-free production API. [Termination and final production][termination] [Final greenery and scoring][endgame] [Test isolation][test-setup]

### Generated choices versus legal-input validation

The engine provides a structured prompt tree, not one flat, exhaustively enumerated action list. `toModel()` exposes nested OR options, selectable cards with bounds/enabled flags, space IDs, amount ranges and payment information. Project play can require further placement/target/resource prompts; payment allocations and card subsets still need controller-side encoding or constrained generation. [Input interface][input-interface] [OR input][or-input] [Card input][card-input] [Space input][space-input] [Payment input][payment-input] [Project-card input][project-input]

Representative enforcement exists beyond UI display:

- Card selections enforce min/max, membership and disabled choices.
- Space selection rejects IDs outside the current candidate spaces.
- Amount selection enforces bounds.
- Payment checks stock/reserves and sufficient value; project payment checks do this through `Player.checkPaymentAndPlayCard()`.
- `Player.getPlayableCards()` filters through affordability and card requirements before offering project choices.

These are source-confirmed examples, not an exhaustive proof that every malformed response is rejected or every offered combination succeeds. Some response guards mainly check keys and type discriminants. `Player.process()` reinstates its pending prompt/callback after an exception, but it does not snapshot and undo every mutation made by nested callbacks. Treat invalid-action atomicity as an experiment, not a guarantee. Avoid bypassing `process()` by invoking callbacks directly. [Card checks][card-input] [Space checks][space-input] [Amount checks][amount-input] [Payment checks][payment-input] [Project payment][project-payment] [Playable-card filtering][playable] [Response guards][responses] [Exception handling][process]

### Actor-visible observations and hidden information

`Server.getPlayerModel(player)` is the likely starting point: own hand, own dealt/drafted/Prelude cards and pending prompt; opponents appear through public models with hand counts and tableau rather than their hands. Game data exposes board spaces, parameters and deck sizes rather than full deck order. Opponent VP is conditional; set `showOtherPlayersVP: false` for the proposed observation format. Participant-view logs filter messages reserved for another player. [Actor view and game view][models] [Public player fields][public-model] [Log filtering][logs]

This establishes an existing visibility boundary, not a complete leak audit. Raw `Game`, `SerializedGame`, both player views combined, deck order and RNG state must stay out of an actor's policy input. Retain an actor's history of seen draft hands if the learning/search design needs it; a current prompt is not a complete information history. Add leakage tests that change only unseen opponent hands or deck order and compare permitted observations; document intentionally public changes and known draft information. Distinguish policy observations from any intentionally privileged critic/training state. This is a proposed adapter requirement.

### Seeded randomness, serialization and branching

The core owns a `SeededRandom` with original seed and current state; initialization and deck shuffles use it. Serialization includes `seed` and `currentSeed`; deserialization rebuilds the RNG and decks. Upstream random tests check equal streams and reconstructed RNG continuity, but were **not run here**. [RNG implementation][rng] [Construction][construction] [Deck operations][deck] [Snapshot][snapshot] [Restoration][restore] [RNG test source][rng-tests]

Two limits matter:

1. The HTTP create route samples `Math.random()` for the board pool and seed, and does not read a caller-supplied seed. Direct construction is the inspected seeded seam. Generated game names use unseeded randomness; timestamps and timers also make literal whole-snapshot equality inappropriate without a defined normalization. This investigation did not audit every random call or prove identical complete trajectories. [HTTP randomness][create-api] [Game metadata][game]
2. `Game.serialize()` writes `deferredActions: []`. Player serialization omits `waitingFor` and its callback; inputs themselves contain functions. `Game.deserialize()` regenerates prompts from phase (draft/research/action/final greenery). This is recovery logic, **not a demonstrated lossless clone at arbitrary nested decisions**. Some Prelude transient fields are explicitly excluded in upstream serialization test source. [Snapshot][snapshot] [Player serialization][player-snapshot] [Input callbacks][input-interface] [Phase restoration][restore] [Serialization test source][snapshot-tests]

The existing `Cloner.clone()` rewrites player/game IDs and mutates its supplied serialized object; the create route loads save version zero for its cloning path. That is useful for repeating an initial game, not evidence for cheap search-node branching. Serialized output also retains object references such as options/logs, so independence requires a deep snapshot and verification. Restore/undo operations can touch the shared database. [Cloner][cloner] [Clone route][create-api] [Snapshot fields][snapshot] [Loader restoration][loader]

Likely adaptation seam (inference): wrap the core in-process with a controlled persistence implementation and a stable actor/prompt scheduler. First prove replay from known safe checkpoints; only then define clone semantics. If nested prompt snapshots fail, consider deterministic action replay from a prior checkpoint or an explicit continuation representation. Do not label `JSON.stringify(game.serialize())` plus `Game.deserialize()` a general clone without evidence.

### Rules and license boundaries

Card manifests include base, Corporate Era and Prelude when enabled; playability, stock/payment, board placement, turn progression and victory scoring live on the server. This is substantially more than a card catalogue or interface mock. Nevertheless, card-by-card official rules fidelity, interactions, two-player ties and all five desired map editions were not exhaustively compared with publisher rules. The Amazonis discrepancy is already concrete. Source test coverage cannot establish completeness without executing and supplementing it. [Card pools][card-pools] [Playable cards][playable] [End-game scoring][endgame] [Map comparison above][publisher-maps]

The repository `LICENSE` contains GNU GPL version 3; its README identifies GPLv3. **Conflicting metadata:** `package.json` says ISC. Record the project as GPLv3 based on its license file/README and flag the discrepancy for clarification; do not infer a permissive license from package metadata. The README separately lists font/icon attribution, so code-license identification is not an asset-rights audit. This report does not decide distribution obligations. [License file][license] [README][readme] [Package metadata][package]

## Focused follow-up experiment

Run this only after choosing map editions and addressing required rules gaps. Preserve the pinned revision, adapter changes and exact configuration in the results.

1. **Setup and completion:** verify two seats, enabled card pools, initial deals, first seat, native map layout/bonuses and milestones; execute complete seeded games on each supported target map to `Phase.END`. Track deadlocks, unexpected inputs, invalid actions, final scores and ties; sample official rule scenarios.
2. **Determinism and visibility:** replay identical seeds and encoded decisions, including alternating draft/research submission order. Compare normalized gameplay state, prompts, RNG progress and terminal scores; test the two actor observation boundaries.
3. **Branch correctness:** test independent round trips at initial selection, drafting (before/after one seat picks), research, normal action, payment/placement/attack subprompts, Prelude effects and final greenery. Compare the original and restored next decisions plus subsequent trajectory; mutate one branch and assert the other is unchanged. Test rejection of invalid inputs for state atomicity.
4. **Measure local cost:** after warm-up, report reset, observation, valid step, serialization, restore/clone latency distributions (median/p95), snapshot bytes, RSS/heap and completed-game costs across early/mid/late states and maps. Count prompt decisions separately from top-level game actions. Compare HTTP with in-process control and storage enabled with isolated persistence.
5. **Resource scaling:** record CPU model, Node/runtime, worker counts and seed corpus; measure worker scaling, GC and memory over sustained runs on the actual 32 GB host. Separately measure batched model inference/training and VRAM on the RTX 4080 Super, then the combined workload. The engine's TypeScript source provides no GPU-training throughput estimate.

No throughput acceptance threshold can be derived from this inspection. Decide required game/search/model budgets from the measured workload; do not turn “runs a local web server” into a training capacity claim.

[game]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/Game.ts
[board-ids]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/common/boards/BoardName.ts#L1-L15
[board-factory]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/GameSetup.ts#L37-L61
[random-pool]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/routes/ApiCreateGame.ts#L75-L89
[vastitas]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/boards/VastitasBorealisBoard.ts#L12-L53
[nova]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/boards/VastitasBorealisNovaBoard.ts#L14-L64
[amazonis]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/boards/AmazonisBoard.ts#L7-L42
[constants]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/common/constants.ts
[milestones]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/milestones/Milestones.ts#L153-L159
[awards]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/awards/Awards.ts#L125-L131
[ocean-cap]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/Game.ts#L1563-L1565
[map-ui]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/client/components/create/CreateGameForm.vue#L178-L185
[publisher-maps]: https://fryxgames.se/wp-content/uploads/2024/09/TM_AV_WRAP_ENG.pdf
[publisher-oceans]: https://fryxgames.se/product/tm-amazonis-tiles/
[construction]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/Game.ts#L257-L332
[options]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/game/GameOptions.ts
[create-api]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/routes/ApiCreateGame.ts#L123-L216
[card-pools]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/GameCards.ts#L45-L81
[ma-compatible]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/ma/MAManifest.ts#L24-L31
[draft]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/Draft.ts#L38-L240
[waiting-api]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/routes/ApiWaitingFor.ts#L19-L43
[paths]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/common/app/paths.ts
[routes]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/server/requestProcessor.ts#L89-L130
[player-api]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/routes/ApiPlayer.ts#L16-L29
[input-api]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/routes/PlayerInput.ts#L73-L108
[process]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/Player.ts#L1684-L1723
[initialize]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/globalInitialize.ts
[package]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/package.json
[termination]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/Game.ts#L774-L802
[endgame]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/Game.ts#L1103-L1172
[test-setup]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/tests/testing/setup.ts
[input-interface]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/PlayerInput.ts
[or-input]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/inputs/OrOptions.ts#L13-L36
[card-input]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/inputs/SelectCard.ts#L56-L95
[space-input]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/inputs/SelectSpace.ts#L19-L37
[amount-input]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/inputs/SelectAmount.ts#L21-L47
[payment-input]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/inputs/SelectPayment.ts#L21-L63
[project-input]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/inputs/SelectCardToPlay.ts#L57-L118
[project-payment]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/Player.ts#L757-L791
[playable]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/Player.ts#L1207-L1276
[responses]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/common/inputs/InputResponse.ts
[models]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/models/ServerModel.ts#L66-L129
[public-model]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/models/ServerModel.ts#L225-L305
[logs]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/routes/GameLogs.ts#L30-L49
[rng]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/common/utils/Random.ts#L13-L35
[deck]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/cards/Deck.ts#L17-L83
[snapshot]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/Game.ts#L471-L532
[restore]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/Game.ts#L1697-L1840
[rng-tests]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/tests/common/utils/Random.spec.ts
[player-snapshot]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/Player.ts#L1758-L1857
[snapshot-tests]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/tests/Game.spec.ts#L771-L813
[cloner]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/database/Cloner.ts#L11-L40
[loader]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/database/GameLoader.ts#L234-L277
[license]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/LICENSE
[readme]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/README.md
