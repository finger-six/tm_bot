# Terraforming Mars TypeScript Engine: Simulator Measurements

Measurement report for [Issue #12](https://github.com/finger-six/tm_bot/issues/12), gating engine selection in [Issue #5](https://github.com/finger-six/tm_bot/issues/5).

## 1. Pinned Revision, Host, and Configuration

- **Engine Submodule:** [`tools/simbench/engine`](file:///C:/2026/tm_bot/tools/simbench/engine) (`terraforming-mars/terraforming-mars`)
  - **Pinned Commit:** `f91e1e2574175c6affa2440a2c66a8fc00412e1e` (unmodified; zero engine patches)
- **Harness:** [`tools/simbench/`](file:///C:/2026/tm_bot/tools/simbench)
- **Host Environment:**
  - **OS:** Windows 11 (`win32` `10.0.26300`, `x64`)
  - **CPU:** `Intel(R) Core(TM) Ultra 7 265F` (`20` logical cores)
  - **RAM:** `31.52 GB` (`33,847,894,016` bytes)
  - **Runtime:** Node `v24.18.0` (engine declares `engines.node = "22.x"`, emits `EBADENGINE` warning on install only)
- **Explicit Game Configuration:** Defined in [`config.ts`](file:///C:/2026/tm_bot/tools/simbench/src/config.ts) (`createExplicitTwoPlayerGame()` / `buildExplicitGameOptions()`) with zero reliance on `DEFAULT_GAME_OPTIONS` or random map/seed defaults:
  - **Seats:** 2 players (`Seat 1` / `red` / `p-1`, `Seat 2` / `blue` / `p-2`), `beginner = false`, `handicap = 0`, explicit `firstPlayer = players[0]`, explicit integer `seed`.
  - **Maps (`TARGET_MAPS`):** `tharsis`, `elysium`, `hellas`, `vastitas borealis nova`.
  - **Expansions:** Corporate Era (`corporateEra: true`, `expansions.corpera: true`) + Prelude (`preludeExtension: true`, `expansions.prelude: true`) only; all other expansions, promos, and community cards `false`.
  - **Variants:** `draftVariant: true`, `initialDraftVariant: false`, `preludeDraftVariant: false`, `ceosDraftVariant: false`, `undoOption: false`, `showTimers: false`, `fastModeOption: false`, `showOtherPlayersVP: false`, `shuffleMapOption: false`, `randomMA: RandomMAOptionType.NONE`, `includeFanMA: false`, `modularMA: false`.
- **In-Process Isolation:** [`initializeEngine()`](file:///C:/2026/tm_bot/tools/simbench/src/engine.ts) installs pure in-memory no-op `Database` and `GameLoader` singletons before calling `globalInitialize()`, avoiding SQLite/Postgres/LocalFS I/O and `Cache` background maintenance timers.
- **Canonical Simultaneous Order:** [`pickNextWaitingPlayer()`](file:///C:/2026/tm_bot/tools/simbench/src/driver.ts) uses `'canonical'` order (lowest seat index first whenever >1 required player is waiting in `RESEARCH` / `SelectInitialCards` / `DRAFTING`). All Stage 1–5 runs below use `'canonical'` order unless explicitly testing order sensitivity in Stage 2.

---

## 2. Verdict Summary vs. Issue #12 Criteria

| Stage | Scope | Criterion | Verdict | Summary |
| --- | --- | --- | --- | --- |
| **Stage 1: Smoke** | 20 seeded games/map (80 games) | Complete to `Phase.END`, 0 deadlocks, 0 unexpected inputs, 0 invalid actions | **PASS** | `80/80` completed (`100%`), `0` deadlocks, `0` invalid actions, `0` step caps, `0` unexpected input types; `18.08–26.64 ms` mean/game. |
| **Stage 2: Determinism** | Strict replay + alternated simultaneous draft/research order (4 maps) | Identical seed + identical action sequence produces identical normalized state & final scores | **PASS (with canonical order)** | Strict replay: `4/4` PASS (`0` mismatches). Alternated Draft-only: `4/4` PASS (`0` mismatches). Alternated Draft+Research: `4/4` FAIL (`discardPile` diverges at step 2; resolved by enforcing canonical seat submission order in the adapter). |
| **Stage 3: Cloning** | 12 decision-point categories × 4 maps (48 points) + atomicity + suffix sweep | Prompt equivalence, continuation equivalence, 20-step branch isolation, atomic invalid-input rejection | **FAIL (Direct) / PASS (Fallback)** | Direct `Game.deserialize(deepCopy(game.serialize()))`: **PASS** `6/12` root categories (`24/24`), **FAIL** `6/12` categories (`research_buy`, `prelude_effect`, `nested_payment`, `nested_space`, `nested_player_attack`, `nested_deferred_action`) due to `deferredActions: []` and unserialized `waitingFor`/cards. **Checkpoint + Replay fallback** passes `24/24` failed points (`12/12` categories overall) at `~0.83–1.76 ms` (`suffix=1`) and `2.125 ms` median at `suffix=40`. Atomicity: `48/48` PASS for tested invalid responses (see Gaps). |
| **Stage 4: Latency** | 80 in-process games (`31,790` decisions) + 4 HTTP games (`1,456` decisions) | **Bar: 1 complete 2P game ≤ 30 s, single worker, bare engine** | **PASS** | Bare engine game wall (no obs): **`17.10 ms` median**, `25.43 ms` p95, **`28.66 ms` max** (**~1,047× faster** than the 30 s bar). With `Server.getPlayerModel`: `44.26 ms` median. Localhost HTTP server (`GET /api/waitingfor` + `POST /player/input`): `31.52 ms` median/decision (`14.82 s` median/game) — **~380.7× slower** per-decision than in-process. |
| **Stage 5: Scaling** | `worker_threads` for 60 s/tier across `1, 2, 4, 8, 20` workers | Informational (no threshold) | **Measured (no threshold)** | `62.83 games/s` (`1w`, `354.86 MB` RSS) → `545.94 games/s` / `224,813 dec/s` (`20w`, `4,680.41 MB` max RSS, `8.69×` speedup), `0` failed games out of `77,487` total across all tiers. |

---

## 3. Stage 1: Smoke (`npm run smoke`)

Executed in canonical simultaneous order across `seeds = 1001..1020` (`20` seeds per map = `80` games total).

### Single-Seed Detailed Run (`seed = 1001`)

| Map | Completed | Generations | Decisions | Invalid | Deadlocks | Unexpected Inputs | Scores (`red` / `blue`) | Tie | Wall Time (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `tharsis` | yes | 21 | 436 | 0 | 0 | `[]` | `94` / `46` | false | 69.40 |
| `elysium` | yes | 20 | 431 | 0 | 0 | `[]` | `98` / `52` | false | 23.86 |
| `hellas` | yes | 23 | 435 | 0 | 0 | `[]` | `87` / `72` | false | 23.36 |
| `vastitas borealis nova` | yes | 19 | 389 | 0 | 0 | `[]` | `104` / `45` | false | 21.90 |

### 20-Seed Corpus per Map

| Map | Completed | Mean Wall (ms) | p95 Wall (ms) | Min / Max Wall (ms) | Mean Gen | Mean Decisions | Invalid / Deadlocks / StepCaps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `tharsis` | `20/20` (`100%`) | 26.64 | 43.59 | 14.77 / 69.40 | 22.75 | 423.1 | `0 / 0 / 0` |
| `elysium` | `20/20` (`100%`) | 19.09 | 24.06 | 12.39 / 25.84 | 23.85 | 435.2 | `0 / 0 / 0` |
| `hellas` | `20/20` (`100%`) | 18.27 | 23.36 | 13.38 / 27.81 | 22.25 | 404.6 | `0 / 0 / 0` |
| `vastitas borealis nova` | `20/20` (`100%`) | 18.08 | 21.90 | 14.54 / 27.30 | 22.15 | 410.8 | `0 / 0 / 0` |

*Note: `tharsis` seed `1001` (`69.40 ms`) includes initial V8 cold-JIT overhead on the first game of the process.*

---

## 4. Stage 2: Determinism (`npm run determinism`)

Baseline games recorded in canonical order (`seeds = 2001..2004`) and normalized via [`normalizeSerializedGame()`](file:///C:/2026/tm_bot/tools/simbench/src/normalize.ts) (stripping `createdTimeMs`, `lastSaveId`, `players[*].timer`, `gameLog[*].timestamp`, and normalizing caller-supplied `id`/`spectatorId`/unseeded `name`).

| Map (Seed) | Actions | Strict Replay (Distinct IDs) | Alternated Draft-Only Order | Alternated Draft+Research Order | First Non-Log Divergence (Draft+Research) |
| --- | --- | --- | --- | --- | --- |
| `tharsis` (`2001`) | 392 | **PASS** (`392/392`, `0` diffs, `65/72`) | **PASS** (`63` swapped, `0` diffs) | **FAIL** (`85` swapped, `307/307` diffs) | `step=2: $.corporationDeck.discardPile[0]: Robinson Industries !== PhoboLog` |
| `elysium` (`2002`) | 469 | **PASS** (`469/469`, `0` diffs, `94/66`) | **PASS** (`78` swapped, `0` diffs) | **FAIL** (`93` swapped, aborted at `step=419`) | `step=2: $.corporationDeck.discardPile[0]: Robinson Industries !== Tharsis Republic`; reshuffle divergence aborts replay at gen 25 (`Card Interstellar Colony Ship not found`) |
| `hellas` (`2003`) | 426 | **PASS** (`426/426`, `0` diffs, `92/57`) | **PASS** (`66` swapped, `0` diffs) | **FAIL** (`89` swapped, `337/337` diffs) | `step=2: $.corporationDeck.discardPile[0]: Saturn Systems !== Tharsis Republic` |
| `vastitas borealis nova` (`2004`) | 386 | **PASS** (`386/386`, `0` diffs, `71/79`) | **PASS** (`54` swapped, `0` diffs) | **FAIL** (`73` swapped, `313/313` diffs) | `step=2: $.corporationDeck.discardPile[0]: Saturn Systems !== Tharsis Republic` |

**Root Cause & Resolution:**
- In `SelectInitialCards.completed()` (`SelectInitialCards.ts:100-124`) and `ChooseCards` -> `keep()` (`ChooseCards.ts:74`), unselected corporations, preludes, and project cards are pushed onto `discardPile` in whichever order players submit. When `projectDeck` exhausts and reshuffles its `discardPile` (as in `elysium` seed `2002` gen 25), the two runs diverge onto different cards.
- By contrast, `DRAFTING` keeps cards in per-player hands (`draftHand`/`draftedCards`) without touching shared decks or logs until both players finish the round, so alternating `DRAFTING` submission order alone is `100%` invariant (`0` diffs across `261` swapped draft pairs).
- **Adapter rule enforced:** Submit simultaneous waiting players in deterministic canonical seat order (`seat 0` first).

---

## 5. Stage 3: Cloning (`npm run cloning`)

Tested in [`cloning.ts`](file:///C:/2026/tm_bot/tools/simbench/src/cloning.ts) across all `12` requested decision-point categories on all `4` target maps (`48/48` points located and tested; `0` missing categories).

- **Direct clone:** `Game.deserialize(JSON.parse(JSON.stringify(game.serialize())))` (`Game.ts:471`, `Game.ts:1697`).
- **Checks per point:**
  - **(a) Prompt Equivalence:** Exact diff of `Server.getWaitingFor(player, wf)` across all players.
  - **(b) Continuation Equivalence:** Continue original and clone with identical PRNG state & policy to `Phase.END`; compare normalized final state and final scores.
  - **(c) Isolation:** Advance clone 20 decisions; assert original's normalized serialized JSON and pending prompt are byte-identical.
  - **(d) Atomicity:** Submit a deliberately invalid `InputResponse` (`player.process(...)`); assert error thrown, normalized serialized state unchanged, and pending prompt unchanged.
- **Fallback clone (`checkpoint + replay`):** Keep serialized JSON at the last safe root restore boundary (`initial_selection`, `drafting`, `prelude_root`, `action_root`, `final_greenery` with `deferredActions.length === 0`) plus recorded `InputResponse` suffix since that checkpoint; clone by deserializing the checkpoint and replaying the suffix.

### Direct Clone & Fallback Results by Decision Point Category (4 Maps Each)

| Category | Direct Prompt (a) | Direct Cont. (b) | Direct Isolation (c) | Atomicity (d) | Direct Overall | Fallback (a, b, c) | Fallback Time (`suffix=1`, ms) | Representative Direct Failure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `initial_selection` | `4/4` | `4/4` | `4/4` | `4/4` | **PASS** (`4/4`) | — | — | — |
| `drafting_before_pick` | `4/4` | `4/4` | `4/4` | `4/4` | **PASS** (`4/4`) | — | — | — |
| `drafting_after_one_pick` | `4/4` | `4/4` | `4/4` | `4/4` | **PASS** (`4/4`) | — | — | — |
| `research_buy` (gen ≥ 2) | `0/4` | `0/4` | `4/4` | `4/4` | **FAIL** (`0/4`) | **PASS** (`4/4`) | `0.828 – 1.028` | `buttonLabel: Buy !== Ok`; `Too many cards selected` (`runResearchPhase` clears `draftedCards` into unserialized `ChooseCards`) |
| `prelude_root` | `4/4` | `4/4` | `4/4` | `4/4` | **PASS** (`4/4`) | — | — | — |
| `prelude_effect` | `0/4` | `0/4` | `4/4` | `4/4` | **FAIL** (`0/4`) | **PASS** (`4/4`) | `0.850 – 0.937` | Restores to outer `action`/`prelude_root` instead of nested Prelude prompt; `Not a valid OrOptionsResponse` |
| `action_root` | `4/4` | `4/4` | `4/4` | `4/4` | **PASS** (`4/4`) | — | — | — |
| `nested_payment` | `0/4` | `0/4` | `4/4` | `4/4` | **FAIL** (`0/4`) | **PASS** (`4/4`) | `1.309 – 1.582` | `model.amount` missing after restore; drops `SelectPaymentDeferred` |
| `nested_space` | `0/4` | `0/4` | `4/4` | `4/4` | **FAIL** (`0/4`) | **PASS** (`4/4`) | `1.032 – 1.229` | `buttonLabel: Save !== Take action`; drops `PlaceTile` deferred action |
| `nested_player_attack` | `0/4` | `0/4` | `4/4` | `4/4` | **FAIL** (`0/4`) | **PASS** (`4/4`) | `1.349 – 1.757` | `buttonLabel: Decrease !== Take action`; drops `DecreaseAnyProduction` / `SelectPlayer` |
| `nested_deferred_action` | `0/4` | `0/4` | `4/4` | `4/4` | **FAIL** (`0/4`) | **PASS** (`4/4`) | `1.073 – 1.379` | `buttonLabel: Save !== Take action`; `deferredActions: []` wipes queued prompts |
| `final_greenery` | `4/4` | `4/4` | `4/4` | `4/4` | **PASS** (`4/4`) | — | — | — |

- **Why `research_buy` fails direct restore even though `Game.deserialize()` calls `gotoResearchPhase()`:** [`Player.runResearchPhase()`](file:///C:/2026/tm_bot/tools/simbench/engine/src/server/Player.ts#L658-L673) calls `copyAndClear(this.draftedCards)` and holds the 4 drawn/drafted cards only inside the unserialized [`ChooseCards`](file:///C:/2026/tm_bot/tools/simbench/engine/src/server/deferredActions/ChooseCards.ts#L27-L76) closure. When `Game.deserialize()` re-invokes `gotoResearchPhase()`, `this.draftedCards` is already empty (`[]`), presenting an `Ok` prompt with `max = 0` cards.

### Fallback Clone Cost vs. Suffix Length (`tharsis`, mid-game action root, `n = 25` samples/length)

| Suffix Length | Median (ms) | p95 (ms) | Max (ms) |
| --- | --- | --- | --- |
| `0` (deserialize only) | 0.845 | 0.994 | 1.016 |
| `1` | 0.905 | 1.222 | 1.433 |
| `2` | 1.298 | 3.013 | 10.189 |
| `5` | 1.226 | 1.717 | 5.789 |
| `10` | 1.200 | 1.395 | 1.461 |
| `20` | 1.354 | 1.863 | 3.550 |
| `40` | 2.125 | 2.356 | 3.748 |

*Note: Within a single action or turn, real nested suffixes before returning to a safe root boundary are typically `1–4` decisions (`~0.9–1.7 ms`), whereas checkpointing only at generation boundaries requires up to `~20–40` decisions (`~1.4–2.1 ms` median).*

---

## 6. Stage 4: Latency (`npm run latency`)

Measured after `4` warm-up games (`1` per map) over `20` games/map = `80` completed games (`31,790` decisions, `0` deserialize errors, canonical order). Generation tertiles split each game's decisions by `generation` into `[1, ceil(G/3)]` (early, `n = 9,820`), `(ceil(G/3), ceil(2G/3)]` (mid, `n = 10,769`), and `(ceil(2G/3), G]` (late, `n = 11,201`).

### In-Process Operation Distributions

| Operation | Scope / Samples | Median | p95 | Max | Mean | Min |
| --- | --- | --- | --- | --- | --- | --- |
| `reset` (`Game.newInstance`) | Overall (`n = 80`) | `0.5974 ms` | `1.0123 ms` | `2.2516 ms` | `0.6641 ms` | `0.5144 ms` |
| `observation` (`Server.getPlayerModel`) | Overall (`n = 31,790`) | `0.0545 ms` | `0.1159 ms` | `1.9785 ms` | `0.0627 ms` | `0.0228 ms` |
| — Early / Mid / Late | `9,820` / `10,769` / `11,201` | `0.0413` / `0.0537` / `0.0706 ms` | `0.0786` / `0.1047` / `0.1388 ms` | `1.1484` / `1.9785` / `1.3152 ms` | — | — |
| `step` (`player.process`) | Overall (`n = 31,790`) | `0.0283 ms` | `0.1161 ms` | `1.6130 ms` | `0.0393 ms` | `0.0005 ms` |
| — Early / Mid / Late | `9,820` / `10,769` / `11,201` | `0.0227` / `0.0277` / `0.0355 ms` | `0.0939` / `0.1149` / `0.1310 ms` | `1.3133` / `1.6130` / `1.1578 ms` | — | — |
| `serialize()` + `JSON.stringify` | Overall (`n = 31,790`) | `0.1747 ms` | `0.3576 ms` | `1.8467 ms` | `0.2012 ms` | `0.0641 ms` |
| — Early / Mid / Late | `9,820` / `10,769` / `11,201` | `0.1138` / `0.1694` / `0.2779 ms` | `0.1778` / `0.2776` / `0.4057 ms` | `1.3324` / `1.4706` / `1.8467 ms` | — | — |
| `JSON.parse` + `Game.deserialize` | Overall (`n = 31,790`) | `0.7295 ms` | `1.1490 ms` | `12.8609 ms` | `0.7708 ms` | `0.4240 ms` |
| — Early / Mid / Late | `9,820` / `10,769` / `11,201` | `0.6166` / `0.7259` / `0.8467 ms` | `0.9683` / `1.0612` / `1.3064 ms` | `6.5604` / `7.2702` / `12.8609 ms` | — | — |
| Snapshot bytes (UTF-8 JSON) | Overall (`n = 31,790`) | `48,939 B` | `84,873 B` | `118,421 B` | `50,214 B` | `14,675 B` |
| — Early / Mid / Late | `9,820` / `10,769` / `11,201` | `26,601` / `47,428` / `71,025 B` | `37,787` / `63,529` / `94,646 B` | `50,801` / `82,267` / `118,421 B` | — | — |
| Process RSS (`process.memoryUsage().rss`) | `160` game-end samples | `429.96 MB` | `541.97 MB` | `550.04 MB` | `422.07 MB` | `121.18 MB` |
| V8 `heapUsed` | `160` game-end samples | `186.85 MB` | `344.02 MB` | `384.48 MB` | `199.81 MB` | `30.07 MB` |

### Completed-Game Wall Time by Map (`n = 20` games/map; canonical order)

| Map | Without Observation: Median / p95 / Max (ms) | With `Server.getPlayerModel()` per Decision: Median / p95 / Max (ms) |
| --- | --- | --- |
| `tharsis` | `16.35` / `26.12` / `28.46` | `38.57` / `53.67` / `68.68` |
| `elysium` | `17.15` / `21.11` / `23.17` | `45.26` / `61.05` / `69.07` |
| `hellas` | `20.61` / `25.51` / `28.66` | `52.05` / `71.41` / `78.56` |
| `vastitas borealis nova` | `15.88` / `22.69` / `25.43` | `39.26` / `65.94` / `67.22` |
| **All Maps (`n = 80`)** | **`17.10` / `25.43` / `28.66`** (**PASS** vs `≤ 30,000 ms`) | **`44.26` / `67.22` / `78.56`** |

### Localhost HTTP Server vs. In-Process Comparison

Measured against the unmodified built engine server (`engine/build/src/server/server.js`) using its default local SQLite backend (`better-sqlite3` in an isolated temp directory) over `1` warm-up game (`tharsis`, `562` decisions, `25,757 ms`) + `4` measured games (`1` per map, `4/4` completed, `1,456` decisions):

| Metric | In-Process (Canonical) | Localhost HTTP (`GET /api/waitingfor` + `POST /player/input`) | Speedup (In-Process vs. HTTP) |
| --- | --- | --- | --- |
| `GET /api/waitingfor` (median / p95 / max) | — | `14.750 ms` / `16.200 ms` / `17.228 ms` | — |
| `POST /player/input` (median / p95 / max) | `0.0283 ms` (`player.process`) | `17.882 ms` / `20.603 ms` / `39.260 ms` | **~631.9×** (step-only) |
| Combined per-decision (`obs+step` vs `waitingfor+input`) | `0.0828 ms` (`0.0545 + 0.0283 ms`) | `31.520 ms` / `35.970 ms` / `55.079 ms` | **~380.7×** |
| Completed-game wall (median / p95 / max) | `44.26 ms` / `67.22 ms` / `78.56 ms` (with obs) | `14,823.64 ms` / `20,339.10 ms` / `20,339.10 ms` | **~334.9×** |

*Why HTTP is ~381× slower:* Every `POST /player/input` round-trips over TCP/JSON and triggers `SQLite.saveGame()` disk commits, and `GET /api/waitingfor` hits SQLite-backed `GameLoader` serialization/lookup overhead. Note also that `POST /api/creategame` uses `Math.random()` rather than an explicit integer seed (`ApiCreateGame.ts`).

---

## 7. Stage 5: Scaling (`npm run scaling`)

Executed in [`scaling.ts`](file:///C:/2026/tm_bot/tools/simbench/src/scaling.ts) using Node `worker_threads` for `60 s` per tier (`1, 2, 4, 8, 20` workers) after `2` warm-up games per worker, rotating the 4 target maps in canonical simultaneous order (`77,487` total games completed across all tiers, `0` failed games).

**Why `worker_threads` instead of `child_process`:** Each `worker_thread` runs in its own isolated V8 isolate, giving each worker independent module-level singletons (`Database`, `GameLoader`, `BehaviorExecutor`, `GlobalEventDealer`, `Tag` caches) while sharing one parent OS process (`process.memoryUsage().rss` captures combined process memory directly).

### Throughput, Speedup, and Memory by Worker Count (`60 s` Sustained Tiers)

| Workers | Actual Wall (s) | Games Completed (Failed) | Games / sec | Decisions / sec | Speedup vs. 1w | Scaling Efficiency | Process RSS Mean / Max (MB) | RSS / Worker Max (MB) | Worker Heap Peak Avg / Max (MB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | 60.00 | `3,770` (`0`) | 62.83 | 25,877.4 | `1.00×` | 100.0% | `333.93` / `354.86` | 354.86 | `171.43` / `171.43` |
| `2` | 60.01 | `7,364` (`0`) | 122.71 | 50,457.8 | `1.95×` | 97.7% | `589.49` / `615.65` | 307.83 | `195.78` / `250.98` |
| `4` | 60.02 | `13,229` (`0`) | 220.42 | 90,816.0 | `3.51×` | 87.7% | `993.06` / `1,036.79` | 259.20 | `162.19` / `192.48` |
| `8` | 60.02 | `20,335` (`0`) | 338.79 | 139,549.1 | `5.39×` | 67.4% | `1,813.90` / `2,269.93` | 283.74 | `148.81` / `206.57` |
| `20` | 60.06 | `32,789` (`0`) | 545.94 | 224,812.6 | `8.69×` | 43.4% | `4,303.00` / `4,680.41` | 234.02 | `151.21` / `204.13` |

### Garbage Collection (`PerformanceObserver`) and Event-Loop Delay (`monitorEventLoopDelay`)

| Workers | GC Total Events | GC Total Pause Time (ms) | GC Major Events | GC Max Pause (ms) | ELD p50 Avg (ms) | ELD p95 Max (ms) | ELD Max Pause (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | 1,639 | 1,079.67 | 62 | 4.69 | 15.23 | 23.86 | 41.22 |
| `2` | 3,166 | 2,157.31 | 121 | 6.41 | 15.54 | 24.58 | 49.91 |
| `4` | 5,752 | 4,230.73 | 228 | 22.52 | 17.15 | 28.26 | 76.48 |
| `8` | 8,950 | 7,880.82 | 360 | 53.92 | 22.69 | 35.68 | 127.27 |
| `20` | 14,768 | 21,800.43 | 518 | 71.02 | 35.12 | 76.74 | 278.40 |

*Notes:*
- At `20` workers (`~14.8%` of `31.52 GB` host RAM), max RSS stays at `4.68 GB` (`~234 MB/worker`), leaving `~26.8 GB` headroom for Python/PyTorch training/inference.
- Efficiency tapers from `97.7%` (`2w`) to `43.4%` (`20w`) as Intel hybrid P+E core scheduling, memory bandwidth, and cross-worker V8 GC pauses (`71.02 ms` max GC pause, `278.40 ms` max event-loop delay) saturate all 20 logical threads; ELD p50 (`15.2–35.1 ms`) primarily reflects each worker's per-game synchronous execution before yielding via `setImmediate()`.

---

## 8. Gaps and Out-of-Scope Boundaries

1. **Direct cloning fails at mid-action, deferred, Prelude-effect, and gen ≥ 2 research prompts:**
   - `Game.serialize()` writes `deferredActions: []` (`Game.ts:483`) and `Player.serialize()` omits `waitingFor` and `waitingForCb` (`Player.ts:95-96`, `1758-1858`). `Game.deserialize()` (`Game.ts:1697-1841`) only reconstructs phase-level root prompts (`gotoInitialResearchPhase`, `restoreDraft`, `gotoResearchPhase`, `takeNextFinalGreeneryAction`, `takeAction(false)`).
   - **Impact:** MCTS / search tree branching cannot use raw `Game.deserialize(game.serialize())` at subprompts, deferred actions, Prelude effects, or post-gen-1 `ChooseCards` research. It **must** use the Stage 3 **checkpoint + replay** fallback from the last safe root boundary (`0.905–2.125 ms` median for suffix lengths `1–40`).
2. **Invalid-input atomicity caveat in `SelectInitialCards`:**
   - In Stage 3, all `48/48` representative invalid-input tests threw and left normalized state and pending prompt unchanged (`atomicityFailures: []`). However, source inspection of [`SelectInitialCards.ts`](file:///C:/2026/tm_bot/tools/simbench/engine/src/server/inputs/SelectInitialCards.ts#L59-L98) shows that `prelude` and `ceo` sub-option callbacks mutate `player.preludeCardsInHand` / `player.ceoCardsInHand` before later sub-options validate; `completed()` only clears `cardsInHand` and `preludeCardsInHand` on the `"Too many cards selected"` affordability check (`SelectInitialCards.ts:94-98`), not if `project` fails inside `SelectCard.process` after valid `corp`+`prelude` sub-responses. Any RL/search adapter should validate responses before calling `player.process()` or restore from checkpoint on caught `InputError`.
3. **Simultaneous prompt submission order (`RESEARCH` / `SelectInitialCards`):**
   - As proven in Stage 2, submitting simultaneous `RESEARCH` or `SelectInitialCards` prompts out of canonical seat order mutates `discardPile` order immediately and causes deck reshuffle divergence later in the game. The RL/search environment adapter must enforce canonical seat order (`seat 0` first).
4. **Amazonis Planitia incompatibility (pre-existing gap, out of scope for Issue #12):**
   - As documented in [`simulator-feasibility.md`](file:///C:/2026/tm_bot/docs/research/simulator-feasibility.md), the engine's `amazonis p.` board differs from FryxGames' publisher Amazonis Planitia rules (9-ocean global cap vs. 11 oceans, 61-hex layout, and different milestones/awards) and was excluded from `TARGET_MAPS`.
5. **Out-of-scope items per Issue #12:**
   - Card-by-card rules audit against official FAQs, human game-log replay fidelity (deferred to [Issue #7](https://github.com/finger-six/tm_bot/issues/7)), observation leakage adversarial testing across private hidden state, and GPU model inference/training throughput on the RTX 4080 Super.
