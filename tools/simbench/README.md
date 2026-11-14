# tm-simbench

In-process benchmark and validation harness for the open-source [Terraforming Mars TypeScript engine](https://github.com/terraforming-mars/terraforming-mars) (`tools/simbench/engine`).

## Pinned Engine Revision

- **Submodule path:** `tools/simbench/engine`
- **Repository:** `https://github.com/terraforming-mars/terraforming-mars.git`
- **Commit:** `f91e1e2574175c6affa2440a2c66a8fc00412e1e`
- **Engine source patches:** None (engine source is unmodified).

## Install & Build

Tested on Windows 11 with **Node v24.18.0** and **npm 11.16.0** (the engine `package.json` declares `engines.node = "22.x"`, which emits an `EBADENGINE` warning on Node 24 without blocking install or build).

```powershell
# 1. Initialize submodule if cloning fresh
git submodule update --init tools/simbench/engine

# 2. Install and build the engine server code (generate static JSON manifests + tsc + tsc-alias)
cd tools/simbench/engine
npm install --no-audit --no-fund
npm run make:json
npm run build:server

# 3. Install and build simbench
cd ..
npm install --no-audit --no-fund
npm run build
```

## Run

Inside `tools/simbench/`:

```powershell
# Stage 1: Smoke (1 detailed game + 20 seeds per target map, canonical simultaneous order)
npm run smoke

# Stage 2: Determinism (strict replay + alternated simultaneous draft/research order)
npm run determinism

# Stage 3: Cloning (direct Game.deserialize + fallback checkpoint+replay + atomicity)
npm run cloning

# Stage 4: Latency (in-process distributions by generation tertile + localhost HTTP comparison)
npm run latency

# Stage 5: Scaling (worker_threads throughput, RSS, GC & event-loop delay across 1..CPU cores)
npm run scaling
```

Optional flags:

```powershell
node dist/smoke.js --seeds 50
node dist/scaling.js --seconds 60
```

Outputs are written to `tools/simbench/results/`:
- `results/summary.json` (tracked in git; combined Stage 1–5 summary)
- `results/smoke.json`, `results/determinism.json`, `results/cloning.json`, `results/latency.json`, `results/scaling.json` (gitignored by `tools/simbench/.gitignore`)

## In-Process Isolation

`src/engine.ts` (`initializeEngine()`) replaces `Database.getInstance` and `GameLoader.getInstance` with pure in-memory no-op fakes before calling `globalInitialize()`. Unlike `engine/tests/testing/setup.ts`, it never calls the real `GameLoader.getInstance()` constructor, so neither `SQLite`/`PostgreSQL`/`LocalFilesystem` nor `Cache`'s background timer is ever created.

## Explicit Game Configuration

Every game is constructed via `createExplicitTwoPlayerGame()` in `src/config.ts` without relying on `DEFAULT_GAME_OPTIONS`:

- **Players:** 2 seats (`Seat 1` red / `p-1`, `Seat 2` blue / `p-2`), `beginner = false`, `handicap = 0`, explicit `firstPlayer = players[0]`, explicit integer `seed`.
- **Boards (`TARGET_MAPS`):** `tharsis`, `elysium`, `hellas`, `vastitas borealis nova` (`BoardName` enum values).
- **Expansions (legacy booleans and `expansions` record kept strictly consistent):**
  - `corporateEra: true` (`expansions.corpera: true`)
  - `preludeExtension: true` (`expansions.prelude: true`)
  - All other expansions, promos, and community cards `false`.
- **Variants & Rules:**
  - `draftVariant: true`
  - `initialDraftVariant: false`, `preludeDraftVariant: false`, `ceosDraftVariant: false`
  - `showOtherPlayersVP: false`, `showTimers: false`, `undoOption: false`, `fastModeOption: false`
  - `shuffleMapOption: false`, `randomMA: RandomMAOptionType.NONE`, `includeFanMA: false`, `modularMA: false`
  - `startingCorporations: 2`, `startingPreludes: 4`, empty custom/banned/included card lists.

## Stage 2 State Normalization & Canonical Submission Order

`normalizeSerializedGame()` in `src/normalize.ts` strips or canonicalizes the following non-gameplay / wall-clock / identity fields before deep structural comparison:

1. `createdTimeMs` — wall-clock timestamp (`new Date()`) from `Game.newInstance()`.
2. `lastSaveId` — database save counter.
3. `id` — caller-supplied game ID (normalized to `'<GAME_ID>'`).
4. `name` — unseeded random game name generated via `UnseededRandom.INSTANCE`.
5. `spectatorId` — caller-supplied spectator ID (normalized to `'<SPECTATOR_ID>'`).
6. `gameLog[*].timestamp` — wall-clock `Date.now()` captured in `LogMessage` constructor.
7. `gameLog[*].data[*].value` — raw `game.id` string logged at `Phase.END` (`"This game id was ${0}"` in `Game.gotoEndGame()`), normalized to `'<GAME_ID>'`.
8. `players[*].timer` — wall-clock `Timer` state (`sumElapsed`, `startedAt`, `running`, `afterFirstAction`, `lastStoppedAt`).

### Simultaneous Order Finding (Stage 2)

- **Exact replay** (same seed, same action sequence and submission order, distinct `gameId`/`spectatorId`) is 100% deterministic across every step on all 4 maps.
- **Alternated `DRAFTING` submission order** (with canonical `RESEARCH`/`SelectInitialCards` order) is also 100% deterministic across all 4 maps (`0` mismatches).
- **Alternated `RESEARCH` / `SelectInitialCards` submission order** causes `corporationDeck.discardPile`, `preludeDeck.discardPile`, `projectDeck.discardPile`, and `gameLog` to reflect submission order immediately at step 2; if `projectDeck` exhausts and reshuffles its `discardPile` later in the game, the shuffled deck permutation diverges. `src/driver.ts` (`pickNextWaitingPlayer`) enforces `'canonical'` seat order (seat 0 first whenever >1 required player is waiting).

## Stages 3–5 Modules

- **Stage 3 (`src/cloning.ts`):** Tests direct `Game.deserialize(deepCopy(game.serialize()))` across 12 decision-point categories per map (prompt equivalence, continuation equivalence, 20-step branch isolation, and invalid-input atomicity), plus fallback **checkpoint + replay** at every point where direct restore fails and a suffix-length cost sweep (`0..40`).
- **Stage 4 (`src/latency.ts`, `src/httpBench.ts`):** Measures reset, `Server.getPlayerModel()` observation, `player.process()` step, `serialize()+JSON.stringify`, `JSON.parse+Game.deserialize`, snapshot bytes (overall & early/mid/late generation tertiles), completed-game wall time with/without observation, RSS/heapUsed, and localhost HTTP `GET /api/waitingfor` + `POST /player/input` comparison against the built engine server.
- **Stage 5 (`src/scaling.ts`):** Benchmarks multi-worker throughput (`games/s`, `decisions/s`), process RSS, worker `heapUsed`, GC pause events (`PerformanceObserver`), and event-loop delay (`monitorEventLoopDelay`) using `worker_threads` (isolated V8 isolates per worker) across 1, 2, 4, 8, and logical-CPU worker counts.
