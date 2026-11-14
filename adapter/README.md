# tm-adapter

Factored-decision TypeScript adapter around the pinned Terraforming Mars engine (`tools/simbench/engine`) for playing complete two-player training-format games.

## Prerequisites

The engine submodule at `tools/simbench/engine` must be initialized and built first:

```bash
git submodule update --init tools/simbench/engine
cd tools/simbench/engine
npm ci --no-audit --no-fund
npm run make:json
npm run build:server
```

## Install, Build, Test, Fuzz

From `adapter/`:

```bash
npm install --no-audit --no-fund
npm run build
npm test
npm run fuzz -- 10000
```

- `npm run build`: compiles TypeScript (`src/`, `test/`, `scripts/`) into `dist/`.
- `npm test`: builds and runs the `node:test` suite (`dist/test/*.test.js`), including 400 random-vs-random games (100 per supported map).
- `npm run fuzz -- N`: non-CI fuzz script that plays `N` random-vs-random games evenly across the four supported maps and reports completion, rejection, step-cap, and deadlock counts.

## Public API

- `createGame({ seedKey, map })`: creates a deterministic two-player training-format game (`drafting` + `Prelude` + `Corporate Era` only, `showOtherPlayersVP: false`). Accepts `'tharsis'`, `'elysium'`, `'hellas'`, `'vastitas borealis nova'`; rejects `'amazonis p.'` and any unsupported map or option.
- `pending()`: returns seats (`0 | 1`) that must currently decide, in canonical seat order (`[0, 1]` when both are pending).
- `choice(seat)`: returns the current atomic choice (`kind`, `prompt`, ordered `candidates` with stable `id`s and human-readable `label`s).
- `choose(seat, candidateIndex)`: applies one atomic choice; submits the full `InputResponse` to `player.process()` once the decision tree is complete. Engine rejections surface as `AdapterBugError` with `seedKey` and full `history` (no silent retries).
- `result()`: returns `null` until game end, then `{ scores, winner, tie, generations }`.
