# Paired Evaluation Harness (`tm_bot.eval`)

Fair, reproducible two-player evaluation over disjoint seed namespaces (`train-*`, `dev-*`, `eval-*`) and seat-swapped pairs across the four supported maps (`tharsis`, `elysium`, `hellas`, `vastitas borealis nova`), with `amazonis planitia` explicitly reported as `not supported`.

## Protocol

- **Seed namespaces** ([`tm_bot/eval/seeds.py`](../tm_bot/eval/seeds.py)): `SeedKey` enforces one of `train`, `dev`, or `eval` prefixes (`<namespace>-<identifier>`), keeping training, checkpoint selection (`dev`), and final evaluation (`eval`) seeds disjoint by construction. Per-game agent seeds are derived deterministically via SHA-256 from `(seed_key, map_name, role)`.
- **Seat-swapped pairing** ([`tm_bot/eval/pairing.py`](../tm_bot/eval/pairing.py)): Every `(seed_key, map_name)` is played twice — `(A, B)` and `(B, A)` with swapped per-agent seeds — so each agent plays every deal from both seats. The default 400-game evaluation uses `50` seeds x `4` supported maps x `2` seats.
- **Agent isolation & search extension** ([`tm_bot/agents/protocol.py`](../tm_bot/agents/protocol.py), [`tm_bot/agents/sampler.py`](../tm_bot/agents/sampler.py), [`tm_bot/agents/runner.py`](../tm_bot/agents/runner.py)): Standard agents (`Agent`) receive ONLY `(observation, choice)` from `agent.act(obs, choice)`. Search agents implementing `SearchAgent` (`act_with_sampler`, plus optional cross-game batch hook `act_batch_with_samplers`) receive `(observation, choice, sampler)` where `sampler` is a seat-scoped `WorldSampler` (`ScopedWorldSampler`) allowing `sample(k, seed)` of actor-consistent determinized clones plus `clone`/`close`/step ops strictly on owned clones. The runner binds each sampler to a *belief root* (a `determinize_clone` of the decision point sampled from the actor observation), so the sampler never holds the live game ID and cannot reach the true hidden state; the runner closes every sampled clone and the root at the end of each decision (including on error). See [search.md](search.md).
- **Scoring & failure isolation**: Engine-declared wins score `1.0`, exact ties score `0.5`, and losses score `0.0`. Failures (`ok=False`: step cap, adapter/bridge errors, or raising/illegal agents) are excluded from win rate, VP margin, generations, and bootstrap CIs, and are reported separately.
- **Paired bootstrap CI** ([`tm_bot/eval/stats.py`](../tm_bot/eval/stats.py)): 95% percentile bootstrap CI resamples whole seeds for the overall win rate (the adapter derives the deal from the seed key alone, so a seed's games on every map and both seats are one cluster) and `(seed_key, map_name)` seat-swapped pairs for per-map rates, with a deterministic RNG seed (`DEFAULT_BOOTSTRAP_SEED = 0`, `DEFAULT_BOOTSTRAP_RESAMPLES = 10_000`).
- **Reproducible reporting** ([`tm_bot/eval/report.py`](../tm_bot/eval/report.py)): Outputs `report.json` (containing separated `deterministic` and `runtime` sections), `report_deterministic.json` (canonical deterministic-only JSON), and `report.md`. Per-decision wall-clock latency (`p50`/`p95`/`mean`, plus searched-decision-only latency when present) is measured inside `tm_bot.agents.runner` without altering the `(observation, choice)` inputs passed to agents.

## CLI Usage

```bash
conda run -n tm_bot python -m tm_bot.eval --a b1 --b b0 --games 400 --namespace eval --out data/reports/b1_vs_b0
```

Pass `--omit-timing` to exclude runtime latency, wall-clock, and hardware sections from `report.json` and `report.md`.

## Baseline Acceptance Run (`B1` vs `B0`, 400 `eval-*` games)

### `b1-v2` (current, #34)

`data/reports/b1v2_vs_b0` (gitignored).

- **Agents**: `b1` (`b1-v2`, `training_seeds=0`) vs `b0` (`b0-v1`, `training_seeds=0`)
- **Games**: `400` total, `400` completed, `0` failures (`0.00%`)
- **Record (`b1`)**: `400W - 0L - 0T`
- **Win Rate (95% paired bootstrap CI)**: **100.00%** `[100.00%, 100.00%]` (target `>= 90%`)
- **Mean VP Margin (`b1 - b0`)**: `+86.38` (`125.56` vs `39.18`)
- **Generations / Game**: `13.23` (`min=10`, `max=18`), `184,888` total decisions (`462.2`/game)

| Map | Status | Games | Failures | Win Rate | 95% CI | Mean VP Margin | Mean Gen |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| Tharsis | supported | 100 | 0 | 100.00% | [100.00%, 100.00%] | +84.22 | 13.33 |
| Elysium | supported | 100 | 0 | 100.00% | [100.00%, 100.00%] | +90.51 | 13.34 |
| Hellas | supported | 100 | 0 | 100.00% | [100.00%, 100.00%] | +86.08 | 13.15 |
| Vastitas Borealis Nova | supported | 100 | 0 | 100.00% | [100.00%, 100.00%] | +84.69 | 13.12 |
| Amazonis Planitia | not supported | 0 | 0 | not supported | not supported | not supported | not supported |

### `b1-v1` (historical, superseded by #34)

`b1-v1` never took a corporation's mandatory first action; its `16` losses below are those games.

Executed twice (`data/reports/b1_vs_b0_run1` and `data/reports/b1_vs_b0_run2`, gitignored); `report_deterministic.json` was byte-identical across both runs (`2,860` bytes).

- **Agents**: `b1` (`b1-v1`, `training_seeds=0`) vs `b0` (`b0-v1`, `training_seeds=0`)
- **Games**: `400` total, `400` completed, `0` failures (`0.00%`)
- **Record (`b1`)**: `384W - 16L - 0T`
- **Win Rate (95% paired bootstrap CI)**: **96.00%** `[92.00%, 99.00%]` (target `>= 90%`)
- **Mean VP Margin (`b1 - b0`)**: `+78.42` (`121.28` vs `42.85`)
- **Generations / Game**: `14.18` (`min=10`, `max=55`), `191,579` total decisions (`478.9`/game)
- **Per-decision latency** (8 workers, batch size 32, Windows 11 20-CPU host): `b1` p50 `0.0110 ms` / p95 `0.0477 ms`; `b0` p50 `0.0008 ms` / p95 `0.0048 ms`

| Map | Status | Games | Failures | Win Rate | 95% CI | Mean VP Margin | Mean Gen |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| Tharsis | supported | 100 | 0 | 96.00% | [92.00%, 99.00%] | +76.64 | 14.20 |
| Elysium | supported | 100 | 0 | 96.00% | [92.00%, 99.00%] | +81.74 | 14.74 |
| Hellas | supported | 100 | 0 | 96.00% | [92.00%, 99.00%] | +78.69 | 13.92 |
| Vastitas Borealis Nova | supported | 100 | 0 | 96.00% | [92.00%, 99.00%] | +76.62 | 13.85 |
| Amazonis Planitia | not supported | 0 | 0 | not supported | not supported | not supported | not supported |
