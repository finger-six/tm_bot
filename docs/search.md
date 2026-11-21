# Search Agent (determinized IS-MCTS)

Ticket #23 adds the main agent's search: determinized information-set MCTS with
PUCT, neural priors and leaf values from the warm-start policy/value network
([warmstart.md](warmstart.md)). Code: [`tm_bot/search/`](../tm_bot/search/),
[`tm_bot/agents/sampler.py`](../tm_bot/agents/sampler.py), and the adapter's
belief sampler `sampleWorldFromObservation` in
[`adapter/src/determinize.ts`](../adapter/src/determinize.ts).

## Where search runs

`is_search_decision(observation, choice)` ([`config.py`](../tm_bot/search/config.py))
selects the decisions that are searched; everything else uses the policy head directly.

| Decision | Searched? |
| --- | --- |
| Action-phase root (`branch` with an action-root prompt) | yes |
| Draft pick (`drafting`, `initial_drafting`, kind `card`) | yes |
| Research / initial project buy (`multi_card` with "buy") | yes |
| Nested sub-choices (payment, space, player, amount, nested card/branch) | no, policy head |
| Initial corporation / prelude picks | no, policy head |
| Single-candidate (forced) choices | no, index 0 |

## Algorithm

For a searched decision the agent:

1. Asks its `WorldSampler` for `K = min(num_worlds, num_simulations)` worlds
   (`sample(K, seed)`); each world is a `determinize_clone` whose hidden state
   (opponent hand, dealt and drafted cards, deck order) is drawn uniformly from the
   actor-consistent belief, using only the acting seat's observation.
2. Runs `num_simulations` simulations in waves of `wave_size`, cycling through the
   worlds. All worlds share one root node (shared root statistics); below the root
   each world keeps its own subtree. Selection is PUCT with virtual loss; Q values are
   sign-adjusted to the perspective of the seat to move, and unvisited children get
   `fpu_value` in that same perspective.
3. Expands one node per simulation by cloning the parent clone and applying the
   action. Nested sub-choices inside the simulation are played greedily by the policy
   head (at most `max_nested_steps`) until the next searched decision or the end of the
   game. The leaf is scored by the value head (or the exact terminal result), from the
   root actor's perspective.
4. Picks the root action: most visits at temperature 0, otherwise sampled from
   visits^(1/T). `action_temperature` overrides `temperature` in the action phase.

Neural evaluation is batched on the GPU across all concurrent games: the runner groups
searched decisions from every active game into one `act_batch_with_samplers` call, and
each wave issues one `predict_batch` and one worker-pool batch for all of them.

If no actor-consistent world can be sampled for a decision, the agent plays the
policy-head choice for that decision instead of failing the game (see
"Known limits").

## Leakage and live-game safety

- The belief sampler `sampleWorldFromObservation(obs, seed)` takes only the actor
  observation and a seed. Tests: `Ticket 23` in
  [`adapter/test/clone.test.ts`](../adapter/test/clone.test.ts) swaps a card between the
  true opponent hand and the deck, checks the actor observation and the sampled world
  are unchanged, and checks that determinizing the mutated game and an unmutated twin
  gives identical hidden worlds.
- The runner never passes the live game to an agent. For each searched-agent decision
  it creates a *belief root* with `determinize_clone(live, seat, 0)` and binds the
  agent's `ScopedWorldSampler` to it. The sampler stores neither the live game id nor
  the true hidden state; it can only address the belief root (via `sample`) and the
  clones it owns, and every other id is rejected with `PermissionError`.
  `test_world_sampler_has_no_reachable_path_to_live_game` walks every object reachable
  from the sampler and from the sampler module and asserts the live id is not among
  them. This is a structural guard against accidental leaks by in-process code, not a
  security boundary: Python code can always introspect the worker pool.
- Clones and the belief root are closed after every decision, also on error.
  `test_search_returns_legal_candidate_never_mutates_live_game_and_cleans_clones` plays
  a full game and checks after every searched decision that the live observations,
  choice and pending seats are unchanged and no clone is left open.

## Configuration

`SearchConfig` fields: `num_simulations`, `num_worlds` (K), `c_puct`, `fpu_value`,
`temperature`, `action_temperature`, `wave_size`, `max_nested_steps`.

The evaluated setting, version tag `ismcts-s8-k4-warmstart-v2`, is the factory
`tm_bot.search:make_warmstart_search_agent` with its defaults:

| Setting | Value |
| --- | --- |
| Checkpoint | `runs/warmstart/best` (warm-start v2, #22) |
| `num_simulations` | 8 |
| `num_worlds` (K) | 4 |
| `wave_size` | 4 |
| `c_puct` | 1.5 |
| `temperature` (draft/buy) | 0.0 |
| `action_temperature` | 1.0 |

Each can be overridden in the agent spec (for example
`tm_bot.search:make_warmstart_search_agent?num_simulations=16&num_worlds=4`) or with
the `TM_SEARCH_*` environment variables. The runner records every decision's
wall-clock latency and, separately, the latency of searched decisions only; both
appear in the evaluation report.

## Results

400 `eval-*` games per pairing (seat-swapped pairs, four supported maps, Amazonis
Planitia reported as not supported), setting `ismcts-s8-k4-warmstart-v2`, 16 workers,
batch size 32, one training seed. Reports: `data/reports/search_vs_b0/` and
`data/reports/search_vs_b1/` (gitignored). Runs with `--resume` or
`--max-games-per-run` also keep per-game rows in `games.jsonl`.

| Pairing | Completed | Failures | Search win rate (95% paired bootstrap CI) | Mean VP margin | Gens/game | Wall time |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Search vs B0 | 400 | 0 | **62.00%** [56.75%, 67.25%] | +6.87 (84.4 vs 77.5) | 23.69 (15-40) | 3,368 s |
| Search vs B1 | 400 | 0 | **4.00%** [1.00%, 8.00%] | -83.38 (43.5 vs 126.8) | 14.82 (11-51) | 2,051 s |

Per map vs B0: Tharsis 53%, Elysium 67%, Hellas 65%, Vastitas Borealis Nova 63%.
Per map vs B1: 4% on every map (each seed key fixes the deal on all four maps, so
outcomes are strongly correlated across maps; win attribution was checked).
The longest games (up to 51 generations vs B1, 40 vs B0) exceed the 6-25 game-shape
range of #10. Both runs had 0 failures and 0 step caps, so these are slow games rather
than adapter errors; the likely cause is the untrained action policy (see "Known limits").
The longest games (up to 51 generations vs B1, 40 vs B0) exceed the 6-25 game-shape
range of #10; they come from the weak action policy stalling terraforming, not from
adapter errors (0 failures, 0 step caps).

Decision latency of the search agent (wall clock, 16 concurrent games sharing one GPU):

| Pairing | All decisions p50 / p95 | Searched decisions p50 / p95 (n) |
| --- | --- | --- |
| vs B0 | 27.7 / 50.4 ms | 34.2 / 53.5 ms (82,321) |
| vs B1 | 28.4 / 51.4 ms | 34.4 / 55.7 ms (48,969) |

Sampler fallbacks (`world sampling failed` / `belief root sampling failed` warnings):
0 in both runs. An earlier pre-fix run vs B0 had 2 of 400 games fail in the sampler
(see "Known limits").

## Known limits

- **Weak play is expected at this stage.** The warm-start network was trained only on
  setup and draft/buy decisions from human games, and its value head was trained on
  setup-time outcomes. Search reuses that value head as a mid-game leaf evaluator and
  that policy head for action-phase priors, neither of which it was trained for, with
  only 8 simulations per decision. Real strength needs the self-play training of #24.
- `action_temperature = 1.0` samples action-phase choices from the visit distribution
  (and nested action-phase sub-choices from the policy head) instead of taking the
  argmax. With an untrained action policy, argmax play tends to repeat the same
  low-value actions; sampling keeps games progressing.
- **Belief sampler failures (fixed, #31).** In the first run vs B0, which used a
  pre-commit version of #23, the actor-consistent belief became unsatisfiable in 2 of
  400 games (`Not enough unseen projects: need 169, have 151`; related:
  `opponentHand exceeded opponentHandSize`). Root cause: two paths left stale cards in
  `knownProjectDiscardPile` after the deck reshuffled the discard pile, so the belief
  claimed more known discards than the pile held. (1) Drafted cards the actor did not
  buy were counted as actor-acquired, so after a reshuffle they stayed in the known
  discard pile. (2) A `Discarded N cards` batch that spans a reshuffle was credited to
  the new pile (the `spansReshuffle` guard in `adapter/src/determinize.ts`). Both were
  fixed in the #23 commit (ea97aa5), before the reported runs above.
  - Regression tests: the two reshuffle tests in `adapter/test/clone.test.ts` make the
    post-reshuffle belief exactly tight and sweep 64 sampler seeds
    (`assertSamplesEverySeed`). With either stale path restored, they throw the original
    `Not enough unseen projects` error.
  - `npm run fuzz:determinize` checks at every truth check that the belief contains the
    truth (`beliefViolations`: deck cards unseen, known discards in the discard pile,
    known and possible opponent cards placed correctly): 7,316 checks, 0 violations.
  - A fresh 400-game search-vs-B0 eval on the fixed code
    (`data/reports/issue31_search_vs_b0/`): 400/400 completed, 0 failures, 0 sampler
    fallback warnings (stderr empty). Its headline numbers match the reported run above
    (62.00% [56.75%, 67.25%], +6.87 VP, 23.69 generations per game); #31's commit
    (b5e3215) changed only tests and the fuzz script, not the runtime.
  - The agent keeps the policy-head fallback for a decision with no sampleable world, as
    a safety net; it is no longer expected to fire.
