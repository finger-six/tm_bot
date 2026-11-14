# Baseline Agents (`B0` and `B1`)

All agents implement `tm_bot.agents.protocol.Agent`:

- `reset(seed: int | None = None) -> None`
- `act(observation: ActorObservation, choice: Choice) -> int`

Agents receive **only** the actor observation (`ActorObservation`) and current atomic choice (`Choice`). The runner (`tm_bot.agents.play_game` / `play_games`) never exposes the game session, game ID, worker client, or the opponent's private state.

## `B0` (`B0_VERSION = "b0-v1"`)

- **Implementation**: [`tm_bot/agents/b0.py`](../tm_bot/agents/b0.py) (`B0Agent`)
- **Behavior**: Selects uniformly at random over `range(len(choice.candidates))` using an isolated `random.Random(seed)` instance (never the global `random` module).
- **Determinism**: Given `(seed_key, map_name, seat0_seed, seat1_seed)`, `B0` produces an identical candidate sequence and game outcome.

## `B1` (`B1_VERSION = "b1-v1"`)

- **Implementation**: [`tm_bot/agents/b1.py`](../tm_bot/agents/b1.py) (`B1Agent`)
- **Freeze policy**: `B1`'s behavior is frozen under `B1_VERSION = "b1-v1"`. Any behavioral change requires bumping `B1_VERSION`. Game outcomes from `play_game` / `play_games` record `agent_versions`.

### Static Card Data Generation

`B1` uses public static rules metadata (`cost`, `cardType`, `tags`, `startingMegaCredits`, `cardCost`, `staticVp`, `production`, `stock`, `tr`, global-parameter steps, `drawCard`, `hasAction`, `discountAmount`, `requirements`) derived directly from the pinned engine (`tools/simbench/engine`):

```bash
cd adapter
npm run export:cards
```

Or from Python:

```bash
conda run -n tm_bot python -m tm_bot.agents.card_data --export
```

This writes the gitignored file `data/cards.json`. `load_card_data()` auto-regenerates it if missing.

### `B1` Decision Rules (`b1-v1`)

1. **Horizon & valuation**:
   - Estimates remaining terraforming steps `rem_steps = (8 - temp)//2 + (14 - oxy) + (9 - oceans)` (max `42`, `rem_frac = rem_steps / 42`).
   - Production horizon scales with `rem_frac`: `prod_horizon = 0.8 + 6.2 * rem_frac`.
   - VP value scales late: `vp_value = 5.0 + 3.0 * (1 - rem_frac)`.
   - Effective TR (capped by remaining global tracks) is valued at `vp_value + 1.0 * prod_horizon`.
   - Card static value sums effective TR, greenery/city value, production box, immediate stock, static/dynamic VP, card draw (`3.5`), active card actions, discounts, and tag count.

2. **Setup, draft, and card buying**:
   - **Corporation**: maximizes `startingMegaCredits + card_static_value - (cardCost - 3) * 4 + tag_synergy`.
   - **Preludes**: maximizes `card_static_value + startingMegaCredits`.
   - **Drafting**: picks highest `card_static_value - 0.85 * cost`.
   - **Buying (`initial` / `research` / look-and-buy)**: buys affordable cards above net-value threshold (`cardCost + 2.0` early/mid, `cardCost + 4.0` when `rem_steps <= 14`; initial buy uses the chosen corporation's `startingMegaCredits` and `cardCost`, capped at 5 cards while reserving `12 M€`); otherwise picks `done`.
   - **Discard / Sell patents**: keeps cards when `done` is offered on Sell Patents; when forced, discards/sells the lowest net-value card.

3. **Action-phase branch priority**:
   1. **Claim a milestone** (always when available).
   2. **Convert plants into greenery** (always when available).
   3. **Convert heat into temperature** (when `temperature < 8`).
   4. **Play project card** (when at least one likely-playable card in hand clears the net-value threshold: `-0.5` early/mid, `-2.0` when `rem_steps <= 10`).
   5. **Perform an action from a played card** (prefers TR/VP/draw card actions).
   6. **Fund an award** (only when strictly ahead in score on an unfunded award by at least the tier lead margin for `8 / 14 / 20 M€`).
   7. **Standard projects** (`Greenery`, `Aquifer`, `Asteroid:SP`, `City` when affordable and justified by remaining tracks/stage; avoids `Power Plant:SP`).
   8. **Pass for this generation** (prevents `End Turn` infinite loops when no worthwhile action remains).

4. **Nested choices**:
   - **Payment**: always selects candidate index `0` (the adapter's canonical minimum-M€ split).
   - **Spaces**: scores each candidate space by its `SpaceBonus` yields plus exact 6-neighbor hex adjacency (favoring greeneries/oceans near own cities, avoiding feeding opponent cities, and placing cities near greeneries/oceans).
   - **Player target**: selects the opponent color over self/neutral.
   - **Resource**: gains highest-priority (`titanium > plants > steel > megacredits > heat > energy`), loses lowest-priority (`energy > heat > megacredits > plants > steel > titanium`).
   - **Amount**: selects `max` for positive conversions (and `Insulation` late-game when `temperature >= 0` or `rem_steps <= 14`), `min` when losing/discarding.

## Quick Gate Check

Run the 200-game seat-swapped quick gate (`100` seeds x `2` seats across all 4 supported maps):

```bash
conda run -n tm_bot python -m tm_bot.agents.gate
```
