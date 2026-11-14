"""Quick 200-game seat-swapped gate script: B1 vs B0 across all 4 supported maps."""

from __future__ import annotations

import argparse
import statistics
import time

from tm_bot.agents.b0 import B0_VERSION, B0Agent
from tm_bot.agents.b1 import B1_VERSION, B1Agent
from tm_bot.agents.card_data import load_card_data
from tm_bot.agents.runner import GameSpec, play_games
from tm_bot.env.types import SUPPORTED_MAPS


def run_gate(
    *,
    seeds_per_map: int = 25,
    num_workers: int = 8,
    batch_size: int = 32,
) -> int:
    card_data = load_card_data()
    specs: list[GameSpec] = []
    b1_seats: list[int] = []

    seed_counter = 0
    for map_name in SUPPORTED_MAPS:
        for i in range(seeds_per_map):
            seed_key = f"gate-b1-b0-{map_name}-{i}"
            s_b1 = 10_000 + seed_counter
            s_b0 = 20_000 + seed_counter
            seed_counter += 1

            # Seat 0 = B1, Seat 1 = B0
            specs.append(
                GameSpec(
                    seed_key=seed_key,
                    map_name=map_name,
                    agents=(B1Agent(seed=s_b1, card_data=card_data), B0Agent(seed=s_b0)),
                    seat_seeds=(s_b1, s_b0),
                )
            )
            b1_seats.append(0)

            # Swapped: Seat 0 = B0, Seat 1 = B1
            specs.append(
                GameSpec(
                    seed_key=seed_key,
                    map_name=map_name,
                    agents=(B0Agent(seed=s_b0), B1Agent(seed=s_b1, card_data=card_data)),
                    seat_seeds=(s_b0, s_b1),
                )
            )
            b1_seats.append(1)

    t0 = time.perf_counter()
    outcomes = play_games(specs, num_workers=num_workers, batch_size=batch_size)
    wall_s = time.perf_counter() - t0

    failures = [out for out in outcomes if not out.ok or out.result is None]
    wins = 0
    losses = 0
    ties = 0
    generations: list[int] = []
    margins: list[int] = []
    decisions: list[int] = []
    per_map: dict[str, list[tuple[float, int, int]]] = {m: [] for m in SUPPORTED_MAPS}

    for out, b1_seat in zip(outcomes, b1_seats):
        if not out.ok or out.result is None:
            continue
        res = out.result
        generations.append(res.generations)
        decisions.append(out.decisions)
        b1_vp = res.scores[b1_seat]
        b0_vp = res.scores[1 - b1_seat]
        margin = b1_vp - b0_vp
        margins.append(margin)
        if res.tie:
            ties += 1
            pts = 0.5
        elif res.winner == b1_seat:
            wins += 1
            pts = 1.0
        else:
            losses += 1
            pts = 0.0
        per_map[out.map].append((pts, res.generations, margin))

    completed = wins + losses + ties
    win_rate = ((wins + 0.5 * ties) / completed) if completed > 0 else 0.0
    mean_gen = statistics.mean(generations) if generations else 0.0
    mean_margin = statistics.mean(margins) if margins else 0.0
    mean_decisions = statistics.mean(decisions) if decisions else 0.0

    print(f"=== Baseline Quick Gate: {B1_VERSION} vs {B0_VERSION} ===")
    print(f"Games: {len(outcomes)} total | {completed} completed | {len(failures)} failures")
    print(f"Result: {wins}W - {losses}L - {ties}T | Win rate: {win_rate * 100:.2f}% (target >= 90.00%)")
    print(f"Mean generations: {mean_gen:.2f} | Mean VP margin (B1 - B0): {mean_margin:+.2f}")
    print(f"Mean decisions/game: {mean_decisions:.1f} | Wall time: {wall_s:.2f}s")
    print("Per map:")
    for map_name in SUPPORTED_MAPS:
        rows = per_map[map_name]
        if rows:
            m_wr = sum(r[0] for r in rows) / len(rows)
            m_gen = statistics.mean(r[1] for r in rows)
            m_mar = statistics.mean(r[2] for r in rows)
            print(f"  {map_name:24s}: n={len(rows):3d} win_rate={m_wr * 100:6.2f}% gen={m_gen:5.2f} margin={m_mar:+6.2f}")
    if failures:
        print("Failures (first 10):")
        for f in failures[:10]:
            print(f"  seed={f.seed_key} map={f.map} type={f.error_type}: {f.error}")

    return 0 if len(failures) == 0 and completed == len(outcomes) and win_rate >= 0.90 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the 200-game B1 vs B0 quick gate.")
    parser.add_argument("--seeds-per-map", type=int, default=25, help="Number of seed pairs per map (default: 25 -> 200 games).")
    parser.add_argument("--workers", type=int, default=8, help="Number of Node worker processes (default: 8).")
    parser.add_argument("--batch-size", type=int, default=32, help="Concurrent active games across pool (default: 32).")
    args = parser.parse_args(argv)
    return run_gate(
        seeds_per_map=args.seeds_per_map,
        num_workers=args.workers,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    raise SystemExit(main())
