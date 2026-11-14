"""Latency benchmark for single-request, batched, and pooled bridge round-trips."""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

from tm_bot.env.client import WorkerClient
from tm_bot.env.pool import WorkerPool
from tm_bot.env.types import SUPPORTED_MAPS, Op


@dataclass(frozen=True, slots=True)
class LatencyReport:
    steps: int
    batch_size: int
    workers: int
    ping_p50_ms: float
    single_step_mean_ms: float
    single_step_p50_ms: float
    single_step_p95_ms: float
    single_observe_mean_ms: float
    batched_batch_step_mean_ms: float
    batched_amortized_step_ms: float
    batched_amortized_observe_ms: float
    pool_batch_step_mean_ms: float
    pool_amortized_step_ms: float


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def measure_bridge_latency(
    *,
    batch_size: int = 16,
    steps: int = 100,
    workers: int = 4,
) -> LatencyReport:
    if batch_size < 1 or steps < 1 or workers < 1:
        raise ValueError("batch_size, steps, and workers must all be >= 1")

    # 1. Single-worker single-request measurements
    with WorkerClient() as client:
        # Warm up V8 JIT and stdio pipes
        warm_gid = client.create("bench-warm", "tharsis")
        for _ in range(5):
            seats = client.pending(warm_gid)
            if not seats:
                break
            client.observe(warm_gid, seats[0])
            client.choice(warm_gid, seats[0])
            client.choose(warm_gid, seats[0], 0)
        client.close_game(warm_gid)

        ping_samples: list[float] = []
        for _ in range(min(steps, 100)):
            t0 = time.perf_counter()
            client.ping()
            ping_samples.append((time.perf_counter() - t0) * 1000.0)

        gid = client.create("bench-single-1", "tharsis")
        step_samples: list[float] = []
        observe_samples: list[float] = []
        recreated = 0
        for i in range(steps):
            t0 = time.perf_counter()
            seats = client.pending(gid)
            if not seats:
                client.close_game(gid)
                recreated += 1
                gid = client.create(f"bench-single-r-{recreated}", "tharsis")
                seats = client.pending(gid)
            seat = seats[0]
            client.choice(gid, seat)
            client.choose(gid, seat, 0)
            step_samples.append((time.perf_counter() - t0) * 1000.0)

            t_obs = time.perf_counter()
            client.observe(gid, seat)
            observe_samples.append((time.perf_counter() - t_obs) * 1000.0)
        client.close_game(gid)

        # 2. Single-worker batched measurements across `batch_size` games
        gids = client.batch(
            [Op.create(f"bench-batch-{i}", SUPPORTED_MAPS[i % len(SUPPORTED_MAPS)]) for i in range(batch_size)]
        )
        batched_step_samples: list[float] = []
        batched_obs_samples: list[float] = []
        for step_idx in range(steps):
            t0 = time.perf_counter()
            pending_lists = client.batch([Op.pending(g) for g in gids])
            active_pairs = [(g, s[0]) for g, s in zip(gids, pending_lists) if s]
            if len(active_pairs) < batch_size:
                # Replace any finished game so batch_size stays constant.
                for idx, (g, s) in enumerate(zip(gids, pending_lists)):
                    if not s:
                        client.close_game(g)
                        gids[idx] = client.create(
                            f"bench-batch-r-{step_idx}-{idx}",
                            SUPPORTED_MAPS[idx % len(SUPPORTED_MAPS)],
                        )
                pending_lists = client.batch([Op.pending(g) for g in gids])
                active_pairs = [(g, s[0]) for g, s in zip(gids, pending_lists)]
            client.batch([Op.choice(g, s) for g, s in active_pairs])
            client.batch([Op.choose(g, s, 0) for g, s in active_pairs])
            batched_step_samples.append((time.perf_counter() - t0) * 1000.0)

            t_obs = time.perf_counter()
            client.batch([Op.observe(g, s) for g, s in active_pairs])
            batched_obs_samples.append((time.perf_counter() - t_obs) * 1000.0)

        client.batch([Op.close(g) for g in gids])

    # 3. WorkerPool batched measurements across `workers` subprocesses
    with WorkerPool(num_workers=workers) as pool:
        pool_gids = pool.batch(
            [Op.create(f"bench-pool-{i}", SUPPORTED_MAPS[i % len(SUPPORTED_MAPS)]) for i in range(batch_size)]
        )
        # Warm up pool workers
        warm_pending = pool.batch([Op.pending(g) for g in pool_gids])
        pool.batch([Op.choice(g, s[0]) for g, s in zip(pool_gids, warm_pending)])

        pool_step_samples: list[float] = []
        for step_idx in range(steps):
            t0 = time.perf_counter()
            pending_lists = pool.batch([Op.pending(g) for g in pool_gids])
            if any(not s for s in pending_lists):
                for idx, (g, s) in enumerate(zip(pool_gids, pending_lists)):
                    if not s:
                        pool.close_game(g)
                        pool_gids[idx] = pool.create(
                            f"bench-pool-r-{step_idx}-{idx}",
                            SUPPORTED_MAPS[idx % len(SUPPORTED_MAPS)],
                        )
                pending_lists = pool.batch([Op.pending(g) for g in pool_gids])
            active_pairs = [(g, s[0]) for g, s in zip(pool_gids, pending_lists)]
            pool.batch([Op.choice(g, s) for g, s in active_pairs])
            pool.batch([Op.choose(g, s, 0) for g, s in active_pairs])
            pool_step_samples.append((time.perf_counter() - t0) * 1000.0)

        pool.batch([Op.close(g) for g in pool_gids])

    batched_mean = statistics.mean(batched_step_samples)
    pool_mean = statistics.mean(pool_step_samples)
    return LatencyReport(
        steps=steps,
        batch_size=batch_size,
        workers=workers,
        ping_p50_ms=statistics.median(ping_samples),
        single_step_mean_ms=statistics.mean(step_samples),
        single_step_p50_ms=statistics.median(step_samples),
        single_step_p95_ms=_percentile(step_samples, 95.0),
        single_observe_mean_ms=statistics.mean(observe_samples),
        batched_batch_step_mean_ms=batched_mean,
        batched_amortized_step_ms=batched_mean / batch_size,
        batched_amortized_observe_ms=statistics.mean(batched_obs_samples) / batch_size,
        pool_batch_step_mean_ms=pool_mean,
        pool_amortized_step_ms=pool_mean / batch_size,
    )


def format_report(report: LatencyReport) -> str:
    return "\n".join(
        [
            f"=== Bridge Latency Benchmark (steps={report.steps}, batch_size={report.batch_size}, workers={report.workers}) ===",
            f"Ping p50 round-trip:          {report.ping_p50_ms:.4f} ms",
            f"Single step (pend+ch+choose): mean={report.single_step_mean_ms:.4f} ms | p50={report.single_step_p50_ms:.4f} ms | p95={report.single_step_p95_ms:.4f} ms",
            f"Single observe:               mean={report.single_observe_mean_ms:.4f} ms",
            f"1-worker batch ({report.batch_size} games):   batch={report.batched_batch_step_mean_ms:.4f} ms | amortized={report.batched_amortized_step_ms:.4f} ms/game-step",
            f"1-worker batch observe:       amortized={report.batched_amortized_observe_ms:.4f} ms/game-obs",
            f"{report.workers}-worker pool ({report.batch_size} games):   batch={report.pool_batch_step_mean_ms:.4f} ms | amortized={report.pool_amortized_step_ms:.4f} ms/game-step",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Python-Node bridge round-trip latency.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    report = measure_bridge_latency(
        batch_size=args.batch_size,
        steps=args.steps,
        workers=args.workers,
    )
    print(format_report(report))


if __name__ == "__main__":
    main()
