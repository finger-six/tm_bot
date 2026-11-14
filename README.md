# Terraforming Mars AI (`tm_bot`)

Local research repository for training and evaluating two-player Terraforming Mars agents (see [`CONTEXT.md`](CONTEXT.md) for domain vocabulary and format constraints).

## Prerequisites

- **Conda** (Miniconda / Anaconda) — Python runs exclusively via the `tm_bot` conda environment (there is no system Python dependency).
- **Node.js 24** (`node`, `npm`) for building and running the pinned Terraforming Mars engine and TypeScript adapter/worker.
- **NVIDIA GPU + CUDA driver** (tested on RTX 4080 SUPER with CUDA 13.x driver and PyTorch `cu130` wheels).

## 1. Conda Environment Setup

Create the `tm_bot` environment from [`environment.yml`](environment.yml):

```bash
conda env create -f environment.yml
```

Or update an existing `tm_bot` environment:

```bash
conda env update -n tm_bot -f environment.yml
```

Verify PyTorch CUDA support:

```bash
conda run -n tm_bot python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## 2. Engine Submodule Build

Initialize and build the pinned Terraforming Mars engine under `tools/simbench/engine` (do not modify engine source):

```bash
git submodule update --init tools/simbench/engine
cd tools/simbench/engine
npm ci --no-audit --no-fund
npm run make:json
npm run build:server
cd ../../..
```

## 3. Adapter & Node Worker Build

Install dependencies, compile TypeScript (including the stdio JSON worker `adapter/dist/src/worker.js`), and run the Node test suite:

```bash
cd adapter
npm install --no-audit --no-fund
npm run build
npm test
cd ..
```

## 4. Running Python Tests

Run the full Python test suite (bridge + data tests):

```bash
conda run -n tm_bot pytest
```

## 5. Bridge Usage (`tm_bot.env`)

`tm_bot.env` provides a Python client (`WorkerClient`), single-game handle (`GameSession`), batched request builders (`Op`), and multi-process worker pool (`WorkerPool`) over the Node worker:

```python
from tm_bot.env import Op, WorkerClient, WorkerPool

# Single worker with GameSession
with WorkerClient() as client:
    with client.open(seed_key="train-1", map="tharsis") as game:
        while seats := game.pending():
            seat = seats[0]
            obs = game.observe(seat)
            choice = game.choice(seat)
            game.choose(seat, 0)
        print(game.result())

# Multi-worker pool with batched requests across games
with WorkerPool(num_workers=4) as pool:
    game_ids = pool.batch([
        Op.create("train-10", "tharsis"),
        Op.create("train-11", "elysium"),
    ])
    pending = pool.batch([Op.pending(gid) for gid in game_ids])
```

Measure single-request, batched, and pooled bridge round-trip latency:

```bash
conda run -n tm_bot python -m tm_bot.env.bench --batch-size 16 --steps 100 --workers 4
```
