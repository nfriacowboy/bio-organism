# bio-organism

> Emergent social behaviour from biologically plausible neural learning rules.

Artificial organisms equipped with **Spiking Neural Networks (SNN)** learn to navigate sparse environments using olfactory gradients, then self-organise into competing tribes with dominance hierarchies, cross-territorial mating, and alpha dynamics — all without any explicit social programming.

---

## Key Results

| Metric | Result |
|---|---|
| Navigation vs. random baseline (Phase 3, 0.15% food density) | **23× above random** |
| Generations reached in 40-minute simulation (10 worlds) | **56 generations** |
| Cross-territorial mating (6-patch worlds) | **38% of all births** |
| Organisms produced in a single 40-min run | **5 819 lifetimes** |
| Genetic convergence (cv_fc1, init → final) | **11.6 → 1.1** |
| Tribal clusters per world (DBSCAN) | **4–5 stable territories** |

---

## What It Does

```
Single organism                    Multi-organism world
─────────────────                  ────────────────────────────────────────────
Inputs (45)                        World 1    World 2  ...  World N
  ├─ 15 visual (ray-cast)            │          │               │
  ├─  3 olfactory (1/d² weighted)    ▼          ▼               ▼
  ├─  3 pheromone (trail sensing)  ┌─────────────────────────────────┐
  ├─  4 efferent (action history)  │  GPU Inference Server (batched) │
  ├─ 15 flow / obstacle            │  batched_forward_multi_snn()    │
  ├─  2 proprioceptive             └─────────────────────────────────┘
  └─  3 internal state                       │
        │                                    ▼
        ▼                          PostgreSQL (async writes)
  H1 LIF (64) → H2 RLIF (32) → Output LIF (4)   Dashboard (Streamlit)
  [recurrent lateral connections]
        │
  R-STDP + novelty gate
  (learning ↔ inference mode)
        │
  KAN (exploration/exploitation bias)
        ▼
  avancar | virar_esq | virar_dir | parar
```

### Learning

Each organism uses **Reward-modulated STDP (R-STDP)** with a novelty gate:
- **Learning mode** — R-STDP active, epsilon high, weights update
- **Inference mode** — R-STDP frozen, epsilon low, exploiting learned policy

The gate switches on CV (coefficient of variation) of reward over a rolling window. No fixed learning schedule — the organism decides when it has learned enough.

### Social Dynamics

In multi-organism worlds each territory (food patch) develops an **alpha** — the organism with the highest food-rate (food/step). Alphas:
- bypass mating cooldown within their territory
- attempt **cross-territorial excursions** (P=0.20/check) — fight the rival alpha, and if victorious mate cross-territorially, placing offspring at the neutral midpoint between patches

This seeds new tribal lineages. Displacement of the weakest organism allows population to exceed `MAX_POP` only via dominance, not random replacement.

---

## Quick Start

### Prerequisites

- Docker + Docker Compose
- Python 3.12+ with [uv](https://github.com/astral-sh/uv)
- (Optional) ROCm 6.3+ or CUDA for GPU acceleration

### 1. Infrastructure

```bash
cp .env.example .env
# edit .env — set POSTGRES_PASSWORD and API_SECRET_KEY

docker compose up -d snn-db snn-mlflow
```

### 2. Install Python dependencies

```bash
cd trainer
uv sync
```

### 3. Train a single organism

```bash
# Runs the 3-phase curriculum: 200×200 → 500×500 → 1000×1000
cd trainer
PYTHONPATH=src PYTHONUNBUFFERED=1 \
  SNN_N_EPISODIOS=2000 \
  SNN_DEVICE=cpu \
  DATABASE_URL="postgresql://snn_user:changeme@localhost:5432/snn_organismo" \
  uv run --no-sync python3 -m training.trainer
```

### 4. Run the multi-world simulation

```bash
# Default: 20 worlds, n_patches sweep [2,3,4,6], 2 hours, GPU
bash launch_multi_world.sh

# Quick validation (5 worlds, 10 minutes)
SNN_N_MUNDOS=5 SNN_FIXED_N_PATCHES=3 SNN_MAX_TEMPO_S=600 bash launch_multi_world.sh
```

### 5. Dashboard

```bash
bash launch_dashboard.sh
# → http://localhost:8501
```

---

## Repository Structure

```
bio-organism/
├── trainer/
│   └── src/
│       ├── snn/              # Neural network: network.py, learning.py, encoding.py, kan_policy.py
│       ├── world/            # Simulation: environment.py, patches.py, olfato.py, feromona.py,
│       │                     #             reproduction.py, predador.py, world_async.py
│       ├── training/         # Training loops: trainer.py, world_runner.py, trainer_multi_world.py,
│       │                     #                 inference_server.py, db_writer.py
│       └── export/           # Result analysis and JSON export
├── dashboard/                # Streamlit dashboard (live 2D canvas + metrics)
├── db/
│   └── migrations/           # PostgreSQL schema (001–009)
├── documentation/            # Architecture docs, training philosophy, full report
│   └── bio_organism_relatorio.md   ← start here for a full project overview
├── tasks/                    # Research roadmap
├── docker-compose.yml        # PostgreSQL + MLflow + optional API/web
├── launch_multi_world.sh     # Main entry point for multi-world runs
└── launch_dashboard.sh       # Start the Streamlit dashboard
```

---

## Environment Variables

Copy `.env.example` to `.env` and adjust. Key variables for the trainer:

| Variable | Default | Description |
|---|---|---|
| `SNN_N_MUNDOS` | `20` | Number of parallel worlds |
| `SNN_WORLD_SIZE` | `500` | World grid size (≥500 for tribal emergence with 5 patches) |
| `SNN_VIDA_MAX` | `300` | Max organism lifespan in steps |
| `SNN_FIXED_N_PATCHES` | `0` | 0 = sweep [2,3,4,6]; >0 = fixed value |
| `SNN_MAX_TEMPO_S` | `7200` | Simulation wall-clock limit |
| `SNN_DEVICE` | `cuda` | `cpu` or `cuda` (CPU is faster for batch=1) |
| `SNN_BASE_SNN_PATH` | — | Path to `.pt` checkpoint for initial weights |
| `DATABASE_URL` | — | PostgreSQL connection string |

---

## Scientific Background

The project tests whether complex social behaviours emerge from minimal physical rules:

| Behaviour | Mechanism | Reference |
|---|---|---|
| Territorial dominance | Alpha = max(food_rate) per patch | Sapolsky (2005) |
| Cross-territorial gene flow | Alpha excursions P=0.20 | Watts & Mitani (2001) |
| Inbreeding avoidance | cosine_sim(SNN_a, SNN_b) < 0.95 | Charlesworth & Charlesworth (1987) |
| Olfactory chemotaxis | R-STDP + 1/d² gradient | Bhattacharya et al. (2014) |
| Novelty-gated plasticity | CV-based learning/inference switch | Frémaux & Gerstner (2016) |

See [`documentation/bio_organism_relatorio.md`](documentation/bio_organism_relatorio.md) for the full project report, training history, and simulation results.

---

## GPU Note

Training is **CPU-faster** for single organisms (batch=1): CPU ~1.7ms/step vs GPU ~5.4ms/step on an AMD Radeon AI PRO R9700 (ROCm 6.3). GPU batching becomes advantageous at N≥8 organisms via `batched_forward_multi_snn()`.

---

## License

MIT — see [LICENSE](LICENSE).
