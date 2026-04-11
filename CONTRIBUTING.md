# Contributing

Contributions are welcome. This is a research project in active development — please open an issue before starting significant work so we can coordinate.

## Getting Started

1. Fork the repository and create a branch from `main`
2. Follow the [Quick Start](README.md#quick-start) to set up the environment
3. Make your changes with tests where applicable
4. Open a pull request with a clear description of what and why

## Areas of Interest

- **Performance**: the simulation is CPU-bound per world; multiprocessing or Rust workers would unlock larger sweeps
- **Tribal analysis**: post-hoc DBSCAN, inter-tribe genetic distance, lineage tracing
- **New sensory modalities**: sound, temperature, visual field expansion
- **Ethical emergence (Phase D)**: tragedy-of-the-commons resource dynamics
- **Visualisation**: richer dashboard, SNN weight visualisation, lineage trees

## Code Style

- Python 3.12+, formatted with `ruff` (no strict enforcement yet)
- Keep new modules in the appropriate `trainer/src/` subdirectory
- New DB schema changes go in `db/migrations/` as numbered SQL files

## Reporting Issues

Please include:
- Python and PyTorch versions
- GPU / ROCm / CUDA version if relevant
- Minimal reproduction script and error output
