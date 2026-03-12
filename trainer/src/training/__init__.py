"""
training package — SNN training utilities.

Exports:
    StepLogger      — buffers per-step observations and bulk-inserts into steps_log
    TreinoManager   — manages treino session rows in the treinos table
    MLflowTracker   — logs metrics, params and model artefacts to MLflow

Imports are lazy (try/except) so the package loads cleanly even when
psycopg2 or mlflow are not installed in the local environment.
"""
from __future__ import annotations

try:
    from .logger import StepLogger, TreinoManager
    _HAS_DB = True
except ImportError:
    StepLogger = None     # type: ignore[assignment,misc]
    TreinoManager = None  # type: ignore[assignment,misc]
    _HAS_DB = False

try:
    from .mlflow_tracker import MLflowTracker
    _HAS_MLFLOW = True
except ImportError:
    MLflowTracker = None  # type: ignore[assignment,misc]
    _HAS_MLFLOW = False

__all__ = [
    "StepLogger",
    "TreinoManager",
    "MLflowTracker",
    "_HAS_DB",
    "_HAS_MLFLOW",
]
