"""
MLflow experiment tracker for SNN training sessions.

Passo 10: MLflowTracker wraps the MLflow client and provides:
- Per-episode metric logging with a rolling tempo_vida_medio window
- Model saving with SHA256 checksums logged as MLflow params
- Clean run lifecycle (start / end)
"""
from __future__ import annotations

import hashlib
import os
from collections import deque
from pathlib import Path
from typing import Any

import mlflow
import torch


# ---------------------------------------------------------------------------
# Standalone helper (also exposed for direct use)
# ---------------------------------------------------------------------------


def guardar_modelo_com_checksum(state_dict: dict, path: Path) -> str:
    """
    Save ``state_dict`` to ``path`` and log its SHA256 checksum as an MLflow
    param under the key ``checksum_<filename>``.

    Returns the hex-encoded checksum string.
    """
    torch.save(state_dict, path)
    checksum = hashlib.sha256(open(path, "rb").read()).hexdigest()
    mlflow.log_param(f"checksum_{path.name}", checksum)
    return checksum


# ---------------------------------------------------------------------------
# MLflowTracker
# ---------------------------------------------------------------------------


class MLflowTracker:
    """
    Tracks a single SNN training experiment in MLflow.

    Lifecycle::

        tracker = MLflowTracker(treino_id, parametros)

        for episodio in range(num_episodios):
            ...
            tracker.log_episodio(episodio, metricas)

        tracker.log_modelo(snn.state_dict(), kan.state_dict(), episodio=num_episodios)
        tracker.end()

    Environment variables used:
        MLFLOW_TRACKING_URI       — MLflow server URL (default: http://localhost:5000)
        MLFLOW_EXPERIMENT_NAME    — experiment name    (default: snn-organismo)
    """

    def __init__(self, treino_id: str, parametros: dict) -> None:
        self._treino_id = treino_id

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "snn-organismo")

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        self._run = mlflow.start_run(run_name=treino_id)

        # Log all hyperparameters (nested dicts are flattened with dot notation)
        flat_params = self._flatten_dict(parametros)
        mlflow.log_params(flat_params)

        # Rolling buffer for tempo_vida_medio (last 10 episodes)
        self._vida_buffer: deque[float] = deque(maxlen=10)

    # ------------------------------------------------------------------
    @property
    def mlflow_run_id(self) -> str:
        """The MLflow run ID for this training session."""
        return self._run.info.run_id

    # ------------------------------------------------------------------
    def log_episodio(self, episodio: int, metricas: dict) -> None:
        """
        Log per-episode metrics at the given step.

        Expected keys in ``metricas`` (all optional — missing values default to 0):
            tempo_vida        — episode length in steps (used to compute rolling average)
            eficiencia        — food items eaten / total energy spent
            taxa_inferencia   — fraction of steps taken in inference mode (0–1)
            novidade_media    — mean AGMP novelty score for the episode
            recompensa_total  — sum of rewards accumulated this episode
            fase_curriculum   — current curriculum phase (integer, logged as float)

        Logged metric names:
            tempo_vida_medio, eficiencia, taxa_inferencia, novidade_media,
            recompensa_total, fase_curriculum
        """
        if "tempo_vida" in metricas:
            self._vida_buffer.append(float(metricas["tempo_vida"]))

        tempo_vida_medio = (
            sum(self._vida_buffer) / len(self._vida_buffer)
            if self._vida_buffer
            else 0.0
        )

        metrics_to_log: dict[str, float] = {
            "tempo_vida_medio": tempo_vida_medio,
            "eficiencia": float(metricas.get("eficiencia", 0.0)),
            "taxa_inferencia": float(metricas.get("taxa_inferencia", 0.0)),
            "novidade_media": float(metricas.get("novidade_media", 0.0)),
            "recompensa_total": float(metricas.get("recompensa_total", 0.0)),
            "fase_curriculum": float(metricas.get("fase_curriculum", 1.0)),
        }

        mlflow.log_metrics(metrics_to_log, step=episodio)

    # ------------------------------------------------------------------
    def log_modelo(
        self,
        snn_state_dict: dict,
        kan_state_dict: dict | None = None,
        episodio: int | None = None,
        path_prefix: str | None = None,
    ) -> dict[str, str]:
        """
        Save SNN (and optionally KAN) model weights to disk with SHA256 checksums.

        File paths and checksums are recorded as MLflow params so that any
        checkpoint can be reproduced exactly from a given run.

        Args:
            snn_state_dict: ``torch.nn.Module.state_dict()`` of the SNN.
            kan_state_dict: ``torch.nn.Module.state_dict()`` of the KAN (optional).
            episodio:       Current episode number, appended to filename when given.
            path_prefix:    Root directory for model artefacts.

        Returns:
            Dict with ``path_snn``, ``checksum_snn`` and (if KAN provided)
            ``path_kan``, ``checksum_kan``.
        """
        import os
        if path_prefix is None:
            path_prefix = os.getenv("MLFLOW_ARTIFACT_ROOT", "./mlartifacts")
        output_dir = Path(path_prefix) / self._treino_id
        output_dir.mkdir(parents=True, exist_ok=True)

        suffix = f"_ep{episodio}" if episodio is not None else ""
        result: dict[str, str] = {}

        # SNN
        snn_path = output_dir / f"snn{suffix}.pt"
        snn_checksum = guardar_modelo_com_checksum(snn_state_dict, snn_path)
        mlflow.log_param(f"path_snn{suffix}", str(snn_path))
        result["path_snn"] = str(snn_path)
        result["checksum_snn"] = snn_checksum

        # KAN (optional)
        if kan_state_dict is not None:
            kan_path = output_dir / f"kan{suffix}.pt"
            kan_checksum = guardar_modelo_com_checksum(kan_state_dict, kan_path)
            mlflow.log_param(f"path_kan{suffix}", str(kan_path))
            result["path_kan"] = str(kan_path)
            result["checksum_kan"] = kan_checksum

        return result

    # ------------------------------------------------------------------
    def end(self) -> None:
        """End the active MLflow run."""
        mlflow.end_run()

    # ------------------------------------------------------------------
    @staticmethod
    def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict[str, str]:
        """
        Recursively flatten a nested dict into dot-notation keys suitable for
        ``mlflow.log_params``.

        Example::
            {"snn": {"camadas": 3}} -> {"snn.camadas": "3"}
        """
        items: dict[str, str] = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            if isinstance(v, dict):
                items.update(MLflowTracker._flatten_dict(v, new_key, sep))
            else:
                items[new_key] = str(v)
        return items
