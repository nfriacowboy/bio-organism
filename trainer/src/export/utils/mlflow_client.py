"""
utils/mlflow_client.py — Cliente MLflow reutilizável para o módulo de exportação.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient


def _client() -> MlflowClient:
    uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(uri)
    return MlflowClient()


def get_run_id(treino_id: str) -> str | None:
    """
    Devolve o MLflow run_id associado ao treino_id.
    Procura por run_name igual a treino_id no experiment snn-organismo.
    Devolve None se não encontrado.
    """
    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "snn-organismo")
    try:
        client = _client()
        exp = client.get_experiment_by_name(exp_name)
        if exp is None:
            return None
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.mlflow.runName = '{treino_id}'",
            max_results=1,
        )
        if runs:
            return runs[0].info.run_id
    except Exception:
        pass
    return None


def download_artifact(run_id: str, artifact_path: str, local_dest: Path) -> Path | None:
    """
    Descarrega um artefacto MLflow para local_dest.
    Devolve o Path local se sucesso, None caso contrário.
    """
    try:
        client = _client()
        local_dest.mkdir(parents=True, exist_ok=True)
        downloaded = client.download_artifacts(run_id, artifact_path, str(local_dest))
        return Path(downloaded)
    except Exception:
        return None


def get_metricas_mlflow(run_id: str) -> dict[str, list[float]]:
    """
    Devolve todas as métricas registadas num run como {nome: [valores]}.
    """
    try:
        client = _client()
        run = client.get_run(run_id)
        result: dict[str, list[float]] = {}
        for k, v in run.data.metrics.items():
            hist = client.get_metric_history(run_id, k)
            result[k] = [m.value for m in hist]
        return result
    except Exception:
        return {}
