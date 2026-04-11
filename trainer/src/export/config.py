"""
config.py — Configuração do módulo de exportação.

Lê treino_id, liga a DB e define o ficheiro JSON de output.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


class ExportConfig:
    """
    Centraliza toda a configuração de uma sessão de exportação.

    Parâmetros
    ----------
    treino_id :
        ID ISO8601 do treino a exportar.
    output_dir :
        Pasta de output (default: ``exports/``).
    episodios_amostra :
        Quantos episódios de referência incluir nos logs de comportamento.
    max_steps_por_episodio :
        Limite de steps por episódio (evita JSONs gigantes).
    """

    def __init__(
        self,
        treino_id: str,
        output_dir: str = "exports",
        episodios_amostra: int = 5,
        max_steps_por_episodio: int = 50_000,
        # parâmetros legados ignorados (mantidos para compatibilidade de chamadas)
        apenas: object = None,
        incluir_modelos: bool = False,
    ) -> None:
        self.treino_id = treino_id
        self.episodios_amostra = episodios_amostra
        self.max_steps_por_episodio = max_steps_por_episodio

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        db_url = os.getenv("DATABASE_URL", "")
        if not db_url:
            raise EnvironmentError(
                "DATABASE_URL não definida. "
                "Defina a variável de ambiente antes de correr o exportador."
            )
        self.engine: Engine = create_engine(db_url, pool_pre_ping=True)
        with self.engine.connect() as c:
            c.execute(text("SELECT 1"))

    @property
    def treino_id_sanitizado(self) -> str:
        """Converte ISO8601 para nome de ficheiro seguro: 20260315T134709UTC"""
        try:
            dt = datetime.fromisoformat(self.treino_id)
            return dt.strftime("%Y%m%dT%H%M%S") + "UTC"
        except ValueError:
            return self.treino_id.replace(":", "").replace("-", "").replace("+", "")

    @property
    def output_file(self) -> Path:
        """Caminho completo para o ficheiro JSON de output."""
        return self.output_dir / f"{self.treino_id_sanitizado}_export.json"
