"""
config.py — Configuração do módulo de exportação.

Lê treino_id, liga a DB e MLflow, cria directoria de output.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
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
        Pasta raiz de output (default: ``exports/``).
    apenas :
        Lista de exportadores a activar. None = todos.
    episodios_amostra :
        Quantos episódios de referência exportar.
    max_steps_por_episodio :
        Limite de steps por episódio (evita ficheiros gigantes).
    incluir_modelos :
        Se True, copia os .pt dos modelos para o output.
    """

    ALL_EXPORTERS = ("metricas", "comportamento", "snn", "kan", "espacial")

    def __init__(
        self,
        treino_id: str,
        output_dir: str = "exports",
        apenas: list[str] | None = None,
        episodios_amostra: int = 5,
        max_steps_por_episodio: int = 50_000,
        incluir_modelos: bool = False,
    ) -> None:
        self.treino_id = treino_id
        self.episodios_amostra = episodios_amostra
        self.max_steps_por_episodio = max_steps_por_episodio
        self.incluir_modelos = incluir_modelos
        self.apenas = set(apenas) if apenas else set(self.ALL_EXPORTERS)

        # Pasta de output
        base = Path(output_dir) / treino_id
        if base.exists():
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            base = Path(output_dir) / f"{treino_id}_{ts}"
        self.output_dir: Path = base
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Sub-pastas por categoria
        for cat in ("metricas", "comportamento", "snn", "kan", "espacial"):
            (self.output_dir / cat).mkdir(exist_ok=True)

        # SQLAlchemy engine
        db_url = os.getenv("DATABASE_URL", "")
        if not db_url:
            raise EnvironmentError(
                "DATABASE_URL não definida. "
                "Defina a variável de ambiente antes de correr o exportador."
            )
        self.engine: Engine = create_engine(db_url, pool_pre_ping=True)
        # Quick check
        with self.engine.connect() as c:
            c.execute(text("SELECT 1"))

    def cat(self, categoria: str) -> Path:
        """Devolve o path da sub-pasta de uma categoria."""
        return self.output_dir / categoria

    def enabled(self, nome: str) -> bool:
        return nome in self.apenas
