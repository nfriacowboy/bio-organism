"""
PostgreSQL structured logger for SNN training sessions.

Passo 9: StepLogger (per-step bulk insert) + TreinoManager (treino lifecycle).

Design notes:
- StepLogger buffers rows in memory and bulk-inserts via psycopg2 execute_values.
  Individual per-step INSERTs are never used.
- Auto-flush fires every flush_interval rows (default 100).
- TreinoManager uses SQLAlchemy text queries for higher-level treino CRUD.
- Both classes accept DATABASE_URL from the environment when not provided explicitly.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any

import psycopg2
import psycopg2.extras
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_FLUSH_INTERVAL = 100

_INSERT_STEPS_SQL = """
    INSERT INTO steps_log (
        treino_id, episodio, step, timestamp_ms,
        pos_x, pos_y, angulo,
        energia_actual, energia_max, visao,
        accao, modo, novidade, recompensa,
        bias_explorar, bias_exploitar, comeu, fase_mundo
    ) VALUES %s
"""


# ---------------------------------------------------------------------------
# StepLogger
# ---------------------------------------------------------------------------


class StepLogger:
    """
    Buffers per-step rows and bulk-inserts them into steps_log.

    Usage::

        logger = StepLogger(treino_id="2025-01-01T12:00:00+00:00")
        for step, (obs, accao, ...) in enumerate(episode):
            logger.log_step(episodio, step, obs, accao, modo, novidade, recompensa)
        logger.close()  # flushes remaining rows and closes the connection
    """

    flush_interval: int = _FLUSH_INTERVAL

    def __init__(self, treino_id: str, database_url: str | None = None) -> None:
        self.treino_id = treino_id
        url = database_url or os.environ.get("DATABASE_URL")
        if not url:
            raise ValueError("DATABASE_URL must be set in the environment or passed explicitly.")

        try:
            self._engine = create_engine(url, pool_pre_ping=True)
            # Obtain a raw psycopg2 connection for execute_values performance
            self._conn: psycopg2.extensions.connection = self._engine.raw_connection()
        except Exception as exc:
            raise ConnectionError(
                f"StepLogger: cannot connect to database — {exc}"
            ) from exc

        self._buffer: list[tuple[Any, ...]] = []

    # ------------------------------------------------------------------
    def log_step(
        self,
        episodio: int,
        step: int,
        obs: dict,
        accao: str,
        modo: str,
        novidade: float | None,
        recompensa: float,
        bias_explorar: float | None = None,
        bias_exploitar: float | None = None,
    ) -> None:
        """
        Append a step row to the internal buffer.

        ``obs`` is the dict returned by ``Mundo._observacao()`` (and optionally
        extended with ``comeu=True/False`` by the training loop).

        Auto-flushes to the database every ``flush_interval`` rows.
        """
        visao_json = psycopg2.extras.Json(obs["visao"])

        row: tuple[Any, ...] = (
            self.treino_id,                                             # treino_id
            int(episodio),                                             # episodio
            int(step),                                                 # step
            int(time.time() * 1000),                                   # timestamp_ms
            int(obs["pos_x"]),                                         # pos_x
            int(obs["pos_y"]),                                         # pos_y
            float(obs["angulo"]),                                      # angulo
            float(obs["energia_actual"]),                              # energia_actual
            float(obs["energia_max"]),                                 # energia_max
            visao_json,                                                # visao (JSONB)
            str(accao),                                                # accao
            str(modo),                                                 # modo
            float(novidade) if novidade is not None else None,         # novidade
            float(recompensa),                                         # recompensa
            float(bias_explorar) if bias_explorar is not None else None,  # bias_explorar
            float(bias_exploitar) if bias_exploitar is not None else None, # bias_exploitar
            bool(obs.get("comeu", False)),                             # comeu
            int(obs.get("fase", obs.get("fase_mundo", 1))),            # fase_mundo
        )

        self._buffer.append(row)

        if len(self._buffer) >= self.flush_interval:
            self.flush()

    # ------------------------------------------------------------------
    def flush(self) -> None:
        """
        Bulk-insert all buffered rows using psycopg2 execute_values.

        This is the only code path that performs INSERTs — individual per-step
        inserts are never used.
        """
        if not self._buffer:
            return

        try:
            with self._conn.cursor() as cur:
                psycopg2.extras.execute_values(cur, _INSERT_STEPS_SQL, self._buffer)
            self._conn.commit()
            self._buffer.clear()
        except psycopg2.Error as exc:
            try:
                self._conn.rollback()
            except Exception:
                pass
            raise RuntimeError(f"StepLogger.flush failed: {exc}") from exc

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Perform a final flush and release the database connection."""
        try:
            self.flush()
        finally:
            try:
                self._conn.close()
            except Exception:
                pass
            try:
                self._engine.dispose()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# TreinoManager
# ---------------------------------------------------------------------------


class TreinoManager:
    """
    Manages training session rows in the ``treinos`` table.

    Responsible for creating new sessions, updating their state, and writing
    final metrics once training completes.
    """

    def __init__(self, database_url: str | None = None) -> None:
        url = database_url or os.environ.get("DATABASE_URL")
        if not url:
            raise ValueError("DATABASE_URL must be set in the environment or passed explicitly.")

        try:
            self._engine = create_engine(url, pool_pre_ping=True)
        except Exception as exc:
            raise ConnectionError(
                f"TreinoManager: cannot connect to database — {exc}"
            ) from exc

    # ------------------------------------------------------------------
    def create_treino(self, parametros: dict, nota: str = "") -> str:
        """
        Insert a new treino row with estado='a_correr'.

        Returns the treino_id, which is an ISO8601 UTC timestamp string used
        as the primary key.
        """
        treino_id = datetime.now(timezone.utc).isoformat()
        data_human = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        try:
            with self._engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO treinos (id, data_human, nota, estado, parametros)
                        VALUES (:id, :data_human, :nota, 'a_correr', CAST(:parametros AS JSONB))
                        """
                    ),
                    {
                        "id": treino_id,
                        "data_human": data_human,
                        "nota": nota,
                        "parametros": json.dumps(parametros),
                    },
                )
        except Exception as exc:
            raise RuntimeError(f"TreinoManager.create_treino failed: {exc}") from exc

        return treino_id

    # ------------------------------------------------------------------
    def update_metricas(self, treino_id: str, metricas_finais: dict) -> None:
        """Persist final metrics into the treinos row."""
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        UPDATE treinos
                        SET metricas_finais = CAST(:metricas AS JSONB)
                        WHERE id = :id
                        """
                    ),
                    {"metricas": json.dumps(metricas_finais), "id": treino_id},
                )
        except Exception as exc:
            raise RuntimeError(f"TreinoManager.update_metricas failed: {exc}") from exc

    # ------------------------------------------------------------------
    def update_estado(self, treino_id: str, estado: str) -> None:
        """
        Update the training session lifecycle state.

        Valid values: 'pendente', 'a_correr', 'concluido', 'erro'
        """
        _VALID_STATES = {"pendente", "a_correr", "concluido", "erro"}
        if estado not in _VALID_STATES:
            raise ValueError(
                f"Invalid estado '{estado}'. Must be one of: {sorted(_VALID_STATES)}"
            )

        try:
            with self._engine.begin() as conn:
                conn.execute(
                    text("UPDATE treinos SET estado = :estado WHERE id = :id"),
                    {"estado": estado, "id": treino_id},
                )
        except Exception as exc:
            raise RuntimeError(f"TreinoManager.update_estado failed: {exc}") from exc


# ---------------------------------------------------------------------------
# EpisodioLogger
# ---------------------------------------------------------------------------

_INSERT_EPISODIO_SQL = text("""
    INSERT INTO episodios_log (
        treino_id, episodio,
        tempo_vida, comida, eficiencia, recompensa_total,
        taxa_inferencia, novidade_media,
        bias_explorar_medio, bias_exploitar_medio,
        fase_curriculum, morreu,
        precisao_direcional, steps_com_visao, taxa_perseguicao,
        seed, p_avancar,
        modo_plasticidade
    ) VALUES (
        :treino_id, :episodio,
        :tempo_vida, :comida, :eficiencia, :recompensa_total,
        :taxa_inferencia, :novidade_media,
        :bias_explorar_medio, :bias_exploitar_medio,
        :fase_curriculum, :morreu,
        :precisao_direcional, :steps_com_visao, :taxa_perseguicao,
        :seed, :p_avancar,
        :modo_plasticidade
    )
    ON CONFLICT (treino_id, episodio) DO UPDATE SET
        tempo_vida            = EXCLUDED.tempo_vida,
        comida                = EXCLUDED.comida,
        eficiencia            = EXCLUDED.eficiencia,
        recompensa_total      = EXCLUDED.recompensa_total,
        taxa_inferencia       = EXCLUDED.taxa_inferencia,
        novidade_media        = EXCLUDED.novidade_media,
        bias_explorar_medio   = EXCLUDED.bias_explorar_medio,
        bias_exploitar_medio  = EXCLUDED.bias_exploitar_medio,
        fase_curriculum       = EXCLUDED.fase_curriculum,
        morreu                = EXCLUDED.morreu,
        precisao_direcional   = EXCLUDED.precisao_direcional,
        steps_com_visao       = EXCLUDED.steps_com_visao,
        taxa_perseguicao      = EXCLUDED.taxa_perseguicao,
        seed                  = EXCLUDED.seed,
        p_avancar             = EXCLUDED.p_avancar,
        modo_plasticidade     = EXCLUDED.modo_plasticidade
""")


class EpisodioLogger:
    """
    Persiste métricas por episódio na tabela episodios_log.

    Garante que precisao_direcional, taxa_perseguicao e todas as métricas
    calculadas pelo trainer ficam na BD (single source of truth).

    Usage::

        logger = EpisodioLogger(treino_id="2026-03-15T...")
        logger.log_episodio(episodio=42, metricas={...})
        logger.close()
    """

    def __init__(self, treino_id: str, database_url: str | None = None) -> None:
        self.treino_id = treino_id
        url = database_url or os.environ.get("DATABASE_URL")
        if not url:
            raise ValueError("DATABASE_URL must be set in the environment or passed explicitly.")
        try:
            self._engine = create_engine(url, pool_pre_ping=True)
        except Exception as exc:
            raise ConnectionError(
                f"EpisodioLogger: cannot connect to database — {exc}"
            ) from exc

    # ------------------------------------------------------------------
    def log_episodio(self, episodio: int, metricas: dict) -> None:
        """
        Persiste as métricas de um episódio.

        Chaves esperadas em ``metricas`` (todas opcionais — default 0/None):
            tempo_vida, comida, eficiencia, recompensa_total,
            taxa_inferencia, novidade_media,
            bias_explorar_medio, bias_exploitar_medio,
            fase_curriculum, morreu,
            precisao_direcional, steps_com_visao, taxa_perseguicao,
            seed
        """
        params = {
            "treino_id":            self.treino_id,
            "episodio":             int(episodio),
            "tempo_vida":           int(metricas.get("tempo_vida", 0)),
            "comida":               int(metricas.get("comida", 0)),
            "eficiencia":           float(metricas.get("eficiencia", 0.0)),
            "recompensa_total":     float(metricas.get("recompensa_total", 0.0)),
            "taxa_inferencia":      float(metricas.get("taxa_inferencia", 0.0)),
            "novidade_media":       float(metricas["novidade_media"]) if metricas.get("novidade_media") is not None else None,
            "bias_explorar_medio":  float(metricas["bias_explorar_medio"]) if metricas.get("bias_explorar_medio") is not None else None,
            "bias_exploitar_medio": float(metricas["bias_exploitar_medio"]) if metricas.get("bias_exploitar_medio") is not None else None,
            "fase_curriculum":      int(metricas.get("fase_curriculum", 1)),
            "morreu":               bool(metricas.get("morreu", False)),
            "precisao_direcional":  float(metricas["precisao_direcional"]) if metricas.get("precisao_direcional") is not None else None,
            "steps_com_visao":      int(metricas.get("steps_com_visao", 0)),
            "taxa_perseguicao":     float(metricas["taxa_perseguicao"]) if metricas.get("taxa_perseguicao") is not None else None,
            "seed":                 int(metricas["seed"]) if metricas.get("seed") is not None else None,
            "p_avancar":            float(metricas["p_avancar"]) if metricas.get("p_avancar") is not None else None,
            "modo_plasticidade":    str(metricas.get("modo_plasticidade", "aprendizagem")),
        }
        try:
            with self._engine.begin() as conn:
                conn.execute(_INSERT_EPISODIO_SQL, params)
        except Exception as exc:
            raise RuntimeError(f"EpisodioLogger.log_episodio failed: {exc}") from exc

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Liberta a ligação à BD."""
        try:
            self._engine.dispose()
        except Exception:
            pass
