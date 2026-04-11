"""
DBWriter — single shared DB connection com queue para escrita assíncrona.

Evita o limite de max_connections do PostgreSQL quando 100 WorldRunners tendem a
abrir 100 conexões simultâneas.

Uso:
    writer = DBWriter(db_url)
    writer.start()
    writer.enqueue_many(sql, rows)   # non-blocking
    writer.execute_one(sql, params)  # non-blocking
    writer.stop()                    # aguarda flush e fecha
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Optional


# Sentinel para parar o worker
_STOP = object()


class DBWriter:
    """
    Faz todos os writes numa única thread com uma única conexão psycopg2.

    Os WorldRunners chamam enqueue_many() / execute_one() sem bloquear —
    os dados são postos numa queue e escritos em background.
    """

    def __init__(self, db_url: str, maxsize: int = 50_000):
        self._url   = db_url
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._conn  = None
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="DBWriter"
        )
        self._started = False

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Tenta ligar à DB e arranca a thread de escrita. Retorna True se OK."""
        if not self._url:
            return False
        try:
            import psycopg2
            self._conn = psycopg2.connect(self._url)
            self._conn.autocommit = True
            self._started = True
            self._thread.start()
            return True
        except Exception as e:
            print(f"[DBWriter] Conexão falhou: {e}")
            return False

    def stop(self, timeout: float = 30.0) -> None:
        """Envia sinal de paragem e aguarda flush (até timeout segundos)."""
        if not self._started:
            return
        self._queue.put(_STOP)
        self._thread.join(timeout=timeout)

    @property
    def is_active(self) -> bool:
        return self._started and self._conn is not None

    # ------------------------------------------------------------------
    # API pública (thread-safe)
    # ------------------------------------------------------------------

    def enqueue_many(self, sql: str, rows: list) -> None:
        """Encafila um executemany(sql, rows). Non-blocking."""
        if not self._started or not rows:
            return
        try:
            self._queue.put_nowait(("many", sql, rows))
        except queue.Full:
            pass  # drop on overflow — métricas são melhor-esforço

    def execute_one(self, sql: str, params: tuple) -> None:
        """Encafila um execute(sql, params). Non-blocking."""
        if not self._started:
            return
        try:
            self._queue.put_nowait(("one", sql, params))
        except queue.Full:
            pass

    # ------------------------------------------------------------------
    # Worker privado
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is _STOP:
                # Drenar a queue antes de fechar
                self._drain()
                break

            self._process(item)

        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass

    def _drain(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
                if item is _STOP:
                    break
                self._process(item)
            except queue.Empty:
                break

    def _process(self, item) -> None:
        kind = item[0]
        try:
            with self._conn.cursor() as cur:
                if kind == "many":
                    _, sql, rows = item
                    cur.executemany(sql, rows)
                elif kind == "one":
                    _, sql, params = item
                    cur.execute(sql, params)
        except Exception as e:
            print(f"[DBWriter] Erro SQL: {e}")
            # Tentar reconectar uma vez
            try:
                import psycopg2
                self._conn = psycopg2.connect(self._url)
                self._conn.autocommit = True
            except Exception:
                pass
