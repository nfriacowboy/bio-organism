"""
ExperimentManager — rastreabilidade científica para simulações multi-organismo.

Toda a simulação deve referenciar:
  - Uma baseline (SNN_IS_BASELINE=1) — esta simulação TORNA-SE uma baseline
  - Um experimento (SNN_EXPERIMENT_FILE=path/to/exp.json) — variação controlada
  - Se nenhum: regista como 'undocumented_run' (trigger DB também prefija nota)

Uso no trainer:
    mgr = ExperimentManager(DATABASE_URL)
    ids = mgr.on_start(simulacao_id, configs_snapshot)
    # → ids = {"experimento_id": ..., "baseline_id": ...} ou {}

    # ... correr simulação ...

    mgr.on_complete(simulacao_id, metricas_finais)
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Variáveis de ambiente lidas pelo manager
IS_BASELINE     = os.getenv("SNN_IS_BASELINE",    "0") == "1"
BASELINE_NOME   = os.getenv("SNN_BASELINE_NOME",  "")
BASELINE_TIPO   = os.getenv("SNN_BASELINE_TIPO",  "experiment_control")
EXPERIMENT_FILE = os.getenv("SNN_EXPERIMENT_FILE", "")


class ExperimentManager:
    """
    Gere o ciclo de vida de baseline/experimento em torno de uma simulação.
    Opera directamente com psycopg2 (conexão dedicada, fora do DBWriter async).
    """

    def __init__(self, db_url: str):
        self._url = db_url
        self._conn = None
        self._mode = "none"          # 'baseline' | 'experiment' | 'undocumented'
        self._exp_id: Optional[str] = None
        self._baseline_id: Optional[str] = None
        self._criterios: list = []

    # ------------------------------------------------------------------
    # on_start — chamado ANTES de inserir na tabela simulacoes
    # ------------------------------------------------------------------

    def on_start(self, simulacao_id: str, config_snapshot: dict) -> dict:
        """
        Prepara o registo de baseline ou experimento.
        Retorna dict com chaves para incluir no INSERT INTO simulacoes:
            {"experimento_id": <uuid|None>, "baseline_id": <uuid|None>}
        """
        if not self._url:
            return {"experimento_id": None, "baseline_id": None}

        try:
            import psycopg2
            import psycopg2.extras
            self._conn = psycopg2.connect(self._url)
            self._conn.autocommit = True
        except Exception as e:
            logger.warning(f"[ExperimentManager] Não foi possível ligar à DB: {e}")
            return {"experimento_id": None, "baseline_id": None}

        if IS_BASELINE:
            return self._start_as_baseline(simulacao_id, config_snapshot)
        elif EXPERIMENT_FILE:
            return self._start_as_experiment(simulacao_id, config_snapshot)
        else:
            self._mode = "undocumented"
            logger.warning(
                "\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "  ⚠  SIMULAÇÃO SEM BASELINE OU EXPERIMENTO DECLARADO\n"
                "  Registada como UNDOCUMENTED_RUN.\n"
                "  Para disciplina científica:\n"
                "    SNN_IS_BASELINE=1 SNN_BASELINE_NOME='nome'  → nova baseline\n"
                "    SNN_EXPERIMENT_FILE=experiments/exp.json    → variação controlada\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
            return {"experimento_id": None, "baseline_id": None}

    # ------------------------------------------------------------------
    # on_complete — chamado após a simulação terminar
    # ------------------------------------------------------------------

    def on_complete(self, simulacao_id: str, metricas_finais: dict) -> None:
        """
        Verifica critérios de validação e actualiza o registo de experimento.
        Para baseline, grava as métricas de referência.
        """
        if not self._conn:
            return

        try:
            if self._mode == "baseline" and self._baseline_id:
                self._conn.cursor().execute(
                    "UPDATE baselines SET metricas_referencia = %s WHERE baseline_id = %s",
                    (json.dumps(metricas_finais), self._baseline_id)
                )
                logger.info(f"[ExperimentManager] Baseline {self._baseline_id} finalizado.")

            elif self._mode == "experiment" and self._exp_id:
                atingidos, relatorio = self._check_criterios(metricas_finais)
                with self._conn.cursor() as cur:
                    cur.execute("""
                        UPDATE experimentos
                        SET estado             = 'concluido',
                            concluido_em       = %s,
                            simulacao_id       = %s,
                            criterios_atingidos = %s,
                            resultado_real     = %s
                        WHERE experimento_id = %s
                    """, (
                        datetime.now(timezone.utc),
                        simulacao_id,
                        atingidos,
                        json.dumps(metricas_finais),
                        self._exp_id
                    ))
                self._log_validation_report(relatorio, atingidos)
        except Exception as e:
            logger.warning(f"[ExperimentManager] on_complete falhou: {e}")
        finally:
            self._conn.close()

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _start_as_baseline(self, simulacao_id: str, config_snapshot: dict) -> dict:
        nome = BASELINE_NOME or f"baseline_{simulacao_id}"
        bid = str(uuid.uuid4())
        try:
            with self._conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO baselines
                        (baseline_id, nome, tipo, simulacao_id, config_snapshot)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (nome) DO UPDATE
                        SET simulacao_id   = EXCLUDED.simulacao_id,
                            config_snapshot = EXCLUDED.config_snapshot
                    RETURNING baseline_id
                """, (bid, nome, BASELINE_TIPO, simulacao_id, json.dumps(config_snapshot)))
                bid = cur.fetchone()[0]
        except Exception as e:
            logger.warning(f"[ExperimentManager] Falha ao criar baseline: {e}")
            return {"experimento_id": None, "baseline_id": None}

        self._mode = "baseline"
        self._baseline_id = str(bid)
        logger.info(f"[ExperimentManager] Baseline criado: {nome} ({bid})")
        return {"experimento_id": None, "baseline_id": str(bid)}

    def _start_as_experiment(self, simulacao_id: str, config_snapshot: dict) -> dict:
        try:
            with open(EXPERIMENT_FILE, "r") as f:
                exp = json.load(f)
        except Exception as e:
            logger.warning(f"[ExperimentManager] Não foi possível ler {EXPERIMENT_FILE}: {e}")
            self._mode = "undocumented"
            return {"experimento_id": None, "baseline_id": None}

        # Validar que a baseline existe
        baseline_id = exp.get("baseline_id")
        if not baseline_id:
            logger.warning("[ExperimentManager] SNN_EXPERIMENT_FILE sem baseline_id. Undocumented.")
            self._mode = "undocumented"
            return {"experimento_id": None, "baseline_id": None}

        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT 1 FROM baselines WHERE baseline_id = %s", (baseline_id,))
                if not cur.fetchone():
                    logger.warning(f"[ExperimentManager] baseline_id {baseline_id} não existe na DB.")
                    self._mode = "undocumented"
                    return {"experimento_id": None, "baseline_id": None}
        except Exception as e:
            logger.warning(f"[ExperimentManager] Erro ao verificar baseline: {e}")
            self._mode = "undocumented"
            return {"experimento_id": None, "baseline_id": None}

        eid = str(uuid.uuid4())
        try:
            with self._conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO experimentos (
                        experimento_id, baseline_id, simulacao_id, nome,
                        sweep_variable, sweep_values, n_replicas_per_value,
                        parametros_alterados, justificacao, resultado_esperado,
                        criterios_validacao, estado, iniciado_em
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'a_correr',%s)
                """, (
                    eid,
                    baseline_id,
                    simulacao_id,
                    exp.get("nome", ""),
                    exp.get("sweep_variable"),
                    json.dumps(exp.get("sweep_values")) if exp.get("sweep_values") else None,
                    exp.get("n_replicas_per_value"),
                    json.dumps(exp.get("parametros_alterados", {})),
                    exp["justificacao"],
                    exp["resultado_esperado"],
                    json.dumps(exp.get("criterios_validacao", [])),
                    datetime.now(timezone.utc),
                ))
        except Exception as e:
            logger.warning(f"[ExperimentManager] Falha ao criar experimento: {e}")
            self._mode = "undocumented"
            return {"experimento_id": None, "baseline_id": None}

        self._mode = "experiment"
        self._exp_id = eid
        self._baseline_id = baseline_id
        self._criterios = exp.get("criterios_validacao", [])
        logger.info(f"[ExperimentManager] Experimento iniciado: {exp.get('nome', eid)} ({eid})")
        return {"experimento_id": eid, "baseline_id": baseline_id}

    def _check_criterios(self, metricas: dict) -> tuple[bool, list]:
        OPERADORES = {">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
                      "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
                      "==": lambda a, b: a == b}
        relatorio = []
        todos_ok = True
        for c in self._criterios:
            metrica = c.get("metrica", "")
            op_str  = c.get("op", ">")
            valor   = c.get("valor", 0)
            real    = metricas.get(metrica)
            if real is None:
                resultado = False
                todos_ok  = False
                relatorio.append({"metrica": metrica, "ok": False, "real": None,
                                   "esperado": f"{op_str} {valor}", "nota": "não disponível"})
                continue
            op_fn = OPERADORES.get(op_str, lambda a, b: False)
            ok = op_fn(float(real), float(valor))
            if not ok:
                todos_ok = False
            relatorio.append({"metrica": metrica, "ok": ok, "real": real,
                               "esperado": f"{op_str} {valor}"})
        return todos_ok, relatorio

    def _log_validation_report(self, relatorio: list, todos_ok: bool) -> None:
        status = "✓ PASS" if todos_ok else "✗ FAIL"
        logger.info(f"\n[ExperimentManager] Validação de critérios: {status}")
        for r in relatorio:
            icon = "✓" if r["ok"] else "✗"
            real = f"{r['real']:.4f}" if isinstance(r["real"], float) else str(r["real"])
            logger.info(f"  {icon} {r['metrica']}: {real} (esperado {r['esperado']})")
        if not todos_ok:
            logger.warning("[ExperimentManager] Critérios NÃO atingidos. "
                           "Actualizar analise_conclusao em DB manualmente.")
