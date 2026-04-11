"""
validate_experiment.py — Verificar/actualizar critérios de validação pós-run.

Útil para re-computar critérios manualmente, ou para adicionar análise_conclusao.

Uso:
    # Re-verificar critérios com métricas actuais da DB
    python3 -m scripts.validate_experiment --experimento_id <uuid>

    # Adicionar análise manual de conclusão
    python3 -m scripts.validate_experiment \\
        --experimento_id <uuid> \\
        --conclusao "cv_fc1 atingido mas gen_max abaixo do esperado. ..."

    # Listar todos os experimentos
    python3 -m scripts.validate_experiment --list
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone


def _check(criterios: list, metricas: dict) -> tuple[bool, list]:
    OPS = {">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
           "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
           "==": lambda a, b: a == b}
    relatorio = []
    todos_ok = True
    for c in criterios:
        metrica = c.get("metrica", "")
        op_str  = c.get("op", ">")
        valor   = c.get("valor", 0)
        real    = metricas.get(metrica)
        if real is None:
            todos_ok = False
            relatorio.append({"metrica": metrica, "ok": False,
                               "real": None, "esperado": f"{op_str} {valor}"})
            continue
        ok = OPS.get(op_str, lambda a, b: False)(float(real), float(valor))
        if not ok:
            todos_ok = False
        relatorio.append({"metrica": metrica, "ok": ok,
                           "real": real, "esperado": f"{op_str} {valor}"})
    return todos_ok, relatorio


def _metricas_da_db(conn, simulacao_id: str) -> dict:
    """Agrega métricas finais do mundos_log para a simulação dada."""
    metricas = {}
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                MAX(gen_max)       AS gen_max,
                AVG(gen_max)       AS gen_media,
                AVG(cv_fc1)        AS cv_fc1_media,
                AVG(n_vivos)       AS n_vivos_media
            FROM mundos_log
            WHERE simulacao_id = %s
              AND t_s = (SELECT MAX(t_s) FROM mundos_log WHERE simulacao_id = %s)
        """, (simulacao_id, simulacao_id))
        row = cur.fetchone()
        if row and row[0] is not None:
            metricas = {
                "gen_max":       int(row[0]),
                "gen_media":     float(row[1] or 0),
                "cv_fc1_media":  float(row[2] or 0),
                "n_vivos_media": float(row[3] or 0),
            }
    return metricas


def cmd_list(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT e.experimento_id, e.nome, b.nome AS baseline,
                   e.estado, e.criterios_atingidos,
                   e.criado_em, e.concluido_em
            FROM experimentos e
            JOIN baselines b ON b.baseline_id = e.baseline_id
            ORDER BY e.criado_em DESC
            LIMIT 30
        """)
        rows = cur.fetchall()
    if not rows:
        print("(nenhum experimento registado)")
        return
    print(f"\n{'ID':36}  {'Nome':30}  {'Baseline':25}  {'Estado':15}  {'OK':4}  Criado")
    print("─" * 130)
    for r in rows:
        eid, nome, bl, estado, ok, criado, concluido = r
        ok_str = "✓" if ok is True else ("✗" if ok is False else "—")
        print(f"  {str(eid):34}  {(nome or ''):28}  {bl:23}  {estado:13}  {ok_str:4}  "
              f"{criado.strftime('%Y-%m-%d %H:%M') if criado else ''}")


def cmd_validate(conn, experimento_id: str, conclusao: str | None) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT e.estado, e.simulacao_id, e.criterios_validacao,
                   e.resultado_real, e.criterios_atingidos, e.nome,
                   b.nome AS baseline_nome
            FROM experimentos e
            JOIN baselines b ON b.baseline_id = e.baseline_id
            WHERE e.experimento_id = %s
        """, (experimento_id,))
        row = cur.fetchone()

    if not row:
        print(f"ERRO: experimento_id '{experimento_id}' não encontrado.", file=sys.stderr)
        sys.exit(1)

    estado, sim_id, criterios_json, resultado_real, ok_prev, nome, bl_nome = row
    criterios = criterios_json if isinstance(criterios_json, list) else json.loads(criterios_json or "[]")

    print(f"\nExperimento : {nome or experimento_id}")
    print(f"Baseline    : {bl_nome}")
    print(f"Estado      : {estado}")
    print(f"Simulação   : {sim_id or '(por iniciar)'}")

    # Obter métricas actuais da DB se a simulação existir
    metricas = {}
    if sim_id:
        metricas = resultado_real if isinstance(resultado_real, dict) else \
                   (json.loads(resultado_real) if resultado_real else {})
        # Re-computar da DB se vazio
        if not metricas:
            metricas = _metricas_da_db(conn, sim_id)
            print("(métricas obtidas do mundos_log)")

    if criterios:
        print(f"\nCritérios de validação ({len(criterios)}):")
        todos_ok, relatorio = _check(criterios, metricas)
        for r in relatorio:
            icon = "✓" if r["ok"] else "✗"
            real_str = f"{r['real']:.4f}" if isinstance(r["real"], float) else str(r["real"])
            print(f"  {icon} {r['metrica']}: {real_str}  (esperado {r['esperado']})")
        print(f"\n  Resultado: {'✓ PASS' if todos_ok else '✗ FAIL'}")

        # Actualizar DB
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE experimentos
                SET criterios_atingidos = %s,
                    resultado_real      = %s,
                    analise_conclusao   = COALESCE(%s, analise_conclusao),
                    estado              = CASE WHEN estado != 'cancelado'
                                              THEN 'concluido' ELSE estado END,
                    concluido_em        = COALESCE(concluido_em, %s)
                WHERE experimento_id = %s
            """, (todos_ok, json.dumps(metricas), conclusao,
                  datetime.now(timezone.utc), experimento_id))
        print("  DB actualizada.")
    else:
        print("(sem critérios de validação definidos)")

    if conclusao:
        print(f"\nConclusão registada: {conclusao}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verificar critérios de validação de um experimento."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--experimento_id", help="UUID do experimento a validar")
    group.add_argument("--list", action="store_true", help="Listar experimentos recentes")
    parser.add_argument("--conclusao", default=None,
                        help="Análise humana de conclusão a registar em DB")
    args = parser.parse_args()

    db_url = os.getenv("DATABASE_URL", "")
    if not db_url:
        print("ERRO: DATABASE_URL não definido.", file=sys.stderr)
        sys.exit(1)

    try:
        import psycopg2
        conn = psycopg2.connect(db_url)
        conn.autocommit = True
    except Exception as e:
        print(f"ERRO: Não foi possível ligar à DB: {e}", file=sys.stderr)
        sys.exit(1)

    if args.list:
        cmd_list(conn)
    else:
        cmd_validate(conn, args.experimento_id, args.conclusao)

    conn.close()


if __name__ == "__main__":
    main()
