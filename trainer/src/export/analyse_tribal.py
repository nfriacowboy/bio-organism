#!/usr/bin/env python3
"""
Análise tribal post-hoc — lê DB e produz relatório de emergência tribal
em função da densidade de comida.

Uso:
    DATABASE_URL="postgresql://..." python3 -m export.analyse_tribal --sim <simulacao_id>
    DATABASE_URL="postgresql://..." python3 -m export.analyse_tribal --last
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://snn_user:snn_secure_pw_2026@localhost:5432/snn_organismo",
)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _connect():
    import psycopg2
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    return conn


def _query(conn, sql: str, params=None) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(sql, params or ())
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# DBSCAN tribal (posições)
# ---------------------------------------------------------------------------

def _dbscan_tribal(positions: list[tuple[int, int]], eps: float = 80) -> dict:
    """
    Aplica DBSCAN sobre posições (x, y) e devolve métricas tribais.
    """
    if not positions:
        return {"n_tribos": 0, "tamanho_medio": 0, "tamanho_max": 0, "isolados": 0}

    try:
        from sklearn.cluster import DBSCAN
        coords = np.array(positions, dtype=float)
        labels = DBSCAN(eps=eps, min_samples=2).fit_predict(coords)
    except ImportError:
        labels = np.zeros(len(positions), dtype=int)

    n_tribos   = int((labels >= 0).any() and labels[labels >= 0].max() + 1) if labels.max() >= 0 else 0
    isolados   = int((labels == -1).sum())
    if n_tribos > 0:
        sizes      = [int((labels == i).sum()) for i in range(n_tribos)]
        tam_medio  = float(np.mean(sizes))
        tam_max    = int(max(sizes))
    else:
        tam_medio = tam_max = 0.0

    return {
        "n_tribos":      n_tribos,
        "tamanho_medio": tam_medio,
        "tamanho_max":   tam_max,
        "isolados":      isolados,
    }


# ---------------------------------------------------------------------------
# Análise principal
# ---------------------------------------------------------------------------

def analisar(simulacao_id: str) -> None:
    conn = _connect()

    # --- Metadados da simulação ---
    sims = _query(conn, "SELECT * FROM simulacoes WHERE simulacao_id = %s", (simulacao_id,))
    if not sims:
        print(f"Simulação '{simulacao_id}' não encontrada.")
        return

    sim = sims[0]
    cfg_raw = sim.get("config_json") or {}
    cfg = cfg_raw if isinstance(cfg_raw, dict) else json.loads(cfg_raw)
    print(f"\n{'='*65}")
    print(f"  Simulação: {simulacao_id}")
    print(f"  Nota:      {sim.get('nota', '')}")
    print(f"  Mundos:    {sim.get('n_mundos', '?')}")
    print(f"  Config:    {json.dumps(cfg, ensure_ascii=False)}")
    print(f"{'='*65}\n")

    # --- Configurações por mundo ---
    configs = _query(conn,
        "SELECT * FROM mundos_config WHERE simulacao_id = %s ORDER BY mundo_id",
        (simulacao_id,)
    )
    if not configs:
        print("Sem dados em mundos_config. Simulação ainda a correr?")
        return

    # --- Métricas por mundo (último registo) ---
    mundos_metrics = _query(conn, """
        SELECT DISTINCT ON (mundo_id)
            mundo_id, t_s, n_vivos, gen_max, gen_media, cv_fc1, comida_rate_s
        FROM mundos_log
        WHERE simulacao_id = %s
        ORDER BY mundo_id, t_s DESC
    """, (simulacao_id,))
    metrics_by_id = {m["mundo_id"]: m for m in mundos_metrics}

    # --- Lifetimes ---
    lt = _query(conn, """
        SELECT mundo_id, COUNT(*) AS n_org,
               AVG(idade) AS idade_media, AVG(comida_total) AS comida_media,
               AVG(n_filhos) AS filhos_medio, MAX(geracao) AS gen_max,
               100.0 * SUM(CASE WHEN causa_morte='fome' THEN 1 ELSE 0 END) / COUNT(*) AS pct_fome
        FROM organismos_lifetimes
        WHERE simulacao_id = %s
        GROUP BY mundo_id
    """, (simulacao_id,))
    lt_by_id = {m["mundo_id"]: m for m in lt}

    # --- Snapshots mais recentes (tribal map) ---
    snaps = _query(conn, """
        SELECT DISTINCT ON (mundo_id)
            mundo_id, t_s, step
        FROM posicoes_snapshot
        WHERE simulacao_id = %s
        ORDER BY mundo_id, t_s DESC
    """, (simulacao_id,))

    tribal_by_id: dict[int, dict] = {}
    for snap in snaps:
        mid = snap["mundo_id"]
        positions_rows = _query(conn, """
            SELECT x, y FROM posicoes_snapshot
            WHERE simulacao_id = %s AND mundo_id = %s AND step = %s
        """, (simulacao_id, mid, snap["step"]))
        positions = [(r["x"], r["y"]) for r in positions_rows]
        tribal_by_id[mid] = _dbscan_tribal(positions, eps=40)

    # --- Agregar por densidade ---
    by_density: dict[float, list[dict]] = defaultdict(list)
    for cfg_m in configs:
        mid  = cfg_m["mundo_id"]
        dpct = round(float(cfg_m.get("food_pct", 0)) * 100, 2)
        row  = {
            "mundo_id": mid,
            "food_pct": dpct,
            "n_food":   cfg_m.get("n_food", 0),
        }
        row.update(metrics_by_id.get(mid, {}))
        row.update(lt_by_id.get(mid, {}))
        row.update({f"tribal_{k}": v for k, v in tribal_by_id.get(mid, {}).items()})
        by_density[dpct].append(row)

    # --- Imprimir tabela de resultados ---
    print(f"  {'Dens%':>7} | {'Mundos':>6} | {'Gen_max':>7} | {'CV_fc1':>7} | "
          f"{'Idade_med':>9} | {'Food/org':>8} | "
          f"{'Tribos_med':>10} | {'TribSz_med':>10} | {'Isolados':>8}")
    print("  " + "-" * 90)

    for dpct in sorted(by_density):
        sl = by_density[dpct]
        def _safe_mean(key):
            vals = [r[key] for r in sl if r.get(key) is not None]
            return float(np.mean(vals)) if vals else float("nan")
        def _safe_max(key):
            vals = [r[key] for r in sl if r.get(key) is not None]
            return int(max(vals)) if vals else -1

        n_tribos_med  = _safe_mean("tribal_n_tribos")
        tam_tribal    = _safe_mean("tribal_tamanho_medio")
        isolados_med  = _safe_mean("tribal_isolados")
        cv_med        = _safe_mean("cv_fc1")
        gen_max_max   = _safe_max("gen_max")
        idade_med     = _safe_mean("idade_media")
        comida_org    = _safe_mean("comida_media")

        print(
            f"  {dpct:>7.2f} | {len(sl):>6} | {gen_max_max:>7} | "
            f"{cv_med:>7.3f} | {idade_med:>9.0f} | {comida_org:>8.1f} | "
            f"{n_tribos_med:>10.1f} | {tam_tribal:>10.1f} | {isolados_med:>8.1f}"
        )

    # --- Insights ---
    print(f"\n{'='*65}")
    print("  INSIGHTS:")

    # Correlação densidade × tribos
    dens_all  = []
    tribos_all = []
    cv_all    = []
    for dpct, sl in sorted(by_density.items()):
        for r in sl:
            if r.get("tribal_n_tribos") is not None:
                dens_all.append(dpct)
                tribos_all.append(r["tribal_n_tribos"])
            if r.get("cv_fc1") is not None:
                cv_all.append((dpct, r["cv_fc1"]))

    if len(dens_all) > 2:
        corr = float(np.corrcoef(dens_all, tribos_all)[0, 1])
        print(f"  Correlação densidade × n_tribos : {corr:+.3f}")
        if corr < -0.3:
            print("  → Densidade ALTA suprime tribos (abundância → mistura)")
        elif corr > 0.3:
            print("  → Densidade ALTA gera mais tribos (mais organismos → mais grupos)")
        else:
            print("  → Fraca correlação linear (dinâmica não-linear)")

    if len(cv_all) > 2:
        cd, cc = zip(*cv_all)
        corr_cv = float(np.corrcoef(cd, cc)[0, 1])
        print(f"  Correlação densidade × cv_fc1   : {corr_cv:+.3f}")
        if corr_cv < -0.2:
            print("  → Escassez promove diversidade genética (pressão selectiva)")
        elif corr_cv > 0.2:
            print("  → Abundância promove diversidade (deriva genética)")

    # Mundo com mais tribos
    best_tribal = max(
        (r for sl in by_density.values() for r in sl if r.get("tribal_n_tribos")),
        key=lambda r: r.get("tribal_n_tribos", 0),
        default=None,
    )
    if best_tribal:
        print(f"  Mundo mais tribal: id={best_tribal['mundo_id']} "
              f"food_pct={best_tribal['food_pct']:.2f}% "
              f"n_tribos={best_tribal.get('tribal_n_tribos',0)}")

    print(f"{'='*65}\n")

    conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Análise tribal multi-mundo")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sim",  type=str, help="simulacao_id explícito")
    group.add_argument("--last", action="store_true", help="última simulação")
    args = parser.parse_args()

    if args.last:
        conn_tmp = _connect()
        rows     = _query(conn_tmp, """
            SELECT simulacao_id FROM simulacoes
            ORDER BY created_at DESC LIMIT 1
        """)
        conn_tmp.close()
        if not rows:
            print("Sem simulações na DB.")
            sys.exit(1)
        sim_id = rows[0]["simulacao_id"]
    else:
        sim_id = args.sim

    analisar(sim_id)
