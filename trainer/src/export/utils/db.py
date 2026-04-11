"""
utils/db.py — Queries PostgreSQL reutilizáveis para o módulo de exportação.

Todas as funções recebem treino_id (str) e conn (SQLAlchemy Connection).
Queries com GROUP BY no servidor para evitar trazer dezenas de milhões de
linhas para Python.
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Connection


# ---------------------------------------------------------------------------
# Treino metadata
# ---------------------------------------------------------------------------

def get_treino_metadata(treino_id: str, conn: Connection) -> dict:
    row = conn.execute(
        text("SELECT * FROM treinos WHERE id = :id"),
        {"id": treino_id},
    ).mappings().fetchone()
    if row is None:
        raise ValueError(f"Treino não encontrado: {treino_id}")
    d = dict(row)
    # parametros / metricas_finais podem estar como string
    for k in ("parametros", "metricas_finais"):
        if isinstance(d.get(k), str):
            try:
                d[k] = json.loads(d[k])
            except Exception:
                pass
    return d


# ---------------------------------------------------------------------------
# Métricas por episódio (aggregated from steps_log)
# ---------------------------------------------------------------------------

def get_metricas_por_episodio(treino_id: str, conn: Connection) -> pd.DataFrame:
    sql = text("""
        SELECT
            episodio,
            COUNT(*)                                         AS tempo_vida,
            SUM(CASE WHEN comeu THEN 1 ELSE 0 END)          AS comida_total,
            SUM(ABS(recompensa))                             AS energia_gasta,
            SUM(recompensa)                                  AS recompensa_total,
            SUM(CASE WHEN modo='aprendizagem' THEN 1 ELSE 0 END)  AS steps_aprendizagem,
            SUM(CASE WHEN modo='inferencia'   THEN 1 ELSE 0 END)  AS steps_inferencia,
            AVG(novidade)                                    AS novidade_media,
            AVG(bias_explorar)                               AS bias_explorar_medio,
            AVG(bias_exploitar)                              AS bias_exploitar_medio,
            MAX(fase_mundo)                                  AS fase_curriculum,
            MIN(energia_actual) <= 0                         AS morreu
        FROM steps_log
        WHERE treino_id = :tid
        GROUP BY episodio
        ORDER BY episodio
    """)
    df = pd.read_sql(sql, conn, params={"tid": treino_id})
    if df.empty:
        return df
    # Eficiência
    df["eficiencia"] = df["comida_total"] / df["energia_gasta"].replace(0, float("nan"))
    df["eficiencia"] = df["eficiencia"].fillna(0.0)
    # Taxa inferência
    df["taxa_inferencia"] = df["steps_inferencia"] / df["tempo_vida"].replace(0, 1)
    # Média móvel 10 episódios
    df["tempo_vida_media_10ep"] = (
        df["tempo_vida"].rolling(window=10, min_periods=1).mean()
    )
    # Seed: não está na steps_log — deixar como 0
    df["seed"] = 0
    return df


# ---------------------------------------------------------------------------
# Log de um único episódio
# ---------------------------------------------------------------------------

def get_log_episodio(
    treino_id: str,
    episodio: int,
    conn: Connection,
    max_steps: int | None = None,
) -> pd.DataFrame:
    limit_clause = f"LIMIT {max_steps}" if max_steps else ""
    sql = text(f"""
        SELECT step, pos_x, pos_y, angulo, energia_actual, accao,
               modo, novidade, recompensa, comeu,
               bias_explorar, bias_exploitar, fase_mundo,
               visao::text AS visao_json
        FROM steps_log
        WHERE treino_id = :tid AND episodio = :ep
        ORDER BY step
        {limit_clause}
    """)
    df = pd.read_sql(sql, conn, params={"tid": treino_id, "ep": episodio})
    if not df.empty:
        # visao_resumo compacto
        def _resumo(v: str) -> str:
            try:
                d = json.loads(v)
                items = []
                for ang, info in list(d.items())[:3]:
                    tipo = info.get("tipo", "?")
                    dist = info.get("dist", "?")
                    items.append(f"{tipo[:4]}@{ang}°={dist:.0f}" if isinstance(dist, (int, float)) else f"?@{ang}")
                return ",".join(items)
            except Exception:
                return ""
        df["visao_resumo"] = df["visao_json"].apply(_resumo)
        df["inten_proxima"] = ""   # reservado — KAN intent não gravado no DB
        df.drop(columns=["visao_json"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Eventos críticos (todos os episódios)
# ---------------------------------------------------------------------------

def get_todos_eventos(
    treino_id: str,
    tipo: str,
    conn: Connection,
) -> pd.DataFrame:
    """
    tipo in: 'comeu', 'quase_morreu', 'modo_mudou', 'novidade_alta', 'sem_visao'
    """
    if tipo == "comeu":
        where = "comeu = true"
    elif tipo == "quase_morreu":
        where = "energia_actual < 10"
    elif tipo == "novidade_alta":
        where = "novidade > 0.8"
    else:
        # modo_mudou e sem_visao tratados em Python após query
        where = "true"

    sql = text(f"""
        SELECT treino_id, episodio, step, pos_x, pos_y, energia_actual,
               modo, novidade, comeu, visao::text AS visao_json
        FROM steps_log
        WHERE treino_id = :tid AND ({where})
        ORDER BY episodio, step
    """)
    df = pd.read_sql(sql, conn, params={"tid": treino_id})
    if df.empty:
        return pd.DataFrame(columns=[
            "treino_id", "episodio", "step", "tipo_evento",
            "pos_x", "pos_y", "energia_actual", "detalhe"
        ])
    df["tipo_evento"] = tipo
    df["detalhe"] = ""

    if tipo == "modo_mudou" and len(df) > 1:
        df = df.sort_values(["episodio", "step"])
        df["modo_ant"] = df.groupby("episodio")["modo"].shift(1)
        df = df[df["modo"] != df["modo_ant"]].copy()
        df["detalhe"] = df["modo_ant"].fillna("") + "→" + df["modo"]
        df = df.dropna(subset=["modo_ant"])
    elif tipo == "sem_visao":
        def _tudo_vazio(v: str) -> bool:
            try:
                return all(i.get("tipo") is None for i in json.loads(v).values())
            except Exception:
                return False
        df = df[df["visao_json"].apply(_tudo_vazio)].copy()
        df["detalhe"] = "todos_raios_vazios"

    return df[["treino_id", "episodio", "step", "tipo_evento",
               "pos_x", "pos_y", "energia_actual", "detalhe"]]


# ---------------------------------------------------------------------------
# Episódios de referência
# ---------------------------------------------------------------------------

def get_episodios_referencia(treino_id: str, conn: Connection) -> dict[str, int | None]:
    sql = text("""
        SELECT episodio, COUNT(*) AS n_steps
        FROM steps_log
        WHERE treino_id = :tid
        GROUP BY episodio
        ORDER BY episodio
    """)
    df = pd.read_sql(sql, conn, params={"tid": treino_id})
    if df.empty:
        return dict(inicial=None, precoce=None, melhor=None,
                    pos_convergencia=None, ultimo=None)

    episodios = sorted(df["episodio"].tolist())
    melhor_row = df.loc[df["n_steps"].idxmax()]
    melhor_ep = int(melhor_row["episodio"])

    inicial = episodios[0]
    precoce = episodios[min(49, len(episodios) - 1)]
    ultimo  = episodios[-1]

    # pos_convergencia: melhor_ep + 50 (or last episode if not available)
    pos_conv = min(melhor_ep + 50, ultimo)

    return dict(
        inicial=inicial,
        precoce=precoce,
        melhor=melhor_ep,
        pos_convergencia=pos_conv,
        ultimo=ultimo,
    )


# ---------------------------------------------------------------------------
# KAN outputs histórico
# ---------------------------------------------------------------------------

def get_kan_outputs_historico(treino_id: str, conn: Connection) -> pd.DataFrame:
    sql = text("""
        SELECT
            episodio,
            AVG(bias_explorar)   AS bias_explorar_medio,
            STDDEV(bias_explorar) AS bias_explorar_std,
            AVG(bias_exploitar)  AS bias_exploitar_medio,
            STDDEV(bias_exploitar) AS bias_exploitar_std
        FROM steps_log
        WHERE treino_id = :tid
          AND bias_explorar IS NOT NULL
        GROUP BY episodio
        ORDER BY episodio
    """)
    df = pd.read_sql(sql, conn, params={"tid": treino_id})
    if df.empty:
        return df
    df["ratio_explorar_exploitar"] = (
        df["bias_explorar_medio"] / (df["bias_exploitar_medio"] + 0.001)
    )
    def _dom(r: float) -> str:
        if r > 1.2:   return "explorar"
        if r < 0.8:   return "exploitar"
        return "equilibrado"
    df["dominante"] = df["ratio_explorar_exploitar"].apply(_dom)
    return df


# ---------------------------------------------------------------------------
# Posições (heatmap)
# ---------------------------------------------------------------------------

def get_posicoes_todas(
    treino_id: str,
    conn: Connection,
    resolucao_celula: int = 10,
) -> pd.DataFrame:
    sql = text(f"""
        SELECT
            (pos_x / {resolucao_celula}) AS celula_x,
            (pos_y / {resolucao_celula}) AS celula_y,
            COUNT(*)                      AS visitas_total,
            COUNT(DISTINCT episodio)      AS episodios_visitou,
            MIN(episodio)                 AS primeira_visita_episodio,
            MAX(episodio)                 AS ultima_visita_episodio
        FROM steps_log
        WHERE treino_id = :tid
        GROUP BY celula_x, celula_y
        ORDER BY celula_x, celula_y
    """)
    df = pd.read_sql(sql, conn, params={"tid": treino_id})
    if df.empty:
        return df
    total = df["visitas_total"].sum()
    df["pct_do_total"] = df["visitas_total"] / total * 100.0

    # visitas_media_por_episodio
    n_ep_sql = text(
        "SELECT COUNT(DISTINCT episodio) AS n FROM steps_log WHERE treino_id = :tid"
    )
    n_ep = conn.execute(n_ep_sql, {"tid": treino_id}).scalar() or 1
    df["visitas_media_por_episodio"] = df["visitas_total"] / n_ep
    return df


# ---------------------------------------------------------------------------
# Episódios log (single source of truth para métricas por episódio)
# ---------------------------------------------------------------------------

def get_episodios_log(treino_id: str, conn: Connection) -> pd.DataFrame:
    """Lê episodios_log — fonte primária para métricas por episódio."""
    sql = text("""
        SELECT
            episodio, tempo_vida, comida, eficiencia, recompensa_total,
            taxa_inferencia, novidade_media,
            bias_explorar_medio, bias_exploitar_medio,
            fase_curriculum, morreu,
            precisao_direcional, steps_com_visao, taxa_perseguicao,
            seed, p_avancar
        FROM episodios_log
        WHERE treino_id = :tid
        ORDER BY episodio
    """)
    return pd.read_sql(sql, conn, params={"tid": treino_id})
