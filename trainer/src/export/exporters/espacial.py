"""
Exportador 5 — Análise espacial.

Gera:
  espacial/heatmap_posicoes.csv
  espacial/heatmap_comida_consumida.csv
  espacial/trajectoria_ep_<tipo>.csv  (para cada episódio de referência)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import text

from ..config import ExportConfig
from ..utils.db import (
    get_episodios_referencia,
    get_log_episodio,
    get_posicoes_todas,
)

COLUNAS_TRAJ = ["episodio", "step", "pos_x", "pos_y", "angulo", "accao", "comeu"]


def exportar(cfg: ExportConfig) -> list[Path]:
    generated: list[Path] = []
    out_dir = cfg.cat("espacial")

    # -----------------------------------------------------------------------
    # 1. heatmap_posicoes.csv
    # -----------------------------------------------------------------------
    hm_path = out_dir / "heatmap_posicoes.csv"
    with cfg.engine.connect() as conn:
        df_hm = get_posicoes_todas(cfg.treino_id, conn, resolucao_celula=10)

    if not df_hm.empty:
        df_hm.to_csv(hm_path, index=False, float_format="%.2f")
        print(f"  [espacial] {hm_path.name} — {len(df_hm)} células")
    else:
        pd.DataFrame(columns=[
            "celula_x", "celula_y", "visitas_total",
            "visitas_media_por_episodio", "pct_do_total",
            "primeira_visita_episodio", "ultima_visita_episodio",
        ]).to_csv(hm_path, index=False)
        print(f"  [espacial] {hm_path.name} — vazio")
    generated.append(hm_path)

    # -----------------------------------------------------------------------
    # 2. heatmap_comida_consumida.csv
    # -----------------------------------------------------------------------
    comida_path = out_dir / "heatmap_comida_consumida.csv"
    sql_comida = text("""
        SELECT episodio, step, pos_x, pos_y,
               energia_actual AS energia_depois,
               fase_mundo     AS fase_curriculum
        FROM steps_log
        WHERE treino_id = :tid AND comeu = true
        ORDER BY episodio, step
    """)
    with cfg.engine.connect() as conn:
        df_comida = pd.read_sql(sql_comida, conn, params={"tid": cfg.treino_id})

    if not df_comida.empty:
        # energia_antes: step anterior (aproximação com shift)
        df_comida["energia_antes"] = df_comida["energia_depois"].shift(1).fillna(
            df_comida["energia_depois"]
        )
        col_order = ["episodio", "step", "pos_x", "pos_y",
                     "energia_antes", "energia_depois", "fase_curriculum"]
        df_comida[col_order].to_csv(comida_path, index=False, float_format="%.2f")
        print(f"  [espacial] {comida_path.name} — {len(df_comida)} eventos alimentação")
    else:
        pd.DataFrame(columns=[
            "episodio", "step", "pos_x", "pos_y",
            "energia_antes", "energia_depois", "fase_curriculum",
        ]).to_csv(comida_path, index=False)
        print(f"  [espacial] {comida_path.name} — vazio")
    generated.append(comida_path)

    # -----------------------------------------------------------------------
    # 3. Trajectórias dos episódios de referência
    # -----------------------------------------------------------------------
    with cfg.engine.connect() as conn:
        refs = get_episodios_referencia(cfg.treino_id, conn)

        ep_map = {
            "inicial":          refs["inicial"],
            "precoce":          refs["precoce"],
            "melhor":           refs["melhor"],
            "pos_convergencia": refs["pos_convergencia"],
            "ultimo":           refs["ultimo"],
        }

        for label, episodio in ep_map.items():
            fname = f"trajectoria_ep_{label}.csv"
            fpath = out_dir / fname
            if episodio is None:
                pd.DataFrame(columns=COLUNAS_TRAJ).to_csv(fpath, index=False)
                print(f"    [espacial] {fname} — episódio não disponível")
            else:
                df_log = get_log_episodio(
                    cfg.treino_id, episodio, conn,
                    max_steps=cfg.max_steps_por_episodio,
                )
                if df_log.empty:
                    traj = pd.DataFrame(columns=COLUNAS_TRAJ)
                else:
                    traj = df_log.copy()
                    traj["episodio"] = episodio
                    for col in COLUNAS_TRAJ:
                        if col not in traj.columns:
                            traj[col] = ""
                    traj = traj[COLUNAS_TRAJ]
                traj.to_csv(fpath, index=False, float_format="%.2f")
                print(f"    [espacial] {fname} — ep={episodio}, {len(traj)} steps")
            generated.append(fpath)

    return generated
