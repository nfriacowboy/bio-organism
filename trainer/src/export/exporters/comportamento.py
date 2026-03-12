"""
Exportador 2 — Logs de comportamento.

Gera:
  comportamento/log_ep_<tipo>.csv  (para cada episódio de referência)
  comportamento/eventos_criticos.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import ExportConfig
from ..utils.db import (
    get_episodios_referencia,
    get_log_episodio,
    get_todos_eventos,
)

# Colunas canónicas de um ficheiro de log de episódio
COLUNAS_LOG = [
    "step", "pos_x", "pos_y", "angulo", "energia_actual",
    "accao", "modo", "novidade", "recompensa", "comeu",
    "visao_resumo", "bias_explorar", "bias_exploitar", "inten_proxima",
]


def _guardar_log(
    df: pd.DataFrame,
    path: Path,
    max_steps: int,
    label: str,
) -> None:
    if df.empty:
        pd.DataFrame(columns=COLUNAS_LOG).to_csv(path, index=False)
        print(f"    [comportamento] {path.name} — vazio (sem dados)")
        return
    # Garantir que todas as colunas estão presentes
    for col in COLUNAS_LOG:
        if col not in df.columns:
            df[col] = ""
    n_antes = len(df)
    if max_steps and len(df) > max_steps:
        df = df.head(max_steps)
    df[COLUNAS_LOG].to_csv(path, index=False, float_format="%.4f")
    print(f"    [comportamento] {path.name} — ep={label}, {len(df)}/{n_antes} steps")


def exportar(cfg: ExportConfig) -> list[Path]:
    generated: list[Path] = []
    out_dir = cfg.cat("comportamento")

    with cfg.engine.connect() as conn:
        refs = get_episodios_referencia(cfg.treino_id, conn)

        # Mapa: label → (episodio, nome_ficheiro)
        ep_map: dict[str, tuple[int | None, str]] = {
            "inicial":          (refs["inicial"],          "log_ep_inicial.csv"),
            "precoce":          (refs["precoce"],          "log_ep_precoce.csv"),
            "melhor":           (refs["melhor"],           "log_ep_melhor.csv"),
            "pos_convergencia": (refs["pos_convergencia"], "log_ep_pos_convergencia.csv"),
            "ultimo":           (refs["ultimo"],           "log_ep_ultimo.csv"),
        }

        for label, (episodio, fname) in ep_map.items():
            fpath = out_dir / fname
            if episodio is None:
                pd.DataFrame(columns=COLUNAS_LOG).to_csv(fpath, index=False)
                print(f"    [comportamento] {fname} — episódio não disponível")
            else:
                df = get_log_episodio(
                    cfg.treino_id, episodio, conn,
                    max_steps=cfg.max_steps_por_episodio,
                )
                _guardar_log(df, fpath, cfg.max_steps_por_episodio, str(episodio))
            generated.append(fpath)

        # Eventos críticos — todos os episódios
        ev_path = out_dir / "eventos_criticos.csv"
        dfs_eventos: list[pd.DataFrame] = []
        for tipo in ("comeu", "quase_morreu", "modo_mudou", "novidade_alta", "sem_visao"):
            try:
                ev = get_todos_eventos(cfg.treino_id, tipo, conn)
                if not ev.empty:
                    dfs_eventos.append(ev)
            except Exception as e:
                print(f"    [comportamento] aviso evento {tipo}: {e}")

        if dfs_eventos:
            ev_all = pd.concat(dfs_eventos, ignore_index=True)
            ev_all = ev_all.sort_values(["episodio", "step"])
        else:
            ev_all = pd.DataFrame(columns=[
                "treino_id", "episodio", "step", "tipo_evento",
                "pos_x", "pos_y", "energia_actual", "detalhe",
            ])

        ev_all.to_csv(ev_path, index=False, float_format="%.4f")
        generated.append(ev_path)
        print(f"  [comportamento] {ev_path.name} — {len(ev_all)} eventos")

    return generated
