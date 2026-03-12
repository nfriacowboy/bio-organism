"""
Exportador 1 — Métricas por episódio.

Gera:
  metricas/metricas_por_episodio.csv
  metricas/resumo_treino.json
  metricas/progressao_fases.csv
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import pandas as pd

from ..config import ExportConfig
from ..utils.db import get_metricas_por_episodio, get_treino_metadata


# ---------------------------------------------------------------------------
# Utilitário: estabilização (taxa_inf > 0.7 por 20+ episódios consecutivos)
# ---------------------------------------------------------------------------

def _episodio_estabilizacao(taxa: pd.Series) -> int | None:
    window = 20
    threshold = 0.7
    rolling_mean = taxa.rolling(window=window, min_periods=window).mean()
    mask = rolling_mean >= threshold
    if mask.any():
        idx = mask.idxmax()   # primeiro índice onde a condição é True
        return int(taxa.index[idx])
    return None


def _mudancas_fase(df: pd.DataFrame) -> dict[str, int | None]:
    result: dict[str, int | None] = {"1_para_2": None, "2_para_3": None}
    prev = None
    for _, row in df.iterrows():
        fase = int(row["fase_curriculum"])
        if prev is None:
            prev = fase
            continue
        if fase == 2 and prev == 1 and result["1_para_2"] is None:
            result["1_para_2"] = int(row["episodio"])
        elif fase == 3 and prev == 2 and result["2_para_3"] is None:
            result["2_para_3"] = int(row["episodio"])
        prev = fase
    return result


# ---------------------------------------------------------------------------
# Exportador principal
# ---------------------------------------------------------------------------

def exportar(cfg: ExportConfig) -> list[Path]:
    generated: list[Path] = []

    with cfg.engine.connect() as conn:
        meta  = get_treino_metadata(cfg.treino_id, conn)
        df    = get_metricas_por_episodio(cfg.treino_id, conn)

    out_dir = cfg.cat("metricas")

    # -----------------------------------------------------------------------
    # 1. metricas_por_episodio.csv
    # -----------------------------------------------------------------------
    csv_path = out_dir / "metricas_por_episodio.csv"

    if df.empty:
        # Criar CSV vazio com cabeçalho correcto
        cols = [
            "episodio", "tempo_vida", "tempo_vida_media_10ep", "comida_total",
            "energia_gasta", "eficiencia", "recompensa_total", "steps_aprendizagem",
            "steps_inferencia", "taxa_inferencia", "novidade_media",
            "bias_explorar_medio", "bias_exploitar_medio", "fase_curriculum",
            "morreu", "seed",
        ]
        pd.DataFrame(columns=cols).to_csv(csv_path, index=False)
    else:
        col_order = [
            "episodio", "tempo_vida", "tempo_vida_media_10ep", "comida_total",
            "energia_gasta", "eficiencia", "recompensa_total", "steps_aprendizagem",
            "steps_inferencia", "taxa_inferencia", "novidade_media",
            "bias_explorar_medio", "bias_exploitar_medio", "fase_curriculum",
            "morreu", "seed",
        ]
        df[col_order].to_csv(csv_path, index=False, float_format="%.4f")

    generated.append(csv_path)
    print(f"  [metricas] {csv_path.name} — {len(df)} episódios")

    # -----------------------------------------------------------------------
    # 2. resumo_treino.json
    # -----------------------------------------------------------------------
    json_path = out_dir / "resumo_treino.json"

    if df.empty:
        resumo: dict = {
            "treino_id": cfg.treino_id,
            "data_human": meta.get("data_human", ""),
            "nota": meta.get("nota", ""),
            "conclusao": meta.get("conclusao", ""),
            "parametros": meta.get("parametros", {}),
            "total_episodios": 0,
            "fase_maxima_atingida": 1,
            "episodio_mudanca_fase": {"1_para_2": None, "2_para_3": None},
            "tempo_vida": {},
            "eficiencia": {},
            "taxa_inferencia": {},
            "convergencia": {"episodio_estabilizacao": None, "criterio": "taxa_inferencia > 0.7 por 20 episodios consecutivos"},
            "mlflow_run_id": meta.get("mlflow_run_id"),
        }
    else:
        tv = df["tempo_vida"].tolist()
        ef = df["eficiencia"].tolist()
        ti = df["taxa_inferencia"]

        melhor_idx = int(df["tempo_vida"].idxmax())
        pior_idx_apos_50 = None
        df_late = df[df["episodio"] > 50]
        if not df_late.empty:
            pior_idx_apos_50 = int(df_late["tempo_vida"].idxmin())

        mudancas = _mudancas_fase(df)
        estab    = _episodio_estabilizacao(ti)

        resumo = {
            "treino_id": cfg.treino_id,
            "data_human": meta.get("data_human", ""),
            "nota": meta.get("nota", ""),
            "conclusao": meta.get("conclusao", ""),
            "parametros": meta.get("parametros", {}),
            "total_episodios": len(df),
            "fase_maxima_atingida": int(df["fase_curriculum"].max()),
            "episodio_mudanca_fase": mudancas,
            "tempo_vida": {
                "min": int(min(tv)),
                "max": int(max(tv)),
                "media": round(statistics.mean(tv), 2),
                "mediana": round(statistics.median(tv), 2),
                "p25": round(float(pd.Series(tv).quantile(0.25)), 2),
                "p75": round(float(pd.Series(tv).quantile(0.75)), 2),
                "desvio_padrao": round(statistics.stdev(tv) if len(tv) > 1 else 0.0, 2),
                "melhor_episodio": int(df.loc[melhor_idx, "episodio"]),
                "pior_episodio_apos_ep50": (
                    int(df.loc[pior_idx_apos_50, "episodio"]) if pior_idx_apos_50 is not None else None
                ),
            },
            "eficiencia": {
                "min": round(min(ef), 4),
                "max": round(max(ef), 4),
                "media": round(statistics.mean(ef), 4),
                "mediana": round(statistics.median(ef), 4),
            },
            "taxa_inferencia": {
                "media_total": round(float(ti.mean()), 4),
                "media_ultimos_100ep": round(float(ti.tail(100).mean()), 4),
            },
            "convergencia": {
                "episodio_estabilizacao": estab,
                "criterio": "taxa_inferencia > 0.7 por 20 episodios consecutivos",
            },
            "mlflow_run_id": meta.get("mlflow_run_id"),
        }

    json_path.write_text(json.dumps(resumo, ensure_ascii=False, indent=2))
    generated.append(json_path)
    print(f"  [metricas] {json_path.name}")

    # -----------------------------------------------------------------------
    # 3. progressao_fases.csv
    # -----------------------------------------------------------------------
    fase_path = out_dir / "progressao_fases.csv"

    if df.empty:
        pd.DataFrame(columns=[
            "episodio_mudanca", "fase_anterior", "fase_nova",
            "tempo_vida_medio_na_mudanca", "eficiencia_na_mudanca",
        ]).to_csv(fase_path, index=False)
    else:
        rows = []
        prev_fase = None
        for _, row in df.iterrows():
            fase = int(row["fase_curriculum"])
            if prev_fase is not None and fase != prev_fase:
                rows.append({
                    "episodio_mudanca":            int(row["episodio"]),
                    "fase_anterior":               prev_fase,
                    "fase_nova":                   fase,
                    "tempo_vida_medio_na_mudanca": round(float(row["tempo_vida_media_10ep"]), 2),
                    "eficiencia_na_mudanca":       round(float(row["eficiencia"]), 4),
                })
            prev_fase = fase
        pd.DataFrame(rows, columns=[
            "episodio_mudanca", "fase_anterior", "fase_nova",
            "tempo_vida_medio_na_mudanca", "eficiencia_na_mudanca",
        ]).to_csv(fase_path, index=False)

    generated.append(fase_path)
    print(f"  [metricas] {fase_path.name}")

    return generated
