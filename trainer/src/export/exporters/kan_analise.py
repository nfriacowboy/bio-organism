"""
Exportador 4 — Análise da KAN.

Gera:
  kan/kan_outputs_historico.csv
  kan/kan_splines_visualization.png
  kan/kan_inputs_correlacao.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from ..config import ExportConfig
from ..utils.db import get_kan_outputs_historico, get_treino_metadata
from ..utils import mlflow_client as mf


# Nome dos inputs e outputs da KAN
INPUT_NAMES  = [
    "energia_actual", "energia_max", "tempo_de_vida",
    "comida_vista_50steps", "distancia_media_comida", "steps_sem_comer",
]
OUTPUT_NAMES = ["bias_explorar", "bias_exploitar"]


def exportar(cfg: ExportConfig) -> list[Path]:
    generated: list[Path] = []
    out_dir = cfg.cat("kan")

    with cfg.engine.connect() as conn:
        meta = get_treino_metadata(cfg.treino_id, conn)
        df_hist = get_kan_outputs_historico(cfg.treino_id, conn)

    run_id = meta.get("mlflow_run_id") or mf.get_run_id(cfg.treino_id)
    kan_path: Path | None = None
    if run_id:
        tmp = out_dir / "_modelos"
        for name in ("kan_final.pt", "kan_ep500.pt", "kan_ep400.pt",
                     "kan_ep300.pt", "kan_ep200.pt", "kan_ep100.pt"):
            p = mf.download_artifact(run_id, name, tmp)
            if p and p.exists():
                kan_path = p
                break

    # -----------------------------------------------------------------------
    # 1. kan_outputs_historico.csv
    # -----------------------------------------------------------------------
    hist_path = out_dir / "kan_outputs_historico.csv"
    if not df_hist.empty:
        df_hist.to_csv(hist_path, index=False, float_format="%.4f")
        print(f"  [kan] {hist_path.name} — {len(df_hist)} episódios")
    else:
        pd.DataFrame(columns=[
            "episodio", "bias_explorar_medio", "bias_explorar_std",
            "bias_exploitar_medio", "bias_exploitar_std",
            "ratio_explorar_exploitar", "dominante",
        ]).to_csv(hist_path, index=False)
        print(f"  [kan] {hist_path.name} — vazio (sem dados de KAN no log)")
    generated.append(hist_path)

    # -----------------------------------------------------------------------
    # 2. kan_splines_visualization.png
    # -----------------------------------------------------------------------
    png_path = out_dir / "kan_splines_visualization.png"
    png_generated = False

    if kan_path and kan_path.exists():
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from kan import KAN as _KAN  # pykan

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            kan_model = _KAN(width=[6, 8, 2], grid=5, k=3, auto_save=False,
                             device=str(device))
            state = torch.load(kan_path, map_location=device)
            kan_model.load_state_dict(state, strict=False)
            kan_model.plot(
                beta=100,
                title="KAN Policy — Organismo SNN (treino final)",
            )
            plt.savefig(png_path, dpi=150, bbox_inches="tight")
            plt.close("all")
            png_generated = True
            print(f"  [kan] {png_path.name} — visualização gerada")
        except Exception as e:
            print(f"  [kan] kan_splines_visualization: {e}")

    if not png_generated:
        # Criar PNG placeholder com matplotlib
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5,
                    "KAN não disponível\n(modelo não encontrado no MLflow)",
                    ha="center", va="center", fontsize=12, transform=ax.transAxes)
            ax.axis("off")
            plt.savefig(png_path, dpi=100, bbox_inches="tight")
            plt.close("all")
            print(f"  [kan] {png_path.name} — placeholder (modelo não disponível)")
        except Exception:
            png_path.write_bytes(b"")
    generated.append(png_path)

    # -----------------------------------------------------------------------
    # 3. kan_inputs_correlacao.csv
    # -----------------------------------------------------------------------
    corr_path = out_dir / "kan_inputs_correlacao.csv"

    # Precisamos de dados de estado_kan + outputs.
    # Não gravamos estado_kan separado — reconstruímos a partir dos logs:
    # energia_norm ↔ bias_explorar/exploitar já está na hist
    # Usar df_hist se disponível, senão vazio
    rows = []

    if not df_hist.empty and len(df_hist) >= 10:
        # Correlações disponíveis: bias_explorar_medio / bias_exploitar_medio
        # com features disponíveis nos logs (limitado)
        try:
            from scipy import stats as sp_stats

            # Construir proxy features por episódio a partir da tabela steps_log
            sql_feat = """
                SELECT
                    episodio,
                    AVG(energia_actual)                 AS energia_actual,
                    AVG(energia_max)                    AS energia_max,
                    COUNT(*) * 1.0                      AS tempo_de_vida,
                    SUM(CASE WHEN comeu THEN 1 ELSE 0 END) AS comida_vista_50steps,
                    NULL::float                         AS distancia_media_comida,
                    0::float                            AS steps_sem_comer
                FROM steps_log
                WHERE treino_id = :tid
                GROUP BY episodio
                ORDER BY episodio
            """
            from sqlalchemy import text
            with cfg.engine.connect() as conn:
                df_feat = pd.read_sql(text(sql_feat), conn,
                                      params={"tid": cfg.treino_id})

            df_merged = df_hist.merge(df_feat, on="episodio", how="inner")

            for inp in INPUT_NAMES:
                if inp not in df_merged.columns:
                    continue
                x = df_merged[inp].dropna()
                for out in OUTPUT_NAMES:
                    out_col = f"{out}_medio"
                    if out_col not in df_merged.columns:
                        continue
                    y = df_merged[out_col].loc[x.index].dropna()
                    common = x.index.intersection(y.index)
                    if len(common) < 5:
                        continue
                    r, p = sp_stats.pearsonr(x.loc[common], y.loc[common])
                    strength = (
                        "forte positivo"  if r > 0.5  else
                        "fraco positivo"  if r > 0.2  else
                        "forte negativo"  if r < -0.5 else
                        "fraco negativo"  if r < -0.2 else
                        "neutro"
                    )
                    rows.append({
                        "input_nome":  inp,
                        "output_nome": out,
                        "pearson_r":   round(r, 4),
                        "pearson_p":   round(p, 6),
                        "interpretacao": f"{inp} → {out}: r={r:.2f} ({strength})",
                    })
        except Exception as e:
            print(f"  [kan] correlações: {e}")

    if rows:
        pd.DataFrame(rows).to_csv(corr_path, index=False)
        print(f"  [kan] {corr_path.name} — {len(rows)} pares input→output")
    else:
        pd.DataFrame(columns=[
            "input_nome", "output_nome", "pearson_r", "pearson_p", "interpretacao",
        ]).to_csv(corr_path, index=False)
        print(f"  [kan] {corr_path.name} — vazio (dados insuficientes)")
    generated.append(corr_path)

    return generated
