"""
Exportador único — produz um único ficheiro JSON com todos os dados do treino.

Ficheiro gerado: exports/{treino_id_sanitizado}_export.json

Secções:
  meta         — identificação e versão do export
  treino        — metadados e parâmetros da sessão de treino
  resumo        — estatísticas agregadas
  episodios     — métricas por episódio (de episodios_log)
  comportamento — logs de referência e eventos críticos (de steps_log)
  snn           — análise AGMP e pesos (de steps_log + .pt)
  kan           — histórico de bias, sensibilidade via plotext (de episodios_log + .pt)
  espacial      — heatmap e trajectórias (de steps_log)
"""
from __future__ import annotations

import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sqlalchemy import text

from ..config import ExportConfig
from ..utils.db import (
    get_episodios_log,
    get_episodios_referencia,
    get_log_episodio,
    get_posicoes_todas,
    get_treino_metadata,
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_VERSAO_EXPORT = "3.0"
_SPARKS = " ▁▂▃▄▅▆▇█"

KAN_INPUT_NAMES = [
    "energia_actual", "energia_max", "tempo_de_vida",
    "comida_vista_50steps", "distancia_media_comida", "steps_sem_comer",
]
KAN_OUTPUT_NAMES = ["bias_explorar", "bias_exploitar"]


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _nan_safe(v: Any) -> Any:
    """Converte NaN/inf para None para JSON válido."""
    if v is None:
        return None
    try:
        if math.isnan(v) or math.isinf(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    """DataFrame → lista de dicts com NaN sanitizado."""
    records = []
    for row in df.to_dict(orient="records"):
        records.append({k: _nan_safe(v) for k, v in row.items()})
    return records


def _sparkline(values: list[float]) -> str:
    """Converte lista de floats em sparkline unicode."""
    if not values:
        return ""
    mn, mx = min(values), max(values)
    rng = mx - mn if mx > mn else 1e-9
    return "".join(_SPARKS[min(8, int((v - mn) / rng * 8))] for v in values)


def _plotext_ascii(x_vals: list[float], y_vals: list[float], title: str = "",
                   width: int = 50, height: int = 10) -> str:
    """Gera gráfico ASCII via plotext para uma curva (x, y), sem códigos ANSI."""
    import re
    try:
        import plotext as plt
        plt.clf()
        plt.plot(x_vals, y_vals)
        if title:
            plt.title(title)
        plt.plotsize(width, height)
        plt.xaxes(True, False)
        plt.yaxes(True, False)
        raw = plt.build()
        # Remover escape codes ANSI para JSON limpo
        return re.sub(r'\x1b\[[0-9;]*m', '', raw)
    except Exception as e:
        return f"[plotext indisponível: {e}]"


# ---------------------------------------------------------------------------
# Secção: meta
# ---------------------------------------------------------------------------

def _build_meta(cfg: ExportConfig) -> dict:
    return {
        "versao_export":      _VERSAO_EXPORT,
        "exportado_em":       datetime.now(timezone.utc).isoformat(),
        "treino_id":          cfg.treino_id,
        "ficheiro":           cfg.output_file.name,
    }


# ---------------------------------------------------------------------------
# Secção: treino
# ---------------------------------------------------------------------------

def _build_treino(meta: dict) -> dict:
    return {
        "id":             meta.get("id") or meta.get("treino_id", ""),
        "data_human":     meta.get("data_human", ""),
        "nota":           meta.get("nota", ""),
        "conclusao":      meta.get("conclusao", ""),
        "estado":         meta.get("estado", ""),
        "parametros":     meta.get("parametros") or {},
        "metricas_finais": meta.get("metricas_finais") or {},
        "mlflow_run_id":  meta.get("mlflow_run_id"),
    }


# ---------------------------------------------------------------------------
# Secção: resumo
# ---------------------------------------------------------------------------

def _build_resumo(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"total_episodios": 0, "aviso": "sem dados em episodios_log"}

    tv = df["tempo_vida"].tolist()
    ef = df["eficiencia"].tolist()
    prec = df["precisao_direcional"].dropna().tolist()
    pers = df["taxa_perseguicao"].dropna().tolist()

    # Mudanças de fase
    mudancas: dict[str, int | None] = {"1_para_2": None, "2_para_3": None}
    prev = None
    for _, row in df.iterrows():
        fase = int(row["fase_curriculum"])
        if prev is not None and fase != prev:
            if fase == 2 and mudancas["1_para_2"] is None:
                mudancas["1_para_2"] = int(row["episodio"])
            elif fase == 3 and mudancas["2_para_3"] is None:
                mudancas["2_para_3"] = int(row["episodio"])
        prev = fase

    # Episódio de estabilização (taxa_inferencia > 0.7 por 20 eps consecutivos)
    estab = None
    ti = df["taxa_inferencia"].fillna(0)
    rolling = ti.rolling(window=20, min_periods=20).mean()
    mask = rolling >= 0.7
    if mask.any():
        estab = int(df.loc[mask.idxmax(), "episodio"])

    melhor_ep = int(df.loc[df["tempo_vida"].idxmax(), "episodio"])

    return {
        "total_episodios":       len(df),
        "fase_maxima_atingida":  int(df["fase_curriculum"].max()),
        "mudancas_fase":         mudancas,
        "melhor_episodio":       melhor_ep,
        "tempo_vida": {
            "min":     int(min(tv)),
            "max":     int(max(tv)),
            "media":   round(statistics.mean(tv), 2),
            "mediana": round(statistics.median(tv), 2),
            "std":     round(statistics.stdev(tv) if len(tv) > 1 else 0.0, 2),
        },
        "eficiencia": {
            "min":    round(min(ef), 4),
            "max":    round(max(ef), 4),
            "media":  round(statistics.mean(ef), 4),
        },
        "precisao_direcional": {
            "media":          round(statistics.mean(prec), 4) if prec else None,
            "media_ult_50ep": round(float(df["precisao_direcional"].tail(50).mean()), 4) if prec else None,
            "max":            round(max(prec), 4) if prec else None,
        },
        "taxa_perseguicao": {
            "media":          round(statistics.mean(pers), 4) if pers else None,
            "media_ult_50ep": round(float(df["taxa_perseguicao"].tail(50).mean()), 4) if pers else None,
            "max":            round(max(pers), 4) if pers else None,
        },
        "taxa_inferencia": {
            "media_total":     round(float(ti.mean()), 4),
            "media_ult_100ep": round(float(ti.tail(100).mean()), 4),
        },
        "convergencia": {
            "episodio_estabilizacao": estab,
            "criterio": "taxa_inferencia > 0.7 por 20 episodios consecutivos",
        },
    }


# ---------------------------------------------------------------------------
# Secção: comportamento
# ---------------------------------------------------------------------------

def _build_comportamento(cfg: ExportConfig) -> dict:
    result: dict[str, Any] = {"episodios_referencia": {}, "logs": {}, "eventos_criticos": []}

    try:
        with cfg.engine.connect() as conn:
            refs = get_episodios_referencia(cfg.treino_id, conn)
            result["episodios_referencia"] = refs

            for label, episodio in refs.items():
                if episodio is None:
                    result["logs"][label] = []
                    continue
                df = get_log_episodio(
                    cfg.treino_id, episodio, conn,
                    max_steps=cfg.max_steps_por_episodio,
                )
                result["logs"][label] = _df_to_records(df)

            # Eventos críticos
            eventos: list[dict] = []
            for tipo, where in [
                ("comeu",          "comeu = true"),
                ("quase_morreu",   "energia_actual < 10"),
                ("novidade_alta",  "novidade > 0.8"),
            ]:
                sql = text(f"""
                    SELECT treino_id, episodio, step, pos_x, pos_y,
                           energia_actual, modo, novidade, comeu
                    FROM steps_log
                    WHERE treino_id = :tid AND ({where})
                    ORDER BY episodio, step
                    LIMIT 10000
                """)
                df_ev = pd.read_sql(sql, conn, params={"tid": cfg.treino_id})
                for row in _df_to_records(df_ev):
                    row["tipo_evento"] = tipo
                    eventos.append(row)
            result["eventos_criticos"] = sorted(eventos, key=lambda r: (r["episodio"], r["step"]))
    except Exception as e:
        result["erro"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Secção: snn
# ---------------------------------------------------------------------------

def _build_snn(cfg: ExportConfig, meta: dict) -> dict:
    result: dict[str, Any] = {"agmp": [], "spike_rates": [], "evolucao_pesos": []}

    # AGMP histórico: novidade por episódio (de steps_log quando disponível)
    try:
        sql_agmp = text("""
            SELECT
                episodio,
                AVG(novidade)  AS novidade_media,
                MAX(novidade)  AS novidade_max,
                MIN(novidade)  AS novidade_min,
                SUM(CASE WHEN modo='aprendizagem' THEN 1 ELSE 0 END)
                    * 100.0 / COUNT(*) AS pct_steps_aprendizagem
            FROM steps_log
            WHERE treino_id = :tid
            GROUP BY episodio
            ORDER BY episodio
        """)
        with cfg.engine.connect() as conn:
            df_agmp = pd.read_sql(sql_agmp, conn, params={"tid": cfg.treino_id})
        result["agmp"] = _df_to_records(df_agmp)
    except Exception as e:
        result["agmp_erro"] = str(e)

    # Modelos .pt para spike_rates e evolucao_pesos
    device = torch.device("cpu")
    artifact_root = Path(os.getenv("MLFLOW_ARTIFACT_ROOT", "./mlartifacts"))
    local_dir = artifact_root / cfg.treino_id

    def _find_pt(names: list[str]) -> Path | None:
        for name in names:
            p = local_dir / name
            if p.exists():
                return p
        return None

    pt_inicial = _find_pt(["snn_ep0.pt", "snn_ep25.pt", "snn_ep50.pt"])
    pt_final   = _find_pt(["snn_ep500.pt", "snn_ep400.pt", "snn_ep300.pt",
                            "snn_ep200.pt", "snn_ep100.pt", "snn_ep50.pt",
                            "snn_ep1500.pt"])

    # Spike rates
    if pt_final:
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parents[4]))
            from snn import OrganismoSNN
            from snn.encoding import observacao_para_tensor, T_DEFAULT
            from world import Mundo

            snn_model = OrganismoSNN(device=device).to(device)
            snn_model.load_state_dict(torch.load(pt_final, map_location=device), strict=False)
            snn_model.eval()

            mundo = Mundo(fase=1, seed=0)
            obs = mundo.reset()
            buffers: dict[str, list] = {"oculta1_lif": [], "oculta2_rlif": [], "output": []}

            def _hook(name):
                def h(module, inp, out):
                    buffers[name].append(out.detach().cpu())
                return h

            h1 = snn_model.lif1.register_forward_hook(_hook("oculta1_lif"))
            h2 = snn_model.lif2.register_forward_hook(_hook("oculta2_rlif"))
            h3 = snn_model.lif3.register_forward_hook(_hook("output"))
            try:
                with torch.no_grad():
                    for _ in range(300):
                        x = observacao_para_tensor(obs, T=T_DEFAULT, device=device)
                        snn_model(x)
                        obs, _, done = mundo.step("avancar")
                        if done:
                            obs = mundo.reset()
            finally:
                h1.remove(); h2.remove(); h3.remove()

            rows = []
            for layer_name, tensors in buffers.items():
                if not tensors:
                    continue
                stacked = torch.stack([t.view(-1) for t in tensors]).float()
                for idx in range(stacked.shape[1]):
                    s = stacked[:, idx]
                    rows.append({
                        "camada": layer_name,
                        "neuronio_idx": idx,
                        "spike_rate_medio": round(float(s.mean()), 4),
                        "spike_rate_std": round(float(s.std()), 4),
                        "pct_silencio": round(float((s == 0).float().mean() * 100), 2),
                    })
            result["spike_rates"] = rows
        except Exception as e:
            result["spike_rates_erro"] = str(e)

    # Evolução de pesos
    if pt_inicial and pt_final:
        try:
            s0 = torch.load(pt_inicial, map_location="cpu")
            sf = torch.load(pt_final,   map_location="cpu")
            rows = []
            for key in s0:
                if key not in sf:
                    continue
                w0 = s0[key].float().flatten()
                wf = sf[key].float().flatten()
                if w0.shape != wf.shape:
                    continue
                delta = (wf - w0).abs()
                rows.append({
                    "camada":          key,
                    "n_pesos":         len(w0),
                    "delta_medio":     round(float(delta.mean()), 6),
                    "delta_max":       round(float(delta.max()), 6),
                    "pct_sinal_mudou": round(float(((w0 * wf) < 0).float().mean() * 100), 2),
                    "norma_inicial":   round(float(w0.norm()), 4),
                    "norma_final":     round(float(wf.norm()), 4),
                })
            result["evolucao_pesos"] = rows
        except Exception as e:
            result["evolucao_pesos_erro"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Secção: kan
# ---------------------------------------------------------------------------

def _build_kan(cfg: ExportConfig, df_eps: pd.DataFrame) -> dict:
    result: dict[str, Any] = {
        "inputs":     KAN_INPUT_NAMES,
        "outputs":    KAN_OUTPUT_NAMES,
        "historico":  [],
        "correlacoes": [],
        "sensibilidade": {},
    }

    # Histórico por episódio (de episodios_log)
    if not df_eps.empty and "bias_explorar_medio" in df_eps.columns:
        cols = ["episodio", "bias_explorar_medio", "bias_exploitar_medio",
                "taxa_perseguicao"]
        available = [c for c in cols if c in df_eps.columns]
        df_hist = df_eps[available].copy()
        if "bias_explorar_medio" in df_hist and "bias_exploitar_medio" in df_hist:
            df_hist["ratio"] = (
                df_hist["bias_explorar_medio"] /
                (df_hist["bias_exploitar_medio"].fillna(0) + 0.001)
            )
            df_hist["dominante"] = df_hist["ratio"].apply(
                lambda r: "explorar" if r > 1.2 else ("exploitar" if r < 0.8 else "equilibrado")
            )
        result["historico"] = _df_to_records(df_hist)

    # Correlações inputs → outputs (de episodios_log via steps_log proxy)
    try:
        from scipy import stats as sp_stats
        sql_feat = text("""
            SELECT
                episodio,
                AVG(energia_actual)  AS energia_actual,
                AVG(energia_max)     AS energia_max,
                COUNT(*)             AS tempo_de_vida
            FROM steps_log
            WHERE treino_id = :tid
            GROUP BY episodio
            ORDER BY episodio
        """)
        with cfg.engine.connect() as conn:
            df_feat = pd.read_sql(sql_feat, conn, params={"tid": cfg.treino_id})

        if not df_eps.empty and not df_feat.empty:
            df_m = df_eps.merge(df_feat, on="episodio", how="inner")
            corr_rows = []
            for inp in ["energia_actual", "energia_max", "tempo_de_vida"]:
                if inp not in df_m.columns:
                    continue
                for out in ["bias_explorar_medio", "bias_exploitar_medio"]:
                    if out not in df_m.columns:
                        continue
                    x = df_m[inp].dropna()
                    y = df_m[out].loc[x.index].dropna()
                    common = x.index.intersection(y.index)
                    if len(common) < 5:
                        continue
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        r, p = sp_stats.pearsonr(x.loc[common], y.loc[common])
                    corr_rows.append({
                        "input":  inp, "output": out,
                        "pearson_r": round(r, 4), "pearson_p": round(p, 6),
                        "forca": ("forte+" if r > 0.5 else "fraco+" if r > 0.2
                                  else "forte-" if r < -0.5 else "fraco-" if r < -0.2
                                  else "neutro"),
                    })
            result["correlacoes"] = corr_rows
    except Exception:
        pass

    # Sensibilidade: variar cada input e registar outputs com plotext
    artifact_root = Path(os.getenv("MLFLOW_ARTIFACT_ROOT", "./mlartifacts"))
    kan_path = None
    for name in ["kan_ep50.pt", "kan_ep25.pt", "kan_ep100.pt",
                 "kan_ep200.pt", "kan_ep500.pt", "kan_ep1500.pt", "kan_final.pt"]:
        p = artifact_root / cfg.treino_id / name
        if p.exists():
            kan_path = p
            break

    if kan_path:
        try:
            from kan import KAN as _KAN
            kan_model = _KAN(width=[6, 8, 2], grid=5, k=3,
                             auto_save=False, device="cpu")
            kan_model.load_state_dict(
                torch.load(kan_path, map_location="cpu"), strict=False
            )
            kan_model.eval()

            N_SWEEP = 25
            x_sweep = [i / (N_SWEEP - 1) for i in range(N_SWEEP)]
            sensib: dict[str, dict] = {}

            with torch.no_grad():
                for i, inp_name in enumerate(KAN_INPUT_NAMES):
                    baseline = [0.5] * 6
                    sensib[inp_name] = {}
                    for j, out_name in enumerate(KAN_OUTPUT_NAMES):
                        y_vals: list[float] = []
                        for xv in x_sweep:
                            vec = baseline.copy()
                            vec[i] = xv
                            t = torch.tensor([vec], dtype=torch.float32)
                            out = kan_model(t)
                            y_vals.append(float(out[0, j].item()))

                        sensib[inp_name][out_name] = {
                            "sparkline":    _sparkline(y_vals),
                            "plotext":      _plotext_ascii(
                                x_sweep, y_vals,
                                title=f"{inp_name} → {out_name}",
                                width=50, height=8,
                            ),
                            "valores":      [round(v, 4) for v in y_vals],
                            "y_min":        round(min(y_vals), 4),
                            "y_max":        round(max(y_vals), 4),
                            "variacao":     round(max(y_vals) - min(y_vals), 4),
                        }

            result["sensibilidade"] = sensib
            result["kan_modelo_usado"] = kan_path.name
        except Exception as e:
            result["sensibilidade_erro"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Secção: espacial
# ---------------------------------------------------------------------------

def _build_espacial(cfg: ExportConfig) -> dict:
    result: dict[str, Any] = {
        "heatmap_posicoes": [],
        "eventos_alimentacao": [],
        "trajectórias": {},
    }

    try:
        with cfg.engine.connect() as conn:
            df_hm = get_posicoes_todas(cfg.treino_id, conn, resolucao_celula=10)
            result["heatmap_posicoes"] = _df_to_records(df_hm)

            sql_comida = text("""
                SELECT episodio, step, pos_x, pos_y,
                       energia_actual AS energia_depois, fase_mundo
                FROM steps_log
                WHERE treino_id = :tid AND comeu = true
                ORDER BY episodio, step
            """)
            df_comida = pd.read_sql(sql_comida, conn, params={"tid": cfg.treino_id})
            result["eventos_alimentacao"] = _df_to_records(df_comida)

            refs = get_episodios_referencia(cfg.treino_id, conn)
            for label, episodio in refs.items():
                if episodio is None:
                    result["trajectórias"][label] = []
                    continue
                df_traj = get_log_episodio(
                    cfg.treino_id, episodio, conn,
                    max_steps=cfg.max_steps_por_episodio,
                )
                cols = ["step", "pos_x", "pos_y", "angulo", "accao", "comeu"]
                available = [c for c in cols if c in df_traj.columns]
                result["trajectórias"][label] = _df_to_records(df_traj[available]) if not df_traj.empty else []
    except Exception as e:
        result["erro"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Exportador principal
# ---------------------------------------------------------------------------

def exportar(cfg: ExportConfig) -> list[Path]:
    print(f"  [json] A construir {cfg.output_file.name} ...")

    with cfg.engine.connect() as conn:
        meta     = get_treino_metadata(cfg.treino_id, conn)
        df_eps   = get_episodios_log(cfg.treino_id, conn)

    if df_eps.empty:
        print("  [json] Aviso: episodios_log vazio — treino sem dados de episódios")

    print(f"  [json] episodios_log: {len(df_eps)} episódios")

    output = {
        "meta":           _build_meta(cfg),
        "treino":         _build_treino(meta),
        "resumo":         _build_resumo(df_eps),
        "episodios":      _df_to_records(df_eps),
        "comportamento":  _build_comportamento(cfg),
        "snn":            _build_snn(cfg, meta),
        "kan":            _build_kan(cfg, df_eps),
        "espacial":       _build_espacial(cfg),
    }

    # Escrever JSON
    json_str = json.dumps(output, ensure_ascii=False, indent=2, default=str)
    cfg.output_file.write_text(json_str, encoding="utf-8")

    size_kb = cfg.output_file.stat().st_size / 1024
    print(f"  [json] {cfg.output_file.name} — {size_kb:.1f} KB")

    return [cfg.output_file]
