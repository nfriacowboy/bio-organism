"""
Exportador 3 — Análise interna da SNN.

Gera:
  snn/spike_rates_por_camada.csv
  snn/evolucao_pesos.csv
  snn/agmp_historico.csv
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import torch

from ..config import ExportConfig
from ..utils.db import get_treino_metadata
from ..utils import mlflow_client as mf


# ---------------------------------------------------------------------------
# Helpers de análise de spikes
# ---------------------------------------------------------------------------

def _analise_spikes(snn, device: torch.device, n_steps: int = 500) -> pd.DataFrame:
    """
    Corre o organismo em modo inferência por n_steps (Fase 1, seed=0) e
    regista a taxa de spikes de cada neurônio por camada.
    Retorna DataFrame com colunas: camada, neuronio_idx, spike_rate_medio,
    spike_rate_std, percentagem_silencio.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parents[4]))  # trainer/src

    from snn.encoding import observacao_para_tensor, T_DEFAULT
    from world import Mundo

    snn.eval()
    mundo = Mundo(fase=1, seed=0)
    obs = mundo.reset()

    buffers: dict[str, list[torch.Tensor]] = {
        "oculta1_lif": [], "oculta2_rlif": [], "output": []
    }

    def _make_hook(name):
        def hook(module, inp, out):
            buffers[name].append(out.detach().cpu())
        return hook

    h1 = snn.lif1.register_forward_hook(_make_hook("oculta1_lif"))
    h2 = snn.lif2.register_forward_hook(_make_hook("oculta2_rlif"))
    h3 = snn.lif3.register_forward_hook(_make_hook("output"))

    try:
        with torch.no_grad():
            for _ in range(n_steps):
                x = observacao_para_tensor(obs, T=T_DEFAULT, device=device)
                snn(x)
                acc = ["avancar", "virar_esquerda", "virar_direita", "parar"]
                obs, _, done = mundo.step(acc[0])
                if done:
                    obs = mundo.reset()
    finally:
        h1.remove(); h2.remove(); h3.remove()

    rows = []

    def _stats_layer(name: str, tensors: list[torch.Tensor]) -> None:
        if not tensors:
            return
        # Cada tensor: [batch=1, N_neurons] ou [N_neurons]
        stacked = torch.stack([t.view(-1) for t in tensors])  # [n_steps, N]
        for idx in range(stacked.shape[1]):
            s = stacked[:, idx].float()
            rows.append({
                "camada":              name,
                "neuronio_idx":        idx,
                "spike_rate_medio":    float(s.mean()),
                "spike_rate_std":      float(s.std()),
                "percentagem_silencio": float((s == 0).float().mean() * 100),
            })

    for name, buf in buffers.items():
        _stats_layer(name, buf)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Exportador principal
# ---------------------------------------------------------------------------

def exportar(cfg: ExportConfig) -> list[Path]:
    generated: list[Path] = []
    out_dir = cfg.cat("snn")

    # Tentar carregar modelos — primeiro local, depois MLflow
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with cfg.engine.connect() as conn:
        meta = get_treino_metadata(cfg.treino_id, conn)

    import os
    artifact_root = Path(os.getenv("MLFLOW_ARTIFACT_ROOT", "./mlartifacts"))
    local_dir = artifact_root / cfg.treino_id

    def _local_or_mlflow(names: list[str]) -> Path | None:
        # 1. Procurar ficheiro local
        for name in names:
            p = local_dir / name
            if p.exists():
                return p
        # 2. Tentar download MLflow (com timeout curto)
        run_id_val = meta.get("mlflow_run_id") or mf.get_run_id(cfg.treino_id)
        if run_id_val:
            tmp = out_dir / "_modelos"
            for name in names:
                try:
                    import signal
                    def _timeout(signum, frame):
                        raise TimeoutError("MLflow download timeout")
                    signal.signal(signal.SIGALRM, _timeout)
                    signal.alarm(10)
                    p = mf.download_artifact(run_id_val, name, tmp)
                    signal.alarm(0)
                    if p and p.exists():
                        return p
                except (TimeoutError, Exception):
                    signal.alarm(0)
        return None

    model_inicial_path = _local_or_mlflow(["snn_ep0.pt", "snn_ep25.pt", "snn_ep50.pt"])
    model_final_path   = _local_or_mlflow(["snn_ep500.pt", "snn_ep400.pt", "snn_ep300.pt",
                                            "snn_ep200.pt", "snn_ep100.pt", "snn_ep50.pt",
                                            "snn_ep25.pt"])

    # -----------------------------------------------------------------------
    # 1. spike_rates_por_camada.csv
    # -----------------------------------------------------------------------
    sr_path = out_dir / "spike_rates_por_camada.csv"
    snn_loaded = False

    if model_final_path and model_final_path.exists():
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parents[4]))
            from snn import OrganismoSNN
            snn_model = OrganismoSNN(device=device).to(device)
            state = torch.load(model_final_path, map_location=device)
            snn_model.load_state_dict(state, strict=False)
            df_spikes = _analise_spikes(snn_model, device)
            df_spikes.to_csv(sr_path, index=False, float_format="%.4f")
            snn_loaded = True
            print(f"  [snn] {sr_path.name} — {len(df_spikes)} neurónios analisados")
        except Exception as e:
            print(f"  [snn] spike_rates: erro ao carregar modelo: {e}")

    if not snn_loaded:
        pd.DataFrame(columns=[
            "camada", "neuronio_idx", "spike_rate_medio",
            "spike_rate_std", "percentagem_silencio",
        ]).to_csv(sr_path, index=False)
        print(f"  [snn] {sr_path.name} — vazio (modelo não disponível)")
    generated.append(sr_path)

    # -----------------------------------------------------------------------
    # 2. evolucao_pesos.csv
    # -----------------------------------------------------------------------
    pw_path = out_dir / "evolucao_pesos.csv"
    rows = []

    if model_inicial_path and model_final_path and \
       model_inicial_path.exists() and model_final_path.exists():
        try:
            s0 = torch.load(model_inicial_path, map_location="cpu")
            sf = torch.load(model_final_path,   map_location="cpu")
            for key in s0:
                if key not in sf:
                    continue
                w0 = s0[key].float().flatten()
                wf = sf[key].float().flatten()
                if w0.shape != wf.shape:
                    continue
                delta = (wf - w0)
                for idx, (wi, wfinal, d) in enumerate(
                        zip(w0.tolist(), wf.tolist(), delta.tolist())):
                    rows.append({
                        "camada":          key,
                        "ligacao_idx":     idx,
                        "peso_inicial":    round(wi, 6),
                        "peso_final":      round(wfinal, 6),
                        "delta_absoluto":  round(abs(d), 6),
                        "delta_percentual": round(abs(d) / (abs(wi) + 1e-8) * 100, 2),
                        "sinal_mudou":     (wi * wfinal < 0),
                    })
            print(f"  [snn] {pw_path.name} — {len(rows)} pesos comparados")
        except Exception as e:
            print(f"  [snn] evolucao_pesos: {e}")

    if rows:
        pd.DataFrame(rows).to_csv(pw_path, index=False, float_format="%.6f")
    else:
        pd.DataFrame(columns=[
            "camada", "ligacao_idx", "peso_inicial", "peso_final",
            "delta_absoluto", "delta_percentual", "sinal_mudou",
        ]).to_csv(pw_path, index=False)
        print(f"  [snn] {pw_path.name} — vazio (modelos não disponíveis)")
    generated.append(pw_path)

    # -----------------------------------------------------------------------
    # 3. agmp_historico.csv
    # -----------------------------------------------------------------------
    agmp_path = out_dir / "agmp_historico.csv"
    sql_agmp = """
        SELECT
            episodio,
            AVG(novidade)     AS novidade_media,
            MAX(novidade)     AS novidade_max,
            MIN(novidade)     AS novidade_min,
            SUM(CASE WHEN modo='aprendizagem' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS pct_steps_aprendizagem
        FROM steps_log
        WHERE treino_id = :tid
        GROUP BY episodio
        ORDER BY episodio
    """
    from sqlalchemy import text
    with cfg.engine.connect() as conn:
        df_agmp = pd.read_sql(text(sql_agmp), conn, params={"tid": cfg.treino_id})

    if not df_agmp.empty:
        # Calcular n_mudancas_modo, primeiro_step_inferencia, ultimo_step_aprendizagem
        # requer query adicional por episódio — demasiado caro para todos;
        # deixamos como null aqui e sugerimos query separada se necessário
        df_agmp["n_mudancas_modo"] = None
        df_agmp["primeiro_step_inferencia"] = None
        df_agmp["ultimo_step_aprendizagem"] = None
        df_agmp.to_csv(agmp_path, index=False, float_format="%.4f")
        print(f"  [snn] {agmp_path.name} — {len(df_agmp)} episódios")
    else:
        pd.DataFrame(columns=[
            "episodio", "novidade_media", "novidade_max", "novidade_min",
            "pct_steps_aprendizagem", "n_mudancas_modo",
            "primeiro_step_inferencia", "ultimo_step_aprendizagem",
        ]).to_csv(agmp_path, index=False)
        print(f"  [snn] {agmp_path.name} — vazio (sem dados)")
    generated.append(agmp_path)

    return generated
