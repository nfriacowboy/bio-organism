"""
validate.py — Verifica se todos os ficheiros foram gerados correctamente.

Uso:
  uv run python -m export.validate --treino_id <ID>
  uv run python -m export.validate --treino_id <ID> --output_dir exports/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Colunas esperadas por ficheiro CSV
COLUNAS_METRICAS = [
    "episodio", "tempo_vida", "tempo_vida_media_10ep", "comida_total",
    "eficiencia", "recompensa_total", "steps_aprendizagem", "steps_inferencia",
    "taxa_inferencia", "novidade_media", "fase_curriculum",
]
COLUNAS_LOG = [
    "step", "pos_x", "pos_y", "angulo", "energia_actual",
    "accao", "modo", "recompensa", "comeu",
]

FICHEIROS_ESPERADOS: dict[str, list[str] | None] = {
    "metricas/metricas_por_episodio.csv":       COLUNAS_METRICAS,
    "metricas/resumo_treino.json":              ["total_episodios", "tempo_vida", "convergencia"],
    "metricas/progressao_fases.csv":            ["episodio_mudanca", "fase_anterior", "fase_nova"],
    "comportamento/log_ep_inicial.csv":         COLUNAS_LOG,
    "comportamento/log_ep_melhor.csv":          COLUNAS_LOG,
    "comportamento/eventos_criticos.csv":       ["tipo_evento", "episodio", "step"],
    "snn/spike_rates_por_camada.csv":           ["camada", "neuronio_idx", "spike_rate_medio"],
    "snn/evolucao_pesos.csv":                   ["camada", "peso_inicial", "peso_final"],
    "snn/agmp_historico.csv":                   ["episodio", "novidade_media", "pct_steps_aprendizagem"],
    "kan/kan_outputs_historico.csv":            ["episodio", "bias_explorar_medio"],
    "kan/kan_splines_visualization.png":        None,   # apenas existência + tamanho > 0
    "kan/kan_inputs_correlacao.csv":            ["input_nome", "output_nome", "pearson_r"],
    "espacial/heatmap_posicoes.csv":            ["celula_x", "celula_y", "visitas_total"],
    "espacial/heatmap_comida_consumida.csv":    ["episodio", "pos_x", "pos_y"],
}


def _check_csv(path: Path, colunas_req: list[str]) -> list[str]:
    erros: list[str] = []
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return [f"Não foi possível ler CSV: {e}"]
    for col in colunas_req:
        if col not in df.columns:
            erros.append(f"coluna '{col}' em falta")
    return erros


def _check_json(path: Path, chaves_req: list[str]) -> list[str]:
    erros: list[str] = []
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        return [f"JSON inválido: {e}"]
    for k in chaves_req:
        if k not in data:
            erros.append(f"chave '{k}' em falta")
    return erros


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Valida os ficheiros exportados do treino SNN Organismo."
    )
    parser.add_argument("--treino_id", required=True)
    parser.add_argument("--output_dir", default="exports")
    args = parser.parse_args()

    base = Path(args.output_dir) / args.treino_id
    if not base.exists():
        # Encontrar pasta com timestamp
        candidates = sorted(Path(args.output_dir).glob(f"{args.treino_id}*"))
        if candidates:
            base = candidates[-1]
        else:
            print(f"[validate] Pasta não encontrada: {base}", file=sys.stderr)
            sys.exit(1)

    print(f"[validate] base: {base}")
    print()

    total = 0
    ok    = 0
    falhas: list[tuple[str, list[str]]] = []

    for rel_path, colunas in FICHEIROS_ESPERADOS.items():
        total += 1
        fpath = base / rel_path
        erros: list[str] = []

        # 1. Existência
        if not fpath.exists():
            erros.append("ficheiro não existe")
        elif fpath.stat().st_size == 0:
            erros.append("ficheiro vazio (0 bytes)")
        elif colunas is None:
            pass   # PNG — verificação de existência já feita
        elif rel_path.endswith(".csv"):
            erros.extend(_check_csv(fpath, colunas))
        elif rel_path.endswith(".json"):
            erros.extend(_check_json(fpath, colunas))

        if erros:
            falhas.append((rel_path, erros))
            print(f"  ✗  {rel_path}")
            for e in erros:
                print(f"       → {e}")
        else:
            ok += 1
            size_kb = fpath.stat().st_size / 1024
            print(f"  ✓  {rel_path:<55} ({size_kb:.1f} KB)")

    print()
    print(f"Resultado: {ok}/{total} ficheiros OK")
    if falhas:
        print(f"Falhas: {len(falhas)} ficheiro(s)")
        sys.exit(1)
    else:
        print("Todos os ficheiros validados com sucesso.")
        sys.exit(0)


if __name__ == "__main__":
    main()
