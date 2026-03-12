"""
run.py — Entrypoint principal do módulo de exportação SNN Organismo.

Uso:
  uv run python -m export.run --treino_id 2026-03-15T14:32:00Z
  uv run python -m export.run --treino_id 2026-03-15T14:32:00Z --apenas metricas comportamento
  uv run python -m export.run --treino_id 2026-03-15T14:32:00Z --episodios_amostra 10
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exporta dados de treino da SNN Organismo para análise externa."
    )
    parser.add_argument("--treino_id", required=True,
                        help="ID do treino a exportar (timestamp ISO8601)")
    parser.add_argument("--output_dir", default="exports",
                        help="Directoria raiz para os ficheiros exportados (default: exports/)")
    parser.add_argument("--apenas", nargs="*",
                        choices=["metricas", "comportamento", "snn", "kan", "espacial"],
                        default=None,
                        help="Lista de exportadores a correr (default: todos)")
    parser.add_argument("--episodios_amostra", type=int, default=5,
                        help="Quantos episódios de amostra exportar (default: 5)")
    parser.add_argument("--max_steps_por_episodio", type=int, default=50_000,
                        help="Limite de steps por episódio (default: 50000)")
    parser.add_argument("--incluir_modelos", action="store_true",
                        help="Se presente, copia os .pt dos modelos para o output")
    args = parser.parse_args()

    # --- Configuração ---
    from .config import ExportConfig
    try:
        cfg = ExportConfig(
            treino_id=args.treino_id,
            output_dir=args.output_dir,
            apenas=args.apenas,
            episodios_amostra=args.episodios_amostra,
            max_steps_por_episodio=args.max_steps_por_episodio,
            incluir_modelos=args.incluir_modelos,
        )
    except (EnvironmentError, ValueError) as e:
        print(f"[export] Erro de configuração: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[export] treino_id = {cfg.treino_id}")
    print(f"[export] output    = {cfg.output_dir}")
    print(f"[export] exporters = {sorted(cfg.apenas)}")
    print()

    t_inicio = time.monotonic()
    todos_gerados: list[Path] = []

    # --- Executar exportadores por prioridade ---
    ORDEM = [
        ("metricas",    ".exporters.metricas",    "Exportador 1 — Métricas"),
        ("comportamento", ".exporters.comportamento", "Exportador 2 — Comportamento"),
        ("snn",         ".exporters.snn_analise", "Exportador 3 — SNN"),
        ("kan",         ".exporters.kan_analise", "Exportador 4 — KAN"),
        ("espacial",    ".exporters.espacial",    "Exportador 5 — Espacial"),
    ]

    for nome, modulo_rel, label in ORDEM:
        if not cfg.enabled(nome):
            continue
        print(f"--- {label} ---")
        try:
            import importlib
            mod = importlib.import_module(modulo_rel, package="export")
            ficheiros = mod.exportar(cfg)
            todos_gerados.extend(ficheiros)
        except Exception as e:
            print(f"  [ERRO] {nome}: {e}")
            import traceback
            traceback.print_exc()
        print()

    # --- Resumo final ---
    elapsed = time.monotonic() - t_inicio
    tamanho_total = sum(p.stat().st_size for p in todos_gerados if p.exists())
    tamanho_mb = tamanho_total / 1024 / 1024

    print("=" * 60)
    print(f"  Ficheiros gerados : {len(todos_gerados)}")
    print(f"  Tamanho total     : {tamanho_mb:.2f} MB")
    print(f"  Tempo de execução : {elapsed:.1f}s")
    print(f"  Output            : {cfg.output_dir.resolve()}")
    print("=" * 60)

    # Listar ficheiros por categoria
    for nome, _, _ in ORDEM:
        cat_dir = cfg.output_dir / nome
        if not cat_dir.exists():
            continue
        ficheiros_cat = sorted(cat_dir.iterdir())
        if ficheiros_cat:
            print(f"\n  {nome}/")
            for f in ficheiros_cat:
                if f.name.startswith("_"):
                    continue
                size_kb = f.stat().st_size / 1024 if f.exists() else 0
                print(f"    {f.name:<45} {size_kb:6.1f} KB")


if __name__ == "__main__":
    main()
