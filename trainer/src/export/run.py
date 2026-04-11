"""
run.py — Entrypoint principal do módulo de exportação SNN Organismo.

Produz um único ficheiro JSON com todos os dados do treino.

Uso CLI:
  uv run python -m export.run --treino_id 2026-03-15T14:32:00Z
  uv run python -m export.run --treino_id 2026-03-15T14:32:00Z --output_dir ../exports

Uso programático (chamado pelo trainer no fim de cada treino):
  from export.run import run_export
  run_export(treino_id="2026-03-15T14:32:00Z", output_dir="exports")
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Lógica de execução
# ---------------------------------------------------------------------------

def _executar(cfg) -> list[Path]:
    print(f"[export] treino_id = {cfg.treino_id}")
    print(f"[export] output    = {cfg.output_dir}")
    print(f"[export] ficheiro  = {cfg.output_file.name}")
    print()

    t_inicio = time.monotonic()

    from .exporters.json_unico import exportar
    gerados = exportar(cfg)

    elapsed = time.monotonic() - t_inicio
    tamanho_total = sum(p.stat().st_size for p in gerados if p.exists())

    print()
    print("=" * 60)
    print(f"  Ficheiros gerados : {len(gerados)}")
    print(f"  Tamanho total     : {tamanho_total / 1024:.1f} KB")
    print(f"  Tempo de execução : {elapsed:.1f}s")
    print(f"  Output            : {cfg.output_dir.resolve()}")
    print("=" * 60)
    for p in gerados:
        if p.exists():
            print(f"    {p.name}")

    return gerados


# ---------------------------------------------------------------------------
# API programática (chamada pelo trainer)
# ---------------------------------------------------------------------------

def run_export(
    treino_id: str,
    output_dir: str = "exports",
    episodios_amostra: int = 5,
    max_steps_por_episodio: int = 50_000,
) -> None:
    """
    Exporta os dados de um treino para um único ficheiro JSON.

    Chamada automaticamente pelo trainer no fim de cada treino quando
    DATABASE_URL está configurada.
    """
    from .config import ExportConfig
    cfg = ExportConfig(
        treino_id=treino_id,
        output_dir=output_dir,
        episodios_amostra=episodios_amostra,
        max_steps_por_episodio=max_steps_por_episodio,
    )
    _executar(cfg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exporta dados de treino da SNN Organismo para um único ficheiro JSON."
    )
    parser.add_argument("--treino_id", required=True,
                        help="ID do treino a exportar (timestamp ISO8601)")
    parser.add_argument("--output_dir", default="exports",
                        help="Directoria de output (default: exports/)")
    parser.add_argument("--episodios_amostra", type=int, default=5,
                        help="Episódios de referência nos logs de comportamento (default: 5)")
    parser.add_argument("--max_steps_por_episodio", type=int, default=50_000,
                        help="Limite de steps por episódio (default: 50000)")
    args = parser.parse_args()

    from .config import ExportConfig
    try:
        cfg = ExportConfig(
            treino_id=args.treino_id,
            output_dir=args.output_dir,
            episodios_amostra=args.episodios_amostra,
            max_steps_por_episodio=args.max_steps_por_episodio,
        )
    except (EnvironmentError, ValueError) as e:
        print(f"[export] Erro de configuração: {e}", file=sys.stderr)
        sys.exit(1)

    _executar(cfg)


if __name__ == "__main__":
    main()
