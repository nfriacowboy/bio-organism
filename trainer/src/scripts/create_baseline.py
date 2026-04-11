"""
create_baseline.py — Promover uma simulação existente a baseline.

Uso:
    python3 -m scripts.create_baseline \\
        --simulacao_id "20260411T153000Z" \\
        --nome "sweep4_world500_v1" \\
        --tipo experiment_control \\
        [--descricao "Referência sweep n_patches=4, world_size=500"]

    DATABASE_URL=postgresql://... python3 -m scripts.create_baseline ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone


TIPOS_VALIDOS = {"random_walk", "fixed_snn", "no_alpha", "experiment_control", "production"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promover simulação existente a baseline de referência."
    )
    parser.add_argument("--simulacao_id", required=True,
                        help="ID da simulação a promover (ex: 20260411T153000Z)")
    parser.add_argument("--nome", required=True,
                        help="Nome único da baseline (ex: sweep4_world500_v1)")
    parser.add_argument("--tipo", required=True, choices=sorted(TIPOS_VALIDOS),
                        help="Tipo da baseline")
    parser.add_argument("--descricao", default="",
                        help="Descrição opcional")
    args = parser.parse_args()

    db_url = os.getenv("DATABASE_URL", "")
    if not db_url:
        print("ERRO: DATABASE_URL não definido.", file=sys.stderr)
        sys.exit(1)

    try:
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(db_url)
        conn.autocommit = True
    except Exception as e:
        print(f"ERRO: Não foi possível ligar à DB: {e}", file=sys.stderr)
        sys.exit(1)

    with conn.cursor() as cur:
        # Verificar que a simulação existe
        cur.execute("SELECT config_json FROM simulacoes WHERE simulacao_id = %s",
                    (args.simulacao_id,))
        row = cur.fetchone()
        if not row:
            print(f"ERRO: simulacao_id '{args.simulacao_id}' não existe na DB.", file=sys.stderr)
            sys.exit(1)
        config_snapshot = row[0] if isinstance(row[0], dict) else json.loads(row[0] or "{}")

        # Verificar nome único
        cur.execute("SELECT baseline_id FROM baselines WHERE nome = %s", (args.nome,))
        if cur.fetchone():
            print(f"AVISO: Já existe uma baseline com o nome '{args.nome}'.", file=sys.stderr)
            print("Usa --nome diferente ou actualiza manualmente em DB.", file=sys.stderr)
            sys.exit(1)

        # Inserir
        bid = str(uuid.uuid4())
        cur.execute("""
            INSERT INTO baselines (baseline_id, nome, descricao, tipo, simulacao_id, config_snapshot)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (bid, args.nome, args.descricao or None, args.tipo,
              args.simulacao_id, json.dumps(config_snapshot)))

    print(f"✓ Baseline criada com sucesso")
    print(f"  baseline_id  : {bid}")
    print(f"  nome         : {args.nome}")
    print(f"  tipo         : {args.tipo}")
    print(f"  simulacao_id : {args.simulacao_id}")
    print()
    print("Para usar como referência num experimento:")
    print(f'  "baseline_id": "{bid}"')

    conn.close()


if __name__ == "__main__":
    main()
