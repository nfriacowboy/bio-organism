"""
create_experiment.py — Registar um experimento antes de correr a simulação.

Usa um ficheiro JSON que define a hipótese, parâmetros alterados e critérios.
O ficheiro deve ser comitado no repo para rastreabilidade.

Uso:
    python3 -m scripts.create_experiment --file experiments/exp_001.json

    DATABASE_URL=postgresql://... python3 -m scripts.create_experiment \\
        --file experiments/exp_001.json

Formato do ficheiro JSON (ver experiments/template.json):
    {
      "baseline_id": "<uuid>",
      "nome": "Sweep n_patches 2-6 world500",
      "sweep_variable": "n_patches",
      "sweep_values": [2, 3, 4, 6],
      "n_replicas_per_value": 5,
      "parametros_alterados": {
        "SNN_WORLD_SIZE": {"de": 200, "para": 500}
      },
      "justificacao": "...",
      "resultado_esperado": "...",
      "criterios_validacao": [
        {"metrica": "cv_fc1_media", "op": ">", "valor": 1.0, "janela_final_s": 300}
      ]
    }
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone


REQUIRED_FIELDS = {"baseline_id", "parametros_alterados", "justificacao", "resultado_esperado"}
OPS_VALIDOS = {">", ">=", "<", "<=", "=="}


def _validar(exp: dict) -> list[str]:
    erros = []
    for f in REQUIRED_FIELDS:
        if f not in exp:
            erros.append(f"Campo obrigatório em falta: '{f}'")
    if exp.get("parametros_alterados") == {}:
        erros.append("'parametros_alterados' não pode ser vazio")
    if not exp.get("justificacao", "").strip():
        erros.append("'justificacao' não pode estar vazia")
    if not exp.get("resultado_esperado", "").strip():
        erros.append("'resultado_esperado' não pode estar vazio")
    for i, c in enumerate(exp.get("criterios_validacao", [])):
        if "metrica" not in c:
            erros.append(f"criterio[{i}]: falta 'metrica'")
        if c.get("op") not in OPS_VALIDOS:
            erros.append(f"criterio[{i}]: 'op' deve ser um de {OPS_VALIDOS}")
        if "valor" not in c:
            erros.append(f"criterio[{i}]: falta 'valor'")
    return erros


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Criar registo de experimento antes de correr simulação."
    )
    parser.add_argument("--file", required=True,
                        help="Caminho para o ficheiro JSON do experimento")
    args = parser.parse_args()

    # Ler e validar ficheiro
    try:
        with open(args.file, "r") as f:
            exp = json.load(f)
    except Exception as e:
        print(f"ERRO: Não foi possível ler '{args.file}': {e}", file=sys.stderr)
        sys.exit(1)

    erros = _validar(exp)
    if erros:
        print("ERRO: Ficheiro de experimento inválido:", file=sys.stderr)
        for e in erros:
            print(f"  • {e}", file=sys.stderr)
        sys.exit(1)

    db_url = os.getenv("DATABASE_URL", "")
    if not db_url:
        print("ERRO: DATABASE_URL não definido.", file=sys.stderr)
        sys.exit(1)

    try:
        import psycopg2
        conn = psycopg2.connect(db_url)
        conn.autocommit = True
    except Exception as e:
        print(f"ERRO: Não foi possível ligar à DB: {e}", file=sys.stderr)
        sys.exit(1)

    with conn.cursor() as cur:
        # Verificar baseline existe
        bid = exp["baseline_id"]
        cur.execute("SELECT nome, tipo FROM baselines WHERE baseline_id = %s", (bid,))
        bl = cur.fetchone()
        if not bl:
            print(f"ERRO: baseline_id '{bid}' não existe. "
                  f"Cria a baseline primeiro com create_baseline.py", file=sys.stderr)
            sys.exit(1)
        bl_nome, bl_tipo = bl

        eid = str(uuid.uuid4())
        cur.execute("""
            INSERT INTO experimentos (
                experimento_id, baseline_id, nome,
                sweep_variable, sweep_values, n_replicas_per_value,
                parametros_alterados, justificacao, resultado_esperado,
                criterios_validacao, estado
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'pendente')
        """, (
            eid, bid,
            exp.get("nome", ""),
            exp.get("sweep_variable"),
            json.dumps(exp.get("sweep_values")) if exp.get("sweep_values") else None,
            exp.get("n_replicas_per_value"),
            json.dumps(exp.get("parametros_alterados", {})),
            exp["justificacao"],
            exp["resultado_esperado"],
            json.dumps(exp.get("criterios_validacao", [])),
        ))

    print(f"✓ Experimento registado com sucesso")
    print(f"  experimento_id : {eid}")
    print(f"  nome           : {exp.get('nome', '(sem nome)')}")
    print(f"  baseline       : {bl_nome} ({bl_tipo})")
    print(f"  params alterados: {list(exp.get('parametros_alterados', {}).keys())}")
    print(f"  critérios      : {len(exp.get('criterios_validacao', []))}")
    print()
    print("Para usar nesta simulação, definir antes do run:")
    print(f"  SNN_EXPERIMENT_FILE={args.file}")
    print()
    print("O experimento será automaticamente associado à simulação ao iniciar o trainer.")

    conn.close()


if __name__ == "__main__":
    main()
