#!/usr/bin/env bash
# launch_tribal.sh — Simulação tribal N=20 organismos, fase 2, patches + feromónas
#
# Requisitos:
#   - Docker com snn-db activo (docker compose up -d)
#   - base_v1 disponível em mlartifacts/base_v1/
#
# Uso:
#   bash launch_tribal.sh
#   bash launch_tribal.sh 2>&1 | tee logs/treino_tribal_$(date +%Y%m%d_%H%M).log
# ---------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/treino_tribal_$(date +%Y%m%d_%H%M).log"

echo "=== Tribal Simulation — $(date) ===" | tee "$LOG_FILE"
echo "Log: $LOG_FILE"
echo ""

cd "$REPO_ROOT/trainer"

PYTHONPATH=src \
PYTHONUNBUFFERED=1 \
SNN_N_ORGANISMOS=20 \
SNN_N_EPISODIOS=1000 \
SNN_FASE_INICIAL=2 \
SNN_DEVICE=cpu \
SNN_USAR_PATCHES=1 \
SNN_USAR_PREDADOR=0 \
SNN_REPRODUCAO=1 \
SNN_MUTACAO_SIGMA=0.01 \
SNN_SNAPSHOT_INTERVAL=5 \
SNN_SAVE_INTERVAL=100 \
SNN_TREINO_NOTA="tribal N=20 fase2 patches feromónas" \
DATABASE_URL="postgresql://snn_user:snn_secure_pw_2026@localhost:5432/snn_organismo" \
  uv run --no-sync python3 -m training.trainer_populacao \
  2>&1 | tee -a "$LOG_FILE"

echo ""
echo "=== Fim — $(date) ===" | tee -a "$LOG_FILE"
echo "Log guardado em: $LOG_FILE"
