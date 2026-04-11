#!/usr/bin/env bash
# Simulação assíncrona tribal — N organismos autónomos, sem tick global
#
# Uso:
#   ./launch_async.sh                 # padrão: N=20, CPU, 1h
#   SNN_N_ORGANISMOS=50 SNN_DEVICE=cuda ./launch_async.sh
#   SNN_MAX_TEMPO_S=7200 ./launch_async.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +%Y%m%dT%H%M%S)"
LOG_FILE="$LOG_DIR/async_${TIMESTAMP}.log"

# --- Parâmetros (override via env) ---
export SNN_N_ORGANISMOS="${SNN_N_ORGANISMOS:-20}"
export SNN_DEVICE="${SNN_DEVICE:-cpu}"
export SNN_USAR_PATCHES="${SNN_USAR_PATCHES:-1}"
export SNN_MAX_TEMPO_S="${SNN_MAX_TEMPO_S:-3600}"
export SNN_LOG_INTERVAL_S="${SNN_LOG_INTERVAL_S:-30}"
export SNN_INFERENCE_BATCH="${SNN_INFERENCE_BATCH:-64}"
export SNN_TREINO_NOTA="${SNN_TREINO_NOTA:-async_tribal}"
export SNN_SNAPSHOT_INTERVAL="${SNN_SNAPSHOT_INTERVAL:-5}"
export SNN_FASE_INICIAL="${SNN_FASE_INICIAL:-1}"

# Base de dados (opcional — só loga se DATABASE_URL definida)
export DATABASE_URL="${DATABASE_URL:-postgresql://snn_user:snn_secure_pw_2026@localhost:5432/snn_organismo}"

echo "====================================================="
echo "  bio-organism — Simulação Tribal Assíncrona"
echo "====================================================="
echo "  N_ORGANISMOS : $SNN_N_ORGANISMOS"
echo "  DEVICE       : $SNN_DEVICE"
echo "  PATCHES      : $SNN_USAR_PATCHES"
echo "  MAX_TEMPO    : ${SNN_MAX_TEMPO_S}s"
echo "  LOG_INTERVAL : ${SNN_LOG_INTERVAL_S}s"
echo "  BATCH        : $SNN_INFERENCE_BATCH"
echo "  FASE_INICIAL : $SNN_FASE_INICIAL"
echo "  SNAPSHOT_INT : $SNN_SNAPSHOT_INTERVAL log intervals"
echo "  LOG_FILE     : $LOG_FILE"
echo "====================================================="
echo ""
echo "  Dashboard: DATABASE_URL=\"$DATABASE_URL\" streamlit run dashboard/dashboard.py"
echo ""

cd "$REPO_ROOT/trainer"

PYTHONPATH=src PYTHONUNBUFFERED=1 \
    uv run --no-sync python3 -m training.trainer_async 2>&1 | tee "$LOG_FILE"
