#!/usr/bin/env bash
# Simulação multi-mundo — n_patches sweep tribal (com pesos v80p)
#
# Uso:
#   ./launch_multi_world.sh                                                # 20 mundos sweep, 2h
#   SNN_N_MUNDOS=5 SNN_FIXED_N_PATCHES=5 SNN_MAX_TEMPO_S=600 ./launch_multi_world.sh  # fase 0 validação

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +%Y%m%dT%H%M%S)"
LOG_FILE="$LOG_DIR/multi_world_${TIMESTAMP}.log"

# Checkpoint v80p — olfato 1/d², fase 3 breakthrough (7.6× random)
V80P_SNN="${REPO_ROOT}/trainer/mlartifacts/2026-03-26T16:27:55.922977+00:00/snn_ep2000.pt"

export SNN_N_MUNDOS="${SNN_N_MUNDOS:-20}"
export SNN_WORLD_SIZE="${SNN_WORLD_SIZE:-500}"
export SNN_N_ORG_PER_WORLD="${SNN_N_ORG_PER_WORLD:-10}"
export SNN_VIDA_MAX="${SNN_VIDA_MAX:-300}"
export SNN_MAX_POP="${SNN_MAX_POP:-25}"
export SNN_PATCH_RADIUS="${SNN_PATCH_RADIUS:-40}"
export SNN_FIXED_N_PATCHES="${SNN_FIXED_N_PATCHES:-0}"
export SNN_BASE_SNN_PATH="${SNN_BASE_SNN_PATH:-$V80P_SNN}"
export SNN_DEVICE="${SNN_DEVICE:-cuda}"
export SNN_MAX_TEMPO_S="${SNN_MAX_TEMPO_S:-7200}"
export SNN_LOG_INTERVAL_S="${SNN_LOG_INTERVAL_S:-60}"
export SNN_INFERENCE_BATCH="${SNN_INFERENCE_BATCH:-256}"
export SNN_TREINO_NOTA="${SNN_TREINO_NOTA:-multi_world_patches_sweep}"
export DATABASE_URL="${DATABASE_URL:-postgresql://snn_user:snn_secure_pw_2026@localhost:5432/snn_organismo}"

echo "======================================================="
echo "  bio-organism — Multi-World N-Patches Sweep"
echo "======================================================="
echo "  N_MUNDOS     : $SNN_N_MUNDOS"
echo "  WORLD_SIZE   : $SNN_WORLD_SIZE"
echo "  N_ORG/MUNDO  : $SNN_N_ORG_PER_WORLD"
echo "  VIDA_MAX     : $SNN_VIDA_MAX"
echo "  PATCH_RADIUS : $SNN_PATCH_RADIUS"
echo "  FIXED_PATCHES: $SNN_FIXED_N_PATCHES (0=sweep)"
echo "  BASE_SNN     : $SNN_BASE_SNN_PATH"
echo "  DEVICE       : $SNN_DEVICE"
echo "  MAX_TEMPO    : ${SNN_MAX_TEMPO_S}s"
echo "  LOG_INTERVAL : ${SNN_LOG_INTERVAL_S}s"
echo "  BATCH        : $SNN_INFERENCE_BATCH"
echo "  LOG_FILE     : $LOG_FILE"
echo "======================================================="
echo ""
echo "  Dashboard: streamlit run dashboard/dashboard.py"
echo ""

cd "$REPO_ROOT/trainer"

PYTHONPATH=src PYTHONUNBUFFERED=1 \
    uv run --no-sync python3 -m training.trainer_multi_world 2>&1 | tee "$LOG_FILE"
