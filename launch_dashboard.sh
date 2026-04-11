#!/usr/bin/env bash
# launch_dashboard.sh — Inicia o Streamlit dashboard de monitorização tribal
#
# Uso:
#   bash launch_dashboard.sh
# ---------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
DASH_DIR="$REPO_ROOT/dashboard"

# Instalar deps se necessário
if [ ! -f "$DASH_DIR/.venv/bin/streamlit" ] || \
   ! "$DASH_DIR/.venv/bin/python3" -c "import transformers" 2>/dev/null; then
    echo "Criando/actualizando venv do dashboard..."
    [ ! -d "$DASH_DIR/.venv" ] && uv venv --python 3.12 "$DASH_DIR/.venv"
    uv pip install --python "$DASH_DIR/.venv/bin/python3" \
        streamlit plotly scikit-learn psycopg2-binary pandas kaleido imageio \
        "transformers>=4.40.0" "accelerate>=0.27.0"
    # Instalar torch ROCm para inferência na GPU
    uv pip install --python "$DASH_DIR/.venv/bin/python3" \
        "torch==2.9.1" --index-url https://download.pytorch.org/whl/rocm6.3
fi

echo "Abrindo dashboard em http://localhost:8501"
echo "Pressiona Ctrl+C para parar"
echo ""

DATABASE_URL="postgresql://snn_user:snn_secure_pw_2026@localhost:5432/snn_organismo" \
    "$DASH_DIR/.venv/bin/streamlit" run "$DASH_DIR/dashboard.py" \
    --server.port 8501 \
    --server.address localhost
