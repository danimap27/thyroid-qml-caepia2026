#!/bin/bash
# =============================================================================
# deploy.sh — Sync a local project to Hercules via rsync.
#
# Usage:
#   bash core/deploy.sh                  # sync only
#   bash core/deploy.sh --launch         # sync + open SSH session
#
# Configure via environment variables or edit the defaults below:
#   HERCULES_USER : your Hercules username
#   HERCULES_HOST : hostname (default: hercules.spc.cica.es)
#   REMOTE_DIR    : destination path on the cluster
# =============================================================================

HERCULES_USER="${HERCULES_USER:-your_user}"
HERCULES_HOST="${HERCULES_HOST:-hercules.spc.cica.es}"
REMOTE_DIR="${REMOTE_DIR:-~/experiment}"

LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================================"
echo "  From : $LOCAL_DIR"
echo "  To   : ${HERCULES_USER}@${HERCULES_HOST}:${REMOTE_DIR}"
echo "============================================================"

rsync -avz --progress \
    --exclude ".git" \
    --exclude "__pycache__" \
    --exclude "*.pyc" \
    --exclude "*.egg-info" \
    --exclude ".env" \
    --exclude "data/raw" \
    --exclude "results" \
    --exclude "logs" \
    --exclude "*.out" \
    --exclude "*.err" \
    "$LOCAL_DIR/" \
    "${HERCULES_USER}@${HERCULES_HOST}:${REMOTE_DIR}/"

echo ""
echo "[OK] Sync complete."

if [[ "$1" == "--launch" ]]; then
    echo "[INFO] Opening SSH session on Hercules..."
    ssh "${HERCULES_USER}@${HERCULES_HOST}" -t "cd ${REMOTE_DIR} && bash"
fi
