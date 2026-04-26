#!/bin/bash

# Configuration (overridable by environment variables)
REMOTE_USER="${REMOTE_USER:-hfzhkn}"
REMOTE_HOST="${REMOTE_HOST:-34.101.41.172}"
REMOTE_DIR="${REMOTE_DIR:-~/tanyadfbot/}"

echo "🚀 Syncing files to VPS..."

rsync -avz --progress \
  --exclude 'venv' \
  --exclude '__pycache__' \
  --exclude '.git' \
  --exclude '*.db*' \
  --exclude '*.session*' \
  --exclude '.env' \
  ./ "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"

echo "✅ Sync complete."
echo "🧹 Cleaning up pycache on VPS..."
ssh "${REMOTE_USER}@${REMOTE_HOST}" "find ${REMOTE_DIR} -name '__pycache__' -type d -exec rm -rf {} +"
echo "🔄 Restarting service..."
ssh -t "${REMOTE_USER}@${REMOTE_HOST}" "sudo systemctl restart tanyadfbot"

echo "🔍 Verifying startup stability (10s watch)..."
sleep 10

ssh "${REMOTE_USER}@${REMOTE_HOST}" "systemctl is-active --quiet tanyadfbot"
if [ $? -eq 0 ]; then
    echo "✨ Deployment finished! Service is stable."
else
    echo "❌ CRITICAL: Service crashed after restart!"
    echo "📜 Fetching recent logs..."
    ssh "${REMOTE_USER}@${REMOTE_HOST}" "journalctl -u tanyadfbot -n 20 --no-pager"
    
    # Optional: Send a local notification if on Linux/macOS
    if command -v notify-send &> /dev/null; then
        notify-send "TanyaDFBot Deployment Failed" "Service crashed immediately after restart." -u critical
    fi
    exit 1
fi
