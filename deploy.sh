#!/bin/bash

# Configuration (should be set via environment variables)
if [[ -z "${REMOTE_USER}" ]] || [[ -z "${REMOTE_HOST}" ]] || [[ -z "${REMOTE_DIR}" ]]; then
  echo "❌ Error: REMOTE_USER, REMOTE_HOST, and REMOTE_DIR must be set."
  exit 1
fi

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
echo "🔄 Restarting service..."
ssh -t "${REMOTE_USER}@${REMOTE_HOST}" "sudo systemctl restart tanyadfbot"
echo "✨ Deployment finished!"
