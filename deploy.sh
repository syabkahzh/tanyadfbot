#!/bin/bash

# Configuration
REMOTE_USER="hfzhkn"
REMOTE_HOST="34.101.41.172"
REMOTE_DIR="~/tanyadfbot/"

echo "🚀 Syncing files to VPS..."

rsync -avz --progress \
  --exclude 'venv' \
  --exclude '__pycache__' \
  --exclude '.git' \
  --exclude '*.db' \
  --exclude '*.session' \
  --exclude '.env' \
  ./ ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}

echo "✅ Sync complete."
echo "🔄 Restarting service..."
ssh -t ${REMOTE_USER}@${REMOTE_HOST} "sudo systemctl restart tanyadfbot"
echo "✨ Deployment finished!"
