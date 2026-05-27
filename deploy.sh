#!/bin/bash
set -euo pipefail

# Deployment modes:
# - local: same-VM cutover, restart a local systemd service
# - remote: legacy sync path for explicit remote targets only
DEPLOY_TARGET="${DEPLOY_TARGET:-local}"
SERVICE_NAME="${SERVICE_NAME:-tanyadfbot-runtime}"
REMOTE_USER="${REMOTE_USER:-hfzhkn}"
REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_DIR="${REMOTE_DIR:-~/tanyadfbot/}"
DEPLOY_PULL="${DEPLOY_PULL:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${DEPLOY_TARGET}" == "local" ]]; then
  echo "🚀 Local same-VM deploy starting..."
  cd "${SCRIPT_DIR}"

  if [[ "${DEPLOY_PULL}" == "1" ]]; then
    echo "⬇️ Pulling latest git changes..."
    git pull --ff-only
  fi

  echo "🔄 Restarting local service: ${SERVICE_NAME}"
  sudo systemctl restart "${SERVICE_NAME}"

  echo "🔍 Verifying startup stability (10s watch)..."
  sleep 10

  if systemctl is-active --quiet "${SERVICE_NAME}"; then
    echo "✨ Deployment finished! Service is stable."
    exit 0
  fi

  echo "❌ CRITICAL: Service crashed after restart!"
  echo "📜 Fetching recent logs..."
  journalctl -u "${SERVICE_NAME}" -n 20 --no-pager
  exit 1
fi

if [[ "${DEPLOY_TARGET}" != "remote" ]]; then
  echo "❌ Unknown DEPLOY_TARGET='${DEPLOY_TARGET}'. Use 'local' or 'remote'."
  exit 1
fi

if [[ -z "${REMOTE_HOST}" ]]; then
  echo "❌ REMOTE_HOST must be set for remote deployments."
  exit 1
fi

if [[ "${REMOTE_HOST}" == "34.101.41.172" ]]; then
  echo "❌ Remote host 34.101.41.172 is deprecated. Use the same-VM local mode instead."
  exit 1
fi

echo "🚀 Syncing files to remote target..."
rsync -avz --progress \
  --exclude 'venv' \
  --exclude '__pycache__' \
  --exclude '.git' \
  --exclude '*.db*' \
  --exclude '*.session*' \
  --exclude '.env' \
  ./ "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"

echo "✅ Sync complete."
echo "🧹 Cleaning up pycache on remote target..."
ssh "${REMOTE_USER}@${REMOTE_HOST}" "find ${REMOTE_DIR} -name '__pycache__' -type d -exec rm -rf {} +"
echo "🔄 Restarting service..."
ssh -t "${REMOTE_USER}@${REMOTE_HOST}" "sudo systemctl restart ${SERVICE_NAME}"

echo "🔍 Verifying startup stability (10s watch)..."
sleep 10

ssh "${REMOTE_USER}@${REMOTE_HOST}" "systemctl is-active --quiet ${SERVICE_NAME}"
if [[ $? -eq 0 ]]; then
  echo "✨ Deployment finished! Service is stable."
else
  echo "❌ CRITICAL: Service crashed after restart!"
  echo "📜 Fetching recent logs..."
  ssh "${REMOTE_USER}@${REMOTE_HOST}" "journalctl -u ${SERVICE_NAME} -n 20 --no-pager"
  exit 1
fi
