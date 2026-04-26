#!/bin/bash
# tools/vps_alert.sh - Out-of-band Telegram Notifier
# Used when the main Python bot is dead/crashing.

# Load environment variables for tokens
source .env 2>/dev/null

# Configuration
BOT_TOKEN="${TELEGRAM_BOT_TOKEN}"
CHAT_ID="${OWNER_ID:-82849559}" # Default to your ID if not set

MESSAGE="🚨 *VPS CRASH ALERT* 🚨
Service: *tanyadfbot*
Host: $(hostname)
Time: $(date)

The bot service has entered a crash loop or failed to start. Check logs immediately with:
'journalctl -u tanyadfbot -n 50 --no-pager'"

if [ -z "$BOT_TOKEN" ]; then
    echo "Error: TELEGRAM_BOT_TOKEN not found in .env"
    exit 1
fi

curl -s -X POST "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
    -d chat_id="${CHAT_ID}" \
    -d text="${MESSAGE}" \
    -d parse_mode="Markdown" > /dev/null

echo "✅ Failure notification sent to Telegram."
