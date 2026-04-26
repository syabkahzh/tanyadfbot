#!/bin/bash
# Auto-cleanup script for TanyaDFBot
# Runs via cron to remove images and media older than 24 hours

BOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find and delete image and media files older than 1 day
# Limits search to 1 level to avoid touching subdirectories
find "$BOT_DIR" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' -o -name '*.webp' -o -name '*.mp4' -o -name '*.oga' \) -mtime +1 -exec rm -f {} +

echo "[$(date)] Cleanup executed. Removed old media files." >> "$BOT_DIR/cleanup.log"
