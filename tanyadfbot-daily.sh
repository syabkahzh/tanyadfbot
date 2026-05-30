#!/usr/bin/env bash
# tanyadfbot daily quality report — called by Hermes cron (no_agent=true)
# stdout is delivered verbatim to the operator.
set -euo pipefail
cd "$(dirname "$0")"
PYTHONPATH=. .venv/bin/python tools/hermes_daily_report.py 2>&1
