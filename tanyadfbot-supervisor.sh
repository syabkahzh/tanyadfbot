#!/usr/bin/env bash
# tanyadfbot supervisor loop — called by Hermes cron (no_agent=true)
# stdout is delivered verbatim to the operator.
set -euo pipefail
cd "$(dirname "$0")"
PYTHONPATH=. .venv/bin/python tools/hermes_supervisor_report.py --hours 1 2>&1
