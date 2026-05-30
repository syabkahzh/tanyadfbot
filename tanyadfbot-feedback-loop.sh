#!/usr/bin/env bash
# tanyadfbot feedback loop analysis — called by Hermes cron
# stdout is injected as context into the agent prompt.
set -euo pipefail
cd "$(dirname "$0")"
PYTHONPATH=. .venv/bin/python tools/hermes_feedback_loop.py 2>&1
