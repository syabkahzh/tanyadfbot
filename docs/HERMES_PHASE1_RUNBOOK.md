# Hermes Phase 1 Runbook

This repo stays the `tanyadfbot` data plane. Hermes Agent runs as a separate service on the same VM and uses the commands below as its read-only inspection surface.

## Stable Commands

Run from the repo root:

```bash
PYTHONPATH=. .venv/bin/python tools/hermes_daily_report.py --hours 24
PYTHONPATH=. .venv/bin/python tools/hermes_health_report.py --hours 24
```

Optional log tail:

```bash
PYTHONPATH=. .venv/bin/python tools/hermes_health_report.py \
  --hours 24 \
  --log-path /var/log/tanyadfbot/runtime.log \
  --tail-lines 40
```

These commands are intentionally:

- read-only
- env-free for normal use
- human-readable markdown output
- safe to hand to Hermes without ad hoc SQL prompts

## Hermes Policy for Phase 1

Hermes may:

- read repo text files
- run the two report scripts above
- inspect `git diff`, `git log`, and read-only service status commands
- read log files that do not expose secrets directly

Hermes must not:

- read `.env`
- read `*.session`
- modify code, prompts, YAML, or docs
- restart services from cron
- connect to or act on the monitored public group directly

## Expected Report Coverage

`tools/hermes_daily_report.py` covers:

- promo volume in the lookback window
- AI correction volume
- false-positive candidates
- missed-signal candidates
- repeat-fire brand hints
- top brands and top correction labels

`tools/hermes_health_report.py` covers:

- queue depth
- total messages and promos
- repeated AI failure counts
- recent failures
- recent warning/error system logs
- optional log tail with secret redaction

## VM Setup Still Required

You still need to do these on the GCP VM:

1. Install Hermes Agent separately from this repo.
2. Configure Hermes gateway with its own Telegram bot token.
3. Point Hermes terminal `cwd` at the live `tanyadfbot` checkout.
4. Keep Hermes approvals in manual mode for Phase 1.
5. Create a daily Hermes cron/job that runs the report commands and sends the result back to your Telegram destination.
