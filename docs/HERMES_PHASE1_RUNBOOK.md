# Hermes Phase 1 Runbook

This repo stays the `tanyadfbot` data plane. Hermes Agent runs as a separate service on the same VM and uses the commands below as its read-only inspection surface. The same-VM contract is the important bit: Hermes should inspect local Tanya state, not a deprecated remote host.

## Stable Commands

Run from the repo root:

```bash
PYTHONPATH=. .venv/bin/python tools/hermes_recent_promos.py --hours 2
PYTHONPATH=. .venv/bin/python tools/hermes_daily_report.py --hours 24
PYTHONPATH=. .venv/bin/python tools/hermes_health_report.py --hours 24
PYTHONPATH=. .venv/bin/python tools/hermes_maestro_report.py --command-hours 2 --review-hours 24
PYTHONPATH=. .venv/bin/python tools/hermes_supervisor_report.py --hours 2
PYTHONPATH=. .venv/bin/python tools/hermes_shadow_watch.py --minutes 5 --quiet-empty
```

Optional log tail:

```bash
PYTHONPATH=. .venv/bin/python tools/hermes_health_report.py \
  --hours 24 \
  --log-path /var/log/tanyadfbot/runtime.log \
  --tail-lines 40
PYTHONPATH=. .venv/bin/python tools/hermes_maestro_report.py \
  --command-hours 2 \
  --review-hours 24 \
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
- run the report scripts above
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

`tools/hermes_recent_promos.py` covers:

- latest promo within a recent time window
- recent promos for an optional brand filter
- explicit local-only guidance so Hermes does not fall back to SSH

`tools/hermes_health_report.py` covers:

- queue depth
- total messages and promos
- repeated AI failure counts
- recent failures
- recent warning/error system logs
- optional log tail with secret redaction

`tools/hermes_maestro_report.py` covers:

- command-center promo lookup within a recent time window
- runtime health snapshot
- review findings and prioritized recommendations
- reviewable tuning proposals grounded in the recent window

`tools/hermes_supervisor_report.py` covers:

- second-chance candidates where Tanya extracted a weak promo or missed a signal
- runtime watch signals such as queue depth, repeated AI failures, and database lock noise
- recommended Hermes actions with operator-DM first and high-confidence `force_alert` only as an auditable escalation

`tools/hermes_shadow_watch.py` covers:

- near-live high-signal messages Tanya has ingested but not extracted into promos
- short polling windows suitable for every-2-minute Hermes cron
- quiet-empty mode so Hermes does not spam the operator DM when there are no findings

## Proactive Supervisor Loop

Hermes should run a scheduled supervisor loop every 15-30 minutes from `/home/haqqanimalang/tanyadfbot`:

```bash
PYTHONPATH=. .venv/bin/python tools/hermes_supervisor_report.py --hours 2
```

The default delivery target should be the Hermes operator DM. The supervisor loop is a second-chance monitor, not the first-line realtime alert path. Tanya still owns immediate alerts; Hermes notices weak extractions, missed juicy context, and runtime problems after the fact.

For near-live shadow monitoring, Hermes may also run this every 2 minutes:

```bash
PYTHONPATH=. .venv/bin/python tools/hermes_shadow_watch.py --minutes 5 --quiet-empty
```

This is intentionally DB-backed instead of a separate canonical Telegram listener. Hermes may use telegram-mcp for richer on-demand group context, but the always-on shadow loop should use Tanya's local DB so actions remain auditable and the group never becomes an instruction source.

## Operator Guardrail

For normal operator questions like "latest promo" or "latest promo in the last 2 hours", Hermes should use `tools/hermes_recent_promos.py` first. It should not open SSH sessions or try alternate SSH ports for that class of question.

## VM Setup Still Required

You still need to do these on the GCP VM:

1. Install Hermes Agent separately from this repo.
2. Configure Hermes gateway with its own Telegram bot token.
3. Point Hermes terminal `cwd` at the live `tanyadfbot` checkout.
4. Keep Hermes approvals in manual mode for Phase 1.
5. Create a frequent Hermes cron/job for `tools/hermes_supervisor_report.py` and a daily job for broader review output.
