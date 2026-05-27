# TanyaDFBot - VM Agent Briefing

You are on a GCP VM (agentdf) running a Telegram deal-aggregator with two services.

## Architecture

Tanya Runtime (tanyadfbot-runtime.service): Telethon user-auth listener on @discountfessofficialbase. Ingests messages, AI extraction via Gemini, stores promos in SQLite. Bot is send-only (alerts/digests) — polling disabled because Hermes owns the webhook.

Hermes Gateway (hermes-gateway.service): Control plane. Webhook mode port 8443. Operator DM interface. Reads Tanya DB via local report tools.

## Key Paths

- Repo: /home/haqqanimalang/tanyadfbot/
- DB: tanya_main.db (aiosqlite, WAL)
- Hermes home: /home/haqqanimalang/.hermes/
- Logs: journalctl -u <service>

## Quick Commands

Status: systemctl is-active tanyadfbot-runtime.service && systemctl is-active hermes-gateway.service
Logs: journalctl -u tanyadfbot-runtime.service -n 50 --no-pager
Promos: cd /home/haqqanimalang/tanyadfbot && PYTHONPATH=. .venv/bin/python tools/hermes_recent_promos.py --hours 6
Maestro: PYTHONPATH=. .venv/bin/python tools/hermes_maestro_report.py --command-hours 6 --review-hours 24
Health: PYTHONPATH=. .venv/bin/python tools/hermes_health_report.py --hours 24
DB: sqlite3 tanya_main.db "SELECT count(*) FROM promos;"
Restart: sudo systemctl restart tanyadfbot-runtime.service

## Hard Rules

- NEVER SSH for promo lookup. Everything is local.
- NEVER use ad hoc SQL when a repo tool exists.
- Read-only by default. Code changes need tests + diff + approval.
- Write-whitelisted: skills/*.md, prompts/*.md, config/*.yaml, docs/*.md
