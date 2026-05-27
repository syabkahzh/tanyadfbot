# TanyaDFBot / Hermes Handover

## Snapshot

- Local repo: clean on `master`
- Local `HEAD`: `0c99913` — `fix: make hermes local reports dotenv-safe`
- GitHub `origin/master`: includes `bd16d5c` and `0c99913`
- VM repo: `/home/haqqanimalang/tanyadfbot` at `0c99913`
- VM `hermes-gateway.service`: `active`
- VM `tanyadfbot-runtime.service`: `inactive`

## What Was Done

### Repo-side maestro work

- Added a repo instruction file: `AGENTS.md`
- Added a combined control-plane report: `tools/hermes_maestro_report.py`
- Added a focused recent promo lookup tool: `tools/hermes_recent_promos.py`
- Expanded `hermes_reports.py` with:
  - command center report
  - recent promo lookup report
  - review + recommendations report
  - tuning proposal report
  - combined maestro report
- Added tests in `tests/test_hermes_reports.py`
- Added same-VM cutover docs:
  - `docs/HERMES_PHASE1_RUNBOOK.md`
  - `docs/MAESTRO_CUTOVER.md`
  - `HERMES_CONTROL_PLANE.md`
- Updated `deploy.sh` so local same-VM deploy is the default and the deprecated remote host is rejected

### Live VM sync work

- Confirmed the live VM repo now matches the pushed repo at `0c99913`
- Preserved older VM-only drift on backup branches:
  - `vm-backup-20260527-123236`
  - `vm-backup-20260527-123255`
- Left unrelated VM-only helper `tools/tanyafetch.py` on the backup branch, not in `master`

### Hermes SSH fallback fix

Root cause of the bad "latest promo" behavior was not just model drift:

1. Hermes memory on the VM was stale and still described Phase 1 as only:
   - `tools/hermes_daily_report.py`
   - `tools/hermes_health_report.py`
2. The new local recent-promo tool initially crashed on the VM because `hermes_reports.py` imported `config.py`, which required `python-dotenv`
3. When the local tool crashed, Hermes fell back to shell improvisation and tried SSH

Fixes applied:

- Added `tools/hermes_recent_promos.py`
- Updated `AGENTS.md` to explicitly say:
  - use local report scripts first
  - do not use SSH for normal promo lookup
  - do not probe alternate SSH ports
- Changed `hermes_reports.py` to fall back to `tanya_main.db` without requiring `dotenv` at import time
- Synced that fix to the VM at `0c99913`
- Verified on the VM:
  - `.venv/bin/python tools/hermes_recent_promos.py --hours 2`
  - now runs successfully
- Updated `/home/haqqanimalang/.hermes/memories/MEMORY.md` to mention:
  - `tools/hermes_recent_promos.py`
  - `tools/hermes_maestro_report.py`
  - never use SSH for normal promo lookup

## What Is Still Wrong

### The data plane is still not live on the same VM

This is the biggest remaining blocker.

- `tanyadfbot-runtime.service` is `inactive`
- Earlier VM inspection showed:
  - local `tanya_main.db` was empty
  - local Tanya runtime dependencies were incomplete at least at one point
  - the bot was not actually running as a live same-VM data plane

That means Hermes may now know the right local tool path, but it still may not have real promo data to answer from until the Tanya runtime is truly up and writing to the local DB.

### Hermes gateway has restart instability / startup noise

Recent journal shows multiple issues:

- earlier gateway starts reported:
  - `No messaging platforms enabled`
  - `no delivery target resolved for deliver=telegram`
- earlier turns still logged failed SSH attempts to the deprecated host `34.101.41.172`
- after a manual restart, journal showed:
  - `Self-improvement review: Skill 'tanyadfbot-operations' created.`
  - then `Main process exited, status=1/FAILURE`
  - systemd restarted it and current state is `active`

So Hermes is currently up, but it has shown unstable startup behavior and should be treated as not fully trusted yet.

## Current Truth About "Latest Promo"

As of this handover:

- The repo and VM now contain a local promo lookup path that works:
  - `PYTHONPATH=. .venv/bin/python tools/hermes_recent_promos.py --hours 2`
- The VM tool currently returns a valid local-only response shape
- But the underlying Tanya runtime on the VM is still inactive, so "latest promo" may legitimately return no data until the data plane is restored

## Recommended Next Checks

1. Bring up the real Tanya runtime on `34.21.211.116`
   - verify `.env`
   - verify Telethon session
   - verify runtime deps in `.venv`
   - verify non-empty live DB
   - verify `tanyadfbot-runtime.service`
2. Confirm the local DB starts receiving fresh promos
3. Re-ask Hermes:
   - "what's the latest promo right now?"
4. Watch `journalctl -u hermes-gateway.service`
   - confirm it uses the local tool path
   - confirm no SSH attempt appears
5. If Hermes still SSHs after the data plane is live:
   - inspect Hermes session/state persistence, not just repo files

## Useful Commands

### Local repo

```bash
git status --short --branch
git log --oneline -n 5
.venv/bin/python -m pytest -q tests/test_hermes_reports.py
```

### VM repo

```bash
ssh haqqanimalang@34.21.211.116
cd /home/haqqanimalang/tanyadfbot
git status --short --branch
git rev-parse --short HEAD
.venv/bin/python tools/hermes_recent_promos.py --hours 2
.venv/bin/python tools/hermes_maestro_report.py --command-hours 2 --review-hours 24
```

### VM services

```bash
systemctl is-active hermes-gateway.service
systemctl is-active tanyadfbot-runtime.service
journalctl -u hermes-gateway.service -n 80 --no-pager
```
