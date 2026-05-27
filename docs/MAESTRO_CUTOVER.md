# Hermes Maestro Cutover

This runbook turns the roadmap into an operator sequence for the same-VM setup.

## Goal

Run Hermes and the Tanya runtime on the same VM, with local reports as the only inspection surface and no dependence on the deprecated `34.101.41.172` host.

## Target State

- `hermes-gateway.service` is active on the VM.
- `tanyadfbot-runtime.service` is active on the same VM.
- Tanya writes to a local live database.
- Hermes reads local report output only.
- The deployment script uses local mode by default.

## Preflight

1. Confirm the VM has both repos or a shared worktree layout.
2. Confirm the Tanya `.env` and Telethon session exist on the VM.
3. Confirm the Tanya virtualenv includes runtime dependencies.
4. Confirm the live database path is non-empty.
5. Confirm Hermes can run the local report commands from the Tanya repo root.

## Local Same-VM Deploy

Use the local mode in `deploy.sh`:

```bash
DEPLOY_TARGET=local SERVICE_NAME=tanyadfbot-runtime ./deploy.sh
```

Optional pull-before-restart:

```bash
DEPLOY_TARGET=local DEPLOY_PULL=1 SERVICE_NAME=tanyadfbot-runtime ./deploy.sh
```

If the service name differs on the VM, override `SERVICE_NAME` explicitly.

## Verification

After restart, verify:

```bash
systemctl is-active tanyadfbot-runtime
systemctl is-active hermes-gateway
PYTHONPATH=. .venv/bin/python tools/hermes_maestro_report.py --command-hours 2 --review-hours 24
PYTHONPATH=. .venv/bin/python tools/hermes_health_report.py --hours 24
```

## Operator Checks

- Hermes answers recent promo and health questions without referencing a remote host.
- Hermes review output includes false positives, missed signals, and recommendations.
- Hermes tuning proposals mention reviewable YAML/prompt/skill assets only.
- No part of the workflow assumes `34.101.41.172` is still live.
