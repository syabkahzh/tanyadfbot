# Hermes Supervisor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a proactive Hermes supervisor loop that monitors Tanya, finds second-chance promo candidates, reports health issues, and queues safe runtime actions through existing local tools.

**Architecture:** Tanya remains the data plane. A new repo-owned supervisor CLI builds on `hermes_reports.py` and `tools/hermes_control.py`, producing a compact Markdown report and optional queued commands. Hermes cron runs the CLI on the same VM and delivers the output to the operator.

**Tech Stack:** Python 3, SQLite via existing `Database`, existing report builders in `hermes_reports.py`, pytest.

---

## File Structure

- Modify `hermes_reports.py`: add a second-chance candidate builder and supervisor report formatter.
- Create `tools/hermes_supervisor_report.py`: CLI wrapper for scheduled Hermes use.
- Modify `tools/hermes_control.py`: add guardrails for known command names and clearer command history output.
- Modify `tests/test_hermes_reports.py`: cover second-chance and supervisor report behavior.
- Modify `README.md` and `docs/HERMES_PHASE1_RUNBOOK.md`: document the proactive supervisor loop.

### Task 1: Second-Chance Report Builder

**Files:**
- Modify: `hermes_reports.py`
- Test: `tests/test_hermes_reports.py`

- [ ] **Step 1: Write failing tests**

Add tests that create recent messages/promos/corrections and assert the report surfaces:

```python
def test_build_supervisor_report_flags_weak_latest_promo(tmp_path: Path) -> None:
    db_path = tmp_path / "tanya.db"
    _init_test_db(db_path)
    with sqlite3.connect(db_path) as conn:
        _insert_message(conn, 8470350, "Xiaomi A27i voucher scroll dapet 900rb", processed=1)
        _insert_promo(conn, brand="Xiaomi", summary="makasi udh kasi tauu", msg_id=8470350)

    report = build_supervisor_report(str(db_path), hours=2)

    assert "# Hermes Supervisor Report" in report
    assert "Second-Chance Candidates" in report
    assert "weak promo summary" in report
    assert "8470350" in report
```

```python
def test_build_supervisor_report_mentions_database_lock_health(tmp_path: Path) -> None:
    db_path = tmp_path / "tanya.db"
    log_path = tmp_path / "runtime.log"
    _init_test_db(db_path)
    log_path.write_text("2026-05-28 ERROR jobs: image_processing_job item error: database is locked\n")

    report = build_supervisor_report(str(db_path), hours=2, log_path=str(log_path))

    assert "Runtime Watch" in report
    assert "database is locked" in report
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_hermes_reports.py
```

Expected: fails because `build_supervisor_report` is not defined.

- [ ] **Step 3: Implement report builder**

Add `build_supervisor_report(db_path: str | None = None, hours: int = 2, log_path: str | None = None) -> str` to `hermes_reports.py`. It should reuse existing connection helpers and report sections, then add:

- `Second-Chance Candidates`: recent promos with suspiciously generic summaries such as thanks-only text, unknown brands, or skipped/corrected messages.
- `Runtime Watch`: queue depth, repeated failures, and log lines containing `database is locked`.
- `Recommended Actions`: operator-DM alert by default; force-alert only for high-confidence candidates.

- [ ] **Step 4: Run tests**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_hermes_reports.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add hermes_reports.py tests/test_hermes_reports.py
git commit -m "feat: add hermes supervisor report"
```

### Task 2: Supervisor CLI

**Files:**
- Create: `tools/hermes_supervisor_report.py`
- Test: `tests/test_hermes_reports.py`

- [ ] **Step 1: Write CLI test**

Add a subprocess test matching existing report CLI tests:

```python
def test_hermes_supervisor_report_cli_runs(tmp_path: Path) -> None:
    db_path = tmp_path / "tanya.db"
    _init_test_db(db_path)

    result = subprocess.run(
        [
            sys.executable,
            "tools/hermes_supervisor_report.py",
            "--db-path",
            str(db_path),
            "--hours",
            "2",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "# Hermes Supervisor Report" in result.stdout
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_hermes_reports.py::test_hermes_supervisor_report_cli_runs
```

Expected: fails because the CLI file is missing.

- [ ] **Step 3: Create CLI**

Create `tools/hermes_supervisor_report.py`:

```python
from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hermes_reports import build_supervisor_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a Hermes supervisor report for proactive monitoring.")
    parser.add_argument("--db-path", help="SQLite DB path. Defaults to Tanya config/local fallback.")
    parser.add_argument("--hours", type=int, default=2, help="Recent monitoring window.")
    parser.add_argument("--log-path", help="Optional Tanya runtime log path.")
    args = parser.parse_args()
    print(build_supervisor_report(db_path=args.db_path, hours=args.hours, log_path=args.log_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_hermes_reports.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tools/hermes_supervisor_report.py tests/test_hermes_reports.py
git commit -m "feat: add hermes supervisor cli"
```

### Task 3: Control CLI Guardrails

**Files:**
- Modify: `tools/hermes_control.py`
- Test: create `tests/test_hermes_control_cli.py`

- [ ] **Step 1: Write failing test**

```python
def test_hermes_control_rejects_unknown_command() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "tools/hermes_control.py",
            "send-command",
            "drop_database",
            "{}",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Unsupported command" in result.stderr
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_hermes_control_cli.py
```

Expected: fails because unknown commands are currently accepted.

- [ ] **Step 3: Implement guardrail**

Add:

```python
SUPPORTED_COMMANDS = {"reprocess", "suppress_brand", "override_alert", "force_alert"}
```

Before JSON validation in `send_command`, reject any command not in that set:

```python
if command not in SUPPORTED_COMMANDS:
    print(
        f"Unsupported command: {command}. Supported commands: {', '.join(sorted(SUPPORTED_COMMANDS))}",
        file=sys.stderr,
    )
    return 2
```

- [ ] **Step 4: Run tests**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_hermes_control_cli.py tests/test_hermes_reports.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tools/hermes_control.py tests/test_hermes_control_cli.py
git commit -m "fix: guard hermes runtime commands"
```

### Task 4: Docs and VM Cron Wiring

**Files:**
- Modify: `README.md`
- Modify: `docs/HERMES_PHASE1_RUNBOOK.md`
- Modify on VM after merge: `/home/haqqanimalang/.hermes/cron/jobs.json` through Hermes scheduler or gateway command, not by hand-editing secrets.

- [ ] **Step 1: Document command**

Add:

```markdown
### `tools/hermes_supervisor_report.py`

Builds a proactive supervisor report for Hermes. It combines recent promo lookup, missed-signal review, and runtime health into one scheduled operator-facing check.

```bash
PYTHONPATH=. .venv/bin/python tools/hermes_supervisor_report.py --hours 2
PYTHONPATH=. .venv/bin/python tools/hermes_supervisor_report.py --hours 2 --log-path /var/log/tanyadfbot/runtime.log
```
```

- [ ] **Step 2: Add runbook schedule**

Document a 15- or 30-minute Hermes cron job named `tanyadfbot supervisor loop` with script:

```bash
cd /home/haqqanimalang/tanyadfbot
PYTHONPATH=. .venv/bin/python tools/hermes_supervisor_report.py --hours 2
```

- [ ] **Step 3: Run tests**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_hermes_reports.py tests/test_hermes_control_cli.py
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/HERMES_PHASE1_RUNBOOK.md
git commit -m "docs: describe hermes supervisor loop"
```

### Task 5: Deploy and Verify

**Files:**
- No source files.

- [ ] **Step 1: Verify local repo state**

Run:

```bash
git status --short
git log --oneline -n 5
```

Expected: clean worktree after commits.

- [ ] **Step 2: Sync VM**

Run on VM:

```bash
cd /home/haqqanimalang/tanyadfbot
git pull --ff-only
.venv/bin/python -m pytest -q tests/test_hermes_reports.py tests/test_hermes_control_cli.py
```

Expected: pull succeeds and tests pass.

- [ ] **Step 3: Run supervisor report on VM**

Run:

```bash
cd /home/haqqanimalang/tanyadfbot
PYTHONPATH=. .venv/bin/python tools/hermes_supervisor_report.py --hours 2
```

Expected: report includes `# Hermes Supervisor Report`, second-chance candidates or a no-candidates line, runtime watch, and recommended actions.

- [ ] **Step 4: Add Hermes scheduled job**

Use the Hermes gateway scheduler command or existing cron-management interface to add a 30-minute job delivering to the operator Telegram channel. The job should run from `/home/haqqanimalang/tanyadfbot` and execute:

```bash
PYTHONPATH=. .venv/bin/python tools/hermes_supervisor_report.py --hours 2
```

- [ ] **Step 5: Verify service stability**

Run:

```bash
systemctl is-active tanyadfbot-runtime.service
systemctl is-active hermes-gateway.service
journalctl -u hermes-gateway.service -n 80 --no-pager
```

Expected: both services active; no new repeated local-tool failure loop.

## Self-Review

- The plan covers the spec's proactive monitoring, second-chance alerts, local-tool contract, runtime health watch, and command guardrails.
- Source-code changes remain gated through tests and commits.
- No step reads secrets, session files, or posts into the public group.
- The only runtime action is scheduling Hermes to deliver operator-facing reports.
