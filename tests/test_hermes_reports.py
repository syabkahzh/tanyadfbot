from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from hermes_reports import (
    build_alert_quality_report,
    build_command_center_report,
    build_maestro_report,
    build_recent_promo_lookup_report,
    build_review_recommendations_report,
    build_service_health_report,
    build_tuning_proposal_report,
)


def _seed_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    now = datetime.now(timezone.utc)

    def ts(minutes: int) -> str:
        return (now + timedelta(minutes=minutes)).isoformat(sep=" ")

    cur.executescript(
        """
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY,
            tg_msg_id INTEGER,
            chat_id INTEGER,
            timestamp TEXT,
            text TEXT,
            processed INTEGER DEFAULT 1,
            skip_reason TEXT,
            ai_failure_count INTEGER DEFAULT 0
        );

        CREATE TABLE promos (
            id INTEGER PRIMARY KEY,
            source_msg_id INTEGER,
            summary TEXT NOT NULL,
            brand TEXT NOT NULL,
            tg_link TEXT,
            status TEXT NOT NULL DEFAULT 'unknown',
            via_fastpath INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        );

        CREATE TABLE ai_corrections (
            id INTEGER PRIMARY KEY,
            original_msg_id INTEGER NOT NULL,
            correction TEXT NOT NULL,
            weight REAL DEFAULT 0.5,
            created_at TEXT NOT NULL
        );

        CREATE TABLE system_logs (
            id INTEGER PRIMARY KEY,
            level TEXT NOT NULL,
            logger_name TEXT,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE failures (
            id INTEGER PRIMARY KEY,
            component TEXT NOT NULL,
            error_msg TEXT,
            created_at TEXT NOT NULL
        );
        """
    )

    cur.executemany(
        "INSERT INTO messages (id, tg_msg_id, chat_id, timestamp, text, processed, skip_reason, ai_failure_count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, 101, -1, ts(-110), "Promo alfamart cashback 50%", 1, None, 0),
            (2, 102, -1, ts(-100), "Noise message", 1, "ai_skip", 0),
            (3, 103, -1, ts(-90), "GoFood diskon ongkir", 1, "ai_skip", 0),
            (4, 104, -1, ts(-10), "Pending queue message", 0, None, 2),
            (5, 105, -1, ts(-5), "Alfamart fresh deal", 1, None, 0),
        ],
    )
    cur.executemany(
        "INSERT INTO promos (id, source_msg_id, summary, brand, tg_link, status, via_fastpath, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, 1, "Cashback 50% Alfamart", "Alfamart", "https://t.me/c/1/101", "active", 0, ts(-109)),
            (2, 3, "GoFood ongkir murah", "GoFood", "https://t.me/c/1/103", "active", 1, ts(-89)),
            (3, None, "GoFood flash sale", "GoFood", "https://t.me/c/1/999", "unknown", 1, ts(-70)),
            (4, 5, "Alfamart fresh deal", "Alfamart", "https://t.me/c/1/105", "active", 1, ts(-4)),
        ],
    )
    cur.executemany(
        "INSERT INTO ai_corrections (id, original_msg_id, correction, weight, created_at) VALUES (?, ?, ?, ?, ?)",
        [
            (1, 1, "NOT_A_PROMO", 1.0, ts(-50)),
            (2, 3, "Missed cashback context", 0.8, ts(-45)),
            (3, 5, "Need more lingo around flash promo", 0.7, ts(-15)),
        ],
    )
    cur.executemany(
        "INSERT INTO system_logs (id, level, logger_name, message, created_at) VALUES (?, ?, ?, ?, ?)",
        [
            (1, "WARNING", "listener", "Queue pressure rising", ts(-8)),
            (2, "ERROR", "processor", "Provider timeout OPENAI_API_KEY=super-secret", ts(-7)),
        ],
    )
    cur.executemany(
        "INSERT INTO failures (id, component, error_msg, created_at) VALUES (?, ?, ?, ?)",
        [
            (1, "processor", "timeout talking to provider", ts(-6)),
        ],
    )

    conn.commit()
    conn.close()


def test_build_alert_quality_report_surfaces_phase1_sections(tmp_path: Path) -> None:
    db_path = tmp_path / "report.db"
    _seed_db(db_path)

    report = build_alert_quality_report(str(db_path), hours=24)

    assert "# Hermes Alert Quality Report" in report
    assert "## Summary" in report
    assert "## False Positive Candidates" in report
    assert "## Missed Signal Candidates" in report
    assert "## Duplicate / Overfire Hints" in report
    assert "## Top Brands" in report
    assert "Alfamart" in report
    assert "GoFood" in report
    assert "NOT_A_PROMO" in report
    assert "Missed cashback context" in report
    assert "https://t.me/c/1/101" in report


def test_build_service_health_report_redacts_secrets_and_reports_queue(tmp_path: Path) -> None:
    db_path = tmp_path / "report.db"
    _seed_db(db_path)
    log_path = tmp_path / "service.log"
    log_path.write_text(
        "Boot ok\n"
        "BOT_TOKEN=123456:abcdef\n"
        "Normal line\n",
        encoding="utf-8",
    )

    report = build_service_health_report(str(db_path), hours=24, log_path=str(log_path), tail_lines=3)

    assert "# Hermes Service Health Report" in report
    assert "## Runtime Snapshot" in report
    assert "Queue depth" in report
    assert "1" in report
    assert "## Recent Failures" in report
    assert "timeout talking to provider" in report
    assert "## Recent System Logs" in report
    assert "Provider timeout" in report
    assert "[REDACTED]" in report
    assert "abcdef" not in report


def test_reports_handle_empty_database_without_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "empty.db"

    alert_report = build_alert_quality_report(str(db_path), hours=24)
    health_report = build_service_health_report(str(db_path), hours=24, log_path=str(tmp_path / "missing.log"))

    assert "# Hermes Alert Quality Report" in alert_report
    assert "Promos observed: 0" in alert_report
    assert "No negative corrections recorded" in alert_report

    assert "# Hermes Service Health Report" in health_report
    assert "Queue depth: 0" in health_report
    assert "No failures recorded" in health_report


def test_build_service_health_report_marks_empty_log_file(tmp_path: Path) -> None:
    db_path = tmp_path / "report.db"
    _seed_db(db_path)
    log_path = tmp_path / "empty.log"
    log_path.touch()

    report = build_service_health_report(str(db_path), hours=24, log_path=str(log_path))

    assert "Log file empty at" in report


def test_build_command_center_report_highlights_latest_promo_and_health(tmp_path: Path) -> None:
    db_path = tmp_path / "report.db"
    _seed_db(db_path)

    report = build_command_center_report(str(db_path), hours=2)

    assert "# Hermes Maestro Command Center" in report
    assert "## Latest Promo" in report
    assert "Alfamart fresh deal" in report
    assert "## What's Hot Right Now" in report
    assert "## Runtime Health" in report
    assert "Queue depth" in report


def test_build_review_recommendations_report_includes_actionable_guidance(tmp_path: Path) -> None:
    db_path = tmp_path / "report.db"
    _seed_db(db_path)

    report = build_review_recommendations_report(str(db_path), hours=24)

    assert "# Hermes Review + Recommendations" in report
    assert "## Findings" in report
    assert "## Recommendations" in report
    assert "false positive" in report.lower()
    assert "missed signal" in report.lower()
    assert "investigate" in report.lower()
    assert "monitor" in report.lower()


def test_build_tuning_proposal_report_mentions_real_control_plane_assets(tmp_path: Path) -> None:
    db_path = tmp_path / "report.db"
    _seed_db(db_path)

    report = build_tuning_proposal_report(str(db_path), hours=24)

    assert "# Hermes Tuning Proposals" in report
    assert "config/trigger_terms.yaml" in report
    assert "skills/discountfess-lingo.md" in report
    assert "skills/false-positive-patterns.md" in report
    assert "prompts/promo_judge.md" in report
    assert "reviewable" in report.lower()


def test_build_maestro_report_combines_operator_views(tmp_path: Path) -> None:
    db_path = tmp_path / "report.db"
    _seed_db(db_path)
    log_path = tmp_path / "service.log"
    log_path.write_text("boot ok\n", encoding="utf-8")

    report = build_maestro_report(str(db_path), command_hours=2, review_hours=24, log_path=str(log_path))

    assert "# Hermes Maestro Report" in report
    assert "## Command Center" in report
    assert "## Review + Recommendations" in report
    assert "## Tuning Proposals" in report


def test_build_recent_promo_lookup_report_answers_latest_prompt_locally(tmp_path: Path) -> None:
    db_path = tmp_path / "report.db"
    _seed_db(db_path)

    report = build_recent_promo_lookup_report(str(db_path), hours=2)

    assert "# Hermes Recent Promo Lookup" in report
    assert "## Latest Promo" in report
    assert "Alfamart fresh deal" in report
    assert "No SSH needed" in report
    assert "same-VM local Tanya database" in report


def test_build_recent_promo_lookup_report_handles_empty_window(tmp_path: Path) -> None:
    db_path = tmp_path / "report.db"
    _seed_db(db_path)

    report = build_recent_promo_lookup_report(str(db_path), hours=0)

    assert "# Hermes Recent Promo Lookup" in report
    assert "No promo found" in report
    assert "do not fall back to SSH" in report
