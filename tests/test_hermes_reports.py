from __future__ import annotations

import sqlite3
from pathlib import Path

from hermes_reports import build_alert_quality_report, build_service_health_report


def _seed_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

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
            (1, 101, -1, "2026-05-27 00:10:00+00:00", "Promo alfamart cashback 50%", 1, None, 0),
            (2, 102, -1, "2026-05-27 00:20:00+00:00", "Noise message", 1, "ai_skip", 0),
            (3, 103, -1, "2026-05-27 00:30:00+00:00", "GoFood diskon ongkir", 1, "ai_skip", 0),
            (4, 104, -1, "2026-05-27 00:40:00+00:00", "Pending queue message", 0, None, 2),
        ],
    )
    cur.executemany(
        "INSERT INTO promos (id, source_msg_id, summary, brand, tg_link, status, via_fastpath, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, 1, "Cashback 50% Alfamart", "Alfamart", "https://t.me/c/1/101", "active", 0, "2026-05-27 00:12:00+00:00"),
            (2, 3, "GoFood ongkir murah", "GoFood", "https://t.me/c/1/103", "active", 1, "2026-05-27 00:35:00+00:00"),
            (3, None, "GoFood flash sale", "GoFood", "https://t.me/c/1/999", "unknown", 1, "2026-05-27 01:00:00+00:00"),
        ],
    )
    cur.executemany(
        "INSERT INTO ai_corrections (id, original_msg_id, correction, weight, created_at) VALUES (?, ?, ?, ?, ?)",
        [
            (1, 1, "NOT_A_PROMO", 1.0, "2026-05-27 02:00:00+00:00"),
            (2, 3, "Missed cashback context", 0.8, "2026-05-27 02:05:00+00:00"),
        ],
    )
    cur.executemany(
        "INSERT INTO system_logs (id, level, logger_name, message, created_at) VALUES (?, ?, ?, ?, ?)",
        [
            (1, "WARNING", "listener", "Queue pressure rising", "2026-05-27 03:00:00+00:00"),
            (2, "ERROR", "processor", "Provider timeout OPENAI_API_KEY=super-secret", "2026-05-27 03:05:00+00:00"),
        ],
    )
    cur.executemany(
        "INSERT INTO failures (id, component, error_msg, created_at) VALUES (?, ?, ?, ?)",
        [
            (1, "processor", "timeout talking to provider", "2026-05-27 03:06:00+00:00"),
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

