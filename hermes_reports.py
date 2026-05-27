from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from config import Config

NEGATIVE_CORRECTIONS = {"NOT_A_PROMO", "SPAM_OR_NOISE"}
_SECRET_PATTERNS = [
    re.compile(r"\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|API_KEY|AUTH)[A-Z0-9_]*)=([^\s]+)"),
    re.compile(r"\b(ghp_[A-Za-z0-9]+)\b"),
]


def _connect(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or Config.DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _redact_secrets(text: str | None) -> str:
    if not text:
        return ""
    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(lambda match: f"{match.group(1)}=[REDACTED]" if match.lastindex and match.lastindex > 1 else "[REDACTED]", redacted)
    return redacted


def _fetchall(conn: sqlite3.Connection, query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    cur = conn.execute(query, params)
    return list(cur.fetchall())


def _fetchone_value(conn: sqlite3.Connection, query: str, params: tuple[Any, ...] = ()) -> Any:
    cur = conn.execute(query, params)
    row = cur.fetchone()
    return row[0] if row else 0


def _format_bullets(rows: list[str], empty_message: str) -> str:
    if not rows:
        return f"- {empty_message}"
    return "\n".join(f"- {row}" for row in rows)


def build_alert_quality_report(db_path: str | None = None, hours: int = 24) -> str:
    conn = _connect(db_path)
    try:
        promos_total = _fetchone_value(
            conn,
            "SELECT COUNT(*) FROM promos WHERE created_at >= strftime('%Y-%m-%d %H:%M:%S+00:00','now', ?)",
            (f"-{hours} hours",),
        )
        corrections_total = _fetchone_value(
            conn,
            "SELECT COUNT(*) FROM ai_corrections WHERE created_at >= strftime('%Y-%m-%d %H:%M:%S+00:00','now', ?)",
            (f"-{hours} hours",),
        )
        fastpath_total = _fetchone_value(
            conn,
            "SELECT COUNT(*) FROM promos WHERE via_fastpath = 1 AND created_at >= strftime('%Y-%m-%d %H:%M:%S+00:00','now', ?)",
            (f"-{hours} hours",),
        )

        false_positive_rows = _fetchall(
            conn,
            """
            SELECT ac.correction, ac.weight, m.text, p.brand, p.summary, p.tg_link
            FROM ai_corrections ac
            LEFT JOIN messages m ON m.id = ac.original_msg_id
            LEFT JOIN promos p ON p.source_msg_id = ac.original_msg_id
            WHERE ac.created_at >= strftime('%Y-%m-%d %H:%M:%S+00:00','now', ?)
              AND ac.correction IN ('NOT_A_PROMO', 'SPAM_OR_NOISE')
            ORDER BY ac.weight DESC, ac.created_at DESC
            LIMIT 5
            """,
            (f"-{hours} hours",),
        )
        missed_signal_rows = _fetchall(
            conn,
            """
            SELECT ac.correction, ac.weight, m.text, m.skip_reason, p.brand, p.summary, p.tg_link
            FROM ai_corrections ac
            LEFT JOIN messages m ON m.id = ac.original_msg_id
            LEFT JOIN promos p ON p.source_msg_id = ac.original_msg_id
            WHERE ac.created_at >= strftime('%Y-%m-%d %H:%M:%S+00:00','now', ?)
              AND ac.correction NOT IN ('NOT_A_PROMO', 'SPAM_OR_NOISE')
            ORDER BY ac.weight DESC, ac.created_at DESC
            LIMIT 5
            """,
            (f"-{hours} hours",),
        )
        brand_rows = _fetchall(
            conn,
            """
            SELECT brand, COUNT(*) AS promo_count
            FROM promos
            WHERE created_at >= strftime('%Y-%m-%d %H:%M:%S+00:00','now', ?)
            GROUP BY brand
            ORDER BY promo_count DESC, brand ASC
            LIMIT 5
            """,
            (f"-{hours} hours",),
        )
        correction_rows = _fetchall(
            conn,
            """
            SELECT correction, COUNT(*) AS correction_count
            FROM ai_corrections
            WHERE created_at >= strftime('%Y-%m-%d %H:%M:%S+00:00','now', ?)
            GROUP BY correction
            ORDER BY correction_count DESC, correction ASC
            LIMIT 5
            """,
            (f"-{hours} hours",),
        )

        false_positive_lines = [
            (
                f"`{row['correction']}` weight={row['weight']}: "
                f"{row['brand'] or 'Unknown'} -> {row['summary'] or 'no AI summary'} | "
                f"msg={_redact_secrets(row['text'])[:160] or 'missing'}"
                + (f" | link={row['tg_link']}" if row["tg_link"] else "")
            )
            for row in false_positive_rows
        ]
        missed_signal_lines = [
            (
                f"`{row['correction']}` weight={row['weight']}: "
                f"skip_reason={row['skip_reason'] or 'none'} | "
                f"brand={row['brand'] or 'Unknown'} | msg={_redact_secrets(row['text'])[:160] or 'missing'}"
                + (f" | link={row['tg_link']}" if row["tg_link"] else "")
            )
            for row in missed_signal_rows
        ]
        overfire_lines = [
            f"{row['brand']}: {row['promo_count']} promos in the last {hours}h"
            + (" (high fast-path volume possible)" if row["promo_count"] > 1 else "")
            for row in brand_rows
        ]
        brand_lines = [f"{row['brand']}: {row['promo_count']}" for row in brand_rows]
        correction_lines = [f"{row['correction']}: {row['correction_count']}" for row in correction_rows]

        return (
            "# Hermes Alert Quality Report\n\n"
            "## Summary\n"
            f"- Lookback window: last {hours} hours\n"
            f"- Promos observed: {promos_total}\n"
            f"- AI corrections received: {corrections_total}\n"
            f"- Fast-path promos: {fastpath_total}\n\n"
            "## False Positive Candidates\n"
            f"{_format_bullets(false_positive_lines, 'No negative corrections recorded in the lookback window.')}\n\n"
            "## Missed Signal Candidates\n"
            f"{_format_bullets(missed_signal_lines, 'No positive corrections recorded in the lookback window.')}\n\n"
            "## Duplicate / Overfire Hints\n"
            f"{_format_bullets(overfire_lines, 'No repeat-fire brands crossed the report threshold.')}\n\n"
            "## Top Brands\n"
            f"{_format_bullets(brand_lines, 'No promos recorded in the lookback window.')}\n\n"
            "## Top Correction Labels\n"
            f"{_format_bullets(correction_lines, 'No corrections recorded in the lookback window.')}\n"
        )
    finally:
        conn.close()


def _tail_lines(path: Path, tail_lines: int) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return lines[-tail_lines:]


def build_service_health_report(
    db_path: str | None = None,
    hours: int = 24,
    log_path: str | None = None,
    tail_lines: int = 20,
) -> str:
    conn = _connect(db_path)
    try:
        queue_depth = _fetchone_value(conn, "SELECT COUNT(*) FROM messages WHERE processed = 0")
        total_messages = _fetchone_value(conn, "SELECT COUNT(*) FROM messages")
        total_promos = _fetchone_value(conn, "SELECT COUNT(*) FROM promos")
        ai_failures = _fetchone_value(conn, "SELECT COUNT(*) FROM messages WHERE ai_failure_count >= 2")

        failure_rows = _fetchall(
            conn,
            """
            SELECT component, error_msg, created_at
            FROM failures
            WHERE created_at >= strftime('%Y-%m-%d %H:%M:%S+00:00','now', ?)
            ORDER BY created_at DESC
            LIMIT 5
            """,
            (f"-{hours} hours",),
        )
        log_rows = _fetchall(
            conn,
            """
            SELECT level, logger_name, message, created_at
            FROM system_logs
            WHERE created_at >= strftime('%Y-%m-%d %H:%M:%S+00:00','now', ?)
            ORDER BY created_at DESC
            LIMIT 8
            """,
            (f"-{hours} hours",),
        )

        failure_lines = [
            f"{row['created_at']} `{row['component']}`: {_redact_secrets(row['error_msg'])}"
            for row in failure_rows
        ]
        system_log_lines = [
            f"{row['created_at']} `{row['level']}` `{row['logger_name'] or 'unknown'}`: {_redact_secrets(row['message'])}"
            for row in log_rows
        ]

        tailed_log_lines: list[str] = []
        if log_path:
            tailed_log_lines = [_redact_secrets(line) for line in _tail_lines(Path(log_path), tail_lines)]

        report = (
            "# Hermes Service Health Report\n\n"
            "## Runtime Snapshot\n"
            f"- Lookback window: last {hours} hours\n"
            f"- Queue depth: {queue_depth}\n"
            f"- Total messages: {total_messages}\n"
            f"- Total promos: {total_promos}\n"
            f"- Messages with repeated AI failures: {ai_failures}\n\n"
            "## Recent Failures\n"
            f"{_format_bullets(failure_lines, 'No failures recorded in the lookback window.')}\n\n"
            "## Recent System Logs\n"
            f"{_format_bullets(system_log_lines, 'No warning/error logs recorded in the lookback window.')}\n"
        )
        if log_path:
            report += (
                "\n## Log Tail\n"
                f"{_format_bullets(tailed_log_lines, f'Log file not found at {log_path}.')}\n"
            )
        return report
    finally:
        conn.close()
