from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

_WIB = timezone(timedelta(hours=7))


def _to_wib(utc_str: str | None) -> str:
    """Convert a UTC timestamp string to WIB (UTC+7) display format."""
    if not utc_str:
        return "unknown"
    try:
        dt = datetime.fromisoformat(utc_str.replace("+00:00", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        wib = dt.astimezone(_WIB)
        return wib.strftime("%Y-%m-%d %H:%M WIB")
    except (ValueError, TypeError):
        return utc_str

NEGATIVE_CORRECTIONS = {"NOT_A_PROMO", "SPAM_OR_NOISE"}
_SECRET_PATTERNS = [
    re.compile(r"\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|API_KEY|AUTH)[A-Z0-9_]*)=([^\s]+)"),
    re.compile(r"\b(ghp_[A-Za-z0-9]+)\b"),
]


def _default_db_path() -> str:
    try:
        from config import Config
    except Exception:
        return "tanya_main.db"
    return Config.DB_PATH


def _connect(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or _default_db_path()
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


def _safe_fetchall(conn: sqlite3.Connection, query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    try:
        return _fetchall(conn, query, params)
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc):
            return []
        raise


def _safe_fetchone_value(conn: sqlite3.Connection, query: str, params: tuple[Any, ...] = ()) -> Any:
    try:
        return _fetchone_value(conn, query, params)
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc):
            return 0
        raise


def _format_bullets(rows: list[str], empty_message: str) -> str:
    if not rows:
        return f"- {empty_message}"
    return "\n".join(f"- {row}" for row in rows)


def _data_plane_warning(total_messages: int, total_promos: int) -> str:
    if total_messages or total_promos:
        return ""
    return (
        "## Data Plane Warning\n"
        "- The local Tanya database has no messages or promos. This can mean the data plane is not running, "
        "is pointed at a different database, or has not ingested data yet.\n"
        "- Verify `tanyadfbot-runtime.service` on the same VM before treating an empty promo result as normal.\n"
    )


def _window_param(hours: int) -> tuple[str]:
    return (f"-{hours} hours",)


def _recent_promo_rows(
    conn: sqlite3.Connection,
    hours: int,
    limit: int = 5,
) -> list[sqlite3.Row]:
    return _safe_fetchall(
        conn,
        """
        SELECT p.created_at, p.brand, p.summary, p.tg_link, p.status, p.via_fastpath, m.text
        FROM promos p
        LEFT JOIN messages m ON m.id = p.source_msg_id
        WHERE p.created_at >= strftime('%Y-%m-%d %H:%M:%S+00:00','now', ?)
        ORDER BY p.created_at DESC
        LIMIT ?
        """,
        (*_window_param(hours), limit),
    )


def _brand_activity_rows(
    conn: sqlite3.Connection,
    hours: int,
    limit: int = 5,
) -> list[sqlite3.Row]:
    return _safe_fetchall(
        conn,
        """
        SELECT brand, COUNT(*) AS promo_count, MAX(created_at) AS latest_created_at
        FROM promos
        WHERE created_at >= strftime('%Y-%m-%d %H:%M:%S+00:00','now', ?)
        GROUP BY brand
        ORDER BY promo_count DESC, latest_created_at DESC, brand ASC
        LIMIT ?
        """,
        (*_window_param(hours), limit),
    )


def _review_payload(conn: sqlite3.Connection, hours: int) -> dict[str, Any]:
    promos_total = _safe_fetchone_value(
        conn,
        "SELECT COUNT(*) FROM promos WHERE created_at >= strftime('%Y-%m-%d %H:%M:%S+00:00','now', ?)",
        _window_param(hours),
    )
    corrections_total = _safe_fetchone_value(
        conn,
        "SELECT COUNT(*) FROM ai_corrections WHERE created_at >= strftime('%Y-%m-%d %H:%M:%S+00:00','now', ?)",
        _window_param(hours),
    )
    fastpath_total = _safe_fetchone_value(
        conn,
        "SELECT COUNT(*) FROM promos WHERE via_fastpath = 1 AND created_at >= strftime('%Y-%m-%d %H:%M:%S+00:00','now', ?)",
        _window_param(hours),
    )
    false_positive_rows = _safe_fetchall(
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
        _window_param(hours),
    )
    missed_signal_rows = _safe_fetchall(
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
        _window_param(hours),
    )
    brand_rows = _brand_activity_rows(conn, hours, limit=5)
    correction_rows = _safe_fetchall(
        conn,
        """
        SELECT correction, COUNT(*) AS correction_count
        FROM ai_corrections
        WHERE created_at >= strftime('%Y-%m-%d %H:%M:%S+00:00','now', ?)
        GROUP BY correction
        ORDER BY correction_count DESC, correction ASC
        LIMIT 5
        """,
        _window_param(hours),
    )
    queue_depth = _safe_fetchone_value(conn, "SELECT COUNT(*) FROM messages WHERE processed = 0")
    ai_failures = _safe_fetchone_value(conn, "SELECT COUNT(*) FROM messages WHERE ai_failure_count >= 2")
    failure_rows = _safe_fetchall(
        conn,
        """
        SELECT component, error_msg, created_at
        FROM failures
        WHERE created_at >= strftime('%Y-%m-%d %H:%M:%S+00:00','now', ?)
        ORDER BY created_at DESC
        LIMIT 5
        """,
        _window_param(hours),
    )
    return {
        "promos_total": promos_total,
        "corrections_total": corrections_total,
        "fastpath_total": fastpath_total,
        "false_positive_rows": false_positive_rows,
        "missed_signal_rows": missed_signal_rows,
        "brand_rows": brand_rows,
        "correction_rows": correction_rows,
        "queue_depth": queue_depth,
        "ai_failures": ai_failures,
        "failure_rows": failure_rows,
    }


def build_alert_quality_report(db_path: str | None = None, hours: int = 24) -> str:
    conn = _connect(db_path)
    try:
        promos_total = _safe_fetchone_value(
            conn,
            "SELECT COUNT(*) FROM promos WHERE created_at >= strftime('%Y-%m-%d %H:%M:%S+00:00','now', ?)",
            (f"-{hours} hours",),
        )
        corrections_total = _safe_fetchone_value(
            conn,
            "SELECT COUNT(*) FROM ai_corrections WHERE created_at >= strftime('%Y-%m-%d %H:%M:%S+00:00','now', ?)",
            (f"-{hours} hours",),
        )
        fastpath_total = _safe_fetchone_value(
            conn,
            "SELECT COUNT(*) FROM promos WHERE via_fastpath = 1 AND created_at >= strftime('%Y-%m-%d %H:%M:%S+00:00','now', ?)",
            (f"-{hours} hours",),
        )

        false_positive_rows = _safe_fetchall(
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
        missed_signal_rows = _safe_fetchall(
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
        brand_rows = _safe_fetchall(
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
        correction_rows = _safe_fetchall(
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


def _tail_lines(path: Path, tail_lines: int) -> list[str] | None:
    if not path.exists():
        return None
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
        queue_depth = _safe_fetchone_value(conn, "SELECT COUNT(*) FROM messages WHERE processed = 0")
        total_messages = _safe_fetchone_value(conn, "SELECT COUNT(*) FROM messages")
        total_promos = _safe_fetchone_value(conn, "SELECT COUNT(*) FROM promos")
        ai_failures = _safe_fetchone_value(conn, "SELECT COUNT(*) FROM messages WHERE ai_failure_count >= 2")
        data_plane_warning = _data_plane_warning(total_messages=total_messages, total_promos=total_promos)

        failure_rows = _safe_fetchall(
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
        log_rows = _safe_fetchall(
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
            f"{_to_wib(row['created_at'])} `{row['component']}`: {_redact_secrets(row['error_msg'])}"
            for row in failure_rows
        ]
        system_log_lines = [
            f"{_to_wib(row['created_at'])} `{row['level']}` `{row['logger_name'] or 'unknown'}`: {_redact_secrets(row['message'])}"
            for row in log_rows
        ]

        tailed_log_lines: list[str] | None = None
        if log_path:
            tailed_log_lines = _tail_lines(Path(log_path), tail_lines)

        report = (
            "# Hermes Service Health Report\n\n"
            "## Runtime Snapshot\n"
            f"- Lookback window: last {hours} hours\n"
            f"- Queue depth: {queue_depth}\n"
            f"- Total messages: {total_messages}\n"
            f"- Total promos: {total_promos}\n"
            f"- Messages with repeated AI failures: {ai_failures}\n\n"
            f"{data_plane_warning}"
            f"{'\n' if data_plane_warning else ''}"
            "## Recent Failures\n"
            f"{_format_bullets(failure_lines, 'No failures recorded in the lookback window.')}\n\n"
            "## Recent System Logs\n"
            f"{_format_bullets(system_log_lines, 'No warning/error logs recorded in the lookback window.')}\n"
        )
        if log_path:
            if tailed_log_lines is None:
                tail_block = f"- Log file not found at {log_path}."
            elif tailed_log_lines:
                tail_block = _format_bullets(
                    [_redact_secrets(line) for line in tailed_log_lines],
                    f"Log file empty at {log_path}.",
                )
            else:
                tail_block = f"- Log file empty at {log_path}."
            report += (
                "\n## Log Tail\n"
                f"{tail_block}\n"
            )
        return report
    finally:
        conn.close()


def build_command_center_report(
    db_path: str | None = None,
    hours: int = 2,
    limit: int = 5,
    log_path: str | None = None,
    tail_lines: int = 20,
) -> str:
    conn = _connect(db_path)
    try:
        promo_rows = _recent_promo_rows(conn, hours=hours, limit=limit)
        brand_rows = _brand_activity_rows(conn, hours=hours, limit=limit)
        queue_depth = _safe_fetchone_value(conn, "SELECT COUNT(*) FROM messages WHERE processed = 0")
        total_promos = _safe_fetchone_value(
            conn,
            "SELECT COUNT(*) FROM promos WHERE created_at >= strftime('%Y-%m-%d %H:%M:%S+00:00','now', ?)",
            _window_param(hours),
        )
        total_messages = _safe_fetchone_value(
            conn,
            "SELECT COUNT(*) FROM messages WHERE timestamp >= strftime('%Y-%m-%d %H:%M:%S+00:00','now', ?)",
            _window_param(hours),
        )
        recent_promo_lines = [
            (
                f"{_to_wib(row['created_at'])} `{row['brand'] or 'Unknown'}`: {row['summary'] or 'no AI summary'}"
                f" | status={row['status'] or 'unknown'}"
                f" | fastpath={'yes' if row['via_fastpath'] else 'no'}"
                + (f" | link={row['tg_link']}" if row["tg_link"] else "")
            )
            for row in promo_rows
        ]
        hot_brand_lines = [
            f"{row['brand'] or 'Unknown'}: {row['promo_count']} promos (latest {row['latest_created_at']})"
            for row in brand_rows
        ]
        health_report = build_service_health_report(
            db_path=db_path,
            hours=hours,
            log_path=log_path,
            tail_lines=tail_lines,
        )
        health_snapshot = [
            "- Recent activity snapshot:",
            f"- Messages seen in the window: {total_messages}",
            f"- Promos seen in the window: {total_promos}",
            f"- Queue depth: {queue_depth}",
        ]
        return (
            "# Hermes Maestro Command Center\n\n"
            "## Latest Promo\n"
            f"{_format_bullets(recent_promo_lines, f'No promo found in the last {hours} hours.')}\n\n"
            "## What's Hot Right Now\n"
            f"{_format_bullets(hot_brand_lines, f'No promo brands observed in the last {hours} hours.')}\n\n"
            "## Runtime Health\n"
            f"{'\n'.join(health_snapshot)}\n\n"
            "## Service Details\n"
            f"{health_report}\n"
        )
    finally:
        conn.close()


def build_recent_promo_lookup_report(
    db_path: str | None = None,
    hours: int = 2,
    brand: str | None = None,
    limit: int = 5,
    today: bool = False,
) -> str:
    conn = _connect(db_path)
    try:
        if today:
            now_wib = datetime.now(_WIB)
            midnight_wib = now_wib.replace(hour=0, minute=0, second=0, microsecond=0)
            hours = max(1, int((now_wib - midnight_wib).total_seconds() / 3600) + 1)
        params: list[Any] = [f"-{hours} hours"]
        brand_filter = ""
        if brand:
            brand_filter = " AND LOWER(p.brand) = LOWER(?)"
            params.append(brand)
        params.append(limit)
        promo_rows = _safe_fetchall(
            conn,
            f"""
            SELECT p.created_at, p.brand, p.summary, p.tg_link, p.status, p.via_fastpath
            FROM promos p
            WHERE p.created_at >= strftime('%Y-%m-%d %H:%M:%S+00:00','now', ?)
            {brand_filter}
            ORDER BY p.created_at DESC
            LIMIT ?
            """,
            tuple(params),
        )
        total_messages = _safe_fetchone_value(conn, "SELECT COUNT(*) FROM messages")
        total_promos = _safe_fetchone_value(conn, "SELECT COUNT(*) FROM promos")
        data_plane_warning = _data_plane_warning(total_messages=total_messages, total_promos=total_promos)
        latest_lines = [
            (
                f"{_to_wib(row['created_at'])} `{row['brand'] or 'Unknown'}`: {row['summary'] or 'no AI summary'}"
                f" | status={row['status'] or 'unknown'}"
                f" | fastpath={'yes' if row['via_fastpath'] else 'no'}"
                + (f" | link={row['tg_link']}" if row["tg_link"] else "")
            )
            for row in promo_rows
        ]
        scope_line = f"- Brand filter: `{brand}`\n" if brand else ""
        empty_message = (
            f"No promo found in the last {hours} hours"
            + (f" for brand `{brand}`." if brand else ".")
            + " Stay local and do not fall back to SSH or remote host probing."
        )
        return (
            "# Hermes Recent Promo Lookup\n\n"
            "## Query Policy\n"
            "- Use the same-VM local Tanya database for recent promo questions.\n"
            "- No SSH needed. Do not inspect deprecated remote hosts for normal promo lookup.\n"
            f"- Lookback window: last {hours} hours\n"
            f"{scope_line}"
            "\n"
            f"{data_plane_warning}"
            f"{'\n' if data_plane_warning else ''}"
            "## Latest Promo\n"
            f"{_format_bullets(latest_lines, empty_message)}\n"
        )
    finally:
        conn.close()


def build_review_recommendations_report(db_path: str | None = None, hours: int = 24) -> str:
    conn = _connect(db_path)
    try:
        payload = _review_payload(conn, hours)
        false_positive_lines = [
            (
                f"`{row['correction']}` weight={row['weight']}: "
                f"{row['brand'] or 'Unknown'} -> {row['summary'] or 'no AI summary'} | "
                f"msg={_redact_secrets(row['text'])[:160] or 'missing'}"
                + (f" | link={row['tg_link']}" if row["tg_link"] else "")
            )
            for row in payload["false_positive_rows"]
        ]
        missed_signal_lines = [
            (
                f"`{row['correction']}` weight={row['weight']}: "
                f"skip_reason={row['skip_reason'] or 'none'} | "
                f"brand={row['brand'] or 'Unknown'} | msg={_redact_secrets(row['text'])[:160] or 'missing'}"
                + (f" | link={row['tg_link']}" if row["tg_link"] else "")
            )
            for row in payload["missed_signal_rows"]
        ]
        overfire_lines = [
            f"{row['brand'] or 'Unknown'}: {row['promo_count']} promos in the last {hours}h"
            + (" (watch for overfire)" if row["promo_count"] > 1 else "")
            for row in payload["brand_rows"]
        ]
        correction_lines = [f"{row['correction']}: {row['correction_count']}" for row in payload["correction_rows"]]
        recommendations: list[str] = []
        if payload["failure_rows"] or payload["queue_depth"] or payload["ai_failures"]:
            recommendations.append("Investigate runtime health: queue pressure, repeated AI failures, or recent failures are visible in the window.")
        if payload["false_positive_rows"]:
            recommendations.append("Investigate false-positive controls: review `config/trigger_terms.yaml`, `skills/false-positive-patterns.md`, and `prompts/promo_judge.md`.")
        if payload["missed_signal_rows"]:
            recommendations.append("Monitor missed-signal recovery: expand `skills/discountfess-lingo.md` and tighten `config/trigger_terms.yaml` for the observed phrasing.")
        if payload["brand_rows"] and any(row["promo_count"] > 1 for row in payload["brand_rows"]):
            recommendations.append("Monitor repeat-fire brands: refine dedupe or brand-specific gating before broadening automation.")
        if not recommendations:
            recommendations.append("Continue monitoring: the current lookback window did not surface a strong tuning signal.")
        return (
            "# Hermes Review + Recommendations\n\n"
            "## Summary\n"
            f"- Lookback window: last {hours} hours\n"
            f"- Promos observed: {payload['promos_total']}\n"
            f"- AI corrections received: {payload['corrections_total']}\n"
            f"- Fast-path promos: {payload['fastpath_total']}\n\n"
            "## Findings\n"
            "### False Positive Candidates\n"
            f"{_format_bullets(false_positive_lines, 'No negative corrections recorded in the lookback window.')}\n\n"
            "### Missed Signal Candidates\n"
            f"{_format_bullets(missed_signal_lines, 'No positive corrections recorded in the lookback window.')}\n\n"
            "### Overfire Hints\n"
            f"{_format_bullets(overfire_lines, 'No repeat-fire brands crossed the report threshold.')}\n\n"
            "### Correction Labels\n"
            f"{_format_bullets(correction_lines, 'No corrections recorded in the lookback window.')}\n\n"
            "## Recommendations\n"
            f"{_format_bullets(recommendations, 'Continue monitoring; no actionable recommendation yet.')}\n"
        )
    finally:
        conn.close()


def build_tuning_proposal_report(db_path: str | None = None, hours: int = 24) -> str:
    conn = _connect(db_path)
    try:
        payload = _review_payload(conn, hours)
        proposals: list[str] = []
        if payload["false_positive_rows"]:
            proposals.append(
                "Update `skills/false-positive-patterns.md` and `config/trigger_terms.yaml` with the observed false-positive phrases before changing any runtime logic."
            )
            proposals.append(
                "Refine `prompts/promo_judge.md` to down-rank the specific false-positive phrasing while preserving strong deal signals."
            )
        if payload["missed_signal_rows"]:
            proposals.append(
                "Expand `skills/discountfess-lingo.md` with the missed-signal phrasing and mirror the new terms into `config/trigger_terms.yaml`."
            )
            proposals.append(
                "Adjust `prompts/promo_judge.md` so the observed missed-signal phrasing gets stronger promo interpretation guidance."
            )
        if payload["brand_rows"] and any(row["promo_count"] > 1 for row in payload["brand_rows"]):
            proposals.append(
                "Review `skills/promo-review.md` for repeat-fire brand guidance and consider a targeted dedupe note in `config/trigger_terms.yaml`."
            )
        if payload["failure_rows"] or payload["queue_depth"] or payload["ai_failures"]:
            proposals.append(
                "Pause tuning changes until runtime health is clear; queue pressure and/or repeated failures should be fixed before adjusting control-plane assets."
            )
        if not proposals:
            proposals.append("No tuning proposal needed from this window; keep observing before proposing YAML, prompt, or skill diffs.")
        evidence_lines = [
            f"Promos: {payload['promos_total']}, corrections: {payload['corrections_total']}, fast-path: {payload['fastpath_total']}, queue depth: {payload['queue_depth']}, repeated AI failures: {payload['ai_failures']}"
        ]
        evidence_lines.extend(
            [
                f"False positives: {len(payload['false_positive_rows'])}",
                f"Missed signals: {len(payload['missed_signal_rows'])}",
            ]
        )
        return (
            "# Hermes Tuning Proposals\n\n"
            "## Evidence\n"
            f"{_format_bullets(evidence_lines, 'No evidence gathered.')}\n\n"
            "## Proposed Diffs\n"
            f"{_format_bullets(proposals, 'No reviewable diffs proposed in the current window.')}\n\n"
            "## Guardrail\n"
            "- These are reviewable proposals only. No file should be mutated automatically in this roadmap.\n"
        )
    finally:
        conn.close()


def build_maestro_report(
    db_path: str | None = None,
    command_hours: int = 2,
    review_hours: int = 24,
    log_path: str | None = None,
    tail_lines: int = 20,
) -> str:
    return (
        "# Hermes Maestro Report\n\n"
        "## Command Center\n"
        f"{build_command_center_report(db_path=db_path, hours=command_hours, log_path=log_path, tail_lines=tail_lines)}\n\n"
        "## Review + Recommendations\n"
        f"{build_review_recommendations_report(db_path=db_path, hours=review_hours)}\n\n"
        "## Tuning Proposals\n"
        f"{build_tuning_proposal_report(db_path=db_path, hours=review_hours)}\n"
    )
