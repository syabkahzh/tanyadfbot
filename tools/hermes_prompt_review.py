"""tools/hermes_prompt_review.py — CLI for Hermes to review the extraction prompt.

Shows the current extraction prompt, recent AI outputs, and corrections
so Hermes can evaluate whether the prompt needs changing.

Usage:
    PYTHONPATH=. .venv/bin/python tools/hermes_prompt_review.py [--hours 24] [--limit 20]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sqlite3
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processor import _EXTRACT_SYSTEM


def _connect(db_path: str | None) -> sqlite3.Connection:
    path = db_path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tanya_main.db",
    )
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def main() -> int:
    parser = argparse.ArgumentParser(description="Review extraction prompt and recent AI outputs.")
    parser.add_argument("--hours", type=int, default=24, help="Lookback window for outputs/corrections.")
    parser.add_argument("--limit", type=int, default=10, help="Max rows to show.")
    parser.add_argument("--db-path", default=None, help="Override DB path.")
    args = parser.parse_args()

    conn = _connect(args.db_path)

    # ── Current prompt ────────────────────────────────────────────────────
    print("# Extraction Prompt Review\n")

    # Check for override
    try:
        row = conn.execute(
            "SELECT value, updated_at FROM hermes_config WHERE key='extraction_prompt'"
        ).fetchone()
    except sqlite3.OperationalError:
        row = None

    if row:
        print(f"**Status:** Custom override (set {row['updated_at']})\n")
        print("```")
        print(row["value"][:2000])
        if len(row["value"]) > 2000:
            print(f"... ({len(row['value'])} chars total)")
        print("```\n")
    else:
        print("**Status:** Using default prompt\n")
        print("```")
        print(_EXTRACT_SYSTEM[:2000])
        if len(_EXTRACT_SYSTEM) > 2000:
            print(f"... ({len(_EXTRACT_SYSTEM)} chars total)")
        print("```\n")

    # ── Recent AI extractions ─────────────────────────────────────────────
    try:
        rows = conn.execute(
            "SELECT p.brand, p.summary, p.status, p.created_at, p.confidence "
            "FROM promos p "
            "WHERE p.created_at >= strftime('%Y-%m-%d %H:%M:%S+00:00','now', ?) "
            "ORDER BY p.created_at DESC LIMIT ?",
            (f"-{args.hours} hours", args.limit),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []

    print(f"## Recent Extractions (last {args.hours}h)\n")
    if rows:
        print("| Time | Brand | Summary | Status | Conf |")
        print("|------|-------|---------|--------|------|")
        for r in rows:
            summary = (r["summary"] or "")[:50]
            print(f"| {r['created_at'][:16]} | {r['brand']} | {summary} | {r['status']} | {r['confidence'] or '-'} |")
    else:
        print("*No extractions in this period.*")
    print()

    # ── Recent corrections ────────────────────────────────────────────────
    try:
        rows = conn.execute(
            "SELECT original_msg_id, brand, summary, correction, created_at "
            "FROM ai_corrections "
            "WHERE created_at >= strftime('%Y-%m-%d %H:%M:%S+00:00','now', ?) "
            "ORDER BY created_at DESC LIMIT ?",
            (f"-{args.hours} hours", args.limit),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []

    print(f"## Recent Corrections (last {args.hours}h)\n")
    if rows:
        print("| Time | Brand | Summary | Correction |")
        print("|------|-------|---------|------------|")
        for r in rows:
            summary = (r["summary"] or "")[:40]
            print(f"| {r['created_at'][:16]} | {r['brand'] or '-'} | {summary} | {r['correction']} |")
    else:
        print("*No corrections in this period.*")
    print()

    # ── AI failure stats ──────────────────────────────────────────────────
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS total FROM messages "
            "WHERE ai_failure_count > 0 AND timestamp >= strftime('%Y-%m-%d %H:%M:%S+00:00','now', ?)",
            (f"-{args.hours} hours",),
        ).fetchone()
        failures = row["total"] if row else 0
    except sqlite3.OperationalError:
        failures = 0

    print(f"## AI Failures: {failures} messages with failures in last {args.hours}h\n")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
