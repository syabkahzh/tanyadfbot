#!/usr/bin/env python3
"""hermes_feedback_loop.py — Analyze ai_corrections and suggest pattern fixes.

Reads feedback from the ai_corrections table, identifies patterns,
and suggests specific code changes to processor.py.

Usage:
    PYTHONPATH=. .venv/bin/python tools/hermes_feedback_loop.py --hours 24
"""

import sqlite3
import re
import argparse
from collections import Counter, defaultdict
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "tanya_main.db"


def get_corrections(hours: int = 24) -> list[dict]:
    """Fetch recent corrections from ai_corrections table."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        """
        SELECT ac.id, ac.original_msg_id, ac.brand, ac.summary, 
               ac.correction, ac.weight, ac.created_at,
               m.text as raw_msg
        FROM ai_corrections ac
        LEFT JOIN messages m ON ac.original_msg_id = m.id
        WHERE ac.created_at > datetime('now', ? || ' hours')
        ORDER BY ac.created_at DESC
        """,
        (f"-{hours}",)
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def analyze_patterns(corrections: list[dict]) -> dict:
    """Identify patterns in corrections."""
    analysis = {
        "total": len(corrections),
        "by_type": Counter(),
        "raw_msg_patterns": defaultdict(list),
        "brand_issues": [],
        "suggested_patterns": [],
    }
    
    for c in corrections:
        correction_type = c.get("correction", "unknown")
        analysis["by_type"][correction_type] += 1
        
        raw_msg = c.get("raw_msg", "") or ""
        
        # Analyze NOT_A_PROMO patterns
        if correction_type == "NOT_A_PROMO":
            # Extract common words/phrases
            words = raw_msg.lower().split()
            analysis["raw_msg_patterns"]["NOT_A_PROMO"].append({
                "msg_id": c["original_msg_id"],
                "text": raw_msg[:100],
                "words": words[:10],
            })
        
        # Analyze Wrong brand patterns
        elif correction_type == "Wrong brand":
            analysis["brand_issues"].append({
                "msg_id": c["original_msg_id"],
                "text": raw_msg[:100],
                "brand": c.get("brand"),
            })
    
    # Generate suggested patterns from NOT_A_PROMO
    not_promo_msgs = [p["text"] for p in analysis["raw_msg_patterns"]["NOT_A_PROMO"]]
    
    # Common complaint words
    complaint_words = [
        "gangguan", "ga bisa", "gak bisa", "belum on", "belum aktif",
        "habis terus", "ga dapet", "gak dapet", "kena refund", "dibatalkan",
        "rusak", "error", "bug", "ga jalan", "gak jalan", "tiba2",
    ]
    
    found_patterns = []
    for word in complaint_words:
        count = sum(1 for msg in not_promo_msgs if word in msg.lower())
        if count >= 2:
            found_patterns.append((word, count))
    
    analysis["suggested_patterns"] = sorted(found_patterns, key=lambda x: -x[1])
    
    return analysis


def print_report(analysis: dict):
    """Print human-readable analysis report."""
    print("=" * 60)
    print("📊 FEEDBACK LOOP ANALYSIS")
    print("=" * 60)
    
    print(f"\nTotal corrections: {analysis['total']}")
    
    print("\nBy Type:")
    for ctype, count in analysis["by_type"].most_common():
        print(f"  {ctype}: {count}")
    
    if analysis["suggested_patterns"]:
        print("\n🔍 Suggested Pattern Additions (NOT_A_PROMO):")
        for pattern, count in analysis["suggested_patterns"]:
            print(f"  • '{pattern}' — found in {count} messages")
    
    if analysis["brand_issues"]:
        print("\n⚠️ Brand Issues:")
        for issue in analysis["brand_issues"][:5]:
            print(f"  • msg {issue['msg_id']}: {issue['text'][:50]}...")
    
    print("\n" + "=" * 60)
    print("ACTION ITEMS:")
    print("1. Review suggested patterns above")
    print("2. Add new patterns to _COMPLAINT_PATTERN in processor.py")
    print("3. Test with: PYTHONPATH=. .venv/bin/python -c 'from processor import _COMPLAINT_PATTERN; print(\"OK\")'")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Analyze feedback patterns")
    parser.add_argument("--hours", type=int, default=24, help="Lookback window")
    args = parser.parse_args()
    
    corrections = get_corrections(args.hours)
    analysis = analyze_patterns(corrections)
    print_report(analysis)


if __name__ == "__main__":
    main()
