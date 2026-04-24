#!/usr/bin/env python3
"""
Export labeled training data from messages.db for FastText binary classifier.
Run locally: python tools/export_training_data.py --db messages.db --out training.txt

Positive = messages that resulted in a promo extraction (promos table join).
Negative = messages Gemini explicitly skipped (skip_reason = 'ai_skip').
           Do NOT include skip_reason = 'regex' — those never hit the AI.
"""
import argparse
import re
import sqlite3
from pathlib import Path


def clean(text: str) -> str:
    """Normalize text for FastText: lowercase, collapse whitespace."""
    if not text: return ""
    # Collapsing whitespace and lowercase
    return re.sub(r"\s+", " ", text.lower()).strip()


def export(db_path: str, out_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    print(f"🔍 Reading data from {db_path}...")

    # Positives: messages that became promos
    cur.execute("""
        SELECT DISTINCT m.text
        FROM messages m
        INNER JOIN promos p ON m.id = p.source_msg_id
        WHERE m.text IS NOT NULL 
        AND length(trim(m.text)) > 5
    """)
    positives = [row["text"] for row in cur.fetchall()]

    # Negatives: messages Gemini explicitly skipped
    # Limit JUNK to roughly 2x the number of positives to prevent class imbalance
    neg_limit = len(positives) * 2
    cur.execute(f"""
        SELECT DISTINCT text
        FROM messages
        WHERE skip_reason = 'ai_skip'
        AND text IS NOT NULL AND length(trim(text)) > 5
        ORDER BY RANDOM()
        LIMIT {neg_limit}
    """)
    negatives = [row["text"] for row in cur.fetchall()]
    conn.close()

    if len(positives) < 100 or len(negatives) < 100:
        print(f"⚠️  Not enough data yet: {len(positives)} pos / {len(negatives)} neg")
        print("   Keep the bot running with skip_reason tagging for 24-48 hours first.")
        return

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for text in positives:
            f.write(f"__label__PROMO {clean(text)}\n")
        for text in negatives:
            f.write(f"__label__JUNK {clean(text)}\n")

    print(f"✅ Wrote {len(positives)} PROMO + {len(negatives)} JUNK → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--out", default="data/training.txt")
    args = p.parse_args()
    export(args.db, args.out)
