import sqlite3
import os
import sys

# Add the parent directory to the path so we can import your Config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import Config
    DB_PATH = Config.DB_PATH
except ImportError:
    # Fallback just in case you run it from a weird directory
    DB_PATH = "tanyadfbot.db"


def export_feedback():
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Pull the corrections, the original message, and the AI's bad guess
    query = """
    SELECT
        ac.id, ac.created_at, ac.weight, ac.correction,
        m.text as original_text,
        p.brand as ai_brand, p.summary as ai_summary
    FROM ai_corrections ac
    LEFT JOIN messages m ON ac.original_msg_id = m.id
    LEFT JOIN promos p ON p.source_msg_id = m.id
    ORDER BY ac.weight DESC, ac.created_at DESC
    LIMIT 50;
    """

    cur.execute(query)
    rows = cur.fetchall()

    if not rows:
        print("🤷‍♂️ No feedback found in the database yet.")
        return

    out_file = "latest_feedback_export.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("# 🔧 AI Feedback Export\n\n")
        for r in rows:
            f.write(f"### Correction ID: {r['id']} | Weight: {r['weight']}\n")
            f.write(f"**Date:** `{r['created_at']}`\n")

            orig_text = str(r['original_text']).replace('\n', ' ')
            f.write(f"**Original Message:**\n> {orig_text}\n\n")

            f.write(f"**What the AI Guessed:**\n* Brand: `{r['ai_brand']}`\n* Summary: `{r['ai_summary']}`\n\n")
            f.write(f"**Your Correction:**\n> 🚨 **{r['correction']}**\n\n")
            f.write("---\n\n")

    print(f"✅ Exported {len(rows)} corrections to {out_file}")
    print("Copy the contents of that file and paste it to the AI for analysis.")


if __name__ == "__main__":
    export_feedback()
