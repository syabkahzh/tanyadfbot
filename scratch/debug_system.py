import sqlite3
import os

db_path = 'tanya_main.db'

def get_db():
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def check():
    conn = get_db()
    cur = conn.cursor()

    print("--- 🛠️ TANYA DF SYSTEM HEALTH CHECK ---")
    
    # 1. Message Processing Stats
    cur.execute("SELECT processed, COUNT(*) as count FROM messages GROUP BY processed")
    rows = cur.fetchall()
    print("\nProcessing Stats:")
    for r in rows:
        status = "Processed" if r['processed'] == 1 else "Unprocessed"
        print(f"  {status}: {r['count']}")

    # 2. Skip Reasons
    cur.execute("SELECT skip_reason, COUNT(*) as count FROM messages WHERE processed=1 AND skip_reason IS NOT NULL GROUP BY skip_reason")
    rows = cur.fetchall()
    print("\nSkip Reasons:")
    for r in rows:
        print(f"  {r['skip_reason']}: {r['count']}")

    # 3. AI Failures
    cur.execute("SELECT COUNT(*) FROM messages WHERE ai_failure_count > 0")
    fail_count = cur.fetchone()[0]
    print(f"\nMessages with AI Failures: {fail_count}")
    if fail_count > 0:
        cur.execute("SELECT id, text, ai_failure_count FROM messages WHERE ai_failure_count > 0 ORDER BY ai_failure_count DESC LIMIT 5")
        for r in cur.fetchall():
            print(f"  ID:{r['id']} | Fails:{r['ai_failure_count']} | Text: {r['text'][:50]}...")

    # 4. Recent System Logs
    print("\nRecent System Logs (Last 10 WARNING/ERROR):")
    cur.execute("SELECT level, message, created_at FROM system_logs WHERE level IN ('WARNING', 'ERROR', 'CRITICAL') ORDER BY created_at DESC LIMIT 10")
    for r in cur.fetchall():
        print(f"  [{r['created_at']}] {r['level']}: {r['message'][:100]}")

    # 5. Failures Table
    print("\nRecent Recorded Failures:")
    cur.execute("SELECT component, error_msg, created_at FROM failures ORDER BY created_at DESC LIMIT 5")
    for r in cur.fetchall():
        print(f"  [{r['created_at']}] {r['component']}: {r['error_msg'][:100]}")

    # 6. Promos Count
    cur.execute("SELECT COUNT(*) FROM promos")
    promos_count = cur.fetchone()[0]
    print(f"\nTotal Promos Extracted: {promos_count}")

    conn.close()

if __name__ == '__main__':
    check()
