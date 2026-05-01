import sqlite3
import sys
from datetime import datetime

def read_logs(limit=10):
    try:
        conn = sqlite3.connect('tanya_main.db')
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        print(f"--- Showing last {limit} System Logs ---")
        cur.execute("SELECT level, logger_name, message, created_at FROM system_logs ORDER BY id DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        
        if not rows:
            print("No logs found.")
            return

        for r in rows:
            icon = "⚠️" if r['level'] == 'WARNING' else "🚨"
            ts = r['created_at']
            print(f"{icon} [{ts}] {r['level']} ({r['logger_name']})")
            print(f"   {r['message']}")
            print("-" * 40)
            
    except Exception as e:
        print(f"Error reading logs: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    n = 10
    if len(sys.argv) > 1:
        try: n = int(sys.argv[1])
        except ValueError: pass
    read_logs(n)
