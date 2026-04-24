import sqlite3
import json
from datetime import datetime, timezone, timedelta

def export_last_hour():
    db_path = "tanya_main_vps.db"
    output_path = "scraped_history.json"
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Calculate cutoff for last hour
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=1.1)).strftime('%Y-%m-%d %H:%M:%S+00:00')
    
    print(f"🔍 Exporting messages since {cutoff}...")
    
    cursor.execute("""
        SELECT tg_msg_id, sender_name, timestamp, text 
        FROM messages 
        WHERE timestamp >= ? 
        ORDER BY timestamp ASC
    """, (cutoff,))
    
    rows = cursor.fetchall()
    messages = [dict(row) for row in rows]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)
        
    print(f"✅ Exported {len(messages)} messages to {output_path}")
    conn.close()

if __name__ == "__main__":
    export_last_hour()
