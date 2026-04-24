import asyncio
import sys
import os
from datetime import datetime, timezone, timedelta

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from db import Database
from listener import TelethonListener

async def main():
    if not Config.validate():
        print("❌ Config invalid.")
        return

    db = Database()
    await db.init()
    
    listener = TelethonListener(db)
    
    print("🚀 Starting Telethon client for manual scrape...")
    await listener.client.start()
    
    chat = await listener.client.get_entity(Config.TARGET_GROUP)
    print(f"✅ Connected to: {chat.title} ({chat.id})")
    
    # Scrape last 1 hour
    hours = 1
    print(f"📥 Scraping history for the last {hours} hour(s)...")
    
    # We use a custom fetch here to ignore existing DB state and force a 1-hour download
    from db import _ts_str
    limit_date = datetime.now(timezone.utc) - timedelta(hours=hours)
    
    count = 0
    buffer = []
    async for message in listener.client.iter_messages(chat.id, offset_date=limit_date, reverse=True):
        if not message.text:
            continue
        
        # Check for time mentions roughly
        from listener import TIME_PATTERN
        has_time = bool(TIME_PATTERN.search(message.text))
        
        buffer.append((
            message.id, chat.id, message.sender_id,
            f"User_{message.sender_id}", _ts_str(message.date),
            message.text, message.reply_to_msg_id,
            0, 1 if message.photo else 0, 1 if has_time else 0
        ))
        
        count += 1
        if len(buffer) >= 100:
            await listener._bulk_save_to_db(buffer)
            buffer = []
            print(f"   Processed {count} messages...")
            
    if buffer:
        await listener._bulk_save_to_db(buffer)
        
    print(f"🏁 Done! Scraped {count} messages from the last hour.")
    await listener.client.disconnect()
    if db.conn:
        await db.conn.close()

if __name__ == "__main__":
    asyncio.run(main())
