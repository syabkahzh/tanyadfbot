import asyncio
import os
import psutil
import shared
from db import Database
from processor import GeminiProcessor
from config import Config
from unittest.mock import MagicMock

async def check_status():
    # Initialize
    db = Database()
    await db.init()
    gp = GeminiProcessor()
    
    # Check basic counts
    msg_count = await db.get_total_messages()
    unprocessed = await db.get_queue_size()
    promo_count = await db.get_total_promos()
    
    print(f"--- 📊 SYSTEM STATUS ---")
    print(f"Total Messages: {msg_count}")
    print(f"Queue Size:     {unprocessed}")
    print(f"Total Promos:   {promo_count}")
    
    # Check AI Slots
    print(f"\nAI Fleet Status:")
    for name, slot in gp._slots.items():
        print(f"  • {name}: {slot.current_usage()}/{slot.limit} RPM")
        
    # Check Failures
    failures = await db.get_recent_failures(limit=5)
    print(f"\nRecent Failures ({len(failures)}):")
    for f in failures:
        print(f"  • [{f['created_at']}] {f['component']}: {f['error_msg']}")

if __name__ == "__main__":
    asyncio.run(check_status())
