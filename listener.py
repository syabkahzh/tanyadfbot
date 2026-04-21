from telethon import TelegramClient, events
from config import Config
from datetime import datetime, timedelta
import re

TIME_PATTERN = re.compile(r'\bjam\s+\d|\bpukul\s+\d|\d{1,2}[:.]\d{2}\s*(wib|wita|wit)?|\bmalem\b|\bsore\b|\bsubuh\b', re.IGNORECASE)

class TelethonListener:
    def __init__(self, db_manager):
        self.db = db_manager
        self.client = TelegramClient(Config.SESSION_NAME, Config.API_ID, Config.API_HASH)

    async def _save_msg_helper(self, event, processed=0):
        sender = await event.get_sender()
        sender_name = getattr(sender, 'first_name', 'Unknown')
        if getattr(sender, 'last_name', None): sender_name += f" {sender.last_name}"
        elif getattr(sender, 'username', None): sender_name = f"@{sender.username}"

        # Native async save
        internal_id = await self.db.save_message(
            tg_msg_id=event.id,
            chat_id=event.chat_id,
            sender_id=event.sender_id,
            sender_name=sender_name,
            timestamp=event.date,
            text=event.text,
            reply_to_msg_id=event.reply_to_msg_id,
            processed=processed,
            has_photo=1 if event.photo else 0
        )
        if internal_id:
            if TIME_PATTERN.search(event.text):
                await self.db.conn.execute(
                    "UPDATE messages SET has_time_mention=1 WHERE id=?", (internal_id,)
                )
                await self.db.conn.commit()

            tag = "[Synced]" if processed else "[Scraped]"
            preview = (event.text[:97] + '...') if len(event.text) > 100 else event.text
            print(f"📥 {tag} Msg ID: {internal_id} | From: {sender_name} | Text: {preview}")
        return internal_id

    async def start(self):
        @self.client.on(events.NewMessage(chats=Config.TARGET_GROUP))
        async def handler(event):
            if event.text:
                await self._save_msg_helper(event, processed=0)
        
        await self.client.start()
        print("Telethon Listener started.")

    async def sync_history(self, hours=6):
        # 1. Resolve chat to get numeric ID
        chat = await self.client.get_entity(Config.TARGET_GROUP)
        chat_id = chat.id
        
        # 2. Find where we left off
        last_id = await self.db.get_last_msg_id(chat_id)
        if not last_id:
            # Fallback: check if we have ANY messages in DB (this bot only does one group)
            async with self.db.conn.execute("SELECT MAX(tg_msg_id) FROM messages") as cur:
                row = await cur.fetchone()
                if row and row[0]:
                    last_id = row[0]
                    print(f"ℹ️ Found global last_id: {last_id} (chat_id mismatch fallback)")
        
        if last_id:
            # Calculate how long we were down
            last_ts = await self.db.get_last_msg_timestamp(chat_id)
            if last_ts:
                from datetime import timezone
                gap_hours = (datetime.now(timezone.utc) - last_ts).total_seconds() / 3600
            else:
                gap_hours = 999
            
            if gap_hours > 1.5:
                # We were down long enough that catching up would flood stale alerts
                # Fetch only last 90 minutes — enough to catch anything still relevant
                print(f"⚠️ Gap of {gap_hours:.1f}h detected. Fetching last 90min only.")
                from datetime import timezone
                limit_date = datetime.now(timezone.utc) - timedelta(hours=1.5)
                async for message in self.client.iter_messages(chat_id, offset_date=limit_date, reverse=True):
                    if message.text:
                        await self._save_msg_helper(message, processed=0)
            else:
                # Short gap — safe to catch up from where we left off
                print(f"🔄 Short gap ({gap_hours:.1f}h). Catching up since Msg ID: {last_id}...")
                async for message in self.client.iter_messages(chat_id, min_id=last_id, reverse=True):
                    if message.text:
                        await self._save_msg_helper(message, processed=0)
        else:
            print(f"⏱️ DB Empty. Syncing last {hours} hours...")
            from datetime import timezone
            limit_date = datetime.now(timezone.utc) - timedelta(hours=hours)
            async for message in self.client.iter_messages(chat_id, offset_date=limit_date, reverse=True):
                if message.text:
                    await self._save_msg_helper(message, processed=0)
                    
        print("✅ History sync/catch-up complete.")
