from telethon import TelegramClient, events
from config import Config
from datetime import datetime, timedelta

class TelethonListener:
    def __init__(self, db_manager):
        self.db = db_manager
        self.client = TelegramClient(Config.SESSION_NAME, Config.API_ID, Config.API_HASH)

    async def _save_msg_helper(self, event):
        sender = await event.get_sender()
        sender_name = getattr(sender, 'first_name', 'Unknown')
        if getattr(sender, 'last_name', None): sender_name += f" {sender.last_name}"
        elif getattr(sender, 'username', None): sender_name = f"@{sender.username}"

        # FIX: Offload the synchronous SQLite write to a separate thread
        import asyncio
        internal_id = await asyncio.to_thread(
            self.db.save_message,
            tg_msg_id=event.id,
            chat_id=event.chat_id,
            sender_id=event.sender_id,
            sender_name=sender_name,
            timestamp=event.date,
            text=event.text,
            reply_to_msg_id=event.reply_to_msg_id
        )
        if internal_id:
            preview = (event.text[:97] + '...') if len(event.text) > 100 else event.text
            print(f"📥 [Scraped] Msg ID: {internal_id} | From: {sender_name} | Text: {preview}")
        return internal_id

    async def start(self):
        @self.client.on(events.NewMessage(chats=Config.TARGET_GROUP))
        async def handler(event):
            if event.text:
                await self._save_msg_helper(event)
        
        await self.client.start()
        print("Telethon Listener started.")

    async def sync_history(self, hours=6):
        print(f"Syncing last {hours} hours of history...")
        limit_date = datetime.now() - timedelta(hours=hours)
        async for message in self.client.iter_messages(Config.TARGET_GROUP, offset_date=limit_date, reverse=True):
            if message.text:
                await self._save_msg_helper(message)
        print("History sync complete.")
