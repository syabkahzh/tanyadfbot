from telethon import TelegramClient, events
from config import Config
from datetime import datetime, timedelta

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
            processed=processed
        )
        if internal_id:
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
        
        if last_id:
            print(f"🔄 Catching up since Msg ID: {last_id}...")
            # Fetch everything newer than last_id
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
