import asyncio
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from db import Database
from processor import GeminiProcessor
from listener import TelethonListener
from bot import TelegramBot
from config import Config

db = Database()
gemini = GeminiProcessor()
listener = TelethonListener(db)
bot = TelegramBot(db, gemini)
scheduler = AsyncIOScheduler()

async def processing_loop():
    print(f"[{datetime.now()}] 🧠 Starting AI Analysis Batch...")
    batch = db.get_unprocessed_batch(50)
    if not batch: 
        print("☕ No new messages in queue.")
        return

    msgs_to_proc = [{"id": r['id'], "text": r['text'], "timestamp": r['timestamp'], "sender_name": r['sender_name'], "tg_msg_id": r['tg_msg_id'], "chat_id": r['chat_id']} for r in batch]
    print(f"🤖 Sending {len(msgs_to_proc)} messages to {Config.MODEL_ID}...")
    
    promos = await gemini.process_batch(msgs_to_proc)
    db.mark_batch_processed([m["id"] for m in msgs_to_proc])
    
    if not promos:
        print("⚠️ No promos extracted or AI API failed. Batch cleared, moving on.")
        return

    msg_id_map = {m["id"]: m for m in msgs_to_proc}
    promo_ids = [p.original_msg_id for p in promos]
    
    for p in promos:
        m = msg_id_map.get(p.original_msg_id)
        if not m: continue
        
        tg_link = f"https://t.me/c/{str(m['chat_id'])[4:]}/{m['tg_msg_id']}"
        db.save_promo(p.original_msg_id, p, tg_link)
        
        print(f"🔥 PROMO FOUND: {p.brand} - {p.summary[:50]}...")
        await bot.send_alert(p, tg_link)

    for m in msgs_to_proc:
        if m['id'] not in promo_ids:
            print(f"⏭️  Skipped (Not a promo): ID {m['id']} | {m['text'][:50]}...")

    print(f"✅ Batch Complete. Extracted {len(promos)} promos.")

async def hourly_digest_job():
    print(f"[{datetime.now()}] 🕒 Generating Hourly Digest...")
    rows = db.get_promos(hours=1)
    if not rows:
        print("No promos in the last hour for digest.")
        return

    context = "\n".join([f"- {r['brand']}: {r['summary']}" for r in rows])
    digest = await gemini.answer_question(
        "Please summarize the last hour of promotional activity into a bulleted list. Highlight the best deals.",
        context
    )
    
    await bot.send_digest(digest, datetime.now().strftime('%H:00'))
    print("✅ Hourly Digest sent.")

async def trend_monitor_job():
    """Detects 'hot topics' in the group and interprets them with context."""
    print(f"[{datetime.now()}] 🔍 Checking for trends...")
    WINDOW_MIN = 20
    
    hot_words_data = db.get_recent_words(minutes=WINDOW_MIN)
    if not hot_words_data: return
    
    top_words = [w for w, count in hot_words_data[:5] if count >= 3]
    if not top_words: return

    all_raw = db.get_recent_messages(minutes=WINDOW_MIN)
    context_msgs = [m['text'] for m in all_raw if any(w in m['text'].lower() for w in top_words)]
    
    if not context_msgs: return

    topic_summary = await gemini.interpret_keywords(top_words, WINDOW_MIN, context_msgs)

    if topic_summary:
        print(f"📈 Trend Detected: {topic_summary}")
        await bot.send_plain(f"💡 *Topik Ramai di Group*\n\n{topic_summary}")

async def main():
    if not Config.validate():
        return
        
    await bot.app.initialize()
    await bot.app.start()
    await bot.app.updater.start_polling()
    
    await listener.start()
    await listener.sync_history(6)
    
    scheduler.add_job(processing_loop, "interval", minutes=2)
    scheduler.add_job(trend_monitor_job, "interval", minutes=15)
    scheduler.add_job(hourly_digest_job, "cron", minute=0)
    scheduler.start()
    
    await processing_loop()
    
    print("TanyaDFBot Full System Online.")
    await asyncio.Event().wait()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot stopped by user.")
