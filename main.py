import asyncio
from datetime import datetime, timezone, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.executors.asyncio import AsyncIOExecutor
import pytz
from db import Database
from processor import GeminiProcessor
from listener import TelethonListener
from bot import TelegramBot
from config import Config

db = Database()
gemini = GeminiProcessor()
listener = TelethonListener(db)
bot = TelegramBot(db, gemini)

WIB = pytz.timezone(Config.TIMEZONE)
scheduler = AsyncIOScheduler(timezone=WIB, executors={"default": AsyncIOExecutor()})

BOOT_TIME = datetime.now(timezone.utc)
_recent_alerts_history: list[dict] = []

def _make_tg_link(chat_id, msg_id):
    cid = str(chat_id)
    if cid.startswith("-100"): cid = cid[4:]
    return f"https://t.me/c/{cid}/{msg_id}"

def _parse_ts(ts) -> datetime:
    """Always returns a UTC-aware datetime from whatever timestamp format we get."""
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    s = str(ts).replace('Z', '+00:00')
    try:
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)

async def processing_loop():
    global _recent_alerts_history

    if not _recent_alerts_history:
        recent_promos = await db.get_promos(hours=3, limit=15)
        for r in recent_promos:
            _recent_alerts_history.append({"brand": r['brand'], "summary": r['summary']})

    print(f"[{datetime.now(WIB).strftime('%H:%M:%S WIB')}] 🧠 Analysis Batch Started...")
    batch = await db.get_unprocessed_batch(100)
    if not batch: return

    msgs_to_proc = [
        {"id": r['id'], "text": r['text'], "timestamp": r['timestamp'],
         "tg_msg_id": r['tg_msg_id'], "chat_id": r['chat_id']}
        for r in batch
    ]
    promos = await gemini.process_batch(msgs_to_proc)
    if promos is None: return

    msg_id_map = {m["id"]: m for m in msgs_to_proc}
    unique_promos = await gemini.filter_duplicates(promos, _recent_alerts_history)

    promos_to_save = []
    now_utc = datetime.now(timezone.utc)

    for p in unique_promos:
        m = msg_id_map.get(p.original_msg_id)
        if not m: continue

        tg_link = _make_tg_link(m['chat_id'], m['tg_msg_id'])
        promos_to_save.append((m['id'], p, tg_link))

        msg_time = _parse_ts(m['timestamp'])
        age_seconds = (now_utc - msg_time).total_seconds()

        # FIX: Only alert fresh promos (≤60 min). Boot grace only for truly recent
        # messages (≤90 min) to handle normal processing delay, not 12h backlog.
        is_fresh = age_seconds < 3600
        is_boot_catchup = age_seconds < 5400 and (now_utc - BOOT_TIME).total_seconds() < 300

        if p.status != 'expired' and (is_fresh or is_boot_catchup):
            await bot.send_alert(p, tg_link, timestamp=m['timestamp'])
            _recent_alerts_history.append({"brand": p.brand, "summary": p.summary})
            if len(_recent_alerts_history) > 30:
                _recent_alerts_history.pop(0)

    success = await db.save_promos_batch(promos_to_save, [m["id"] for m in msgs_to_proc])
    if success:
        print(f"✅ Batch: {len(msgs_to_proc)} msgs → {len(unique_promos)} new promos saved.")

async def hourly_digest_job():
    rows = await db.get_promos(hours=1)
    if not rows:
        print("ℹ️ Hourly digest: no promos this hour, skipping.")
        return
    context = "\n".join([f"- {r['brand']}: {r['summary']}" for r in rows])
    now_wib = datetime.now(WIB)
    digest = await gemini.answer_question(
        f"Summarize these deals concisely in Indonesian. Hour: {now_wib.strftime('%H:00 WIB')}",
        context
    )
    await bot.send_digest(digest, now_wib.strftime('%H:00 WIB'))

async def main():
    if not Config.validate(): return
    await db.init()
    await bot.app.initialize()
    await bot.app.start()
    await bot.app.updater.start_polling()

    await listener.start()
    # Sync only 2h on boot — aggressive 12h was the root of stale alert flooding
    asyncio.create_task(listener.sync_history(2))

    scheduler.add_job(processing_loop, "interval", minutes=1, id="process")
    # FIX: cron now uses WIB timezone (scheduler is WIB-aware)
    scheduler.add_job(hourly_digest_job, "cron", minute=0, id="digest")
    scheduler.start()

    await processing_loop()
    now_wib = datetime.now(WIB).strftime('%H:%M WIB')
    print(f"✅ TanyaDFBot Online — {now_wib}")
    try:
        await asyncio.Event().wait()
    finally:
        await db.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
