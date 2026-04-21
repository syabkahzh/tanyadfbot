import asyncio
import html
import json
from datetime import datetime, timezone, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import pytz
from db import Database
from processor import GeminiProcessor, PromoExtraction
from listener import TelethonListener
from bot import TelegramBot
from config import Config

db = Database()
gemini = GeminiProcessor()
listener = TelethonListener(db)
bot = TelegramBot(db, gemini)

WIB = pytz.timezone("Asia/Jakarta")
scheduler = AsyncIOScheduler(timezone=WIB)

BOOT_TIME = datetime.now(timezone.utc)
BOOT_CATCHUP_WINDOW = 3600  # Default 1h catch-up
_recent_alerts_history: list[dict] = []
_buffer_flush_task = None
_alerted_hot_threads: dict[int, int] = {}  # tg_msg_id -> last alerted count
_last_trend_alert = ""  # debounce narratives
_last_halfhour_digest_count = 0  # track new promos in 30m window
_last_spike_alert = datetime.min.replace(tzinfo=timezone.utc) # cooldown for spikes

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

def _esc(text: str) -> str:
    """Escapes common markdown characters."""
    if not text: return ""
    return text.replace("*", "\\*").replace("_", "\\_").replace("[", "\\[").replace("]", "\\]").replace("`", "\\`")

async def _flush_alert_buffer():
    await asyncio.sleep(20)  # collect for 20s (reduced from 90s for speed)
    global _buffer_flush_task
    
    # Get but don't clear yet to prevent loss on crash
    async with db.conn.execute("SELECT brand, p_data_json, tg_link, timestamp FROM pending_alerts") as cur:
        rows = await cur.fetchall()
    
    if not rows:
        _buffer_flush_task = None
        return

    # Group by brand
    snapshot = {}
    for r in rows:
        brand = r['brand']
        p_data = PromoExtraction.model_validate_json(r['p_data_json'])
        if brand not in snapshot:
            snapshot[brand] = []
        snapshot[brand].append((p_data, r['tg_link'], r['timestamp']))
    
    _buffer_flush_task = None
    
    try:
        # Separate per brand (topic) to ensure unique deals aren't buried
        for brand_key, items in snapshot.items():
            if len(items) == 1:
                p, link, ts = items[0]
                await bot.send_alert(p, link, timestamp=ts)
            else:
                # Grouped alert: multiple people talking about the SAME topic
                await bot.send_grouped_alert(brand_key, items)
        
        # Clear from DB only after successful sending
        await db.conn.execute("DELETE FROM pending_alerts")
        await db.conn.commit()
    except Exception as e:
        print(f"⚠️ Alert flush failed: {e}")

async def image_processing_job():
    """Background job to process high-value images (posters/screenshots)."""
    try:
        # Find messages with photos that haven't been image-processed
        async with db.conn.execute("""
            SELECT id, tg_msg_id, chat_id, text, timestamp 
            FROM messages 
            WHERE has_photo=1 AND image_processed=0
            ORDER BY id DESC LIMIT 5
        """) as cur:
            rows = await cur.fetchall()
            
        for r in rows:
            msg_id, tg_msg_id, chat_id, caption, ts = r
            
            # Check popularity (replies)
            async with db.conn.execute("SELECT COUNT(*) FROM messages WHERE reply_to_msg_id=? AND chat_id=?", (tg_msg_id, chat_id)) as ccur:
                replies = (await ccur.fetchone())[0]
                is_popular = replies >= 3
            
            is_filtered = gemini._is_worth_checking(caption or "")
            
            if not (is_popular or is_filtered):
                await db.conn.execute("UPDATE messages SET image_processed=1 WHERE id=?", (msg_id,))
                continue
                
            print(f"🖼️ Processing high-value image: Msg {tg_msg_id} ({replies} replies)")
            
            try:
                tg_msg = await listener.client.get_messages(chat_id, ids=tg_msg_id)
                if not tg_msg or not tg_msg.photo:
                    await db.conn.execute("UPDATE messages SET image_processed=1 WHERE id=?", (msg_id,))
                    continue
                    
                photo_bytes = await listener.client.download_media(tg_msg.photo, file=bytes)
                if photo_bytes:
                    p = await gemini.process_image(photo_bytes, caption or "", msg_id)
                    if p:
                        tg_link = _make_tg_link(chat_id, tg_msg_id)
                        await db.save_promos_batch([(msg_id, p, tg_link)], [])
                        
                        # Alert if fresh
                        now_utc = datetime.now(timezone.utc)
                        if (now_utc - _parse_ts(ts)).total_seconds() < 5400:
                             await db.save_pending_alert(p.brand.lower().strip(), p.model_dump_json(), tg_link, ts)
                             global _buffer_flush_task
                             if _buffer_flush_task is None or _buffer_flush_task.done():
                                 _buffer_flush_task = asyncio.create_task(_flush_alert_buffer())

                await db.conn.execute("UPDATE messages SET image_processed=1 WHERE id=?", (msg_id,))
                await db.conn.commit()
            except Exception as e:
                print(f"❌ image_processing_job item error (Msg {tg_msg_id}): {e}")
                
    except Exception as e:
        print(f"❌ image_processing_job critical error: {e}")

async def processing_loop():
    global _recent_alerts_history

    if not _recent_alerts_history:
        recent_promos = await db.get_recent_alert_brands(hours=2, limit=100)
        for r in recent_promos:
            _recent_alerts_history.append({"brand": r['brand'], "summary": r['summary']})

    print(f"🚀 Processing Loop Started (Permanent Mode)")
    while True:
        try:
            batch = await db.get_unprocessed_batch(200) 
            if not batch:
                await asyncio.sleep(2) # Tiny breather if no data
                continue
            
            # ... (rest of the logic remains same, just ensure it doesn't 'break' the while loop)

        msgs_to_proc = [
            {"id": r['id'], "text": r['text'], "timestamp": r['timestamp'],
             "tg_msg_id": r['tg_msg_id'], "chat_id": r['chat_id'], "reply_to_msg_id": r['reply_to_msg_id']}
            for r in batch
        ]
        
        # 1. Filter first (save AI cost/tokens)
        filtered_msgs = [m for m in msgs_to_proc if gemini._is_worth_checking(m['text'])]
        
        # 2. Enrich only filtered messages with context via bulk fetch
        if filtered_msgs:
            chat_id = filtered_msgs[0]['chat_id']
            min_id = min(m['id'] for m in filtered_msgs) - 5
            max_id = max(m['id'] for m in filtered_msgs)
            
            # Bulk fetch context messages
            ctx_rows = await db.get_context_bulk(chat_id, min_id, max_id)
            ctx_map = {r['id']: r['text'] for r in ctx_rows}
            
            # Bulk fetch reply sources
            reply_ids = [m['reply_to_msg_id'] for m in filtered_msgs if m.get('reply_to_msg_id')]
            reply_map = await db.get_reply_sources_bulk(reply_ids, chat_id) if reply_ids else {}
            
            for m in filtered_msgs:
                # Local reconstruction of context
                preceding = [ctx_map[i] for i in range(m['id']-5, m['id']) if i in ctx_map]
                m['context'] = " → ".join(preceding)
                
                if m.get('reply_to_msg_id') and m['reply_to_msg_id'] in reply_map:
                    m['context'] = f"[REPLYING TO: {reply_map[m['reply_to_msg_id']]}] " + m.get('context', '')
            
        # 3. Process only the filtered and enriched batch
        promos = await gemini.process_batch(filtered_msgs)
        if promos is None:
            # Mark failed batch as processed to unstick the queue, log for review
            failed_ids = [m["id"] for m in msgs_to_proc]
            await db.mark_batch_processed(failed_ids)
            print(f"⚠️ Gemini failed — {len(failed_ids)} msgs skipped and marked processed.")
            await asyncio.sleep(5)
            continue  # NOT break — keep the loop going

        msg_id_map = {m["id"]: m for m in filtered_msgs}
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
            
            is_fresh = age_seconds < 3600
            # Cap catch-up alerts at 1.5 hours (5400s) to avoid stale noise
            is_boot_catchup = age_seconds < 5400 and (now_utc - BOOT_TIME).total_seconds() < 900

            if p.status != 'expired' and (is_fresh or is_boot_catchup):
                brand_key = p.brand.lower().strip()
                # Save to persistent DB buffer instead of memory
                await db.save_pending_alert(
                    brand=brand_key,
                    p_data_json=p.model_dump_json(),
                    tg_link=tg_link,
                    timestamp=m['timestamp']
                )

                # Schedule flush if not already pending
                global _buffer_flush_task
                if _buffer_flush_task is None or _buffer_flush_task.done():
                    _buffer_flush_task = asyncio.create_task(_flush_alert_buffer())

                _recent_alerts_history.append({"brand": p.brand, "summary": p.summary})
                if len(_recent_alerts_history) > 100:
                    _recent_alerts_history.pop(0)

        success = await db.save_promos_batch(promos_to_save, [m["id"] for m in msgs_to_proc])
        if success:
            print(f"✅ Batch: {len(msgs_to_proc)} msgs → {len(unique_promos)} new promos saved.")
        
        # Don't hammer the AI too fast even in catch-up mode
        await asyncio.sleep(1)
        
    except Exception as e:
        print(f"❌ processing_loop batch error: {e}")
        await asyncio.sleep(5)

async def hourly_digest_job():
    try:
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
    except Exception as e:
        print(f"❌ hourly_digest_job error: {e}")

async def midnight_digest_job():
    """Covers 2am–5am WIB sleep window, sent at 5am."""
    try:
        rows = await db.get_promos(hours=3)  # last 3 hrs = 2-5am
        if not rows:
            print("ℹ️ Midnight digest: no promos, skipping.")
            return
        context = "\n".join([f"- {r['brand']}: {r['summary']}" for r in rows])
        digest = await gemini.answer_question(
            "Rangkum promo yang masuk jam 02:00–05:00 WIB tadi malam. Singkat, padat, santai.",
            context
        )
        await bot.send_digest(digest, "02:00–05:00 WIB (Rangkuman Tengah Malam)")
    except Exception as e:
        print(f"❌ midnight_digest_job error: {e}")

async def halfhour_digest_job():
    global _last_halfhour_digest_count
    try:
        now_wib = datetime.now(WIB)
        hour = now_wib.hour
        
        # Skip 02:00–05:00 WIB — midnight_digest_job covers this window
        if 2 <= hour < 5:
            return
        
        rows = await db.get_promos(hours=0.5)  # last 30 min
        if not rows or len(rows) <= _last_halfhour_digest_count:
            if not rows: _last_halfhour_digest_count = 0
            return
        
        _last_halfhour_digest_count = len(rows)
        context = "\n".join([f"- {r['brand']}: {r['summary']}" for r in rows])
        label = now_wib.strftime('%H:%M WIB')
        digest = await gemini.answer_question(
            f"Ringkas promo 30 menit terakhir. Waktu: {label}. Singkat dan padat.",
            context
        )
        await bot.send_digest(digest, f"{label} (30 menit)")
    except Exception as e:
        print(f"❌ halfhour_digest_job error: {e}")

async def heartbeat_job():
    try:
        now_wib = datetime.now(WIB).strftime('%H:%M WIB')
        async with db.conn.execute("SELECT COUNT(*) FROM messages WHERE processed=0") as cur:
            queue = (await cur.fetchone())[0]
        async with db.conn.execute("SELECT MAX(timestamp) FROM messages") as cur:
            last_ts = (await cur.fetchone())[0]
        
        # Lag detection — if newest message is >15min old, group might be dead or listener dropped
        lag_warn = ""
        if last_ts:
            last_dt = _parse_ts(last_ts)
            lag_min = (datetime.now(timezone.utc) - last_dt).total_seconds() / 60
            if lag_min > 15:
                lag_warn = f"\n⚠️ *Listener lag: {int(lag_min)}m — cek koneksi!*"
        
        text = f"💓 `{now_wib}` | queue: `{queue}`{lag_warn}"
        await bot.send_plain(text)
    except Exception as e:
        print(f"❌ heartbeat_job error: {e}")

async def hot_thread_job():
    try:
        threads = await db.get_hot_threads(minutes=30, min_replies=3)
        
        # Prune old threads from memory (keep last 100)
        if len(_alerted_hot_threads) > 100:
            # Remove the oldest half
            keys = list(_alerted_hot_threads.keys())
            for k in keys[:50]:
                del _alerted_hot_threads[k]

        for t in threads:
            last_count = _alerted_hot_threads.get(t['tg_msg_id'], 0)
            
            # Alert if new or if it gained significant traction (+5 replies) since last alert
            if t['reply_count'] >= last_count + 5:
                _alerted_hot_threads[t['tg_msg_id']] = t['reply_count']
                
                link = _make_tg_link(t['chat_id'], t['tg_msg_id'])
                msg_wib = _parse_ts(t['timestamp']).astimezone(WIB).strftime('%H:%M')

                # Fetch replies for summarization
                reply_rows = await db.get_thread_replies(t['tg_msg_id'], t['chat_id'])
                replies = [r['text'] for r in reply_rows]
                summary = await gemini.summarize_thread(t['text'], replies)

                text = (
                    f"🧵 <b>Thread Ramai</b> ({t['reply_count']} balasan)\n\n"
                    f"💬 <i>{html.escape(t['text'][:300])}</i>\n"
                    f"— {html.escape(t['sender_name'])} (<code>{msg_wib} WIB</code>)\n\n"
                    f"🤖 <b>Inti:</b> {html.escape(summary)}\n\n"
                    f"🔗 <a href='{link}'>Lihat pesan asli</a>"
                )
                from telegram.constants import ParseMode
                await bot.send_plain(text, parse_mode=ParseMode.HTML)

    except Exception as e:
        print(f"❌ hot_thread_job error: {e}")

async def time_mention_job():
    try:
        async with db.conn.execute("""
            SELECT id, tg_msg_id, chat_id, text, sender_name, timestamp
            FROM messages
            WHERE has_time_mention=1 AND time_alerted=0
            ORDER BY id DESC LIMIT 10
        """) as cur:
            rows = await cur.fetchall()
        
        for r in rows:
            link = _make_tg_link(r['chat_id'], r['tg_msg_id'])
            msg_wib = _parse_ts(r['timestamp']).astimezone(WIB).strftime('%H:%M')
            
            # Use HTML for better robustness with arbitrary user names/text
            text = (
                f"🕐 <b>Mention Waktu</b>\n"
                f"<code>{msg_wib} WIB</code> · {html.escape(r['sender_name'])}\n\n"
                f"{html.escape(r['text'][:300])}\n\n<a href='{link}'>→ Lihat Pesan</a>"
            )
            from telegram.constants import ParseMode
            await bot.send_plain(text, parse_mode=ParseMode.HTML)
            await db.conn.execute("UPDATE messages SET time_alerted=1 WHERE id=?", (r['id'],))
        await db.conn.commit()
    except Exception as e:
        print(f"❌ time_mention_job error: {e}")

async def trend_job():
    try:
        global _last_trend_alert
        minutes = 20
        words_freq = await db.get_recent_words(minutes=minutes)
        
        # Filter for words appearing 4+ times
        hot_words = [w for w, c in words_freq if c >= 4][:10]
        if not hot_words:
            return

        recent_msgs = await db.get_recent_messages(minutes=minutes)
        context_msgs = [m['text'] for m in recent_msgs]

        narrative = await gemini.interpret_keywords(hot_words, minutes, context_msgs)
        
        if narrative and narrative != "NO_TREND":
            if narrative == _last_trend_alert:
                return
            _last_trend_alert = narrative
            text = (
                f"📈 <b>Trend Obrolan ({minutes} mnt terakhir)</b>\n"
                f"🔥 <code>{', '.join(hot_words)}</code>\n\n"
                f"🤖 {html.escape(narrative)}"
            )
            from telegram.constants import ParseMode
            await bot.send_plain(text, parse_mode=ParseMode.HTML)
    except Exception as e:
        print(f"❌ trend_job error: {e}")

async def spike_detection_job():
    """Detects sudden volume spikes (>30 msgs/min) and triggers summary."""
    global _last_spike_alert
    try:
        # Check cooldown (10 minutes)
        if (datetime.now(timezone.utc) - _last_spike_alert).total_seconds() < 600:
            return

        async with db.conn.execute("""
            SELECT COUNT(*) FROM messages 
            WHERE timestamp >= strftime('%Y-%m-%d %H:%M:%S+00:00', 'now', '-1 minute')
        """) as cur:
            count = (await cur.fetchone())[0]

        if count >= 30:
            print(f"🚀 SPIKE DETECTED: {count} msgs/min. Summarizing...")
            recent_msgs = await db.get_recent_messages(minutes=3)
            texts = [f"{m['sender_name']}: {m['text']}" for m in reversed(recent_msgs)]
            
            summary = await gemini.summarize_raw(texts)
            
            text = (
                f"🚀 <b>Lonjakan Pesan Terdeteksi!</b>\n"
                f"📊 Volume: <code>{count} pesan/menit</code>\n\n"
                f"🤖 <b>Rangkuman Cepat:</b>\n{html.escape(summary)}"
            )
            from telegram.constants import ParseMode
            await bot.send_plain(text, parse_mode=ParseMode.HTML)
            _last_spike_alert = datetime.now(timezone.utc)

    except Exception as e:
        print(f"❌ spike_detection_job error: {e}")

async def db_maintenance_job():
    try:
        await db.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        # Rolling DB: delete processed messages older than 1 day
        await db.conn.execute("DELETE FROM messages WHERE processed=1 AND timestamp < strftime('%Y-%m-%d %H:%M:%S+00:00','now','-1 day')")
        await db.conn.commit()
        print("🔧 DB maintenance: WAL checkpointed, old messages (1d+) pruned.")
    except Exception as e:
        print(f"❌ db_maintenance_job error: {e}")

async def main():
    if not Config.validate(): return
    await db.init()
    
    # Dynamic catch-up: check how long we were offline to alert missed deals
    global BOOT_CATCHUP_WINDOW
    async with db.conn.execute("SELECT MAX(timestamp) FROM messages") as cur:
        row = await cur.fetchone()
    if row and row[0]:
        gap = (datetime.now(timezone.utc) - _parse_ts(row[0])).total_seconds()
        BOOT_CATCHUP_WINDOW = max(3600, min(gap + 1800, 8 * 3600)) 
        print(f"🔄 Boot Catch-up Window: {BOOT_CATCHUP_WINDOW/3600:.1f} hours")
    
    # Check for leftover alerts from previous crash
    global _buffer_flush_task
    _buffer_flush_task = asyncio.create_task(_flush_alert_buffer())
    
    await bot.app.initialize()
    await bot.app.start()
    await bot.app.updater.start_polling()

    # Start the permanent processing loop
    asyncio.create_task(processing_loop())

    await listener.start()
    # Sync only 2h on boot
    asyncio.create_task(listener.sync_history(2))

    # Image processing job every 3 minutes
    scheduler.add_job(image_processing_job, "interval", minutes=3, id="images", misfire_grace_time=120)
    # Hourly: skip 2am, 3am, 4am WIB to avoid sleep alerts
    scheduler.add_job(hourly_digest_job, "cron", minute=0, hour="0,1,5-23", id="digest", misfire_grace_time=120)
    # Midnight catch-up for 2am-5am window
    scheduler.add_job(midnight_digest_job, "cron", hour=5, minute=0, id="midnight_digest", misfire_grace_time=300)
    # 30-min digest at :15 and :45 to complement hourly at :00
    scheduler.add_job(halfhour_digest_job, "cron", minute="15,45", id="halfhour_digest", misfire_grace_time=60)
    scheduler.add_job(heartbeat_job, "cron", minute=30, id="heartbeat", misfire_grace_time=60)
    scheduler.add_job(hot_thread_job, "interval", minutes=5, id="hot_threads", misfire_grace_time=30)
    scheduler.add_job(time_mention_job, "interval", minutes=2, id="time_mentions", misfire_grace_time=20)
    scheduler.add_job(trend_job, "interval", minutes=10, id="trend_job", misfire_grace_time=60)
    scheduler.add_job(spike_detection_job, "interval", minutes=1, id="spike_check", misfire_grace_time=20)
    scheduler.add_job(db_maintenance_job, "cron", hour="*/4", minute=0, id="db_maint", misfire_grace_time=300)
    scheduler.start()

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
