import asyncio
import html
import json
import uuid
import re
from collections import deque
from datetime import datetime, timezone, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import pytz
from db import Database, normalize_brand
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
_alerted_aman_parents: set[int] = set()  # tg_msg_id of parents already aman-alerted
_alerted_aman_parents_deque: deque[int] = deque(maxlen=500)
_queue_emergency_mode = False  # True when queue > 500, activates auto-triage
_listener_reconnecting = False  # prevent overlapping reconnect attempts

_BRAND_KEYWORDS = {
    'kopken': 'Kopi Kenangan', 'kopi kenangan': 'Kopi Kenangan',
    'hokben': 'HokBen', 'hoka ben': 'HokBen',
    'hophop': 'HopHop', 'hop hop': 'HopHop',
    'sfood': 'ShopeeFood', 'shopeefood': 'ShopeeFood',
    'gfood': 'GoFood', 'gofood': 'GoFood',
    'spx': 'SPX', 'shopee xpress': 'SPX',
    'alfamart': 'Alfamart', 'indomaret': 'Indomaret',
    'chatime': 'Chatime', 'starbucks': 'Starbucks',
    'ismaya': 'Ismaya', 'gindaco': 'Gindaco',
    'cgv': 'CGV', 'xxi': 'XXI',
    'mcd': 'McD', 'kfc': 'KFC',
    'gopay': 'GoPay', 'spay': 'ShopeePay', 'ovo': 'OVO',
    'grab': 'Grab', 'gojek': 'Gojek',
    'pln': 'PLN', 'pulsa': 'Pulsa',
    'mm': 'Mall Monday', 'mall monday': 'Mall Monday',
    'cetem': 'Cetem', 'pubg': 'PUBG',
    'pchematapril': 'PC HematApril',
}

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

def _guess_brand(text: str) -> str:
    if not text: return 'Unknown'
    t = text.lower()
    for kw, brand in _BRAND_KEYWORDS.items():
        if kw in t:
            return brand
    return 'Unknown'

async def _flush_alert_buffer():
    await asyncio.sleep(10)  # Reduced from 45s for faster path delivery
    global _buffer_flush_task
    
    # Atomic claim of rows using flush_id to prevent concurrent task overlap
    flush_id = str(uuid.uuid4())
    await db.conn.execute("UPDATE pending_alerts SET flush_id=? WHERE flush_id IS NULL", (flush_id,))
    await db.conn.commit()

    async with db.conn.execute(
        "SELECT brand, p_data_json, tg_link, timestamp FROM pending_alerts WHERE flush_id=?", (flush_id,)
    ) as cur:
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
        await db.conn.execute("DELETE FROM pending_alerts WHERE flush_id=?", (flush_id,))
        await db.conn.commit()
    except Exception as e:
        print(f"⚠️ Alert flush failed: {e}")
        # RESET flush_id so these alerts get retried on next flush
        await db.conn.execute(
            "UPDATE pending_alerts SET flush_id=NULL WHERE flush_id=?", (flush_id,)
        )
        await db.conn.commit()

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

async def _fast_path_check(msgs):
    global _alerted_aman_parents
    global _buffer_flush_task

    # During queue emergency, skip fast-path to reduce noise flood
    if _queue_emergency_mode:
        return

    # ── FAST PATH ────────────────────────────────────────────────────────────    INSTANT_SIGNALS = {'on', 'jp', 'work', 'restock', 'ristok', 'luber', 'pecah', 'mantap'}
    AMAN_SIGNALS = {'aman'}  # separate — deduplicated per parent

    for m in msgs:
        text = (m['text'] or '').strip()
        text_lower = text.lower()
        tg_link = _make_tg_link(m['chat_id'], m['tg_msg_id'])

        is_instant = (text_lower in INSTANT_SIGNALS and '?' not in text)
        is_allcaps = (len(text) > 4
                    and text == text.upper()
                    and bool(re.search(r'[A-Z]', text))
                    and '?' not in text)
        is_aman = text_lower in AMAN_SIGNALS and '?' not in text

        if not (is_instant or is_allcaps or is_aman):
            continue

        # For "aman" — one alert per unique parent only
        if is_aman and not (is_instant or is_allcaps):
            parent_id = m.get('reply_to_msg_id')
            # No parent = standalone "aman" with no context, skip (too ambiguous)
            if not parent_id:
                continue
            # Already alerted for this parent
            if parent_id in _alerted_aman_parents:
                continue
            
            _alerted_aman_parents.add(parent_id)
            _alerted_aman_parents_deque.append(parent_id)
            if len(_alerted_aman_parents) > 500:
                # FIFO eviction using deque
                _alerted_aman_parents = set(_alerted_aman_parents_deque)

        # Resolve parent for context
        parent_text = None
        if m.get('reply_to_msg_id'):
            async with db.conn.execute(
                "SELECT text FROM messages WHERE tg_msg_id=? AND chat_id=?",
                (m['reply_to_msg_id'], m['chat_id'])
            ) as cur:
                row = await cur.fetchone()
                if row: parent_text = row['text']

        brand = normalize_brand(_guess_brand(parent_text or text))
        label = "aman ✅" if is_aman else text[:40]

        fast_promo = PromoExtraction(
            original_msg_id=m['id'],
            summary=f"{label}" + (f" — {parent_text[:80]}" if parent_text else ""),
            brand=brand,
            conditions="",
            valid_until="",
            status="active",
        )

        await db.save_pending_alert(
            brand=brand,
            p_data_json=fast_promo.model_dump_json(),
            tg_link=tg_link,
            timestamp=m['timestamp']
        )
        if _buffer_flush_task is None or _buffer_flush_task.done():
            _buffer_flush_task = asyncio.create_task(_flush_alert_buffer())

        print(f"⚡ Fast-path: {repr(text)} brand={brand}")
    # ── END FAST PATH ─────────────────────────────────────────────────────────

def _score_confidence(p: PromoExtraction, msg: dict, recent_alerts: list) -> int:
    """
    Returns confidence score 0-100.
    >= 70 → fire immediately
    < 70  → hold for corroboration
    """
    score = 0

    # Known brand in canon = high trust
    from db import normalize_brand
    if normalize_brand(p.brand) != "Unknown":
        score += 30

    # Summary contains price/discount signal = concrete info
    if re.search(r'(rp|rb|ribu|disc|diskon|gratis|free|off|%|cashback)', p.summary, re.IGNORECASE):
        score += 25

    # Active status from AI (not unknown)
    if p.status == 'active':
        score += 15

    # Is a reply (has context from parent)
    if msg.get('reply_to_msg_id'):
        score += 10

    # Already have a recent alert for this brand = corroboration
    brand_key = normalize_brand(p.brand).lower()
    matching = [r for r in recent_alerts if normalize_brand(r.get('brand', '')).lower() == brand_key]
    if matching:
        score += 20

    return min(score, 100)

async def _auto_triage_queue():
    """
    Emergency queue surgery: when backlog > 500, bulk-discard obvious noise
    without AI. Applies _is_worth_checking directly at DB level.
    Only runs if queue is critically large. Silent — no alerts sent.
    """
    global _queue_emergency_mode

    async with db.conn.execute("SELECT COUNT(*) FROM messages WHERE processed=0") as cur:
        queue = (await cur.fetchone())[0]

    if queue < 500:
        if _queue_emergency_mode:
            _queue_emergency_mode = False
            print("✅ Queue Surgeon: queue back to normal, exiting emergency mode.")
        return

    _queue_emergency_mode = True
    print(f"🔪 Queue Surgeon: {queue} unprocessed — triaging noise...")

    # Fetch the oldest unprocessed chunk (not the newest — let live msgs breathe)
    async with db.conn.execute("""
        SELECT id, text, reply_to_msg_id FROM messages
        WHERE processed=0
        ORDER BY id ASC LIMIT 500
    """) as cur:
        rows = await cur.fetchall()

    discard_ids = []
    for r in rows:
        text = r['text'] or ''
        # Keep if: has promo keyword OR is a reply (context-dependent) OR long enough to matter
        if r['reply_to_msg_id']:
            continue  # replies always need context — don't discard
        if gemini._is_worth_checking(text):
            continue  # has signal — let AI handle it
        if len(text.split()) >= 8:
            continue  # substance — let AI decide
        discard_ids.append(r['id'])

    if discard_ids:
        await db.mark_batch_processed(discard_ids)
        print(f"🗑️ Queue Surgeon: discarded {len(discard_ids)} noise messages. Remaining: {queue - len(discard_ids)}")
    else:
        print("🔪 Queue Surgeon: nothing safe to discard in this batch.")

async def processing_loop():
    global _buffer_flush_task
    
    # Seed dedup history once on boot
    recent_promos = await db.get_recent_alert_brands(hours=6, limit=300)
    for rp in recent_promos:
        _recent_alerts_history.append({"brand": rp['brand'], "summary": rp['summary']})

    # Semaphore: max 2 concurrent Gemini calls at once (fits within 8 RPM safely)
    sem = asyncio.Semaphore(2)
    _processing_active = False

    async def process_one_batch(msgs):
        async with sem:
            promos = await gemini.process_batch(msgs, db)
            if promos is None:
                print(f"⚠️ Gemini failed — {len(msgs)} msgs left unprocessed, will retry.")
                return

            filtered = await gemini.filter_duplicates(promos, _recent_alerts_history)
            if filtered:
                promos_to_save = []
                now_utc = datetime.now(timezone.utc)
                msg_id_map = {m['id']: m for m in msgs}

                for p in filtered:
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
                        brand_key = normalize_brand(p.brand)
                        confidence = _score_confidence(p, m, _recent_alerts_history)

                        if confidence >= 70:
                            # High confidence — fire immediately
                            await db.save_pending_alert(
                                brand=brand_key,
                                p_data_json=p.model_dump_json(),
                                tg_link=tg_link,
                                timestamp=m['timestamp']
                            )
                            global _buffer_flush_task
                            if _buffer_flush_task is None or _buffer_flush_task.done():
                                _buffer_flush_task = asyncio.create_task(_flush_alert_buffer())
                        else:
                            # Low confidence — hold for 15min corroboration window
                            expires_at = (datetime.now(timezone.utc) + timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M:%S')
                            await db.conn.execute("""
                                INSERT INTO pending_confirmations
                                    (brand, p_data_json, tg_link, timestamp, confidence, expires_at)
                                VALUES (?, ?, ?, ?, ?, ?)
                            """, (brand_key, p.model_dump_json(), tg_link, m['timestamp'], confidence, expires_at))
                            await db.conn.commit()
                            print(f"🔒 Gated (conf={confidence}): {brand_key} — {p.summary[:40]}")

                        _recent_alerts_history.append({"brand": normalize_brand(p.brand), "summary": p.summary})
                        if len(_recent_alerts_history) > 500:
                            _recent_alerts_history.pop(0)

                await db.save_promos_batch(promos_to_save, [m['id'] for m in msgs])
            else:
                await db.mark_batch_processed([m['id'] for m in msgs])

    print(f"🚀 Processing Loop Started (Concurrent Mode)")
    while True:
        try:
            # Auto-triage if queue is critically large
            await _auto_triage_queue()

            if _processing_active:
                await asyncio.sleep(5)
                continue

            # Reduce batch to what the rate limiter can actually handle per cycle
            # 8 RPM / 2 concurrent = 4 calls each, ~50 msgs per call = 100 max per cycle
            batch = await db.get_unprocessed_batch(batch_size=100)
            if not batch:
                await asyncio.sleep(3)
                continue

            msgs_to_proc = [
                {"id": r['id'], "text": r['text'], "timestamp": r['timestamp'],
                 "tg_msg_id": r['tg_msg_id'], "chat_id": r['chat_id'], "reply_to_msg_id": r['reply_to_msg_id']}
                for r in batch
            ]

            # Fast path first (no AI, immediate)
            await _fast_path_check(msgs_to_proc)

            batch_a = msgs_to_proc[:50]
            batch_b = msgs_to_proc[50:]

            tasks = [asyncio.create_task(process_one_batch(batch_a))]
            if batch_b:
                tasks.append(asyncio.create_task(process_one_batch(batch_b)))

            _processing_active = True
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                _processing_active = False

            await asyncio.sleep(3)
            
        except Exception as e:
            print(f"❌ processing_loop error: {e}")
            _processing_active = False
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
    try:
        now_wib = datetime.now(WIB)
        hour = now_wib.hour
        
        # Skip 02:00–05:00 WIB — midnight_digest_job covers this window
        if 2 <= hour < 5:
            return
        
        rows = await db.get_promos(hours=0.5)  # last 30 min
        if not rows:
            return
        
        context = "\n".join([f"- {r['brand']}: {r['summary']}" for r in rows])
        label = now_wib.strftime('%H:%M WIB')
        digest = await gemini.answer_question(
            f"Ringkas promo 30 menit terakhir. Waktu: {label}. Singkat dan padat.",
            context
        )
        await bot.send_digest(digest, f"{label} (30 menit)")
    except Exception as e:
        print(f"❌ halfhour_digest_job error: {e}")

async def _reconnect_listener(gap_minutes: float):
    """Disconnect and reconnect Telethon, then replay missed messages."""
    global _listener_reconnecting
    if _listener_reconnecting:
        return
    _listener_reconnecting = True
    try:
        print(f"🔌 Listener reconnect triggered (lag: {int(gap_minutes)}m)...")
        try:
            await listener.client.disconnect()
        except Exception:
            pass  # already disconnected — fine
        await asyncio.sleep(3)
        await listener.client.connect()
        # Replay the gap window (cap at 3h to avoid flood)
        catchup_hours = min(gap_minutes / 60 + 0.25, 3.0)
        await listener.sync_history(hours=catchup_hours)
        print(f"✅ Listener reconnected. Replayed {catchup_hours:.1f}h of history.")
    except Exception as e:
        print(f"❌ Listener reconnect failed: {e}")
    finally:
        _listener_reconnecting = False

async def heartbeat_job():
    try:
        now_wib = datetime.now(WIB).strftime('%H:%M WIB')
        async with db.conn.execute("SELECT COUNT(*) FROM messages WHERE processed=0") as cur:
            queue = (await cur.fetchone())[0]
        async with db.conn.execute("SELECT MAX(timestamp) FROM messages") as cur:
            last_ts = (await cur.fetchone())[0]

        lag_note = ""
        if last_ts:
            last_dt = _parse_ts(last_ts)
            lag_min = (datetime.now(timezone.utc) - last_dt).total_seconds() / 60

            if lag_min > 20 and not _listener_reconnecting:
                # Self-heal: reconnect silently, don't just warn
                asyncio.create_task(_reconnect_listener(lag_min))
                lag_note = f"\n🔌 *Lag {int(lag_min)}m — auto-reconnecting...*"
            elif lag_min > 5:
                lag_note = f"\n⚠️ *Lag: {int(lag_min)}m*"

        # RPM pressure indicator
        now_loop = asyncio.get_event_loop().time()
        rpm_used = len([t for t, _ in gemini._last_calls if now_loop - t < 60])
        rpm_note = f" | rpm: `{rpm_used}/{gemini._rpm_limit}`"

        text = f"💓 `{now_wib}` | queue: `{queue}`{rpm_note}{lag_note}"
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

        calls_this_run = 0
        for t in threads:
            if calls_this_run >= 3:
                break
            last_count = _alerted_hot_threads.get(t['tg_msg_id'], 0)
            
            # Alert if new or if it gained significant traction (+5 replies) since last alert
            if t['reply_count'] >= last_count + 5:
                _alerted_hot_threads[t['tg_msg_id']] = t['reply_count']
                calls_this_run += 1
                
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

        # Skip if we've already used 6+ slots this minute to stay under RPM
        now = asyncio.get_event_loop().time()
        recent = [t for t in gemini._last_calls if now - t[0] < 60]
        if len(recent) >= 6:
            return  # loop is busy, don't pile on

        one_min_ago = (datetime.now(timezone.utc) - timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M:%S+00:00')
        async with db.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE timestamp >= ?", (one_min_ago,)
        ) as cur:
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
        # Rolling DB: delete processed messages older than 1 day that don't back any promo
        await db.conn.execute("""
            DELETE FROM messages 
            WHERE processed=1 
            AND timestamp < strftime('%Y-%m-%d %H:%M:%S+00:00','now','-1 day')
            AND id NOT IN (SELECT source_msg_id FROM promos WHERE source_msg_id IS NOT NULL)
        """)
        await db.conn.commit()
        print("🔧 DB maintenance: WAL checkpointed, old messages (1d+) pruned.")
    except Exception as e:
        print(f"❌ db_maintenance_job error: {e}")

async def dead_promo_reaper_job():
    """
    Scans promos marked 'active' and checks their reply threads for expiry signals.
    If found → auto-flips to 'expired'. No AI needed — pure regex on stored text.
    """
    EXPIRY_SIGNALS = re.compile(
        r'\b(nt|abis|habis|sold.?out|expired|kehabisan|ga bisa|gabisa|udah mati|mati|nonaktif|hangus|error terus|ga work|gak work|off)\b',
        re.IGNORECASE
    )
    try:
        # Get active promos from last 6 hours that still have a source message
        async with db.conn.execute("""
            SELECT p.id, p.source_msg_id, p.brand, p.summary,
                   m.tg_msg_id, m.chat_id
            FROM promos p
            JOIN messages m ON p.source_msg_id = m.id
            WHERE p.status = 'active'
            AND p.created_at >= strftime('%Y-%m-%d %H:%M:%S','now','-6 hours')
        """) as cur:
            active_promos = await cur.fetchall()

        if not active_promos:
            return

        reaped = 0
        for promo in active_promos:
            # Fetch replies to this promo's source message
            reply_rows = await db.get_thread_replies(promo['tg_msg_id'], promo['chat_id'], limit=30)
            if not reply_rows:
                continue

            expiry_votes = 0
            active_votes = 0
            for r in reply_rows:
                text = r['text'] or ''
                if EXPIRY_SIGNALS.search(text):
                    expiry_votes += 1
                # Also count confirmations to avoid false reaping
                if re.search(r'\b(aman|on|work|jp|masih|masih bisa)\b', text, re.IGNORECASE):
                    active_votes += 1

            # Reap if: expiry signals outnumber active signals AND at least 2 expiry votes
            if expiry_votes >= 2 and expiry_votes > active_votes:
                await db.conn.execute(
                    "UPDATE promos SET status='expired' WHERE id=?", (promo['id'],)
                )
                reaped += 1
                print(f"💀 Reaper: expired promo '{promo['brand']} — {promo['summary'][:40]}' ({expiry_votes} expiry votes vs {active_votes} active)")

        if reaped:
            await db.conn.commit()
            print(f"💀 Dead Promo Reaper: flipped {reaped} promos to expired.")

    except Exception as e:
        print(f"❌ dead_promo_reaper_job error: {e}")

async def confirmation_gate_job():
    """
    Checks pending_confirmations table. Fires alerts if:
    - corroborations >= 1 (someone else confirmed), OR
    - item has aged past 15 minutes (time-based release, fire anyway)
    Deletes expired entries that never got corroborated.
    """
    global _buffer_flush_task
    try:
        now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        async with db.conn.execute("""
            SELECT id, brand, p_data_json, tg_link, timestamp, corroborations, expires_at
            FROM pending_confirmations
            WHERE expires_at <= ? OR corroborations >= 1
            ORDER BY id ASC
        """, (now_str,)) as cur:
            ready = await cur.fetchall()

        if not ready:
            return

        fired_ids = []
        for r in ready:
            p_data = PromoExtraction.model_validate_json(r['p_data_json'])
            # Only fire if corroborated OR expired (time-based release)
            if r['corroborations'] >= 1:
                print(f"✅ Confirmation gate: CORROBORATED — {r['brand']}: {p_data.summary[:40]}")
            else:
                # Timed out with no corroboration — fire but with lower priority
                print(f"⏱️ Confirmation gate: TIMEOUT RELEASE — {r['brand']}: {p_data.summary[:40]}")

            await db.save_pending_alert(r['brand'], r['p_data_json'], r['tg_link'], r['timestamp'])
            if _buffer_flush_task is None or _buffer_flush_task.done():
                _buffer_flush_task = asyncio.create_task(_flush_alert_buffer())
            fired_ids.append(r['id'])

        if fired_ids:
            placeholders = ','.join('?' * len(fired_ids))
            await db.conn.execute(f"DELETE FROM pending_confirmations WHERE id IN ({placeholders})", fired_ids)
            await db.conn.commit()

    except Exception as e:
        print(f"❌ confirmation_gate_job error: {e}")

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
    # Sync history on boot
    asyncio.create_task(listener.sync_history(2, catchup_hours=BOOT_CATCHUP_WINDOW/3600))

    # Image processing job every 3 minutes
    scheduler.add_job(image_processing_job, "interval", minutes=3, id="images", misfire_grace_time=120)
    # Hourly: skip 2am, 3am, 4am WIB to avoid sleep alerts
    scheduler.add_job(hourly_digest_job, "cron", minute=0, hour="0,1,5-23", id="digest", misfire_grace_time=120)
    # Midnight catch-up for 2am-5am window
    scheduler.add_job(midnight_digest_job, "cron", hour=5, minute=0, id="midnight_digest", misfire_grace_time=300)
    # 30-min digest at :15 and :45 to complement hourly at :00
    scheduler.add_job(halfhour_digest_job, "cron", minute="15,45", id="halfhour_digest", misfire_grace_time=60)
    scheduler.add_job(heartbeat_job, "cron", minute=30, id="heartbeat", misfire_grace_time=60)
    scheduler.add_job(hot_thread_job, "interval", minutes=8, id="hot_threads", misfire_grace_time=60)
    scheduler.add_job(time_mention_job, "interval", minutes=2, id="time_mentions", misfire_grace_time=20)
    scheduler.add_job(trend_job, "interval", minutes=10, id="trend_job", misfire_grace_time=60)
    scheduler.add_job(spike_detection_job, "interval", minutes=1, id="spike", misfire_grace_time=30)
    scheduler.add_job(dead_promo_reaper_job, "interval", minutes=15, id="reaper", misfire_grace_time=120)
    scheduler.add_job(confirmation_gate_job, "interval", minutes=3, id="confirm_gate", misfire_grace_time=60)
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
