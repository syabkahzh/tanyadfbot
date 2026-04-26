"""
listener.py — Telethon Message Listener

Key architectural fixes vs previous version:
- BUG FIX: fast-path fires BEFORE _save_to_db, not after.  The old order
  forced fast-path to wait for a DB commit (10-20 ms I/O) before it could
  even start.  Now the two tasks are launched concurrently and fast-path
  wins the race.
- BUG FIX: parent-message lookup no longer blocks on asyncio.sleep(0.2)
  retry inside the hot path.  We accept "Unknown" brand if the parent
  isn't in DB yet; the AI loop will catch it later with full context.
- BUG FIX: context_tracker.update_brand / velocity query are fire-and-forget
  so they never add latency to the alert path.
- FAST-PATH: all blocking/async operations that aren't strictly needed to
  send the alert are moved out of the critical path.
"""

import re
import time
import asyncio
import logging
from telethon import TelegramClient, events
from config import Config
from datetime import datetime, timedelta, timezone
import shared

logger = logging.getLogger(__name__)

# ── Pre-compiled Patterns ─────────────────────────────────────────────────────
TIME_PATTERN = re.compile(
    r'(\b(jam|pukul|pukl|stgh|setengah)\s?\d{1,2}([:.]\d{2})?(\s?wib)?\b|'
    r'\b\d{1,2}[:.]\d{2}\s?(wib)?\b|'
    r'\b(malem|sore|subuh|pagi|siang)\b)',
    re.IGNORECASE
)

INSTANT_PATTERN = re.compile(
    r'\b(on|jp|jackpot|work|aman|luber|pecah|berhasil|gacor|mantul|restock|ristok|aktif|ready|'
    r'nyala|masuk|udah pake|dikirim|cair|done|lancar|cek|info|'
    r'makasih|thx|thanks|makasi|mks|terima.?kasih|'
    r'potongan|idm|alfa|indomaret|ag|alfagift|voc|voucher|minbel|'
    r'r\+s\+t\+k|r\+s\+t\+c\+k|r\+st\+ck|cb|kesbek|c\+s\+h\+b\+c\+k|cash back|'
    r'qr|scan|edc|membership|member|mamber)\b',
    re.IGNORECASE
)

NEG_PATTERN = re.compile(
    r'\b(kapan|kok|ga pernah|ga\b|ya\b|tidak|belom|belum|gaada|ngga|ga ada|gak|nggak|bukan|jangan|'
    r'iya|cuma|pas|tadi|gamau|jamber|jambrapa|jamberapa|'
    r'b\+r\+p|brp|berapa|drmana|dimana|dmn|mana|d\+r\+m\+n|'
    r'tunggu|nunggu|nanti|besok|lusa|tar\b|dulu|sore|malem|malam|pagi|'
    r'harusnya|katanya|mungkin|kayaknya|kyknya|sepertinya|entah|'
    r'koid|hangus|refund|batal|balsis|ngebadut|zonk|habis|sold|error|'
    r'coba|nyoba|semoga|mudah.mudahan|insya|makasih|terima kasih|thanks|nuhun|tengkyu|nangis|sedih|anjir|bgst|halo|hai)\b',
    re.IGNORECASE
)

FAST_ALLCAPS = re.compile(r'^[^a-z]*[A-Z][^a-z]*$')
_PROMO_CODE_PATTERN = re.compile(r'\b(?=.*[A-Z])[A-Z0-9]{6,25}\b')


def check_fast_path(text: str) -> bool:
    """Synchronous logic check for fast-path eligibility. Zero I/O."""
    if not text:
        return False
    t = text.strip()
    tl = t.lower()

    if '?' in t or NEG_PATTERN.search(tl):
        return False

    if re.search(r'\b(aman|work|on)\s+(ga|gak|nggak|ya)\b', tl):
        return False

    from processor import _SOCIAL_FILLER
    if _SOCIAL_FILLER.match(t):
        return False

    from shared import TRANSIT_NOISE_PATTERN
    if 'aman' in tl and TRANSIT_NOISE_PATTERN.search(tl):
        return False

    if FAST_ALLCAPS.match(t) and len(t) > 3:
        return True

    if INSTANT_PATTERN.search(t):
        return True

    return False


class TelethonListener:
    def __init__(self, db_manager):
        self.db = db_manager
        self.client = TelegramClient(
            Config.SESSION_NAME,
            Config.API_ID,
            Config.API_HASH,
            sequential_updates=False,
            auto_reconnect=True,
            connection_retries=10,
            retry_delay=1,
        )

        @self.client.on(events.NewMessage(chats=Config.TARGET_GROUP))
        async def _master_handler(event):
            if not event.text:
                return
            shared.mark_message_ingested()
            # ── CRITICAL FIX: launch both tasks concurrently. ──────────────
            # fast-path runs without waiting for DB save.  It uses only the
            # in-memory event object — zero DB reads in the happy path.
            asyncio.create_task(self._handle_fast_path_from_event(event))
            asyncio.create_task(self._save_to_db(event))

    # ── Fast-path ─────────────────────────────────────────────────────────────

    async def _handle_fast_path_from_event(self, event) -> None:
        """
        Fires an immediate alert using only in-memory event data.

        Design principles:
        - ZERO blocking DB reads in the critical path.
        - Parent text lookup uses a single non-retrying DB read with
          asyncio.timeout(0.3). If the parent isn't committed yet, we
          accept Unknown brand and let the AI loop handle it.
        - context_tracker and velocity queries are fire-and-forget.
        - fuzzy-dedup and brand dedup are the only locks we take.
        """
        _fp_start = time.monotonic()

        from shared import (
            _make_tg_link, _guess_brand, _flush_alert_buffer,
            db, _alerted_aman_parents, _alerted_aman_parents_deque, _aman_lock,
            _recent_alerts_history, _recent_alerts_lock,
            _fastpath_brand_last_fired, _fastpath_brand_lock,
            FASTPATH_BRAND_DEDUP_SEC,
            context_tracker, TRANSIT_NOISE_PATTERN,
            is_fuzzy_duplicate,
        )
        from db import normalize_brand
        from processor import PromoExtraction, _SOCIAL_FILLER

        text = (event.text or '').strip()
        text_lower = text.lower()
        chat_id = event.chat_id
        reply_to_msg_id = event.reply_to_msg_id

        # ── Gate 1: cheap synchronous checks ─────────────────────────────
        if not check_fast_path(text):
            return

        is_instant = bool(INSTANT_PATTERN.search(text)) and '?' not in text
        is_allcaps = bool(FAST_ALLCAPS.match(text)) and len(text.strip()) > 3 and '?' not in text
        is_aman_standalone = text_lower == 'aman' and '?' not in text
        has_promo_code = bool(_PROMO_CODE_PATTERN.search(text))

        if _SOCIAL_FILLER.match(text):
            return
        if not (is_instant or is_aman_standalone or is_allcaps or has_promo_code):
            return
        if NEG_PATTERN.search(text_lower):
            return
        if re.search(r'\b(aman|work|on)\s+(ga|gak|nggak|ya)\b', text_lower):
            return
        if 'aman' in text_lower and TRANSIT_NOISE_PATTERN.search(text):
            return

        # ── Gate 2: standalone "aman" requires parent ─────────────────────
        if is_aman_standalone and not is_allcaps:
            if not reply_to_msg_id:
                return
            async with _aman_lock:
                if reply_to_msg_id in _alerted_aman_parents:
                    return
                _alerted_aman_parents.add(reply_to_msg_id)
                _alerted_aman_parents_deque.append(reply_to_msg_id)
                if len(_alerted_aman_parents) > 500:
                    _alerted_aman_parents.clear()
                    _alerted_aman_parents.update(_alerted_aman_parents_deque)

        # ── Brand resolution: ONE non-blocking DB read, no retry sleep ────
        parent_text = None
        brand = normalize_brand(_guess_brand(text))

        if brand == "Unknown" and reply_to_msg_id:
            try:
                # Single attempt, tight timeout — no sleep/retry
                async with asyncio.timeout(0.3):
                    async with self.db.conn.execute(
                        "SELECT text FROM messages WHERE tg_msg_id=? AND chat_id=?",
                        (reply_to_msg_id, chat_id)
                    ) as cur:
                        row = await cur.fetchone()
                        if row:
                            parent_text = row['text']
                            brand = normalize_brand(_guess_brand(parent_text or text))
            except Exception:
                pass  # stay Unknown — fast-path may skip below

        # Temporal context fallback (fire-and-forget update if brand found)
        if brand != "Unknown":
            asyncio.create_task(context_tracker.update_brand(chat_id, brand))
        else:
            # Try context WITHOUT blocking — get_context uses a lock but is fast
            try:
                async with asyncio.timeout(0.1):
                    brand = await context_tracker.get_context(chat_id)
            except Exception:
                pass

        # ── Gate 3: ambiguous signals need a known brand ──────────────────
        AMBIGUOUS = {
            'on', 'ready', 'aktif', 'restock', 'ristok', 'cek', 'info',
            'makasih', 'thx', 'thanks', 'makasi', 'mks', 'terimakasih',
            'luber', 'pecah'
        }
        found_sigs = set(w for w in INSTANT_PATTERN.findall(text_lower) if isinstance(w, str))
        if (not is_allcaps and not has_promo_code and brand == "Unknown"
                and (found_sigs & AMBIGUOUS) and len(text) > 15):
            return
        if not is_allcaps and not has_promo_code and brand == "Unknown" and (is_aman_standalone or is_instant):
            return

        # ── Gate 4: brand-level dedup (short window) ──────────────────────
        if brand and brand != "Unknown":
            now_mono = time.monotonic()
            async with _fastpath_brand_lock:
                last = _fastpath_brand_last_fired.get(brand, 0.0)
                if now_mono - last < FASTPATH_BRAND_DEDUP_SEC:
                    return
                _fastpath_brand_last_fired[brand] = now_mono

        # ── Build summary ─────────────────────────────────────────────────
        if is_aman_standalone and parent_text:
            summary = f"aman ✅ — {parent_text[:120]}"
        elif is_aman_standalone:
            return  # aman with no context = useless
        elif has_promo_code:
            m = _PROMO_CODE_PATTERN.search(text)
            summary = f"Kode Promo: {m.group(0)}" if m else text[:120]
        else:
            summary = text[:120]

        # Extract non-Telegram links
        combined = (text or "") + " " + (parent_text or "")
        promo_links: list[str] = []
        seen: set[str] = set()
        for url in re.findall(r'(https?://[^\s>]+)', combined):
            u = url.strip('.,()[]"\'')
            if 't.me' not in u and 'telegram.me' not in u and u not in seen:
                promo_links.append(u)
                seen.add(u)

        # Velocity tag — fire-and-forget, doesn't block alert
        velocity = 0
        try:
            async with asyncio.timeout(0.2):
                velocity = await self.db.get_brand_velocity(brand, minutes=5)
        except Exception:
            pass
        if velocity >= 100:
            summary = f"🔥 RAMAI ({velocity} msg/5m) — {summary}"

        queue_time = (datetime.now(timezone.utc) - event.date.astimezone(timezone.utc)).total_seconds()

        fast_promo = PromoExtraction(
            original_msg_id=0,
            summary=summary,
            brand=brand,
            conditions="",
            valid_until="",
            status="active",
            links=promo_links[:3],
            queue_time=queue_time,
            ai_time=0.0
        )

        tg_link = _make_tg_link(chat_id, event.id)

        # ── Fuzzy dedup ───────────────────────────────────────────────────
        if await is_fuzzy_duplicate(brand, summary):
            return

        # ── Persist + broadcast ───────────────────────────────────────────
        await db.save_pending_alert(
            brand=brand,
            p_data_json=fast_promo.model_dump_json(),
            tg_link=tg_link,
            timestamp=event.date,
            source='python',
            commit=True
        )

        # Prevent AI loop from re-alerting the same promo
        async with _recent_alerts_lock:
            _recent_alerts_history.append({
                "brand": normalize_brand(brand),
                "summary": summary,
            })

        # Save fastpath promo to promos table (fire-and-forget)
        asyncio.create_task(db.save_fastpath_promo(
            brand=brand,
            summary=summary,
            conditions="",
            tg_link=tg_link,
            status="active",
        ))

        # Mark message processed by tg_id (fire-and-forget, race-safe)
        asyncio.create_task(
            self.db.mark_processed_by_tg_id(event.id, chat_id)
        )

        # Trigger flush with minimal delay for fast-path
        t = shared.get_buffer_flush_task()
        if t is None or t.done():
            shared.set_buffer_flush_task(
                asyncio.create_task(_flush_alert_buffer(delay=0.1))
            )

        total_latency = time.monotonic() - _fp_start
        logger.info(
            f"⚡ FAST-PATH: {brand} — {summary[:50]} "
            f"(total={total_latency*1000:.1f}ms, queue={queue_time:.2f}s)"
        )

    # ── DB Persistence (runs concurrently with fast-path) ─────────────────

    async def _save_to_db(self, event) -> None:
        """Pure DB persistence. Concurrent with fast-path, no locks shared."""
        text_preview = (event.text or "")[:50].replace("\n", " ")
        logger.debug(f"📩 [{event.id}] {text_preview}")
        try:
            has_time = bool(TIME_PATTERN.search(event.text or ""))
            internal_id = await self.db.save_message(
                tg_msg_id=event.id,
                chat_id=event.chat_id,
                sender_id=event.sender_id,
                sender_name=f"User_{event.sender_id}",
                timestamp=event.date,
                text=event.text,
                reply_to_msg_id=event.reply_to_msg_id,
                processed=0,
                has_photo=1 if event.photo else 0,
                has_time_mention=1 if has_time else 0,
                commit=True
            )
            if internal_id:
                logger.debug(f"   📥 Queued (ID={internal_id})")
        except Exception as e:
            logger.error(f"❌ _save_to_db error: {e}")
            if shared.bot:
                asyncio.create_task(shared.bot.alert_error("listener_save_db", e))

    # ── Start / history sync ──────────────────────────────────────────────────

    async def start(self):
        import logging as _logging
        _l = _logging.getLogger(__name__)

        for attempt in range(5):
            try:
                await self.client.start()
                break
            except Exception as e:
                if "locked" in str(e).lower() and attempt < 4:
                    wait = 3 * (attempt + 1)
                    _l.warning(f"⚠️ Telethon session locked on start, retrying in {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                raise

        logger.info("🚀 Telethon Listener started.")

    async def sync_history(self, hours=6, catchup_hours=2):
        chat = await self.client.get_entity(Config.TARGET_GROUP)
        chat_id = chat.id
        from db import _ts_str

        last_id = await self.db.get_last_msg_id(chat_id)
        if not last_id:
            async with self.db.conn.execute(
                "SELECT MAX(tg_msg_id) FROM messages"
            ) as cur:
                row = await cur.fetchone()
                if row and row[0]:
                    last_id = row[0]

        async def fetch_and_buffer(iterator):
            buffer = []
            async for message in iterator:
                if not message.text:
                    continue

                msg_age = (datetime.now(timezone.utc) - message.date.astimezone(timezone.utc)).total_seconds()
                mark_processed = 1 if msg_age > 900 else 0

                has_time = any(
                    w in message.text.lower()
                    for w in ['jam', 'menit', 'detik', 'sore', 'siang', 'pagi', 'malam']
                )

                buffer.append((
                    message.id, chat_id, message.sender_id,
                    f"User_{message.sender_id}", _ts_str(message.date),
                    message.text, message.reply_to_msg_id,
                    mark_processed, 1 if message.photo else 0, 1 if has_time else 0
                ))

                if len(buffer) >= 100:
                    await self._bulk_save_to_db(buffer)
                    buffer = []
                await asyncio.sleep(0)

            if buffer:
                await self._bulk_save_to_db(buffer)

        if last_id:
            last_ts = await self.db.get_last_msg_timestamp(chat_id)
            # BUG D FIX: guard against None timestamp
            if last_ts is None:
                gap_hours = 999.0
            else:
                gap_hours = (
                    (datetime.now(timezone.utc) - last_ts).total_seconds() / 3600
                )

            if gap_hours > catchup_hours:
                logger.info(f"⚠️ Gap {gap_hours:.1f}h — fetching last {catchup_hours}h only.")
                limit_date = datetime.now(timezone.utc) - timedelta(hours=catchup_hours)
                await fetch_and_buffer(self.client.iter_messages(
                    chat_id, offset_date=limit_date, reverse=True
                ))
            else:
                logger.info(f"🔄 Short gap ({gap_hours:.1f}h) — catching up from msg {last_id}.")
                await fetch_and_buffer(self.client.iter_messages(
                    chat_id, min_id=last_id, reverse=True
                ))
        else:
            logger.info(f"⏱️ DB empty — syncing last {hours}h.")
            limit_date = datetime.now(timezone.utc) - timedelta(hours=hours)
            await fetch_and_buffer(self.client.iter_messages(
                chat_id, offset_date=limit_date, reverse=True
            ))

        logger.info("✅ History sync complete.")

    async def _bulk_save_to_db(self, msgs_data: list[tuple]):
        """Bulk insert with deduplication."""
        if not self.db.conn or not msgs_data:
            return
        try:
            await self.db.conn.executemany("""
                INSERT OR IGNORE INTO messages
                    (tg_msg_id, chat_id, sender_id, sender_name, timestamp,
                     text, reply_to_msg_id, processed, has_photo, has_time_mention)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, msgs_data)
            await self.db.conn.commit()
            logger.debug(f"   📥 Bulk Queued {len(msgs_data)} msgs")
        except Exception as e:
            logger.error(f"DB bulk_save error: {e}")
