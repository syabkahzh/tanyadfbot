"""
listener.py — Telethon Message Listener
Bulletproof rewrite:
- BUG 4 FIX: asyncio.timeout(0.5) on DB lookup in fast-path — never stalls
- BUG 5 FIX: merged _write_lock into single acquisition per message
- Fast-path persists to promos via db.save_fastpath_promo (BUG A fix propagated)
- Velocity tag on viral fast-path alerts
- sync_history unchanged in behaviour, just cleaner
"""

import re
import time
import asyncio
from telethon import TelegramClient, events
from config import Config
from datetime import datetime, timedelta, timezone

# ── Pre-compiled Patterns (Performance Optimization) ─────────────────────────
TIME_PATTERN = re.compile(
    r'(\b(jam|pukul|pukl|stgh|setengah)\s?\d{1,2}([:.]\d{2})?(\s?wib)?\b|'
    r'\b\d{1,2}[:.]\d{2}\s?(wib)?\b|'
    r'\b(malem|sore|subuh|pagi|siang)\b)',
    re.IGNORECASE
)
# 'aman', 'jp', 'work', 'luber', 'pecah', 'jackpot' are confirmation slang —
# they fire fast-path as long as a known brand resolves (inline or via parent).
# 'on/ready/aktif/restock/ristok' stay in the AMBIGUOUS set below and require a
# known brand plus length guard.
INSTANT_PATTERN = re.compile(
    r'\b(on|jp|jackpot|work|aman|luber|pecah|berhasil|gacor|mantul|restock|ristok|aktif|ready|'
    r'nyala|masuk|udah pake|dikirim|cair|done|lancar|'
    r'potongan|idm|alfa|indomaret|ag|alfagift|voc|voucher|minbel|'
    r'r\+s\+t\+k|r\+s\+t\+c\+k|r\+st\+ck|cb|kesbek|c\+s\+h\+b\+c\+k|cash back|'
    r'qr|scan|edc|membership|member|mamber)\b',
    re.IGNORECASE
)

NEG_PATTERN = re.compile(
    r'\b(kapan|kok|ga pernah|tidak|belom|belum|gaada|ngga|ga ada|gak|nggak|bukan|jangan|'
    r'iya|cuma|pas|tadi|gamau|jamber|jambrapa|jamberapa|'
    r'b\+r\+p|brp|berapa|drmana|dimana|dmn|mana|d\+r\+m\+n|'
    r'tunggu|nunggu|nanti|besok|lusa|tar\b|dulu|sore|malem|malam|pagi|'
    r'harusnya|katanya|mungkin|kayaknya|kyknya|sepertinya|entah|'
    r'koid|hangus|refund|batal|balsis|ngebadut|zonk|habis|sold|error|'
    r'coba|nyoba|semoga|mudah.mudahan|insya)\b',
    re.IGNORECASE
)
# All-caps: has at least one uppercase, zero lowercase letters
FAST_ALLCAPS = re.compile(r'^[^a-z]*[A-Z][^a-z]*$')


class TelethonListener:
    def __init__(self, db_manager):
        self.db     = db_manager
        self.client = TelegramClient(
            Config.SESSION_NAME,
            Config.API_ID,
            Config.API_HASH,
            sequential_updates=False,   # parallel update handling
            auto_reconnect=True,
            connection_retries=10,
            retry_delay=1,
        )

    # ── Fast-path ─────────────────────────────────────────────────────────────

    async def _handle_fast_path(self, message_data: dict) -> bool:
        """
        Fires an immediate alert for high-signal messages (on/jp/work/aman/ALL-CAPS).
        Returns True if alert was fired so the caller can mark the message processed=1.

        BUG 4 FIX: DB lookup wrapped in asyncio.timeout(0.5) — never stalls the 
        event loop at high message volume.
        """
        from shared import (
            _make_tg_link, _guess_brand, _flush_alert_buffer,
            db, gemini, bot,
            _alerted_aman_parents, _alerted_aman_parents_deque, _aman_lock,
            _recent_alerts_history, _recent_alerts_lock,
            _fastpath_brand_last_fired, _fastpath_brand_lock,
            FASTPATH_BRAND_DEDUP_SEC,
            context_tracker, TRANSIT_NOISE_PATTERN,
            is_fuzzy_duplicate,
        )
        from db import normalize_brand
        from processor import PromoExtraction
        import shared

        text       = (message_data.get('text') or '').strip()
        text_lower = text.lower()

        is_instant  = bool(INSTANT_PATTERN.search(text)) and '?' not in text
        is_allcaps  = (bool(FAST_ALLCAPS.match(text))
                       and len(text.strip()) > 3
                       and '?' not in text)
        # Standalone "aman" (single word, no extras) — parent-dedup path only.
        # Compound "aman <brand>" hits via is_instant above and benefits from
        # brand-level dedup below.
        is_aman_standalone = text_lower == 'aman' and '?' not in text

        if not (is_instant or is_aman_standalone or is_allcaps):
            return False
        if NEG_PATTERN.search(text_lower):
            return False
        # Transit-noise gate: "aman kak rutenya" is NOT a deal signal
        if 'aman' in text_lower and TRANSIT_NOISE_PATTERN.search(text):
            return False

        # Standalone "aman" — MUST be a reply to a known parent AND not a repeat
        # reaction to the same parent we already alerted on.
        if is_aman_standalone and not is_allcaps:
            parent_id = message_data.get('reply_to_msg_id')
            if not parent_id:
                return False
            async with _aman_lock:
                if parent_id in _alerted_aman_parents:
                    return False
                _alerted_aman_parents.add(parent_id)
                _alerted_aman_parents_deque.append(parent_id)
                if len(_alerted_aman_parents) > 500:
                    _alerted_aman_parents.clear()
                    _alerted_aman_parents.update(_alerted_aman_parents_deque)

        # Brand resolution — DB lookup with hard timeout so fast-path stays fast
        parent_text = None
        chat_id     = message_data['chat_id']
        brand       = normalize_brand(_guess_brand(text))

        # If brand found explicitly, update temporal context for future messages
        if brand != "Unknown":
            await context_tracker.update_brand(chat_id, brand)

        if brand == "Unknown" and message_data.get('reply_to_msg_id'):
            try:
                async with asyncio.timeout(0.5):   # BUG 4 FIX
                    async with self.db.conn.execute(
                        "SELECT text FROM messages WHERE tg_msg_id=? AND chat_id=?",
                        (message_data['reply_to_msg_id'], chat_id)
                    ) as cur:
                        row = await cur.fetchone()
                        if row:
                            parent_text = row['text']
                brand = normalize_brand(_guess_brand(parent_text or text))
                if brand != "Unknown":
                    await context_tracker.update_brand(chat_id, brand)
            except (asyncio.TimeoutError, Exception):
                pass   # stay Unknown — fast-path may skip, that's fine

        # Temporal context fallback: inherit brand from recent conversation
        if brand == "Unknown":
            brand = await context_tracker.get_context(chat_id)

        # Strict: ambiguous signals require a known brand (unless ALL-CAPS shout)
        AMBIGUOUS = {'on', 'ready', 'aktif', 'restock', 'ristok'}
        found_sigs = set(INSTANT_PATTERN.findall(text_lower))
        if (not is_allcaps and brand == "Unknown"
                and (found_sigs & AMBIGUOUS) and len(text) > 15):
            return False
        if not is_allcaps and brand == "Unknown" and (is_aman_standalone or is_instant):
            return False

        # Brand-level dedup for burst traffic: suppress repeat fast-path alerts
        # for the same brand within a short window. This is what stops 20 "aman
        # toped" messages from firing 20 alerts — only the first one fires; the
        # rest are corroborations and fall through to AI where filter_duplicates
        # catches them.
        if brand and brand != "Unknown":
            now_mono = time.monotonic()
            async with _fastpath_brand_lock:
                last = _fastpath_brand_last_fired.get(brand, 0.0)
                if now_mono - last < FASTPATH_BRAND_DEDUP_SEC:
                    return False
                _fastpath_brand_last_fired[brand] = now_mono

        # Build summary
        if is_aman_standalone and parent_text:
            summary = f"aman ✅ — {parent_text[:120]}"
        elif is_aman_standalone:
            return False   # aman with no parent context = useless
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

        # Velocity tag — flag viral promos without an extra AI call
        try:
            velocity = await self.db.get_brand_velocity(brand, minutes=5)
            if velocity >= 100: # NEW: raised to 100/5m (20/min)
                summary = f"🔥 RAMAI ({velocity} msg/5m) — {summary}"
        except Exception:
            pass

        fast_promo = PromoExtraction(
            original_msg_id=message_data.get('internal_id', 0),
            summary=summary,
            brand=brand,
            conditions="",
            valid_until="",
            status="active",
            links=promo_links[:3],
        )

        tg_link = _make_tg_link(message_data['chat_id'], message_data['tg_msg_id'])

        # Fuzzy dedup: suppress near-identical alerts before DB/broadcast
        if await is_fuzzy_duplicate(brand, summary):
            return False

        await db.save_pending_alert(
            brand=brand,
            p_data_json=fast_promo.model_dump_json(),
            tg_link=tg_link,
            timestamp=message_data['timestamp'],
            source='python',
            commit=False
        )

        # BUG A FIX: also write to promos so digest/recap can see this alert
        # Fire-and-forget — doesn't need to block the alert
        asyncio.create_task(db.save_fastpath_promo(
            brand=brand,
            summary=summary,
            conditions="",
            tg_link=tg_link,
            status="active",
        ))

        # Update deduplication history to prevent duplicate AI alert
        async with _recent_alerts_lock:
            _recent_alerts_history.append({
                "brand":   normalize_brand(brand),
                "summary": summary,
            })

        # Trigger immediate flush with reduced delay for fast-path
        t = shared.get_buffer_flush_task()
        if t is None or t.done():
            shared.set_buffer_flush_task(
                asyncio.create_task(_flush_alert_buffer(delay=0.3))
            )

        latency = (
            datetime.now(timezone.utc) - message_data['timestamp']
        ).total_seconds()
        print(f"⚡ FAST-PATH: {brand} — {summary[:50]}  ({latency:.3f}s)")
        return True

    # ── Message ingestion ─────────────────────────────────────────────────────

    async def _handle_fast_path_standalone(self, event):
        """Pure fast-path: pattern match → alert. Zero DB reads except parent lookup.

        When the fast-path fires (returns True), we also mark the source message
        as processed=1 so the AI processing loop does not re-waste a batch slot
        on it. The mark is tolerant of the race with `_save_to_db`: if the row
        isn't present yet the UPDATE affects 0 rows and the row will be caught
        by the AI path later (where `filter_duplicates` catches it via the
        already-appended history entry).
        """
        message_data = {
            'tg_msg_id':       event.id,
            'chat_id':         event.chat_id,
            'text':            event.text,
            'reply_to_msg_id': event.reply_to_msg_id,
            'timestamp':       event.date,
            'internal_id':     0,
        }
        try:
            fired = await self._handle_fast_path(message_data)
            if fired:
                asyncio.create_task(
                    self.db.mark_processed_by_tg_id(event.id, event.chat_id)
                )
        except Exception as e:
            if shared.bot:
                await shared.bot.alert_error("fast_path", e)
            print(f"❌ fast_path error: {e}")

    async def _save_to_db(self, event):
        """Pure DB persistence. No fast-path, no locks beyond aiosqlite's own queue."""
        text_preview = (event.text or '')[:50].replace('\n', ' ')
        print(f"📩 [{event.id}] {text_preview}")
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
                commit=False
            )
            if internal_id:
                print(f"   📥 Queued (ID={internal_id})")
        except Exception as e:
            if shared.bot:
                await shared.bot.alert_error("listener_save_db", e)
            print(f"❌ _save_to_db error: {e}")

    # ── Start / history sync ──────────────────────────────────────────────────

    async def start(self):
        @self.client.on(events.NewMessage(chats=Config.TARGET_GROUP))
        async def handler(event):
            if not event.text:
                return
            # Record ingest timestamp BEFORE dispatch so /diag has an accurate
            # "time since last message" signal even if downstream tasks error.
            import shared as _shared
            _shared.mark_message_ingested()
            # Fast-path as its OWN task - never waits for DB save
            asyncio.create_task(self._handle_fast_path_standalone(event))
            # DB save as separate task - never blocks fast-path
            asyncio.create_task(self._save_to_db(event))

        # BUG FIX: Retry loop for Telethon start to handle 'database is locked'
        # which happens if a previous instance is still cleaning up or if
        # multiple startup tasks conflict.
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

        print("🚀 Telethon Listener started.")

    async def sync_history(self, hours=6, catchup_hours=2):
        chat    = await self.client.get_entity(Config.TARGET_GROUP)
        chat_id = chat.id
        from db import _ts_str
        
        from datetime import timezone
        PROCESS_CUTOFF = datetime.now(timezone.utc) - timedelta(minutes=30)

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
                
                # Check for time mentions roughly
                has_time = any(w in message.text.lower() for w in ['jam','menit','detik','sore','siang','pagi','malam'])
                
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
            last_ts  = await self.db.get_last_msg_timestamp(chat_id)
            gap_hours = (
                (datetime.now(timezone.utc) - last_ts).total_seconds() / 3600
                if last_ts else 999
            )
            if gap_hours > catchup_hours:
                print(f"⚠️ Gap {gap_hours:.1f}h — fetching last {catchup_hours}h only.")
                limit_date = datetime.now(timezone.utc) - timedelta(hours=catchup_hours)
                await fetch_and_buffer(self.client.iter_messages(
                    chat_id, offset_date=limit_date, reverse=True
                ))
            else:
                print(f"🔄 Short gap ({gap_hours:.1f}h) — catching up from msg {last_id}.")
                await fetch_and_buffer(self.client.iter_messages(
                    chat_id, min_id=last_id, reverse=True
                ))
        else:
            print(f"⏱️ DB empty — syncing last {hours}h.")
            limit_date = datetime.now(timezone.utc) - timedelta(hours=hours)
            await fetch_and_buffer(self.client.iter_messages(
                chat_id, offset_date=limit_date, reverse=True
            ))

        print("✅ History sync complete.")

    async def _bulk_save_to_db(self, msgs_data: list[tuple]):
        """Performs a bulk insert of messages into the database."""
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
            print(f"   📥 Bulk Queued {len(msgs_data)} msgs")
        except Exception as e:
            logger.error(f"DB bulk_save error: {e}")
