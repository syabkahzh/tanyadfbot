"""jobs.py — Background Scheduled Jobs.

Implements all asynchronous periodic tasks including digests, trend analysis, 
spike detection, and database maintenance.
"""

import asyncio
import html
import re
import sys
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Sequence, cast

from db import Database
from processor import GeminiProcessor, PromoExtraction
from bot import TelegramBot
from utils import _esc
import shared
from shared import _make_tg_link, _flush_alert_buffer, _reconnect_listener

logger = logging.getLogger(__name__)

# ── Digest jobs ──────────────────────────────────────────────────────────────

async def brewing_digest_job(bot: TelegramBot) -> None:
    """Sends a mock info message before the actual hourly digest."""
    try:
        await bot.send_plain(
            "🍵 Brewing hourly digest for ya, will be ready 1 minute again mawmaw",
            parse_mode='HTML'
        )
    except Exception as e:
        logger.error(f"brewing_digest_job error: {e}")

async def hourly_digest_job(db: Database, gemini: GeminiProcessor, bot: TelegramBot, WIB: Any) -> None:
    """Generates and sends a summary of promotions from the last hour."""
    logger.info("⏰ [Job] Starting hourly_digest_job...")
    try:
        rows        = await db.get_promos(hours=1)
        if not rows:
            return

        # Simple dedup on summary
        seen = set()
        unique_rows = []
        for r in rows:
            if r['summary'] not in seen:
                unique_rows.append(r)
                seen.add(r['summary'])

        lines = []
        for r in unique_rows:
            brand     = (r['brand'] if r['brand'] 
                         and r['brand'].lower() not in ('unknown', 'sunknown', '') else '❓')
            fast_tag  = " ⚡" if r['via_fastpath'] else ""
            line      = f"• <b>{html.escape(brand)}</b>: {html.escape(r['summary'])}{fast_tag}"
            if r['tg_link']:
                line += f" <a href='{r['tg_link']}'>[→]</a>"
            lines.append(line)

        full_text = "\n".join(lines)
        
        # Check cache to avoid duplicate exact digests
        from shared import _last_hourly_digest
        if full_text == _last_hourly_digest:
            await bot.send_plain(
                shared._last_hourly_digest + "\n\n(<i>cached</i>)", 
                parse_mode='HTML'
            )
        else:
            await bot.send_digest(full_text, "1 Jam Terakhir")
            shared._last_hourly_digest = full_text # Cache it

        logger.info("✅ [Job] Finished hourly_digest_job")
    except Exception as e:
        logger.error(f"hourly_digest_job error: {e}", exc_info=True)
        await bot.alert_error("hourly_digest_job", e)


async def halfhour_digest_job(db: Database, gemini: GeminiProcessor, bot: TelegramBot, WIB: Any) -> None:
    """Sends a smaller digest for the last 30 minutes."""
    logger.info("⏰ [Job] Starting halfhour_digest_job...")
    try:
        rows = await db.get_promos(hours=0.5)
        if not rows:
            return

        lines = []
        for r in rows:
            brand = r['brand'] if r['brand'] and r['brand'].lower() != 'unknown' else '❓'
            line  = f"• <b>{html.escape(brand)}</b>: {html.escape(r['summary'])}"
            if r['tg_link']:
                line += f" <a href='{r['tg_link']}'>[→]</a>"
            lines.append(line)

        await bot.send_digest("\n".join(lines), "30 Menit Terakhir")
        logger.info("✅ [Job] Finished halfhour_digest_job")
    except Exception as e:
        logger.error(f"halfhour_digest_job error: {e}", exc_info=True)
        await bot.alert_error("halfhour_digest_job", e)


async def midnight_digest_job(db: Database, gemini: GeminiProcessor, bot: TelegramBot) -> None:
    """Sends a consolidated summary of the previous day's best deals."""
    logger.info("⏰ [Job] Starting midnight_digest_job...")
    try:
        # Get last 18 hours of promos
        rows = await db.get_promos(hours=18)
        if not rows:
            return

        lines = []
        # Group by brand
        grouped: dict[str, list[str]] = {}
        for r in rows:
            b = normalize_brand(r['brand'])
            if b not in grouped: grouped[b] = []
            if r['summary'] not in grouped[b]:
                grouped[b].append(r['summary'])

        for brand, summaries in grouped.items():
            lines.append(f"🏪 <b>{html.escape(brand)}</b>")
            for s in summaries[:3]: # top 3 per brand
                lines.append(f"  • {html.escape(s)}")
            lines.append("")

        header = "🌅 <b>REKAP PROMO SEMALAM</b>\n\n"
        await bot.send_plain(header + "\n".join(lines), parse_mode='HTML')
        logger.info("✅ [Job] Finished midnight_digest_job")
    except Exception as e:
        logger.error(f"midnight_digest_job error: {e}", exc_info=True)
        await bot.alert_error("midnight_digest_job", e)


# ── Utility jobs ─────────────────────────────────────────────────────────────

async def heartbeat_job(db: Database, gemini: GeminiProcessor, bot: TelegramBot, WIB: Any) -> None:
    """Periodically reports system health and AI quota status."""
    logger.info("⏰ [Job] Starting heartbeat_job...")
    try:
        # Collect model stats
        stats_lines = ["🤖 <b>Gemini Quota Status:</b>"]
        for mid, slot in gemini._model_stats.items():
            rpm_used  = slot.current_usage()
            rpm_limit = slot.limit
            rpd_used  = slot.daily_usage()
            # Calculate safety percent
            usage_pct = (rpm_used / rpm_limit) * 100 if rpm_limit > 0 else 0
            status_emoji = "🟢" if usage_pct < 70 else "🟡" if usage_pct < 90 else "🔴"
            
            stats_lines.append(
                f"{status_emoji} <code>{mid}</code>\n"
                f"   • RPM: <code>{rpm_used}/{rpm_limit}</code>\n"
                f"   • RPD: <code>{rpd_used}</code>"
            )
        
        # Get queue depth
        q_size = await db.get_queue_size()
        stats_lines.append(f"\n📥 Queue Depth: <code>{q_size}</code> messages")
        
        await bot.send_plain("\n".join(stats_lines), parse_mode='HTML')
        logger.info("✅ [Job] Finished heartbeat_job")
    except Exception as e:
        logger.error(f"heartbeat_job error: {e}", exc_info=True)
        await bot.alert_error("heartbeat_job", e)


async def image_processing_job(db: Database, gemini: GeminiProcessor, listener: Any) -> None:
    """Processes messages with images that haven't been analyzed yet."""
    logger.info("⏰ [Job] Starting image_processing_job...")
    try:
        if not db.conn:
            return

        async with db.conn.execute(
            "SELECT id, tg_msg_id, chat_id, text, timestamp FROM messages "
            "WHERE has_photo=1 AND image_processed=0 ORDER BY id ASC LIMIT 20"
        ) as cur:
            rows = await cur.fetchall()

        if not rows:
            return
        
        # Process at most 5 per cycle to keep it fast
        rows = rows[:5]
        processed_ids = []

        for r in rows:
            msg_id, tg_msg_id, chat_id, text, ts = (
                r['id'], r['tg_msg_id'], r['chat_id'], r['text'], r['timestamp']
            )
            try:
                # Need to fetch message from Telethon to get photo
                msg_obj = await listener.client.get_messages(chat_id, ids=tg_msg_id)
                if not msg_obj or not msg_obj.photo:
                    processed_ids.append(msg_id)
                    continue

                # Download media
                downloaded = await listener.client.download_media(msg_obj.photo, bytes)
                if not downloaded:
                    photo_bytes = None
                elif isinstance(downloaded, str):
                    import os
                    if os.path.exists(downloaded):
                        with open(downloaded, 'rb') as f:
                            photo_bytes = f.read()
                        os.remove(downloaded)
                    else:
                        photo_bytes = None
                else:
                    photo_bytes = downloaded

                if not photo_bytes:
                    processed_ids.append(msg_id)
                    continue

                promo = await gemini.process_image(photo_bytes, text or "", msg_id)

                if promo:
                    # Correct common mis-attribution where the vision model
                    # latches onto a cross-promo PAYMENT banner (e.g.
                    # "Cashback Saldo ShopeePay") instead of the MERCHANT
                    # (the store where the deal redeems).
                    caption_l = (text or "").lower()
                    PAY_BRANDS = {
                        'shopeepay', 'spay', 'gopay', 'gpy', 'dana',
                        'ovo', 'astrapay', 'aspay', 'linkaja', 'qris'
                    }
                    if promo.brand and promo.brand.lower().strip() in PAY_BRANDS:
                        if re.search(r'\b(jsm|psm)\b', caption_l):
                            promo.brand = 'Alfamart'
                        elif re.search(r'\bafm\b', caption_l):
                            promo.brand = 'Alfamart'
                        elif re.search(r'\bidm\b', caption_l):
                            promo.brand = 'Indomaret'

                    tg_link  = _make_tg_link(chat_id, tg_msg_id)
                    now_utc  = datetime.now(timezone.utc)
                    if (now_utc - shared._parse_ts(ts)).total_seconds() < 5400:
                        await db.save_pending_alert(
                            promo.brand.lower().strip(),
                            promo.model_dump_json(), tg_link, ts,
                            source='ai', commit=False
                        )
                        t = shared.get_buffer_flush_task()
                        if t is None or t.done():
                            shared.set_buffer_flush_task(
                                asyncio.create_task(_flush_alert_buffer(delay=0.5))
                            )

                processed_ids.append(msg_id)
            except Exception as e:
                logger.error(f"image_processing_job item (msg {tg_msg_id}) error: {e}")
        
        if processed_ids:
            ph = ','.join('?' * len(processed_ids))
            await db.conn.execute(
                f"UPDATE messages SET image_processed=1 WHERE id IN ({ph})", processed_ids
            )
            await db.conn.commit()

        logger.info("✅ [Job] Finished image_processing_job")
    except Exception as e:
        logger.error(f"image_processing_job critical error: {e}", exc_info=True)
        if shared.bot:
            await shared.bot.alert_error("image_processing_job", e)


async def hot_thread_job(db: Database, gemini: GeminiProcessor, listener: Any, bot: TelegramBot, WIB: Any, alerted_map: dict) -> None:
    """Identifies and summarizes active discussion threads."""
    logger.info("⏰ [Job] Starting hot_thread_job...")
    try:
        # Get threads with 5+ unique senders in last 15m
        threads = await db.get_hot_threads(minutes=15, min_replies=5, limit=5)
        if not threads:
            return

        now = datetime.now(timezone.utc)
        for t in threads:
            # Skip if alerted in last 1 hour
            tid = t['id']
            if tid in alerted_map:
                last_time = alerted_map[tid][1]
                if (now - last_time).total_seconds() < 3600:
                    continue

            # Summarize thread
            replies = await db.get_thread_replies(t['tg_msg_id'], t['chat_id'], limit=20)
            reply_texts = [r['text'] for r in replies if r['text']]
            
            parent_photo = None
            if t['has_photo']:
                # Optional: download parent photo for vision summary
                pass

            summary = await gemini.summarize_thread(t['text'] or "(tanpa teks)", reply_texts, parent_photo)
            
            if summary:
                alerted_map[tid] = (t['reply_count'], now)
                msg_wib = _to_wib(t['timestamp'], WIB)
                link = _make_tg_link(t['chat_id'], t['tg_msg_id'])
                
                snippets = []
                for r in replies[:4]:
                    snip = (r['text'] or '').strip()
                    if snip:
                        snippets.append(f"  • <i>\"{html.escape(snip[:60])}\"</i>")

                from telegram.constants import ParseMode
                text = (
                    f"🔥 <b>Thread Hot! ({t['reply_count']} balasan)</b>\n"
                    f"⏰ Jam: <code>{msg_wib}</code>\n"
                    f"📌 <b>Pesan Awal:</b>\n<i>{html.escape((t['text'] or '')[:100])}</i>\n\n"
                    f"💬 <b>Topik:</b>\n{html.escape(summary)}\n\n"
                    f"📜 <b>Cuplikan:</b>\n" + "\n".join(snippets) + "\n\n"
                    f"🔗 <a href='{link}'>Lihat Thread</a>"
                )
                await bot.send_plain(text, parse_mode=ParseMode.HTML)
        logger.info("✅ [Job] Finished hot_thread_job")
    except Exception as e:
        logger.error(f"hot_thread_job error: {e}", exc_info=True)
        await bot.alert_error("hot_thread_job", e)


def _to_wib(ts: str | datetime, WIB: Any) -> str:
    """Helper for job timestamps."""
    try:
        if isinstance(ts, str):
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        else:
            dt = ts
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(WIB).strftime('%H:%M')
    except:
        return "??"


# ─────────────────────────────────────────────────────────────────────────────
# Time mentions
# ─────────────────────────────────────────────────────────────────────────────

# Patterns for filtering "Sinyal Waktu" false positives.
# Questions, speculation, retrospective, and OOT chatter should NOT fire alerts.
_TIME_QUESTION_PATTERN = re.compile(
    r'(\?'                                # ends/contains question mark
    r'|^(apa|gimana|berapa|kapan|dimana|kenapa|mana|siapa|kok)\b'  # starts with question word
    r'|\b(brp|jam brp|jam berapa|kah|ya\??)\s*$'  # trailing question markers
    r'|\b(apa aja|apa saja|yg apa|yang apa|ada apa|war apa)\b'  # embedded question phrases
    r'|\b(dong|dongg)\b'  # request/plea markers
    r')',
    re.IGNORECASE,
)
_TIME_SPECULATION_PATTERN = re.compile(
    r'\b(kyny|kayaknya|kyknya|sepertinya|mungkin|entah|semoga|mudah.mudahan|'
    r'klo gk slh|kalo ga salah|deh$|gatau|gak tau|ga tau|blm tau)\b',
    re.IGNORECASE,
)
_TIME_RETROSPECTIVE_PATTERN = re.compile(
    r'\b(tadi|kemarin|kemaren|semalam|tdi|pernah|waktu itu|dulu|barusan|'
    r'td pagi|tadi pagi|tadi malem|kmrn|kmrin)\b',
    re.IGNORECASE,
)
_TIME_COMPLAINT_FILLER = re.compile(
    r'\b(males|lag mulu|lelet|lemot|kesel|sebel|nyesel|nyesek)\b|😭{2,}',
    re.IGNORECASE,
)
# Actionable promo signals — stronger than generic _PROMO. A time-mention
# should only fire if it indicates a concrete upcoming or active deal event.
_TIME_STRONG_SIGNAL = re.compile(
    r'\b(promo|diskon|cashback|voucher|gratis|flash|sale|deal|potongan|hemat|'
    r'minbel|restock|ristok|limit|kuota|slot|claim|klaim|redeem|'
    r'voc|vcr|cb|kesbek|cash back|ongkir|bonus)\b',
    re.IGNORECASE,
)
_TIME_ACTIVE_STATUS = re.compile(
    r'\b(on|aman|nyala|cair|masuk|berhasil|lancar|jp|work|luber|pecah|gacor)\b',
    re.IGNORECASE,
)
_TIME_BRAND = re.compile(
    r'\b(sfood|gfood|shopee|tokped|tokopedia|grab|gojek|alfamart|alfa|idm|indomaret|'
    r'cgv|xxi|tsel|telkomsel|sopi|tukpo|gopay|spay|ovo|dana|'
    r'solaria|kopken|chatime|gindaco|rotio|hero|tmrw|neo|seabank|saqu|'
    r'shopee ?food|grab ?food|alfagift|alfamidi|hokben|mcd|kfc|jco)\b',
    re.IGNORECASE,
)


def _is_time_signal_worthy(text: str) -> bool:
    """Determine if a time-mentioning message deserves a Sinyal Waktu alert.

    A worthy time signal is an *informational statement* about a future/current
    promo event at a specific time. Reject:
    - Questions about timing ("war apa jam 12?", "jam brp ya?")
    - Speculation ("tengah malem on deh klo gk slh")
    - Retrospective accounts ("tadi pagi", "kemarin", "semalam")
    - Complaints/venting ("males bgt")
    - Messages without both a brand AND an action/promo signal
    """
    if not text or not text.strip():
        return False

    t = text.strip()
    tl = t.lower()

    # ── Gate 1: reject questions ──
    if _TIME_QUESTION_PATTERN.search(t):
        return False

    # ── Gate 2: reject speculation / uncertainty ──
    if _TIME_SPECULATION_PATTERN.search(tl):
        return False

    # ── Gate 3: reject retrospective / past-tense ──
    if _TIME_RETROSPECTIVE_PATTERN.search(tl):
        return False

    # ── Gate 4: reject complaints / emotional filler ──
    if _TIME_COMPLAINT_FILLER.search(tl):
        return False

    # ── Gate 5: require BOTH a brand mention AND a promo/status signal ──
    has_brand = bool(_TIME_BRAND.search(tl))
    has_promo_signal = bool(_TIME_STRONG_SIGNAL.search(tl))
    has_active_status = bool(_TIME_ACTIVE_STATUS.search(tl))

    # Must have a brand AND at least one deal/status signal
    if not has_brand:
        return False
    if not (has_promo_signal or has_active_status):
        return False

    return True


async def time_mention_job(db: Database, bot: TelegramBot) -> None:
    """Monitors and alerts on relevant time-sensitive messages."""
    logger.info("⏰ [Job] Starting time_mention_job...")
    try:
        if not db.conn:
            return

        async with db.conn.execute(
            "SELECT id, text, timestamp, chat_id, tg_msg_id FROM messages "
            "WHERE has_time_mention=1 AND time_alerted=0 ORDER BY id ASC"
        ) as cur:
            rows = await cur.fetchall()
        if not rows:
            return

        from telegram.constants import ParseMode

        all_ids = []
        noise_count = 0
        for r in rows:
            text  = r['text'] or ""
            
            # ALL IDs in the batch will be marked as "processed" by this job
            # to avoid infinite scanning of the same old messages.
            all_ids.append(r['id'])

            if not _is_time_signal_worthy(text):
                noise_count += 1
                continue

            link  = _make_tg_link(r['chat_id'], r['tg_msg_id'])
            alert = (
                f"🕒 <b>Sinyal Waktu:</b>\n{_esc(text)}\n\n"
                f"🔗 <a href='{link}'>Lihat Pesan</a>"
            )
            await bot.send_plain(alert, parse_mode=ParseMode.HTML)

        if all_ids:
            # Chunk updates to avoid SQLite variable limits (standard is 999)
            chunk_size = 900
            for i in range(0, len(all_ids), chunk_size):
                chunk = all_ids[i:i + chunk_size]
                ph = ','.join('?' * len(chunk))
                await db.conn.execute(
                    f"UPDATE messages SET time_alerted=1 WHERE id IN ({ph})", chunk
                )
            await db.conn.commit()

            if noise_count > 0:
                logger.info(f"⏰ [Job] Marked {noise_count} noise time-mentions as alerted.")

        logger.info("✅ [Job] Finished time_mention_job")
    except Exception as e:
        logger.error(f"time_mention_job error: {e}", exc_info=True)
        await bot.alert_error("time_mention_job", e)


# ── Trend detection ──────────────────────────────────────────────────────────

async def trend_job(db: Database, gemini: GeminiProcessor, bot: TelegramBot) -> None:
    """Analyzes recent traffic to identify and describe discussion trends."""
    logger.info("⏰ [Job] Starting trend_job...")
    try:
        # Get last 15 mins of activity
        msgs = await db.get_recent_messages(minutes=15)
        if len(msgs) < 20:
            return
        
        try:
            async with asyncio.timeout(60):
                trends = await gemini.generate_narrative(msgs, db=db)
        except TimeoutError:
            logger.warning("AI timeout in trend_job. Skipping.")
            return
        
        if not trends:
            return

        # Use combined topics as a simple string for the dedup check
        current_summary = " ".join([t.topic for t in trends])
        from shared import _last_trend_alert
        if current_summary == _last_trend_alert:
            return
        shared._last_trend_alert = current_summary

        lines = []
        for t in trends:
            link = _make_tg_link(Config.TARGET_GROUP, t.msg_id)
            lines.append(f"• {html.escape(t.topic)}\n  🔗 <a href='{link}'>Lihat Pesan</a>")

        full_text = (
            "📈 <b>Narasi Tren (15m):</b>\n\n"
            + "\n\n".join(lines)
        )
        
        await bot.send_plain(full_text, parse_mode='HTML')
        logger.info("✅ [Job] Finished trend_job")
    except Exception as e:
        logger.error(f"trend_job error: {e}", exc_info=True)
        await bot.alert_error("trend_job", e)


async def spike_detection_job(db: Database, gemini: GeminiProcessor, bot: TelegramBot) -> None:
    """Detects and provides context for sudden traffic bursts."""
    logger.info("⏰ [Job] Starting spike_detection_job...")
    try:
        now_utc = datetime.now(timezone.utc)
        from shared import _last_spike_alert
        
        # 5m cooldown
        if (now_utc - _last_spike_alert).total_seconds() < 300:
            return
            
        if not db.conn:
            return

        # 1. Measure current speed (last 1 min)
        one_min_ago = (now_utc - timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M:%S+00:00')
        async with db.conn.execute(
            "SELECT COUNT(*), MAX(id) FROM messages WHERE timestamp >= ?", (one_min_ago,)
        ) as cur:
            row = await cur.fetchone()
            count = cast(int, row[0]) if row else 0

        # 2. Measure baseline (last 5 min)
        five_min_ago = (now_utc - timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S+00:00')
        async with db.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE timestamp >= ?", (five_min_ago,)
        ) as cur:
            row_5 = await cur.fetchone()
            five_min_count = cast(int, row_5[0]) if row_5 else 0
        avg_per_min = five_min_count / 5

        # 3. Check for Spike (thresholds: 15 msg/min AND 2x baseline)
        if count >= 15 and count >= max(avg_per_min * 2.0, 5):
            # Fetch the actual messages for analysis
            recent_msgs = await db.get_recent_messages(minutes=2)
            if not recent_msgs:
                return
            
            # Fetch hot words from these SPECIFIC messages
            top_words = await db.get_recent_words(minutes=2)
            hot_words = [w[0] for w in top_words[:8] if len(w[0]) > 2]
            
            try:
                async with asyncio.timeout(30):
                    narrative = await gemini.interpret_keywords(
                        hot_words[:5], 2, [cast(str, m['text']) for m in recent_msgs if m['text']]
                    )
            except TimeoutError:
                logger.warning("AI timeout in spike_detection_job. Skipping narrative.")
                narrative = "Aktivitas meningkat tajam."

            # 4. Smart Snippets: Show messages that contain top words
            sample_lines = []
            seen_texts = set()
            for msg in recent_msgs:
                text = (msg['text'] or '').strip()
                if not text or text in seen_texts: continue
                
                # Check if this msg contains any of our hot words
                if any(hw.lower() in text.lower() for hw in hot_words[:3]):
                    clean = text.replace('\n', ' ')
                    sample_lines.append(f"• {html.escape(clean[:100])}")
                    seen_texts.add(text)
                
                if len(sample_lines) >= 5: break
            
            # Fallback to random if no keyword match
            if not sample_lines:
                sample_lines = [f"• {html.escape((m['text'] or '').strip()[:100])}" for m in recent_msgs[:5]]

            from telegram.constants import ParseMode
            header_lines = [
                "🚀 <b>Lonjakan Pesan!</b>",
                f"📊 Kecepatan: <code>{count} msg/min</code>",
                f"📈 Rata-rata: <code>{avg_per_min:.1f} msg/min</code>",
            ]
            
            text = (
                "\n".join(header_lines) + "\n\n"
                f"🏷 <b>Top Kata:</b> <code>{', '.join(hot_words[:5])}</code>\n"
                f"🤖 <b>Analisis:</b>\n{html.escape(narrative or 'Aktivitas meningkat tajam.')}\n\n"
                f"<b>Cuplikan Terkait:</b>\n" + "\n".join(sample_lines)
            )
            await bot.send_plain(text, parse_mode='HTML')
            shared._last_spike_alert = now_utc
            
        logger.info(f"✅ [Job] Finished spike_detection_job (Count: {count})")
    except Exception as e:
        logger.error(f"spike_detection_job error: {e}", exc_info=True)
        await bot.alert_error("spike_detection_job", e)


# ── Database maintenance ─────────────────────────────────────────────────────

async def dead_promo_reaper_job(db: Database, bot: TelegramBot) -> None:
    """Closes expired promotions based on subsequent community chat signals."""
    logger.info("⏰ [Job] Starting dead_promo_reaper_job...")
    try:
        # Get active promos from last 6 hours
        async with db.conn.execute("""
            SELECT p.id, p.brand, p.summary, m.tg_msg_id, m.chat_id
            FROM promos p
            JOIN messages m ON p.source_msg_id = m.id
            WHERE p.status = 'active'
              AND p.created_at >= strftime('%Y-%m-%d %H:%M:%S+00:00','now','-6 hours')
        """) as cur:
            active_promos = await cur.fetchall()

        if not active_promos:
            return

        reaped_count = 0
        EXPIRY_SIGNALS = re.compile(
            r'\b(nt|abis|habis|sold.?out|expired|kehabisan|ga bisa|gabisa|'
            r'udah mati|mati|nonaktif|hangus|error terus|ga work|gak work|off)\b',
            re.IGNORECASE
        )

        for p in active_promos:
            # Check messages *after* this promo for negative signals about this brand
            async with db.conn.execute("""
                SELECT text FROM messages
                WHERE id > ? AND (text LIKE ? OR text LIKE ?)
                ORDER BY id DESC LIMIT 10
            """, (p['id'], f"%{p['brand']}%", "%abis%")) as cur:
                recent_mentions = await cur.fetchall()
            
            signals = 0
            for m in recent_mentions:
                if m[0] and EXPIRY_SIGNALS.search(m[0]):
                    signals += 1
            
            if signals >= 3:
                # Mark as expired
                await db.conn.execute(
                    "UPDATE promos SET status='expired' WHERE id=?", (p['id'],)
                )
                reaped_count += 1
                logger.info(f"💀 [Reaper] Expired promo for {p['brand']} due to community signals.")

        if reaped_count > 0:
            await db.conn.commit()
            
        logger.info(f"✅ [Job] Finished dead_promo_reaper_job (Reaped: {reaped_count})")
    except Exception as e:
        logger.error(f"dead_promo_reaper_job error: {e}", exc_info=True)


async def confirmation_gate_job(db: Database) -> None:
    """Promotes pending confirmations to alerts once threshold is reached."""
    logger.info("⏰ [Job] Starting confirmation_gate_job...")
    try:
        now_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        
        async with db.conn.execute(
            "SELECT * FROM pending_confirmations WHERE expires_at > ? AND corroborations >= 1",
            (now_utc,)
        ) as cur:
            ready = await cur.fetchall()

        for r in ready:
            # Move to pending_alerts
            await db.save_pending_alert(
                r['brand'], r['p_data_json'], r['tg_link'], r['timestamp'],
                corroborations=r['corroborations'],
                corroboration_texts=r['corroboration_texts'],
                source='ai', commit=False
            )
            # Delete from confirmations
            await db.conn.execute("DELETE FROM pending_confirmations WHERE id=?", (r['id'],))
        
        # Also cleanup truly expired ones
        await db.conn.execute("DELETE FROM pending_confirmations WHERE expires_at <= ?", (now_utc,))
        
        await db.conn.commit()
        
        if ready:
            # Trigger flush
            t = shared.get_buffer_flush_task()
            if t is None or t.done():
                shared.set_buffer_flush_task(
                    asyncio.create_task(shared._flush_alert_buffer(delay=0.5))
                )
            logger.info(f"🚪 [Gate] Promoted {len(ready)} alerts via corroboration.")

        logger.info("✅ [Job] Finished confirmation_gate_job")
    except Exception as e:
        logger.error(f"confirmation_gate_job error: {e}")


async def db_maintenance_job(db: Database, bot: TelegramBot) -> None:
    """Performs routine database cleanup and health checks."""
    logger.info("⏰ [Job] Starting db_maintenance_job...")
    try:
        await db.prune_old_messages()
        logger.info("✅ [Job] Finished db_maintenance_job")
    except Exception as e:
        logger.error(f"db_maintenance_job error: {e}", exc_info=True)
        await bot.alert_error("db_maintenance_job", e)
