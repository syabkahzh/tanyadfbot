"""jobs.py — Background Scheduled Jobs.

Implements all asynchronous periodic tasks including digests, trend analysis, 
spike detection, and database maintenance.
"""

import asyncio
import html
import re
import pytz
import logging
import time
import os
from datetime import datetime, timezone, timedelta
from typing import Any, cast

from db import Database
from processor import GeminiProcessor
from bot import TelegramBot
from utils import _esc
import shared
from shared import _make_tg_link, _flush_alert_buffer, _reconnect_listener

logger = logging.getLogger(__name__)


def _read_file_bytes(path: str) -> bytes:
    """Synchronous helper to read a file — intended for asyncio.to_thread."""
    with open(path, 'rb') as f:
        return f.read()


# ─────────────────────────────────────────────────────────────────────────────
# Time reminder job — "sinyal waktu"
# Scans active promos for an HH:MM time of day in valid_until/summary/conditions
# and fires a T-2min reminder so users don't miss a time-bounded promo window.
# ─────────────────────────────────────────────────────────────────────────────

# HH:MM / HH.MM / HH:MM WIB — anchored to word boundaries.
_TIME_OF_DAY_RE = re.compile(
    r'\b(?:jam|pukul|pkl|pk|s[./]?d|sampai|sebelum|pada)?\s*'
    r'(\d{1,2})[:.](\d{2})\s*(?:wib|wit|wita)?\b',
    re.IGNORECASE,
)
# "jam 10" / "pukul 14" without minutes (interpreted as :00).
_TIME_HOUR_RE = re.compile(
    r'\b(?:jam|pukul|pkl|pk|s[./]?d|sampai|sebelum|pukul|pada)\s+(\d{1,2})(?:\s*(wib|wit|wita|pagi|siang|sore|malem|malam))?\b',
    re.IGNORECASE,
)


def _extract_time_of_day(text: str) -> tuple[int, int] | None:
    """Return (hour, minute) if text explicitly names a time of day.

    Improved to handle more variations and common typos.
    """
    if not text:
        return None
    # Pre-clean: "jam 10.00wib" -> "jam 10.00 wib"
    text = re.sub(r'(\d{2})(wib|wit|wita)', r'\1 \2', text, flags=re.IGNORECASE)
    
    m = _TIME_OF_DAY_RE.search(text)
    if m:
        hh, mm = int(m.group(1)), int(m.group(2))
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return (hh, mm)
            
    m = _TIME_HOUR_RE.search(text)
    if m:
        hh = int(m.group(1))
        if 0 <= hh <= 23:
            tod = (m.group(2) or '').lower()
            # "jam 3 sore" → 15:00, "jam 7 malam" → 19:00
            if tod in ('sore',) and hh < 12:
                hh += 12
            elif tod in ('malem', 'malam') and hh < 12:
                hh += 12
            elif tod in ('pagi',) and hh == 12:
                hh = 0
            return (hh, 0)
    
    # Special case: "tengah malam" -> 00:00
    if "tengah malam" in text.lower() or "jam 12 malam" in text.lower():
        return (0, 0)
    if "nanti siang" in text.lower() or "jam 12 siang" in text.lower():
        return (12, 0)
        
    return None


async def time_reminder_job(db: Database, bot: TelegramBot, WIB: Any) -> None:
    """Fire T-2min reminders for active promos with an explicit time-of-day.

    This is the "sinyal waktu" feature: for any active promo whose
    summary/conditions/valid_until mentions an explicit HH:MM WIB,
    schedule one reminder ~2 minutes before the stated time. Each promo
    fires at most one reminder (tracked via the `reminder_fired` column).
    """
    logger.info("⏰ [Job] Starting time_reminder_job...")
    try:
        import pytz
        if not db.conn:
            return

        # Ensure the idempotency column exists. Cheap no-op if already there.
        try:
            await db.conn.execute(
                "ALTER TABLE promos ADD COLUMN reminder_fired INTEGER DEFAULT 0"
            )
            await db.conn.commit()
        except Exception:
            pass   # column already exists

        now_wib = datetime.now(WIB)

        # Only look at active, recent promos to keep this fast.
        # Join with messages to get reply context.
        async with db.conn.execute("""
            SELECT p.id, p.brand, p.summary, p.conditions, p.valid_until, p.tg_link,
                   m.reply_to_msg_id, m.chat_id,
                   (SELECT text FROM messages WHERE tg_msg_id = m.reply_to_msg_id AND chat_id = m.chat_id LIMIT 1) as parent_text
            FROM promos p
            JOIN messages m ON p.source_msg_id = m.id
            WHERE p.status='active' AND p.reminder_fired=0 
              AND p.created_at >= strftime('%Y-%m-%d %H:%M:%S+00:00','now','-24 hours') 
            ORDER BY p.id DESC LIMIT 200
        """) as cur:
            rows = await cur.fetchall()

        if not rows:
            logger.info("✅ [Job] time_reminder_job: no candidates")
            return

        # Pre-classify summaries in batch to weigh them with FastText
        summaries = [r['summary'] for r in rows if r['summary']]
        ft_results = {}
        if summaries:
            classifications = await shared.classify_batch(summaries)
            ft_results = {summaries[i]: classifications[i] for i in range(len(summaries))}

        fired_ids = []
        from telegram.constants import ParseMode

        for r in rows:
            # 1. FastText Weighting: Skip if it's very likely JUNK (> 0.90 confidence)
            ft_label, ft_conf = ft_results.get(r['summary'], ("__label__PROMO", 0.0))
            if ft_label == "__label__JUNK" and ft_conf > 0.90:
                # Mark as fired so we don't re-scan noise.
                fired_ids.append(r['id'])
                continue

            # Prefer valid_until if it contains a time; otherwise scan the
            # summary and conditions for an explicit HH:MM.
            haystack = " ".join(
                x for x in (r['valid_until'], r['summary'], r['conditions']) if x
            )
            tod = _extract_time_of_day(haystack)
            if tod is None:
                # Mark as checked-no-time so we don't re-scan next minute.
                fired_ids.append(r['id'])
                continue

            hh, mm = tod
            target = now_wib.replace(hour=hh, minute=mm, second=0, microsecond=0)
            
            # If target already passed today, and it was more than 12 hours ago,
            # it's likely a stale mention or a time meant for tomorrow.
            # We only want to fire for upcoming times in the next ~12 hours.
            if target <= now_wib:
                if (now_wib - target).total_seconds() > 43200: # 12h
                    target = target + timedelta(days=1)
                else:
                    # Time has very recently passed, skip to prevent late alerts
                    fired_ids.append(r['id'])
                    continue

            minutes_to = (target - now_wib).total_seconds() / 60.0

            # Fire when the stated time is between 1 and 6 minutes away
            if 1.0 <= minutes_to <= 6.0:
                now_str = now_wib.strftime('%H:%M:%S')
                
                # Format reply context if available
                context_header = ""
                if r['parent_text']:
                    p_text = r['parent_text'].replace('\n', ' ')[:100]
                    context_header = f"💬 _Balasan untuk: \"{p_text}...\"_\n\n"

                text = (
                    f"⏰ **Sinyal Waktu — 2 menit lagi!**\n"
                    f"⏰ Skrg: `{now_str}`\n"
                    f"🏪 **{r['brand']}**\n"
                    f"🕒 Target: `{target.strftime('%H:%M:%S WIB')}`\n\n"
                    f"{context_header}"
                    f"📝 {r['summary']}\n"
                )
                
                if r['conditions']:
                    text += f"ℹ️ _{r['conditions']}_\n"
                
                # Show FastText status for transparency
                text += f"\n🛡️ FT: `{ft_label.replace('__label__', '')} ({ft_conf:.2f})`"
                
                if r['tg_link']:
                    text += f"\n\n🔗 Lihat Promo:\n{r['tg_link']}"

                try:
                    await bot.send_plain(text)
                    fired_ids.append(r['id'])
                except Exception as e:
                    logger.error(f"time_reminder send failed for promo {r['id']}: {e}")
                    continue

        if fired_ids:
            ph = ','.join(['?'] * len(fired_ids))
            await db.conn.execute(f"UPDATE promos SET reminder_fired=1 WHERE id IN ({ph})", fired_ids)
            await db.conn.commit()
            logger.info(f"✅ [Job] time_reminder_job fired/marked {len(fired_ids)} promos")
        else:
            logger.info("✅ [Job] time_reminder_job: no promos within 2-3 min window")
    except Exception as e:
        logger.error(f"time_reminder_job error: {e}", exc_info=True)
        try:
            await bot.alert_error("time_reminder_job", e)
        except Exception:
            pass


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
        now_wib     = datetime.now(WIB)
        hour_label  = now_wib.strftime('%H:00 WIB')

        if not rows:
            await bot.send_plain(
                f"📊 **Rekap Promo {hour_label}**\n\n"
                "😴 _Tidak ada promo yang terdeteksi 1 jam terakhir._"
            )
            return

        # Group by brand
        by_brand = {}
        for r in rows:
            brand = (r['brand'] if r['brand'] and r['brand'].lower() not in ('unknown', 'sunknown', '') else '❓')
            by_brand.setdefault(brand, []).append(r)

        # Build output structure
        body_lines = []
        for brand, promos in sorted(by_brand.items()):
            body_lines.append(f"**{brand}**")
            for r in promos:
                status_icon = ("🟢" if r['status'] == 'active' else ("🔴" if r['status'] == 'expired' else "⚪"))
                fast_tag    = " ⚡" if r['via_fastpath'] else ""
                link_part   = f" [🔗 Lihat Pesan]({r['tg_link']})" if r['tg_link'] else ""
                body_lines.append(f"  {status_icon} {r['summary']}{fast_tag}{link_part}")

        context    = "\n".join([f"- {r['brand']}: {r['summary']}" for r in rows])
        try:
            async with asyncio.timeout(60):
                ai_summary = await gemini.answer_question(
                    f"Rangkum promo berikut dalam 2-3 kalimat santai. Jam: {hour_label}.", context
                )
        except TimeoutError:
            logger.warning("AI timeout in hourly_digest_job. Skipping summary.")
            ai_summary = "_Summary unavailable._"

        now_str = datetime.now(WIB).strftime('%H:%M:%S')
        full_text = (
            f"📊 **Rekap Promo {hour_label}**\n"
            f"🕒 _({len(rows)} promo)_ · ⏱ `{now_str}`\n\n"
            f"{ai_summary}\n\n"
            f"**Detail:**\n" + "\n".join(body_lines)
        )
        shared._last_hourly_digest = full_text
        await bot.send_plain(full_text)
        logger.info("✅ [Job] Finished hourly_digest_job")
    except Exception as e:
        logger.error(f"hourly_digest_job error: {e}", exc_info=True)
        await bot.alert_error("hourly_digest_job", e)


async def midnight_digest_job(db: Database, gemini: GeminiProcessor, bot: TelegramBot) -> None:
    """Generates a re-cap of overnight activity (02:00–05:00 WIB)."""
    logger.info("⏰ [Job] Starting midnight_digest_job...")
    try:
        import pytz
        jakarta_tz = pytz.timezone("Asia/Jakarta")
        now_wib = datetime.now(jakarta_tz)
        # Anchor: today at 02:00 WIB
        since_wib = now_wib.replace(hour=2, minute=0, second=0, microsecond=0)
        rows      = await db.get_promos(since_dt=since_wib)

        if not rows:
            await bot.send_plain(
                "📊 **Rekap Promo 02:00–05:00 WIB**\n\n"
                "😴 _Tidak ada promo yang masuk tadi malam._"
            )
            return

        # Group by brand
        by_brand = {}
        for r in rows:
            brand = (r['brand'] if r['brand'] and r['brand'].lower() not in ('unknown', 'sunknown', '') else '❓')
            by_brand.setdefault(brand, []).append(r)

        # Build output structure
        body_lines = []
        for brand, promos in sorted(by_brand.items()):
            body_lines.append(f"**{brand}**")
            for r in promos:
                fast_tag    = " ⚡" if r['via_fastpath'] else ""
                link_part   = f" [🔗 Lihat Pesan]({r['tg_link']})" if r['tg_link'] else ""
                body_lines.append(f"  • {r['summary']}{fast_tag}{link_part}")

        context = "\n".join([f"- {r['brand']}: {r['summary']}" for r in rows])
        try:
            async with asyncio.timeout(60):
                digest  = await gemini.answer_question(
                    "Rangkum promo yang masuk jam 02:00–05:00 WIB tadi malam. Singkat, padat, santai.",
                    context
                )
        except TimeoutError:
            logger.warning("AI timeout in midnight_digest_job. Skipping summary.")
            digest = "_Summary unavailable._"

        now_str = now_wib.strftime('%H:%M:%S')
        full_text = (
            f"📊 **Rekap Promo 02:00–05:00 WIB**\n"
            f"🕒 _({len(rows)} promo)_ · ⏱ `{now_str}`\n\n"
            f"{digest}\n\n"
            f"**Detail:**\n" + "\n".join(body_lines)
        )
        await bot.send_plain(full_text)
        logger.info("✅ [Job] Finished midnight_digest_job")
    except Exception as e:
        logger.error(f"midnight_digest_job error: {e}", exc_info=True)
        await bot.alert_error("midnight_digest_job", e)


async def halfhour_digest_job(db: Database, gemini: GeminiProcessor, bot: TelegramBot, WIB: Any) -> None:
    """Generates a quick status update every 30 minutes during active hours."""
    logger.info("⏰ [Job] Starting halfhour_digest_job...")
    try:
        now_wib = datetime.now(WIB)
        hour    = now_wib.hour
        # Skip 02:00–05:00 WIB — midnight_digest_job covers this window
        if 2 <= hour < 5:
            return
        rows = await db.get_promos(hours=0.5)
        if not rows:
            return

        # Group by brand
        by_brand = {}
        for r in rows:
            brand = (r['brand'] if r['brand'] and r['brand'].lower() not in ('unknown', 'sunknown', '') else '❓')
            by_brand.setdefault(brand, []).append(r)

        # Build output structure
        body_lines = []
        for brand, promos in sorted(by_brand.items()):
            body_lines.append(f"**{brand}**")
            for r in promos:
                fast_tag    = " ⚡" if r['via_fastpath'] else ""
                link_part   = f" [🔗 Lihat Pesan]({r['tg_link']})" if r['tg_link'] else ""
                body_lines.append(f"  • {r['summary']}{fast_tag}{link_part}")

        label   = now_wib.strftime('%H:%M:%S WIB')
        context = "\n".join([f"- {r['brand']}: {r['summary']}" for r in rows])
        
        try:
            async with asyncio.timeout(45):
                digest  = await gemini.answer_question(
                    f"Ringkas promo 30 menit terakhir. Waktu: {label}. Singkat dan padat.", context
                )
        except TimeoutError:
            logger.warning("AI timeout in halfhour_digest_job. Skipping.")
            return

        now_str = now_wib.strftime('%H:%M:%S')
        full_text = (
            f"⚡ **Update {label}** ({len(rows)} promo)\n"
            f"⏰ Waktu: `{now_str}`\n\n"
            f"{digest}\n\n"
            f"**Detail:**\n" + "\n".join(body_lines)
        )
        await bot.send_plain(full_text)
        logger.info("✅ [Job] Finished halfhour_digest_job")
    except Exception as e:
        logger.error(f"halfhour_digest_job error: {e}", exc_info=True)
        await bot.alert_error("halfhour_digest_job", e)


# ─────────────────────────────────────────────────────────────────────────────
# Image processing
# ─────────────────────────────────────────────────────────────────────────────

# ⚡ Bolt: Performance Optimization
# Moving these pre-compiled regular expressions to the module level prevents
# the Python interpreter from attempting to recompile them (or look them up
# in its internal cache) every time `image_processing_job` runs. This saves
# micro-seconds per execution and reduces overhead on the event loop.
_IMG_SKIP = re.compile(
    r'\b(setting|pengaturan|config|tutorial|cara|gimana|help|tolong|'
    r'oot|random|selfie|meme|lucu|haha|wkwk|ini kak|kak ini)\b',
    re.IGNORECASE
)
_IMG_KEEP = re.compile(
    r'(\b(promo|diskon|cashback|voucher|gratis|murah|hemat|sale|deal|'
    r'sfood|gfood|grab|shopee|gojek|aman|jp|work|flash|limit)\b|%|rp\s?\d)',
    re.IGNORECASE
)

async def image_processing_job(db: Database, gemini: GeminiProcessor, listener: Any) -> None:
    """Processes unhandled photo messages with the vision model."""
    logger.info("⏰ [Job] Starting image_processing_job...")
    try:
        import pytz
        if not db.conn:
            return

        async with db.conn.execute("""
            SELECT id, tg_msg_id, chat_id, text, timestamp
            FROM messages
            WHERE has_photo=1 AND image_processed=0
            ORDER BY id DESC LIMIT 20
        """) as cur:
            all_rows = await cur.fetchall()
        if not all_rows:
            return

        rows: list[Any] = []
        for r in all_rows:
            caption = (r['text'] or '').strip()
            if not caption:
                rows.append(r)
                continue
            if _IMG_SKIP.search(caption) and not _IMG_KEEP.search(caption):
                await db.conn.execute(
                    "UPDATE messages SET image_processed=1 WHERE id=?", (r['id'],)
                )
                continue
            rows.append(r)
        await db.conn.commit()

        if not rows:
            return
        rows = rows[:10]
        
        async def _process_single_image(r):
            msg_id, tg_msg_id, chat_id, text, ts = (
                r['id'], r['tg_msg_id'], r['chat_id'], r['text'], r['timestamp']
            )
            try:
                # Fetch message and photo bytes
                msg_obj = await listener.client.get_messages(chat_id, ids=tg_msg_id)
                if not msg_obj or not msg_obj.photo:
                    await db.conn.execute("UPDATE messages SET image_processed=1 WHERE id=?", (msg_id,))
                    return msg_id

                downloaded = await listener.client.download_media(msg_obj, bytes)
                if not downloaded:
                    photo_bytes = None
                elif isinstance(downloaded, str):
                    if await asyncio.to_thread(os.path.exists, downloaded):
                        photo_bytes = await asyncio.to_thread(_read_file_bytes, downloaded)
                        await asyncio.to_thread(os.remove, downloaded)
                    else:
                        photo_bytes = None
                else:
                    photo_bytes = downloaded

                if not photo_bytes:
                    await db.conn.execute("UPDATE messages SET image_processed=1 WHERE id=?", (msg_id,))
                    return msg_id

                start_ai = time.monotonic()
                promo = await gemini.process_image(photo_bytes, text or "", msg_id)
                ai_duration = time.monotonic() - start_ai

                if promo:
                    promo.ai_time = ai_duration
                    promo.queue_time = (datetime.now(timezone.utc) - shared._parse_ts(ts)).total_seconds() - ai_duration
                    
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

                # Commit image_processed status immediately per item
                await db.conn.execute(
                    "UPDATE messages SET image_processed=1 WHERE id=?", (msg_id,)
                )
                return msg_id
            except Exception as e:
                logger.error(f"image_processing_job item (msg {tg_msg_id}) error: {e}")
                return None

        tasks = [_process_single_image(r) for r in rows]
        results = await asyncio.gather(*tasks)
        processed_ids = [res for res in results if res is not None]
        
        if processed_ids:
            await db.conn.commit()

        logger.info(f"✅ [Job] Finished image_processing_job (processed {len(processed_ids)} imgs)")
    except Exception as e:
        logger.error(f"image_processing_job critical error: {e}", exc_info=True)
        if shared.bot:
            await shared.bot.alert_error("image_processing_job", e)


# ─────────────────────────────────────────────────────────────────────────────
# Heartbeat
# ─────────────────────────────────────────────────────────────────────────────

async def heartbeat_job(db: Database, gemini: GeminiProcessor, bot: TelegramBot, WIB: Any) -> None:
    """Monitors system health, RPM pressure, and listener lag."""
    logger.info("⏰ [Job] Starting heartbeat_job...")
    try:
        if not await db.ensure_connection():
            logger.error("❌ Heartbeat failed: Could not ensure database connection.")
            return

        now_wib = datetime.now(WIB).strftime('%H:%M:%S WIB')
        queue   = await db.get_queue_size()

        async with db.conn.execute("SELECT MAX(timestamp) FROM messages") as cur:
            row = await cur.fetchone()
            last_ts = row[0] if row else None

        lag_note = ""
        if last_ts:
            last_dt = shared._parse_ts(last_ts)
            lag_min = (datetime.now(timezone.utc) - last_dt).total_seconds() / 60
            if lag_min > 20 and not shared._listener_reconnecting:
                asyncio.create_task(_reconnect_listener(lag_min))
                lag_note = f"\n🔌 *Lag {int(lag_min)}m — auto-reconnecting...*"
            elif lag_min > 5:
                lag_note = f"\n⚠️ *Lag: {int(lag_min)}m*"

        # AI Army Fleet Status
        fleet_status = []
        total_active = 0
        total_limit = 0
        
        # Sort by usage to show busiest ones first
        sorted_slots = sorted(gemini._slots.values(), key=lambda s: s.current_usage(), reverse=True)
        
        for slot in sorted_slots:
            usage = slot.current_usage()
            total_active += usage
            total_limit += slot.limit
            if usage > 0:
                fleet_status.append(f"{slot.name}: `{usage}/{slot.limit}`")
        
        army_note = f" | 🪖 `{total_active}/{total_limit}`"
        if fleet_status:
            army_note += "\n" + " · ".join(fleet_status[:3]) # Show top 3 active
        
        text = f"💓 `{now_wib}` | queue: `{queue}`{army_note}{lag_note}"
        await bot.send_plain(text)
        logger.info("✅ [Job] Finished heartbeat_job")
    except Exception as e:
        logger.error(f"heartbeat_job error: {e}", exc_info=True)
        await bot.alert_error("heartbeat_job", e)


# ─────────────────────────────────────────────────────────────────────────────
# Hot threads
# ─────────────────────────────────────────────────────────────────────────────

async def hot_thread_job(db: Database, gemini: GeminiProcessor, listener: Any, bot: TelegramBot, 
                         WIB: Any, alerted_hot_threads: dict[int, tuple[int, datetime]]) -> None:
    """Identifies highly active discussion threads and summarizes them."""
    logger.info("⏰ [Job] Starting hot_thread_job...")
    try:
        threads = await db.get_hot_threads(minutes=15, min_replies=3, limit=10)
        if not threads:
            return

        if len(alerted_hot_threads) > 1000:
            keys = sorted(alerted_hot_threads.keys())
            for k in keys[:50]:
                del alerted_hot_threads[k]

        calls_this_run = 0
        for t in threads:
            if calls_this_run >= 3:
                break
            now_ts    = datetime.now(timezone.utc)
            last_data = alerted_hot_threads.get(t['tg_msg_id'], (0, cast(datetime, datetime(1, 1, 1, tzinfo=timezone.utc))))
            last_count, last_alerted_at = last_data
            
            cooldown_ok = (now_ts - last_alerted_at).total_seconds() > 900
            
            if t['reply_count'] >= last_count + 3 and cooldown_ok:
                alerted_hot_threads[t['tg_msg_id']] = (t['reply_count'], now_ts)
                calls_this_run += 1

                link     = _make_tg_link(t['chat_id'], t['tg_msg_id'])
                msg_wib  = shared._parse_ts(t['timestamp']).astimezone(WIB).strftime('%H:%M:%S')

                reply_rows  = await db.get_thread_replies(t['tg_msg_id'], t['chat_id'], limit=20)
                reply_texts = [r['text'] for r in reply_rows if r['text']]

                parent_photo: bytes | None = None
                if t['has_photo']:
                    try:
                        downloaded = await listener.client.download_media(
                            await listener.client.get_messages(t['chat_id'], ids=t['tg_msg_id']),
                            bytes
                        )
                        if isinstance(downloaded, bytes):
                            parent_photo = downloaded
                        elif isinstance(downloaded, str):
                            parent_photo = await asyncio.to_thread(_read_file_bytes, downloaded)
                            await asyncio.to_thread(os.remove, downloaded)
                    except Exception as e:
                        logger.error(f"Hot thread photo download failed: {e}")

                try:
                    async with asyncio.timeout(30):
                        summary = await gemini.summarize_thread(
                            t['text'] or "", reply_texts, parent_photo=parent_photo
                        )
                except TimeoutError:
                    logger.warning("AI timeout in hot_thread_job. Skipping.")
                    continue

                snippets: list[str] = []
                for r in reply_rows[:4]:
                    snip = (r['text'] or '').strip()
                    if snip:
                        snip = (snip[:57] + "...") if len(snip) > 60 else snip
                        snippets.append(f"• _{snip}_ ")

                parent_snippet = (t['text'] or '').strip()
                if len(parent_snippet) > 100:
                    parent_snippet = parent_snippet[:97] + "..."

                age_min = (now_ts - shared._parse_ts(t['timestamp'])).total_seconds() / 60
                text = (
                    f"🔥 **Thread Hot! ({t['reply_count']} balasan)**\n"
                    f"⏰ Jam: `{msg_wib}` (`{age_min:.0f}m ago`)\n"
                    f"📌 **Pesan Awal:**\n_{parent_snippet}_\n\n"
                    f"💬 **Topik:**\n{summary}\n\n"
                    f"📜 **Cuplikan:**\n" + "\n".join(snippets) + "\n\n"
                    f"🔗 [Lihat Thread]({link})"
                )
                await bot.send_plain(text)
        logger.info("✅ [Job] Finished hot_thread_job")
    except Exception as e:
        logger.error(f"hot_thread_job error: {e}", exc_info=True)
        await bot.alert_error("hot_thread_job", e)


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
        import pytz
        if not db.conn:
            return

        # Single query fetching message and its parent context via self-join
        async with db.conn.execute("""
            SELECT m1.id, m1.text, m1.timestamp, m1.chat_id, m1.tg_msg_id,
                   m2.text as parent_text
            FROM messages m1
            LEFT JOIN messages m2 ON m1.reply_to_msg_id = m2.tg_msg_id AND m1.chat_id = m2.chat_id
            WHERE m1.has_time_mention=1 AND m1.time_alerted=0 
            ORDER BY m1.id ASC
        """) as cur:
            rows = await cur.fetchall()
            
        if not rows:
            return

        all_done = []
        noise_count = 0
        for r in rows:
            text = r['text'] or ""
            msg_id = r['id']
            all_done.append(msg_id)

            # FastText weighting: skip noise
            ft_label, ft_conf = await shared.classify_one(text)
            if (ft_label == "__label__JUNK" and ft_conf > 0.85) or not _is_time_signal_worthy(text):
                noise_count += 1
                continue

            link = _make_tg_link(r['chat_id'], r['tg_msg_id'])
            now_str = datetime.now(pytz.timezone('Asia/Jakarta')).strftime('%H:%M:%S')
            
            context_header = ""
            if r['parent_text']:
                p_text = r['parent_text'].replace('\n', ' ')[:100]
                context_header = f"💬 _Balasan untuk: \"{p_text}...\"_\n\n"

            alert = (
                f"🕒 **Sinyal Waktu:**\n"
                f"{context_header}"
                f"{text}\n"
                f"⏰ Waktu: `{now_str}`\n\n"
                f"🛡️ FT: `{ft_label.replace('__label__', '')} ({ft_conf:.2f})`"
                f"\n\n🔗 Lihat Pesan:\n{link}"
            )
            await bot.send_plain(alert)

        if all_done:
            # Chunk updates to avoid SQLite variable limits (standard is 999)
            chunk_size = 900
            for i in range(0, len(all_done), chunk_size):
                chunk = all_done[i:i + chunk_size]
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
        # Attempt to find a reference ID if we crashed inside the loop
        ref_id = locals().get('msg_id')
        await bot.alert_error("time_mention_job", e, source_msg_id=ref_id)


# ─────────────────────────────────────────────────────────────────────────────
# Trend detection
# ─────────────────────────────────────────────────────────────────────────────

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
                trends = await gemini.generate_narrative([dict(m) for m in msgs], db=db)
        except TimeoutError:
            logger.warning("AI timeout in trend_job. Skipping.")
            return
        
        if not trends:
            return

        # Use combined topics as a simple string for the dedup check
        current_summary = " ".join([t.topic for t in trends])
        import time as _time
        _TREND_REPEAT_COOLDOWN = 1800  # 30 min, not "exact same string"
        now_mono = _time.monotonic()
        if (current_summary == shared._last_trend_alert 
                and (now_mono - shared._last_trend_alert_ts) < _TREND_REPEAT_COOLDOWN):
            return

        lines: list[str] = []
        for t in trends:
            chat_id = msgs[0]['chat_id']
            link = _make_tg_link(chat_id, t.msg_id)
            lines.append(f"• {t.topic}\n  🔗 [Lihat Pesan]({link})")

        now_wib = datetime.now(pytz.timezone("Asia/Jakarta")).strftime('%H:%M:%S')
        model_info = f" (via {trends[0].model_name})" if trends[0].model_name else ""
        full_text = (
            f"📈 **Narasi Tren (15m){model_info}:**\n"
            f"⏰ Waktu: `{now_wib}`\n\n"
            + "\n\n".join(lines)
        )

        await bot.send_plain(full_text)

        shared._last_trend_alert = current_summary
        shared._last_trend_alert_ts = _time.monotonic()
        logger.info("✅ [Job] Finished trend_job")
    except Exception as e:
        logger.error(f"trend_job error: {e}", exc_info=True)
        await bot.alert_error("trend_job", e)


# ─────────────────────────────────────────────────────────────────────────────
# Spike detection
# ─────────────────────────────────────────────────────────────────────────────

async def spike_detection_job(db: Database, gemini: GeminiProcessor, bot: TelegramBot) -> None:
    """Detects and provides context for sudden traffic bursts."""
    logger.info("⏰ [Job] Starting spike_detection_job...")
    try:
        now_utc = datetime.now(timezone.utc)
        from shared import _last_spike_alert
        # Reduced cooldown from 10m to 5m
        if (now_utc - _last_spike_alert).total_seconds() < 300:
            return
            
        one_min_ago = (now_utc - timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M:%S+00:00')
        if not db.conn:
            return

        async with db.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE timestamp >= ?", (one_min_ago,)
        ) as cur:
            row = await cur.fetchone()
            count = cast(int, row[0]) if row else 0

        five_min_ago = (now_utc - timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S+00:00')
        async with db.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE timestamp >= ?", (five_min_ago,)
        ) as cur:
            row_5 = await cur.fetchone()
            five_min_count = cast(int, row_5[0]) if row_5 else 0
        avg_per_min = five_min_count / 5

        # More sensitive: 10 msg/min and 1.5x average (helps catch sustained bursts)
        if count >= 10 and count >= max(avg_per_min * 1.5, 5):
            recent_msgs  = await db.get_recent_messages(minutes=1)
            sample_lines = [
                f"• {html.escape((m['text'] or '').strip()[:80])}" for m in recent_msgs[:5]
            ]

            # Find the dominant brand: if any brand shows up in ≥40% of the
            # last minute's messages, name it explicitly and pick its most
            # recent message as a clickable "start here" link. Previously
            # the alert was keyword-only and felt generic.
            from shared import _guess_brand
            from db import normalize_brand
            brand_counts: dict[str, int] = {}
            brand_latest_msg: dict[str, Any] = {}
            for m in recent_msgs:
                b = normalize_brand(_guess_brand(m['text']))
                if b == "Unknown":
                    continue
                brand_counts[b] = brand_counts.get(b, 0) + 1
                brand_latest_msg.setdefault(b, m)   # first = most recent
            dominant_brand: str | None = None
            dominant_count = 0
            for b, c in brand_counts.items():
                if c > dominant_count:
                    dominant_brand = b
                    dominant_count = c
            # Require ≥40% of sample to agree on brand before naming it.
            if dominant_brand and dominant_count < max(3, int(count * 0.4)):
                dominant_brand = None

            top_words = await db.get_recent_words(minutes=3)
            hot_words = [w[0] for w in top_words[:5]]

            try:
                async with asyncio.timeout(30):
                    narrative = await gemini.interpret_keywords(
                        hot_words, 3, [cast(str, m['text']) for m in recent_msgs if m['text']]
                    )
            except TimeoutError:
                logger.warning("AI timeout in spike_detection_job. Skipping narrative.")
                narrative = "Aktivitas meningkat tajam."

            from telegram.constants import ParseMode
            header_lines = [
                "🚀 **Lonjakan Pesan!**",
                f"📊 Kecepatan: `{count} msg/min`",
                f"📈 Rata-rata: `{avg_per_min:.1f} msg/min`",
                f"⏰ Waktu: `{datetime.now(pytz.timezone('Asia/Jakarta')).strftime('%H:%M:%S')}`",
            ]
            if dominant_brand:
                m = brand_latest_msg[dominant_brand]
                link = _make_tg_link(m['chat_id'], m['tg_msg_id'])
                header_lines.append(
                    f"🏪 Dominan: **{dominant_brand}** "
                    f"({dominant_count}/{count}) · "
                    f"[mulai di sini]({link})"
                )

            text = (
                "\n".join(header_lines) + "\n\n"
                f"🤖 **Analisis:**\n{narrative or 'Aktivitas meningkat tajam.'}\n\n"
                f"**Cuplikan:**\n" + "\n".join(sample_lines)
            )
            await bot.send_plain(text)
            shared._last_spike_alert = now_utc
        logger.info("✅ [Job] Finished spike_detection_job")
    except Exception as e:
        logger.error(f"spike_detection_job error: {e}", exc_info=True)
        await bot.alert_error("spike_detection_job", e)


# ─────────────────────────────────────────────────────────────────────────────
# Dead promo reaper
# ─────────────────────────────────────────────────────────────────────────────

async def dead_promo_reaper_job(db: Database, bot: TelegramBot) -> None:
    """Closes expired promotions based on subsequent community chat signals."""
    logger.info("💀 [Job] Starting dead_promo_reaper_job...")
    try:
        import pytz
        if not db.conn:
            return
            
        # More robust keywords for expiry and active status
        # expiry: nt, abis, habis, zonk, limit, sold out, koid, mati, gabisa, ga bisa, koin, koid, telat, ketinggalan, hangus, off
        # active: aman, on, work, jp, cair, nyala, masih, bisa, dapet, dapat, pecah, luber
        
        async with db.conn.execute("""
            WITH ActivePromos AS (
                SELECT p.id, m.tg_msg_id, m.chat_id
                FROM promos p
                JOIN messages m ON p.source_msg_id = m.id
                WHERE p.status = 'active'
                  AND p.created_at >= strftime('%Y-%m-%d %H:%M:%S+00:00','now','-6 hours')
            ),
            ReplySignals AS (
                SELECT
                    ap.id AS promo_id,
                    SUM(CASE WHEN 
                        mrt.text LIKE '% nt%' OR mrt.text LIKE 'nt %' OR mrt.text = 'nt' OR
                        mrt.text LIKE '%abis%' OR mrt.text LIKE '%habis%' OR 
                        mrt.text LIKE '%sold out%' OR mrt.text LIKE '%expired%' OR 
                        mrt.text LIKE '%kehabisan%' OR mrt.text LIKE '%ga bisa%' OR 
                        mrt.text LIKE '%gabisa%' OR mrt.text LIKE '%mati%' OR 
                        mrt.text LIKE '%hangus%' OR mrt.text LIKE '%ga work%' OR 
                        mrt.text LIKE '%zonk%' OR mrt.text LIKE '%limit%' OR
                        mrt.text LIKE '%off%' OR mrt.text LIKE '%koid%' OR
                        mrt.text LIKE '%telat%' OR mrt.text LIKE '%ketinggalan%'
                        THEN 1 ELSE 0 END) AS expiry_votes,
                    SUM(CASE WHEN 
                        mrt.text LIKE '%aman%' OR mrt.text LIKE '% on%' OR mrt.text LIKE 'on %' OR
                        mrt.text LIKE '%work%' OR mrt.text LIKE '%jp%' OR 
                        mrt.text LIKE '%masih%' OR mrt.text LIKE '%bisa%' OR
                        mrt.text LIKE '%cair%' OR mrt.text LIKE '%nyala%' OR
                        mrt.text LIKE '%pecah%' OR mrt.text LIKE '%luber%'
                        THEN 1 ELSE 0 END) AS active_votes
                FROM ActivePromos ap
                JOIN messages mrt ON mrt.reply_to_msg_id = ap.tg_msg_id AND mrt.chat_id = ap.chat_id
                GROUP BY ap.id
            )
            SELECT promo_id FROM ReplySignals
            WHERE expiry_votes >= 2 AND expiry_votes > active_votes
        """) as cur:
            promo_ids_to_reap = [r[0] for r in await cur.fetchall()]

        if promo_ids_to_reap:
            logger.info(f"💀 Reaper: found {len(promo_ids_to_reap)} promos to mark as expired.")
            placeholders = ','.join('?' * len(promo_ids_to_reap))
            await db.conn.execute(
                f"UPDATE promos SET status='expired' WHERE id IN ({placeholders})",
                promo_ids_to_reap
            )
            await db.conn.commit()
        logger.info("✅ [Job] Finished dead_promo_reaper_job")
    except Exception as e:
        logger.error(f"dead_promo_reaper_job error: {e}", exc_info=True)
        await bot.alert_error("dead_promo_reaper_job", e)


# ─────────────────────────────────────────────────────────────────────────────
# Confirmation gate
# ─────────────────────────────────────────────────────────────────────────────

async def confirmation_gate_job(db: Database) -> None:
    """Processes low-confidence promotions that were waiting for confirmation."""
    logger.info("⏰ [Job] Starting confirmation_gate_job...")
    try:
        import pytz
        if not db.conn:
            return
            
        now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        async with db.conn.execute("""
            SELECT id, brand, p_data_json, tg_link, timestamp, corroborations, corroboration_texts, expires_at
            FROM pending_confirmations
            WHERE expires_at <= ? OR corroborations >= 1
            ORDER BY id ASC
        """, (now_str,)) as cur:
            ready = await cur.fetchall()
        if not ready:
            return

        fired_ids: list[int] = []
        from processor import PromoExtraction
        for r in ready:
            if r['corroborations'] >= 1:
                p_data = PromoExtraction.model_validate_json(r['p_data_json'])
                logger.info(f"Confirm gate CORROBORATED: {r['brand']}")
                await db.save_pending_alert(
                    r['brand'], r['p_data_json'], r['tg_link'],
                    r['timestamp'], corroborations=r['corroborations'],
                    corroboration_texts=r['corroboration_texts'],
                    source='ai', commit=False
                )
                t = shared.get_buffer_flush_task()
                if t is None or t.done():
                    shared.set_buffer_flush_task(
                        asyncio.create_task(_flush_alert_buffer(delay=0.3))
                    )
                async with shared._recent_alerts_lock:
                    shared._recent_alerts_history.append({
                        "brand": r['brand'], "summary": p_data.summary
                    })
            fired_ids.append(r['id'])

        if fired_ids:
            ph = ','.join('?' * len(fired_ids))
            await db.conn.execute(
                f"DELETE FROM pending_confirmations WHERE id IN ({ph})", list(fired_ids)
            )
            await db.conn.commit()
        logger.info("✅ [Job] Finished confirmation_gate_job")
    except Exception as e:
        logger.error(f"confirmation_gate_job error: {e}", exc_info=True)
        if shared.bot:
            await shared.bot.alert_error("confirmation_gate_job", e)


# ─────────────────────────────────────────────────────────────────────────────
# DB maintenance
# ─────────────────────────────────────────────────────────────────────────────

async def db_maintenance_job(db: Database, bot: TelegramBot) -> None:
    """Executes periodic database cleanup and optimization."""
    logger.info("⏰ [Job] Starting db_maintenance_job...")
    try:
        # Skip heavy VACUUM during peak deal hours (9 AM - 11 PM WIB)
        import pytz
        jakarta_tz = pytz.timezone("Asia/Jakarta")
        now_wib = datetime.now(jakarta_tz)
        
        ignore_vacuum = False
        if 9 <= now_wib.hour <= 23:
            logger.info("⚡ Peak hours detected (9 AM - 11 PM), skipping VACUUM.")
            ignore_vacuum = True
            
        await db.prune_old_messages(ignore_vacuum=ignore_vacuum)
        logger.info("✅ [Job] Finished db_maintenance_job")
    except Exception as e:
        logger.error(f"db_maintenance_job error: {e}", exc_info=True)
        await bot.alert_error("db_maintenance_job", e)


# ─────────────────────────────────────────────────────────────────────────────
# AI Model Retraining
# ─────────────────────────────────────────────────────────────────────────────

async def fasttext_retrain_job(db: Database, bot: TelegramBot) -> None:
    """Automatically exports data and retrains the FastText model."""
    logger.info("⏰ [Job] Starting FastText autonomous retraining...")
    try:
        import os
        import subprocess
        import sys
        
        # 1. Path setup
        venv_python = os.path.join(os.getcwd(), "venv", "bin", "python3")
        if not os.path.exists(venv_python):
            venv_python = sys.executable  # Fallback to current python
            
        export_script = os.path.join(os.getcwd(), "tools", "export_training_data.py")
        train_script = os.path.join(os.getcwd(), "tools", "train_model.py")
        
        # 2. Export Data
        logger.info("📡 Exporting training data...")
        process = await asyncio.create_subprocess_exec(
            venv_python, export_script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"Export failed: {stderr.decode()}")
            
        # 3. Train Model
        logger.info("🧠 Training FastText model...")
        process = await asyncio.create_subprocess_exec(
            "nice", "-n", "19", venv_python, train_script, "--data", "data/training.txt", "--out", "model.ftz",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"Training failed: {stderr.decode()}")
            
        # 4. Reload Model in memory
        import shared
        reloaded = await shared.load_classifier("model.ftz")
        
        if reloaded:
            logger.info("✅ [Job] FastText retraining complete and reloaded.")
        else:
            logger.warning("⚠️ [Job] FastText trained but failed to reload.")
            
    except Exception as e:
        logger.error(f"fasttext_retrain_job error: {e}", exc_info=True)
        await bot.alert_error("fasttext_retrain_job", e)


# ─────────────────────────────────────────────────────────────────────────────
# Visual Trends
# ─────────────────────────────────────────────────────────────────────────────

async def visual_trend_job(db: Database, bot: TelegramBot) -> None:
    """Generates and sends a visual chart of top brands."""
    logger.info("⏰ [Job] Starting visual_trend_job...")
    try:
        rows = await db.get_brand_stats(hours=24)
        if not rows:
            logger.info("✅ [Job] No brand stats to visualize.")
            return

        # Attempt to generate chart using matplotlib
        import io
        try:
            import matplotlib.pyplot as plt
            import io
            
            # Encapsulate synchronous plotting logic
            def _draw_chart(b, c):
                plt.switch_backend('Agg')
                plt.figure(figsize=(10, 6))
                colors = plt.cm.Paired(range(len(b)))
                bars = plt.bar(b, c, color=colors)
                plt.title('Top 10 Brands (Last 24h)', fontsize=15, pad=20)
                plt.xlabel('Brand', fontsize=12)
                plt.ylabel('Promo Count', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                for bar in bars:
                    yval = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, yval, ha='center', va='bottom')
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150)
                plt.close('all') # Critical to prevent memory leak
                buf.seek(0)
                return buf

            brands = [r['brand'] for r in rows]
            counts = [r['count'] for r in rows]
            
            # Execute in a separate thread to prevent blocking the async loop
            buf = await asyncio.to_thread(_draw_chart, brands, counts)
            
            WIB = pytz.timezone("Asia/Jakarta")
            now_str = datetime.now(WIB).strftime('%d %b %H:%M:%S')
            caption = f"📊 **Brand activity summary**\n🕒 `{now_str} WIB`"
            await bot.send_photo(buf.read(), caption=caption)
            logger.info("✅ [Job] Visual trend chart sent.")
            
        except ImportError:
            logger.warning("⚠️ matplotlib not installed. Skipping visual chart.")
            # Fallback to text summary
            text = "📊 **Top 10 Brands (Last 24h)**\n\n"
            for r in rows:
                text += f"• **{r['brand']}**: {r['count']}\n"
            await bot.send_plain(text)
            
    except Exception as e:
        logger.error(f"visual_trend_job error: {e}", exc_info=True)
        await bot.alert_error("visual_trend_job", e)

