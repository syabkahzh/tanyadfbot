"""jobs.py — Background Scheduled Jobs.

Implements all asynchronous periodic tasks including digests, trend analysis, 
spike detection, and database maintenance.
"""

import asyncio
import html
import re
import pytz
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, cast

from db import Database
from processor import GeminiProcessor
from bot import TelegramBot
from utils import _esc
import shared
from shared import _make_tg_link, _flush_alert_buffer, _reconnect_listener

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Time reminder job — "sinyal waktu"
# Scans active promos for an HH:MM time of day in valid_until/summary/conditions
# and fires a T-2min reminder so users don't miss a time-bounded promo window.
# ─────────────────────────────────────────────────────────────────────────────

# HH:MM / HH.MM / HH:MM WIB — anchored to word boundaries to avoid price numbers.
_TIME_OF_DAY_RE = re.compile(
    r'\b(?:jam|pukul|pkl|pk|s[./]?d|sampai|sebelum)?\s*'
    r'(\d{1,2})[:.](\d{2})\s*(wib)?\b',
    re.IGNORECASE,
)
# "jam 10" / "pukul 14" without minutes (interpreted as :00).
_TIME_HOUR_RE = re.compile(
    r'\b(?:jam|pukul|pkl|pk|s[./]?d|sampai|sebelum)\s+(\d{1,2})(?:\s*(wib|pagi|siang|sore|malem|malam))?\b',
    re.IGNORECASE,
)


def _extract_time_of_day(text: str) -> tuple[int, int] | None:
    """Return (hour, minute) if text explicitly names a time of day in WIB.

    Conservative: only returns when we're confident the number is a clock
    time, not a price or date. Prefers the HH:MM form; falls back to
    "jam/pukul N" (minute=0) only when anchored by a time preposition.
    """
    if not text:
        return None
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
            return (hh, 0)
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
        async with db.conn.execute(
            "SELECT id, brand, summary, conditions, valid_until, tg_link "
            "FROM promos "
            "WHERE status='active' AND reminder_fired=0 "
            "  AND created_at >= strftime('%Y-%m-%d %H:%M:%S+00:00','now','-24 hours') "
            "ORDER BY id DESC LIMIT 200"
        ) as cur:
            rows = await cur.fetchall()

        if not rows:
            logger.info("✅ [Job] time_reminder_job: no candidates")
            return

        fired = 0
        from telegram.constants import ParseMode

        for r in rows:
            # Prefer valid_until if it contains a time; otherwise scan the
            # summary and conditions for an explicit HH:MM.
            haystack = " ".join(
                x for x in (r['valid_until'], r['summary'], r['conditions']) if x
            )
            tod = _extract_time_of_day(haystack)
            if tod is None:
                # Mark as checked-no-time so we don't re-scan next minute.
                await db.conn.execute(
                    "UPDATE promos SET reminder_fired=1 WHERE id=?", (r['id'],)
                )
                continue

            hh, mm = tod
            target = now_wib.replace(hour=hh, minute=mm, second=0, microsecond=0)
            # If target already passed today, assume tomorrow.
            if target <= now_wib:
                target = target + timedelta(days=1)

            minutes_to = (target - now_wib).total_seconds() / 60.0

            # Fire when the stated time is between 2 and 3 minutes away — we
            # run this job every minute so this gives us exactly one window.
            if 2.0 <= minutes_to <= 3.0:
                now_str = datetime.now(pytz.timezone('Asia/Jakarta')).strftime('%H:%M:%S')
                text = (
                    f"⏰ <b>Sinyal Waktu — 2 menit lagi!</b>\n"
                    f"⏰ Skrg: <code>{now_str}</code>\n"
                    f"🏪 <b>{html.escape(r['brand'])}</b>\n"
                    f"🕒 Target: <code>{target.strftime('%H:%M WIB')}</code>\n"
                    f"📝 {_esc(r['summary'])}\n"
                )
                if r['conditions']:
                    text += f"ℹ️ <i>{_esc(r['conditions'])}</i>\n"
                if r['tg_link']:
                    text += f"\n🔗 <a href='{html.escape(r['tg_link'])}'>Lihat Promo</a>"

                try:
                    await bot.send_plain(text, parse_mode=ParseMode.HTML)
                    fired += 1
                except Exception as e:
                    logger.error(f"time_reminder send failed for promo {r['id']}: {e}")
                    continue

                await db.conn.execute(
                    "UPDATE promos SET reminder_fired=1 WHERE id=?", (r['id'],)
                )

        if fired:
            await db.conn.commit()
            logger.info(f"✅ [Job] time_reminder_job fired {fired} reminders")
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
                f"📊 <b>Digest {hour_label}</b>\n\n"
                "😴 Tidak ada promo yang terdeteksi 1 jam terakhir.",
                parse_mode='HTML'
            )
            return

        lines = []
        for r in rows:
            brand       = (r['brand'] if r['brand']
                           and r['brand'].lower() not in ('unknown', 'sunknown', '') else '❓')
            status_icon = ("🟢" if r['status'] == 'active'
                           else ("🔴" if r['status'] == 'expired' else "⚪"))
            link_part   = f" <a href='{r['tg_link']}'>[→]</a>" if r['tg_link'] else ""
            fast_tag    = " ⚡" if r['via_fastpath'] else ""
            lines.append(
                f"{status_icon} <b>{html.escape(brand)}</b>"
                f"{fast_tag}: {html.escape(r['summary'])}{link_part}"
            )

        context    = "\n".join([f"- {r['brand']}: {r['summary']}" for r in rows])
        try:
            async with asyncio.timeout(60):
                ai_summary = await gemini.answer_question(
                    f"Rangkum promo berikut dalam 2-3 kalimat santai. Jam: {hour_label}.", context
                )
        except TimeoutError:
            logger.warning("AI timeout in hourly_digest_job. Skipping summary.")
            ai_summary = ""
        now_str = datetime.now(pytz.timezone('Asia/Jakarta')).strftime('%H:%M:%S')
        full_text = (
            f"📊 <b>Digest {hour_label}</b> ({len(rows)} promo)\n"
            f"⏰ Waktu: <code>{now_str}</code>\n\n"
            f"{ai_summary}\n\n"
            f"<b>Detail:</b>\n" + "\n".join(lines)
        )
        shared._last_hourly_digest = full_text
        await bot.send_plain(full_text, parse_mode='HTML')
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
                "📊 <b>Rekap 02:00–05:00 WIB</b>\n\n"
                "😴 Tidak ada promo yang masuk tadi malam.",
                parse_mode='HTML'
            )
            return

        lines = []
        for r in rows:
            brand     = (r['brand'] if r['brand']
                         and r['brand'].lower() not in ('unknown', 'sunknown', '') else '❓')
            fast_tag  = " ⚡" if r['via_fastpath'] else ""
            link_part = f" <a href='{r['tg_link']}'>[→]</a>" if r['tg_link'] else ""
            lines.append(
                f"• <b>{html.escape(brand)}</b>{fast_tag}: "
                f"{html.escape(r['summary'])}{link_part}"
            )

        context = "\n".join([f"- {r['brand']}: {r['summary']}" for r in rows])
        try:
            async with asyncio.timeout(60):
                digest  = await gemini.answer_question(
                    "Rangkum promo yang masuk jam 02:00–05:00 WIB tadi malam. Singkat, padat, santai.",
                    context
                )
        except TimeoutError:
            logger.warning("AI timeout in midnight_digest_job. Skipping summary.")
            digest = ""
        now_str = datetime.now(jakarta_tz).strftime('%H:%M:%S')
        full_text = (
            f"📊 <b>Rekap 02:00–05:00 WIB</b> ({len(rows)} promo)\n"
            f"⏰ Waktu: <code>{now_str}</code>\n\n"
            f"{digest}\n\n"
            f"<b>Detail:</b>\n" + "\n".join(lines)
        )
        await bot.send_plain(full_text, parse_mode='HTML')
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
        lines = []
        for r in rows:
            brand     = (r['brand'] if r['brand']
                         and r['brand'].lower() not in ('unknown', 'sunknown', '') else '❓')
            fast_tag  = " ⚡" if r['via_fastpath'] else ""
            link_part = f" <a href='{r['tg_link']}'>[→]</a>" if r['tg_link'] else ""
            lines.append(
                f"• <b>{html.escape(brand)}</b>{fast_tag}: "
                f"{html.escape(r['summary'])}{link_part}"
            )
        label   = now_wib.strftime('%H:%M WIB')
        context = "\n".join([f"- {r['brand']}: {r['summary']}" for r in rows])
        
        try:
            async with asyncio.timeout(45):
                digest  = await gemini.answer_question(
                    f"Ringkas promo 30 menit terakhir. Waktu: {label}. Singkat dan padat.", context
                )
        except TimeoutError:
            logger.warning("AI timeout in halfhour_digest_job. Skipping.")
            return

        now_str = datetime.now(WIB).strftime('%H:%M:%S')
        full_text = (
            f"⚡ <b>Update {label}</b> ({len(rows)} promo)\n"
            f"⏰ Waktu: <code>{now_str}</code>\n\n"
            f"{digest}\n\n" + "\n".join(lines)
        )
        await bot.send_plain(full_text, parse_mode='HTML')
        logger.info("✅ [Job] Finished halfhour_digest_job")
    except Exception as e:
        logger.error(f"halfhour_digest_job error: {e}", exc_info=True)
        await bot.alert_error("halfhour_digest_job", e)


# ─────────────────────────────────────────────────────────────────────────────
# Image processing
# ─────────────────────────────────────────────────────────────────────────────

async def image_processing_job(db: Database, gemini: GeminiProcessor, listener: Any) -> None:
    """Processes unhandled photo messages with the vision model."""
    logger.info("⏰ [Job] Starting image_processing_job...")
    try:
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
        rows = rows[:5]
        processed_ids = []

        for r in rows:
            msg_id, tg_msg_id, chat_id, text, ts = (
                r['id'], r['tg_msg_id'], r['chat_id'], r['text'], r['timestamp']
            )
            try:
                # Fetch message and photo bytes
                msg_obj = await listener.client.get_messages(chat_id, ids=tg_msg_id)
                if not msg_obj or not msg_obj.photo:
                    processed_ids.append(msg_id)
                    continue

                downloaded = await listener.client.download_media(msg_obj, bytes)
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

                processed_ids.append(msg_id)
            except Exception as e:
                logger.error(f"image_processing_job item (msg {tg_msg_id}) error: {e}")

        if processed_ids:
            ph = ','.join('?' * len(processed_ids))
            await db.conn.execute(
                f"UPDATE messages SET image_processed=1 WHERE id IN ({ph})",
                processed_ids
            )
            await db.conn.commit()

        logger.info("✅ [Job] Finished image_processing_job")
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

        now_wib = datetime.now(WIB).strftime('%H:%M WIB')
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

        # RPM pressure
        active_calls = sum(
            slot.current_usage() for slot in gemini._slots.values()
        )
        total_limit  = sum(slot.limit for slot in gemini._slots.values())
        rpm_note     = f" | rpm: `{active_calls}/{total_limit}`"

        text = f"💓 `{now_wib}` | queue: `{queue}`{rpm_note}{lag_note}"
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
        threads = await db.get_hot_threads(limit=10)
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
            last_data = alerted_hot_threads.get(t['tg_msg_id'], (0, cast(datetime, datetime.min.replace(tzinfo=timezone.utc))))
            last_count, last_alerted_at = last_data
            
            cooldown_ok = (now_ts - last_alerted_at).total_seconds() > 900
            
            if t['reply_count'] >= last_count + 5 and cooldown_ok:
                alerted_hot_threads[t['tg_msg_id']] = (t['reply_count'], now_ts)
                calls_this_run += 1

                link     = _make_tg_link(t['chat_id'], t['tg_msg_id'])
                msg_wib  = shared._parse_ts(t['timestamp']).astimezone(WIB).strftime('%H:%M')

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
                            with open(downloaded, 'rb') as f:
                                parent_photo = f.read()
                            import os
                            os.remove(downloaded)
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
                        snippets.append(f"• <i>{html.escape(snip)}</i>")

                parent_snippet = (t['text'] or '').strip()
                if len(parent_snippet) > 100:
                    parent_snippet = parent_snippet[:97] + "..."

                from telegram.constants import ParseMode
                age_min = (now_ts - shared._parse_ts(t['timestamp'])).total_seconds() / 60
                text = (
                    f"🔥 <b>Thread Hot! ({t['reply_count']} balasan)</b>\n"
                    f"⏰ Jam: <code>{msg_wib}</code> (<code>{age_min:.0f}m ago</code>)\n"
                    f"📌 <b>Pesan Awal:</b>\n<i>{html.escape(parent_snippet)}</i>\n\n"
                    f"💬 <b>Topik:</b>\n{html.escape(summary)}\n\n"
                    f"📜 <b>Cuplikan:</b>\n" + "\n".join(snippets) + "\n\n"
                    f"🔗 <a href='{link}'>Lihat Thread</a>"
                )
                await bot.send_plain(text, parse_mode=ParseMode.HTML)
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
            now_str = datetime.now(pytz.timezone('Asia/Jakarta')).strftime('%H:%M:%S')
            alert = (
                f"🕒 <b>Sinyal Waktu:</b>\n{_esc(text)}\n"
                f"⏰ Waktu: <code>{now_str}</code>\n\n"
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

        lines: list[str] = []
        for t in trends:
            chat_id = msgs[0]['chat_id']
            link = _make_tg_link(chat_id, t.msg_id)
            lines.append(f"• {html.escape(t.topic)}\n  🔗 <a href='{link}'>Lihat Pesan</a>")

        now_wib = datetime.now(pytz.timezone("Asia/Jakarta")).strftime('%H:%M:%S')
        full_text = (
            f"📈 <b>Narasi Tren (15m):</b>\n"
            f"⏰ Waktu: <code>{now_wib}</code>\n\n"
            + "\n\n".join(lines)
        )
        
        await bot.send_plain(full_text, parse_mode='HTML')
        shared._last_trend_alert = current_summary
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

        # More sensitive: 15 msg/min and 2.0x average (was 25 and 2.5x)
        if count >= 15 and count >= max(avg_per_min * 2.0, 5):
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
                "🚀 <b>Lonjakan Pesan!</b>",
                f"📊 Kecepatan: <code>{count} msg/min</code>",
                f"📈 Rata-rata: <code>{avg_per_min:.1f} msg/min</code>",
                f"⏰ Waktu: <code>{datetime.now(pytz.timezone('Asia/Jakarta')).strftime('%H:%M:%S')}</code>",
            ]
            if dominant_brand:
                m = brand_latest_msg[dominant_brand]
                link = _make_tg_link(m['chat_id'], m['tg_msg_id'])
                header_lines.append(
                    f"🏪 Dominan: <b>{html.escape(dominant_brand)}</b> "
                    f"({dominant_count}/{count}) · "
                    f"<a href='{link}'>mulai di sini</a>"
                )

            text = (
                "\n".join(header_lines) + "\n\n"
                f"🤖 <b>Analisis:</b>\n{html.escape(narrative or 'Aktivitas meningkat tajam.')}\n\n"
                f"<b>Cuplikan:</b>\n" + "\n".join(sample_lines)
            )
            await bot.send_plain(text, parse_mode=ParseMode.HTML)
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
    logger.info("⏰ [Job] Starting dead_promo_reaper_job...")
    re.compile(
        r'\b(nt|abis|habis|sold.?out|expired|kehabisan|ga bisa|gabisa|'
        r'udah mati|mati|nonaktif|hangus|error terus|ga work|gak work|off)\b',
        re.IGNORECASE
    )
    try:
        if not db.conn:
            return
            
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
                    SUM(CASE WHEN mrt.text LIKE '%nt%' OR mrt.text LIKE '%abis%' OR mrt.text LIKE '%habis%' OR mrt.text LIKE '%sold out%' OR mrt.text LIKE '%expired%' OR mrt.text LIKE '%kehabisan%' OR mrt.text LIKE '%ga bisa%' OR mrt.text LIKE '%gabisa%' OR mrt.text LIKE '%mati%' OR mrt.text LIKE '%hangus%' OR mrt.text LIKE '%ga work%' OR mrt.text LIKE '%off%' THEN 1 ELSE 0 END) AS expiry_votes,
                    SUM(CASE WHEN mrt.text LIKE '%aman%' OR mrt.text LIKE '%on%' OR mrt.text LIKE '%work%' OR mrt.text LIKE '%jp%' OR mrt.text LIKE '%masih%' OR mrt.text LIKE '%bisa%' THEN 1 ELSE 0 END) AS active_votes
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
            venv_python, train_script, "--data", "data/training.txt", "--out", "model.ftz",
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

