"""jobs.py — Background Scheduled Jobs.

Implements all asynchronous periodic tasks including digests, trend analysis, 
spike detection, and database maintenance.
"""

import asyncio
import html
import re
import sys
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Sequence, cast

from db import Database
from processor import GeminiProcessor, PromoExtraction
from bot import TelegramBot
import shared
from shared import _make_tg_link, _flush_alert_buffer, _esc, _reconnect_listener

logger = logging.getLogger(__name__)

# ── Digest jobs ──────────────────────────────────────────────────────────────

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
        full_text = (
            f"📊 <b>Digest {hour_label}</b> ({len(rows)} promo)\n\n"
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
        full_text = (
            f"📊 <b>Rekap 02:00–05:00 WIB</b> ({len(rows)} promo)\n\n"
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

        full_text = (
            f"⚡ <b>Update {label}</b> ({len(rows)} promo)\n\n"
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

        for r in rows:
            msg_id, tg_msg_id, chat_id, text, ts = (
                r['id'], r['tg_msg_id'], r['chat_id'], r['text'], r['timestamp']
            )
            try:
                downloaded = await listener.client.download_media(
                    await listener.client.get_messages(chat_id, ids=tg_msg_id),
                    bytes
                )
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
                    await db.conn.execute(
                        "UPDATE messages SET image_processed=1 WHERE id=?", (msg_id,)
                    )
                    await db.conn.commit()
                    continue

                promo = await gemini.process_image(photo_bytes, text or "", msg_id)

                if promo:
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

                await db.conn.execute(
                    "UPDATE messages SET image_processed=1 WHERE id=?", (msg_id,)
                )
                await db.conn.commit()
            except Exception as e:
                logger.error(f"image_processing_job item (msg {tg_msg_id}) error: {e}")
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
                text = (
                    f"🔥 <b>Thread Hot! ({t['reply_count']} balasan)</b>\n"
                    f"⏰ Jam: <code>{msg_wib}</code>\n"
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

        from processor import _PROMO
        from telegram.constants import ParseMode
        for r in rows:
            text  = r['text'] or ""
            # Only alert on time-mentions that also have deal signals
            if not _PROMO.search(text):
                await db.conn.execute(
                    "UPDATE messages SET time_alerted=1 WHERE id=?", (r['id'],)
                )
                continue

            link  = _make_tg_link(r['chat_id'], r['tg_msg_id'])
            alert = (
                f"🕒 <b>Sinyal Waktu:</b>\n{_esc(text)}\n\n"
                f"🔗 <a href='{link}'>Lihat Pesan</a>"
            )
            await bot.send_plain(alert, parse_mode=ParseMode.HTML)
            await db.conn.execute(
                "UPDATE messages SET time_alerted=1 WHERE id=?", (r['id'],)
            )
        await db.conn.commit()
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
                trends = await gemini.generate_narrative(msgs)
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

        full_text = (
            f"📈 <b>Narasi Tren (15m):</b>\n\n"
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
            text = (
                f"🚀 <b>Lonjakan Pesan!</b>\n"
                f"📊 Kecepatan: <code>{count} msg/min</code>\n"
                f"📈 Rata-rata: <code>{avg_per_min:.1f} msg/min</code>\n\n"
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
    EXPIRY_SIGNALS = re.compile(
        r'\b(nt|abis|habis|sold.?out|expired|kehabisan|ga bisa|gabisa|'
        r'udah mati|mati|nonaktif|hangus|error terus|ga work|gak work|off)\b',
        re.IGNORECASE
    )
    try:
        if not db.conn:
            return
            
        async with db.conn.execute("""
            SELECT p.id, p.source_msg_id, p.brand, p.summary,
                   m.tg_msg_id, m.chat_id
            FROM promos p
            LEFT JOIN messages m ON p.source_msg_id = m.id
            WHERE p.status = 'active'
              AND p.created_at >= strftime('%Y-%m-%d %H:%M:%S+00:00','now','-6 hours')
        """) as cur:
            active_promos = await cur.fetchall()
        if not active_promos:
            return

        reaped = 0
        for promo in active_promos:
            if not promo['tg_msg_id'] or not promo['chat_id']:
                continue

            reply_rows = await db.get_thread_replies(
                promo['tg_msg_id'], promo['chat_id'], limit=30
            )

            if not reply_rows:
                continue
            expiry_votes = sum(
                1 for r in reply_rows if EXPIRY_SIGNALS.search(r['text'] or '')
            )
            active_votes = sum(
                1 for r in reply_rows
                if re.search(r'\b(aman|on|work|jp|masih|masih bisa)\b',
                             r['text'] or '', re.IGNORECASE)
            )
            if expiry_votes >= 2 and expiry_votes > active_votes:
                await db.conn.execute(
                    "UPDATE promos SET status='expired' WHERE id=?", (promo['id'],)
                )
                reaped += 1
                logger.info(f"💀 Reaper: expired '{promo['brand']} — {promo['summary'][:40]}'")
                
        if reaped:
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
        await db.prune_old_messages()
        logger.info("✅ [Job] Finished db_maintenance_job")
    except Exception as e:
        logger.error(f"db_maintenance_job error: {e}", exc_info=True)
        await bot.alert_error("db_maintenance_job", e)
