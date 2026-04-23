"""main.py — TanyaDFBot Orchestrator.

Central coordination for message ingestion, AI processing, scheduled jobs, 
and alert broadcasting.
"""

import asyncio
import html
import json
import uuid
import re
import sys
import time
import logging
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Any, Sequence

from apscheduler.schedulers.asyncio import AsyncIOScheduler
import pytz

from db import Database, normalize_brand
from processor import GeminiProcessor, PromoExtraction, TrendItem
from listener import TelethonListener
from bot import TelegramBot
from config import Config
import jobs
import shared

from shared import (
    db, gemini, bot, listener,
    _recent_alerts_history, _recent_alerts_lock,
    _parse_ts,
    _alerted_aman_parents_deque, _aman_lock,
    _BRAND_KEYWORDS, _make_tg_link, _guess_brand,
    _flush_alert_buffer, _score_confidence,
    _listener_reconnecting, _last_trend_alert, _last_spike_alert,
    _reconnect_listener
)
from utils import _esc

logger = logging.getLogger(__name__)

listener: TelethonListener = TelethonListener(db)
bot: TelegramBot           = TelegramBot(db, gemini)

# Assign to shared for cross-module access
shared.listener = listener
shared.bot = bot

WIB: pytz.BaseTzInfo = pytz.timezone("Asia/Jakarta")
scheduler: AsyncIOScheduler = AsyncIOScheduler(timezone=WIB)

BOOT_TIME: datetime            = datetime.now(timezone.utc)
BOOT_CATCHUP_WINDOW: int       = 3600  # extended dynamically in main()

_queue_emergency_mode: bool  = False
_in_progress_ids: set[int]   = set()
_alerted_hot_threads: dict[int, tuple[int, datetime]] = {}
_triage_cycle_counter: int   = 0
_AI_SEMAPHORE: asyncio.Semaphore = asyncio.Semaphore(8)  # max 8 concurrent AI batches

# ── Queue triage ───────────────────────────────────────────────────────────────

async def _auto_triage_queue() -> None:
    """Automated noise clearance for the message queue.

    Triggers 'emergency mode' if queue size exceeds threshold, aggressively 
    discarding low-signal messages to preserve processing bandwidth.
    """
    global _queue_emergency_mode

    queue = await db.get_queue_size()
    if queue < 30:
        if _queue_emergency_mode:
            _queue_emergency_mode = False
            logger.info("Queue Surgeon: pressure normalized.")
        return

    _queue_emergency_mode = True
    triage_limit = min(queue, 1000)
    logger.warning(f"Queue Surgeon: {queue} unprocessed — triaging up to {triage_limit}...")

    if not db.conn:
        return

    async with db.conn.execute(
        "SELECT id, text, reply_to_msg_id FROM messages "
        "WHERE processed=0 ORDER BY id ASC LIMIT ?",
        (triage_limit + 200,)
    ) as cur:
        rows = await cur.fetchall()

    discard_ids: list[int] = []
    for r in rows:
        if r['id'] in _in_progress_ids:
            continue
        text = r['text'] or ''
        # Replies always worth checking (parent context might have signal)
        if r['reply_to_msg_id']:
            continue
        if gemini._is_worth_checking(text):
            continue
        discard_ids.append(r['id'])
        if len(discard_ids) >= triage_limit:
            break

    if discard_ids:
        await db.mark_batch_processed(discard_ids)
        logger.info(f"Triage discarded {len(discard_ids)} noise msgs ({queue - len(discard_ids)} remain).")


async def commit_worker() -> None:
    """Background task that periodically commits database transactions.

    Reduces disk I/O by batching multiple writes into a single commit cycle.
    """
    logger.info("🗄️ Commit Worker started.")
    while not shared.get_stop_event().is_set():
        try:
            await asyncio.sleep(3.0)
            if db.conn:
                await db.conn.commit()
        except Exception as e:
            logger.error(f"commit_worker error: {e}")
            await asyncio.sleep(1)


# ── Processing loop ────────────────────────────────────────────────────────────

async def processing_loop() -> None:
    """Main background loop for message analysis and promotion extraction."""
    global _triage_cycle_counter

    # Seed dedup history from DB
    recent_promos = await db.get_recent_alert_brands(hours=6, limit=300)
    async with _recent_alerts_lock:
        for rp in recent_promos:
            _recent_alerts_history.append({
                "brand":   normalize_brand(rp['brand']),
                "summary": rp['summary'],
            })

    async def process_one_batch(msgs: Sequence[dict[str, Any]]) -> None:
        """Processes a single batch of messages.

        Args:
            msgs: Sequence of message records to analyze.
        """
        msg_ids = [m['id'] for m in msgs]
        
        # SCOPE FIX: Semaphore only for the actual AI call
        async with _AI_SEMAPHORE:
            shared._active_ai_tasks += 1
            ai_start = time.monotonic()
            try:
                promos = await gemini.process_batch(msgs, db)
            except Exception as e:
                logger.error(f"AI processing error: {e}", exc_info=True)
                _in_progress_ids.difference_update(msg_ids)
                shared._active_ai_tasks -= 1
                return
            finally:
                ai_duration = time.monotonic() - ai_start
                shared._active_ai_tasks -= 1

        if promos is None:
            logger.warning(f"AI returned None — incrementing failure count for {len(msgs)} msgs.")
            await db.increment_ai_failure_count(msg_ids)
            _in_progress_ids.difference_update(msg_ids)
            return

        try:
            async with _recent_alerts_lock:
                history_snapshot = list(_recent_alerts_history)
                filtered = await gemini.filter_duplicates(promos, history_snapshot)

            if filtered:
                promos_to_save: list[tuple[int, PromoExtraction, str]] = []
                now_utc        = datetime.now(timezone.utc)
                msg_id_map     = {m['id']: m for m in msgs}

                # Pre-calculate recently alerted brands for confidence scoring
                recently_alerted_brands = {
                    normalize_brand(r.get('brand', '')).lower()
                    for r in history_snapshot[-20:]
                }

                for p in filtered:
                    m = msg_id_map.get(p.original_msg_id)
                    if not m:
                        continue

                    # Supplement links from raw text
                    urls = re.findall(r'(https?://[^\s>]+)', m['text'] or "")
                    seen = set(p.links)
                    for url in urls:
                        u = url.strip('.,()[]"\'')
                        if 't.me' not in u and 'telegram.me' not in u and u not in seen:
                            p.links.append(u)
                            seen.add(u)
                    p.links = p.links[:3]

                    tg_link    = _make_tg_link(m['chat_id'], m['tg_msg_id'])
                    promos_to_save.append((m['id'], p, tg_link))

                    msg_time   = _parse_ts(m['timestamp'])
                    now_utc    = datetime.now(timezone.utc)
                    age_sec    = (now_utc - msg_time).total_seconds()
                    
                    p.queue_time = (now_utc - msg_time).total_seconds() - ai_duration
                    p.ai_time    = ai_duration

                    if p.status != 'expired' and age_sec < 7200:
                        brand_key  = normalize_brand(p.brand)
                        confidence = _score_confidence(p, m, recently_alerted_brands)

                        if confidence >= 55:
                            await db.save_pending_alert(
                                brand_key, p.model_dump_json(), tg_link, m['timestamp'],
                                source='ai', commit=True # commit=True to avoid flush races
                            )
                            t = shared.get_buffer_flush_task()
                            if t is None or t.done():
                                shared.set_buffer_flush_task(
                                    asyncio.create_task(_flush_alert_buffer(delay=0.6))
                                )
                            async with _recent_alerts_lock:
                                _recent_alerts_history.append({
                                    "brand":   brand_key,
                                    "summary": p.summary,
                                })
                            recently_alerted_brands.add(brand_key.lower())
                        else:
                            if not db.conn:
                                continue
                            async with db.conn.execute(
                                "SELECT id, corroboration_texts FROM pending_confirmations WHERE brand=? LIMIT 1",
                                (brand_key,)
                            ) as cur:
                                existing = await cur.fetchone()
                            
                            snippet = (m['text'] or '')[:100].strip()
                            if existing:
                                try:
                                    texts = json.loads(existing['corroboration_texts'])
                                except:
                                    texts = []
                                if snippet and snippet not in texts:
                                    texts.append(snippet)
                                
                                await db.conn.execute(
                                    "UPDATE pending_confirmations "
                                    "SET corroborations=corroborations+1, corroboration_texts=? WHERE id=?",
                                    (json.dumps(texts), existing['id'])
                                )
                                await db.conn.commit()
                            else:
                                expires_at = (
                                    datetime.now(timezone.utc) + timedelta(minutes=5)
                                ).strftime('%Y-%m-%d %H:%M:%S')
                                texts = [snippet] if snippet else []
                                await db.conn.execute(
                                    "INSERT INTO pending_confirmations "
                                    "(brand, p_data_json, tg_link, timestamp, confidence, corroboration_texts, expires_at) "
                                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                                    (brand_key, p.model_dump_json(), tg_link,
                                     m['timestamp'], confidence, json.dumps(texts), expires_at)
                                )
                                await db.conn.commit()

                await db.save_promos_batch(promos_to_save, msg_ids)
            else:
                await db.mark_batch_processed(msg_ids)
        finally:
            _in_progress_ids.difference_update(msg_ids)

        logger.debug(f"Batch completed: {len(msgs)} messages.")

    logger.info("Processing Loop started.")
    while True:
        try:
            await db.ensure_connection()
            # BUG 6 FIX: triage every 2 cycles
            _triage_cycle_counter += 1
            if _triage_cycle_counter % 2 == 0:
                await _auto_triage_queue()

            # Interleaved drain: Priority is VERY recent, history is the rest
            priority_raw = await db.get_unprocessed_recent(minutes=2, batch_size=30)
            priority = [r for r in priority_raw if r['id'] not in _in_progress_ids][:20]

            now_m = time.monotonic()
            total_used = sum(
                len([t for t in slot._calls if now_m - t < 60])
                for slot in gemini._slots.values()
            )
            total_cap    = sum(s.limit for s in gemini._slots.values())
            headroom_pct = 1.0 - (total_used / max(total_cap, 1))

            # Dynamically adjust drain size based on AI headroom and queue pressure
            queue_size = await db.get_queue_size()
            if _queue_emergency_mode or queue_size > 100:
                drain_size = int(80 + 100 * headroom_pct)   # 80–180
            else:
                drain_size = int(35 + 65 * headroom_pct)    # 35–100

            old_raw = await db.get_unprocessed_batch(batch_size=drain_size + 100)
            seen_ids = {m['id'] for m in priority}
            old_batch = [r for r in old_raw if r['id'] not in seen_ids and r['id'] not in _in_progress_ids]

            # Merge and cap
            batch = (priority + old_batch)[:drain_size]

            if not batch:
                await asyncio.sleep(2)
                continue

            # PRE-FILTER OPTIMIZATION: 
            # If a message is clearly noise, mark it processed NOW instead of sending to AI batch.
            to_ai = []
            noise_ids = []
            for r in batch:
                if gemini._is_worth_checking(r['text']):
                    to_ai.append({
                        "id":               r['id'],
                        "text":             r['text'],
                        "timestamp":        r['timestamp'],
                        "tg_msg_id":        r['tg_msg_id'],
                        "chat_id":          r['chat_id'],
                        "reply_to_msg_id":  r['reply_to_msg_id'],
                    })
                else:
                    noise_ids.append(r['id'])

            if noise_ids:
                await db.mark_batch_processed(noise_ids)
                _in_progress_ids.difference_update(noise_ids)
                logger.info(f"🧹 Filtered {len(noise_ids)} noise messages from batch.")

            if not to_ai:
                continue

            logger.info(f"🧬 [Pipeline] Sending {len(to_ai)} messages to AI "
                        f"(Headroom: {headroom_pct:.0%}, Noise: {len(noise_ids)})")

            if shared._active_ai_tasks < 8:
                asyncio.create_task(process_one_batch(to_ai))
            else:
                logger.warning(f"AI saturated ({shared._active_ai_tasks} tasks). Sleeping.")
                _in_progress_ids.difference_update([m['id'] for m in to_ai])
                await asyncio.sleep(3)
                continue

            await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"processing_loop error: {e}", exc_info=True)
            await bot.alert_error("processing_loop", e)
            await asyncio.sleep(5)

# ── Alert watchdog (runtime) ───────────────────────────────────────────────────

async def _alert_watchdog():
    """Periodically un-sticks pending_alerts rows orphaned by mid-flush crashes."""
    await db.recover_stuck_alerts()


# ── Main Entry Point ───────────────────────────────────────────────────────────

async def main() -> None:
    """Initializes and starts all application components."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Silence only the most repetitive polling logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("telegram.ext").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    # Custom logger for pipeline visibility
    pipeline_logger = logging.getLogger("pipeline")
    pipeline_logger.setLevel(logging.INFO)

    if not Config.validate():
        sys.exit(1)

    logger.info("--- TanyaDFBot Booting ---")
    await db.init()

    global BOOT_CATCHUP_WINDOW
    if not db.conn:
        logger.critical("Database connection failed.")
        sys.exit(1)

    async with db.conn.execute("SELECT MAX(timestamp) FROM messages") as cur:
        row = await cur.fetchone()
        if row and row[0]:
            last_ts = _parse_ts(row[0])
            gap     = (datetime.now(timezone.utc) - last_ts).total_seconds()
            if gap > 3600:
                BOOT_CATCHUP_WINDOW = min(int(gap) + 600, 28800)

    # Catchup logic logs
    if BOOT_CATCHUP_WINDOW > 3600:
        logger.info(f"Large gap detected: catching up {BOOT_CATCHUP_WINDOW/3600:.1f} hours.")

    # Initialize and start Bot/Updater
    await bot.app.initialize()
    await bot.app.start()
    
    # NEW: Check for pending failures that were marked fixed but not retried
    pending = await db.get_pending_failures()
    fixed_pending = [f for f in pending if f['fixed'] and not f['retried']]
    if fixed_pending:
        count = len(fixed_pending)
        logger.info(f"🔍 Found {count} fixed failures pending retry.")
        msg = (f"🔄 <b>Startup Recovery</b>\n\nFound {count} errors marked as fixed. "
               "You can retry them from their original error messages.")
        await bot.send_plain(msg, parse_mode='HTML')

    await bot.app.updater.start_polling()

    # Launch background loops
    asyncio.create_task(commit_worker())
    asyncio.create_task(processing_loop())
    await listener.start()
    asyncio.create_task(
        listener.sync_history(2, catchup_hours=BOOT_CATCHUP_WINDOW / 3600)
    )

    # ── Scheduled jobs ─────────────────────────────────────────────────────────
    # Relaxed schedule to reduce event loop congestion
    scheduler.add_job(
        jobs.image_processing_job, "interval", minutes=5,
        id="images", args=[db, gemini, listener]
    )
    scheduler.add_job(
        jobs.hourly_digest_job, "cron", minute=1, hour="0,1,5-23", # offset by 1m
        id="digest", args=[db, gemini, bot, WIB]
    )
    scheduler.add_job(
        jobs.midnight_digest_job, "cron", hour=5, minute=1, # offset by 1m
        id="midnight_digest", args=[db, gemini, bot]
    )
    scheduler.add_job(
        jobs.halfhour_digest_job, "cron", minute="16,46", # offset by 1m
        id="halfhour_digest", args=[db, gemini, bot, WIB]
    )
    scheduler.add_job(
        jobs.heartbeat_job, "cron", minute=31, # offset by 1m
        id="heartbeat", args=[db, gemini, bot, WIB]
    )
    scheduler.add_job(
        jobs.hot_thread_job, "interval", minutes=15,
        id="hot_threads",
        args=[db, gemini, listener, bot, WIB, _alerted_hot_threads]
    )
    scheduler.add_job(
        jobs.time_mention_job, "interval", minutes=5,
        id="time_mentions", args=[db, bot]
    )
    scheduler.add_job(
        jobs.trend_job, "interval", minutes=20,
        id="trend_job", args=[db, gemini, bot]
    )
    scheduler.add_job(
        jobs.spike_detection_job, "interval", minutes=5,
        id="spike", args=[db, gemini, bot]
    )
    scheduler.add_job(
        jobs.dead_promo_reaper_job, "interval", minutes=20,
        id="reaper", args=[db, bot]
    )
    scheduler.add_job(
        jobs.confirmation_gate_job, "interval", minutes=4,
        id="confirm_gate", args=[db]
    )
    scheduler.add_job(
        jobs.db_maintenance_job, "cron", hour="*/4", minute=5,
        id="db_maint", args=[db, bot]
    )
    # NEW: runtime watchdog for stuck pending_alerts
    scheduler.add_job(
        _alert_watchdog, "interval", minutes=1, id="alert_watchdog"
    )

    scheduler.start()
    logger.info("✅ TanyaDFBot Online")

    try:
        await shared.get_stop_event().wait()
        logger.info("🛑 Shutdown signal received.")
    except (KeyboardInterrupt, SystemExit):
        logger.info("🛑 Interrupted by user.")
    except Exception as e:
        logger.critical(f"🛑 Critical crash: {e}", exc_info=True)
    finally:
        scheduler.shutdown()
        await bot.app.stop()
        await bot.app.shutdown()
        await listener.client.disconnect()
        if db.conn:
            await db.conn.close()
            logger.info("🗄️ Database connection closed.")
        logger.info("👋 Shutdown complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
