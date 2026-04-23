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

# ── FIXED: Use a proper asyncio.Lock-guarded set so tasks can safely
#           claim IDs before yielding control.
_in_progress_ids: set[int]   = set()
_in_progress_lock: asyncio.Lock = asyncio.Lock()

_alerted_hot_threads: dict[int, tuple[int, datetime]] = {}
_triage_cycle_counter: int   = 0

# ── FIXED: Reduced to 4 concurrent AI batches (2 models x 11 RPM = 22 RPM total capacity)
_AI_SEMAPHORE: asyncio.Semaphore = asyncio.Semaphore(4)

_META_PATTERNS = re.compile(
    r"(user bertanya|tidak ada informasi|tidak disebutkan|no information|"
    r"pesan ini|pertanyaan tentang|menanyakan|mencari tahu|"
    r"meminta konfirmasi|menginformasikan bahwa)",
    re.IGNORECASE,
)


# ── Queue triage ───────────────────────────────────────────────────────────────

async def _auto_triage_queue() -> None:
    """Automated noise clearance for the message queue.

    Triggers 'emergency mode' if queue size exceeds threshold, aggressively 
    discarding low-signal messages to preserve processing bandwidth.
    """
    global _queue_emergency_mode

    queue = await db.get_queue_size()
    if queue < 20:
        if _queue_emergency_mode:
            _queue_emergency_mode = False
            logger.info("Queue Surgeon: pressure normalized.")
        return

    _queue_emergency_mode = queue > 100
    triage_limit = min(queue, 2000)
    logger.warning(f"Queue Surgeon: {queue} unprocessed — triaging up to {triage_limit}...")

    if not db.conn:
        return

    async with _in_progress_lock:
        current_in_progress = frozenset(_in_progress_ids)

    async with db.conn.execute(
        "SELECT id, text, reply_to_msg_id FROM messages "
        "WHERE processed=0 ORDER BY id ASC LIMIT ?",
        (triage_limit + 500,)
    ) as cur:
        rows = await cur.fetchall()

    discard_ids: list[int] = []
    for r in rows:
        if r['id'] in current_in_progress:
            continue
        text = r['text'] or ''
        
        # FIX: replies are no longer unconditionally preserved.
        # A reply with no promo signal is still noise.
        if not text:
            continue
        if not gemini._is_worth_checking(text):
            discard_ids.append(r['id'])
        if len(discard_ids) >= triage_limit:
            break

    if discard_ids:
        await db.mark_batch_processed(discard_ids)
        logger.info(f"Triage discarded {len(discard_ids)} noise msgs ({queue - len(discard_ids)} remain).")


# ── Processing loop ────────────────────────────────────────────────────────────

async def processing_loop() -> None:
    """Main background loop for message analysis and promotion extraction."""
    global _triage_cycle_counter, _queue_emergency_mode

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
        ai_start = time.monotonic()
        
        # SCOPE FIX: Semaphore only for the actual AI call
        async with _AI_SEMAPHORE:
            shared._active_ai_tasks += 1
            try:
                promos = await gemini.process_batch(msgs, db)
            except TimeoutError:
                logger.warning(f"AI rate limits exhausted — requeuing {len(msgs)} msgs for later.")
                return
            except Exception as e:
                logger.error(f"AI processing error: {e}", exc_info=True)
                await db.increment_ai_failure_count(msg_ids)
                return
            finally:
                shared._active_ai_tasks -= 1

        ai_duration = time.monotonic() - ai_start

        if promos is None:
            logger.warning(f"AI returned None — incrementing failure count for {len(msgs)} msgs.")
            await db.increment_ai_failure_count(msg_ids)
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

                    # ── FIXED: Reject obviously bad extractions early ──────
                    brand_norm = normalize_brand(p.brand)
                    summary_stripped = (p.summary or "").strip()

                    # Never alert "Unknown" brand with vague summary
                    if brand_norm == "Unknown" and len(summary_stripped) < 30:
                        continue

                    # Never alert meta-commentary
                    if _META_PATTERNS.search(summary_stripped):
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
                    age_sec    = (now_utc - msg_time).total_seconds()
                    
                    p.queue_time = age_sec - ai_duration
                    p.ai_time    = ai_duration

                    if p.status != 'expired' and age_sec < 7200:
                        brand_key  = normalize_brand(p.brand)
                        confidence = _score_confidence(p, m, recently_alerted_brands)

                        if confidence >= 45:
                            await db.save_pending_alert(
                                brand_key, p.model_dump_json(), tg_link, m['timestamp'],
                                source='ai', commit=False # commit=False + single commit at end
                            )
                            async with _recent_alerts_lock:
                                _recent_alerts_history.append({
                                    "brand":   brand_key,
                                    "summary": p.summary,
                                })
                            recently_alerted_brands.add(brand_key.lower())
                        else:
                            if brand_norm == "Unknown":
                                continue # drop silently

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
                                except (json.JSONDecodeError, TypeError, ValueError):
                                    texts = []
                                if snippet and snippet not in texts:
                                    texts.append(snippet)
                                
                                await db.conn.execute(
                                    "UPDATE pending_confirmations "
                                    "SET corroborations=corroborations+1, corroboration_texts=? WHERE id=?",
                                    (json.dumps(texts), existing['id'])
                                )
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

                # Commit all pending_alerts and confirmations in one shot
                if db.conn:
                    await db.conn.commit()

                # Trigger flush for any new pending alerts
                t = shared.get_buffer_flush_task()
                if t is None or t.done():
                    shared.set_buffer_flush_task(
                        asyncio.create_task(_flush_alert_buffer(delay=0.6))
                    )

                await db.save_promos_batch(promos_to_save, msg_ids)
            else:
                await db.mark_batch_processed(msg_ids)
        except Exception as e:
            logger.error(f"Post-AI processing error: {e}", exc_info=True)
            await db.mark_batch_processed(msg_ids)
        finally:
            async with _in_progress_lock:
                _in_progress_ids.difference_update(msg_ids)

        logger.debug(f"Batch completed: {len(msgs)} messages.")

    logger.info("Processing Loop started.")
    while True:
        try:
            await db.ensure_connection()

            queue_size = await db.get_queue_size()

            # ── FIXED: Triage every cycle when queue is pressured
            if queue_size > 20:
                await _auto_triage_queue()
                queue_size = await db.get_queue_size()

            # ── FIXED: Cap maximum concurrent AI tasks strictly.
            current_tasks = shared._active_ai_tasks
            if current_tasks >= 4:
                await asyncio.sleep(1.0)
                continue

            # Dynamic drain size based on pressure
            now_m = time.monotonic()
            total_used = sum(
                len([t for t in slot._calls if now_m - t < 60])
                for slot in gemini._slots.values()
            )
            total_cap    = sum(s.limit for s in gemini._slots.values())
            headroom_pct = max(0.0, 1.0 - (total_used / max(total_cap, 1)))

            if _queue_emergency_mode:
                batch_size = int(30 + 20 * headroom_pct)   # 30-50
            else:
                batch_size = int(15 + 15 * headroom_pct)    # 15-30

            # ── CORE FIX: fetch AND claim inside the same lock acquisition ──────
            combined: list[Any] = []
            async with _in_progress_lock:
                priority_raw = await db.get_unprocessed_recent(minutes=10, batch_size=batch_size + 20)
                priority = [r for r in priority_raw if r['id'] not in _in_progress_ids]

                backlog_size = max(0, batch_size - len(priority))
                if backlog_size > 0:
                    old_raw = await db.get_unprocessed_batch(batch_size=backlog_size + 20)
                    seen = {r['id'] for r in priority}
                    old_batch = [r for r in old_raw if r['id'] not in seen and r['id'] not in _in_progress_ids]
                else:
                    old_batch = []

                combined = (priority + old_batch)[:batch_size]
                if combined:
                    for r in combined:
                        _in_progress_ids.add(r['id'])

            if not combined:
                await asyncio.sleep(2)
                continue

            # PRE-FILTER OPTIMIZATION: 
            to_ai = []
            noise_ids = []
            for r in combined:
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
                async with _in_progress_lock:
                    _in_progress_ids.difference_update(noise_ids)
                logger.debug(f"🧹 Filtered {len(noise_ids)} noise messages from batch.")

            if not to_ai:
                await asyncio.sleep(0.5)
                continue

            logger.info(f"🧬 Spawning batch: {len(to_ai)} msgs "
                        f"(tasks={shared._active_ai_tasks}, headroom={headroom_pct:.0%}, queue={queue_size})")

            # ONE task per iteration. Sleep briefly so task registers in _active_ai_tasks.
            asyncio.create_task(process_one_batch(to_ai))
            await asyncio.sleep(0.3)
        except Exception as e:
            logger.error(f"processing_loop error: {e}", exc_info=True)
            try:
                await bot.alert_error("processing_loop", e)
            except Exception:
                pass
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
    asyncio.create_task(processing_loop())
    await listener.start()
    asyncio.create_task(
        listener.sync_history(2, catchup_hours=BOOT_CATCHUP_WINDOW / 3600)
    )

    # ── Scheduled jobs ─────────────────────────────────────────────────────────
    # Adaptive Jitter: varied timing to avoid patterns and 'rush hour' spikes
    scheduler.add_job(
        jobs.image_processing_job, "interval", minutes=5,
        id="images", args=[db, gemini, listener], jitter=30
    )
    scheduler.add_job(
        jobs.brewing_digest_job, "cron", minute=0, hour="0,1,5-23",
        id="brewing_digest", args=[bot]
    )
    scheduler.add_job(
        jobs.hourly_digest_job, "cron", minute=1, hour="0,1,5-23",
        id="digest", args=[db, gemini, bot, WIB], jitter=60 # 01:00–01:01
    )
    scheduler.add_job(
        jobs.midnight_digest_job, "cron", hour=5, minute=0,
        id="midnight_digest", args=[db, gemini, bot], jitter=300 # 05:00–05:05
    )
    scheduler.add_job(
        jobs.halfhour_digest_job, "cron", minute="14,44",
        id="halfhour_digest", args=[db, gemini, bot, WIB], jitter=240 # 14-18 & 44-48
    )
    scheduler.add_job(
        jobs.heartbeat_job, "cron", minute=30,
        id="heartbeat", args=[db, gemini, bot, WIB], jitter=120 # 30-32
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
        if bot.app.updater and bot.app.updater.running:
            await bot.app.updater.stop()
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
