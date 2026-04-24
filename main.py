"""main.py — TanyaDFBot Orchestrator.

Central coordination for message ingestion, AI processing, scheduled jobs, 
and alert broadcasting.
"""

import asyncio
import json
import re
import sys
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Sequence

from apscheduler.schedulers.asyncio import AsyncIOScheduler
import pytz

from db import normalize_brand
from processor import PromoExtraction
from listener import TelethonListener
from bot import TelegramBot
from config import Config
import jobs
import shared

from shared import (
    db, gemini, bot, listener,
    _recent_alerts_history, _recent_alerts_lock,
    _parse_ts,
    _make_tg_link, _flush_alert_buffer, _score_confidence,
    _reconnect_listener
)

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

_in_progress_ids: dict[int, float] = {}
_in_progress_lock: asyncio.Lock = asyncio.Lock()
_IN_PROGRESS_MAX_AGE_SEC: float = 130.0

_alerted_hot_threads: dict[int, tuple[int, datetime]] = {}
_triage_cycle_counter: int   = 0

_AI_MAX_INFLIGHT: int = 16
_AI_SEMAPHORE: asyncio.Semaphore = asyncio.Semaphore(_AI_MAX_INFLIGHT)

_active_spawn_tasks: set[asyncio.Task] = set()

_META_PATTERNS = re.compile(
    r"(user bertanya|tidak ada informasi|tidak disebutkan|no information|"
    r"pesan ini|pertanyaan tentang|menanyakan|mencari tahu|"
    r"meminta konfirmasi|menginformasikan bahwa)",
    re.IGNORECASE,
)


# ── Queue triage ───────────────────────────────────────────────────────────────

async def _auto_triage_queue() -> None:
    """Automated noise clearance for the message queue."""
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
        current_in_progress = frozenset(_in_progress_ids.keys())

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
        if not gemini._is_worth_checking(text):
            discard_ids.append(r['id'])
        if len(discard_ids) >= triage_limit:
            break

    if discard_ids:
        await db.mark_batch_processed(discard_ids, skip_reason="triage")
        logger.info(f"Triage discarded {len(discard_ids)} noise msgs ({queue - len(discard_ids)} remain).")


# ── Processing loop ────────────────────────────────────────────────────────────

async def processing_loop() -> None:
    """Main background loop for message analysis and promotion extraction."""
    global _triage_cycle_counter, _queue_emergency_mode

    recent_promos = await db.get_recent_alert_brands(hours=6, limit=300)
    async with _recent_alerts_lock:
        for rp in recent_promos:
            _recent_alerts_history.append({
                "brand":   normalize_brand(rp['brand']),
                "summary": rp['summary'],
            })

    shared.set_loop_tick()

    async def process_one_batch(msgs: Sequence[dict[str, Any]]) -> None:
        """Processes a single batch of messages."""
        msg_ids = [m['id'] for m in msgs]
        ai_start = time.monotonic()

        async def _release_claims() -> None:
            async with _in_progress_lock:
                for mid in msg_ids:
                    _in_progress_ids.pop(mid, None)

        shared._active_ai_tasks += 1
        promos: list[Any] | None = None
        ai_failed = False
        try:
            try:
                async with _AI_SEMAPHORE:
                    try:
                        promos = await gemini.process_batch(msgs, db)
                    except TimeoutError:
                        logger.warning(f"AI rate limits exhausted — requeuing {len(msgs)} msgs for later.")
                        ai_failed = True
                    except Exception as e:
                        logger.error(f"AI processing error: {e}", exc_info=True)
                        await db.increment_ai_failure_count(msg_ids)
                        ai_failed = True
            finally:
                shared._active_ai_tasks -= 1
        except BaseException:
            await _release_claims()
            raise

        if ai_failed:
            shared.record_ai_outcome(success=False)
            await _release_claims()
            return

        ai_duration = time.monotonic() - ai_start

        if promos is None:
            logger.warning(f"AI returned None — incrementing failure count for {len(msgs)} msgs.")
            await db.increment_ai_failure_count(msg_ids)
            shared.record_ai_outcome(success=False)
            await _release_claims()
            return

        shared.record_ai_outcome(success=True)

        try:
            async with _recent_alerts_lock:
                history_snapshot = list(_recent_alerts_history)
                filtered = await gemini.filter_duplicates(promos, history_snapshot)

            # ── Training Data Collection ──
            success_ids_ai = {p.original_msg_id for p in promos}
            ai_skip_ids = [mid for mid in msg_ids if mid not in success_ids_ai]
            logic_skip_ids = []
            filtered_ids = {p.original_msg_id for p in filtered}
            duplicate_ids = success_ids_ai - filtered_ids

            if filtered:
                promos_to_save: list[tuple[int, PromoExtraction, str]] = []
                now_utc        = datetime.now(timezone.utc)
                msg_id_map     = {m['id']: m for m in msgs}

                recently_alerted_brands = {
                    normalize_brand(r.get('brand', '')).lower()
                    for r in history_snapshot[-20:]
                }

                for p in filtered:
                    m = msg_id_map.get(p.original_msg_id)
                    if not m: continue

                    brand_norm = normalize_brand(p.brand)
                    summary_stripped = (p.summary or "").strip()

                    if brand_norm != "Unknown":
                        await shared.context_tracker.update_brand(m['chat_id'], brand_norm)

                    if brand_norm == "Unknown" and len(summary_stripped) < 30:
                        logic_skip_ids.append(m['id'])
                        continue

                    if _META_PATTERNS.search(summary_stripped):
                        logic_skip_ids.append(m['id'])
                        continue

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
                            if await shared.is_fuzzy_duplicate(brand_key, p.summary):
                                continue
                            await db.save_pending_alert(
                                brand_key, p.model_dump_json(), tg_link, m['timestamp'],
                                source='ai', commit=False 
                            )
                            t = shared.get_buffer_flush_task()
                            if t is None or t.done():
                                shared.set_buffer_flush_task(asyncio.create_task(_flush_alert_buffer(delay=0.8)))
                            async with _recent_alerts_lock:
                                _recent_alerts_history.append({"brand": brand_key, "summary": p.summary})
                            recently_alerted_brands.add(brand_key.lower())
                        else:
                            if brand_norm == "Unknown": continue
                            if not db.conn: continue
                            async with db.conn.execute(
                                "SELECT id, corroboration_texts FROM pending_confirmations WHERE brand=? LIMIT 1",
                                (brand_key,)
                            ) as cur:
                                existing = await cur.fetchone()
                            
                            snippet = (m['text'] or '')[:100].strip()
                            if existing:
                                try:
                                    texts = json.loads(existing['corroboration_texts'])
                                except: texts = []
                                if snippet and snippet not in texts: texts.append(snippet)
                                await db.conn.execute(
                                    "UPDATE pending_confirmations SET corroborations=corroborations+1, corroboration_texts=? WHERE id=?",
                                    (json.dumps(texts), existing['id'])
                                )
                            else:
                                expires_at = (datetime.now(timezone.utc) + timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S')
                                texts = [snippet] if snippet else []
                                await db.conn.execute(
                                    "INSERT INTO pending_confirmations (brand, p_data_json, tg_link, timestamp, confidence, corroboration_texts, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                    (brand_key, p.model_dump_json(), tg_link, m['timestamp'], confidence, json.dumps(texts), expires_at)
                                )

                if db.conn: await db.conn.commit()

                t = shared.get_buffer_flush_task()
                if t is None or t.done():
                    shared.set_buffer_flush_task(asyncio.create_task(_flush_alert_buffer(delay=0.6)))

                await db.save_promos_batch(promos_to_save, [p[0] for p in promos_to_save])
            
            final_skip_ids = list(set(ai_skip_ids) | set(logic_skip_ids))
            if final_skip_ids:
                await db.mark_batch_processed(final_skip_ids, skip_reason="ai_skip")
            
            if duplicate_ids:
                await db.mark_batch_processed(list(duplicate_ids))

        except Exception as e:
            logger.error(f"Post-AI processing error: {e}", exc_info=True)
            await db.mark_batch_processed(msg_ids)
        finally:
            await _release_claims()

    logger.info("Processing Loop started.")
    while True:
        try:
            shared.set_loop_tick()
            await db.ensure_connection()

            queue_size = await db.get_queue_size()
            if queue_size > 20:
                await _auto_triage_queue()
                queue_size = await db.get_queue_size()

            current_tasks = shared._active_ai_tasks
            if current_tasks >= _AI_MAX_INFLIGHT:
                await asyncio.sleep(0.2)
                continue

            circuit_wait = shared.ai_circuit_open_remaining()
            if circuit_wait > 0:
                await asyncio.sleep(min(circuit_wait, 2.0))
                continue

            now_m = time.monotonic()
            total_used = sum(len([t for t in slot._calls if now_m - t < 60]) for slot in gemini._slots.values())
            total_cap    = sum(s.limit for s in gemini._slots.values())
            headroom_pct = max(0.0, 1.0 - (total_used / max(total_cap, 1)))

            if _queue_emergency_mode:
                batch_size = int(15 + 10 * headroom_pct)
            else:
                batch_size = int(10 + 10 * headroom_pct)

            # Optimization: Fetch candidates OUTSIDE of the lock to minimize contention
            ancient_reserve = int(batch_size * 0.3)
            if queue_size > 200: backlog_reserve = int(batch_size * 0.3)
            elif queue_size > 50: backlog_reserve = int(batch_size * 0.2)
            else: backlog_reserve = 0
            priority_cap = max(1, batch_size - ancient_reserve - backlog_reserve)

            # Query DB (can be slow under load)
            ancient_raw = await db.get_unprocessed_ancient(min_age_minutes=15, batch_size=ancient_reserve + 100)
            priority_raw = await db.get_unprocessed_recent(minutes=10, batch_size=priority_cap + 50)

            combined: list[Any] = []
            async with _in_progress_lock:
                ancient = [r for r in ancient_raw if r['id'] not in _in_progress_ids][:ancient_reserve]
                seen_ids = {r['id'] for r in ancient}
                priority = [r for r in priority_raw if r['id'] not in _in_progress_ids and r['id'] not in seen_ids][:priority_cap]

                filled = len(ancient) + len(priority)
                backlog_needed = batch_size - filled
                if backlog_needed > 0:
                    old_raw = await db.get_unprocessed_batch(batch_size=backlog_needed + 200)
                    seen_ids.update(r['id'] for r in priority)
                    old_batch = [r for r in old_raw if r['id'] not in seen_ids and r['id'] not in _in_progress_ids][:backlog_needed]
                else: old_batch = []

                combined = ancient + priority + old_batch

                if combined:
                    claim_ts = time.monotonic()
                    for r in combined: _in_progress_ids[r['id']] = claim_ts

                if ancient:
                    try:
                        oldest_age = max((datetime.now(timezone.utc) - _parse_ts(r['timestamp'])).total_seconds() for r in ancient)
                        shared._last_observed_ancient_age = oldest_age
                    except: pass

            if not combined:
                await asyncio.sleep(2)
                continue

            # PRE-FILTER OPTIMIZATION: 
            to_ai = []
            regex_noise_ids = []
            fasttext_noise_ids = []
            
            # Level 2 Safeguard: Protect strong signals from model error
            _PROTECTED_SIGNALS = re.compile(
                r'\b(jsm|psm|aman|on|jp|work|luber|pecah|cair|nyala|'
                r'berhasil|lancar|masuk|murce|murmer|big|badut|syarat|snk|serbu)\b', 
                re.IGNORECASE
            )
            
            candidates = []
            for r in combined:
                text = r['text'] or ""
                has_photo = bool(r['has_photo'])
                # Tier 1: Dumb Regex (Zero cost)
                if not gemini._is_worth_checking(text, has_photo):
                    regex_noise_ids.append(r['id'])
                    continue
                
                # Safeguard: ALWAYS pass holy-grail keywords or photos to AI
                if has_photo or _PROTECTED_SIGNALS.search(text):
                    to_ai.append({
                        "id":               r['id'],
                        "text":             r['text'],
                        "timestamp":        r['timestamp'],
                        "tg_msg_id":        r['tg_msg_id'],
                        "chat_id":          r['chat_id'],
                        "reply_to_msg_id":  r['reply_to_msg_id'],
                        "has_photo":        has_photo,
                    })
                    continue

                candidates.append(r)

            # Tier 2: FastText Batch Semantic Filter
            if candidates:
                texts = [c['text'] or "" for c in candidates]
                results = await shared.classify_batch(texts)
                
                for r, (label, confidence) in zip(candidates, results):
                    if label == "__label__JUNK" and confidence >= 0.88:
                        fasttext_noise_ids.append(r['id'])
                    else:
                        to_ai.append({
                            "id":               r['id'],
                            "text":             r['text'],
                            "timestamp":        r['timestamp'],
                            "tg_msg_id":        r['tg_msg_id'],
                            "chat_id":          r['chat_id'],
                            "reply_to_msg_id":  r['reply_to_msg_id'],
                            "has_photo":        r['has_photo'],
                        })

            if regex_noise_ids:
                await db.mark_batch_processed(regex_noise_ids, skip_reason="regex")
                async with _in_progress_lock:
                    for nid in regex_noise_ids: _in_progress_ids.pop(nid, None)

            if fasttext_noise_ids:
                await db.mark_batch_processed(fasttext_noise_ids, skip_reason="fasttext")
                async with _in_progress_lock:
                    for nid in fasttext_noise_ids: _in_progress_ids.pop(nid, None)
                logger.debug(f"🛡️ Traffic Cop: Filtered {len(fasttext_noise_ids)} messages (reason: fasttext).")

            if not to_ai:
                await asyncio.sleep(0.5)
                continue

            logger.info(f"🧬 Spawning batch: {len(to_ai)} msgs (tasks={shared._active_ai_tasks}, headroom={headroom_pct:.0%}, queue={queue_size})")
            shared.mark_batch_spawned()
            _spawned_task = asyncio.create_task(process_one_batch(to_ai))
            _active_spawn_tasks.add(_spawned_task)
            _spawned_task.add_done_callback(_active_spawn_tasks.discard)
            await asyncio.sleep(0.05)
        except Exception as e:
            logger.error(f"processing_loop error: {e}", exc_info=True)
            try: await bot.alert_error("processing_loop", e)
            except: pass
            await asyncio.sleep(5)

# ── Runtime self-heal watchdogs ────────────────────────────────────────────────

async def _alert_watchdog():
    await db.recover_stuck_alerts()

async def _active_ai_tasks_reconciler() -> None:
    live_tasks = sum(1 for t in _active_spawn_tasks if not t.done())
    counter = shared._active_ai_tasks
    if counter > live_tasks:
        drift = counter - live_tasks
        logger.warning(f"⚠️ _active_ai_tasks counter drift: counter={counter} live_tasks={live_tasks} — reconciling down by {drift}")
        shared._active_ai_tasks = max(0, shared._active_ai_tasks - drift)

async def _in_progress_reaper() -> None:
    now_m = time.monotonic()
    evicted: list[int] = []
    async with _in_progress_lock:
        for mid, claim_ts in list(_in_progress_ids.items()):
            if now_m - claim_ts > _IN_PROGRESS_MAX_AGE_SEC:
                evicted.append(mid)
                _in_progress_ids.pop(mid, None)
    if evicted: logger.warning(f"⚠️ In-progress reaper freed {len(evicted)} stuck claims. Messages will be retried.")

_last_loop_alert_ts: float = 0.0
_LOOP_ALERT_COOLDOWN_SEC: float = 600.0

async def _loop_heartbeat_watchdog() -> None:
    global _last_loop_alert_ts
    last_tick = shared.get_loop_tick()
    if last_tick is None: return
    age = time.monotonic() - last_tick
    if age < 90.0: return
    now_m = time.monotonic()
    if now_m - _last_loop_alert_ts < _LOOP_ALERT_COOLDOWN_SEC: return
    _last_loop_alert_ts = now_m
    logger.error(f"💔 processing_loop heartbeat stale: last tick {age:.0f}s ago")
    try: await bot.alert_error("processing_loop_heartbeat", RuntimeError(f"processing_loop stalled: no tick for {age:.0f}s"))
    except: pass

_LISTENER_WATCHDOG_QUIET_SEC: float = 60.0
_last_listener_reconnect_ts: float = 0.0
_LISTENER_RECONNECT_COOLDOWN_SEC: float = 45.0
_listener_reconnect_attempts: int = 0

async def _listener_health_watchdog() -> None:
    global _last_listener_reconnect_ts, _listener_reconnect_attempts
    try:
        ingest_age = shared.seconds_since_last_ingest()
        if ingest_age is not None and ingest_age < _LISTENER_WATCHDOG_QUIET_SEC:
            if _listener_reconnect_attempts > 0: _listener_reconnect_attempts = 0
            return
        try: mtproto_connected = bool(shared.listener.client.is_connected())
        except: mtproto_connected = False
        if mtproto_connected: return
        backoff = min(15.0 * (_listener_reconnect_attempts + 1), _LISTENER_RECONNECT_COOLDOWN_SEC)
        now_m = time.monotonic()
        if now_m - _last_listener_reconnect_ts < backoff: return
        if shared._listener_reconnecting: return
        _last_listener_reconnect_ts = now_m
        _listener_reconnect_attempts += 1
        logger.warning(f"🔌 Listener watchdog: socket disconnected — forcing reconnect.")
        from shared import _reconnect_listener
        asyncio.create_task(_reconnect_listener(gap_minutes=0.5))
    except Exception as e: logger.error(f"listener_health_watchdog error: {e}", exc_info=True)


# ── Main Entry Point ───────────────────────────────────────────────────────────

async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("telegram.ext").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    if not Config.validate(): sys.exit(1)

    logger.info("--- TanyaDFBot Booting ---")
    await db.init()
    await shared.load_classifier("model.ftz")

    global BOOT_CATCHUP_WINDOW
    if not db.conn:
        logger.critical("Database connection failed.")
        sys.exit(1)

    async with db.conn.execute("SELECT MAX(timestamp) FROM messages") as cur:
        row = await cur.fetchone()
        if row and row[0]:
            last_ts = _parse_ts(row[0])
            gap     = (datetime.now(timezone.utc) - last_ts).total_seconds()
            if gap > 3600: BOOT_CATCHUP_WINDOW = min(int(gap) + 600, 28800)

    if BOOT_CATCHUP_WINDOW > 3600: logger.info(f"Large gap detected: catching up {BOOT_CATCHUP_WINDOW/3600:.1f} hours.")
    await bot.app.initialize()
    await bot.app.start()
    pending = await db.get_pending_failures()
    fixed_pending = [f for f in pending if f['fixed'] and not f['retried']]
    if fixed_pending:
        count = len(fixed_pending)
        msg = (f"🔄 <b>Startup Recovery</b>\n\nFound {count} errors marked as fixed.")
        await bot.send_plain(msg, parse_mode='HTML')

    await bot.app.updater.start_polling()
    asyncio.create_task(processing_loop())
    await listener.start()
    asyncio.create_task(_reconnect_listener(BOOT_CATCHUP_WINDOW / 60))

    # ── Scheduled jobs ─────────────────────────────────────────────────────────
    scheduler.add_job(jobs.image_processing_job, "interval", minutes=5, id="images", args=[db, gemini, listener], jitter=30)
    scheduler.add_job(jobs.brewing_digest_job, "cron", minute=0, hour="0,1,5-23", id="brewing_digest", args=[bot])
    scheduler.add_job(jobs.hourly_digest_job, "cron", minute=1, hour="0,1,5-23", id="digest", args=[db, gemini, bot, WIB], jitter=60)
    scheduler.add_job(jobs.midnight_digest_job, "cron", hour=5, minute=0, id="midnight_digest", args=[db, gemini, bot], jitter=300)
    scheduler.add_job(jobs.halfhour_digest_job, "cron", minute="14,44", id="halfhour_digest", args=[db, gemini, bot, WIB], jitter=240)
    scheduler.add_job(jobs.heartbeat_job, "cron", minute=30, id="heartbeat", args=[db, gemini, bot, WIB], jitter=120)
    scheduler.add_job(jobs.hot_thread_job, "interval", minutes=15, id="hot_threads", args=[db, gemini, listener, bot, WIB, _alerted_hot_threads])
    scheduler.add_job(jobs.time_mention_job, "interval", minutes=5, id="time_mentions", args=[db, bot])
    scheduler.add_job(jobs.trend_job, "interval", minutes=20, id="trend_job", args=[db, gemini, bot])
    scheduler.add_job(jobs.spike_detection_job, "interval", minutes=5, id="spike", args=[db, gemini, bot])
    scheduler.add_job(jobs.dead_promo_reaper_job, "interval", minutes=20, id="reaper", args=[db, bot])
    scheduler.add_job(jobs.confirmation_gate_job, "interval", minutes=1, id="confirm_gate", args=[db])
    scheduler.add_job(jobs.db_maintenance_job, "cron", hour="*/4", minute=5, id="db_maint", args=[db, bot])
    scheduler.add_job(_alert_watchdog, "interval", minutes=1, id="alert_watchdog")
    scheduler.add_job(_in_progress_reaper, "interval", minutes=1, id="in_progress_reaper")
    scheduler.add_job(_active_ai_tasks_reconciler, "interval", minutes=1, id="ai_tasks_reconciler")
    scheduler.add_job(_loop_heartbeat_watchdog, "interval", seconds=30, id="loop_heartbeat")
    scheduler.add_job(_listener_health_watchdog, "interval", seconds=30, id="listener_health")
    scheduler.add_job(jobs.time_reminder_job, "interval", minutes=1, id="time_reminder", args=[db, bot, WIB])

    scheduler.start()
    logger.info("✅ TanyaDFBot Online")

    try: await shared.get_stop_event().wait()
    except (KeyboardInterrupt, SystemExit): pass
    except Exception as e: logger.critical(f"🛑 Critical crash: {e}", exc_info=True)
    finally:
        scheduler.shutdown()
        if bot.app.updater and bot.app.updater.running: await bot.app.updater.stop()
        await bot.app.stop()
        await bot.app.shutdown()
        await listener.client.disconnect()
        if db.conn: await db.conn.close()
        logger.info("👋 Shutdown complete.")

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: pass
