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

# ── FIXED: Use a proper asyncio.Lock-guarded dict so tasks can safely
#           claim IDs before yielding control. The value is the monotonic
#           timestamp at claim time, used by the stuck-claim reaper below.
_in_progress_ids: dict[int, float] = {}
_in_progress_lock: asyncio.Lock = asyncio.Lock()

# Claims older than this are considered stuck (AI crashed silently, task was
# cancelled, etc.) and get evicted so the underlying message can be retried.
# NOTE: must be larger than the AI-call timeout in processor._AI_CALL_TIMEOUT_SEC
# (90s) so a normal slow call doesn't get double-claimed by the reaper. 120s is
# the sweet spot: recovers from a hung claim within ~2 min of detection.
_IN_PROGRESS_MAX_AGE_SEC: float = 120.0

_alerted_hot_threads: dict[int, tuple[int, datetime]] = {}
_triage_cycle_counter: int   = 0

# AI concurrency cap. Two Gemma models × 11 RPM = 22 RPM aggregate, and the
# `_ModelSlot` limiter in `processor.py` already enforces that rate cap cleanly.
# So this semaphore just exists to cap in-flight asyncio tasks (memory / event
# loop scheduling pressure) — it is NOT the rate limit. Raised from 4 → 16 so
# the pipeline can actually saturate 22 RPM during bursts. At ~1GB RAM / 2% CPU
# budget, 16 in-flight batches is comfortably safe.
_AI_MAX_INFLIGHT: int = 16
_AI_SEMAPHORE: asyncio.Semaphore = asyncio.Semaphore(_AI_MAX_INFLIGHT)

# Strong references to spawned `process_one_batch` tasks. Without this set,
# asyncio can GC a task whose only reference is the scheduler's weak queue
# mid-flight, leaving claims stuck in `_in_progress_ids` until the reaper.
_active_spawn_tasks: set[asyncio.Task] = set()

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

    # Mark loop alive for the heartbeat watchdog below.
    shared.set_loop_tick()

    async def process_one_batch(msgs: Sequence[dict[str, Any]]) -> None:
        """Processes a single batch of messages.

        Args:
            msgs: Sequence of message records to analyze.
        """
        msg_ids = [m['id'] for m in msgs]
        ai_start = time.monotonic()

        async def _release_claims() -> None:
            """Clear this batch from _in_progress_ids so the rows are fetchable again.

            MUST be called on every exit path — leaving IDs stuck here is what
            caused the major-latency bug: stuck IDs at the head of the queue
            blocked `get_unprocessed_batch` from ever surfacing fresher backlog.
            """
            async with _in_progress_lock:
                for mid in msg_ids:
                    _in_progress_ids.pop(mid, None)

        # Count this batch as in-flight from the moment it exists (including
        # time spent waiting on the semaphore), so the processing_loop's
        # spawn-gate accurately reflects total pending work — not just tasks
        # that have already acquired the semaphore.
        shared._active_ai_tasks += 1
        # Outer try/finally catches CancelledError / BaseException too, so a
        # task that gets cancelled mid-AI-call still releases its claims.
        # (Pre-fix: reaper had to mop up stuck claims every 5 min, causing
        # multi-minute silences during API hiccups.)
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
            # CancelledError or any other BaseException: release claims so
            # the rows become eligible for re-processing, then re-raise.
            await _release_claims()
            raise

        if ai_failed:
            await _release_claims()
            return

        ai_duration = time.monotonic() - ai_start

        if promos is None:
            logger.warning(f"AI returned None — incrementing failure count for {len(msgs)} msgs.")
            await db.increment_ai_failure_count(msg_ids)
            await _release_claims()
            return

        try:
            # Atomically filter duplicates AND pre-reserve the surviving keys
            # in `_recent_alerts_history` in a single critical section. This
            # fixes a cross-batch race where N parallel AI batches could each
            # read the same stale history, conclude "no duplicate", and all
            # fire near-identical alerts seconds apart (the 4x Sayurbox /
            # "Daging Sapi Rendang" alerts reported in production).
            #
            # The reservation is optimistic: we append every survivor even
            # before we know its final confidence. If it ends up being
            # dropped (low confidence → pending_confirmations, or rejected
            # meta-commentary), leaving it in history is harmless — it just
            # prevents re-extracting the same phrasing in the next batch,
            # which is the correct behavior for dedup purposes.
            async with _recent_alerts_lock:
                history_snapshot = list(_recent_alerts_history)
                filtered = await gemini.filter_duplicates(promos, history_snapshot)
                for _p in filtered:
                    _recent_alerts_history.append({
                        "brand":   normalize_brand(_p.brand),
                        "summary": _p.summary,
                    })

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
                            # History was already appended inside the dedup
                            # critical section above — don't double-append.
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
            await _release_claims()

        logger.debug(f"Batch completed: {len(msgs)} messages.")

    logger.info("Processing Loop started.")
    while True:
        try:
            shared.set_loop_tick()
            await db.ensure_connection()

            queue_size = await db.get_queue_size()

            # ── FIXED: Triage every cycle when queue is pressured
            if queue_size > 20:
                await _auto_triage_queue()
                queue_size = await db.get_queue_size()

            # Cap maximum concurrent AI tasks. The real rate limit is enforced
            # by `_ModelSlot` in processor.py (22 RPM total). This cap exists
            # only to bound event-loop / memory pressure from too many
            # simultaneously-spawned batches. When we're at the cap, we yield
            # briefly rather than sleeping a full second so drain resumes ASAP.
            current_tasks = shared._active_ai_tasks
            if current_tasks >= _AI_MAX_INFLIGHT:
                await asyncio.sleep(0.2)
                continue

            # Dynamic drain size based on RPM pressure
            now_m = time.monotonic()
            total_used = sum(
                len([t for t in slot._calls if now_m - t < 60])
                for slot in gemini._slots.values()
            )
            total_cap    = sum(s.limit for s in gemini._slots.values())
            headroom_pct = max(0.0, 1.0 - (total_used / max(total_cap, 1)))

            # Smaller batches → lower per-message latency + more parallelism.
            # Sweet spot for gemma-4-31b-it: 15–25 msgs per call (~3–6s each).
            # Under emergency pressure we can size up a bit for throughput, but
            # never so large that a single batch stalls the drain pipeline.
            # Batch-size cap at 25 is deliberate: Gemini's Gemma-4 models
            # time out erratically on very large prompts (observed ≥50-msg
            # batches hanging forever with no response), so we keep any
            # single batch small enough that (a) it completes fast under
            # normal latency, and (b) a stall on one batch doesn't take
            # down a huge chunk of the queue.
            if _queue_emergency_mode:
                batch_size = int(15 + 10 * headroom_pct)   # 15–25
            else:
                batch_size = int(10 + 10 * headroom_pct)   # 10–20

            # ── CORE FIX: fetch AND claim inside the same lock acquisition ──────
            #
            # 3-tier queue policy — designed so NO row can sit >~3 min under
            # any realistic queue shape:
            #
            #   ancient_reserve  — rows >= 15 min old. Always claim ≥ 30% of
            #                      the batch from the very tail of the queue.
            #                      This is the anti-starvation guarantee: a
            #                      row that got stuck for 15 min skips ahead
            #                      of fresher backlog and gets processed.
            #   priority         — rows < 10 min old, FIFO within that window.
            #   backlog_reserve  — rows between 10–15 min old, catching the
            #                      middle tier so the handoff is smooth.
            #
            # Previously a row that missed the 10 min window drained at 25%
            # of the batch, and the `_last_observed_ancient_age` telemetry
            # showed rows occasionally sitting 20+ min. The 30% ancient
            # reserve caps the observed age tightly.
            combined: list[Any] = []
            async with _in_progress_lock:
                ancient_reserve = int(batch_size * 0.3)  # always 30%
                if queue_size > 200:
                    backlog_reserve = int(batch_size * 0.3)
                elif queue_size > 50:
                    backlog_reserve = int(batch_size * 0.2)
                else:
                    backlog_reserve = 0

                priority_cap = max(1, batch_size - ancient_reserve - backlog_reserve)

                # Tier 1: ancient rows (>15 min old) — always first to claim
                ancient_raw = await db.get_unprocessed_ancient(
                    min_age_minutes=15, batch_size=ancient_reserve + 100,
                )
                ancient = [
                    r for r in ancient_raw
                    if r['id'] not in _in_progress_ids
                ][:ancient_reserve]

                # Tier 2: fresh priority (<10 min old)
                priority_raw = await db.get_unprocessed_recent(
                    minutes=10, batch_size=priority_cap + 20,
                )
                seen_ids = {r['id'] for r in ancient}
                priority = [
                    r for r in priority_raw
                    if r['id'] not in _in_progress_ids and r['id'] not in seen_ids
                ][:priority_cap]

                # Tier 3: middle backlog (10–15 min old or whatever's left)
                filled = len(ancient) + len(priority)
                backlog_needed = batch_size - filled
                if backlog_needed > 0:
                    # Oversample generously so any residual stuck rows in
                    # `_in_progress_ids` can't drown out fresher backlog.
                    old_raw = await db.get_unprocessed_batch(batch_size=backlog_needed + 200)
                    seen_ids.update(r['id'] for r in priority)
                    old_batch = [
                        r for r in old_raw
                        if r['id'] not in seen_ids and r['id'] not in _in_progress_ids
                    ][:backlog_needed]
                else:
                    old_batch = []

                combined = ancient + priority + old_batch
                if combined:
                    claim_ts = time.monotonic()
                    for r in combined:
                        _in_progress_ids[r['id']] = claim_ts

                # Telemetry — max observed claim age so /diag can report it
                if ancient:
                    try:
                        oldest_age = max(
                            (datetime.now(timezone.utc)
                             - _parse_ts(r['timestamp'])).total_seconds()
                            for r in ancient
                        )
                        shared._last_observed_ancient_age = oldest_age
                    except Exception:
                        pass

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
                    for nid in noise_ids:
                        _in_progress_ids.pop(nid, None)
                logger.debug(f"🧹 Filtered {len(noise_ids)} noise messages from batch.")

            if not to_ai:
                await asyncio.sleep(0.5)
                continue

            logger.info(f"🧬 Spawning batch: {len(to_ai)} msgs "
                        f"(tasks={shared._active_ai_tasks}, headroom={headroom_pct:.0%}, queue={queue_size})")

            # Spawn one task per iteration and yield briefly so the new task
            # registers in `_active_ai_tasks` before we re-check the cap.
            # 50ms is enough for the asyncio scheduler to run the task up to
            # its first `await`, without artificially throttling drain rate.
            shared.mark_batch_spawned()
            # Save a reference to the task so it isn't GC'd mid-flight.
            # Under heavy load Python's task finalizer can collect a task
            # whose only reference is the asyncio loop's internal deque
            # if no one awaits it; if that fires before `finally` in
            # process_one_batch, the claims leak. Saving and discarding
            # via add_done_callback closes the window entirely.
            _spawned_task = asyncio.create_task(process_one_batch(to_ai))
            _active_spawn_tasks.add(_spawned_task)
            _spawned_task.add_done_callback(_active_spawn_tasks.discard)
            await asyncio.sleep(0.05)
        except Exception as e:
            logger.error(f"processing_loop error: {e}", exc_info=True)
            try:
                await bot.alert_error("processing_loop", e)
            except Exception:
                pass
            await asyncio.sleep(5)

# ── Runtime self-heal watchdogs ────────────────────────────────────────────────

async def _alert_watchdog():
    """Un-stick `pending_alerts` rows orphaned by mid-flush crashes."""
    await db.recover_stuck_alerts()


async def _in_progress_reaper() -> None:
    """Evict `_in_progress_ids` entries older than _IN_PROGRESS_MAX_AGE_SEC.

    An entry stuck here means a batch claimed rows but never released them —
    possibly because `process_one_batch` was cancelled mid-flight (e.g. the
    model slot was lost, a connection timed out, or the task was GC'd before
    finally ran). Left unchecked, these rows sit at `processed=0` forever and
    eventually cause the same multi-minute tail latencies as the original bug.
    """
    now_m = time.monotonic()
    evicted: list[int] = []
    async with _in_progress_lock:
        for mid, claim_ts in list(_in_progress_ids.items()):
            if now_m - claim_ts > _IN_PROGRESS_MAX_AGE_SEC:
                evicted.append(mid)
                _in_progress_ids.pop(mid, None)
    if evicted:
        logger.warning(
            f"⚠️ In-progress reaper freed {len(evicted)} stuck claims "
            f"(ages > {_IN_PROGRESS_MAX_AGE_SEC:.0f}s). Messages will be retried."
        )


_last_loop_alert_ts: float = 0.0
_LOOP_ALERT_COOLDOWN_SEC: float = 600.0


async def _loop_heartbeat_watchdog() -> None:
    """Alert the owner if `processing_loop` hasn't ticked in >90s.

    The loop ticks every iteration (top of the `while True`). If no tick has
    been observed for 90s, something has blocked the loop — a runaway AI call
    holding the event loop, a DB deadlock, or an exception we're not catching.
    We rate-limit the alert to at most once per 10 minutes so a long block
    doesn't produce a storm of pings.
    """
    global _last_loop_alert_ts
    last_tick = shared.get_loop_tick()
    if last_tick is None:
        return   # loop hasn't started yet
    age = time.monotonic() - last_tick
    if age < 90.0:
        return
    now_m = time.monotonic()
    if now_m - _last_loop_alert_ts < _LOOP_ALERT_COOLDOWN_SEC:
        return
    _last_loop_alert_ts = now_m
    logger.error(f"💔 processing_loop heartbeat stale: last tick {age:.0f}s ago")
    try:
        await bot.alert_error(
            "processing_loop_heartbeat",
            RuntimeError(f"processing_loop stalled: no tick for {age:.0f}s"),
        )
    except Exception:
        pass


_LISTENER_WATCHDOG_QUIET_SEC: float = 180.0   # 3 min without an ingest
_last_listener_reconnect_ts: float = 0.0
_LISTENER_RECONNECT_COOLDOWN_SEC: float = 300.0


async def _listener_health_watchdog() -> None:
    """Reconnect Telethon proactively if the listener appears silent.

    Runs every 60s. Logic:
      - If we've ingested a message in the last _LISTENER_WATCHDOG_QUIET_SEC
        seconds, do nothing (listener is healthy).
      - Otherwise check MTProto socket state. If disconnected, attempt a
        reconnect (rate-limited to once per cooldown window).

    This complements `heartbeat_job`'s 20-min lag check with a much tighter
    detection loop for the "silent dead listener" failure mode.
    """
    global _last_listener_reconnect_ts
    try:
        ingest_age = shared.seconds_since_last_ingest()
        # Fresh ingest → healthy, done.
        if ingest_age is not None and ingest_age < _LISTENER_WATCHDOG_QUIET_SEC:
            return

        try:
            mtproto_connected = bool(shared.listener.client.is_connected())
        except Exception:
            mtproto_connected = False

        # Socket says connected AND we haven't been silent that long → fine.
        # We only intervene when the socket itself reports disconnected.
        if mtproto_connected:
            return

        # Rate-limit reconnect attempts.
        now_m = time.monotonic()
        if now_m - _last_listener_reconnect_ts < _LISTENER_RECONNECT_COOLDOWN_SEC:
            return
        if shared._listener_reconnecting:
            return

        _last_listener_reconnect_ts = now_m
        logger.warning(
            f"🔌 Listener watchdog: socket disconnected and no ingest for "
            f"{(ingest_age or 0):.0f}s — forcing reconnect."
        )
        from shared import _reconnect_listener
        asyncio.create_task(_reconnect_listener(gap_minutes=0.5))
    except Exception as e:
        logger.error(f"listener_health_watchdog error: {e}", exc_info=True)


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
        jobs.confirmation_gate_job, "interval", minutes=1, id="confirm_gate", args=[db]
    )

    scheduler.add_job(
        jobs.db_maintenance_job, "cron", hour="*/4", minute=5,
        id="db_maint", args=[db, bot]
    )
    # runtime watchdog for stuck pending_alerts
    scheduler.add_job(
        _alert_watchdog, "interval", minutes=1, id="alert_watchdog"
    )
    # Evict stale `_in_progress_ids` claims so messages stuck from a crashed
    # batch don't block the queue indefinitely.
    scheduler.add_job(
        _in_progress_reaper, "interval", minutes=1, id="in_progress_reaper"
    )
    # Alert if processing_loop stops ticking (heartbeat > 90s stale).
    scheduler.add_job(
        _loop_heartbeat_watchdog, "interval", seconds=30, id="loop_heartbeat"
    )
    # Listener watchdog: every 60s, if we haven't ingested a message in 3 min
    # AND MTProto socket claims disconnected, force a reconnect. Complements
    # the existing heartbeat_job (which only reconnects on 20min+ lag).
    scheduler.add_job(
        _listener_health_watchdog, "interval", seconds=60, id="listener_health"
    )
    # Sinyal waktu: T-2min reminders for time-bounded promos.
    scheduler.add_job(
        jobs.time_reminder_job, "interval", minutes=1,
        id="time_reminder", args=[db, bot, WIB]
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
