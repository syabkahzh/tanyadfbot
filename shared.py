import asyncio
import difflib
import logging
import re
import time as _time_mod
import uuid
import os
from collections import OrderedDict, deque
from datetime import datetime, timedelta, timezone
from typing import Any

import fasttext
# NumPy 2.0+ compatibility fix for FastText
try:
    import numpy as np
    _orig_array = np.array
    def _fixed_array(obj, *args, **kwargs):
        if kwargs.get("copy") is False:
            kwargs.pop("copy")
            return np.asarray(obj, *args, **kwargs)
        return _orig_array(obj, *args, **kwargs)
    np.array = _fixed_array
except ImportError:
    pass

from db import Database, normalize_brand, get_brand_canon
from processor import GeminiProcessor, PromoExtraction, _CURRENCY_DISCOUNT_PATTERN

logger = logging.getLogger(__name__)

# Shared instances to avoid __main__ vs module import issues
db: Database = Database()
gemini: GeminiProcessor = GeminiProcessor()
listener: Any = None
bot: Any = None

# ── Local NLP "Traffic Cop" ──────────────────────────────────────────────────
_ft_model = None

async def load_classifier(model_path: str = "model.ftz") -> bool:
    """Load FastText model at startup. Non-blocking."""
    global _ft_model
    if not os.path.exists(model_path):
        logger.warning(f"⚠️  No {model_path} found — Tier 2 classifier disabled (regex only)")
        return False
        
    try:
        # Load in thread — fasttext.load_model is synchronous and slow (~200ms)
        _ft_model = await asyncio.to_thread(fasttext.load_model, model_path)
        logger.info(f"🛡️ Traffic Cop: FastText model loaded ({model_path}).")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load FastText model: {e}")
        return False

async def classify(text: str) -> tuple[str, float]:
    """
    Classify text as PROMO or JUNK.

    Returns:
        (label, confidence) where label is '__label__PROMO' or '__label__JUNK'
    """
    res = await classify_batch([text])
    return res[0]

async def classify_batch(texts: list[str]) -> list[tuple[str, float]]:
    """
    Classify a batch of texts efficiently using native FastText batch prediction.

    Returns:
        List of (label, confidence) tuples.
    """
    if _ft_model is None:
        logger.warning("FastText model not loaded — traffic cop disabled, all messages will pass through")
        return [("__label__UNKNOWN", 0.0)] * len(texts)
    if not texts:
        return []
        
    try:
        # FastText.predict handles lists natively and is MUCH faster than looping
        # Clean and truncate each text for consistency
        cleaned = [t.lower().replace("\n", " ")[:512] for t in texts]
        
        labels_list, probs_list = await asyncio.to_thread(_ft_model.predict, cleaned, k=1)
        
        # FastText returns [ [l1], [l2] ... ], [ [p1], [p2] ... ]
        return [(l[0], float(p[0])) for l, p in zip(labels_list, probs_list)]
    except Exception as exc:
        logger.warning(f"FastText batch inference error: {exc}")
        return [("__label__UNKNOWN", 0.0)] * len(texts)

async def classify_one(text: str) -> tuple[str, float]:
    """Predicts label for a single string."""
    results = await classify_batch([text])
    return results[0]

def _parse_ts(ts: str | datetime | Any) -> datetime:
    """Always returns a UTC-aware datetime from various timestamp formats.

    Args:
        ts: The timestamp as a string or datetime object.

    Returns:
        A UTC-aware datetime object.
    """
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    
    s = str(ts).replace('Z', '+00:00')
    try:
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        logger.warning(f"_parse_ts: could not parse {ts!r}, defaulting to epoch")
        return datetime.fromtimestamp(0, tz=timezone.utc)

# Global state that needs to be shared across main and listener
_buffer_flush_task: asyncio.Task[None] | None = None
_active_ai_tasks: int = 0
_active_retry_sends: int = 0

# Use deque(maxlen=500) for O(1) append+eviction
_recent_alerts_history: deque[dict[str, Any]] = deque(maxlen=500)
_recent_alerts_lock: asyncio.Lock = asyncio.Lock()
_flush_lock: asyncio.Lock = asyncio.Lock()

# Alert rate limiting state
_last_error_alerts: dict[str, float] = {}
_ERROR_ALERT_COOLDOWN: float = 120.0

_alerted_aman_parents: set[int] = set()
_alerted_aman_parents_deque: deque[int] = deque(maxlen=500)
_aman_lock: asyncio.Lock = asyncio.Lock()

# Brand-level dedup for the fast-path. Suppresses repeat alerts for the same
# brand within a short window so a 20-message "aman toped" burst produces ONE
# alert instead of 20. Remaining messages fall through to the AI path, where
# `filter_duplicates` catches them via `_recent_alerts_history`.
FASTPATH_BRAND_DEDUP_SEC: float = 90.0
_fastpath_brand_last_fired: dict[str, float] = {}
_fastpath_brand_lock: asyncio.Lock = asyncio.Lock()

_stop_event: asyncio.Event = asyncio.Event()

_listener_reconnecting: bool = False

# ── Temporal Brand Context Tracker ────────────────────────────────────────────
#
# Tracks the *active conversational brand* per chat within a TTL window.
# When a message explicitly mentions a brand (e.g. "sfood 80%"), we record
# that brand. Subsequent brand-less signals ("nyala", "aman", "udah bisa")
# within the TTL inherit the brand from context, preventing them from
# falling to "Unknown" and being dropped by the fast-path.
#
# This is a long-term architectural fix: the previous code treated every
# message in a vacuum, which is fundamentally incompatible with how
# Indonesian deal-hunter chats actually flow (short bursts about one brand,
# then a topic switch).

class TemporalBrandTracker:
    """TTL-based conversational brand context cache."""

    def __init__(self, ttl_seconds: int = 180) -> None:
        self.ttl = ttl_seconds
        # chat_id → (brand, monotonic_timestamp)
        self._active: OrderedDict[int, tuple[str, float]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def update_brand(self, chat_id: int, brand: str) -> None:
        """Record that `brand` is the active topic in `chat_id`."""
        if not brand or brand == "Unknown":
            return
        async with self._lock:
            self._active[chat_id] = (brand, _time_mod.monotonic())
            self._active.move_to_end(chat_id)
            # Evict stale entries beyond a generous cap
            while len(self._active) > 500:
                self._active.popitem(last=False)

    async def get_context(self, chat_id: int) -> str:
        """Return the active brand for `chat_id`, or 'Unknown' if expired/absent."""
        async with self._lock:
            if chat_id not in self._active:
                return "Unknown"
            brand, ts = self._active[chat_id]
            if _time_mod.monotonic() - ts > self.ttl:
                del self._active[chat_id]
                return "Unknown"
            return brand

context_tracker: TemporalBrandTracker = TemporalBrandTracker()


# ── Transit-noise gate for "aman" false positives ─────────────────────────────
# "aman kak rutenya", "aman perjalanannya" should NOT trigger fast-path.

TRANSIT_NOISE_PATTERN = re.compile(
    r'\b(rute|jalan|macet|kereta|stasiun|paket|kirim|kurir|perjalanan|nyampe)(nya|an)?\b',
    re.IGNORECASE,
)


# ── Fuzzy semantic dedup ─────────────────────────────────────────────────────
#
# Rolling in-memory queue with difflib.SequenceMatcher fuzzy matching.
# Applied BEFORE DB write / Telegram broadcast so near-identical alerts
# ("CGV Tsel On" vs "CGV On ges") are collapsed in-memory without any
# DB round-trip.

_fuzzy_dedup_queue: list[dict[str, Any]] = []
_fuzzy_dedup_lock = asyncio.Lock()

async def is_fuzzy_duplicate(brand: str, summary: str,
                              window_minutes: int = 15,
                              threshold: float = 0.6) -> bool:
    """Return True if a near-identical alert was seen within the window."""
    global _fuzzy_dedup_queue
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=window_minutes)

    async with _fuzzy_dedup_lock:
        # Prune expired entries
        _fuzzy_dedup_queue = [
            a for a in _fuzzy_dedup_queue if a['time'] >= cutoff
        ]

        norm_brand = normalize_brand(brand).lower()
        norm_summary = summary.lower()

        for alert in _fuzzy_dedup_queue:
            if alert['brand'] == norm_brand:
                # Offload CPU-bound SequenceMatcher to thread to avoid blocking event loop
                similarity = await asyncio.to_thread(
                    difflib.SequenceMatcher(None, alert['summary'], norm_summary).ratio
                )
                if similarity > threshold:
                    return True

        _fuzzy_dedup_queue.append({
            'brand': norm_brand,
            'summary': norm_summary,
            'time': now,
        })
        return False

# Monotonic timestamp of the last processing_loop iteration. A dedicated
# watchdog (main._loop_heartbeat_watchdog) alerts the owner if this goes
# stale for >90s — i.e. the loop is blocked.
_last_loop_tick: float | None = None

def set_loop_tick() -> None:
    """Record that the processing_loop just ticked (called at top of each iter)."""
    global _last_loop_tick
    import time as _time
    _last_loop_tick = _time.monotonic()

def get_loop_tick() -> float | None:
    return _last_loop_tick

# Max age (seconds) of the oldest row we observed in the ancient tier on the
# last batch fetch. Used by /diag to surface tail-latency risk.
_last_observed_ancient_age: float = 0.0

# Monotonic timestamp of the last time any AI batch was spawned. /diag
# reports (now - this) so the owner can see if batching has stalled even
# when the loop is still ticking (e.g. queue empty or semaphore saturated).
_last_batch_spawn_ts: float | None = None

def mark_batch_spawned() -> None:
    global _last_batch_spawn_ts
    import time as _time
    _last_batch_spawn_ts = _time.monotonic()

# Wall-clock timestamp of the last message we ingested from the target group.
# Primary signal for "is Telethon actually delivering updates" — much more
# reliable than `client.is_connected()`, which returns False transiently
# during MTProto reconnects even while messages are actively flowing.
_last_message_ingest_ts: float | None = None

def mark_message_ingested() -> None:
    global _last_message_ingest_ts
    import time as _time
    _last_message_ingest_ts = _time.monotonic()

def seconds_since_last_ingest() -> float | None:
    import time as _time
    if _last_message_ingest_ts is None:
        return None
    return _time.monotonic() - _last_message_ingest_ts


# ── AI circuit breaker ──────────────────────────────────────────────────────
#
# When Gemini has a provider-side incident (500 storms, rolling timeouts),
# every batch fails in ~90s and we just pound the dead provider while msgs
# pile up. The circuit breaker tracks recent consecutive failures; when we
# cross the threshold we pause AI spawns for a cooldown so the queue can
# drain through fast-path / poison-retirement rather than block behind
# doomed batches.
_ai_consecutive_failures: int = 0
_ai_circuit_open_until: float = 0.0   # monotonic; AI paused until this ts

def record_ai_outcome(success: bool) -> None:
    """Called from process_one_batch after every AI call.

    `success=True` resets the failure counter and closes the breaker.
    `success=False` bumps the counter; once we hit the threshold we open
    the breaker for _AI_CIRCUIT_COOLDOWN_SEC.
    """
    global _ai_consecutive_failures, _ai_circuit_open_until
    import time as _time
    if success:
        _ai_consecutive_failures = 0
        _ai_circuit_open_until = 0.0
        return
    _ai_consecutive_failures += 1
    if _ai_consecutive_failures >= _AI_CIRCUIT_FAILURE_THRESHOLD:
        _ai_circuit_open_until = _time.monotonic() + _AI_CIRCUIT_COOLDOWN_SEC

def ai_circuit_open_remaining() -> float:
    """Seconds remaining in the current AI pause, or 0 if AI is available."""
    import time as _time
    remaining = _ai_circuit_open_until - _time.monotonic()
    return remaining if remaining > 0 else 0.0

# Threshold + cooldown. Kept here (not main.py) so /diag can read them.
_AI_CIRCUIT_FAILURE_THRESHOLD: int = 50
_AI_CIRCUIT_COOLDOWN_SEC: float = 5.0
_last_trend_alert: str = ""
_last_trend_alert_ts: float = 0.0
_last_spike_alert: datetime = datetime.min.replace(tzinfo=timezone.utc)
_last_hourly_digest: str = ""

_ACTIVE_SLANG = re.compile(r'\b(jp|aman|on|work|luber|pecah)\b', re.IGNORECASE)

def _score_confidence(p: PromoExtraction, msg: dict, recently_alerted_brands: set[str]) -> int:
    """Calculates a confidence score for a promotion.

    Args:
        p: Extracted promotion data.
        msg: Original message data.
        recently_alerted_brands: Set of normalized brand names alerted recently.

    Returns:
        Confidence score from 0 to 100.
    """
    score = 0
    if normalize_brand(p.brand) != "Unknown": score += 30
    if _CURRENCY_DISCOUNT_PATTERN.search(p.summary):
        score += 30
        
    # BUG S1 FIX: Slang active signals count as implicit confirmation
    if _ACTIVE_SLANG.search(p.summary or 
''):
        score += 15
        
    if p.status == 'active': score += 15
    if msg.get('reply_to_msg_id'): score += 5

    brand_key = normalize_brand(p.brand).lower()
    if brand_key in recently_alerted_brands:
        score -= 15

    return max(0, min(score, 100))


def get_buffer_flush_task() -> asyncio.Task[None] | None:
    """Retrieves the current buffer flush task.

    Returns:
        The active asyncio Task or None.
    """
    return _buffer_flush_task

def set_buffer_flush_task(task: asyncio.Task[None] | None) -> None:
    """Sets the current buffer flush task.

    Args:
        task: The asyncio Task to set as active.
    """
    global _buffer_flush_task
    _buffer_flush_task = task

def get_stop_event() -> asyncio.Event:
    """Retrieves the global application stop event.

    Returns:
        The asyncio Event used to signal shutdown.
    """
    return _stop_event

def _make_tg_link(chat_id: int | str, msg_id: int | str) -> str:
    """Generates a direct Telegram deep-link for a message."""
    cid = str(chat_id)
    if cid.startswith("-100"):
        cid = cid[4:]
    return f"https://t.me/c/{cid}/{msg_id}"


async def _reconnect_listener(gap_minutes: float) -> None:
    """Handles Telethon client reconnection and history catchup with lock resilience."""
    global _listener_reconnecting
    if _listener_reconnecting:
        return
    _listener_reconnecting = True
    connected = False
    try:
        logger.info(f"🔄 Reconnecting listener (lag: {int(gap_minutes)}m)...")
        try:
            await listener.client.disconnect()
        except Exception:
            pass
        await asyncio.sleep(1)

        # Phase 1: reconnect with overall 30s timeout
        try:
            async with asyncio.timeout(30):
                for attempt in range(5):
                    try:
                        await listener.client.connect()
                        connected = True
                        break
                    except Exception as e:
                        if "locked" in str(e).lower() and attempt < 4:
                            wait = min(2 * (attempt + 1), 8)
                            logger.warning(f"⚠️ Telethon session locked, retrying in {wait}s...")
                            await asyncio.sleep(wait)
                            continue
                        raise
        except (asyncio.TimeoutError, TimeoutError):
            logger.error("❌ _reconnect_listener: connect phase timed out (30s)")
        except Exception as e:
            logger.error(f"❌ _reconnect_listener connect failed: {e}")
    finally:
        _listener_reconnecting = False

    # Phase 2: sync history OUTSIDE the reconnecting guard
    if connected:
        try:
            async with asyncio.timeout(60):
                await listener.sync_history(hours=min(gap_minutes / 60 + 0.25, 3.0))
        except (asyncio.TimeoutError, TimeoutError):
            logger.warning("⚠️ _reconnect_listener: sync_history timed out (60s), skipping.")
        except Exception as e:
            logger.error(f"⚠️ _reconnect_listener sync_history failed: {e}")

_ELONGATION_RE = re.compile(r'([a-zA-Z])\1{2,}')

def _guess_brand(text: str | None) -> str:
    """Fast, pattern-based brand identification with strict short-word matching."""
    if not text:
        return 'Unknown'

    t_raw = text.lower()
    t_norm = _ELONGATION_RE.sub(r'\1', t_raw)

    brand_canon = get_brand_canon()
    for kw, brand in brand_canon.items():
        if len(kw) <= 3 or '+' in kw:
            pattern = rf'(^|[^a-zA-Z0-9]){re.escape(kw)}($|[^a-zA-Z0-9])'
            if re.search(pattern, t_raw) or re.search(pattern, t_norm):
                return brand
        elif len(kw) <= 5:
            pattern = rf'(^|[^a-zA-Z0-9]){re.escape(kw)}($|[^a-zA-Z0-9])'
            if re.search(pattern, t_raw) or re.search(pattern, t_norm):
                return brand
        elif kw in t_raw or kw in t_norm:
            return brand

    return 'Unknown'


async def _flush_alert_buffer(delay: float = 0.5) -> None:
    """Consolidates and broadcasts pending alerts to Telegram."""
    if delay > 0:
        await asyncio.sleep(delay)

    async with _flush_lock:
        flush_id: str = str(uuid.uuid4())
        
        if not db.conn:
            logger.error("Database connection missing in _flush_alert_buffer.")
            return

        await db.conn.execute(
            "UPDATE pending_alerts SET flush_id=? WHERE flush_id IS NULL", (flush_id,)
        )
        await db.conn.commit()

        async with db.conn.execute(
            "SELECT brand, p_data_json, tg_link, timestamp, corroborations, corroboration_texts, source "
            "FROM pending_alerts WHERE flush_id=?", (flush_id,)
        ) as cur:
            rows = await cur.fetchall()

        if not rows:
            set_buffer_flush_task(None)
            return

        snapshot: dict[str, list[tuple[PromoExtraction, str, Any, int, str, str]]] = {}
        for r in rows:
            brand  = r['brand']
            try:
                p_data = PromoExtraction.model_validate_json(r['p_data_json'])
                snapshot.setdefault(brand, []).append(
                    (p_data, r['tg_link'], r['timestamp'], r['corroborations'], r['corroboration_texts'], r['source'])
                )
            except Exception as e:
                logger.error(f"Failed to parse p_data_json in flush: {e}")

    try:
        tasks = []
        task_to_item = []
        from telegram.constants import ParseMode
        for brand_key, items in snapshot.items():
            if len(items) == 1:
                p, link, ts, corr, ctexts, src = items[0]
                tasks.append(bot.send_alert(p, link, timestamp=ts, corroborations=corr, corroboration_texts=ctexts, source=src))
                task_to_item.append((brand_key, 0))
            else:
                tasks.append(bot.send_grouped_alert(brand_key, items))
                task_to_item.append((brand_key, None))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Identify which alerts succeeded vs failed
            failed_brands: set[str] = set()
            for idx, res in enumerate(results):
                if isinstance(res, Exception):
                    brand_key, _ = task_to_item[idx]
                    failed_brands.add(brand_key)
            
            if failed_brands:
                # Only delete alerts for brands that succeeded
                brands_to_delete = [b for b, _ in task_to_item if b not in failed_brands]
                if brands_to_delete:
                    ph = ",".join("?" * len(brands_to_delete))
                    await db.conn.execute(
                        f"DELETE FROM pending_alerts WHERE flush_id=? AND brand IN ({ph})",
                        (flush_id, *brands_to_delete)
                    )
                    await db.conn.commit()
                # Reset flush_id for failed brands so they retry
                if failed_brands:
                    ph = ",".join("?" * len(failed_brands))
                    await db.conn.execute(
                        f"UPDATE pending_alerts SET flush_id=NULL WHERE flush_id=? AND brand IN ({ph})",
                        (flush_id, *failed_brands)
                    )
                    await db.conn.commit()
            else:
                await db.conn.execute(
                    "DELETE FROM pending_alerts WHERE flush_id=?", (flush_id,)
                )
                await db.conn.commit()
        set_buffer_flush_task(None)
    except Exception as e:
        set_buffer_flush_task(None)
        logger.error(f"⚠️ Alert flush failed: {e}")
        if bot:
            await bot.alert_error("_flush_alert_buffer", e)
        await db.conn.execute(
            "UPDATE pending_alerts SET flush_id=NULL WHERE flush_id=?", (flush_id,)
        )
        await db.conn.commit()
