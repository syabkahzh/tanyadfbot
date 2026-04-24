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
        if kwargs.get('copy') is False:
            kwargs.pop('copy')
            return np.asarray(obj, *args, **kwargs)
        return _orig_array(obj, *args, **kwargs)
    np.array = _fixed_array
except ImportError:
    pass

from db import Database, normalize_brand
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
        Returns ('__label__UNKNOWN', 0.0) if model not loaded.
    """
    if _ft_model is None:
        return "__label__UNKNOWN", 0.0
    if not text or not text.strip():
        return "__label__JUNK", 1.0
        
    try:
        t = text.lower().replace("\n", " ")[:512]
        labels, probs = await asyncio.to_thread(_ft_model.predict, t, k=1)
        return labels[0], float(probs[0])
    except Exception as exc:
        logger.warning(f"FastText inference error: {exc}")
        return "__label__UNKNOWN", 0.0

def _parse_ts(ts: str | datetime | Any) -> datetime:
    """Always returns a UTC-aware datetime from various timestamp formats."""
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

_alerted_aman_parents: set[int] = set()
_alerted_aman_parents_deque: deque[int] = deque(maxlen=500)
_aman_lock: asyncio.Lock = asyncio.Lock()

FASTPATH_BRAND_DEDUP_SEC: float = 90.0
_fastpath_brand_last_fired: dict[str, float] = {}
_fastpath_brand_lock: asyncio.Lock = asyncio.Lock()

_stop_event: asyncio.Event = asyncio.Event()

_listener_reconnecting: bool = False

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

TRANSIT_NOISE_PATTERN = re.compile(
    r'\b(rute|jalan|macet|kereta|stasiun|paket|kirim|kurir|perjalanan|nyampe)(nya|an)?\b',
    re.IGNORECASE,
)

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
                similarity = difflib.SequenceMatcher(
                    None, alert['summary'], norm_summary
                ).ratio()
                if similarity > threshold:
                    return True

        _fuzzy_dedup_queue.append({
            'brand': norm_brand,
            'summary': norm_summary,
            'time': now,
        })
        return False

_last_loop_tick: float | None = None

def set_loop_tick() -> None:
    """Record that the processing_loop just ticked."""
    global _last_loop_tick
    import time as _time
    _last_loop_tick = _time.monotonic()

def get_loop_tick() -> float | None:
    return _last_loop_tick

_last_observed_ancient_age: float = 0.0
_last_batch_spawn_ts: float | None = None

def mark_batch_spawned() -> None:
    global _last_batch_spawn_ts
    import time as _time
    _last_batch_spawn_ts = _time.monotonic()

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

_ai_consecutive_failures: int = 0
_ai_circuit_open_until: float = 0.0   # monotonic; AI paused until this ts

def record_ai_outcome(success: bool) -> None:
    """Tracks consecutive AI failures to manage the circuit breaker."""
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

_AI_CIRCUIT_FAILURE_THRESHOLD: int = 5
_AI_CIRCUIT_COOLDOWN_SEC: float = 30.0
_last_trend_alert: str = ""
_last_spike_alert: datetime = datetime.min.replace(tzinfo=timezone.utc)
_last_hourly_digest: str = ""

_ACTIVE_SLANG = re.compile(r'\b(jp|aman|on|work|luber|pecah)\b', re.IGNORECASE)

def _score_confidence(p: PromoExtraction, msg: dict, recently_alerted_brands: set[str]) -> int:
    """Calculates a confidence score for a promotion."""
    score = 0
    if normalize_brand(p.brand) != "Unknown": score += 30
    if _CURRENCY_DISCOUNT_PATTERN.search(p.summary):
        score += 30
    if _ACTIVE_SLANG.search(p.summary or ''):
        score += 15
    if p.status == 'active': score += 15
    if msg.get('reply_to_msg_id'): score += 5
    brand_key = normalize_brand(p.brand).lower()
    if brand_key in recently_alerted_brands:
        score -= 15
    return max(0, min(score, 100))

def get_buffer_flush_task() -> asyncio.Task[None] | None:
    return _buffer_flush_task

def set_buffer_flush_task(task: asyncio.Task[None] | None) -> None:
    global _buffer_flush_task
    _buffer_flush_task = task

def get_stop_event() -> asyncio.Event:
    return _stop_event
