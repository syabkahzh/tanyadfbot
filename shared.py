import asyncio
import html
import logging
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any, Sequence, cast

from db import Database, normalize_brand
from processor import GeminiProcessor, PromoExtraction, _CURRENCY_DISCOUNT_PATTERN
from utils import _esc

logger = logging.getLogger(__name__)

# Shared instances to avoid __main__ vs module import issues
db: Database = Database()
gemini: GeminiProcessor = GeminiProcessor()
listener: Any = None
bot: Any = None

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

_alerted_aman_parents: set[int] = set()
_alerted_aman_parents_deque: deque[int] = deque(maxlen=500)
_aman_lock: asyncio.Lock = asyncio.Lock()

_stop_event: asyncio.Event = asyncio.Event()

_listener_reconnecting: bool = False
_last_trend_alert: str = ""
_last_spike_alert: datetime = datetime.min.replace(tzinfo=timezone.utc)
_last_hourly_digest: str = ""

_BRAND_KEYWORDS: dict[str, str] = {
    'kopken': 'Kopi Kenangan', 'kopi kenangan': 'Kopi Kenangan',
    'kenangan': 'Kopi Kenangan', 'k+p+k+n': 'Kopi Kenangan',
    'hokben': 'HokBen', 'hoka ben': 'HokBen', 'h+k+b+n': 'HokBen', 'h+o+k+b+e+n': 'HokBen',
    'hophop': 'HopHop', 'hop hop': 'HopHop', 'h+p+h+p': 'HopHop',
    'sfood': 'ShopeeFood', 'shopeefood': 'ShopeeFood',
    's+f+d': 'ShopeeFood', 'sfud': 'ShopeeFood',
    'sopifut': 'ShopeeFood', 'sopifud': 'ShopeeFood', 's+p+f+d': 'ShopeeFood',
    'gfood': 'GoFood', 'gofood': 'GoFood', 'g+f+d': 'GoFood', 'g+o+f+o+o+d': 'GoFood',
    'spx': 'SPX', 'shopee xpress': 'SPX', 's+p+x': 'SPX',
    'alfamart': 'Alfamart', 'alfa': 'Alfamart', 'a+l+f+a': 'Alfamart', 'a+l+f+a+m+a+r+t': 'Alfamart',
    'indomaret': 'Indomaret', 'idm': 'Indomaret', 'i+d+m': 'Indomaret', 'i+n+d+o': 'Indomaret',
    'chatime': 'Chatime', 'chtm': 'Chatime',
    'c+h+t+m': 'Chatime', 'ctm': 'Chatime', 'c+t+m': 'Chatime',
    'starbucks': 'Starbucks', 's+t+a+r+b+u+c+k+s': 'Starbucks',
    'ismaya': 'Ismaya', 'gindaco': 'Gindaco', 'g+i+n+d+a+c+o': 'Gindaco',
    'g+n+d+c': 'Gindaco',
    'kawanlama': 'Kawan Lama', 'kawan lama': 'Kawan Lama', 'k+w+n+l+m': 'Kawan Lama',
    'cgv': 'CGV', 'xxi': 'XXI', 'c+g+v': 'CGV', 'x+x+i': 'XXI',
    'mcd': 'McD', 'kfc': 'KFC', 'm+c+d': 'McD', 'k+f+c': 'KFC',
    'gopay': 'GoPay', 'gpy': 'GoPay', 'g+p+y': 'GoPay', 'g+o+p+a+y': 'GoPay',
    'spay': 'ShopeePay', 'shopeepay': 'ShopeePay', 's+p+a+y': 'ShopeePay', 's+h+o+p+e+e+p+a+y': 'ShopeePay',
    'ovo': 'OVO', 'o+v+o': 'OVO',
    'goco': 'GoPay Coins', 'g+o+c+o': 'GoPay Coins',
    'grab': 'Grab', 'gojek': 'Gojek', 'g+r+a+b': 'Grab', 'g+o+j+e+k': 'Gojek',
    'goride': 'GoRide', 'grd': 'GoRide', 'g+r+d': 'GoRide', 'gored': 'GoRide',
    'pln': 'PLN', 'pulsa': 'Pulsa', 'ag': 'Alfagift', 'alfagift': 'Alfagift', 'a+g': 'Alfagift', 'a+l+f+a+g+i+f+t': 'Alfagift',
    'astrapay': 'AstraPay', 'alt': 'Astra', 'a+s+t+r+a+p+a+y': 'AstraPay', 'a+l+t': 'Astra',
    'tsel': 'Telkomsel', 'mytsel': 'Telkomsel', 'mytelkomsel': 'Telkomsel', 't+s+e+l': 'Telkomsel',
    'tokopedia': 'Tokopedia', 'tokped': 'Tokopedia',
    'toped': 'Tokopedia', 'tkpd': 'Tokopedia', 't+k+p+d': 'Tokopedia',
    'lazada': 'Lazada', 'lzd': 'Lazada', 'l+z+d': 'Lazada', 'l+a+z+a+d+a': 'Lazada',
    'dilan': 'Dilan', 'bandung': 'Bandung',
    'cetem': 'Cetem', 'cetam': 'Cetem', 'pubg': 'PUBG', 'pugb': 'PUBG', 'p+u+b+g': 'PUBG',
    'rotio': 'Roti O', 'roti o': 'Roti O', 'roti-o': 'Roti O',
    'tomoro': 'Tomoro Coffee', 'tomoro coffee': 'Tomoro Coffee',
    'jago': 'Bank Jago', 'saqu': 'Bank Saqu', 'seabank': 'SeaBank', 'aladin': 'Bank Aladin',
    'g+b+s': 'GaBisa',
    'r+s+t+k': 'Restock', 'r+s+t+c+k': 'Restock', 'r+st+ck': 'Restock',
    'cb': 'Cashback', 'kesbek': 'Cashback', 'c+s+h+b+c+k': 'Cashback', 'cash back': 'Cashback',
    'pchematapril': 'PC HematApril',
}


def _make_tg_link(chat_id: int | str, msg_id: int | str) -> str:
    """Generates a direct Telegram deep-link for a message."""
    cid = str(chat_id)
    if cid.startswith("-100"):
        cid = cid[4:]
    return f"https://t.me/c/{cid}/{msg_id}"




async def _reconnect_listener(gap_minutes: float) -> None:
    """Handles Telethon client reconnection and history catchup."""
    from shared import listener
    if shared._listener_reconnecting:
        return
    shared._listener_reconnecting = True
    try:
        logger.info(f"Reconnecting listener (lag: {int(gap_minutes)}m)...")
        try:
            await listener.client.disconnect()
        except Exception:
            pass
        await asyncio.sleep(3)
        await listener.client.connect()
        await listener.sync_history(hours=min(gap_minutes / 60 + 0.25, 3.0))
    finally:
        shared._listener_reconnecting = False


def _guess_brand(text: str | None) -> str:
    """Fast, pattern-based brand identification with strict short-word matching."""
    if not text:
        return 'Unknown'
    
    t = text.lower()
    import re
    
    for kw, brand in _BRAND_KEYWORDS.items():
        # For very short or commonly substringed keywords, require word boundaries
        if len(kw) <= 3 or '+' in kw:
            # We use a custom boundary check to handle the '+' literal in some keywords
            pattern = rf'(^|[^a-zA-Z0-9]){re.escape(kw)}($|[^a-zA-Z0-9])'
            if re.search(pattern, t):
                return brand
        elif kw in t:
            return brand
            
    return 'Unknown'


async def _flush_alert_buffer(delay: float = 0.5) -> None:
    """Consolidates and broadcasts pending alerts to Telegram.

    Group alerts by brand and delivers them after a short delay to allow 
    for concurrent batch processing.

    Args:
        delay: Seconds to wait before starting the flush.
    """
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

        # 3. Process data into snapshot while still under lock to ensure consistency
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

        # Clear task early so new alerts can trigger a new flush cycle while this one is sending
        set_buffer_flush_task(None)

    # --- LOCK RELEASED ---
    # Delivery happens outside the lock so the next flush (e.g. fast-path) 
    # can start claiming its own rows immediately.

    try:
        tasks = []
        for brand_key, items in snapshot.items():
            if len(items) == 1:
                p, link, ts, corr, ctexts, src = items[0]
                tasks.append(bot.send_alert(p, link, timestamp=ts, corroborations=corr, corroboration_texts=ctexts, source=src))
            else:
                tasks.append(bot.send_grouped_alert(brand_key, items))
        
        if tasks:
            await asyncio.gather(*tasks)

        await db.conn.execute(
            "DELETE FROM pending_alerts WHERE flush_id=?", (flush_id,)
        )
        await db.conn.commit()
    except Exception as e:
        print(f"⚠️ Alert flush failed: {e}")
        if bot:
            await bot.alert_error("_flush_alert_buffer", e)
        # Unlock rows so they'll be retried on next flush cycle
        await db.conn.execute(
            "UPDATE pending_alerts SET flush_id=NULL WHERE flush_id=?", (flush_id,)
        )
        await db.conn.commit()


def _score_confidence(p: PromoExtraction, msg: dict, recent_alerts: list) -> int:
    """Calculates a confidence score for a promotion."""
    score = 0
    if normalize_brand(p.brand) != "Unknown": score += 30
    if _CURRENCY_DISCOUNT_PATTERN.search(p.summary):
        score += 30
    if p.status == 'active': score += 15
    if msg.get('reply_to_msg_id'): score += 5

    brand_key = normalize_brand(p.brand).lower()
    recently_alerted_brands = {
        normalize_brand(r.get('brand', '')).lower()
        for r in recent_alerts[-20:]
    }
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
