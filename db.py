"""db.py — TanyaDFBot Database Layer.

Robust SQLite management with WAL support, automated triage, and proactive 
integrity recovery.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Sequence, cast
import functools

import aiosqlite
from config import Config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Brand normalisation (single source of truth — shared by processor & main)
# ─────────────────────────────────────────────────────────────────────────────

_BRAND_CANON: dict[str, str] = {
    'hokben': 'HokBen', 'hoka ben': 'HokBen', 'h+k+b+n': 'HokBen', 'h+o+k+b+e+n': 'HokBen',
    'hophop': 'HopHop', 'hop hop': 'HopHop', 'h+p+h+p': 'HopHop',
    'sfood': 'ShopeeFood', 'shopeefood': 'ShopeeFood', 'shopee food': 'ShopeeFood',
    's+f+d': 'ShopeeFood', 'sfud': 'ShopeeFood',
    'sopifut': 'ShopeeFood', 'sopifud': 'ShopeeFood', 's+p+f+d': 'ShopeeFood',
    'gfood': 'GoFood', 'gofood': 'GoFood', 'go food': 'GoFood', 'g+f+d': 'GoFood', 'g+o+f+o+o+d': 'GoFood',
    'gopay': 'GoPay', 'gpy': 'GoPay', 'g+p+y': 'GoPay', 'g+o+p+a+y': 'GoPay',
    'kopken': 'Kopi Kenangan', 'kopi kenangan': 'Kopi Kenangan',
    'kenangan': 'Kopi Kenangan', 'k+p+k+n': 'Kopi Kenangan',
    'alfamart': 'Alfamart', 'alfa': 'Alfamart', 'a+l+f+a': 'Alfamart', 'a+l+f+a+m+a+r+t': 'Alfamart',
    'jsm': 'Alfamart', 'j+s+m': 'Alfamart',
    'psm': 'Alfamart', 'p+s+m': 'Alfamart',
    'afm': 'Alfamart', 'a+f+m': 'Alfamart',
    'indomaret': 'Indomaret', 'idm': 'Indomaret', 'i+d+m': 'Indomaret', 'i+n+d+o': 'Indomaret',
    'spx': 'SPX', 'spx express': 'SPX', 'shopee xpress': 'SPX', 's+p+x': 'SPX',
    'chatime': 'Chatime', 'chtm': 'Chatime',
    'c+h+t+m': 'Chatime', 'ctm': 'Chatime', 'c+t+m': 'Chatime',
    "the people's cafe": 'The Peoples Cafe',
    'the peoples cafe': 'The Peoples Cafe', 'tpc': 'The Peoples Cafe', 't+p+c': 'The Peoples Cafe',
    'ismaya+': 'Ismaya', 'ismaya+ delivery': 'Ismaya', 'i+s+m+a+y+a': 'Ismaya',
    'cupbob': 'Cupbop', 'c+u+p+b+o+p': 'Cupbop',
    'pubg': 'PUBG', 'pugb': 'PUBG', 'p+u+b+g': 'PUBG',
    'tokopedia': 'Tokopedia', 'tokped': 'Tokopedia',
    'toped': 'Tokopedia', 'tkpd': 'Tokopedia', 't+k+p+d': 'Tokopedia',
    'lazada': 'Lazada', 'lzd': 'Lazada', 'l+z+d': 'Lazada', 'l+a+z+a+d+a': 'Lazada',
    'goride': 'GoRide', 'grd': 'GoRide', 'g+r+d': 'GoRide', 'gored': 'GoRide',
    'ag': 'Alfagift', 'alfagift': 'Alfagift', 'a+l+f+a+g+i+f+t': 'Alfagift', 'a+g': 'Alfagift',
    'astrapay': 'AstraPay', 'a+s+t+r+a+p+a+y': 'AstraPay', 'a+s+p+a+y': 'AstraPay',
    'alt': 'Astra', 'a+l+t': 'Astra',
    'tsel': 'Telkomsel', 'mytsel': 'Telkomsel', 'mytelkomsel': 'Telkomsel', 't+s+e+l': 'Telkomsel',
    'dilan': 'Dilan', 'bandung': 'Bandung',
    'goco': 'GoPay Coins', 'g+o+c+o': 'GoPay Coins',
    'gindaco': 'Gindaco', 'g+n+d+c': 'Gindaco',
    'kawanlama': 'Kawan Lama', 'kawan lama': 'Kawan Lama', 'k+w+n+l+m': 'Kawan Lama',
    # junk sentinels → Unknown
    'brand': 'Unknown', 'tidak diketahui': 'Unknown',
    'tidak disebutkan': 'Unknown', 'sunknown': 'Unknown',
    'bunknown': 'Unknown', 'n/a': 'Unknown',
}

@functools.lru_cache(maxsize=1024)
def normalize_brand(brand: str | None) -> str:
    """Normalizes brand names to a canonical representation.

    Args:
        brand: The raw brand name to normalize.

    Returns:
        The normalized canonical brand name.
    """
    if not brand:
        return "Unknown"
    b = brand.strip()
    if b.lower() in ('unknown', 'sunknown', 'bunknown', ''):
        return "Unknown"
    if b.lower() in _BRAND_CANON:
        return _BRAND_CANON[b.lower()]
    return b[0].upper() + b[1:] if b else "Unknown"


def _ts_str(ts: str | datetime | Any) -> str:
    """Normalizes any timestamp to canonical UTC string format.

    Args:
        ts: The timestamp as string or datetime.

    Returns:
        ISO8601 formatted string in UTC.
    """
    if isinstance(ts, datetime):
        return ts.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M:%S+00:00')
    s = str(ts).replace('T', ' ').replace('Z', '+00:00')
    return s


# ─────────────────────────────────────────────────────────────────────────────

class Database:
    """Handles all SQLite database operations for the application."""

    def __init__(self) -> None:
        """Initializes the Database instance."""
        self.db_path: str = Config.DB_PATH
        self.conn: aiosqlite.Connection | None = None

    async def ensure_connection(self) -> bool:
        """Checks connection health and attempts reconnection if dropped.
        
        Returns:
            True if connection is active or successfully restored.
        """
        if self.conn:
            try:
                await self.conn.execute("SELECT 1")
                return True
            except Exception:
                logger.warning("🗄️ Database connection lost. Attempting to reconnect...")
                try:
                    await self.conn.close()
                except Exception:
                    pass
                self.conn = None
        
        try:
            await self.init()
            logger.info("✅ Database connection restored.")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to restore database connection: {e}")
            return False

    # ── Initialisation ────────────────────────────────────────────────────────

    async def init(self) -> None:
        """Connects to the database and initializes schemas and indexes."""
        self.conn = await aiosqlite.connect(self.db_path)
        self.conn.row_factory = aiosqlite.Row

        # Performance pragmas — safe for WAL on a single-writer VPS
        await self.conn.execute("PRAGMA journal_mode=WAL")   # keep for crash safety
        await self.conn.execute("PRAGMA foreign_keys=ON")
        await self.conn.execute("PRAGMA wal_autocheckpoint=200")  # checkpoint more often
        await self.conn.execute("PRAGMA cache_size=-16000")   # 16 MB page cache
        await self.conn.execute("PRAGMA synchronous=NORMAL")
        await self.conn.execute("PRAGMA temp_store=MEMORY")
        await self.conn.execute("PRAGMA busy_timeout=5000")   # wait 5s instead of instant SQLITE_BUSY

        # ── messages ──────────────────────────────────────────────────────────
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                tg_msg_id        INTEGER NOT NULL,
                chat_id          INTEGER NOT NULL,
                sender_id        INTEGER,
                sender_name      TEXT,
                timestamp        TEXT NOT NULL,   -- ISO8601 UTC '+00:00'
                text             TEXT,
                reply_to_msg_id  INTEGER,
                processed        INTEGER DEFAULT 0,
                has_photo        INTEGER DEFAULT 0,
                image_processed  INTEGER DEFAULT 0,
                has_time_mention INTEGER DEFAULT 0,
                time_alerted     INTEGER DEFAULT 0,
                UNIQUE(tg_msg_id, chat_id)
            )
        """)

        # ── promos ────────────────────────────────────────────────────────────
        # source_msg_id is nullable — fast-path promos have no DB message row
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS promos (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                source_msg_id INTEGER,            -- NULL for fast-path alerts
                summary       TEXT NOT NULL,
                brand         TEXT NOT NULL,
                conditions    TEXT,
                tg_link       TEXT,
                status        TEXT NOT NULL DEFAULT 'unknown'
                              CHECK(status IN ('active','expired','unknown')),
                via_fastpath  INTEGER DEFAULT 0,  -- 1 = came from fast-path
                created_at    TEXT DEFAULT (strftime('%Y-%m-%d %H:%M:%S+00:00','now')),
                FOREIGN KEY (source_msg_id) REFERENCES messages(id)
            )
        """)

        # ── pending_alerts ────────────────────────────────────────────────────
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS pending_alerts (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                brand          TEXT,
                p_data_json    TEXT,
                tg_link        TEXT,
                timestamp      TEXT,
                corroborations INTEGER DEFAULT 0,
                corroboration_texts TEXT DEFAULT '[]',
                source         TEXT DEFAULT 'ai', -- 'ai' or 'python'
                flush_id       TEXT DEFAULT NULL,
                created_at     TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%S+00:00','now'))
            )
        """)

        # ── pending_confirmations ─────────────────────────────────────────────
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS pending_confirmations (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                brand          TEXT,
                p_data_json    TEXT,
                tg_link        TEXT,
                timestamp      TEXT,
                confidence     INTEGER DEFAULT 0,
                corroborations INTEGER DEFAULT 0,
                corroboration_texts TEXT DEFAULT '[]', -- JSON list of snippets
                expires_at     TEXT,
                created_at     TEXT DEFAULT (strftime('%Y-%m-%d %H:%M:%S+00:00','now'))
            )
        """)

        # ── failures (Error Tracking) ─────────────────────────────────────────
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS failures (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                component    TEXT NOT NULL,
                error_msg    TEXT,
                traceback    TEXT,
                fixed        INTEGER DEFAULT 0, -- 1 if a fix was applied
                retried      INTEGER DEFAULT 0, -- 1 if already retried
                created_at   TEXT DEFAULT (strftime('%Y-%m-%d %H:%M:%S+00:00','now'))
            )
        """)

        await self.conn.commit()

        # ── Safe migrations (idempotent) ──────────────────────────────────────
        migrations = [
            "ALTER TABLE messages ADD COLUMN has_time_mention INTEGER DEFAULT 0",
            "ALTER TABLE messages ADD COLUMN time_alerted INTEGER DEFAULT 0",
            "ALTER TABLE messages ADD COLUMN has_photo INTEGER DEFAULT 0",
            "ALTER TABLE messages ADD COLUMN image_processed INTEGER DEFAULT 0",
            "ALTER TABLE messages ADD COLUMN ai_failure_count INTEGER DEFAULT 0",
            "ALTER TABLE pending_alerts ADD COLUMN flush_id TEXT DEFAULT NULL",
            "ALTER TABLE pending_alerts ADD COLUMN corroborations INTEGER DEFAULT 0",
            "ALTER TABLE pending_alerts ADD COLUMN corroboration_texts TEXT DEFAULT '[]'",
            "ALTER TABLE pending_alerts ADD COLUMN source TEXT DEFAULT 'ai'",
            "ALTER TABLE pending_confirmations ADD COLUMN corroboration_texts TEXT DEFAULT '[]'",
            "ALTER TABLE promos ADD COLUMN valid_until TEXT DEFAULT ''",
            "ALTER TABLE promos ADD COLUMN status_history TEXT DEFAULT '[]'",
            "ALTER TABLE promos ADD COLUMN via_fastpath INTEGER DEFAULT 0",
            "ALTER TABLE promos ADD COLUMN reminder_fired INTEGER DEFAULT 0",
            # created_at format fix for older rows
        ]
        for sql in migrations:
            try:
                await self.conn.execute(sql)
            except Exception:
                pass  # column already exists — fine

        await self.conn.commit()

        # ── Indexes: drop exact duplicates that existed in old schema ─────────
        # Keep the canonical name; drop the old aliases
        for old_idx in ("idx_msg_processed", "idx_msg_timestamp", "idx_msg_reply"):
            await self.conn.execute(f"DROP INDEX IF EXISTS {old_idx}")

        # Canonical indexes
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_queue "
            "ON messages(processed, id ASC)"
        )
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_queue_ts "
            "ON messages(processed, timestamp DESC) WHERE processed=0"
        )
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_timestamp "
            "ON messages(timestamp)"
        )
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_reply "
            "ON messages(reply_to_msg_id, chat_id)"
        )
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_photo "
            "ON messages(has_photo, image_processed) WHERE has_photo=1"
        )
        # promos: UNIQUE dedup index + lookup indexes
        await self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_promos_dedup "
            "ON promos(brand, summary)"
        )
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_promo_created "
            "ON promos(created_at)"
        )
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_promo_brand "
            "ON promos(brand)"
        )

        # ⚡ Bolt Optimization: Add index to pending_confirmations(brand)
        # This speeds up the frequent lookup in main.py:
        # "SELECT id, corroboration_texts FROM pending_confirmations WHERE brand=? LIMIT 1"
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pending_conf_brand "
            "ON pending_confirmations(brand)"
        )
        await self.conn.commit()

        # ── BUG D FIX: Recover stuck pending_alerts on every boot ─────────────
        # Rows with flush_id set mid-crash will never be picked up again.
        # Resetting them here makes them available on the next flush cycle.
        await self.conn.execute(
            "UPDATE pending_alerts SET flush_id=NULL WHERE flush_id IS NOT NULL"
        )
        await self.conn.commit()

        logger.info("DB init complete — indexes clean, stuck alerts recovered.")

    # ── Failures (Error Tracking) ─────────────────────────────────────────────

    async def log_failure(self, component: str, error_msg: str, traceback: str) -> int:
        """Logs a system failure to the database.

        Args:
            component: The component that failed.
            error_msg: The error message.
            traceback: Full traceback string.

        Returns:
            The ID of the logged failure.
        """
        if not self.conn: return 0
        try:
            cursor = await self.conn.execute("""
                INSERT INTO failures (component, error_msg, traceback)
                VALUES (?, ?, ?)
            """, (component, error_msg, traceback))
            fid = cursor.lastrowid or 0
            await self.conn.commit()
            return fid
        except Exception as e:
            logger.error(f"Failed to log failure: {e}")
            return 0

    async def get_pending_failures(self) -> Sequence[Any]:
        """Retrieves failures that haven't been successfully retried after a fix.

        Returns:
            Sequence of failure rows.
        """
        if not self.conn: return []
        async with self.conn.execute(
            "SELECT * FROM failures WHERE retried = 0 ORDER BY id DESC"
        ) as cur:
            return await cur.fetchall()

    async def mark_failure_fixed(self, failure_id: int) -> None:
        """Marks a failure as fixed (ready for retry)."""
        if not self.conn: return
        await self.conn.execute(
            "UPDATE failures SET fixed = 1 WHERE id = ?", (failure_id,)
        )
        await self.conn.commit()

    async def mark_failure_retried(self, failure_id: int) -> None:
        """Marks a failure as retried."""
        if not self.conn: return
        await self.conn.execute(
            "UPDATE failures SET retried = 1 WHERE id = ?", (failure_id,)
        )
        await self.conn.commit()

    # ── Messages ──────────────────────────────────────────────────────────────

    async def save_message(
        self, 
        tg_msg_id: int, 
        chat_id: int, 
        sender_id: int | None, 
        sender_name: str | None,
        timestamp: datetime | str, 
        text: str | None, 
        reply_to_msg_id: int | None,
        processed: int = 0, 
        has_photo: int = 0, 
        has_time_mention: int = 0,
        commit: bool = True
    ) -> int | None:
        """Saves a new message to the database.

        Args:
            tg_msg_id: Telegram message ID.
            chat_id: Telegram chat ID.
            sender_id: ID of the sender.
            sender_name: Name of the sender.
            timestamp: Message timestamp.
            text: Message text content.
            reply_to_msg_id: ID of the message being replied to.
            processed: Initial processed state (0 or 1).
            has_photo: Whether the message has a photo (0 or 1).
            has_time_mention: Whether the message mentions a time (0 or 1).
            commit: Whether to commit immediately.

        Returns:
            The internal database ID if saved, None if it's a duplicate or failed.
        """
        if not self.conn:
            logger.error("Attempted to save message without DB connection.")
            return None
            
        ts_str = _ts_str(timestamp)
        try:
            cursor = await self.conn.execute("""
                INSERT OR IGNORE INTO messages
                    (tg_msg_id, chat_id, sender_id, sender_name, timestamp,
                     text, reply_to_msg_id, processed, has_photo, has_time_mention)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (tg_msg_id, chat_id, sender_id, sender_name, ts_str,
                  text, reply_to_msg_id, processed, has_photo, has_time_mention))
            if commit:
                await self.conn.commit()
            if cursor.rowcount == 0:
                return None   # duplicate — already in DB
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"DB save_message error: {e}")
            return None

    async def get_unprocessed_batch(self, batch_size: int = 50) -> list[aiosqlite.Row]:
        """Retrieves a batch of unprocessed messages.

        Args:
            batch_size: Maximum number of messages to retrieve.

        Returns:
            A list of database rows for unprocessed messages.
        """
        if not self.conn:
            return []
            
        try:
            async with self.conn.execute("""
                SELECT id, text, timestamp, sender_name, tg_msg_id, chat_id, reply_to_msg_id
                FROM messages WHERE processed=0 ORDER BY id ASC LIMIT ?
            """, (batch_size,)) as cur:
                rows = await cur.fetchall()
                return list(rows) if rows else []
        except Exception as e:
            logger.error(f"DB get_unprocessed_batch error: {e}")
            return []

    async def get_unprocessed_ancient(self, min_age_minutes: int = 15,
                                       batch_size: int = 20) -> list[aiosqlite.Row]:
        """Rows older than `min_age_minutes` that haven't been processed.

        Used by the 3-tier queue policy in main.processing_loop to guarantee
        a starvation cap: any row this old skips ahead of fresher backlog.
        Returns oldest first so the very tail of the queue drains first.
        """
        if not self.conn:
            return []
        try:
            cutoff = _ts_str(datetime.now(timezone.utc) - timedelta(minutes=min_age_minutes))
            async with self.conn.execute(
                "SELECT id, text, timestamp, tg_msg_id, chat_id, reply_to_msg_id "
                "FROM messages WHERE processed=0 AND timestamp < ? "
                "ORDER BY timestamp ASC LIMIT ?",
                (cutoff, batch_size)
            ) as cur:
                rows = await cur.fetchall()
                return list(rows) if rows else []
        except Exception as e:
            logger.error(f"DB get_unprocessed_ancient error: {e}")
            return []

    async def get_oldest_unprocessed_age_sec(self) -> float | None:
        """Seconds since the oldest unprocessed message was received, or None."""
        if not self.conn:
            return None
        try:
            async with self.conn.execute(
                "SELECT MIN(timestamp) FROM messages WHERE processed=0"
            ) as cur:
                row = await cur.fetchone()
            if not row or not row[0]:
                return None
            try:
                oldest = datetime.fromisoformat(str(row[0]).replace("Z", "+00:00"))
                if oldest.tzinfo is None:
                    oldest = oldest.replace(tzinfo=timezone.utc)
            except Exception:
                return None
            return (datetime.now(timezone.utc) - oldest).total_seconds()
        except Exception as e:
            logger.error(f"DB get_oldest_unprocessed_age_sec error: {e}")
            return None

    async def get_unprocessed_recent(self, minutes: int = 3, batch_size: int = 20) -> list[aiosqlite.Row]:
        """Retrieves recent unprocessed messages within a time window, oldest-first.

        FIFO ordering (``ORDER BY timestamp ASC``) is intentional: processing
        newest-first under bursty load starves messages that arrived 5–10 minutes
        ago until they age out of the priority window entirely, causing the
        multi-minute alert latencies observed in production.

        Args:
            minutes: Time window in minutes.
            batch_size: Maximum number of messages to retrieve.

        Returns:
            A list of database rows for recent unprocessed messages, oldest first.
        """
        if not self.conn:
            return []

        try:
            cutoff = _ts_str(datetime.now(timezone.utc) - timedelta(minutes=minutes))
            # FIFO within the priority window: oldest-first so new arrivals can't
            # starve a message that's been sitting for 9 minutes.
            async with self.conn.execute(
                "SELECT id, text, timestamp, tg_msg_id, chat_id, reply_to_msg_id "
                "FROM messages WHERE processed=0 AND timestamp >= ? ORDER BY timestamp ASC LIMIT ?",
                (cutoff, batch_size)
            ) as cur:
                rows = await cur.fetchall()
                return list(rows) if rows else []
        except Exception as e:
            logger.error(f"DB get_unprocessed_recent error: {e}")
            return []

    async def mark_batch_processed(self, ids: Sequence[int]) -> None:
        """Marks a sequence of message IDs as processed.

        Args:
            ids: The sequence of internal database IDs to mark.
        """
        if not self.conn or not ids:
            return
            
        ph = ','.join('?' * len(ids))
        try:
            await self.conn.execute(
                f"UPDATE messages SET processed=1 WHERE id IN ({ph})", list(ids)
            )
            await self.conn.commit()
        except Exception as e:
            logger.error(f"DB mark_batch_processed error: {e}")

    async def mark_processed_by_tg_id(self, tg_msg_id: int, chat_id: int) -> None:
        """Marks a message as processed by its Telegram id (race-safe).

        Called from the listener's fast-path after a successful alert so the AI
        loop doesn't re-process a message that already fired. Tolerant of the
        race with `_save_to_db`: if the row isn't yet in `messages`, the UPDATE
        affects 0 rows (no error) and the row will be picked up normally by the
        AI loop later (where `_recent_alerts_history` dedup catches it).
        """
        if not self.conn:
            return
        try:
            await self.conn.execute(
                "UPDATE messages SET processed=1 "
                "WHERE tg_msg_id=? AND chat_id=? AND processed=0",
                (tg_msg_id, chat_id),
            )
            await self.conn.commit()
        except Exception as e:
            logger.error(f"DB mark_processed_by_tg_id error: {e}")

    async def increment_ai_failure_count(self, ids: Sequence[int]) -> None:
        """Increments failure count for a batch of messages. 
        
        Messages with high failure counts (>=3) are eventually marked processed 
        to prevent permanent loop on 'poison' messages.
        """
        if not self.conn or not ids:
            return
        ph = ','.join('?' * len(ids))
        try:
            # Increment count
            await self.conn.execute(
                f"UPDATE messages SET ai_failure_count = ai_failure_count + 1 WHERE id IN ({ph})",
                list(ids)
            )
            # Mark those that reached 3 failures as processed so they don't block the queue forever
            await self.conn.execute(
                f"UPDATE messages SET processed=1 WHERE id IN ({ph}) AND ai_failure_count >= 3",
                list(ids)
            )
            await self.conn.commit()
        except Exception as e:
            logger.error(f"DB increment_ai_failure_count error: {e}")

    async def get_last_msg_id(self, chat_id: int) -> int:
        """Retrieves the highest Telegram message ID seen in a chat.

        Args:
            chat_id: The chat ID to check.

        Returns:
            The maximum tg_msg_id found, or 0.
        """
        if not self.conn:
            return 0
            
        async with self.conn.execute(
            "SELECT MAX(tg_msg_id) FROM messages WHERE chat_id=?", (chat_id,)
        ) as cur:
            row = await cur.fetchone()
            return cast(int, row[0]) if row and row[0] else 0

    async def get_last_msg_timestamp(self, chat_id: int) -> datetime | None:
        """Retrieves the timestamp of the last message in a chat.

        Args:
            chat_id: The chat ID to check.

        Returns:
            A UTC-aware datetime or None.
        """
        if not self.conn:
            return None
            
        async with self.conn.execute(
            "SELECT MAX(timestamp) FROM messages WHERE chat_id=?", (chat_id,)
        ) as cur:
            row = await cur.fetchone()
            ts = row[0] if row and row[0] else None
            
        if not ts:
            async with self.conn.execute("SELECT MAX(timestamp) FROM messages") as cur:
                row = await cur.fetchone()
                ts = row[0] if row and row[0] else None
                
        if ts:
            s = str(ts).replace('Z', '+00:00')
            from shared import _parse_ts
            return _parse_ts(s)
        return None

    # ── Promos ────────────────────────────────────────────────────────────────

    async def save_promos_batch(self, promos_to_save: Sequence[tuple[int, Any, str]], processed_msg_ids: Sequence[int]) -> bool:
        """Persists AI-extracted promos and marks source messages as processed.

        Uses UPSERT logic to handle re-active promos.

        Args:
            promos_to_save: Sequence of (source_id, promo_data, tg_link) tuples.
            processed_msg_ids: Sequence of internal database IDs to mark processed.

        Returns:
            True if successful, False otherwise.
        """
        if not self.conn:
            return False
            
        try:
            await self.conn.execute("BEGIN")
            
            # Prepare bulk data
            promo_data = []
            for source_id, p, link in promos_to_save:
                clean_brand = normalize_brand(p.brand)
                if clean_brand == "Unknown" and (not p.summary or len(p.summary) < 15):
                    continue
                status = p.status if p.status in ('active', 'expired', 'unknown') else 'unknown'
                valid_until = (getattr(p, 'valid_until', '') or '').strip()
                promo_data.append((source_id, p.summary, clean_brand, p.conditions or '',
                                   link, status, valid_until))

            if promo_data:
                # Persist `valid_until` too so the time-reminder job can find
                # promos with a time-of-day window (e.g. "s/d 12:00", "jam 14").
                await self.conn.executemany("""
                    INSERT INTO promos
                        (source_msg_id, summary, brand, conditions, tg_link, status, via_fastpath, valid_until)
                    VALUES (?, ?, ?, ?, ?, ?, 0, ?)
                    ON CONFLICT(brand, summary) DO UPDATE SET
                        status        = excluded.status,
                        tg_link       = excluded.tg_link,
                        source_msg_id = COALESCE(excluded.source_msg_id, source_msg_id),
                        valid_until   = CASE
                            WHEN excluded.valid_until != '' THEN excluded.valid_until
                            ELSE valid_until
                        END,
                        created_at    = strftime('%Y-%m-%d %H:%M:%S+00:00','now')
                """, promo_data)

            if processed_msg_ids:
                ph = ','.join('?' * len(processed_msg_ids))
                await self.conn.execute(
                    f"UPDATE messages SET processed=1 WHERE id IN ({ph})",
                    list(processed_msg_ids)
                )
            await self.conn.commit()
            return True
        except Exception as e:
            await self.conn.rollback()
            logger.error(f"DB save_promos_batch error: {e}")
            return False

    async def save_fastpath_promo(self, brand: str, summary: str, conditions: str,
                                  tg_link: str, status: str = 'active') -> None:
        """Persists a fast-path detected promo to the database.

        Args:
            brand: The brand name.
            summary: Brief summary of the promo.
            conditions: Any extracted conditions.
            tg_link: Direct link to the Telegram message.
            status: Initial status ('active', 'expired', etc.).
        """
        if not self.conn:
            return
            
        clean_brand = normalize_brand(brand)
        status = status if status in ('active', 'expired', 'unknown') else 'active'
        try:
            await self.conn.execute("""
                INSERT INTO promos
                    (source_msg_id, summary, brand, conditions, tg_link, status, via_fastpath)
                VALUES (NULL, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(brand, summary) DO UPDATE SET
                    status     = excluded.status,
                    tg_link    = excluded.tg_link,
                    created_at = strftime('%Y-%m-%d %H:%M:%S+00:00','now')
            """, (summary, clean_brand, conditions or '', tg_link, status))
            await self.conn.commit()
        except Exception as e:
            logger.error(f"DB save_fastpath_promo error: {e}")

    async def get_promos(self, hours: float | None = None, limit: int | None = None, 
                         since_dt: datetime | None = None) -> list[aiosqlite.Row]:
        """Retrieves promos from the database based on time or limit.

        Args:
            hours: Lookback window in hours.
            limit: Maximum number of promos to return.
            since_dt: Absolute UTC datetime to look back to.

        Returns:
            A list of database rows for matching promos.
        """
        if not self.conn:
            return []
            
        query = """
            SELECT p.brand, p.summary, p.conditions, p.tg_link, p.status,
                   COALESCE(m.timestamp, p.created_at) AS msg_time,
                   p.created_at, p.via_fastpath
            FROM promos p
            LEFT JOIN messages m ON p.source_msg_id = m.id
        """
        params: list[Any] = []
        conds: list[str]  = []

        if since_dt:
            cutoff = _ts_str(since_dt)
            conds.append("COALESCE(m.timestamp, p.created_at) >= ?")
            params.append(cutoff)
        elif hours is not None:
            conds.append(
                "COALESCE(m.timestamp, p.created_at) >= "
                "strftime('%Y-%m-%d %H:%M:%S+00:00','now',?)"
            )
            params.append(f'-{hours} hours')

        if conds:
            query += " WHERE " + " AND ".join(conds)
        query += " ORDER BY COALESCE(m.timestamp, p.created_at) ASC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)

        async with self.conn.execute(query, params) as cur:
            rows = await cur.fetchall()
            return list(rows)

    async def get_recent_alert_brands(self, hours: int = 6, limit: int = 300) -> list[aiosqlite.Row]:
        """Retrieves recently alerted brands for deduplication history.

        Args:
            hours: Lookback window in hours.
            limit: Maximum number of entries to return.

        Returns:
            A list of database rows (brand, summary).
        """
        if not self.conn:
            return []
            
        async with self.conn.execute("""
            SELECT DISTINCT brand, summary FROM promos
            WHERE created_at >= strftime('%Y-%m-%d %H:%M:%S+00:00','now',?)
            ORDER BY id DESC LIMIT ?
        """, (f'-{hours} hours', limit)) as cur:
            rows = await cur.fetchall()
            return list(rows)

    # ── Pending alerts ────────────────────────────────────────────────────────

    async def save_pending_alert(self, brand: str, p_data_json: str,
                                  tg_link: str, timestamp: str | datetime,
                                  corroborations: int = 0,
                                  corroboration_texts: str = '[]',
                                  source: str = 'ai',
                                  commit: bool = True) -> None:
        """Saves an alert that is waiting to be flushed to Telegram.

        Args:
            brand: Normalized brand name.
            p_data_json: JSON string of promo data.
            tg_link: Link to source message.
            timestamp: Original message timestamp.
            corroborations: Initial corroboration count.
            corroboration_texts: JSON list of snippets.
            source: Source of the alert ('ai' or 'python').
            commit: Whether to commit immediately.
        """
        if not self.conn:
            return

        ts = _ts_str(timestamp) if not isinstance(timestamp, str) else timestamp
        try:
            await self.conn.execute(
                "INSERT INTO pending_alerts "
                "(brand, p_data_json, tg_link, timestamp, corroborations, corroboration_texts, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (brand, p_data_json, tg_link, ts, corroborations, corroboration_texts, source)
            )
            if commit:
                await self.conn.commit()

        except Exception as e:
            logger.error(f"DB save_pending_alert error: {e}")

    # ── Velocity ──────────────────────────────────────────────────────────────

    async def get_brand_velocity(self, brand: str, minutes: int = 5) -> int:
        """Counts total messages in a time window as an activity proxy.

        Args:
            brand: Brand name (unused in current optimized implementation).
            minutes: Lookback window in minutes.

        Returns:
            Total message count in the window.
        """
        if not self.conn:
            return 0
            
        cutoff = _ts_str(datetime.now(timezone.utc) - timedelta(minutes=minutes))
        async with self.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE timestamp >= ?",
            (cutoff,)
        ) as cur:
            row = await cur.fetchone()
            return cast(int, row[0]) if row else 0

    async def get_queue_size(self) -> int:
        """Retrieves the current count of unprocessed messages.

        Returns:
            Total count of unprocessed messages.
        """
        if not self.conn:
            return 0
            
        async with self.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE processed=0"
        ) as cur:
            row = await cur.fetchone()
            return cast(int, row[0]) if row else 0

    # ── Context / thread helpers ──────────────────────────────────────────────

    async def get_reply_sources_bulk(self, reply_msg_ids: Sequence[int], chat_id: int) -> dict[int, str]:
        """Bulk retrieves text of messages being replied to.

        Args:
            reply_msg_ids: Sequence of Telegram message IDs to look up.
            chat_id: Telegram chat ID.

        Returns:
            A dictionary mapping tg_msg_id to its text content.
        """
        if not self.conn or not reply_msg_ids:
            return {}
            
        unique_ids = list({i for i in reply_msg_ids if i})
        if not unique_ids:
            return {}
            
        ph = ','.join('?' * len(unique_ids))
        async with self.conn.execute(
            f"SELECT tg_msg_id, text FROM messages "
            f"WHERE tg_msg_id IN ({ph}) AND chat_id=?",
            (*unique_ids, chat_id)
        ) as cur:
            rows = await cur.fetchall()
            return {r['tg_msg_id']: r['text'] for r in rows}

    async def get_deep_context_bulk(self, reply_ids: Sequence[int], chat_id: int, max_depth: int = 3) -> dict[int, str]:
        """Bulk retrieves recursive text context for multiple messages.

        Traverses up the reply chain to provide more context for the AI.

        Returns:
            Dictionary mapping the original reply_id to a combined context string.
        """
        if not self.conn or not reply_ids:
            return {}

        results: dict[int, str] = {}
        current_to_fetch = list(set(reply_ids))
        
        # Mapping for current level lookup: child_tg_id -> parent_tg_id
        # We need to know which parent belongs to which original child.
        original_to_latest = {rid: rid for rid in current_to_fetch}
        combined_context: dict[int, list[str]] = {rid: [] for rid in current_to_fetch}

        for _ in range(max_depth):
            if not current_to_fetch:
                break
            
            ph = ','.join('?' * len(current_to_fetch))
            async with self.conn.execute(
                f"SELECT tg_msg_id, text, reply_to_msg_id FROM messages "
                f"WHERE tg_msg_id IN ({ph}) AND chat_id=?",
                (*current_to_fetch, chat_id)
            ) as cur:
                rows = await cur.fetchall()
            
            if not rows:
                break

            found_map = {r['tg_msg_id']: r for r in rows}
            new_fetch_ids = []
            
            # For each original ID, see if its latest parent was found
            for orig_id, latest_id in list(original_to_latest.items()):
                if latest_id in found_map:
                    row = found_map[latest_id]
                    if row['text']:
                        combined_context[orig_id].append(row['text'])
                    
                    # Move up the chain
                    parent_id = row['reply_to_msg_id']
                    if parent_id and parent_id != latest_id:
                        original_to_latest[orig_id] = parent_id
                        new_fetch_ids.append(parent_id)
                    else:
                        # Chain ended
                        del original_to_latest[orig_id]
                else:
                    # Parent not in DB
                    del original_to_latest[orig_id]
            
            current_to_fetch = list(set(new_fetch_ids))

        return {rid: " | ".join(reversed(texts)) for rid, texts in combined_context.items() if texts}

    async def get_thread_replies(self, parent_tg_msg_id: int,
                                  chat_id: int, limit: int = 20) -> list[aiosqlite.Row]:
        """Retrieves recent replies to a specific message.

        Args:
            parent_tg_msg_id: The Telegram ID of the parent message.
            chat_id: Telegram chat ID.
            limit: Maximum number of replies to return.

        Returns:
            A list of database rows for replies.
        """
        if not self.conn:
            return []
            
        async with self.conn.execute("""
            SELECT text, sender_name, timestamp FROM messages
            WHERE reply_to_msg_id=? AND chat_id=?
            ORDER BY id ASC LIMIT ?
        """, (parent_tg_msg_id, chat_id, limit)) as cur:
            rows = await cur.fetchall()
            return list(rows)

    async def get_hot_threads(self, minutes: int = 15, min_replies: int = 5, 
                              limit: int = 10) -> list[aiosqlite.Row]:
        """Identifies highly active discussion threads.

        Args:
            minutes: Lookback window for replies in minutes.
            min_replies: Minimum number of unique senders in the thread.
            limit: Maximum number of threads to return.

        Returns:
            A list of database rows for hot threads.
        """
        if not self.conn:
            return []
            
        async with self.conn.execute(f"""
            SELECT
                m_parent.id, m_parent.tg_msg_id, m_parent.chat_id,
                m_parent.text, m_parent.sender_name, m_parent.timestamp,
                m_parent.has_photo,
                COUNT(DISTINCT m_reply.sender_id) AS reply_count
            FROM messages m_parent
            JOIN messages m_reply
                ON  m_reply.reply_to_msg_id = m_parent.tg_msg_id
                AND m_reply.chat_id = m_parent.chat_id
            WHERE m_reply.timestamp  >= strftime('%Y-%m-%d %H:%M:%S+00:00','now',?)
              AND m_parent.timestamp >= strftime('%Y-%m-%d %H:%M:%S+00:00','now','-120 minutes')
            GROUP BY m_parent.id
            HAVING reply_count >= ?
            ORDER BY reply_count DESC
            LIMIT ?
        """, (f"-{minutes} minutes", min_replies, limit)) as cur:
            rows = await cur.fetchall()
            return list(rows)

    async def get_recent_messages(self, minutes: int = 10) -> list[aiosqlite.Row]:
        """Retrieves messages from the last N minutes.

        Args:
            minutes: Lookback window in minutes.

        Returns:
            A list of database rows for recent messages.
        """
        if not self.conn:
            return []
            
        async with self.conn.execute("""
            SELECT id, tg_msg_id, chat_id, text, sender_id,
                   sender_name, reply_to_msg_id, timestamp
            FROM messages
            WHERE timestamp >= strftime('%Y-%m-%d %H:%M:%S+00:00','now',?)
            ORDER BY timestamp DESC
        """, (f'-{minutes} minutes',)) as cur:
            rows = await cur.fetchall()
            return list(rows)

    async def get_last_n_messages(self, n: int = 50) -> list[aiosqlite.Row]:
        """Retrieves the last N messages regardless of time.

        Args:
            n: Number of messages to retrieve.

        Returns:
            A list of database rows.
        """
        if not self.conn:
            return []
            
        async with self.conn.execute("""
            SELECT text, sender_name, timestamp
            FROM messages ORDER BY id DESC LIMIT ?
        """, (n,)) as cur:
            rows = await cur.fetchall()
            return list(rows)

    async def get_recent_words(self, minutes: int = 20) -> list[tuple[str, int]]:
        """Calculates word frequency for recent traffic.

        Args:
            minutes: Lookback window in minutes.

        Returns:
            A sorted list of (word, count) tuples.
        """
        if not self.conn:
            return []
            
        STOP = {
            'yang','yg','iya','gak','ada','aku','kak','juga','gais','bisa','ga','ya',
            'di','ke','dan','atau','tapi','pls','nih','ini','itu','udah','dah','nggk',
            'lagi','aja','kok','kalo','smpe','dri','dr','tdk','blm','mau','tanya','tau',
            'lho','sih','dong','deh','banget','emang','kayak','terus','jadi','sama',
            'kaya','punya','abis','habis','dengan','untuk','dari','buat','biasa','kalau',
            'kayaknya','pake','boleh','nuker','beli','tadi','masuk','minta','coba','kasih',
        }
        async with self.conn.execute("""
            SELECT text FROM messages
            WHERE timestamp >= strftime('%Y-%m-%d %H:%M:%S+00:00','now',?)
        """, (f'-{minutes} minutes',)) as cur:
            rows = await cur.fetchall()

        freq: dict[str, int] = {}
        for row in rows:
            seen_in_msg: set[str] = set()
            for word in (row[0] or '').lower().split():
                word = word.strip('.,!?:;()"\'')
                if (len(word) > 3 and word not in STOP
                        and not word.startswith('http')
                        and word not in seen_in_msg):
                    freq[word] = freq.get(word, 0) + 1
                    seen_in_msg.add(word)
        return sorted(freq.items(), key=lambda x: -x[1])

    async def get_message_with_context(self, msg_id: int, chat_id: int, window: int = 5) -> list[aiosqlite.Row]:
        """Retrieves a message and its immediate surrounding context.

        Args:
            msg_id: The internal database ID of the message.
            chat_id: Telegram chat ID.
            window: Number of messages to include in the context.

        Returns:
            A list of database rows.
        """
        if not self.conn:
            return []
            
        async with self.conn.execute("""
            SELECT text, sender_name, timestamp FROM messages
            WHERE chat_id=? AND id<=? AND id>?-?
            ORDER BY id ASC
        """, (chat_id, msg_id, msg_id, window)) as cur:
            rows = await cur.fetchall()
            return list(rows)

    async def get_reply_source(self, reply_to_msg_id: int | None, chat_id: int) -> str | None:
        """Retrieves the text of a single message by Telegram ID.

        Args:
            reply_to_msg_id: The Telegram ID of the message to look up.
            chat_id: Telegram chat ID.

        Returns:
            The text content if found, else None.
        """
        if not self.conn or not reply_to_msg_id:
            return None
            
        async with self.conn.execute(
            "SELECT text FROM messages WHERE tg_msg_id=? AND chat_id=?",
            (reply_to_msg_id, chat_id)
        ) as cur:
            row = await cur.fetchone()
            return cast(str, row['text']) if row else None

    # ── Maintenance ───────────────────────────────────────────────────────────

    async def prune_old_messages(self) -> None:
        """Deletes processed messages older than 1 day that are not backing a promo."""
        if not self.conn:
            return
            
        try:
            await self.conn.execute("""
                DELETE FROM messages
                WHERE processed=1
                AND timestamp < strftime('%Y-%m-%d %H:%M:%S+00:00','now','-1 day')
                AND id NOT IN (
                    SELECT source_msg_id FROM promos
                    WHERE source_msg_id IS NOT NULL
                )
            """)
            await self.conn.commit()
            
            # Checkpoint after commit. Use PASSIVE to avoid 'database is locked' errors
            # if other tasks are reading/writing.
            await self.conn.execute("PRAGMA wal_checkpoint(PASSIVE)")

            # VACUUM only if fragmentation is significant — check freelist ratio
            async with self.conn.execute("PRAGMA freelist_count") as cur:
                free = (await cur.fetchone())[0]
            async with self.conn.execute("PRAGMA page_count") as cur:
                total = (await cur.fetchone())[0]
            if total > 0 and free / total > 0.20:
                logger.info(f"Running VACUUM (fragmentation: {100*free//total}%)")
                await self.conn.execute("VACUUM")

            logger.info("Database maintenance: OLD messages pruned, WAL checkpointed.")
        except Exception as e:
            logger.error(f"DB prune_old_messages error: {e}")

    async def recover_stuck_alerts(self) -> int:
        """Un-sticks pending_alerts orphaned by mid-flush crashes.

        Returns:
            The number of alerts unstuck.
        """
        if not self.conn:
            return 0
            
        try:
            cutoff_dt = datetime.now(timezone.utc) - timedelta(minutes=3)
            # Use BOTH formats to handle legacy T-separator rows and space-separator rows
            cutoff_space = _ts_str(cutoff_dt)                                      # space format
            cutoff_T     = cutoff_dt.strftime('%Y-%m-%dT%H:%M:%S+00:00')           # T format
            async with self.conn.execute(
                "SELECT COUNT(*) FROM pending_alerts "
                "WHERE flush_id IS NOT NULL AND (created_at < ? OR created_at < ?)",
                (cutoff_space, cutoff_T)
            ) as cur:
                row = await cur.fetchone()
                stuck = cast(int, row[0]) if row else 0
                
            if stuck:
                logger.warning(f"Watchdog found {stuck} stuck alerts. Recovering...")
                await self.conn.execute(
                    "UPDATE pending_alerts SET flush_id=NULL "
                    "WHERE flush_id IS NOT NULL AND (created_at < ? OR created_at < ?)",
                    (cutoff_space, cutoff_T)
                )
                await self.conn.commit()
                return stuck
            
            return 0
        except Exception as e:
            logger.error(f"DB recover_stuck_alerts error: {e}")
            return 0
