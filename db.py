import aiosqlite
from datetime import datetime, timezone
from config import Config

def _normalize_brand(brand: str) -> str:
    if not brand: return "Unknown"
    b = brand.strip()
    # Fix model hallucination artifact
    if b.lower() in ('unknown', 'sunknown', 'bunknown', ''):
        return "Unknown"
    
    # Normalize common variants
    canon = {'hokben': 'HokBen', 'hophop': 'HopHop', 'hop hop': 'HopHop',
             'shopeefood': 'ShopeeFood', 'shopee food': 'ShopeeFood',
             'gofood': 'GoFood', 'go food': 'GoFood', 'kopken': 'Kopi Kenangan',
             'kopi kenangan': 'Kopi Kenangan', 'alfamart': 'Alfamart',
             'indomaret': 'Indomaret', 'spx': 'SPX'}
    
    return canon.get(b.lower(), b.title())

class Database:
    def __init__(self):
        self.db_path = Config.DB_PATH
        self.conn = None

    async def init(self):
        self.conn = await aiosqlite.connect(self.db_path)
        self.conn.row_factory = aiosqlite.Row
        await self.conn.execute("PRAGMA journal_mode=WAL")
        await self.conn.execute("PRAGMA wal_autocheckpoint=1000")
        await self.conn.execute("PRAGMA cache_size=-8000")
        await self.conn.execute("PRAGMA synchronous=NORMAL")
        await self.conn.execute("PRAGMA timezone = 'UTC'")
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tg_msg_id INTEGER,
                chat_id INTEGER,
                sender_id INTEGER,
                sender_name TEXT,
                timestamp TEXT NOT NULL,  -- always stored as ISO8601 UTC with +00:00
                text TEXT,
                reply_to_msg_id INTEGER,
                processed BOOLEAN DEFAULT 0,
                has_photo INTEGER DEFAULT 0,
                image_processed INTEGER DEFAULT 0,
                UNIQUE(tg_msg_id, chat_id)
            )
        """)
        # Safe to run even if columns already exist — will error silently
        for col_sql in [
            "ALTER TABLE messages ADD COLUMN has_time_mention INTEGER DEFAULT 0",
            "ALTER TABLE messages ADD COLUMN time_alerted INTEGER DEFAULT 0",
            "ALTER TABLE messages ADD COLUMN has_photo INTEGER DEFAULT 0",
            "ALTER TABLE messages ADD COLUMN image_processed INTEGER DEFAULT 0",
        ]:
            try:
                await self.conn.execute(col_sql)
            except:
                pass  # column already exists
        await self.conn.commit()

        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS promos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_msg_id INTEGER,
                summary TEXT,
                brand TEXT,
                conditions TEXT,
                tg_link TEXT,
                status TEXT NOT NULL DEFAULT 'unknown' CHECK(status IN ('active', 'expired', 'unknown')),
                -- FIX: store as explicit UTC ISO string, not SQLite CURRENT_TIMESTAMP
                created_at TEXT DEFAULT (strftime('%Y-%m-%d %H:%M:%S+00:00', 'now')),
                FOREIGN KEY (source_msg_id) REFERENCES messages (id)
            )
        """)
        await self.conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_promos_dedup ON promos(brand, summary);")
        
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS pending_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                brand TEXT,
                p_data_json TEXT,
                tg_link TEXT,
                timestamp TEXT,
                created_at TEXT DEFAULT (strftime('%Y-%m-%d %H:%M:%S+00:00', 'now'))
            )
        """)
        await self.conn.commit()

    async def save_pending_alert(self, brand: str, p_data_json: str, tg_link: str, timestamp: str):
        await self.conn.execute(
            "INSERT INTO pending_alerts (brand, p_data_json, tg_link, timestamp) VALUES (?, ?, ?, ?)",
            (brand, p_data_json, tg_link, timestamp)
        )
        await self.conn.commit()

    async def get_and_clear_pending_alerts(self):
        async with self.conn.execute("SELECT brand, p_data_json, tg_link, timestamp FROM pending_alerts") as cur:
            rows = await cur.fetchall()
        await self.conn.execute("DELETE FROM pending_alerts")
        await self.conn.commit()
        return rows

    async def get_recent_alert_brands(self, hours=2, limit=100):
        """Fetch recent alerts for deduplication seeding on boot."""
        async with self.conn.execute("""
            SELECT DISTINCT brand, summary FROM promos
            WHERE created_at >= strftime('%Y-%m-%d %H:%M:%S+00:00','now',?)
            ORDER BY id DESC LIMIT ?
        """, (f'-{hours} hours', limit)) as cur:
            return await cur.fetchall()

    async def get_hot_threads(self, minutes=30, min_replies=5):
        """Find messages that have received replies from many unique users recently."""
        async with self.conn.execute("""
            SELECT 
                m_parent.id,
                m_parent.tg_msg_id,
                m_parent.chat_id,
                m_parent.text,
                m_parent.sender_name,
                m_parent.timestamp,
                COUNT(DISTINCT m_reply.sender_id) as reply_count
            FROM messages m_parent
            JOIN messages m_reply ON m_reply.reply_to_msg_id = m_parent.tg_msg_id
                AND m_reply.chat_id = m_parent.chat_id
            WHERE m_reply.timestamp >= strftime('%Y-%m-%d %H:%M:%S+00:00','now',?)
            GROUP BY m_parent.id
            HAVING reply_count >= ?
            ORDER BY reply_count DESC
            LIMIT 10
        """, (f'-{minutes} minutes', min_replies)) as cur:
            return await cur.fetchall()

    async def close(self):
        if self.conn:
            await self.conn.close()

    async def save_message(self, tg_msg_id, chat_id, sender_id, sender_name, timestamp, text, reply_to_msg_id, processed=0, has_photo=0):
        # Normalize timestamp to UTC ISO8601 string
        if isinstance(timestamp, datetime):
            ts_str = timestamp.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M:%S+00:00')
        else:
            ts_str = str(timestamp)

        try:
            cursor = await self.conn.execute("""
                INSERT OR IGNORE INTO messages (tg_msg_id, chat_id, sender_id, sender_name, timestamp, text, reply_to_msg_id, processed, has_photo)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (tg_msg_id, chat_id, sender_id, sender_name, ts_str, text, reply_to_msg_id, processed, has_photo))
            await self.conn.commit()
            
            # Use rowcount to detect if insertion actually happened (INSERT OR IGNORE returns 0 changes if ignored)
            if cursor.rowcount == 0:
                return None
            return cursor.lastrowid
        except Exception as e:
            print(f"DB Error (save_message): {e}")
            return None

    async def get_unprocessed_batch(self, batch_size=50):
        async with self.conn.execute("""
            SELECT id, text, timestamp, sender_name, tg_msg_id, chat_id, reply_to_msg_id
            FROM messages WHERE processed = 0 ORDER BY id ASC LIMIT ?
        """, (batch_size,)) as cursor:
            return await cursor.fetchall()

    async def save_promos_batch(self, promos_to_save: list, processed_msg_ids: list):
        try:
            for source_id, p, link in promos_to_save:
                # Sanitize status before insert
                status = p.status if p.status in ('active', 'expired', 'unknown') else 'unknown'
                await self.conn.execute("""
                    INSERT OR IGNORE INTO promos (source_msg_id, summary, brand, conditions, tg_link, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (source_id, p.summary, _normalize_brand(p.brand), p.conditions, link, status))

            if processed_msg_ids:
                placeholders = ','.join(['?'] * len(processed_msg_ids))
                await self.conn.execute(
                    f"UPDATE messages SET processed = 1 WHERE id IN ({placeholders})",
                    processed_msg_ids
                )
            await self.conn.commit()
            return True
        except Exception as e:
            await self.conn.rollback()
            print(f"❌ DB Batch Error: {e}")
            return False

    async def mark_batch_processed(self, ids):
        if not ids: return
        await self.conn.execute(
            f"UPDATE messages SET processed = 1 WHERE id IN ({','.join(['?']*len(ids))})", ids
        )
        await self.conn.commit()

    async def get_promos(self, hours=None, limit=None):
        """
        Returns promos joined with source message timestamps.
        FIX: compare using UTC-aware strings consistently.
        """
        query = """
            SELECT p.brand, p.summary, p.conditions, p.tg_link, p.status,
                   m.timestamp as msg_time, p.created_at
            FROM promos p
            JOIN messages m ON p.source_msg_id = m.id
        """
        params = []
        conds = []
        if hours:
            # Use strftime comparison on stored UTC ISO strings (safe for our format)
            conds.append("m.timestamp >= strftime('%Y-%m-%d %H:%M:%S+00:00', 'now', ?)")
            params.append(f'-{hours} hours')
        if conds:
            query += " WHERE " + " AND ".join(conds)
        query += " ORDER BY p.id DESC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)

        async with self.conn.execute(query, params) as cursor:
            return await cursor.fetchall()

    async def get_message_with_context(self, msg_id: int, chat_id: int, window: int = 5):
        """Fetch a message plus N messages before it in the same chat — the thread context."""
        async with self.conn.execute("""
            SELECT text, sender_name, timestamp FROM messages
            WHERE chat_id = ? AND id <= ? AND id > ? - ?
            ORDER BY id ASC
        """, (chat_id, msg_id, msg_id, window)) as cur:
            return await cur.fetchall()

    async def get_reply_source(self, reply_to_msg_id: int, chat_id: int):
        """If this message is a reply, fetch what it's replying to."""
        if not reply_to_msg_id: return None
        async with self.conn.execute("""
            SELECT text FROM messages WHERE tg_msg_id = ? AND chat_id = ?
        """, (reply_to_msg_id, chat_id)) as cur:
            row = await cur.fetchone()
            return row['text'] if row else None

    async def get_last_n_messages(self, n=50):
        async with self.conn.execute("""
            SELECT text, sender_name, timestamp
            FROM messages ORDER BY id DESC LIMIT ?
        """, (n,)) as cursor:
            return await cursor.fetchall()

    async def get_last_msg_id(self, chat_id):
        async with self.conn.execute(
            "SELECT MAX(tg_msg_id) FROM messages WHERE chat_id = ?", (chat_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row and row[0] else 0

    async def get_last_msg_timestamp(self, chat_id):
        async with self.conn.execute(
            "SELECT MAX(timestamp) FROM messages WHERE chat_id=?", (chat_id,)
        ) as cur:
            row = await cur.fetchone()
            ts = row[0] if row and row[0] else None
            
        if not ts:
            # Fallback: check global newest timestamp
            async with self.conn.execute("SELECT MAX(timestamp) FROM messages") as cur:
                row = await cur.fetchone()
                ts = row[0] if row and row[0] else None

        if ts:
            s = str(ts).replace('Z', '+00:00')
            dt = datetime.fromisoformat(s)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        return None

    async def get_context_bulk(self, chat_id: int, min_id: int, max_id: int):
        async with self.conn.execute(
            "SELECT id, text FROM messages WHERE chat_id=? AND id>=? AND id<=? ORDER BY id ASC",
            (chat_id, min_id, max_id)
        ) as cur:
            return await cur.fetchall()

    async def get_reply_sources_bulk(self, reply_msg_ids: list, chat_id: int):
        if not reply_msg_ids: return {}
        # Filter out duplicates and None
        unique_ids = list(set([i for i in reply_msg_ids if i]))
        if not unique_ids: return {}
        placeholders = ','.join('?' * len(unique_ids))
        async with self.conn.execute(
            f"SELECT tg_msg_id, text FROM messages WHERE tg_msg_id IN ({placeholders}) AND chat_id=?",
            (*unique_ids, chat_id)
        ) as cur:
            rows = await cur.fetchall()
            return {r['tg_msg_id']: r['text'] for r in rows}

    async def get_recent_messages(self, minutes=10):
        async with self.conn.execute("""
            SELECT id, text, sender_id, sender_name, reply_to_msg_id, timestamp
            FROM messages
            WHERE timestamp >= strftime('%Y-%m-%d %H:%M:%S+00:00', 'now', ?)
            ORDER BY timestamp DESC
        """, (f'-{minutes} minutes',)) as cursor:
            return await cursor.fetchall()

    async def get_recent_words(self, minutes=20):
        STOP = {
            'yang','yg','iya','gak','ada','aku','kak','juga','gais','bisa','ga','ya',
            'di','ke','dan','atau','tapi','pls','nih','ini','itu','udah','dah','nggk',
            'lagi','aja','kok','kalo','smpe','dri','dr','tdk','blm','mau','tanya','tau',
            'lho','sih','dong','deh','banget','emang','kayak','terus','jadi','sama',
            'kaya','punya','abis','habis','dengan','untuk','dari','buat','biasa','kalau',
            'kayaknya','pake','boleh','nuker','beli','tadi','masuk','minta','coba','kasih'
        }
        async with self.conn.execute("""
            SELECT text FROM messages
            WHERE timestamp >= strftime('%Y-%m-%d %H:%M:%S+00:00', 'now', ?)
        """, (f'-{minutes} minutes',)) as cursor:
            rows = await cursor.fetchall()

        freq = {}
        for row in rows:
            seen_in_msg = set()
            for word in row[0].lower().split():
                word = word.strip('.,!?:;()"\'')
                if len(word) > 3 and word not in STOP and not word.startswith('http') and word not in seen_in_msg:
                    freq[word] = freq.get(word, 0) + 1
                    seen_in_msg.add(word)
        return sorted(freq.items(), key=lambda x: -x[1])
