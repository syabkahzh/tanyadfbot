import aiosqlite
from datetime import datetime, timezone
from config import Config

class Database:
    def __init__(self):
        self.db_path = Config.DB_PATH
        self.conn = None

    async def init(self):
        self.conn = await aiosqlite.connect(self.db_path)
        self.conn.row_factory = aiosqlite.Row
        await self.conn.execute("PRAGMA journal_mode=WAL")
        await self.conn.execute("PRAGMA timezone = 'UTC'")  # explicit
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
                UNIQUE(tg_msg_id, chat_id)
            )
        """)
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS promos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_msg_id INTEGER,
                summary TEXT,
                brand TEXT,
                conditions TEXT,
                tg_link TEXT,
                status TEXT DEFAULT 'active',
                -- FIX: store as explicit UTC ISO string, not SQLite CURRENT_TIMESTAMP
                created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%S+00:00', 'now')),
                FOREIGN KEY (source_msg_id) REFERENCES messages (id)
            )
        """)
        await self.conn.commit()

    async def close(self):
        if self.conn:
            await self.conn.close()

    async def save_message(self, tg_msg_id, chat_id, sender_id, sender_name, timestamp, text, reply_to_msg_id, processed=0):
        # Normalize timestamp to UTC ISO8601 string
        if isinstance(timestamp, datetime):
            ts_str = timestamp.astimezone(timezone.utc).isoformat()
        else:
            ts_str = str(timestamp)

        try:
            cursor = await self.conn.execute("""
                INSERT OR IGNORE INTO messages (tg_msg_id, chat_id, sender_id, sender_name, timestamp, text, reply_to_msg_id, processed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (tg_msg_id, chat_id, sender_id, sender_name, ts_str, text, reply_to_msg_id, processed))
            await self.conn.commit()
            # lastrowid is 0 on IGNORE — return None to signal "not new"
            return cursor.lastrowid if cursor.lastrowid else None
        except Exception as e:
            print(f"DB Error (save_message): {e}")
            return None

    async def get_unprocessed_batch(self, batch_size=50):
        async with self.conn.execute("""
            SELECT id, text, timestamp, sender_name, tg_msg_id, chat_id
            FROM messages WHERE processed = 0 ORDER BY id ASC LIMIT ?
        """, (batch_size,)) as cursor:
            return await cursor.fetchall()

    async def save_promos_batch(self, promos_to_save: list, processed_msg_ids: list):
        try:
            for source_id, p, link in promos_to_save:
                await self.conn.execute("""
                    INSERT INTO promos (source_msg_id, summary, brand, conditions, tg_link, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (source_id, p.summary, p.brand, p.conditions, link, p.status))

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
            conds.append("m.timestamp >= strftime('%Y-%m-%dT%H:%M:%S+00:00', 'now', ?)")
            params.append(f'-{hours} hours')
        if conds:
            query += " WHERE " + " AND ".join(conds)
        query += " ORDER BY p.id DESC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)

        async with self.conn.execute(query, params) as cursor:
            return await cursor.fetchall()

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

    async def get_recent_messages(self, minutes=10):
        async with self.conn.execute("""
            SELECT id, text, sender_id, sender_name, reply_to_msg_id, timestamp
            FROM messages
            WHERE timestamp >= strftime('%Y-%m-%dT%H:%M:%S+00:00', 'now', ?)
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
            WHERE timestamp >= strftime('%Y-%m-%dT%H:%M:%S+00:00', 'now', ?)
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
