import sqlite3
from datetime import datetime
from config import Config

class Database:
    def __init__(self):
        self.conn = sqlite3.connect(Config.DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tg_msg_id INTEGER,
                    chat_id INTEGER,
                    sender_id INTEGER,
                    sender_name TEXT,
                    timestamp DATETIME,
                    text TEXT,
                    reply_to_msg_id INTEGER,
                    processed BOOLEAN DEFAULT 0,
                    UNIQUE(tg_msg_id, chat_id)
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS promos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_msg_id INTEGER,
                    summary TEXT,
                    brand TEXT,
                    conditions TEXT,
                    tg_link TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_msg_id) REFERENCES messages (id)
                )
            """)

    # ── Core message/promo ops ───────────────────────────────────────────────

    def save_message(self, tg_msg_id, chat_id, sender_id, sender_name, timestamp, text, reply_to_msg_id):
        try:
            with self.conn:
                cursor = self.conn.execute("""
                    INSERT OR IGNORE INTO messages (tg_msg_id, chat_id, sender_id, sender_name, timestamp, text, reply_to_msg_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (tg_msg_id, chat_id, sender_id, sender_name, timestamp, text, reply_to_msg_id))
                return cursor.lastrowid
        except Exception as e:
            print(f"DB Error (save_message): {e}")
            return None

    def get_unprocessed_batch(self, batch_size=50):
        cursor = self.conn.execute("""
            SELECT id, text, timestamp, sender_name, tg_msg_id, chat_id
            FROM messages WHERE processed = 0 LIMIT ?
        """, (batch_size,))
        return cursor.fetchall()

    def mark_batch_processed(self, ids):
        if not ids: return
        with self.conn:
            self.conn.execute(
                f"UPDATE messages SET processed = 1 WHERE id IN ({','.join(['?']*len(ids))})", ids
            )

    def save_promo(self, source_msg_id, p_data, tg_link):
        with self.conn:
            self.conn.execute("""
                INSERT INTO promos (source_msg_id, summary, brand, conditions, tg_link)
                VALUES (?, ?, ?, ?, ?)
            """, (source_msg_id, p_data.summary, p_data.brand, p_data.conditions, tg_link))

    def get_promos(self, hours=None, limit=None):
        query = "SELECT brand, summary, conditions, tg_link, created_at FROM promos"
        params = []
        conds = []
        if hours:
            conds.append("created_at >= datetime('now', ?)")
            params.append(f'-{hours} hours')
        if conds:
            query += " WHERE " + " AND ".join(conds)
        query += " ORDER BY id DESC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        return self.conn.execute(query, params).fetchall()

    # ── Trend / monitoring helpers ───────────────────────────────────────────

    def get_message_count_in_window(self, minutes=5):
        row = self.conn.execute("""
            SELECT COUNT(*) FROM messages
            WHERE timestamp >= datetime('now', ?)
        """, (f'-{minutes} minutes',)).fetchone()
        return row[0]

    def get_recent_messages(self, minutes=10):
        cursor = self.conn.execute("""
            SELECT id, text, sender_id, sender_name, reply_to_msg_id, timestamp
            FROM messages
            WHERE timestamp >= datetime('now', ?)
            ORDER BY timestamp DESC
        """, (f'-{minutes} minutes',))
        return cursor.fetchall()

    def get_hot_threads(self, minutes=20, min_replies=3):
        """
        Return hot threads with the original message text + link info,
        by joining reply_to_msg_id back to the originating message.
        Only returns threads not already seen before (new since last check).
        """
        cursor = self.conn.execute("""
            SELECT
                m.reply_to_msg_id AS tg_msg_id,
                COUNT(*) AS reply_count,
                orig.text AS orig_text,
                orig.sender_name AS orig_sender,
                orig.chat_id AS chat_id
            FROM messages m
            LEFT JOIN messages orig
                ON orig.tg_msg_id = m.reply_to_msg_id
                AND orig.chat_id = m.chat_id
            WHERE m.reply_to_msg_id IS NOT NULL
              AND m.timestamp >= datetime('now', ?)
            GROUP BY m.reply_to_msg_id
            HAVING reply_count >= ?
            ORDER BY reply_count DESC
        """, (f'-{minutes} minutes', min_replies))
        return cursor.fetchall()

    def get_recent_words(self, minutes=20):
        """
        Returns list of (word, count) from recent messages,
        filtered to meaningful tokens only. Caller decides what to do with them.
        """
        STOP = {
            'yang','yg','iya','gak','ada','aku','kak','juga','gais','bisa','ga','ya',
            'di','ke','dan','atau','tapi','pls','nih','ini','itu','udah','dah','nggk',
            'lagi','aja','kok','kalo','smpe','dri','dr','tdk','blm','mau','tanya',
            'tau','lho','sih','dong','deh','banget','emang','kayak','terus','jadi',
            'sama','kaya','punya','udah','ngga','nggak','enggak','abis','habis',
            'dengan','untuk','dari','juga','buat','biasa','kalau','kayaknya',
            # slang that's too generic to signal anything
            'pake','boleh','nuker','beli','tadi','masuk','minta','coba','kasih',
        }
        rows = self.conn.execute("""
            SELECT text FROM messages
            WHERE timestamp >= datetime('now', ?)
        """, (f'-{minutes} minutes',)).fetchall()

        freq = {}
        for row in rows:
            seen_in_msg = set()  # count each word once per message (prevents one person spamming)
            for word in row['text'].lower().split():
                word = word.strip('.,!?:;()"\'')
                if (
                    len(word) > 3
                    and word not in STOP
                    and not word.startswith('http')
                    and word not in seen_in_msg
                ):
                    freq[word] = freq.get(word, 0) + 1
                    seen_in_msg.add(word)

        return sorted(freq.items(), key=lambda x: -x[1])

    def get_time_mentions(self, minutes=5):
        """
        Return recent messages that contain a time reference (jam X, HH:MM, etc).
        Only looks in the freshest window so we catch them as they arrive.
        """
        import re
        TIME_RE = re.compile(
            r'\bjam\s*\d{1,2}(?:[.:]\d{2})?\b'
            r'|\b\d{1,2}[.:]\d{2}\b'
            r'|\b\d{1,2}\s*(?:pagi|siang|sore|malam|wib|wita|wit)\b',
            re.IGNORECASE
        )
        rows = self.conn.execute("""
            SELECT id, tg_msg_id, chat_id, sender_name, text, timestamp
            FROM messages
            WHERE timestamp >= datetime('now', ?)
            ORDER BY timestamp DESC
        """, (f'-{minutes} minutes',)).fetchall()

        return [r for r in rows if TIME_RE.search(r['text'])]
