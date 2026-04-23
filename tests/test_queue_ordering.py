"""Regression tests for the processing_loop priority/backlog queue.

Covers the "major delay bug" where messages could sit unprocessed for tens of
minutes because:
  1. `get_unprocessed_recent` returned rows in timestamp DESC (LIFO) order
     within the priority window, so older-but-still-fresh messages kept getting
     jumped by newer arrivals.
  2. Backlog was only fetched when priority was underfull, which never happened
     in a busy group, so messages that aged out of the 10-min window sat
     forever as `processed=0`.
"""
import asyncio
import tempfile
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

from db import Database, _ts_str


@pytest_asyncio.fixture
async def db_fixture():
    """A fresh Database backed by a tempfile per test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    import config
    orig = config.Config.DB_PATH
    config.Config.DB_PATH = path
    try:
        db = Database()  # picks up the patched Config.DB_PATH in __init__
        await db.init()
        yield db
    finally:
        if db.conn:
            await db.conn.close()
        config.Config.DB_PATH = orig


async def _insert(db: Database, tg_msg_id: int, ts: datetime, text: str = "promo sfood aman 50rb") -> int:
    """Insert a single unprocessed message and return its internal id."""
    assert db.conn is not None
    cur = await db.conn.execute(
        """
        INSERT INTO messages
            (tg_msg_id, chat_id, sender_id, sender_name, timestamp,
             text, reply_to_msg_id, processed, has_photo, has_time_mention)
        VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, 0)
        """,
        (tg_msg_id, -100123, 1, "u", _ts_str(ts), text, None),
    )
    await db.conn.commit()
    assert cur.lastrowid is not None
    return cur.lastrowid


@pytest.mark.asyncio
async def test_get_unprocessed_recent_returns_oldest_first(db_fixture):
    """Regression: the priority window must drain FIFO (oldest first).

    Previously used ORDER BY timestamp DESC, which let new arrivals starve
    older-but-still-fresh messages indefinitely under sustained traffic.
    """
    db = db_fixture
    now = datetime.now(timezone.utc)

    oldest_id = await _insert(db, 1, now - timedelta(minutes=9))
    middle_id = await _insert(db, 2, now - timedelta(minutes=5))
    newest_id = await _insert(db, 3, now - timedelta(seconds=10))

    rows = await db.get_unprocessed_recent(minutes=10, batch_size=10)
    ids = [r["id"] for r in rows]
    assert ids == [oldest_id, middle_id, newest_id], (
        "priority window must be oldest-first to prevent starvation"
    )


@pytest.mark.asyncio
async def test_priority_window_respects_cutoff(db_fixture):
    """Messages older than the cutoff must NOT appear in the priority window.

    Those stale messages must be drained by the backlog query
    (`get_unprocessed_batch`) instead.
    """
    db = db_fixture
    now = datetime.now(timezone.utc)

    stale_id = await _insert(db, 10, now - timedelta(minutes=30))
    fresh_id = await _insert(db, 11, now - timedelta(minutes=2))

    rows = await db.get_unprocessed_recent(minutes=10, batch_size=10)
    ids = [r["id"] for r in rows]
    assert ids == [fresh_id]

    backlog = await db.get_unprocessed_batch(batch_size=10)
    backlog_ids = [r["id"] for r in backlog]
    assert stale_id in backlog_ids, "stale messages must be reachable via backlog"
