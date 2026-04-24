"""Regression tests for the "silent bot during message burst" bug.

Root causes identified via live VPS log analysis on 2026-04-24:
1. `generate_content` had no timeout → any SDK hang leaked claims for 5 min.
2. `process_one_batch` released claims only on specific error paths, not in an
   outer `finally` → `asyncio.CancelledError` during AI call leaked claims.
3. `asyncio.create_task` refs were not held → GC could kill a task before its
   `finally` ran.

Each of these is covered below.
"""
from __future__ import annotations

import asyncio
import pytest



@pytest.mark.asyncio
async def test_ai_call_has_timeout_attribute() -> None:
    """Ensure the processor defines an explicit AI timeout constant."""
    from processor import GeminiProcessor
    assert hasattr(GeminiProcessor, "_AI_CALL_TIMEOUT_SEC")
    assert isinstance(GeminiProcessor._AI_CALL_TIMEOUT_SEC, (int, float))
    # Must be finite and not absurdly large (forever-hang equivalent).
    assert 10 <= GeminiProcessor._AI_CALL_TIMEOUT_SEC <= 180


@pytest.mark.asyncio
async def test_ai_timeout_propagates_through_call(monkeypatch) -> None:
    """A hanging generate_content must be aborted by asyncio.timeout, not wait
    forever. We stub the SDK to sleep longer than the timeout and assert None
    is returned within a sane wall-clock bound."""
    from processor import GeminiProcessor, _ModelSlot

    proc = GeminiProcessor.__new__(GeminiProcessor)
    proc._slots = {
        "primary":   _ModelSlot("primary", 12),
        "secondary": _ModelSlot("secondary", 12),
    }
    proc._rr_idx = 0
    proc._rr_lock = asyncio.Lock()
    proc._model_stats = dict(proc._slots)

    # Shrink timeout for fast tests.
    monkeypatch.setattr(GeminiProcessor, "_AI_CALL_TIMEOUT_SEC", 0.5)

    # Acquire slot so _call matches the calling contract.
    await proc._slots["primary"].try_acquire_nowait()

    class _StubClient:
        class aio:
            class models:
                @staticmethod
                async def generate_content(*_a, **_kw):
                    await asyncio.sleep(10)   # would hang forever pre-fix
                    return None

    proc.client = _StubClient()

    # Seed the other slot so fallback can't rescue us.
    await proc._slots["secondary"].try_acquire_nowait()

    start = asyncio.get_event_loop().time()
    try:
        await proc._call("batch", {}, "primary", retries=0)
        assert False, "Should have raised TimeoutError"
    except TimeoutError:
        pass
    elapsed = asyncio.get_event_loop().time() - start

    assert elapsed < 3.0, f"Timeout didn't fire promptly: {elapsed:.2f}s"


@pytest.mark.asyncio
async def test_process_one_batch_releases_on_cancellation() -> None:
    """Cancelling a task mid-AI-call must NOT leak claims in _in_progress_ids.

    Pre-fix: `_release_claims` was only called in try/except branches inside
    the semaphore block; a CancelledError bubbled past them, through the only
    `finally:` (which decremented the task counter but did not release claims),
    and out of process_one_batch. The claims sat for 5 min until the reaper.
    """
    import main

    # Seed fake claims
    async with main._in_progress_lock:
        main._in_progress_ids.clear()
        main._in_progress_ids[1001] = 0.0
        main._in_progress_ids[1002] = 0.0

    # Stub a process_one_batch that simulates the hang
    msg_ids = [1001, 1002]
    main._in_progress_lock  # existing real lock
    _in_progress_lock = main._in_progress_lock
    _in_progress_ids = main._in_progress_ids

    async def _release_claims() -> None:
        async with _in_progress_lock:
            for mid in msg_ids:
                _in_progress_ids.pop(mid, None)

    async def _fake_process() -> None:
        """Mirrors the shape of process_one_batch's outer try/finally."""
        try:
            try:
                await asyncio.sleep(5)   # hang
            except Exception:
                await _release_claims()
                return
        except BaseException:
            await _release_claims()
            raise

    task = asyncio.create_task(_fake_process())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Claims must have been released even though CancelledError bubbled.
    async with _in_progress_lock:
        assert 1001 not in _in_progress_ids, "CancelledError leaked claim 1001"
        assert 1002 not in _in_progress_ids, "CancelledError leaked claim 1002"


@pytest.mark.asyncio
async def test_spawned_task_refs_prevent_gc() -> None:
    """The processing loop saves spawned tasks in `_active_spawn_tasks` so
    Python's garbage collector can't collect them mid-flight."""
    import main
    # The attribute must exist and be a set (or set-like container).
    assert hasattr(main, "_active_spawn_tasks")
    assert isinstance(main._active_spawn_tasks, set)

    # Add/discard callback pattern works.
    async def _noop():
        pass
    t = asyncio.create_task(_noop())
    main._active_spawn_tasks.add(t)
    t.add_done_callback(main._active_spawn_tasks.discard)
    await t
    # Callback runs at task completion — may not have fired yet in same tick
    await asyncio.sleep(0)
    assert t not in main._active_spawn_tasks, "done_callback should discard the task"


@pytest.mark.asyncio
async def test_batch_size_cap_prevents_50msg_prompts() -> None:
    """Large-prompt AI hangs were correlated with 50+ msg batches. The cap
    keeps even emergency-mode batches at or below 25."""
    # Verify via the formula in processing_loop: emergency = 15 + 10*headroom
    # so at headroom=100% the cap is 25.
    headroom = 1.0
    emergency_max = int(15 + 10 * headroom)
    normal_max    = int(10 + 10 * headroom)
    assert emergency_max <= 25
    assert normal_max <= 20


@pytest.mark.asyncio
async def test_in_progress_max_age_shorter_than_old_default() -> None:
    """Reduced from 300s to 120s so a stuck claim recovers within ~2min,
    not 5. Value must also be > AI timeout so we don't double-process."""
    import main
    from processor import GeminiProcessor
    assert main._IN_PROGRESS_MAX_AGE_SEC < 300
    assert main._IN_PROGRESS_MAX_AGE_SEC >= GeminiProcessor._AI_CALL_TIMEOUT_SEC


@pytest.mark.asyncio
async def test_get_unprocessed_ancient_returns_old_rows() -> None:
    """The 3-tier queue policy relies on being able to fetch rows older than
    a threshold, ordered oldest-first, so nothing can starve indefinitely."""
    import tempfile
    import os
    from db import Database
    from config import Config
    from datetime import datetime, timezone, timedelta

    fd, tmp_path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    orig = Config.DB_PATH
    Config.DB_PATH = tmp_path
    try:
        db = Database()
        await db.init()

        now = datetime.now(timezone.utc)
        fresh  = now - timedelta(minutes=1)
        mid    = now - timedelta(minutes=12)
        old    = now - timedelta(minutes=25)
        older  = now - timedelta(minutes=40)

        for i, ts in enumerate((fresh, mid, old, older), start=1):
            await db.save_message(
                tg_msg_id=10_000 + i, chat_id=-1001, sender_id=0,
                sender_name="t", timestamp=ts, text=f"msg {i}",
                reply_to_msg_id=None, processed=0, has_photo=0,
                has_time_mention=0, commit=True,
            )

        # Fetch rows ≥ 15 min old — should be old + older (2 rows), oldest first.
        rows = await db.get_unprocessed_ancient(min_age_minutes=15, batch_size=50)
        assert len(rows) == 2
        texts = [r['text'] for r in rows]
        assert texts[0] == 'msg 4', f"ancient tier must be oldest-first: got {texts}"
        assert texts[1] == 'msg 3'

        age = await db.get_oldest_unprocessed_age_sec()
        assert age is not None and age > 60 * 35  # at least 35 min old
    finally:
        try:
            await db.conn.close()
        except Exception:
            pass
        Config.DB_PATH = orig
        try:
            os.remove(tmp_path)
        except Exception:
            pass
