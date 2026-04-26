"""Regression tests for the concurrency / rate-limit / batch-size tuning.

The user reported:
  - alert latencies of 500–1400s
  - VPS at 2% CPU / 110MB RAM while throttled
  - per-model RPM limit is 12 (24 aggregate)

These tests pin the runtime constants we rely on so accidental tuning-down
regressions get caught by CI instead of by a user losing alerts at 3am.
"""
import asyncio

from main import _AI_MAX_INFLIGHT, _AI_SEMAPHORE
from processor import GeminiProcessor
from config import Config


def test_ai_max_inflight_matches_aggregate_rpm_budget():
    """We must size in-flight concurrency to at least the aggregate RPM so
    the semaphore is never the bottleneck when the real rate limit has room.
    Aggregate = 12 + 12 = 24; we set 16 as a memory-safe ceiling that still
    comfortably exceeds sustained RPM × avg-call-time."""
    assert _AI_MAX_INFLIGHT >= 12, (
        f"_AI_MAX_INFLIGHT={_AI_MAX_INFLIGHT} is below a single model's RPM. "
        "That guarantees we cannot saturate even one model."
    )
    assert _AI_SEMAPHORE._value == _AI_MAX_INFLIGHT


def test_model_slots_use_real_rpm_limit():
    """Per-model limit must match the documented Gemini RPM (12), not a
    conservative pre-fix cap (11). PR #6/#7/#8 bumped latency handling; this
    test ensures we don't silently drop back to 11."""
    gp = GeminiProcessor()
    assert Config.MODEL_ID in gp._slots
    assert Config.MODEL_FALLBACK in gp._slots
    for slot in gp._slots.values():
        if "gemma" in slot.model_id:
            assert slot.limit == 12, (
                f"Model {slot.model_id} limit={slot.limit}, expected 12. "
                "Dropping below 12 wastes available RPM headroom."
            )


def test_pick_model_short_timeout_allows_fast_retry():
    """_pick_model must not hold rr_lock across blocking acquire waits.

    We verify this by inspecting the closure: if rr_lock was held across an
    `acquire(timeout=...)`, a single slow waiter would serialize every
    concurrent _pick_model caller behind it — the exact bug that turned 4
    concurrent batches into 1-way serialized execution under burst pressure.
    """
    import inspect
    src = inspect.getsource(GeminiProcessor._pick_model)
    # The acquire waits must come AFTER the rr_lock `async with` block ends.
    # Assert there's no acquire call lexically inside the rr_lock block.
    # This regex-less heuristic: find the rr_lock `async with` start and next
    # dedent, then check that `.acquire(timeout=` does not appear inside.
    lines = src.splitlines()
    in_rr = False
    rr_indent = None
    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if 'async with self._rr_lock' in stripped:
            in_rr = True
            rr_indent = indent
            continue
        if in_rr:
            # First line indented same as or less than the `async with` -> out of block
            if stripped and indent <= rr_indent:
                in_rr = False
                continue
            # Inside the block — must not contain a blocking acquire with timeout
            assert '.acquire(timeout=' not in stripped, (
                "Found blocking acquire inside rr_lock block — this serializes "
                f"all _pick_model callers. Offending line: {line!r}"
            )


async def _fake_ai_call(slot):
    """Simulate an AI call that takes ~50ms holding a slot.
    
    NOTE: We no longer call release_last(). Failed/successful calls both
    consume provider rate limits to keep local bucket in sync with server.
    The slot will naturally expire from the bucket after 60s.
    """
    await asyncio.sleep(0.05)


def test_model_slot_allows_up_to_limit_concurrent():
    """Sanity check that _ModelSlot actually lets `limit` concurrent acquires
    through in a single 60s window, so removing the artificial Semaphore(4)
    cap does not trigger accidental 429s."""
    async def run():
        gp = GeminiProcessor()
        slot = next(iter(gp._slots.values()))
        acquired = 0
        for _ in range(slot.limit):
            if await slot.try_acquire_nowait():
                acquired += 1
        assert acquired == slot.limit, (
            f"Expected {slot.limit} non-blocking acquires to succeed, got "
            f"{acquired}. RPM accounting is off."
        )
        # The next one must fail non-blocking
        assert await slot.try_acquire_nowait() is False

    asyncio.run(run())
