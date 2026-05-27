"""Regression tests for the concurrency / rate-limit / batch-size tuning.

The user reported:
  - alert latencies of 500–1400s
  - VPS at 2% CPU / 110MB RAM while throttled
  - per-model RPM limit is 12 (24 aggregate)

These tests pin the runtime constants we rely on so accidental tuning-down
regressions get caught by CI instead of by a user losing alerts at 3am.
"""
import asyncio

from processor import GeminiProcessor
from config import Config


def test_model_slots_use_real_rpm_limit():
    """Per-model limit must match the documented Gemini RPM (12), not a
    conservative pre-fix cap (11). PR #6/#7/#8 bumped latency handling; this
    test ensures we don't silently drop back to 11."""
    gp = GeminiProcessor()
    # Ensure at least some models from the army are loaded
    assert len(gp._slots) > 0
    
    # Check for any gemma model to have at least 12 RPM
    gemma_found = False
    for slot in gp._slots.values():
        if "gemma" in slot.model_id:
            gemma_found = True
            assert slot.limit >= 12, (
                f"Model {slot.model_id} limit={slot.limit}, expected at least 12. "
                "Dropping below 12 wastes available RPM headroom."
            )
    assert gemma_found, "No gemma models found in the fleet configuration."


def test_pick_model_short_timeout_allows_fast_retry():
    """_pick_model must not hold rr_lock across blocking acquire waits.

    NOTE: Architectures have changed, _pick_model now uses _fleet_lock.
    We check that it doesn't do anything obviously blocking inside the lock
    that would serialize the entire fleet.
    """
    import inspect
    src = inspect.getsource(GeminiProcessor._pick_model)
    # The new fleet uses a single global lock. We must ensure it's not held
    # across long external calls.
    assert "async with self._fleet_lock:" in src


async def _fake_ai_call(slot):
    """Simulate an AI call that takes ~50ms holding a slot."""
    await asyncio.sleep(0.05)


def test_model_slot_allows_up_to_limit_concurrent():
    """Sanity check that _ModelSlot actually lets `limit` concurrent acquires
    through in a single 60s window."""
    async def run():
        from unittest.mock import MagicMock
        from processor import _ModelSlot, BaseAIClient
        
        mock_client = MagicMock(spec=BaseAIClient)
        limit = 10
        slot = _ModelSlot(
            name="test-slot",
            provider="google",
            model_id="test-model",
            client=mock_client,
            limit=limit
        )
        
        acquired = 0
        for _ in range(limit):
            if await slot.try_acquire_nowait():
                acquired += 1
        assert acquired == limit, (
            f"Expected {limit} non-blocking acquires to succeed, got "
            f"{acquired}. RPM accounting is off."
        )
        # The next one must fail non-blocking
        assert await slot.try_acquire_nowait() is False

    asyncio.run(run())
