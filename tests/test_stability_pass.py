"""Regression tests for the stability pass (April 2026).

Covers:
- Atomic dedup reservation (cross-batch dup race fix — 4x Sayurbox bug)
- Intra-batch fuzzy dedup (single batch can't fire 4 near-dupes)
- filter_duplicates no longer self-locks (race-free under concurrent calls)
- _extract_time_of_day correctness for sinyal waktu reminder
- _in_progress_ids stuck-claim reaper semantics
- bot.alert_error rate-limiting
"""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from processor import GeminiProcessor, PromoExtraction


def _mk_promo(brand: str, summary: str, status: str = "active") -> PromoExtraction:
    return PromoExtraction(
        original_msg_id=1,
        summary=summary,
        brand=brand,
        conditions="",
        valid_until="",
        status=status,  # type: ignore[arg-type]
        links=[],
    )


# ── filter_duplicates: intra-batch dedup ──────────────────────────────────────

@pytest.mark.asyncio
async def test_filter_duplicates_intra_batch_dedup():
    """A single batch with 4 phrasings of the same Sayurbox promo keeps ≤ 1."""
    proc = GeminiProcessor.__new__(GeminiProcessor)
    proc._dedup_lock = asyncio.Lock()

    promos = [
        _mk_promo("Sayurbox", "Promo Guncang 5.5 Sayurbox: Daging Sapi Rendang 500g diskon 81% Rp16.480"),
        _mk_promo("Sayurbox", "Sayurbox Daging Sapi Rendang 500g harga Rp20.600 per pack"),
        _mk_promo("Sayurbox", "Sayurbox menawarkan Daging Sapi Rendang 500 gram seharga Rp20.600"),
        _mk_promo("Sayurbox", "Sayurbox promo Daging Sapi Rendang 500 gr seharga Rp20.600 per unit"),
    ]
    out = await proc.filter_duplicates(promos, recent_alerts=[])
    assert len(out) == 1, f"expected 1 survivor, got {len(out)}: {[p.summary for p in out]}"


@pytest.mark.asyncio
async def test_filter_duplicates_different_brands_all_pass():
    """Different-brand promos in the same batch all pass through."""
    proc = GeminiProcessor.__new__(GeminiProcessor)
    proc._dedup_lock = asyncio.Lock()

    promos = [
        _mk_promo("Sayurbox", "Daging Sapi Rendang 500g Rp20.600"),
        _mk_promo("Alfamart", "Cashback ShopeePay 100rb"),
        _mk_promo("Tokopedia", "Flash sale serba 5rb"),
    ]
    out = await proc.filter_duplicates(promos, recent_alerts=[])
    assert len(out) == 3


@pytest.mark.asyncio
async def test_filter_duplicates_respects_history_snapshot():
    """If history already contains a similar promo, the new one is dropped."""
    proc = GeminiProcessor.__new__(GeminiProcessor)
    proc._dedup_lock = asyncio.Lock()

    history = [{"brand": "Sayurbox", "summary": "Sayurbox Daging Sapi Rendang 500g Rp20.600"}]
    new = [_mk_promo("Sayurbox", "Sayurbox menawarkan Daging Sapi Rendang 500 gram seharga Rp20.600")]

    out = await proc.filter_duplicates(new, recent_alerts=history)
    assert out == []


@pytest.mark.asyncio
async def test_concurrent_batches_with_shared_history_only_one_survives():
    """Simulate the 4x-Sayurbox race: two batches filter against the SAME
    shared history under a SHARED outer lock. The outer lock + pre-reserve
    must make the reservation atomic so batch 2 sees batch 1's reservation.
    """
    proc = GeminiProcessor.__new__(GeminiProcessor)
    proc._dedup_lock = asyncio.Lock()

    shared_history: list[dict] = []
    outer_lock = asyncio.Lock()

    async def one_batch(summary: str):
        """Mimic main.process_one_batch's atomic reserve pattern."""
        async with outer_lock:
            filtered = await proc.filter_duplicates(
                [_mk_promo("Sayurbox", summary)], shared_history
            )
            # Pre-reserve inside the lock (exactly like main.py now does).
            for p in filtered:
                shared_history.append({"brand": p.brand, "summary": p.summary})
            return filtered

    # Fire off 4 near-identical batches concurrently, same as the production bug.
    results = await asyncio.gather(
        one_batch("Sayurbox Daging Sapi Rendang 500g Rp20.600"),
        one_batch("Sayurbox menawarkan Daging Sapi Rendang 500 gram Rp20.600"),
        one_batch("Sayurbox promo Daging Sapi Rendang 500 gr Rp20.600 per unit"),
        one_batch("Promo Guncang Sayurbox Daging Sapi Rendang 500g diskon 81%"),
    )
    total_survivors = sum(len(r) for r in results)
    assert total_survivors == 1, (
        f"race reintroduced: expected 1 alert, got {total_survivors}. "
        f"Survivors: {[r for r in results if r]}"
    )


@pytest.mark.asyncio
async def test_filter_duplicates_no_longer_takes_internal_lock():
    """Concurrent filter_duplicates calls don't serialise on the old _dedup_lock.

    Regression pin: if someone re-adds `async with self._dedup_lock` around the
    whole filter, this will still pass (correctness), but the OUTER caller
    lock in main.process_one_batch is what makes it atomic. This test just
    proves the function is safe to call concurrently without deadlock.
    """
    proc = GeminiProcessor.__new__(GeminiProcessor)
    proc._dedup_lock = asyncio.Lock()

    async def one():
        return await proc.filter_duplicates(
            [_mk_promo("Tokopedia", "Flash sale 5rb hari ini")],
            recent_alerts=[],
        )

    results = await asyncio.gather(*[one() for _ in range(8)])
    # All 8 calls complete; each sees an empty history and returns 1 promo.
    assert all(len(r) == 1 for r in results)


# ── sinyal waktu: _extract_time_of_day ───────────────────────────────────────

def test_extract_time_of_day_hhmm():
    from jobs import _extract_time_of_day
    assert _extract_time_of_day("promo sampai 12:00 WIB") == (12, 0)
    assert _extract_time_of_day("jam 14:30 buruan") == (14, 30)
    assert _extract_time_of_day("s/d 23:59") == (23, 59)
    assert _extract_time_of_day("pukul 09.00 WIB") == (9, 0)


def test_extract_time_of_day_jam_only():
    from jobs import _extract_time_of_day
    assert _extract_time_of_day("jam 10 aman") == (10, 0)
    assert _extract_time_of_day("pukul 3 sore") == (15, 0)
    assert _extract_time_of_day("sampai jam 7 malam") == (19, 0)


def test_extract_time_of_day_returns_none_for_prices():
    from jobs import _extract_time_of_day
    # Bare numbers that aren't times should not match.
    assert _extract_time_of_day("diskon 50rb") is None
    assert _extract_time_of_day("harga Rp20.600") is None


# ── stuck-claim reaper ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_in_progress_reaper_evicts_stale():
    """Entries older than _IN_PROGRESS_MAX_AGE_SEC are evicted; fresh stay."""
    import main as mainmod

    # Snapshot & reset
    saved = dict(mainmod._in_progress_ids)
    mainmod._in_progress_ids.clear()
    try:
        now = time.monotonic()
        mainmod._in_progress_ids[1] = now - 400   # stale (>130s)
        mainmod._in_progress_ids[2] = now - 50    # fresh (<130s)
        mainmod._in_progress_ids[3] = now - 600   # very stale

        await mainmod._in_progress_reaper()

        assert 1 not in mainmod._in_progress_ids
        assert 3 not in mainmod._in_progress_ids
        assert 2 in mainmod._in_progress_ids
    finally:
        mainmod._in_progress_ids.clear()
        mainmod._in_progress_ids.update(saved)


# ── bot.alert_error rate-limit ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_alert_error_rate_limits_same_component():
    """Second identical alert within cooldown is suppressed on Telegram."""
    from bot import TelegramBot

    # Build a minimal bot instance without going through __init__.
    bot = TelegramBot.__new__(TelegramBot)
    bot.auth_ids = [42]
    bot._send_with_retry = AsyncMock()
    bot.db = MagicMock()
    bot.db.log_failure = AsyncMock(return_value=1)
    bot._error_alert_cooldown = {}
    bot._ERROR_ALERT_COOLDOWN_SEC = 120.0

    # First call: should send.
    await bot.alert_error("spike_detection_job", ValueError("boom"))
    # Second call with same (component, error-class): should be throttled.
    await bot.alert_error("spike_detection_job", ValueError("boom again"))

    # _send_with_retry invoked once (first call) — second was throttled.
    assert bot._send_with_retry.await_count == 1, (
        f"expected 1 send, got {bot._send_with_retry.await_count}"
    )
    # But DB log was called both times (cheap + useful).
    assert bot.db.log_failure.await_count == 2


@pytest.mark.asyncio
async def test_alert_error_distinct_components_not_throttled():
    from bot import TelegramBot

    bot = TelegramBot.__new__(TelegramBot)
    bot.auth_ids = [42]
    bot._send_with_retry = AsyncMock()
    bot.db = MagicMock()
    bot.db.log_failure = AsyncMock(return_value=1)
    bot._error_alert_cooldown = {}
    bot._ERROR_ALERT_COOLDOWN_SEC = 120.0

    await bot.alert_error("job_a", RuntimeError("x"))
    await bot.alert_error("job_b", RuntimeError("x"))

    assert bot._send_with_retry.await_count == 2
