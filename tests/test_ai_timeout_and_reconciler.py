"""Regression tests for the 30s AI timeout, reaper threshold, and counter self-heal."""
from __future__ import annotations

import asyncio

import pytest


def test_ai_timeout_is_30_seconds() -> None:
    """The per-attempt AI call timeout must be 30s so total (3×30=90s)
    stays bounded even when Gemini is flaky."""
    from processor import GeminiProcessor
    assert GeminiProcessor._AI_CALL_TIMEOUT_SEC == 30.0


def test_in_progress_reaper_threshold_matches_ai_timeout_total() -> None:
    """_IN_PROGRESS_MAX_AGE_SEC must exceed the MAX legitimate AI call time
    including inter-retry slot-acquire overhead (3 × 30s + 2 × 8s + 12s ~= 118s),
    but not much more, to recover fast from genuinely stuck claims."""
    import main
    from processor import GeminiProcessor
    # 3 attempts × timeout + 2 inter-retry acquires × 8s + pick_model overhead 12s
    max_legit_time = 3 * GeminiProcessor._AI_CALL_TIMEOUT_SEC + 2 * 8 + 12
    assert main._IN_PROGRESS_MAX_AGE_SEC >= max_legit_time, (
        f"reaper threshold {main._IN_PROGRESS_MAX_AGE_SEC}s must exceed "
        f"max legit AI time {max_legit_time}s to prevent double-claiming"
    )
    assert main._IN_PROGRESS_MAX_AGE_SEC <= 180, (
        "reaper threshold should be <=3min for prompt stuck-claim recovery"
    )


@pytest.mark.asyncio
async def test_reconciler_noop_when_counter_matches_live_tasks() -> None:
    import main
    import shared

    # Clean state
    main._active_spawn_tasks.clear()
    shared._active_ai_tasks = 0

    # Counter matches live tasks (both 0) → no-op
    await main._active_ai_tasks_reconciler()
    assert shared._active_ai_tasks == 0


@pytest.mark.asyncio
async def test_reconciler_fixes_counter_drift_high() -> None:
    """If counter leaked high above live task count, reconcile down."""
    import main
    import shared

    main._active_spawn_tasks.clear()
    # Simulate leak: counter says 3, no live tasks
    shared._active_ai_tasks = 3

    await main._active_ai_tasks_reconciler()
    assert shared._active_ai_tasks == 0, (
        f"Counter should drop to 0 (live task count), got {shared._active_ai_tasks}"
    )


@pytest.mark.asyncio
async def test_reconciler_preserves_counter_when_live_tasks_exist() -> None:
    """If live tasks match the counter, no reconciliation."""
    import main
    import shared

    main._active_spawn_tasks.clear()

    async def _noop():
        try:
            await asyncio.sleep(60)   # long-running
        except asyncio.CancelledError:
            pass

    t1 = asyncio.create_task(_noop())
    t2 = asyncio.create_task(_noop())
    main._active_spawn_tasks.add(t1)
    main._active_spawn_tasks.add(t2)
    shared._active_ai_tasks = 2
    try:
        await main._active_ai_tasks_reconciler()
        assert shared._active_ai_tasks == 2
    finally:
        t1.cancel()
        t2.cancel()
        await asyncio.gather(t1, t2, return_exceptions=True)
        main._active_spawn_tasks.clear()
        shared._active_ai_tasks = 0


@pytest.mark.asyncio
async def test_reconciler_ignores_done_tasks() -> None:
    """Tasks that have already completed should not count as live."""
    import main
    import shared

    main._active_spawn_tasks.clear()

    async def _done():
        return

    t1 = asyncio.create_task(_done())
    await t1   # definitely done
    main._active_spawn_tasks.add(t1)
    shared._active_ai_tasks = 1   # counter thinks 1 is running

    await main._active_ai_tasks_reconciler()
    # 0 live tasks, counter should drop to 0
    assert shared._active_ai_tasks == 0

    main._active_spawn_tasks.clear()
