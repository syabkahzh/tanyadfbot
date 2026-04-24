"""Tests for shared.mark_message_ingested / seconds_since_last_ingest.

Motivation: client.is_connected() returns False transiently during MTProto
reconnects even while messages are actively flowing. That caused /diag to
report `Listener: DISCONNECTED` right after startup when Telethon had just
reconnected. The real signal is "have we seen a message recently".
"""
from __future__ import annotations

import asyncio
import pytest


@pytest.mark.asyncio
async def test_seconds_since_last_ingest_none_at_startup() -> None:
    import shared
    shared._last_message_ingest_ts = None  # reset
    assert shared.seconds_since_last_ingest() is None


@pytest.mark.asyncio
async def test_mark_message_ingested_sets_timestamp() -> None:
    import shared
    shared._last_message_ingest_ts = None
    shared.mark_message_ingested()
    age = shared.seconds_since_last_ingest()
    assert age is not None
    assert 0.0 <= age < 1.0, f"Expected fresh ingest, got {age:.2f}s"


@pytest.mark.asyncio
async def test_ingest_age_advances() -> None:
    import shared
    shared.mark_message_ingested()
    await asyncio.sleep(0.1)
    age = shared.seconds_since_last_ingest()
    assert age is not None and age >= 0.1


@pytest.mark.asyncio
async def test_fresh_ingest_overrides_is_connected_false() -> None:
    """Verdict logic: if we've seen a message <5min ago, listener is healthy
    REGARDLESS of is_connected(). This prevents /diag from reporting
    DISCONNECTED when Telethon is in a transient reconnect state."""
    import shared
    shared.mark_message_ingested()
    ingest_age = shared.seconds_since_last_ingest()
    mtproto_connected = False   # simulate transient reconnect

    # Mirror the verdict logic from bot.cmd_diag
    if ingest_age is not None and ingest_age < 300:
        listener_health = f"receiving (last msg {ingest_age:.0f}s ago)"
        listener_degraded = False
    elif mtproto_connected:
        listener_health = "connected, quiet"
        listener_degraded = False
    else:
        listener_health = "DISCONNECTED"
        listener_degraded = True

    assert "receiving" in listener_health
    assert not listener_degraded


@pytest.mark.asyncio
async def test_no_ingest_and_not_connected_is_disconnected() -> None:
    """If we've never seen a message AND socket is down, we ARE disconnected."""
    import shared
    shared._last_message_ingest_ts = None
    ingest_age = shared.seconds_since_last_ingest()
    mtproto_connected = False

    if ingest_age is not None and ingest_age < 300:
        listener_degraded = False
    elif mtproto_connected:
        listener_degraded = False
    else:
        listener_degraded = True

    assert listener_degraded
