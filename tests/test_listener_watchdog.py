"""Tests for the listener health watchdog and /diag reconnect trigger."""
from __future__ import annotations

import asyncio
import types

import pytest


@pytest.mark.asyncio
async def test_watchdog_noop_when_ingest_is_recent(monkeypatch) -> None:
    import main
    import shared

    # Recent ingest → watchdog does nothing.
    shared.mark_message_ingested()
    main._last_listener_reconnect_ts = 0.0

    # Sentinel: if _reconnect_listener is called we'll flag it
    called = {"count": 0}

    async def _fake_reconn(gap_minutes):
        called["count"] += 1

    monkeypatch.setattr(shared, "_reconnect_listener", _fake_reconn)

    await main._listener_health_watchdog()
    assert called["count"] == 0


@pytest.mark.asyncio
async def test_watchdog_noop_when_socket_connected(monkeypatch) -> None:
    import main
    import shared

    # Simulate stale ingest but socket is up.
    shared._last_message_ingest_ts = None
    main._last_listener_reconnect_ts = 0.0

    class _FakeClient:
        def is_connected(self):
            return True

    fake_listener = types.SimpleNamespace(client=_FakeClient())
    monkeypatch.setattr(shared, "listener", fake_listener)

    called = {"count": 0}
    async def _fake_reconn(gap_minutes):
        called["count"] += 1
    monkeypatch.setattr(shared, "_reconnect_listener", _fake_reconn)

    await main._listener_health_watchdog()
    # socket up → no reconnect even without ingest
    assert called["count"] == 0


@pytest.mark.asyncio
async def test_watchdog_triggers_reconnect_when_truly_disconnected(monkeypatch) -> None:
    """No ingest + socket reports disconnected → reconnect must fire."""
    import main
    import shared

    shared._last_message_ingest_ts = None
    main._last_listener_reconnect_ts = 0.0
    shared._listener_reconnecting = False

    class _FakeClient:
        def is_connected(self):
            return False

    fake_listener = types.SimpleNamespace(client=_FakeClient())
    monkeypatch.setattr(shared, "listener", fake_listener)

    called = {"args": []}
    async def _fake_reconn(gap_minutes):
        called["args"].append(gap_minutes)
    monkeypatch.setattr(shared, "_reconnect_listener", _fake_reconn)

    await main._listener_health_watchdog()
    # Allow spawned task to run
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert len(called["args"]) == 1, f"Reconnect should have fired exactly once: {called}"


@pytest.mark.asyncio
async def test_watchdog_respects_cooldown(monkeypatch) -> None:
    """Reconnect attempts are rate-limited so we don't hammer MTProto."""
    import main
    import shared
    import time as _t

    shared._last_message_ingest_ts = None
    # Last reconnect was 10s ago → within 300s cooldown
    main._last_listener_reconnect_ts = _t.monotonic() - 10.0
    shared._listener_reconnecting = False

    class _FakeClient:
        def is_connected(self):
            return False

    fake_listener = types.SimpleNamespace(client=_FakeClient())
    monkeypatch.setattr(shared, "listener", fake_listener)

    called = {"count": 0}
    async def _fake_reconn(gap_minutes):
        called["count"] += 1
    monkeypatch.setattr(shared, "_reconnect_listener", _fake_reconn)

    await main._listener_health_watchdog()
    await asyncio.sleep(0)
    assert called["count"] == 0, "Cooldown should have suppressed reconnect"


@pytest.mark.asyncio
async def test_watchdog_respects_in_flight_reconnect(monkeypatch) -> None:
    """If a reconnect is already in progress, don't spawn another."""
    import main
    import shared

    shared._last_message_ingest_ts = None
    main._last_listener_reconnect_ts = 0.0
    shared._listener_reconnecting = True   # simulate in-flight

    class _FakeClient:
        def is_connected(self):
            return False

    fake_listener = types.SimpleNamespace(client=_FakeClient())
    monkeypatch.setattr(shared, "listener", fake_listener)

    called = {"count": 0}
    async def _fake_reconn(gap_minutes):
        called["count"] += 1
    monkeypatch.setattr(shared, "_reconnect_listener", _fake_reconn)

    await main._listener_health_watchdog()
    await asyncio.sleep(0)
    assert called["count"] == 0, "In-flight reconnect should suppress duplicate"

    # Cleanup for other tests
    shared._listener_reconnecting = False
