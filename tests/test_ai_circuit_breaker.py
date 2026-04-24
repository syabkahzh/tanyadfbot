"""Circuit breaker behavior when Gemini is in a failure streak."""
from __future__ import annotations




def _reset() -> None:
    import shared
    shared._ai_consecutive_failures = 0
    shared._ai_circuit_open_until = 0.0


def test_circuit_closed_initially() -> None:
    import shared
    _reset()
    assert shared.ai_circuit_open_remaining() == 0.0


def test_circuit_opens_after_threshold_consecutive_failures() -> None:
    import shared
    _reset()
    for _ in range(shared._AI_CIRCUIT_FAILURE_THRESHOLD - 1):
        shared.record_ai_outcome(success=False)
    # Still closed just under the threshold
    assert shared.ai_circuit_open_remaining() == 0.0

    shared.record_ai_outcome(success=False)
    # Now open for ~cooldown seconds
    remaining = shared.ai_circuit_open_remaining()
    assert 0 < remaining <= shared._AI_CIRCUIT_COOLDOWN_SEC + 0.5


def test_circuit_resets_on_single_success() -> None:
    import shared
    _reset()
    # Fail a bunch (but not to threshold)
    for _ in range(shared._AI_CIRCUIT_FAILURE_THRESHOLD - 1):
        shared.record_ai_outcome(success=False)
    assert shared._ai_consecutive_failures == shared._AI_CIRCUIT_FAILURE_THRESHOLD - 1

    shared.record_ai_outcome(success=True)
    assert shared._ai_consecutive_failures == 0
    assert shared.ai_circuit_open_remaining() == 0.0


def test_circuit_success_closes_open_breaker() -> None:
    """Even if the breaker is open, a single success closes it immediately."""
    import shared
    _reset()
    # Force open
    for _ in range(shared._AI_CIRCUIT_FAILURE_THRESHOLD):
        shared.record_ai_outcome(success=False)
    assert shared.ai_circuit_open_remaining() > 0

    shared.record_ai_outcome(success=True)
    assert shared.ai_circuit_open_remaining() == 0.0
    assert shared._ai_consecutive_failures == 0


def test_poison_message_threshold_is_2() -> None:
    """Poison messages must be marked processed after 2 failures (not 3) so
    queue can drain during AI provider outages."""
    import db as db_module
    import inspect
    src = inspect.getsource(db_module.Database.increment_ai_failure_count)
    assert "ai_failure_count >= 2" in src, (
        "increment_ai_failure_count should retire messages at >=2 failures"
    )
    assert "ai_failure_count >= 3" not in src, (
        "Old threshold of 3 is now 2 — update the SQL and this test"
    )


def test_reaper_threshold_is_130() -> None:
    """Reaper threshold must match the per-retry + acquire-overhead worst case."""
    import main
    assert main._IN_PROGRESS_MAX_AGE_SEC == 130.0
