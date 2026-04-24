import pytest
from datetime import datetime, timezone

from shared import _parse_ts

def test_parse_ts_aware_datetime():
    dt = datetime(2023, 1, 1, 12, 0, tzinfo=timezone.utc)
    result = _parse_ts(dt)
    assert result == dt
    assert result.tzinfo == timezone.utc

def test_parse_ts_naive_datetime():
    dt = datetime(2023, 1, 1, 12, 0)
    result = _parse_ts(dt)
    assert result.tzinfo == timezone.utc
    assert result == dt.replace(tzinfo=timezone.utc)

def test_parse_ts_iso_with_z():
    ts_str = "2023-01-01T12:00:00Z"
    expected = datetime(2023, 1, 1, 12, 0, tzinfo=timezone.utc)
    result = _parse_ts(ts_str)
    assert result == expected
    assert result.tzinfo == timezone.utc

def test_parse_ts_iso_with_offset():
    ts_str = "2023-01-01T12:00:00+00:00"
    expected = datetime(2023, 1, 1, 12, 0, tzinfo=timezone.utc)
    result = _parse_ts(ts_str)
    assert result == expected
    assert result.tzinfo == timezone.utc

def test_parse_ts_naive_iso():
    ts_str = "2023-01-01T12:00:00"
    expected = datetime(2023, 1, 1, 12, 0, tzinfo=timezone.utc)
    result = _parse_ts(ts_str)
    assert result == expected
    assert result.tzinfo == timezone.utc

def test_parse_ts_invalid_string():
    ts_str = "invalid-date"
    expected = datetime.fromtimestamp(0, tz=timezone.utc)
    result = _parse_ts(ts_str)
    assert result == expected
    assert result.tzinfo == timezone.utc

def test_parse_ts_invalid_object():
    ts_obj = ["list", "of", "junk"]
    expected = datetime.fromtimestamp(0, tz=timezone.utc)
    result = _parse_ts(ts_obj)
    assert result == expected
    assert result.tzinfo == timezone.utc

def test_parse_ts_naive_iso_space_format():
    ts_str = "2023-01-01 12:00:00"
    expected = datetime(2023, 1, 1, 12, 0, tzinfo=timezone.utc)
    result = _parse_ts(ts_str)
    assert result == expected
    assert result.tzinfo == timezone.utc
