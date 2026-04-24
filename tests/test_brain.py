import pytest
from datetime import datetime, timezone
from shared import _parse_ts, TRANSIT_NOISE_PATTERN
from db import _ts_str
from listener import INSTANT_PATTERN

def test_timestamp_integrity():
    # Test datetime to string (canonical space format)
    dt = datetime(2026, 4, 24, 15, 30, 45, tzinfo=timezone.utc)
    s = _ts_str(dt)
    assert s == "2026-04-24 15:30:45+00:00"
    
    # Test string to datetime parsing
    # Should handle our space format
    parsed = _parse_ts("2026-04-24 15:30:45+00:00")
    assert parsed == dt
    
    # Should handle legacy T format (sync_history fix verification)
    parsed_t = _parse_ts("2026-04-24T15:30:45+00:00")
    assert parsed_t == dt

def test_instant_pattern_signals():
    # Valid triggers
    assert INSTANT_PATTERN.search("aman")
    assert INSTANT_PATTERN.search("toped jp")
    assert INSTANT_PATTERN.search("mcd on")
    assert INSTANT_PATTERN.search("grab work")
    
    # Slang triggers
    assert INSTANT_PATTERN.search("luber gais")
    assert INSTANT_PATTERN.search("pecah")
    
    # No-signal noise
    assert not INSTANT_PATTERN.search("halo apa kabar")
    assert not INSTANT_PATTERN.search("tanya dong")

def test_transit_noise_filtering():
    # These should be identified as transit noise (skip alert)
    assert TRANSIT_NOISE_PATTERN.search("aman kak rutenya")
    assert TRANSIT_NOISE_PATTERN.search("jalannya macet")
    assert TRANSIT_NOISE_PATTERN.search("paketnya nyampe")
    
    # Regular promos should NOT match transit noise
    assert not TRANSIT_NOISE_PATTERN.search("aman gfood")
    assert not TRANSIT_NOISE_PATTERN.search("shopee voucher work")

def test_instant_pattern_boundaries():
    # Should not match 'aman' inside other words
    assert not INSTANT_PATTERN.search("keamanan")
    assert not INSTANT_PATTERN.search("paman")
    
    # Should match standalone
    assert INSTANT_PATTERN.search("aman!")
    assert INSTANT_PATTERN.search("aman...")
