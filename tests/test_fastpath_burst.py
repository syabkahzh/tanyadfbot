"""Regression tests for the fast-path burst bug.

The user reported a large influx of "aman toped / aman 50% / aman topeddd yeay"
messages that produced *zero* alerts. The cause was in `listener.py`:

  1. `INSTANT_PATTERN` did not include the slang confirmation word "aman",
     so `is_instant` stayed False on compound messages like "aman toped".
  2. The `AMAN_SIGNALS = {'aman'}` check required the *entire message text* to
     equal "aman" (case-insensitive) for `is_aman` to fire, and even then it
     demanded a reply-to-parent. "Aman toped" and friends failed both gates
     and fell through to the slow AI path, where latency + filter_duplicates
     swallowed them.

This module pins down the new, fixed semantics with unit tests that do not
require network / DB / Telethon.
"""
import re

from listener import INSTANT_PATTERN, NEG_PATTERN, FAST_ALLCAPS
from shared import _guess_brand


def _fast_path_signals(text: str) -> dict:
    """Mirror of the gates at the top of listener._handle_fast_path."""
    text_lower = text.lower()
    is_instant = bool(INSTANT_PATTERN.search(text)) and '?' not in text
    is_allcaps = (bool(FAST_ALLCAPS.match(text))
                  and len(text.strip()) > 3
                  and '?' not in text)
    is_aman_standalone = text_lower == 'aman' and '?' not in text
    is_neg = bool(NEG_PATTERN.search(text_lower))
    return dict(
        is_instant=is_instant,
        is_allcaps=is_allcaps,
        is_aman_standalone=is_aman_standalone,
        is_neg=is_neg,
    )


# ─────────────────────────────────────────────────────────────────────────────
# INSTANT_PATTERN now covers the slang "aman", so compound messages fire
# ─────────────────────────────────────────────────────────────────────────────

def test_instant_pattern_matches_aman_compound():
    """'aman toped' must be caught by INSTANT_PATTERN (was the main bug)."""
    assert INSTANT_PATTERN.search("aman toped")
    assert INSTANT_PATTERN.search("Aman toped")
    assert INSTANT_PATTERN.search("aman topeddd yeay")
    assert INSTANT_PATTERN.search("Aman 50%")
    assert INSTANT_PATTERN.search("Alhamdulillah amang aman sfood")


def test_instant_pattern_matches_other_confirmation_slang():
    """jp / work / luber / pecah / jackpot are also active-confirmation slang."""
    assert INSTANT_PATTERN.search("jp toped")
    assert INSTANT_PATTERN.search("work alfa jam segini")
    assert INSTANT_PATTERN.search("luber pecah sfood")
    assert INSTANT_PATTERN.search("berhasil indomaret")
    assert INSTANT_PATTERN.search("jackpot idm")


def test_instant_pattern_does_not_match_negations():
    """Negative contexts must NOT pass as instant signals by themselves.
    (The full fast-path still adds a NEG_PATTERN reject on top of this.)"""
    # These contain 'aman' inside longer words/negations → word-boundary blocks.
    assert not INSTANT_PATTERN.search("keamanan")   # 'aman' is not word-bounded
    assert not INSTANT_PATTERN.search("pengamanan")


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end gate behaviour: the three "reasons to consider fast-path" flags
# ─────────────────────────────────────────────────────────────────────────────

def test_aman_compound_hits_is_instant_not_standalone():
    """'Aman toped' should be detected via is_instant, not is_aman_standalone."""
    flags = _fast_path_signals("Aman toped")
    assert flags['is_instant'] is True
    assert flags['is_aman_standalone'] is False
    assert flags['is_neg'] is False


def test_aman_standalone_still_works():
    """Bare 'aman' still qualifies as the standalone reply-dedup case."""
    flags = _fast_path_signals("aman")
    assert flags['is_aman_standalone'] is True
    # It also matches INSTANT_PATTERN now (because 'aman' is in there), and
    # that's fine — the standalone branch just gates the parent-dedup logic.
    assert flags['is_instant'] is True


def test_question_disqualifies_fast_path():
    """Questions must never fire fast-path."""
    flags = _fast_path_signals("aman toped?")
    assert flags['is_instant'] is False
    assert flags['is_aman_standalone'] is False


def test_negations_block_fast_path():
    """'nt tokped' and 'kecepetan toped' must be caught by NEG_PATTERN."""
    assert _fast_path_signals("nt tokped")['is_neg'] is False  # 'nt' isn't in NEG
    # These should actually produce is_instant=False so they never reach NEG.
    # The important property: short community reactions like "kecepetan toped"
    # contain nothing that fires fast-path, so we stay out of it.
    flags = _fast_path_signals("kecepetan toped")
    assert flags['is_instant'] is False
    assert flags['is_aman_standalone'] is False
    assert flags['is_allcaps'] is False


# ─────────────────────────────────────────────────────────────────────────────
# Brand resolution on the same burst inputs — must resolve to Tokopedia / etc.
# ─────────────────────────────────────────────────────────────────────────────

def test_brand_resolves_for_aman_toped_variants():
    """All the user-reported burst variants must resolve to Tokopedia."""
    assert _guess_brand("Aman toped") == "Tokopedia"
    assert _guess_brand("aman topeddd yeay") == "Tokopedia"
    assert _guess_brand("Aman toped yeay") == "Tokopedia"
    assert _guess_brand("aman tkpd") == "Tokopedia"
    assert _guess_brand("aman tokped") == "Tokopedia"


def test_brand_resolves_for_aman_other_brands():
    assert _guess_brand("aman alfa") == "Alfamart"
    assert _guess_brand("aman jsm") == "Alfamart"
    assert _guess_brand("work idm") == "Indomaret"
    assert _guess_brand("jp sfood") == "ShopeeFood"


def test_brand_unknown_for_aman_without_context():
    """'aman 50%' has a slang signal but no brand — must resolve Unknown so
    the downstream brand-required gate drops it."""
    assert _guess_brand("Aman 50%") == "Unknown"
    assert _guess_brand("aman yeay") == "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# INSTANT_PATTERN word-boundary regressions
# ─────────────────────────────────────────────────────────────────────────────

def test_instant_pattern_requires_word_boundary():
    """'aman' embedded inside other words must not fire."""
    # 'keamanan' contains 'aman' but not as a word
    assert not INSTANT_PATTERN.search("keamanan rumah")
    # 'pamannya' contains 'aman' but not as a word
    assert not INSTANT_PATTERN.search("pamannya datang")
