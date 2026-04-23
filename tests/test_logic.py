import pytest
from db import normalize_brand
from shared import _guess_brand
from processor import GeminiProcessor

def test_normalize_brand():
    # Canonical matches
    assert normalize_brand("hokben") == "HokBen"
    assert normalize_brand("hoka ben") == "HokBen"
    assert normalize_brand("sfood") == "ShopeeFood"
    assert normalize_brand("gfood") == "GoFood"
    assert normalize_brand("kopken") == "Kopi Kenangan"
    assert normalize_brand("idm") == "Indomaret"
    assert normalize_brand("tpc") == "The Peoples Cafe"
    
    # Capitalization of non-canonical
    assert normalize_brand("brandbaru") == "Brandbaru"
    
    # Junk sentinels
    assert normalize_brand("Unknown") == "Unknown"
    assert normalize_brand("sunknown") == "Unknown"
    assert normalize_brand("brand") == "Unknown"
    assert normalize_brand(None) == "Unknown"
    assert normalize_brand("") == "Unknown"

def test_guess_brand():
    # Simple keyword match
    assert _guess_brand("ada promo hokben nih") == "HokBen"
    assert _guess_brand("vcr sfood ready") == "ShopeeFood"
    
    # Word boundary checks for short words
    assert _guess_brand("ag") == "Alfagift"
    assert _guess_brand("mcd") == "McD"
    
    # Custom boundary checks for '+' keywords
    assert _guess_brand("c+h+t+m") == "Chatime"
    assert _guess_brand("k+p+k+n") == "Kopi Kenangan"
    
    # No match
    assert _guess_brand("halo apa kabar") == "Unknown"
    assert _guess_brand(None) == "Unknown"

def test_is_worth_checking():
    gp = GeminiProcessor()
    
    # High signal
    assert gp._is_worth_checking("promo shopeefood 50rb") is True
    assert gp._is_worth_checking("hokben aman work") is True
    assert gp._is_worth_checking("sfood jp mantap") is True
    
    # Low signal / noise
    assert gp._is_worth_checking("wkwk") is False
    assert gp._is_worth_checking("apa kabar?") is False
    assert gp._is_worth_checking("masih ada?") is False
    assert gp._is_worth_checking("saya membisukan dia") is False
    
    # Meta / Social filler
    assert gp._is_worth_checking("siap noted makasih") is False
    
    # Length based
    assert gp._is_worth_checking("ini ada promo menarik di gerai terdekat") is True
    assert gp._is_worth_checking("cek") is False
