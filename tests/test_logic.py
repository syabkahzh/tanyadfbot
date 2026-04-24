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
    assert normalize_brand("spx") == "SPX"
    assert normalize_brand("ismaya+") == "Ismaya"
    assert normalize_brand("solaria") == "Solaria"
    assert normalize_brand("neo") == "Bank Neo Commerce"
    assert normalize_brand("astrapay") == "AstraPay"
    assert normalize_brand("alfamart") == "Alfamart"
    assert normalize_brand("chatime") == "Chatime"
    assert normalize_brand("tokopedia") == "Tokopedia"
    assert normalize_brand("pubg") == "PUBG"
    assert normalize_brand("kawanlama") == "Kawan Lama"
    
    # Capitalization of non-canonical
    assert normalize_brand("brandbaru") == "Brandbaru"
    
    # Whitespace and Casing Edge Cases
    assert normalize_brand("  hOkBeN  ") == "HokBen"
    assert normalize_brand("sFoOd") == "ShopeeFood"
    assert normalize_brand(" gOpAy ") == "GoPay"
    
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
    
    # Merchant priority (slang)
    assert _guess_brand("aman jsm") == "Alfamart"
    assert _guess_brand("Jsm alfa jam segini masih") == "Alfamart"
    assert _guess_brand("jumat jsm luber") == "Alfamart"
    assert _guess_brand("psm aja") == "Alfamart"
    # Receipt header abbreviation
    assert _guess_brand("AFM RAYA TUBAN") == "Alfamart"
    assert _guess_brand("afm hero bogor") == "Alfamart"
    
    # Elongation handling
    assert _guess_brand("topedddd") == "Tokopedia"
    assert _guess_brand("alfaaaaa") == "Alfamart"
    
    # Word boundary checks for short words
    assert _guess_brand("ag") == "Alfagift"
    assert _guess_brand("mcd") == "McD"
    
    # Overlapping/Specific match
    assert _guess_brand("bayar pake spay") == "ShopeePay"
    
    # Custom boundary checks for '+' keywords
    assert _guess_brand("c+h+t+m") == "Chatime"
    assert _guess_brand("k+p+k+n") == "Kopi Kenangan"
    
    # No match
    assert _guess_brand("halo ges") == "Unknown"
    assert _guess_brand("tanya dong") == "Unknown"
    assert _guess_brand(None) == "Unknown"

def test_check_fast_path():
    from listener import check_fast_path
    
    # ALL-CAPS (len > 3)
    assert check_fast_path("HOKBEN") is True
    assert not check_fast_path("MCD")  # too short
    
    # Instant Triggers
    assert check_fast_path("jp gais") is True
    assert check_fast_path("luber") is True
    
    # Negation handling
    assert not check_fast_path("kapan on")
    assert not check_fast_path("kok gak work")
    
    # Transit noise gate
    assert not check_fast_path("aman rutenya")
    assert not check_fast_path("jalannya macet")
    
    # valid activation + brand
    assert check_fast_path("aman toped") is True

def test_is_worth_checking():
    gp = GeminiProcessor()
    
    # High signal
    assert gp._is_worth_checking("promo shopeefood 50rb") is True
    assert gp._is_worth_checking("hokben aman work") is True
    assert gp._is_worth_checking("sfood jp mantap") is True
    assert gp._is_worth_checking("membership alfagift murah") is True
    assert gp._is_worth_checking("info member shopee") is True
    assert gp._is_worth_checking("mamber indomaret") is True
    
    # Low signal / noise (multi-word support verified)
    assert gp._is_worth_checking("wkwk") is False
    assert gp._is_worth_checking("wkwk haha") is False
    assert gp._is_worth_checking("siap noted makasih") is False
    # Verified: expanded _SOCIAL_FILLER now catches this as noise
    assert gp._is_worth_checking("oke mantap bos") is False
    assert gp._is_worth_checking("apa kabar?") is False
    assert gp._is_worth_checking("masih ada?") is False
    assert gp._is_worth_checking("saya membisukan dia") is False
    
    # Meta / Social filler (already covered by multi-word tests above)
    assert gp._is_worth_checking("siap noted makasih") is False
    
    # Length based
    assert gp._is_worth_checking("ini ada promo menarik di gerai terdekat") is True
    assert gp._is_worth_checking("cek") is False


def test_fast_path_improved():
    from listener import check_fast_path

    # Noise phrases with 'aman'
    assert not check_fast_path("kak beli di toko itu aman ga ya")
    assert not check_fast_path("aman ga")
    assert not check_fast_path("aman ngga")

    # Real signal with 'aman'
    assert check_fast_path("aman spx")
    assert check_fast_path("spx aman")

    # Social filler
    assert not check_fast_path("📝 ya allah baru masuk ke rmh😭")
    assert not check_fast_path("makasih kak")

def test_is_worth_checking_improved():
    gp = GeminiProcessor()

    # Noise phrases with 'aman'
    assert not gp._is_worth_checking("kak beli di toko itu aman ga ya")
    assert not gp._is_worth_checking("aman ga")
    assert not gp._is_worth_checking("aman ngga")

    # Social filler
    assert not gp._is_worth_checking("📝 ya allah baru masuk ke rmh😭")
    assert not gp._is_worth_checking("makasih kak")
    assert not gp._is_worth_checking("nangis beneran")

    # Valid short phrase
    assert gp._is_worth_checking("aman spx")


def test_check_fast_path_comprehensive():
    from listener import check_fast_path

    # Empty and None cases
    assert not check_fast_path(None)
    assert not check_fast_path("")

    # Fast path ALL CAPS
    assert check_fast_path("PROMO") is True
    assert check_fast_path("GRATIS") is True
    assert not check_fast_path("PRO") # <= 3
    assert check_fast_path("PROMO123")

    # Instant Pattern Matches (Positive Cases)
    assert check_fast_path("gacor bosku") is True
    assert check_fast_path("restock kak") is True
    assert check_fast_path("sudah cair") is True
    assert check_fast_path("mantul gan") is True
    assert check_fast_path("ag") is True
    assert check_fast_path("voc") is True

    # Negation Pattern Matches (Negative Cases)
    assert not check_fast_path("kapan on")
    assert not check_fast_path("gaada promo")
    assert not check_fast_path("kok gak bisa")
    assert not check_fast_path("belum masuk")
    assert not check_fast_path("nunggu restock")
    assert not check_fast_path("coba dulu")
    assert not check_fast_path("error kak")
    assert not check_fast_path("sold out")

    # Transit noise logic
    assert not check_fast_path("aman di jalan")
    assert not check_fast_path("paketnya aman nyampe")

    # Real World Samples
    assert not check_fast_path("kak mau tanya dongg dlu udh pernah buka linknya di email tp blm dipake nah pas aku buka lagi kok udh ke reedem ya")
    assert not check_fast_path("Bentar buka appny dulu lemot")
    assert not check_fast_path("Kena colong kah?soale pas awal2 pubg ad case bgini dia kecolong")
    assert not check_fast_path("Ih kok ak blm buka plz😭")
