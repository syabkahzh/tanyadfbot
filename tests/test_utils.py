from utils import _esc

def test_esc_none():
    assert _esc(None) == ""

def test_esc_empty():
    assert _esc("") == ""

def test_esc_normal_text():
    assert _esc("Hello World") == "Hello World"

def test_esc_all_special_chars():
    special_chars = "_*[]()~`>#+-=|{}.!\\"
    expected = "\\_\\*\\[\\]\\(\\)\\~\\`\\>\\#\\+\\-\\=\\|\\{\\}\\.\\!\\\\"
    assert _esc(special_chars) == expected

def test_esc_complex_string():
    text = "Check out this deal: [Brand] - 50% Off! (Conditions apply) #awesome *Hot*"
    expected = "Check out this deal: \\[Brand\\] \\- 50% Off\\! \\(Conditions apply\\) \\#awesome \\*Hot\\*"
    assert _esc(text) == expected

def test_esc_backslash_path():
    assert _esc("C:\\Users\\Admin") == "C:\\\\Users\\\\Admin"

def test_esc_backslash_literal():
    assert _esc("\\") == "\\\\"

def test_esc_double_backslash():
    assert _esc("\\\\") == "\\\\\\\\"
