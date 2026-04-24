import os
from unittest.mock import patch
from config import Config, get_int

def test_get_int():
    with patch.dict(os.environ, {"TEST_KEY": "123"}):
        assert get_int("TEST_KEY") == 123
    
    with patch.dict(os.environ, {"TEST_KEY": "abc"}):
        assert get_int("TEST_KEY", default=456) == 456
        
    with patch.dict(os.environ, {"TEST_KEY": ""}):
        assert get_int("TEST_KEY", default=789) == 789

def test_config_validate():
    # Test missing values
    with patch.object(Config, 'API_ID', 0):
        assert Config.validate() is False
        
    # Test all present (mocking class attributes)
    with patch.object(Config, 'API_ID', 12345), \
         patch.object(Config, 'API_HASH', "hash"), \
         patch.object(Config, 'BOT_TOKEN', "token"), \
         patch.object(Config, 'GEMINI_API_KEY', "key"), \
         patch.object(Config, 'TARGET_GROUP', "group"):
        assert Config.validate() is True
