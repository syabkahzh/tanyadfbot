import os
from dotenv import load_dotenv

load_dotenv()

def get_int(key, default=0):
    val = os.getenv(key)
    if not val or not val.strip():
        return default
    return int(val)

class Config:
    API_ID = get_int("TG_API_ID", 0)
    API_HASH = os.getenv("TG_API_HASH", "")
    BOT_TOKEN = os.getenv("BOT_TOKEN", "")
    OWNER_ID = get_int("MY_TELEGRAM_ID", 0)
    TARGET_GROUP = os.getenv("TARGET_GROUP", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

    DB_PATH = "tanya_main.db"
    SESSION_NAME = "tg_session"
    MODEL_ID = "gemma-4-31b-it"


    @classmethod
    def validate(cls):
        missing = []
        if not cls.API_ID: missing.append("TG_API_ID")
        if not cls.API_HASH: missing.append("TG_API_HASH")
        if not cls.BOT_TOKEN: missing.append("BOT_TOKEN")
        if not cls.GEMINI_API_KEY: missing.append("GEMINI_API_KEY")
        if not cls.TARGET_GROUP: missing.append("TARGET_GROUP")

        if missing:
            print(f"CRITICAL ERROR: Missing values in .env for: {', '.join(missing)}")
            return False
        return True
