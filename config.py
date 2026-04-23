import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def get_int(key: str, default: int = 0) -> int:
    """Safely retrieves an integer from environment variables.

    Args:
        key: The environment variable key.
        default: The default value if not found or invalid.

    Returns:
        The integer value.
    """
    val = os.getenv(key)
    if not val or not val.strip():
        return default
    try:
        return int(val)
    except ValueError:
        return default

class Config:
    """Application configuration management using environment variables."""
    
    API_ID: int = get_int("TG_API_ID", 0)
    API_HASH: str = os.getenv("TG_API_HASH", "")
    BOT_TOKEN: str = os.getenv("BOT_TOKEN", "")
    OWNER_ID: int = get_int("MY_TELEGRAM_ID", 0)
    EXTRA_AUTH_ID: int = get_int("EXTRA_AUTH_ID", 0)
    TARGET_GROUP: str = os.getenv("TARGET_GROUP", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Paths using pathlib
    BASE_DIR: Path = Path(__file__).parent
    DB_PATH: str = "tanya_main.db"
    SESSION_NAME: str = "tg_session"
    
    # Models
    MODEL_ID: str = "gemma-4-31b-it"
    MODEL_FALLBACK: str = "gemma-4-26b-a4b-it"
    MODEL_LAST_RESORT: str = "gemini-1.5-flash-lite-preview"
    MODEL_LAST_RESORT_RPD: int = 450

    # Timezone: WIB = UTC+7
    TIMEZONE: str = "Asia/Jakarta"
    UTC_OFFSET_HOURS: int = 7

    @classmethod
    def validate(cls) -> bool:
        """Validates that all required configuration values are present.

        Returns:
            True if configuration is valid, False otherwise.
        """
        missing: list[str] = []
        if not cls.API_ID: missing.append("TG_API_ID")
        if not cls.API_HASH: missing.append("TG_API_HASH")
        if not cls.BOT_TOKEN: missing.append("BOT_TOKEN")
        if not cls.GEMINI_API_KEY: missing.append("GEMINI_API_KEY")
        if not cls.TARGET_GROUP: missing.append("TARGET_GROUP")

        if missing:
            print(f"CRITICAL ERROR: Missing values in .env for: {', '.join(missing)}")
            return False
        return True
