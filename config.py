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

    @classmethod
    def get_ai_army(cls) -> list[dict]:
        """Loads the multi-provider fleet configuration from models_config.json."""
        import json
        import os
        path = os.path.join(os.path.dirname(__file__), "models_config.json")
        try:
            with open(path, "r") as f:
                army = json.load(f)
            
            # Inject API keys from environment
            for p in army:
                env_key = p.get("api_key_env")
                p["api_key"] = os.getenv(env_key) if env_key else ""
            
            return army
        except Exception as e:
            import logging
            logging.error(f"Failed to load AI Army: {e}")
            return []

    @classmethod
    def get_ai_keys(cls) -> dict[str, str]:
        """Returns all configured AI keys for monitoring."""
        return {
            "google": cls.GEMINI_API_KEY,
            "groq": os.getenv("GROQ_API_KEY", ""),
            "glm": os.getenv("GLM_API_KEY", ""),
        }
