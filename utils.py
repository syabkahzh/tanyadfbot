"""utils.py - General utility functions."""
import logging
import asyncio

def _esc(text: str | None) -> str:
    """Escapes MarkdownV2 special characters.

    Required by Telegram MarkdownV2:
    _ * [ ] ( ) ~ ` > # + - = | { } . ! \
    """
    if not text:
        return ""
    # We must escape \ first to avoid double-escaping other backslashes.
    escaped = text.replace("\\", "\\\\")
    for char in "_*[]()~`>#+-=|{}.!":
        escaped = escaped.replace(char, f"\\{char}")
    return escaped

class AsyncDBHandler(logging.Handler):
    """Logging handler that saves records to the database asynchronously."""
    
    def __init__(self, db, loop=None):
        super().__init__()
        self.db = db
        self.loop = loop or asyncio.get_event_loop()

    def emit(self, record):
        # We only care about WARNING and above
        if record.levelno < logging.WARNING:
            return
            
        try:
            msg = self.format(record)
            tb = None
            if record.exc_info:
                import traceback
                tb = "".join(traceback.format_exception(*record.exc_info))
            
            # Use call_soon_threadsafe to schedule the async save from any thread
            if self.loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self.db.save_system_log(record.levelname, record.name, msg, tb),
                    self.loop
                )
        except Exception:
            self.handleError(record)
