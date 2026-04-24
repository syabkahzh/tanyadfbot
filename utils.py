"""utils.py - General utility functions."""

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
