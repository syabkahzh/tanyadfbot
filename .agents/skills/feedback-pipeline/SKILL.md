# TanyaDFBot Feedback & Correction Pipeline

This skill documents how to handle and leverage the "Fix Details" feedback provided by the bot owner.

## Feature Overview

The bot includes a **Feedback Pipeline** designed to collect "Ground Truth" data from the owner to improve AI detection accuracy.

1. **User Interface**: Every promo alert sent by the bot includes a **"🔧 Feedback"** inline button.
2. **Interaction**: Clicking the button triggers a stateful "Correction Mode" for that specific user.
3. **Data Collection**: The next message sent by the user is captured as a correction/feedback for the original promotion.
4. **Storage**: Feedback is saved in the `ai_corrections` table in the SQLite database.

## Database Schema (`ai_corrections`)

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | INTEGER | Primary Key |
| `original_msg_id` | INTEGER | The Telegram message ID of the original post |
| `brand` | TEXT | (Optional) The correct brand if known |
| `summary` | TEXT | (Optional) The correct summary if known |
| `correction` | TEXT | The raw text of the user's feedback |
| `created_at` | TEXT | Timestamp of the correction |

## How to use this for AI Learning

When tasking an agent to "improve the bot's accuracy" or "update the prompt", the agent should:

1. **Audit Corrections**: Run a query like `SELECT * FROM ai_corrections ORDER BY created_at DESC LIMIT 50;`.
2. **Compare with Reality**: Cross-reference the `original_msg_id` with the `messages` and `promos` tables to see what the bot originally extracted.
3. **Identify Patterns**: Look for common mistakes (e.g., "Always misidentifying ShopeePay as the merchant").
4. **Update System Prompt**: Modify `_EXTRACT_SYSTEM` in `processor.py` or `_BRAND_CANON` in `db.py` based on this feedback.
5. **Mark as Resolved**: (Future implementation) Add a way to track which corrections have already been "learned" by the model.

## Implementation Details

- **State Management**: Handled via `self._awaiting_feedback` dict in `TelegramBot` class (`bot.py`).
- **Callback Handler**: `handle_callback` identifies the `feed_` prefix.
- **Message Handler**: `handle_qa` checks for the user state before proceeding to normal AI Q&A.
