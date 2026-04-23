"""bot.py — Telegram Bot Layer.

Handles all user interactions, command dispatching, alert broadcasting, 
and reliable message delivery with retry logic.
"""

import asyncio
import html
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Sequence, Callable, TypeVar, cast

import pytz
from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup, 
    InlineQueryResultArticle, InputMessageContent
)
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, 
    filters, ContextTypes, CallbackQueryHandler, InlineQueryHandler
)
from telegram.constants import ParseMode
from telegram.request import HTTPXRequest

from config import Config
from processor import PromoExtraction
from utils import _esc
import shared

logger = logging.getLogger(__name__)
WIB: pytz.BaseTzInfo = pytz.timezone(Config.TIMEZONE)

T = TypeVar("T", bound=Callable[..., Any])

def _to_wib(ts: str | datetime | Any) -> str:
    """Converts any timestamp to a WIB HH:MM string.

    Args:
        ts: The raw timestamp as a string, datetime, or generic object.

    Returns:
        Formatted time string (HH:MM) in WIB, or '??' on failure.
    """
    try:
        if isinstance(ts, str):
            ts = ts.replace('Z', '+00:00')
            dt = datetime.fromisoformat(ts)
        elif isinstance(ts, datetime):
            dt = ts
        else:
            return "??"
        
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(WIB).strftime('%H:%M:%S')
    except Exception:
        return "??:??:??"

def _esc(text: str | None) -> str:
    """Escapes common Markdown characters to prevent formatting errors.

    Args:
        text: The raw string to escape.

    Returns:
        The escaped string.
    """
    if not text:
        return ""
    return (text.replace("*", "\\*").replace("_", "\\_")
                .replace("[", "\\[").replace("]", "\\]")
                .replace("`", "\\`"))

class TelegramBot:
    """Main Telegram Bot implementation."""

    def __init__(self, db_manager: Any, gemini_processor: Any) -> None:
        """Initializes the TelegramBot.

        Args:
            db_manager: Database management instance.
            gemini_processor: AI processing instance.
        """
        self.db: Any = db_manager
        self.gemini: Any = gemini_processor
        self.start_time: datetime = datetime.now(timezone.utc)
        
        # Authorized IDs
        self.auth_ids: set[int] = {
            uid for uid in (Config.OWNER_ID, Config.EXTRA_AUTH_ID) if uid
        }

        request = HTTPXRequest(connect_timeout=30.0, read_timeout=60.0)
        self.app = ApplicationBuilder().token(Config.BOT_TOKEN).request(request).build()
        self._setup_handlers()

    def _owner_only(func: T) -> T:
        """Decorator to restrict command access to authorized users only."""
        async def wrapper(self: "TelegramBot", update: Update, context: ContextTypes.DEFAULT_TYPE) -> Any:
            user = update.effective_user
            uid = user.id if user else 0
            username = user.username if user else "Unknown"
            msg_text = update.message.text if update.message else 'callback/inline'
            
            logger.info(f"Incoming command: {msg_text} from {username} ({uid})")
            
            if uid not in self.auth_ids:
                logger.warning(f"Unauthorized access attempt from UID: {uid}")
                return
            return await func(self, update, context)
        return cast(T, wrapper)

    def _setup_handlers(self) -> None:
        """Registers all command and interaction handlers with the application."""
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("ping", self.cmd_ping))
        self.app.add_handler(CommandHandler("help", self.cmd_help))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("logs", self.cmd_logs))
        self.app.add_handler(CommandHandler("debug", self.cmd_debug))
        self.app.add_handler(CommandHandler("summary", self.cmd_summary))
        self.app.add_handler(CommandHandler("today", self.cmd_today))
        self.app.add_handler(CommandHandler("aisummary", self.cmd_aisummary))
        self.app.add_handler(CommandHandler("hourly", self.cmd_hourly))
        self.app.add_handler(CommandHandler("restart", self.cmd_restart))
        self.app.add_handler(CallbackQueryHandler(self.handle_callback))
        self.app.add_handler(InlineQueryHandler(self.handle_inline_query))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_qa))

    def _fmt_raw_list(self, rows: Sequence[Any], title: str) -> str:
        """Formats a raw list of promo rows into a clean Markdown message.

        Args:
            rows: Sequence of database rows.
            title: Header title for the list.

        Returns:
            Formatted Markdown string.
        """
        by_brand: dict[str, list[Any]] = {}
        for r in rows:
            brand = r['brand'] if r['brand'] and r['brand'].lower() not in ('unknown', 'sunknown', '') else '❓ Lainnya'
            by_brand.setdefault(brand, []).append(r)

        text = f"📋 *{title}* ({len(rows)} promo)\n\n"
        for brand, items in sorted(by_brand.items(), key=lambda x: (x[0] == '❓ Lainnya', x[0])):
            text += f"🏪 *{brand}*\n"
            for r in items:
                ts_key = 'msg_time' if 'msg_time' in r.keys() else 'created_at'
                m_time = _to_wib(r[ts_key])
                status_icon = "🟢" if r['status'] == 'active' else ("🔴" if r['status'] == 'expired' else "⚪")
                text += f"  {status_icon} `[{m_time}]` {_esc(r['summary'])}"
                cond = f" _({_esc(r['conditions'])})_" if r['conditions'] and r['conditions'].lower() not in ('none', '') else ""
                link = f" [→]({r['tg_link']})" if r['tg_link'] else ""
                text += f"{cond}{link}\n"
            text += "\n"
        return text

    async def _send_long(self, update: Update, text: str, edit_msg: Any = None, 
                         parse_mode: str = ParseMode.MARKDOWN) -> None:
        """Sends a potentially very long message by splitting it into chunks.

        Args:
            update: The current update object.
            text: The long text to send.
            edit_msg: Optional message object to edit instead of sending new.
            parse_mode: Telegram parse mode (default Markdown).
        """
        chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
        for i, chunk in enumerate(chunks):
            try:
                if i == 0 and edit_msg:
                    await edit_msg.edit_text(chunk, parse_mode=parse_mode, disable_web_page_preview=True)
                elif update.message:
                    await update.message.reply_text(chunk, parse_mode=parse_mode, disable_web_page_preview=True)
            except Exception as e:
                logger.error(f"Failed to send long message chunk: {e}")
                # Fallback to plain text if formatting fails
                if i == 0 and edit_msg:
                    await edit_msg.edit_text(chunk, disable_web_page_preview=True)
                elif update.message:
                    await update.message.reply_text(chunk, disable_web_page_preview=True)

    @_owner_only
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Responds to the /start command."""
        now_wib = datetime.now(WIB).strftime('%H:%M WIB')
        if update.message:
            await update.message.reply_text(f"👋 *TanyaDFBot Active!*\n🕒 {now_wib}", parse_mode=ParseMode.MARKDOWN)

    @_owner_only
    async def cmd_restart(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Safe-restarts the bot by setting the stop event."""
        if update.message:
            await update.message.reply_text("🔄 *Restarting service...*", parse_mode=ParseMode.MARKDOWN)
        shared.get_stop_event().set()

    @_owner_only
    async def cmd_ping(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Standard health-check command."""
        uptime = (datetime.now(timezone.utc) - self.start_time)
        h, m = divmod(int(uptime.total_seconds()), 3600)
        if update.message:
            await update.message.reply_text(
                f"🟢 *Online*\n⏱ Uptime: `{h}h {m//60}m`\n📂 DB: `{Config.DB_PATH}`",
                parse_mode=ParseMode.MARKDOWN
            )

    @_owner_only
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Displays available commands."""
        help_text = (
            "💰 *Perintah Bot*\n"
            "• `/summary [jam]` — Promo X jam terakhir\n"
            "• `/today` — Promo hari ini (WIB)\n"
            "• `/aisummary [n]` — Rangkum N pesan mentah\n"
            "• `/status` — Queue & kesehatan sistem\n"
            "• `/logs` — Tampilkan log sistem\n"
            "• `/ping` — Uptime & koneksi\n"
            "• `/restart` — Restart bot (safe)\n\n"
            "📊 *Jadwal Otomatis*\n"
            "• `Hourly` — Tiap jam :00\n"
            "• `15m Digest` — Tiap menit :15 & :45\n"
            "• `Midnight` — Rekap 02:00–05:00 (dikirim jam 05:00)\n"
            "• `Trends` — Tiap 15 menit jika ramai"
        )
        if update.message:
            await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

    @_owner_only
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Provides full transparency into system health and background tasks."""
        if not update.message:
            return

        status_msg = await update.message.reply_text("⏳ Checking...", parse_mode=ParseMode.MARKDOWN)

        # 1. Database Stats
        async with self.db.conn.execute("SELECT COUNT(*) FROM messages") as cur:
            msg_count = (await cur.fetchone())[0]
        async with self.db.conn.execute("SELECT COUNT(*) FROM promos") as cur:
            promo_count = (await cur.fetchone())[0]
        async with self.db.conn.execute("SELECT COUNT(*) FROM messages WHERE processed = 0") as cur:
            unprocessed = (await cur.fetchone())[0]
        async with self.db.conn.execute("SELECT MAX(timestamp) FROM messages") as cur:
            latest_ts = (await cur.fetchone())[0]

        import os
        db_size_mb = await asyncio.to_thread(os.path.getsize, Config.DB_PATH) / (1024 * 1024)

        # 2. System Resource Usage
        import psutil
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / (1024 * 1024)
        cpu_usage = process.cpu_percent(interval=None)

        # 3. Model Usage
        from shared import gemini
        model_details = []
        total_active = 0
        total_limit  = 0
        for m_id, slot in gemini._slots.items():
            active = slot.current_usage()
            total_active += active
            total_limit  += slot.limit
            m_label = "31B" if "31b" in m_id.lower() else ("26B" if "26b" in m_id.lower() else "Lite")
            usage_str = f"`{m_label}: {active}/{slot.limit}`"
            if slot.daily_limit > 0:
                daily = slot.daily_usage()
                usage_str += f" `({daily}/{slot.daily_limit} d)`"
            model_details.append(usage_str)

        rpm_status = " | ".join(model_details)

        # 4. Recent Activity
        from shared import _recent_alerts_history, _recent_alerts_lock
        recent_log = ""
        async with _recent_alerts_lock:
            if _recent_alerts_history:
                seen = set()
                last_3 = []
                for item in reversed(_recent_alerts_history):
                    if item['brand'] not in seen:
                        seen.add(item['brand'])
                        last_3.append(item['brand'])
                    if len(last_3) >= 3: break
                recent_log = f"\n🔥 *Recent:* `{', '.join(last_3)}`"

        # 5. Triage Status
        import main
        triage_icon = "🚑" if getattr(main, '_queue_emergency_mode', False) else "✅"

        # 6. Background Task Tracking
        flush_status = "Active 🔄" if shared._buffer_flush_task and not shared._buffer_flush_task.done() else "Idle 😴"
        bg_tasks = (
            f"🤖 AI Batches: `{shared._active_ai_tasks}`\n"
            f"⏳ Retrying Sends: `{shared._active_retry_sends}`\n"
            f"🔔 Alert Flush: `{flush_status}`"
        )

        now_wib = datetime.now(WIB).strftime('%H:%M:%S WIB')
        latest_wib = _to_wib(latest_ts) + " WIB" if latest_ts else "N/A"

        text = (
            f"📊 *Full Transparency Status*\n"
            f"🕒 `{now_wib}`\n\n"
            f"📩 Total Msgs: `{msg_count}`\n"
            f"🔥 Total Promos: `{promo_count}`\n"
            f"🔄 Queue: `{unprocessed}` {triage_icon}\n"
            f"🕒 Latest: `{latest_wib}`\n\n"
            f"⚙️ *Background Tasks:*\n{bg_tasks}\n\n"
            f"🤖 *AI Pressure:*\n{rpm_status}\n"
            f"📈 Total RPM: `{total_active}/{total_limit}`\n"
            f"{recent_log}\n\n"
            f"💻 *System:*\n"
            f"📁 DB Size: `{db_size_mb:.1f} MB`\n"
            f"🧠 RAM: `{ram_mb:.1f} MB`\n"
            f"⚡ CPU: `{cpu_usage:.1f}%`"
        )
        await status_msg.edit_text(text, parse_mode=ParseMode.MARKDOWN)

    @_owner_only
    async def cmd_hourly(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Triggers a manual generation or retrieval of the hourly digest."""
        if not update.message: return
        
        # 1. Try cache first
        if shared._last_hourly_digest:
            await update.message.reply_text(
                shared._last_hourly_digest + "\n\n(<i>cached</i>)", 
                parse_mode=ParseMode.HTML
            )
            return

        # 2. Generate if cache empty
        wait_msg = await update.message.reply_text("⏳ Generating digest...")
        rows = await self.db.get_promos(hours=1)
        if not rows:
            await wait_msg.edit_text("😔 Tidak ada promo 1 jam terakhir.")
            return
        
        context_text = "\n".join([f"- {r['brand']}: {r['summary']}" for r in rows])
        now_wib = datetime.now(WIB)
        hour_label = now_wib.strftime('%H:00 WIB')
        
        digest = await self.gemini.answer_question(f"Summarize deals for {hour_label}", context_text)
        
        full_text = f"📊 <b>Digest {hour_label}</b>\n\n{digest}"
        shared._last_hourly_digest = full_text # Cache it
        
        await wait_msg.edit_text(full_text, parse_mode=ParseMode.HTML)

    @_owner_only
    async def cmd_debug(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Shows the last 5 messages for debugging purposes."""
        async with self.db.conn.execute(
            "SELECT id, text, processed, timestamp FROM messages ORDER BY id DESC LIMIT 5"
        ) as cur:
            rows = await cur.fetchall()
        text = "🔍 *Latest 5 messages:*\n\n"
        for r in rows:
            proc = "✅" if r['processed'] else "⏳"
            wib_time = _to_wib(r['timestamp'])
            text += f"{proc} *ID {r['id']}* `[{wib_time}]`\n{r['text'][:80]}\n\n"
        if update.message:
            await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

    @_owner_only
    async def cmd_summary(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Shows a list of promos from the last N hours."""
        if not update.message: return
        hours = 8
        if context.args:
            try: hours = int(context.args[0])
            except (ValueError, IndexError): pass
        wait_msg = await update.message.reply_text(f"⏳ Checking {hours}h...", parse_mode=ParseMode.MARKDOWN)
        rows = await self.db.get_promos(hours=hours)
        if not rows:
            await wait_msg.edit_text(f"😔 Tidak ada promo dalam {hours} jam terakhir.")
            return
        text = self._fmt_raw_list(rows, f"Promo {hours} Jam Terakhir")
        await self._send_long(update, text, edit_msg=wait_msg)

    @_owner_only
    async def cmd_aisummary(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Summarizes the last N raw chat messages using AI."""
        if not update.message: return
        n = 50
        if context.args:
            try: n = min(int(context.args[0]), 200)
            except (ValueError, IndexError): pass
        wait_msg = await update.message.reply_text(f"⏳ Summarizing {n} pesan...", parse_mode=ParseMode.MARKDOWN)
        rows = await self.db.get_last_n_messages(n)
        if not rows:
            await wait_msg.edit_text("😔 No messages found.")
            return
        texts = [f"{r['sender_name']}: {r['text']}" for r in reversed(rows)]
        summary = await self.gemini.summarize_raw(texts)
        header = f"🤖 <b>AI Summary ({n} Pesan Terakhir)</b>\n\n"
        await self._send_long(update, header + summary, edit_msg=wait_msg, parse_mode=ParseMode.HTML)

    @_owner_only
    async def cmd_today(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Shows promos from today WIB (strictly since 00:00:00 WIB)."""
        if not update.message: return
        wait_msg = await update.message.reply_text("⏳ Checking today (WIB)...", parse_mode=ParseMode.MARKDOWN)

        now_wib = datetime.now(WIB)
        today_midnight_wib = now_wib.replace(hour=0, minute=0, second=0, microsecond=0)

        rows = await self.db.get_promos(since_dt=today_midnight_wib)

        if not rows:
            await wait_msg.edit_text("😔 Belum ada promo hari ini.")
            return
        today_str = now_wib.strftime('%d %b %Y')
        text = self._fmt_raw_list(rows, f"Promo Hari Ini — {today_str} WIB")
        await self._send_long(update, text, edit_msg=wait_msg)

    @_owner_only
    async def handle_qa(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handles natural language questions about recent promos."""
        if not update.message or not update.message.text: return
        wait_msg = await update.message.reply_text("🤔 <b>Thinking...</b>", parse_mode=ParseMode.HTML)
        rows = await self.db.get_promos(hours=12, limit=50)
        context_text = "\n".join([f"- {r['brand']}: {r['summary']}" for r in rows])
        answer = await self.gemini.answer_question(update.message.text, context_text)
        await wait_msg.edit_text(answer, parse_mode=ParseMode.HTML)

    @_owner_only
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handles interactive button callbacks (e.g., marking expired, fixing errors)."""
        query = update.callback_query
        if not query: return
        await query.answer()
        
        data = query.data or ""
        
        if data.startswith("expire_"):
            internal_id = int(data.split("_")[1])
            await self.db.conn.execute(
                "UPDATE promos SET status = 'expired' WHERE source_msg_id = ?", (internal_id,)
            )
            await self.db.conn.commit()
            if query.message:
                await query.edit_message_text(
                    text=f"{query.message.text}\n\n✅ *Marked Expired*", parse_mode=ParseMode.MARKDOWN
                )

        elif data.startswith("fix_"):
            fid = int(data.split("_")[1])
            await self.db.mark_failure_fixed(fid)
            
            keyboard = [[InlineKeyboardButton("🔄 Retry Now", callback_data=f"retry_{fid}")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            if query.message:
                await query.edit_message_text(
                    text=f"{query.message.text_html}\n\n🛠 <b>Fix Marked!</b> Ready to retry.",
                    parse_mode=ParseMode.HTML,
                    reply_markup=reply_markup
                )

        elif data.startswith("retry_"):
            fid = int(data.split("_")[1])
            # Retrieve failure details
            async with self.db.conn.execute("SELECT * FROM failures WHERE id = ?", (fid,)) as cur:
                f = await cur.fetchone()
            
            if not f:
                await query.edit_message_text("❌ Failure record not found.")
                return

            await query.edit_message_text(
                text=f"{query.message.text_html}\n\n⏳ <b>Retrying {f['component']}...</b>",
                parse_mode=ParseMode.HTML
            )
            
            # Logic to retry based on component
            success = await self._perform_retry(f['component'], f)
            
            if success:
                await self.db.mark_failure_retried(fid)
                await query.edit_message_text(
                    text=f"{query.message.text_html}\n\n✅ <b>Retry Successful!</b>",
                    parse_mode=ParseMode.HTML
                )
            else:
                await query.edit_message_text(
                    text=f"{query.message.text_html}\n\n❌ <b>Retry Failed again.</b> Check logs.",
                    parse_mode=ParseMode.HTML,
                    reply_markup=query.message.reply_markup
                )

    async def _perform_retry(self, component: str, failure_row: Any) -> bool:
        """Internal logic to attempt retrying a failed component/job."""
        try:
            import jobs
            import sys
            import main
            # This is a bit tricky as we need the right arguments.
            # For now, we'll map common background jobs.
            if component == "hourly_digest_job":
                await jobs.hourly_digest_job(self.db, self.gemini, self, WIB)
                return True
            elif component == "midnight_digest_job":
                await jobs.midnight_digest_job(self.db, self.gemini, self)
                return True
            elif component == "halfhour_digest_job":
                await jobs.halfhour_digest_job(self.db, self.gemini, self, WIB)
                return True
            elif component == "trend_job":
                await jobs.trend_job(self.db, self.gemini, self, sys.modules['main'])
                return True
            elif component == "spike_detection_job":
                await jobs.spike_detection_job(self.db, self.gemini, self, sys.modules['main'])
                return True
            elif component == "image_processing_job":
                await jobs.image_processing_job(self.db, self.gemini, shared.listener, sys.modules['main'])
                return True
            # Add more as needed
            logger.warning(f"No explicit retry logic for {component}")
            return False
        except Exception as e:
            logger.error(f"Retry of {component} failed: {e}")
            return False


    @_owner_only
    async def handle_inline_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handles inline searches for recent promos.

        Args:
            update: Telegram update object.
            context: Context for the update.
        """
        if not update.inline_query:
            return
            
        query = update.inline_query.query
        if not query:
            return
        
        async with self.db.conn.execute(
            "SELECT brand, summary, tg_link, status FROM promos "
            "WHERE (brand LIKE ? OR summary LIKE ?) AND status='active' "
            "ORDER BY id DESC LIMIT 10",
            (f'%{query}%', f'%{query}%')
        ) as cur:
            rows = await cur.fetchall()

        results = []
        for i, r in enumerate(rows):
            results.append(InlineQueryResultArticle(
                id=str(i),
                title=f"{r['brand']}: {r['summary'][:50]}",
                input_message_content=InputMessageContent(
                    f"🏪 *{_esc(r['brand'])}*\n📝 {_esc(r['summary'])}\n🔗 [Lihat]({r['tg_link']})",
                    parse_mode=ParseMode.MARKDOWN
                )
            ))
        await update.inline_query.answer(results)

    async def _send_with_retry(
        self, 
        chat_id: int | str, 
        text: str, 
        parse_mode: str, 
        reply_markup: InlineKeyboardMarkup | None = None, 
        disable_preview: bool = True
    ) -> Any:
        """Sends a message with exponential backoff retries on transient failures.

        Args:
            chat_id: Target chat identifier.
            text: Message text.
            parse_mode: Telegram parse mode.
            reply_markup: Optional inline keyboard.
            disable_preview: Whether to disable link previews.

        Returns:
            The sent message object.

        Raises:
            Exception: If maximum retries are reached or a permanent error occurs.
        """
        shared._active_retry_sends += 1
        try:
            for attempt in range(10):
                try:
                    return await self.app.bot.send_message(
                        chat_id=chat_id,
                        text=text,
                        parse_mode=parse_mode,
                        reply_markup=reply_markup,
                        disable_web_page_preview=disable_preview
                    )
                except Exception as e:
                    err_str = str(e).lower()
                    if "message is too long" in err_str or "can't parse entities" in err_str:
                        logger.error(f"Permanent send error to {chat_id}: {e}")
                        raise e
                    
                    wait = min(1.5 ** attempt, 300)
                    logger.warning(f"Send failed (attempt {attempt+1}) to {chat_id}, retrying in {wait:.1f}s: {e}")
                    await asyncio.sleep(wait)
            raise Exception("Max retries reached for send_message")
        finally:
            shared._active_retry_sends -= 1

    async def send_alert(self, p_data: PromoExtraction, tg_link: str, 
                         timestamp: Any = None, corroborations: int = 0,
                         corroboration_texts: str = '[]',
                         source: str = 'ai') -> None:
        """Broadcasts a new promo alert to all authorized owners.

        Args:
            p_data: The extracted promo data.
            tg_link: Deep link to the source message.
            timestamp: Original message timestamp.
            corroborations: Number of users who confirmed the promo.
            corroboration_texts: JSON list of snippets from corroborating users.
            source: Source of the alert ('ai' or 'python').
        """
        keyboard = [[
            InlineKeyboardButton("🛒 Buka", url=tg_link),
            InlineKeyboardButton("❌ Expired", callback_data=f"expire_{p_data.original_msg_id}")
        ]]
        brand_label = p_data.brand if p_data.brand.lower() not in ('unknown', 'sunknown', '') else "❓ Unknown"

        msg_wib = _to_wib(timestamp) if timestamp else "??"
        now_dt = datetime.now(timezone.utc)
        now_wib = now_dt.astimezone(WIB).strftime('%H:%M:%S')
        status_icon = "🟢" if p_data.status == 'active' else ("🔴" if p_data.status == 'expired' else "⚪")

        latency_str = ""
        if timestamp:
            ts = shared._parse_ts(timestamp)
            latency = (now_dt - ts).total_seconds()
            
            detail = []
            if p_data.queue_time: detail.append(f"Q: {p_data.queue_time:.1f}s")
            if p_data.ai_time: detail.append(f"AI: {p_data.ai_time:.1f}s")
            
            detail_str = f" ({' · '.join(detail)})" if detail else ""
            latency_str = f"\n⚡ Latency: <code>{latency:.2f}s</code>{detail_str}"

        source_chip = "🤖 <code>processed by ai</code>" if source == 'ai' else "🐍 <code>triggered pythonly</code>"

        alert_text = (
            f"🔥 <b>PROMO BARU</b> {status_icon}\n"
            f"🕒 Msg: <code>{msg_wib}</code> · Det: <code>{now_wib}</code> WIB{latency_str}\n"
            f"🛠 Source: {source_chip}\n\n"
            f"🏪 <b>{html.escape(brand_label)}</b>\n"
            f"📝 {html.escape(p_data.summary)}\n"
        )
        if p_data.conditions and p_data.conditions.lower() not in ('none', ''):
            alert_text += f"ℹ️ <i>{html.escape(p_data.conditions)}</i>\n"
        if p_data.valid_until and p_data.valid_until.lower() not in ('none', 'unknown', ''):
            alert_text += f"⏳ s/d {html.escape(p_data.valid_until)}\n"
        
        if corroborations > 0:
            alert_text += f"\n✅ <b>Confirmed by {corroborations} users</b>\n"
            try:
                import json
                snippets = json.loads(corroboration_texts)
                for snip in snippets[:3]: # Show up to 3
                    alert_text += f"🔥 <i>\"{html.escape(snip)}\"</i>\n"
            except:
                pass

        async def _safe_send(uid):
            try:
                logger.info(f"📢 [Broadcast] Sending alert to {uid}")
                await self._send_with_retry(
                    chat_id=uid,
                    text=alert_text,
                    parse_mode=ParseMode.HTML,
                    reply_markup=InlineKeyboardMarkup(keyboard)
                )
            except Exception as e:
                logger.error(f"Failed to send alert to {uid} after retries: {e}")

        if self.auth_ids:
            await asyncio.gather(*[_safe_send(uid) for uid in self.auth_ids])

    async def send_grouped_alert(self, brand_key: str, items: list[tuple[PromoExtraction, str, Any, int, str, str]]) -> None:
        """Broadcasts a consolidated alert for multiple promos of the same brand.

        Args:
            brand_key: Normalized brand identifier.
            items: List of (PromoExtraction, tg_link, timestamp, corroborations, corroboration_texts, source).
        """
        first_p = items[0][0]
        brand_label = first_p.brand if first_p.brand.lower() not in ('unknown', 'sunknown', '') else brand_key.upper()
        
        seen = set()
        unique_items = []
        sources = set()
        for p, link, ts, corr, ctexts, src in items:
            key = p.summary[:40].lower()
            sources.add(src)
            if key not in seen:
                seen.add(key)
                unique_items.append((p, link, ts, corr, ctexts, src))
        
        if len(unique_items) == 1:
            p, link, ts, corr, ctexts, src = unique_items[0]
            await self.send_alert(p, link, timestamp=ts, corroborations=corr, corroboration_texts=ctexts, source=src)
            return

        now_wib = datetime.now(WIB).strftime('%H:%M:%S')
        first_ts = items[0][2]
        msg_wib = _to_wib(first_ts)
        
        # Calculate latency for the group (based on the first item)
        latency_str = ""
        if first_ts:
            ts = shared._parse_ts(first_ts)
            latency = (datetime.now(timezone.utc) - ts).total_seconds()
            latency_str = f"\n⚡ Latency: <code>{latency:.2f}s</code>"

        lines = []
        all_ext_links: list[str] = []
        total_corr = sum(it[3] for it in items)
        all_snippets = []
        for p, link, ts, corr, ctexts, src in unique_items:
            line = f"  • {html.escape(p.summary)}" + (f" <a href='{link}'>[→]</a>" if link else "")
            lines.append(line)
            if p.links:
                all_ext_links.extend(p.links)
            try:
                import json
                snips = json.loads(ctexts)
                for s in snips:
                    if s not in all_snippets:
                        all_snippets.append(s)
            except:
                pass

        lines_block = "\n".join(lines)
        
        source_chip = "🤖 <code>processed by ai</code>"
        if 'python' in sources:
            source_chip = "🐍 <code>triggered pythonly</code>"
        if len(sources) > 1:
            source_chip = "🖇️ <code>hybrid detection</code>"

        ext_links_block = ""
        if all_ext_links:
            unique_links = []
            seen_l = set()
            for l in all_ext_links:
                if l not in seen_l:
                    unique_links.append(l)
                    seen_l.add(l)
            ext_links_block = "\n" + "\n".join([f"🔗 {l}" for l in unique_links[:5]]) + "\n"

        text_parts = [
            f"🔥 <b>PROMO GRUP: {html.escape(brand_label)}</b>",
            f"🕒 Msg: <code>{msg_wib}</code> · Det: <code>{now_wib}</code> WIB{latency_str}",
            f"🛠 Source: {source_chip}\n",
            lines_block,
            ext_links_block
        ]

        if total_corr > 0:
            text_parts.append(f"\n✅ <b>Confirmed by {total_corr} users</b>")
            for snip in all_snippets[:3]:
                text_parts.append(f"🔥 <i>\"{html.escape(snip)}\"</i>")

        text = "\n".join(text_parts)

        async def _safe_send(uid):
            try:
                await self._send_with_retry(
                    chat_id=uid, text=text,
                    parse_mode=ParseMode.HTML
                )
            except Exception as e:
                logger.error(f"Failed to send grouped alert to {uid} after retries: {e}")

        if self.auth_ids:
            await asyncio.gather(*[_safe_send(uid) for uid in self.auth_ids])

    async def send_mega_alert(self, snapshot: dict[str, list[tuple[PromoExtraction, str, Any, int]]]) -> None:
        """Sends a consolidated alert for multiple brands at once.

        Args:
            snapshot: Dictionary of grouped promos.
        """
        count = sum(len(items) for items in snapshot.values())
        now_wib = datetime.now(WIB).strftime('%H:%M:%S')
        
        text = f"🚀 <b>DEALS BUNDLE</b> ({len(snapshot)} Brands · {count} Alerts)\n"
        text += f"🕒 <code>{now_wib}</code> WIB\n\n"
        
        for brand_key, items in sorted(snapshot.items()):
            first_p = items[0][0]
            brand_label = first_p.brand if first_p.brand.lower() not in ('unknown', 'sunknown', '') else brand_key.upper()
            
            text += f"🏪 <b>{html.escape(brand_label)}</b> ({len(items)})\n"
            seen_summaries = set()
            for p, link, ts, corr in items:
                s_key = p.summary[:40].lower()
                if s_key not in seen_summaries:
                    seen_summaries.add(s_key)
                    text += f"  • {html.escape(p.summary)} <a href='{link}'>[→]</a>\n"
            text += "\n"

        async def _safe_send(uid):
            try:
                await self._send_with_retry(chat_id=uid, text=text, parse_mode=ParseMode.HTML)
            except Exception as e:
                logger.error(f"Failed to send mega alert to {uid}: {e}")

        if self.auth_ids:
            await asyncio.gather(*[_safe_send(uid) for uid in self.auth_ids])

    async def send_digest(self, digest_text: str, hour_label: str) -> None:
        """Sends a formatted digest of recent promotions.

        Args:
            digest_text: The pre-formatted digest text.
            hour_label: Label for the time period.
        """
        if not digest_text: 
            return
            
        full = f"📊 <b>Ringkasan Promo {hour_label}</b>\n\n{digest_text}"
        
        async def _safe_send(uid):
            try:
                await self._send_with_retry(chat_id=uid, text=full, parse_mode=ParseMode.HTML)
            except Exception as e:
                logger.error(f"Failed to send digest to {uid}: {e}")

        if self.auth_ids:
            await asyncio.gather(*[_safe_send(uid) for uid in self.auth_ids])

    async def send_plain(self, text: str, parse_mode: str = ParseMode.MARKDOWN) -> None:
        """Sends a simple text message to all authorized owners.

        Args:
            text: The message text.
            parse_mode: Telegram parse mode.
        """
        async def _safe_send(uid):
            try:
                await self._send_with_retry(
                    chat_id=uid, text=text, parse_mode=parse_mode
                )
            except Exception as e:
                logger.error(f"Failed to send plain msg to {uid} after retries: {e}")

        if self.auth_ids:
            await asyncio.gather(*[_safe_send(uid) for uid in self.auth_ids])

    async def alert_error(self, component: str, error: Exception) -> None:
        """Notifies the owner of a critical system error and logs it for retry.

        Args:
            component: The system component where the error occurred.
            error: The exception instance.
        """
        now_wib = datetime.now(WIB).strftime('%H:%M:%S WIB')
        import traceback
        tb = traceback.format_exc()

        # Log to DB
        fid = await self.db.log_failure(component, str(error), tb)

        tb_short = tb[-1000:] if len(tb) > 1000 else tb

        text = (
            f"❌ <b>SYSTEM ERROR</b> (ID: {fid})\n"
            f"⏰ <code>{now_wib}</code>\n"
            f"🔧 Component: <code>{html.escape(component)}</code>\n\n"
            f"⚠️ <b>Error:</b>\n<code>{html.escape(str(error))}</code>\n\n"
            f"📜 <b>Traceback:</b>\n<code>{html.escape(tb_short)}</code>"
        )

        # Add button to mark as fixed
        reply_markup = None
        if fid > 0:
            keyboard = [[InlineKeyboardButton("🛠 Apply Fix", callback_data=f"fix_{fid}")]]
            reply_markup = InlineKeyboardMarkup(keyboard)

        for uid in self.auth_ids:
            try:
                await self._send_with_retry(
                    chat_id=uid, text=text, parse_mode=ParseMode.HTML, reply_markup=reply_markup
                )
            except Exception as e:
                logger.error(f"Failed to send error alert to {uid}: {e}")
    @_owner_only
    async def cmd_logs(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Retrieves and displays recent system logs.

        Args:
            update: Telegram update object.
            context: Context for the update.
        """
        import subprocess
        try:
            res = await asyncio.to_thread(
                subprocess.check_output,
                ["journalctl", "-u", "tanyadfbot", "-n", "20", "--no-pager"], 
                text=True
            )
            lines = res.split("\n")
            clean_log = "\n".join(lines[-20:])
            if update.message:
                await update.message.reply_text(f"📜 *Recent System Logs:*\n\n`{clean_log}`", parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            if update.message:
                await update.message.reply_text(f"❌ Failed to fetch logs: {e}")
