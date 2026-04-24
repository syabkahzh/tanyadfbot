"""bot.py — Telegram Interface and Broadcast Layer.

Handles all Telegram bot interactions including command processing, alert 
broadcasting with MarkdownV2 formatting, and administrative diagnostics.
"""

import asyncio
import html
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, TypeVar, cast

from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup, 
    LinkPreviewOptions, constants
)
from telegram.ext import (
    Application, ApplicationBuilder, CommandHandler, 
    MessageHandler, CallbackQueryHandler, ContextTypes, filters
)
from telegram.request import HTTPXRequest
from telegram.constants import ParseMode

from telegramify_markdown import convert, telegramify
from telegramify_markdown.content import ContentType

import shared
from db import Database, normalize_brand
from config import Config
from processor import GeminiProcessor, PromoExtraction
from utils import _esc

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=Callable[..., Any])


class TelegramBot:
    """Manages Telegram bot interactions and broadcasting."""

    def __init__(self, db_manager: Database, gemini_processor: GeminiProcessor) -> None:
        """Initializes the TelegramBot."""
        self.db = db_manager
        self.gemini = gemini_processor
        self.auth_ids = {
            uid for uid in (Config.OWNER_ID, Config.EXTRA_AUTH_ID) if uid
        }

        request = HTTPXRequest(connect_timeout=30.0, read_timeout=60.0)
        self.app = ApplicationBuilder().token(Config.BOT_TOKEN).request(request).build()
        
        # Track users in feedback flow: {user_id: original_msg_id}
        self._awaiting_feedback: dict[int, int] = {}
        
        self._setup_handlers()

    def _owner_only(func: T) -> T:
        """Decorator to restrict command access to authorized users only."""
        async def wrapper(self: "TelegramBot", update: Update, context: ContextTypes.DEFAULT_TYPE) -> Any:
            uid = update.effective_user.id if update.effective_user else None
            if uid not in self.auth_ids:
                logger.warning(f"Unauthorized access attempt from UID: {uid}")
                return
            return await func(self, update, context)
        return cast(T, wrapper)

    def _setup_handlers(self) -> None:
        """Configures Telegram command and message handlers."""
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("help", self.cmd_help))
        self.app.add_handler(CommandHandler("ping", self.cmd_ping))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("diag", self.cmd_diag))
        self.app.add_handler(CommandHandler("today", self.cmd_today))
        self.app.add_handler(CommandHandler("chart", self.cmd_chart))
        self.app.add_handler(CommandHandler("clear", self.cmd_clear))
        self.app.add_handler(CommandHandler("debug", self.cmd_debug))
        self.app.add_handler(CallbackQueryHandler(self.handle_callback))
        
        # Priority handler for feedback flow
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_qa))
        
        # Global error handler
        self.app.add_error_handler(self.error_handler)

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handles the /start command."""
        text = (
            "🚀 **TanyaDFBot Online**\n\n"
            "I'm scanning for promos and hot discussion trends in real-time.\n"
            "Use /status to see system health."
        )
        await self._send_markdown(update, text)

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Dynamically lists all available commands from registered handlers."""
        cmds = []
        for group in self.app.handlers.values():
            for handler in group:
                if isinstance(handler, CommandHandler):
                    for command in handler.commands:
                        cmds.append(f"/{command}")
        
        # Deduplicate and sort
        active_cmds = sorted(list(set(cmds)))
        
        text = (
            "📖 **TanyaDFBot Help**\n\n"
            "I'm scanning for promos and hot discussion trends in real-time.\n\n"
            "🛡 **Active Commands:**\n"
            + "\n".join([f"• {c}" for c in active_cmds]) +
            "\n\n_Note: Some commands are restricted to authorized users._"
        )
        await self._send_markdown(update, text)

    async def cmd_ping(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Simple health check to confirm the bot is responsive."""
        now = datetime.now(pytz.timezone("Asia/Jakarta")).strftime('%H:%M:%S')
        await self._send_markdown(update, f"🏓 **Pong!**\n🕒 Time: `{now} WIB`")

    async def _send_markdown(self, update: Update | object, text: str, reply_markup: Any = None) -> None:
        """Helper to send Markdown text using telegramify-markdown."""
        safe_text, entities = convert(text)
        try:
            if isinstance(update, Update) and update.message:
                await update.message.reply_text(
                    safe_text,
                    entities=[e.to_dict() for e in entities],
                    reply_markup=reply_markup,
                    link_preview_options=LinkPreviewOptions(is_disabled=True)
                )
            elif hasattr(update, 'message'): # For callback queries or other objects
                 await update.message.reply_text(
                    safe_text,
                    entities=[e.to_dict() for e in entities],
                    reply_markup=reply_markup,
                    link_preview_options=LinkPreviewOptions(is_disabled=True)
                )
        except Exception as e:
            logger.error(f"Failed to send markdown msg: {e}")

    @_owner_only
    async def cmd_clear(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Clears the message queue."""
        count = await self.db.clear_queue()
        await update.message.reply_text(f"🧹 Cleared `{count}` unprocessed messages.")

    @_owner_only
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Surfaces core system metrics."""
        status_msg = await update.message.reply_text("📊 *Calculating metrics...*", parse_mode=ParseMode.MARKDOWN)
        
        # 1. Database Stats
        msg_count   = await self.db.get_total_messages()
        promo_count = await self.db.get_total_promos()
        unprocessed = await self.db.get_queue_size()
        db_size_mb  = await self.db.get_db_size_mb()
        
        # 2. Latest Event
        latest_ts = await self.db.get_latest_message_ts()
        
        # 3. System Load
        import psutil
        import os
        process = psutil.Process(os.getpid())
        ram_mb  = process.memory_info().rss / (1024 * 1024)
        cpu_usage = psutil.cpu_percent()

        # 4. Triage Status
        from main import _queue_emergency_mode
        triage_icon = "⚠️ EMERGENCY" if _queue_emergency_mode else "🟢 Normal"
        
        # 4b. Traffic Cop Status
        ft_status = "🟢 Active" if shared._ft_model is not None else "⚪️ Disabled"

        # 5. AI RPM Status
        rpm_lines = []
        total_active = 0
        total_limit = 0
        for mid, slot in self.gemini._slots.items():
            active = slot.current_usage()
            total_active += active
            total_limit += slot.limit
            rpm_lines.append(f"  • `{mid}`: `{active}/{slot.limit}`")
        rpm_status = "\n".join(rpm_lines)

        # 6. Recent Failures
        recent_log = ""
        failures = await self.db.get_recent_failures(limit=3)
        if failures:
            lines = []
            for f in failures:
                comp = f['component']
                err  = f['error_msg'][:60]
                lines.append(f"• {comp}: {err}...")
            recent_log = "\n\n❌ **Recent Failures:**\n" + "\n".join(lines)

        # 7. Background Task Tracking
        flush_status = "Active 🔄" if shared._buffer_flush_task and not shared._buffer_flush_task.done() else "Idle 😴"
        bg_tasks = (
            f"🤖 AI Batches: `{shared._active_ai_tasks}`\n"
            f"⏳ Retrying Sends: `{shared._active_retry_sends}`\n"
            f"🔔 Alert Flush: `{flush_status}`"
        )

        WIB = pytz.timezone("Asia/Jakarta")
        now_wib = datetime.now(WIB).strftime('%H:%M:%S WIB')
        latest_wib = _to_wib(latest_ts) + " WIB" if latest_ts else "N/A"

        text = (
            f"📊 **Full Transparency Status**\n"
            f"🕒 `{now_wib}`\n\n"
            f"📩 Total Msgs: `{msg_count}`\n"
            f"🔥 Total Promos: `{promo_count}`\n"
            f"🔄 Queue: `{unprocessed}` {triage_icon}\n"
            f"🛡️ Traffic Cop: `{ft_status}`\n"
            f"🕒 Latest: `{latest_wib}`\n\n"
            f"⚙️ **Background Tasks:**\n{bg_tasks}\n\n"
            f"🤖 **AI Pressure:**\n{rpm_status}\n"
            f"📈 Total RPM: `{total_active}/{total_limit}`\n"
            f"{recent_log}\n\n"
            f"💻 **System:**\n"
            f"📁 DB Size: `{db_size_mb:.1f} MB`\n"
            f"🧠 RAM: `{ram_mb:.1f} MB`\n"
            f"⚡ CPU: `{cpu_usage:.1f}%`"
        )
        safe_text, entities = convert(text)
        await status_msg.edit_text(
            safe_text, 
            entities=[e.to_dict() for e in entities],
            link_preview_options=LinkPreviewOptions(is_disabled=True)
        )

    @_owner_only
    async def cmd_diag(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Deep-diagnosis of the processing pipeline."""
        if not update.message: return

        import time as _time
        import main as _main

        now_m = _time.monotonic()
        loop_tick = shared.get_loop_tick()
        loop_age = (now_m - loop_tick) if loop_tick else -1
        spawn_ts = getattr(shared, "_last_batch_spawn_ts", None)
        spawn_age = (now_m - spawn_ts) if spawn_ts else -1

        try: oldest_age_sec = await self.db.get_oldest_unprocessed_age_sec()
        except Exception: oldest_age_sec = None

        async with self.db.conn.execute("SELECT COUNT(*) FROM messages WHERE processed=0") as cur:
            queue = (await cur.fetchone())[0]

        async with _main._in_progress_lock:
            stuck_claims = len(_main._in_progress_ids)

        # Listener signals
        try:
            tg_client = shared.listener.client
            mtproto_connected = bool(tg_client.is_connected())
        except Exception: mtproto_connected = False
        ingest_age = shared.seconds_since_last_ingest()

        listener_health = "DISCONNECTED"
        if ingest_age is not None and ingest_age < 300:
            listener_health = f"🟢 receiving ({ingest_age:.0f}s)"
        elif mtproto_connected:
            listener_health = "🟡 connected (quiet)"

        # AI signals
        total_limit = sum(s.limit for s in self.gemini._slots.values())
        total_active = sum(s.current_usage() for s in self.gemini._slots.values())
        headroom_pct = 1.0 - (total_active / max(total_limit, 1))

        # Status strings
        loop_status = "🟢 ACTIVE" if loop_age >= 0 and loop_age < 45 else f"🔴 STALE ({loop_age:.0f}s)"
        spawn_status = "🟢 ACTIVE" if spawn_age >= 0 and spawn_age < 180 else "⚪️ IDLE" if queue == 0 else f"🟡 DELAYED ({spawn_age:.0f}s)"
        
        oldest_fmt = f"{int(oldest_age_sec)}s" if oldest_age_sec else "N/A"
        
        # 🛡️ Traffic Cop Status
        ft_active = "🟢 Active" if shared._ft_model is not None else "⚪️ Disabled"

        verdict = "✅ HEALTHY"
        if loop_age > 90 or (queue > 50 and spawn_age > 120):
            verdict = "🔴 CRITICAL"

        text = (
            f"🩺 **Pipeline Diagnostics**\n\n"
            f"⚙️ **Heartbeats:**\n"
            f"🔄 Loop: `{loop_status}`\n"
            f"🛰 Ingest: `{listener_health}`\n"
            f"🧬 Spawn: `{spawn_status}`\n\n"
            f"🛡 **Traffic Cop:**\n"
            f"🧠 FastText: `{ft_active}`\n\n"
            f"🤖 **AI Pressure:**\n"
            f"🔥 Concurrent: `{shared._active_ai_tasks}`\n"
            f"🚦 Headroom: `{headroom_pct:.0%}`\n\n"
            f"📥 **Queue Backlog:**\n"
            f"📦 Unprocessed: `{queue}`\n"
            f"🕒 Max Age: `{oldest_fmt}`\n"
            f"⚠️ Stuck Claims: `{stuck_claims}`\n\n"
            f"**Verdict:** {verdict}"
        )
        await self._send_markdown(update, text)

    @_owner_only
    async def cmd_today(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Summary of promos from the start of the current day (WIB) with pagination."""
        await self._render_today_page(update, page=1)

    @_owner_only
    async def cmd_chart(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Manually triggers a visual trend chart."""
        import jobs
        await update.message.reply_text("📊 *Generating chart...*", parse_mode=ParseMode.MARKDOWN)
        await jobs.visual_trend_job(self.db, self)

    async def _render_today_page(self, update: Update | object, page: int = 1) -> None:
        """Helper to render a specific page of today's promos."""
        WIB = pytz.timezone("Asia/Jakarta")
        now_wib = datetime.now(WIB)
        start_of_day_wib = now_wib.replace(hour=0, minute=0, second=0, microsecond=0)
        
        rows = await self.db.get_promos(since_dt=start_of_day_wib)
        
        if not rows:
            msg_text = "Belum ada promo yang terdeteksi hari ini."
            if isinstance(update, Update) and update.message:
                await self._send_markdown(update, msg_text)
            elif hasattr(update, 'edit_message_text'):
                safe_text, entities = convert(msg_text)
                await update.edit_message_text(safe_text, entities=[e.to_dict() for e in entities])
            return

        page_size = 15
        total_promos = len(rows)
        total_pages = (total_promos + page_size - 1) // page_size
        page = max(1, min(page, total_pages))
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_rows = rows[start_idx:end_idx]
        
        today_str = now_wib.strftime('%d %b')
        header = f"📅 **Promo Hari Ini — {today_str}**\n"
        header += f"🔥 Total: **{total_promos} promos** (Hal {page}/{total_pages})\n\n"
        
        lines = []
        for r in page_rows:
            brand = r['brand'] if r['brand'] and r['brand'].lower() not in ('unknown', 'sunknown', '') else '❓'
            time_str = _to_wib(r['msg_time'])
            link = r['tg_link'] or "#"
            lines.append(f"• `[{time_str}]` **{brand}**: {r['summary']} [➤]({link})")
            
        text = header + "\n".join(lines)
        
        # Build Navigation Buttons
        buttons = []
        if total_pages > 1:
            # First and Prev
            if page > 1:
                buttons.append(InlineKeyboardButton("⏮ First", callback_data="today_page:1"))
                buttons.append(InlineKeyboardButton("⬅️ Prev", callback_data=f"today_page:{page-1}"))
            
            # Next and Last
            if page < total_pages:
                buttons.append(InlineKeyboardButton("Next ➡️", callback_data=f"today_page:{page+1}"))
                buttons.append(InlineKeyboardButton("Last ⏭", callback_data=f"today_page:{total_pages}"))
            
        reply_markup = InlineKeyboardMarkup([buttons]) if buttons else None
        
        safe_text, entities = convert(text)
        try:
            if isinstance(update, Update) and update.message:
                await update.message.reply_text(
                    safe_text,
                    entities=[e.to_dict() for e in entities],
                    reply_markup=reply_markup,
                    link_preview_options=LinkPreviewOptions(is_disabled=True)
                )
            elif hasattr(update, 'edit_message_text'): # CallbackQuery
                await update.edit_message_text(
                    safe_text,
                    entities=[e.to_dict() for e in entities],
                    reply_markup=reply_markup,
                    link_preview_options=LinkPreviewOptions(is_disabled=True)
                )
        except Exception as e:
            logger.error(f"Failed to render today page {page}: {e}")

    @_owner_only
    async def cmd_debug(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Dumps raw database rows for debugging."""
        rows = await self.db.get_raw_messages(limit=5)
        text = "🔍 **Raw Debug (Latest 5):**\n\n"
        
        lines = []
        for r in rows:
            clean_text = (r['text'] or '').replace('\n', ' ')[:80]
            status = "✅" if r['processed'] else "⏳"
            lines.append(f"{status} `[{r['id']}]` {clean_text}...")
        
        await self._send_markdown(update, text + "\n".join(lines))

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Processes inline button clicks."""
        query = update.callback_query
        if not query: return
        await query.answer()
        
        data = query.data or ""
        
        if data.startswith("feed_"):
            orig_msg_id = int(data.split("_")[1])
            
            keyboard = [
                [
                    InlineKeyboardButton("🏷 Wrong Brand", callback_data=f"fopt_{orig_msg_id}_brand"),
                    InlineKeyboardButton("⏰ Expired", callback_data=f"fopt_{orig_msg_id}_expired")
                ],
                [
                    InlineKeyboardButton("👯 Duplicate", callback_data=f"fopt_{orig_msg_id}_dup"),
                    InlineKeyboardButton("⌨️ Custom Text", callback_data=f"fopt_{orig_msg_id}_custom")
                ]
            ]
            
            await self._send_markdown(
                query,
                "📝 **Feedback Mode**\n\n"
                "What's wrong with this promo? Choose a quick option or select Custom to type it out.",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            return

        elif data.startswith("fopt_"):
            parts = data.split("_")
            orig_msg_id = int(parts[1])
            option = parts[2]
            
            if option == "custom":
                user_id = update.effective_user.id
                self._awaiting_feedback[user_id] = orig_msg_id
                await query.message.reply_text(
                    "⌨️ **Please type your correction now:**",
                    parse_mode=ParseMode.MARKDOWN
                )
            else:
                mapping = {"brand": "Wrong brand", "expired": "Expired", "dup": "Duplicate"}
                correction = mapping.get(option, "Feedback")
                try:
                    await self.db.conn.execute(
                        "INSERT INTO ai_corrections (original_msg_id, correction) VALUES (?, ?)",
                        (orig_msg_id, correction)
                    )
                    await self.db.conn.commit()
                    await query.edit_message_text(
                        text=f"✅ **Feedback Saved: {correction}**\n\nThank you, mawmaw!",
                        parse_mode=ParseMode.MARKDOWN
                    )
                except Exception as e:
                    logger.error(f"Failed to save fopt: {e}")
            return

        elif data.startswith("poll_"):
            parts = data.split("_")
            orig_msg_id = int(parts[1])
            vote = parts[2]
            
            if vote == "custom":
                user_id = update.effective_user.id
                self._awaiting_feedback[user_id] = orig_msg_id
                await query.message.reply_text(
                    "⌨️ **Please type your correction for this poll now:**",
                    parse_mode=ParseMode.MARKDOWN
                )
            else:
                mapping = {"yes": "VERIFIED_PROMO", "no": "NOT_A_PROMO", "spam": "SPAM_OR_NOISE"}
                correction = mapping.get(vote, "VOTED")
                
                try:
                    if vote in ("no", "spam"):
                        await self.db.conn.execute(
                            "INSERT INTO ai_corrections (original_msg_id, correction) VALUES (?, ?)",
                            (orig_msg_id, correction)
                        )
                        await self.db.conn.commit()
                    
                    status_emoji = "✅" if vote == "yes" else ("❌" if vote == "no" else "🚫")
                    await query.edit_message_text(
                        text=f"{query.message.text}\n\n{status_emoji} **Verdict: {vote.upper()}**",
                        parse_mode=ParseMode.MARKDOWN
                    )
                except Exception as e:
                    logger.error(f"Failed poll vote: {e}")
            return

        elif data.startswith("fix_"):
            fid = int(data.split("_")[1])
            await self.db.mark_failure_fixed(fid)
            if query.message:
                reply_markup = query.message.reply_markup
                new_text = f"{query.message.text}\n\n🛠 **Fix Marked!** Ready to retry."
                safe_text, entities = convert(new_text)
                await query.edit_message_text(
                    text=safe_text,
                    entities=[e.to_dict() for e in entities],
                    reply_markup=reply_markup
                )
            return

        elif data.startswith("retry_"):
            fid = int(data.split("_")[1])
            await self.db.mark_failure_retried(fid)
            async with self.db.conn.execute("SELECT source_msg_id FROM failures WHERE id = ?", (fid,)) as cur:
                row = await cur.fetchone()
                msg_id = row[0] if row else None
            
            requeued = False
            if msg_id: requeued = await self.db.requeue_message(msg_id)
            
            if query.message:
                reply_markup = query.message.reply_markup
                status_text = "🔄 **Retry Marked!** Re-queueing..." if requeued else "🔄 **Retry Marked!**"
                new_text = f"{query.message.text}\n\n{status_text}"
                safe_text, entities = convert(new_text)
                await query.edit_message_text(
                    text=safe_text,
                    entities=[e.to_dict() for e in entities],
                    reply_markup=reply_markup
                )
            return

        elif data.startswith("today_page:"):
            page = int(data.split(":")[1])
            await self._render_today_page(query, page=page)
            return

    @_owner_only
    async def handle_qa(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handles natural language questions or feedback corrections."""
        if not update.message or not update.message.text: return
        user_id = update.effective_user.id
        
        if user_id in self._awaiting_feedback:
            orig_msg_id = self._awaiting_feedback.pop(user_id)
            correction = update.message.text
            try:
                await self.db.conn.execute(
                    "INSERT INTO ai_corrections (original_msg_id, correction) VALUES (?, ?)",
                    (orig_msg_id, correction)
                )
                await self.db.conn.commit()
                await self._send_markdown(update, "✅ **Feedback Saved!**\n\nI will analyze this to improve my detection. Thank you, mawmaw!")
            except Exception as e:
                logger.error(f"Failed to save ai_correction: {e}")
                await self._send_markdown(update, f"❌ Failed to save feedback: {e}")
            return

        wait_msg = await update.message.reply_text("🤔 **Thinking...**")
        rows = await self.db.get_promos(hours=4)
        context_text = "\n".join([f"- {r['brand']}: {r['summary']}" for r in rows])
        answer = await self.gemini.answer_question(update.message.text, context_text)
        
        safe_text, entities = convert(answer)
        await wait_msg.edit_text(safe_text, entities=[e.to_dict() for e in entities])

    async def send_alert(self, p_data: PromoExtraction, tg_link: str, 
                         timestamp: str | None = None, 
                         corroborations: int = 0,
                         corroboration_texts: str = "[]",
                         source: str = 'ai') -> None:
        """Broadcasts a single promotion alert to the owner."""
        keyboard = [[
            InlineKeyboardButton("🛒 Buka", url=tg_link),
            InlineKeyboardButton("🔧 Feedback", callback_data=f"feed_{p_data.original_msg_id}")
        ]]
        brand_label = p_data.brand if p_data.brand.lower() not in ('unknown', 'sunknown', '') else "❓ Unknown"

        msg_wib = _to_wib(timestamp) if timestamp else "??"
        
        # Visual cues for status
        status_emoji = "🔥" if p_data.status == 'active' else "💀"
        brand_tag = f"**{brand_label}**"
        
        # Source & Latency breakdown
        source_tag = "🤖 **AI Processed**" if source == 'ai' else "⚡ **Triggered Pythonly**"
        
        perf_tag = ""
        if source == 'ai':
            total_lat = (p_data.queue_time or 0) + (p_data.ai_time or 0)
            perf_tag = f"\n⏱ `Total: {total_lat:.1f}s | Q: {p_data.queue_time or 0:.0f}s | AI: {p_data.ai_time or 0:.1f}s`"
        else:
            # Fast-path: queue_time was set to total latency in listener.py
            perf_tag = f"\n⏱ `Latency: {p_data.queue_time or 0:.2f}s`"

        text = (
            f"{status_emoji} {brand_tag}\n"
            f"📝 {p_data.summary}\n"
            f"⏰ Jam: `{msg_wib}`\n"
            f"{source_tag}{perf_tag}"
        )

        if p_data.conditions and p_data.conditions.lower() != 'none':
            text += f"\nℹ️ _{p_data.conditions}_"
        
        if corroborations > 0:
            text += f"\n\n👥 **+{corroborations} users** also mentioned this."
            try:
                snippets = json.loads(corroboration_texts)
                if snippets:
                    text += "\n" + "\n".join([f"  • _\"{s}\"_" for s in snippets[:3]])
            except Exception: pass

        if p_data.links:
            text += "\n\n🔗 " + " ".join([f"[Link]({l})" for l in p_data.links])

        safe_text, entities = convert(text)
        try:
            await self.app.bot.send_message(
                chat_id=Config.OWNER_ID,
                text=safe_text,
                entities=[e.to_dict() for e in entities],
                reply_markup=InlineKeyboardMarkup(keyboard),
                link_preview_options=LinkPreviewOptions(is_disabled=True)
            )
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    async def send_grouped_alert(self, brand: str, items: list) -> None:
        """Broadcasts a grouped alert for multiple promos of the same brand."""
        if not items: return
        
        # Link to the latest message in the group
        latest_link = items[-1][1]
        keyboard = [[
            InlineKeyboardButton("🛒 Buka Terakhir", url=latest_link),
            InlineKeyboardButton("🔧 Feedback", callback_data=f"feed_{items[-1][0].original_msg_id}")
        ]]
        
        header = f"🚀 **{brand.upper()} BURST!** ({len(items)} alerts)\n"
        lines = []
        for p, link, ts, corr, ctexts, src in items:
            msg_wib = _to_wib(ts)
            source_icon = "🤖" if src == 'ai' else "⚡"
            if src == 'ai':
                total_lat = (p.queue_time or 0) + (p.ai_time or 0)
                lat_info = f"`{total_lat:.1f}s (Q{p.queue_time or 0:.0f}/A{p.ai_time or 0:.1f})`"
            else:
                lat_info = f"`{p.queue_time or 0:.2f}s`"
            
            lines.append(f"• {source_icon} {p.summary} (`{msg_wib}`) | {lat_info}")

        text = header + "\n" + "\n".join(lines)
        safe_text, entities = convert(text)
        try:
            await self.app.bot.send_message(
                chat_id=Config.OWNER_ID,
                text=safe_text,
                entities=[e.to_dict() for e in entities],
                reply_markup=InlineKeyboardMarkup(keyboard),
                link_preview_options=LinkPreviewOptions(is_disabled=True)
            )
        except Exception as e:
            logger.error(f"Failed to send grouped alert: {e}")

    async def send_plain(self, text: str, parse_mode: Any = None) -> None:
        """Sends a plain text message to the owner using telegramify for auto-chunking."""
        try:
            results = await telegramify(text)
            for item in results:
                if item.content_type == ContentType.TEXT:
                    await self.app.bot.send_message(
                        chat_id=Config.OWNER_ID,
                        text=item.text,
                        entities=[e.to_dict() for e in item.entities],
                        link_preview_options=LinkPreviewOptions(is_disabled=True)
                    )
        except Exception as e:
            logger.error(f"Failed to send plain msg: {e}")

    async def send_photo(self, photo: bytes, caption: str | None = None) -> None:
        """Sends a photo to the owner."""
        try:
            await self.app.bot.send_photo(
                chat_id=Config.OWNER_ID,
                photo=photo,
                caption=caption,
                parse_mode=ParseMode.HTML
            )
        except Exception as e:
            logger.error(f"Failed to send photo: {e}")

    async def send_verification_poll(self, p_data: PromoExtraction, tg_link: str) -> None:
        """Sends a verification poll for low-confidence AI extractions."""
        keyboard = [
            [
                InlineKeyboardButton("✅ Deal", callback_data=f"poll_{p_data.original_msg_id}_yes"),
                InlineKeyboardButton("❌ No Deal", callback_data=f"poll_{p_data.original_msg_id}_no")
            ],
            [
                InlineKeyboardButton("🚫 Spam", callback_data=f"poll_{p_data.original_msg_id}_spam"),
                InlineKeyboardButton("⌨️ Custom", callback_data=f"poll_{p_data.original_msg_id}_custom")
            ],
            [InlineKeyboardButton("🛒 Buka Pesan", url=tg_link)]
        ]
        
        text = (
            f"🗳 **Deal or No Deal?**\n"
            f"AI is unsure about this extraction (Conf: `{p_data.confidence:.2f}`)\n\n"
            f"🏪 **{p_data.brand}**\n"
            f"📝 {p_data.summary}\n"
            f"ℹ️ _{p_data.conditions or 'No conditions'}_"
        )
        
        safe_text, entities = convert(text)
        try:
            await self.app.bot.send_message(
                chat_id=Config.OWNER_ID,
                text=safe_text,
                entities=[e.to_dict() for e in entities],
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        except Exception as e:
            logger.error(f"Failed to send verification poll: {e}")

    async def alert_error(self, component: str, error: Exception, source_msg_id: int | None = None) -> None:
        """Alerts the owner about a system error (rate-limited)."""
        err_msg = str(error)
        
        # Save to DB first
        import traceback
        tb = traceback.format_exc()
        fid = await self.db.log_failure(component, err_msg, tb, source_msg_id=source_msg_id)

        # Rate-limit notifications: 120s per component
        import time; now = time.monotonic()
        if not hasattr(self, '_error_alert_cooldown'):
            self._error_alert_cooldown = {}
            self._ERROR_ALERT_COOLDOWN_SEC = 120.0

        if component in self._error_alert_cooldown and (now - self._error_alert_cooldown[component]) < self._ERROR_ALERT_COOLDOWN_SEC:
            return
        
        self._error_alert_cooldown[component] = now
        
        keyboard = [[
            InlineKeyboardButton("🛠 Mark Fixed", callback_data=f"fix_{fid}"),
            InlineKeyboardButton("🔄 Retry Msg", callback_data=f"retry_{fid}")
        ]]
        now_wib = datetime.now(pytz.timezone("Asia/Jakarta")).strftime('%H:%M:%S')
        text = (
            f"🚨 **Error in {component}**\n"
            f"⏰ Time: `{now_wib}`\n\n"
            f"`{err_msg[:300]}`"
        )
        safe_text, entities = convert(text)
        try:
            await self.app.bot.send_message(
                chat_id=Config.OWNER_ID,
                text=safe_text,
                entities=[e.to_dict() for e in entities],
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        except Exception as e:
            logger.error(f"Failed to send error alert: {e}")

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Global error handler for the Telegram application."""
        logger.error(f"Update {update} caused error: {context.error}")

    def _fmt_raw_list(self, rows: list, title: str) -> str:
        """Formatting helper for lists of promotions."""
        lines = [f"📅 **{title}**\n"]
        for r in rows:
            brand = r['brand'] if r['brand'] and r['brand'].lower() != 'unknown' else '❓'
            lines.append(f"• **{brand}**: {r['summary']}")
        return "\n".join(lines)

    async def _send_long(self, update: Update, text: str) -> None:
        """Sends potentially long messages by splitting into chunks safely using telegramify."""
        try:
            results = await telegramify(text)
            for item in results:
                if item.content_type == ContentType.TEXT:
                    await update.message.reply_text(
                        item.text,
                        entities=[e.to_dict() for e in item.entities],
                        link_preview_options=LinkPreviewOptions(is_disabled=True)
                    )
        except Exception as e:
            logger.error(f"Failed to send long markdown msg: {e}")


def _to_wib(ts_str: str) -> str:
    """Helper to convert ISO timestamp string to WIB HH:MM:SS."""
    if not ts_str:
        return "??:??:??"
    try:
        from shared import _parse_ts
        dt = _parse_ts(ts_str)
        WIB = pytz.timezone("Asia/Jakarta")
        return dt.astimezone(WIB).strftime('%H:%M:%S')
    except Exception:
        return "??:??:??"

import pytz
