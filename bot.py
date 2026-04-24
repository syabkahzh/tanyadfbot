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
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("diag", self.cmd_diag))
        self.app.add_handler(CommandHandler("today", self.cmd_today))
        self.app.add_handler(CommandHandler("clear", self.cmd_clear))
        self.app.add_handler(CommandHandler("debug", self.cmd_debug))
        self.app.add_handler(CallbackQueryHandler(self.handle_callback))
        
        # Priority handler for feedback flow
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_qa))
        
        # Global error handler
        self.app.add_error_handler(self.error_handler)

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handles the /start command."""
        await update.message.reply_text(
            "🚀 <b>TanyaDFBot Online</b>\n\n"
            "I'm scanning for promos and hot discussion trends in real-time.\n"
            "Use /status to see system health.",
            parse_mode=ParseMode.HTML
        )

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
                comp = html.escape(f['component'])
                err  = html.escape(f['error_msg'][:60])
                lines.append(f"• {comp}: {err}...")
            recent_log = "\n\n❌ <b>Recent Failures:</b>\n" + "\n".join(lines)

        # 7. Background Task Tracking
        flush_status = "Active 🔄" if shared._buffer_flush_task and not shared._buffer_flush_task.done() else "Idle 😴"
        bg_tasks = (
            f"🤖 AI Batches: <code>{shared._active_ai_tasks}</code>\n"
            f"⏳ Retrying Sends: <code>{shared._active_retry_sends}</code>\n"
            f"🔔 Alert Flush: <code>{html.escape(flush_status)}</code>"
        )

        WIB = pytz.timezone("Asia/Jakarta")
        now_wib = datetime.now(WIB).strftime('%H:%M:%S WIB')
        latest_wib = _to_wib(latest_ts) + " WIB" if latest_ts else "N/A"

        text = (
            f"📊 <b>Full Transparency Status</b>\n"
            f"🕒 <code>{now_wib}</code>\n\n"
            f"📩 Total Msgs: <code>{msg_count}</code>\n"
            f"🔥 Total Promos: <code>{promo_count}</code>\n"
            f"🔄 Queue: <code>{unprocessed}</code> {html.escape(triage_icon)}\n"
            f"🛡️ Traffic Cop: <code>{html.escape(ft_status)}</code>\n"
            f"🕒 Latest: <code>{latest_wib}</code>\n\n"
            f"⚙️ <b>Background Tasks:</b>\n{bg_tasks}\n\n"
            f"🤖 <b>AI Pressure:</b>\n{rpm_status}\n"
            f"📈 Total RPM: <code>{total_active}/{total_limit}</code>\n"
            f"{recent_log}\n\n"
            f"💻 <b>System:</b>\n"
            f"📁 DB Size: <code>{db_size_mb:.1f} MB</code>\n"
            f"🧠 RAM: <code>{ram_mb:.1f} MB</code>\n"
            f"⚡ CPU: <code>{cpu_usage:.1f}%</code>"
        )
        await status_msg.edit_text(text, parse_mode=ParseMode.HTML)

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
        except: oldest_age_sec = None

        async with self.db.conn.execute("SELECT COUNT(*) FROM messages WHERE processed=0") as cur:
            queue = (await cur.fetchone())[0]

        async with _main._in_progress_lock:
            stuck_claims = len(_main._in_progress_ids)

        # Listener signals
        try:
            tg_client = shared.listener.client
            mtproto_connected = bool(tg_client.is_connected())
        except: mtproto_connected = False
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
            f"🩺 <b>Pipeline Diagnostics</b>\n\n"
            f"⚙️ <b>Heartbeats:</b>\n"
            f"🔄 Loop: <code>{html.escape(loop_status)}</code>\n"
            f"🛰 Ingest: <code>{html.escape(listener_health)}</code>\n"
            f"🧬 Spawn: <code>{html.escape(spawn_status)}</code>\n\n"
            f"🛡 <b>Traffic Cop:</b>\n"
            f"🧠 FastText: <code>{html.escape(ft_active)}</code>\n\n"
            f"🤖 <b>AI Pressure:</b>\n"
            f"🔥 Concurrent: <code>{shared._active_ai_tasks}</code>\n"
            f"🚦 Headroom: <code>{headroom_pct:.0%}</code>\n\n"
            f"📥 <b>Queue Backlog:</b>\n"
            f"📦 Unprocessed: <code>{queue}</code>\n"
            f"🕒 Max Age: <code>{oldest_fmt}</code>\n"
            f"⚠️ Stuck Claims: <code>{stuck_claims}</code>\n\n"
            f"<b>Verdict:</b> {verdict}"
        )
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    @_owner_only
    async def cmd_today(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Summary of promos from today."""
        rows = await self.db.get_promos(hours=24)
        if not rows:
            await update.message.reply_text("Belum ada promo yang terdeteksi hari ini.")
            return
        
        WIB = pytz.timezone("Asia/Jakarta")
        today_str = datetime.now(WIB).strftime('%d %b')
        text = self._fmt_raw_list(rows, f"Promo Hari Ini — {today_str} WIB")
        await self._send_long(update, text)

    @_owner_only
    async def cmd_debug(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Dumps raw database rows for debugging."""
        rows = await self.db.get_raw_messages(limit=5)
        text = "🔍 <b>Raw Debug (Latest 5):</b>\n\n"
        
        lines = []
        for r in rows:
            clean_text = html.escape((r['text'] or '').replace('\n', ' ')[:80])
            status = "✅" if r['processed'] else "⏳"
            lines.append(f"{status} <code>[{r['id']}]</code> {clean_text}...")
        
        await update.message.reply_text(text + "\n".join(lines), parse_mode=ParseMode.HTML)

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Processes inline button clicks."""
        query = update.callback_query
        if not query: return
        await query.answer()
        
        data = query.data or ""
        
        if data.startswith("feed_"):
            orig_msg_id = int(data.split("_")[1])
            user_id = update.effective_user.id
            self._awaiting_feedback[user_id] = orig_msg_id
            
            await query.message.reply_text(
                "📝 <b>Correction Mode</b>\n\n"
                "What's wrong with this promo? Send me the correct details "
                "(e.g., 'Wrong brand, it should be McD' or 'Expired').\n\n"
                "I will learn from this!",
                parse_mode=ParseMode.HTML
            )
            return

        elif data.startswith("fix_"):
            fid = int(data.split("_")[1])
            await self.db.mark_failure_fixed(fid)
            
            # Update original message to show it's marked
            if query.message:
                reply_markup = query.message.reply_markup
                await query.edit_message_text(
                    text=f"{query.message.text_html}\n\n🛠 <b>Fix Marked!</b> Ready to retry.",
                    parse_mode=ParseMode.HTML,
                    reply_markup=reply_markup
                )

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
                await update.message.reply_text("✅ <b>Feedback Saved!</b>\n\nI will analyze this to improve my detection. Thank you, mawmaw!", parse_mode=ParseMode.HTML)
            except Exception as e:
                logger.error(f"Failed to save ai_correction: {e}")
                await update.message.reply_text(f"❌ Failed to save feedback: {e}")
            return

        wait_msg = await update.message.reply_text("🤔 <b>Thinking...</b>", parse_mode=ParseMode.HTML)
        rows = await self.db.get_promos(hours=4)
        context_text = "\n".join([f"- {r['brand']}: {r['summary']}" for r in rows])
        answer = await self.gemini.answer_question(update.message.text, context_text)
        await wait_msg.edit_text(answer, parse_mode=ParseMode.HTML)

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
        brand_tag = f"<b>{_esc(brand_label)}</b>"
        
        # Latency breakdown
        perf_tag = ""
        if p_data.queue_time is not None and p_data.ai_time is not None:
            perf_tag = f"\n⏱ <code>Q:{p_data.queue_time:.0f}s | AI:{p_data.ai_time:.1f}s</code>"

        text = (
            f"{status_emoji} {brand_tag}\n"
            f"📝 {_esc(p_data.summary)}\n"
            f"⏰ Jam: <code>{msg_wib}</code>"
            f"{perf_tag}"
        )

        if p_data.conditions and p_data.conditions.lower() != 'none':
            text += f"\nℹ️ <i>{_esc(p_data.conditions)}</i>"
        
        if corroborations > 0:
            text += f"\n\n👥 <b>+{corroborations} users</b> also mentioned this."
            try:
                snippets = json.loads(corroboration_texts)
                if snippets:
                    text += "\n" + "\n".join([f"  • <i>\"{_esc(s)}\"</i>" for s in snippets[:3]])
            except: pass

        if p_data.links:
            text += "\n\n🔗 " + " ".join([f"<a href='{l}'>Link</a>" for l in p_data.links])

        try:
            await self.app.bot.send_message(
                chat_id=Config.OWNER_ID,
                text=text,
                parse_mode=ParseMode.HTML,
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
        
        header = f"🚀 <b>{_esc(brand.upper())} BURST!</b> ({len(items)} alerts)\n"
        lines = []
        for p, link, ts, corr, ctexts, src in items:
            msg_wib = _to_wib(ts)
            lines.append(f"• {_esc(p.summary)} (<code>{msg_wib}</code>)")

        text = header + "\n".join(lines)
        try:
            await self.app.bot.send_message(
                chat_id=Config.OWNER_ID,
                text=text,
                parse_mode=ParseMode.HTML,
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        except Exception as e:
            logger.error(f"Failed to send grouped alert: {e}")

    async def send_plain(self, text: str, parse_mode: Any = ParseMode.HTML) -> None:
        """Sends a plain text message to the owner."""
        try:
            await self.app.bot.send_message(
                chat_id=Config.OWNER_ID,
                text=text,
                parse_mode=parse_mode,
                link_preview_options=LinkPreviewOptions(is_disabled=True)
            )
        except Exception as e:
            logger.error(f"Failed to send plain msg: {e}")

    async def alert_error(self, component: str, error: Exception) -> None:
        """Alerts the owner about a system error (rate-limited)."""
        err_msg = str(error)
        
        # Save to DB first
        import traceback
        tb = traceback.format_exc()
        fid = await self.db.log_failure(component, err_msg, tb)

        # Rate-limit notifications: 120s per component
        from shared import _last_error_alerts, _ERROR_ALERT_COOLDOWN
        now = time.monotonic()
        if component in _last_error_alerts and (now - _last_error_alerts[component]) < _ERROR_ALERT_COOLDOWN:
            return
        
        _last_error_alerts[component] = now
        
        keyboard = [[InlineKeyboardButton("🛠 Mark Fixed", callback_data=f"fix_{fid}")]]
        text = (
            f"🚨 <b>Error in {component}</b>\n\n"
            f"<code>{_esc(err_msg[:300])}</code>"
        )
        try:
            await self.app.bot.send_message(
                chat_id=Config.OWNER_ID,
                text=text,
                parse_mode=ParseMode.HTML,
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        except Exception as e:
            logger.error(f"Failed to send error alert: {e}")

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Global error handler for the Telegram application."""
        logger.error(f"Update {update} caused error: {context.error}")

    def _fmt_raw_list(self, rows: list, title: str) -> str:
        """Formatting helper for lists of promotions."""
        lines = [f"📅 <b>{title}</b>\n"]
        for r in rows:
            brand = r['brand'] if r['brand'] and r['brand'].lower() != 'unknown' else '❓'
            lines.append(f"• <b>{_esc(brand)}</b>: {_esc(r['summary'])}")
        return "\n".join(lines)

    async def _send_long(self, update: Update, text: str) -> None:
        """Sends potentially long messages by splitting into chunks."""
        if len(text) < 4000:
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)
            return
        
        for i in range(0, len(text), 4000):
            await update.message.reply_text(text[i:i+4000], parse_mode=ParseMode.HTML)


def _to_wib(ts_str: str) -> str:
    """Helper to convert ISO timestamp string to WIB HH:MM."""
    try:
        dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        WIB = pytz.timezone("Asia/Jakarta")
        return dt.astimezone(WIB).strftime('%H:%M')
    except:
        return "??"

import pytz
