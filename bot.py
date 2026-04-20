import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InlineQueryResultArticle, InputMessageContent
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler, InlineQueryHandler
from telegram.constants import ParseMode
from telegram.request import HTTPXRequest
from config import Config
from datetime import datetime, timezone, timedelta
import pytz

WIB = pytz.timezone(Config.TIMEZONE)

def _to_wib(ts) -> str:
    """Convert any timestamp (str, datetime, naive) to WIB HH:MM string."""
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
        return dt.astimezone(WIB).strftime('%H:%M')
    except Exception:
        return "??"

class TelegramBot:
    def __init__(self, db_manager, gemini_processor):
        self.db = db_manager
        self.gemini = gemini_processor

        request = HTTPXRequest(connect_timeout=30.0, read_timeout=60.0)
        self.app = ApplicationBuilder().token(Config.BOT_TOKEN).request(request).build()
        self._setup_handlers()

    def _owner_only(func):
        async def wrapper(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
            if not update.effective_user or update.effective_user.id != Config.OWNER_ID:
                return
            return await func(self, update, context)
        return wrapper

    def _setup_handlers(self):
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("help", self.cmd_help))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("debug", self.cmd_debug))
        self.app.add_handler(CommandHandler("summary", self.cmd_summary))
        self.app.add_handler(CommandHandler("today", self.cmd_today))
        self.app.add_handler(CommandHandler("aisummary", self.cmd_aisummary))
        self.app.add_handler(CommandHandler("hourly", self.cmd_hourly))
        self.app.add_handler(CallbackQueryHandler(self.handle_callback))
        self.app.add_handler(InlineQueryHandler(self.handle_inline_query))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_qa))

    def _fmt_raw_list(self, rows: list, title: str) -> str:
        by_brand = {}
        for r in rows:
            brand = r['brand'] if r['brand'] and r['brand'].lower() not in ('unknown', 'sunknown', '') else '❓ Lainnya'
            by_brand.setdefault(brand, []).append(r)

        text = f"📋 *{title}* ({len(rows)} promo)\n\n"
        for brand, items in sorted(by_brand.items(), key=lambda x: (x[0] == '❓ Lainnya', x[0])):
            text += f"🏪 *{brand}*\n"
            for r in items:
                # FIX: use msg_time (the actual promo time), not created_at
                ts_key = 'msg_time' if 'msg_time' in r.keys() else 'created_at'
                m_time = _to_wib(r[ts_key])
                status_icon = "🟢" if r.get('status') == 'active' else ("🔴" if r.get('status') == 'expired' else "⚪")
                text += f"  {status_icon} `[{m_time}]` {r['summary']}"
                cond = f" _({r['conditions']})_" if r.get('conditions') and r['conditions'].lower() not in ('none', '') else ""
                link = f" [→]({r['tg_link']})" if r.get('tg_link') else ""
                text += f"{cond}{link}\n"
            text += "\n"
        return text

    async def _send_long(self, update, text: str, edit_msg=None):
        chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
        for i, chunk in enumerate(chunks):
            try:
                if i == 0 and edit_msg:
                    await edit_msg.edit_text(chunk, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True)
                else:
                    await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True)
            except Exception:
                if i == 0 and edit_msg:
                    await edit_msg.edit_text(chunk, disable_web_page_preview=True)
                else:
                    await update.message.reply_text(chunk, disable_web_page_preview=True)

    @_owner_only
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        now_wib = datetime.now(WIB).strftime('%H:%M WIB')
        await update.message.reply_text(f"👋 *TanyaDFBot Active!*\n🕒 {now_wib}", parse_mode=ParseMode.MARKDOWN)

    @_owner_only
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        help_text = (
            "💰 *Cek Promo*\n"
            "• `/summary [jam]` — Promo X jam terakhir (default: 8)\n"
            "• `/today` — Promo hari ini (WIB)\n"
            "• `/aisummary [n]` — Rangkum N pesan terakhir\n"
            "• `/hourly` — Trigger digest manual\n"
            "• `/status` — Status sistem\n"
            "• `/debug` — 5 pesan terbaru di DB"
        )
        await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

    @_owner_only
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        status_msg = await update.message.reply_text("⏳ Checking...", parse_mode=ParseMode.MARKDOWN)
        async with self.db.conn.execute("SELECT COUNT(*) FROM messages") as cur:
            msg_count = (await cur.fetchone())[0]
        async with self.db.conn.execute("SELECT COUNT(*) FROM promos") as cur:
            promo_count = (await cur.fetchone())[0]
        async with self.db.conn.execute("SELECT COUNT(*) FROM messages WHERE processed = 0") as cur:
            unprocessed = (await cur.fetchone())[0]
        async with self.db.conn.execute("SELECT MAX(timestamp) FROM messages") as cur:
            latest_ts = (await cur.fetchone())[0]

        now_wib = datetime.now(WIB).strftime('%H:%M:%S WIB')
        latest_wib = _to_wib(latest_ts) + " WIB" if latest_ts else "N/A"
        text = (
            f"📊 *Status* — `{now_wib}`\n"
            f"📩 Total pesan: `{msg_count}`\n"
            f"🔥 Promos: `{promo_count}`\n"
            f"🔄 Queue: `{unprocessed}`\n"
            f"🕒 Pesan terbaru: `{latest_wib}`"
        )
        await status_msg.edit_text(text, parse_mode=ParseMode.MARKDOWN)

    @_owner_only
    async def cmd_hourly(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Manual trigger for hourly digest."""
        wait_msg = await update.message.reply_text("⏳ Generating digest...", parse_mode=ParseMode.MARKDOWN)
        rows = await self.db.get_promos(hours=1)
        if not rows:
            await wait_msg.edit_text("😔 Tidak ada promo 1 jam terakhir.")
            return
        context_text = "\n".join([f"- {r['brand']}: {r['summary']}" for r in rows])
        now_wib = datetime.now(WIB)
        digest = await self.gemini.answer_question(
            f"Summarize these deals concisely in Indonesian. Hour: {now_wib.strftime('%H:00 WIB')}",
            context_text
        )
        full = f"📊 *Ringkasan Promo {now_wib.strftime('%H:00 WIB')}*\n\n{digest}"
        await wait_msg.edit_text(full, parse_mode=ParseMode.MARKDOWN)

    @_owner_only
    async def cmd_debug(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        async with self.db.conn.execute(
            "SELECT id, text, processed, timestamp FROM messages ORDER BY id DESC LIMIT 5"
        ) as cur:
            rows = await cur.fetchall()
        text = "🔍 *Latest 5 messages:*\n\n"
        for r in rows:
            proc = "✅" if r['processed'] else "⏳"
            wib_time = _to_wib(r['timestamp'])
            text += f"{proc} *ID {r['id']}* `[{wib_time}]`\n{r['text'][:80]}\n\n"
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

    @_owner_only
    async def cmd_summary(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        hours = 8
        if context.args:
            try: hours = int(context.args[0])
            except: pass
        wait_msg = await update.message.reply_text(f"⏳ Checking {hours}h...", parse_mode=ParseMode.MARKDOWN)
        rows = await self.db.get_promos(hours=hours)
        if not rows:
            await wait_msg.edit_text(f"😔 Tidak ada promo dalam {hours} jam terakhir.")
            return
        text = self._fmt_raw_list(rows, f"Promo {hours} Jam Terakhir")
        await self._send_long(update, text, edit_msg=wait_msg)

    @_owner_only
    async def cmd_aisummary(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        n = 50
        if context.args:
            try: n = min(int(context.args[0]), 200)
            except: pass
        wait_msg = await update.message.reply_text(f"⏳ Summarizing {n} pesan...", parse_mode=ParseMode.MARKDOWN)
        rows = await self.db.get_last_n_messages(n)
        if not rows:
            await wait_msg.edit_text("😔 No messages found.")
            return
        texts = [f"{r['sender_name']}: {r['text']}" for r in reversed(rows)]
        summary = await self.gemini.summarize_raw(texts)
        header = f"🤖 *AI Summary ({n} Pesan Terakhir)*\n\n"
        await self._send_long(update, header + summary, edit_msg=wait_msg)

    @_owner_only
    async def cmd_today(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Shows promos from today in WIB (midnight WIB = 17:00 UTC prev day)."""
        wait_msg = await update.message.reply_text("⏳ Checking today (WIB)...", parse_mode=ParseMode.MARKDOWN)
        # Calculate hours since WIB midnight
        now_wib = datetime.now(WIB)
        hours_since_midnight = now_wib.hour + now_wib.minute / 60
        rows = await self.db.get_promos(hours=max(1, int(hours_since_midnight) + 1))
        if not rows:
            await wait_msg.edit_text("😔 Belum ada promo hari ini.")
            return
        today_str = now_wib.strftime('%d %b %Y')
        text = self._fmt_raw_list(rows, f"Promo Hari Ini — {today_str} WIB")
        await self._send_long(update, text, edit_msg=wait_msg)

    @_owner_only
    async def handle_qa(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        wait_msg = await update.message.reply_text("🤔 *Thinking...*", parse_mode=ParseMode.MARKDOWN)
        rows = await self.db.get_promos(hours=12, limit=50)
        context_text = "\n".join([f"- {r['brand']}: {r['summary']}" for r in rows])
        answer = await self.gemini.answer_question(update.message.text, context_text)
        await wait_msg.edit_text(answer, parse_mode=ParseMode.MARKDOWN)

    @_owner_only
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        if query.data.startswith("expire_"):
            internal_id = int(query.data.split("_")[1])
            await self.db.conn.execute(
                "UPDATE promos SET status = 'expired' WHERE source_msg_id = ?", (internal_id,)
            )
            await self.db.conn.commit()
            await query.edit_message_text(
                text=f"{query.message.text}\n\n✅ *Marked Expired*", parse_mode=ParseMode.MARKDOWN
            )

    async def handle_inline_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.effective_user or update.effective_user.id != Config.OWNER_ID: return
        query = update.inline_query.query
        if not query: return
        async with self.db.conn.execute(
            "SELECT brand, summary, tg_link FROM promos WHERE brand LIKE ? OR summary LIKE ? ORDER BY id DESC LIMIT 10",
            (f"%{query}%", f"%{query}%")
        ) as cur:
            results = await cur.fetchall()
        articles = [
            InlineQueryResultArticle(
                id=str(i),
                title=f"{r['brand']}: {r['summary'][:60]}",
                input_message_content=InputMessageContent(
                    f"🔥 *{r['brand']}*\n{r['summary']}\n🔗 [Link]({r['tg_link']})",
                    parse_mode=ParseMode.MARKDOWN
                )
            ) for i, r in enumerate(results)
        ]
        await update.inline_query.answer(articles)

    async def send_alert(self, p_data, tg_link, timestamp=None):
        keyboard = [[
            InlineKeyboardButton("🛒 Buka", url=tg_link),
            InlineKeyboardButton("❌ Expired", callback_data=f"expire_{p_data.original_msg_id}")
        ]]
        brand_label = p_data.brand if p_data.brand.lower() not in ('unknown', 'sunknown', '') else "❓ Unknown"

        msg_wib = _to_wib(timestamp) if timestamp else "??"
        now_wib = datetime.now(WIB).strftime('%H:%M:%S')
        status_icon = "🟢" if p_data.status == 'active' else ("🔴" if p_data.status == 'expired' else "⚪")

        alert_text = (
            f"🔥 *PROMO BARU* {status_icon}\n"
            f"🕒 Msg: `{msg_wib}` · Det: `{now_wib}` WIB\n\n"
            f"🏪 *{brand_label}*\n"
            f"📝 {p_data.summary}\n"
        )
        if p_data.conditions and p_data.conditions.lower() not in ('none', ''):
            alert_text += f"ℹ️ _{p_data.conditions}_\n"
        if p_data.valid_until and p_data.valid_until.lower() not in ('none', 'unknown', ''):
            alert_text += f"⏳ s/d {p_data.valid_until}\n"

        await self.app.bot.send_message(
            chat_id=Config.OWNER_ID,
            text=alert_text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def send_digest(self, digest_text, hour_label):
        if not digest_text: return
        full = f"📊 *Ringkasan Promo {hour_label}*\n\n{digest_text}"
        await self.app.bot.send_message(
            chat_id=Config.OWNER_ID, text=full, parse_mode=ParseMode.MARKDOWN
        )

    async def send_plain(self, text):
        await self.app.bot.send_message(
            chat_id=Config.OWNER_ID, text=text, parse_mode=ParseMode.MARKDOWN
        )
