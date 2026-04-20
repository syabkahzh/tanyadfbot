import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InlineQueryResultArticle, InputMessageContent
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler, InlineQueryHandler
from telegram.constants import ParseMode
from telegram.request import HTTPXRequest
from config import Config
from datetime import datetime

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
        self.app.add_handler(CommandHandler("help", self.cmd_start))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("debug", self.cmd_debug))
        self.app.add_handler(CommandHandler("summary", self.cmd_summary))
        self.app.add_handler(CommandHandler("today", self.cmd_today))
        self.app.add_handler(CallbackQueryHandler(self.handle_callback))
        self.app.add_handler(InlineQueryHandler(self.handle_inline_query))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_qa))

    # ── Utils ───────────────────────────────────────────────────────────────

    def _fmt_raw_list(self, rows: list, title: str) -> str:
        """
        Format promo rows as a clean grouped list.
        Groups by brand, dedupes near-identical summaries by word overlap.
        """
        # Group by brand
        by_brand: dict[str, list] = {}
        for r in rows:
            brand = r['brand'] if r['brand'] and r['brand'].lower() != 'unknown' else '❓ Lainnya'
            by_brand.setdefault(brand, []).append(r)

        text = f"📋 *{title}* ({len(rows)} item)\n\n"

        for brand, items in sorted(by_brand.items(), key=lambda x: (x[0] == '❓ Lainnya', x[0])):
            text += f"🏪 *{brand}*\n"
            seen_keywords: list[set] = []
            for r in items:
                # Simple dupe filter: skip if >60% word overlap with an already-shown summary
                words = set(r['summary'].lower().split())
                is_dupe = any(
                    len(words & prev) / max(len(words | prev), 1) > 0.6
                    for prev in seen_keywords
                )
                if is_dupe:
                    continue
                seen_keywords.append(words)

                cond = f" _({r['conditions']})_" if r['conditions'] else ""
                link = f" [→]({r['tg_link']})" if r['tg_link'] else ""
                text += f"  • {r['summary']}{cond}{link}\n"
            text += "\n"

        return text

    async def _send_long(self, update_or_chat_id, text: str, edit_msg=None):
        """Send text, splitting at 4000 chars if needed."""
        chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
        for i, chunk in enumerate(chunks):
            try:
                if i == 0 and edit_msg:
                    await edit_msg.edit_text(chunk, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True)
                else:
                    target = update_or_chat_id if isinstance(update_or_chat_id, int) else update_or_chat_id.message.chat_id
                    await self.app.bot.send_message(target, chunk, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True)
            except Exception as e:
                print(f"Send error: {e}")

    # ── Commands ────────────────────────────────────────────────────────────

    @_owner_only
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "👋 *TanyaDFBot Active!*\n\n"
            "• `/summary` — promo 8 jam terakhir (list)\n"
            "• `/summary ai` — promo 8 jam, dirangkum AI\n"
            "• `/summary 4` — promo 4 jam terakhir (list)\n"
            "• `/summary 4 ai` — promo 4 jam, dirangkum AI\n"
            "• `/today` — promo 24 jam (list)\n"
            "• `/status` — cek kondisi bot\n"
            "• Atau tanya langsung aja!",
            parse_mode=ParseMode.MARKDOWN
        )

    @_owner_only
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        wait_msg = await update.message.reply_text("⏳ Bentar...", parse_mode=ParseMode.MARKDOWN)
        msg_count  = self.db.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        promo_count = self.db.conn.execute("SELECT COUNT(*) FROM promos").fetchone()[0]
        unprocessed = self.db.conn.execute("SELECT COUNT(*) FROM messages WHERE processed=0").fetchone()[0]
        recent_5min = self.db.get_message_count_in_window(5)
        text = (
            "📊 *Bot Status*\n\n"
            f"📩 Total Pesan: `{msg_count}`\n"
            f"🔥 Promo Terdeteksi: `{promo_count}`\n"
            f"🔄 Belum Diproses: `{unprocessed}`\n"
            f"⚡ Pesan 5 menit terakhir: `{recent_5min}`\n"
            f"🕒 Waktu: `{datetime.now().strftime('%H:%M:%S')}`"
        )
        await wait_msg.edit_text(text, parse_mode=ParseMode.MARKDOWN)

    @_owner_only
    async def cmd_debug(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        rows = self.db.conn.execute("SELECT id, text, processed FROM messages ORDER BY id DESC LIMIT 5").fetchall()
        text = "🔍 *Latest 5 Messages:*\n\n"
        for r in rows:
            proc = "✅" if r['processed'] else "⏳"
            snippet = r['text'][:80] + ('...' if len(r['text']) > 80 else '')
            text += f"{proc} *ID {r['id']}:* {snippet}\n\n"
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

    @_owner_only
    async def cmd_summary(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Usage:
          /summary          → list, 8 jam
          /summary 4        → list, 4 jam
          /summary ai       → AI narrative, 8 jam
          /summary 4 ai     → AI narrative, 4 jam
        """
        args = context.args or []
        hours = 8
        ai_mode = False

        for arg in args:
            if arg.lower() == 'ai':
                ai_mode = True
            else:
                try:
                    hours = int(arg)
                except ValueError:
                    await update.message.reply_text(
                        "❌ Format: `/summary [jam] [ai]`\nContoh: `/summary 4 ai`",
                        parse_mode=ParseMode.MARKDOWN
                    )
                    return

        rows = self.db.get_promos(hours=hours)

        if not rows:
            await update.message.reply_text(f"😔 Gak ada promo dalam {hours} jam terakhir.")
            return

        if ai_mode:
            wait_msg = await update.message.reply_text(
                f"🤖 Lagi merangkum {len(rows)} promo ({hours}j terakhir) dengan AI...",
                parse_mode=ParseMode.MARKDOWN
            )
            result = self.gemini.smart_summary(rows)
            if result:
                header = f"🧠 *AI Summary — {hours} Jam Terakhir*\n\n"
                await self._send_long(update, header + result, edit_msg=wait_msg)
            else:
                # fallback to raw list if AI fails
                await wait_msg.edit_text("⚠️ AI gagal, nampilin list biasa:")
                text = self._fmt_raw_list(rows, f"Promo {hours} Jam Terakhir")
                await self._send_long(update, text)
        else:
            text = self._fmt_raw_list(rows, f"Promo {hours} Jam Terakhir")
            await self._send_long(update, text)

    @_owner_only
    async def cmd_today(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        rows = self.db.get_promos(hours=24)
        if not rows:
            await update.message.reply_text("😔 Belum ada promo hari ini.")
            return
        text = self._fmt_raw_list(rows, "Promo 24 Jam Terakhir")
        await self._send_long(update, text)

    @_owner_only
    async def handle_qa(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        wait_msg = await update.message.reply_text("🤔 *Mikir dulu ya...*", parse_mode=ParseMode.MARKDOWN)
        query = update.message.text
        rows = self.db.get_promos(limit=50)
        context_text = "\n".join([f"- {r['brand']}: {r['summary']} Link: {r['tg_link']}" for r in rows])
        # AWAITING the async call
        answer = await self.gemini.answer_question(query, context_text)
        await wait_msg.edit_text(answer, parse_mode=ParseMode.MARKDOWN)


    @_owner_only
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()

    async def handle_inline_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.effective_user or update.effective_user.id != Config.OWNER_ID:
            return
        q = update.inline_query.query
        if not q:
            return
        rows = self.db.conn.execute(
            "SELECT brand, summary, tg_link, conditions FROM promos WHERE brand LIKE ? OR summary LIKE ? ORDER BY id DESC LIMIT 10",
            (f"%{q}%", f"%{q}%")
        ).fetchall()
        articles = [
            InlineQueryResultArticle(
                id=str(i),
                title=f"{r['brand']}: {r['summary'][:60]}",
                input_message_content=InputMessageContent(
                    f"🔥 *{r['brand']}*\n{r['summary']}\n🔗 [Link]({r['tg_link']})",
                    parse_mode=ParseMode.MARKDOWN
                )
            ) for i, r in enumerate(rows)
        ]
        await update.inline_query.answer(articles)

    # ── Outbound ────────────────────────────────────────────────────────────

    async def send_alert(self, p_data, tg_link: str):
        """Per-promo real-time alert."""
        buttons = [[InlineKeyboardButton("🛒 Open", url=tg_link)]] if tg_link else []
        text = (
            f"🔥 *Promo Baru — {p_data.brand}*\n"
            f"{p_data.summary}\n"
            f"📌 _{p_data.conditions}_"
        )
        await self.app.bot.send_message(
            chat_id=Config.OWNER_ID,
            text=text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup(buttons) if buttons else None,
        )

    async def send_digest(self, digest_text: str, hour_label: str):
        """Hourly cron digest."""
        if not digest_text:
            return
        full = f"📊 *Ringkasan Promo {hour_label}*\n\n{digest_text}"
        chunks = [full[i:i+4000] for i in range(0, len(full), 4000)]
        for chunk in chunks:
            await self.app.bot.send_message(Config.OWNER_ID, chunk, parse_mode=ParseMode.MARKDOWN)

    async def send_plain(self, text: str):
        """Generic owner notification (trend alerts, digest preamble, etc)."""
        await self.app.bot.send_message(Config.OWNER_ID, text, parse_mode=ParseMode.MARKDOWN)
