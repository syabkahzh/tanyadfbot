"""processor.py — Gemini AI Layer.

Advanced model orchestration with dual-primary load balancing, automated fallback 
mechanisms, and rate-limit aware token buckets.
"""

import re
import asyncio
import time
import logging
from typing import List, Literal, Optional, Any, Sequence, cast
from pydantic import BaseModel

from google import genai
from config import Config
from db import normalize_brand

logger = logging.getLogger(__name__)

# ── Pre-compiled Patterns (Performance Optimization) ─────────────────────────
_WORD_BOUNDARY_KEYWORDS = re.compile(
    r'\b(off|on|aman|work|bs|jp|mm)\b', re.IGNORECASE
)
_SOCIAL_FILLER = re.compile(
    r'^(wkwk|haha|hehe|iya|noted|oke|ok|makasih|thanks|thx|mantap|gas|'
    r'siap|sip|lol|anjir|anjay|btw|oot|gws|semangat)[!.\s]*$',
    re.IGNORECASE
)
_NON_PROMO = re.compile(
    r'\b(setting|pengaturan|config|tutorial|cara|gimana|help|tolong|ini kak|'
    r'oot|random|foto|selfie|meme|lucu|haha|wkwk)\b', re.IGNORECASE
)
_PROMO = re.compile(
    r'\b(promo|diskon|cashback|voucher|gratis|murah|hemat|sale|off|deal|potongan|'
    r'sfood|gfood|grab|shopee|gojek|aman|on|jp|work|flash|limit|idm|alfa|indomaret|'
    r'nt|abis|habis|gabisa|gaada|gamau|minbel|r\+s\+t\+k|r\+s\+t\+c\+k|r\+st\+ck|'
    r'cb|kesbek|c\+s\+h\+b\+c\+k|cash back|kuota|slot|redeem|qr|scan|edc)\b', re.IGNORECASE
)
_JUNK_SUMMARY_PATTERN = re.compile(
    r'\b(tidak ada|none|n/a|tidak ditemukan|no promo)\b', re.IGNORECASE
)
_CURRENCY_DISCOUNT_PATTERN = re.compile(
    r'(rp\s?\d|rb\s?\d|\d+[kK]|disc|diskon|gratis|free|\d+\s*%|cashback)',
    re.IGNORECASE
)

# ── Response schemas ─────────────────────────────────────────────────────────

class PromoExtraction(BaseModel):
    """Structured promotion data extracted from chat text or images."""
    original_msg_id: int
    summary: str
    brand: str
    conditions: str
    valid_until: str
    status: Literal["active", "expired", "unknown"]
    links: List[str] = []
    detected_at: Optional[str] = None

class BatchResponse(BaseModel):
    """Wrapper for batch AI extraction results."""
    promos: List[PromoExtraction]

class TrendItem(BaseModel):
    """A single identified trend or topic from recent discussions."""
    topic: str
    msg_id: int

class TrendResponse(BaseModel):
    """Wrapper for aggregate trend analysis results."""
    trends: List[TrendItem]


# ─────────────────────────────────────────────────────────────────────────────
# Prompt constants
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACT_SYSTEM = """Kamu ekstrak promo dari percakapan grup deal hunter Indonesia (Discountfess).

ISTILAH KUNCI: mm=Mall Monday, bs=BerburuSales, nt=gagal/expired, jp=jackpot, sfood=ShopeeFood, gfood=GoFood

STATUS: active jika ada "aman/on/jp/work/restock/berhasil" | expired jika "abis/nt/sold out/ga bisa" | unknown jika ambigu

EKSTRAK jika: ada sinyal aktif/expired + info brand/platform. SKIP jika: pertanyaan murni, OOT, curhat tanpa info promo.
JANGAN EKSTRAK: form isi data pribadi (NIK/KTP/alamat), kuis berhadiah yang butuh upload foto/data, konten OOT, atau pesan yang tidak menyebut diskon/harga/voucher/cashback konkret.

Context ditulis sebagai C: sebelum MSG: — gunakan untuk resolve brand jika pesan utama cuma "aman" atau "on".
Summary: 1 kalimat informatif, sertakan harga/diskon jika ada.
Brand: Gunakan nama yang konsisten — "HopHop" bukan "Hophop". Jika ragu → "Unknown" (bukan "sunknown" atau variasi lain).

CONTOH YANG HARUS DI-SKIP:
- 'wkwkwk iya bener' → OOT
- 'hasilnya masih sama kak' → konteks tidak cukup, skip
- 'mau tanya dong, masih on gak?' → pertanyaan murni
- 'noted makasih' → bukan promo

CONTOH YANG HARUS DIEKSTRAK:
- 'sfood masih on 40k dapet 2' → active, ShopeeFood
- 'gobiz 30% off s/d jam 12 aman dicoba' → active, GoBiz
- 'NT gaes, udah abis' → expired, status flip"""

_DEDUP_SYSTEM = "Kamu agen deteksi duplikasi. Output HANYA angka indeks dipisah koma."

_DIGEST_SYSTEM = "Kamu asisten ringkasan promo Indonesia. Jawab singkat dan informatif dalam bahasa Indonesia santai."

_VISION_SYSTEM = """Kamu analis visual grup promo Indonesia (Discountfess).

TUGASMU: Tentukan apakah gambar ini berisi informasi promo yang bisa dimanfaatkan, lalu ekstrak detailnya.

PROMO VALID — ekstrak jika gambar berisi:
- Poster/banner promo brand (diskon, cashback, voucher, harga spesial)
- Screenshot aplikasi yang menampilkan harga/voucher/deal aktif
- Bukti transaksi dengan promo (struk, order confirmation dengan diskon)
- Screenshot chat/grup yang membahas promo konkret dengan angka/brand jelas

TOLAK (isi summary="SKIP", brand="SKIP") jika gambar adalah:
- Screenshot settings/UI aplikasi tanpa info promo
- Foto makanan/produk biasa tanpa harga promo
- Meme, stiker, foto personal
- Screenshot chat yang tidak membahas promo konkret
- Gambar blur/tidak jelas
- Konten OOT apapun

ATURAN OUTPUT:
- Jika SKIP: {"summary": "SKIP", "brand": "SKIP", "conditions": "", "valid_until": "", "status": "unknown", "original_msg_id": 0}
- Jika promo valid: summary 1 kalimat padat dengan brand + diskon/harga + syarat utama
- Brand: nama konsisten, "Unknown" jika tidak jelas tapi promo valid"""


# ─────────────────────────────────────────────────────────────────────────────
# Pre-filter keyword sets (no AI cost)
# ─────────────────────────────────────────────────────────────────────────────

_STRONG_KEYWORDS: set[str] = {
    'sfood','gfood','grab','shopee','gojek','tokped','tokopedia',
    'voucher','vcr','voc','diskon','promo','cashback','gratis','potongan',
    'idm','indomaret','alfa','alfamart','alfagift','hokben',
    'klaim','claim','restock','ristok','nt','abis','habis',
    'gabisa','gaada','g+b+s','gamau','minbel',
    'kuota','limit','slot','redeem','qr','scan','edc',
    'r+s+t+k','r+s+t+c+k','r+st+ck',
    'cb','kesbek','c+s+h+b+c+k','cash back',
    'luber','pecah','flash','sale','deal','murah','hemat','bonus',
    'ongkir','gratis ongkir',
}

_JUNK_SUMMARIES: set[str] = {'summary','none','n/a','-','tidak ada','tidak ditemukan'}


# ─────────────────────────────────────────────────────────────────────────────
# Token-bucket rate limiter (per model)
# ─────────────────────────────────────────────────────────────────────────────

class _ModelSlot:
    """Sliding-window token bucket (RPM) + Daily Quota (RPD) manager.

    Attributes:
        model_id: The identifier of the AI model.
        limit: Requests per minute (RPM) limit.
        daily_limit: Optional daily requests (RPD) limit.
    """

    def __init__(self, model_id: str, limit: int, daily_limit: int = 0) -> None:
        """Initializes the ModelSlot.

        Args:
            model_id: Model identifier.
            limit: RPM limit.
            daily_limit: RPD limit.
        """
        self.model_id: str = model_id
        self.limit: int    = limit
        self.daily_limit: int = daily_limit
        self._calls: list[float] = []
        self._daily_calls: list[float] = []  # Track 24h window
        self._lock: asyncio.Lock = asyncio.Lock()

    def available(self, now: float) -> int:
        """Calculates current available RPM capacity.

        Args:
            now: Current monotonic timestamp.

        Returns:
            Number of slots remaining in the current minute.
        """
        self._calls = [t for t in self._calls if now - t < 60]
        return self.limit - len(self._calls)

    async def acquire(self, timeout: float = 90.0) -> bool:
        """Waits for and claims a rate-limit slot.

        Args:
            timeout: Maximum seconds to wait before failing.

        Returns:
            True if slot was acquired, False if timeout or daily quota hit.
        """
        deadline = time.monotonic() + timeout
        while True:
            now = time.monotonic()
            async with self._lock:
                # 1. Cleanup old records
                self._calls = [t for t in self._calls if now - t < 60]
                if self.daily_limit > 0:
                    self._daily_calls = [t for t in self._daily_calls if now - t < 86400]
                
                # 2. Check Daily Limit
                if self.daily_limit > 0 and len(self._daily_calls) >= self.daily_limit:
                    return False

                # 3. Check RPM Limit
                if len(self._calls) < self.limit:
                    self._calls.append(now)
                    if self.daily_limit > 0:
                        self._daily_calls.append(now)
                    return True
            
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(2)

    def release_last(self) -> None:
        """Removes the most recent call record from the trackers."""
        async def _sync_release() -> None:
            async with self._lock:
                if self._calls:
                    self._calls.pop()
                if self.daily_limit > 0 and self._daily_calls:
                    self._daily_calls.pop()
        asyncio.create_task(_sync_release())

    def current_usage(self) -> int:
        """Returns the number of slots used in the last 60 seconds.

        Returns:
            Used RPM count.
        """
        now = time.monotonic()
        return len([t for t in self._calls if now - t < 60])
    
    def daily_usage(self) -> int:
        """Returns the total number of calls in the last 24 hours.

        Returns:
            Used RPD count.
        """
        now = time.monotonic()
        return len([t for t in self._daily_calls if now - t < 86400])


# ─────────────────────────────────────────────────────────────────────────────

class GeminiProcessor:
    """Orchestrates AI analysis using various Gemini models with load balancing."""

    def __init__(self) -> None:
        """Initializes the GeminiProcessor."""
        self.client = genai.Client(api_key=Config.GEMINI_API_KEY)
        self._dedup_lock = asyncio.Lock()

        # Per-model slots — 11-12 RPM each
        self._slots: dict[str, _ModelSlot] = {
            Config.MODEL_ID:           _ModelSlot(Config.MODEL_ID,           11),
            Config.MODEL_FALLBACK:     _ModelSlot(Config.MODEL_FALLBACK,     11),
            Config.MODEL_LAST_RESORT:  _ModelSlot(Config.MODEL_LAST_RESORT,  12, daily_limit=Config.MODEL_LAST_RESORT_RPD),
        }
        # Round-robin index (which primary model to try first)
        self._rr_idx: int = 0

        # Expose _model_stats compatible shape for heartbeat_job
        self._model_stats: dict[str, _ModelSlot] = {
            mid: slot for mid, slot in self._slots.items()
        }

    @staticmethod
    def _estimate_tokens(text: str | list[Any]) -> int:
        """Roughly estimates token count for rate-limit awareness.

        Args:
            text: Input text or list of strings.

        Returns:
            Estimated token count.
        """
        chars = sum(len(str(p)) for p in text) if isinstance(text, list) else len(str(text))
        return int(chars / 3.5) + 200

    async def _pick_model(self) -> str:
        """Round-robin across primary models, picking the one with most headroom.

        Acquires a slot atomically so it can never double-count.

        Returns:
            Model identifier of the slot that was acquired.
        """
        primaries = [Config.MODEL_ID, Config.MODEL_FALLBACK]
        # Try both in round-robin order, non-blocking check first
        for _ in range(2):
            self._rr_idx = (self._rr_idx + 1) % len(primaries)
            mid = primaries[self._rr_idx]
            slot = self._slots[mid]
            now = time.monotonic()
            async with slot._lock:
                slot._calls = [t for t in slot._calls if now - t < 60]
                if len(slot._calls) < slot.limit:
                    slot._calls.append(now)
                    return mid

        # Both full — sequential wait:
        for mid in primaries:
            slot = self._slots[mid]
            acquired = await slot.acquire(timeout=45.0)
            if acquired:
                return mid

        # Absolute fallback: check RPD and RPM for Last Resort
        lr_slot = self._slots.get(Config.MODEL_LAST_RESORT)
        if lr_slot:
            acquired = await lr_slot.acquire(timeout=5.0)
            if acquired:
                return Config.MODEL_LAST_RESORT

        logger.warning("All models at capacity or quota exhausted. Falling back to primary.")
        return primaries[0]

    async def _call(self, contents: Any, config: dict[str, Any], model_id: str, retries: int = 2) -> Any:
        """Single-responsibility caller: takes an already-acquired model_id.

        Retries on 429 by switching to the other primary, then last-resort.

        Args:
            contents: Content parts for the AI.
            config: Model configuration.
            model_id: Target model ID.
            retries: Max retries on failure.

        Returns:
            The AI response or None on failure.
        """
        primaries = [Config.MODEL_ID, Config.MODEL_FALLBACK]
        target = model_id

        for attempt in range(retries + 1):
            try:
                logger.info(f"🤖 [AI] Requesting Gemini ({target})...")
                res = await self.client.aio.models.generate_content(
                    model=target, contents=contents, config=config
                )
                logger.info(f"✨ [AI] Received response from {target}")
                return res
            except Exception as e:
                err = str(e)
                is_rate = "429" in err or "resource_exhausted" in err.lower()

                if is_rate and attempt < retries:
                    # Switch to the other primary
                    other = [m for m in primaries if m != target]
                    if other:
                        if target in self._slots:
                            self._slots[target].release_last()

                        slot = self._slots[other[0]]
                        acquired = await slot.acquire(timeout=10.0)
                        if acquired:
                            target = other[0]
                            logger.info(f"Rate-limited on {model_id}, switched to {target}")
                            continue
                    await asyncio.sleep(3 * (attempt + 1))
                    continue

                if attempt == retries:
                    # Last-ditch: LAST_RESORT fallback
                    lr_slot = self._slots.get(Config.MODEL_LAST_RESORT)
                    if lr_slot:
                        if target in self._slots:
                            self._slots[target].release_last()
                        
                        acquired = await lr_slot.acquire(timeout=5.0)
                        if acquired:
                            try:
                                return await self.client.aio.models.generate_content(
                                    model=Config.MODEL_LAST_RESORT, contents=contents, config=config
                                )
                            except Exception as e2:
                                logger.error(f"{Config.MODEL_LAST_RESORT} failed: {e2}")
                                return None

                await asyncio.sleep(1.5 ** attempt)

        return None

    # ── Public interface ──────────────────────────────────────────────────────

    def _is_worth_checking(self, text: str | None) -> bool:
        """Pre-filter — skip low-signal messages without any AI call.

        Args:
            text: Message text to check.

        Returns:
            True if high-signal, False otherwise.
        """
        if not text or not text.strip():
            return False
        t = text.strip().lower()
        if "saya membisukan dia" in t or "@dfautokick_bot" in t:
            return False

        words = t.split()
        if len(words) < 2:
            return False

        question_words = {'ga','gak','nggak','apa','gimana','berapa','kapan','dimana','kenapa'}
        if t.endswith('?') and words and words[0] in question_words:
            return False
        if len(words) <= 3 and t.endswith('?'):
            return False

        if _SOCIAL_FILLER.match(t):
            return False

        if any(kw in t for kw in _STRONG_KEYWORDS):
            return True
        if _WORD_BOUNDARY_KEYWORDS.search(t):
            return True
        if len(words) <= 3:
            return False
        return False

    async def process_batch(self, messages: Sequence[dict[str, Any]], db: Any = None) -> list[PromoExtraction] | None:
        """Extracts promos from a batch of messages using AI.

        Args:
            messages: List of message records.
            db: Optional Database instance for context.

        Returns:
            List of extracted promos or None if AI failed.
        """
        if not messages:
            return []

        logger.debug(f"AI Batch: {len(messages)} messages")
        filtered = [m for m in messages if self._is_worth_checking(m.get('text'))]

        if not filtered:
            logger.debug("All messages filtered as noise.")
            return []

        # Enrich with reply context
        if db:
            chat_id  = filtered[0]['chat_id']
            reply_ids = [m['reply_to_msg_id'] for m in filtered if m.get('reply_to_msg_id')]
            # Use deep context (3 levels) to catch brand mentions further up the chain
            reply_map = await db.get_deep_context_bulk(reply_ids, chat_id, max_depth=3) if reply_ids else {}
            for m in filtered:
                if m.get('reply_to_msg_id') and m['reply_to_msg_id'] in reply_map:
                    ctx_text = reply_map[m['reply_to_msg_id']]
                    # Take last 150 chars of context to stay lean but capture recent details
                    m['context'] = f"[context: {ctx_text[-150:]}] "
                else:
                    m['context'] = ""
        else:
            for m in filtered:
                m['context'] = ""

        batch_text = "\n---\n".join(
            f"ID:{m['id']} {m['context']}MSG:{m['text'] or ''}"
            for m in filtered
        )
        config = {
            "response_mime_type": "application/json",
            "response_schema": BatchResponse,
            "system_instruction": _EXTRACT_SYSTEM,
        }

        target_model = await self._pick_model()
        logger.debug(f"Using {target_model} for {len(filtered)} msgs")

        response = await self._call(
            contents=f"Batch pesan:\n\n{batch_text}",
            config=config,
            model_id=target_model,
        )

        if response is None:
            logger.error("AI call failed in batch process.")
            return None

        if not response.parsed:
            return []

        valid = [
            p for p in response.parsed.promos
            if (p.summary or '').strip()
               and len((p.summary or '').strip()) >= 8
               and (p.summary or '').strip().lower() not in _JUNK_SUMMARIES
        ]
        logger.info(f"Extracted {len(valid)} promos from batch.")
        return valid

    async def filter_duplicates(self, new_promos: Sequence[PromoExtraction],
                                 recent_alerts: Sequence[dict[str, Any]]) -> list[PromoExtraction]:
        """Aggressively filters duplicates using brand context and keyword overlap."""
        async with self._dedup_lock:
            if not new_promos:
                return []
            if not recent_alerts:
                return list(new_promos)

            recent_keys = {
                f"{normalize_brand(r.get('brand', '')).lower()}:{r.get('summary', '')[:35].lower()}"
                for r in recent_alerts
            }
            recent_alerts = list(recent_alerts)   # snapshot inside lock
        recent_brands_tail = [
            normalize_brand(r.get('brand', '')).lower()
            for r in recent_alerts[-50:]
        ]
        recent_brands_set = set(recent_brands_tail)

        unique: list[PromoExtraction] = []
        for p in new_promos:
            brand_key = normalize_brand(p.brand).lower()
            key = f"{brand_key}:{p.summary[:35].lower()}"

            if (brand_key in recent_brands_set
                    and brand_key != 'unknown'
                    and p.status == 'active'):
                p_words = set(re.findall(r'\w+', p.summary.lower())[:6])
                is_dupe = False
                for r in reversed(list(recent_alerts[-50:])):
                    if normalize_brand(r.get('brand', '')).lower() == brand_key:
                        r_words = set(re.findall(r'\w+', r.get('summary', '').lower())[:6])
                        if len(p_words & r_words) >= 2:
                            is_dupe = True
                            break
                if is_dupe:
                    continue

            if key not in recent_keys:
                unique.append(p)
                recent_keys.add(key)
                recent_brands_set.add(brand_key)

        return unique

    async def summarize_raw(self, texts: Sequence[str]) -> str:
        """Summarizes a set of raw chat messages.

        Args:
            texts: Raw message strings.

        Returns:
            The AI-generated summary.
        """
        if not texts:
            return "Tidak ada pesan."
        context  = "\n---\n".join(texts)
        target   = await self._pick_model()
        response = await self._call(
            contents=f"Rangkum pesan ini:\n\n{context}",
            config={"system_instruction": _DIGEST_SYSTEM},
            model_id=target,
        )
        return cast(str, response.text) if response else "❌ Gagal merangkum."

    async def summarize_thread(self, parent_text: str, replies: Sequence[str],
                                parent_photo: bytes | None = None) -> str:
        """Summarizes a specific conversation thread.

        Args:
            parent_text: The root message.
            replies: List of reply texts.
            parent_photo: Optional photo attached to parent.

        Returns:
            The thread summary.
        """
        if not replies:
            return "Thread ini sedang ramai dibicarakan."
        reply_context = "\n- ".join(replies[:20])
        prompt = (
            f"PESAN UTAMA (Thread Starter): {parent_text}\n\n"
            f"BEBERAPA BALASAN DARI USER LAIN:\n- {reply_context}\n\n"
            "TUGASMU: Rangkum diskusi ini dalam 1-2 kalimat informatif."
        )
        contents = [prompt]
        if parent_photo:
            contents.append(genai.types.Part.from_bytes(data=parent_photo, mime_type="image/jpeg"))

        target = await self._pick_model()
        response = await self._call(
            contents=contents,
            config={"system_instruction": _DIGEST_SYSTEM},
            model_id=target,
        )
        return cast(str, response.text) if response else "Thread ini sedang ramai dibicarakan."

    async def answer_question(self, question: str, context: str) -> str:
        """Answers a specific user inquiry based on provided context.

        Args:
            question: User question.
            context: Contextual data.

        Returns:
            AI response.
        """
        target = await self._pick_model()
        response = await self._call(
            contents=f"Pertanyaan: {question}\n\nKonteks:\n{context}",
            config={"system_instruction": _DIGEST_SYSTEM},
            model_id=target,
        )
        return cast(str, response.text) if response else "❌ AI Busy."

    async def process_image(self, image_bytes: bytes, caption: str | None,
                             original_msg_id: int) -> PromoExtraction | None:
        """Processes an image to extract promotional info.

        Args:
            image_bytes: Raw image data.
            caption: Optional text caption.
            original_msg_id: Original Telegram message ID.

        Returns:
            Extraction data or None.
        """
        has_promo   = bool(_PROMO.search(caption))  if caption else False
        has_nonpro  = bool(_NON_PROMO.search(caption)) if caption else False
        if has_nonpro and not has_promo:
            return None

        prompt = (f'Caption: "{caption}"' if caption else "Analisis gambar saja.")
        config = {
            "response_mime_type": "application/json",
            "response_schema": PromoExtraction,
            "system_instruction": _VISION_SYSTEM,
        }
        target = await self._pick_model()
        response = await self._call(
            contents=[prompt, genai.types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")],
            config=config,
            model_id=target,
        )
        if not response or not response.parsed:
            return None
        res = response.parsed
        JUNK = {'tidak ada','none','n/a','tidak ada promo','no promo','tidak ditemukan','-'}
        if (res.brand == "SKIP" or res.summary == "SKIP"
                or not res.summary or len(res.summary) < 10
                or res.summary.lower().strip() in JUNK
                or res.brand.lower().strip() in JUNK):
            return None
        res.original_msg_id = original_msg_id
        return res

    async def generate_narrative(self, messages: Sequence[dict[str, Any]]) -> list[TrendItem]:
        """Generates structured trend narratives for recent traffic.

        Args:
            messages: List of message records.

        Returns:
            List of TrendItem objects.
        """
        if not messages:
            return []
        context = "\n- ".join([f"ID:{m['tg_msg_id']} {m['sender_name']}: {m['text']}" for m in messages[:50]])
        target  = await self._pick_model()
        config = {
            "response_mime_type": "application/json",
            "response_schema": TrendResponse,
            "system_instruction": "Kamu analis tren. Simpulkan 1-3 tren utama dengan link ID pesan.",
        }
        response = await self._call(contents=f"Pesan grup:\n{context}", config=config, model_id=target)
        return cast(list[TrendItem], response.parsed.trends) if response and response.parsed else []

    async def interpret_keywords(self, hot_words: Sequence[str], window: int,
                                  context_msgs: Sequence[str]) -> str | None:
        """Interprets the context behind a burst of specific keywords.

        Args:
            hot_words: Frequent words detected.
            window: Time window.
            context_msgs: Full messages for context.

        Returns:
            Brief explanation or None.
        """
        if not context_msgs:
            return None
        system = f"Ada kenaikan penggunaan kata: {', '.join(hot_words)} dlm {window} menit. Jelaskan singkat."
        target = await self._pick_model()
        response = await self._call(
            contents="\n- ".join(context_msgs[:30]),
            config={"system_instruction": system},
            model_id=target
        )
        return cast(str, response.text.strip()) if response and response.text and "NO_TREND" not in response.text else None
