import re
import asyncio
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import Optional, List, Literal, Union
from config import Config
from db import normalize_brand

class PromoExtraction(BaseModel):
    original_msg_id: int
    summary: str
    brand: str
    conditions: str
    valid_until: str
    status: Literal["active", "expired", "unknown"]
    detected_at: Optional[str] = None

class BatchResponse(BaseModel):
    promos: List[PromoExtraction]

# ── Prompt constants ──────────────────────────────────────────────────────────
_EXTRACT_SYSTEM = """Kamu ekstrak promo dari percakapan grup deal hunter Indonesia (Discountfess).

ISTILAH KUNCI: mm=Mall Monday, bs=BerburuSales, nt=gagal/expired, jp=jackpot, sfood=ShopeeFood, gfood=GoFood

STATUS: active jika ada "aman/on/jp/work/restock/berhasil" | expired jika "abis/nt/sold out/ga bisa" | unknown jika ambigu

EKSTRAK jika: ada sinyal aktif/expired + info brand/platform. SKIP jika: pertanyaan murni, OOT, curhat tanpa info promo.

Context ditulis sebagai C: sebelum MSG: — gunakan untuk resolve brand jika pesan utama cuma "aman" atau "on".
Summary: 1 kalimat informatif, sertakan harga/diskon jika ada.
Brand: Gunakan nama yang konsisten — "HopHop" bukan "Hophop". Jika ragu → "Unknown" (bukan "sunknown" atau variasi lain)."""

_DEDUP_SYSTEM = "Kamu agen deteksi duplikasi. Output HANYA angka indeks dipisah koma."

_DIGEST_SYSTEM = "Kamu asisten ringkasan promo Indonesia. Jawab singkat dan informatif dalam bahasa Indonesia santai."

_VISION_SYSTEM = """Kamu analis visual grup promo (Discountfess). Tugasmu ekstrak promo dari GAMBAR (poster/screenshot) + Caption.

ATURAN:
1. Analisis gambar dengan teliti (diskon, brand, syarat).
2. Jika gambar adalah screenshot chat, prioritaskan info promo yang sedang dibahas.
3. Summary harus 1 kalimat padat.
4. Output harus valid JSON sesuai schema."""

_WORD_BOUNDARY_KEYWORDS = re.compile(
    r'\b(off|on|aman|work|bs|jp|mm)\b', re.IGNORECASE
)
_STRONG_KEYWORDS = {
    'sfood','gfood','grab','shopee','gojek','tokped','tokopedia',
    'voucher','vcr','voc','diskon','promo','cashback','gratis',
    'klaim','claim','limit','error','bug','ready','restock','ristok',
    'luber','pecah','flash','sale','deal','murah','hemat','bonus',
    'bayar','checkout','order','ongkir','gratis ongkir','cod'
}

# ─────────────────────────────────────────────────────────────────────────────

class GeminiProcessor:
    def __init__(self):
        self.client = genai.Client(api_key=Config.GEMINI_API_KEY)
        self._last_calls = []       # list of (timestamp, tokens_used)
        self._rate_lock = asyncio.Lock()

        # Limits — set conservatively below free tier ceiling
        self._rpm_limit = 8
        self._tpm_limit = 200_000   # free tier is ~250k, keep 50k headroom
        self._consecutive_429s = 0          # tracks back-to-back rate limit hits
        self._rpm_floor = 5                 # never go below this
        self._rpm_ceiling = 10              # never go above this
        self._last_rpm_adjust = 0.0         # loop time of last adjustment

    @staticmethod
    def _estimate_tokens(text: Union[str, list]) -> int:
        """Rough token estimate: Indonesian text ~3.5 chars/token, 200 overhead."""
        if isinstance(text, list):
            chars = sum(len(str(p)) for p in text)
        else:
            chars = len(str(text))
        return int(chars / 3.5) + 200

    async def _rate_limit(self, estimated_tokens: int = 1000):
        """Block until both RPM and TPM windows have capacity."""
        while True:
            async with self._rate_lock:
                now = asyncio.get_event_loop().time()
                # Evict entries older than 60s
                self._last_calls = [(t, tok) for t, tok in self._last_calls if now - t < 60]

                current_rpm = len(self._last_calls)
                current_tpm = sum(tok for _, tok in self._last_calls)

                rpm_ok = current_rpm < self._rpm_limit
                tpm_ok = (current_tpm + estimated_tokens) <= self._tpm_limit

                if rpm_ok and tpm_ok:
                    self._last_calls.append((now, estimated_tokens))
                    # Successful call — gradually recover RPM if we've been quiet
                    if self._consecutive_429s > 0:
                        self._consecutive_429s = max(0, self._consecutive_429s - 1)
                    elif (now - self._last_rpm_adjust > 120 and
                          self._rpm_limit < self._rpm_ceiling and
                          len(self._last_calls) < self._rpm_limit * 0.5):
                        # Quiet: < 50% RPM used, been 2+ min since last adjust → creep up
                        self._rpm_limit = min(self._rpm_ceiling, self._rpm_limit + 1)
                        self._last_rpm_adjust = now
                        print(f"📈 Adaptive RPM: recovered to {self._rpm_limit} RPM")
                    return

                # Calculate wait needed
                if self._last_calls:
                    wait_time = 60 - (now - self._last_calls[0][0])
                else:
                    wait_time = 1.0

            await asyncio.sleep(max(0.5, wait_time))

    async def _call_with_retry(self, contents, config, model_id=None, max_retries=2, estimated_tokens=1000):
        """Calls Gemini with automated fallback across 3 tiers of models."""
        await self._rate_limit(estimated_tokens)
        
        # 1. Gemma 31B (Primary)
        # 2. Gemma 26B (Fallback)
        # 3. Gemini 3.1 (Last Resort)
        models = [model_id or Config.MODEL_ID, Config.MODEL_FALLBACK, Config.MODEL_LAST_RESORT]
        
        for mid in models:
            if not mid: continue
            for attempt in range(max_retries):
                try:
                    response = await self.client.aio.models.generate_content(
                        model=mid, contents=contents, config=config
                    )
                    return response
                except Exception as e:
                    err_msg = str(e).lower()
                    if "429" in err_msg or "resource_exhausted" in err_msg:
                        self._consecutive_429s += 1
                        # Back off: subtract 1 RPM per hit, floor at 5
                        if self._consecutive_429s >= 2:
                            new_limit = max(self._rpm_floor, self._rpm_limit - 1)
                            if new_limit != self._rpm_limit:
                                self._rpm_limit = new_limit
                                print(f"📉 Adaptive RPM: backed off to {self._rpm_limit} RPM (429 count: {self._consecutive_429s})")
                        break  # Try next model tier
                    
                    if attempt == max_retries - 1:
                        print(f"❌ {mid} failed after {max_retries} attempts, trying next tier...")
                        continue # Try next model tier
                    
                    wait = 1.5 ** attempt
                    await asyncio.sleep(wait)
        return None

    def _is_worth_checking(self, text: str) -> bool:
        """Pre-filter to skip low-signal messages without AI calls (RPM efficiency)."""
        if not text or not text.strip():
            return False
        t = text.strip().lower()
        
        # Hard exclusions: known bot noise or non-promo system messages
        if "saya membisukan dia" in t or "@dfautokick_bot" in t:
            return False

        words = t.split()

        # Definite skip: pure question with no promo content
        question_words = {'ga','gak','nggak','apa','gimana','berapa','kapan','dimana','kenapa'}
        if t.endswith('?') and words and words[0] in question_words:
            return False
        if len(words) <= 2 and t.endswith('?'):
            return False

        # Strong keyword match → always worth it
        if any(kw in t for kw in _STRONG_KEYWORDS):
            return True

        # Word-boundary match for ambiguous shorts
        if _WORD_BOUNDARY_KEYWORDS.search(t):
            return True

        return False

    async def process_batch(self, messages: List[dict], db=None) -> List[PromoExtraction]:
        if not messages: return []

        filtered = [m for m in messages if self._is_worth_checking(m['text'])]
        if not filtered: return []

        # Enrich with context ONLY for replies if DB is provided
        if db:
            chat_id = filtered[0]['chat_id']
            reply_ids = [m['reply_to_msg_id'] for m in filtered if m.get('reply_to_msg_id')]
            reply_map = await db.get_reply_sources_bulk(reply_ids, chat_id) if reply_ids else {}
            
            for m in filtered:
                if m.get('reply_to_msg_id') and m['reply_to_msg_id'] in reply_map:
                    # fetch parent only, truncated
                    parent_text = (reply_map[m['reply_to_msg_id']] or "")[:60]
                    m['context'] = f"[reply to: {parent_text}] "
                else:
                    m['context'] = ""
        else:
            for m in filtered: m['context'] = ""

        batch_text = "\n---\n".join([
            f"ID:{m['id']} {m['context']}MSG:{m['text'] or ''}"
            for m in filtered
        ])

        config = {
            "response_mime_type": "application/json",
            "response_schema": BatchResponse,
            "system_instruction": _EXTRACT_SYSTEM,
        }

        response = await self._call_with_retry(
            contents=f"Batch pesan:\n\n{batch_text}",
            config=config,
            model_id=Config.MODEL_ID,
            estimated_tokens=self._estimate_tokens(batch_text)
        )

        if response is None: return None
        if not response.parsed: return []

        JUNK_SUMMARIES = {'summary', 'none', 'n/a', '-', 'tidak ada', 'tidak ditemukan'}
        valid = []
        for p in response.parsed.promos:
            s = (p.summary or "").strip()
            if s and len(s) >= 15 and s.lower() not in JUNK_SUMMARIES:
                valid.append(p)
        return valid

    async def filter_duplicates(self, new_promos: List[PromoExtraction], recent_alerts: List[dict]) -> List[PromoExtraction]:
        """Local fast deduplication using brand + summary matching to save AI costs."""
        if not new_promos: return []
        if not recent_alerts: return new_promos

        recent_keys = {f"{normalize_brand(r['brand']).lower()}:{r['summary'][:35].lower()}" for r in recent_alerts}
        unique = []
        for p in new_promos:
            key = f"{normalize_brand(p.brand).lower()}:{p.summary[:35].lower()}"
            if key not in recent_keys:
                unique.append(p)
                # Prevent intra-batch dupes
                recent_keys.add(key)
        return unique

    async def summarize_raw(self, texts: List[str]) -> str:
        if not texts: return "Tidak ada pesan."
        context = "\n---\n".join(texts)
        response = await self._call_with_retry(
            contents=f"Rangkum pesan ini:\n\n{context}",
            config={"system_instruction": _DIGEST_SYSTEM},
            model_id=Config.MODEL_FAST,
            estimated_tokens=self._estimate_tokens(context)
        )
        return response.text if response else "❌ Gagal merangkum."

    async def summarize_thread(self, parent_text: str, replies: List[str]) -> str:
        """Summarizes a hot thread based on the root message and its replies."""
        if not replies: return "Thread ini sedang ramai dibicarakan."

        reply_context = "\n- ".join(replies[:20])
        prompt = (
            f"PESAN UTAMA: {parent_text}\n\n"
            f"BEBERAPA BALASAN:\n- {reply_context}\n\n"
            "Berdasarkan percakapan di atas, apa intinya? (Misal: orang-orang konfirmasi promo ini work, atau lagi bahas error). Jawab 1 kalimat singkat saja."
        )

        response = await self._call_with_retry(
            contents=prompt,
            config={"system_instruction": _DIGEST_SYSTEM},
            model_id=Config.MODEL_FAST,
            estimated_tokens=self._estimate_tokens(prompt)
        )
        return response.text if response else "Thread ini sedang ramai dibicarakan."

    async def answer_question(self, question: str, context: str) -> str:

        response = await self._call_with_retry(
            contents=f"Pertanyaan: {question}\n\nKonteks:\n{context}",
            config={"system_instruction": _DIGEST_SYSTEM},
            model_id=Config.MODEL_FALLBACK,
            estimated_tokens=self._estimate_tokens(question + context)
        )
        return response.text if response else "❌ AI Busy."

    async def process_image(self, image_bytes: bytes, caption: str, original_msg_id: int) -> PromoExtraction | None:
        """Analyzes an image using a multimodal model (Gemma vision-enabled)."""
        prompt = f"Caption: {caption}" if caption else "Ekstrak info promo dari gambar ini."
        
        config = {
            "response_mime_type": "application/json",
            "response_schema": PromoExtraction,
            "system_instruction": _VISION_SYSTEM,
        }

        # Use the primary multimodal model
        response = await self._call_with_retry(
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
            ],
            config=config,
            model_id=Config.MODEL_ID,
            estimated_tokens=self._estimate_tokens(prompt) + 1000  # +1000 for image encoding overhead
        )

        if not response or not response.parsed:
            return None
            
        res = response.parsed
        res.original_msg_id = original_msg_id
        return res

    async def interpret_keywords(self, hot_words: List[str], window: int, context_msgs: List[str]) -> str | None:
        if not context_msgs: return None
        word_str = ", ".join(hot_words)
        context_str = "\n- ".join(context_msgs[:30])
        prompt = f"Trending words: {word_str}\nContext:\n{context_str}"
        
        system = (
            "Kamu analis grup promo. Simpulkan ADA KEJADIAN APA dalam 1-2 kalimat narasi.\n"
            "Jika tidak ada yang penting, tulis: NO_TREND"
        )
        
        # Use FAST model for trend interpretation
        response = await self._call_with_retry(
            contents=prompt,
            config={"system_instruction": system},
            model_id=Config.MODEL_FAST,
            estimated_tokens=self._estimate_tokens(prompt)
        )
        
        if response and response.text and "NO_TREND" not in response.text:
            return response.text.strip()
        return None
