import re
import asyncio
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import Optional, List, Literal
from config import Config

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

# ─────────────────────────────────────────────────────────────────────────────

class GeminiProcessor:
    def __init__(self):
        self.client = genai.Client(api_key=Config.GEMINI_API_KEY)
        self._last_calls = [] # List of timestamps

    async def _rate_limit(self):
        """Ensures max 10 calls per minute."""
        while True:
            now = asyncio.get_event_loop().time()
            # Remove calls older than 60s
            self._last_calls = [t for t in self._last_calls if now - t < 60]
            if len(self._last_calls) < 10:
                self._last_calls.append(now)
                return
            wait_time = 60 - (now - self._last_calls[0])
            await asyncio.sleep(max(0.1, wait_time))

    async def _call_with_retry(self, contents, config, model_id=None, max_retries=2):
        """Calls Gemini with automated fallback and rate limiting."""
        await self._rate_limit()
        # 1. Start with requested model (or default)
        # 2. Fallback to Config.MODEL_FALLBACK (gemini-3.1-flash-lite-preview)
        models = [model_id or Config.MODEL_ID, Config.MODEL_FALLBACK]
        
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
                        print(f"⚠️ Rate limited on {mid}, trying next fallback...")
                        break # Try next model
                    
                    if attempt == max_retries - 1:
                        print(f"❌ {mid} failed after {max_retries} attempts, trying fallback...")
                        continue # Try next model
                    
                    wait = 1.5 ** attempt
                    await asyncio.sleep(wait)
        return None

    def _is_worth_checking(self, text: str) -> bool:
        """Pre-filter to skip low-signal messages without AI calls (RPM efficiency)."""
        t = text.strip().lower()
        if not t: return False
        
        # Hard exclusions: known bot noise or non-promo system messages
        if "saya membisukan dia" in t or "@dfautokick_bot" in t:
            return False

        words = t.split()
        
        # 1. Immediate triggers for very short signals
        short_signals = {
            'aman', 'on', 'jp', 'mm', 'bs', 'ywwa', 'work', 'ready', 
            'mantap', 'masuk', 'pecah', 'luber', 'ristok', 'restock'
        }
        if t in short_signals:
            return True

        # 2. Pattern detection
        has_percent = bool(re.search(r'\d+\s*(%|persen)', t))
        has_price = bool(re.search(r'\d+\s*(k|rb|ribu)', t))
        
        question_patterns = ['ga?', 'gak?', 'nggak?', 'kah?', 'ya?', 'dong?', '?']
        is_question = any(t.endswith(p) for p in question_patterns) or t.count('?') > t.count('!')

        # 3. Expanded Keyword sets
        keywords = {
            # Platforms/Brands
            'sfood', 'gfood', 'grab', 'shopee', 'gojek', 'tokped', 'blibli',
            'spay', 'gopay', 'ovo', 'dana', 'linkaja', 'pln', 'bpjs', 'pulsa',
            'hokben', 'mcd', 'kfc', 'chatime', 'starbucks', 'kopken',
            # Promo signals
            'voucher', 'vcr', 'diskon', 'promo', 'off', 'cashback', 'gratis',
            'klaim', 'claim', 'limit', 'error', 'bug', 'mumpung', 'gercep',
            'masih bisa', 'bisa pake', 'nyantol', 'kejar'
        }
        
        has_keyword = any(kw in t for kw in keywords)
        
        # Logic:
        # - Always check if it has a price, percentage, or high-signal keyword
        if has_percent or has_price or has_keyword:
            return True
            
        # - If it's a question with no signals, skip if it's short
        if is_question and len(words) < 5:
            return False
            
        # - Fallback for longer messages that might be complex descriptions
        return len(words) >= 6

    async def process_batch(self, messages: List[dict]) -> List[PromoExtraction]:
        if not messages: return []

        filtered = [m for m in messages if self._is_worth_checking(m['text'])]
        if not filtered: return []

        batch_text = "\n---\n".join([
            f"ID:{m['id']} C:{m.get('context','')} MSG:{m['text']}"
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
            model_id=Config.MODEL_ID
        )

        if not response or not response.parsed: return []

        valid = []
        for p in response.parsed.promos:
            if p.summary and p.summary.strip():
                valid.append(p)
        return valid

    async def filter_duplicates(self, new_promos: List[PromoExtraction], recent_alerts: List[dict]) -> List[PromoExtraction]:
        """Local fast deduplication using brand + summary matching to save AI costs."""
        if not new_promos: return []
        if not recent_alerts: return new_promos

        recent_keys = {f"{r['brand'].lower()}:{r['summary'][:35].lower()}" for r in recent_alerts}
        unique = []
        for p in new_promos:
            key = f"{p.brand.lower()}:{p.summary[:35].lower()}"
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
            model_id=Config.MODEL_FALLBACK  # Use cheaper model for generic summaries
        )
        return response.text if response else "❌ Gagal merangkum."

    async def answer_question(self, question: str, context: str) -> str:
        response = await self._call_with_retry(
            contents=f"Pertanyaan: {question}\n\nKonteks:\n{context}",
            config={"system_instruction": _DIGEST_SYSTEM},
            model_id=Config.MODEL_FALLBACK
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
            model_id=Config.MODEL_ID
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
            model_id=Config.MODEL_FAST
        )
        
        if response and response.text and "NO_TREND" not in response.text:
            return response.text.strip()
        return None
