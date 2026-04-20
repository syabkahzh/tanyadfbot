import re
import asyncio
from google import genai
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
_EXTRACT_SYSTEM = """Kamu adalah analis grup promo Indonesia. Tugasmu: baca pesan-pesan dari grup dan ekstrak HANYA pesan yang menyatakan promo/voucher/deal sedang aktif atau berhasil digunakan.

ATURAN EKSTRAKSI:
1. EKSTRAK jika pesan berisi:
   - Pernyataan bahwa promo/voucher AKTIF: "aman", "on", "ready", "masih bisa", "berhasil", "nyantol", "jp", "mantap", "work"
   - Harga spesifik yang sangat murah disebutkan bersama brand/item: "hokben 4k", "ndog 5k", "sfood 80%"
   - Konfirmasi voucher berhasil diklaim atau dipakai
   - ALLCAPS keyword + brand → hampir pasti promo aktif

2. ABAIKAN jika:
   - Pesan <5 kata tanpa brand/item spesifik (contoh: "Aman", "Mantap", "Oke", "Makasih")
   - Pertanyaan murni tanpa konfirmasi (contoh: "ada yg tau sfood?", "masih on ga?")
   - Keluhan/drama tanpa info promo (contoh: "gagal terus", "sedih banget")
   - Reply/tag orang tanpa info deal

3. BRAND: Selalu isi dengan nama brand yang jelas. Jika tidak ada brand, tulis "Unknown" — JANGAN ekstrak jika brand benar-benar tidak terdeteksi.

4. STATUS:
   - "active": ada kata "aman", "on", "berhasil", "masih bisa", atau harga murah dikonfirmasi
   - "expired": ada kata "habis", "abis", "udah ga bisa", "kehabisan", "sold out"
   - "unknown": status tidak jelas dari konteks

5. SUMMARY: singkat, informatif, max 1 kalimat. Sertakan harga jika disebutkan. Bahasa Indonesia.

BRAND UMUM yang sering muncul:
Shopee Food (sfood), GoFood, GrabFood, Hokben, McDonald's (mcd/mekdi), KFC, Chatime (kopken), 
Kopi Kenangan, Starbucks, PLN, BPJS, Telkom/Indihome, Pulsa/Kuota, Aice, Gindaco, HopHop, Fore Coffee

CONTOH BENAR:
- "aman sfood" → brand: ShopeeFood, summary: "Voucher ShopeeFood aktif", status: active ✅
- "aman ndog 5k, habis 16kg" → brand: Ndog, summary: "Telur 5k aktif, stok 16kg sudah habis", status: expired ✅  
- "berhasil dapet hokben 4k" → brand: Hokben, summary: "Hokben berhasil di-redeem harga 4k", status: active ✅
- "pln on" → brand: PLN, summary: "Promo PLN sedang aktif", status: active ✅
- "sfood 80% bs semua resto" → brand: ShopeeFood, summary: "Diskon 80% berlaku di semua resto", status: active ✅

CONTOH SALAH (jangan ekstrak):
- "Makasih kk" → tidak ada brand/promo ❌
- "masih on ga?" → pertanyaan, bukan konfirmasi ❌
- "Kurang tau 🙏" → tidak ada info ❌
- "ihh kesel bgt telor busuqq" → keluhan bukan promo ❌
"""

_DEDUP_SYSTEM = "Kamu agen deteksi duplikasi. Output HANYA angka indeks dipisah koma."

_DIGEST_SYSTEM = "Kamu asisten ringkasan promo Indonesia. Jawab singkat dan informatif dalam bahasa Indonesia santai."

# ─────────────────────────────────────────────────────────────────────────────

class GeminiProcessor:
    def __init__(self):
        self.client = genai.Client(api_key=Config.GEMINI_API_KEY)

    async def _call_with_retry(self, model_id, contents, config, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = await self.client.aio.models.generate_content(
                    model=model_id, contents=contents, config=config
                )
                return response
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"❌ AI API failed after {max_retries} attempts: {e}")
                    return None
                wait = 2 ** attempt
                print(f"⚠️ Retry {attempt+1}/{max_retries} in {wait}s: {e}")
                await asyncio.sleep(wait)
        return None

    async def process_batch(self, messages: List[dict]) -> List[PromoExtraction]:
        if not messages: return []

        # Pre-filter: skip clearly useless messages before sending to LLM
        def is_worth_checking(text: str) -> bool:
            t = text.strip()
            if len(t) < 5: return False
            words = t.split()
            if len(words) < 2:
                # Single word only worth it if it's a known strong signal
                return t.lower() in {'aman', 'on', 'mantap', 'jp', 'berhasil'}
            # Questions without confirmation keywords
            question_patterns = ['ga?', 'gak?', 'nggak?', 'kah?', 'ya?', 'dong?', '?']
            has_question = any(t.endswith(p) for p in question_patterns) or t.count('?') > t.count('!')
            promo_signals = any(w in t.lower() for w in [
                'aman', 'on', 'ready', 'berhasil', 'work', 'mantap', 'jp', 'nyantol',
                'voucher', 'vcr', 'diskon', 'promo', 'off', 'cashback', 'gratis',
                'sfood', 'gofood', 'grabfood', 'pln', 'bpjs', 'pulsa', 'kuota',
                'hokben', 'mcd', 'kfc', 'chatime', 'shopee', 'gojek', 'grab',
                'masih bisa', 'bisa pake', 'klaim', '4k', '5k', '6k', '10k', '20k',
                '%', 'ribu', 'rb', 'grb'
            ])
            # Short pure questions with no promo signal → skip
            if has_question and not promo_signals and len(words) < 6:
                return False
            return promo_signals or len(words) >= 4

        filtered = [m for m in messages if is_worth_checking(m['text'])]
        skipped = len(messages) - len(filtered)
        if skipped > 0:
            print(f"🔍 Pre-filter: skipped {skipped} low-signal msgs, sending {len(filtered)} to LLM")

        if not filtered:
            return []

        batch_text = "\n---\n".join([
            f"MSG_ID:{m['id']} | {m['text']}"
            for m in filtered
        ])

        config = {
            "response_mime_type": "application/json",
            "response_schema": BatchResponse,
            "system_instruction": _EXTRACT_SYSTEM,
        }

        response = await self._call_with_retry(
            model_id=Config.MODEL_ID,
            contents=f"Batch pesan grup promo Indonesia:\n\n{batch_text}",
            config=config
        )

        if response is None: return None
        if not response.parsed: return []

        # Validate: drop results where brand is clearly bad
        valid = []
        for p in response.parsed.promos:
            if not p.summary or p.summary.strip() == '':
                continue
            valid.append(p)
        return valid

    async def filter_duplicates(self, new_promos: List[PromoExtraction], recent_alerts: List[dict]) -> List[PromoExtraction]:
        if not new_promos: return []
        if not recent_alerts: return new_promos

        recent_context = "\n".join([f"- {r['brand']}: {r['summary']}" for r in recent_alerts])
        new_context = "\n".join([f"IDX {i}: {p.brand} — {p.summary}" for i, p in enumerate(new_promos)])

        prompt = (
            f"PROMO SUDAH ADA:\n{recent_context}\n\n"
            f"PROMO BARU (cek duplikat):\n{new_context}\n\n"
            "Return indeks IDX dari promo yang BUKAN duplikat, dipisah koma. "
            "Duplikat = brand sama + penawaran sama. Jika semua unik, return semua indeks."
        )

        response = await self._call_with_retry(
            model_id=Config.MODEL_ID,
            contents=prompt,
            config={"system_instruction": _DEDUP_SYSTEM}
        )

        if not response or not response.text:
            return new_promos

        try:
            valid_indices = [int(i) for i in re.findall(r'\d+', response.text) if int(i) < len(new_promos)]
            if not valid_indices:
                return new_promos
            return [new_promos[i] for i in valid_indices]
        except Exception:
            return new_promos

    async def summarize_raw(self, texts: List[str]) -> str:
        if not texts: return "Tidak ada pesan."
        context = "\n---\n".join(texts)
        prompt = (
            "Rangkum pesan-pesan berikut dari grup chat promo secara singkat dan padat "
            "dalam bahasa Indonesia santai. Fokus pada deal/promo yang disebutkan. "
            "Gunakan bullet points."
        )
        response = await self._call_with_retry(
            model_id=Config.MODEL_ID,
            contents=f"{prompt}\n\nPesan:\n{context}",
            config={"system_instruction": _DIGEST_SYSTEM}
        )
        return response.text if response else "❌ Gagal merangkum."

    async def answer_question(self, question: str, context: str) -> str:
        response = await self._call_with_retry(
            model_id=Config.MODEL_ID,
            contents=f"Pertanyaan: {question}\n\nKonteks promo:\n{context}",
            config={"system_instruction": _DIGEST_SYSTEM}
        )
        return response.text if response else "❌ AI API Busy."

    async def interpret_keywords(self, hot_words: List[str], window: int, context_msgs: List[str]) -> str | None:
        if not context_msgs: return None
        word_str = ", ".join(hot_words)
        context_str = "\n- ".join(context_msgs[:20])
        prompt = f"Dalam {window} menit terakhir, kata ini sering muncul: {word_str}\n\nKonteks:\n{context_str}"
        system = "Kamu analis grup promo. Jika tidak ada promo/event jelas, ketik NO_TREND saja."
        response = await self._call_with_retry(
            model_id=Config.MODEL_ID,
            contents=prompt,
            config={"system_instruction": system}
        )
        if response and response.text and "NO_TREND" not in response.text:
            return response.text.strip()
        return None
