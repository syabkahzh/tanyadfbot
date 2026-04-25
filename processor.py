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

# ── Pre-compiled Patterns ─────────────────────────────────────────────────────
_WORD_BOUNDARY_KEYWORDS = re.compile(
    r'\b(off|on|aman|work|bs|jp|mm)\b', re.IGNORECASE
)
_SOCIAL_FILLER = re.compile(
    r'^((wkwk|haha|hehe|iya|noted|oke|ok|makasih|thanks|thx|mantap|gas|bos|guys|gais|bang|kak|siap|sip|lol|anjir|anjay|btw|oot|gws|semangat|ya allah|nangis|sedih|beneran|kah|'
    r'ywwa|bau|goib|zonk|cuan|nt|jp|aman)[!.\s]*)+$',
    re.IGNORECASE
)
_NON_PROMO = re.compile(
    r'\b(setting|pengaturan|config|tutorial|cara|gimana|help|tolong|ini kak|'
    r'oot|random|foto|selfie|meme|lucu|haha|wkwk)\b', re.IGNORECASE
)
_PURE_QUESTION_PATTERN = re.compile(
    r'^(kak|ka|guys?|gais|ges|gaes|bun|mba|mas|bang)\s+'
    r'(ada|mau|bisa|boleh|tanya|gimana|berapa|kapan|dimana|apa|gmn|brp)\b',
    re.IGNORECASE
)
_PROMO = re.compile(
    r'\b(promo|diskon|cashback|voucher|gratis|murah|hemat|sale|off|deal|potongan|'
    r'sfood|gfood|grab|shopee|gojek|aman|on|jp|work|flash|limit|idm|alfa|indomaret|'
    r'nt|abis|habis|gabisa|gaada|gamau|minbel|r\+s\+t\+k|r\+s\+t\+c\+k|r\+st\+ck|'
    r'cb|kesbek|c\+s\+h\+b\+c\+k|cash back|kuota|slot|redeem|qr|scan|edc|'
    r'membership|member|mamber|cek|info|luber|pecah|'
    r'makasih|thx|thanks|makasi|mks|terima.?kasih|'
    r'yang butuh aja|ymma|'
    r'tukpo|murce|murmer|sopi|tsel|cgv|xxi|svip|badut|war|begal|kreator|'
    r'kopken|chatime|gindaco|solaria|rotio|spx|gopay|spay|ovo|'
    r'neo|tmrw|saqu|seabank|hero|'
    r'garap|serbabu|goib|emados|tts|blibli|famima|supin|flip|superbank|gacoan|sei|azko|pc|ndog|'
    r'periode|last day|reset|dom)\b', re.IGNORECASE
)
_JUNK_SUMMARY_PATTERN = re.compile(
    r'\b(tidak ada|none|n/a|tidak ditemukan|no promo)\b', re.IGNORECASE
)
_CURRENCY_DISCOUNT_PATTERN = re.compile(
    r'(rp\s?\d|rb\s?\d|\d+[kK]|disc|diskon|gratis|free|\d+\s*%|cashback)',
    re.IGNORECASE
)
# Summaries that describe the message rather than the promo — always reject
_META_SUMMARY_PATTERN = re.compile(
    r'(user bertanya|tidak ada informasi|tidak disebutkan|no information|'
    r'pesan ini|pertanyaan tentang|menanyakan|mencari tahu|'
    r'meminta konfirmasi|menginformasikan bahwa)',
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
    confidence: float # 0.0 to 1.0 based on AI certainty
    links: List[str] = []
    detected_at: Optional[str] = None
    queue_time: Optional[float] = None
    ai_time: Optional[float] = None

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


# ── Prompt constants ──────────────────────────────────────────────────────────

_EXTRACT_SYSTEM = """Kamu ekstrak promo dari percakapan grup deal hunter Indonesia (Discountfess).

ISTILAH KUNCI: mm=Mall Monday, bs=BerburuSales, nt=gagal/expired, jp=jackpot, sfood=ShopeeFood, gfood=GoFood,
tukpo=Tokopedia, murce=murah meriah, sopi=Shopee, tsel=Telkomsel, badut=beli-ada-duit (cashback trick)

STATUS: active jika ada "aman/on/jp/work/restock/berhasil/nyala/cair/lancar/masuk/cek/info/makasih" | expired jika "abis/nt/sold out/ga bisa/koid/hangus/zonk" | unknown jika ambigu

ATURAN UTAMA:
1. Pahami konteks obrolan (slang). Jika pengguna bilang "nyala", "on", "cair", "masuk", "cek", "info", "makasih", atau "udah pake" terkait suatu brand, itu ADALAH event promo yang valid. Ekstrak sebagai active.
2. JANGAN MEMAKSAKAN ANGKA KONKRET. Jika diskon/harga tidak disebutkan tapi status promo jelas aktif, biarkan field harga null/kosong, tetapi TETAP ekstrak sebagai data valid.
3. Abaikan obrolan OOT (Out of Topic) seperti review barang fisik (parfum, makanan tanpa konteks promo), keluhan rute/kurir, atau curhatan non-promo.

EKSTRAK jika: ada sinyal aktif/expired + info brand/platform (bahkan tanpa angka harga/diskon). SKIP jika: pertanyaan murni, OOT, curhat tanpa info promo.
JANGAN EKSTRAK: form isi data pribadi (NIK/KTP/alamat), kuis berhadiah yang butuh upload foto/data, konten OOT.

Context ditulis sebagai C: sebelum MSG: — gunakan untuk resolve brand jika pesan utama cuma "aman" atau "on".
Summary: 1 kalimat informatif, sertakan harga/diskon jika ada. Jika tidak ada angka, cukup sebutkan status + brand.
Brand: Gunakan nama yang konsisten — "HopHop" bukan "Hophop". Jika ragu → "Unknown" (bukan "sunknown" atau variasi lain).

PENTING: Jika kamu tidak yakin ada promo nyata, JANGAN isi summary with deskripsi tentang pesan itu sendiri seperti "User bertanya tentang..." atau "Pesan ini membahas...". Lebih baik SKIP sama sekali.

ATURAN BRAND (PENTING — sering salah):
- `brand` = MERCHANT/TOKO tempat promo ditebus, BUKAN metode pembayaran.
- Metode pembayaran (ShopeePay/Spay, GoPay, DANA, OVO, AstraPay, kartu kredit, QRIS) masuk ke `conditions`, TIDAK PERNAH jadi `brand` kecuali promo itu murni promo aplikasi dompet tanpa merchant spesifik.
- Struk/bukti transaksi: brand = nama toko di struk. Contoh: struk "AFM RAYA TUBAN" (AFM = Alfamart) + banner "Cashback Saldo ShopeePay" → brand = **Alfamart**, conditions menyebut ShopeePay.
- Singkatan struk yang umum: `AFM` = Alfamart, `IDM` = Indomaret, `AFMD` = Alfamidi.
- Slang caption: `jsm` (Jumat Sabtu Minggu) & `psm` (Promo Spesial Minggu) selalu berarti Alfamart. Caption "aman jsm" pada struk = konfirmasi promo Alfamart JSM berhasil.
- Cross-promo (toko × dompet): `brand` = toko. Jika bingung mana toko mana dompet, pilih yang muncul di bagian BADGE/HEADER struk atau logo fisik toko, bukan logo banner promosi di atasnya.

ATURAN OUTPUT:
- Jika SKIP: {"summary": "SKIP", "brand": "SKIP", "conditions": "", "valid_until": "", "status": "unknown", "original_msg_id": 0}
- Jika promo valid: summary 1 kalimat padat dengan brand + status. Sertakan harga/diskon jika disebutkan, tapi boleh kosong jika status jelas.
- Confidence: Berikan skor 0.0 - 1.0. Skor tinggi (>0.85) jika ada bukti kuat (struk/screenshot/keyword jelas). Skor rendah (<0.7) jika ambigu atau cuma diskusi singkat tanpa bukti konkrit.
- Brand: nama konsisten (Alfamart, Indomaret, Tokopedia, Shopee, ShopeeFood, GoFood, ShopeePay, GoPay, Solaria, CGV, Telkomsel, dll). "Unknown" hanya jika benar-benar tidak jelas.

CONTOH YANG HARUS DI-SKIP:
- 'wkwkwk iya bener' → OOT
- 'hasilnya masih sama kak' → konteks tidak cukup, skip
- 'mau tanya dong, masih on gak?' → pertanyaan murni
- 'noted makasih' → bukan promo
- 'aman kak rutenya' → OOT (transportasi, bukan promo)

CONTOH YANG HARUS DIEKSTRAK:
- 'sfood masih on 40k dapet 2' → active, ShopeeFood
- 'gobiz 30% off s/d jam 12 aman dicoba' → active, GoBiz
- 'NT gaes, udah abis' → expired, status flip
- 'tsel nyala' → active, Telkomsel (tanpa angka pun tetap valid)
- 'sopi cair' → active, Shopee (status jelas aktif meski tanpa angka)
- 'cgv on ges' → active, CGV"""

_DEDUP_SYSTEM = "Kamu agen deteksi duplikasi. Output HANYA angka indeks dipisah koma."

_DIGEST_SYSTEM = "Kamu asisten ringkasan promo Indonesia. Jawab singkat dan informatif dalam bahasa Indonesia santai."

_VISION_SYSTEM = """Kamu analis visual grup promo Indonesia (Discountfess).

TUGASMU: Tentukan apakah gambar ini berisi informasi promo yang bisa dimanfaatkan, lalu ekstrak detailnya.

PROMO VALID — ekstrak jika gambar berisi:
- Poster/banner promo brand (diskon, cashback, voucher, harga spesial)
- Screenshot aplikasi yang menampilkan harga/voucher/deal aktif
- Bukti transaksi with promo (struk, order confirmation with diskon)
- Screenshot chat/grup yang membahas promo konkret with angka/brand jelas

TOLAK (isi summary="SKIP", brand="SKIP") jika gambar adalah:
- Screenshot settings/UI aplikasi tanpa info promo
- Foto makanan/produk biasa tanpa harga promo
- Meme, stiker, foto personal
- Screenshot chat yang tidak membahas promo konkret
- Gambar blur/tidak jelas
- Konten OOT apapun

PENTING (KASUS PERTANYAAN):
Jika caption berbentuk pertanyaan (misal: "lazlive jadwalnya ini kah??" atau "ini promo bukan?"), JANGAN LANGSUNG DI-SKIP. Periksa gambarnya! Jika gambar tersebut valid (poster jadwal, screenshot diskon/voucher), maka EKSTRAK informasinya. Anggap caption pertanyaan tersebut sebagai cara user membagikan info.

ATURAN OUTPUT:
- Jika SKIP: {"summary": "SKIP", "brand": "SKIP", "conditions": "", "valid_until": "", "status": "unknown", "original_msg_id": 0, "confidence": 0.0}
- Jika promo valid: summary 1 kalimat padat dengan brand + diskon/harga + syarat utama
- Brand: nama konsisten, "Unknown" jika tidak jelas tapi promo valid
- Confidence: Berikan skor 0.0 - 1.0 berdasarkan tingkat kepastian bahwa gambar tersebut adalah promo nyata. (0.9+ untuk poster resmi/struk jelas, <0.7 untuk gambar samar/tidak meyakinkan)"""



# ── Pre-filter keyword sets ───────────────────────────────────────────────────

_SLANG_KAMUS = {
    'ywwa': 'yang wangi-wangi aja (pamer hoki/promo)',
    'bau': 'tidak hoki / tidak dapat promo / amsyong',
    'cibu': 'cashback ribu (biasanya promo receh 1k)',
    'yang butuh aja': 'terkait promo bagi-bagi kuota/voucher',
    'ymma': 'yang mau mau aja (promo terbatas)',
    'on': 'promo masih aktif / work / bisa ditebus',
    'jp': 'jackpot / berhasil tembus promo / hoki',
    'nt': 'nice try / gagal / habis / kuota limit',
    'luber': 'melimpah / restock besar / banjir promo',
    'nyantol': 'promo berhasil didapatkan / masuk ke akun',
    'klem': 'klaim',
    'burek': 'buka rekening (biasanya bank digital untuk dapat promo)',
    'gratong': 'gratis ongkir',
    'kompen': 'kompensasi (voucher ganti rugi dari CS)',
    'selfre': 'self reward (membeli barang tanpa promo / bayar harga normal)',
    'md': 'milidetik (waktu spesifik untuk war promo)',
    'mnt': 'menit',
    'spl': 'Shopee PayLater',
    'gpl': 'GoPayLater',
    'getok': 'getok telur (game berhadiah / promo)',
    'gosok': 'gosok kartu (game berhadiah / promo)',
    'skinenjel': 'brand skincare skin1004',
    'mapok': 'brand popok Mamy Poko',
    'semar': 'Toko Emas Semar Nusantara (sering dibeli pakai voucher diskon)',
}

_STRONG_KEYWORDS: set[str] = {
    'sfood','gfood','grab','shopee','gojek','tokped','tokopedia',
    'voucher','vcr','voc','diskon','promo','cashback','gratis','potongan',
    'idm','indomaret','alfa','alfamart','alfagift','hokben',
    'klaim','claim','restock','ristok','nt','abis','habis',
    'gabisa','gaada','g+b+s','gamau','minbel',
    'kuota','limit','slot','redeem','qr','scan','edc',
    'r+s+t+k',r'r+s+t+c\+k',r'r\+st\+ck',
    'cb','kesbek',r'c\+s\+h\+b\+c\+k','cash back',
    'luber','pecah','flash','sale','deal','murah','hemat','bonus',
    'ongkir','gratis ongkir',
    'membership','member','mamber',
    'yang butuh aja','ymma',
    'tukpo','murce','murmer','sopi','tsel','cgv','xxi','svip','badut','war','begal','kreator','live kreator',
    'kopken','chatime','gindaco','solaria','rotio','spx','gopay','spay','shopeepay','ovo','neo','tmrw','saqu','seabank','hero',
    'blibli', 'serbabu', 'famima', 'familymart',
    'supin', 'superindo', 'gacoan', 'mie gacoan',
    'wingstop', 'yoshinoya', 'azko', 'sei',
    'flip', 'superbank', 'dana',
    'tts', 'emados', 'ndog', 'pc', 'garap',
}

_WEAK_KEYWORDS: set[str] = {
    'cek', 'info', 'makasih', 'thx', 'thanks', 'makasi', 'mks', 'terimakasih', 'terima kasih'
}

_JUNK_SUMMARIES: set[str] = {'summary','none','n/a','-','tidak ada','tidak ditemukan'}


# ── Rate limiter ──────────────────────────────────────────────────────────────

class _ModelSlot:
    """Sliding-window RPM + daily RPD limiter.

    KEY DESIGN: acquire() is the ONLY way to register a call.
    release_last() is synchronous so counts are accurate immediately.
    """

    def __init__(self, model_id: str, limit: int, daily_limit: int = 0) -> None:
        self.model_id = model_id
        self.limit = limit
        self.daily_limit = daily_limit
        self._calls: list[float] = []
        self._daily_calls: list[float] = []
        self._lock = asyncio.Lock()

    def _cleanup(self, now: float) -> None:
        """Remove expired timestamps. Must be called under self._lock."""
        self._calls = [t for t in self._calls if now - t < 60]
        if self.daily_limit > 0:
            self._daily_calls = [t for t in self._daily_calls if now - t < 86400]

    def available(self, now: float) -> int:
        """Current available RPM slots. Approximate — does not lock."""
        cutoff = now - 60
        active = sum(1 for t in self._calls if t > cutoff)
        return max(0, self.limit - active)

    async def try_acquire_nowait(self) -> bool:
        """Non-blocking attempt. Returns True and records the call if a slot is free."""
        now = time.monotonic()
        async with self._lock:
            self._cleanup(now)
            if self.daily_limit > 0 and len(self._daily_calls) >= self.daily_limit:
                return False
            if len(self._calls) < self.limit:
                self._calls.append(now)
                if self.daily_limit > 0:
                    self._daily_calls.append(now)
                return True
        return False

    async def acquire(self, timeout: float = 90.0) -> bool:
        """Blocking acquire. Returns True if a slot was obtained before timeout."""
        deadline = time.monotonic() + timeout
        while True:
            if await self.try_acquire_nowait():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(2.0, remaining))

    def release_last(self) -> None:
        """Synchronously remove the most recent call record.

        This is intentionally synchronous so the count is accurate the instant
        this method returns — no fire-and-forget task races.
        """
        if self._calls:
            self._calls.pop()
        if self.daily_limit > 0 and self._daily_calls:
            self._daily_calls.pop()

    def current_usage(self) -> int:
        now = time.monotonic()
        return sum(1 for t in self._calls if now - t < 60)

    def daily_usage(self) -> int:
        now = time.monotonic()
        return sum(1 for t in self._daily_calls if now - t < 86400)


# ─────────────────────────────────────────────────────────────────────────────

class GeminiProcessor:
    """Orchestrates AI analysis using Gemini/Gemma models with balanced load."""

    def __init__(self) -> None:
        """Initializes the GeminiProcessor."""
        self.client = genai.Client(api_key=Config.GEMINI_API_KEY)
        self._dedup_lock = asyncio.Lock()

        # Both models get the same RPM limit (12 each = 24 total aggregate).
        self._slots: dict[str, _ModelSlot] = {
            Config.MODEL_ID:       _ModelSlot(Config.MODEL_ID,       12),
            Config.MODEL_FALLBACK: _ModelSlot(Config.MODEL_FALLBACK, 12),
        }

        # Strict round-robin index — incremented BEFORE use
        self._rr_idx: int = 0
        self._rr_lock: asyncio.Lock = asyncio.Lock()

        # Expose _model_stats for heartbeat_job compatibility
        self._model_stats: dict[str, _ModelSlot] = dict(self._slots)

    @staticmethod
    def _estimate_tokens(text: str | list[Any]) -> int:
        """Roughly estimates token count for rate-limit awareness."""
        chars = sum(len(str(p)) for p in text) if isinstance(text, list) else len(str(text))
        return int(chars / 3.5) + 200

    async def _pick_model(self) -> str:
        """Strictly alternating round-robin with fallback to the other model.

        Returns the model_id whose slot has been acquired.
        """
        primaries = [Config.MODEL_ID, Config.MODEL_FALLBACK]

        async with self._rr_lock:
            self._rr_idx = (self._rr_idx + 1) % len(primaries)
            primary_idx = self._rr_idx

            primary = primaries[primary_idx]
            secondary = primaries[1 - primary_idx]

            # 1. Try primary non-blocking
            if await self._slots[primary].try_acquire_nowait():
                return primary

            # 2. Try secondary non-blocking
            if await self._slots[secondary].try_acquire_nowait():
                return secondary

        # 3. Both full — pick the one with more headroom and wait briefly.
        now = time.monotonic()
        p_avail = self._slots[primary].available(now)
        s_avail = self._slots[secondary].available(now)
        wait_on = primary if p_avail >= s_avail else secondary
        other   = secondary if wait_on == primary else primary

        logger.debug(f"Both models at capacity, waiting briefly on {wait_on}...")
        acquired = await self._slots[wait_on].acquire(timeout=8.0)
        if acquired:
            return wait_on

        # Last resort: quick try on the other one before giving up.
        acquired = await self._slots[other].acquire(timeout=4.0)
        if acquired:
            return other

        raise TimeoutError("Both model slots exhausted — rate limit exceeded")

    # Hard ceiling on a single generate_content call.
    _AI_CALL_TIMEOUT_SEC: float = 30.0

    async def _call(self, contents: Any, config: dict[str, Any], model_id: str, retries: int = 2) -> Any:
        """Execute an AI call on an already-acquired model slot."""
        primaries = [Config.MODEL_ID, Config.MODEL_FALLBACK]
        target = model_id
        slot_acquired = target  # track which slot we're holding

        for attempt in range(retries + 1):
            try:
                logger.info(f"🤖 [AI] Requesting {target} (attempt {attempt + 1})...")
                async with asyncio.timeout(self._AI_CALL_TIMEOUT_SEC):
                    res = await self.client.aio.models.generate_content(
                        model=target, contents=contents, config=config
                    )
                logger.info(f"✨ [AI] Response from {target}")
                return res
            except asyncio.TimeoutError:
                logger.warning(
                    f"AI ({target}) timed out after {self._AI_CALL_TIMEOUT_SEC:.0f}s on attempt {attempt + 1}."
                )
                self._slots[slot_acquired].release_last()
                if attempt < retries:
                    other = [m for m in primaries if m != target]
                    if other:
                        acquired = await self._slots[other[0]].acquire(timeout=8.0)
                        if acquired:
                            target = other[0]
                            slot_acquired = target
                            continue
                    acquired = await self._slots[target].acquire(timeout=8.0)
                    if acquired:
                        slot_acquired = target
                    else:
                        return None
                    continue
                logger.error(f"AI call timed out after {retries + 1} attempts.")
                return None
            except Exception as e:
                err = str(e)
                is_rate = "429" in err or "resource_exhausted" in err.lower()
                is_internal = "500" in err or "internal" in err.lower()

                if (is_rate or is_internal) and attempt < retries:
                    reason = "Rate-limited" if is_rate else "Internal error"
                    logger.warning(f"AI ({target}) {reason}: {err[:120]}. Retrying...")

                    # Release current slot synchronously
                    self._slots[slot_acquired].release_last()

                    # Try the other model
                    other = [m for m in primaries if m != target]
                    if other:
                        acquired = await self._slots[other[0]].try_acquire_nowait()
                        if not acquired:
                            await asyncio.sleep(2.0)
                            acquired = await self._slots[other[0]].acquire(timeout=10.0)
                        if acquired:
                            target = other[0]
                            slot_acquired = target
                            continue

                    # Can't switch — back off and retry same model
                    wait = (3.0 * (attempt + 1)) if is_internal else (1.5 ** attempt)
                    await asyncio.sleep(wait)
                    # Re-acquire the original slot before retrying
                    acquired = await self._slots[target].acquire(timeout=15.0)
                    if acquired:
                        slot_acquired = target
                    else:
                        logger.error(f"Could not re-acquire slot for {target}")
                        return None
                    continue

                if attempt == retries:
                    logger.error(f"AI call failed after {retries + 1} attempts: {err[:200]}")
                    self._slots[slot_acquired].release_last()
                    if is_internal:
                        # 500 internals shouldn't penalize the message's failure count permanently.
                        # We return a specific sentinel string that the caller can interpret as a transient error.
                        raise Exception("TEMP_500_INTERNAL")
                    return None

                await asyncio.sleep(1.5 ** attempt)

        self._slots[slot_acquired].release_last()
        return None

    # ── Public interface ──────────────────────────────────────────────────────

    def _is_worth_checking(self, text: str | None, has_photo: bool = False) -> bool:
        """Pre-filter: skip low-signal messages without any AI call."""
        if has_photo:
            return True
        if not text or not text.strip():
            return False
        t = text.strip().lower()
        if "saya membisukan dia" in t or "@dfautokick_bot" in t:
            return False
        if len(t) < 4:
            return False

        if _SOCIAL_FILLER.match(t):
            return False

        words = t.split()

        # Heuristic scoring
        score = 0

        # Strong indicators (+)
        if any(kw in t for kw in _STRONG_KEYWORDS):
            score += 10
        if any(kw in t for kw in _WEAK_KEYWORDS):
            score += 3
        if _WORD_BOUNDARY_KEYWORDS.search(t):
            score += 5
        if bool(_PROMO.search(t)):
            score += 2

        # Question / Noise indicators (-)
        question_words = {'ga', 'gak', 'nggak', 'apa', 'gimana', 'berapa', 'kapan', 'dimana', 'kenapa', 'ada', 'masih', 'ya'}
        has_question_word = any(w in question_words for w in words)
        
        if '?' in t:
            score -= 10
            
        # Phrases like "aman ga", "aman ya" -> high penalty
        if re.search(r'\b(aman|work|on)\s+(ga|gak|nggak|ya)\b', t):
            score -= 15

        if t.endswith('?') and words and words[0] in question_words:
            score -= 5

        if has_question_word and ('aman' in t or 'work' in t or 'on' in t):
            # Probably asking if it's working
            score -= 8

        # specifically penalize single short word followed by ngga
        if re.search(r'\b(aman|work|on)\s+(ngga)\b', t):
            score -= 15

        # Short message penalty
        if len(words) <= 4 and score < 5:
            return False

        return score > 2

    async def process_batch(self, messages: Sequence[dict[str, Any]], db: Any = None) -> list[PromoExtraction] | None:
        """Extracts promos from a batch of messages using AI."""
        if not messages:
            return []

        filtered = [m for m in messages if self._is_worth_checking(m.get('text'), bool(m['has_photo']))]
        if not filtered:
            return []

        # Enrich with reply context
        if db:
            chat_id  = filtered[0]['chat_id']
            reply_ids = [m['reply_to_msg_id'] for m in filtered if m.get('reply_to_msg_id')]
            reply_map = await db.get_deep_context_bulk(reply_ids, chat_id, max_depth=3) if reply_ids else {}
            for m in filtered:
                if m.get('reply_to_msg_id') and m['reply_to_msg_id'] in reply_map:
                    ctx_text = reply_map[m['reply_to_msg_id']]
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
            return None

        if not response.parsed:
            return []

        valid = []
        for p in response.parsed.promos:
            summary = (p.summary or "").strip()
            if not summary or len(summary) < 8:
                continue
            if summary.lower() in _JUNK_SUMMARIES:
                continue
            if _META_SUMMARY_PATTERN.search(summary):
                logger.debug(f"Rejected meta-summary: {summary[:60]}")
                continue
            valid.append(p)

        logger.info(f"Extracted {len(valid)} promos from batch of {len(filtered)} msgs.")
        return valid

    async def filter_duplicates(self, new_promos: Sequence[PromoExtraction],
                                 recent_alerts: Sequence[dict[str, Any]]) -> list[PromoExtraction]:
        """Aggressively filters duplicates using brand context and keyword overlap."""
        if not new_promos:
            return []

        # Cross-batch history (what the caller already alerted on recently)
        history_tail = list(recent_alerts)[-50:]
        recent_keys = {
            f"{normalize_brand(r['brand']).lower()}:{r['summary'][:35].lower()}"
            for r in recent_alerts
        }
        recent_brands_set = {
            normalize_brand(r['brand']).lower() for r in history_tail
        }

        unique: list[PromoExtraction] = []
        # Intra-batch reservation: what we've already accepted in THIS call.
        intra_batch_keys: set[str] = set()
        intra_batch_by_brand: dict[str, list[set[str]]] = {}

        for p in new_promos:
            brand_key = normalize_brand(p.brand).lower()
            key = f"{brand_key}:{p.summary[:35].lower()}"

            # Exact key match against either history or this batch = dupe
            if key in recent_keys or key in intra_batch_keys:
                continue

            p_words = set(re.findall(r'\w+', p.summary.lower())[:8])

            # Cross-batch fuzzy dedup
            if (brand_key in recent_brands_set
                    and brand_key != 'unknown'
                    and p.status == 'active'):
                is_dupe = False
                for r in reversed(history_tail):
                    if normalize_brand(r['brand']).lower() == brand_key:
                        r_words = set(re.findall(r'\w+', r['summary'].lower())[:8])
                        if len(p_words & r_words) >= 2:
                            is_dupe = True
                            break
                if is_dupe:
                    continue

            # Intra-batch fuzzy dedup
            if brand_key != 'unknown':
                for prev_words in intra_batch_by_brand.get(brand_key, ()):
                    if len(p_words & prev_words) >= 2:
                        break
                else:
                    unique.append(p)
                    recent_keys.add(key)
                    intra_batch_keys.add(key)
                    intra_batch_by_brand.setdefault(brand_key, []).append(p_words)
                continue

            unique.append(p)
            recent_keys.add(key)
            intra_batch_keys.add(key)

        return unique

    async def summarize_raw(self, texts: Sequence[str]) -> str:
        """Summarizes a set of raw chat messages."""
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
        """Summarizes a specific conversation thread."""
        if not replies:
            return "Thread ini sedang ramai dibicarakan."
        reply_context = "\n- ".join(replies[:20])
        prompt = (
            f"PESAN UTAMA (Thread Starter): {parent_text}\n\n"
            f"BEBERAPA BALASAN DARI USER LAIN:\n- {reply_context}\n\n"
            "TUGASMU: Rangkum diskusi ini dalam 1-2 kalimat informatif."
        )
        contents: list[Any] = [prompt]
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
        """Answers a specific user inquiry based on provided context."""
        target = await self._pick_model()
        response = await self._call(
            contents=f"Pertanyaan: {question}\n\nKonteks:\n{context}",
            config={"system_instruction": _DIGEST_SYSTEM},
            model_id=target,
        )
        return cast(str, response.text) if response else "❌ AI Busy."

    async def process_image(self, image_bytes: bytes, caption: str | None,
                             original_msg_id: int) -> PromoExtraction | None:
        """Processes an image to extract promotional info."""
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

    async def generate_narrative(self, messages: Sequence[dict[str, Any]],
                                  db: Any = None) -> list[TrendItem]:
        """Generates structured trend narratives for recent traffic."""
        if not messages:
            return []

        # Enrich with reply-parent text so the model can weight thread context.
        parent_map: dict[int, str] = {}
        if db is not None:
            try:
                chat_id = messages[0]['chat_id']
                reply_ids = [m['reply_to_msg_id'] for m in messages
                             if m['reply_to_msg_id']]
                if chat_id is not None and reply_ids:
                    parent_map = await db.get_deep_context_bulk(
                        reply_ids, chat_id, max_depth=2
                    )
            except Exception as e:
                logger.warning(f"generate_narrative: reply enrichment failed: {e}")

        lines: list[str] = []
        for m in messages[:50]:
            ctx = ""
            rid = m['reply_to_msg_id']
            if rid and rid in parent_map:
                parent_txt = (parent_map[rid] or "")[-120:].replace("\n", " ")
                if parent_txt:
                    ctx = f" [reply→ {parent_txt}]"
            lines.append(f"ID:{m['tg_msg_id']} {m['sender_name']}:{ctx} {m['text']}")
        context = "\n- ".join(lines)
        target  = await self._pick_model()
        
        slang_desc = "\n".join([f"- `{k}` = {v}" for k, v in _SLANG_KAMUS.items()])
        config = {
            "response_mime_type": "application/json",
            "response_schema": TrendResponse,
            "system_instruction": (
                "Kamu analis sentimen deal-hunter Indonesia. Simpulkan 1-3 tren utama dengan link ID pesan.\n"
                "Gunakan konteks kamus slang berikut agar tidak salah paham:\n"
                f"{slang_desc}"
            ),
        }
        response = await self._call(contents=f"Pesan grup:\n{context}", config=config, model_id=target)
        
        # Cross-trend dedup
        seen_topics: list[set[str]] = []
        unique_trends: list[TrendItem] = []
        if response and response.parsed:
            for t in response.parsed.trends:
                words = set(re.findall(r'\w+', t.topic.lower()))
                if any(len(words & s) >= 3 for s in seen_topics):
                    continue
                unique_trends.append(t)
                seen_topics.append(words)
        
        return unique_trends

    async def interpret_keywords(self, hot_words: Sequence[str], window: int,
                                  context_msgs: Sequence[str]) -> str | None:
        """Interprets the context behind a burst of specific keywords."""
        if not context_msgs:
            return None
            
        word_counts = {}
        for w in hot_words:
            word_counts[w] = sum(1 for msg in context_msgs if w.lower() in msg.lower())
        
        counts_str = ", ".join([f"'{w}' ({c}x)" for w, c in word_counts.items()])
        
        slang_desc = "\n".join([f"- `{k}` = {v}" for k, v in _SLANG_KAMUS.items()])
        system = (
            "Kamu analis sentimen real-time. Ada lonjakan aktivitas di grup.\n"
            f"Kata kunci dominan dlm {window} menit terakhir: {counts_str}.\n"
            "Gunakan konteks kamus slang berikut agar tidak salah paham:\n"
            f"{slang_desc}\n\n"
            "TUGASMU: Jelaskan APA yang sedang dibahas berdasarkan pesan-pesan berikut.\n"
            "JANGAN menebak jika tidak ada informasi. Jawab dalam 1-2 kalimat padat."
        )
        
        target = await self._pick_model()
        context_block = "\n".join([f"- {msg[:150]}" for msg in context_msgs[-40:]])
        
        response = await self._call(
            contents=f"Pesan context:\n{context_block}",
            config={"system_instruction": system},
            model_id=target
        )
        return cast(str, response.text.strip()) if response and response.text and "NO_TREND" not in response.text else None
