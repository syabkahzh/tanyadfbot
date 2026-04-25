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
    brand: Optional[str] = "Unknown"
    conditions: Optional[str] = ""
    valid_until: Optional[str] = ""
    status: Literal["active", "expired", "unknown"] = "unknown"
    confidence: float = 1.0 # 0.0 to 1.0 based on AI certainty
    links: List[str] = []
    detected_at: Optional[str] = None
    queue_time: Optional[float] = None
    ai_time: Optional[float] = None
    model_name: Optional[str] = None

class BatchResponse(BaseModel):
    """Wrapper for batch AI extraction results."""
    promos: List[PromoExtraction]

class TrendItem(BaseModel):
    """A single identified trend or topic from recent discussions."""
    topic: str
    msg_id: int
    model_name: Optional[str] = None

class TrendResponse(BaseModel):
    """Wrapper for aggregate trend analysis results."""
    trends: List[TrendItem]


# ── Prompt constants ──────────────────────────────────────────────────────────

_EXTRACT_SYSTEM = """Kamu sistem ekstraksi data promo dari grup deal-hunter Indonesia (Discountfess).
Fokus utama: deteksi apakah pesan menginformasikan keberhasilan/kegagalan promo, atau membagikan promo baru.

INPUT FORMAT:
ID:<id> C:<konteks pesan yang dibalas jika ada> MSG:<pesan utama user>

ATURAN KONTEKS (C: vs MSG:):
- `MSG:` adalah pesan dari user saat ini. Ini adalah penentu utama.
- `C:` adalah pesan sebelumnya yang dibalas oleh user.
- Gunakan `C:` HANYA untuk mencari tahu brand apa yang sedang dibahas jika `MSG:` berupa konfirmasi pendek, misalnya `MSG: aman kak` dan `C: sfood 50%`.
- JANGAN mengekstrak promo dari `C:` jika `MSG:` hanyalah pertanyaan murni, misalnya `MSG: ini gimana caranya?` dengan `C: promo gopay`. Output wajib = SKIP.
- Jika `MSG:` berisi validasi seperti `jp`, `nyala`, atau `udah habis`, maka ekstrak promo dengan memakai brand/info dari `C:`.

ISTILAH KUNCI SLANG:
- ACTIVE: aman, on, jp, work, nyala, cair, masuk, luber, pecah, nyantol, dapet.
- EXPIRED: abis, habis, nt, sold out, zonk, gabisa, limit, koid.
- BRAND: sfood/spud=ShopeeFood, gfood=GoFood, tukpo/toped=Tokopedia, sopi=Shopee, tsel=Telkomsel, idm=Indomaret, afm/jsm/psm=Alfamart.

ATURAN EKSTRAKSI (WAJIB DIIKUTI):
1. Brand harus konsisten. Metode bayar (ShopeePay, OVO, DANA) = `conditions`, bukan brand, kecuali promo aplikasi murni tanpa merchant spesifik.
2. Status: `active` (berhasil/ada), `expired` (habis/gagal), `unknown` (ambigu).
3. Summary: 1 kalimat padat, misalnya `ShopeeFood diskon 50k aktif.` Jangan berisi meta-deskripsi seperti `User menanyakan promo`.
4. Jika bukan promo, wajib isi `brand="SKIP"` dan `summary="SKIP"`.

WAJIB SKIP JIKA `MSG:` ADALAH:
- Pertanyaan murni, misalnya `masih bisa?` atau `caranya gimana?`
- OOT / curhat, misalnya `lagi di jalan`, `wkwk`, `iya makasih`
- Tidak jelas membahas promo apa sama sekali.
"""

_DEDUP_SYSTEM = "Kamu agen deteksi duplikasi. Output HANYA angka indeks dipisah koma."

_DIGEST_SYSTEM = "Kamu asisten ringkasan promo Indonesia. Jawab singkat dan informatif dalam bahasa Indonesia santai."

_VISION_SYSTEM = """Kamu analis visual grup promo Indonesia (Discountfess).

TUGASMU: Ekstrak detail promo hanya dari gambar yang diberikan.

ATURAN CAPTION & GAMBAR:
- Jika caption berupa pertanyaan, misalnya `ini promo bukan?` atau `cara makenya gimana?`, abaikan caption tersebut. Jangan langsung di-skip, lihat isi gambarnya.
- Jika gambar berisi poster diskon, voucher aktif, atau struk keberhasilan, ekstrak data tersebut meskipun captionnya bertanya.
- Jika gambar berupa meme, foto pribadi, atau screenshot OOT, barulah output SKIP.

PROMO VALID - ekstrak jika gambar berisi:
- Poster/banner promo brand (diskon, cashback, voucher, harga spesial)
- Screenshot aplikasi yang menampilkan harga/voucher/deal aktif
- Bukti transaksi (struk, order confirmation)

TOLAK (isi summary="SKIP", brand="SKIP") jika gambar adalah:
- Screenshot settings/UI aplikasi tanpa info promo
- Foto produk fisik tanpa harga promo, misalnya foto kopi biasa
- Screenshot chat/grup tanpa bukti promo konkret

ATURAN OUTPUT:
- Jika SKIP: {"summary": "SKIP", "brand": "SKIP", "conditions": "", "valid_until": "", "status": "unknown", "original_msg_id": 0, "confidence": 0.0}
- Jika valid: summary 1 kalimat padat, berisi brand + diskon/harga + syarat.
- Confidence: 0.9+ untuk poster resmi/struk jelas, <0.7 untuk gambar samar.
"""



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


# ── AI Clients ───────────────────────────────────────────────────────────────

class BaseAIClient:
    """Abstract base class for all AI providers."""
    async def generate_content(self, model: str, contents: Any, config: dict[str, Any]) -> Any:
        raise NotImplementedError

class WrappedResponse:
    """Compatibility wrapper for AI responses."""
    def __init__(self, res=None, text=None, parsed=None, model_name=None):
        self.res = res
        self._text = text
        self._parsed = parsed
        self.model_name = model_name
    @property
    def text(self):
        return self._text if self._text is not None else getattr(self.res, 'text', "")
    @property
    def parsed(self):
        return self._parsed if self._parsed is not None else getattr(self.res, 'parsed', None)

class GoogleClient(BaseAIClient):
    """Client for Google GenAI models."""
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    async def generate_content(self, model: str, contents: Any, config: dict[str, Any]) -> Any:
        # Some models (like gemma-3/4) might not support system_instruction or JSON mode as separate fields
        is_gemma = "gemma-" in model
        if is_gemma:
            system = config.pop("system_instruction", None)
            if system:
                sys_text = system
                if hasattr(system, 'parts'): sys_text = system.parts[0].text
                elif isinstance(system, dict) and 'parts' in system: sys_text = system['parts'][0].get('text', '')
                
                if isinstance(contents, str): contents = f"INSTRUCTION: {sys_text}\n\n{contents}"
                elif isinstance(contents, list): contents.insert(0, f"INSTRUCTION: {sys_text}")

            if config.get("response_mime_type") == "application/json":
                config.pop("response_mime_type")
                config.pop("response_schema", None)
                json_msg = "OUTPUT MUST BE VALID JSON."
                if isinstance(contents, str): contents += f"\n\n{json_msg}"
                elif isinstance(contents, list): contents.append(json_msg)

        # Safety settings can be problematic on some models, omitting to use defaults
        # if you need to override them, use the correct HARM_CATEGORY_ prefix.

        res = await self.client.aio.models.generate_content(
            model=model, contents=contents, config=config
        )
        return WrappedResponse(res)

class OpenAICompatibleClient(BaseAIClient):
    """Generic client for OpenAI-compatible providers like Groq and GLM."""
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        from openai import AsyncOpenAI
        if not api_key:
            logger.error(f"OpenAICompatibleClient initialized with EMPTY API KEY for {base_url}")
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def generate_content(self, model: str, contents: Any, config: dict[str, Any]) -> Any:
        # Map Google-style config to OpenAI-style
        system = config.get("system_instruction", "")
        response_schema = config.get("response_schema")
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        
        if isinstance(contents, str):
            messages.append({"role": "user", "content": contents})
        elif isinstance(contents, list):
            for item in contents:
                if isinstance(item, str):
                    messages.append({"role": "user", "content": item})
        
        kwargs = {}
        if response_schema:
            kwargs["response_format"] = {"type": "json_object"}
            schema_str = response_schema.model_json_schema()
            messages[0]["content"] += (
                f"\n\nYou MUST respond with ONLY valid JSON. No explanation, no markdown, no preamble."
                f"\nThe JSON must match this exact schema:\n{schema_str}"
                f"\nFor promo extraction: if not a promo, set summary='SKIP' and brand='SKIP'."
                f"\nNever write descriptions of messages. Extract or SKIP."
            )

        res = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        
        text = res.choices[0].message.content
        parsed = None
        if response_schema and text:
            try:
                clean_text = re.sub(r'^```json\s*|\s*```$', '', text.strip(), flags=re.MULTILINE)
                import json
                try:
                    # Attempt standard validation
                    parsed = response_schema.model_validate_json(clean_text)
                except Exception:
                    # Fallback: manually parse and clean
                    try:
                        raw = json.loads(clean_text)
                        # Case 1: Model returned a single object or list instead of {"promos": [...]}
                        if response_schema.__name__ == "BatchResponse":
                            if isinstance(raw, dict) and "promos" not in raw:
                                if "summary" in raw:
                                    raw = {"promos": [raw]}
                            elif isinstance(raw, list):
                                raw = {"promos": raw}
                        
                        # Case 2: Deeply clean nulls into empty strings for required fields
                        def clean_nulls(obj):
                            if isinstance(obj, list): return [clean_nulls(x) for x in obj]
                            if isinstance(obj, dict):
                                for k, v in obj.items():
                                    if v is None:
                                        if k in ('queue_time', 'ai_time', 'confidence', 'original_msg_id'):
                                            pass
                                        else:
                                            obj[k] = ""
                                    else: obj[k] = clean_nulls(v)
                            return obj
                        
                        raw = clean_nulls(raw)
                        parsed = response_schema.model_validate(raw)
                    except Exception as e2:
                        logger.warning(f"OpenAI client deep parse failed: {e2}")
                        raise
            except Exception as e:
                logger.warning(f"OpenAI client failed to parse JSON: {e}")
                logger.debug(f"Raw text that failed: {text}")
        
        return WrappedResponse(res, text=text, parsed=parsed)

class OllamaClient(BaseAIClient):
    """Client for Ollama models via ollamafreeapi."""
    def __init__(self):
        try:
            from ollamafreeapi import Ollama
            self.client = Ollama()
        except ImportError:
            self.client = None

    async def generate_content(self, model: str, contents: Any, config: dict[str, Any]) -> Any:
        if not self.client: return None
        text_content = str(contents)
        response = self.client.chat(model=model, messages=[{'role': 'user', 'content': text_content}])
        return WrappedResponse(text=response['message']['content'])

# ── Rate limiter ──────────────────────────────────────────────────────────────

class _ModelSlot:
    """Sliding-window RPM + daily RPD limiter with provider awareness and TPM/TPD support."""

    def __init__(self, name: str, provider: str, model_id: str, client: BaseAIClient, 
                 limit: int, daily_limit: int = 0, 
                 tpm_limit: int = 0, tpd_limit: int = 0,
                 priority: int = 3) -> None:
        self.name = name
        self.provider = provider
        self.model_id = model_id
        self.client = client
        self.limit = limit
        self.daily_limit = daily_limit
        self.tpm_limit = tpm_limit
        self.tpd_limit = tpd_limit
        self.priority = priority
        
        self._calls: list[float] = []
        self._daily_calls: list[float] = []
        
        # Tracking tokens (timestamp, tokens)
        self._tokens: list[tuple[float, int]] = []
        self._daily_tokens: list[tuple[float, int]] = []
        
        self._lock = asyncio.Lock()

    def _cleanup(self, now: float) -> None:
        """Remove expired timestamps. Must be called under self._lock."""
        self._calls = [t for t in self._calls if now - t < 60]
        self._tokens = [t for t in self._tokens if now - t[0] < 60]
        
        if self.daily_limit > 0:
            self._daily_calls = [t for t in self._daily_calls if now - t < 86400]
        if self.tpd_limit > 0:
            self._daily_tokens = [t for t in self._daily_tokens if now - t[0] < 86400]

    def available(self, now: float) -> int:
        """Current available RPM slots. Approximate — does not lock."""
        cutoff = now - 60
        active = sum(1 for t in self._calls if t > cutoff)
        return max(0, self.limit - active)

    async def try_acquire_nowait(self, estimated_tokens: int = 0) -> bool:
        """Non-blocking attempt. Returns True and records the call if a slot is free."""
        now = time.monotonic()
        async with self._lock:
            self._cleanup(now)
            
            # Check Call Limits
            if self.daily_limit > 0 and len(self._daily_calls) >= self.daily_limit:
                return False
            if len(self._calls) >= self.limit:
                return False
                
            # Check Token Limits
            if estimated_tokens > 0:
                if self.tpm_limit > 0:
                    current_tpm = sum(t[1] for t in self._tokens)
                    if current_tpm + estimated_tokens > self.tpm_limit:
                        return False
                if self.tpd_limit > 0:
                    current_tpd = sum(t[1] for t in self._daily_tokens)
                    if current_tpd + estimated_tokens > self.tpd_limit:
                        return False

            # Record Usage
            self._calls.append(now)
            self._tokens.append((now, estimated_tokens))
            if self.daily_limit > 0:
                self._daily_calls.append(now)
            if self.tpd_limit > 0:
                self._daily_tokens.append((now, estimated_tokens))
                
            return True

    async def acquire(self, estimated_tokens: int = 0, timeout: float = 90.0) -> bool:
        """Blocking acquire. Returns True if a slot was obtained before timeout."""
        deadline = time.monotonic() + timeout
        while True:
            if await self.try_acquire_nowait(estimated_tokens):
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
        if self._tokens:
            self._tokens.pop()
        if self.daily_limit > 0 and self._daily_calls:
            self._daily_calls.pop()
        if self.tpd_limit > 0 and self._daily_tokens:
            self._daily_tokens.pop()

    def current_usage(self) -> int:
        now = time.monotonic()
        return sum(1 for t in self._calls if now - t < 60)

    def daily_usage(self) -> int:
        now = time.monotonic()
        return sum(1 for t in self._daily_calls if now - t < 86400)


# ─────────────────────────────────────────────────────────────────────────────

class GeminiProcessor:
    """Orchestrates AI analysis using the AI Army (multiple free providers)."""

    

    def __init__(self) -> None:
        """Initializes the AI Army fleet from configuration."""
        self.reinitialize()

    def reinitialize(self) -> None:
        """Reloads the fleet configuration and rebuilds model slots."""
        self._slots = {}
        army = Config.get_ai_army()
        for p in army:
            client = None
            api_key = p.get('api_key')
            
            if p['provider'] != 'ollama' and not api_key:
                logger.warning(f"Skipping unit {p['name']}: {p['api_key_env']} is not set.")
                continue

            if p['provider'] == 'google':
                client = GoogleClient(api_key=api_key)
            elif p['provider'] in ('groq', 'glm'):
                client = OpenAICompatibleClient(
                    api_key=api_key, 
                    base_url=p.get('base_url')
                )
            elif p['provider'] == 'ollama':
                client = OllamaClient()
            
            if client:
                slot = _ModelSlot(
                    name=p['name'],
                    provider=p['provider'],
                    model_id=p['model_id'],
                    client=client,
                    limit=p['rpm'],
                    daily_limit=p.get('rpd', 0),
                    tpm_limit=p.get('tpm', 0),
                    tpd_limit=p.get('tpd', 0),
                    priority=p.get('priority', 3)
                )
                self._slots[p['name']] = slot

        self._priority_list = [
            s.name for s in sorted(self._slots.values(), key=lambda x: x.priority)
        ]
        logger.info(f"AI Army (re)initialized with {len(self._slots)} units. Priority: {self._priority_list}")

    def update_model_priority(self, name: str, priority: int) -> bool:
        """Updates a model's priority in models_config.json and reloads the fleet."""
        import json
        import os
        path = os.path.join(os.path.dirname(__file__), "models_config.json")
        try:
            with open(path, "r") as f:
                army = json.load(f)
            
            found = False
            for p in army:
                if p['name'] == name:
                    p['priority'] = priority
                    found = True
                    break
            
            if not found: return False
            
            with open(path, "w") as f:
                json.dump(army, f, indent=4)
            
            self.reinitialize()
            return True
        except Exception as e:
            logger.error(f"Failed to update priority for {name}: {e}")
            return False

    async def _pick_model(self, exclude: Optional[str | list[str]] = None, provider: Optional[str] = None, estimated_tokens: int = 0) -> str:
        """Picks a model using Least-Utilized Load Balancing to utilize the whole AI Army simultaneously."""
        excludes = [exclude] if isinstance(exclude, str) else (exclude or [])
        
        valid_candidates = [n for n in self._priority_list if n not in excludes]
        if provider:
            valid_candidates = [n for n in valid_candidates if self._slots[n].provider == provider]
            
        if not valid_candidates:
            # If everything was excluded, reset to all valid provider models
            valid_candidates = [n for n in self._priority_list if not provider or self._slots[n].provider == provider]

        def get_utilization(name: str) -> float:
            slot = self._slots[name]
            # Calculate load percentage (e.g., 5/15 RPM = 0.33)
            return slot.current_usage() / max(1, slot.limit)

        # 1. Sort by lowest utilization %, then highest priority
        candidates_by_load = sorted(valid_candidates, key=lambda n: (get_utilization(n), self._slots[n].priority))

        # 2. Distribute load: try to acquire from the least loaded API first
        for name in candidates_by_load:
            if await self._slots[name].try_acquire_nowait(estimated_tokens):
                return name
        
        # 3. If ALL models are fully saturated, poll until ANY model frees up.
        # (The previous code hard-blocked on valid_candidates[0] and ignored the rest)
        timeout = 90.0
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            # Re-sort to grab whoever freed up first and has the lowest relative load
            candidates_by_load = sorted(valid_candidates, key=lambda n: (get_utilization(n), self._slots[n].priority))
            for name in candidates_by_load:
                if await self._slots[name].try_acquire_nowait(estimated_tokens):
                    return name
            await asyncio.sleep(1.0)
            
        # 4. Fallback if timeout expires
        if not valid_candidates:
            # Fallback to the absolute highest priority model if everything is exhausted
            best_name = self._priority_list[0]
        else:
            best_name = valid_candidates[0]
        await self._slots[best_name].acquire(estimated_tokens, timeout=0.1)
        return best_name

    def _estimate_tokens(self, content: Any) -> int:
        """Rough token estimation for rate limiting purposes.
        
        Uses simple heuristic: ~4 chars ≈ 1 token (standard GPT approximation).
        """
        # CRITICAL FIX: Account for the ~600 token overhead of the massive system prompt
        base_overhead = 600

        if isinstance(content, str):
            return max(1, (len(content) // 4) + base_overhead)
        elif isinstance(content, list):
            total = base_overhead
            for item in content:
                if isinstance(item, str):
                    total += len(item) // 4
                elif hasattr(item, '__str__'):
                    total += len(str(item)) // 4
            return max(1, total)
        else:
            return max(1, (len(str(content)) // 4) + base_overhead)


    async def _call(self, contents: Any, config: dict, slot_name: str, 
                    attempt: int = 1, max_attempts: int = 3, tried: list[str] = None) -> Optional[WrappedResponse]:
        """Executes an AI call with automatic cross-provider fallback."""
        if tried is None: tried = []
        tried.append(slot_name)
        
        slot = self._slots[slot_name]
        try:
            logger.info(f"🤖 [AI] Requesting {slot.model_id} on {slot.provider} (attempt {attempt})...")
            
            # CRITICAL FIX: Cut timeout to 18s to prevent internal SDK retries from hanging the fleet.
            response = await asyncio.wait_for(
                slot.client.generate_content(
                    model=slot.model_id,
                    contents=contents,
                    config=config.copy()
                ),
                timeout=18.0
            )
            
            if response is None:
                raise Exception("Provider returned empty response")
                
            # CRITICAL FIX: Attach the name of the slot that actually succeeded
            response.model_name = slot_name 
            return response

        except Exception as e:
            logger.warning(f"AI ({slot.model_id}) failed on attempt {attempt}: {type(e).__name__}: {repr(e)}")
            slot.release_last() # Don't count failed calls against RPM

            if attempt < max_attempts:
                # Determine if vision is needed
                is_vision = False
                if isinstance(contents, list):
                    is_vision = any(hasattr(item, 'data') or (isinstance(item, dict) and 'image' in str(item).lower()) for item in contents)
                
                provider_filter = "google" if is_vision else None
                
                # CRITICAL FIX: Provider-Level Banning
                # If the server timed out or threw a 50x error, ban the ENTIRE provider for the retry chain
                exclude_list = list(tried)
                err_str = str(e).lower()
                is_server_death = isinstance(e, (asyncio.TimeoutError, TimeoutError)) or "timeout" in err_str or "50" in err_str
                
                if is_server_death and not (is_vision and slot.provider == "google"):
                    for n, s in self._slots.items():
                        if s.provider == slot.provider and n not in exclude_list:
                            exclude_list.append(n)
                            
                next_slot = await self._pick_model(exclude=exclude_list, provider=provider_filter)
                return await self._call(contents, config, next_slot, attempt + 1, max_attempts, tried)
            
            logger.error(f"AI call failed after {attempt} attempts.")
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
                    m['context'] = f"C:{ctx_text[-150:]} "
                else:
                    m['context'] = ""
        else:
            for m in filtered:
                m['context'] = ""

        batch_text = "\n---\n".join(
            f"ID:{m['id']} {m['context']}MSG: {m['text'] or ''}"
            for m in filtered
        )
        config = {
            "response_mime_type": "application/json",
            "response_schema": BatchResponse,
            "system_instruction": _EXTRACT_SYSTEM,
        }

        

        tokens = self._estimate_tokens(batch_text)
        target_model = await self._pick_model(estimated_tokens=tokens)
        logger.debug(f"Using {target_model} for {len(filtered)} msgs")

        response = await self._call(
            contents=f"Batch pesan:\n\n{batch_text}",
            config=config,
            slot_name=target_model,
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
            # CRITICAL FIX: Use the actual model that succeeded, not the initial target
            p.model_name = response.model_name
            valid.append(p)

        logger.info(f"Extracted {len(valid)} promos from batch of {len(filtered)} msgs. (Model: {target_model})")
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
        tokens = self._estimate_tokens(context)
        target   = await self._pick_model(estimated_tokens=tokens)
        response = await self._call(
            contents=f"Rangkum pesan ini:\n\n{context}",
            config={"system_instruction": _DIGEST_SYSTEM},
            slot_name=target,
        )
        res = response.text if response else "❌ Gagal merangkum."
        return f"{res}\n\n— via {target}" if response else res

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

        tokens = self._estimate_tokens(contents)
        # Vision requirement only if parent_photo exists
        provider = "google" if parent_photo else None
        target = await self._pick_model(provider=provider, estimated_tokens=tokens)
        response = await self._call(
            contents=contents,
            config={"system_instruction": _DIGEST_SYSTEM},
            slot_name=target,
        )
        res = response.text if response else "Thread ini sedang ramai dibicarakan."
        return f"{res}\n\n— via {target}" if response else res

    async def answer_question(self, question: str, context: str) -> str:
        """Answers a specific user inquiry based on provided context."""
        tokens = self._estimate_tokens(question + context)
        target = await self._pick_model(estimated_tokens=tokens)
        response = await self._call(
            contents=f"Pertanyaan: {question}\n\nKonteks:\n{context}",
            config={"system_instruction": _DIGEST_SYSTEM},
            slot_name=target,
        )
        res = response.text if response else "❌ AI Busy."
        return f"{res}\n\n— via {target}" if response else res

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
        contents = [prompt, genai.types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")]
        tokens = self._estimate_tokens(contents)
        target = await self._pick_model(provider="google", estimated_tokens=tokens)
        response = await self._call(
            contents=contents,
            config=config,
            slot_name=target,
        )
        if not response or not response.parsed:
            return None
        res = response.parsed
        res.model_name = target
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
        tokens = self._estimate_tokens(context)
        target  = await self._pick_model(estimated_tokens=tokens)
        
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
        response = await self._call(contents=f"Pesan grup:\n{context}", config=config, slot_name=target)
        # Cross-trend dedup
        seen_topics: list[set[str]] = []
        unique_trends: list[TrendItem] = []
        if response and response.parsed:
            for t in response.parsed.trends:
                words = set(re.findall(r'\w+', t.topic.lower()))
                if any(len(words & s) >= 3 for s in seen_topics):
                    continue
                t.model_name = target
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
        
        context_block = "\n".join([f"- {msg[:150]}" for msg in context_msgs[-40:]])
        tokens = self._estimate_tokens(system + context_block)
        target = await self._pick_model(estimated_tokens=tokens)
        
        response = await self._call(
            contents=f"Pesan context:\n{context_block}",
            config={"system_instruction": system},
            slot_name=target
        )
        return cast(str, response.text.strip()) if response and response.text and "NO_TREND" not in response.text else None
