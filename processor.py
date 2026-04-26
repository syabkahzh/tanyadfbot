"""processor.py — Gemini AI Layer.

Advanced model orchestration with dual-primary load balancing, automated fallback 
mechanisms, and rate-limit aware token buckets.
"""

import re
import asyncio
import time
import logging
from typing import List, Literal, Optional, Any, Sequence, cast
from pydantic import BaseModel, Field

from google import genai
from config import Config
from db import normalize_brand

logger = logging.getLogger(__name__)

# ── Pre-compiled Patterns ─────────────────────────────────────────────────────
_WORD_BOUNDARY_KEYWORDS = re.compile(
    r'\b(off|on|aman|work|bs|jp|mm)\b', re.IGNORECASE
)
_SOCIAL_FILLER = re.compile(
    r'^((wkwk|haha|hehe|iya|noted|oke|ok|makasih|thanks|thx|mantap|gas|bos|guys|gais|bang|kak|siap|sip|lol|anjir|anjay|btw|oot|gws|semangat|ya allah|nangis|sedih|beneran|kah)[!.\s]*)+$',
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
    original_msg_id: int = 0
    summary: str = Field(default="", description="1 kalimat ringkasan WAJIB DALAM BAHASA INDONESIA. Isi 'SKIP' jika bukan promo.")
    brand: str = Field(default="", description="Nama brand yang tepat, atau 'SKIP'.")
    conditions: str = Field(default="", description="Syarat dan ketentuan DALAM BAHASA INDONESIA, atau string kosong.")
    valid_until: Optional[str] = ""
    # CRITICAL FIX: Forcing strict enums mathematically prevents hallucinated statuses
    status: Literal['active', 'expired', 'unknown'] = Field(default='unknown', description="Strictly select one.")
    confidence: float = Field(default=1.0, description="Confidence score from 0.0 to 1.0.")
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

_EXTRACT_SYSTEM = """Kamu adalah TanyaDFBot, sistem ekstraksi intelijen promo untuk grup Discountfess.
DILARANG KERAS MENGGUNAKAN TABEL ATAU MARKDOWN SELAIN JSON.

INSTRUKSI ANALISIS (LAKUKAN DALAM URUTAN INI):
1. IDENTIFIKASI SUBJEK: Periksa C: (Konteks) dan MSG: (Pesan). Apakah mereka membahas spesifik suatu brand/toko/layanan? Jika obrolan oot/random/tanya jawab biasa, set brand="SKIP".
2. EVALUASI STATUS: Gunakan terminologi slang ini:
   - ACTIVE: "jp", "aman", "on", "work", "nyala", "makasih", "dapet", "alhamdulillah", "nyantol".
   - EXPIRED: "abis", "nt", "sold", "gabisa", "limit", "koid", "gaib", "goib", "badut", "zonk".
   - QUESTION: Jika hanya bertanya (misal: "aman ga?", "ada yg tau?").
3. RINGKASAN: Buat 1 kalimat padat berbahasa Indonesia. Gunakan Bold (**) pada nama brand.

CONTOH:
Input: ID:1 C:sfood diskon 50k MSG:nyala bang
Output: {"promos": [{"original_msg_id": 1, "brand": "ShopeeFood", "summary": "ShopeeFood diskon 50k aktif.", "status": "active", "confidence": 0.95}]}

Input: ID:2 C:gopay coins 100% MSG:nt
Output: {"promos": [{"original_msg_id": 2, "brand": "GoPay", "summary": "Promo GoPay Coins 100% sudah habis.", "status": "expired", "confidence": 0.90}]}

ATURAN WAJIB:
- Jika MSG hanya kata pendek (e.g., "aman"), WAJIB baca C: untuk mengetahui brand yang dimaksud.
- Output HARUS valid JSON yang sesuai dengan skema. Tidak ada teks pembuka/penutup.
- Brand harus konsisten (e.g. sfood -> ShopeeFood, tsel -> Telkomsel).
"""

_DEDUP_SYSTEM = "Kamu agen deteksi duplikasi. Output HANYA angka indeks dipisah koma."

_DIGEST_SYSTEM = """Kamu adalah TanyaDFBot, admin paling gercep di grup Discountfess. 
Tugasmu merangkum promo terbaru dengan gaya bahasa santai, informatif, dan pake slang dikit (gais, mantul, sikat). 

ATURAN:
- DILARANG KERAS PAKAI TABEL. 
- Gunakan Bullet points dan Bold untuk nama Brand.
- Jawab singkat padat, jangan bertele-tele.
"""

_VISION_SYSTEM = """Kamu analis visual TanyaDFBot untuk grup Discountfess. 

TUGASMU: Ekstrak detail promo dari gambar/screenshot.

ATURAN VERIFIKASI:
1. Jika gambar adalah STRUK PEMBAYARAN atau BUKTI SUKSES (ada nominal terbayar, centang hijau, atau "Pesanan Selesai") -> set status='active' dan confidence=1.0.
2. Jika gambar adalah POSTER PROMO atau BANNER -> set status='active' dan confidence=0.85.
3. Jika gambar adalah meme, foto oot, atau UI settings -> set summary='SKIP', brand='SKIP'.

DILARANG KERAS MENGGUNAKAN TABEL. Gunakan bold untuk brand.
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
    async def generate_content(self, model: str, contents: Any, config: dict[str, Any], capabilities: dict[str, Any]) -> Any:
        raise NotImplementedError

class WrappedResponse:
    """Compatibility wrapper for AI responses."""
    def __init__(self, res=None, text=None, parsed=None, model_name=None, usage=None, headers=None):
        self.res = res
        self._text = text
        self._parsed = parsed
        self.model_name = model_name
        self.usage = usage  # Actual tokens (prompt, completion, total)
        self.headers = headers or {} # Rate limit headers

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

    async def generate_content(self, model: str, contents: Any, config: dict[str, Any], capabilities: dict[str, Any]) -> Any:
        config = config.copy()
        
        # Adaptive JSON handling based on capabilities
        if not capabilities.get("native_json", True):
            schema = config.pop("response_schema", None)
            config.pop("response_mime_type", None)
            if schema:
                schema_text = f"\n\nOUTPUT MUST BE VALID JSON matching this schema:\n{schema.model_json_schema()}"
                if isinstance(contents, str): contents += schema_text
                elif isinstance(contents, list):
                    if isinstance(contents[0], str): contents[0] += schema_text
                    else: contents.append(schema_text)

        # Adaptive system instruction handling
        if not capabilities.get("system_instruction", True):
            system = config.pop("system_instruction", None)
            if system:
                sys_text = system
                if hasattr(system, 'parts'): sys_text = system.parts[0].text
                if isinstance(contents, str): contents = f"SYSTEM: {sys_text}\n\n{contents}"
                elif isinstance(contents, list): 
                    contents.insert(0, f"SYSTEM: {sys_text}")

        # Ensure contents is converted to genai Parts if it contains dicts
        if isinstance(contents, list):
            final_contents = []
            for item in contents:
                if isinstance(item, dict) and "data" in item:
                    final_contents.append(genai.types.Part.from_bytes(data=item["data"], mime_type=item["mime_type"]))
                else:
                    final_contents.append(item)
            contents = final_contents

        # Explicitly disable Automatic Function Calling (AFC)
        config["automatic_function_calling"] = {"disable": True}

        res = await self.client.aio.models.generate_content(
            model=model, contents=contents, config=config
        )
        usage = {
            "prompt_tokens": getattr(res.usage_metadata, 'prompt_token_count', 0),
            "completion_tokens": getattr(res.usage_metadata, 'candidates_token_count', 0),
            "total_tokens": getattr(res.usage_metadata, 'total_token_count', 0),
        }
        return WrappedResponse(res, usage=usage)

class OpenAICompatibleClient(BaseAIClient):
    """Generic client for OpenAI-compatible providers."""
    def __init__(self, api_key: str, base_url: Optional[str] = None, provider: str = "openai"):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.provider = provider

    async def generate_content(self, model: str, contents: Any, config: dict[str, Any], capabilities: dict[str, Any]) -> Any:
        system = config.get("system_instruction", "")
        response_schema = config.get("response_schema")
        
        messages = []
        if system: messages.append({"role": "system", "content": system})
        
        if isinstance(contents, str): messages.append({"role": "user", "content": contents})
        elif isinstance(contents, list):
            user_content = []
            for item in contents:
                if isinstance(item, str):
                    user_content.append({"type": "text", "text": item})
                elif isinstance(item, dict) and "data" in item:
                    # Image handling (base64)
                    import base64
                    img_b64 = base64.b64encode(item["data"]).decode("utf-8")
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{item['mime_type']};base64,{img_b64}"}
                    })
            
            if user_content:
                messages.append({"role": "user", "content": user_content})
            else:
                # Fallback for empty or non-standard list
                messages.append({"role": "user", "content": str(contents)})
        
        kwargs = {}
        # Adaptive reasoning format handling (e.g., Groq's hidden reasoning)
        reasoning_mode = capabilities.get("reasoning")
        if reasoning_mode:
            kwargs["extra_body"] = {"reasoning_format": reasoning_mode}

        if response_schema:
            if capabilities.get("native_tools", False):
                # Native Tool Calling (Function Calling)
                schema_dict = response_schema.model_json_schema()
                tool_name = "extract_data"
                kwargs["tools"] = [{
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": f"Extract structured {response_schema.__name__} data",
                        "parameters": schema_dict
                    }
                }]
                # Use 'auto' instead of forced function to be more resilient to models 
                # that output JSON directly but skip the tool envelope.
                kwargs["tool_choice"] = "auto" 
            elif capabilities.get("native_json", True):
                # Fallback to JSON object mode
                kwargs["response_format"] = {"type": "json_object"}
                schema_str = response_schema.model_json_schema()
                messages[0]["content"] += (
                    f"\n\nYou MUST respond with ONLY valid JSON. No preamble.\nSchema:\n{schema_str}"
                )

        try:
            # Use with_raw_response to access rate limit headers
            raw_res = await self.client.chat.completions.with_raw_response.create(model=model, messages=messages, **kwargs)
            res = raw_res.parse()
            headers = {k.lower(): v for k, v in raw_res.headers.items() if k.lower().startswith('x-ratelimit')}
        except Exception as e:
            # GROQ RESCUE: Handle 'Tool choice is required, but model did not call a tool'
            # Some models (like GPT-OSS or Qwen) might generate perfect JSON but skip the tool envelope.
            err_str = str(e).lower()
            if ("tool_use_failed" in err_str or "tool choice" in err_str) and hasattr(e, 'response'):
                try:
                    body = e.response.json()
                    text = body.get('error', {}).get('failed_generation')
                    if text:
                        logger.info(f"💡 [AI] Rescued content from failed tool call for {model}")
                        headers = {k.lower(): v for k, v in e.response.headers.items() if k.lower().startswith('x-ratelimit')}
                        return WrappedResponse(text=text, model_name=model, headers=headers)
                except: pass
            raise e

        message = res.choices[0].message
        
        text = None
        if message.tool_calls:
            text = message.tool_calls[0].function.arguments
        else:
            text = message.content

        # Robust thinking/reasoning removal for 2026 models (DeepSeek, o3, etc)
        if text: 
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            # Also handle some models that use markdown blocks for reasoning
            text = re.sub(r'--- reasoning ---.*?--- reasoning ---', '', text, flags=re.DOTALL | re.IGNORECASE).strip()

        usage = {
            "prompt_tokens": getattr(res.usage, 'prompt_tokens', 0),
            "completion_tokens": getattr(res.usage, 'completion_tokens', 0),
            "total_tokens": getattr(res.usage, 'total_tokens', 0),
        }

        parsed = None
        if response_schema and text:
            try:
                clean_text = re.sub(r'^```json\s*|\s*```$', '', text.strip(), flags=re.MULTILINE)
                import json
                try: parsed = response_schema.model_validate_json(clean_text)
                except:
                    raw = json.loads(clean_text)
                    if response_schema.__name__ == "BatchResponse":
                        if isinstance(raw, dict) and "promos" not in raw:
                            if "summary" in raw: raw = {"promos": [raw]}
                        elif isinstance(raw, list): raw = {"promos": raw}
                    
                    def clean_nulls(obj):
                        if isinstance(obj, list): return [clean_nulls(x) for x in obj]
                        if isinstance(obj, dict):
                            for k, v in list(obj.items()):
                                if v is None:
                                    if k in ('queue_time', 'ai_time'): pass
                                    elif k == 'confidence': obj[k] = 1.0
                                    elif k == 'original_msg_id': obj[k] = 0
                                    else: obj[k] = ""
                                else: obj[k] = clean_nulls(v)
                        return obj
                    parsed = response_schema.model_validate(clean_nulls(raw))
            except Exception as e:
                logger.warning(f"AI client failed to parse JSON: {e}")
        
        return WrappedResponse(res, text=text, parsed=parsed, usage=usage, headers=headers)

class OllamaClient(BaseAIClient):
    """Client for Ollama models."""
    def __init__(self):
        try:
            from ollamafreeapi import Ollama
            self.client = Ollama()
        except: self.client = None

    async def generate_content(self, model: str, contents: Any, config: dict[str, Any], capabilities: dict[str, Any]) -> Any:
        if not self.client: return None
        text_content = str(contents)
        response = self.client.chat(model=model, messages=[{'role': 'user', 'content': text_content}])
        return WrappedResponse(text=response['message']['content'])

# ── Rate limiter ──────────────────────────────────────────────────────────────

class _ModelSlot:
    """Sliding-window RPM + daily RPD limiter with capability awareness."""

    def __init__(self, name: str, provider: str, model_id: str, client: BaseAIClient, 
                 limit: int, daily_limit: int = 0, 
                 tpm_limit: int = 0, tpd_limit: int = 0,
                 priority: int = 3, capabilities: dict = None) -> None:
        self.name = name
        self.provider = provider
        self.model_id = model_id
        self.client = client
        self.limit = limit
        self.daily_limit = daily_limit
        self.tpm_limit = tpm_limit
        self.tpd_limit = tpd_limit
        self.priority = priority
        self.capabilities = capabilities or {}
        
        self._calls: list[float] = []
        self._daily_calls: list[float] = []
        
        # Tracking tokens (timestamp, tokens)
        self._tokens: list[tuple[float, int]] = []
        self._daily_tokens: list[tuple[float, int]] = []
        
        self._lock = asyncio.Lock()
        self.exhausted_until: float = 0.0

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
        if now < self.exhausted_until:
            return False
            
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

    def current_usage(self) -> int:
        now = time.monotonic()
        return sum(1 for t in self._calls if now - t < 60)

    def daily_usage(self) -> int:
        now = time.monotonic()
        return sum(1 for t in self._daily_calls if now - t < 86400)

    def release_last(self) -> None:
        """Remove the most recent call from all counters. Used when a call fails."""
        if self._calls:
            self._calls.pop()
        if self._tokens:
            self._tokens.pop()
        if self._daily_calls:
            self._daily_calls.pop()
        if self._daily_tokens:
            self._daily_tokens.pop()

    def update_actual_usage(self, estimated: int, actual: int) -> None:
        """Corrects the token bucket after a successful call with real usage data."""
        if estimated == actual or actual <= 0:
            return
            
        diff = actual - estimated
        # We only correct the tokens, not the call counts.
        # Find the entry and update it, or just add the diff to the current buckets.
        # Simplest way: append a 'correction' entry.
        now = time.monotonic()
        self._tokens.append((now, diff))
        if self.tpd_limit > 0:
            self._daily_tokens.append((now, diff))
        
        logger.debug(f"📊 [{self.name}] Token Correction: Estimated {estimated} -> Actual {actual} (Diff: {diff})")

    def saturate_locally(self, used: int, limit: int) -> None:
        """Artificially fills the local bucket based on authoritative API data."""
        if limit <= 0: return
        
        now = time.monotonic()
        current_local = sum(t[1] for t in self._daily_tokens)
        diff = used - current_local
        
        if diff > 0:
            self._daily_tokens.append((now, diff))
            logger.warning(f"🎯 [{self.name}] Syncing with API: Added {diff} tokens to local tracker. (API Used: {used}/{limit})")

    def sync_from_headers(self, headers: dict) -> None:
        """Synchronizes local tracking using HTTP rate limit headers."""
        if not headers:
            return
            
        now = time.monotonic()
        
        # 1. Sync Requests Per Day (RPD)
        limit_req = headers.get('x-ratelimit-limit-requests')
        rem_req   = headers.get('x-ratelimit-remaining-requests')
        if limit_req and rem_req:
            try:
                limit_req_v = int(limit_req)
                rem_req_v   = int(rem_req)
                used_req_v  = limit_req_v - rem_req_v
                
                # Check RPD bucket
                current_daily_calls = len([t for t in self._daily_calls if now - t < 86400])
                diff_req = used_req_v - current_daily_calls
                if diff_req > 0:
                    for _ in range(diff_req):
                        self._daily_calls.append(now - 1.0) # slightly in past
            except: pass

        # 2. Sync Tokens (Usually TPM in Groq, but we use it for both)
        limit_tokens = headers.get('x-ratelimit-limit-tokens')
        rem_tokens   = headers.get('x-ratelimit-remaining-tokens')
        if limit_tokens and rem_tokens:
            try:
                limit_tokens_v = int(limit_tokens)
                rem_tokens_v   = int(rem_tokens)
                used_tokens_v  = limit_tokens_v - rem_tokens_v
                
                # Sync TPM bucket
                current_tpm = sum(t[1] for t in self._tokens)
                diff_tpm = used_tokens_v - current_tpm
                if diff_tpm > 0:
                    self._tokens.append((now, diff_tpm))
            except: pass

        # 3. Sync Tokens Per Day (TPD) if header exists (some providers use this)
        limit_tpd = headers.get('x-ratelimit-limit-tokens-day')
        rem_tpd   = headers.get('x-ratelimit-remaining-tokens-day')
        if limit_tpd and rem_tpd:
            try:
                limit_tpd_v = int(limit_tpd)
                rem_tpd_v   = int(rem_tpd)
                used_tpd_v  = limit_tpd_v - rem_tpd_v
                self.saturate_locally(used_tpd_v, limit_tpd_v)
            except: pass


# ─────────────────────────────────────────────────────────────────────────────

class GeminiProcessor:
    """Orchestrates AI analysis using the AI Army (multiple free providers)."""

    _AI_CALL_TIMEOUT_SEC = 30.0

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
                logger.warning(f"Skipping unit {p['name']}: {p.get('api_key_env', 'API_KEY')} is not set.")
                continue

            if p['provider'] == 'google':
                client = GoogleClient(api_key=api_key)
            elif p['provider'] in ('groq', 'glm', 'openrouter', 'mistral', 'siliconflow', 'cerebras'):
                client = OpenAICompatibleClient(
                    api_key=api_key, 
                    base_url=p.get('base_url'),
                    provider=p['provider']
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
                    priority=p.get('priority', 3),
                    capabilities=p.get('capabilities', {})
                )
                self._slots[p['name']] = slot

        self._priority_list = [
            s.name for s in sorted(self._slots.values(), key=lambda x: x.priority)
        ]
        logger.info(f"AI Army (re)initialized with {len(self._slots)} units.")

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

    async def _pick_model(self, exclude: Optional[str | list[str]] = None, provider: Optional[str] = None, 
                    estimated_tokens: int = 0, require_vision: bool = False) -> str:
        """Picks a model using Multi-Dimensional Load Balancing (RPM, TPM, TPD)."""
        excludes = set([exclude] if isinstance(exclude, str) else (exclude or []))
        
        # 1. Start with all units not in excludes
        valid_candidates = [n for n in self._priority_list if n not in excludes]
        
        # 2. Filter by vision if required
        if require_vision:
            valid_candidates = [n for n in valid_candidates if self._slots[n].capabilities.get("vision")]

        # 3. Apply provider filter if requested
        if provider:
            p_candidates = [n for n in valid_candidates if self._slots[n].provider == provider]
            if p_candidates:
                valid_candidates = p_candidates
            
        if not valid_candidates:
            # If everything is excluded, pick the best one from the whole list 
            # as a last resort, but this should be rare.
            valid_candidates = [n for n in self._priority_list if n not in excludes]
            if not valid_candidates: 
                valid_candidates = self._priority_list

        def get_max_utilization(name: str) -> float:
            slot = self._slots[name]
            now = time.monotonic()
            
            # RPM Util
            rpm_active = sum(1 for t in slot._calls if now - t < 60)
            rpm_util = rpm_active / max(1, slot.limit)
            
            # TPM Util
            tpm_util = 0.0
            if slot.tpm_limit > 0:
                current_tpm = sum(t[1] for t in slot._tokens if now - t[0] < 60)
                tpm_util = current_tpm / slot.tpm_limit
                
            # TPD Util
            tpd_util = 0.0
            if slot.tpd_limit > 0:
                current_tpd = sum(t[1] for t in slot._daily_tokens if now - t[0] < 86400)
                tpd_util = current_tpd / slot.tpd_limit
                
            return max(rpm_util, tpm_util, tpd_util)

        # 1. Sort by:
        #    a) Priority (1 to 5) -> QUALITY FIRST. Use the best models until they are saturated.
        #    b) Max Utilization (0.0 to 1.0) -> Balance load within the same quality tier.
        #    c) -Daily Capacity (TPD) -> Tie-breaker: use model with more daily runway.
        candidates_by_load = sorted(valid_candidates, key=lambda n: (
            self._slots[n].priority,
            get_max_utilization(n), 
            -self._slots[n].tpd_limit
        ))

        # 2. Try least-loaded first
        for name in candidates_by_load:
            if await self._slots[name].try_acquire_nowait(estimated_tokens):
                return name
        
        # 3. Block until any model frees up
        timeout = 90.0
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            candidates_by_load = sorted(valid_candidates, key=lambda n: (get_max_utilization(n), self._slots[n].priority))
            for name in candidates_by_load:
                if await self._slots[name].try_acquire_nowait(estimated_tokens):
                    return name
            await asyncio.sleep(1.0)
            
        best_name = valid_candidates[0]
        await self._slots[best_name].acquire(estimated_tokens, timeout=0.1)
        return best_name

    def _estimate_tokens(self, content: Any) -> int:
        """Conservative token estimation with schema overhead padding."""
        base_overhead = 400 # Increased for function calling overhead
        
        def _count(obj):
            if isinstance(obj, str): return int(len(obj) / 3.5) # Indonesian/slang requires more tokens per char
            if isinstance(obj, list): return sum(_count(x) for x in obj)
            return int(len(str(obj)) / 3.5)

        # 1.5x multiplier for pessimistic reservation during concurrent bursts
        return int((max(1, _count(content)) + base_overhead) * 1.5)


    async def _call(self, contents: Any, config: dict, slot_name: str, 
                    attempt: int = 1, max_attempts: int = 6, tried: list[str] = None,
                    banned_providers: set[str] = None) -> Optional[WrappedResponse]:
        """Executes an AI call with automatic cross-provider fallback and jittered backoff."""
        if tried is None: tried = []
        if banned_providers is None: banned_providers = set()
        
        tried.append(slot_name)

        slot = self._slots[slot_name]
        estimated_tokens = self._estimate_tokens(contents)
        try:
            logger.info(f"🛰️ [AI] Requesting {slot.model_id} ({slot.provider}) | Attempt {attempt}...")
            response = await asyncio.wait_for(
                slot.client.generate_content(
                    model=slot.model_id,
                    contents=contents,
                    config=config.copy(),
                    capabilities=slot.capabilities
                ),
                timeout=60.0
            )
            
            if response is None:
                raise Exception("Provider returned empty response")
                
            response.model_name = slot_name

            # Correct token counts with actual data
            actual_tokens = response.usage.get("total_tokens", 0) if response.usage else 0
            if actual_tokens > 0:
                slot.update_actual_usage(estimated_tokens, actual_tokens)

            # Sync from headers if available
            if response.headers:
                slot.sync_from_headers(response.headers)

            return response

        except Exception as e:
            import random
            err_str = str(e).lower()
            logger.warning(f"🔄 AI ({slot.model_id}) Failure | Attempt {attempt} | {type(e).__name__}: {repr(e)}")
            slot.release_last()

            # Dynamic 429 Rate Limit Handling
            is_rate_limit = "429" in err_str or "rate limit" in err_str or "resource has been exhausted" in err_str
            if is_rate_limit:
                sleep_sec = 60.0

                # OPENROUTER DAILY LIMIT: If we see this, back off for 12 hours (it resets daily)
                if slot.provider == "openrouter" and ("50 requests" in err_str or "daily limit" in err_str):
                    sleep_sec = 3600 * 12
                    logger.warning(f"🛑 [OpenRouter] Daily free limit (50) reached! Backing off for 12h.")
                
                # ADAPTIVE SYNC: Parse "Used X, Limit Y" from Groq error message
                # Example: "Tokens per day (TPD): Limit 500000, Used 498659"
                usage_match = re.search(r'limit (\d+), used (\d+)', err_str)
                if usage_match:
                    limit_val = int(usage_match.group(1))
                    used_val  = int(usage_match.group(2))
                    slot.saturate_locally(used_val, limit_val)

                wait_match = re.search(r'try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)', err_str)
                if wait_match:
                    h = int(wait_match.group(1) or 0)
                    m = int(wait_match.group(2) or 0)
                    s = float(wait_match.group(3) or 0)
                    sleep_sec = (h * 3600) + (m * 60) + s + 1.0
                elif "per day" in err_str or "tpd" in err_str or "rpd" in err_str:
                    sleep_sec = 3600 * 4 # Back off for 4 hours on daily limit

                # Add jitter to prevent thundering herds on recovery
                jitter = random.uniform(0.8, 1.2)
                sleep_sec = sleep_sec * jitter

                logger.warning(f"⏳ [{slot.name}] Rate Limited. Sleeping {sleep_sec:.1f}s.")
                slot.exhausted_until = time.monotonic() + sleep_sec
            if attempt < max_attempts:
                is_vision = False
                if isinstance(contents, list):
                    # Check if any part is a dict with 'data' (our image format)
                    is_vision = any(isinstance(item, dict) and 'data' in item for item in contents)

                # Cross-provider ban ONLY on severe server errors (5xx).
                # Client errors (4xx) should ONLY exclude the specific model.
                exclude_list = list(tried)
                
                # Check for 5xx in the error string or response status code
                is_server_death = any(s in err_str for s in ["500", "502", "503", "504"])
                if hasattr(e, 'status_code') and e.status_code >= 500:
                    is_server_death = True
                
                if is_server_death:
                    banned_providers.add(slot.provider)
                    logger.warning(f"🚫 [{slot.provider}] Provider-wide ban triggered due to server death (5xx).")

                # Add all slots from banned providers to exclude_list
                for n, s in self._slots.items():
                    if s.provider in banned_providers and n not in exclude_list:
                        exclude_list.append(n)
                            
                next_slot = await self._pick_model(
                    exclude=exclude_list, 
                    estimated_tokens=self._estimate_tokens(contents),
                    require_vision=is_vision
                )
                if not next_slot:
                    logger.error(f"❌ AI Fallback failed: No suitable models available after {slot_name} failure.")
                    return None
                
                return await self._call(contents, config, next_slot, attempt + 1, max_attempts, tried, banned_providers)
            
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
            score -= 5
            
        # Phrases like "aman ga", "aman ya" -> moderate penalty
        if re.search(r'\b(aman|work|on)\s+(ga|gak|nggak|ya)\b', t):
            score -= 8

        if t.endswith('?') and words and words[0] in question_words:
            score -= 5

        if has_question_word and ('aman' in t or 'work' in t or 'on' in t):
            # Probably asking if it's working
            score -= 8

        # specifically penalize single short word followed by ngga
        if re.search(r'\b(aman|work|on)\s+(ngga)\b', t):
            score -= 15

        # If we have a strong keyword, we almost always want to check it unless it's pure junk
        if any(kw in t for kw in _STRONG_KEYWORDS) and score >= 0:
            return True

        # Short message penalty
        if len(words) <= 4 and score < 2:
            return False

        return score >= 2

    async def process_batch(self, messages: Sequence[dict[str, Any]], db: Any = None) -> list[PromoExtraction] | None:
        """Extracts promos from a batch of messages using AI with concurrent chunking."""
        if not messages:
            return []

        filtered = [m for m in messages if self._is_worth_checking(m.get('text'), bool(m.get('has_photo', False)))]
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
                    m['context'] = f"C:{ctx_text[-1000:]} "
                else:
                    m['context'] = ""
        else:
            for m in filtered:
                m['context'] = ""

        config = {
            "response_mime_type": "application/json",
            "response_schema": BatchResponse,
            "system_instruction": _EXTRACT_SYSTEM,
        }

        # ── Scatter-Gather Chunking ──
        # Distribute the load across the fleet concurrently instead of one giant batch
        CHUNK_SIZE = 10 
        chunks = [filtered[i:i + CHUNK_SIZE] for i in range(0, len(filtered), CHUNK_SIZE)]
        
        async def _process_chunk(chunk: list[dict]):
            chunk_text = "\n---\n".join(
                f"ID:{m['id']} {m.get('context', '')}MSG: {m['text'] or ''}"
                for m in chunk
            )
            tokens = self._estimate_tokens(chunk_text)
            target_model = await self._pick_model(estimated_tokens=tokens, require_vision=False)
            
            logger.debug(f"🛰️ Chunk ({len(chunk)} msgs) -> {target_model}")
            
            return await self._call(
                contents=f"Batch pesan:\n\n{chunk_text}",
                config=config,
                slot_name=target_model,
            )

        # Fire all chunks to different models simultaneously
        tasks = [_process_chunk(c) for c in chunks]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        valid = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"❌ Chunk {i} processing crashed: {response}")
                continue
            if response is None or not response.parsed:
                continue

            for p in response.parsed.promos:
                summary = (p.summary or "").strip()
                if not summary or len(summary) < 8:
                    continue
                if summary.lower() in _JUNK_SUMMARIES:
                    continue
                if _META_SUMMARY_PATTERN.search(summary):
                    continue
                
                verified_brand = normalize_brand(p.brand)
                p.brand = verified_brand
                p.model_name = response.model_name
                valid.append(p)

        logger.info(f"✅ Scatter-Gather complete. Extracted {len(valid)} promos from {len(filtered)} msgs across {len(chunks)} chunks.")
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
            contents.append({"mime_type": "image/jpeg", "data": parent_photo})

        tokens = self._estimate_tokens(contents)
        # Prefer google for vision, but allow fallback
        target = await self._pick_model(provider="google" if parent_photo else None, estimated_tokens=tokens)
        if not target and parent_photo:
            # Fallback to any vision-capable model
            target = await self._pick_model(estimated_tokens=tokens)
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
        contents = [prompt, {"mime_type": "image/jpeg", "data": image_bytes}]
        tokens = self._estimate_tokens(contents)
        
        # Prefer google for vision, but allow fallback
        target = await self._pick_model(provider="google", estimated_tokens=tokens)
        if not target:
            # Fallback to any vision-capable model (like Llama Scout)
            target = await self._pick_model(estimated_tokens=tokens)

        if not target:
            logger.error("No vision models available for process_image")
            return None

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
        
        config = {
            "response_mime_type": "application/json",
            "response_schema": TrendResponse,
            "system_instruction": (
                "Kamu adalah TanyaDFBot, analis tren grup Discountfess. "
                "Simpulkan 1-3 tren utama dengan link ID pesan. "
                "KATEGORI: [PROMO_BARU], [SYSTEM_EROR], [DISKUSI_HANGAT], atau [RESTOCK]. "
                "DILARANG KERAS MENGGUNAKAN TABEL. Gunakan bold untuk brand."
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
        
        system = (
            "Kamu adalah TanyaDFBot, analis sentimen real-time. Ada lonjakan aktivitas di grup.\n"
            f"Kata kunci dominan dlm {window} menit terakhir: {counts_str}.\n\n"
            "TUGASMU: Jelaskan APA yang sedang dibahas berdasarkan pesan-pesan berikut.\n"
            "DILARANG KERAS PAKAI TABEL. Jawab singkat padat dalam 1-2 kalimat."
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
