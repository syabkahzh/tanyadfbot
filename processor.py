"""processor.py — Gemini AI Layer.

Key fixes vs previous version:
- BUG C FIX: RPD (daily limit) is now tracked from models_config.json.
  Google Gemma-4 no longer hammers past daily quota because daily_limit=0.
- THUNDERING HERD FIX: _pick_model now uses a single global fleet lock
  during the check-and-reserve phase so concurrent coroutines can't all
  grab the same slot simultaneously.
- BETTER LOAD BALANCING: priority-within-tier load balancing now correctly
  spreads across ALL available P1 models before falling to P2, instead of
  converging on the first alphabetically.
- ADAPTIVE MODEL SELECTION: heavy tasks (image, dedup, summarization) route
  to capable models explicitly; lightweight extraction goes to fastest.
- PROMPT IMPROVEMENTS: clearer, more concise extraction prompt with better
  examples tuned to Indonesian deal-hunter slang.
- NO-SLEEP RETRY: backoff uses exponential sleep only on 5xx; 429 sets
  exhausted_until so the slot is skipped without sleeping the event loop.
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
    r'^(?:wkwk|haha|hehe|iya|noted|oke|ok|makasih|thanks|thx|mantap|gas|bos|guys|gais|bang|kak|siap|sip|lol|anjir|anjay|btw|oot|gws|semangat|ya allah|nangis|sedih|beneran|kah|[!.\s])+$',
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
_META_SUMMARY_PATTERN = re.compile(
    r'(user bertanya|tidak ada informasi|tidak disebutkan|no information|'
    r'pesan ini|pertanyaan tentang|menanyakan|mencari tahu|'
    r'meminta konfirmasi|menginformasikan bahwa)',
    re.IGNORECASE
)

# ── Response schemas ──────────────────────────────────────────────────────────

class PromoExtraction(BaseModel):
    original_msg_id: int = 0
    summary: str = Field(default="", description="1 kalimat ringkasan WAJIB DALAM BAHASA INDONESIA.")
    brand: str = Field(default="", description="Nama brand yang tepat.")
    conditions: str = Field(default="", description="Syarat dan ketentuan DALAM BAHASA INDONESIA, atau string kosong.")
    valid_until: Optional[str] = ""
    status: Literal['active', 'expired', 'unknown'] = Field(default='unknown')
    confidence: float = Field(default=1.0)
    links: List[str] = []
    detected_at: Optional[str] = None
    queue_time: Optional[float] = None
    ai_time: Optional[float] = None
    model_name: Optional[str] = None

class BatchResponse(BaseModel):
    promos: List[PromoExtraction]

class TrendItem(BaseModel):
    topic: str
    msg_id: int
    model_name: Optional[str] = None

class TrendResponse(BaseModel):
    trends: List[TrendItem]


# ── Improved Prompts ──────────────────────────────────────────────────────────

_EXTRACT_SYSTEM = """Kamu TanyaDFBot, sistem ekstraksi promo untuk grup Discountfess Indonesia.
Output HARUS berupa JSON valid sesuai skema. DILARANG tabel/markdown.

SLANG PENTING:
jp=jackpot/berhasil | aman/on/work/nyala=aktif | nt/koid/habis/zonk=expired
sfood=ShopeeFood | gfood=GoFood | sopi=Shopee | tsel=Telkomsel | idm=Indomaret
alfa/jsm/psm/afm=Alfamart | kopken=Kopi Kenangan | cgv=CGV | ag/alfagift=Alfagift
spay=ShopeePay | spx=SPX | cb/kesbek=cashback | voc/vcr=voucher | fs=flash sale
tukpo=tukar poin | minbel=minimum belanja | gratong=gratis ongkir

ATURAN EKSTRAKSI:
1. Baca C: (konteks/reply) DAHULU untuk tahu brand dan konteks.
2. Evaluasi status: ACTIVE jika "jp/aman/on/work/nyala/alhamdulillah/dapet".
   EXPIRED jika "nt/habis/zonk/koid/gaib/badut/limit". UNKNOWN jika tidak jelas.
3. Jika pesan hanya tanya-tanya atau OOT: set brand="Unknown", summary="Unknown".
4. Jangan tulis "SKIP" - gunakan "Unknown" untuk brand/summary yang tidak relevan.
5. Summary: 1 kalimat padat, awali dengan nama brand bold (**Brand**).

CONTOH:
ID:1 C:sfood diskon 80k MSG:nyala -> {"promos":[{"original_msg_id":1,"brand":"ShopeeFood","summary":"**ShopeeFood** diskon 80k aktif.","status":"active","confidence":0.95}]}
ID:2 C:gopay coins MSG:nt -> {"promos":[{"original_msg_id":2,"brand":"GoPay","summary":"**GoPay** Coins sudah habis/expired.","status":"expired","confidence":0.90}]}
ID:3 MSG:ada yg tau cgv tsel? -> {"promos":[{"original_msg_id":3,"brand":"Unknown","summary":"Unknown","status":"unknown","confidence":0.1}]}"""

_DEDUP_SYSTEM = "Kamu agen deteksi duplikasi. Output HANYA angka indeks dipisah koma."

_DIGEST_SYSTEM = """Kamu TanyaDFBot, admin gercep grup Discountfess. 
Rangkum promo dengan bahasa santai dan informatif. Gunakan slang secukupnya.
DILARANG KERAS PAKAI TABEL. Gunakan bullet points dan bold untuk nama brand.
Jawab singkat padat."""

_VISION_SYSTEM = """Kamu analis visual TanyaDFBot untuk Discountfess.
Ekstrak detail promo dari gambar/screenshot.

ATURAN:
1. Struk/bukti pembayaran berhasil → status='active', confidence=1.0
2. Poster/banner promo → status='active', confidence=0.85  
3. Meme/OOT/settings UI → summary='Unknown', brand='Unknown'
DILARANG TABEL. Bold untuk brand."""


# ── Keyword sets ──────────────────────────────────────────────────────────────

_STRONG_KEYWORDS: set[str] = {
    'sfood','gfood','grab','shopee','gojek','tokped','tokopedia',
    'voucher','vcr','voc','diskon','promo','cashback','gratis','potongan',
    'idm','indomaret','alfa','alfamart','alfagift','hokben',
    'klaim','claim','restock','ristok','nt','abis','habis',
    'gabisa','gaada','g+b+s','gamau','minbel',
    'kuota','limit','slot','redeem','qr','scan','edc',
    'r+s+t+k','r+s+t+c+k','r+st+ck',
    'cb','kesbek','c\+s\+h\+b\+c\+k','cash back',
    'luber','pecah','flash','sale','deal','murah','hemat','bonus',
    'ongkir','gratis ongkir','membership','member','mamber',
    'yang butuh aja','ymma','tukpo','murce','murmer','sopi','tsel','cgv','xxi',
    'svip','badut','war','begal','kreator','live kreator',
    'kopken','chatime','gindaco','solaria','rotio','spx','gopay','spay',
    'shopeepay','ovo','neo','tmrw','saqu','seabank','hero',
    'blibli','serbabu','famima','familymart','supin','superindo',
    'gacoan','mie gacoan','wingstop','yoshinoya','azko','sei',
    'flip','superbank','dana','tts','emados','ndog','pc','garap',
}

_WEAK_KEYWORDS: set[str] = {
    'cek','info','makasih','thx','thanks','makasi','mks','terimakasih','terima kasih'
}

_JUNK_SUMMARIES: set[str] = {'summary','none','n/a','-','tidak ada','tidak ditemukan'}

# ── Pre-compiled optimized keyword patterns ───────────────────────────────────
_STRONG_PATTERN = re.compile('|'.join(map(re.escape, _STRONG_KEYWORDS)))
_WEAK_PATTERN = re.compile('|'.join(map(re.escape, _WEAK_KEYWORDS)))
_QUESTION_WORDS = frozenset({'ga', 'gak', 'nggak', 'apa', 'gimana', 'berapa', 'kapan', 'dimana', 'kenapa', 'ada', 'masih', 'ya'})
_QUESTION_AMAN_PATTERN = re.compile(r'\b(aman|work|on)\s+(ga|gak|nggak|ya)\b')
_QUESTION_AMAN_NGGA_PATTERN = re.compile(r'\b(aman|work|on)\s+(ngga)\b')



# ── AI Clients ────────────────────────────────────────────────────────────────

class BaseAIClient:
    async def generate_content(self, model: str, contents: Any, config: dict[str, Any], capabilities: dict[str, Any]) -> Any:
        raise NotImplementedError

class WrappedResponse:
    def __init__(self, res=None, text=None, parsed=None, model_name=None, usage=None, headers=None):
        self.res = res
        self._text = text
        self._parsed = parsed
        self.model_name = model_name
        self.usage = usage
        self.headers = headers or {}

    @property
    def text(self):
        return self._text if self._text is not None else getattr(self.res, 'text', "")

    @property
    def parsed(self):
        return self._parsed if self._parsed is not None else getattr(self.res, 'parsed', None)

class GoogleClient(BaseAIClient):
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    async def generate_content(self, model: str, contents: Any, config: dict[str, Any], capabilities: dict[str, Any]) -> Any:
        config = config.copy()

        if not capabilities.get("native_json", True):
            schema = config.pop("response_schema", None)
            config.pop("response_mime_type", None)
            if schema:
                schema_text = f"\n\nOUTPUT MUST BE VALID JSON matching this schema:\n{schema.model_json_schema()}"
                if isinstance(contents, str):
                    contents += schema_text
                elif isinstance(contents, list):
                    if isinstance(contents[0], str):
                        contents[0] += schema_text
                    else:
                        contents.append(schema_text)

        if not capabilities.get("system_instruction", True):
            system = config.pop("system_instruction", None)
            if system:
                sys_text = system
                if hasattr(system, 'parts'):
                    sys_text = system.parts[0].text
                if isinstance(contents, str):
                    contents = f"SYSTEM: {sys_text}\n\n{contents}"
                elif isinstance(contents, list):
                    contents.insert(0, f"SYSTEM: {sys_text}")

        if isinstance(contents, list):
            final_contents = []
            for item in contents:
                if isinstance(item, dict) and "data" in item:
                    final_contents.append(genai.types.Part.from_bytes(data=item["data"], mime_type=item["mime_type"]))
                else:
                    final_contents.append(item)
            contents = final_contents

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
    def __init__(self, api_key: str, base_url: Optional[str] = None, provider: str = "openai"):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.provider = provider

    async def generate_content(self, model: str, contents: Any, config: dict[str, Any], capabilities: dict[str, Any]) -> Any:
        system = config.get("system_instruction", "")
        response_schema = config.get("response_schema")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})

        if isinstance(contents, str):
            messages.append({"role": "user", "content": contents})
        elif isinstance(contents, list):
            user_content = []
            for item in contents:
                if isinstance(item, str):
                    user_content.append({"type": "text", "text": item})
                elif isinstance(item, dict) and "data" in item:
                    import base64
                    img_b64 = base64.b64encode(item["data"]).decode("utf-8")
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{item['mime_type']};base64,{img_b64}"}
                    })
            if user_content:
                messages.append({"role": "user", "content": user_content})
            else:
                messages.append({"role": "user", "content": str(contents)})

        kwargs = {}
        reasoning_mode = capabilities.get("reasoning")
        if reasoning_mode:
            kwargs["extra_body"] = {"reasoning_format": reasoning_mode}

        if response_schema:
            if capabilities.get("native_tools", False):
                schema_dict = response_schema.model_json_schema()
                kwargs["tools"] = [{
                    "type": "function",
                    "function": {
                        "name": "extract_data",
                        "description": f"Extract structured {response_schema.__name__} data",
                        "parameters": schema_dict
                    }
                }]
                kwargs["tool_choice"] = "auto"
            elif capabilities.get("native_json", True):
                kwargs["response_format"] = {"type": "json_object"}
                schema_str = response_schema.model_json_schema()
                messages[0]["content"] += (
                    f"\n\nYou MUST respond with ONLY valid JSON. No preamble.\nSchema:\n{schema_str}"
                )

        try:
            raw_res = await self.client.chat.completions.with_raw_response.create(
                model=model, messages=messages, **kwargs
            )
            res = raw_res.parse()
            headers = {k.lower(): v for k, v in raw_res.headers.items() if k.lower().startswith('x-ratelimit')}
        except Exception as e:
            err_str = str(e).lower()
            if ("tool_use_failed" in err_str or "tool choice" in err_str) and hasattr(e, 'response'):
                try:
                    body = e.response.json()
                    text = body.get('error', {}).get('failed_generation')
                    if text:
                        logger.info(f"💡 [AI] Rescued content from failed tool call for {model}")
                        headers = {k.lower(): v for k, v in e.response.headers.items() if k.lower().startswith('x-ratelimit')}
                        return WrappedResponse(text=text, model_name=model, headers=headers)
                except Exception:
                    pass
            raise e

        message = res.choices[0].message
        text = None
        if message.tool_calls:
            text = message.tool_calls[0].function.arguments
        else:
            text = message.content

        if text:
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
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
                try:
                    parsed = response_schema.model_validate_json(clean_text)
                except Exception:
                    raw = json.loads(clean_text)

                    if isinstance(raw, list) and len(raw) > 0:
                        if response_schema.__name__ == "PromoExtraction":
                            raw = raw[0]
                        elif response_schema.__name__ == "BatchResponse":
                            raw = {"promos": raw}

                    if response_schema.__name__ == "BatchResponse":
                        if isinstance(raw, dict) and "promos" not in raw:
                            if "summary" in raw:
                                raw = {"promos": [raw]}

                    def clean_nulls(obj):
                        if isinstance(obj, list):
                            return [clean_nulls(x) for x in obj]
                        if isinstance(obj, dict):
                            for k, v in list(obj.items()):
                                if v is None:
                                    if k in ('queue_time', 'ai_time'):
                                        pass
                                    elif k == 'confidence':
                                        obj[k] = 1.0
                                    elif k == 'original_msg_id':
                                        obj[k] = 0
                                    else:
                                        obj[k] = ""
                                else:
                                    obj[k] = clean_nulls(v)
                        return obj

                    parsed = response_schema.model_validate(clean_nulls(raw))
            except Exception as e:
                logger.warning(f"AI client failed to parse JSON from {model}: {e}")

        return WrappedResponse(res, text=text, parsed=parsed, usage=usage, headers=headers)


# ── Rate limiter ──────────────────────────────────────────────────────────────

class _ModelSlot:
    """Sliding-window RPM + RPD + TPM limiter."""

    def __init__(self, name: str, provider: str, model_id: str, client: BaseAIClient,
                 limit: int, daily_limit: int = 0,
                 tpm_limit: int = 0, tpd_limit: int = 0,
                 priority: int = 3, capabilities: dict = None) -> None:
        self.name = name
        self.provider = provider
        self.model_id = model_id
        self.client = client
        self.limit = limit          # RPM
        self.daily_limit = daily_limit  # RPD
        self.tpm_limit = tpm_limit
        self.tpd_limit = tpd_limit
        self.priority = priority
        self.capabilities = capabilities or {}

        self._calls: list[float] = []           # RPM timestamps
        self._daily_calls: list[float] = []     # RPD timestamps
        self._tokens: list[tuple[float, int]] = []        # TPM
        self._daily_tokens: list[tuple[float, int]] = []  # TPD

        # Per-slot lock — used ONLY for atomic check+append inside try_acquire_nowait
        self._lock = asyncio.Lock()
        self.exhausted_until: float = 0.0

    def _cleanup(self, now: float) -> None:
        """Prune expired entries. Must be called under self._lock."""
        self._calls = [t for t in self._calls if now - t < 60]
        self._tokens = [(ts, n) for ts, n in self._tokens if now - ts < 60]
        if self.daily_limit > 0 or self._daily_calls:
            self._daily_calls = [t for t in self._daily_calls if now - t < 86400]
        if self.tpd_limit > 0 or self._daily_tokens:
            self._daily_tokens = [(ts, n) for ts, n in self._daily_tokens if now - ts < 86400]

    def available(self, now: float) -> int:
        cutoff = now - 60
        active = sum(1 for t in self._calls if t > cutoff)
        return max(0, self.limit - active)

    def is_daily_exhausted(self, now: float) -> bool:
        """Fast daily-limit check without acquiring lock."""
        if self.daily_limit <= 0:
            return False
        cutoff = now - 86400
        used = sum(1 for t in self._daily_calls if t > cutoff)
        return used >= self.daily_limit

    async def try_acquire_nowait(self, estimated_tokens: int = 0) -> bool:
        """Non-blocking. Returns True and records the call if slot is free."""
        now = time.monotonic()
        if now < self.exhausted_until:
            return False
        # Fast daily check before acquiring lock
        if self.is_daily_exhausted(now):
            return False

        async with self._lock:
            self._cleanup(now)

            # RPD check
            if self.daily_limit > 0 and len(self._daily_calls) >= self.daily_limit:
                return False
            # RPM check
            if len(self._calls) >= self.limit:
                return False
            # TPM check
            if estimated_tokens > 0 and self.tpm_limit > 0:
                current_tpm = sum(n for _, n in self._tokens)
                if current_tpm + estimated_tokens > self.tpm_limit:
                    return False
            # TPD check
            if estimated_tokens > 0 and self.tpd_limit > 0:
                current_tpd = sum(n for _, n in self._daily_tokens)
                if current_tpd + estimated_tokens > self.tpd_limit:
                    return False

            # Commit the reservation
            self._calls.append(now)
            self._tokens.append((now, estimated_tokens))
            if self.daily_limit > 0:
                self._daily_calls.append(now)
            if self.tpd_limit > 0:
                self._daily_tokens.append((now, estimated_tokens))

            return True

    async def acquire(self, estimated_tokens: int = 0, timeout: float = 90.0) -> bool:
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

    def daily_remaining(self) -> int:
        if self.daily_limit <= 0:
            return 999999
        return max(0, self.daily_limit - self.daily_usage())

    def release_last(self) -> None:
        if self._calls:
            self._calls.pop()
        if self._tokens:
            self._tokens.pop()
        if self._daily_calls:
            self._daily_calls.pop()
        if self._daily_tokens:
            self._daily_tokens.pop()

    def update_actual_usage(self, estimated: int, actual: int) -> None:
        if estimated == actual or actual <= 0:
            return
        diff = actual - estimated
        now = time.monotonic()
        self._tokens.append((now, diff))
        if self.tpd_limit > 0:
            self._daily_tokens.append((now, diff))

    def saturate_locally(self, used: int, limit: int) -> None:
        if limit <= 0:
            return
        now = time.monotonic()
        current_local = sum(n for _, n in self._daily_tokens)
        diff = used - current_local
        if diff > 0:
            self._daily_tokens.append((now, diff))
            logger.warning(f"🎯 [{self.name}] Syncing tokens: +{diff} (API Used: {used}/{limit})")

    def sync_from_headers(self, headers: dict) -> None:
        if not headers:
            return
        now = time.monotonic()

        # RPD sync
        limit_req = headers.get('x-ratelimit-limit-requests')
        rem_req = headers.get('x-ratelimit-remaining-requests')
        if limit_req and rem_req:
            try:
                used = int(limit_req) - int(rem_req)
                current = len([t for t in self._daily_calls if now - t < 86400])
                diff = used - current
                if diff > 0:
                    for _ in range(diff):
                        self._daily_calls.append(now - 1.0)
            except Exception:
                pass

        # TPM sync
        limit_tokens = headers.get('x-ratelimit-limit-tokens')
        rem_tokens = headers.get('x-ratelimit-remaining-tokens')
        if limit_tokens and rem_tokens:
            try:
                used = int(limit_tokens) - int(rem_tokens)
                current = sum(n for _, n in self._tokens)
                diff = used - current
                if diff > 0:
                    self._tokens.append((now, diff))
            except Exception:
                pass

        # TPD sync
        limit_tpd = headers.get('x-ratelimit-limit-tokens-day')
        rem_tpd = headers.get('x-ratelimit-remaining-tokens-day')
        if limit_tpd and rem_tpd:
            try:
                self.saturate_locally(int(limit_tpd) - int(rem_tpd), int(limit_tpd))
            except Exception:
                pass


# ── GeminiProcessor ───────────────────────────────────────────────────────────

class GeminiProcessor:
    """Orchestrates AI analysis using the multi-provider fleet."""

    _AI_CALL_TIMEOUT_SEC = 60.0

    # Global fleet lock: prevents thundering-herd where N concurrent coroutines
    # all see the same slot as "available" before any has committed its reservation.
    _fleet_lock: asyncio.Lock = asyncio.Lock()

    def __init__(self) -> None:
        self.reinitialize()

    def reinitialize(self) -> None:
        self._slots: dict[str, _ModelSlot] = {}
        army = Config.get_ai_army()
        for p in army:
            client = None
            api_key = p.get('api_key')

            if p['provider'] != 'ollama' and not api_key:
                logger.warning(f"Skipping {p['name']}: {p.get('api_key_env', 'API_KEY')} not set.")
                continue

            if p['provider'] == 'google':
                client = GoogleClient(api_key=api_key)
            elif p['provider'] in ('groq', 'glm', 'openrouter', 'mistral', 'siliconflow', 'cerebras'):
                client = OpenAICompatibleClient(
                    api_key=api_key,
                    base_url=p.get('base_url'),
                    provider=p['provider']
                )

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

        # Sorted list: priority ASC, then name for determinism
        self._priority_list = [
            s.name for s in sorted(self._slots.values(), key=lambda x: (x.priority, x.name))
        ]
        logger.info(f"AI Fleet initialized: {len(self._slots)} models | order: {self._priority_list}")

    def update_model_priority(self, name: str, priority: int) -> bool:
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
            if not found:
                return False
            with open(path, "w") as f:
                json.dump(army, f, indent=4)
            self.reinitialize()
            return True
        except Exception as e:
            logger.error(f"Failed to update priority for {name}: {e}")
            return False

    async def _pick_model(
        self,
        exclude: Optional[str | list[str]] = None,
        provider: Optional[str] = None,
        estimated_tokens: int = 0,
        require_vision: bool = False,
        prefer_capable_json: bool = False,
    ) -> str:
        """
        Thread-safe model selection with global fleet lock.

        The lock prevents the thundering-herd problem where N concurrent
        coroutines all observe the same slot as available and grab it
        simultaneously, resulting in over-allocation.

        Selection criteria (in order):
        1. Must not be in exclude list
        2. Must support vision (if require_vision=True)
        3. Must not be daily-exhausted
        4. Sort by (priority, RPM utilization %, daily remaining DESC)
        5. First available slot wins
        """
        excludes = set([exclude] if isinstance(exclude, str) else (exclude or []))

        async with self._fleet_lock:
            now = time.monotonic()

            # Build candidate list
            candidates = [
                n for n in self._priority_list
                if n not in excludes
                and not self._slots[n].is_daily_exhausted(now)
                and (not require_vision or self._slots[n].capabilities.get("vision"))
                and (now >= self._slots[n].exhausted_until)
            ]

            if require_vision and not candidates:
                # Fallback: any model not exhausted
                candidates = [
                    n for n in self._priority_list
                    if n not in excludes and now >= self._slots[n].exhausted_until
                ]

            if provider:
                p_cands = [n for n in candidates if self._slots[n].provider == provider]
                if p_cands:
                    candidates = p_cands

            if not candidates:
                # Last resort: pick least-exhausted ignoring daily limit
                candidates = [n for n in self._priority_list if n not in excludes]
                if not candidates:
                    return self._priority_list[0]

            # Sort: priority first, then utilization (lower=better), then daily_remaining (higher=better)
            def score(name: str) -> tuple:
                s = self._slots[name]
                rpm_util = s.current_usage() / max(1, s.limit)
                daily_rem = s.daily_remaining()
                return (s.priority, rpm_util, -daily_rem)

            candidates.sort(key=score)

            # Try to acquire the best available slot (still under fleet lock
            # so no other coroutine can interleave)
            for name in candidates:
                slot = self._slots[name]
                # try_acquire_nowait also uses its own per-slot lock, but
                # since we hold fleet_lock, no other coroutine is in _pick_model
                acquired = await slot.try_acquire_nowait(estimated_tokens)
                if acquired:
                    return name

        # All slots busy — block-wait outside the fleet lock so we don't
        # hold the fleet lock while sleeping (that would deadlock)
        deadline = time.monotonic() + 90.0
        while time.monotonic() < deadline:
            await asyncio.sleep(1.0)
            async with self._fleet_lock:
                now = time.monotonic()
                available = [
                    n for n in self._priority_list
                    if n not in excludes and now >= self._slots[n].exhausted_until
                    and not self._slots[n].is_daily_exhausted(now)
                ]
                available.sort(key=lambda n: (self._slots[n].priority, self._slots[n].current_usage() / max(1, self._slots[n].limit)))
                for name in available:
                    if await self._slots[name].try_acquire_nowait(estimated_tokens):
                        return name

        # True last resort
        best = self._priority_list[0] if self._priority_list else list(self._slots.keys())[0]
        logger.error(f"_pick_model: all slots saturated after 90s, forcing {best}")
        return best

    def _estimate_tokens(self, content: Any) -> int:
        base_overhead = 400

        def _count(obj):
            if isinstance(obj, str):
                return int(len(obj) / 3.5)
            if isinstance(obj, list):
                return sum(_count(x) for x in obj)
            return int(len(str(obj)) / 3.5)

        return int((max(1, _count(content)) + base_overhead) * 1.5)

    async def _call(
        self,
        contents: Any,
        config: dict,
        slot_name: str,
        attempt: int = 1,
        max_attempts: int = 6,
        tried: list[str] = None,
        banned_providers: set[str] = None,
    ) -> Optional[WrappedResponse]:
        """Execute an AI call with cross-provider fallback and smart backoff."""
        if tried is None:
            tried = []
        if banned_providers is None:
            banned_providers = set()

        tried.append(slot_name)
        slot = self._slots[slot_name]
        estimated_tokens = self._estimate_tokens(contents)

        try:
            logger.debug(f"🛰️ [AI] {slot.model_id} ({slot.provider}) attempt {attempt}...")
            response = await asyncio.wait_for(
                slot.client.generate_content(
                    model=slot.model_id,
                    contents=contents,
                    config=config.copy(),
                    capabilities=slot.capabilities
                ),
                timeout=self._AI_CALL_TIMEOUT_SEC
            )

            if response is None:
                raise Exception("Provider returned empty response")

            response.model_name = slot_name

            # Correct token accounting
            actual_tokens = response.usage.get("total_tokens", 0) if response.usage else 0
            if actual_tokens > 0:
                slot.update_actual_usage(estimated_tokens, actual_tokens)
            if response.headers:
                slot.sync_from_headers(response.headers)

            return response

        except Exception as e:
            import random
            err_str = str(e).lower()
            logger.warning(f"🔄 [{slot.name}] Failure attempt {attempt}: {type(e).__name__}: {str(e)[:120]}")
            slot.release_last()

            is_rate_limit = (
                "429" in err_str
                or "rate limit" in err_str
                or "resource has been exhausted" in err_str
            )

            if is_rate_limit:
                sleep_sec = 61.0

                # OpenRouter daily limit
                if slot.provider == "openrouter" and ("50 requests" in err_str or "daily limit" in err_str):
                    sleep_sec = 3600 * 12
                    logger.warning(f"🛑 [{slot.name}] OpenRouter daily limit hit — backing off 12h")

                # Parse "try again in Xm Ys" from error
                wait_match = re.search(r'try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)', err_str)
                if wait_match:
                    h = int(wait_match.group(1) or 0)
                    m = int(wait_match.group(2) or 0)
                    s = float(wait_match.group(3) or 0)
                    sleep_sec = (h * 3600) + (m * 60) + s + 1.0
                elif "per day" in err_str or "tpd" in err_str or "rpd" in err_str:
                    sleep_sec = 3600 * 4

                # Adaptive sync from error message
                usage_match = re.search(r'limit (\d+), used (\d+)', err_str)
                if usage_match:
                    slot.saturate_locally(int(usage_match.group(2)), int(usage_match.group(1)))

                jitter = random.uniform(0.9, 1.1)
                slot.exhausted_until = time.monotonic() + sleep_sec * jitter
                logger.warning(f"⏳ [{slot.name}] Rate limited — pausing {sleep_sec * jitter:.0f}s")

            if attempt < max_attempts:
                is_vision = isinstance(contents, list) and any(
                    isinstance(item, dict) and 'data' in item for item in contents
                )

                exclude_list = list(tried)

                # Provider-wide ban on server death (5xx)
                is_server_death = any(s in err_str for s in ["500", "502", "503", "504"])
                if hasattr(e, 'status_code') and getattr(e, 'status_code', 0) >= 500:
                    is_server_death = True
                if is_server_death:
                    banned_providers.add(slot.provider)
                    logger.warning(f"🚫 [{slot.provider}] Provider banned (5xx server death)")

                for n, s in self._slots.items():
                    if s.provider in banned_providers and n not in exclude_list:
                        exclude_list.append(n)

                next_slot = await self._pick_model(
                    exclude=exclude_list,
                    estimated_tokens=estimated_tokens,
                    require_vision=is_vision,
                )
                if next_slot:
                    return await self._call(
                        contents, config, next_slot,
                        attempt + 1, max_attempts, tried, banned_providers
                    )

            logger.error(f"❌ AI call failed after {attempt} attempts")
            return None

    # ── Public interface ──────────────────────────────────────────────────────

    def _is_worth_checking(self, text: str | None, has_photo: bool = False) -> bool:
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
        score = 0

        has_strong = bool(_STRONG_PATTERN.search(t))
        if has_strong:
            score += 10
        if _WEAK_PATTERN.search(t):
            score += 3
        if _WORD_BOUNDARY_KEYWORDS.search(t):
            score += 5
        if _PROMO.search(t):
            score += 2

        if '?' in t:
            score -= 5
        if _QUESTION_AMAN_PATTERN.search(t):
            score -= 8
        if t.endswith('?') and words and words[0] in _QUESTION_WORDS:
            score -= 5
        if any(w in _QUESTION_WORDS for w in words) and ('aman' in t or 'work' in t or 'on' in t):
            score -= 8
        if _QUESTION_AMAN_NGGA_PATTERN.search(t):
            score -= 15

        if has_strong and score >= 0:
            return True
        if len(words) <= 4 and score < 2:
            return False

        return score >= 2

    async def process_batch(self, messages: Sequence[dict[str, Any]], db: Any = None) -> list[PromoExtraction] | None:
        """
        Extract promos from a batch of messages using concurrent scatter-gather.

        Each chunk is sent to a DIFFERENT model concurrently, maximizing
        throughput across the entire fleet.
        """
        if not messages:
            return []

        filtered = [m for m in messages if self._is_worth_checking(m.get('text'), bool(m.get('has_photo', False)))]
        if not filtered:
            return []

        # Enrich with reply context
        if db:
            chat_id = filtered[0]['chat_id']
            reply_ids = [m['reply_to_msg_id'] for m in filtered if m.get('reply_to_msg_id')]
            reply_map = await db.get_deep_context_bulk(reply_ids, chat_id, max_depth=3) if reply_ids else {}
            for m in filtered:
                if m.get('reply_to_msg_id') and m['reply_to_msg_id'] in reply_map:
                    ctx_text = reply_map[m['reply_to_msg_id']]
                    m['context'] = f"C:{ctx_text[-800:]} "
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

        # Scatter-gather: each chunk gets its own model from the fleet
        CHUNK_SIZE = 10
        chunks = [filtered[i:i + CHUNK_SIZE] for i in range(0, len(filtered), CHUNK_SIZE)]

        # Pre-select models for all chunks (under fleet lock, atomically)
        # This ensures chunks get DIFFERENT models when possible
        selected_models: list[str] = []
        used_this_batch: list[str] = []
        for chunk in chunks:
            tokens = self._estimate_tokens("\n".join(
                f"ID:{m['id']} {m.get('context','')[:200]}MSG:{m['text'] or ''}"
                for m in chunk
            ))
            model = await self._pick_model(
                exclude=used_this_batch if len(used_this_batch) < len(self._priority_list) - 1 else None,
                estimated_tokens=tokens,
            )
            selected_models.append(model)
            if model not in used_this_batch:
                used_this_batch.append(model)

        async def _process_chunk(chunk: list[dict], model_name: str):
            chunk_text = "\n---\n".join(
                f"ID:{m['id']} {m.get('context', '')}MSG: {m['text'] or ''}"
                for m in chunk
            )
            # Slot was already acquired during _pick_model; just call
            return await self._call(
                contents=f"Batch pesan:\n\n{chunk_text}",
                config=config,
                slot_name=model_name,
            )

        tasks = [_process_chunk(c, m) for c, m in zip(chunks, selected_models)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        valid: list[PromoExtraction] = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"❌ Chunk {i} crashed: {response}")
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
                if verified_brand == "Unknown":
                    continue
                p.brand = verified_brand
                p.model_name = response.model_name
                valid.append(p)

        if valid:
            model_names = set(p.model_name for p in valid)
            logger.info(
                f"✅ Batch complete: {len(valid)} promos from {len(filtered)} msgs "
                f"via {len(chunks)} chunks | models: {model_names}"
            )
        return valid

    async def filter_duplicates(self, new_promos: Sequence[PromoExtraction],
                                recent_alerts: Sequence[dict[str, Any]]) -> list[PromoExtraction]:
        if not new_promos:
            return []

        history_tail = list(recent_alerts)[-50:]
        recent_keys = {
            f"{normalize_brand(r['brand']).lower()}:{r['summary'][:35].lower()}"
            for r in recent_alerts
        }
        recent_brands_set = {normalize_brand(r['brand']).lower() for r in history_tail}

        unique: list[PromoExtraction] = []
        intra_batch_keys: set[str] = set()
        intra_batch_by_brand: dict[str, list[set[str]]] = {}

        for p in new_promos:
            brand_key = normalize_brand(p.brand).lower()
            key = f"{brand_key}:{p.summary[:35].lower()}"

            if key in recent_keys or key in intra_batch_keys:
                continue

            p_words = set(re.findall(r'\w+', p.summary.lower())[:8])

            if brand_key in recent_brands_set and brand_key != 'unknown' and p.status == 'active':
                is_dupe = False
                for r in reversed(history_tail):
                    if normalize_brand(r['brand']).lower() == brand_key:
                        r_words = set(re.findall(r'\w+', r['summary'].lower())[:8])
                        if len(p_words & r_words) >= 2:
                            is_dupe = True
                            break
                if is_dupe:
                    continue

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
        if not texts:
            return "Tidak ada pesan."
        context = "\n---\n".join(texts)
        tokens = self._estimate_tokens(context)
        target = await self._pick_model(estimated_tokens=tokens)
        response = await self._call(
            contents=f"Rangkum pesan ini:\n\n{context}",
            config={"system_instruction": _DIGEST_SYSTEM},
            slot_name=target,
        )
        res = response.text if response else "❌ Gagal merangkum."
        return f"{res}\n\n— via {target}" if response else res

    async def summarize_thread(self, parent_text: str, replies: Sequence[str],
                               parent_photo: bytes | None = None) -> str:
        if not replies:
            return "Thread ini sedang ramai dibicarakan."
        reply_context = "\n- ".join(replies[:20])
        prompt = (
            f"PESAN UTAMA: {parent_text}\n\n"
            f"BALASAN:\n- {reply_context}\n\n"
            "Rangkum diskusi ini dalam 1-2 kalimat informatif."
        )
        contents: list[Any] = [prompt]
        if parent_photo:
            contents.append({"mime_type": "image/jpeg", "data": parent_photo})

        tokens = self._estimate_tokens(contents)
        target = await self._pick_model(
            provider="google" if parent_photo else None,
            estimated_tokens=tokens,
            require_vision=bool(parent_photo)
        )
        response = await self._call(
            contents=contents,
            config={"system_instruction": _DIGEST_SYSTEM},
            slot_name=target,
        )
        res = response.text if response else "Thread ini sedang ramai dibicarakan."
        return f"{res}\n\n— via {target}" if response else res

    async def answer_question(self, question: str, context: str) -> str:
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
        has_promo = bool(_PROMO.search(caption)) if caption else False
        has_nonpro = bool(_NON_PROMO.search(caption)) if caption else False
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

        target = await self._pick_model(
            provider="google",
            estimated_tokens=tokens,
            require_vision=True
        )

        response = await self._call(
            contents=contents,
            config=config,
            slot_name=target,
        )
        if not response or not response.parsed:
            return None
        res = response.parsed
        res.model_name = target

        verified_brand = normalize_brand(res.brand)
        if verified_brand == "Unknown":
            return None
        res.brand = verified_brand

        JUNK = {'tidak ada', 'none', 'n/a', 'tidak ada promo', 'no promo', 'tidak ditemukan', '-', 'unknown'}
        if (not res.summary or len(res.summary) < 10
                or res.summary.lower().strip() in JUNK):
            return None
        res.original_msg_id = original_msg_id
        return res

    async def generate_narrative(self, messages: Sequence[dict[str, Any]],
                                 db: Any = None) -> list[TrendItem]:
        if not messages:
            return []

        parent_map: dict[int, str] = {}
        if db is not None:
            try:
                chat_id = messages[0]['chat_id']
                reply_ids = [m['reply_to_msg_id'] for m in messages if m['reply_to_msg_id']]
                if chat_id is not None and reply_ids:
                    parent_map = await db.get_deep_context_bulk(reply_ids, chat_id, max_depth=2)
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
        target = await self._pick_model(estimated_tokens=tokens)

        config = {
            "response_mime_type": "application/json",
            "response_schema": TrendResponse,
            "system_instruction": (
                "Kamu TanyaDFBot, analis tren grup Discountfess. "
                "Simpulkan 1-3 tren utama dengan ID pesan. "
                "KATEGORI: [PROMO_BARU], [SYSTEM_EROR], [DISKUSI_HANGAT], [RESTOCK]. "
                "DILARANG TABEL. Bold untuk brand."
            ),
        }
        response = await self._call(
            contents=f"Pesan grup:\n{context}",
            config=config,
            slot_name=target
        )

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
        if not context_msgs:
            return None

        word_counts = {}
        for w in hot_words:
            word_counts[w] = sum(1 for msg in context_msgs if w.lower() in msg.lower())

        counts_str = ", ".join([f"'{w}' ({c}x)" for w, c in word_counts.items()])
        system = (
            f"Analis sentimen real-time TanyaDFBot. Lonjakan aktivitas di grup.\n"
            f"Kata kunci dominan {window} menit: {counts_str}.\n"
            "Jelaskan APA yang dibahas berdasarkan pesan-pesan berikut.\n"
            "DILARANG TABEL. Jawab 1-2 kalimat."
        )

        context_block = "\n".join([f"- {msg[:150]}" for msg in context_msgs[-40:]])
        tokens = self._estimate_tokens(system + context_block)
        target = await self._pick_model(estimated_tokens=tokens)

        response = await self._call(
            contents=f"Pesan context:\n{context_block}",
            config={"system_instruction": system},
            slot_name=target
        )
        result = response.text.strip() if response and response.text else None
        return result if result and "NO_TREND" not in result else None
