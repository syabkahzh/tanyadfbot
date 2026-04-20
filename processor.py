import os
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

class BatchResponse(BaseModel):
    promos: List[PromoExtraction]

class GeminiProcessor:
    def __init__(self):
        self.client = genai.Client(api_key=Config.GEMINI_API_KEY)

    async def _call_with_retry(self, model_id, contents, config, max_retries=3):
        """Internal helper to handle API failures with exponential backoff."""
        for attempt in range(max_retries):
            try:
                response = await self.client.aio.models.generate_content(
                    model=model_id,
                    contents=contents,
                    config=config
                )
                return response
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"❌ AI API failed after {max_retries} attempts: {e}")
                    return None
                wait_time = 2 ** attempt
                print(f"⚠️ AI API error: {e}. Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(wait_time)
        return None

    async def process_batch(self, messages: List[dict]) -> List[PromoExtraction]:
        if not messages: return []
        
        batch_text = "\n---\n".join([
            f"ID: {m['id']} | Text: {m['text']}"
            for m in messages
        ])

        config = {
            "response_mime_type": "application/json",
            "response_schema": BatchResponse,
            "system_instruction": (
                "You are a highly sensitive expert at finding deals in Indonesian Telegram chats. "
                "Indonesian users use slang: 'jd' (jadi), 'cmn' (cuman), 'hrga' (harga), 'pake' (pakai). "
                "EXTRACT ANYTHING that looks like a deal, a price drop, a coupon code, or a 'deals' mention. "
                "Example: 'pake deals jd cmn 9.5k' is a PROMO for a store (unknown) at price 9.5k. "
                "If you see a link and a price, it's a promo. "
                "If you see a brand (Shopee, Grab, Foodpanda, etc) and a discount, it's a promo. "
                "For each, extract: "
                "- summary: short description (1-2 sentences) "
                "- brand: store or company name "
                "- conditions: any requirements (e.g., 'min spend') "
                "- valid_until: expiry date if mentioned "
                "- status: 'active' if it looks current. "
                "Return a JSON array. Be extremely aggressive—include anything that might be a deal."
            )
        }

        response = await self._call_with_retry(
            model_id=Config.MODEL_ID,
            contents=f"Batch of messages from an Indonesian discount group:\n\n{batch_text}",
            config=config
        )

        if not response or not response.parsed:
            return []
            
        return response.parsed.promos

    async def answer_question(self, question: str, context: str) -> str:
        # ... existing answer_question logic ...

    async def interpret_keywords(self, hot_words: List[str], window: int, context_msgs: List[str]) -> Optional[str]:
        """Interprets trending keywords using specific Indonesian context and NO_TREND safety."""
        if not context_msgs: return None

        word_str = ", ".join(hot_words)
        # Limit to 20 messages to save tokens
        context_str = "\n- ".join(context_msgs[:20]) 
        
        prompt = (
            f"Dalam {window} menit terakhir, kata-kata ini sering muncul: {word_str}\n\n"
            f"Konteks pesan asli:\n{context_str}\n\n"
            "Berdasarkan pesan-pesan asli ini, apa yang sebenarnya sedang dibahas group? "
            "Jawab dalam 2-3 kalimat bahasa Indonesia. "
            "JIKA pesan-pesan ini tidak saling berhubungan dan hanya kebetulan menggunakan kata umum, "
            "WAJIB jawab persis dengan kata: NO_TREND"
        )
        
        system = (
            "Kamu analis group promo. Jangan menebak. "
            "Jika konteks tidak menunjukkan promo/event yang jelas, ketik NO_TREND."
        )

        response = await self._call_with_retry(
            model_id=Config.MODEL_ID,
            contents=prompt,
            config={"system_instruction": system}
        )

        if response and response.text and "NO_TREND" not in response.text:
            return response.text.strip()
            
        return None
