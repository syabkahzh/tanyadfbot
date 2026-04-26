import asyncio
import os
from dotenv import load_dotenv
from processor import OpenAICompatibleClient, BatchResponse, PromoExtraction

load_dotenv()

async def test_openrouter():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in .env")
        return

    print("🚀 Initializing OpenRouter Test...")
    # OpenRouter uses OpenAI-compatible API
    client = OpenAICompatibleClient(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        provider="openrouter"
    )

    # Test model ID for free routing
    model_id = "openrouter/auto:free"
    
    # Simple extraction test
    content = "ID:999 MSG: sfood aman kak diskon 50k"
    
    config = {
        "response_schema": BatchResponse,
        "system_instruction": "Extract promo JSON. Output ONLY JSON.",
    }
    
    capabilities = {
        "native_json": True
    }

    try:
        print(f"🛰️ Requesting {model_id}...")
        res = await client.generate_content(
            model=model_id,
            contents=content,
            config=config,
            capabilities=capabilities
        )
        
        print(f"✅ Success! Response from {res.model_name}")
        if res.parsed:
            import json
            print(f"📊 Extracted: {json.dumps(res.parsed.model_dump(), indent=2)}")
        else:
            print(f"📝 Raw Text: {res.text}")
            
    except Exception as e:
        print(f"❌ OpenRouter Test Failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_openrouter())
