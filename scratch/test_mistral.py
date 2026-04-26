import asyncio
import os
import json
from dotenv import load_dotenv
from processor import OpenAICompatibleClient, BatchResponse

load_dotenv()

async def test_mistral():
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("❌ MISTRAL_API_KEY not found in .env")
        return

    print("🚀 Initializing Mistral Test...")
    client = OpenAICompatibleClient(
        api_key=api_key,
        base_url="https://api.mistral.ai/v1",
        provider="mistral"
    )

    model_id = "mistral-small-latest" # Using latest alias for test
    
    content = "ID:777 MSG: Promo Hokben diskon 50% pake kartu sakti"
    
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
            print(f"📊 Extracted: {json.dumps(res.parsed.model_dump(), indent=2)}")
        else:
            print(f"📝 Raw Text: {res.text}")
            
    except Exception as e:
        print(f"❌ Mistral Test Failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_mistral())
