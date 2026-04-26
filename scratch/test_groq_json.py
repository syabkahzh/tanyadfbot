import asyncio
import logging
import sys
import os

# Add parent dir to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from processor import GeminiProcessor, PromoExtraction, _EXTRACT_SYSTEM

async def test_groq_json():
    logging.basicConfig(level=logging.INFO)
    print("🚀 Initializing Groq JSON Test...")
    gp = GeminiProcessor()
    
    # Realistic Indonesian promo message
    test_msg = "sfood masih on gaes, tadi baru aja garap emados dapet diskon 40k lumayan buat makan siang"
    
    config = {
        "response_mime_type": "application/json",
        "response_schema": PromoExtraction,
        "system_instruction": _EXTRACT_SYSTEM,
    }
    
    groq_units = [name for name in gp._slots if "groq" in name]
    
    if not groq_units:
        print("❌ No Groq units found in config.")
        return

    for name in groq_units:
        print(f"\n💎 Testing JSON Extraction on: {name}...")
        slot = gp._slots[name]
        
        try:
            res = await gp._call(test_msg, config, name, max_attempts=1)
            if res and res.parsed:
                print(f"✅ {name} SUCCESS!")
                print(f"📊 Extracted Data: {res.parsed.model_dump_json(indent=2)}")
            elif res:
                print(f"⚠️ {name} returned text but failed schema: {res.text[:100]}...")
            else:
                print(f"❌ {name} FAILED: Empty response")
        except Exception as e:
            print(f"❌ {name} ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(test_groq_json())
