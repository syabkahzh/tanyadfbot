import asyncio
import logging
import sys
import os

# Add parent dir to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from processor import GeminiProcessor

async def test_fleet():
    logging.basicConfig(level=logging.DEBUG)
    print("🚀 Initializing AI Army...")
    gp = GeminiProcessor()
    
    test_prompt = "Say 'OK' if you are working."
    config = {"system_instruction": "You are a health check bot."}
    
    results = {}
    
    for name in gp._priority_list:
        print(f"\n📡 Testing unit: {name}...")
        slot = gp._slots[name]
        
        # Acquire slot
        if not await slot.try_acquire_nowait():
            print(f"❌ Could not acquire slot for {name}")
            continue
            
        try:
            res = await gp._call(test_prompt, config, name, max_attempts=1)
            if res and res.text:
                print(f"✅ {name} SUCCESS: {res.text.strip()}")
                results[name] = "OK"
            else:
                print(f"❌ {name} FAILED: Empty response")
                results[name] = "FAIL"
        except Exception as e:
            print(f"❌ {name} ERROR: {e}")
            results[name] = "ERROR"
        finally:
            slot.release_last()
            
    print("\n" + "="*30)
    print("📊 FLEET STATUS SUMMARY")
    print("="*30)
    for name, status in results.items():
        print(f"{name:20}: {status}")

if __name__ == "__main__":
    asyncio.run(test_fleet())
