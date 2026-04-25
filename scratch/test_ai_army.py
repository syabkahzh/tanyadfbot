import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from processor import GeminiProcessor
from config import Config

logging.basicConfig(level=logging.INFO)

async def test_init():
    print("Testing AI Army initialization...")
    try:
        processor = GeminiProcessor()
        print(f"Loaded {len(processor._slots)} slots.")
        for name, slot in processor._slots.items():
            print(f" - {name}: {slot.provider} ({slot.model_id}), RPM: {slot.limit}, Priority: {slot.priority}")
        
        print("\nPriority list:", processor._priority_list)
        
        # Test pick model
        picked = await processor._pick_model()
        print(f"\nPicked model: {picked}")
        
        # Release it
        processor._slots[picked].release_last()
        print(f"Released {picked}")
        
    except Exception as e:
        print(f"Error during initialization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_init())
