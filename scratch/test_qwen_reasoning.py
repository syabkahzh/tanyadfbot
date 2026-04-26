
import asyncio
import os
import sys
import time
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processor import GeminiProcessor

async def test_qwen_reasoning():
    load_dotenv()
    gp = GeminiProcessor()
    
    model_name = "groq-qwen-32b"
    slot = gp._slots.get(model_name)
    if not slot:
        print(f"Model {model_name} not found.")
        return

    tasks = [
        ("Simple", "Say 'Hello' and nothing else."),
        ("Complex", "How many 'r's are in the word 'strawberry'? Explain your reasoning step by step.")
    ]

    for task_name, prompt in tasks:
        print(f"\n--- Testing {task_name} Task ---")
        start_time = time.monotonic()
        
        # We want to see the RAW text before re.sub strips the think tags
        # So we'll reach into the client directly
        try:
            capabilities = slot.capabilities
            messages = [{"role": "user", "content": prompt}]
            
            kwargs = {}
            reasoning_mode = capabilities.get("reasoning")
            if reasoning_mode:
                kwargs["extra_body"] = {"reasoning_format": reasoning_mode}
            
            print(f"Requesting with extra_body: {kwargs.get('extra_body')}")
            
            res = await slot.client.client.chat.completions.create(
                model=slot.model_id,
                messages=messages,
                **kwargs
            )
            
            duration = time.monotonic() - start_time
            raw_text = res.choices[0].message.content
            
            print(f"Duration: {duration:.2f}s")
            print(f"Raw Text Preview: {raw_text[:200]}...")
            
            if "<think>" in raw_text:
                print("🔍 Found <think> tags in output!")
            else:
                print("🚫 No <think> tags found in output.")
                
            # Check if Groq returns usage with reasoning tokens
            usage = getattr(res, 'usage', None)
            if usage:
                print(f"Usage: {usage}")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_qwen_reasoning())
