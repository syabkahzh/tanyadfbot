import os
import asyncio
from google import genai
from dotenv import load_dotenv

load_dotenv()

async def test():
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    model_id = "gemma-4-31b-it" # Or "models/gemma-4-31b-it"
    try:
        print(f"Testing model {model_id}...")
        res = await client.aio.models.generate_content(
            model=model_id,
            contents="Hello"
        )
        print(f"Response: {res.text}")
    except Exception as e:
        print(f"Error with {model_id}: {e}")

    model_id = "gemini-1.5-flash"
    try:
        print(f"Testing model {model_id}...")
        res = await client.aio.models.generate_content(
            model=model_id,
            contents="Hello"
        )
        print(f"Response: {res.text}")
    except Exception as e:
        print(f"Error with {model_id}: {e}")

asyncio.run(test())
