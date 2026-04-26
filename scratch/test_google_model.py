import asyncio
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

async def main():
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    model_id = "gemma-4-31b-it"
    print(f"Testing model: {model_id}")
    try:
        response = await client.aio.models.generate_content(
            model=model_id,
            contents="Hello, identify yourself."
        )
        print(f"Success: {response.text}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        print(f"Repr: {repr(e)}")

if __name__ == "__main__":
    asyncio.run(main())
