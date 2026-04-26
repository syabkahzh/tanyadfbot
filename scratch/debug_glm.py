import asyncio
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

async def test_glm():
    client = AsyncOpenAI(
        api_key=os.getenv("GLM_API_KEY"),
        base_url="https://api.z.ai/api/paas/v4"
    )
    
    model = "glm-4.7-flash"
    print(f"Testing model: {model} at {client.base_url}")
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}]
        )
        print("✅ SUCCESS!")
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(test_glm())
