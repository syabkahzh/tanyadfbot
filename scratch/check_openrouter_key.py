import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

def check_key():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in .env")
        return

    print(f"🚀 Checking OpenRouter Key Status...")
    response = requests.get(
      url="https://openrouter.ai/api/v1/key",
      headers={
        "Authorization": f"Bearer {api_key}"
      }
    )
    
    if response.status_code == 200:
        data = response.json().get('data', {})
        print("✅ Success!")
        print(f"  • Label: {data.get('label')}")
        print(f"  • Is Free Tier: {data.get('is_free_tier')} (True means never paid)")
        print(f"  • Credit Remaining: ${data.get('limit_remaining') if data.get('limit_remaining') is not None else 'Unlimited'}")
        print(f"  • Usage (Daily): ${data.get('usage_daily')}")
        
        # In 2026, we should also check for any per-model limits if reported
        print("\nNote: Per OpenRouter docs, if you have paid < $10, you are limited to 50 free requests/day total across all free models.")
    else:
        print(f"❌ Failed to check key: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    check_key()
