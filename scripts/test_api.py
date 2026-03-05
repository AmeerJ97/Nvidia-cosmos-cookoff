#!/usr/bin/env python3
"""
Quick API key + model availability test.
Run: python test_api.py --key nvapi-YOUR-KEY
Or:  NGC_API_KEY=nvapi-YOUR-KEY python test_api.py
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import aiohttp

BASE_URL = "https://integrate.api.nvidia.com/v1"

MODELS_TO_TEST = [
    "nvidia/cosmos-reason2-8b",
    "meta/llama-3.2-11b-vision-instruct",
    "meta/llama-3.1-8b-instruct",
]


async def test_model(session, key, model):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say: READY"}],
        "max_tokens": 5, "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        async with session.post(
            f"{BASE_URL}/chat/completions", json=payload, headers=headers,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                reply = data["choices"][0]["message"]["content"]
                print(f"  ✓ {model}: {reply.strip()}")
                return True
            else:
                print(f"  ✗ {model}: HTTP {resp.status}")
                return False
    except Exception as e:
        print(f"  ✗ {model}: {e}")
        return False


async def main(key):
    print(f"Testing NIM API key: {key[:12]}...{key[-4:]}")
    async with aiohttp.ClientSession() as session:
        for model in MODELS_TO_TEST:
            await test_model(session, key, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", default=os.environ.get("NGC_API_KEY", ""))
    args = parser.parse_args()
    if not args.key:
        print("Provide key via --key or NGC_API_KEY env var")
        sys.exit(1)
    asyncio.run(main(args.key))
