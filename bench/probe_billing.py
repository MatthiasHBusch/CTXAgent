"""Probe: are requests via OpenRouter's Anthropic (/messages) and OpenAI (/responses)
endpoints billed to the key like /chat/completions? Sends one tiny request per endpoint and
polls the key's usage counter. Cost: well below $0.01."""
import time

import httpx

from bench.run import _dotenv_keys

KEY = _dotenv_keys()["OPENROUTER_API_KEY"]
H = {"Authorization": f"Bearer {KEY}"}


def usage() -> float:
    return httpx.get("https://openrouter.ai/api/v1/key", headers=H, timeout=60).json()["data"]["usage"]


def main():
    u0 = usage()
    print("usage before", u0)
    r1 = httpx.post("https://openrouter.ai/api/v1/messages", timeout=120,
                    headers={**H, "anthropic-version": "2023-06-01", "x-api-key": KEY},
                    json={"model": "anthropic/claude-haiku-5.5", "max_tokens": 20,
                          "messages": [{"role": "user", "content": "Say OK."}]})
    print("/messages", r1.status_code, {k: r1.json().get(k) for k in ("id", "model", "usage")})
    r2 = httpx.post("https://openrouter.ai/api/v1/responses", timeout=120, headers=H,
                    json={"model": "openai/gpt-5-mini", "input": "Say OK.", "max_output_tokens": 200})
    print("/responses", r2.status_code, {k: r2.json().get(k) for k in ("id", "model", "usage")})
    r3 = httpx.post("https://openrouter.ai/api/v1/chat/completions", timeout=120, headers=H,
                    json={"model": "openai/gpt-5-mini", "messages": [{"role": "user", "content": "Say OK."}],
                          "usage": {"include": True}})
    print("/chat/completions", r3.status_code, {k: r3.json().get(k) for k in ("id", "model", "usage")})
    for i in range(12):
        time.sleep(20)
        u = usage()
        print(f"t+{(i + 1) * 20}s usage {u}  delta {u - u0:.6f}", flush=True)


main()
