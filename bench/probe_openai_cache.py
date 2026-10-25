"""Probe: does OpenRouter's `session_id` prevent cross-chat prompt-cache reuse on OpenAI models?

Sends the real ctxagent static prefix (system prompt + tool schemas) from chat A, then
from chat B (different workdir, different question), once with distinct session_ids and
once without session_id. Reports cache_read of B's request. ~$0.05 with GPT-6.1 Sol.
"""
import asyncio
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.run import _dotenv_keys  # noqa: E402

os.environ.update(_dotenv_keys())
from ctxagent import agent, llm, tools as toolmod  # noqa: E402
from ctxagent.config import SessionConfig  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-6.1-sol"


async def one(workdir: str, question: str, session_id: str | None) -> dict:
    cfg = SessionConfig(model=MODEL, workdirs=[workdir])
    msgs = [{"role": "system", "content": agent.build_system_prompt(cfg)}, {"role": "user", "content": question}]
    specs = toolmod.openai_schema(cfg.enabled_tools)
    usage = {}
    async for ev in llm.stream_chat(model=MODEL, messages=msgs, tools=specs, max_tokens=50, frozen_msg_count=1,
                                    reasoning_effort="low", session_id=session_id):
        if ev.get("type") == "usage":
            usage = ev["usage"]
    d = usage.get("prompt_tokens_details") or {}
    return {"prompt": usage.get("prompt_tokens"), "cache_read": d.get("cached_tokens"),
            "cache_write": d.get("cache_write_tokens")}


async def main():
    tag = uuid.uuid4().hex[:6]
    modes = sys.argv[2].split(",") if len(sys.argv) > 2 else ["distinct", "none", "shared"]
    for label in modes:
        sid = {"distinct": lambda x: f"probe-{tag}-{x}", "none": lambda x: None,
               "shared": lambda x: f"probe-{tag}-shared"}[label]
        a = await one(f"C:/probe/{tag}/{label}/a", "Say A.", sid("a"))
        time.sleep(3)
        b = await one(f"D:/probe/{tag}/{label}/b", "Say B.", sid("b"))
        time.sleep(3)
        c = await one(f"E:/probe/{tag}/{label}/c", "Say C.", sid("c"))
        print(f"{label:10} A={a}  B={b}  C={c}", flush=True)


asyncio.run(main())
