"""Fake OpenRouter for OFFLINE adapter tests (no model calls, no cost).

Serves /api/v1/chat/completions (OpenAI format), /api/v1/messages (Anthropic format, also under
/api/v1/v1/messages as Letta's pi-ai builds it) and /api/v1/models. Every reply carries a usage
object with a fake `cost`, so the real UsageProxy (started with upstream = this server, see
BENCH_UPSTREAM in bench.run) logs billed cost exactly as it does with OpenRouter.

Reply policy: if the request offers a bash-like tool and the last message is a plain user message,
answer with ONE tool call `echo fake-tool-ok`; otherwise answer with text that reports how many user
messages the request carried, so session continuation is visible in the recorded streams (for a
bash-only tool list, i.e. mini-swe-agent, the text comes with its submit command). With
FAKE_LLM_LOG=<file> every request (model, tools, reasoning fields, message roles) is appended as
JSON, which is how the flags each adapter sets are verified without a live call.

    python -m bench.fake_llm [port]      # prints the port, serves until killed
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid

from aiohttp import web

PRICE_IN, PRICE_OUT = 1e-6, 5e-6  # fake $/token


def _is_tool_result(m: dict) -> bool:
    if m.get("role") == "tool":
        return True
    c = m.get("content")
    return isinstance(c, list) and any(isinstance(p, dict) and p.get("type") == "tool_result" for p in c)


def _n_user(messages: list) -> int:
    return sum(1 for m in messages if m.get("role") == "user" and not _is_tool_result(m))


def _bash_tool(tools) -> str | None:
    for t in tools or []:
        name = (t.get("function") or {}).get("name") or t.get("name")
        if name in ("bash", "Bash", "shell", "exec_command"):
            return name
    return None


def _want_tool(messages: list, tools) -> bool:
    last = messages[-1] if messages else {}
    return bool(_bash_tool(tools)) and last.get("role") == "user" and not _is_tool_result(last)


def _log(req: web.Request, body: dict) -> None:
    path = os.environ.get("FAKE_LLM_LOG")
    if not path:
        return
    msgs = body.get("messages") or []
    rec = {"ts": time.time(), "path": req.path, "model": body.get("model"), "stream": body.get("stream"),
           "max_tokens": body.get("max_tokens"), "reasoning": body.get("reasoning"),
           "reasoning_effort": body.get("reasoning_effort"), "thinking": body.get("thinking"),
           "tools": [((t.get("function") or {}).get("name") or t.get("name")) for t in body.get("tools") or []],
           "roles": [m.get("role") for m in msgs], "n_user": _n_user(msgs),
           "system_chars": len(json.dumps(body.get("system") or "")) if "system" in body else None,
           "headers": {k: v for k, v in req.headers.items()
                       if k.lower() in ("anthropic-beta", "anthropic-version", "http-referer", "x-title", "user-agent")}}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _usage_openai(prompt: int, completion: int) -> dict:
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion,
            "prompt_tokens_details": {"cached_tokens": max(prompt - 500, 0)},
            "cost": round(prompt * PRICE_IN + completion * PRICE_OUT, 8), "is_byok": False}


async def chat(req: web.Request):
    b = await req.json()
    _log(req, b)
    msgs, tools = b.get("messages") or [], b.get("tools")
    prompt = sum(len(json.dumps(m)) for m in msgs) // 4 + 300
    rid, model = "gen-fake-" + uuid.uuid4().hex[:12], b.get("model", "fake")
    if _want_tool(msgs, tools):
        args = json.dumps({"command": "echo fake-tool-ok", "description": "fake"})
        tc = {"index": 0, "id": "call_" + uuid.uuid4().hex[:8], "type": "function",
              "function": {"name": _bash_tool(tools), "arguments": args}}
        delta, finish, text, completion = {"role": "assistant", "content": None, "tool_calls": [tc]}, "tool_calls", None, 40
    elif [((t.get("function") or {}).get("name")) for t in tools or []] == ["bash"]:
        # bash-only harness (mini-swe-agent): answer in text AND submit, as its prompt demands
        text = f"FAKE ANSWER user_messages={_n_user(msgs)}"
        tc = {"index": 0, "id": "call_" + uuid.uuid4().hex[:8], "type": "function", "function": {
            "name": "bash", "arguments": json.dumps({"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"})}}
        delta, finish, completion = {"role": "assistant", "content": text, "tool_calls": [tc]}, "tool_calls", 30
    else:
        text = f"FAKE ANSWER user_messages={_n_user(msgs)}"
        delta, finish, completion = {"role": "assistant", "content": text}, "stop", 12
    usage = _usage_openai(prompt, completion)
    if b.get("stream"):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(req)
        base = {"id": rid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model}
        for chunk in ({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                      {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
                      {**base, "choices": [], "usage": usage}):
            await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp
    msg: dict = {"role": "assistant", "content": text}
    if delta.get("tool_calls"):
        msg["tool_calls"] = [{k: v for k, v in tc.items() if k != "index"} for tc in delta["tool_calls"]]
    return web.json_response({"id": rid, "object": "chat.completion", "created": int(time.time()), "model": model,
                              "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage})


async def messages(req: web.Request):
    b = await req.json()
    _log(req, b)
    msgs, tools = b.get("messages") or [], b.get("tools")
    prompt = (len(json.dumps(b.get("system") or "")) + sum(len(json.dumps(m)) for m in msgs)) // 4 + 300
    rid, model = "msg_fake_" + uuid.uuid4().hex[:12], b.get("model", "fake")
    cache_read, cache_write, unc = max(prompt - 500, 0), 400, 100
    if _want_tool(msgs, tools):
        block = {"type": "tool_use", "id": "toolu_" + uuid.uuid4().hex[:10], "name": _bash_tool(tools),
                 "input": {"command": "echo fake-tool-ok", "description": "fake"}}
        stop, completion = "tool_use", 40
    else:
        block = {"type": "text", "text": f"FAKE ANSWER user_messages={_n_user(msgs)}"}
        stop, completion = "end_turn", 12
    usage = {"input_tokens": unc, "cache_creation_input_tokens": cache_write, "cache_read_input_tokens": cache_read,
             "output_tokens": completion, "cost": round(prompt * PRICE_IN + completion * PRICE_OUT, 8)}
    if not b.get("stream"):
        return web.json_response({"id": rid, "type": "message", "role": "assistant", "model": model, "content": [block],
                                  "stop_reason": stop, "usage": usage})
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(req)

    async def ev(name: str, data: dict):
        await resp.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())

    await ev("message_start", {"type": "message_start", "message": {
        "id": rid, "type": "message", "role": "assistant", "model": model, "content": [], "stop_reason": None,
        "usage": {**usage, "output_tokens": 1}}})
    if block["type"] == "text":
        await ev("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
        await ev("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": block["text"]}})
    else:
        await ev("content_block_start", {"type": "content_block_start", "index": 0,
                                         "content_block": {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}})
        await ev("content_block_delta", {"type": "content_block_delta", "index": 0,
                                         "delta": {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}})
    await ev("content_block_stop", {"type": "content_block_stop", "index": 0})
    await ev("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None}, "usage": usage})
    await ev("message_stop", {"type": "message_stop"})
    await resp.write_eof()
    return resp


async def models(req: web.Request):
    ids = ("anthropic/claude-haiku-5.5", "anthropic/claude-haiku-4.5", "anthropic/claude-opus-5.5",
           "openai/gpt-5-mini", "openai/gpt-6.1-sol")
    return web.json_response({"object": "list", "data": [
        {"id": m, "object": "model", "name": m, "context_length": 200000,
         "pricing": {"prompt": str(PRICE_IN), "completion": str(PRICE_OUT)},
         "supported_parameters": ["reasoning", "tools", "max_tokens"]} for m in ids]})


async def other(req: web.Request):
    try:
        _log(req, await req.json())
    except Exception:  # noqa: BLE001
        _log(req, {})
    return web.json_response({"error": {"message": f"fake: no route for {req.method} {req.path}"}}, status=404)


class FakeLLM:
    """In-process server for tests: `with FakeLLM() as f: ... f.url`."""

    def __init__(self, port: int = 0):
        self.port, self.url = port, None
        self._thread = None

    def __enter__(self):
        import threading
        ready = threading.Event()

        def run():
            self._loop = asyncio.new_event_loop()
            runner = _runner()
            self._loop.run_until_complete(runner.setup())
            site = web.TCPSite(runner, "127.0.0.1", self.port)
            self._loop.run_until_complete(site.start())
            self.port = site._server.sockets[0].getsockname()[1]
            self.url = f"http://127.0.0.1:{self.port}"
            ready.set()
            self._stop = asyncio.Event()
            self._loop.run_until_complete(self._stop.wait())
            self._loop.run_until_complete(runner.cleanup())

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        ready.wait(30)
        return self

    def __exit__(self, *exc):
        self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(10)


def _runner() -> web.AppRunner:
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app.router.add_post("/api/v1/chat/completions", chat)
    app.router.add_post("/api/v1/messages", messages)
    app.router.add_post("/api/v1/v1/messages", messages)
    app.router.add_get("/api/v1/models", models)
    app.router.add_route("*", "/{tail:.*}", other)
    return web.AppRunner(app, access_log=None)


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    loop = asyncio.new_event_loop()
    runner = _runner()
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, "127.0.0.1", port)
    loop.run_until_complete(site.start())
    print(site._server.sockets[0].getsockname()[1], flush=True)
    loop.run_forever()


if __name__ == "__main__":
    main()
