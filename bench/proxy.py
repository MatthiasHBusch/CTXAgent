"""Transparent logging proxy between a harness and OpenRouter.

Every harness (ctxagent, Claude Code, Codex, OpenCode, Letta Code, mini-swe-agent) sends its model traffic through one of these
per trial. Requests and responses pass through unchanged (streaming preserved); the
proxy only *reads* the response, extracts every `usage` object (OpenRouter adds the
billed `cost` to all three API formats: /chat/completions, /messages, /responses) and
appends one JSON line per request to the log (with the request's tool names and reasoning
setting, so the tool list in force is recorded from the wire for every harness). This gives billed cost and per-request
token counts for every harness from one code path, independent of how each CLI reports
(or mis-reports) usage.

Provider pinning (REDESIGN.md section 5.4): for models of a pinned family the proxy
rewrites the request body before forwarding, injecting `provider: {order: [...],
allow_fallbacks: false}` so every harness hits the same upstream and the same cache
implementation. The injected block is recorded in the log (`provider_pin`); other
bodies are forwarded byte-for-byte. `UsageProxy(..., pin=False)` disables it.

    proxy = UsageProxy(Path("proxy.jsonl")); port = proxy.start()
    proxy.turn = 1   # requests are tagged with the current turn
    ...; proxy.stop()
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import aiohttp
from aiohttp import web

HOP = {"host", "content-length", "transfer-encoding", "connection", "accept-encoding", "keep-alive",
       "content-encoding"}

# Model family prefix -> OpenRouter provider slugs to pin (in order, no fallbacks).
PROVIDER_PIN = {"anthropic/": ["anthropic"], "openai/": ["openai"], "meta/": ["meta"]}


def pin_provider(body: bytes, pins: dict | None = None) -> tuple[bytes, dict | None]:
    """Inject `provider` into a JSON request body if the model belongs to a pinned family.
    Returns (new_body, injected_block_or_None); any non-JSON / non-model body is unchanged."""
    pins = PROVIDER_PIN if pins is None else pins
    try:
        j = json.loads(body) if body else None
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body, None
    if not isinstance(j, dict) or not isinstance(j.get("model"), str):
        return body, None
    model = j["model"].lstrip("~").split("[", 1)[0]
    for prefix, order in pins.items():
        if model.startswith(prefix):
            block = {"order": list(order), "allow_fallbacks": False}
            j["provider"] = {**(j.get("provider") if isinstance(j.get("provider"), dict) else {}), **block}
            return json.dumps(j, ensure_ascii=False).encode("utf-8"), j["provider"]
    return body, None


def _usages(obj, out: list) -> None:
    """Collect every dict stored under a 'usage' key, depth-first, in document order."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "usage" and isinstance(v, dict):
                out.append(v)
            else:
                _usages(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _usages(v, out)


def _first_id(obj) -> str | None:
    if isinstance(obj, dict):
        for k in ("id",):
            if isinstance(obj.get(k), str):
                return obj[k]
        for k in ("message", "response"):
            if isinstance(obj.get(k), dict) and isinstance(obj[k].get("id"), str):
                return obj[k]["id"]
    return None


def summarize(body: bytes, content_type: str) -> dict:
    """Merge all usage objects of a response (SSE or JSON) into one record."""
    docs = []
    text = body.decode("utf-8", "replace")
    if "event-stream" in content_type:
        for line in text.splitlines():
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload and payload != "[DONE]":
                    try:
                        docs.append(json.loads(payload))
                    except json.JSONDecodeError:
                        pass
    else:
        try:
            docs.append(json.loads(text))
        except json.JSONDecodeError:
            pass
    merged, rid = {}, None
    for d in docs:
        rid = rid or _first_id(d)
        found = []
        _usages(d, found)
        for u in found:
            for k, v in u.items():
                if v is not None:
                    merged[k] = v  # later events (final totals) override earlier partials
    return {"usage": merged, "id": rid, "events": len(docs)}


class UsageProxy:
    def __init__(self, log: Path, upstream: str = "https://openrouter.ai", pin: bool = True):
        self.log = Path(log)
        self.upstream = upstream.rstrip("/")
        self.pin = pin
        self.turn = 0
        self.port = None
        self._loop = None
        self._thread = None
        self._ready = threading.Event()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ server
    async def _handle(self, request: web.Request) -> web.StreamResponse:
        t0 = time.time()
        turn = self.turn
        body = await request.read()
        req_model, req_stream, pinned, req_tools, req_effort = None, None, None, None, None
        try:
            j = json.loads(body) if body else {}
            req_model, req_stream = j.get("model"), j.get("stream")
            if isinstance(j.get("tools"), list):  # tool list in force, as the model saw it
                req_tools = [(t.get("function") or {}).get("name") or t.get("name") or t.get("type")
                             for t in j["tools"] if isinstance(t, dict)]
            req_effort = (j.get("reasoning") or j.get("reasoning_effort") or j.get("thinking")
                          or (j.get("output_config") or {}).get("effort"))
        except (json.JSONDecodeError, AttributeError):
            pass
        if self.pin and request.method == "POST" and req_model:
            body, pinned = pin_provider(body)
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
        headers["Accept-Encoding"] = "identity"  # read usage without decompressing
        url = self.upstream + request.path_qs
        buf = bytearray()
        status, ctype, error = None, "", None
        resp = web.StreamResponse()
        try:
            async with self._session.request(request.method, url, data=body, headers=headers) as up:
                status, ctype = up.status, up.headers.get("Content-Type", "")
                resp.set_status(up.status)
                for k, v in up.headers.items():
                    if k.lower() not in HOP:
                        resp.headers[k] = v
                await resp.prepare(request)
                client_ok = True
                async for chunk in up.content.iter_any():
                    buf.extend(chunk)  # keep draining upstream even if the client left:
                    if client_ok:      # the usage/cost event is usually the last one
                        try:
                            await resp.write(chunk)
                        except (ConnectionError, RuntimeError) as exc:
                            client_ok, error = False, f"client gone: {type(exc).__name__}: {exc}"
                if client_ok:
                    try:
                        await resp.write_eof()
                    except (ConnectionError, RuntimeError) as exc:
                        error = f"client gone at eof: {type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001 - log and surface as 502 if not started
            error = f"{type(exc).__name__}: {exc}"
            if not resp.prepared:
                resp = web.Response(status=502, text=error)
        rec = {"ts": t0, "dur": round(time.time() - t0, 2), "turn": turn, "method": request.method,
               "path": request.path, "status": status, "model": req_model, "stream": req_stream,
               "req_bytes": len(body), "resp_bytes": len(buf), "error": error, "provider_pin": pinned}
        if req_tools is not None:
            rec["tools"] = req_tools
        if req_effort is not None:
            rec["effort"] = req_effort
        if request.method == "POST":
            rec.update(summarize(bytes(buf), ctype))
            if status and status >= 400:
                rec["error_body"] = bytes(buf[:500]).decode("utf-8", "replace")
        with self._lock, open(self.log, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return resp

    async def _main(self):
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600, sock_read=1800),
                                              auto_decompress=False)
        app = web.Application(client_max_size=256 * 1024 * 1024)
        app.router.add_route("*", "/{tail:.*}", self._handle)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]
        self._ready.set()
        self._stop = asyncio.Event()
        await self._stop.wait()
        await self._runner.cleanup()
        await self._session.close()

    def start(self) -> int:
        def run():
            self._loop = asyncio.new_event_loop()
            # Never let loop noise (Windows proactor 'connection lost' etc.) reach stderr:
            # a blocked stderr write would freeze the proxy and with it the harness.
            self._loop.set_exception_handler(self._loop_exception)
            self._loop.run_until_complete(self._main())

        self._thread = threading.Thread(target=run, daemon=True, name="usage-proxy")
        self._thread.start()
        if not self._ready.wait(30):
            raise RuntimeError("proxy did not start")
        return self.port

    def _loop_exception(self, loop, context) -> None:
        exc = context.get("exception")
        rec = {"ts": time.time(), "turn": self.turn, "method": "LOOP", "status": None,
               "error": f"{type(exc).__name__}: {exc}" if exc else str(context.get("message"))[:300]}
        try:
            with self._lock, open(self.log, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def stop(self) -> None:
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._stop.set)
            self._thread.join(15)


def read_log(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def normalize(rec: dict) -> dict | None:
    """Disjoint token categories (uncached input, cache write, cache read, output) for one
    proxied model call, from whichever API format it used. None for non-model requests."""
    u = rec.get("usage") or {}
    if rec.get("method") != "POST" or not u:
        return None
    path = rec.get("path") or ""
    if path.endswith("/messages"):  # Anthropic: input_tokens excludes cache reads/writes
        unc, cw, cr = u.get("input_tokens") or 0, u.get("cache_creation_input_tokens") or 0, \
            u.get("cache_read_input_tokens") or 0
        out = u.get("output_tokens") or 0
    elif path.endswith("/responses"):  # OpenAI Responses: input_tokens includes cached
        d = u.get("input_tokens_details") or {}
        cr, cw = d.get("cached_tokens") or 0, d.get("cache_write_tokens") or 0
        unc, out = max((u.get("input_tokens") or 0) - cr - cw, 0), u.get("output_tokens") or 0
    else:  # chat/completions: prompt_tokens includes cached and cache writes
        d = u.get("prompt_tokens_details") or {}
        cr, cw = d.get("cached_tokens") or 0, d.get("cache_write_tokens") or 0
        unc, out = max((u.get("prompt_tokens") or 0) - cr - cw, 0), u.get("completion_tokens") or 0
    return {"turn": rec.get("turn"), "model": rec.get("model"), "path": path, "uncached": unc,
            "cache_write": cw, "cache_read": cr, "output": out, "context": unc + cw + cr,
            "billed": u.get("cost") if isinstance(u.get("cost"), (int, float)) else None,
            "status": rec.get("status"), "ts": rec.get("ts")}


def billed(records: list[dict]) -> dict:
    """Billed cost and token totals per turn from proxy records (model calls only)."""
    out: dict = {}
    for r in records:
        u = r.get("usage") or {}
        if r.get("method") != "POST" or not u:
            continue
        t = out.setdefault(r["turn"], {"cost": 0.0, "requests": 0, "missing_cost": 0})
        t["requests"] += 1
        if isinstance(u.get("cost"), (int, float)):
            t["cost"] += u["cost"]
        else:
            t["missing_cost"] += 1
    return out
