"""FastAPI server + SSE event stream for the GUI."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
import json
import time
import hashlib
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Literal

from . import agent, llm, providers as provmod, readers, store, tools as toolmod
from .config import APP_DIR, BACKEND_BUILD, SessionConfig, anthropic_api_key, api_key, ensure_dirs, key_problem, load_env
from . import defaults, delegation

WEB_DIR = APP_DIR / "web"
SERVER_STARTED_AT = time.time()

def source_fingerprint():
    digest = hashlib.sha256()
    for file in sorted((APP_DIR / 'ctxagent').rglob('*.py')):
        digest.update(str(file.relative_to(APP_DIR)).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()[:16]

LOADED_SOURCE_FINGERPRINT = source_fingerprint()

app = FastAPI(title="ctxagent")


@app.on_event("startup")
async def _startup() -> None:
    load_env()
    ensure_dirs()
    store.init()
    delegation.init()
    delegation.reap()
    from .defaults import apply_pending
    apply_pending()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class NewChat(BaseModel):
    title: str | None = None
    config: dict[str, Any] | None = None
    source_chat: str | None = None


class ConfigPatch(BaseModel):
    config: dict[str, Any]


class SendMessage(BaseModel):
    message: str


class CompactRequest(BaseModel):
    instructions: str = ""


class BrowseRequest(BaseModel):
    path: str = ""


class DocumentOpen(BaseModel):
    path: str
    select: str | list[str] | None = None
    max_tokens: int = Field(default=8000, ge=1, le=20000)


class DocumentClose(BaseModel):
    summary: str = ""


# ---------------------------------------------------------------------------
# Chats
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health() -> dict:
    return {
        "ok": True,
        "backend_build": BACKEND_BUILD,
        "server_pid": os.getpid(),
        "server_started_at": SERVER_STARTED_AT,
        "loaded_source_fingerprint": LOADED_SOURCE_FINGERPRINT,
        "restart_required": source_fingerprint() != LOADED_SOURCE_FINGERPRINT,
        "accounting_version": 2,
        "has_key": bool(api_key() or anthropic_api_key()),
        "has_openrouter_key": bool(api_key()),
        "has_anthropic_key": bool(anthropic_api_key()),
        "tools": toolmod.catalogue(),
        "default_tools": defaults.get()["enabled_tools"],
        "defaults": defaults.get(),
        "static_levels": readers.STATIC_LEVELS,
        "llm_levels": {str(k): v for k, v in readers.LLM_LEVEL_TOKENS.items()},
        "levels_text": "static: structural depth 0-5 · llm: "
                       + readers.describe_llm_levels(),
    }


@app.get("/api/chats")
async def get_chats() -> list[dict]:
    return store.list_chats()


@app.post("/api/chats")
async def new_chat(body: NewChat) -> dict:
    try:
        if body.source_chat is not None:
            source = store.get_chat(body.source_chat)
            if source is None:
                raise HTTPException(404, "source chat no longer exists")
            # Normalize the full saved snapshot; never copy history or runtime state.
            saved = SessionConfig.from_dict(source["config"]).to_dict()
            cfg = SessionConfig.from_dict(defaults.merged(saved, body.config))
        else:
            cfg = defaults.create_config(body.config)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc))
    cid = store.create_chat(body.title or "New chat", cfg.to_dict())
    return {"id": cid}


@app.get("/api/chats/{cid}")
async def read_chat(cid: str) -> dict:
    row = store.get_chat(cid)
    if not row:
        raise HTTPException(404, "no such chat")
    s = agent.get_session(cid)
    return {
        "id": row["id"],
        "title": row["title"],
        "config": row["config"],
        "segments": [seg.to_dict() for seg in s.conv.segments] if s else [],
        "compaction_log": row["compaction_log"],
        "usage": row["usage"],
        "stats": s.stats() if s else None,
        "running": bool(s and s.turn_lock.locked()) or delegation.active_chat(cid),
        "delegated": bool(delegation.for_chat(cid)),
    }


@app.delete("/api/chats/{cid}")
async def remove_chat(cid: str) -> dict:
    if delegation.active_chat(cid) or any(a["state"] not in delegation.TERMINAL for a in delegation.list_agents(cid, 100)):
        raise HTTPException(409, "Cancel/wait for delegated agents before deleting this chat")
    session = agent.get_session(cid)
    if session and session.turn_lock.locked():
        raise HTTPException(409, "Stop the active turn before deleting this chat")
    delegation.archive_chat_usage(cid)
    agent.drop_session(cid)
    store.delete_chat(cid)
    return {"ok": True}


@app.patch("/api/chats/{cid}/config")
async def patch_config(cid: str, body: ConfigPatch) -> dict:
    _ordinary_chat(cid)
    s = agent.get_session(cid)
    if not s:
        raise HTTPException(404, "no such chat")
    if s.turn_lock.locked():
        raise HTTPException(409, "Wait for the active turn before changing configuration")
    merged = s.cfg.to_dict()
    for k, v in body.config.items():
        if k == "compaction" and isinstance(v, dict):
            merged["compaction"].update(v)
        else:
            merged[k] = v
    try:
        cfg = SessionConfig.from_dict(merged)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc))
    s.cfg = cfg
    s.engine.cfg = s.cfg.compaction
    s._sync_system()
    s.persist()
    return {"config": s.cfg.to_dict(), "stats": s.stats()}


@app.post("/api/chats/{cid}/defaults")
async def save_defaults(cid: str) -> dict:
    # Copy SAVED settings, not partially edited browser controls.
    row = store.get_chat(cid)
    if not row:
        raise HTTPException(404, "no such chat")
    return {"defaults": defaults.save(row["config"])}


@app.patch("/api/chats/{cid}/title")
async def patch_title(cid: str, body: NewChat) -> dict:
    store.save_chat(cid, title=body.title or "Untitled")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Streaming turn
# ---------------------------------------------------------------------------

def _sse(ev: dict) -> str:
    return f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"


@app.post("/api/chats/{cid}/send")
async def send(cid: str, body: SendMessage) -> StreamingResponse:
    _ordinary_chat(cid)
    s = agent.get_session(cid)
    if not s:
        raise HTTPException(404, "no such chat")
    if s.turn_lock.locked():
        raise HTTPException(409, "A turn is already running")
    if problem := key_problem(s.cfg.model):
        raise HTTPException(400, problem)
    if s.cfg.compaction.enabled and (problem := key_problem(s.cfg.compaction.summary_model)):
        raise HTTPException(400, f"{problem} (needed by the compaction model {s.cfg.compaction.summary_model})")

    # First user message names the chat.
    row = store.get_chat(cid)
    if row and row["title"] in ("New chat", "Untitled"):
        t = " ".join(body.message.split())[:60]
        if t:
            store.save_chat(cid, title=t)

    async def gen():
        try:
            async with aclosing(s.run_turn(body.message)) as stream:
                async for ev in stream:
                    yield _sse(ev)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            yield _sse({"type": "error", "error": f"{type(e).__name__}: {e}"})
        yield _sse({"type": "end"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/chats/{cid}/stop")
async def stop(cid: str) -> dict:
    task = delegation.for_chat(cid)
    if task:
        return {"ok": True, "agent": await asyncio.to_thread(delegation.cancel, task["id"], task["parent"])}
    s = agent.get_session(cid)
    if s:
        s.cancel.set()
        task = getattr(s, "browser_task", None)
        if task and not task.done():
            task.cancel()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Delegated workers (owned by parent/root; browser inspection never owns their turns)
# ---------------------------------------------------------------------------

class DelegateRequest(BaseModel):
    task: str
    context: str = ""
    overrides: dict[str, Any] = Field(default_factory=dict)
    inherit_documents: bool = True
    read_only: bool = False
    timeout: int | None = Field(default=None, ge=1, le=604800)


class AgentFollowup(BaseModel):
    message: str
    overrides: dict[str, Any] = Field(default_factory=dict)
    timeout: int | None = Field(default=None, ge=1, le=604800)


def _agent_parent(cid, permission):
    s = agent.get_session(cid)
    if not s:
        raise HTTPException(404, "No such chat")
    if permission not in s.cfg.enabled_tools:
        raise HTTPException(403, "Enable " + permission + " first")
    return s


@app.get("/api/chats/{cid}/agents")
async def delegated_list(cid: str):
    if not store.get_chat(cid):
        raise HTTPException(404, "No such chat")
    return {"agents": await asyncio.to_thread(delegation.list_agents, cid),
            "cost": await asyncio.to_thread(delegation.cost_summary, cid)}


@app.post("/api/chats/{cid}/agents")
async def delegated_start(cid: str, body: DelegateRequest):
    _ordinary_chat(cid)
    s = _agent_parent(cid, "agent_start")
    if s.turn_lock.locked():
        raise HTTPException(409, "Wait for the parent turn before manual delegation")
    if problem := key_problem(s.cfg.model):
        raise HTTPException(400, problem)
    async with s.turn_lock:
        try:
            return await asyncio.to_thread(delegation.start, s, body.task, body.context, body.overrides, body.inherit_documents, body.read_only, body.timeout)
        except (ValueError, TypeError, PermissionError) as exc:
            raise HTTPException(422, str(exc))


@app.get("/api/chats/{cid}/agents/{aid}")
async def delegated_status(cid: str, aid: str):
    try:
        return await asyncio.to_thread(delegation.status, aid, cid)
    except (ValueError, PermissionError) as exc:
        raise HTTPException(404, str(exc))


@app.get("/api/chats/{cid}/agents/{aid}/result")
async def delegated_result(cid: str, aid: str, offset: int = 0, max_chars: int = 12000):
    try:
        return await asyncio.to_thread(delegation.result, aid, cid, offset, max_chars)
    except (ValueError, PermissionError) as exc:
        raise HTTPException(422, str(exc))


@app.post("/api/chats/{cid}/agents/{aid}/cancel")
async def delegated_cancel(cid: str, aid: str):
    _agent_parent(cid, "agent_cancel")
    try:
        return await asyncio.to_thread(delegation.cancel, aid, cid)
    except (ValueError, PermissionError) as exc:
        raise HTTPException(404, str(exc))


@app.post("/api/chats/{cid}/agents/{aid}/send")
async def delegated_followup(cid: str, aid: str, body: AgentFollowup):
    s = _agent_parent(cid, "agent_send")
    if problem := key_problem(s.cfg.model):
        raise HTTPException(400, problem)
    try:
        return await asyncio.to_thread(delegation.send, aid, cid, body.message, body.overrides, body.timeout)
    except (ValueError, TypeError, PermissionError) as exc:
        raise HTTPException(422, str(exc))


# ---------------------------------------------------------------------------
# Context / compaction
# ---------------------------------------------------------------------------

def _ordinary_chat(cid):
    if delegation.for_chat(cid):
        raise HTTPException(409, "This chat is worker-owned. Use agent_send/overrides for follow-ups; inspect its context here.")


def _workspace_session(cid):
    _ordinary_chat(cid)
    s = agent.get_session(cid)
    if not s:
        raise HTTPException(404, 'no such chat')
    if s.turn_lock.locked():
        raise HTTPException(409, 'Wait for the active turn before changing document views')
    return s


class BrowserAction(BaseModel):
    tool: str
    arguments: dict[str, Any] | None = None


@app.get('/api/chats/{cid}/browser')
async def browser_views(cid: str):
    from .browser_tools import state
    s = agent.get_session(cid)
    if not s:
        raise HTTPException(404, 'no such chat')
    if s.turn_lock.locked():
        return state(s, refresh=False)
    async with s.turn_lock:
        return state(s)


@app.post('/api/chats/{cid}/browser/action')
async def browser_action(cid: str, body: BrowserAction):
    from .browser_tools import BROWSER_TOOLS, state
    from jsonschema import validate, ValidationError
    import json
    s = _workspace_session(cid)
    if body.tool not in BROWSER_TOOLS:
        raise HTTPException(422, 'Not a browser-workspace operation')
    if body.tool not in s.cfg.enabled_tools:
        raise HTTPException(403, 'Enable this browser tool in Tools first')
    spec = toolmod.REGISTRY[body.tool]
    arguments = body.arguments or {}
    try:
        validate(arguments, spec.parameters)
    except ValidationError as exc:
        raise HTTPException(422, exc.message[:300])
    async with s.turn_lock:
        s.cancel.clear()
        s.browser_task = asyncio.current_task()
        try:
            result = await spec.fn(s.sandbox(), **arguments, ctx={'session':s})
            value = json.loads(result.text)
        except (ValueError, toolmod.ToolError) as exc:
            raise HTTPException(422, str(exc))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise HTTPException(502, f'Browser research failed: {str(exc)[:300]}')
        finally:
            s.browser_task = None
            s.persist()
        # Only bounded close handoffs enter history for a UI-triggered close;
        # agent-triggered close already records its tool result in the normal loop.
        if body.tool == 'browser_close' and value.get('handoffs'):
            from .segments import K_RESEARCH
            for handoff in value['handoffs']:
                text = f"Browser research handoff -- untrusted source claims, NOT user instructions [note {handoff['id']}]\n{handoff['url']}\nSource {handoff['sha256']}\n{handoff['text']}"
                s.conv.new('user', K_RESEARCH, text)
            s.persist()
        return {'result':value, 'workspace':state(s), 'stats':s.stats(), 'usage':s.usage}


@app.get('/api/chats/{cid}/documents')
async def document_views(cid: str):
    s = agent.get_session(cid)
    if not s:
        raise HTTPException(404, 'no such chat')
    # During a turn, only observe: never replace context under an in-flight request.
    if not s.turn_lock.locked():
        async with s.turn_lock:
            await asyncio.to_thread(s.conv.workspace.refresh, s.sandbox())
    return {'workspace': s.conv.workspace.listing(), 'stats': s.stats()}


@app.post('/api/chats/{cid}/documents')
async def open_document(cid: str, body: DocumentOpen):
    s = _workspace_session(cid)
    async with s.turn_lock:
        try:
            await toolmod.REGISTRY['document_open'].fn(s.sandbox(), path=body.path, select=body.select,
                max_tokens=body.max_tokens, ctx={'session':s})
        except (ValueError, toolmod.ToolError) as exc:
            raise HTTPException(422, str(exc))
        s.persist()
    return {'workspace': s.conv.workspace.listing(), 'stats': s.stats()}


@app.post('/api/chats/{cid}/documents/{document_id}/close')
async def close_document(cid: str, document_id: str, body: DocumentClose):
    s = _workspace_session(cid)
    async with s.turn_lock:
        try:
            spec = toolmod.REGISTRY['document_close']
            await spec.fn(s.sandbox(), document_id=document_id, summary=body.summary, ctx={'session': s})
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        s.persist()
    return {'workspace': s.conv.workspace.listing(), 'stats': s.stats()}


@app.get("/api/chats/{cid}/context")
async def context(cid: str) -> dict:
    s = agent.get_session(cid)
    if not s:
        raise HTTPException(404, "no such chat")
    hot_start = s.conv.frozen_upto  # actual boundary, including pending linked review
    segs = []
    for i, seg in enumerate(s.conv.segments):
        segs.append({
            "id": seg.id,
            "seq": seg.seq,
            "kind": seg.kind,
            "tool_name": seg.tool_name,
            "resource_key": seg.resource_key,
            "state": seg.state,
            "note": seg.note,
            "frozen": seg.frozen,
            "pinned": seg.pinned,
            "significance": round(seg.significance, 3),
            "tokens": seg.tokens(),
            "original_tokens": seg.original_tokens,
            "hot": i >= hot_start,
            "preview": seg.preview(),
        })
    return {
        "stats": s.stats(),
        "segments": segs,
        "hot_start": hot_start,
        "compaction_log": s.compaction_log[-50:],
        "usage": s.usage,
        "config": s.cfg.compaction.to_dict(),
    }


@app.post("/api/chats/{cid}/compact")
async def compact(cid: str, body: CompactRequest) -> StreamingResponse:
    _ordinary_chat(cid)
    s = agent.get_session(cid)
    if not s:
        raise HTTPException(404, "no such chat")

    async def gen():
        try:
            async with aclosing(s.compact_now(body.instructions)) as stream:
                async for ev in stream:
                    yield _sse(ev)
        except Exception as e:
            yield _sse({"type": "error", "error": f"{type(e).__name__}: {e}"})
        yield _sse({"type": "end"})

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/chats/{cid}/segments/{sid}/restore")
async def restore_segment(cid: str, sid: str) -> dict:
    _ordinary_chat(cid)
    s = agent.get_session(cid)
    if not s:
        raise HTTPException(404, "no such chat")
    if s.turn_lock.locked():
        raise HTTPException(409, "Wait for the active turn")
    seg = s.conv.by_id(sid)
    if not seg:
        raise HTTPException(404, "no such segment")
    from .blobs import get_blob

    if seg.blob_id:
        blob = get_blob(seg.blob_id)
        if blob is not None:
            seg.original_body = blob
    for item in s.conv.segments:
        item.frozen = False
    s.conv.frozen_upto = 0
    seg.restore()
    seg.pin_scope = "manual"
    s.persist()
    return {"ok": True, "stats": s.stats()}


@app.post("/api/chats/{cid}/segments/{sid}/pin")
async def pin_segment(cid: str, sid: str) -> dict:
    _ordinary_chat(cid)
    s = agent.get_session(cid)
    if not s:
        raise HTTPException(404, "no such chat")
    if s.turn_lock.locked():
        raise HTTPException(409, "Wait for the active turn")
    seg = s.conv.by_id(sid)
    if not seg:
        raise HTTPException(404, "no such segment")
    seg.pinned = not seg.pinned
    seg.pin_scope = "manual"
    s.persist()
    return {"ok": True, "pinned": seg.pinned}


@app.get("/api/chats/{cid}/segments/{sid}/full")
async def segment_full(cid: str, sid: str) -> dict:
    s = agent.get_session(cid)
    if not s:
        raise HTTPException(404, "no such chat")
    if s.turn_lock.locked():
        raise HTTPException(409, "Wait for the active turn")
    seg = s.conv.by_id(sid)
    if not seg:
        raise HTTPException(404, "no such segment")
    from .blobs import get_blob

    body = seg.original_body
    if seg.blob_id and not body:
        body = get_blob(seg.blob_id)
    return {"current": seg.body, "original": body or seg.body}


# ---------------------------------------------------------------------------
# Filesystem browser (for picking working directories)
# ---------------------------------------------------------------------------

@app.post("/api/browse")
async def browse(body: BrowseRequest) -> dict:
    p = body.path.strip()
    if not p:
        roots = []
        home = Path.home()
        roots.append({"name": str(home), "path": str(home)})
        if os.name == "nt":
            import string
            from ctypes import windll  # type: ignore

            bits = windll.kernel32.GetLogicalDrives()
            for i, letter in enumerate(string.ascii_uppercase):
                if bits & (1 << i):
                    roots.append({"name": f"{letter}:\\", "path": f"{letter}:\\"})
        else:
            roots.append({"name": "/", "path": "/"})
        return {"path": "", "parent": None, "dirs": roots}

    d = Path(p)
    if not d.exists() or not d.is_dir():
        raise HTTPException(400, f"not a directory: {p}")
    dirs = []
    try:
        for e in sorted(d.iterdir(), key=lambda x: x.name.lower()):
            if e.is_dir() and not e.name.startswith("$"):
                dirs.append({"name": e.name, "path": str(e)})
    except PermissionError:
        raise HTTPException(403, "permission denied")
    return {
        "path": str(d),
        "parent": str(d.parent) if d.parent != d else None,
        "dirs": dirs,
    }


class ProviderCheck(BaseModel):
    model: str
    policy: str = "eu_us"


@app.post("/api/providers/check")
async def providers_check(body: ProviderCheck) -> dict:
    """Can this model be served under this region policy?"""
    return await provmod.check_model(body.model, body.policy)


class FlexCheck(BaseModel):
    model: str = Field(min_length=1, max_length=256)
    policy: Literal['eu_us', 'any'] = 'eu_us'
    prompt_tokens: int = Field(default=0, ge=0, le=10_000_000)
    needs_tools: bool = False
    refresh: bool = False


@app.post('/api/providers/flex')
async def flex_check(body: FlexCheck) -> dict:
    """Metadata-only discovery. No model generation or prompt transmission."""
    from .flex import discover
    return await discover(body.model, body.policy, prompt_tokens=body.prompt_tokens,
                          needs_tools=body.needs_tools, refresh=body.refresh)


@app.get("/api/providers")
async def providers_list() -> dict:
    provs = await provmod.providers()
    allowed = [p for p in provs if provmod.region_ok(p)]
    return {
        "total": len(provs),
        "allowed": sorted(
            ({"slug": p["slug"], "name": p.get("name"),
              "hq": p.get("headquarters"),
              "datacenters": p.get("datacenters")} for p in allowed),
            key=lambda p: p["slug"],
        ),
        "regions": sorted(provmod.EU_US),
    }



@app.get('/api/chats/{cid}/artifacts/{aid}')
async def image_artifact(cid: str, aid: str):
    from .artifacts import artifact_path
    session = agent.get_session(cid)
    if session is None:
        raise HTTPException(404, 'No such chat')
    # Artifact possession alone is not authority: it must be referenced by this chat.
    allowed = any(image.get('id') == aid for seg in session.conv for image in seg.images)
    allowed = allowed or any(image.get('id') == aid for view in session.conv.browser.views for image in view.get('images', []))
    if not allowed:
        for sid in session.conv.archive:
            seg = session.conv.archived_by_id(sid)
            if seg and any(image.get('id') == aid for image in seg.images):
                allowed = True
                break
    if not allowed:
        raise HTTPException(404, 'No such artifact in this chat')
    try:
        path = artifact_path(aid)
    except ValueError:
        raise HTTPException(404, 'Invalid artifact')
    if not path.is_file():
        raise HTTPException(404, 'Artifact missing from disk')
    return FileResponse(path, media_type='image/png', headers={'X-Content-Type-Options': 'nosniff'})


@app.get("/api/models")
async def models() -> dict:
    try:
        data = await llm.list_models()
    except Exception as e:
        return {"error": str(e), "models": []}
    out = []
    for m in data:
        ctx = m.get("context_length") or 0
        pricing = m.get("pricing") or {}
        out.append({
            "id": m.get("id"),
            "name": m.get("name"),
            "provider": m.get("provider", "openrouter"),
            "context_length": ctx,
            "prompt_price": pricing.get("prompt"),
            "completion_price": pricing.get("completion"),
            "input_modalities": (m.get("architecture") or {}).get("input_modalities", []),
        })
    out.sort(key=lambda m: (m["id"] or ""))
    return {"models": out}


# ---------------------------------------------------------------------------
# Static UI
# ---------------------------------------------------------------------------

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.middleware("http")
async def revalidate_ui(request, call_next):
    # Without a Cache-Control header browsers may serve a cached index.html
    # together with a newer app.js (or vice versa) after an update, and the
    # mismatched pair fails at startup. Revalidation is a cheap local 304.
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(WEB_DIR / "index.html"))


def main() -> None:
    import uvicorn

    load_env()
    ensure_dirs()
    store.init()
    port = int(os.environ.get("CTXAGENT_PORT", "8765"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
