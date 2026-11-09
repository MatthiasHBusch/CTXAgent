"""/api/settings/...: API keys, settings profiles and app preferences.

Included from server.py with one line. Every route here is guarded against
cross-site use: requests must address a loopback host (defeats DNS rebinding)
and, when the browser sends an Origin, it must be a loopback origin too.
Mutations take JSON bodies, so a foreign page cannot send them without a CORS
preflight, which this server never grants.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from . import __version__, config, insights, keys, profiles, store
from .version import APP_NAME

_LOOPBACK = {"127.0.0.1", "localhost", "::1", "[::1]"}


def _loopback(host: str | None) -> bool:
    host = (host or "").lower()
    return host in _LOOPBACK or host.endswith(".localhost")


def local_only(request: Request) -> None:
    host = urlsplit("//" + (request.headers.get("host") or "")).hostname
    if not _loopback(host):
        raise HTTPException(403, "Settings are only available on the local app address")
    origin = request.headers.get("origin")
    if origin is not None and not _loopback(urlsplit(origin).hostname):
        raise HTTPException(403, "Cross-origin settings requests are refused")


router = APIRouter(prefix="/api/settings", dependencies=[Depends(local_only)])


# ---------------------------------------------------------------------------
# Overview / preferences / about
# ---------------------------------------------------------------------------

def _logs_dir() -> Path:
    return Path(getattr(config, "LOG_DIR", None) or Path(config.DATA_DIR) / "logs")


@router.get("")
async def overview() -> dict[str, Any]:
    return {
        "product": APP_NAME,
        "version": __version__,
        "backend_build": config.BACKEND_BUILD,
        "data_dir": str(Path(config.DATA_DIR).resolve()),
        "logs_dir": str(_logs_dir().resolve()),
        "keystore": keys.backend_info(),
        "keys": keys.status(),
        "prefs": profiles.load_prefs(),
        "default_profile_id": profiles.default_profile_id(),
    }


class PrefsPatch(BaseModel):
    prefs: dict[str, Any]


@router.patch("/prefs")
async def patch_prefs(body: PrefsPatch) -> dict[str, Any]:
    try:
        prefs = profiles.update_prefs(body.prefs)
    except profiles.ProfileError as exc:
        raise HTTPException(422, str(exc))
    insights.refresh()
    return {"prefs": prefs}


class FolderRequest(BaseModel):
    which: str = Field(pattern="^(data|logs|insights)$")


def _open_in_file_manager(path: Path) -> None:  # replaced in tests
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


@router.post("/open-folder")
async def open_folder(body: FolderRequest) -> dict[str, Any]:
    """Opens one of the app's own folders (never an arbitrary path)."""
    path = {"data": Path(config.DATA_DIR), "logs": _logs_dir(), "insights": insights.folder()}[body.which]
    path.mkdir(parents=True, exist_ok=True)
    try:
        _open_in_file_manager(path)
    except Exception as exc:
        raise HTTPException(500, f"Could not open the folder: {type(exc).__name__}")
    return {"ok": True, "path": str(path.resolve())}


# ---------------------------------------------------------------------------
# Usage insights (insights.py, insights_report.py)
# ---------------------------------------------------------------------------

def _insights_summary() -> dict[str, Any]:
    folder = insights.folder()
    files = sorted(folder.glob("*.jsonl")) if folder.is_dir() else []
    report = folder / "report.html"
    return {"enabled": bool(profiles.load_prefs().get("insights")), "folder": str(folder.resolve()),
            "chats": len(files), "bytes": sum(f.stat().st_size for f in files),
            "report": str(report.resolve()) if report.is_file() else None}


@router.get("/insights")
async def insights_summary() -> dict[str, Any]:
    return _insights_summary()


class ReportRequest(BaseModel):
    open: bool = True


@router.post("/insights/report")
async def insights_report(body: ReportRequest) -> dict[str, Any]:
    """Builds DATA_DIR/insights/report.html (and report.md) and opens it in the default browser."""
    import asyncio
    from . import insights_report as report
    folder = insights.folder()
    info = await asyncio.to_thread(report.build, folder, md_path=folder / "report.md")
    if body.open and info["chats"]:
        try:
            _open_in_file_manager(Path(info["html"]))
        except Exception as exc:
            raise HTTPException(500, f"Report written to {info['html']} but could not be opened: {type(exc).__name__}")
    return {**info, **_insights_summary()}


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------

def _key_name(name: str) -> str:
    if not keys.valid_name(name):
        raise HTTPException(422, "Key names are upper-case letters, digits and underscores")
    return name


def _keys_payload() -> dict[str, Any]:
    return {"keys": keys.list_keys(), "keystore": keys.backend_info(), "status": keys.status()}


@router.get("/keys")
async def list_keys() -> dict[str, Any]:
    return _keys_payload()


# No pydantic length constraints on secrets: FastAPI's 422 body echoes the
# rejected input. keys.set_key validates and never repeats the value.
class KeyValue(BaseModel):
    value: str


@router.put("/keys/{name}")
async def set_key(name: str, body: KeyValue) -> dict[str, Any]:
    _key_name(name)
    try:
        keys.set_key(name, body.value)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except keys.KeyStoreError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:  # backend failure: never echo the value
        raise HTTPException(503, f"The credential store refused the key ({type(exc).__name__})")
    return {"key": keys.describe(name), **_keys_payload()}


@router.delete("/keys/{name}")
async def delete_key(name: str) -> dict[str, Any]:
    _key_name(name)
    try:
        keys.delete_key(name)
    except keys.KeyStoreError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        raise HTTPException(503, f"The credential store refused the change ({type(exc).__name__})")
    return {"key": keys.describe(name), **_keys_payload()}


class KeyTest(BaseModel):
    value: str | None = None


@router.post("/keys/{name}/test")
async def test_key(name: str, body: KeyTest) -> dict[str, Any]:
    """Free check (no generation): OpenRouter GET /api/v1/key, Anthropic GET /v1/models."""
    _key_name(name)
    if body.value is not None and (len(body.value) > keys.MAX_VALUE_CHARS or any(c in body.value for c in "\r\n\0")):
        raise HTTPException(422, "value must be a single line of at most 4096 characters")
    return await keys.test_key(name, body.value)


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

def _profile_errors(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except profiles.NotFound as exc:
        raise HTTPException(404, str(exc))
    except profiles.ReadOnly as exc:
        raise HTTPException(409, str(exc))
    except profiles.ProfileError as exc:
        raise HTTPException(422, str(exc))


def _profiles_payload() -> dict[str, Any]:
    return {"profiles": profiles.list_profiles(), "default_profile_id": profiles.default_profile_id(),
            "excluded_keys": sorted(profiles.EXCLUDED)}


@router.get("/profiles")
async def list_profiles() -> dict[str, Any]:
    return _profiles_payload()


class ProfileCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    config: dict[str, Any] | None = None
    from_chat: str | None = None


@router.post("/profiles")
async def create_profile(body: ProfileCreate) -> dict[str, Any]:
    cfg = body.config
    if body.from_chat:
        row = store.get_chat(body.from_chat)
        if not row:
            raise HTTPException(404, "no such chat")
        cfg = profiles.snapshot_from_chat(row["config"])
    profile = _profile_errors(profiles.create, body.name, cfg, body.description)
    return {"profile": profile, **_profiles_payload()}


class ProfilesExport(BaseModel):
    ids: list[str] | None = None


@router.post("/profiles/export")
async def export_profiles(body: ProfilesExport) -> dict[str, Any]:
    """Profile JSON for download. Contains settings only: no keys, no chats."""
    return profiles.export(body.ids)


class ProfilesImport(BaseModel):
    data: Any


@router.post("/profiles/import")
async def import_profiles(body: ProfilesImport) -> dict[str, Any]:
    created = _profile_errors(profiles.import_profiles, body.data)
    return {"created": created, **_profiles_payload()}


@router.get("/profiles/{pid}")
async def read_profile(pid: str) -> dict[str, Any]:
    return {"profile": _profile_errors(profiles.get, pid)}


class ProfilePatch(BaseModel):
    name: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    config: dict[str, Any] | None = None


@router.patch("/profiles/{pid}")
async def patch_profile(pid: str, body: ProfilePatch) -> dict[str, Any]:
    profile = _profile_errors(profiles.update, pid, name=body.name, description=body.description,
                              config_dict=body.config)
    return {"profile": profile, **_profiles_payload()}


@router.delete("/profiles/{pid}")
async def delete_profile(pid: str) -> dict[str, Any]:
    _profile_errors(profiles.delete, pid)
    return _profiles_payload()


class ProfileDuplicate(BaseModel):
    name: str | None = Field(default=None, max_length=200)


@router.post("/profiles/{pid}/duplicate")
async def duplicate_profile(pid: str, body: ProfileDuplicate) -> dict[str, Any]:
    profile = _profile_errors(profiles.duplicate, pid, body.name)
    return {"profile": profile, **_profiles_payload()}


# ---------------------------------------------------------------------------
# A chat and its profile
# ---------------------------------------------------------------------------

def _chat_row(cid: str) -> dict[str, Any]:
    row = store.get_chat(cid)
    if not row:
        raise HTTPException(404, "no such chat")
    from . import agent
    s = agent.get_session(cid)
    if s is not None and s.pending_config is not None:
        # Saved during the running turn: the settings as they apply from its next request.
        row = {**row, "config": s.pending_config.to_dict()}
    return row


def _editable_session(cid: str):
    """Same rules as PATCH /api/chats/{cid}/config."""
    from . import agent, delegation
    if delegation.for_chat(cid):
        raise HTTPException(409, "This chat is worker-owned; its settings come from the parent agent.")
    s = agent.get_session(cid)
    if not s:
        raise HTTPException(404, "no such chat")
    return s


def _install(s, cfg) -> None:
    """Now, or while a turn runs before its next model request (AgentSession.set_config)."""
    s.set_config(cfg)


def _current(s):
    """The chat's settings as they apply from the next request on."""
    return s.pending_config or s.cfg


@router.get("/chats/{cid}/profile")
async def chat_profile(cid: str) -> dict[str, Any]:
    return profiles.chat_status(_chat_row(cid)["config"])


class ChatProfileApply(BaseModel):
    profile_id: str | None = None


@router.post("/chats/{cid}/profile/reset")
async def reset_chat_to_profile(cid: str, body: ChatProfileApply) -> dict[str, Any]:
    """Replace the chat's settings with a profile's (its own by default, or the
    given one). Working directories, history and usage are kept."""
    s = _editable_session(cid)
    pid = body.profile_id or _current(s).profile_id
    if not pid:
        raise HTTPException(422, "This chat has no profile; choose one")
    cfg = _profile_errors(profiles.config_for_chat, pid, _current(s).to_dict())
    _install(s, cfg)
    return {"config": cfg.to_dict(), "pending": s.pending_config is not None, "stats": s.stats(),
            "status": profiles.chat_status(cfg.to_dict())}


@router.post("/chats/{cid}/profile/save")
async def save_chat_into_profile(cid: str) -> dict[str, Any]:
    """Overwrite the chat's profile with the chat's saved settings. Other chats
    using the profile are not changed (they may now show as modified)."""
    row = _chat_row(cid)
    pid = row["config"].get("profile_id") or ""
    if not pid:
        raise HTTPException(422, "This chat has no profile; use 'save as new profile'")
    _profile_errors(profiles.update, pid, config_dict=profiles.snapshot_from_chat(row["config"]))
    return {"status": profiles.chat_status(row["config"]), **_profiles_payload()}


class ChatProfileSaveAs(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)


@router.post("/chats/{cid}/profile/save-as")
async def save_chat_as_profile(cid: str, body: ChatProfileSaveAs) -> dict[str, Any]:
    """New profile from the chat's saved settings; the chat is linked to it."""
    row = _chat_row(cid)
    profile = _profile_errors(profiles.create, body.name,
                              profiles.snapshot_from_chat(row["config"]), body.description)
    linked = False
    try:
        s = _editable_session(cid)
    except HTTPException:
        s = None  # worker-owned: the profile exists, the link waits
    config = row["config"]
    if s is not None:
        cfg = _current(s).to_dict()
        cfg["profile_id"] = profile["id"]
        from .config import SessionConfig
        _install(s, SessionConfig.from_dict(cfg))
        linked = True
        config = _current(s).to_dict()
    return {"profile": profile, "linked": linked, "status": profiles.chat_status(config),
            **_profiles_payload()}
