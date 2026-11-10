"""Settings profiles and app-level preferences.

A profile is a named bundle of chat settings: a *partial* SessionConfig dict
(model, reasoning, output/context limits, temperature, provider policy, pricing,
tools, system prompt addition, the compaction config, delegation limits, and
whatever fields SessionConfig grows later). Missing keys resolve to the
SessionConfig defaults at the moment the profile is used, so a field another
module adds tomorrow is carried automatically.

Semantics
---------
* A chat created from a profile stores a FULL snapshot of the resolved settings
  plus ``profile_id``. Everything stays editable in the chat.
* A chat is "modified" when its snapshot differs from the profile's resolved
  settings (``differences``). Working directories are per chat and never part
  of a profile.
* Editing or deleting a profile never rewrites existing chats. They keep their
  snapshot; the profile slot may then show "(modified)" or "(deleted)".
* Built-in profiles are read-only starters; duplicate one to edit it.

Files (both in config.DATA_DIR, no secrets): ``profiles.json`` holds the user
profiles, ``settings.json`` the app preferences (default profile, default
working directory, new-chat behaviour, theme, zoom).
"""

from __future__ import annotations

import copy
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from . import config
from .config import CompactionConfig, SessionConfig

PROFILES_FILE = "profiles.json"
PREFS_FILE = "settings.json"
EXPORT_FORMAT = "ctxagent-profiles"
#: Exports written while the app was called PiAgent; imported like current ones.
LEGACY_EXPORT_FORMATS = ("piagent-profiles",)

#: Per-chat state, never part of a profile or of the "modified" comparison.
EXCLUDED = frozenset({"workdirs", "profile_id"})
ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
MAX_PROFILES = 200

_lock = threading.RLock()


class ProfileError(ValueError):
    """Bad input (422)."""


class NotFound(LookupError):
    pass


class ReadOnly(PermissionError):
    pass


# ---------------------------------------------------------------------------
# Built-in starters (OpenRouter ids; see MODELS.md and anthropic_api.PRICES)
# ---------------------------------------------------------------------------
#
# Not every model manages its context reliably. Fable, Opus, Sol, Astra, Sonnet and
# Terra are self-steered at a 1M window: they get compact_context, curate_context and
# the "Managing your own context" prompt section. Gemini 4 Argon, which never compacted
# when self-steered (GUIDANCE_EXPERIMENTS.md section 10), runs at a 400k window with the
# engine managing the context. Every starter enables every tool
# the user can switch (tools.catalogue()); the summariser is the chat model.

def _all_tools() -> list[str]:
    from . import tools
    return [t["name"] for t in tools.catalogue()]


#: Compaction of the self-steered starters: the configuration the benchmark's frozen
#: ctxagent_live arm records (bench/configs/ctxagent_live.json), with the chat model as
#: summariser. Review hints come at most every 25 tool batches, the deep safety net at
#: 90 % of the window.
_SELF_STEERED = {
    "summary_model": "", "rewrite_policy": "agent", "rewrite_interval_batches": 25,
    "initial_result_max_tokens": 8_000,
    "deep_trigger_ratio": 0.9, "summary_max_tokens": 3_000,
}
#: Engine-managed starters: the engine defaults (deep compaction at 75 % of the window),
#: no context tools for the model.
_ENGINE = {"summary_model": "", "rewrite_policy": "auto"}


def _starter(pid: str, name: str, model: str, label: str, price: str, *, steered: bool, effort: str,
             max_out: int = 32_000, window: int | None = None, compaction: dict | None = None) -> dict[str, Any]:
    window = window or (1_000_000 if steered else 400_000)
    if steered:
        how = (f"Self-steered context management at a {window // 1000}k window: the model compacts and "
               "curates its own context; the engine's deep compaction stays as the safety net.")
    else:
        how = ("400k window, context managed by the compaction engine; the model has no "
               "compact_context or curate_context.")
    return {
        "id": pid, "name": name,
        "description": f"{label} via OpenRouter ({price} per Mtok), {effort} reasoning effort. {how} All tools.",
        "config": {"model": model, "reasoning_effort": effort,
                   "context_window": window, "max_output_tokens": max_out,
                   "pricing_mode": "standard", "enabled_tools": _all_tools(),
                   "compaction": dict(compaction or (_SELF_STEERED if steered else _ENGINE))},
    }


BUILTINS: list[dict[str, Any]] = [
    _starter("builtin-fable-5-1", "Fable 5.1 (self-steered, 1M)", "anthropic/claude-fable-5.1", "Claude Fable 5.1",
             "$10 / $50", steered=True, effort="high", max_out=64_000),
    # The app's default chat (SessionConfig defaults): xhigh, 700k, unlimited output,
    # self-steered with deep compaction at 90 %.
    _starter("builtin-opus-5-5", "Opus 5.5 (self-steered, 700k)", "anthropic/claude-opus-5.5", "Claude Opus 5.5",
             "$4 / $20", steered=True, effort="xhigh", max_out=0, window=700_000,
             compaction={"summary_model": "", **config.NEW_CHAT_COMPACTION}),
    _starter("builtin-sol-6-1", "Sol 6.1 (self-steered, 1M)", "openai/gpt-6.1-sol", "GPT-6.1 Sol",
             "$2 / $10", steered=True, effort="high"),
    _starter("builtin-astra-6", "Astra 6 (self-steered, 1M)", "openai/gpt-6-astra", "GPT-6 Astra",
             "$10 / $50", steered=True, effort="high"),
    _starter("builtin-sonnet-5-5", "Sonnet 5.5 (self-steered, 1M)", "anthropic/claude-sonnet-5.5", "Claude Sonnet 5.5",
             "$2 / $10", steered=True, effort="medium"),
    _starter("builtin-terra-6-1", "Terra 6.1 (self-steered, 1M)", "openai/gpt-6.1-terra", "GPT-6.1 Terra",
             "$1 / $5", steered=True, effort="medium"),
    _starter("builtin-argon-4", "Gemini 4 Argon (engine, 400k)", "google/gemini-4-argon", "Gemini 4 Argon",
             "$2 / $10", steered=False, effort="high"),
]
_BUILTIN_IDS = {p["id"] for p in BUILTINS}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def merged(base: dict, patch: dict | None) -> dict:
    """Deep-enough merge: top-level keys replace, `compaction` merges by key."""
    result = copy.deepcopy(base)
    for key, value in (patch or {}).items():
        if key == "compaction" and isinstance(value, dict):
            result.setdefault("compaction", {}).update(copy.deepcopy(value))
        else:
            result[key] = copy.deepcopy(value)
    return result


def base_config() -> dict[str, Any]:
    return SessionConfig().to_dict()


def _known_keys() -> tuple[set[str], set[str]]:
    base = base_config()
    return set(base) - EXCLUDED, set(base["compaction"])


def normalize(partial: dict | None) -> dict[str, Any]:
    """Validate a partial config with SessionConfig.from_dict and keep only the
    keys that were given and are real settings, in normalised form."""
    if partial is None:
        return {}
    if not isinstance(partial, dict):
        raise ProfileError("config must be an object")
    try:
        full = SessionConfig.from_dict(merged(base_config(), partial)).to_dict()
    except (ValueError, TypeError) as exc:
        raise ProfileError(str(exc))
    top, comp = _known_keys()
    out: dict[str, Any] = {}
    for key, value in partial.items():
        if key == "compaction":
            if not isinstance(value, dict):
                raise ProfileError("compaction must be an object")
            sub = {k: full["compaction"][k] for k in value if k in comp}
            if sub:
                out["compaction"] = sub
        elif key in top:
            out[key] = full[key]
    return out


def resolve(partial: dict | None) -> dict[str, Any]:
    """Full settings a profile stands for (without per-chat keys)."""
    full = SessionConfig.from_dict(merged(base_config(), partial or {})).to_dict()
    for key in EXCLUDED:
        full.pop(key, None)
    return full


def comparable(chat_config: dict | None) -> dict[str, Any]:
    full = SessionConfig.from_dict(chat_config or {}).to_dict()
    for key in EXCLUDED:
        full.pop(key, None)
    # The context tools follow the self-steered setting, not the tool list (older chats
    # and profiles still list them): ignore them when comparing.
    full["enabled_tools"] = [n for n in full.get("enabled_tools") or [] if n not in config.SELF_STEERED_TOOLS]
    return full


def differences(a: dict, b: dict) -> list[str]:
    """Dotted keys whose values differ between two full (comparable) configs."""
    out = []
    for key in sorted(set(a) | set(b)):
        if key == "compaction" and isinstance(a.get(key), dict) and isinstance(b.get(key), dict):
            for sub in sorted(set(a[key]) | set(b[key])):
                if a[key].get(sub) != b[key].get(sub):
                    out.append(f"compaction.{sub}")
        elif a.get(key) != b.get(key):
            out.append(key)
    return out


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def _path(name: str) -> Path:
    return Path(config.DATA_DIR) / name


def _read_json(name: str, default):
    try:
        return json.loads(_path(name).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except (OSError, ValueError):
        # Keep the unreadable file for inspection; start clean rather than crash.
        try:
            broken = _path(name)
            broken.replace(broken.with_name(broken.name + f".broken-{int(time.time())}"))
        except OSError:
            pass
        return default


def _write_json(name: str, data) -> None:
    path = _path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _user_profiles() -> list[dict[str, Any]]:
    data = _read_json(PROFILES_FILE, {})
    items = data.get("profiles", []) if isinstance(data, dict) else []
    out = []
    for p in items:
        if isinstance(p, dict) and isinstance(p.get("id"), str) and ID_RE.match(p["id"]) \
                and p["id"] not in _BUILTIN_IDS and isinstance(p.get("config"), dict):
            out.append({"id": p["id"], "name": str(p.get("name") or "Untitled")[:80],
                        "description": str(p.get("description") or "")[:500],
                        "config": p["config"], "created_at": p.get("created_at"),
                        "updated_at": p.get("updated_at")})
    return out


def _save_user_profiles(items: list[dict[str, Any]]) -> None:
    _write_json(PROFILES_FILE, {"version": 1, "profiles": items})


def _public(p: dict[str, Any], builtin: bool) -> dict[str, Any]:
    try:
        effective = resolve(p["config"])
        error = None
    except (ValueError, TypeError) as exc:
        effective, error = None, str(exc)
    return {"id": p["id"], "name": p["name"], "description": p.get("description", ""),
            "builtin": builtin, "config": copy.deepcopy(p["config"]), "effective": effective,
            "error": error, "created_at": p.get("created_at"), "updated_at": p.get("updated_at")}


def list_profiles() -> list[dict[str, Any]]:
    with _lock:
        return [_public(p, True) for p in BUILTINS] + [_public(p, False) for p in _user_profiles()]


def get(pid: str) -> dict[str, Any]:
    with _lock:
        for p in BUILTINS:
            if p["id"] == pid:
                return _public(p, True)
        for p in _user_profiles():
            if p["id"] == pid:
                return _public(p, False)
    raise NotFound(f"no such profile: {pid}")


def find(pid: str | None) -> dict[str, Any] | None:
    if not pid:
        return None
    try:
        return get(pid)
    except NotFound:
        return None


def _clean_name(name, items, exclude_id=None) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ProfileError("name must not be empty")
    name = " ".join(name.split())[:80]
    taken = {p["name"].lower() for p in BUILTINS} | {p["name"].lower() for p in items if p["id"] != exclude_id}
    if name.lower() not in taken:
        return name
    for i in range(2, 1000):
        candidate = f"{name[:74]} ({i})"
        if candidate.lower() not in taken:
            return candidate
    raise ProfileError("too many profiles with this name")


def _description(value) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ProfileError("description must be a string")
    return value.strip()[:500]


def create(name: str, config_dict: dict | None = None, description: str = "") -> dict[str, Any]:
    cfg = normalize(config_dict or {})
    with _lock:
        items = _user_profiles()
        if len(items) >= MAX_PROFILES:
            raise ProfileError(f"at most {MAX_PROFILES} profiles")
        now = time.time()
        p = {"id": "p-" + uuid.uuid4().hex[:12], "name": _clean_name(name, items),
             "description": _description(description), "config": cfg,
             "created_at": now, "updated_at": now}
        items.append(p)
        _save_user_profiles(items)
        return _public(p, False)


def update(pid: str, *, name: str | None = None, description: str | None = None,
           config_dict: dict | None = None) -> dict[str, Any]:
    if pid in _BUILTIN_IDS:
        raise ReadOnly("Built-in profiles are read-only. Duplicate it to make an editable copy.")
    with _lock:
        items = _user_profiles()
        for p in items:
            if p["id"] == pid:
                if name is not None:
                    p["name"] = _clean_name(name, items, exclude_id=pid)
                if description is not None:
                    p["description"] = _description(description)
                if config_dict is not None:
                    p["config"] = normalize(config_dict)
                p["updated_at"] = time.time()
                _save_user_profiles(items)
                return _public(p, False)
    raise NotFound(f"no such profile: {pid}")


def delete(pid: str) -> None:
    if pid in _BUILTIN_IDS:
        raise ReadOnly("Built-in profiles cannot be deleted.")
    with _lock:
        items = _user_profiles()
        kept = [p for p in items if p["id"] != pid]
        if len(kept) == len(items):
            raise NotFound(f"no such profile: {pid}")
        _save_user_profiles(kept)
        if load_prefs().get("default_profile_id") == pid:
            _save_prefs({**load_prefs(), "default_profile_id": ""})


def duplicate(pid: str, name: str | None = None) -> dict[str, Any]:
    source = get(pid)
    if source["effective"] is None:
        raise ProfileError(f"profile cannot be resolved: {source['error']}")
    # A duplicate is a full snapshot of what the source means today.
    return create(name or f"{source['name']} copy", source["effective"], source.get("description", ""))


def export(ids: list[str] | None = None) -> dict[str, Any]:
    chosen = [p for p in list_profiles() if (ids is None and not p["builtin"]) or (ids and p["id"] in ids)]
    return {"format": EXPORT_FORMAT, "version": 1, "exported_at": time.time(),
            "profiles": [{"name": p["name"], "description": p["description"],
                          "config": p["effective"] if p["builtin"] else p["config"]}
                         for p in chosen if p["effective"] is not None]}


def import_profiles(data: Any) -> list[dict[str, Any]]:
    """Accepts an export file, a list of profiles, or a single profile. Every
    entry is validated before anything is written; ids are always new."""
    if isinstance(data, dict) and "profiles" in data:
        if data.get("format") not in (None, EXPORT_FORMAT, *LEGACY_EXPORT_FORMATS):
            raise ProfileError("not a CTXAgent profile export")
        entries = data["profiles"]
    elif isinstance(data, list):
        entries = data
    elif isinstance(data, dict) and "config" in data:
        entries = [data]
    else:
        raise ProfileError("expected a CTXAgent profile export")
    if not isinstance(entries, list) or not entries:
        raise ProfileError("no profiles to import")
    if len(entries) > 50:
        raise ProfileError("at most 50 profiles per import")
    prepared = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict) or not isinstance(entry.get("config"), dict):
            raise ProfileError(f"profile {i + 1}: config must be an object")
        try:
            cfg = normalize(entry["config"])
        except ProfileError as exc:
            raise ProfileError(f"profile {i + 1} ({entry.get('name') or 'unnamed'}): {exc}")
        prepared.append((entry.get("name") or "Imported profile", cfg, entry.get("description") or ""))
    return [create(name, cfg, description) for name, cfg, description in prepared]


# ---------------------------------------------------------------------------
# Chats and profiles
# ---------------------------------------------------------------------------

def chat_status(chat_config: dict | None) -> dict[str, Any]:
    pid = (chat_config or {}).get("profile_id") or ""
    profile = find(pid)
    status = {"profile_id": pid, "profile": None, "missing": bool(pid) and profile is None,
              "modified": False, "differences": []}
    if profile is not None:
        status["profile"] = {k: profile[k] for k in ("id", "name", "builtin", "description")}
        if profile["effective"] is not None:
            diff = differences(comparable(chat_config), profile["effective"])
            status["differences"] = diff
            status["modified"] = bool(diff)
    return status


def config_for_chat(pid: str, chat_config: dict | None = None,
                    overrides: dict | None = None) -> SessionConfig:
    """The profile's settings as a chat config. Keeps the chat's working
    directories (or applies the default working directory to a new chat)."""
    profile = get(pid)
    if profile["effective"] is None:
        raise ProfileError(f"profile cannot be resolved: {profile['error']}")
    full = merged(profile["effective"], overrides)
    full["profile_id"] = pid
    if "workdirs" not in (overrides or {}):
        full["workdirs"] = (list(chat_config.get("workdirs") or []) if chat_config is not None
                            else default_workdirs())
    return SessionConfig.from_dict(full)


def snapshot_from_chat(chat_config: dict) -> dict[str, Any]:
    return comparable(chat_config)


# ---------------------------------------------------------------------------
# App preferences (DATA_DIR/settings.json)
# ---------------------------------------------------------------------------

DEFAULT_PREFS: dict[str, Any] = {
    "default_profile_id": "",      # "" = saved new-chat template / built-in defaults
    "default_workdir": "",         # applied to new chats that have no working directory
    "new_chat_mode": "inherit",    # "inherit": New chat copies the open chat; "profile": default profile
    "theme": "",                   # PiTheme name; "" = the GUI's default
    "zoom": 100,                   # percent, 80..150
    "insights": False,             # record usage insights (insights.py); off by default
}
NEW_CHAT_MODES = ("inherit", "profile")


def load_prefs() -> dict[str, Any]:
    with _lock:
        data = _read_json(PREFS_FILE, {})
        prefs = dict(DEFAULT_PREFS)
        if isinstance(data, dict):
            for key in DEFAULT_PREFS:
                if key in data:
                    prefs[key] = data[key]
        return prefs


def _save_prefs(prefs: dict[str, Any]) -> None:
    _write_json(PREFS_FILE, {k: prefs[k] for k in DEFAULT_PREFS})


def update_prefs(patch: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(patch, dict):
        raise ProfileError("preferences must be an object")
    unknown = set(patch) - set(DEFAULT_PREFS)
    if unknown:
        raise ProfileError("unknown preferences: " + ", ".join(sorted(unknown)))
    clean: dict[str, Any] = {}
    if "default_profile_id" in patch:
        pid = patch["default_profile_id"] or ""
        if not isinstance(pid, str) or (pid and find(pid) is None):
            raise ProfileError("default_profile_id must name an existing profile (or be empty)")
        clean["default_profile_id"] = pid
    if "default_workdir" in patch:
        wd = patch["default_workdir"] or ""
        if not isinstance(wd, str) or len(wd) > 1000:
            raise ProfileError("default_workdir must be a path")
        wd = wd.strip()
        if wd and not (Path(wd).is_absolute() and Path(wd).is_dir()):
            raise ProfileError("default_workdir must be an existing absolute directory")
        clean["default_workdir"] = wd
    if "new_chat_mode" in patch:
        if patch["new_chat_mode"] not in NEW_CHAT_MODES:
            raise ProfileError("new_chat_mode must be 'inherit' or 'profile'")
        clean["new_chat_mode"] = patch["new_chat_mode"]
    if "theme" in patch:
        theme = patch["theme"] or ""
        if not isinstance(theme, str) or not re.fullmatch(r"[A-Za-z0-9 _-]{0,40}", theme):
            raise ProfileError("theme must be a short name")
        clean["theme"] = theme
    if "zoom" in patch:
        zoom = patch["zoom"]
        if isinstance(zoom, bool) or not isinstance(zoom, (int, float)) or not 80 <= zoom <= 150:
            raise ProfileError("zoom must be between 80 and 150 (percent)")
        clean["zoom"] = int(round(zoom))
    if "insights" in patch:
        if not isinstance(patch["insights"], bool):
            raise ProfileError("insights must be true or false")
        clean["insights"] = patch["insights"]
    with _lock:
        prefs = {**load_prefs(), **clean}
        _save_prefs(prefs)
        return prefs


def default_profile_id() -> str:
    pid = load_prefs().get("default_profile_id") or ""
    return pid if find(pid) is not None else ""


def default_workdirs() -> list[str]:
    wd = load_prefs().get("default_workdir") or ""
    return [wd] if wd and Path(wd).is_dir() else []


def new_chats_inherit_open_chat() -> bool:
    return load_prefs().get("new_chat_mode", "inherit") != "profile" or not default_profile_id()
