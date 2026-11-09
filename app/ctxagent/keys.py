"""Provider API keys: OS credential store first, environment / .env second.

Where keys live
---------------
* The OS credential store via `keyring` (Windows Credential Manager, macOS
  Keychain, Secret Service on Linux), service name ``CTXAgent``, one entry per
  key name. This is where the Settings menu writes. A key stored by the app under
  its former name (service ``PiAgent``) is read from there and copied over once.
* If no keyring backend works (a headless Linux box without Secret Service),
  a fallback file ``DATA_DIR/credentials.json``: created with mode 0600 inside a
  0700 directory on POSIX; on Windows it inherits the per-user ACL of the data
  folder (%LOCALAPPDATA%). Plain text: the OS account is the protection.
* Environment variables, which `config.load_env()` fills from a ``.env`` file
  next to the app. Still supported for development and for the benchmarks.

Lookup order is store, then environment, so a key saved in Settings wins over a
stale ``.env``. Removing a stored key falls back to the environment again.

What never happens
------------------
Keys are not copied into ``os.environ`` (shell commands the agent runs would
inherit them), not written into chat configs, profiles or settings.json, not
logged, never returned unmasked by the API, and never placed in a URL. The
names of stored keys (not the values) are indexed in ``DATA_DIR/keys.json``
because a credential store cannot be enumerated.

Delegated child agents are separate processes. They resolve keys the same way:
the store is per OS user and the data folder / environment is inherited, so no
key ever has to travel on a command line or in an environment block.

``CTXAGENT_KEYSTORE`` (formerly ``PIAGENT_KEYSTORE``, still read) selects the
backend: ``auto`` (default: keyring, else file), ``keyring``, ``file``, or ``env``
(environment only, read-only; the test suite uses this so it never touches the
real credential store).
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

from . import config
from .version import APP_NAME, LEGACY_APP_NAMES

SERVICE = APP_NAME
#: Services the app stored keys under before (its former names), read as a fallback.
LEGACY_SERVICES = LEGACY_APP_NAMES
INDEX_FILE = "keys.json"
FALLBACK_FILE = "credentials.json"
_CACHE_SECONDS = 30.0

#: Known keys. Other names matching NAME_RE can be stored too (room for more
#: providers); they show up in the list once stored or present in the env.
KNOWN: dict[str, dict[str, Any]] = {
    "OPENROUTER_API_KEY": {
        "label": "OpenRouter API key", "provider": "openrouter", "secret": True,
        "help": "Used for every model id with a provider prefix (anthropic/..., openai/..., ...).",
        "placeholder": "sk-or-v1-...",
    },
    "ANTHROPIC_API_KEY": {
        "label": "Anthropic API key", "provider": "anthropic", "secret": True,
        "help": "Used for bare Claude ids (claude-opus-5-5, ...), which go to the Anthropic API directly.",
        "placeholder": "sk-ant-...",
    },
    "ANTHROPIC_WORKSPACE_ID": {
        "label": "Anthropic workspace ID", "provider": "anthropic", "secret": False,
        "help": "Only for user-scoped keys (sk-ant-usr-...): the wrkspc_... id from the Claude Console.",
        "placeholder": "wrkspc_...",
    },
}
NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")
MAX_VALUE_CHARS = 4096


class KeyStoreError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class KeyringBackend:
    name = "keyring"

    def __init__(self, kr=None):
        if kr is None:
            import keyring
            kr = keyring.get_keyring()
        self.kr = kr

    @property
    def label(self) -> str:
        cls = type(self.kr)
        return f"OS credential store ({cls.__module__.rsplit('.', 1)[-1]}.{cls.__name__})"

    def get(self, name: str) -> str | None:
        value = self.kr.get_password(SERVICE, name)
        if value is not None:
            return value
        for service in LEGACY_SERVICES:   # saved under the app's former name
            value = self.kr.get_password(service, name)
            if value is not None:
                try:
                    self.kr.set_password(SERVICE, name, value)
                except Exception:  # noqa: BLE001 - copying is a convenience; the read stands
                    pass
                return value
        return None

    def set(self, name: str, value: str) -> None:
        self.kr.set_password(SERVICE, name, value)

    def delete(self, name: str) -> None:
        # The former name's copy too: removing a key must not bring the old one back.
        for service in (SERVICE, *LEGACY_SERVICES):
            try:
                self.kr.delete_password(service, name)
            except Exception as exc:  # keyring.errors.PasswordDeleteError: not stored
                if type(exc).__name__ != "PasswordDeleteError":
                    raise


class FileBackend:
    """Plain JSON file in the data folder, readable only by the OS user."""
    name = "file"

    def __init__(self, path: Path | None = None, reason: str = ""):
        self._path = path
        self.reason = reason  # why the OS store was not used, shown in Settings

    @property
    def path(self) -> Path:
        return self._path or (Path(config.DATA_DIR) / FALLBACK_FILE)

    @property
    def label(self) -> str:
        text = f"File in the data folder ({self.path.name}, user-only permissions)"
        return f"{text}; {self.reason}" if self.reason else text

    def _read(self) -> dict[str, str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            raise KeyStoreError(f"credential file unreadable: {type(exc).__name__}")
        return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}

    def _write(self, data: dict[str, str]) -> None:
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            try:
                os.chmod(path.parent, 0o700)
            except OSError:
                pass
        tmp = path.with_name(path.name + ".tmp")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        if os.name != "nt":
            os.chmod(tmp, 0o600)
        os.replace(tmp, path)

    def get(self, name: str) -> str | None:
        return self._read().get(name)

    def set(self, name: str, value: str) -> None:
        data = self._read()
        data[name] = value
        self._write(data)

    def delete(self, name: str) -> None:
        data = self._read()
        if data.pop(name, None) is not None:
            self._write(data)


class EnvOnlyBackend:
    """No persistent store: environment / .env only (tests, locked-down hosts)."""
    name = "env"
    label = "Environment only (CTXAGENT_KEYSTORE=env)"

    def get(self, name: str) -> str | None:
        return None

    def set(self, name: str, value: str) -> None:
        raise KeyStoreError("No writable key store: CTXAGENT_KEYSTORE=env. Set the key in the environment or .env.")

    def delete(self, name: str) -> None:
        return None


_lock = threading.RLock()
_backend = None
_cache: dict[str, tuple[float, str | None]] = {}


def _keyring_usable(kr) -> bool:
    try:
        from keyring.backends import fail, null
        if isinstance(kr, (fail.Keyring, null.Keyring)):
            return False
        chained = getattr(kr, "backends", None)
        if chained is not None and not chained:
            return False
        return float(getattr(kr, "priority", 1) or 0) > 0
    except Exception:
        return False


def _select_backend():
    mode = (os.environ.get("CTXAGENT_KEYSTORE") or os.environ.get("PIAGENT_KEYSTORE") or "auto").strip().lower()
    if mode == "env":
        return EnvOnlyBackend()
    if mode == "file":
        return FileBackend()
    try:
        import keyring
        kr = keyring.get_keyring()
        if _keyring_usable(kr):
            return KeyringBackend(kr)
        if mode == "keyring":
            raise KeyStoreError("CTXAGENT_KEYSTORE=keyring but no keyring backend is available")
    except KeyStoreError:
        raise
    except Exception:
        if mode == "keyring":
            raise KeyStoreError("CTXAGENT_KEYSTORE=keyring but keyring failed to load")
    if os.name != "nt" and sys.platform != "darwin":
        return FileBackend(reason="no Secret Service keyring (GNOME Keyring, KWallet) is running")
    return FileBackend()


def backend():
    global _backend
    with _lock:
        if _backend is None:
            _backend = _select_backend()
        return _backend


def use_backend(new) -> None:
    """Swap the backend (tests, or after the user picks one). Clears the cache."""
    global _backend
    with _lock:
        _backend = new
        _cache.clear()


def backend_info() -> dict[str, Any]:
    b = backend()
    return {"name": b.name, "label": b.label, "writable": b.name != "env"}


# ---------------------------------------------------------------------------
# Index of stored names (values never written here)
# ---------------------------------------------------------------------------

def _index_path() -> Path:
    return Path(config.DATA_DIR) / INDEX_FILE


def _read_index() -> list[str]:
    try:
        data = json.loads(_index_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [n for n in data.get("names", []) if isinstance(n, str) and NAME_RE.match(n)]


def _write_index(names: list[str]) -> None:
    path = _index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"names": sorted(set(names))}), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def valid_name(name: str) -> bool:
    return isinstance(name, str) and bool(NAME_RE.match(name))


def _stored(name: str) -> str | None:
    now = time.monotonic()
    with _lock:
        hit = _cache.get(name)
        if hit and now - hit[0] < _CACHE_SECONDS:
            return hit[1]
    try:
        value = backend().get(name)
    except Exception:
        value = None  # a locked/unavailable store must not break env fallback
    value = value.strip() if isinstance(value, str) and value.strip() else None
    with _lock:
        _cache[name] = (now, value)
    return value


def lookup(name: str) -> tuple[str, str | None]:
    """(value, source) with source "store", "env" or None. Value "" if unset."""
    if not valid_name(name):
        return "", None
    value = _stored(name)
    if value:
        return value, "store"
    env = os.environ.get(name, "").strip()
    if env:
        return env, "env"
    return "", None


def get(name: str) -> str:
    return lookup(name)[0]


def set_key(name: str, value: str) -> None:
    if not valid_name(name):
        raise ValueError("Key names are upper-case letters, digits and underscores (e.g. MISTRAL_API_KEY)")
    if not isinstance(value, str):
        raise ValueError("value must be a string")
    value = value.strip()
    if not value:
        raise ValueError("value must not be empty (use delete to remove a key)")
    if len(value) > MAX_VALUE_CHARS or any(c in value for c in "\r\n\0"):
        raise ValueError("value must be a single line of at most 4096 characters")
    with _lock:
        backend().set(name, value)
        _cache.pop(name, None)
        names = _read_index()
        if name not in names:
            _write_index(names + [name])


def delete_key(name: str) -> None:
    if not valid_name(name):
        raise ValueError("invalid key name")
    with _lock:
        backend().delete(name)
        _cache.pop(name, None)
        names = _read_index()
        if name in names:
            _write_index([n for n in names if n != name])


def forget_all() -> list[str]:
    """Delete every key this app stored (indexed names and the known ones) from
    the credential store. Environment and .env values are not touched. Used by
    the uninstaller when the user asks to remove their data."""
    removed = []
    with _lock:
        for name in sorted(set(_read_index()) | set(KNOWN)):
            try:
                had = backend().get(name)
                backend().delete(name)
            except Exception:
                continue
            if had:
                removed.append(name)
        _cache.clear()
        try:
            _write_index([])
        except OSError:
            pass
    return removed


def mask(value: str) -> str:
    """`sk-or-v1-0123...a1b2` -> `sk-or-…a1b2`. Never reveals more than a short
    vendor prefix and the last four characters, and nothing of short values."""
    if not value:
        return ""
    m = re.match(r"^((?:sk|pk|rk)-[a-z]{2,4}-|[a-z]{2,8}_)", value)
    prefix = m.group(1) if m and len(value) >= 12 else ""
    suffix = value[-4:] if len(value) >= 16 else ""
    return f"{prefix}…{suffix}" if (prefix or suffix) else "••••"


def redact(text: str, *values: str) -> str:
    """Replace any occurrence of the given secrets in text (error messages)."""
    for v in values:
        if v and len(v) >= 6:
            text = text.replace(v, mask(v))
    return text


def describe(name: str) -> dict[str, Any]:
    value, source = lookup(name)
    meta = KNOWN.get(name, {})
    stored = bool(_stored(name))
    return {
        "name": name,
        "label": meta.get("label", name),
        "provider": meta.get("provider"),
        "help": meta.get("help", ""),
        "placeholder": meta.get("placeholder", ""),
        "secret": meta.get("secret", True),
        "known": name in KNOWN,
        "configured": bool(value),
        "source": source,
        "stored": stored,
        # Env value still present underneath a stored one?
        "env_present": bool(os.environ.get(name, "").strip()),
        "masked": mask(value),
        "testable": name in TESTERS,
    }


def list_keys() -> list[dict[str, Any]]:
    names = list(KNOWN)
    for n in _read_index():
        if n not in names:
            names.append(n)
    return [describe(n) for n in names]


def status() -> dict[str, bool]:
    return {"openrouter": bool(get("OPENROUTER_API_KEY")),
            "anthropic": bool(get("ANTHROPIC_API_KEY"))}


# ---------------------------------------------------------------------------
# Key checks against free, non-generating endpoints
# ---------------------------------------------------------------------------

OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"
ANTHROPIC_MODELS_URL = "https://api.anthropic.com/v1/models"


async def _test_openrouter(value: str, client) -> dict[str, Any]:
    r = await client.get(OPENROUTER_KEY_URL, headers={"Authorization": f"Bearer {value}"})
    if r.status_code == 200:
        try:
            data = (r.json() or {}).get("data") or {}
        except ValueError:
            data = {}
        detail = {k: data.get(k) for k in ("label", "usage", "limit", "limit_remaining", "is_free_tier") if k in data}
        if isinstance(detail.get("label"), str):
            detail["label"] = redact(detail["label"], value)
        return {"ok": True, "status": 200, "message": "OpenRouter accepted the key.", "detail": detail}
    if r.status_code in (401, 403):
        return {"ok": False, "status": r.status_code, "message": "OpenRouter rejected the key (invalid or revoked)."}
    return {"ok": False, "status": r.status_code, "message": f"OpenRouter answered HTTP {r.status_code}; the key could not be verified."}


async def _test_anthropic(value: str, client) -> dict[str, Any]:
    headers = {"x-api-key": value, "anthropic-version": "2023-06-01"}
    workspace = get("ANTHROPIC_WORKSPACE_ID")
    if workspace:
        headers["anthropic-workspace-id"] = workspace
    r = await client.get(ANTHROPIC_MODELS_URL, headers=headers, params={"limit": 1})
    if r.status_code == 200:
        return {"ok": True, "status": 200, "message": "Anthropic accepted the key.",
                "detail": {"workspace_header": bool(workspace)}}
    try:
        message = str(((r.json() or {}).get("error") or {}).get("message") or "")
    except ValueError:
        message = ""
    message = redact(message, value)[:300]
    if r.status_code in (401, 403):
        return {"ok": False, "status": r.status_code, "message": "Anthropic rejected the key (invalid or revoked)."}
    if "workspace" in message.lower():
        return {"ok": False, "status": r.status_code,
                "message": "This is a user-scoped key: also set ANTHROPIC_WORKSPACE_ID (wrkspc_...) from the Claude Console."}
    return {"ok": False, "status": r.status_code,
            "message": f"Anthropic answered HTTP {r.status_code}" + (f": {message}" if message else "")}


TESTERS = {"OPENROUTER_API_KEY": _test_openrouter, "ANTHROPIC_API_KEY": _test_anthropic}


async def test_key(name: str, value: str | None = None, transport=None) -> dict[str, Any]:
    """Verify a key (the stored/effective one, or `value` before saving) with a
    free metadata request. Nothing is generated or billed. The key travels only
    in a request header, never in the URL, and never appears in the result."""
    tester = TESTERS.get(name)
    if tester is None:
        return {"ok": False, "status": None, "message": f"{name} has no online check."}
    candidate = (value or "").strip() or get(name)
    if not candidate:
        return {"ok": False, "status": None, "message": f"{name} is not configured."}
    import httpx
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0), transport=transport) as client:
            result = await tester(candidate, client)
    except httpx.HTTPError as exc:
        return {"ok": False, "status": None,
                "message": redact(f"Network error: {type(exc).__name__}", candidate)}
    result["masked"] = mask(candidate)
    return result


if __name__ == "__main__":  # CTXAgentCLI.exe --run-module ctxagent.keys --forget-all
    import sys
    if sys.argv[1:] == ["--forget-all"]:
        gone = forget_all()
        print(f"removed {len(gone)} stored key(s)")
        raise SystemExit(0)
    print("usage: python -m ctxagent.keys --forget-all", file=sys.stderr)
    raise SystemExit(2)
