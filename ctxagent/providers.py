"""Provider-region routing.

A model's *origin* and its *serving location* are different things. `z-ai/glm-5.3-flash`
is a Chinese model, but OpenRouter serves it from ~20 endpoints, most of them run
by US companies on US hardware; `qwen/qwen3.5-flash-02-23` looks similar and is in
fact served from exactly one place, Alibaba. You cannot tell these apart from the
model id, so this module asks.

OpenRouter's `/api/v1/providers` exposes `headquarters` and `datacenters` per
provider. We keep providers whose HQ is in the EU/EEA/UK/CH/US/CA **and** whose
declared datacenters are all in that set, then pin each request with
`provider: {"only": [...], "allow_fallbacks": true}`.

`only` is a **hard filter**, not a preference, and it stays hard with fallbacks
enabled -- verified directly: a model served solely from Singapore returns
`404 No allowed providers are available` under both `allow_fallbacks: false` and
`true`. So fallbacks can only ever move the request to *another provider in the
allowlist*, which is exactly what we want:

    allow_fallbacks: false   glm-5.3-flash, 3 calls -> 1 success, 2x HTTP 429
    allow_fallbacks: true    glm-5.3-flash, 3 calls -> 3 successes, 2-7s
                                                       (GMICloud, Friendli, Novita)

Disabling fallbacks bought no extra safety and cost most of the requests: with
one endpoint pinned, a busy provider is a failed call rather than a reroute to
the next US datacentre.

If a model has no allowed endpoint at all the request fails loudly, so
`check_model()` lets the UI and the model evaluator say so up front instead.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from .config import DATA_DIR, ensure_dirs

PROVIDERS_URL = "https://openrouter.ai/api/v1/providers"
ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{model}/endpoints"
CACHE_PATH = lambda: DATA_DIR / "providers.json"  # noqa: E731
CACHE_TTL = 24 * 3600

#: EU / EEA / UK / Switzerland / US / Canada.
EU_US = {
    "US", "CA",
    "GB", "UK", "IE", "IS", "NO", "CH",
    "DE", "FR", "NL", "BE", "LU", "AT", "DK", "SE", "FI",
    "ES", "PT", "IT", "GR", "MT", "CY",
    "PL", "CZ", "SK", "HU", "SI", "HR", "RO", "BG",
    "EE", "LV", "LT",
}

POLICY_EU_US = "eu_us"
POLICY_ANY = "any"

_cache: dict[str, Any] = {"ts": 0.0, "providers": []}


# ---------------------------------------------------------------------------

def _load_disk() -> None:
    try:
        raw = json.loads(CACHE_PATH().read_text(encoding="utf-8"))
        if time.time() - raw.get("ts", 0) < CACHE_TTL and raw.get("providers"):
            _cache.update(raw)
    except Exception:
        pass


def _save_disk() -> None:
    try:
        ensure_dirs()
        CACHE_PATH().write_text(json.dumps(_cache), encoding="utf-8")
    except Exception:
        pass


async def providers(force: bool = False) -> list[dict]:
    """The provider directory, cached on disk for a day."""
    if not _cache["providers"]:
        _load_disk()
    if _cache["providers"] and not force and time.time() - _cache["ts"] < CACHE_TTL:
        return _cache["providers"]
    try:
        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.get(PROVIDERS_URL, headers={"Accept": "application/json"})
            r.raise_for_status()
            data = r.json().get("data", [])
    except Exception:
        return _cache["providers"]  # stale beats nothing
    _cache["providers"] = data
    _cache["ts"] = time.time()
    _save_disk()
    return data


def region_ok(p: dict) -> bool:
    hq = (p.get("headquarters") or "").upper()
    if hq not in EU_US:
        return False
    # An unspecified datacenter list is not evidence of anything, but a
    # specified one that includes a non-EU/US site disqualifies the provider.
    return all((d or "").upper() in EU_US for d in (p.get("datacenters") or []))


async def allowed_slugs(policy: str = POLICY_EU_US) -> list[str]:
    if policy != POLICY_EU_US:
        return []
    return sorted(p["slug"] for p in await providers() if region_ok(p))


def routing_payload(allowed: list[str]) -> dict | None:
    """The `provider` block for a chat-completions request.

    Fallbacks are ON: `only` already constrains them to the allowlist, so they
    add resilience without weakening the region guarantee. See the module
    docstring for the measurement behind that.
    """
    if not allowed:
        return None
    return {"only": allowed, "allow_fallbacks": True}


# ---------------------------------------------------------------------------

async def model_endpoints(model: str) -> list[dict]:
    try:
        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.get(ENDPOINTS_URL.format(model=model),
                            headers={"Accept": "application/json"})
            r.raise_for_status()
            return (r.json().get("data") or {}).get("endpoints", [])
    except Exception:
        return []


async def check_model(model: str, policy: str = POLICY_EU_US) -> dict:
    """Can `model` be served under `policy`? Returns a UI-friendly verdict."""
    if model.startswith("claude-"):
        # Direct Anthropic API: one US-headquartered provider, no OpenRouter routing.
        return {"model": model, "policy": policy, "known": True, "ok": True,
                "allowed": [{"provider": "Anthropic (direct API)", "slug": None, "tag": None,
                             "hq": "US", "price_in": None, "uptime_1d": None}],
                "blocked": [], "message": "Direct Anthropic API (US); OpenRouter region routing does not apply."}
    provs = await providers()
    by_name = {p.get("name"): p for p in provs}
    eps = await model_endpoints(model)
    if not eps:
        return {"model": model, "policy": policy, "known": False,
                "allowed": [], "blocked": [], "ok": policy == POLICY_ANY,
                "message": "Could not read this model's endpoint list."}

    allowed, blocked = [], []
    from .flex import endpoint_tier
    for e in eps:
        if endpoint_tier(e) != 'default':
            continue  # Flex/priority are opt-in and have their own discovery panel
        p = by_name.get(e.get("provider_name"))
        row = {
            "provider": e.get("provider_name"),
            "slug": (p or {}).get("slug"),
            "tag": e.get("tag"),
            "hq": (p or {}).get("headquarters"),
            "price_in": float((e.get("pricing") or {}).get("prompt") or 0) * 1e6,
            "uptime_1d": e.get("uptime_last_1d"),
        }
        (allowed if (p and region_ok(p)) else blocked).append(row)
    allowed.sort(key=lambda r: r["price_in"])

    if policy == POLICY_ANY:
        msg = f"{len(allowed) + len(blocked)} endpoint(s); region policy is off."
        ok = True
    elif allowed:
        best = allowed[0]
        ok = True
        msg = (f"{len(allowed)} EU/US endpoint(s) available "
               f"(cheapest: {best['provider']}, {best['hq']}, "
               f"${best['price_in']:.3f}/M). "
               f"{len(blocked)} endpoint(s) excluded by region.")
    else:
        ok = False
        names = ", ".join(sorted({b["provider"] or "?" for b in blocked}))
        msg = (f"NO EU/US endpoint serves this model — it is available only via "
               f"{names}. Requests would fail under the EU/US policy. Pick a "
               f"different model or set the region policy to 'any'.")
    return {"model": model, "policy": policy, "known": True, "ok": ok,
            "allowed": allowed, "blocked": blocked, "message": msg}
