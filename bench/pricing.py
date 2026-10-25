"""Uniform cost from token counts x OpenRouter list prices, identical for every harness.

Token categories (disjoint): uncached input, cache write, cache read, output
(output includes reasoning). Provider-reported cost is recorded separately as a
cross-check; this function is what the paper's cost numbers use.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

CACHE = Path(__file__).resolve().parent / "results" / "prices.json"


def prices(refresh: bool = False) -> dict:
    """{model_id: {in, out, cache_read, cache_write}} in $/token, snapshot cached on disk
    (the snapshot date is stored, so the paper can cite the price date)."""
    if CACHE.exists() and not refresh:
        return json.loads(CACHE.read_text(encoding="utf-8"))["models"]
    data = httpx.get("https://openrouter.ai/api/v1/models", timeout=60).json()["data"]
    models = {}
    for m in data:
        p = m.get("pricing") or {}
        f = lambda k: float(p.get(k) or 0)  # noqa: E731
        models[m["id"]] = {"in": f("prompt"), "out": f("completion"),
                           "cache_read": f("input_cache_read"), "cache_write": f("input_cache_write")}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps({"fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"), "models": models}),
                     encoding="utf-8")
    return models


def cost(model: str, uncached: int, cache_write: int, cache_read: int, output: int) -> float:
    model = model.split("[", 1)[0]  # Claude Code's "[1m]" context suffix is not part of the id
    p = prices().get(model.lstrip("~")) or prices().get(model)
    if p is None:
        raise KeyError(f"no price for {model}")
    cw = p["cache_write"] or p["in"]
    cr = p["cache_read"] or p["in"]
    return uncached * p["in"] + cache_write * cw + cache_read * cr + output * p["out"]
