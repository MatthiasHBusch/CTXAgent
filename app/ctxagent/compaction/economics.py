"""Cache economics of a context rewrite, from this chat's own prices and cache layout.

A rewrite (a curate_context drop, a compact_context) frees S
tokens from every later request but makes the provider re-read the prompt only up to the
last cache breakpoint before the first changed byte; the R surviving tokens after that
point are written again (cache-write price, or plain input without a write tariff). With
per-token prices r (cache read) and w (cache write):

    one-time cost    R * (w - r)            (R tokens written instead of read once)
    saving           S * r per later request
    break-even       n = R * (w - r) / (S * r)   later requests

so the same drop that pays after a dozen requests on a 1:12.5 price list (Haiku, Astra)
needs fifty on 1:50 (Fable). `R` is what the provider actually re-sends: with explicit
breakpoints (Anthropic family) everything after the frozen prefix, not just the suffix
after the dropped result. The expected number of later requests is the Copernican
estimate: as many as since the last compaction, clamped to [rewrite_horizon_min,
rewrite_horizon_max].
"""
from __future__ import annotations

from ..segments import Conversation


def rates(usage: dict | None, fallback_ratio: float = 20.0) -> dict:
    """{'read', 'write', 'input', 'output', 'known'} per token from the newest priced agent
    request of this chat; a cache miss without a write tariff is billed as input. Unknown
    prices fall back to the configured write/read ratio (read = 1 unit)."""
    for r in reversed((usage or {}).get('recent_requests') or []):
        if r.get('source') != 'agent':
            continue
        rt = (r.get('pricing') or {}).get('rates') or {}
        prompt, read = rt.get('prompt'), rt.get('input_cache_read')
        if prompt and read:
            write = rt.get('input_cache_write') or prompt
            return {'read': float(read), 'write': float(write), 'input': float(prompt),
                    'output': float(rt.get('completion') or 0.0), 'known': True}
    return {'read': 1.0, 'write': float(fallback_ratio), 'input': float(fallback_ratio) / 1.25,
            'output': 0.0, 'known': False}


def cache_family(model: str) -> str:
    from ..cache_policy import cache_family as fam
    return 'breakpoints' if fam(model or '') == 'anthropic' else 'prefix'


def cached_boundary(conv: Conversation, first_change: int, family: str, tool_overhead: int = 0) -> int:
    """Prompt tokens the next request can still read from cache when segment `first_change`
    is the first one to change: the frozen prefix under explicit breakpoints (its end is a
    breakpoint every request writes), the whole unchanged prefix under automatic caching."""
    segs = conv.segments
    end = first_change if family == 'prefix' else min(first_change, conv.frozen_upto)
    return tool_overhead + sum(s.tokens() for s in segs[:max(0, end)])


def horizon(conv: Conversation, cfg) -> int:
    st = conv.compaction_state
    since = min(int(st.get('batch_clock', 0) or 0), int(st.get('turns_since_deep', 10 ** 9) or 0))
    return max(cfg.rewrite_horizon_min, min(cfg.rewrite_horizon_max, since))


def verdict(freed: int, invalidated: int, price: dict, horizon_requests: int, calibration: float = 1.0) -> dict:
    """Break-even of one rewrite. Token counts are local estimates; `calibration` converts
    them to provider tokens for the dollar figures only (the ratio needs no conversion)."""
    r, w = price['read'], price['write']
    once = max(0, invalidated) * max(0.0, w - r)
    per = max(0, freed) * r
    n = (once / per) if per > 0 else None
    out = {'freed_tokens': int(freed), 'invalidated_tokens': int(invalidated),
           'breakeven_requests': round(n, 1) if n is not None else None,
           'expected_requests': int(horizon_requests),
           'worth_it': bool(n is not None and n <= horizon_requests),
           'write_over_read': round(w / r, 1) if r else None}
    if price.get('known'):
        out['one_time_usd'] = round(once * calibration, 4)
        out['saving_per_request_usd'] = round(per * calibration, 5)
    return out
