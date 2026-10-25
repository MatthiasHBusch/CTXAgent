"""Measured usage and labelled tariff estimates. Reasoning is a completion subset.

Never infer a missing reasoning count as zero or split historical totals by the
current model's price. Actual cost is authoritative; category estimates and their
reconciliation residual are separate. No prompt or reasoning text is logged.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import re
import time

from . import config, providers
from .flex import endpoint_tier

CATEGORIES = ('input', 'cache_read', 'cache_write', 'output', 'reasoning', 'output_unsplit')
_PRICE_CACHE: dict[str, tuple[float, list[dict]]] = {}


def number(value) -> float | None:
    try:
        n = float(value)
        return n if math.isfinite(n) and n >= 0 else None
    except (ValueError, TypeError):
        return None


def count(value) -> int:
    return int(number(value) or 0)


def normalize(raw: dict) -> dict:
    details = raw.get('prompt_tokens_details') or {}
    completion_details = raw.get('completion_tokens_details') or {}
    prompt, completion = count(raw.get('prompt_tokens')), count(raw.get('completion_tokens'))
    read = count(details.get('cached_tokens', raw.get('cache_read_input_tokens')))
    write = count(details.get('cache_write_tokens', details.get('cache_creation_tokens',
                  raw.get('cache_creation_input_tokens'))))
    reasoning = number(completion_details.get('reasoning_tokens'))
    warnings = []
    if read + write > prompt:
        warnings.append('cache counts exceed total prompt tokens; input split unavailable')
    if reasoning is not None and reasoning > completion:
        warnings.append('reasoning exceeds completion tokens; output split unavailable')
        reasoning = None
    return {'prompt_tokens': prompt, 'completion_tokens': completion,
            'input': max(0, prompt - read - write) if read + write <= prompt else None,
            'cache_read': read, 'cache_write': write,
            'reasoning': int(reasoning) if reasoning is not None else None,
            'output': completion - int(reasoning) if reasoning is not None else None,
            'output_unsplit': completion if reasoning is None else 0,
            'cost': number(raw.get('cost')), 'warnings': warnings}


def resolve_price(endpoints: list[dict], provider: str | None, prompt_tokens: int) -> dict:
    candidates = [e for e in endpoints if not provider or e.get('provider_name') == provider
                  or e.get('tag') == provider]
    prices = []
    for endpoint in candidates:
        price = copy.deepcopy(endpoint.get('pricing') or {})
        for override in sorted(price.get('overrides') or [], key=lambda o: o.get('min_prompt_tokens', 0)):
            if prompt_tokens >= int(override.get('min_prompt_tokens', 0)):
                price.update({k: v for k, v in override.items() if k != 'min_prompt_tokens'})
        prices.append(price)
    # A provider may have multiple service tiers with different tariffs. Do not
    # guess which endpoint served the request; retain only agreed rate fields.
    rates = {}
    for key in ('prompt', 'completion', 'input_cache_read', 'input_cache_write'):
        values = {number(price.get(key)) for price in prices}
        if len(values) == 1 and None not in values:
            rates[key] = values.pop()
    return {'rates': rates, 'provider': provider, 'candidate_endpoints': len(candidates),
            'basis': 'published endpoint tariff estimate; not an invoice allocation',
            'fetched_at': time.time()}


async def pricing(model: str, provider: str | None, prompt_tokens: int, raw: dict | None = None, *, reported_tier: str | None = None) -> dict:
    from . import anthropic_api
    if anthropic_api.is_direct(model):
        rates = anthropic_api.rates(model, prompt_tokens)
        return {'rates': rates or {}, 'provider': 'Anthropic', 'candidate_endpoints': 1 if rates else 0,
                'basis': 'Anthropic list price table (ctxagent/anthropic_api.py)' if rates else 'unknown direct model',
                'fetched_at': time.time(), 'reported_service_tier': reported_tier}
    cached = _PRICE_CACHE.get(model)
    if cached is None or time.time() - cached[0] > 3600:
        eps = await providers.model_endpoints(model)
        _PRICE_CACHE[model] = (time.time(), eps)
    else:
        eps = cached[1]
    if reported_tier in ('default', 'flex', 'priority'):
        eps = [e for e in eps if endpoint_tier(e) == reported_tier]
    resolved = resolve_price(eps, provider, prompt_tokens)
    resolved['reported_service_tier'] = reported_tier
    if raw and number(raw.get('cost')) is not None:
        matches = []
        for endpoint in eps:
            if provider and provider not in (endpoint.get('provider_name'), endpoint.get('tag')):
                continue
            candidate = resolve_price([endpoint], None, prompt_tokens)
            candidate['provider'] = provider
            candidate['endpoint_tag'] = endpoint.get('tag')
            estimated = estimate(normalize(raw), candidate)
            residual = estimated['residual']
            if residual is not None and abs(residual) <= max(1e-8, float(raw['cost']) * 1e-5):
                matches.append(candidate)
        tariffs = {json.dumps(item['rates'], sort_keys=True) for item in matches}
        if len(tariffs) == 1:
            resolved = matches[0]
            resolved['reported_service_tier'] = reported_tier
            resolved['basis'] = 'published tariff consistent with reported total; inferred, not provider itemization'
    return resolved


def estimate(normalized: dict, price: dict) -> dict:
    mapping = {'input': 'prompt', 'cache_read': 'input_cache_read', 'cache_write': 'input_cache_write',
               'output': 'completion', 'reasoning': 'completion', 'output_unsplit': 'completion'}
    rates = price.get('rates', {})
    out = {}
    for category, rate_key in mapping.items():
        n = normalized[category]
        rate = rates.get(rate_key)
        out[category] = 0.0 if n == 0 else (n * rate if n is not None and rate is not None else None)
    # When reasoning count is unreported, output_unsplit represents ALL completion
    # cost; output/reasoning remain unknown, not additional cost.
    active = ('input', 'cache_read', 'cache_write', 'output_unsplit') if normalized['reasoning'] is None else CATEGORIES
    complete = all(out[k] is not None for k in active)
    total = sum(out[k] or 0 for k in active)
    return {'components': out, 'total': total if complete else None, 'complete': complete,
            'residual': normalized['cost'] - total if complete and normalized['cost'] is not None else None}


def initialize(usage: dict) -> dict:
    if usage.get('accounting_version') == 2:
        return usage
    usage.update(accounting_version=2, legacy_unallocated_cost=float(usage.get('cost') or 0),
                 component_estimates={k: 0.0 for k in CATEGORIES},
                 component_unknown_requests={k: 0 for k in CATEGORIES},
                 token_components={k: 0 for k in CATEGORIES}, recent_requests=[],
                 reported_cost_requests=0, missing_cost_requests=0, reconciliation_residual=0.0,
                 unreconciled_reported_cost=0.0, by_source={})
    return usage


def record(usage: dict, raw: dict, *, model: str, provider: str | None = None,
           generation_id: str | None = None, source: str = 'agent', price: dict | None = None,
           diagnostic: dict | None = None, routing: dict | None = None) -> dict | None:
    initialize(usage)
    if generation_id and any(r.get('generation_id') == generation_id for r in usage['recent_requests']):
        return None
    n = normalize(raw)
    price = price or {'rates': {}, 'basis': 'unavailable'}
    e = estimate(n, price)
    entry = {'timestamp': time.time(), 'generation_id': generation_id, 'model': model,
             'provider': provider, 'source': source, 'usage': n, 'raw_usage': copy.deepcopy(raw),
             'pricing': price, 'estimate': e, 'cache_diagnostic': diagnostic or {}, 'routing': routing or {}}
    for key in ('prompt_tokens', 'completion_tokens', 'cache_read', 'cache_write'):
        usage[key] = usage.get(key, 0) + n[key]
    usage['requests'] = usage.get('requests', 0) + 1
    if n['cost'] is None:
        usage['missing_cost_requests'] += 1
    else:
        usage['reported_cost_requests'] += 1
        usage['cost'] = usage.get('cost', 0.0) + n['cost']
        if e['residual'] is not None:
            usage['reconciliation_residual'] += e['residual']
        else:
            usage['unreconciled_reported_cost'] += n['cost']
    for category in CATEGORIES:
        usage['token_components'][category] += n[category] or 0
        if e['components'][category] is None:
            usage['component_unknown_requests'][category] += 1
        else:
            usage['component_estimates'][category] += e['components'][category]
    by_source = usage['by_source'].setdefault(source, {'requests': 0, 'cost': 0.0})
    by_source['requests'] += 1
    by_source['cost'] += n['cost'] or 0
    usage['cache_hit_rate'] = usage['cache_read'] / usage['prompt_tokens'] if usage['prompt_tokens'] else None
    usage['recent_requests'] = (usage['recent_requests'] + [entry])[-100:]
    recent = [r for r in usage['recent_requests'] if r['source'] == 'agent'][-10:]
    prompt = sum(r['usage']['prompt_tokens'] for r in recent)
    usage['recent_cache_hit_rate'] = sum(r['usage']['cache_read'] for r in recent) / prompt if prompt else None
    usage['last_cache_hit_rate'] = recent[-1]['usage']['cache_read'] / recent[-1]['usage']['prompt_tokens'] if recent and recent[-1]['usage']['prompt_tokens'] else None
    return entry


def append_ledger(chat_id: str, entry: dict) -> None:
    # Names are local IDs, never arbitrary paths. JSONL is append-only telemetry;
    # aggregate state remains in SQLite. No prompt or reasoning text is recorded.
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', chat_id):
        raise ValueError('Invalid chat ID for usage ledger')
    folder = config.DATA_DIR / 'usage'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f'{chat_id}.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + '\n')
