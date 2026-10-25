"""Explicit service-tier discovery and bounded fallback planning.

Never send just service_tier=flex: OpenRouter may use standard rates if a model
has no flex endpoints. Exact /flex tags in provider.only prevent that ambiguity.
Discovery is metadata-only; it does not send prompts or buy test generations.
"""
from __future__ import annotations

import copy
import math
import re
import time

from . import providers

MODES = ('standard', 'prefer_flex', 'flex_only')
# Only explicit HTTP rejections before successful stream establishment may fall
# back. No network timeout, auth/credit failure, or partial generation is replayed.
FALLBACK_STATUSES = {404, 408, 429, 502, 503, 504}
TTL = 300
_CACHE: dict[str, tuple[float, list[dict]]] = {}


def validate_mode(value: str) -> str:
    if value not in MODES:
        raise ValueError('pricing_mode must be standard, prefer_flex or flex_only')
    return value


def endpoint_tier(endpoint: dict) -> str:
    suffix = (endpoint.get('tag') or '').rsplit('/', 1)[-1]
    if suffix == 'flex':
        return 'flex'
    if suffix in ('fast', 'priority'):
        return 'priority'
    return 'default'


def rate(value):
    try:
        number = float(value)
        return number if math.isfinite(number) and number >= 0 else None
    except (TypeError, ValueError):
        return None


def effective_rates(endpoint: dict, prompt_tokens: int = 0) -> dict:
    price = copy.deepcopy(endpoint.get('pricing') or {})
    for override in sorted(price.get('overrides') or [], key=lambda row: row.get('min_prompt_tokens', 0)):
        if prompt_tokens >= override.get('min_prompt_tokens', 0):
            price.update(override)
    return {k: rate(price.get(k)) for k in ('prompt','completion','input_cache_read','input_cache_write')}


def catalogue(model: str, endpoints: list[dict], directory: list[dict], policy: str,
              prompt_tokens: int = 0, needs_tools: bool = False) -> dict:
    by_name = {p.get('name'): p for p in directory}
    candidates, excluded = [], []
    for e in endpoints:
        if endpoint_tier(e) != 'flex':
            continue
        tag = e.get('tag') or ''
        provider = by_name.get(e.get('provider_name'))
        reason = None
        if not re.fullmatch(r'[a-z0-9_-]+(?:/[a-z0-9_.-]+)*/flex', tag):
            reason = 'unrecognized exact Flex endpoint tag'
        elif policy != providers.POLICY_ANY and not (provider and providers.region_ok(provider)):
            reason = 'not eligible under provider-region policy'
        elif provider and not tag.startswith(provider['slug'].rstrip('/') + '/'):
            reason = 'endpoint tag does not match provider identity'
        elif needs_tools and e.get('supported_parameters') is not None and 'tools' not in e['supported_parameters']:
            reason = 'endpoint does not advertise tool use'
        # Status is advisory. A listed unhealthy endpoint can recover before the
        # actual request; gateway capacity errors are handled explicitly.
        prices = effective_rates(e, prompt_tokens)
        base = [b for b in endpoints if b.get('tag') == tag[:-5] and endpoint_tier(b) == 'default']
        base_rates = effective_rates(base[0], prompt_tokens) if len(base) == 1 else {}
        savings = {k: 1 - prices[k] / base_rates[k]
                   if prices[k] is not None and base_rates.get(k) not in (None, 0) else None
                   for k in prices}
        row = {'tag': tag, 'provider': e.get('provider_name'), 'status': e.get('status'),
               'rates': prices, 'standard_rates': base_rates, 'discount_fractions': savings,
               'context_length': e.get('context_length'), 'max_completion_tokens': e.get('max_completion_tokens')}
        if reason:
            excluded.append({**row, 'reason': reason})
        else:
            candidates.append(row)
    candidates.sort(key=lambda row: (row['rates']['prompt'] if row['rates']['prompt'] is not None else math.inf, row['tag']))
    # Deduplicate tags: provider.only is a routing set, not repeated attempts.
    candidates = list({row['tag']: row for row in candidates}.values())
    return {'model': model, 'policy': policy, 'known': bool(endpoints), 'available': bool(candidates),
            'candidates': candidates, 'excluded': excluded, 'prompt_tokens_estimate': prompt_tokens,
            'checked_at': time.time(), 'note': 'Advertised tariffs, not a reservation. Flex may be slower or unavailable. '
                    'Discounts compare the matching standard endpoint, not the cheapest provider. Actual returned tier and cost govern billing.'}


async def discover(model: str, policy: str, *, prompt_tokens: int = 0, needs_tools: bool = False,
                   refresh: bool = False) -> dict:
    if policy not in (providers.POLICY_EU_US, providers.POLICY_ANY):
        raise ValueError('Unknown provider policy')
    # Model variants are not endpoint catalogue identities. Do not rewrite the
    # request model; only remove a known routing variant for metadata discovery.
    base = re.sub(r':(?:floor|nitro)$', '', model).lstrip('~')
    cached = _CACHE.get(base)
    if refresh or cached is None or time.time() - cached[0] > TTL:
        endpoints = await providers.model_endpoints(base)
        _CACHE[base] = (time.time(), endpoints)
    else:
        endpoints = cached[1]
    directory = await providers.providers()
    return catalogue(model, endpoints, directory, policy, prompt_tokens, needs_tools)


async def plans(model: str, policy: str, mode: str, *, needs_tools: bool = False) -> list[dict]:
    validate_mode(mode)
    if policy not in (providers.POLICY_ANY, providers.POLICY_EU_US):
        raise ValueError('Unknown provider policy')
    allowed = await providers.allowed_slugs(policy)
    if policy == providers.POLICY_EU_US and not allowed:
        # An empty allowlist must never become an unrestricted provider payload.
        raise ValueError('No eligible providers known for the selected regional policy; refusing unrestricted routing')
    standard = {'tier': 'default', 'provider': providers.routing_payload(allowed),
                'notice': None, 'mode': mode}
    if mode == 'standard':
        return [standard]
    info = await discover(model, policy, needs_tools=needs_tools)
    tags = [row['tag'] for row in info['candidates']]
    # Recheck against the actual allowlist to avoid trusting mismatched directory snapshots.
    if policy == providers.POLICY_EU_US:
        tags = [tag for tag in tags if any(tag.startswith(slug.rstrip('/') + '/') for slug in allowed)]
    if not tags:
        if mode == 'flex_only':
            raise ValueError('Flex only selected, but no eligible Flex endpoints were discovered. No generation sent.')
        return [{**standard, 'notice': 'Flex unavailable or discovery failed; using standard routing/rates as permitted.'}]
    plan = {'tier': 'flex', 'mode': mode,
            'provider': {'only': tags, 'allow_fallbacks': True, 'sort': 'price'},
            'notice': 'Trying Flex endpoints: ' + ', '.join(tags) + '. Lower availability/higher latency are possible.'}
    return [plan] + ([{**standard, 'notice': 'Flex rejected/unavailable; falling back to standard routing/rates as permitted.'}]
                     if mode == 'prefer_flex' else [])


def fallback_allowed(status: int, body: str = '') -> bool:
    if status in FALLBACK_STATUSES:
        return True
    # Avoid replaying arbitrary malformed requests; only a clearly tier-specific
    # 400 is eligible for compatibility fallback.
    message = body.lower()
    return status == 400 and ('service_tier' in message or 'flex' in message) and any(
        term in message for term in ('unsupported', 'not supported', 'unavailable', 'not available', 'invalid service_tier'))


def metadata(plan: dict, reported_tier=None) -> dict:
    tier = reported_tier if reported_tier in ('default', 'flex', 'priority', 'standard') else None
    warning = None
    if plan['tier'] == 'flex' and tier in ('default', 'standard', 'priority'):
        warning = 'Provider reported a non-Flex tier despite Flex-only endpoint routing. Actual tier/cost applies; no repeat request was sent.'
    return {'pricing_mode': plan['mode'], 'requested_tier': plan['tier'],
            'reported_service_tier': 'default' if tier == 'standard' else tier,
            'candidate_endpoint_tags': (plan.get('provider') or {}).get('only', []) if plan['tier'] == 'flex' else [],
            'flex_confirmed': tier == 'flex', 'routing_notice': plan.get('notice'), 'tier_warning': warning}
