"""OpenRouter client.

Two things beyond a plain wrapper:

* **Cache breakpoint injection.** Anthropic-family models on OpenRouter cache
  only where you put an explicit `cache_control` marker. We place one at the end
  of the system prompt (so it survives every compaction) and one at the end of
  the frozen prefix (so the stable history is cached as a unit), plus a moving
  one on the last block so tool iterations re-read the hot zone. This is the
  other half of the "never rewrite the prefix" invariant -- freezing the bytes is
  pointless if nothing tells the provider to cache them.

* **Usage passthrough** including cache-read/-write counts, so the UI can show
  whether the cache strategy is actually working rather than assuming it.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from typing import Any, AsyncIterator

import httpx

from . import providers as provmod
from .config import OPENROUTER_URL, api_key, model_output_limit
from .artifacts import expand_messages
from . import anthropic_api, cache_policy, flex

_TIMEOUT = httpx.Timeout(connect=20.0, read=600.0, write=120.0, pool=20.0)

_ANTHROPIC_PREFIXES = ("anthropic/",)


def supports_explicit_cache(model: str) -> bool:
    return model.startswith(_ANTHROPIC_PREFIXES) or anthropic_api.is_direct(model)


def _headers() -> dict[str, str]:
    key = api_key()
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Put it in a .env file next to the app "
            "or export it in the environment."
        )
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8765",
        "X-Title": "ctxagent",
    }


def _as_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return content
    return [{"type": "text", "text": content or ""}]


def apply_cache_breakpoints(messages: list[dict[str, Any]], model: str,
                            frozen_msg_count: int, previous_anchor: int | None = None,
                            ttl: str = "5m", tail: bool = True, trailer: int = 0) -> list[dict[str, Any]]:
    return cache_policy.apply(messages, model, frozen_msg_count, previous_anchor, ttl, tail=tail,
                              trailer=trailer)[0]


def _merge_usage(target: dict, update: dict) -> None:
    # SSE usage events may be cumulative snapshots, not increments.
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_usage(target[key], value)
        else:
            target[key] = value


def _merge_reasoning(acc: list[dict], blocks: list[dict]) -> None:
    """Preserve opaque provider reasoning blocks verbatim in original order."""
    for block in blocks:
        index = block.get('index')
        found = next((item for item in acc if index is not None and item.get('index') == index
                      and item.get('type') == block.get('type')), None)
        if found is None:
            acc.append(dict(block))
        else:
            for key, value in block.items():
                if key in ('text', 'data', 'signature', 'summary') and isinstance(value, str):
                    found[key] = found.get(key, '') + value
                else:
                    found[key] = value


#: Models that refuse `reasoning: {"enabled": false}`. Learned at runtime so the
#: 400 is paid once per process, not once per call.
_REASONING_MANDATORY: set[str] = set()

REASONING_LEVELS = ("off", "low", "medium", "high", "xhigh", "max", "default")


def reasoning_payload(effort: str) -> dict | None:
    """OpenRouter `reasoning` block for a configured effort level.

    A level the model does not offer is mapped by OpenRouter to the nearest one it
    does (Gemini 4 Argon runs xhigh as high), so no per-model table is kept here."""
    effort = (effort or "default").lower()
    if effort == "off":
        return {"enabled": False}
    if effort in ("low", "medium", "high", "xhigh", "max"):
        return {"effort": effort}
    return None


def _reasoning_ladder(effort: str, model: str) -> list[dict | None]:
    """The configured effort, then progressively cheaper fallbacks.

    Needed because reasoning tokens are drawn from the SAME `max_tokens` budget
    as the answer: a model can reason until the budget is gone and return an
    empty string. We honour the request first, then step down rather than fail.
    """
    first = reasoning_payload(effort)
    if first is None:
        # "default": say nothing and let the provider decide. Adding fallbacks
        # here would silently override a deliberate choice not to specify.
        return [None]
    ladder: list[dict | None] = []
    if not (first == {"enabled": False} and model in _REASONING_MANDATORY):
        ladder.append(first)
    for step in ({"effort": "low"}, None):
        if step not in ladder:
            ladder.append(step)
    return ladder

#: Sentinel: this attempt must be retried with a different `reasoning` setting.
_RETRY_REASONING = object()

#: Retry budget for transient upstream failures. This matters more under a
#: region policy than it would otherwise: `allow_fallbacks: false` is what makes
#: the EU/US guarantee real, and it also removes OpenRouter's own failover, so a
#: momentarily busy endpoint surfaces as a 429 instead of being routed around.
_RETRY_STATUSES = {408, 409, 429, 500, 502, 503, 504}
_MAX_RETRIES = 4
_BACKOFF_BASE = 1.5


#: Transparent retries of a streamed request that failed before producing anything
#: (connection reset, 5xx, provider_unavailable). Recorded in the sessions: upstream
#: transport errors and in-band provider errors that each ended a turn.
STREAM_RETRIES = 3
_STREAM_BACKOFF = (2.0, 6.0, 15.0)
_sleep = asyncio.sleep  # patched by tests


def _retryable_status(status: int) -> bool:
    return status in _RETRY_STATUSES or 520 <= status <= 529


def _retry_after(value) -> float | None:
    try:
        return float(value) if value else None
    except (TypeError, ValueError):
        return None


def _backoff(attempt: int, suggested: float | None) -> float:
    base = _STREAM_BACKOFF[min(attempt, len(_STREAM_BACKOFF)) - 1]
    return min(30.0, max(base, suggested or 0.0))


def _error_code(error) -> str:
    if isinstance(error, dict):
        return str(error.get('code') or (error.get('metadata') or {}).get('error_type') or '?')
    return '?'


def _retryable_error(error) -> bool:
    """An in-band provider error that says "try again", not "this request is wrong"."""
    if isinstance(error, dict):
        try:
            code = int(error.get('code') or 0)
        except (TypeError, ValueError):
            code = 0
        meta = error.get('metadata') if isinstance(error.get('metadata'), dict) else {}
        message = str(error.get('message', '')).lower()
        if code and _retryable_status(code):
            return True
        if meta.get('error_type') in ('provider_unavailable', 'overloaded', 'rate_limit', 'timeout'):
            return True
    else:
        message = str(error or '').lower()
    return any(k in message for k in ('overloaded', 'temporarily unavailable', 'internal server error',
                                      'stream ended before', 'try again', 'upstream connect'))


def _affordable_max_tokens(status: int, body: str, requested: int | None) -> int | None:
    """OpenRouter's 402 "requires more credits, or fewer max_tokens ... can only afford N":
    the same request fits with a smaller max_tokens. Recorded: one sweep lost 137 turns."""
    if status != 402:
        return None
    m = re.search(r"can only afford (\d+)", body)
    if not m:
        return None
    n = int(m.group(1)) - 64
    return n if 1024 <= n and (requested is None or n < requested) else None


async def _post_with_backoff(payload: dict, reasoning: dict | None, model: str):
    """POST with exponential backoff on transient status codes."""
    delay = 1.0
    last: Exception | None = None
    max_attempts = 1 if payload.get("service_tier") == "flex" else _MAX_RETRIES
    for attempt in range(max_attempts):
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.post(OPENROUTER_URL, headers=_headers(), json=payload)

        if r.status_code == 400:
            body = r.text
            if reasoning == {"enabled": False} and "reasoning" in body.lower():
                _REASONING_MANDATORY.add(model)
                return _RETRY_REASONING
            r.raise_for_status()

        if r.status_code in _RETRY_STATUSES and attempt < max_attempts - 1:
            retry_after = r.headers.get("retry-after")
            try:
                wait = float(retry_after) if retry_after else delay
            except ValueError:
                wait = delay
            await asyncio.sleep(min(wait, 20.0))
            delay *= _BACKOFF_BASE
            last = RuntimeError(f"HTTP {r.status_code} from {model}")
            continue

        r.raise_for_status()
        return r.json()
    raise last or RuntimeError(f"no response from {model}")


def _apply_route(payload: dict, plan: dict) -> dict:
    routed = dict(payload)
    routed.pop('provider', None)
    if plan.get('provider'):
        routed['provider'] = plan['provider']
    if plan['mode'] != 'standard':
        routed['service_tier'] = plan['tier']
    return routed


def _early_flex_rejection(data: dict, progressed: bool = False) -> bool:
    """Only explicit capacity rejection before any generation can change tiers.

    HTTP 200 can carry an OpenRouter error in SSE or JSON. Never replay partial
    text/tool calls/reasoning, reported usage, or an unknown/malformed outcome.
    """
    error = data.get('error')
    if progressed or not isinstance(error, dict) or data.get('usage'):
        return False
    for choice in data.get('choices') or []:
        for key in ('message', 'delta'):
            value = choice.get(key) or {}
            if any(v for k, v in value.items() if k != 'role'):
                return False
    try:
        code = int(error.get('code', 0))
    except (ValueError, TypeError):
        return False
    message = str(error.get('message', '')).lower()
    meta = error.get('metadata')
    kind = meta.get('error_type') if isinstance(meta, dict) else None
    explicit = kind == 'provider_unavailable' or ('flex' in message and any(
        word in message for word in ('unavailable', 'capacity', 'exhausted', 'not supported')))
    return explicit and flex.fallback_allowed(code, message)


async def _post_routed(payload: dict, reasoning, model: str, route_plans: list[dict]):
    for index, plan in enumerate(route_plans):
        try:
            data = await _post_with_backoff(_apply_route(payload, plan), reasoning, model)
            if isinstance(data, dict) and index + 1 < len(route_plans) and plan['tier'] == 'flex' and _early_flex_rejection(data):
                continue
            return data, plan
        except httpx.HTTPStatusError as exc:
            if index + 1 < len(route_plans) and plan['tier'] == 'flex' and flex.fallback_allowed(
                    exc.response.status_code, exc.response.text[:4000]):
                continue
            raise
    raise RuntimeError('No routing plan')


async def complete(
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int = 2000,
    temperature: float = 0.0,
    response_format: dict | None = None,
    reasoning_effort: str = "high",
    provider_policy: str = provmod.POLICY_EU_US,
    session_id: str | None = None,
    usage_callback=None,
    pricing_mode: str = "standard",
) -> str:
    """Non-streaming completion. Used by the summariser and the doc readers.

    `reasoning_effort` is honoured first, then stepped down. On a reasoning
    model the reasoning tokens come out of the SAME `max_tokens` budget as the
    answer, so a high-effort call can consume the whole budget and return an
    empty string -- which looks exactly like a model that "can't follow the
    format", and silently turned several good candidates into apparent failures
    when this was first measured. Some endpoints also refuse to have reasoning
    disabled at all (`"Reasoning is mandatory for this endpoint"`), which is why
    the ladder exists rather than a single setting.
    """
    if anthropic_api.is_direct(model):
        return await anthropic_api.complete(
            model=model, messages=messages, max_tokens=max_tokens, reasoning_effort=reasoning_effort,
            response_format=response_format, usage_callback=usage_callback, pricing_mode=pricing_mode)
    attempts = _reasoning_ladder(reasoning_effort, model)

    route_plans = await flex.plans(model, provider_policy, pricing_mode)

    last_error: Exception | None = None
    for reasoning in attempts:
        payload: dict[str, Any] = {
            "model": model,
            "messages": anthropic_api.strip_tool_changes(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if session_id:
            payload["session_id"] = session_id
        if response_format:
            payload["response_format"] = response_format
        if reasoning is not None:
            payload["reasoning"] = reasoning
        try:
            data, used_plan = await _post_routed(payload, reasoning, model, route_plans)
            route_plans = route_plans[route_plans.index(used_plan):]
            if data is _RETRY_REASONING:
                last_error = RuntimeError("endpoint mandates reasoning")
                continue
        except (httpx.HTTPStatusError, httpx.RequestError):
            raise  # do not retry an unavailable tier by changing reasoning effort
        except Exception:
            raise  # unknown execution outcome must not be replayed with another effort

        if usage_callback and data.get("usage"):
            await usage_callback(data['usage'], model=data.get('model') or model,
                                 provider=data.get('provider'), generation_id=data.get('id'),
                                 routing=flex.metadata(used_plan, data.get('service_tier')))
        if "choices" not in data or not data["choices"]:
            raise RuntimeError(f"unexpected response: {json.dumps(data)[:400]}")
        content = data["choices"][0]["message"].get("content") or ""
        if content.strip():
            return content
        # Empty content with budget burned => reasoning ate it. Try the next
        # attempt, which asks for less of it.
        last_error = RuntimeError(
            f"empty completion from {model} "
            f"(completion_tokens={(data.get('usage') or {}).get('completion_tokens')})"
        )

    raise last_error or RuntimeError(f"no completion from {model}")


async def stream_chat(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    max_tokens: int | None = 8000,
    temperature: float = 0.0,
    frozen_msg_count: int = 0,
    reasoning_effort: str = "high",
    provider_policy: str = provmod.POLICY_EU_US,
    session_id: str | None = None,
    previous_cache_anchor: int | None = None,
    cache_ttl: str = "5m",
    pricing_mode: str = "standard",
    uncached_tail: int = 0,
) -> AsyncIterator[dict[str, Any]]:
    """Stream a chat completion. The last `uncached_tail` messages (the live canvas) stay
    behind the tail cache mark.

    Yields dicts:
      {"type":"text","text":...}
      {"type":"tool_calls","tool_calls":[...]}   (once, when complete)
      {"type":"usage","usage":{...}}
      {"type":"done","finish_reason":...}
      {"type":"error","error":...}

    `max_tokens=None` (unlimited output, model maximum unknown): left out, the provider
    decides. The direct Anthropic API always needs one; it gets the model's maximum.
    """
    if anthropic_api.is_direct(model):
        async for ev in anthropic_api.stream_chat(
                model=model, messages=messages, tools=tools,
                max_tokens=max_tokens or model_output_limit(model) or 32_000,
                frozen_msg_count=frozen_msg_count, reasoning_effort=reasoning_effort,
                previous_cache_anchor=previous_cache_anchor, cache_ttl=cache_ttl,
                pricing_mode=pricing_mode, uncached_tail=uncached_tail):
            yield ev
        return
    msgs = anthropic_api.strip_tool_changes(expand_messages(apply_cache_breakpoints(
        messages, model, frozen_msg_count, previous_cache_anchor, cache_ttl, trailer=uncached_tail)))
    payload: dict[str, Any] = {
        "model": model,
        "messages": msgs,
        "temperature": temperature,
        "stream": True,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if session_id:
        payload["session_id"] = session_id
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    route_plans = await flex.plans(model, provider_policy, pricing_mode, needs_tools=bool(tools))
    reasoning = reasoning_payload(reasoning_effort)
    if reasoning is not None:
        payload["reasoning"] = reasoning

    plan_index = 0
    retries = 0
    credit_retry = False
    announced: set[int] = set()
    while plan_index < len(route_plans):
        plan = route_plans[plan_index]
        acc_tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason = None
        usage_snapshot = {}
        metadata = {"model": model}
        reasoning_blocks = []
        progressed = False
        retry_route = False
        saw_done = False
        retry: tuple[str, float | None] | None = None  # (why, server-suggested wait)
        # Transparent retries only on the standard tier: a Flex rejection has its own
        # fallback rules, and flex_only must surface capacity errors (see flex.py).
        may_retry = plan['tier'] != 'flex' and retries < STREAM_RETRIES

        if plan.get('notice') and plan_index not in announced:
            announced.add(plan_index)
            yield {'type': 'routing', **flex.metadata(plan)}
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                async with client.stream(
                    "POST", OPENROUTER_URL, headers=_headers(), json=_apply_route(payload, plan)
                ) as r:
                    if r.status_code >= 400:
                        body = (await r.aread()).decode("utf-8", "replace")
                        affordable = _affordable_max_tokens(r.status_code, body, payload.get("max_tokens"))
                        if plan_index + 1 < len(route_plans) and plan['tier'] == 'flex' and flex.fallback_allowed(r.status_code, body):
                            retry_route = True  # response rejected BEFORE any stream content
                        elif affordable and not credit_retry:
                            credit_retry = True
                            yield {"type": "notice", "text": (
                                f"OpenRouter credits cover only {affordable} output tokens; this request "
                                f"is retried once with max_tokens={affordable} instead of "
                                f"{payload.get('max_tokens') or 'the model maximum'}.")}
                            payload["max_tokens"] = affordable
                            retry = ("credits", 0.0)
                        elif may_retry and _retryable_status(r.status_code):
                            retry = (f"HTTP {r.status_code}", _retry_after(r.headers.get("retry-after")))
                        else:
                            yield {"type": "error", "error": f"HTTP {r.status_code}: {body[:1200]}"}
                            return
                    elif 'application/json' in r.headers.get('content-type', ''):
                        # Handle pretty-printed JSON as a whole, not one line at a time.
                        try:
                            rejection = json.loads(await r.aread())
                        except (ValueError, TypeError):
                            yield {'type':'error','error':'Invalid JSON response to streaming request; outcome uncertain.'}
                            return
                        if isinstance(rejection, dict) and plan_index + 1 < len(route_plans) and plan['tier'] == 'flex' and _early_flex_rejection(rejection):
                            retry_route = True
                        elif (isinstance(rejection, dict) and may_retry and not rejection.get('usage')
                              and not rejection.get('choices') and _retryable_error(rejection.get('error'))):
                            retry = ("provider error " + _error_code(rejection.get('error')), None)
                        else:
                            if isinstance(rejection, dict) and rejection.get('usage'):
                                yield {'type':'usage','usage':rejection['usage'],'model':rejection.get('model') or model,
                                       'provider':rejection.get('provider'),'generation_id':rejection.get('id'),
                                       'routing':flex.metadata(plan,rejection.get('service_tier'))}
                            yield {'type':'error','error':str(rejection.get('error', 'Unexpected non-stream response')) if isinstance(rejection,dict) else 'Unexpected non-object response'}
                            return
                    else:
                        async for line in r.aiter_lines():
                            if not line:
                                continue
                            if line.startswith('data:'):
                                data = line[5:].strip()
                            elif 'application/json' in r.headers.get('content-type', '') and line.lstrip().startswith('{'):
                                data = line.strip()  # HTTP 200 JSON error, not SSE
                            else:
                                continue
                            if data == "[DONE]":
                                saw_done = True
                                break
                            try:
                                chunk = json.loads(data)
                            except json.JSONDecodeError:
                                continue

                            for key in ('id', 'model', 'provider', 'service_tier'):
                                if chunk.get(key):
                                    metadata[key] = chunk[key]
                            if chunk.get("usage"):
                                _merge_usage(usage_snapshot, chunk['usage'])
                            if chunk.get("error"):
                                if plan_index + 1 < len(route_plans) and plan['tier'] == 'flex' and not usage_snapshot and _early_flex_rejection(chunk, progressed):
                                    retry_route = True
                                    break  # close the rejected response before switching tiers
                                if may_retry and not progressed and not usage_snapshot and _retryable_error(chunk["error"]):
                                    # Nothing was generated: the same request is safe to send again.
                                    retry = ("provider error " + _error_code(chunk["error"]), None)
                                    break
                                if usage_snapshot:
                                    yield {"type": "usage", "usage": usage_snapshot, "model": metadata['model'],
                                           "provider": metadata.get('provider'), "generation_id": metadata.get('id'),
                                           "routing": flex.metadata(plan, metadata.get('service_tier'))}
                                yield {"type": "error", "error": str(chunk["error"]),
                                       **({"interrupted": True} if progressed and _retryable_error(chunk["error"]) else {})}
                                return

                            for choice in chunk.get("choices") or []:
                                delta = choice.get("delta") or {}
                                if any(v for k, v in delta.items() if k != 'role'):
                                    progressed = True
                                if choice.get("finish_reason"):
                                    progressed = True
                                    finish_reason = choice["finish_reason"]
                                if delta.get("reasoning_details"):
                                    _merge_reasoning(reasoning_blocks, delta["reasoning_details"])
                                txt = delta.get("content")
                                if txt:
                                    yield {"type": "text", "text": txt}
                                for tc in delta.get("tool_calls") or []:
                                    idx = tc.get("index", 0)
                                    slot = acc_tool_calls.setdefault(
                                        idx,
                                        {
                                            "id": "",
                                            "type": "function",
                                            "function": {"name": "", "arguments": ""},
                                        },
                                    )
                                    if tc.get("id"):
                                        slot["id"] = tc["id"]
                                    fn = tc.get("function") or {}
                                    if fn.get("name"):
                                        slot["function"]["name"] += fn["name"]
                                    if fn.get("arguments"):
                                        slot["function"]["arguments"] += fn["arguments"]
        except httpx.TransportError as exc:
            # Connection reset, read timeout, peer closed mid-stream, DNS failure, ...
            # A partially streamed tool call is never executed: calls are only handed out
            # once the stream completes. Without progress the request is sent again.
            if may_retry and not progressed and not usage_snapshot:
                retry = (type(exc).__name__, None)
            elif plan['tier'] == 'flex':
                raise  # unknown Flex outcome: never replayed, surfaced as before
            else:
                if usage_snapshot:
                    yield {"type": "usage", "usage": usage_snapshot, "model": metadata['model'],
                           "provider": metadata.get('provider'), "generation_id": metadata.get('id'),
                           "routing": flex.metadata(plan, metadata.get('service_tier'))}
                yield {"type": "error", "error": f"{type(exc).__name__}: {exc or 'connection failed'}",
                       **({"interrupted": True} if progressed else {})}
                return

        if retry_route:
            plan_index += 1
            continue
        if retry is None and not saw_done and finish_reason is None:
            # The connection closed cleanly but the response never finished ([DONE] and a
            # finish_reason are both missing). Tool-call arguments may be cut off.
            if may_retry and not progressed and not usage_snapshot:
                retry = ("stream ended before any output", None)
            else:
                if usage_snapshot:
                    yield {"type": "usage", "usage": usage_snapshot, "model": metadata['model'],
                           "provider": metadata.get('provider'), "generation_id": metadata.get('id'),
                           "routing": flex.metadata(plan, metadata.get('service_tier'))}
                yield {"type": "error", "error": "The provider stream ended before the response was complete "
                       "(no finish_reason); partial output was not used.",
                       **({"interrupted": True} if progressed else {})}
                return
        if retry is not None:
            why, suggested = retry
            if why != "credits":
                retries += 1
                wait = _backoff(retries, suggested)
                yield {"type": "notice", "text": f"Provider request failed before any output ({why}); "
                       f"retrying in {wait:.0f}s ({retries}/{STREAM_RETRIES})."}
                await _sleep(wait)
            continue
        tier_info = flex.metadata(plan, metadata.get('service_tier'))
        if tier_info['tier_warning']:
            yield {'type': 'routing', **tier_info, 'routing_notice': tier_info['tier_warning']}
        if usage_snapshot:
            yield {"type": "usage", "usage": usage_snapshot, "model": metadata['model'],
                   "provider": metadata.get('provider'), "generation_id": metadata.get('id'),
                   "routing": flex.metadata(plan, metadata.get('service_tier'))}
        if reasoning_blocks:
            yield {"type": "reasoning_details", "reasoning_details": reasoning_blocks}
        if acc_tool_calls:
            calls = [acc_tool_calls[k] for k in sorted(acc_tool_calls)]
            for c in calls:
                if not c["id"]:
                    # Unique, not positional: a provider that sends no ids would otherwise
                    # repeat call_0 every iteration, and duplicate ids are rejected.
                    c["id"] = "call_" + uuid.uuid4().hex[:24]
            yield {"type": "tool_calls", "tool_calls": calls}

        yield {"type": "done", "finish_reason": finish_reason}
        return


async def list_models() -> list[dict[str, Any]]:
    """Both catalogues, each entry tagged with the key it needs. One failing
    source must not hide the other; only both failing is an error."""
    errors = []
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            r = await client.get(
                "https://openrouter.ai/api/v1/models", headers={"Accept": "application/json"}
            )
            r.raise_for_status()
            models = [{**m, "provider": "openrouter"} for m in r.json().get("data", [])]
    except Exception as exc:
        models, errors = [], [f"OpenRouter: {exc}"]
    try:
        direct = [{**m, "provider": "anthropic"} for m in await anthropic_api.list_models()]
    except Exception as exc:
        direct = []
        errors.append(f"Anthropic: {exc}")
    if errors and not (models or direct):
        raise RuntimeError("; ".join(errors))
    return direct + models
