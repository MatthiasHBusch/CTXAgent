"""Direct Anthropic Messages API backend.

Selected by model id: a bare Claude id (`claude-opus-5-5`) goes here, anything
with a provider prefix (`anthropic/claude-sonnet-5`) still goes to OpenRouter.
The rest of ctxagent speaks OpenAI chat format, so this module translates in
both directions and yields exactly the events `llm.stream_chat` yields.

Things that differ from the OpenRouter path and are easy to get wrong:

* **Caching is opt-in.** The Anthropic API caches nothing without
  `cache_control`. `cache_policy.apply` already marks the system prompt and the
  frozen prefix; on top of that we send top-level automatic caching, which puts
  a moving breakpoint on the last block. Within a tool loop every iteration then
  reads the previous iteration's prefix instead of re-billing the hot zone.
* **Preserved thinking.** Opus 5.5 / Sonnet 5.5 / Fable 5.1 bind each thinking
  block's signature to the exact prefix before it. Compaction rewrites that
  prefix, which is a 400 on accounts that are enforced. We send
  `prefix_mismatch_behavior: "drop_block"`, so the API drops stale blocks and
  the request proceeds; blocks from untouched turns keep working.
* **No sampling parameters.** Current models reject `temperature`; it is never
  sent.
* Reasoning is stored in OpenRouter's `reasoning_details` shape
  (`format: anthropic-claude-v1`), so a chat can move between the OpenRouter
  and direct routes without losing its thinking blocks.
"""

from __future__ import annotations

import json
import re
from typing import Any, AsyncIterator

from .artifacts import expand_messages
from .config import anthropic_api_key, anthropic_workspace_id
from . import cache_policy

BLOCK_BINDING_BETA = "thinking-binding-controls-2026-08-01"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
REASONING_FORMAT = "anthropic-claude-v1"

#: Models whose server-side refusal fallback (`fallbacks: "default"`) we enable.
_FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5")

#: Features the account/endpoint rejected once; learned at runtime so the 400
#: is paid once per process (same pattern as llm._REASONING_MANDATORY).
_REJECTED: set[str] = set()

#: USD per million tokens: (input, output, cache read). Cache writes are 1.25x
#: input for the 5-minute TTL. Anthropic list prices as of 2026-10.
PRICES: dict[str, tuple[float, float, float]] = {
    "claude-fable-5-1": (10.0, 50.0, 0.25),
    "claude-fable-5": (10.0, 50.0, 1.0),
    "claude-opus-5-5": (4.0, 20.0, 0.20),
    "claude-opus-5": (5.0, 25.0, 0.50),
    "claude-opus-4-8": (5.0, 25.0, 0.50),
    "claude-opus-4-7": (5.0, 25.0, 0.50),
    "claude-opus-4-6": (5.0, 25.0, 0.50),
    "claude-sonnet-5-5": (2.0, 10.0, 0.20),
    "claude-sonnet-5": (2.0, 10.0, 0.20),
    "claude-sonnet-4-6": (3.0, 15.0, 0.30),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "claude-haiku-4-5": (1.0, 5.0, 0.10),
}
#: Claude Haiku 5.5 is billed at a higher rate above 100k prompt tokens.
_LONG_PROMPT = {"claude-haiku-5-5": (100_000, (0.50, 2.50, 0.05))}


def is_direct(model: str) -> bool:
    return model.startswith("claude-")


def _base(model: str) -> str:
    """`claude-opus-5-5-20260101` -> `claude-opus-5-5` for table lookups."""
    return re.sub(r"-\d{8}$", "", model)


def rates(model: str, prompt_tokens: int = 0) -> dict[str, float] | None:
    """Per-token rates in accounting's key names, or None for an unknown model."""
    base = _base(model)
    price = PRICES.get(base)
    if price is None:
        return None
    threshold, long_price = _LONG_PROMPT.get(base, (None, None))
    if threshold is not None and prompt_tokens > threshold:
        price = long_price
    i, o, r = (p / 1e6 for p in price)
    return {"prompt": i, "completion": o, "input_cache_read": r, "input_cache_write": i * 1.25}


# ---------------------------------------------------------------------------
# Thinking / effort per model
# ---------------------------------------------------------------------------

def _thinking_config(model: str, effort: str) -> tuple[dict | None, str | None]:
    """(`thinking` param, `output_config.effort`) for a configured effort level.

    "off" cannot be expressed the same way everywhere: Opus 5.5 and Fable reject
    every way of disabling thinking (lowest effort is the closest), Sonnet 5.5
    uses `between_tools`, the rest accept `disabled`.
    """
    base = _base(model)
    effort = (effort or "default").lower()
    always_on = base.startswith(("claude-fable", "claude-mythos")) or base == "claude-opus-5-5"
    on_by_default = always_on or base in ("claude-opus-5", "claude-sonnet-5-5", "claude-sonnet-5", "claude-haiku-5-5")
    adaptive_capable = on_by_default or base in ("claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6")
    if not adaptive_capable:
        return None, None  # older models: budget_tokens era; leave provider defaults
    if effort == "off":
        if always_on:
            return {"type": "adaptive"}, "low"
        if base == "claude-sonnet-5-5":
            return {"type": "between_tools"}, None
        return {"type": "disabled"}, None
    if effort in ("low", "medium", "high", "xhigh", "max"):
        return {"type": "adaptive"}, effort
    # "default": keep the model's own default, but say so explicitly where it
    # thinks by default so the block-binding control can ride along.
    return ({"type": "adaptive"} if on_by_default else None), None


# ---------------------------------------------------------------------------
# OpenAI chat format -> Anthropic Messages format
# ---------------------------------------------------------------------------

def _tool_id(raw: str | None, fallback: str) -> str:
    """Anthropic ids must match ^[a-zA-Z0-9_-]+$; other providers' ids may not."""
    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", raw or "")
    return clean or fallback


def _cache(block: dict) -> dict:
    return {"cache_control": block["cache_control"]} if block.get("cache_control") else {}


def _content_blocks(content: Any) -> list[dict]:
    """User-side content (text and data-URL images) as Anthropic blocks."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content.strip() else []
    out = []
    for block in content:
        kind = block.get("type")
        if kind == "text":
            if block.get("text", "").strip():
                out.append({"type": "text", "text": block["text"], **_cache(block)})
        elif kind == "image_url":
            url = (block.get("image_url") or {}).get("url", "")
            match = re.match(r"data:([^;]+);base64,(.*)", url, re.S)
            source = ({"type": "base64", "media_type": match[1], "data": match[2]} if match
                      else {"type": "url", "url": url})
            out.append({"type": "image", "source": source, **_cache(block)})
    return out


def _thinking_blocks(details: list[dict] | None) -> list[dict]:
    """Replay only blocks Anthropic produced; other providers' reasoning is opaque to it."""
    out = []
    for item in details or []:
        if not str(item.get("format", "")).startswith("anthropic-claude"):
            continue
        if item.get("type") == "reasoning.text" and item.get("signature"):
            out.append({"type": "thinking", "thinking": item.get("text") or "", "signature": item["signature"]})
        elif item.get("type") == "reasoning.encrypted" and item.get("data"):
            out.append({"type": "redacted_thinking", "data": item["data"]})
    return out


def _tool_calls(calls: list[dict], start: int) -> list[dict]:
    out = []
    for i, call in enumerate(calls or []):
        fn = call.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {"_unparsed_arguments": fn.get("arguments")}
        if not isinstance(args, dict):
            args = {"value": args}
        out.append({"type": "tool_use", "id": _tool_id(call.get("id"), f"call_{start + i}"),
                    "name": fn.get("name") or "unknown", "input": args})
    return out


def convert_tools(tools: list[dict] | None) -> list[dict]:
    out = []
    for tool in tools or []:
        fn = tool.get("function", tool)
        out.append({"name": fn["name"], "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters") or {"type": "object", "properties": {}}})
    return out


def convert_messages(messages: list[dict], keep_thinking: bool = True) -> tuple[list[dict], list[dict]]:
    """Return (system blocks, messages) for the Messages API.

    Consecutive tool results become ONE user message (parallel tool results
    belong together), followed by any user content in the same run.
    """
    system: list[dict] = []
    out: list[dict] = []
    index = 0
    while index < len(messages) and messages[index].get("role") in ("system", "developer"):
        system.extend(_content_blocks(messages[index].get("content")))
        index += 1

    def push(role: str, blocks: list[dict]) -> None:
        if not blocks:
            return
        if out and out[-1]["role"] == role == "user":
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": blocks})

    call_index = 0
    for message in messages[index:]:
        role = message.get("role")
        if role == "assistant":
            blocks = _thinking_blocks(message.get("reasoning_details")) if keep_thinking else []
            blocks += [b for b in _content_blocks(message.get("content")) if b["type"] == "text"]
            calls = _tool_calls(message.get("tool_calls") or [], call_index)
            call_index += len(calls)
            blocks += calls
            if any(b["type"] in ("text", "tool_use") for b in blocks):
                push("assistant", blocks)
        elif role == "tool":
            content = message.get("content")
            inner = _content_blocks(content) or [{"type": "text", "text": "(empty)"}]
            cache = next((b["cache_control"] for b in inner if b.get("cache_control")), None)
            for b in inner:
                b.pop("cache_control", None)
            push("user", [{"type": "tool_result", "tool_use_id": _tool_id(message.get("tool_call_id"), "call_missing"),
                           "content": inner, **({"cache_control": cache} if cache else {})}])
        elif role in ("system", "developer"):
            push("user", [{"type": "text", "text": "[system note]\n" + "".join(
                b.get("text", "") for b in _content_blocks(message.get("content")))}])
        else:
            push("user", _content_blocks(message.get("content")) or [{"type": "text", "text": "(empty)"}])
    if out and out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": [{"type": "text", "text": "(conversation continues)"}]})
    return system, out


# ---------------------------------------------------------------------------
# Usage in OpenRouter's shape so accounting.normalize needs no second dialect
# ---------------------------------------------------------------------------

def usage_payload(model: str, usage: dict) -> dict:
    uncached = int(usage.get("input_tokens") or 0)
    read = int(usage.get("cache_read_input_tokens") or 0)
    write = int(usage.get("cache_creation_input_tokens") or 0)
    output = int(usage.get("output_tokens") or 0)
    prompt = uncached + read + write
    raw: dict[str, Any] = {
        "prompt_tokens": prompt, "completion_tokens": output,
        "prompt_tokens_details": {"cached_tokens": read, "cache_write_tokens": write},
        "anthropic_usage": usage,
    }
    rate = rates(model, prompt)
    if rate:
        # 1-hour cache writes cost 2x input instead of 1.25x.
        write_1h = int((usage.get("cache_creation") or {}).get("ephemeral_1h_input_tokens") or 0)
        raw["cost"] = (uncached * rate["prompt"] + read * rate["input_cache_read"]
                       + (write - write_1h) * rate["input_cache_write"] + write_1h * rate["prompt"] * 2
                       + output * rate["completion"])
        raw["cost_basis"] = "computed from Anthropic list prices (ctxagent/anthropic_api.py)"
    return raw


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

_client = None
_client_key = None

_WORKSPACE_HINT = (" Set ANTHROPIC_WORKSPACE_ID in .env to the workspace ID (wrkspc_...) shown under "
                   "Settings > Workspaces in the Claude Console, or use a workspace-scoped key.")


def _get_client():
    global _client, _client_key
    key = anthropic_api_key()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Put it in the .env file next to the app.")
    workspace = anthropic_workspace_id()
    if _client is None or (key, workspace) != _client_key:
        import anthropic
        headers = {"anthropic-workspace-id": workspace} if workspace else None
        _client = anthropic.AsyncAnthropic(api_key=key, max_retries=4, default_headers=headers)
        _client_key = (key, workspace)
    return _client


def _describe(exc) -> str:
    message = str(getattr(exc, "message", "") or exc)
    return message + (_WORKSPACE_HINT if "anthropic-workspace-id" in message else "")


def build_request(*, model: str, messages: list[dict], tools: list[dict] | None, max_tokens: int,
                  reasoning_effort: str, frozen_msg_count: int, previous_cache_anchor: int | None,
                  cache_ttl: str, response_format: dict | None = None, apply_cache: bool = True) -> dict:
    """The full `beta.messages.create` kwargs. Pure; unit-testable offline."""
    thinking, effort = _thinking_config(model, reasoning_effort)
    adaptive = bool(thinking and thinking["type"] == "adaptive")
    binding = adaptive and "block_binding" not in _REJECTED
    # Without the drop_block control a replayed block after compaction can be a
    # 400; when thinking is off there is nothing to gain from replaying anyway.
    keep_thinking = binding
    anchors: list[int] = []
    if apply_cache:
        messages, anchors = cache_policy.apply(messages, model, frozen_msg_count, previous_cache_anchor, cache_ttl)
    system, converted = convert_messages(expand_messages(messages), keep_thinking=keep_thinking)

    kwargs: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "messages": converted}
    betas: list[str] = []
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = convert_tools(tools)
        kwargs["tool_choice"] = {"type": "auto"}
    if thinking:
        thinking = dict(thinking)
        if binding:
            thinking["block_binding"] = {"prefix_mismatch_behavior": "drop_block"}
            betas.append(BLOCK_BINDING_BETA)
        kwargs["thinking"] = thinking
    output_config: dict[str, Any] = {}
    if effort:
        output_config["effort"] = effort
    if response_format and response_format.get("type") == "json_schema":
        schema = (response_format.get("json_schema") or {}).get("schema")
        if schema:
            output_config["format"] = {"type": "json_schema", "schema": schema}
    if output_config:
        kwargs["output_config"] = output_config
    # Moving tail breakpoint. Counts as one of the 4 slots; cache_policy uses <= 3.
    # Same TTL as the explicit markers: a differing TTL on the last block is a 400.
    if apply_cache and len(anchors) < 4:
        kwargs["cache_control"] = {"type": "ephemeral", **({"ttl": "1h"} if cache_ttl == "1h" else {})}
    if (_base(model) in _FALLBACK_MODELS and "fallbacks" not in _REJECTED
            and not (thinking and thinking["type"] == "between_tools")):
        kwargs["fallbacks"] = "default"
        betas.append(FALLBACK_BETA)
    if betas:
        kwargs["betas"] = betas
    return kwargs


def _rejected_feature(exc: Exception) -> str | None:
    """Which optional feature a 400 complains about, if any (then retry without it)."""
    text = str(getattr(exc, "message", "") or exc).lower()
    if "block_binding" in text or BLOCK_BINDING_BETA in text:
        return "block_binding"
    if "fallback" in text:
        return "fallbacks"
    return None


async def stream_chat(*, model: str, messages: list[dict], tools: list[dict] | None = None,
                      max_tokens: int = 8000, frozen_msg_count: int = 0, reasoning_effort: str = "high",
                      previous_cache_anchor: int | None = None, cache_ttl: str = "5m",
                      pricing_mode: str = "standard", response_format: dict | None = None,
                      apply_cache: bool = True, **_ignored) -> AsyncIterator[dict[str, Any]]:
    """Same event protocol as `llm.stream_chat`."""
    import anthropic

    if pricing_mode == "flex_only":
        yield {"type": "error", "error": "Flex pricing is an OpenRouter tier; the direct Anthropic API has none. No request sent."}
        return
    client = _get_client()
    stream = None
    for _attempt in range(3):
        kwargs = build_request(model=model, messages=messages, tools=tools, max_tokens=max_tokens,
                               reasoning_effort=reasoning_effort, frozen_msg_count=frozen_msg_count,
                               previous_cache_anchor=previous_cache_anchor, cache_ttl=cache_ttl,
                               response_format=response_format, apply_cache=apply_cache)
        try:
            stream = await client.beta.messages.create(stream=True, **kwargs)
            break
        except anthropic.BadRequestError as exc:
            feature = _rejected_feature(exc)
            if feature and feature not in _REJECTED:
                _REJECTED.add(feature)
                continue
            yield {"type": "error", "error": f"HTTP 400: {_describe(exc)}"}
            return
        except anthropic.APIStatusError as exc:
            yield {"type": "error", "error": f"HTTP {exc.status_code}: {_describe(exc)}"}
            return
        except anthropic.APIConnectionError as exc:
            yield {"type": "error", "error": f"connection error: {exc}"}
            return
    if stream is None:
        yield {"type": "error", "error": "Anthropic request rejected repeatedly; no response."}
        return

    meta: dict[str, Any] = {"model": model}
    usage: dict[str, Any] = {}
    blocks: dict[int, dict] = {}
    stop_reason = None
    stop_details = None
    try:
        async for event in stream:
            kind = event.type
            if kind == "message_start":
                meta["id"] = event.message.id
                meta["model"] = event.message.model or model
                usage.update(event.message.usage.model_dump(exclude_none=True))
            elif kind == "content_block_start":
                block = event.content_block.model_dump(exclude_none=True)
                if block.get("type") == "fallback":
                    # A model declined mid-output and another continues. Per the API
                    # contract only text before the boundary is echoed back.
                    blocks = {i: b for i, b in blocks.items() if b.get("type") == "text"}
                    continue
                if block.get("type") == "tool_use":
                    block["partial_json"] = ""
                blocks[event.index] = block
            elif kind == "content_block_delta":
                block = blocks.get(event.index)
                delta = event.delta
                if block is None:
                    continue
                if delta.type == "text_delta":
                    block["text"] = block.get("text", "") + delta.text
                    yield {"type": "text", "text": delta.text}
                elif delta.type == "input_json_delta":
                    block["partial_json"] += delta.partial_json
                elif delta.type == "thinking_delta":
                    block["thinking"] = block.get("thinking", "") + delta.thinking
                elif delta.type == "signature_delta":
                    block["signature"] = block.get("signature", "") + delta.signature
            elif kind == "message_delta":
                stop_reason = event.delta.stop_reason or stop_reason
                stop_details = getattr(event.delta, "stop_details", None) or stop_details
                if event.usage:
                    usage.update(event.usage.model_dump(exclude_none=True))
    except anthropic.APIError as exc:
        if usage:
            yield _usage_event(meta, usage)
        yield {"type": "error", "error": f"stream interrupted: {exc}"}
        return
    finally:
        await stream.close()

    if usage:
        yield _usage_event(meta, usage)
    if stop_reason == "refusal":
        detail = stop_details.model_dump() if hasattr(stop_details, "model_dump") else (stop_details or {})
        yield {"type": "error", "error": "Claude declined this request (stop_reason=refusal"
               + (f", category={detail.get('category')}" if detail.get("category") else "") + ")."}
        return

    reasoning, calls = [], []
    for i in sorted(blocks):
        b = blocks[i]
        if b["type"] == "thinking":
            reasoning.append({"type": "reasoning.text", "text": b.get("thinking", ""), "signature": b.get("signature", ""),
                              "format": REASONING_FORMAT, "index": len(reasoning)})
        elif b["type"] == "redacted_thinking":
            reasoning.append({"type": "reasoning.encrypted", "data": b.get("data", ""),
                              "format": REASONING_FORMAT, "index": len(reasoning)})
        elif b["type"] == "tool_use":
            calls.append({"id": b["id"], "type": "function",
                          "function": {"name": b["name"], "arguments": b["partial_json"] or "{}"}})
    if reasoning:
        yield {"type": "reasoning_details", "reasoning_details": reasoning}
    if calls:
        yield {"type": "tool_calls", "tool_calls": calls}
    finish = {"end_turn": "stop", "stop_sequence": "stop", "tool_use": "tool_calls",
              "max_tokens": "length"}.get(stop_reason, stop_reason)
    yield {"type": "done", "finish_reason": finish}


def _usage_event(meta: dict, usage: dict) -> dict:
    return {"type": "usage", "usage": usage_payload(meta["model"], usage), "model": meta["model"],
            "provider": "Anthropic", "generation_id": meta.get("id"), "routing": None}


async def complete(*, model: str, messages: list[dict], max_tokens: int = 2000,
                   reasoning_effort: str = "high", response_format: dict | None = None,
                   usage_callback=None, pricing_mode: str = "standard", **_ignored) -> str:
    """Non-streaming contract of `llm.complete`, implemented over the stream
    (long summariser inputs must not hit HTTP timeouts)."""
    text: list[str] = []
    error = None
    finish = None
    async for ev in stream_chat(model=model, messages=messages, max_tokens=max_tokens,
                                reasoning_effort=reasoning_effort, response_format=response_format,
                                pricing_mode=pricing_mode, apply_cache=False):
        if ev["type"] == "text":
            text.append(ev["text"])
        elif ev["type"] == "usage" and usage_callback:
            await usage_callback(ev["usage"], model=ev["model"], provider=ev["provider"],
                                 generation_id=ev["generation_id"], routing=None)
        elif ev["type"] == "error":
            error = ev["error"]
        elif ev["type"] == "done":
            finish = ev["finish_reason"]
    if error:
        raise RuntimeError(error)
    content = "".join(text)
    if not content.strip():
        raise RuntimeError(f"empty completion from {model} (finish_reason={finish})")
    return content


async def list_models() -> list[dict[str, Any]]:
    """Models visible to this key, in the shape /api/models already returns."""
    if not anthropic_api_key():
        return []
    out = []
    async for m in _get_client().models.list():
        rate = rates(m.id) or {}
        out.append({"id": m.id, "name": f"{m.display_name} (Anthropic direct)",
                    "context_length": getattr(m, "max_input_tokens", None) or 0,
                    "pricing": {"prompt": rate.get("prompt"), "completion": rate.get("completion")},
                    "architecture": {"input_modalities": ["text", "image"]}})
    return out
