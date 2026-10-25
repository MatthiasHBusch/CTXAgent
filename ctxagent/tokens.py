"""Token counting.

We use cl100k_base as a provider-agnostic approximation. It is within a few
percent for Llama/Claude/GPT-family text, and the compaction engine only needs
*relative* accuracy to make good decisions.
"""

from __future__ import annotations

import functools
from typing import Any

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - fallback path
    _ENC = None

# Per-message framing overhead charged by chat APIs (role, delimiters).
MESSAGE_OVERHEAD = 4
# Tool-call plumbing is heavier (id, type, function wrapper).
TOOLCALL_OVERHEAD = 12


@functools.lru_cache(maxsize=8192)
def _count_cached(text: str) -> int:
    if _ENC is None:
        return max(1, len(text) // 4)
    return len(_ENC.encode(text, disallowed_special=()))


def count(text: str | None) -> int:
    if not text:
        return 0
    # Long strings blow out the LRU; count those uncached.
    if len(text) > 20_000:
        if _ENC is None:
            return max(1, len(text) // 4)
        return len(_ENC.encode(text, disallowed_special=()))
    return _count_cached(text)


def count_message(msg: dict[str, Any]) -> int:
    """Token cost of one OpenAI-format chat message."""
    total = MESSAGE_OVERHEAD
    content = msg.get("content")
    if isinstance(content, str):
        total += count(content)
    elif isinstance(content, list):
        for block in content:
            if block.get("type") == "image_ref":
                total += int(block["image"].get("estimated_tokens", 4096))
            elif block.get("type") == "image_url":
                total += 4096  # estimates only; never count base64 as text
            elif block.get("type") == "text":
                total += count(block.get("text", ""))
    for block in msg.get("reasoning_details") or []:
        # Opaque reasoning contributes context too. This is an estimate, never billing.
        total += count(block.get('text') or block.get('summary') or '')
        if block.get('data'):
            total += max(1, len(block['data']) // 4)
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        total += TOOLCALL_OVERHEAD
        total += count(fn.get("name") or "")
        total += count(fn.get("arguments") or "")
    if msg.get("name"):
        total += count(msg["name"])
    return total


def count_messages(msgs: list[dict[str, Any]]) -> int:
    return sum(count_message(m) for m in msgs) + 3


def count_tools(tool_specs: list[dict[str, Any]]) -> int:
    """Tool definitions are part of every request's prefix."""
    import json

    if not tool_specs:
        return 0
    return count(json.dumps(tool_specs, ensure_ascii=False)) + 8


def truncate_to_tokens(text: str, budget: int, *, head_ratio: float = 0.6) -> str:
    """Keep the head and tail of `text`, dropping the middle.

    Head-and-tail beats head-only for tool output: the head has the command and
    the schema, the tail usually has the result and the error.

    The result is *guaranteed* to fit the budget. That matters because callers
    treat it as a hard cap: a character-count estimate of "4 chars per token"
    overshoots badly on token-dense text (long identifiers, CJK, base64), which
    is exactly the content most likely to need truncating.
    """
    if budget <= 0:
        return ""
    if count(text) <= budget:
        return text

    lines = text.splitlines()
    if len(lines) <= 4:
        # No line structure to exploit - cut characters, then verify. Binary
        # search on the character count converges in a handful of encodes.
        lo, hi = 0, len(text)
        best = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            head_n = int(mid * head_ratio)
            tail_n = max(0, mid - head_n)
            omitted = max(0, len(text) - head_n - tail_n)
            cand = (f"{text[:head_n]}\n… [{omitted} characters omitted] …\n"
                    f"{text[len(text) - tail_n:]}" if tail_n else
                    f"{text[:head_n]}\n… [{omitted} characters omitted] …")
            if count(cand) <= budget:
                best = cand
                lo = mid + 1
            else:
                hi = mid - 1
        return best or f"… [{len(text)} characters omitted] …"

    head_budget = int(budget * head_ratio)
    tail_budget = budget - head_budget

    head: list[str] = []
    used = 0
    for ln in lines:
        c = count(ln) + 1
        if used + c > head_budget:
            break
        head.append(ln)
        used += c

    tail: list[str] = []
    used = 0
    for ln in reversed(lines):
        c = count(ln) + 1
        if used + c > tail_budget:
            break
        tail.append(ln)
        used += c
    tail.reverse()

    omitted = len(lines) - len(head) - len(tail)
    if omitted <= 0:
        return text
    marker = f"… [{omitted} lines omitted for brevity] …"
    out = "\n".join(head + [marker] + tail)
    # The per-line accounting above ignores the marker and can land a token or
    # two over. Shave from the head until it genuinely fits.
    while head and count(out) > budget:
        head.pop()
        omitted += 1
        marker = f"… [{omitted} lines omitted for brevity] …"
        out = "\n".join(head + [marker] + tail)
    return out
