"""Cache hints plus local, content-free prefix diagnostics.

Frozen means immutable locally, NOT a guarantee of a provider cache hit. Keep an
old anchor when a growing prefix advances by >20 blocks (Anthropic look-back).
The rewritable suffix gets at most a moving tail marker (see `apply(tail=...)`);
its anchor is never reused as a stable one.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import time

from . import tokens


# The system prompt is STATIC text + SESSION_SEPARATOR + per-chat text (working
# directories, user additions). Splitting there and marking only the static block
# lets every chat -- new chats, sub-agents, benchmark trials with different
# workdirs -- share one cached tools+static-system prefix, instead of writing it
# cold per chat. Uses the existing system breakpoint, so the 4-breakpoint budget
# is unchanged; the small session block is covered by the next breakpoint.
SESSION_SEPARATOR = "\n\n## Session\n"


def split_system(message: dict) -> dict:
    """Return the system message as [static, session] text blocks when it has a
    session part; unchanged otherwise (custom templates, already-split content)."""
    content = message.get('content')
    if not isinstance(content, str) or SESSION_SEPARATOR not in content:
        return message
    static, session = content.split(SESSION_SEPARATOR, 1)
    if not static.strip():
        return message
    return {**message, 'content': [{'type': 'text', 'text': static},
                                   {'type': 'text', 'text': SESSION_SEPARATOR + session}]}


def cache_family(model: str) -> str | None:
    model = model.lstrip('~')
    if model.startswith(('anthropic/', 'claude-')):
        return 'anthropic'
    match = re.match(r'openai/gpt-(\d+)(?:\.(\d+))?', model)
    if match and (int(match[1]), int(match[2] or 0)) >= (5, 6):
        return 'openai'
    return None


def eligible_block(message: dict, images: bool = True) -> int | None:
    content = message.get('content')
    if isinstance(content, str):
        return 0 if content.strip() else None
    if isinstance(content, list):
        for i in range(len(content) - 1, -1, -1):
            block = content[i]
            if block.get('type') == 'text' and block.get('text', '').strip():
                return i
            if images and block.get('type') in ('image_ref', 'image_url'):
                return i
    return None


def apply(messages: list[dict], model: str, frozen_msg_count: int,
          previous_anchor: int | None = None, ttl: str = '5m',
          tail: bool = False) -> tuple[list[dict], list[int]]:
    """Mark cache breakpoints. Returned anchors are the STABLE ones only.

    `tail=True` (Anthropic family only) also marks the last cacheable block. The
    suffix may still be rewritten, but between compaction passes every tool
    iteration re-reads it, so a write (1.25x on the new delta) pays for itself on
    the first reuse. After a rewrite the request still reads the longest prefix
    that matches an earlier breakpoint (frozen end, or an older tail) -- a
    partial hit -- and the wasted write is only the delta since the last request.
    """
    family = cache_family(model)
    if not family or not messages:
        return messages, []
    out = copy.deepcopy(messages)
    anchors = []
    if out[0].get('role') == 'system' and eligible_block(out[0], family != 'openai') is not None:
        anchors.append(0)
    end = min(frozen_msg_count, len(out)) - 1
    # Empty assistant text with tool_calls is not a cacheable text block. Walk
    # back to actual content rather than silently losing the entire breakpoint.
    for requested in (previous_anchor, end):
        if requested is None or requested < 0 or requested > end:
            continue
        for i in range(requested, -1, -1):
            if eligible_block(out[i], family != 'openai') is not None:
                if i not in anchors:
                    anchors.append(i)
                break
    marks = list(anchors)
    if tail and family == 'anthropic':
        for i in range(len(out) - 1, -1, -1):
            if eligible_block(out[i]) is not None:
                if i not in marks and len(marks) < 4:
                    marks.append(i)
                break
    for index in marks:
        message = out[index]
        static_system = index == 0 and message.get('role') == 'system' and \
            isinstance(message.get('content'), str) and SESSION_SEPARATOR in message['content']
        if static_system:
            out[index] = message = split_system(message)
            static_system = isinstance(message['content'], list)
        if isinstance(message.get('content'), str):
            message['content'] = [{'type': 'text', 'text': message['content']}]
        block = message['content'][0 if static_system else eligible_block(message, family != 'openai')]
        if family == 'anthropic':
            block['cache_control'] = {'type': 'ephemeral', **({'ttl': '1h'} if ttl == '1h' else {})}
        else:
            block['prompt_cache_breakpoint'] = {'mode': 'explicit'}
    return out, anchors


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


def diagnose(messages: list[dict], schema: list[dict], model: str, frozen_count: int,
             previous: dict | None, reasoning_effort: str, now: float | None = None) -> dict:
    """Persist hashes/counts, never duplicate message contents or base64 in telemetry."""
    now = time.time() if now is None else now
    hashes = [digest(message) for message in messages]
    sizes = [tokens.count_message(message) for message in messages]
    schema_hash = digest(schema)
    common = 0
    reasons = []
    if previous:
        if previous.get('model') != model:
            reasons.append('model changed')
        if previous.get('schema_hash') != schema_hash:
            reasons.append('tool schema changed')
        if previous.get('reasoning_effort') != reasoning_effort:
            reasons.append('reasoning settings changed')
        if not reasons:
            for a, b in zip(previous.get('message_hashes', []), hashes):
                if a != b:
                    break
                common += 1
            if common < min(len(previous.get('message_hashes', [])), len(hashes)):
                reasons.append(f'prefix changed at message {common} (0-based)')
            if common < previous.get('frozen_count', 0):
                reasons.append('previously frozen prefix changed (e.g. deep compaction or restore)')
        if now - previous.get('sent_at', now) > 300:
            reasons.append('gap over 5 minutes; short-lived caches may have expired')
    else:
        reasons.append('first locally tracked request')
    shared_tokens = sum(sizes[:common]) + (tokens.count_tools(schema) if common else 0)
    return {'model': model, 'schema_hash': schema_hash, 'message_hashes': hashes,
            'message_tokens': sizes, 'frozen_count': frozen_count, 'sent_at': now,
            'reasoning_effort': reasoning_effort, 'shared_messages': common,
            'shared_prefix_tokens_estimate': shared_tokens,
            'frozen_tokens_estimate': sum(sizes[:frozen_count]),
            'prompt_tokens_estimate': sum(sizes) + tokens.count_tools(schema),
            'cache_family': cache_family(model) or 'provider-managed', 'reasons': reasons}


def public_diagnostic(data: dict) -> dict:
    return {k: v for k, v in data.items() if k not in ('message_hashes', 'message_tokens', 'schema_hash')}
