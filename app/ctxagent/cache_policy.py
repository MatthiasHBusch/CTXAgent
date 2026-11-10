"""Cache breakpoints plus local, content-free prefix diagnostics.

Anthropic caches only at the breakpoints a request sends (at most 4), and an entry lives 5
minutes from the last request that read or wrote it. Each request marks the static system
prompt, up to two "rungs" in the history (plan_rungs) and the tail. A change at message c
(curate_context replaced an item, a live view changed) is re-written from the newest valid
cache entry before c; the request after it marks the last unchanged message, so the next
change behind that point costs only what follows it. Live views that appear or change go
behind the previous request's end, and the message before them stays a rung while they are
open: a view change re-writes the views and the history that came after them, not more.
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


#: Breakpoints Anthropic accepts per request.
MAX_BREAKPOINTS = 4
#: Rungs kept in the history: the 4 breakpoints are the system prompt, two rungs, the tail.
MAX_RUNGS = 2
#: A new rung (the previous request's tail) once the history behind the newest one has grown
#: by this many local tokens: a change in the newer part then re-writes at most about this much.
RUNG_SPACING_TOKENS = 20_000
#: A breakpoint looks back at most 20 positions for an older cache entry. A request that
#: appends more messages than this keeps the previous tail as a rung, so its entry is found.
LOOKBACK_GUARD = 12


def plan_rungs(prior: dict | None, history_hashes: list[str], history_tokens: list[int],
               views_hash: str, views_count: int = 0) -> dict:
    """Rungs and the live-view position for the next request, as indices into the history
    messages (live views not counted). `prior` is the previous request's cache_state, None
    when its cache cannot serve this one (model, tools or reasoning changed).

    A rung is
      * the last unchanged message before a change (an item replaced, the views changed), so
        the next change behind it re-writes only what follows, or
      * the previous request's tail, once the history behind the newest rung has grown by
        RUNG_SPACING_TOKENS, or when this request appends more than LOOKBACK_GUARD messages.
    Every request marks its rungs, which keeps their entries warm; the newest MAX_RUNGS stay.

    Live views stay where they are while they and the history before them are unchanged.
    When they appear, change, or a change before them makes their place invalid, they go
    behind the previous request's end (`history_tail`): this request writes everything
    behind the change anyway, and the next view change then re-writes only what came after.
    While views are open the message before them is one of the rungs (the other is the
    newest remaining one), so its entry is warm whenever they change. Views in a history
    that shrank (a compaction) go behind the newest rung, as do views without a prior request.

    Returns {'rungs', 'views_at' (views go after this message), 'common' (unchanged leading
    history messages), 'edit_at' (last unchanged message before a change, or None)}."""
    prior = prior or {}
    previous = prior.get('history_hashes') or []
    common = 0
    for a, b in zip(previous, history_hashes):
        if a != b:
            break
        common += 1
    old_views = prior.get('views_at')
    had_views = bool(prior.get('views_count')) and old_views is not None
    views_changed = bool(previous) and prior.get('views_hash') != views_hash
    # A cache entry at message h survives while messages 0..h are unchanged (and, behind
    # views the last request sent, those views too).
    limit = common
    if views_changed and had_views:
        limit = min(limit, old_views + 1)

    def valid(h) -> bool:
        return isinstance(h, int) and 0 < h < limit

    rungs = [h for h in prior.get('rungs') or [] if valid(h)]
    edit_at = None
    if previous and limit < len(previous) and limit >= 2:
        edit_at = limit - 1
        rungs.append(edit_at)
    tail = prior.get('history_tail')
    if valid(tail):
        newest = max(rungs, default=0)
        grown = sum(history_tokens[newest + 1:tail + 1])
        if grown >= RUNG_SPACING_TOKENS or len(history_hashes) - 1 - tail > LOOKBACK_GUARD:
            rungs.append(tail)
    rungs = sorted(set(rungs))
    keep_views = had_views and views_count > 0 and not views_changed and old_views < common
    if keep_views:
        views_at = old_views
    elif views_count > 0 and isinstance(tail, int) and 0 < tail < len(history_hashes) \
            and len(history_hashes) >= len(previous):
        views_at = tail
    else:
        views_at = max(rungs, default=0)
    if views_count > 0 and views_at > 0:
        others = [h for h in rungs if h != views_at]
        rungs = sorted(others[len(others) - (MAX_RUNGS - 1):] + [views_at])
    else:
        rungs = rungs[-MAX_RUNGS:]
    return {'rungs': rungs, 'views_at': views_at, 'common': common, 'edit_at': edit_at}


def apply(messages: list[dict], model: str, anchors=(), ttl: str = '5m',
          tail: bool = False, trailer: int = 0) -> tuple[list[dict], list[int]]:
    """Mark cache breakpoints: the system prompt, the messages in `anchors` (the rungs), and
    with `tail=True` (Anthropic family only) the last cacheable block. Returns the marked
    stable ones (system prompt and rungs, not the tail).

    `trailer` messages at the end (the live canvas) stay behind the tail mark: they change
    from request to request, and the history before them should be cached as usual.
    """
    family = cache_family(model)
    if not family or not messages:
        return messages, []
    out = copy.deepcopy(messages)
    marks_left = MAX_BREAKPOINTS - int(bool(tail and family == 'anthropic'))
    found = []
    if out[0].get('role') == 'system' and eligible_block(out[0], family != 'openai') is not None:
        found.append(0)
    # Empty assistant text with tool_calls is not a cacheable text block. Walk
    # back to actual content rather than silently losing the entire breakpoint.
    for requested in sorted({a for a in anchors or () if isinstance(a, int) and 0 <= a < len(out)}):
        for i in range(requested, -1, -1):
            if eligible_block(out[i], family != 'openai') is not None:
                if i not in found:
                    found.append(i)
                break
    if len(found) > marks_left:   # the system prompt (or oldest) and the newest rungs
        found = found[:1] + found[len(found) - (marks_left - 1):]
    anchors = found
    marks = list(anchors)
    if tail and family == 'anthropic':
        for i in range(len(out) - 1 - max(0, trailer), -1, -1):
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


def same_cache(previous: dict | None, schema: list[dict], model: str, reasoning_effort: str) -> bool:
    """Whether the previous request's cache can serve this one at all."""
    return bool(previous) and previous.get('model') == model and \
        previous.get('schema_hash') == digest(schema) and previous.get('reasoning_effort') == reasoning_effort


def diagnose(messages: list[dict], schema: list[dict], model: str,
             previous: dict | None, reasoning_effort: str, now: float | None = None, *,
             hashes: list[str] | None = None, sizes: list[int] | None = None) -> dict:
    """Persist hashes/counts, never duplicate message contents or base64 in telemetry.
    `hashes` and `sizes` are the per-message digests and token counts when the caller has
    them already (the agent computes them once, for plan_rungs)."""
    now = time.time() if now is None else now
    if hashes is None:
        hashes = [digest(message) for message in messages]
    if sizes is None:
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
        if now - previous.get('sent_at', now) > 300:
            reasons.append('gap over 5 minutes; short-lived caches may have expired')
    else:
        reasons.append('first locally tracked request')
    shared_tokens = sum(sizes[:common]) + (tokens.count_tools(schema) if common else 0)
    return {'model': model, 'schema_hash': schema_hash, 'message_hashes': hashes,
            'message_tokens': sizes, 'sent_at': now,
            'reasoning_effort': reasoning_effort, 'shared_messages': common,
            'shared_prefix_tokens_estimate': shared_tokens,
            'prompt_tokens_estimate': sum(sizes) + tokens.count_tools(schema),
            'cache_family': cache_family(model) or 'provider-managed', 'reasons': reasons}


def public_diagnostic(data: dict) -> dict:
    return {k: v for k, v in data.items() if k not in ('message_hashes', 'message_tokens', 'schema_hash')}


def legacy_anchors(cache_anchors, frozen_msg_count: int = 0, previous_cache_anchor: int | None = None) -> list[int]:
    """Rungs to mark: `cache_anchors`, or for callers of the frozen-prefix era (benchmark
    probes, replays) the end of their frozen prefix and their previous anchor."""
    if cache_anchors is not None:
        return [a for a in cache_anchors if isinstance(a, int)]
    return [a for a in (previous_cache_anchor, (frozen_msg_count or 0) - 1) if isinstance(a, int) and a >= 0]
