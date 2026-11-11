"""Cache breakpoints plus local, content-free prefix diagnostics.

Anthropic caches only at the breakpoints a request sends (at most 4), and an entry lives 5
minutes from the last request that read or wrote it. Each request marks the static system
prompt, up to two "rungs" in the history (plan_rungs) and the tail. The near rung lies about
50k tokens before the end of the history, the far rung halfway to it. A change at
message c (curate_context replaced an item, a live view changed) is re-written from the
newest rung before c; that request marks the last unchanged message as the near rung, so the
next change behind that point costs only what follows it. Live views that appear or change go
behind the previous request's end, and the message before them stays a rung while they are
open: a view change re-writes the views and the history that came after them, not more.
A view that is being worked on (restless_views) is not placed in the history at all: it is
sent at the end, behind the tail mark, where a new version costs only itself.
"""
from __future__ import annotations

import bisect
import copy
import hashlib
import itertools
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
#: The near rung lies about this many provider tokens before the end of the history: a change
#: in the newest part, which is what a curate usually replaces, re-writes about this much.
#: In a history of less than twice as much it lies in the middle. The far rung lies halfway
#: between the start and the near rung's place: a change further back re-writes the newer
#: part and not everything. (Half of the whole history would be the near rung's own place
#: at 100k and within 50k of it up to 200k.)
NEAR_RUNG_TOKENS = 50_000
#: A rung stays where it is while it is off its place by no more than this share, so that one
#: cache entry serves many requests: the near rung up to (1 + NEAR_RUNG_SLACK) times its
#: distance before the end, the far rung within FAR_RUNG_SLACK of its place.
NEAR_RUNG_SLACK = 0.5
FAR_RUNG_SLACK = 0.25
#: A breakpoint looks back at most 20 positions for an older cache entry. A request that
#: appends more messages than this keeps the previous tail as a rung, so its entry is found.
LOOKBACK_GUARD = 12


def plan_rungs(prior: dict | None, history_hashes: list[str], history_tokens: list[int],
               views_hash: str, views_count: int = 0, before_step: int = 0, scale: float = 1.0) -> dict:
    """Rungs and the live-view position for the next request, as indices into the history
    messages (live views not counted). `prior` is the previous request's cache_state, None
    when its cache cannot serve this one (model, tools or reasoning changed). The views are
    the ones placed in the history (the agent sends restless ones at the end instead).
    `scale` is provider tokens per token of `history_tokens`.

    While nothing changed, the two rungs are placed by position:
      * the near rung NEAR_RUNG_TOKENS before the end of the history, and not before its middle,
      * the far rung halfway between the start and the near rung's place.
    A breakpoint on a message writes an entry there at no extra cost, so a rung can be put
    on any message; it serves from the next request on. A rung of the previous request
    stays while it is within the slack of its place. A request that appends more than
    LOOKBACK_GUARD messages has the previous request's tail as its near rung, so that
    entry is found.

    When something changed (an item replaced, the views changed), the request marks the
    newest rung before the change, from which it reads, and the last unchanged message:
    the next change behind it re-writes only what follows. That message then serves as the
    near rung while it is within the slack.

    Live views stay where they are while they and the history before them are unchanged.
    When they appear, change, or a change before them makes their place invalid, they go
    behind the previous request's end (`history_tail`): this request writes everything
    behind the change anyway, and the next view change then re-writes only what came after.
    While views are open the message before them is one of the rungs, in place of the one
    nearest to it, so its entry is warm when they change. Where no previous
    end is known (the history shrank in a compaction, or `prior` is None) they go before the
    newest step: behind `before_step`, the message before the last assistant or user message.

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

    kept = sorted(h for h in prior.get('rungs') or [] if valid(h))
    tail = prior.get('history_tail')
    ends = list(itertools.accumulate(t * scale for t in history_tokens))   # tokens up to the end of each message
    total = ends[-1] if ends else 0
    edit_at = None
    changed = bool(previous) and limit < len(previous)
    if changed:
        if limit >= 2:
            edit_at = limit - 1
        rungs = kept[-1:] + ([edit_at] if edit_at is not None else [])
    else:
        def at(tokens):
            """The last message that ends within `tokens`; neither the system prompt nor the tail."""
            i = bisect.bisect_right(ends, tokens) - 1
            return i if 1 <= i < len(ends) - 1 else None

        near_place = max(total - NEAR_RUNG_TOKENS, total / 2)
        far_place = near_place / 2
        if valid(tail) and len(history_hashes) - 1 - tail > LOOKBACK_GUARD:
            near = tail
        else:
            reach = (total - near_place) * (1 + NEAR_RUNG_SLACK)
            near = max((h for h in kept if total - ends[h] <= reach), default=None)
            if near is None:
                near = at(near_place)
        far = min((h for h in kept if (near is None or h < near)
                   and abs(ends[h] - far_place) <= FAR_RUNG_SLACK * far_place),
                  key=lambda h: abs(ends[h] - far_place), default=None)
        if far is None:
            far = at(far_place)
        rungs = [h for h in (far, near) if h is not None]
    keep_views = had_views and views_count > 0 and not views_changed and old_views < common
    if keep_views:
        views_at = old_views
    elif views_count > 0 and isinstance(tail, int) and 0 < tail < len(history_hashes) \
            and len(history_hashes) >= len(previous):
        views_at = tail
    else:
        views_at = max(0, min(before_step, len(history_hashes) - 1))
    rungs = sorted(set(rungs))
    if views_count > 0 and views_at > 0 and views_at not in rungs:
        if len(rungs) < MAX_RUNGS:
            rungs.append(views_at)
        elif not changed:   # a change request keeps the rung it reads from and the one before the change
            nearest = min(rungs, key=lambda h: abs(ends[h] - ends[views_at]))
            rungs = [h for h in rungs if h != nearest] + [views_at]
    return {'rungs': sorted(rungs), 'views_at': views_at, 'common': common, 'edit_at': edit_at}


#: A view that changes again within this many requests of its last change is restless, and
#: stays so until it has been unchanged for as many.
RESTLESS_WITHIN = 2


def restless_views(prior: dict | None, hashes: dict[str, str]) -> tuple[dict[str, int], set[str]]:
    """Which live views go to the end of the request instead of into the history.

    A view placed in the history is read from the cache while it stays the same, but a new
    version of it re-writes itself, the other views beside it and the history behind them.
    At the end, behind the tail mark, it is never cached and a new version costs nothing
    else. So a view that changes again within RESTLESS_WITHIN requests of its last change
    (a document being edited, with or without a test run in between) is sent at the end
    until it has been unchanged for RESTLESS_WITHIN requests.

    `hashes` maps each view's key to the digest of its message; `prior` is the previous
    request's cache_state. Returns (calm, restless): per key the number of requests in a row
    without a change (0 = new or changed now), and the keys to send at the end."""
    prior = prior or {}
    known, before = prior.get('view_hashes') or {}, prior.get('views_calm') or {}
    was = set(prior.get('views_restless') or [])
    calm = {key: before.get(key, 0) + 1 if known.get(key) == h else 0 for key, h in hashes.items()}
    restless = {key for key, n in calm.items() if n < RESTLESS_WITHIN and (
        key in was or (n == 0 and before.get(key, RESTLESS_WITHIN) < RESTLESS_WITHIN))}
    return calm, restless


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
