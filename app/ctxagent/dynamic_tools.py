"""Native dynamic tools on the direct Anthropic API (beta mid-conversation-tool-changes).

Elsewhere the late-bound operations (document edits, canvas operations) are reached
through `document_call` and a catalog message rebuilt per request (document_catalog).
On models that support mid-conversation tool changes they are declared up front in
`tools[]` with `defer_loading: true` instead, and the history records when they become
available or go away: a `tool_change` segment, rendered as a `role: "system"` message
of `tool_addition` / `tool_removal` blocks. The history stays append-only, so the
cache and the thinking blocks survive an open or close.

What the model can call is the fold over those segments that are still in the history;
before each request `sync` compares it with what is open now and appends the
difference. Compaction may archive tool-change segments; the next `sync` re-adds what
is still open. anthropic_api places the system messages where the API accepts them
(after a user turn, before the next assistant turn) and drops them on other paths.
"""
from __future__ import annotations

from . import document_catalog, tokens

BETA = "mid-conversation-tool-changes-2026-07-01"
#: Models with mid-conversation tool changes (not Claude Sonnet 5).
MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-opus-4-8", "claude-fable-5-1", "claude-fable-5",
          "claude-mythos-5-1", "claude-mythos-5", "claude-sonnet-5-5")
REJECTED = "tool_changes"  # anthropic_api._REJECTED key once the API refused the beta


def supported(model: str) -> bool:
    from . import anthropic_api
    return (anthropic_api.is_direct(model) and anthropic_api._base(model) in MODELS
            and REJECTED not in anthropic_api._REJECTED)


def active(session) -> bool:
    return supported(session.cfg.model) and document_catalog.enabled(session)


def deferred_names(session) -> list[str]:
    """Every enabled late-bound operation, open or not: the fixed deferred part of tools[]."""
    return sorted(n for n in session.tool_names(live_only=False)
                  if n in document_catalog.LATE_BOUND)


def schemas(names) -> list[dict]:
    from .tools import REGISTRY
    return [{"type": "function", "defer_loading": True,
             "function": {"name": n, "description": REGISTRY[n].description,
                          "parameters": document_catalog.parameters(n)}} for n in names]


def surfaced(conv) -> set[str]:
    """What the history has made callable (same fold as anthropic_api.convert_messages)."""
    from .segments import K_TOOL_CHANGE
    names: set[str] = set()
    for s in conv.segments:
        if s.kind == K_TOOL_CHANGE:
            names |= set(s.observation.get("add") or [])
            names -= set(s.observation.get("remove") or [])
    return names


def sync(session):
    """Append the tool-change segment that makes the history match what is open; None
    when nothing changed."""
    from .segments import K_TOOL_CHANGE
    want = set(document_catalog.operation_names(session))
    have = surfaced(session.conv)
    add, remove = sorted(want - have), sorted(have - want)
    if not add and not remove:
        return None
    body = "; ".join(part for part in (
        f"tools available: {', '.join(add)}" if add else "",
        f"tools removed: {', '.join(remove)}" if remove else "") if part)
    return session.conv.new("system", K_TOOL_CHANGE, body, observation={"add": add, "remove": remove})


def surfaced_tokens(session) -> int:
    """Schema tokens of the surfaced operations (what the model holds besides tools[])."""
    names = sorted(surfaced(session.conv) & set(deferred_names(session)))
    return tokens.count_tools(schemas(names)) if names else 0
