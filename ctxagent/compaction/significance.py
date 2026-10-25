"""Tier 2a -- significance scoring. Zero LLM calls.

Every published scheme I found compresses purely by **age**. Age is a proxy for
what you actually care about, which is whether a segment still carries decision-
relevant information. Two observations of the same age can differ enormously:
a 300-line stack trace from an error that was fixed three turns ago is worthless;
a one-line note recording *why* a design choice was made is load-bearing forever.

So we score explicitly. All features are deterministic and cheap; the score is a
weighted blend in [0,1] which the condensation pass and the deep-compaction
summariser both consume.

Design rules that fall out of the literature:

* User messages are never compacted (Codex deletes assistant/tool messages but
  "preserves all user messages verbatim"). Score pinned to 1.0.
* Tool *calls* and reasoning are "internal" in TokenPilot's taxonomy: cheap,
  high-signal, kept. Tool *results* are "environmental": expensive, mostly
  stale, compressed.
* Forward reference matters most: if a later message mentions this file path,
  symbol or error string, the observation is still live.
"""

from __future__ import annotations

import re
from collections import Counter

from ..segments import (
    Conversation,
    Segment,
    K_USER,
    K_SYSTEM,
    K_SUMMARY,
    K_ASSISTANT,
    K_TOOL_CALL,
    K_TOOL_RESULT,
)
from .. import tokens

_PATH_RE = re.compile(r"[A-Za-z]:[\\/][\w\-./\\ ]+|(?:\./|/)?[\w\-]+(?:/[\w\-.]+)+\.\w+")
_SYMBOL_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b")
_ERROR_RE = re.compile(
    r"\b(error|exception|traceback|failed|fatal|panic|denied|not found|cannot)\b", re.I
)
_RESOLVED_RE = re.compile(
    r"\b(fixed|resolved|works now|passes|passing|succeeded|success|done|ok)\b", re.I
)
_DECISION_RE = re.compile(
    r"\b(i(?:'ll| will)|let's|we should|decided|because|instead of|the reason|"
    r"note that|important|caveat|trade-?off|chose|approach)\b",
    re.I,
)
_NOISE_RE = re.compile(r"^(\s*[-=*_#]{3,}\s*|\s*)$")


def _tokens_of(text: str) -> set[str]:
    return set(m.group(0).lower() for m in _SYMBOL_RE.finditer(text or ""))


def _density(body: str) -> float:
    """Fraction of lines that carry information. Catches padded output."""
    lines = (body or "").splitlines()
    if not lines:
        return 0.0
    useful = sum(0 if _NOISE_RE.match(ln) else 1 for ln in lines)
    uniq = len({ln.strip() for ln in lines})
    repeat_factor = uniq / max(1, len(lines))
    return (useful / len(lines)) * (0.4 + 0.6 * repeat_factor)


def score_all(conv: Conversation, hot_start: int) -> None:
    """Assign `significance` to every segment. Idempotent."""
    segs = conv.segments
    n = len(segs)
    if n == 0:
        return

    # Forward index: which vocabulary appears *after* position i.
    # Built once by suffix accumulation so the pass stays O(n * |vocab|).
    suffix_vocab: list[set[str]] = [set() for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        s = segs[i]
        own: set[str] = set()
        if s.kind in (K_USER, K_ASSISTANT, K_SUMMARY, K_TOOL_CALL):
            own = _tokens_of(s.body)
            for tc in s.tool_calls or []:
                own |= _tokens_of(((tc.get("function") or {}).get("arguments")) or "")
        # Cap growth: only carry forward terms that look like identifiers/paths.
        suffix_vocab[i] = own | suffix_vocab[i + 1]
        if len(suffix_vocab[i]) > 4000:
            suffix_vocab[i] = set(list(suffix_vocab[i])[:4000])

    last_seq = segs[-1].seq or 1

    for i, s in enumerate(segs):
        # --- hard rules -------------------------------------------------
        if s.kind in (K_SYSTEM, K_USER, K_SUMMARY):
            s.significance = 1.0
            continue
        if s.pinned:
            s.significance = 1.0
            continue
        if s.kind == K_TOOL_CALL:
            # Tool calls are tiny and define the trajectory. Always keep.
            s.significance = 0.9
            continue

        # --- graded scoring ---------------------------------------------
        recency = (i + 1) / n  # 0..1, newer is higher

        body = s.original_body or s.body
        vocab = _tokens_of(body) | set(
            m.group(0).lower() for m in _PATH_RE.finditer(body)
        )
        future = suffix_vocab[i + 1]
        if vocab:
            overlap = len(vocab & future) / len(vocab)
        else:
            overlap = 0.0

        density = _density(body)

        has_error = bool(_ERROR_RE.search(body))
        # An error that a *later* segment reports as resolved is dead weight.
        resolved_later = False
        if has_error:
            for later in segs[i + 1 : i + 12]:
                if _RESOLVED_RE.search(later.body or ""):
                    resolved_later = True
                    break

        if s.kind == K_ASSISTANT:
            decisionality = 1.0 if _DECISION_RE.search(body) else 0.35
            has_code = 1.0 if "```" in body else 0.0
            score = (
                0.28 * recency
                + 0.24 * overlap
                + 0.30 * decisionality
                + 0.10 * has_code
                + 0.08 * density
            )
            # Assistant prose is cheap; floor it so we never gut the trace.
            score = max(score, 0.45)
        else:  # K_TOOL_RESULT
            size_penalty = min(1.0, s.original_tokens / 4000.0)
            score = (
                0.34 * recency
                + 0.34 * overlap
                + 0.22 * density
                - 0.18 * size_penalty
            )
            if has_error and not resolved_later:
                score += 0.20  # unresolved errors stay legible
            if resolved_later:
                score -= 0.22
            if s.mutates:
                score += 0.12  # the record that a change happened matters

        s.significance = max(0.0, min(1.0, score))


def rank_for_condensation(conv: Conversation, hot_start: int, threshold: float) -> list[Segment]:
    """Hot-zone segments worth condensing, least significant first."""
    out = [
        s
        for i, s in enumerate(conv.segments)
        if i >= hot_start
        and not s.frozen
        and not s.pinned
        and s.kind in (K_TOOL_RESULT, K_ASSISTANT)
        and s.significance < threshold
        and s.tokens() > 250
    ]
    out.sort(key=lambda s: (s.significance, -s.tokens()))
    return out


def distribution(conv: Conversation) -> dict:
    buckets = Counter()
    for s in conv.segments:
        b = min(4, int(s.significance * 5))
        buckets[b] += s.tokens()
    return {str(k): v for k, v in sorted(buckets.items())}
