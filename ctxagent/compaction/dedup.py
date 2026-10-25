"""Tier 1a -- supersession and duplicate collapse. Zero LLM calls.

This is the pass that answers the motivating example: the agent reads a long
document, edits it, reads it again, edits it again. Three full copies of a 4k-line
file sit in context; only the last one is true.

Two distinct mechanisms, deliberately separated because the evidence differs:

* **Causal supersession.** Segment A and segment B observe the same resource,
  A is older, and a *mutating* operation on that resource happened between
  them. A is then not merely redundant -- it is known-wrong. We can collapse it
  hard and say so. Prior systems dedup by textual similarity, which misses this
  case entirely: v1 and v3 of an edited file are *not* near-duplicates, yet v1
  is exactly the thing you most want gone.

* **Redundant re-observation.** Same resource, no mutation in between. The two
  are near-certainly identical; collapse the older one to a pointer.

Both are non-destructive: `original_body` is retained and the model can pull the
content back with `recall`, which pins the segment against future truncation.
"""

from __future__ import annotations

from ..observations import covers

import hashlib
from collections import defaultdict

from ..segments import (
    Conversation,
    Segment,
    K_TOOL_RESULT,
    S_SUPERSEDED,
)
from .. import tokens


def _fingerprint(body: str) -> str:
    return hashlib.sha1(body.encode("utf-8", "ignore")).hexdigest()[:16]


def _label(seg: Segment) -> str:
    key = seg.resource_key or "?"
    if ":" in key:
        kind, _, val = key.partition(":")
        return val or key
    return key


def _stub(seg: Segment, newer: Segment, *, causal: bool, mutators: list[str],
          sufficient: bool = True) -> str:
    lines = (seg.original_body or seg.body).count("\n") + 1
    tok = seg.original_tokens
    target = _label(seg)
    if causal:
        via = ", ".join(sorted(set(mutators))) or "a modification"
        reason = (
            f"This content is OBSOLETE: {via} changed `{target}` after this "
            f"observation was made."
        )
    else:
        reason = f"Identical re-observation of `{target}` appears later."
    if sufficient:
        pointer = (f"The current version is in the later "
                   f"`{newer.tool_name or 'tool'}` result (step #{newer.seq}). "
                   f"Use that one.")
    else:
        # Every later observation of this resource is shallower than the one we
        # are collapsing, so do not send the model somewhere that cannot answer
        # its question.
        pointer = (f"No later observation captured this resource in full. "
                   f"Re-read `{target}` if you need its current contents.")
    return (
        f"[superseded observation — {lines} lines / ~{tok} tokens elided]\n"
        f"{reason}\n"
        f"{pointer}\n"
        f"If you truly need this historical version, call "
        f"`recall(segment_id=\"{seg.id}\")`."
    )


def run(conv: Conversation, hot_start: int, *, eligible_ids: set[str] | None = None) -> dict:
    """Collapse superseded observations inside the hot zone.

    Only segments at index >= hot_start are eligible; everything before that is
    frozen for cache stability.
    """
    eligible = [
        (i, s)
        for i, s in enumerate(conv.segments)
        if i >= hot_start
        and s.kind == K_TOOL_RESULT
        and not s.mutates
        and not s.frozen
        and not s.pinned
        and s.state not in (S_SUPERSEDED,)
    ]
    if len(eligible) < 2:
        return {"freed": 0, "superseded": 0, "duplicates": 0, "details": []}

    # Index every mutating event per resource, across the *whole* conversation
    # (a mutation in the frozen prefix still invalidates a later-but-older read).
    mutations: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for i, s in enumerate(conv.segments):
        if s.mutates and s.resource_key:
            mutations[s.resource_key].append((s.seq, s.tool_name or "an edit"))

    by_resource: dict[str, list[tuple[int, Segment]]] = defaultdict(list)
    for i, s in eligible:
        if s.resource_key:
            by_resource[s.resource_key].append((i, s))

    freed = 0
    n_super = 0
    n_dup = 0
    details: list[dict] = []

    for key, group in by_resource.items():
        if len(group) < 2:
            continue
        group.sort(key=lambda t: t[1].seq)
        newest = group[-1][1]
        muts = mutations.get(key, [])
        for i, (_, seg) in enumerate(group[:-1]):
            if eligible_ids is not None and seg.id not in eligible_ids:
                continue  # newer hot evidence can replace, but cannot be rewritten
            # Cheap guard: don't stub something already smaller than the stub.
            if seg.tokens() < 120:
                continue
            # Was there a mutation strictly between this observation and the newest?
            between = [name for seq, name in muts if seg.seq < seq <= newest.seq]
            causal = bool(between)
            # A later observation may only supersede this one if it is in the
            # SAME channel and captured at least as much. Without the detail
            # test a level-0 outline would destroy the full read it followed;
            # without the channel test a structural outline would destroy an
            # LLM summary, which contains things no outline can express.
            deeper = [s for _, s in group[i + 1:]
                      if s.channel == seg.channel and s.detail >= seg.detail]
            # Mutation alone proves staleness, NOT that replacement covers it.
            deeper = [s for s in deeper if covers(s.observation, seg.observation)]
            if not deeper:
                continue
            target_seg = deeper[-1] if deeper else newest
            body = _stub(seg, target_seg, causal=causal, mutators=between,
                         sufficient=bool(deeper))
            gained = seg.rewrite(
                body,
                S_SUPERSEDED,
                "obsolete after edit" if causal else "duplicate read",
            )
            if gained > 0:
                freed += gained
                if causal:
                    n_super += 1
                else:
                    n_dup += 1
                details.append(
                    {
                        "segment": seg.id,
                        "resource": key,
                        "causal": causal,
                        "freed": gained,
                    }
                )

    # Cross-resource exact-duplicate collapse (e.g. the same command run twice,
    # the same grep repeated). Keyed on the body fingerprint.
    seen: dict[str, Segment] = {}
    for _, s in eligible:
        if s.state == S_SUPERSEDED or s.tokens() < 200:
            continue
        fp = _fingerprint(s.original_body or s.body)
        prev = seen.get(fp)
        if prev is None:
            seen[fp] = s
            continue
        # keep the newest, stub the older
        older, newer = (prev, s) if prev.seq < s.seq else (s, prev)
        seen[fp] = newer
        if older.state == S_SUPERSEDED or (eligible_ids is not None and older.id not in eligible_ids):
            continue
        if newer.observation.get('delivery_limited') and not older.observation.get('delivery_limited'):
            continue  # archived originals are equal; delivered evidence is NOT
        gained = older.rewrite(
            f"[duplicate output — identical to the later `{newer.tool_name or 'tool'}` "
            f"result at step #{newer.seq}; ~{older.original_tokens} tokens elided]\n"
            f"Call `recall(segment_id=\"{older.id}\")` to restore.",
            S_SUPERSEDED,
            "byte-identical duplicate",
        )
        if gained > 0:
            freed += gained
            n_dup += 1
            details.append(
                {"segment": older.id, "resource": "identical-output", "causal": False, "freed": gained}
            )

    return {
        "freed": freed,
        "superseded": n_super,
        "duplicates": n_dup,
        "details": details,
    }
