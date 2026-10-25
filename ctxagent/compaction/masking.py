"""Tier 1b -- age-tiered observation masking + disk offload. Zero LLM calls.

Straight from "The Complexity Trap" (arXiv 2508.21433): replacing older tool
observations with placeholders while keeping the agent's reasoning and tool
*calls* intact halves cost and matches the solve rate of LLM summarisation.
Cheaper and better, so it runs before anything that costs a model call.

Two refinements over the paper:

* **Progressive tiers** rather than a binary keep/mask (cf. MatClaw's zone
  scheme): the Nth-newest observation gets a shrinking token budget, so context
  decays smoothly with age instead of falling off a cliff.
* **Offload instead of discard** for large bodies (Claude Code's microcompaction):
  the full text goes to a blob on disk and the stub carries a handle, so
  `recall` is lossless rather than best-effort.

Protection (never touched here): pinned segments, the newest
``protected_recent_observations`` results, and EVERY result of the newest
``protected_recent_batches`` tool batches. The batch rule exists because a
single batch can hold 6+ results; a count-only rule then masks results the
agent has not even read yet.

Stub quality: a stub always carries real content - a head/tail excerpt of at
least ``mask_floor_tokens`` - plus, for line-numbered file reads and grep
output, the line range it covered, so the agent can decide whether a recall is
needed without guessing.
"""

from __future__ import annotations

import re

from ..segments import (
    Conversation,
    Segment,
    K_TOOL_CALL,
    K_TOOL_RESULT,
    S_MASKED,
    S_OFFLOADED,
    S_SUPERSEDED,
    S_CONDENSED,
)
from .. import tokens
from ..blobs import put_blob


def _budget_for_rank(rank: int, cfg) -> int:
    """rank 0 = the newest *unprotected* observation."""
    tiers = list(cfg.mask_tiers) or [1000]
    if rank < len(tiers):
        return max(tiers[rank], cfg.mask_floor_tokens)
    return cfg.mask_floor_tokens


def recent_batch_ids(conv: Conversation, n_batches: int) -> set[str]:
    """Ids of every segment from the start of the n-th newest tool batch to the end."""
    if n_batches <= 0:
        return set()
    segs = conv.segments
    seen = 0
    start = None
    for i in range(len(segs) - 1, -1, -1):
        if segs[i].kind == K_TOOL_CALL:
            seen += 1
            start = i
            if seen >= n_batches:
                break
    if start is None:
        return set()
    return {s.id for s in segs[start:]}


_LINE_RX = re.compile(r"^\s*(\d+)\t", re.M)


def _shape(full: str) -> str:
    """One line describing what the full body covered (line span of a numbered read)."""
    first = full.split("\n", 1)[0][:200]
    nums = [int(m.group(1)) for m in _LINE_RX.finditer(full)]
    total_lines = full.count("\n") + 1
    span = f"; numbered lines {nums[0]}-{nums[-1]}" if len(nums) >= 2 else ""
    return f"{total_lines} lines{span}; first line: {first}"


def run(conv: Conversation, hot_start: int, cfg, session_id: str, *, eligible_ids: set[str] | None = None) -> dict:
    """Truncate / offload old observations inside the hot zone."""
    batch_protected = recent_batch_ids(conv, getattr(cfg, "protected_recent_batches", 0))
    obs = [
        s
        for i, s in enumerate(conv.segments)
        if i >= hot_start
        and s.kind == K_TOOL_RESULT
        and not s.frozen
        and not s.pinned
        and s.id not in batch_protected
    ]
    if not obs:
        return {"freed": 0, "masked": 0, "offloaded": 0}

    # Newest first. The most recent `protected_recent_observations` are sacred.
    obs.sort(key=lambda s: s.seq, reverse=True)
    candidates = obs[cfg.protected_recent_observations :]

    freed = 0
    n_masked = 0
    n_offload = 0

    for rank, seg in enumerate(candidates):
        if eligible_ids is not None and seg.id not in eligible_ids:
            continue
        if seg.state in (S_SUPERSEDED, S_CONDENSED):
            continue  # already handled by a stronger pass
        budget = _budget_for_rank(rank, cfg)
        cur = seg.tokens()
        if cur <= budget + 40:
            continue

        full = seg.original_body or seg.body

        # Big bodies: park them on disk so recall is exact.
        if cfg.offload_enabled and seg.original_tokens >= cfg.offload_min_tokens:
            blob_id = seg.blob_id or put_blob(session_id, seg.id, full)
            seg.blob_id = blob_id
            header = (f"[compacted: offloaded ({seg.original_tokens} tokens, {_shape(full)}); "
                      f"recall(segment_id=\"{seg.id}\") restores it exactly]\n")
            kept = tokens.truncate_to_tokens(full, max(budget - tokens.count(header), cfg.mask_floor_tokens))
            gained = seg.rewrite(header + kept, S_OFFLOADED, f"offloaded (age rank {rank})")
            if gained > 0:
                freed += gained
                n_offload += 1
            continue

        kept = tokens.truncate_to_tokens(full, budget)
        if kept == full:
            continue
        body = (
            f"{kept}\n"
            f"[compacted to ~{budget} of {seg.original_tokens} tokens (age rank {rank}) — "
            f"recall(segment_id=\"{seg.id}\") restores it exactly]"
        )
        gained = seg.rewrite(body, S_MASKED, f"masked (age rank {rank})")
        if gained > 0:
            freed += gained
            n_masked += 1

    return {"freed": freed, "masked": n_masked, "offloaded": n_offload}
