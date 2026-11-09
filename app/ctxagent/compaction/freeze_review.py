"""Plan a once-only review of a retiring prefix, never an unseen tool batch.

A token hot budget is a target, not permission to freeze fresh evidence.
Every boundary is snapped backwards to a whole tool group. Frozen bytes never
move backwards, including across restarts or a change from the legacy policy.
"""
from ..segments import K_TOOL_CALL, K_TOOL_RESULT, K_MEDIA
from .deep_plan import groups


def recent_batch_ids(conv, n_batches: int) -> set[str]:
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


def plan(conv, cfg):
    segs = conv.segments
    start = conv.frozen_upto
    observations = [s for s in segs if s.kind == K_TOOL_RESULT]
    count = max(0, cfg.protected_recent_observations)
    protected = {s.id for s in observations[-count:]} if count else set()
    # Every result of the newest N tool batches (a batch may hold many results).
    protected |= {sid for sid in recent_batch_ids(conv, getattr(cfg, 'protected_recent_batches', 0))
                  if (seg := conv.by_id(sid)) is not None and seg.kind == K_TOOL_RESULT}
    # Pinned results stay mutable (unfrozen) so their pin can expire / be lifted.
    protected |= {s.id for s in segs[start:] if s.pinned and s.kind == K_TOOL_RESULT
                  and s.pin_scope == 'manual' and s.pin_until_batch is not None}
    # Even a batch with 100 results must reach the model before retirement.
    if segs and segs[-1].kind in (K_TOOL_CALL, K_TOOL_RESULT, K_MEDIA):
        last_call = next((i for i in range(len(segs) - 1, -1, -1)
                          if segs[i].kind == K_TOOL_CALL), None)
        if last_call is not None:
            protected.update(s.id for s in segs[last_call:])
        else:
            protected.add(segs[-1].id)
    # Already-seen images may freeze verbatim; otherwise two old images could pin
    # hundreds of thousands of later tokens indefinitely. Fresh/unseen media is
    # protected above with its complete tool batch.
    protected = {sid for sid in protected if not conv.by_id(sid).frozen}
    end = conv.compute_hot_start(cfg.hot_zone_tokens)
    # Snap backwards to the beginning of an atomic call/result/media group.
    offset = 0
    for group in groups(segs):
        group_end = offset + len(group)
        if offset < end < group_end or any(s.id in protected for s in group):
            end = min(end, offset)
        offset = group_end
    end = max(start, end)  # legacy frozen boundaries are not secretly thawed
    return {'start': start, 'end': end,
            'eligible_ids': {s.id for s in segs[start:end]},
            'protected_ids': protected}
