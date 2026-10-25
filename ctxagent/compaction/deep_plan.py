"""Token-budgeted deep compaction. Never split tool groups or silently drop pins."""
from ..segments import K_SYSTEM, K_USER, K_SUMMARY, K_TOOL_CALL, K_TOOL_RESULT, K_MEDIA


def groups(segments):
    out = []
    for seg in segments:
        if out and seg.kind in (K_TOOL_RESULT, K_MEDIA) and out[-1][0].kind == K_TOOL_CALL:
            out[-1].append(seg)
        else:
            out.append([seg])
    return out


def plan(conv, cfg, tool_overhead=0):
    """Budget TOTAL resulting prompt including schema, system, pins and summary.

    The newest group is kept (its output may be unseen). An oversized mandatory
    group raises the attainable floor, explicitly reported instead of hiding it.
    """
    all_groups = groups(conv.segments)
    before = conv.total_tokens() + tool_overhead
    target = max(1, int(before * cfg.deep_target_ratio))
    users = [s.id for s in conv if s.kind == K_USER]
    recent = set(users[-cfg.protected_recent_user_turns:]) if cfg.protected_recent_user_turns > 0 else set()
    mandatory = set()
    for i, group in enumerate(all_groups):
        if i == len(all_groups) - 1 or any(
            s.kind == K_SYSTEM or s.id in recent or
            (s.pinned and s.kind != K_SUMMARY and s.pin_scope != 'recall') for s in group
        ):
            mandatory.add(i)
    fixed = tool_overhead + 3 + sum(s.tokens() for i in mandatory for s in all_groups[i])
    # Reserve wrapper/message overhead as well as readable summary content.
    wrapper_reserve = 160
    summary_floor = max(128, min(cfg.summary_min_tokens, cfg.summary_max_tokens))
    summary_budget = max(summary_floor, min(cfg.summary_max_tokens, (target - fixed) // 2 - wrapper_reserve))
    tail_budget = max(0, target - fixed - summary_budget - wrapper_reserve)
    keep = set(mandatory)
    for i in reversed(range(len(all_groups))):
        if i in mandatory:
            continue
        group = all_groups[i]
        if any(s.kind == K_SUMMARY for s in group):
            break
        size = sum(s.tokens() for s in group)
        if size > tail_budget:
            break
        keep.add(i)
        tail_budget -= size
    retained = [s for i, group in enumerate(all_groups) if i in keep for s in group]
    victims = [s for i, group in enumerate(all_groups) if i not in keep for s in group]
    return dict(before=before, target_tokens=target, fixed_tokens=fixed,
                summary_budget=summary_budget, retained=retained, victims=victims,
                minimum_tokens=fixed + wrapper_reserve + summary_budget,
                target_ratio=cfg.deep_target_ratio)
