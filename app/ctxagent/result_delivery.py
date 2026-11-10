"""Bound the first prompt delivery, not the underlying tool result.

Exact bytes are archived before any excerpt is substituted. Errors, recalls,
pins and native visual evidence are exempt. No model call or invented summary.
This is deliberately explicit: head/tail excerpts can omit important evidence.
"""
from . import tokens
from .blobs import put_blob
from .segments import K_TOOL_RESULT, S_OFFLOADED


def prepare(seg, cfg, session_id, *, error=False):
    cap = cfg.initial_result_max_tokens
    if (not cfg.enabled or not cap or seg.kind != K_TOOL_RESULT
            or seg.pinned or seg.tool_name == 'recall' or seg.images or error
            or seg.tokens() <= cap):
        return 0
    full = seg.original_body if seg.original_body is not None else seg.body
    # A failed disk write raises BEFORE substitution; the caller keeps full evidence.
    blob = put_blob(session_id, seg.id, full)
    header = (f'[INCOMPLETE head/tail excerpt of a tool result of ~{seg.original_tokens} tok; the middle '
              f'is omitted. recall(segment_id="{seg.id}") returns all of it]\n')
    kept = tokens.truncate_to_tokens(full, max(40, cap - tokens.count(header) - 16))
    gained = seg.rewrite(header + kept, S_OFFLOADED, 'bounded initial delivery; exact original archived')
    seg.blob_id = blob
    seg.observation = {**seg.observation, 'complete': False, 'delivery_limited': True}
    return gained
