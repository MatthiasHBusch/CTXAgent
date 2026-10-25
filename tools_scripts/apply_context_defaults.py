"""Apply the 2026-10-15 context-control settings to saved chats and new-chat defaults.

Run with the server STOPPED (it keeps sessions in memory and would overwrite
the rows on its next persist):

    .venv\\Scripts\\python.exe tools_scripts\\apply_context_defaults.py            # defaults + every chat
    .venv\\Scripts\\python.exe tools_scripts\\apply_context_defaults.py <chat_id>  # defaults + one chat

What it changes (only these keys, everything else is kept):
  hot_zone_tokens 10000 -> 40000      the mutable working tail; 10k was ~5 tool batches,
                                      so fresh results froze/masked within minutes
  protected_recent_observations -> 12 newest results never touched
  protected_recent_batches -> 3       every result of the last 3 tool batches untouched
  mask_tiers -> [2000,1500,1000,600]  stubs keep real content as they age
  mask_floor_tokens -> 300            (was 60: a stub then held one header line)
  rewrite_policy -> 'agent'           soft due cycles only hint; the agent curates with
                                      curate_context (auto masking stays the safety net at
                                      deep pressure or after 30 batches without a review)
  enabled_tools += pin, job_wait, compact_context, curate_context
  max_output_tokens -> at least 32000 (8000 caused the silent stop)
"""
import json
import sqlite3
import sys
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / '.ctxagent' / 'ctxagent.db'
COMPACTION = {
    'hot_zone_tokens': 40000,
    'protected_recent_observations': 12,
    'protected_recent_batches': 3,
    'mask_tiers': [2000, 1500, 1000, 600],
    'mask_floor_tokens': 300,
    'rewrite_policy': 'agent',
    'agent_review_overdue_batches': 30,
}
NEW_TOOLS = ['pin', 'job_wait', 'compact_context', 'curate_context']


def patch(cfg: dict) -> dict:
    # 8000 output tokens is too small for a high-effort reasoning model: a long
    # think consumed the whole budget and the turn ended with no text and no tool
    # call (the 2026-10-15 "silent stop"). Only raised, never lowered.
    if int(cfg.get('max_output_tokens', 0) or 0) < 32000:
        cfg['max_output_tokens'] = 32000
    comp = cfg.setdefault('compaction', {})
    for k, v in COMPACTION.items():
        if k == 'hot_zone_tokens':
            comp[k] = max(int(comp.get(k, 0) or 0), v)
        else:
            comp[k] = v
    tools = cfg.get('enabled_tools')
    if isinstance(tools, list):
        for t in NEW_TOOLS:
            if t not in tools:
                tools.append(t)
    return cfg


def main() -> None:
    con = sqlite3.connect(DB)
    only = sys.argv[1] if len(sys.argv) > 1 else None
    rows = con.execute('select id, config from chats' + (' where id=?' if only else ''),
                       ((only,) if only else ())).fetchall()
    for cid, cfg in rows:
        new = patch(json.loads(cfg or '{}'))
        con.execute('update chats set config=? where id=?', (json.dumps(new, ensure_ascii=False), cid))
    row = con.execute("select value from settings where key='new_chat_defaults'").fetchone()
    if row:
        d = patch(json.loads(row[0]))
        con.execute("update settings set value=? where key='new_chat_defaults'", (json.dumps(d, ensure_ascii=False),))
    con.commit()
    print(f'updated {len(rows)} chat(s)' + (' and new-chat defaults' if row else ''))


if __name__ == '__main__':
    main()
