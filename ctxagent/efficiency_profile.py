"""Explicit opt-in migration, safe while a chat is running; applied at restart.

Default CLI invocation is read-only. --apply queues a patch in settings, never
updates the active chat row. --new-chat-defaults separately opts in the template.
Existing models, pricing preferences, prompts and other chats remain unchanged.
"""
import argparse
import json
from pathlib import Path
import sqlite3
import uuid

from . import defaults, store
from .config import DATA_DIR, DB_PATH, SessionConfig

PROFILE = {'enabled': True, 'document_compaction_enabled': True,
           'review_before_freeze': True, 'hot_zone_tokens': 10000,
           'rewrite_interval_batches': 10, 'rewrite_max_new_tokens': 20000,
           'initial_result_max_tokens': 8000, 'protected_recent_observations': 6,
           'dedup_enabled': True, 'masking_enabled': True, 'offload_enabled': True,
           'significance_enabled': True, 'deep_enabled': True}


def queue(chat_id, *, new_chat_defaults=False):
    patch = {'compaction': PROFILE}
    with store._conn() as c:
        c.execute('BEGIN IMMEDIATE')
        row = c.execute('SELECT config FROM chats WHERE id=?', (chat_id,)).fetchone()
        if row is None:
            raise ValueError('Chat not found')
        cfg = json.loads(row['config'])
        SessionConfig.from_dict(defaults.merged(cfg, patch))  # validate before writing
        settings = {r['key']: json.loads(r['value']) for r in c.execute(
            'SELECT key,value FROM settings WHERE key IN (?,?)', (defaults.KEY, defaults.PENDING))}
        backup = DATA_DIR / 'backups' / f'freeze-review-settings-{uuid.uuid4().hex}.json'
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_text(json.dumps({'chat_id': chat_id, 'config': cfg,
                                      'settings': settings}, indent=2), encoding='utf-8')
        pending = settings.get(defaults.PENDING, {})
        pending[chat_id] = defaults.merged(pending.get(chat_id, {}), patch)
        updates = {defaults.PENDING: pending}
        if new_chat_defaults:
            template = defaults.merged(settings.get(defaults.KEY, {}), patch)
            updates[defaults.KEY] = SessionConfig.from_dict(template).to_dict()
        for key, value in updates.items():
            c.execute('INSERT OR REPLACE INTO settings(key,value) VALUES (?,?)',
                      (key, json.dumps(value)))
    return {'queued_chat': chat_id, 'applies': 'next backend startup',
            'new_chat_defaults_updated': new_chat_defaults, 'profile': PROFILE,
            'settings_backup': str(backup)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chat-id', required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--new-chat-defaults', action='store_true')
    args = parser.parse_args()
    if args.apply:
        result = queue(args.chat_id, new_chat_defaults=args.new_chat_defaults)
    else:
        c = sqlite3.connect(DB_PATH.resolve().as_uri() + '?mode=ro', uri=True)
        try:
            row = c.execute('SELECT config FROM chats WHERE id=?', (args.chat_id,)).fetchone()
        finally:
            c.close()
        if row is None:
            parser.error('Chat not found')
        SessionConfig.from_dict(defaults.merged(json.loads(row[0]), {'compaction': PROFILE}))
        result = {'dry_run': True, 'chat_id': args.chat_id, 'profile': PROFILE,
                  'new_chat_defaults': args.new_chat_defaults}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
