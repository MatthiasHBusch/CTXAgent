"""Explicit saved new-chat template, independent of whichever chat is open."""
from copy import deepcopy
from . import store
from .config import SessionConfig

KEY = 'new_chat_defaults'
PENDING = 'pending_chat_config'


def merged(base, patch):
    result = deepcopy(base)
    for key, value in (patch or {}).items():
        if key == 'compaction' and isinstance(value, dict):
            result.setdefault(key, {}).update(deepcopy(value))
        else:
            result[key] = deepcopy(value)
    return result


def get():
    return SessionConfig.from_dict(store.get_setting(KEY) or {}).to_dict()


def create_config(overrides=None):
    return SessionConfig.from_dict(merged(get(), overrides))


def save(config):
    template = SessionConfig.from_dict(config).to_dict()
    store.set_setting(KEY, template)
    return template


def apply_pending():
    """Apply explicitly queued configuration patches on startup, before sessions load.

    Transactional and idempotent. Does not touch conversation, usage, or timestamps.
    Used to avoid overwriting an active session's config from an external process.
    """
    import json
    with store._conn() as c:
        c.execute('BEGIN IMMEDIATE')
        row = c.execute('SELECT value FROM settings WHERE key=?', (PENDING,)).fetchone()
        if row is None:
            return
        for cid, patch in json.loads(row['value']).items():
            chat = c.execute('SELECT config FROM chats WHERE id=?', (cid,)).fetchone()
            if chat is not None:
                cfg = SessionConfig.from_dict(merged(json.loads(chat['config']), patch)).to_dict()
                c.execute('UPDATE chats SET config=? WHERE id=?', (json.dumps(cfg), cid))
        c.execute('DELETE FROM settings WHERE key=?', (PENDING,))
