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
    """Settings a new chat starts from: the default profile when one is chosen
    in Settings (profiles.py), else the saved template / built-in defaults."""
    from . import profiles
    pid = profiles.default_profile_id()
    if pid:
        return profiles.config_for_chat(pid).to_dict()
    return SessionConfig.from_dict(store.get_setting(KEY) or {}).to_dict()


def keyed_route(cfg):
    """A Claude model on the route the user has a key for: with only an Anthropic key the
    chat uses `claude-opus-5-5` instead of `anthropic/claude-opus-5.5` (and the other way
    round); the summariser likewise. Other models and configured routes stay as they are."""
    from .config import key_problem, provider_counterpart
    for attr, owner in (("model", cfg), ("summary_model", cfg.compaction)):
        current = getattr(owner, attr)
        other = provider_counterpart(current) if current else None
        if other and key_problem(current) and not key_problem(other):
            setattr(owner, attr, other)
    return cfg


def create_config(overrides=None):
    from . import profiles
    cfg = SessionConfig.from_dict(merged(get(), overrides))
    if not cfg.workdirs and 'workdirs' not in (overrides or {}):
        cfg.workdirs = profiles.default_workdirs()
    return keyed_route(cfg)


def save(config):
    """Explicit "use these settings as new-chat defaults": the most recent
    explicit choice wins, so it also clears a default profile."""
    from . import profiles
    template = SessionConfig.from_dict(config).to_dict()
    store.set_setting(KEY, template)
    if profiles.load_prefs().get('default_profile_id'):
        profiles.update_prefs({'default_profile_id': ''})
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
