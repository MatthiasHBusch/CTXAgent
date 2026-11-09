"""The shared scratchpad: one short text per chat that the user and the agent both edit.

Stored apart from the conversation (table `scratchpads`), so typing in it never changes
the prompt or the cache. The agent sees it when the user sends it (Ctrl+Enter in the UI:
`message()`, delivered at once or with the running turn's next step) or when it calls
scratchpad_read; it writes with scratchpad_write.

Concurrent edits: the UI saves the user's whole text together with the version it was
based on. If the agent wrote in between, its operations since that version are applied
again to the user's text; an operation that no longer applies (its old text is gone) is
dropped and reported, the user's text wins.
"""
from __future__ import annotations

import json
import time

from . import store

MAX_CHARS = 20_000
#: Agent operations kept for rebasing a user save that started from an older version.
OPS_KEPT = 30

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scratchpads (
    chat_id    TEXT PRIMARY KEY,
    text       TEXT NOT NULL,
    version    INTEGER NOT NULL,
    updated_at REAL NOT NULL,
    updated_by TEXT NOT NULL,
    ops        TEXT NOT NULL DEFAULT '[]'
);
"""


class ScratchpadError(ValueError):
    pass


def _row(c, chat_id):
    c.executescript(_SCHEMA)
    r = c.execute("SELECT * FROM scratchpads WHERE chat_id=?", (chat_id,)).fetchone()
    if r is None:
        return {"text": "", "version": 0, "updated_at": None, "updated_by": None, "ops": []}
    return {"text": r["text"], "version": r["version"], "updated_at": r["updated_at"],
            "updated_by": r["updated_by"], "ops": json.loads(r["ops"] or "[]")}


def _store(c, chat_id, state):
    c.execute("INSERT INTO scratchpads (chat_id,text,version,updated_at,updated_by,ops) VALUES (?,?,?,?,?,?) "
              "ON CONFLICT(chat_id) DO UPDATE SET text=excluded.text, version=excluded.version, "
              "updated_at=excluded.updated_at, updated_by=excluded.updated_by, ops=excluded.ops",
              (chat_id, state["text"], state["version"], state["updated_at"], state["updated_by"],
               json.dumps(state["ops"][-OPS_KEPT:], ensure_ascii=False)))


def public(state: dict) -> dict:
    return {k: state[k] for k in ("text", "version", "updated_at", "updated_by")}


def get(chat_id: str) -> dict:
    with store._conn() as c:
        return public(_row(c, chat_id))


def _check(text: str) -> str:
    if not isinstance(text, str):
        raise ScratchpadError("text must be a string")
    if len(text) > MAX_CHARS:
        raise ScratchpadError(f"The scratchpad holds at most {MAX_CHARS} characters ({len(text)} given)")
    return text


def _apply(text: str, op: dict) -> str:
    """One agent operation on `text`; ScratchpadError when it no longer applies."""
    kind = op["kind"]
    if kind == "replace":
        return op["text"]
    if kind == "append":
        return text + ("\n" if text and not text.endswith("\n") else "") + op["text"]
    for old, new in op["edits"]:
        n = text.count(old)
        if n != 1:
            raise ScratchpadError(f"old_string {old[:60]!r} occurs {n} times; it must occur exactly once")
        text = text.replace(old, new, 1)
    return text


def save_user(chat_id: str, text: str, base_version: int) -> dict:
    """The user's whole text, edited from `base_version`."""
    text = _check(text)
    with store._conn() as c:
        state = _row(c, chat_id)
        dropped = 0
        if base_version != state["version"]:
            for op in state["ops"]:
                if op["version"] <= base_version or op["by"] != "agent":
                    continue
                if op["kind"] == "replace":  # a full rewrite cannot be merged into the user's text
                    dropped += 1
                    continue
                try:
                    text = _check(_apply(text, op))
                except ScratchpadError:
                    dropped += 1
        if text == state["text"]:
            return {**public(state), "dropped_agent_edits": dropped}
        state.update(text=text, version=state["version"] + 1, updated_at=time.time(), updated_by="user")
        state["ops"].append({"version": state["version"], "by": "user", "kind": "replace", "text": ""})
        _store(c, chat_id, state)
    return {**public(state), "dropped_agent_edits": dropped}


def agent_write(chat_id: str, *, text: str | None = None, append: str | None = None,
                edits: list | None = None) -> dict:
    given = [x is not None for x in (text, append, edits)]
    if sum(given) != 1:
        raise ScratchpadError("Pass exactly one of text (replace all), append, or edits")
    if text is not None:
        op = {"kind": "replace", "text": _check(text)}
    elif append is not None:
        if not append.strip():
            raise ScratchpadError("append is empty")
        op = {"kind": "append", "text": append}
    else:
        pairs = []
        for e in edits or []:
            old, new = (e.get("old_string"), e.get("new_string")) if isinstance(e, dict) else (None, None)
            if not isinstance(old, str) or not old or not isinstance(new, str):
                raise ScratchpadError("each edit needs a non-empty old_string and a new_string")
            pairs.append((old, new))
        if not pairs:
            raise ScratchpadError("edits is empty")
        op = {"kind": "edits", "edits": pairs}
    with store._conn() as c:
        state = _row(c, chat_id)
        new_text = _check(_apply(state["text"], op))
        state.update(text=new_text, version=state["version"] + 1, updated_at=time.time(), updated_by="agent")
        state["ops"].append({"version": state["version"], "by": "agent", **op})
        _store(c, chat_id, state)
    return public(state)


def delete(chat_id: str) -> None:
    with store._conn() as c:
        if c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='scratchpads'").fetchone():
            c.execute("DELETE FROM scratchpads WHERE chat_id=?", (chat_id,))


PREFIX = "[scratchpad]"


def message(chat_id: str) -> str:
    """The user message that hands the scratchpad to the agent (Ctrl+Enter)."""
    state = get(chat_id)
    body = state["text"].strip() or "(the scratchpad is empty)"
    return (f"{PREFIX} The user asks you to look at the shared scratchpad (version {state['version']}). "
            "Its current text, written by the user and you together:\n"
            f"<scratchpad>\n{body}\n</scratchpad>")
