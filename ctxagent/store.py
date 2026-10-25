"""SQLite persistence for chats.

The whole Conversation (segments + frozen boundary + compaction state) is stored
as JSON on the chat row. Sessions are small in count and large in content, so a
document-style row beats a normalised schema here — and it means a chat reloads
with its compaction state exactly as it was, including which segments are frozen.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from typing import Any

from .config import DB_PATH, ensure_dirs

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    config      TEXT NOT NULL,
    conversation TEXT NOT NULL,
    events      TEXT NOT NULL DEFAULT '[]',
    compaction_log TEXT NOT NULL DEFAULT '[]',
    usage       TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


from contextlib import contextmanager


@contextmanager
def _conn():
    ensure_dirs()
    c = sqlite3.connect(DB_PATH, timeout=30.0)
    try:
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        with c:
            yield c
    finally:
        # sqlite3.Connection.__exit__ commits/rolls back but does not close.
        c.close()


def init() -> None:
    with _conn() as c:
        c.executescript(_SCHEMA)


def create_chat(title: str, config: dict[str, Any]) -> str:
    cid = uuid.uuid4().hex[:16]
    now = time.time()
    with _conn() as c:
        c.execute(
            "INSERT INTO chats (id,title,created_at,updated_at,config,conversation,events,compaction_log,usage)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (cid, title, now, now, json.dumps(config), json.dumps({"segments": [], "frozen_upto": 0}),
             "[]", "[]", "{}"),
        )
    return cid


def list_chats() -> list[dict[str, Any]]:
    with _conn() as c:
        # Delegation is optional; read only persisted hierarchy, never worker status.
        hierarchy = "NULL AS agent_parent,NULL AS agent_root"
        join = ""
        if c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='agent_tasks'").fetchone():
            hierarchy = "agent_tasks.parent AS agent_parent,agent_tasks.root AS agent_root"
            join = " LEFT JOIN agent_tasks ON agent_tasks.child=chats.id"
        rows = c.execute(
            "SELECT chats.id,chats.title,chats.created_at,chats.updated_at,chats.config,chats.usage,"
            + hierarchy + " FROM chats" + join + " ORDER BY chats.updated_at DESC"
        ).fetchall()
    out = []
    for r in rows:
        cfg = json.loads(r["config"])
        out.append({
            "id": r["id"],
            "title": r["title"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "model": cfg.get("model"),
            "usage": json.loads(r["usage"] or "{}"),
            "agent_parent": r["agent_parent"],
            "agent_root": r["agent_root"],
        })
    return out


def get_chat(cid: str) -> dict[str, Any] | None:
    with _conn() as c:
        r = c.execute("SELECT * FROM chats WHERE id=?", (cid,)).fetchone()
    if not r:
        return None
    return {
        "id": r["id"],
        "title": r["title"],
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
        "config": json.loads(r["config"]),
        "conversation": json.loads(r["conversation"]),
        "events": json.loads(r["events"] or "[]"),
        "compaction_log": json.loads(r["compaction_log"] or "[]"),
        "usage": json.loads(r["usage"] or "{}"),
    }


def save_chat(
    cid: str,
    *,
    conversation: dict | None = None,
    config: dict | None = None,
    title: str | None = None,
    events: list | None = None,
    compaction_log: list | None = None,
    usage: dict | None = None,
) -> None:
    sets, vals = ["updated_at=?"], [time.time()]
    if conversation is not None:
        sets.append("conversation=?"); vals.append(json.dumps(conversation, ensure_ascii=False))
    if config is not None:
        sets.append("config=?"); vals.append(json.dumps(config, ensure_ascii=False))
    if title is not None:
        sets.append("title=?"); vals.append(title)
    if events is not None:
        sets.append("events=?"); vals.append(json.dumps(events[-400:], ensure_ascii=False))
    if compaction_log is not None:
        sets.append("compaction_log=?"); vals.append(json.dumps(compaction_log[-200:], ensure_ascii=False))
    if usage is not None:
        sets.append("usage=?"); vals.append(json.dumps(usage, ensure_ascii=False))
    vals.append(cid)
    with _conn() as c:
        c.execute(f"UPDATE chats SET {','.join(sets)} WHERE id=?", vals)


def delete_chat(cid: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM chats WHERE id=?", (cid,))
        if c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='browser_notes'").fetchone():
            c.execute("DELETE FROM browser_notes WHERE owner=?", (cid,))


def get_setting(key: str, default: Any = None) -> Any:
    with _conn() as c:
        r = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return json.loads(r["value"]) if r else default


def set_setting(key: str, value: Any) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO settings (key,value) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value, ensure_ascii=False)),
        )
