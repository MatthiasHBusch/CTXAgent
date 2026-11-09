"""Insights events reconstructed from chats recorded before (or without) insights.py.

A chat database keeps the history (tool calls with their arguments and results, user
messages, answers, with timestamps; deep-compacted segments in the archive blobs), the
compaction log, and DATA_DIR/usage/<chat>.jsonl keeps every request. That is enough to
rebuild most of what the recorder writes: requests, tools, context events and turns.
What it cannot rebuild: tool durations, dry runs and review hints (not logged), and the
error flag of a result (its class is read from the result text instead).

    python -m ctxagent.insights_report --from-db PATH/ctxagent.db --html old.html

Read-only: the database is copied to a temp folder before it is opened (a live app may
hold it in WAL mode), and nothing is written next to it.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from . import insights

_CONTEXT_TOOLS = ("compact_context", "curate_context")


def _copy(db: Path, tmp: Path) -> Path:
    dest = tmp / "chats.db"
    shutil.copyfile(db, dest)
    for ext in ("-wal", "-shm"):
        side = Path(str(db) + ext)
        if side.exists():
            shutil.copyfile(side, Path(str(dest) + ext))
    return dest


def _blob(data_dir: Path, handle: str) -> dict | None:
    try:
        p = data_dir / "blobs" / f"{handle}.txt"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except (OSError, ValueError):
        return None


def _history(conv: dict, data_dir: Path) -> list[dict]:
    archived = [d for d in (_blob(data_dir, h) for h in (conv.get("archive") or {}).values()) if d]
    seen, out = set(), []
    for d in sorted(archived, key=lambda d: d.get("seq", 0)) + (conv.get("segments") or []):
        key = (d.get("id"), d.get("seq"))
        if key not in seen:
            seen.add(key)
            out.append(d)
    return sorted(out, key=lambda d: d.get("seq", 0))


def _body(seg: dict) -> str:
    return seg.get("original_body") if seg.get("original_body") is not None else (seg.get("body") or "")


def _ledger(data_dir: Path, cid: str) -> list[dict]:
    path = data_dir / "usage" / f"{cid}.jsonl"
    out = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if entry.get("source", "agent") == "agent":
                out.append(entry)
    return out


def _is_task(seg: dict) -> bool:
    return seg.get("kind") == "user" and not _body(seg).startswith("[harness note]")


def chat_events(cid: str, cfg: dict, conv: dict, usage: dict, log: list, data_dir: Path) -> list[dict]:
    model = cfg.get("model") or "?"
    window = cfg.get("context_window")
    comp = cfg.get("compaction") or {}
    steered = bool(comp.get("enabled", True)) and comp.get("rewrite_policy") == "agent"
    history = _history(conv, data_dir)
    events: list[dict] = []

    def ev(kind: str, ts: float, **fields: Any) -> None:
        events.append({"v": insights.SCHEMA, "ts": round(float(ts or 0), 3), "chat": cid, "kind": kind,
                       "imported": True, **fields})

    # Turns: every message the user typed.
    starts = [s for s in history if _is_task(s)]
    for n, seg in enumerate(starts, start=1):
        ev("turn_start", seg.get("created_at"), turn=n, model=model, window=window, self_steered=steered,
           user=insights.excerpt(_body(seg)), user_chars=len(_body(seg)))
    start_ts = [s.get("created_at") or 0 for s in starts]

    def turn_at(ts: float) -> int:
        return sum(1 for t in start_ts if t <= ts) or 1

    # Requests from the usage ledger (or the last 100 the chat row keeps).
    entries = _ledger(data_dir, cid) or [r for r in (usage or {}).get("recent_requests") or []
                                         if r.get("source", "agent") == "agent"]
    for entry in entries:
        u = entry.get("usage") or {}
        ts = entry.get("timestamp") or 0
        ev("request", ts, turn=turn_at(ts), model=entry.get("model") or model, provider=entry.get("provider"),
           prompt_tokens=u.get("prompt_tokens"), uncached=u.get("input"), cache_read=u.get("cache_read"),
           cache_write=u.get("cache_write"), output=u.get("completion_tokens"), reasoning=u.get("reasoning"),
           cost=u.get("cost"), window=window, tool_calls=entry.get("tool_calls"))

    # Tool calls with their results.
    results = {s.get("tool_call_id"): s for s in history if s.get("kind") == "tool_result" and s.get("tool_call_id")}
    batches_at: list[float] = []
    for seg in history:
        if seg.get("kind") != "tool_call":
            continue
        batches_at.append(seg.get("created_at") or 0)
        for tc in seg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            name = fn.get("name") or "?"
            raw = fn.get("arguments")
            try:
                args = json.loads(raw) if isinstance(raw, str) and raw.strip() else (raw if isinstance(raw, dict) else {})
            except ValueError:
                args = {"_unparsable_arguments": {"chars": len(raw or "")}}
            res = results.get(tc.get("id"))
            body = _body(res) if res else ""
            cls = insights.classify_result(body) if res else "result.no_result"
            fields: dict[str, Any] = {}
            if name == "run_command":
                fields["exit"] = insights.exit_code(body)
                fields["program"] = insights.command_program((args or {}).get("command") if isinstance(args, dict) else "")
                if fields["exit"] not in (None, 0):
                    fields["tail"] = insights.tail(body)
            if insights.category(name) == "search" and res:
                fields["empty"] = body.rstrip().endswith("(no matches)")
            if name == "write_file" and res and not cls:
                fields["overwrote"] = body.startswith("Overwrote")
            ts = (res or seg).get("created_at") or 0
            ev("tool", ts, turn=turn_at(ts), name=name, category=insights.category(name),
               args=insights.summarize_args(name, args if isinstance(args, dict) else {}),
               ok=not (cls or "").startswith("result."), error_class=cls,
               error=insights.excerpt(body, 300) if cls else None, ms=None,
               tokens=(res or {}).get("original_tokens") or len(body) // 4, chars=len(body),
               mutates=bool((res or {}).get("mutates")), segment=(res or {}).get("id"),
               repeated=insights.REPEAT_NOTE in body, **fields)

    # Context events from the compaction log.
    summaries = {s.get("id"): _body(s) for s in history if s.get("kind") == "summary"}
    last_agent_ts = 0.0
    for entry in log or []:
        ts = entry.get("ts") or 0
        tiers = [t for t in entry.get("tiers") or [] if isinstance(t, dict)]
        names = [t.get("tier") for t in tiers]
        source = ("compact_context" if "manual" in names else "curate_context" if "agent-review" in names
                  else "engine")
        first = tiers[0] if tiers else {}
        before_ts = [s for s in history if (s.get("created_at") or 0) <= ts]
        task = next((_body(s) for s in reversed(before_ts) if _is_task(s)), "")
        answer = next((_body(s) for s in reversed(before_ts) if s.get("kind") == "assistant_text" and _body(s).strip()), "")
        call = next((s for s in reversed(before_ts) if s.get("kind") == "tool_call" and any(
            (c.get("function") or {}).get("name") == source for c in s.get("tool_calls") or [])), None)
        fields: dict[str, Any] = {
            "turn": turn_at(ts), "source": source, "dry_run": False, "held": False, "applied": True,
            "before": entry.get("before"), "after": entry.get("after"), "freed": entry.get("freed"),
            "rewrite_tokens": first.get("cache_rewrite_tokens"), "breakeven_requests": first.get("cache_breakeven_requests"),
            "window": window, "tiers": names, "deep": entry.get("deep"),
            "since": {"batches": sum(1 for b in batches_at if last_agent_ts < b <= ts)},
            "task": insights.excerpt(task), "last_answer": insights.excerpt(answer, 400),
            "call_text": insights.excerpt(_body(call), 400) if call and source in _CONTEXT_TOOLS else None,
        }
        if first.get("dropped"):
            fields["dropped"] = [str(d)[:200] for d in first["dropped"][:40]]
        if first.get("summary_id") in summaries:
            fields["summary"] = insights.excerpt(summaries[first["summary_id"]], 1200)
        ev("context", ts, **fields)
        if source in _CONTEXT_TOOLS:
            last_agent_ts = ts

    # Turn ends: the last answer before the next message the user typed.
    for n, seg in enumerate(starts, start=1):
        lo = seg.get("created_at") or 0
        hi = start_ts[n] if n < len(start_ts) else float("inf")
        inside = [s for s in history if lo <= (s.get("created_at") or 0) < hi]
        answer = next((_body(s) for s in reversed(inside) if s.get("kind") == "assistant_text" and _body(s).strip()), "")
        reqs = [e for e in events if e["kind"] == "request" and lo <= e["ts"] < hi]
        tools = [e for e in events if e["kind"] == "tool" and lo <= e["ts"] < hi]
        end = max([s.get("created_at") or lo for s in inside] or [lo])
        ev("turn_end", end, turn=n, outcome="final" if answer else "unknown", answer=insights.excerpt(answer),
           answer_chars=len(answer), tools=len(tools), tool_errors=sum(1 for t in tools if not t["ok"]),
           requests=len(reqs), cost=sum((e.get("cost") or 0) for e in reqs),
           prompt_tokens=sum((e.get("prompt_tokens") or 0) for e in reqs),
           completion_tokens=sum((e.get("output") or 0) for e in reqs))
    events.sort(key=lambda e: e["ts"])
    return events


def events_from_db(db: Path, data_dir: Path | None = None) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """(events per chat, chat metadata) for every chat in a CTXAgent database."""
    db = Path(db)
    data_dir = Path(data_dir or db.parent)
    out: dict[str, list[dict]] = {}
    meta: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="insights-import-") as tmp:
        local = _copy(db, Path(tmp))
        con = sqlite3.connect(f"file:{local.as_posix()}?mode=ro", uri=True)
        try:
            cols = {r[1] for r in con.execute("pragma table_info(chats)")}
            log_col = "compaction_log" if "compaction_log" in cols else "NULL"
            parent = {}
            if con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='agent_tasks'").fetchone():
                parent = dict(con.execute("select child, parent from agent_tasks").fetchall())
            rows = con.execute(f"select id, title, config, conversation, usage, {log_col} from chats").fetchall()
        finally:
            con.close()
    for cid, title, cfg_raw, conv_raw, usage_raw, log_raw in rows:
        try:
            cfg = json.loads(cfg_raw or "{}")
            conv = json.loads(conv_raw or "{}")
            usage = json.loads(usage_raw or "{}")
            log = json.loads(log_raw or "[]") if log_raw else []
        except ValueError:
            continue
        events = chat_events(cid, cfg, conv, usage, log, data_dir)
        if events:
            out[cid] = events
            meta[cid] = {"id": cid, "title": title or cid, "model": cfg.get("model"), "agent_parent": parent.get(cid)}
    return out, meta
