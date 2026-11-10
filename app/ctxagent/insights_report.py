"""Report on recorded usage insights (insights.py).

    python -m ctxagent.insights_report                  # all chats -> DATA_DIR/insights/report.html
    python -m ctxagent.insights_report --md report.md   # plus a Markdown summary
    python -m ctxagent.insights_report --folder DIR --chat ID --json stats.json
    python -m ctxagent.insights_report --from-db PATH/ctxagent.db --model opus --html opus.html

What it answers:

* how the context window grew and shrank in each chat, and what the chat was doing
  (the user's task, the text that came with the call, the last answer) at each
  compact_context, curate_context, engine compaction and review hint;
* where the tokens and the money went: uncached input, cache reads and writes, output,
  reasoning, per model and per turn;
* how each tool fared: calls, failures by class, duration, result size, how often a
  result was dropped again by curate_context; with sections for the editing tools
  (failure classes, recovery, reading the file first, files touched) and the inspection
  tools (partial and repeated reads, empty searches), and per model the tools that fail
  most and the ones that work.

Read-only: the insights files are never changed.
"""
from __future__ import annotations

import argparse
import collections
import html
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from . import insights

MIN_CALLS = 5          # tools with fewer calls are not ranked
RECOVERY_LOOKAHEAD = 8  # tool calls after a failed edit in which a fix counts as recovery
HINT_FOLLOW = 5        # requests after a review hint in which a curate/compact counts as following it
CHART_CHATS = 24       # context charts drawn (most recent chats first)
READ_FIRST_LABELS = (("read", "after a read or search of the file earlier in the chat"),
                     ("own", "on a file the chat wrote or edited before, without reading it"),
                     ("neither", "with neither"))
AGENT_SOURCES = ("compact_context", "curate_context")


# --------------------------------------------------------------------------- loading
def load(folder: Path | None = None, chats: Iterable[str] | None = None) -> dict[str, list[dict]]:
    folder = Path(folder or insights.folder())
    wanted = set(chats or [])
    out: dict[str, list[dict]] = {}
    if not folder.is_dir():
        return out
    for path in sorted(folder.glob("*.jsonl")):
        if wanted and path.stem not in wanted:
            continue
        events = []
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if isinstance(ev, dict) and ev.get("kind"):
                    events.append(ev)
        if events:
            events.sort(key=lambda e: e.get("ts") or 0)
            out[path.stem] = events
    return out


def chat_titles() -> dict[str, dict]:
    try:
        from . import store
        return {c["id"]: c for c in store.list_chats()}
    except Exception:
        return {}


# --------------------------------------------------------------------------- helpers
def _num(v) -> float:
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else 0


def _median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def _p90(xs: list[float]) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    return ys[min(len(ys) - 1, int(round(0.9 * (len(ys) - 1))))]


def _rate(a: float, b: float) -> float | None:
    return a / b if b else None


def _path_key(v: Any) -> str | None:
    if not isinstance(v, str) or not v.strip():
        return None
    p = v.strip().replace("\\", "/").lower()
    while p.startswith("./"):
        p = p[2:]
    return p


def _arg_path(ev: dict) -> str | None:
    args = ev.get("args") or {}
    for key in ("path", "file", "file_path", "target"):
        k = _path_key(args.get(key))
        if k:
            return k
    return None


def _size(arg: Any) -> int:
    if isinstance(arg, dict):
        return int(arg.get("chars") or 0)
    return len(arg) if isinstance(arg, str) else 0


_DROP_ID = re.compile(r"^\s*([0-9a-f]{6,})\b")


def _failed(ev: dict) -> bool:
    cls = ev.get("error_class") or ""
    return not ev.get("ok", True) or cls.startswith("result.")


# --------------------------------------------------------------------------- analysis
def analyse(by_chat: dict[str, list[dict]], titles: dict[str, dict] | None = None) -> dict[str, Any]:
    titles = titles or {}
    tok_keys = ("uncached", "cache_read", "cache_write", "output", "reasoning")
    overview = collections.Counter()
    models: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    tools: dict[str, dict] = {}
    tools_by_model: dict[tuple[str, str], dict] = {}
    chats_out, context_events, turns = [], [], []
    edit_records, inspect_records, command_records = [], [], []
    dropped_ids: set[str] = set()
    hints = followed = 0
    first_ts, last_ts = None, None

    def tool_row(table, key):
        row = table.get(key)
        if row is None:
            row = table[key] = {"calls": 0, "failed": 0, "classes": collections.Counter(), "ms": [], "tokens": [],
                                "repeated": 0, "mutates": 0, "segments": [], "nonzero_exit": 0, "examples": {}}
        return row

    for cid, events in by_chat.items():
        meta = titles.get(cid) or {}
        model = meta.get("model") or ""
        reqs, cevents, chat_tools = [], [], []
        window = 0
        for ev in events:
            ts = ev.get("ts") or 0
            first_ts = ts if first_ts is None else min(first_ts, ts)
            last_ts = ts if last_ts is None else max(last_ts, ts)
            kind = ev.get("kind")
            if kind == "turn_start":
                model = ev.get("model") or model
                window = ev.get("window") or window
                overview["turns"] += 1
            elif kind == "request":
                model = ev.get("model") or model
                window = ev.get("window") or window
                tokens = ev.get("prompt_tokens") or ev.get("context") or 0
                reasoning = ev.get("reasoning")
                output = _num(ev.get("output")) - _num(reasoning)
                row = {"ts": ts, "turn": ev.get("turn"), "tokens": int(tokens), "cost": ev.get("cost"),
                       "cache_read": _num(ev.get("cache_read")), "cache_write": _num(ev.get("cache_write")),
                       "uncached": _num(ev.get("uncached")), "output": max(0, output), "reasoning": _num(reasoning),
                       "model": model, "finish": ev.get("finish"), "error": ev.get("error")}
                if ev.get("keepalive"):
                    # Prompt-cache keep-alives: real spend, but no model answer (not a request
                    # in the charts and per-turn figures).
                    overview["keepalives"] += 1
                    overview["keepalive_cost"] += _num(ev.get("cost"))
                    overview["cost"] += _num(ev.get("cost"))
                    models[model]["cost"] += _num(ev.get("cost"))
                    for k in tok_keys:
                        overview[k] += row[k]
                        models[model][k] += row[k]
                    continue
                reqs.append(row)
                overview["requests"] += 1
                overview["cost"] += _num(ev.get("cost"))
                models[model]["requests"] += 1
                models[model]["cost"] += _num(ev.get("cost"))
                for k in tok_keys:
                    overview[k] += row[k]
                    models[model][k] += row[k]
                if ev.get("error"):
                    overview["request_errors"] += 1
                if str(ev.get("finish") or "").lower() in ("length", "max_tokens", "max_output_tokens"):
                    overview["capped_responses"] += 1
            elif kind == "tool":
                name = ev.get("name") or "?"
                ev = {**ev, "_model": model, "_chat": cid, "_i": len(chat_tools)}
                chat_tools.append(ev)
                for row in (tool_row(tools, name), tool_row(tools_by_model, (model, name))):
                    row["calls"] += 1
                    if isinstance(ev.get("ms"), (int, float)):
                        row["ms"].append(ev["ms"])
                    row["tokens"].append(_num(ev.get("tokens")))
                    row["repeated"] += int(bool(ev.get("repeated")))
                    row["mutates"] += int(bool(ev.get("mutates")))
                    if ev.get("segment"):
                        row["segments"].append(ev["segment"])
                    cls = ev.get("error_class")
                    if cls == "info.command_nonzero_exit":
                        row["nonzero_exit"] += 1
                    if _failed(ev):
                        row["failed"] += 1
                        row["classes"][cls or "result.unclassified"] += 1
                        row["examples"].setdefault(cls or "result.unclassified", ev.get("error") or "")
                overview["tool_calls"] += 1
                overview["tool_failures"] += int(_failed(ev))
                models[model]["tool_calls"] += 1
                models[model]["tool_failures"] += int(_failed(ev))
                if name == "run_command":
                    command_records.append(ev)
                if ev.get("category") == "edit":
                    edit_records.append(ev)
                elif ev.get("category") in ("inspect", "search"):
                    inspect_records.append(ev)
            elif kind == "context":
                src = ev.get("source")
                pos = max(0, len(reqs) - 1)
                cevents.append({**ev, "_req": pos, "_chat": cid, "_model": model})
                if src == "curate_context" and ev.get("applied"):
                    for d in ev.get("dropped") or []:
                        m = _DROP_ID.match(str(d))
                        if m:
                            dropped_ids.add(m.group(1))
            elif kind == "turn_end":
                turns.append({**ev, "_chat": cid, "_model": model})

        # Did the agent act on a review hint within HINT_FOLLOW requests?
        for k, ev in enumerate(cevents):
            if ev.get("source") != "hint":
                continue
            hints += 1
            for later in cevents[k + 1:]:
                if later["_req"] - ev["_req"] > HINT_FOLLOW:
                    break
                if later.get("source") in AGENT_SOURCES and later.get("applied"):
                    followed += 1
                    break
        context_events.extend(cevents)
        peak = max((r["tokens"] for r in reqs), default=0)
        chats_out.append({
            "id": cid, "title": meta.get("title") or cid, "model": model, "window": window,
            "child": bool(meta.get("agent_parent")),
            "requests": reqs, "events": cevents, "turns": sum(1 for e in events if e.get("kind") == "turn_start"),
            "tool_calls": len(chat_tools), "peak": peak, "final": reqs[-1]["tokens"] if reqs else 0,
            "cost": sum(_num(r["cost"]) for r in reqs), "last_ts": events[-1].get("ts") or 0,
            "compactions": sum(1 for e in cevents if e.get("source") == "compact_context" and e.get("applied")),
            "curates": sum(1 for e in cevents if e.get("source") == "curate_context" and e.get("applied")),
            "engine": sum(1 for e in cevents if e.get("source") == "engine"),
        })

    def finish_row(name, row, model=None):
        out = {"tool": name, "category": insights.category(name), "calls": row["calls"], "failed": row["failed"],
               "failure_rate": _rate(row["failed"], row["calls"]),
               "classes": row["classes"].most_common(), "median_ms": _median(row["ms"]),
               "mean_tokens": (sum(row["tokens"]) / row["calls"]) if row["calls"] else None,
               "total_tokens": sum(row["tokens"]), "repeated": row["repeated"],
               "nonzero_exit": row["nonzero_exit"],
               "dropped_later": sum(1 for s in row["segments"] if s in dropped_ids),
               "examples": row["examples"]}
        if model is not None:
            out["model"] = model
        return out

    tool_rows = sorted((finish_row(n, r) for n, r in tools.items()), key=lambda r: -r["calls"])
    by_model: dict[str, list[dict]] = collections.defaultdict(list)
    for (model, name), row in tools_by_model.items():
        by_model[model].append(finish_row(name, row, model))
    difficult, reliable = {}, {}
    for model, rows in by_model.items():
        ranked = [r for r in rows if r["calls"] >= MIN_CALLS]
        difficult[model] = sorted([r for r in ranked if r["failed"]],
                                  key=lambda r: (-(r["failure_rate"] or 0), -r["calls"]))[:12]
        reliable[model] = sorted([r for r in ranked if (r["failure_rate"] or 0) <= 0.02],
                                 key=lambda r: -r["calls"])[:12]

    cat = collections.defaultdict(lambda: collections.Counter())
    for r in tool_rows:
        c = cat[r["category"]]
        c["calls"] += r["calls"]
        c["failed"] += r["failed"]
        c["tokens"] += r["total_tokens"]
        c["dropped_later"] += r["dropped_later"]

    stats = {
        "generated": time.time(), "span": [first_ts, last_ts],
        "overview": dict(overview), "models": {m: dict(c) for m, c in models.items()},
        "chats": sorted(chats_out, key=lambda c: -c["last_ts"]),
        "tools": tool_rows, "tools_by_model": {m: sorted(rs, key=lambda r: -r["calls"]) for m, rs in by_model.items()},
        "categories": {k: dict(v) for k, v in sorted(cat.items(), key=lambda kv: -kv[1]["calls"])},
        "difficult": difficult, "reliable": reliable,
        "edits": _edits(edit_records, by_chat),
        "inspection": _inspection(inspect_records, dropped_ids, by_chat),
        "commands": _commands(command_records),
        "context": _context_summary(context_events, hints, followed),
        "context_events": context_events,
        "turns": _turns(turns),
    }
    return stats


def _edits(records: list[dict], by_chat: dict[str, list[dict]]) -> dict[str, Any]:
    out: dict[str, Any] = {"calls": len(records)}
    if not records:
        return out
    per_tool = collections.defaultdict(collections.Counter)
    classes = collections.Counter()
    files = collections.Counter()
    sizes_old, sizes_new = [], []
    writes = collections.Counter()
    recovered = unrecovered = 0
    attempts_to_fix: list[int] = []
    stale = collections.Counter()
    read_first = {"read": collections.Counter(), "own": collections.Counter(), "neither": collections.Counter()}
    # Tool sequences per chat, to look back for a read and ahead for a fix.
    seq: dict[str, list[dict]] = collections.defaultdict(list)
    for cid, events in by_chat.items():
        seq[cid] = [e for e in events if e.get("kind") == "tool"]
    for ev in records:
        name = ev.get("name")
        failed = _failed(ev)
        per_tool[name]["calls"] += 1
        per_tool[name]["failed"] += int(failed)
        path = _arg_path(ev)
        if failed:
            classes[ev.get("error_class") or "result.unclassified"] += 1
        elif path:
            files[path] += 1
        if name == "write_file" and not failed and ev.get("overwrote") is not None:
            writes["overwrote" if ev["overwrote"] else "created"] += 1
        if name == "edit_file":
            args = ev.get("args") or {}
            sizes_old.append(_size(args.get("old_string")))
            sizes_new.append(_size(args.get("new_string")))
        chat_seq = seq.get(ev["_chat"], [])
        i = ev["_i"]
        if path and name in ("edit_file", "apply_edit_plan", "document_apply_edits"):
            earlier = [p for p in chat_seq[:i] if _arg_path(p) == path]
            if any(p.get("category") in ("inspect", "search") for p in earlier):
                bucket = read_first["read"]
            elif any(p.get("category") == "edit" and not _failed(p) for p in earlier):
                bucket = read_first["own"]
            else:
                bucket = read_first["neither"]
            bucket["calls"] += 1
            bucket["failed"] += int(failed)
        if failed and path and ev.get("error_class") in ("result.edit_not_found", "result.edit_ambiguous"):
            reads = [k for k, p in enumerate(chat_seq[:i]) if _arg_path(p) == path and p.get("category") in ("inspect", "search")]
            last = reads[-1] if reads else -1
            between = chat_seq[last + 1:i]
            if last < 0:
                stale["not read"] += 1
            elif any(_arg_path(p) == path and p.get("category") == "edit" and not _failed(p) for p in between):
                stale["edited since the last read"] += 1
            elif any(p.get("name") == "run_command" for p in between):
                stale["a command ran since the last read"] += 1
            else:
                stale["unchanged since the last read"] += 1
        if failed:
            fix = None
            for k, later in enumerate(chat_seq[i + 1:i + 1 + RECOVERY_LOOKAHEAD], start=1):
                if later.get("category") == "edit" and not _failed(later) and (_arg_path(later) == path or not path):
                    fix = k
                    break
            if fix is None:
                unrecovered += 1
            else:
                recovered += 1
                attempts_to_fix.append(fix)
    out.update({
        "per_tool": {k: dict(v) for k, v in sorted(per_tool.items(), key=lambda kv: -kv[1]["calls"])},
        "classes": classes.most_common(), "files": files.most_common(15),
        "edit_file_old_chars_median": _median(sizes_old), "edit_file_new_chars_median": _median(sizes_new),
        "edit_file_old_chars_p90": _p90(sizes_old),
        "recovered": recovered, "unrecovered": unrecovered, "calls_to_fix_median": _median(attempts_to_fix),
        "miss_context": stale.most_common(),
        "write_file": dict(writes),
        "read_first": {k: dict(v) for k, v in read_first.items()},
    })
    return out


def _inspection(records: list[dict], dropped_ids: set[str], by_chat: dict[str, list[dict]]) -> dict[str, Any]:
    out: dict[str, Any] = {"calls": len(records)}
    if not records:
        return out
    identical = after_edit = 0
    for events in by_chat.values():
        last_read: dict[tuple, int] = {}
        last_edit: dict[str, int] = {}
        for k, ev in enumerate(e for e in events if e.get("kind") == "tool"):
            path = _arg_path(ev)
            if not path:
                continue
            if ev.get("category") == "edit" and not _failed(ev):
                last_edit[path] = k
            elif ev.get("name") == "read_file" and not _failed(ev):
                key = (path, json.dumps(ev.get("args") or {}, sort_keys=True))
                if key in last_read and last_edit.get(path, -1) < last_read[key]:
                    identical += 1
                if path in last_edit:
                    after_edit += 1
                last_read[key] = k
    out["identical_rereads"] = identical
    out["reads_after_edit"] = after_edit
    reads = [r for r in records if r.get("name") == "read_file"]
    partial_keys = ("offset", "limit", "start_line", "end_line", "start", "end", "lines")
    per_path = collections.Counter(_arg_path(r) for r in reads if _arg_path(r))
    searches = [r for r in records if r.get("category") == "search" and r.get("ok")]
    per_tool = collections.defaultdict(lambda: {"calls": 0, "tokens": [], "dropped_later": 0, "failed": 0})
    for r in records:
        row = per_tool[r.get("name")]
        row["calls"] += 1
        row["tokens"].append(_num(r.get("tokens")))
        row["dropped_later"] += int(r.get("segment") in dropped_ids)
        row["failed"] += int(_failed(r))
    out.update({
        "read_file_calls": len(reads),
        "read_file_partial": sum(1 for r in reads if any(k in (r.get("args") or {}) for k in partial_keys)),
        "read_file_tokens_median": _median([_num(r.get("tokens")) for r in reads]),
        "read_file_tokens_p90": _p90([_num(r.get("tokens")) for r in reads]),
        "files_read": len(per_path),
        "repeat_reads": sum(n - 1 for n in per_path.values() if n > 1),
        "most_read": per_path.most_common(10),
        "searches": len(searches), "empty_searches": sum(1 for r in searches if r.get("empty")),
        "per_tool": {k: {"calls": v["calls"], "failed": v["failed"], "mean_tokens": sum(v["tokens"]) / v["calls"],
                         "dropped_later": v["dropped_later"]}
                     for k, v in sorted(per_tool.items(), key=lambda kv: -kv[1]["calls"])},
    })
    return out


#: (label, regex) over the end of a failing command's output; the first match wins.
SIGNATURES = [
    ("PowerShell: not a command", r"is not recognized as|wurde nicht als Name eines Cmdlet|CommandNotFoundException"),
    ("PowerShell: parser error", r"ParserError|Unexpected token|Unerwartetes Token|leeres Pipeelement|empty pipe element|"
                                 r"not a valid statement separator|kein gültiges Anweisungstrennzeichen|Missing (?:closing|expression)|"
                                 r"Fehlende (?:schließende|Abschlusszeichen)"),
    ("PowerShell: path not found", r"Cannot find path|ItemNotFoundException|kann nicht gefunden werden, da er nicht vorhanden ist"),
    ("bash: command not found", r"command not found"),
    ("Python: module not found", r"ModuleNotFoundError|ImportError: "),
    ("Python: syntax error", r"SyntaxError|IndentationError"),
    ("tests failed", r"\bFAILED\b|AssertionError|failures=\d|\b\d+ failed\b|Tests? failed"),
    ("Python: other exception", r"Traceback \(most recent call last\)|\b[A-Z][A-Za-z]+Error: "),
    ("Godot / GDScript error", r"SCRIPT ERROR|Parse Error:|Invalid call|res://"),
    ("compiler or build error", r"error CS\d+|error C\d+|Build FAILED|\berror: "),
    ("file not found", r"No such file or directory|cannot find the (?:file|path)|FileNotFoundError"),
    ("permission denied", r"Permission denied|Access is denied|Zugriff verweigert|UnauthorizedAccess"),
    ("git", r"\bfatal: "),
    ("PowerShell: other error", r"In Zeile:\d+ Zeichen:|At line:\d+ char:|CategoryInfo"),
]
#: Programs whose exit code 1 means "no match", not an error.
_SEARCHERS = ("rg", "grep", "findstr", "select-string")
_SIG_RX = [(label, re.compile(rx)) for label, rx in SIGNATURES]


def _commands(records: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {"calls": len(records)}
    if not records:
        return out
    heads = collections.defaultdict(collections.Counter)
    sigs = collections.Counter()
    examples: dict[str, str] = {}
    nonzero = timeouts = 0
    shell_reads, shell_read_tokens = 0, 0
    for ev in records:
        head = ev.get("program") or insights.command_program((ev.get("args") or {}).get("command"))
        heads[head]["calls"] += 1
        if head in insights.SHELL_INSPECTION:
            shell_reads += 1
            shell_read_tokens += _num(ev.get("tokens"))
        if ev.get("error_class") == "result.command_timeout":
            timeouts += 1
        if ev.get("exit") not in (None, 0):
            nonzero += 1
            heads[head]["nonzero"] += 1
            text = (ev.get("tail") or "") + " " + (ev.get("error") or "")
            label = next((lbl for lbl, rx in _SIG_RX if rx.search(text)), None)
            if label is None:
                label = "search found nothing (exit 1)" if head in _SEARCHERS and ev.get("exit") == 1 else "other"
            sigs[label] += 1
            examples.setdefault(label, ev.get("tail") or ev.get("error") or "")
    out.update({"nonzero": nonzero, "timeouts": timeouts, "shell_inspection": shell_reads,
                "shell_inspection_mean_tokens": shell_read_tokens / shell_reads if shell_reads else None,
                "heads": sorted(([h, c["calls"], c["nonzero"]] for h, c in heads.items()), key=lambda r: -r[1])[:15],
                "signatures": sigs.most_common(), "examples": examples})
    return out


def _context_summary(events: list[dict], hints: int, followed: int) -> dict[str, Any]:
    by = collections.defaultdict(lambda: {"applied": 0, "dry_run": 0, "held": 0, "freed": [], "before": [],
                                          "since_batches": [], "since_requests": [], "rewrite": []})
    for ev in events:
        row = by[ev.get("source") or "?"]
        if ev.get("dry_run"):
            row["dry_run"] += 1
        elif ev.get("held"):
            row["held"] += 1
        elif ev.get("applied"):
            row["applied"] += 1
            row["freed"].append(_num(ev.get("freed")))
            row["before"].append(_num(ev.get("before")))
            row["rewrite"].append(_num(ev.get("rewrite_tokens")))
            since = ev.get("since") or {}
            row["since_batches"].append(_num(since.get("batches")))
            row["since_requests"].append(_num(since.get("requests")))
    out = {}
    for src, row in by.items():
        out[src] = {"applied": row["applied"], "dry_run": row["dry_run"], "held": row["held"],
                    "freed_total": sum(row["freed"]), "freed_median": _median(row["freed"]),
                    "before_median": _median(row["before"]), "rewrite_median": _median(row["rewrite"]),
                    "since_batches_median": _median(row["since_batches"]),
                    "since_requests_median": _median(row["since_requests"])}
    return {"by_source": out, "hints": hints, "hints_followed": followed}


def _turns(turns: list[dict]) -> dict[str, Any]:
    if not turns:
        return {"count": 0}
    cost = [_num(t.get("cost")) for t in turns]
    reqs = [_num(t.get("requests")) for t in turns]
    tools = [_num(t.get("tools")) for t in turns]
    return {"count": len(turns), "outcomes": collections.Counter(t.get("outcome") for t in turns).most_common(),
            "cost_median": _median(cost), "cost_p90": _p90(cost), "requests_median": _median(reqs),
            "requests_p90": _p90(reqs), "tools_median": _median(tools), "tools_p90": _p90(tools)}


# --------------------------------------------------------------------------- formatting
def _k(n: float | None) -> str:
    if n is None:
        return "–"
    n = float(n)
    if abs(n) >= 1e6:
        return f"{n / 1e6:.2f}M"
    if abs(n) >= 1e4:
        return f"{n / 1e3:.0f}k"
    if abs(n) >= 1e3:
        return f"{n / 1e3:.1f}k"
    return f"{n:.0f}"


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.0f} %" if x >= 0.1 or x == 0 else f"{100 * x:.1f} %"


def _usd(x: float | None) -> str:
    return "–" if x is None else f"${x:,.2f}" if x >= 1 else f"${x:.3f}"


def _when(ts: float | None) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "–"


# --------------------------------------------------------------------------- Markdown
def render_markdown(stats: dict) -> str:
    o = stats["overview"]
    L: list[str] = []
    a = L.append
    a("# Usage insights")
    a("")
    a(f"{_when(stats['span'][0])} to {_when(stats['span'][1])}: {len(stats['chats'])} chats, {o.get('turns', 0)} turns, "
      f"{o.get('requests', 0)} requests, {o.get('tool_calls', 0)} tool calls, {_usd(o.get('cost'))}.")
    prompt = sum(o.get(k, 0) for k in ("uncached", "cache_read", "cache_write"))
    a("")
    a("## Tokens")
    a("")
    a("| model | requests | cost | uncached | cache read | cache write | output | reasoning | cache-read share | tool calls | tool failures |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for m, c in sorted(stats["models"].items(), key=lambda kv: -kv[1].get("cost", 0)):
        p = sum(c.get(k, 0) for k in ("uncached", "cache_read", "cache_write"))
        a(f"| {m or '?'} | {c.get('requests', 0)} | {_usd(c.get('cost'))} | {_k(c.get('uncached'))} | {_k(c.get('cache_read'))} | "
          f"{_k(c.get('cache_write'))} | {_k(c.get('output'))} | {_k(c.get('reasoning'))} | {_pct(_rate(c.get('cache_read', 0), p))} | "
          f"{c.get('tool_calls', 0)} | {_pct(_rate(c.get('tool_failures', 0), c.get('tool_calls', 0)))} |")
    a("")
    a(f"All models: cache-read share {_pct(_rate(o.get('cache_read', 0), prompt))}; "
      f"{o.get('capped_responses', 0)} responses hit the output cap; {o.get('request_errors', 0)} requests failed."
      + (f" {o['keepalives']} cache keep-alives during long tool runs ({_usd(o.get('keepalive_cost'))})."
         if o.get('keepalives') else ""))
    t = stats["turns"]
    if t.get("count"):
        a(f"Per turn: median {_usd(t['cost_median'])} (p90 {_usd(t['cost_p90'])}), {t['requests_median']:.0f} requests "
          f"(p90 {t['requests_p90']:.0f}), {t['tools_median']:.0f} tool calls (p90 {t['tools_p90']:.0f}).")
    a("")
    a("## Context")
    a("")
    a("| chat | model | window | turns | requests | peak | last | compact | curate | engine | cost |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in stats["chats"]:
        a(f"| {c['title'][:60]} | {c['model']} | {_k(c['window'])} | {c['turns']} | {len(c['requests'])} | {_k(c['peak'])} | "
          f"{_k(c['final'])} | {c['compactions']} | {c['curates']} | {c['engine']} | {_usd(c['cost'])} |")
    cs = stats["context"]
    a("")
    a("| source | applied | dry runs | held | freed total | freed median | context before (median) | rewrite (median) | tool batches since last (median) |")
    a("|---|---|---|---|---|---|---|---|---|")
    for src, r in cs["by_source"].items():
        a(f"| {src} | {r['applied']} | {r['dry_run']} | {r['held']} | {_k(r['freed_total'])} | {_k(r['freed_median'])} | "
          f"{_k(r['before_median'])} | {_k(r['rewrite_median'])} | {_k(r['since_batches_median'])} |")
    a("")
    a(f"Review hints: {cs['hints']}, followed by a compact or curate within {HINT_FOLLOW} requests: {cs['hints_followed']}.")
    a("")
    a("### Context events")
    a("")
    for ev in [e for e in stats["context_events"] if e.get("source") in AGENT_SOURCES][-60:]:
        state = "dry run" if ev.get("dry_run") else "held" if ev.get("held") else "applied"
        a(f"- {_when(ev.get('ts'))} · turn {ev.get('turn')} · **{ev.get('source')}** ({state}) · "
          f"{_k(ev.get('before'))} → {_k(ev.get('after'))}, freed {_k(ev.get('freed') or ev.get('freed_estimate'))}"
          f", {(ev.get('since') or {}).get('batches', '–')} tool batches since the last one")
        if ev.get("task"):
            a(f"  - task: {ev['task']}")
        if ev.get("call_text"):
            a(f"  - with the call: {ev['call_text']}")
        if ev.get("summary"):
            a(f"  - summary: {ev['summary'][:400]}")
        if ev.get("dropped"):
            a(f"  - dropped: {len(ev['dropped'])} results")
    a("")
    a("## Tools")
    a("")
    a("| tool | category | calls | failed | failure rate | main failure | median ms | mean result tok | dropped later | repeated failures |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for r in stats["tools"]:
        main = r["classes"][0][0].replace("result.", "") if r["classes"] else ""
        a(f"| {r['tool']} | {r['category']} | {r['calls']} | {r['failed']} | {_pct(r['failure_rate'])} | {main} | "
          f"{_k(r['median_ms'])} | {_k(r['mean_tokens'])} | {r['dropped_later']} | {r['repeated']} |")
    e = stats["edits"]
    a("")
    a("## Editing")
    a("")
    if e.get("calls"):
        a(f"{e['calls']} edit calls. Failed edits fixed within {RECOVERY_LOOKAHEAD} tool calls: {e['recovered']}; "
          f"not fixed: {e['unrecovered']} (median calls to the fix: {e['calls_to_fix_median'] or '–'}).")
        rf = e["read_first"]
        for k, label in READ_FIRST_LABELS:
            b = rf.get(k) or {}
            if b.get("calls"):
                a(f"- {label}: {b['calls']} edits, {_pct(_rate(b.get('failed', 0), b['calls']))} failed")
        a(f"- edit_file old_string: median {_k(e['edit_file_old_chars_median'])} chars (p90 {_k(e['edit_file_old_chars_p90'])}), "
          f"new_string median {_k(e['edit_file_new_chars_median'])} chars")
        a("- failure classes: " + ", ".join(f"{c.replace('result.', '')} {n}" for c, n in e["classes"]))
        if e.get("write_file"):
            a(f"- write_file: created {e['write_file'].get('created', 0)} files, overwrote "
              f"{e['write_file'].get('overwrote', 0)} existing ones (a whole-file rewrite instead of an edit)")
        if e.get("miss_context"):
            a("- old_string not found / not unique, by what happened to the file: "
              + ", ".join(f"{k} {n}" for k, n in e["miss_context"]))
        a("- files edited most: " + ", ".join(f"{p} ({n})" for p, n in e["files"][:10]))
    i = stats["inspection"]
    a("")
    a("## Inspection")
    a("")
    if i.get("calls"):
        a(f"read_file: {i['read_file_calls']} calls ({i['read_file_partial']} partial), median {_k(i['read_file_tokens_median'])} "
          f"tokens (p90 {_k(i['read_file_tokens_p90'])}), {i['files_read']} files, {i['repeat_reads']} repeat reads. "
          f"Searches: {i['searches']}, {i['empty_searches']} without a match. Identical re-reads (same file and "
          f"range, no edit in between): {i['identical_rereads']}; reads of a file after editing it: {i['reads_after_edit']}.")
        a("")
        a("| tool | calls | failed | mean result tok | dropped later |")
        a("|---|---|---|---|---|")
        for name, r in i["per_tool"].items():
            a(f"| {name} | {r['calls']} | {r['failed']} | {_k(r['mean_tokens'])} | {r['dropped_later']} |")
    c = stats.get("commands") or {}
    a("")
    a("## Shell commands")
    a("")
    if c.get("calls"):
        a(f"run_command: {c['calls']} calls, {c['nonzero']} with a non-zero exit ({_pct(_rate(c['nonzero'], c['calls']))}), "
          f"{c['timeouts']} timed out. {c['shell_inspection']} of them only read or searched files (Get-Content, cat, "
          f"Select-String, grep, ls, ...; mean {_k(c['shell_inspection_mean_tokens'])} result tokens), the job of "
          f"read_file, grep_search and list_dir.")
        a("")
        a("| command | calls | non-zero exit |")
        a("|---|---|---|")
        for h, n, z in c["heads"]:
            a(f"| {h} | {n} | {z} ({_pct(_rate(z, n))}) |")
        if c["signatures"]:
            a("")
            a("| why it failed (end of the output) | commands | example |")
            a("|---|---|---|")
            for label, n in c["signatures"]:
                a(f"| {label} | {n} | {insights.excerpt(c['examples'].get(label), 160)} |")
    a("")
    a("## Per model: tools that fail most and tools that work")
    for model in sorted(stats["difficult"]):
        a("")
        a(f"### {model or '?'}")
        a("")
        rows = stats["difficult"][model]
        if rows:
            a("| tool | calls | failure rate | classes | example |")
            a("|---|---|---|---|---|")
            for r in rows:
                cls = ", ".join(f"{c.replace('result.', '')} {n}" for c, n in r["classes"][:3])
                ex = (r["examples"].get(r["classes"][0][0]) if r["classes"] else "") or ""
                a(f"| {r['tool']} | {r['calls']} | {_pct(r['failure_rate'])} | {cls} | {insights.excerpt(ex, 140)} |")
        ok = stats["reliable"].get(model) or []
        if ok:
            a("")
            a(f"Works (≤ 2 % failures, ≥ {MIN_CALLS} calls): " + ", ".join(f"{r['tool']} ({r['calls']})" for r in ok))
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- HTML
_CSS = """
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--page:#f9f9f7;--text-primary:#0b0b0b;--text-secondary:#52514e;
--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--series-1:#2a78d6;--series-2:#eb6834;
--series-3:#1baf7a;--series-4:#eda100;--series-5:#e87ba4;--critical:#d03b3b}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--surface-1:#1a1a19;
--page:#0d0d0d;--text-primary:#f0efec;--text-secondary:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;
--border:rgba(255,255,255,.10);--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--series-4:#c98500;--series-5:#d55181}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-1:#1a1a19;--page:#0d0d0d;--text-primary:#f0efec;
--text-secondary:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--series-1:#3987e5;
--series-2:#d95926;--series-3:#199e70;--series-4:#c98500;--series-5:#d55181}
body{margin:0;background:var(--page,#f9f9f7)}
.viz-root{background:var(--page);color:var(--text-primary);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;
padding:24px 16px 48px;min-height:100vh;box-sizing:border-box}
.wrap{max-width:1100px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 10px}h3{font-size:15px;margin:20px 0 8px}
p.lede,.muted{color:var(--text-secondary)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:16px 0}
.tile{background:var(--surface-1);border:1px solid var(--border);border-radius:8px;padding:10px 12px}
.tile .v{font-size:20px;font-weight:600}.tile .l{color:var(--text-secondary);font-size:12px}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:8px;padding:12px 14px;margin:10px 0}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--grid);vertical-align:top}
th{color:var(--text-secondary);font-weight:600;font-size:12px}
td.n,th.n{text-align:right}
.scroll{overflow-x:auto}
.stack{display:flex;gap:2px;height:14px;margin:8px 0;background:var(--surface-1)}
.stack span{display:block;height:100%}.stack span:first-child{border-radius:4px 0 0 4px}.stack span:last-child{border-radius:0 4px 4px 0}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;color:var(--text-secondary);font-size:12px;margin:6px 0}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.bar{height:6px;border-radius:3px;background:var(--series-1);min-width:2px}
.chart{position:relative}.chart svg{width:100%;height:auto;display:block;overflow:visible}
.tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.gridline{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--axis);stroke-width:1}
.line{fill:none;stroke:var(--series-1);stroke-width:2;stroke-linejoin:round}
.win{stroke:var(--muted);stroke-width:1;stroke-dasharray:4 4}
.cross{stroke:var(--muted);stroke-width:1;visibility:hidden}
.m{stroke:var(--surface-1);stroke-width:2;cursor:default}
.m-compact{fill:var(--series-2)}.m-curate{fill:var(--series-3)}.m-engine{fill:var(--series-4)}.m-hint{fill:none;stroke:var(--muted);stroke-width:1.5}
#tip{position:fixed;pointer-events:none;z-index:9;max-width:420px;background:var(--surface-1);color:var(--text-primary);
border:1px solid var(--border);border-radius:6px;padding:7px 9px;font-size:12px;box-shadow:0 4px 14px rgba(0,0,0,.18);
white-space:pre-wrap;display:none}
#tip b{font-weight:600}
details{margin:6px 0}summary{cursor:pointer;color:var(--text-secondary)}
.ex{color:var(--text-secondary);font-size:12px}
code{font:12px ui-monospace,SFMono-Regular,Consolas,monospace}
"""

_JS = r"""
(() => {
  const tip = document.getElementById('tip');
  function place(ev) {
    const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
    let x = ev.clientX + pad, y = ev.clientY + pad;
    if (x + w > innerWidth - 8) x = ev.clientX - w - pad;
    if (y + h > innerHeight - 8) y = ev.clientY - h - pad;
    tip.style.left = Math.max(8, x) + 'px'; tip.style.top = Math.max(8, y) + 'px';
  }
  function show(lines, ev) {
    tip.replaceChildren();
    lines.forEach((line, i) => {
      const row = document.createElement('div');
      if (i === 0) { const b = document.createElement('b'); b.textContent = line; row.append(b); }
      else row.textContent = line;
      tip.append(row);
    });
    tip.style.display = 'block'; place(ev);
  }
  const hide = () => { tip.style.display = 'none'; };
  document.querySelectorAll('[data-tip]').forEach(el => {
    const lines = () => el.dataset.tip.split('\n');
    el.addEventListener('pointermove', ev => { ev.stopPropagation(); show(lines(), ev); });
    el.addEventListener('pointerleave', hide);
    el.addEventListener('focus', () => { const r = el.getBoundingClientRect(); show(lines(), {clientX: r.right, clientY: r.bottom}); });
    el.addEventListener('blur', hide);
  });
  const k = n => n >= 1e6 ? (n / 1e6).toFixed(2) + 'M' : n >= 1e4 ? Math.round(n / 1e3) + 'k' : n >= 1e3 ? (n / 1e3).toFixed(1) + 'k' : String(n);
  document.querySelectorAll('.chart[data-series]').forEach(fig => {
    const d = JSON.parse(fig.dataset.series), svg = fig.querySelector('svg'), cross = svg.querySelector('.cross');
    const n = d.t.length;
    svg.addEventListener('pointermove', ev => {
      if (ev.target.closest('[data-tip]')) return;
      const box = svg.getBoundingClientRect(), xr = (ev.clientX - box.left) / box.width * d.W;
      const i = Math.max(0, Math.min(n - 1, Math.round((xr - d.L) / (d.W - d.L - d.R) * Math.max(1, n - 1))));
      const x = d.L + (d.W - d.L - d.R) * (n > 1 ? i / (n - 1) : 0);
      cross.setAttribute('x1', x); cross.setAttribute('x2', x); cross.style.visibility = 'visible';
      const lines = [k(d.t[i]) + ' tokens in context', 'request ' + (i + 1) + ' of ' + n + ', turn ' + d.u[i]];
      if (d.r[i] != null) lines.push('cache read ' + d.r[i] + ' %');
      if (d.c[i] != null) lines.push('cost $' + d.c[i]);
      show(lines, ev);
    });
    svg.addEventListener('pointerleave', () => { cross.style.visibility = 'hidden'; hide(); });
  });
})();
"""


def _e(text: Any) -> str:
    return html.escape(str(text if text is not None else ""), quote=True)


def _nice(maxv: float) -> tuple[float, list[float]]:
    if maxv <= 0:
        return 1, [0, 1]
    raw = maxv / 4
    mag = 10 ** int(f"{raw:e}".split("e")[1])
    step = next(s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw)
    top = step * (int(maxv / step) + (1 if maxv % step else 0))
    ticks, v = [], 0.0
    while v <= top + 1e-9:
        ticks.append(v)
        v += step
    return top, ticks


def _event_tip(ev: dict) -> str:
    state = "dry run" if ev.get("dry_run") else "held" if ev.get("held") else "applied"
    lines = [f"{ev.get('source')} ({state}), turn {ev.get('turn')}"]
    if ev.get("before") is not None:
        lines.append(f"{_k(ev.get('before'))} → {_k(ev.get('after'))}, freed {_k(ev.get('freed') or ev.get('freed_estimate'))}")
    if ev.get("rewrite_tokens"):
        lines.append(f"rewrite {_k(ev['rewrite_tokens'])} tokens, break-even {ev.get('breakeven_requests')} requests")
    since = ev.get("since") or {}
    counts = [f"{since[k]} {label}" for k, label in (("batches", "tool batches"), ("requests", "requests")) if k in since]
    if counts:
        lines.append("since the last compact/curate: " + ", ".join(counts))
    if ev.get("tiers"):
        lines.append("tiers: " + ", ".join(map(str, ev["tiers"])))
    if ev.get("task"):
        lines.append("task: " + insights.excerpt(ev["task"], 220))
    if ev.get("call_text"):
        lines.append("with the call: " + insights.excerpt(ev["call_text"], 220))
    return "\n".join(lines)


def _context_chart(chat: dict) -> str:
    reqs = chat["requests"]
    if not reqs:
        return '<p class="muted">No requests recorded.</p>'
    W, H, L, R, T, B = 760, 230, 52, 14, 10, 26
    n = len(reqs)
    peak = max(r["tokens"] for r in reqs) or 1
    window = chat.get("window") or 0
    show_window = window and peak >= 0.5 * window
    top, ticks = _nice(max(peak, window if show_window else 0))

    def x(i):
        return L + (W - L - R) * (i / (n - 1) if n > 1 else 0)

    def y(v):
        return T + (H - T - B) * (1 - v / top)

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Context tokens per request in {_e(chat["title"])}">']
    for t in ticks:
        parts.append(f'<line class="gridline" x1="{L}" x2="{W - R}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>'
                     f'<text class="tick" x="{L - 6}" y="{y(t) + 4:.1f}" text-anchor="end">{_k(t)}</text>')
    parts.append(f'<line class="axis" x1="{L}" x2="{W - R}" y1="{y(0):.1f}" y2="{y(0):.1f}"/>')
    for j in range(5):
        i = round(j * (n - 1) / 4) if n > 1 else 0
        parts.append(f'<text class="tick" x="{x(i):.1f}" y="{H - 8}" text-anchor="middle">{i + 1}</text>')
    if show_window:
        parts.append(f'<line class="win" x1="{L}" x2="{W - R}" y1="{y(window):.1f}" y2="{y(window):.1f}"/>'
                     f'<text class="tick" x="{W - R}" y="{y(window) - 4:.1f}" text-anchor="end">window {_k(window)}</text>')
    pts = " ".join(f"{x(i):.1f},{y(r['tokens']):.1f}" for i, r in enumerate(reqs))
    parts.append(f'<polyline class="line" points="{pts}"/>')
    parts.append(f'<line class="cross" x1="0" x2="0" y1="{T}" y2="{y(0):.1f}"/>')
    for ev in chat["events"]:
        src = ev.get("source")
        if src not in ("compact_context", "curate_context", "engine", "hint"):
            continue
        if src in AGENT_SOURCES and not ev.get("applied"):
            continue
        i = min(ev["_req"], n - 1)
        # On the line: the event's own before/after count the history without the tool
        # schemas and live views, so they sit below the tokens actually sent.
        cx, cy = x(i), y(reqs[i]["tokens"])
        tip = _e(_event_tip(ev))
        if src == "compact_context":
            parts.append(f'<path class="m m-compact" tabindex="0" data-tip="{tip}" d="M{cx - 5.5:.1f},{cy - 5:.1f} '
                         f'L{cx + 5.5:.1f},{cy - 5:.1f} L{cx:.1f},{cy + 5:.1f} Z"/>')
        elif src == "curate_context":
            parts.append(f'<circle class="m m-curate" tabindex="0" data-tip="{tip}" cx="{cx:.1f}" cy="{cy:.1f}" r="4.5"/>')
        elif src == "engine":
            parts.append(f'<rect class="m m-engine" tabindex="0" data-tip="{tip}" x="{cx - 4:.1f}" y="{cy - 4:.1f}" '
                         f'width="8" height="8" rx="1"/>')
        else:
            parts.append(f'<circle class="m m-hint" tabindex="0" data-tip="{tip}" cx="{cx:.1f}" cy="{y(0) - 6:.1f}" r="3"/>')
    parts.append("</svg>")
    series = {"W": W, "L": L, "R": R, "t": [r["tokens"] for r in reqs], "u": [r["turn"] for r in reqs],
              "r": [round(100 * r["cache_read"] / max(1, r["cache_read"] + r["cache_write"] + r["uncached"]))
                    if (r["cache_read"] + r["cache_write"] + r["uncached"]) else None for r in reqs],
              "c": [round(r["cost"], 4) if isinstance(r["cost"], (int, float)) else None for r in reqs]}
    note = "" if show_window or not window else f'<span>window {_k(window)} (off the scale)</span>'
    legend = ('<div class="legend"><span><i style="background:var(--series-1)"></i>context tokens per request</span>'
              '<span><i style="background:var(--series-2)"></i>▼ compact_context</span>'
              '<span><i style="background:var(--series-3);border-radius:50%"></i>● curate_context</span>'
              '<span><i style="background:var(--series-4)"></i>■ engine compaction</span>'
              '<span><i style="border:1.5px solid var(--muted);border-radius:50%;background:none"></i>review hint</span>'
              f'{note}</div>')
    return (f'<figure class="chart" data-series="{_e(json.dumps(series))}" style="margin:0">{"".join(parts)}'
            f'<figcaption class="muted" style="font-size:12px">x: request number, y: tokens sent</figcaption></figure>' + legend)


def _stack(o: dict) -> str:
    keys = [("uncached", "uncached input", 1), ("cache_read", "cache read", 2), ("cache_write", "cache write", 3),
            ("output", "output", 4), ("reasoning", "reasoning", 5)]
    total = sum(o.get(k, 0) for k, _, _ in keys)
    if not total:
        return ""
    segs = "".join(f'<span data-tip="{_e(label)}\n{_k(o.get(k, 0))} tokens ({_pct(o.get(k, 0) / total)})" '
                   f'style="flex:{o.get(k, 0)} 0 0;background:var(--series-{slot})"></span>'
                   for k, label, slot in keys if o.get(k, 0))
    legend = "".join(f'<span><i style="background:var(--series-{slot})"></i>{_e(label)} {_k(o.get(k, 0))} '
                     f'({_pct(o.get(k, 0) / total)})</span>' for k, label, slot in keys)
    return f'<div class="stack" role="img" aria-label="Token composition">{segs}</div><div class="legend">{legend}</div>'


def _table(head: list[str], rows: list[list[Any]], numeric: set[int] | None = None) -> str:
    numeric = numeric or set()
    th = "".join(f'<th class="{"n" if i in numeric else ""}">{_e(h)}</th>' for i, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td class="{"n" if i in numeric else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>"
                   for r in rows)
    return f'<div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def render_html(stats: dict) -> str:
    o = stats["overview"]
    prompt = sum(o.get(k, 0) for k in ("uncached", "cache_read", "cache_write"))
    out: list[str] = []
    a = out.append
    a('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
      f'<title>Usage insights</title><style>{_CSS}</style></head><body><div class="viz-root"><div class="wrap">')
    a("<h1>Usage insights</h1>")
    a(f'<p class="lede">{_e(_when(stats["span"][0]))} to {_e(_when(stats["span"][1]))} · generated {_e(_when(stats["generated"]))}</p>')
    if not stats["chats"]:
        a('<p>No insights recorded yet. Turn on <b>Record usage insights</b> in Settings &gt; Insights and work in a chat.</p>')
        a('</div></div></body></html>')
        return "".join(out)
    tiles = [(len(stats["chats"]), "chats"), (o.get("turns", 0), "turns"), (o.get("requests", 0), "requests"),
             (o.get("tool_calls", 0), "tool calls"), (_pct(_rate(o.get("tool_failures", 0), o.get("tool_calls", 0))), "tool calls failed"),
             (_usd(o.get("cost")), "cost"), (_pct(_rate(o.get("cache_read", 0), prompt)), "of input read from cache")]
    a('<div class="tiles">' + "".join(f'<div class="tile"><div class="v">{_e(v)}</div><div class="l">{_e(l)}</div></div>'
                                      for v, l in tiles) + "</div>")

    a("<h2>Tokens</h2>")
    a('<div class="card">' + _stack(o) + "</div>")
    rows = []
    for m, c in sorted(stats["models"].items(), key=lambda kv: -kv[1].get("cost", 0)):
        p = sum(c.get(k, 0) for k in ("uncached", "cache_read", "cache_write"))
        rows.append([_e(m or "?"), c.get("requests", 0), _usd(c.get("cost")), _k(c.get("uncached")), _k(c.get("cache_read")),
                     _k(c.get("cache_write")), _k(c.get("output")), _k(c.get("reasoning")), _pct(_rate(c.get("cache_read", 0), p)),
                     c.get("tool_calls", 0), _pct(_rate(c.get("tool_failures", 0), c.get("tool_calls", 0)))])
    a(_table(["model", "requests", "cost", "uncached", "cache read", "cache write", "output", "reasoning",
              "cache-read share", "tool calls", "tool failures"], rows, set(range(1, 11))))
    t = stats["turns"]
    if t.get("count"):
        a(f'<p class="muted">Per turn: median {_usd(t["cost_median"])} (p90 {_usd(t["cost_p90"])}), '
          f'{t["requests_median"]:.0f} requests (p90 {t["requests_p90"]:.0f}), {t["tools_median"]:.0f} tool calls '
          f'(p90 {t["tools_p90"]:.0f}). {o.get("capped_responses", 0)} responses hit the output cap, '
          f'{o.get("request_errors", 0)} requests failed.'
          + (f' {o["keepalives"]} cache keep-alives during long tool runs ({_usd(o.get("keepalive_cost"))}).'
             if o.get("keepalives") else '') + '</p>')

    a("<h2>Context window over time</h2>")
    a('<p class="muted">One chart per chat, newest first. Hover a marker for what the chat was doing at that moment.</p>')
    for chat in stats["chats"][:CHART_CHATS]:
        a(f'<div class="card"><h3 style="margin-top:0">{_e(chat["title"])}{" (delegated)" if chat["child"] else ""}</h3>'
          f'<div class="muted" style="font-size:12px">{_e(chat["model"])} · {chat["turns"]} turns · {len(chat["requests"])} requests · '
          f'peak {_k(chat["peak"])} · {chat["compactions"]} compact, {chat["curates"]} curate, {chat["engine"]} engine · '
          f'{_usd(chat["cost"])}</div>{_context_chart(chat)}</div>')
    if len(stats["chats"]) > CHART_CHATS:
        a(f'<p class="muted">{len(stats["chats"]) - CHART_CHATS} older chats are in the table below and the Markdown report.</p>')
    rows = [[_e(c["title"][:70]), _e(c["model"]), _k(c["window"]), c["turns"], len(c["requests"]), _k(c["peak"]), _k(c["final"]),
             c["compactions"], c["curates"], c["engine"], _usd(c["cost"])] for c in stats["chats"]]
    a(_table(["chat", "model", "window", "turns", "requests", "peak", "last", "compact", "curate", "engine", "cost"],
             rows, set(range(2, 11))))

    cs = stats["context"]
    a("<h2>Compaction and curation</h2>")
    rows = [[_e(src), r["applied"], r["dry_run"], r["held"], _k(r["freed_total"]), _k(r["freed_median"]), _k(r["before_median"]),
             _k(r["rewrite_median"]), _k(r["since_batches_median"]), _k(r["since_requests_median"])]
            for src, r in cs["by_source"].items()]
    a(_table(["source", "applied", "dry runs", "held", "freed total", "freed median", "context before (median)",
              "rewrite (median)", "tool batches since last (median)", "requests since last (median)"], rows, set(range(1, 10))))
    a(f'<p class="muted">Review hints: {cs["hints"]}; followed by a compact or curate within {HINT_FOLLOW} requests: '
      f'{cs["hints_followed"]}.</p>')
    agent_events = [e for e in stats["context_events"] if e.get("source") in AGENT_SOURCES]
    rows = []
    titles = {c["id"]: c["title"] for c in stats["chats"]}
    for ev in reversed(agent_events[-200:]):
        state = "dry run" if ev.get("dry_run") else "held" if ev.get("held") else "applied"
        detail = []
        for label, key in (("task", "task"), ("with the call", "call_text"), ("last answer", "last_answer"), ("summary", "summary")):
            if ev.get(key):
                detail.append(f"<div class='ex'><b>{label}:</b> {_e(ev[key])}</div>")
        if ev.get("dropped"):
            detail.append(f"<div class='ex'><b>dropped:</b> {_e(', '.join(ev['dropped'][:12]))}</div>")
        since = ev.get("since") or {}
        rows.append([_e(_when(ev.get("ts"))), _e(titles.get(ev["_chat"], ev["_chat"])[:40]), ev.get("turn"),
                     _e(ev.get("source").replace("_context", "")), state,
                     f'{_k(ev.get("before"))} → {_k(ev.get("after"))}', _k(ev.get("freed") or ev.get("freed_estimate")),
                     since.get("batches", "–"), "<details><summary>show</summary>" + "".join(detail) + "</details>" if detail else ""])
    if rows:
        a(_table(["time", "chat", "turn", "tool", "state", "context", "freed", "batches since last", "what the chat was doing"],
                 rows, {2, 6, 7}))

    a("<h2>Tools</h2>")
    cats = stats["categories"]
    rows = [[_e(k), v.get("calls", 0), v.get("failed", 0), _pct(_rate(v.get("failed", 0), v.get("calls", 0))),
             _k(v.get("tokens", 0)), v.get("dropped_later", 0)] for k, v in cats.items()]
    a(_table(["category", "calls", "failed", "failure rate", "result tokens", "results dropped later"], rows, {1, 2, 3, 4, 5}))
    top = max((r["calls"] for r in stats["tools"]), default=1)
    rows = []
    for r in stats["tools"]:
        main = r["classes"][0][0].replace("result.", "") if r["classes"] else ""
        rows.append([f"<code>{_e(r['tool'])}</code>", _e(r["category"]),
                     f'{r["calls"]}<div class="bar" style="width:{100 * r["calls"] / top:.0f}%"></div>', r["failed"],
                     _pct(r["failure_rate"]), _e(main), _k(r["median_ms"]), _k(r["mean_tokens"]), r["dropped_later"],
                     r["repeated"]])
    a(_table(["tool", "category", "calls", "failed", "failure rate", "main failure", "median ms", "mean result tokens",
              "dropped later", "repeated failures"], rows, {3, 4, 6, 7, 8, 9}))

    e = stats["edits"]
    a("<h2>Editing</h2>")
    if e.get("calls"):
        rf = e["read_first"]
        bits = [f"{e['calls']} edit calls. Failed edits fixed within {RECOVERY_LOOKAHEAD} tool calls: {e['recovered']}; "
                f"not fixed: {e['unrecovered']} (median calls to the fix: {e['calls_to_fix_median'] or '–'})."]
        for k, label in READ_FIRST_LABELS:
            b = rf.get(k) or {}
            if b.get("calls"):
                bits.append(f"{label[0].upper() + label[1:]}: {b['calls']} edits, "
                            f"{_pct(_rate(b.get('failed', 0), b['calls']))} failed.")
        if e.get("write_file"):
            bits.append(f"write_file created {e['write_file'].get('created', 0)} files and overwrote "
                        f"{e['write_file'].get('overwrote', 0)} existing ones (a whole-file rewrite instead of an edit).")
        bits.append(f"edit_file old_string: median {_k(e['edit_file_old_chars_median'])} characters "
                    f"(p90 {_k(e['edit_file_old_chars_p90'])}); new_string median {_k(e['edit_file_new_chars_median'])}.")
        a('<div class="card">' + "".join(f"<p style='margin:4px 0'>{_e(b)}</p>" for b in bits) + "</div>")
        rows = [[f"<code>{_e(k)}</code>", v.get("calls", 0), v.get("failed", 0), _pct(_rate(v.get("failed", 0), v.get("calls", 0)))]
                for k, v in e["per_tool"].items()]
        a(_table(["tool", "calls", "failed", "failure rate"], rows, {1, 2, 3}))
        if e["classes"]:
            a("<h3>Why edits failed</h3>" + _table(["class", "edits"], [[_e(c.replace("result.", "")), n] for c, n in e["classes"]], {1}))
        if e.get("miss_context"):
            a("<h3>Missed old_string, by what happened to the file</h3>"
              + _table(["the file was", "edits"], [[_e(k), n] for k, n in e["miss_context"]], {1}))
        if e["files"]:
            a("<h3>Files edited most</h3>" + _table(["file", "successful edits"], [[f"<code>{_e(p)}</code>", n] for p, n in e["files"]], {1}))
    else:
        a('<p class="muted">No edits recorded.</p>')

    i = stats["inspection"]
    a("<h2>Inspection</h2>")
    if i.get("calls"):
        a(f'<div class="card"><p style="margin:4px 0">read_file: {i["read_file_calls"]} calls, {i["read_file_partial"]} of them partial; '
          f'median {_k(i["read_file_tokens_median"])} tokens per result (p90 {_k(i["read_file_tokens_p90"])}); {i["files_read"]} files, '
          f'{i["repeat_reads"]} repeat reads of a file already read.</p><p style="margin:4px 0">Searches: {i["searches"]}, '
          f'{i["empty_searches"]} without a match. Identical re-reads (same file and range, no edit in between): '
          f'{i["identical_rereads"]}; reads of a file after editing it: {i["reads_after_edit"]}.</p></div>')
        rows = [[f"<code>{_e(k)}</code>", v["calls"], v["failed"], _k(v["mean_tokens"]), v["dropped_later"]] for k, v in i["per_tool"].items()]
        a(_table(["tool", "calls", "failed", "mean result tokens", "dropped later"], rows, {1, 2, 3, 4}))
        if i["most_read"]:
            a("<h3>Files read most</h3>" + _table(["file", "reads"], [[f"<code>{_e(p)}</code>", n] for p, n in i["most_read"]], {1}))
    else:
        a('<p class="muted">No inspection calls recorded.</p>')

    c = stats.get("commands") or {}
    a("<h2>Shell commands</h2>")
    if c.get("calls"):
        a(f'<div class="card"><p style="margin:4px 0">run_command: {c["calls"]} calls, {c["nonzero"]} with a non-zero exit '
          f'({_pct(_rate(c["nonzero"], c["calls"]))}), {c["timeouts"]} timed out. {c["shell_inspection"]} of them only read '
          f'or searched files (Get-Content, cat, Select-String, grep, ls, ...; mean {_k(c["shell_inspection_mean_tokens"])} '
          f'result tokens), the job of read_file, grep_search and list_dir.</p></div>')
        a(_table(["command", "calls", "non-zero exit", "share"],
                 [[f"<code>{_e(h)}</code>", n, z, _pct(_rate(z, n))] for h, n, z in c["heads"]], {1, 2, 3}))
        if c["signatures"]:
            a("<h3>Why commands failed</h3>" + _table(
                ["end of the output", "commands", "example"],
                [[_e(label), n, f"<span class='ex'>{_e(insights.excerpt(c['examples'].get(label), 220))}</span>"]
                 for label, n in c["signatures"]], {1}))
    else:
        a('<p class="muted">No shell commands recorded.</p>')
    a("<h2>Per model: tools that fail most and tools that work</h2>")
    a(f'<p class="muted">Tools with at least {MIN_CALLS} calls. A failure is an error result or a classified error '
      '(a non-zero exit of run_command is listed separately, not as a failure).</p>')
    for model in sorted(stats["difficult"]):
        a(f"<h3>{_e(model or '?')}</h3>")
        rows = []
        for r in stats["difficult"][model]:
            cls = ", ".join(f"{c.replace('result.', '')} {n}" for c, n in r["classes"][:3])
            ex = (r["examples"].get(r["classes"][0][0]) if r["classes"] else "") or ""
            rows.append([f"<code>{_e(r['tool'])}</code>", r["calls"], _pct(r["failure_rate"]), _e(cls),
                         f"<span class='ex'>{_e(insights.excerpt(ex, 200))}</span>"])
        if rows:
            a(_table(["tool", "calls", "failure rate", "classes", "example"], rows, {1, 2}))
        ok = stats["reliable"].get(model) or []
        if ok:
            a('<p class="muted">Works (≤ 2 % failures): ' + ", ".join(f"<code>{_e(r['tool'])}</code> ({r['calls']})" for r in ok) + "</p>")
    a('</div><div id="tip" role="tooltip"></div></div>')
    a(f"<script>{_JS}</script></body></html>")
    return "".join(out)


# --------------------------------------------------------------------------- entry points
def _only_models(events: list[dict], models: Iterable[str], meta: dict | None = None) -> list[dict]:
    """The events of a chat that ran on one of `models` (substrings of the model id). A chat
    can switch models between requests; every event belongs to the model of the latest turn
    start or request before it."""
    wanted = [m.lower() for m in models]
    current = str((meta or {}).get("model") or "")
    kept = []
    for ev in events:
        if ev.get("kind") in ("turn_start", "request") and ev.get("model"):
            current = ev["model"]
        if any(w in current.lower() for w in wanted):
            kept.append(ev)
    return kept


# --------------------------------------------------------------------------- entry points
def build(folder: Path | None = None, chats: Iterable[str] | None = None, *, html_path: Path | None = None,
          md_path: Path | None = None, json_path: Path | None = None,
          from_db: Iterable[Path] | None = None, models: Iterable[str] | None = None) -> dict[str, Any]:
    """Writes the report. `from_db` reads chat databases instead of the insights folder
    (insights_import.py); `models` keeps only chats whose model contains one of the strings."""
    if from_db:
        from . import insights_import
        by_chat, titles = {}, {}
        for db in from_db:
            events, meta = insights_import.events_from_db(Path(db))
            for cid, evs in events.items():
                if chats and cid not in set(chats):
                    continue
                key = cid if cid not in by_chat else f"{Path(db).parent.name}-{cid}"
                by_chat[key], titles[key] = evs, meta.get(cid, {})
    else:
        by_chat, titles = load(folder, chats), chat_titles()
    if models:
        by_chat = {cid: evs for cid, evs in ((cid, _only_models(evs, models, titles.get(cid))) for cid, evs in by_chat.items())
                   if evs}
    stats = analyse(by_chat, titles)
    target = Path(folder or insights.folder())
    html_path = html_path or target / "report.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(render_html(stats), encoding="utf-8")
    if md_path:
        md_path.write_text(render_markdown(stats), encoding="utf-8")
    if json_path:
        json_path.write_text(json.dumps(stats, indent=1, default=str), encoding="utf-8")
    return {"html": str(html_path), "chats": len(by_chat), "events": sum(len(v) for v in by_chat.values()),
            "tool_calls": stats["overview"].get("tool_calls", 0), "requests": stats["overview"].get("requests", 0)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--folder", type=Path, help="insights folder (default: DATA_DIR/insights)")
    ap.add_argument("--chat", action="append", help="only this chat id (repeatable)")
    ap.add_argument("--html", type=Path, help="HTML report path (default: <folder>/report.html)")
    ap.add_argument("--md", type=Path, help="also write a Markdown summary")
    ap.add_argument("--json", type=Path, help="also write the statistics as JSON")
    ap.add_argument("--from-db", type=Path, action="append",
                    help="build from a chat database instead (repeatable; read-only, see insights_import.py)")
    ap.add_argument("--model", action="append", help="only chats whose model id contains this (repeatable)")
    args = ap.parse_args(argv)
    if args.from_db and not args.html:
        ap.error("--from-db needs --html (the report is not written into the insights folder)")
    info = build(args.folder, args.chat, html_path=args.html, md_path=args.md, json_path=args.json,
                 from_db=args.from_db, models=args.model)
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
