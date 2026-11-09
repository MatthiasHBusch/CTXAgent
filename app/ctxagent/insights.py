"""Usage insights: an opt-in record of how a chat uses its context window and its tools.

Off by default (Settings > Insights, preference `insights`). When it is on, every chat
appends one JSON object per line to DATA_DIR/insights/<chat_id>.jsonl:

  turn_start  the user's message (excerpt), model, effort, window, context estimate
  request     one model request: prompt tokens split into uncached / cache read / cache
              write, output and reasoning tokens, cost, context estimate, finish reason,
              number of tool calls, duration, the reasons a cache break was expected
  tool        one tool call: name, category, argument summary (paths, flags and sizes;
              file contents and edit strings only as sizes), outcome, error class and
              excerpt, result tokens, duration, the result's segment id
  context     a compact_context / curate_context call (applied, dry run or held), an
              engine compaction or a review hint: tokens before and after, what was
              dropped with its notes, the summary excerpt, and what the chat was doing at
              that moment (the user's task, the text that came with the call, the last
              answer, tool batches and answers since the previous context event)
  turn_end    outcome, answer excerpt, the turn's tokens, cost and tool calls

Excerpts are short (EXCERPT characters) and stay on this computer. The usage ledger
(DATA_DIR/usage, always on) keeps the full per-request accounting; this file adds the
tool and context events around it. insights_report.py turns the files into a report.
Recording never raises: a full disk or a locked file loses an event, not a turn.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

from . import config

SCHEMA = 1
EXCERPT = 600
SHORT = 160

_ID = re.compile(r"[A-Za-z0-9_-]{1,100}")
_write_lock = threading.Lock()
_pref = [0.0, False]  # (checked at, enabled)
_PREF_TTL = 2.0

#: Tool name -> category. Names not listed fall back by prefix (category()).
CATEGORIES: dict[str, tuple[str, ...]] = {
    "edit": ("write_file", "edit_file", "prepare_edit_plan", "apply_edit_plan", "document_prepare_edits",
             "document_apply_edits", "document_json_update", "document_table_update", "document_docx_replace",
             "document_xlsx_update", "document_obj_transform"),
    "inspect": ("read_file", "list_dir", "focus_file", "summarize_file", "summarize_folder", "read_document",
                "view_image", "render_pdf", "document_open", "document_close", "document_list",
                "document_pdf_inspect", "document_pdf_text", "document_pdf_render", "document_docx_inspect",
                "document_xlsx_inspect", "document_mesh_inspect", "latex_inspect", "latex_diagnostics",
                "environment_report"),
    "search": ("glob_search", "grep_search"),
    "shell": ("run_command", "start_job", "job_status", "job_wait", "list_jobs", "read_job_log", "cancel_job",
              "latex_build"),
    "context": ("recall", "pin", "compact_context", "curate_context", "context_report"),
    "collaboration": ("scratchpad_read", "scratchpad_write"),
    "canvas": ("canvas_open", "canvas_close", "canvas_draw", "canvas_script", "canvas_focus", "canvas_undo",
               "canvas_save"),
    "web": ("fetch_url", "scholarly_search", "doi_lookup"),
    "research": ("record_evidence", "experiment_manifest", "finalize_experiment"),
}
_CATEGORY = {name: cat for cat, names in CATEGORIES.items() for name in names}

#: Argument keys whose values are content (file text, edit strings, summaries): recorded as
#: sizes only, however short.
CONTENT_KEYS = frozenset({
    "content", "old_string", "new_string", "text", "body", "summary", "code", "replacement", "patch",
    "new_text", "old_text", "data", "value", "script", "note", "message", "instructions", "prompt",
    "task", "query_text", "edits", "operations", "rows", "cells",
})

#: (class, regex on the start of an error result). Shared with tools_scripts/scan_tool_failures.py.
RESULT_PATTERNS = [
    ("result.bad_arguments", r"Bad arguments for "),
    ("result.unknown_tool", r"Unknown tool: "),
    ("result.disabled_tool", r"Tool '.*' is not enabled for this chat"),
    ("result.denied_read_only", r"Tool denied: this delegated task is read-only"),
    ("result.bad_document_call", r"Invalid document call"),
    ("result.cancelled_in_batch", r"Tool not executed: user cancelled"),
    ("result.interrupted", r"\[interrupted\] No result recorded"),
    ("result.command_timeout", r"Error: Command timed out after"),
    ("result.edit_not_found", r"Error: old_string not found"),
    ("result.edit_ambiguous", r"Error: old_string occurs \d+ times"),
    ("result.edit_empty", r"Error: old_string must not be empty"),
    ("result.not_utf8", r"Error: .* is not valid UTF-8"),
    ("result.path_not_found", r"Error: Not found: "),
    ("result.path_outside", r"Error: Path is outside the allowed"),
    ("result.no_workdir", r"Error: No working directory"),
    ("result.is_directory", r"Error: .* is (?:a|not a) directory"),
    ("result.bad_regex", r"Error: Bad regex: "),
    ("result.recall_missing", r"No active or archived segment"),
    ("result.context_tool_error", r"(?:compact_context|curate_context): "),
    ("result.pin_error", r"pin: pass segment_ids"),
    ("result.tool_error", r"Error: "),
    ("result.exception", r"(?:[A-Z][A-Za-z]+(?:Error|Exception)|ValueError|KeyError|TypeError|"
                         r"RuntimeError|OSError|PermissionError|FileNotFoundError|WorkspaceError|"
                         r"TimeoutError|CancelledError|JSONDecodeError)\b[:(]"),
]
_RESULT_RX = [(c, re.compile(p)) for c, p in RESULT_PATTERNS]
#: run_command results start with "$ <command>   (cwd=..., exit=N)"; the command may span lines.
_CMD_EXIT = re.compile(r"\(cwd=[^\n]*?, exit=(-?\d+)\)")
REPEAT_NOTE = "[harness] This exact call already failed the same way"


def category(name: str) -> str:
    if name in _CATEGORY:
        return _CATEGORY[name]
    for prefix, cat in (("browser_", "web"), ("web_", "web"), ("agent_", "agents"), ("document_", "inspect")):
        if name.startswith(prefix):
            return cat
    return "other"


def classify_result(body: str) -> str | None:
    """Error class of a tool result body, or None for a plain success."""
    head = (body or "")[:400]
    for cls, rx in _RESULT_RX:
        if rx.match(head):
            return cls
    code = exit_code(body)
    if code not in (None, 0):
        return "info.command_nonzero_exit"
    if head.startswith("{") and '"error"' in head[:200]:
        return "result.json_error"
    return None


def exit_code(body: str) -> int | None:
    if not (body or "").startswith("$ "):
        return None
    m = _CMD_EXIT.search(body[:20000])
    return int(m.group(1)) if m else None


#: Statements that set up a command line rather than run a program.
_SETUP = re.compile(r"^(?:cd|set-location|sl|pushd|chdir|export|chcp)\b|^\$[\w:]+\s*=|^set\s+-[a-z]+|"
                    r"^\[Console\]::|^\$ProgressPreference|^\$ErrorActionPreference", re.I)


#: Programs that only read or search files: the shell doing read_file's or grep_search's job.
SHELL_INSPECTION = frozenset({"get-content", "gc", "cat", "type", "head", "tail", "less", "more", "select-string", "sls",
                              "grep", "rg", "findstr", "get-childitem", "gci", "ls", "dir", "find", "tree", "wc",
                              "measure-object", "sed", "awk"})

#: PowerShell blocks that are a script rather than one program.
_SCRIPT_WORDS = ("for", "foreach", "function", "if", "while", "do", "try", "switch", "$(", "{")


def command_program(command: Any) -> str:
    """The program a shell command runs, past `cd`, variable assignments and here-strings:
    `cd x; & C:\\py\\python.exe -m pytest` -> python, `$c = @' ... '@; $c | python -` and
    `@' ... '@ | python -` -> python, `for (...) { ... }` -> "powershell script"."""
    text = re.sub(r"@(['\"])\r?\n.*?\r?\n\1@", "@HERE@", str(command or ""), flags=re.S)
    for stmt in re.split(r"\s*(?:;|&&|\|\||\n)\s*", text):
        stmt = re.sub(r"^&\s*", "", stmt.strip())
        if not stmt or _SETUP.match(stmt):
            continue
        if stmt.startswith(("@HERE@", "$")) and "|" in stmt:
            stmt = stmt.split("|", 1)[1].strip()
            stmt = re.sub(r"^&\s*", "", stmt)
        first = re.split(r"[\s(]", stmt, maxsplit=1)[0].lower() if stmt[:1] not in "{$" else stmt[:2]
        if first in _SCRIPT_WORDS or stmt.startswith(("{", "$(")):
            return "powershell script"
        token = re.match(r"\"([^\"]+)\"|'([^']+)'|(\S+)", stmt)
        return _program(next(g for g in token.groups() if g)) if token else "?"
    return "?"


def _program(token: str) -> str:
    base = re.split(r"[\\/]", token.strip("'\"()"))[-1].lower()
    return re.sub(r"\.(exe|cmd|bat|ps1|sh)$", "", base) or "?"


def tail(body: str, limit: int = 300) -> str:
    """The end of an output, where a failing command usually says why."""
    flat = re.sub(r"\s+", " ", (body or "")[-(limit * 3):]).strip()
    return flat if len(flat) <= limit else "…" + flat[-(limit - 1):]


def folder() -> Path:
    return Path(config.DATA_DIR) / "insights"


def enabled() -> bool:
    now = time.monotonic()
    if now - _pref[0] < _PREF_TTL:
        return _pref[1]
    try:
        from . import profiles
        on = bool(profiles.load_prefs().get("insights"))
    except Exception:
        on = False
    _pref[0], _pref[1] = now, on
    return on


def refresh() -> None:
    """Forget the cached preference (after Settings changed it)."""
    _pref[0] = 0.0


def write(chat_id: str, kind: str, **fields: Any) -> None:
    if not enabled() or not _ID.fullmatch(chat_id or ""):
        return
    event = {"v": SCHEMA, "ts": round(time.time(), 3), "chat": chat_id, "kind": kind, **fields}
    try:
        line = json.dumps(event, ensure_ascii=False, default=str, allow_nan=False)
        with _write_lock:
            target = folder()
            target.mkdir(parents=True, exist_ok=True)
            with (target / f"{chat_id}.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
    except (OSError, ValueError, TypeError):
        pass


def excerpt(text: Any, limit: int = EXCERPT) -> str:
    flat = re.sub(r"\s+", " ", str(text or "")).strip()
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _size(value: str) -> dict[str, int]:
    return {"chars": len(value), "lines": value.count("\n") + 1}


def summarize_args(name: str, args: dict | None) -> dict[str, Any]:
    """Arguments without their contents: paths, flags and numbers as they are, text as sizes."""
    out: dict[str, Any] = {}
    for key, value in (args or {}).items():
        if key == "ctx":
            continue
        if key == "command" and isinstance(value, str):
            out[key] = excerpt(value, 240)
        elif isinstance(value, bool) or value is None or isinstance(value, (int, float)):
            out[key] = value
        elif isinstance(value, str):
            out[key] = _size(value) if key in CONTENT_KEYS or len(value) > SHORT or "\n" in value else value
        elif isinstance(value, list):
            simple = len(value) <= 8 and all(isinstance(v, str) and len(v) <= 80 for v in value)
            out[key] = value if simple and key not in CONTENT_KEYS else {"items": len(value)}
        elif isinstance(value, dict):
            out[key] = {"keys": sorted(map(str, value))[:12]}
        else:
            out[key] = type(value).__name__
    return out


def _last_user_task(conv) -> str:
    """The latest message the user typed (harness notes are not the user)."""
    for seg in reversed(conv.segments):
        if seg.role == "user" and seg.kind == "user" and not (seg.body or "").startswith("[harness note]"):
            return seg.body or ""
    return ""


def _last_answer(conv) -> str:
    """The latest finished answer (text without tool calls)."""
    for seg in reversed(conv.segments):
        if seg.kind == "assistant_text" and (seg.body or "").strip():
            return seg.body
    return ""


class Recorder:
    """Per-session state: turn numbers and the counts since the last context event."""

    def __init__(self, session) -> None:
        self.s = session
        self.turn = None  # counted from the file on first use, so it survives restarts
        self.since = {"batches": 0, "answers": 0, "requests": 0}
        self.turn_start_usage: dict[str, float] = {}
        self.turn_tools = 0
        self.turn_errors = 0

    @property
    def chat(self) -> str:
        return self.s.chat_id

    def _next_turn(self) -> int:
        if self.turn is None:
            self.turn = 0
            try:
                path = folder() / f"{self.chat}.jsonl"
                if path.is_file():
                    with path.open(encoding="utf-8") as stream:
                        self.turn = sum(1 for line in stream if '"kind": "turn_start"' in line)
            except OSError:
                pass
        self.turn += 1
        return self.turn

    def _context_estimate(self) -> int:
        try:
            from . import document_catalog
            return (self.s.conv.total_tokens() + self.s.tool_overhead() + self.s.conv.workspace.token_count()
                    + self.s.conv.browser.token_count() + document_catalog.token_count(self.s))
        except Exception:
            return 0

    def _usage_totals(self) -> dict[str, float]:
        u = self.s.usage or {}
        return {k: u.get(k) or 0 for k in ("prompt_tokens", "completion_tokens", "cache_read", "cache_write",
                                           "cost", "requests")}

    # -- events --------------------------------------------------------------
    def turn_start(self, user_text: str) -> None:
        if not enabled():
            return
        cfg = self.s.cfg
        self.turn_start_usage = self._usage_totals()
        self.turn_tools = self.turn_errors = 0
        write(self.chat, "turn_start", turn=self._next_turn(), model=cfg.model, effort=cfg.reasoning_effort,
              window=cfg.context_window, self_steered=cfg.compaction.self_steered,
              context=self._context_estimate(), user=excerpt(user_text), user_chars=len(user_text or ""))

    def turn_end(self, outcome: str, answer: str = "") -> None:
        if not enabled():
            return
        now, start = self._usage_totals(), self.turn_start_usage or {}
        delta = {k: round(now[k] - (start.get(k) or 0), 6) for k in now}
        write(self.chat, "turn_end", turn=self.turn, outcome=outcome, answer=excerpt(answer),
              answer_chars=len(answer or ""), tools=self.turn_tools, tool_errors=self.turn_errors,
              context=self._context_estimate(), **delta)

    def request(self, *, iteration: int, effort: str, entry: dict | None, finish: str | None, tool_calls: int,
                text_chars: int, ms: int, error: str | None = None, diagnostic: dict | None = None) -> None:
        if not enabled():
            return
        self.since["requests"] += 1
        u = (entry or {}).get("usage") or {}
        reasons = [r for r in ((diagnostic or {}).get("reasons") or []) if r]
        write(self.chat, "request", turn=self.turn, iteration=iteration, model=(entry or {}).get("model")
              or self.s.cfg.model, provider=(entry or {}).get("provider"), effort=effort,
              prompt_tokens=u.get("prompt_tokens"), uncached=u.get("input"), cache_read=u.get("cache_read"),
              cache_write=u.get("cache_write"), output=u.get("completion_tokens"), reasoning=u.get("reasoning"),
              cost=u.get("cost"), context=self._context_estimate(), window=self.s.cfg.context_window,
              finish=finish, tool_calls=tool_calls, text_chars=text_chars, ms=ms,
              error=excerpt(error, 300) if error else None, cache_break=reasons[:4] or None)

    def answer(self) -> None:
        self.since["answers"] += 1

    def batch_done(self) -> None:
        self.since["batches"] += 1

    def tool(self, *, name: str, args: dict, result, ms: int, segment_id: str | None, tokens: int,
             iteration: int) -> None:
        if not enabled():
            return
        body = result.text or ""
        cls = classify_result(body) if (result.error or body.startswith(("Error:", "Bad arguments"))) else None
        if cls is None and name == "run_command":
            cls = classify_result(body)
        self.turn_tools += 1
        self.turn_errors += int(bool(result.error))
        fields: dict[str, Any] = {}
        if name == "document_call":
            fields["inner"] = (args or {}).get("tool") or (args or {}).get("name")
        if name == "run_command":
            fields["exit"] = exit_code(body)
            fields["program"] = command_program((args or {}).get("command"))
            if fields["exit"] not in (None, 0):
                fields["tail"] = tail(body)
        if category(name) == "search" and not result.error:
            fields["empty"] = body.rstrip().endswith("(no matches)")
        if name == "write_file" and not result.error:
            fields["overwrote"] = body.startswith("Overwrote")
        write(self.chat, "tool", turn=self.turn, iteration=iteration, name=name, category=category(name),
              args=summarize_args(name, args), ok=not result.error, error_class=cls,
              error=excerpt(body, 300) if (result.error or cls) else None, ms=ms, tokens=tokens,
              chars=len(body), mutates=bool(getattr(result, "mutates", False)), segment=segment_id,
              repeated=REPEAT_NOTE in body, **fields)

    def context(self, *, source: str, report: dict, call_text: str = "", dry_run: bool = False) -> None:
        """A context event: an agent's compact/curate call or an engine compaction."""
        if not enabled():
            return
        conv = self.s.conv
        applied = not dry_run and not report.get("held") and source != "hint"
        fields: dict[str, Any] = {
            "turn": self.turn, "source": source, "dry_run": bool(dry_run), "held": bool(report.get("held")),
            "applied": applied,
            "before": report.get("before_tokens", report.get("before")),
            "after": report.get("after_tokens", report.get("after")),
            "freed": report.get("freed", report.get("saved_tokens")),
            "freed_estimate": report.get("freed_estimate"),
            "rewrite_tokens": report.get("cache_rewrite_tokens"),
            "breakeven_requests": report.get("cache_breakeven_requests"),
            "window": self.s.cfg.context_window,
            "since": dict(self.since),
            "task": excerpt(_last_user_task(conv)),
            "call_text": excerpt(call_text, 400) if call_text else None,
            "last_answer": excerpt(_last_answer(conv), 400),
        }
        if report.get("tiers"):
            fields["tiers"] = [t.get("tier") for t in report["tiers"] if isinstance(t, dict)]
        if report.get("dropped"):
            fields["dropped"] = [str(d)[:200] for d in report["dropped"][:40]]
        if report.get("would_drop"):
            fields["would_drop"] = list(report["would_drop"][:40])
        if report.get("problems"):
            fields["problems"] = [str(p)[:200] for p in report["problems"][:10]]
        if report.get("summary_tokens") is not None:
            fields["summary_tokens"] = report.get("summary_tokens")
        if report.get("summary"):
            fields["summary"] = excerpt(report.get("summary"), 1200)
        if report.get("review_hint"):
            fields["hint"] = excerpt(report.get("review_hint"), 400)
        if report.get("deep") is not None:
            fields["deep"] = bool(report.get("deep"))
        if report.get("pressure") is not None:
            fields["pressure"] = report.get("pressure")
        write(self.chat, "context", **fields)
        if applied and source in ("compact_context", "curate_context") and (fields["freed"] or 0) > 0:
            self.since = {"batches": 0, "answers": 0, "requests": 0}


def recorder(session) -> Recorder:
    rec = session.__dict__.get("_insights")
    if rec is None:
        rec = session.__dict__["_insights"] = Recorder(session)
    return rec
