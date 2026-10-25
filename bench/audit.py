"""Transcript audit for the web ban (REDESIGN.md 4.2).

After each trial the runner scans the raw transcript of the harness for

  * banned tool names (web / browser / document tools by harness), and
  * network-reaching shell commands (curl, wget, Invoke-WebRequest, irm, pip install,
    npm install, git clone <url>, requests.get, urllib, ...),

including sub-agent activity where it is recorded in the same place (Claude Code's
Task sub-agents stream into the same file; ctxagent worker chats live in the same store).
Hits are returned as a list of {"where", "tool", "hit"} with the offending call quoted,
and the runner stores them as `web_use` on the trial row. A hit is a flag for hand
review, not a verdict: `pip install` of an already-installed package breaks the
instruction without leaking information, and is reported as such.

    python -m bench.audit bench/.work/<tag> [claude_code|codex|ctxagent]
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

# Tool names that must not appear, per harness (exact names; prefixes end with '*').
BANNED_TOOLS = {
    "claude_code": ["WebFetch", "WebSearch", "mcp__*"],
    "codex": ["web_search", "web_search_call"],
    "ctxagent": ["web_*", "browser_*", "fetch_url", "scholarly_search", "doi_lookup"],
}

# Network-reaching commands. Each pattern is searched in the command string.
NETWORK_PATTERNS = [
    r"\bcurl\b", r"\bwget\b", r"\bInvoke-WebRequest\b", r"\bInvoke-RestMethod\b", r"\biwr\b", r"\birm\b",
    r"\bpip3?\s+install\b", r"\bpython3?\s+-m\s+pip\s+install\b", r"\buv\s+pip\s+install\b",
    r"\bnpm\s+(?:install|i|ci)\b", r"\bnpx\b", r"\byarn\s+add\b", r"\bpnpm\s+(?:install|add)\b",
    r"\bgit\s+clone\s+(?:https?|git|ssh)://", r"\bgit\s+clone\s+\S+@", r"\bgit\s+(?:fetch|pull|push)\b",
    r"\brequests\.(?:get|post|put|head|request|Session)\b", r"\burllib\b", r"\bhttpx\.", r"\baiohttp\b",
    r"\bsocket\.create_connection\b", r"\bInvoke-Expression\s*\(\s*\(?New-Object\s+Net\.WebClient",
    r"\bNet\.WebClient\b", r"\bStart-BitsTransfer\b", r"\bapt(?:-get)?\s+install\b", r"\bconda\s+install\b",
    r"\bssh\s+\S+@", r"\bscp\b", r"\bnc\s+-", r"\btelnet\b", r"https?://(?!127\.0\.0\.1|localhost)",
]
_NET = [re.compile(p, re.I) for p in NETWORK_PATTERNS]


def _banned(name: str, harness: str) -> bool:
    for b in BANNED_TOOLS.get(harness, []):
        if b.endswith("*") and name.startswith(b[:-1]) or name == b:
            return True
    return False


def network_hits(command: str) -> list[str]:
    """Matched fragments (with a little context) of network-reaching commands in `command`."""
    out = []
    for rx in _NET:
        for m in rx.finditer(command or ""):
            a, b = max(0, m.start() - 20), min(len(command), m.end() + 60)
            out.append(command[a:b].replace("\n", " "))
    return out


def _hit(where: str, tool: str, text: str) -> dict:
    return {"where": where, "tool": tool, "hit": (text or "")[:300]}


# ---------------------------------------------------------------- Claude Code
def scan_claude_stream(path: Path) -> list[dict]:
    """claude_stream.jsonl: every assistant message's tool_use blocks (main agent and the
    Task sub-agents, which stream into the same file)."""
    hits = []
    for n, line in enumerate(_lines(path), 1):
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            continue
        if j.get("type") != "assistant":
            continue
        for c in (j.get("message") or {}).get("content") or []:
            if c.get("type") != "tool_use":
                continue
            name, inp = c.get("name") or "", c.get("input") or {}
            where = f"claude_stream.jsonl:{n}"
            if _banned(name, "claude_code"):
                hits.append(_hit(where, name, json.dumps(inp, ensure_ascii=False)))
            cmd = inp.get("command") if isinstance(inp, dict) else None
            if name in ("Bash", "PowerShell") or isinstance(cmd, str):
                for frag in network_hits(str(cmd or "")):
                    hits.append(_hit(where, name, frag))
            if name == "Task":  # sub-agent brief: check the prompt text for explicit network asks
                for frag in network_hits(str(inp.get("prompt") or "")):
                    hits.append(_hit(where, "Task(prompt)", frag))
    return hits


# ---------------------------------------------------------------------- Codex
def scan_codex_stream(path: Path) -> list[dict]:
    """codex_stream.jsonl (`codex exec --json`): item.started/completed items of type
    command_execution carry `command`; a web search would appear as its own item type."""
    hits = []
    for n, line in enumerate(_lines(path), 1):
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = j.get("item") or {}
        t = item.get("type") or ""
        where = f"codex_stream.jsonl:{n}"
        if j.get("type") != "item.completed" and t != "command_execution":
            continue
        if _banned(t, "codex") or "web_search" in t:
            hits.append(_hit(where, t, json.dumps(item, ensure_ascii=False)))
        if t == "command_execution" and j.get("type") == "item.completed":
            for frag in network_hits(str(item.get("command") or "")):
                hits.append(_hit(where, "command_execution", frag))
    return hits


# ------------------------------------------------------------------- ctxagent
def scan_ctxagent_db(path: Path) -> list[dict]:
    """ctxagent.db: every chat in the trial's store (the main chat and any agent_* worker
    chats), tool-call segments by function name and arguments. The archive of compacted
    segments is scanned too."""
    hits = []
    if not Path(path).exists():
        return hits
    con = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT id, title, conversation FROM chats").fetchall()
    finally:
        con.close()
    for cid, title, conv in rows:
        try:
            d = json.loads(conv)
        except (json.JSONDecodeError, TypeError):
            continue
        segs = list(d.get("segments") or [])
        for arch in (d.get("archive") or {}).values() if isinstance(d.get("archive"), dict) else (d.get("archive") or []):
            if isinstance(arch, dict):
                segs.append(arch)
        for s in segs:
            for tc in s.get("tool_calls") or []:
                fn = tc.get("function") or {}
                name, args = fn.get("name") or "", fn.get("arguments") or ""
                where = f"ctxagent.db chat {cid} ({title}) seg {s.get('id')}"
                if _banned(name, "ctxagent"):
                    hits.append(_hit(where, name, str(args)))
                if name in ("run_command", "start_job"):
                    try:
                        cmd = (json.loads(args) if isinstance(args, str) else args).get("command") or args
                    except (json.JSONDecodeError, AttributeError):
                        cmd = args
                    for frag in network_hits(str(cmd)):
                        hits.append(_hit(where, name, frag))
    return hits


def _lines(path: Path):
    p = Path(path)
    if not p.exists():
        return []
    return p.read_text(encoding="utf-8", errors="replace").splitlines()


def scan_trial(tdir: Path, harness: str) -> list[dict]:
    """All hits for one trial directory (bench/.work/<tag>)."""
    tdir = Path(tdir)
    if harness == "claude_code":
        return scan_claude_stream(tdir / "claude_stream.jsonl")
    if harness == "codex":
        return scan_codex_stream(tdir / "codex_stream.jsonl")
    return scan_ctxagent_db(tdir / "ctxdata" / "ctxagent.db")


def guess_harness(tdir: Path) -> str:
    tdir = Path(tdir)
    if (tdir / "claude_stream.jsonl").exists():
        return "claude_code"
    if (tdir / "codex_stream.jsonl").exists():
        return "codex"
    return "ctxagent"


if __name__ == "__main__":
    d = Path(sys.argv[1])
    h = sys.argv[2] if len(sys.argv) > 2 else guess_harness(d)
    res = scan_trial(d, h)
    print(json.dumps({"dir": str(d), "harness": h, "hits": len(res)}))
    for r in res:
        print(" ", r["where"], r["tool"], "|", r["hit"][:160])
