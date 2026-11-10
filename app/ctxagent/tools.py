"""Tool registry and implementations.

Every tool returns a ToolResult carrying, besides the text the model sees:

  resource_key -- a normalised identity for the thing observed ("file:c:/a/b.py").
                  Two results with the same key observe the same object; the
                  context panel shows it.
  mutates      -- whether this call *changed* that resource (a successful change
                  clears the memory of identical failed calls).

Access is confined to the session's configured working directories.
"""

from __future__ import annotations

import asyncio
import contextvars
import fnmatch
import hashlib
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx

from . import readers
from .observations import observation, version
from .readers import fold as foldmod, folder, llm as readers_llm

MAX_TOOL_CHARS = 200_000


#: How much of a resource an observation captured. Supersession may only
#: collapse an older observation when a later one is at least as detailed --
#: otherwise a level-0 outline would "supersede", and destroy, a full read of
#: the same file. Full reads sit above every outline level by construction.
DETAIL_FULL = 100

#: Observation channels. `detail` is only comparable WITHIN a channel:
#: a structural outline and an LLM summary of the same file say different
#: kinds of thing, so neither may supersede the other however deep it is.
CH_RAW = "raw"            # full content
CH_OUTLINE = "outline"    # deterministic structure
CH_SEMANTIC = "semantic"  # model-generated meaning


@dataclass
class ToolResult:
    text: str
    resource_key: str | None = None
    mutates: bool = False
    error: bool = False
    detail: int = DETAIL_FULL
    channel: str = CH_RAW
    meta: dict[str, Any] = field(default_factory=dict)
    observation: dict[str, Any] = field(default_factory=dict)
    images: list[dict[str, Any]] = field(default_factory=list)
    pinned: bool = False


class ToolError(Exception):
    pass


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------

def _norm(p: str | Path) -> str:
    return os.path.normcase(str(Path(p).resolve())).replace("\\", "/")


class Sandbox:
    def __init__(self, roots: list[str]):
        live = [r for r in roots if r and Path(r).exists()]
        # Comparison uses normcase (Windows paths are case-insensitive), but
        # display and joining use the original casing -- the model sees these
        # paths and echoes them back, so they should look like the real thing.
        self.display_roots = [str(Path(r).resolve()) for r in live]
        self.roots = [_norm(r) for r in self.display_roots]

    def resolve(self, path: str, *, must_exist: bool = False) -> Path:
        if not self.roots:
            raise ToolError(
                "No working directory is configured for this chat. "
                "Add one in the sidebar before using file tools."
            )
        p = Path(path)
        if not p.is_absolute():
            p = Path(self.display_roots[0]) / p
        full = _norm(p)
        ok = any(full == r or full.startswith(r.rstrip("/") + "/") for r in self.roots)
        if not ok:
            raise ToolError(
                f"Path is outside the allowed working directories: {path}\n"
                f"Allowed: {', '.join(self.roots)}"
            )
        rp = Path(str(Path(p).resolve()))
        if must_exist and not rp.exists():
            raise ToolError(f"Not found: {rp}")
        return rp

    def key(self, path: Path) -> str:
        return f"file:{_norm(path)}"


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------

async def t_read_file(sb: Sandbox, path: str, offset: int = 1, limit: int = 2000) -> ToolResult:
    p = sb.resolve(path, must_exist=True)
    if p.is_dir():
        raise ToolError(f"{p} is a directory; use list_dir.")
    try:
        raw = p.read_bytes()
        text = raw.decode("utf-8", errors="replace")
    except Exception as e:
        raise ToolError(f"Could not read {p}: {e}")
    lines = text.splitlines()
    offset = max(1, int(offset))
    limit = max(1, int(limit))
    chunk = lines[offset - 1 : offset - 1 + limit]
    numbered = "\n".join(f"{i + offset:6d}\t{ln}" for i, ln in enumerate(chunk))
    header = f"{p}  ({len(lines)} lines"
    if offset > 1 or offset - 1 + limit < len(lines):
        header += f", showing {offset}-{min(len(lines), offset - 1 + limit)}"
    header += ")"
    return ToolResult(
        text=f"{header}\n{numbered}"[:MAX_TOOL_CHARS],
        resource_key=sb.key(p),
        meta={"lines": len(lines)},
        observation={"version": hashlib.sha256(raw).hexdigest(), "representation": "source-lines",
            "coverage": {"kind": "lines", "start": offset, "end": offset + len(chunk) - 1},
            "complete": len(header) + len(numbered) + 1 <= MAX_TOOL_CHARS},
    )


def _detect_newline(raw: bytes) -> str:
    """The file's line ending: CRLF only if every LF in it is part of a CRLF."""
    lf = raw.count(b"\n")
    return "\r\n" if lf and raw.count(b"\r\n") == lf else "\n"


def _write_preserving(p: Path, text: str, newline: str, bom: bool = False) -> None:
    """Write `text` (LF-normalised in memory) with an explicit line ending.

    Path.write_text on Windows translates every '\\n' to CRLF, which silently
    converted LF repositories to CRLF on each edit (and broke tests that search
    source text for '\\n}\\n'). Bytes are written exactly; nothing is translated.
    """
    body = text.replace("\r\n", "\n")
    if newline != "\n":
        body = body.replace("\n", newline)
    raw = body.encode("utf-8")
    if bom:
        raw = b"\xef\xbb\xbf" + raw
    with open(p, "wb") as fh:
        fh.write(raw)


async def t_write_file(sb: Sandbox, path: str, content: str) -> ToolResult:
    p = sb.resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existed = p.exists()
    # Keep an existing file's line ending (and BOM); a new file gets what the
    # content itself uses (LF unless the content is CRLF throughout).
    if existed:
        old = p.read_bytes()
        nl, bom = _detect_newline(old), old.startswith(b"\xef\xbb\xbf")
    else:
        nl, bom = _detect_newline(content.encode("utf-8")), False
    _write_preserving(p, content, nl, bom)
    n = content.count("\n") + 1
    return ToolResult(
        text=f"{'Overwrote' if existed else 'Created'} {p} ({n} lines, {len(content)} chars).",
        resource_key=sb.key(p),
        mutates=True,
    )


async def t_edit_file(
    sb: Sandbox, path: str, old_string: str, new_string: str, replace_all: bool = False
) -> ToolResult:
    p = sb.resolve(path, must_exist=True)
    if not old_string:
        raise ToolError("old_string must not be empty.")
    raw = p.read_bytes()
    newline = _detect_newline(raw)
    bom = raw.startswith(b"\xef\xbb\xbf")
    try:
        text = raw.decode("utf-8-sig" if bom else "utf-8", errors="strict")
    except UnicodeDecodeError as e:
        raise ToolError(f"{p} is not valid UTF-8 ({e}); edit it another way.")
    # Match and replace in LF space whatever the file uses on disk, then write back with
    # the file's own line ending (see _write_preserving).
    text = text.replace("\r\n", "\n")
    old_string = old_string.replace("\r\n", "\n")
    new_string = new_string.replace("\r\n", "\n")
    count = text.count(old_string)
    if count == 0:
        raise ToolError(
            f"old_string not found in {p}. Read the file again — it may have changed."
        )
    if count > 1 and not replace_all:
        raise ToolError(
            f"old_string occurs {count} times in {p}. Make it unique or set replace_all=true."
        )
    new_text = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
    _write_preserving(p, new_text, newline, bom)
    return ToolResult(
        text=f"Edited {p}: {count if replace_all else 1} replacement(s).",
        resource_key=sb.key(p),
        mutates=True,
    )


async def t_list_dir(sb: Sandbox, path: str = ".") -> ToolResult:
    p = sb.resolve(path, must_exist=True)
    if not p.is_dir():
        raise ToolError(f"{p} is not a directory.")
    rows = []
    for entry in sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())):
        if entry.name.startswith(".") and entry.name not in (".env", ".gitignore"):
            continue
        if entry.is_dir():
            rows.append(f"  {entry.name}/")
        else:
            try:
                size = entry.stat().st_size
            except OSError:
                size = 0
            rows.append(f"  {entry.name}  ({size} B)")
    return ToolResult(
        text=f"{p}\n" + ("\n".join(rows) if rows else "  (empty)"),
        resource_key=f"dir:{_norm(p)}",
    )


_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", "dist", "build", ".ctxagent", ".ctxagent-build"}


def _walk(root: Path, max_files: int = 20000, *, include_dirs: bool = False):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")
                      and not (Path(dirpath) / d).is_symlink()
                      and not getattr(Path(dirpath) / d, "is_junction", lambda: False)()]
        if include_dirs:
            for dn in dirnames:
                yield Path(dirpath) / dn
        for fn in filenames:
            if (Path(dirpath) / fn).is_symlink():
                continue
            yield Path(dirpath) / fn
            max_files -= 1
            if max_files <= 0:
                return


async def t_glob_search(sb: Sandbox, pattern: str, path: str = ".") -> ToolResult:
    from . import search
    p = sb.resolve(path, must_exist=True)
    hits = []
    pat = pattern.replace("\\", "/")
    # Directories are matched too: a pattern like "**/readers" is asking where a
    # package lives, and returning nothing sends the agent on a grep hunt.
    fs = await asyncio.to_thread(search.list_files, p, include_dirs=True, use_git=False)
    gf = search.GlobFilter(pat, p)
    for f in fs.files:
        rel = str(f.relative_to(p)).replace("\\", "/")
        if gf(f) or fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch("/" + rel, pat):
            try:
                mtime = f.stat().st_mtime
            except OSError:
                mtime = 0.0
            hits.append((mtime, str(f) + ("/" if f.is_dir() else "")))
    hits.sort(reverse=True)
    body = "\n".join(h[1] for h in hits[:500]) or "(no matches)"
    more = f"\n… {len(hits) - 500} more" if len(hits) > 500 else ""
    notes = ""
    if fs.pruned_dirs:
        names = sorted(fs.pruned_dirs)
        notes += ("\n(not searched below root: " + ", ".join(names[:8]) + (" …" if len(names) > 8 else "")
                  + " — pass one as `path` to search it)")
    if fs.truncated:
        notes += f"\n⚠ INCOMPLETE file list ({fs.limit_note}): narrow `path`"
    return ToolResult(
        text=f"glob {pattern!r} in {p} — {len(hits)} match(es){notes}\n{body}{more}",
        resource_key=f"glob:{_norm(p)}:{pattern}",
    )


async def t_grep_search(
    sb: Sandbox, pattern: str, path: str = ".", glob: str = "*", max_results: int = 200,
    context: int = 0, case_insensitive: bool = False, literal: bool = False,
    mode: str = "content",
) -> ToolResult:
    from . import search
    p = sb.resolve(path, must_exist=True)
    if literal:
        pattern_rx = re.escape(pattern)
    else:
        pattern_rx = pattern
    try:
        rx = re.compile(pattern_rx, re.IGNORECASE if case_insensitive else 0)
    except re.error as e:
        raise ToolError(f"Bad regex: {e}. Set literal=true to search the text as-is.")
    mode = (mode or "content").lower()
    if mode not in ("content", "files", "count"):
        raise ToolError("mode must be 'content', 'files' or 'count'")
    max_results = max(1, min(5000, int(max_results)))
    context = max(0, min(20, int(context)))
    r = await asyncio.to_thread(
        search.grep, p, rx, glob=glob or "*", max_results=max_results, context=context,
        mode=mode, resolve=lambda f: sb.resolve(str(f), must_exist=True))
    text = search.format_grep(pattern, p, r, mode=mode, max_results=max_results)
    return ToolResult(
        text=text[:MAX_TOOL_CHARS],
        resource_key=f"grep:{_norm(p)}:{pattern}:{glob}:{mode}:{context}:{int(case_insensitive)}",
    )


#: run_command limits. Longer work belongs in start_job (it returns at once and survives
#: Stop); a model passing timeout=60000 (milliseconds, the habit from other harnesses)
#: held one recorded turn for the whole 3600 s turn cap.
RUN_COMMAND_DEFAULT_TIMEOUT = 120
RUN_COMMAND_MAX_TIMEOUT = 600
#: After the shell exits, how long to wait for its output pipes to close. A process the
#: command left running in the background (`server &`, Start-Process) keeps them open.
_PIPE_GRACE_S = 3.0
_MAX_CAPTURE_BYTES = 4 * MAX_TOOL_CHARS
#: The chat's Stop event (an asyncio.Event) while a tool batch runs: Stop kills a running
#: command instead of letting it run to its timeout.
RUN_CANCEL: "contextvars.ContextVar[asyncio.Event | None]" = contextvars.ContextVar("ctxagent_run_cancel", default=None)


def command_timeout(value) -> tuple[int, str]:
    """(seconds, note) for a model-supplied run_command timeout."""
    try:
        t = int(float(value))
    except (TypeError, ValueError):
        return RUN_COMMAND_DEFAULT_TIMEOUT, f"timeout {value!r} is not a number; used {RUN_COMMAND_DEFAULT_TIMEOUT} s"
    if t <= 0:
        return RUN_COMMAND_DEFAULT_TIMEOUT, ""
    note = ""
    if t > 3600 and t % 1000 == 0:
        note = f"timeout={t} read as milliseconds ({t // 1000} s); run_command timeouts are in seconds"
        t //= 1000
    if t > RUN_COMMAND_MAX_TIMEOUT:
        note = (note + "; " if note else "") + (
            f"timeout capped at {RUN_COMMAND_MAX_TIMEOUT} s; use start_job for longer work")
        t = RUN_COMMAND_MAX_TIMEOUT
    return t, note


def command_env() -> dict[str, str]:
    """Non-interactive environment: no pager, no credential prompt, unbuffered Python (so a
    timed-out command still shows its output). stdin is closed separately.

    Python writes to a pipe in the ANSI code page on Windows (cp1252): a script printing a
    character outside it (a check mark, an arrow) stopped with UnicodeEncodeError, and umlauts
    arrived garbled because the output is read as UTF-8. PYTHONIOENCODING makes Python's own
    output UTF-8, as start_job already does; a value the user set is kept. PYTHONUTF8 is not
    set: it would also change the default encoding of open()."""
    env = dict(os.environ)
    env.update({"PAGER": "cat", "GIT_PAGER": "cat", "GIT_TERMINAL_PROMPT": "0",
                "DEBIAN_FRONTEND": "noninteractive", "PYTHONUNBUFFERED": "1"})
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


class _Drain:
    """Read a pipe to EOF on a daemon thread, keeping at most `_MAX_CAPTURE_BYTES`."""

    def __init__(self, pipe):
        import threading
        self.chunks: list[bytes] = []
        self.size = self.dropped = 0
        self.keep = True
        self.thread = threading.Thread(target=self._run, args=(pipe,), daemon=True)
        self.thread.start()

    def _run(self, pipe):
        try:
            for chunk in iter(lambda: pipe.read1(65536) if hasattr(pipe, "read1") else pipe.read(65536), b""):
                if self.keep and self.size < _MAX_CAPTURE_BYTES:
                    self.chunks.append(chunk)
                    self.size += len(chunk)
                else:
                    self.dropped += len(chunk)
        except (OSError, ValueError):
            pass
        finally:
            try:
                pipe.close()
            except OSError:
                pass

    def text(self) -> str:
        self.keep = False  # anything read later is discarded
        return b"".join(self.chunks).decode("utf-8", errors="replace")


def _kill_tree(proc) -> None:
    """Kill the command and everything it started (descendants first, collected while the
    shell is alive: on Windows orphans cannot be found from a dead parent)."""
    try:
        import psutil
        try:
            root = psutil.Process(proc.pid)
            victims = list(reversed(root.children(recursive=True))) + [root]
        except psutil.NoSuchProcess:
            victims = []
        for p in victims:
            try:
                p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        psutil.wait_procs(victims, timeout=3)
    except ImportError:
        pass
    if os.name != "nt":
        import signal
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # the command's own session (start_new_session)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        proc.kill()
    except OSError:
        pass


def _run_shell(args, cwd: str, timeout: int, stop) -> dict:
    """Run without inheriting stdin; kill the process tree on timeout or `stop`; never wait
    unboundedly for output pipes. subprocess.run(timeout=) killed only the shell and then
    waited for EOF again, which never came while a grandchild held the pipe: one recorded
    command 'timed out after 60s' and returned after 3753 s."""
    kw: dict[str, Any] = dict(cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, env=command_env())
    if os.name == "nt":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    import time as _time
    t0 = _time.monotonic()
    proc = subprocess.Popen(args, **kw)
    out, err = _Drain(proc.stdout), _Drain(proc.stderr)
    reason = None
    while True:
        try:
            rc = proc.wait(timeout=0.2)
            break
        except subprocess.TimeoutExpired:
            if stop.is_set():
                reason = "stopped"
            elif _time.monotonic() - t0 >= timeout:
                reason = "timeout"
            if reason:
                _kill_tree(proc)
                try:
                    rc = proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    rc = None
                break
    deadline = _time.monotonic() + _PIPE_GRACE_S
    for d in (out, err):
        d.thread.join(max(0.0, deadline - _time.monotonic()))
    lingering = out.thread.is_alive() or err.thread.is_alive()
    return {"rc": rc, "stdout": out.text(), "stderr": err.text(), "reason": reason, "lingering": lingering,
            "dropped": out.dropped + err.dropped, "seconds": _time.monotonic() - t0}


async def t_run_command(
    sb: Sandbox, command: str, cwd: str = ".", timeout: int = RUN_COMMAND_DEFAULT_TIMEOUT
) -> ToolResult:
    import threading
    wd = sb.resolve(cwd, must_exist=True)
    if not wd.is_dir():
        wd = wd.parent
    if not isinstance(command, str) or not command.strip():
        raise ToolError("command must be a non-empty string")
    timeout, timeout_note = command_timeout(timeout)
    if sys.platform == "win32":
        shell_exe = shutil.which("pwsh") or shutil.which("powershell") or "powershell"
        args = [shell_exe, "-NoProfile", "-NonInteractive", "-Command", command]
    else:
        args = ["/bin/bash", "-lc", command]

    stop = threading.Event()
    fut = asyncio.get_running_loop().run_in_executor(None, _run_shell, args, str(wd), timeout, stop)
    cancel = RUN_CANCEL.get()  # the chat's Stop event, set by the tool batch
    watcher = None
    if cancel is not None:
        async def watch():
            await cancel.wait()
            stop.set()
        watcher = asyncio.create_task(watch())
    try:
        r = await asyncio.shield(fut)
    except asyncio.CancelledError:
        stop.set()  # disconnect: kill the command instead of leaving it running
        raise
    finally:
        if watcher is not None:
            watcher.cancel()
    rc, so, se = r["rc"], r["stdout"], r["stderr"]
    # Arguments already contain the exact command. Do not echo a multi-kB
    # inline script into context a second time merely to label its output.
    preview = command if len(command) <= 240 else command[:180] + '\n[command preview shortened; full command is in the tool call]'
    parts = []
    if r["reason"] == "timeout":
        parts.append(f"Error: Command timed out after {timeout}s and was stopped (its process tree was killed). "
                     "Output up to that point follows. For builds, test suites or servers that need longer, "
                     "use start_job (returns at once; job_wait / read_job_log follow it).")
    elif r["reason"] == "stopped":
        parts.append("Command stopped by the user (its process tree was killed).")
    parts.append(f"$ {preview}   (cwd={wd}, exit={rc if rc is not None else 'killed'})")
    if timeout_note:
        parts.append(f"[{timeout_note}]")
    if so.strip():
        parts.append(so.rstrip())
    if se.strip():
        parts.append("--- stderr ---\n" + se.rstrip())
    if not so.strip() and not se.strip():
        parts.append("(no output)")
    if r["dropped"]:
        parts.append(f"[{r['dropped']} more bytes of output were not kept]")
    if r["lingering"] and not r["reason"]:
        parts.append("[note: a process started by this command is still running and holds its output; "
                     "later output is not captured. Use start_job for servers and other long-running processes.]")
    return ToolResult(
        text="\n".join(parts)[:MAX_TOOL_CHARS],
        # A shell command can change anything; key it to the cwd and the command,
        # but do not claim it changed a particular resource.
        resource_key=f"cmd:{_norm(wd)}:{command.strip()[:120]}",
        mutates=False,
        error=rc != 0 or bool(r["reason"]),
        meta={"exit": rc, "timed_out": r["reason"] == "timeout"},
    )


async def t_fetch_url(sb: Sandbox, url: str, max_chars: int = 40000) -> ToolResult:
    from . import browsing
    if not 1 <= max_chars <= 40000:
        raise ToolError('max_chars must be 1..40000; use web_open for paged browsing')
    receipt = await browsing._get(url)
    data = await asyncio.to_thread(browsing._extract, receipt['raw'], receipt['url'], receipt['content_type'], receipt['encoding'])
    suffix = '\n[More text available: use web_open for a cached paged snapshot.]' if len(data['text']) > max_chars else ''
    return ToolResult(text=f"{receipt['url']}\n\n{data['text'][:max_chars]}" + suffix,
                      resource_key='url:' + receipt['url'])


# recall is special: it needs the conversation, so the agent injects it.
async def t_recall(sb: Sandbox, segment_id: str) -> ToolResult:  # pragma: no cover
    raise ToolError("recall is handled by the agent loop")


# compact_context too (agent-directed compaction; compaction/manual.py).
# (prepare_arguments checks a call against this signature: it names what the handler takes.)
async def t_compact_context(sb: Sandbox, summary: str = "", until=None) -> ToolResult:  # pragma: no cover
    raise ToolError("compact_context is handled by the agent loop")


# curate_context likewise (the agent replaces items by its own text; compaction/curate.py).
async def t_curate_context(sb: Sandbox, pairs=None, **other) -> ToolResult:  # pragma: no cover
    raise ToolError("curate_context is handled by the agent loop")


# pin likewise operates on the conversation.
async def t_pin(sb: Sandbox, segment_ids=None, batches: int = 0, unpin: bool = False,
                list_pins: bool = False) -> ToolResult:  # pragma: no cover
    raise ToolError("pin is handled by the agent loop")


# ---------------------------------------------------------------------------
# Document reading and summarisation
# ---------------------------------------------------------------------------

async def t_read_document(
    sb: Sandbox, path: str, offset: int = 1, limit: int = 3000
) -> ToolResult:
    p = sb.resolve(path, must_exist=True)
    if p.is_dir():
        raise ToolError(f"{p} is a directory; use list_dir or summarize_folder.")
    source_version = version(p)
    kind = readers.detect_kind(p)
    try:
        text = readers.extract_text(p, kind)
    except Exception as e:
        raise ToolError(f"Could not extract text from {p} ({kind}): {type(e).__name__}: {e}")
    lines = text.splitlines()
    offset = max(1, int(offset))
    limit = max(1, int(limit))
    chunk = lines[offset - 1: offset - 1 + limit]
    shown = f", showing lines {offset}-{min(len(lines), offset - 1 + limit)}" \
        if (offset > 1 or offset - 1 + limit < len(lines)) else ""
    header = f"{p}  [{kind}, {len(lines)} lines extracted{shown}]"
    body = "\n".join(chunk)
    if offset - 1 + limit < len(lines):
        body += (f"\n\n[{len(lines) - (offset - 1 + limit)} more lines — "
                 f"call again with offset={offset + limit}, or use "
                 f"summarize_file for an overview]")
    return ToolResult(
        text=f"{header}\n{body}"[:MAX_TOOL_CHARS],
        resource_key=sb.key(p),
        detail=DETAIL_FULL,
        meta={"kind": kind, "lines": len(lines)},
        observation=observation(p, "extracted-lines",
            {"kind": "lines", "start": offset, "end": offset + len(chunk) - 1},
            complete=len(header) + len(body) + 1 <= MAX_TOOL_CHARS, source_version=source_version),
    )


async def t_summarize_file(
    sb: Sandbox, path: str, level: int = 2, mode: str = "static", ctx=None
) -> ToolResult:
    p = sb.resolve(path, must_exist=True)
    if p.is_dir():
        raise ToolError(f"{p} is a directory; use summarize_folder.")
    level = max(0, min(readers.MAX_LEVEL, int(level)))
    mode = (mode or "static").lower().strip()
    source_version = version(p)
    kind = readers.detect_kind(p)

    if mode in ("static", "structure", "outline"):
        try:
            text = readers.outline(p, level, kind)
        except Exception as e:
            raise ToolError(f"Could not outline {p} ({kind}): {type(e).__name__}: {e}")
        return ToolResult(
            text=text[:MAX_TOOL_CHARS],
            resource_key=sb.key(p),
            detail=level,
            channel=CH_OUTLINE,
            meta={"kind": kind, "level": level, "mode": "static"},
            observation=observation(p, "static-outline", {"kind": "whole"},
                                    complete=len(text) <= MAX_TOOL_CHARS, source_version=source_version),
        )

    if mode in ("llm", "semantic", "generated"):
        if ctx is None or not ctx.get("complete_fn"):
            raise ToolError("LLM summaries are unavailable in this context.")
        try:
            text, meta = await readers_llm.summarize(
                p, level, ctx.get("summary_model"), ctx["complete_fn"], kind
            )
        except Exception as e:
            raise ToolError(f"LLM summary failed for {p}: {type(e).__name__}: {e}")
        note = (f"[LLM summary of {p.name} — kind={meta['kind']}, level={meta['level']}, "
                f"model={ctx.get('summary_model')}"
                + (", source truncated" if meta["truncated"] else "") + "]")
        return ToolResult(
            text=f"{note}\n{text}"[:MAX_TOOL_CHARS],
            resource_key=sb.key(p),
            detail=min(level, 8),
            channel=CH_SEMANTIC,
            meta={**meta, "mode": "llm"},
        )

    raise ToolError(f"mode must be 'static' or 'llm', got {mode!r}")


async def t_focus_file(
    sb: Sandbox, path: str, select, level: int = 1, max_tokens: int = 24000
) -> ToolResult:
    p = sb.resolve(path, must_exist=True)
    if p.is_dir():
        raise ToolError(f"{p} is a directory; use summarize_folder.")
    if isinstance(select, str):
        selectors = [select]
    elif isinstance(select, (list, tuple)):
        selectors = [str(x) for x in select if str(x).strip()]
    else:
        raise ToolError("select must be a string or a list of strings.")
    if not selectors:
        raise ToolError("select must name at least one part of the document.")
    source_version = version(p)
    kind = readers.detect_kind(p)
    try:
        text = foldmod.fold(p, selectors, level=max(0, min(4, int(level))),
                            kind=kind, max_tokens=max(500, int(max_tokens)))
    except Exception as e:
        raise ToolError(f"Could not fold {p}: {type(e).__name__}: {e}")
    return ToolResult(
        text=text[:MAX_TOOL_CHARS],
        resource_key=sb.key(p),
        # A folded view shows part of the file in full but not the rest, so it
        # must never supersede a full read -- nor be superseded by one, since it
        # carries structure a flat read does not.
        detail=level,
        channel=CH_OUTLINE,
        meta={"kind": kind, "selectors": selectors, "mode": "fold"},
        observation=observation(p, "fold", {"kind": "selectors", "values": selectors},
            complete=len(text) <= MAX_TOOL_CHARS and "NO MATCH" not in text
                     and "TRUNCATED" not in text and "nodes unfolded;" in text,
            source_version=source_version),
    )


async def t_summarize_folder(
    sb: Sandbox, path: str = ".", level: int = 2, token_budget: int = 0
) -> ToolResult:
    p = sb.resolve(path, must_exist=True)
    if not p.is_dir():
        raise ToolError(f"{p} is not a directory; use summarize_file.")
    level = max(0, min(readers.MAX_LEVEL + 1, int(level)))
    budget = int(token_budget) or folder.default_budget(level)
    try:
        text = folder.overview(p, level, token_budget=max(1000, budget))
    except Exception as e:
        raise ToolError(f"Could not survey {p}: {type(e).__name__}: {e}")
    return ToolResult(
        text=text[:MAX_TOOL_CHARS],
        resource_key=f"folder:{_norm(p)}:L{level}",
        detail=level,
        channel=CH_OUTLINE,
        meta={"level": level},
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]
    group: str = "general"
    dangerous: bool = False
    #: receives a `ctx` kwarg carrying {complete_fn, summary_model} -- for tools
    #: that need to call a model themselves (the LLM summariser).
    needs_ctx: bool = False


def _p(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


REGISTRY: dict[str, ToolSpec] = {}

#: Shared description of what `level` means per format. Kept in one place so the
#: tool schema the model sees always matches what the readers actually do.
LEVELS_DOC = (
    "mode='static' (default, no model call): level 0-5 is structural depth, 0 = barest "
    "skeleton, 5 = full text. Code: class names, function names, signatures, docstrings. "
    "Markdown/rst/html: one more heading depth per level. Latex: sections, then "
    "figures/tables/equation labels, then first sentences. Tables: dtypes, column "
    "statistics, sample rows. PDF: bookmarks, then first lines per page. JSON/YAML/XML: "
    "one more nesting depth per level. "
    "mode='llm': level is the token budget of a generated summary: "
    "0=100, 1=300, 2=1000, 3=3000, 4=10000, 5=uncapped."
)


def _reg(spec: ToolSpec) -> None:
    REGISTRY[spec.name] = spec


_reg(ToolSpec(
    "read_file",
    "Read a text file with line numbers.",
    _p({
        "path": {"type": "string", "description": "Absolute, or relative to the working directory."},
        "offset": {"type": "integer", "description": "First line, 1-based."},
        "limit": {"type": "integer", "description": "Lines to read. Default 2000."},
    }, ["path"]),
    t_read_file, "files",
))

_reg(ToolSpec(
    "write_file",
    "Create or overwrite a file with the given content.",
    _p({
        "path": {"type": "string"},
        "content": {"type": "string"},
    }, ["path", "content"]),
    t_write_file, "files", dangerous=True,
))

_reg(ToolSpec(
    "edit_file",
    "Replace an exact string in a file. Fails if old_string occurs more than once, unless replace_all.",
    _p({
        "path": {"type": "string"},
        "old_string": {"type": "string"},
        "new_string": {"type": "string"},
        "replace_all": {"type": "boolean"},
    }, ["path", "old_string", "new_string"]),
    t_edit_file, "files", dangerous=True,
))

_reg(ToolSpec(
    "list_dir",
    "List the entries of a directory.",
    _p({"path": {"type": "string"}}, []),
    t_list_dir, "files",
))

_reg(ToolSpec(
    "glob_search",
    "Find files by glob pattern (e.g. '**/*.py'), newest first.",
    _p({"pattern": {"type": "string"}, "path": {"type": "string"}}, ["pattern"]),
    t_glob_search, "search",
))

_reg(ToolSpec(
    "grep_search",
    "Search file contents with a Python regex; returns path:line: match. In a git repo the "
    "file list follows .gitignore, and dot-dirs below the root are skipped unless given as "
    "path. The header states what was searched and whether the result is incomplete.",
    _p({
        "pattern": {"type": "string"},
        "path": {"type": "string", "description": "File or directory. Default: the working directory."},
        "glob": {"type": "string", "description": "'*.py', 'src/**/*.cpp', comma-separated, '!x' excludes."},
        "max_results": {"type": "integer", "description": "Default 200."},
        "context": {"type": "integer", "description": "Lines around each match, 0-20."},
        "case_insensitive": {"type": "boolean"},
        "literal": {"type": "boolean", "description": "Pattern is plain text."},
        "mode": {"type": "string", "enum": ["content", "files", "count"],
                 "description": "Matching lines (default), paths only, or hits per file."},
    }, ["pattern"]),
    t_grep_search, "search",
))

_reg(ToolSpec(
    "run_command",
    "Run a shell command (PowerShell on Windows, bash elsewhere) in a working directory. "
    "Non-interactive: stdin is closed, so a command waiting for input fails instead of hanging. "
    "On timeout the command and everything it started are killed and the output so far is "
    "returned. Servers, watchers and anything longer than a few minutes: use start_job.",
    _p({
        "command": {"type": "string"},
        "cwd": {"type": "string"},
        "timeout": {"type": "integer", "description": f"Seconds (not milliseconds). Default "
                    f"{RUN_COMMAND_DEFAULT_TIMEOUT}, max {RUN_COMMAND_MAX_TIMEOUT}."},
    }, ["command"]),
    t_run_command, "shell", dangerous=True,
))

_reg(ToolSpec(
    "fetch_url",
    "Fetch a web page and return its text content.",
    _p({"url": {"type": "string"}, "max_chars": {"type": "integer"}}, ["url"]),
    t_fetch_url, "web",
))

_reg(ToolSpec(
    "recall",
    "Return the original content behind a segment_id (a shortened, replaced or compacted "
    "item) as a new result, which automatic compaction then leaves alone.",
    _p({"segment_id": {"type": "string"}}, ["segment_id"]),
    t_recall, "context",
))

_reg(ToolSpec(
    "pin",
    "Keep tool results verbatim: a pinned result survives deep compaction. Pass the segment ids of earlier tool "
    "results ('last' = the previous tool batch's results). batches>0 makes the pin "
    "expire after that many further tool batches; 0 = until unpinned. A shortened "
    "result is restored to its full original when pinned. unpin=true releases; "
    "list_pins=true lists current pins with their token cost. Pin sparingly: pins "
    "count against the context window.",
    _p({
        "segment_ids": {"type": "array", "items": {"type": "string"},
                        "description": "Ids from markers or context_report; 'last' = results of the previous batch."},
        "batches": {"type": "integer", "description": "Pin expires after this many tool batches; 0 = stays."},
        "unpin": {"type": "boolean"},
        "list_pins": {"type": "boolean"},
    }, []),
    t_pin, "context",
))

_reg(ToolSpec(
    "compact_context",
    "Replace the history up to an item by your summary. `until` counts back like a "
    "curate_context reference: 1 is your newest tool result, tool call or answer before this "
    "call. A tool batch is not split: a cut inside one moves before it. The summary is followed "
    "by an index of the replaced tool results with their recall ids.",
    _p({
        "summary": {"type": "string"},
        "until": {"type": "integer", "minimum": 1,
                  "description": "Last item replaced. Default: everything before the newest user message."},
    }, ["summary"]),
    t_compact_context, "context",
))

_reg(ToolSpec(
    "curate_context",
    "curate_context(reference1, replacement1, reference2, replacement2, as many following as you "
    "like...). A reference counts back from the newest item before this call: 1 is your newest tool "
    "result, tool call or answer, 2 the one before, and so on. Each item is replaced by its "
    "replacement text; recall(segment_id) in the stub restores the original.",
    _p({
        "pairs": {"type": "array", "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                  "description": "reference1, replacement1, reference2, replacement2, ..."},
    }, []),
    t_curate_context, "context",
))

_reg(ToolSpec(
    "read_document",
    "Extract the readable text of a document: PDF, DOCX, PPTX, Jupyter notebook, "
    "CSV/TSV/Excel/Parquet, HTML, JSON/YAML/TOML/XML, plain text and source files.",
    _p({
        "path": {"type": "string"},
        "offset": {"type": "integer", "description": "First extracted line, 1-based."},
        "limit": {"type": "integer", "description": "Lines to return. Default 3000."},
    }, ["path"]),
    t_read_document, "documents",
))

_reg(ToolSpec(
    "summarize_file",
    "Outline or summary of one file: its real structure (static) or a small model's "
    "summary of the content (llm). " + LEVELS_DOC,
    _p({
        "path": {"type": "string"},
        "level": {"type": "integer", "minimum": 0, "maximum": 5, "description": "Default 2."},
        "mode": {"type": "string", "enum": ["static", "llm"]},
    }, ["path"]),
    t_summarize_file, "documents", needs_ctx=True,
))

_reg(ToolSpec(
    "focus_file",
    "Folded view of a file: the selected parts (function, section, key, column, page) in "
    "full inside a collapsed outline of the rest. A selector that matches nothing returns "
    "the addressable names.",
    _p({
        "path": {"type": "string"},
        "select": {
            "anyOf": [{"type": "string"},
                      {"type": "array", "items": {"type": "string"}}],
            "description": "Name (`run`), dotted path (`Engine.run`, `server.tls`), /regex/, "
                           "or line range L120-160 (any file).",
        },
        "level": {"type": "integer", "minimum": 0, "maximum": 4,
                  "description": "Detail of the collapsed lines: 0 labels, 1 adds line spans and sizes (default)."},
        "max_tokens": {"type": "integer", "description": "Cap on unfolded content. Default 24000."},
    }, ["path", "select"]),
    t_focus_file, "documents",
))

_reg(ToolSpec(
    "summarize_folder",
    "Survey a directory tree. level 0 is a plain listing; at level N each file gets its "
    "static outline at level N-1 and each subfolder is surveyed at level N-2.",
    _p({
        "path": {"type": "string"},
        "level": {"type": "integer", "minimum": 0, "maximum": 6, "description": "Default 2."},
        "token_budget": {"type": "integer", "description": "Cap on output size."},
    }, []),
    t_summarize_folder, "documents",
))


async def t_context_report(sb: Sandbox, top: int = 5, recent: int = 20, ctx=None) -> ToolResult:
    from .context_report import report
    import json
    session = ctx['session']
    return ToolResult(text=json.dumps(report(session.conv, session.cfg, session.usage,
                                            top=top, recent=recent), ensure_ascii=True))


_reg(ToolSpec(
    'context_report',
    'Context sizes of this chat, its largest tool schemas and segments (with ids), recorded cache hits and '
    'costs, inactive settings. Local token counts are estimates. Defaults: top=5, recent=20 requests.',
    _p({'top': {'type': 'integer', 'minimum': 1, 'maximum': 10},
        'recent': {'type': 'integer', 'minimum': 1, 'maximum': 100}}, []),
    t_context_report, 'context', needs_ctx=True,
))


DEFAULT_TOOLS = [
    "read_file", "write_file", "edit_file", "list_dir",
    "glob_search", "grep_search", "run_command", "recall", "pin",
    "read_document", "summarize_file", "summarize_folder", "focus_file",
    "view_image", "render_pdf", "start_job", "job_status", "job_wait", "list_jobs",
    "read_job_log", "cancel_job", "context_report",
    "document_open", "document_close", "document_list",
    "document_prepare_edits", "document_json_update", "document_table_update", "document_apply_edits",
]


#: Tool names models bring from other harnesses (recorded: `bash`, `bash_placeholder`).
_ALIASES = {
    "bash": "run_command", "shell": "run_command", "sh": "run_command", "exec": "run_command",
    "terminal": "run_command", "execute_command": "run_command", "run_shell": "run_command",
    "python": "run_command", "powershell": "run_command",
    "str_replace_editor": "edit_file", "str_replace_based_edit_tool": "edit_file", "str_replace": "edit_file",
    "edit": "edit_file", "write": "write_file", "create_file": "write_file",
    "read": "read_file", "view": "read_file", "cat": "read_file", "open_file": "read_file",
    "ls": "list_dir", "list_files": "list_dir", "grep": "grep_search", "search": "grep_search",
    "glob": "glob_search", "find": "glob_search", "find_files": "glob_search",
    "wait_job": "job_wait", "job_result": "job_status",
}


def unknown_tool_text(name: str, available: list[str], *, disabled: bool = False) -> str:
    """Error for a call to a tool that does not exist (or is not enabled), with the likely
    intended tool named, so the next call does not guess again."""
    import difflib
    base = re.sub(r"[^a-z0-9_]", "", str(name).lower().split(".")[-1].split(":")[-1]).removesuffix("_placeholder")
    guesses = []
    alias = _ALIASES.get(base)
    if alias and alias in available:
        guesses.append(alias)
    guesses += [g for g in difflib.get_close_matches(base, available, n=3, cutoff=0.6) if g not in guesses]
    head = (f"Tool {name!r} is not enabled for this chat." if disabled else f"Unknown tool: {name}.")
    hint = (f" Did you mean {', '.join(guesses)}?" if guesses else
            " Use only the tools listed in this request's tool definitions.")
    return head + hint + " Nothing was executed."


def _type_label(prop: dict) -> str:
    t = prop.get("type")
    if t:
        return t if isinstance(t, str) else "/".join(t)
    alts = prop.get("anyOf") or prop.get("oneOf") or []
    return "/".join(str(a.get("type", "?")) for a in alts) or "any"


def prepare_arguments(spec: ToolSpec, args: dict) -> tuple[dict, str | None]:
    """Check a call's arguments against the implementation before running it.

    Unknown or missing arguments used to surface as Python's TypeError text
    ("t_write_file() missing 1 required positional argument: 'content'"); the message now
    names the tool's actual parameters. Strings for boolean/integer/number parameters are
    converted ("false" was truthy: replace_all="false" meant replace all); anything that is
    not convertible is an error. Returns (arguments to use, error text or None)."""
    import inspect
    try:
        params = list(inspect.signature(spec.fn).parameters.values())[1:]  # [0] is the sandbox
    except (TypeError, ValueError):
        return args, None
    kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    names = {p.name for p in params if p.kind in kinds and p.name != "ctx"}
    var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
    props = (spec.parameters or {}).get("properties") or {}
    required = set((spec.parameters or {}).get("required") or [])
    unknown = [] if var_kw else [k for k in args if k not in names and k != "ctx"]
    missing = [p.name for p in params if p.kind in kinds and p.name != "ctx"
               and p.default is inspect.Parameter.empty and p.name not in args]
    problems = []
    if missing:
        problems.append("missing required argument" + ("s " if len(missing) > 1 else " ")
                        + ", ".join(repr(m) for m in missing))
    if unknown:
        problems.append("unknown argument" + ("s " if len(unknown) > 1 else " ")
                        + ", ".join(repr(u) for u in unknown))
    out = dict(args)
    for key, value in args.items():
        want = (props.get(key) or {}).get("type")
        if not isinstance(value, str) or want not in ("boolean", "integer", "number"):
            continue
        low = value.strip().lower()
        if want == "boolean":
            if low in ("true", "1", "yes"):
                out[key] = True
            elif low in ("false", "0", "no", ""):
                out[key] = False
            else:
                problems.append(f"argument {key!r} must be true or false, got {value!r}")
        else:
            try:
                number = float(low)
                out[key] = int(number) if want == "integer" and number.is_integer() else number
                if want == "integer" and not number.is_integer():
                    problems.append(f"argument {key!r} must be an integer, got {value!r}")
            except ValueError:
                problems.append(f"argument {key!r} must be a number, got {value!r}")
    if not problems:
        return out, None
    listing = ", ".join(f"{k} ({_type_label(v)}{', required' if k in required else ''})"
                        for k, v in props.items()) or "none"
    return args, (f"Bad arguments for {spec.name}: " + "; ".join(problems)
                  + f". Parameters: {listing}. Nothing was executed.")


def openai_schema(names: list[str]) -> list[dict[str, Any]]:
    out = []
    for n in names:
        spec = REGISTRY.get(n)
        if not spec:
            continue
        out.append({
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        })
    return out


def catalogue() -> list[dict[str, Any]]:
    """The tools a user switches one by one. compact_context and curate_context are not in
    it: they come and go with self-steered context management (config.SELF_STEERED_TOOLS).
    Neither is document_call: the catalog's dispatcher for document and canvas operations
    where the provider has no native deferred tools (OpenRouter). The app adds it whenever
    such an operation is on (AgentSession.tool_names)."""
    from .config import SELF_STEERED_TOOLS
    from .document_catalog import DISPATCHER
    return [
        {
            "name": s.name,
            "description": s.description,
            "group": s.group,
            "dangerous": s.dangerous,
            "needs_ctx": s.needs_ctx,
        }
        for s in REGISTRY.values()
        if s.name not in SELF_STEERED_TOOLS and s.name != DISPATCHER
    ]


# Adapters are separate; services do not import the registry.
from .productivity_tools import register as _register_productivity
_register_productivity(REGISTRY, ToolSpec, ToolResult, ToolError, _p)

from .workflow_tools import register as _register_workflows
_register_workflows(REGISTRY, ToolSpec, ToolResult, _p)

from .research_tools import register as _register_research
_register_research(REGISTRY, ToolSpec, ToolResult, _p)

from .workspace_tools import register as _register_workspace
_register_workspace()

from .document_format_tools import register as _register_document_formats
_register_document_formats(REGISTRY, ToolSpec, ToolResult, _p)
from .document_formats import NAMES as _document_operations
DEFAULT_TOOLS.extend(n for n in _document_operations if n not in DEFAULT_TOOLS)

from .agent_tools import register as _register_agents, AGENT_TOOLS, WEB_TOOLS
_register_agents(REGISTRY, ToolSpec, ToolResult, _p)
DEFAULT_TOOLS.extend(AGENT_TOOLS + WEB_TOOLS)

from .browser_tools import register as _register_browser, BROWSER_TOOLS
_register_browser(REGISTRY, ToolSpec, _p)
DEFAULT_TOOLS.extend(BROWSER_TOOLS)

from .collab_tools import register as _register_collab, SCRATCHPAD_TOOLS
_register_collab(REGISTRY, ToolSpec, ToolResult, ToolError, _p)
DEFAULT_TOOLS.extend(SCRATCHPAD_TOOLS)

from .canvas_tools import register as _register_canvas, CANVAS_TOOLS
_register_canvas(REGISTRY, ToolSpec, ToolResult, ToolError, _p)
DEFAULT_TOOLS.extend(CANVAS_TOOLS)

# A new chat has every tool the user can switch on (the list above keeps its order first).
DEFAULT_TOOLS.extend(t["name"] for t in catalogue() if t["name"] not in DEFAULT_TOOLS)
