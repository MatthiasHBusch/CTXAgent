"""File listing and content search for grep_search / glob_search.

Design rule: NEVER return a silent false negative. Every reason a file was not
searched (size, binary, unreadable, walk limit, time budget, skipped directory)
is counted and reported in the result header, so "(no matches)" means "searched
everything in scope and found nothing", or the header says exactly what was not
searched.

File set:
* Inside a git work tree, the candidate list comes from
  ``git ls-files --cached --others --exclude-standard`` (fast, honours
  .gitignore, never descends into submodules or vendored ignored trees).
* Otherwise (or if git fails) an os.walk that prunes well-known heavy or
  generated directories and dot-directories; pruned directory names are reported.
* An explicit ``path`` that points INTO a pruned/dot directory is searched
  normally: the pruning only applies below the search root.

Content scan streams files line by line, so there is no "too big to search"
cap (AGENT_QUEUE.md at 10 MB is searched like any other file); only binary
files (a NUL byte in the first 8 KiB) are skipped, and counted.
"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

# Pruned by the walker below the search root (git mode uses .gitignore instead).
PRUNE_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache",
    ".pytest_cache", ".tox", "dist", ".ctxagent", ".ctxagent-build", ".godot",
    ".idea", ".vs", ".gradle", "target", "site-packages",
}
BINARY_PROBE = 8192
MAX_LINE_CHARS = 300


@dataclass
class FileSet:
    files: list[Path]
    mode: str                          # "git" | "walk" | "file"
    pruned_dirs: set[str] = field(default_factory=set)
    truncated: bool = False            # walk/time limit hit: the list is incomplete
    limit_note: str = ""


def _is_link(p: Path) -> bool:
    try:
        return p.is_symlink() or bool(getattr(p, "is_junction", lambda: False)())
    except OSError:
        return True


def _git_root(path: Path) -> Path | None:
    try:
        r = subprocess.run(["git", "-C", str(path), "rev-parse", "--show-toplevel"],
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    return Path(r.stdout.strip())


def _git_files(root: Path) -> list[Path] | None:
    try:
        r = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    out = []
    for raw in r.stdout.split(b"\0"):
        if raw:
            out.append(root / raw.decode("utf-8", errors="replace"))
    return out


def list_files(root: Path, *, max_files: int = 200_000, time_budget: float = 30.0,
               use_git: bool = True, include_dirs: bool = False) -> FileSet:
    """Every candidate file below ``root`` (or ``root`` itself if it is a file)."""
    if root.is_file():
        return FileSet([root], "file")
    t0 = time.monotonic()
    if use_git and not include_dirs:
        groot = _git_root(root)
        if groot is not None:
            files = _git_files(groot)
            if files is not None:
                try:
                    rel_root = root.resolve().relative_to(groot.resolve())
                except ValueError:
                    rel_root = None
                if rel_root is not None:
                    base = groot / rel_root
                    prefix = os.path.normcase(os.path.abspath(str(base))).rstrip("\\/") + os.sep
                    sel = []
                    pruned: set[str] = set()
                    for f in files:
                        fs = str(f)
                        if rel_root != Path(".") and not os.path.normcase(os.path.abspath(fs)).startswith(prefix):
                            continue
                        # Dot directories BELOW the root stay pruned in git mode too
                        # (agent worktrees like .claude/ are untracked-but-not-ignored).
                        try:
                            parts = Path(fs).relative_to(base).parts[:-1]
                        except ValueError:
                            parts = ()
                        hidden = next((p for p in parts if p.startswith(".") or p in PRUNE_DIRS), None)
                        if hidden:
                            pruned.add(hidden)
                            continue
                        sel.append(Path(fs))
                    # git lists deleted-but-tracked files too (dropped), and a submodule
                    # as one directory entry (reported: its files are not in this list).
                    kept = []
                    for f in sel:
                        if f.is_file():
                            kept.append(f)
                        elif f.is_dir():
                            pruned.add(f"{f.name} (submodule)")
                    sel = kept
                    return FileSet(sel[:max_files], "git", pruned, len(sel) > max_files,
                                   f"file limit {max_files}" if len(sel) > max_files else "")
    out: list[Path] = []
    pruned = set()
    truncated = False
    note = ""
    for dirpath, dirnames, filenames in os.walk(root):
        keep = []
        for d in dirnames:
            full = Path(dirpath) / d
            if d in PRUNE_DIRS or d.startswith(".") or _is_link(full):
                pruned.add(d)
                continue
            keep.append(d)
        dirnames[:] = keep
        if include_dirs:
            out.extend(Path(dirpath) / d for d in keep)
        for fn in filenames:
            f = Path(dirpath) / fn
            if _is_link(f):
                continue
            out.append(f)
        if len(out) >= max_files:
            truncated, note = True, f"file limit {max_files}"
            out = out[:max_files]
            break
        if time.monotonic() - t0 > time_budget:
            truncated, note = True, f"walk time budget {time_budget:.0f}s"
            break
    return FileSet(out, "walk", pruned, truncated, note)


def glob_to_regex(pat: str) -> re.Pattern:
    """gitignore-style glob -> regex over a '/'-separated relative path.
    '**/' = zero or more directories, '**' = anything, '*' = anything but '/',
    '?' = one char but '/', '[...]' = a class. Case-insensitive on Windows."""
    i, out = 0, []
    while i < len(pat):
        c = pat[i]
        if pat.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pat.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif c == "[":
            j = pat.find("]", i + 1)
            if j < 0:
                out.append(re.escape(c))
                i += 1
            else:
                body = pat[i + 1:j].replace("\\", "\\\\")
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append(f"[{body}]")
                i = j + 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("".join(out) + r"\Z", re.IGNORECASE if os.name == "nt" else 0)


class GlobFilter:
    """A glob without a slash matches the file NAME (at any depth); with a slash,
    the path relative to the search root. Several globs separated by commas or
    spaces; a leading '!' excludes. Empty or '*' = everything."""

    def __init__(self, glob: str, root: Path):
        self.root = root
        self.all = not glob or glob.strip() == "*"
        pats = [g for g in re.split(r"[,\s]+", (glob or "").replace("\\", "/")) if g]
        self.inc = [(("/" in p.lstrip("/")), glob_to_regex(p.lstrip("/"))) for p in pats if not p.startswith("!")]
        self.exc = [(("/" in p[1:].lstrip("/")), glob_to_regex(p[1:].lstrip("/"))) for p in pats if p.startswith("!")]

    def __call__(self, f: Path) -> bool:
        if self.all:
            return True
        try:
            rel = str(f.relative_to(self.root)).replace("\\", "/")
        except ValueError:
            rel = f.name

        def m(entry) -> bool:
            has_slash, rx = entry
            return bool(rx.match(rel if has_slash else f.name))

        if any(m(e) for e in self.exc):
            return False
        return not self.inc or any(m(e) for e in self.inc)


def _glob_match(f: Path, root: Path, glob: str) -> bool:
    return GlobFilter(glob, root)(f)


@dataclass
class GrepResult:
    lines: list[str]
    hits: int
    files_matched: int
    files_searched: int
    skipped_binary: int
    skipped_unreadable: int
    stopped_early: bool
    file_counts: list[tuple[str, int]]
    fileset: FileSet
    timed_out: bool


def grep(root: Path, rx: re.Pattern, *, glob: str = "*", max_results: int = 200,
         context: int = 0, mode: str = "content", time_budget: float = 60.0,
         resolve=None) -> GrepResult:
    fs = list_files(root, time_budget=min(30.0, time_budget))
    t0 = time.monotonic()
    out: list[str] = []
    hits = files_matched = searched = binary = unreadable = 0
    counts: list[tuple[str, int]] = []
    stopped = timed_out = False
    # A file named explicitly as `path` is always searched; the glob filters directories.
    gf = GlobFilter(glob if root.is_dir() else "*", root if root.is_dir() else root.parent)
    for f in fs.files:
        if not gf(f):
            continue
        if time.monotonic() - t0 > time_budget:
            timed_out = True
            break
        try:
            if resolve is not None:
                f = resolve(f)
            with open(f, "rb") as fh:
                probe = fh.read(BINARY_PROBE)
                if b"\0" in probe:
                    binary += 1
                    continue
                fh.seek(0)
                searched += 1
                n_in_file = 0
                before: deque = deque(maxlen=context) if context else deque(maxlen=0)
                after_left = 0
                last_printed = 0
                for i, raw in enumerate(fh, 1):
                    ln = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                    if rx.search(ln):
                        n_in_file += 1
                        hits += 1
                        if mode == "content" and len(out) < max_results:
                            if context:
                                start = i - len(before)
                                if last_printed and start > last_printed + 1:
                                    out.append("--")
                                for j, b in enumerate(before):
                                    out.append(f"{f}-{start + j}- {b[:MAX_LINE_CHARS]}")
                                before.clear()
                            out.append(f"{f}:{i}: {ln.strip()[:MAX_LINE_CHARS] if not context else ln[:MAX_LINE_CHARS]}")
                            last_printed = i
                            after_left = context
                        elif mode == "content":
                            stopped = True
                            break
                        if mode != "content" and hits >= max_results * 50:
                            stopped = True
                            break
                    elif after_left > 0 and mode == "content" and len(out) < max_results + context * 2:
                        out.append(f"{f}-{i}- {ln[:MAX_LINE_CHARS]}")
                        last_printed = i
                        after_left -= 1
                    elif context:
                        before.append(ln)
                if n_in_file:
                    files_matched += 1
                    counts.append((str(f), n_in_file))
                    if mode == "files" and files_matched >= max_results:
                        stopped = True
        except (OSError, ValueError):
            unreadable += 1
            continue
        except Exception as e:  # sandbox rejection etc.
            if type(e).__name__ == "ToolError":
                unreadable += 1
                continue
            raise
        if stopped:
            break
    return GrepResult(out, hits, files_matched, searched, binary, unreadable, stopped,
                      counts, fs, timed_out)


def format_grep(pattern: str, root: Path, r: GrepResult, *, mode: str, max_results: int) -> str:
    head = f"grep {pattern!r} in {root} — "
    plus = "+" if (r.stopped_early or r.timed_out) else ""
    if mode == "files":
        head += f"{r.files_matched}{plus} file(s) with matches"
        body = "\n".join(p for p, _ in r.file_counts[:max_results])
    elif mode == "count":
        head += f"{r.hits}{plus} hit(s) in {r.files_matched}{plus} file(s)"
        body = "\n".join(f"{n:6d}  {p}" for p, n in sorted(r.file_counts, key=lambda x: -x[1])[:max_results])
    else:
        head += f"{r.hits}{plus} hit(s) in {r.files_matched}{plus} file(s)"
        if plus:
            head += f", showing the first {max_results}"
        body = "\n".join(r.lines)
    notes = [f"searched {r.files_searched} file(s) [{r.fileset.mode} file list]"]
    if r.skipped_binary:
        notes.append(f"{r.skipped_binary} binary skipped")
    if r.skipped_unreadable:
        notes.append(f"{r.skipped_unreadable} unreadable")
    if r.fileset.pruned_dirs:
        names = sorted(r.fileset.pruned_dirs)
        notes.append("not searched below root: " + ", ".join(names[:8]) + (" …" if len(names) > 8 else "")
                     + " (pass one as `path` to search it)")
    warn = []
    if r.files_searched == 0 and not r.skipped_binary and not r.skipped_unreadable and r.fileset.files:
        warn.append(f"the glob matched none of the {len(r.fileset.files)} candidate file(s) — check `glob`")
    if r.stopped_early:
        warn.append(f"STOPPED at max_results={max_results}: more matches exist — narrow the pattern/glob, "
                    "raise max_results, or use mode='count'")
    if r.fileset.truncated:
        warn.append(f"INCOMPLETE file list ({r.fileset.limit_note}): narrow `path`")
    if r.timed_out:
        warn.append("INCOMPLETE: search time budget exhausted — narrow `path` or `glob`")
    text = head + "\n(" + "; ".join(notes) + ")"
    if warn:
        text += "\n" + "\n".join("⚠ " + w for w in warn)
    return text + "\n" + (body or "(no matches)")
